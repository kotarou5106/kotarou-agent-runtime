from __future__ import annotations

import asyncio
import logging
import sys
import re
from pathlib import Path
from typing import Awaitable, Callable

from agent_runtime.config_models import Config
from bootstrap.channels import start_channels
from dashboard.backend.api import build_dashboard_server
from bootstrap.memory import build_memory_runtime
from bootstrap.proactive import build_memory_optimizer_task, build_proactive_runtime
from bootstrap.providers import build_providers
from bootstrap.tools import CoreRuntime, build_core_runtime
from agent_runtime.events.event_bus import EventBus
from agent_runtime.core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def _run_cleanup_steps(*steps: tuple[str, Callable[[], Awaitable[None]]]) -> None:
    first_error: Exception | None = None
    for name, step in steps:
        try:
            await step()
        except Exception as exc:
            if first_error is None:
                first_error = exc
            logger.warning("shutdown step failed: %s: %s", name, exc)
    if first_error is not None:
        raise first_error


async def _noop_async() -> None:
    return None


class AppRuntime:
    def __init__(self, config: Config, workspace: Path) -> None:
        self.config = config
        self.workspace = workspace
        self.http_resources = SharedHttpResources()
        self.ipc = None
        self.tg_channel = None
        self.qq_channel = None
        self.qqbot_channel = None
        self.core: CoreRuntime | None = None
        self.agent_loop = None
        self.bus = None
        self.event_bus: EventBus | None = None
        self.tools = None
        self.push_tool = None
        self.session_manager = None
        self.scheduler = None
        self.provider = None
        self.light_provider = None
        self.mcp_registry = None
        self.memory_runtime = None
        self.presence = None
        self.proactive_loop = None
        self.peer_process_manager = None
        self.peer_poller = None
        self.dashboard_server = None
        self.dashboard_task: asyncio.Task[None] | None = None
        self.tasks: list[Awaitable[None]] = []
        self._memory_optimizer = None
        self._shutdown = False
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        configure_default_shared_http_resources(self.http_resources)
        try:
            self.core = build_core_runtime(
                self.config,
                self.workspace,
                self.http_resources,
            )
            self.agent_loop = self.core.loop
            self.bus = self.core.bus
            event_bus = self.core.event_bus
            self.event_bus = event_bus
            self.tools = self.core.tools
            self.push_tool = self.core.push_tool
            self.session_manager = self.core.session_manager
            self.scheduler = self.core.scheduler
            self.provider = self.core.provider
            self.light_provider = self.core.light_provider
            self.mcp_registry = self.core.mcp_registry
            self.memory_runtime = self.core.memory_runtime
            self.presence = self.core.presence
            self.peer_process_manager = self.core.peer_process_manager
            self.peer_poller = self.core.peer_poller
            await self.core.start()
            self._register_telegram_daily_tasks()

            plugin_manager = getattr(self.core, "plugin_manager", None)
            self.ipc, self.tg_channel, self.qq_channel, self.qqbot_channel = await start_channels(
                self.config,
                bus=self.bus,
                session_manager=self.session_manager,
                push_tool=self.push_tool,
                http_resources=self.http_resources,
                event_bus=event_bus,
                bot_commands=(
                    plugin_manager.telegram_bot_commands
                    if plugin_manager
                    else None
                ),
                interrupt_controller=self.agent_loop,
                telegram_daily_task_runner=self.run_telegram_daily_task_command,
            )

            self.tasks = [
                self.agent_loop.run(),
                self.bus.dispatch_outbound(),
                self.scheduler.run(),
            ]
            optimizer_tasks, self._memory_optimizer = build_memory_optimizer_task(
                self.config,
                provider=self.provider,
                memory_store=self.memory_runtime.markdown.store,
            )
            self.tasks.extend(optimizer_tasks)
            self.dashboard_server = build_dashboard_server(
                workspace=self.workspace,
                manual_consolidator=self.agent_loop,
                manual_memory_optimizer=self._memory_optimizer,
                memory_admin=self.memory_runtime.engine,
                memory_store=self.memory_runtime.markdown.store,
            )
            self.dashboard_task = asyncio.create_task(
                self.dashboard_server.serve(),
                name="dashboard_server",
            )
            proactive_tasks, self.proactive_loop = build_proactive_runtime(
                self.config,
                self.workspace,
                session_manager=self.session_manager,
                provider=self.provider,
                light_provider=self.light_provider,
                push_tool=self.push_tool,
                memory_store=self.memory_runtime,
                presence=self.presence,
                agent_loop=self.agent_loop,
                tool_hooks=list(plugin_manager.tool_hooks) if plugin_manager else None,
            )
            self.tasks.extend(proactive_tasks)
            if self.proactive_loop is not None:
                self.ipc.set_proactive_loop(self.proactive_loop)

            self._started = True
        except Exception:
            await self.shutdown()
            raise

    def _register_telegram_job_monitor_task(self) -> None:
        proactive_cfg = getattr(self.config, "proactive", None)
        if proactive_cfg is None:
            return

        monitor_cfg = proactive_cfg.telegram_job_monitor
        if not monitor_cfg.enabled:
            return
        if not monitor_cfg.target_chat_id:
            logger.warning(
                "[telegram_job_monitor] enabled but target_chat_id is empty; task not registered"
            )
            return
        if self.scheduler is None:
            logger.warning("[telegram_job_monitor] scheduler unavailable; task not registered")
            return

        from agent_runtime.background.telegram_job_monitor import (
            TelegramJobMonitorConfig,
            run_telegram_job_monitor_once,
        )

        target_chat_id = _parse_chat_id(monitor_cfg.target_chat_id)
        job_config = TelegramJobMonitorConfig(
            target_chat_id=target_chat_id,
            read_limit=monitor_cfg.read_limit,
            debug=monitor_cfg.debug,
        )
        cron_expr = _schedule_to_daily_cron(monitor_cfg.schedule)

        async def _run_once() -> None:
            try:
                await run_telegram_job_monitor_once(job_config)
            except Exception:
                logger.exception("[telegram_job_monitor] execution failed")

        self.scheduler.register_recurring_callback(
            name="telegram_job_monitor",
            cron_expr=cron_expr,
            timezone=monitor_cfg.timezone,
            callback=_run_once,
        )
        logger.info(
            "[telegram_job_monitor] registered schedule=%s timezone=%s target_chat_id=%s read_limit=%s debug=%s",
            monitor_cfg.schedule,
            monitor_cfg.timezone,
            monitor_cfg.target_chat_id,
            monitor_cfg.read_limit,
            monitor_cfg.debug,
        )

    def _register_telegram_daily_tasks(self) -> None:
        proactive_cfg = getattr(self.config, "proactive", None)
        if proactive_cfg is None:
            return
        if self.scheduler is None:
            logger.warning("[telegram_daily_tasks] scheduler unavailable; tasks not registered")
            return

        summary_cfg = proactive_cfg.telegram_daily_summary
        if summary_cfg.enabled:
            send_channel, send_chat_id = _resolve_send_target(
                summary_cfg.send_to or summary_cfg.target_chat_id,
                proactive_cfg.default_channel,
                proactive_cfg.default_chat_id,
            )
            if not send_channel or not send_chat_id:
                logger.warning(
                    "[telegram_daily_summary] enabled but send target is empty; task not registered"
                )
            elif not summary_cfg.summary_targets:
                logger.warning(
                    "[telegram_daily_summary] enabled but summary_targets is empty; task not registered"
                )
            else:
                from agent_runtime.background.telegram_daily_tasks import (
                    TelegramDailySummaryConfig,
                    TelegramTarget,
                    parse_telegram_chat_id,
                    run_telegram_daily_summary_once,
                )

                task_config = TelegramDailySummaryConfig(
                    summary_targets=[
                        TelegramTarget(
                            chat_id=parse_telegram_chat_id(group.chat_id),
                            chat_title=group.chat_title,
                            topic_id=group.topic_id,
                            expected_topic_title=group.expected_topic_title,
                        )
                        for group in summary_cfg.summary_targets
                    ],
                    lookback_hours=summary_cfg.lookback_hours,
                    max_messages_per_target=summary_cfg.max_messages_per_target,
                    include_original_links=summary_cfg.include_original_links,
                )

                async def _run_summary_once() -> None:
                    try:
                        await run_telegram_daily_summary_once(
                            config=task_config,
                            agent_loop=self.agent_loop,
                            push_tool=self.push_tool,
                            send_channel=send_channel,
                            send_chat_id=send_chat_id,
                        )
                    except Exception:
                        logger.exception("[telegram_daily_summary] execution failed")

                self.scheduler.register_recurring_callback(
                    name="telegram_daily_summary",
                    cron_expr=_schedule_to_daily_cron(summary_cfg.time),
                    timezone=summary_cfg.timezone,
                    callback=_run_summary_once,
                )
                logger.info(
                    "[telegram_daily_summary] registered schedule=%s timezone=%s target_chat_id=%s groups=%s",
                    summary_cfg.time,
                    summary_cfg.timezone,
                    send_chat_id,
                    len(summary_cfg.summary_targets),
                )

        greeting_cfg = proactive_cfg.telegram_morning_greeting
        if greeting_cfg.enabled:
            send_channel, send_chat_id = _resolve_send_target(
                greeting_cfg.send_to or greeting_cfg.target_chat_id,
                proactive_cfg.default_channel,
                proactive_cfg.default_chat_id,
            )
            if not send_channel or not send_chat_id:
                logger.warning(
                    "[telegram_morning_greeting] enabled but send target is empty; task not registered"
                )
            else:
                from agent_runtime.background.telegram_daily_tasks import (
                    MorningGreetingConfig,
                    run_telegram_morning_greeting_once,
                )

                task_config = MorningGreetingConfig(
                    style_pool=list(greeting_cfg.style_pool),
                    avoid_recent_days=greeting_cfg.avoid_recent_days,
                    state_path=self.workspace / "data" / "morning_greeting_history.json",
                    model=self.config.light_model or self.config.model,
                )

                async def _run_greeting_once() -> None:
                    try:
                        await run_telegram_morning_greeting_once(
                            config=task_config,
                            llm_provider=self.light_provider or self.provider,
                            push_tool=self.push_tool,
                            send_channel=send_channel,
                            send_chat_id=send_chat_id,
                        )
                    except Exception:
                        logger.exception("[telegram_morning_greeting] execution failed")

                self.scheduler.register_recurring_callback(
                    name="morning_greeting",
                    cron_expr=_schedule_to_daily_cron(greeting_cfg.time),
                    timezone=greeting_cfg.timezone,
                    callback=_run_greeting_once,
                )
                logger.info(
                    "[telegram_morning_greeting] registered schedule=%s timezone=%s target_chat_id=%s",
                    greeting_cfg.time,
                    greeting_cfg.timezone,
                    send_chat_id,
                )

        audio_cfg = proactive_cfg.telegram_audio_collector
        if audio_cfg.enabled:
            send_channel, send_chat_id = _resolve_send_target(
                audio_cfg.send_to or audio_cfg.target_chat_id,
                proactive_cfg.default_channel,
                proactive_cfg.default_chat_id,
            )
            if not send_channel or not send_chat_id:
                logger.warning(
                    "[telegram_audio_collector] enabled but send target is empty; task not registered"
                )
            elif not audio_cfg.audio_targets:
                logger.warning(
                    "[telegram_audio_collector] enabled but audio_targets is empty; task not registered"
                )
            else:
                from agent_runtime.background.telegram_daily_tasks import (
                    TelegramAudioCollectorConfig,
                    TelegramTarget,
                    parse_telegram_chat_id,
                    run_telegram_audio_collector_once,
                )

                download_dir = Path(audio_cfg.audio_download_dir)
                if not download_dir.is_absolute():
                    download_dir = self.workspace / download_dir
                task_config = TelegramAudioCollectorConfig(
                    audio_targets=[
                        TelegramTarget(
                            chat_id=parse_telegram_chat_id(group.chat_id),
                            chat_title=group.chat_title,
                            topic_id=group.topic_id,
                        )
                        for group in audio_cfg.audio_targets
                    ],
                    lookback_hours=audio_cfg.lookback_hours,
                    max_messages_per_target=audio_cfg.max_messages_per_target,
                    download_audio=audio_cfg.download_audio,
                    audio_download_dir=download_dir,
                    include_original_links=audio_cfg.include_original_links,
                    keywords=list(audio_cfg.keywords),
                    exclude_keywords=list(audio_cfg.exclude_keywords),
                )

                async def _run_audio_once() -> None:
                    try:
                        await run_telegram_audio_collector_once(
                            config=task_config,
                            push_tool=self.push_tool,
                            send_channel=send_channel,
                            send_chat_id=send_chat_id,
                        )
                    except Exception:
                        logger.exception("[telegram_audio_collector] execution failed")

                self.scheduler.register_recurring_callback(
                    name="telegram_audio_collector",
                    cron_expr=_schedule_to_daily_cron(audio_cfg.time),
                    timezone=audio_cfg.timezone,
                    callback=_run_audio_once,
                )
                logger.info(
                    "[telegram_audio_collector] registered schedule=%s timezone=%s target_chat_id=%s targets=%s",
                    audio_cfg.time,
                    audio_cfg.timezone,
                    send_chat_id,
                    len(audio_cfg.audio_targets),
                )

    async def run_telegram_daily_task_command(self, task_name: str, chat_id: str) -> None:
        proactive_cfg = getattr(self.config, "proactive", None)
        if proactive_cfg is None:
            raise RuntimeError("proactive config is unavailable")

        if task_name == "telegram_daily_tasks":
            for name in (
                "telegram_daily_summary",
                "morning_greeting",
                "telegram_audio_collector",
            ):
                try:
                    await self.run_telegram_daily_task_command(name, chat_id)
                except Exception as exc:
                    if self.push_tool is None:
                        raise
                    send_channel = (
                        self.config.channels.telegram.channel_name
                        if self.config.channels.telegram is not None
                        else proactive_cfg.default_channel
                    )
                    await self.push_tool.execute(
                        channel=send_channel,
                        chat_id=chat_id,
                        message=f"{name} 执行失败：{type(exc).__name__}: {exc}",
                    )
            return

        if self.push_tool is None:
            raise RuntimeError("message_push is unavailable")
        send_channel = (
            self.config.channels.telegram.channel_name
            if self.config.channels.telegram is not None
            else proactive_cfg.default_channel
        )

        if task_name == "telegram_daily_summary":
            summary_cfg = proactive_cfg.telegram_daily_summary
            from agent_runtime.background.telegram_daily_tasks import (
                TelegramDailySummaryConfig,
                TelegramTarget,
                parse_telegram_chat_id,
                run_telegram_daily_summary_once,
            )

            await run_telegram_daily_summary_once(
                config=TelegramDailySummaryConfig(
                    summary_targets=[
                        TelegramTarget(
                            chat_id=parse_telegram_chat_id(group.chat_id),
                            chat_title=group.chat_title,
                            topic_id=group.topic_id,
                            expected_topic_title=group.expected_topic_title,
                        )
                        for group in summary_cfg.summary_targets
                    ],
                    lookback_hours=summary_cfg.lookback_hours,
                    max_messages_per_target=summary_cfg.max_messages_per_target,
                    include_original_links=summary_cfg.include_original_links,
                ),
                agent_loop=self.agent_loop,
                push_tool=self.push_tool,
                send_channel=send_channel,
                send_chat_id=chat_id,
            )
            return

        if task_name == "morning_greeting":
            greeting_cfg = proactive_cfg.telegram_morning_greeting
            from agent_runtime.background.telegram_daily_tasks import (
                MorningGreetingConfig,
                run_telegram_morning_greeting_once,
            )

            await run_telegram_morning_greeting_once(
                config=MorningGreetingConfig(
                    style_pool=list(greeting_cfg.style_pool),
                    avoid_recent_days=greeting_cfg.avoid_recent_days,
                    state_path=self.workspace / "data" / "morning_greeting_history.json",
                    model=self.config.light_model or self.config.model,
                ),
                llm_provider=self.light_provider or self.provider,
                push_tool=self.push_tool,
                send_channel=send_channel,
                send_chat_id=chat_id,
            )
            return

        if task_name == "telegram_audio_collector":
            audio_cfg = proactive_cfg.telegram_audio_collector
            from agent_runtime.background.telegram_daily_tasks import (
                TelegramAudioCollectorConfig,
                TelegramTarget,
                parse_telegram_chat_id,
                run_telegram_audio_collector_once,
            )

            download_dir = Path(audio_cfg.audio_download_dir)
            if not download_dir.is_absolute():
                download_dir = self.workspace / download_dir
            await run_telegram_audio_collector_once(
                config=TelegramAudioCollectorConfig(
                    audio_targets=[
                        TelegramTarget(
                            chat_id=parse_telegram_chat_id(group.chat_id),
                            chat_title=group.chat_title,
                            topic_id=group.topic_id,
                        )
                        for group in audio_cfg.audio_targets
                    ],
                    lookback_hours=audio_cfg.lookback_hours,
                    max_messages_per_target=audio_cfg.max_messages_per_target,
                    download_audio=audio_cfg.download_audio,
                    audio_download_dir=download_dir,
                    include_original_links=audio_cfg.include_original_links,
                    keywords=list(audio_cfg.keywords),
                    exclude_keywords=list(audio_cfg.exclude_keywords),
                ),
                push_tool=self.push_tool,
                send_channel=send_channel,
                send_chat_id=chat_id,
            )
            return

        raise ValueError(f"unknown telegram daily task: {task_name}")

    async def run(self) -> None:
        try:
            await self.start()
            await asyncio.gather(*self.tasks)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        try:
            if self.dashboard_server is not None:
                self.dashboard_server.should_exit = True
            if self.dashboard_task is not None:
                try:
                    await self.dashboard_task
                except asyncio.CancelledError:
                    pass
            await _run_cleanup_steps(
                ("agent_runtime.core.stop", self.core.stop if self.core else _noop_async),
                ("ipc.stop", self.ipc.stop if self.ipc else _noop_async),
                (
                    "telegram.stop",
                    self.tg_channel.stop if self.tg_channel else _noop_async,
                ),
                ("qq.stop", self.qq_channel.stop if self.qq_channel else _noop_async),
                (
                    "qqbot.stop",
                    self.qqbot_channel.stop if self.qqbot_channel else _noop_async,
                ),
                (
                    "memory_runtime.aclose",
                    self.memory_runtime.aclose if self.memory_runtime else _noop_async,
                ),
                ("http_resources.aclose", self.http_resources.aclose),
            )
        finally:
            clear_default_shared_http_resources(self.http_resources)


def build_app_runtime(config: Config, workspace: Path | None = None) -> AppRuntime:
    return AppRuntime(config, workspace or (Path.home() / ".kotarou" / "workspace"))


def _parse_chat_id(value: str) -> int | str:
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def _schedule_to_daily_cron(value: str) -> str:
    text = str(value).strip()
    if re.match(r"^\d{1,2}:\d{2}$", text):
        hour_s, minute_s = text.split(":", 1)
        hour = int(hour_s)
        minute = int(minute_s)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"Invalid telegram job monitor schedule: {value!r}")
        return f"{minute} {hour} * * *"
    return text


def _resolve_send_target(
    send_to: str,
    default_channel: str,
    default_chat_id: str,
) -> tuple[str, str]:
    text = str(send_to or "").strip()
    if not text or text == "me":
        return default_channel, default_chat_id
    if ":" in text:
        channel, chat_id = text.split(":", 1)
        return channel.strip() or default_channel, chat_id.strip()
    return default_channel, text
