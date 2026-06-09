from __future__ import annotations

from pathlib import Path

from agent_runtime.config_models import Config
from bootstrap.app import AppRuntime
from proactive_system.config import (
    ProactiveConfig,
    TelegramAudioCollectorProactiveConfig,
    TelegramDailySummaryProactiveConfig,
    TelegramJobMonitorProactiveConfig,
    TelegramMorningGreetingProactiveConfig,
    TelegramTargetConfig,
)


class _Scheduler:
    def __init__(self) -> None:
        self.registered: list[dict[str, object]] = []

    def register_recurring_callback(self, **kwargs) -> None:
        self.registered.append(kwargs)


def test_app_runtime_registers_telegram_job_monitor_callback(tmp_path: Path) -> None:
    config = Config(
        provider="fake",
        model="fake",
        api_key="",
        system_prompt="",
        proactive=ProactiveConfig(
            telegram_job_monitor=TelegramJobMonitorProactiveConfig(
                enabled=True,
                schedule="06:00",
                timezone="Asia/Shanghai",
                target_chat_id="-1001570628112",
                read_limit=50,
                debug=False,
            )
        ),
    )
    runtime = AppRuntime(config, tmp_path)
    scheduler = _Scheduler()
    runtime.scheduler = scheduler

    runtime._register_telegram_job_monitor_task()

    assert len(scheduler.registered) == 1
    registered = scheduler.registered[0]
    assert registered["name"] == "telegram_job_monitor"
    assert registered["cron_expr"] == "0 6 * * *"
    assert registered["timezone"] == "Asia/Shanghai"
    assert callable(registered["callback"])


def test_app_runtime_skips_telegram_job_monitor_without_target(tmp_path: Path) -> None:
    config = Config(
        provider="fake",
        model="fake",
        api_key="",
        system_prompt="",
        proactive=ProactiveConfig(
            telegram_job_monitor=TelegramJobMonitorProactiveConfig(
                enabled=True,
                target_chat_id="",
            )
        ),
    )
    runtime = AppRuntime(config, tmp_path)
    scheduler = _Scheduler()
    runtime.scheduler = scheduler

    runtime._register_telegram_job_monitor_task()

    assert scheduler.registered == []


def test_app_runtime_registers_telegram_daily_summary_and_greeting(tmp_path: Path) -> None:
    config = Config(
        provider="fake",
        model="fake",
        api_key="",
        system_prompt="",
        proactive=ProactiveConfig(
            default_channel="telegram",
            default_chat_id="12345",
            telegram_daily_summary=TelegramDailySummaryProactiveConfig(
                enabled=True,
                time="06:00",
                timezone="Asia/Shanghai",
                summary_targets=[
                    TelegramTargetConfig(
                        chat_id="-1001570628112",
                        chat_title="study-group",
                        topic_id=13017,
                    )
                ],
                lookback_hours=24,
                max_messages_per_target=500,
            ),
            telegram_morning_greeting=TelegramMorningGreetingProactiveConfig(
                enabled=True,
                time="06:05",
                timezone="Asia/Shanghai",
            ),
            telegram_audio_collector=TelegramAudioCollectorProactiveConfig(
                enabled=True,
                time="06:10",
                timezone="Asia/Shanghai",
                audio_targets=[
                    TelegramTargetConfig(
                        chat_id="-1001570628112",
                        chat_title="study-group",
                        topic_id=13017,
                    )
                ],
            ),
        ),
    )
    runtime = AppRuntime(config, tmp_path)
    scheduler = _Scheduler()
    runtime.scheduler = scheduler
    runtime.agent_loop = object()
    runtime.push_tool = object()

    runtime._register_telegram_daily_tasks()

    assert len(scheduler.registered) == 3
    by_name = {str(item["name"]): item for item in scheduler.registered}
    assert by_name["telegram_daily_summary"]["cron_expr"] == "0 6 * * *"
    assert by_name["telegram_daily_summary"]["timezone"] == "Asia/Shanghai"
    assert callable(by_name["telegram_daily_summary"]["callback"])
    assert by_name["morning_greeting"]["cron_expr"] == "5 6 * * *"
    assert by_name["morning_greeting"]["timezone"] == "Asia/Shanghai"
    assert callable(by_name["morning_greeting"]["callback"])
    assert by_name["telegram_audio_collector"]["cron_expr"] == "10 6 * * *"
    assert by_name["telegram_audio_collector"]["timezone"] == "Asia/Shanghai"
    assert callable(by_name["telegram_audio_collector"]["callback"])
