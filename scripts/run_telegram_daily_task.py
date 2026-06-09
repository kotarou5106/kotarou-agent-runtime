from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_runtime.background.telegram_daily_tasks import (  # noqa: E402
    MorningGreetingConfig,
    TelegramAudioCollectorConfig,
    TelegramDailySummaryConfig,
    TelegramTarget,
    parse_telegram_chat_id,
    run_telegram_audio_collector_once,
    run_telegram_daily_summary_once,
    run_telegram_morning_greeting_once,
)
from agent_runtime.config import Config  # noqa: E402
from bootstrap.app import build_app_runtime  # noqa: E402


def _resolve_send_target(send_to: str, default_channel: str, default_chat_id: str) -> tuple[str, str]:
    text = str(send_to or "").strip()
    if not text or text == "me":
        return default_channel, default_chat_id
    if ":" in text:
        channel, chat_id = text.split(":", 1)
        return channel.strip() or default_channel, chat_id.strip()
    return default_channel, text


def _target(item) -> TelegramTarget:
    return TelegramTarget(
        chat_id=parse_telegram_chat_id(item.chat_id),
        chat_title=item.chat_title,
        topic_id=item.topic_id,
    )


async def run_task(task: str, config_path: str, workspace: Path) -> None:
    config = Config.load(config_path)
    runtime = build_app_runtime(config, workspace)
    await runtime.start()
    try:
        if task == "summary":
            cfg = config.proactive.telegram_daily_summary
            channel, chat_id = _resolve_send_target(
                cfg.send_to,
                config.proactive.default_channel,
                config.proactive.default_chat_id,
            )
            result = await run_telegram_daily_summary_once(
                config=TelegramDailySummaryConfig(
                    summary_targets=[_target(item) for item in cfg.summary_targets],
                    lookback_hours=cfg.lookback_hours,
                    max_messages_per_target=cfg.max_messages_per_target,
                    include_original_links=cfg.include_original_links,
                ),
                agent_loop=runtime.agent_loop,
                push_tool=runtime.push_tool,
                send_channel=channel,
                send_chat_id=chat_id,
            )
            print(result)
        elif task == "greeting":
            cfg = config.proactive.telegram_morning_greeting
            channel, chat_id = _resolve_send_target(
                cfg.send_to,
                config.proactive.default_channel,
                config.proactive.default_chat_id,
            )
            result = await run_telegram_morning_greeting_once(
                config=MorningGreetingConfig(
                    style_pool=list(cfg.style_pool),
                    avoid_recent_days=cfg.avoid_recent_days,
                    state_path=workspace / "data" / "morning_greeting_history.json",
                    model=config.light_model or config.model,
                ),
                llm_provider=runtime.light_provider or runtime.provider,
                push_tool=runtime.push_tool,
                send_channel=channel,
                send_chat_id=chat_id,
            )
            print(result)
        elif task == "audio":
            cfg = config.proactive.telegram_audio_collector
            channel, chat_id = _resolve_send_target(
                cfg.send_to,
                config.proactive.default_channel,
                config.proactive.default_chat_id,
            )
            download_dir = Path(cfg.audio_download_dir)
            if not download_dir.is_absolute():
                download_dir = workspace / download_dir
            result = await run_telegram_audio_collector_once(
                config=TelegramAudioCollectorConfig(
                    audio_targets=[_target(item) for item in cfg.audio_targets],
                    lookback_hours=cfg.lookback_hours,
                    max_messages_per_target=cfg.max_messages_per_target,
                    download_audio=cfg.download_audio,
                    audio_download_dir=download_dir,
                    include_original_links=cfg.include_original_links,
                    keywords=list(cfg.keywords),
                    exclude_keywords=list(cfg.exclude_keywords),
                ),
                push_tool=runtime.push_tool,
                send_channel=channel,
                send_chat_id=chat_id,
            )
            print(result)
        else:
            raise SystemExit(f"unknown task: {task}")
    finally:
        await runtime.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manually run Telegram daily background tasks.")
    parser.add_argument("--task", choices=["summary", "greeting", "audio"], required=True)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--workspace", default=str(Path.home() / ".kotarou" / "workspace"))
    args = parser.parse_args()
    asyncio.run(run_task(args.task, args.config, Path(args.workspace)))


if __name__ == "__main__":
    main()
