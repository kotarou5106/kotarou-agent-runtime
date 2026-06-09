from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_runtime.background.telegram_job_monitor import (
    JobMonitorResult,
    JobSections,
    TelegramJobMonitorConfig,
    _format_notify_message,
    _split_for_telegram,
    build_telegram_message_link,
    evaluate_job_message,
    load_job_monitor_config_from_env,
    main,
    parse_job_sections,
    run_telegram_job_monitor_once,
)

__all__ = [
    "JobMonitorResult",
    "JobSections",
    "TelegramJobMonitorConfig",
    "_format_notify_message",
    "_split_for_telegram",
    "build_telegram_message_link",
    "evaluate_job_message",
    "load_job_monitor_config_from_env",
    "parse_job_sections",
    "run_telegram_job_monitor_once",
]


if __name__ == "__main__":
    asyncio.run(main())
