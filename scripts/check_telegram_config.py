from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_runtime.config import Config  # noqa: E402
from connectors.channels.telegram_token import (  # noqa: E402
    is_valid_telegram_bot_token,
    mask_telegram_bot_token,
)


def _resolve_env_placeholders(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("${") and text.endswith("}"):
        return os.getenv(text[2:-1], "").strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Telegram Bot channel configuration.")
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()

    config_path = Path(args.config)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    raw_tg = raw.get("channels", {}).get("telegram", {}) or {}
    raw_token = _resolve_env_placeholders(str(raw_tg.get("token", "") or ""))
    raw_allow_from = [str(item) for item in raw_tg.get("allow_from", raw_tg.get("allowFrom", []))]

    config = Config.load(config_path)
    tg = config.channels.telegram
    token = tg.token if tg else raw_token
    token_present = bool(token)
    token_valid = is_valid_telegram_bot_token(token)
    allow_from = tg.allow_from if tg else raw_allow_from
    proactive_chat_id = config.proactive.default_chat_id

    print("Telegram Bot config check")
    print(f"config: {config_path}")
    print(f"token_present: {token_present}")
    print(f"token_masked: {mask_telegram_bot_token(token)}")
    print(f"token_looks_valid: {token_valid}")
    print(f"allow_from_configured: {bool(allow_from)}")
    print(f"allow_from: {allow_from if allow_from else '[]'}")
    print(f"proactive.target.chat_id_empty: {not bool(proactive_chat_id)}")
    print(f"telegram_channel_can_start: {bool(tg and token_valid)}")
    if not token_valid:
        print(
            "warning: [telegram] invalid or missing bot token; Telegram Bot channel disabled."
        )
        print("Please set TELEGRAM_BOT_TOKEN or configure [channels.telegram].token.")


if __name__ == "__main__":
    main()
