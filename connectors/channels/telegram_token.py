from __future__ import annotations

import re

_BOT_TOKEN_RE = re.compile(r"^\d{6,}:[A-Za-z0-9_-]{20,}$")
_PLACEHOLDER_MARKERS = {
    "你的 BotFather token",
    "your botfather token",
    "botfather token",
    "your_token",
    "telegram_bot_token",
}


def is_valid_telegram_bot_token(token: str) -> bool:
    text = str(token or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if "${" in text:
        return False
    if any(marker.lower() in lowered for marker in _PLACEHOLDER_MARKERS):
        return False
    return _BOT_TOKEN_RE.fullmatch(text) is not None


def mask_telegram_bot_token(token: str) -> str:
    text = str(token or "").strip()
    if not text:
        return "<empty>"
    if len(text) <= 10:
        return text[:2] + "..."
    return f"{text[:6]}...{text[-4:]}"
