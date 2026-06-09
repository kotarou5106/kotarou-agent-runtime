from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_telegram_user_login import build_telegram_client


def _load_target_chat() -> int | str:
    raw = os.getenv("TELEGRAM_TARGET_CHAT_ID", "").strip()
    if not raw:
        raise SystemExit(
            "Missing required environment variable: TELEGRAM_TARGET_CHAT_ID\n"
            "Fill it in .env or export it before running this script."
        )
    try:
        return int(raw)
    except ValueError:
        return raw


def _load_read_limit() -> int:
    raw = os.getenv("TELEGRAM_READ_LIMIT", "").strip()
    if not raw:
        return 20
    try:
        limit = int(raw)
    except ValueError as exc:
        raise SystemExit("TELEGRAM_READ_LIMIT must be an integer.") from exc
    return max(1, limit)


def _message_text(message: object) -> str:
    text = str(getattr(message, "message", "") or "")
    if not text.strip():
        return "[non-text message]"
    text = text.strip()
    return text[:500]


async def main() -> None:
    client, phone = build_telegram_client()
    target_chat = _load_target_chat()
    limit = _load_read_limit()
    try:
        await client.start(phone=phone)
        entity = await client.get_entity(target_chat)
        title = (
            getattr(entity, "title", None)
            or getattr(entity, "username", None)
            or getattr(entity, "first_name", None)
            or str(target_chat)
        )
        print(f"Reading latest {limit} messages from: {title}")
        print(f"target: {target_chat}")
        print("")
        count = 0
        async for message in client.iter_messages(entity, limit=limit):
            count += 1
            print(f"[{count:03d}] message_id: {getattr(message, 'id', '')}")
            print(f"  date: {getattr(message, 'date', '')}")
            print(f"  sender_id: {getattr(message, 'sender_id', '')}")
            print("  text:")
            print(_message_text(message))
            print("")
        if count == 0:
            print("No messages found.")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
