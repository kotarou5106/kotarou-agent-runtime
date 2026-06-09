from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from telethon.tl.types import Channel, Chat, User

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_telegram_user_login import build_telegram_client


def _entity_type(entity: object) -> str:
    if isinstance(entity, Channel):
        return "Channel"
    if isinstance(entity, Chat):
        return "Chat"
    if isinstance(entity, User):
        return "User"
    return type(entity).__name__


def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


def _dialog_lines(index: int, dialog: object) -> list[str]:
    entity = getattr(dialog, "entity", None)
    name = str(getattr(dialog, "name", "") or getattr(entity, "title", "") or "")
    entity_id = getattr(entity, "id", "")
    username = getattr(entity, "username", "") or ""
    entity_type = _entity_type(entity)
    is_channel = isinstance(entity, Channel)
    is_chat = isinstance(entity, Chat)
    is_megagroup = bool(getattr(entity, "megagroup", False))
    is_forum = bool(getattr(entity, "forum", False))
    is_group = is_chat or is_megagroup

    return [
        f"[{index:03d}] {name}",
        f"  copy_id: {entity_id}",
        f"  type: {entity_type}",
        f"  username: {('@' + username) if username else '-'}",
        f"  is_channel: {_format_bool(is_channel)}",
        f"  is_group: {_format_bool(is_group)}",
        f"  is_megagroup: {_format_bool(is_megagroup)}",
        f"  is_forum: {_format_bool(is_forum)}",
    ]


async def main() -> None:
    client, phone = build_telegram_client()
    try:
        await client.start(phone=phone)
        print("Telegram dialogs (first 100)")
        print("Use copy_id for Telethon chat/channel lookup.")
        print("")
        index = 0
        async for dialog in client.iter_dialogs(limit=100):
            index += 1
            print("\n".join(_dialog_lines(index, dialog)))
            print("")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
