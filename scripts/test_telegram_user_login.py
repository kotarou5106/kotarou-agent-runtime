from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient

TelegramProxy = tuple[str, str, int]


def _load_telegram_user_config() -> tuple[int, str, str, str, TelegramProxy | None]:
    load_dotenv(Path.cwd() / ".env")

    api_id_raw = os.getenv("TELEGRAM_API_ID", "").strip()
    api_hash = os.getenv("TELEGRAM_API_HASH", "").strip()
    phone = os.getenv("TELEGRAM_PHONE", "").strip()
    session_name = os.getenv("TELEGRAM_JOB_SESSION_NAME", "").strip()
    if not session_name:
        session_name = "job_monitor_session"

    missing = [
        name
        for name, value in (
            ("TELEGRAM_API_ID", api_id_raw),
            ("TELEGRAM_API_HASH", api_hash),
            ("TELEGRAM_PHONE", phone),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            "Missing required environment variables: "
            + ", ".join(missing)
            + "\nFill them in .env or export them before running this script."
        )

    try:
        api_id = int(api_id_raw)
    except ValueError as exc:
        raise SystemExit("TELEGRAM_API_ID must be an integer.") from exc

    return api_id, api_hash, phone, session_name, _load_telegram_proxy()


def _load_telegram_proxy() -> TelegramProxy | None:
    proxy_type = os.getenv("TELEGRAM_PROXY_TYPE", "").strip()
    proxy_host = os.getenv("TELEGRAM_PROXY_HOST", "").strip()
    proxy_port_raw = os.getenv("TELEGRAM_PROXY_PORT", "").strip()

    if not any((proxy_type, proxy_host, proxy_port_raw)):
        return None
    if not all((proxy_type, proxy_host, proxy_port_raw)):
        raise SystemExit(
            "TELEGRAM_PROXY_TYPE, TELEGRAM_PROXY_HOST, and TELEGRAM_PROXY_PORT "
            "must all be set to enable Telegram proxy."
        )
    try:
        proxy_port = int(proxy_port_raw)
    except ValueError as exc:
        raise SystemExit("TELEGRAM_PROXY_PORT must be an integer.") from exc
    return proxy_type, proxy_host, proxy_port


def build_telegram_client() -> tuple[TelegramClient, str]:
    api_id, api_hash, phone, session_name, proxy = _load_telegram_user_config()
    client = TelegramClient(session_name, api_id, api_hash, proxy=proxy)
    return client, phone


async def main() -> None:
    client, phone = build_telegram_client()
    try:
        await client.start(phone=phone)
        me = await client.get_me()
        display_name = (
            getattr(me, "username", None)
            or getattr(me, "first_name", None)
            or str(getattr(me, "id", ""))
        )
        print("Login OK")
        print(display_name)
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
