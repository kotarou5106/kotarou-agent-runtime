from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_runtime.config import Config  # noqa: E402
from agent_runtime.env_loader import load_dotenv_for_config, mask_secret  # noqa: E402
from agent_runtime.network_proxy import PROXY_ENV_KEYS, proxy_env_status  # noqa: E402
from connectors.channels.telegram_token import is_valid_telegram_bot_token  # noqa: E402


async def _check_telegram_get_me(token: str, timeout_s: float) -> tuple[bool, str]:
    if not token:
        return False, "skipped: TELEGRAM_BOT_TOKEN missing"
    if not is_valid_telegram_bot_token(token):
        return False, "skipped: token format is not a valid BotFather token"
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        async with httpx.AsyncClient(timeout=timeout_s, trust_env=True) as client:
            resp = await client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            user = data.get("result", {}) if isinstance(data, dict) else {}
            return True, f"ok: @{user.get('username', '<unknown>')} id={user.get('id', '<unknown>')}"
        return False, f"http {resp.status_code}: {resp.text[:160]}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


async def _check_deepseek(config: Config, timeout_s: float) -> tuple[bool, str]:
    api_key = config.api_key or os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return False, "skipped: DEEPSEEK_API_KEY missing"
    base_url = (config.base_url or "https://api.deepseek.com/v1").rstrip("/")
    model = config.model or "deepseek-chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_s, trust_env=True) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        if 200 <= resp.status_code < 300:
            return True, f"ok: model={model}"
        return False, f"http {resp.status_code}: {resp.text[:200]}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


async def _amain() -> None:
    parser = argparse.ArgumentParser(description="Check dotenv, proxy, Telegram, and DeepSeek network access.")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--timeout", type=float, default=12.0)
    args = parser.parse_args()

    config_path = Path(args.config)
    dotenv = load_dotenv_for_config(config_path)
    config = Config.load(config_path)
    status = proxy_env_status()
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN") or (
        config.channels.telegram.token if config.channels.telegram else ""
    )

    print("Network check")
    print(f"cwd: {Path.cwd()}")
    print(f"config_path: {config_path.resolve()}")
    print(f"dotenv_path: {dotenv.dotenv_path or ''}")
    print(f"dotenv_exists: {dotenv.dotenv_exists}")
    print(f"dotenv_loaded: {dotenv.dotenv_loaded}")
    for key in PROXY_ENV_KEYS:
        print(f"proxy.{key}: present={bool(os.getenv(key))}")
    print(f"proxy.status: HTTP={status.http_proxy} HTTPS={status.https_proxy} ALL={status.all_proxy}")
    print(f"TELEGRAM_BOT_TOKEN: present={bool(telegram_token)} masked={mask_secret(telegram_token)}")
    print(f"TELEGRAM_BOT_TOKEN_valid_shape: {is_valid_telegram_bot_token(telegram_token)}")
    print(f"DEEPSEEK_API_KEY: present={bool(os.getenv('DEEPSEEK_API_KEY') or config.api_key)} masked={mask_secret(os.getenv('DEEPSEEK_API_KEY') or config.api_key)}")

    tg_ok, tg_message = await _check_telegram_get_me(telegram_token, args.timeout)
    print(f"telegram.getMe: ok={tg_ok} {tg_message}")
    ds_ok, ds_message = await _check_deepseek(config, args.timeout)
    print(f"deepseek.minimal_request: ok={ds_ok} {ds_message}")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
