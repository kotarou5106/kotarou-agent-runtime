from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_runtime.config import Config  # noqa: E402
from agent_runtime.env_loader import (  # noqa: E402
    WATCH_ENV_KEYS,
    load_dotenv_for_config,
    mask_secret,
)
from agent_runtime.network_proxy import PROXY_ENV_KEYS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local .env loading for this runtime.")
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()
    config_path = Path(args.config)

    result = load_dotenv_for_config(config_path)
    config = Config.load(config_path)

    print("Environment check")
    print(f"cwd: {Path.cwd()}")
    print(f"config_path: {config_path.resolve()}")
    print(f"dotenv_path: {result.dotenv_path or ''}")
    print(f"dotenv_exists: {result.dotenv_exists}")
    print(f"dotenv_loaded: {result.dotenv_loaded}")
    print("searched_paths:")
    for path in result.searched_paths:
        print(f"  {path}")
    for key in WATCH_ENV_KEYS:
        value = os.getenv(key, "")
        print(f"{key}: present={bool(value)} masked={mask_secret(value)}")
    for key in PROXY_ENV_KEYS:
        print(f"proxy.{key}: present={bool(os.getenv(key))}")
    print(f"config.llm.main.api_key_present: {bool(config.api_key)}")
    print(f"config.llm.main.api_key_placeholder: {'${' in (config.api_key or '')}")
    print(f"config.channels.telegram.enabled: {bool(config.channels.telegram)}")
    if result.dotenv_exists and not result.dotenv_loaded:
        print(f"warning: .env exists but was not loaded: {result.dotenv_path}")


if __name__ == "__main__":
    main()
