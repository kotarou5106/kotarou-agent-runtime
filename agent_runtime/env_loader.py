from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from agent_runtime.network_proxy import PROXY_ENV_KEYS

logger = logging.getLogger(__name__)

WATCH_ENV_KEYS = (
    "DEEPSEEK_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "DASHSCOPE_API_KEY",
)


@dataclass(frozen=True)
class DotenvLoadResult:
    dotenv_path: Path | None
    dotenv_exists: bool
    dotenv_loaded: bool
    searched_paths: tuple[Path, ...]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dotenv_candidates(config_path: str | Path = "config.toml") -> list[Path]:
    config_dir = Path(config_path).expanduser()
    if not config_dir.is_absolute():
        config_dir = Path.cwd() / config_dir
    config_dir = config_dir.resolve().parent
    roots = [config_dir, Path.cwd().resolve(), _project_root()]

    seen: set[Path] = set()
    candidates: list[Path] = []
    for root in roots:
        for name in (".env", "env"):
            path = (root / name).resolve()
            if path not in seen:
                seen.add(path)
                candidates.append(path)
    return candidates


def load_dotenv_for_config(
    config_path: str | Path = "config.toml",
    *,
    log: bool = True,
) -> DotenvLoadResult:
    candidates = dotenv_candidates(config_path)
    dotenv_path = next((path for path in candidates if path.exists()), None)
    loaded = False
    if dotenv_path is not None:
        loaded = bool(load_dotenv(dotenv_path=dotenv_path, override=False))

    result = DotenvLoadResult(
        dotenv_path=dotenv_path,
        dotenv_exists=dotenv_path is not None,
        dotenv_loaded=loaded,
        searched_paths=tuple(candidates),
    )
    if log:
        _log_dotenv_result(result)
    return result


def _log_dotenv_result(result: DotenvLoadResult) -> None:
    logger.info(
        "[env] dotenv_path=%s dotenv_loaded=%s",
        str(result.dotenv_path) if result.dotenv_path else "",
        result.dotenv_loaded,
    )
    for key in WATCH_ENV_KEYS:
        logger.info("[env] %s present=%s", key, bool(os.getenv(key)))
    for key in PROXY_ENV_KEYS:
        logger.info("[env] proxy.%s present=%s", key, bool(os.getenv(key)))
    if result.dotenv_exists and not result.dotenv_loaded:
        logger.warning("[env] .env exists but no values were loaded: %s", result.dotenv_path)
    if not result.dotenv_exists:
        logger.warning(
            "[env] no .env/env file found. searched=%s",
            [str(path) for path in result.searched_paths],
        )


def mask_secret(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "<empty>"
    if len(text) <= 10:
        return text[:2] + "..."
    return f"{text[:6]}...{text[-4:]}"
