from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
_LOGGED_CONTEXTS: set[str] = set()


@dataclass(frozen=True)
class ProxyEnvStatus:
    http_proxy: bool
    https_proxy: bool
    all_proxy: bool


@dataclass(frozen=True)
class ProxySelection:
    url: str
    source: str


def proxy_env_status() -> ProxyEnvStatus:
    return ProxyEnvStatus(
        http_proxy=bool(os.getenv("HTTP_PROXY")),
        https_proxy=bool(os.getenv("HTTPS_PROXY")),
        all_proxy=bool(os.getenv("ALL_PROXY")),
    )


def log_proxy_env(context: str, *, once: bool = False) -> None:
    if once and context in _LOGGED_CONTEXTS:
        return
    _LOGGED_CONTEXTS.add(context)
    status = proxy_env_status()
    logger.info("[%s] proxy.HTTP_PROXY present=%s", context, status.http_proxy)
    logger.info("[%s] proxy.HTTPS_PROXY present=%s", context, status.https_proxy)
    logger.info("[%s] proxy.ALL_PROXY present=%s", context, status.all_proxy)


def select_http_proxy() -> ProxySelection | None:
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        value = os.getenv(key, "").strip()
        if value:
            return ProxySelection(url=value, source=key)
    return None


def parse_proxy_url(url: str) -> tuple[str, str, int] | None:
    parsed = urlparse(str(url or "").strip())
    if not parsed.scheme or not parsed.hostname:
        return None
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https", "socks4", "socks5"}:
        return None
    if parsed.port is None:
        default_port = 443 if scheme == "https" else 1080 if scheme.startswith("socks") else 80
        port = default_port
    else:
        port = parsed.port
    telethon_scheme = "http" if scheme == "https" else scheme
    return telethon_scheme, parsed.hostname, port


def select_telethon_proxy() -> tuple[tuple[str, str, int] | None, str | None]:
    for key in ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY"):
        value = os.getenv(key, "").strip()
        if not value:
            continue
        parsed = parse_proxy_url(value)
        if parsed is not None:
            return parsed, key
    return None, None
