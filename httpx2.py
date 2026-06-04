"""Compatibility shim for Starlette's optional httpx2 import."""

from __future__ import annotations

import httpx as _httpx
from httpx import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_httpx, name)
