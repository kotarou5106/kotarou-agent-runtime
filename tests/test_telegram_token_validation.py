from __future__ import annotations

import sys
import types
from typing import Any, cast

import pytest

from agent_runtime.config_models import (
    ChannelsConfig,
    Config,
    TelegramChannelConfig,
)
from agent_runtime.events.event_bus import EventBus
from agent_runtime.core.net.http import SharedHttpResources
from bootstrap.channels import start_channels
from connectors.channels.telegram_token import is_valid_telegram_bot_token


VALID_TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabc"


def test_telegram_token_validation_cases() -> None:
    assert not is_valid_telegram_bot_token("")
    assert not is_valid_telegram_bot_token("你的 BotFather token")
    assert not is_valid_telegram_bot_token("${TELEGRAM_BOT_TOKEN}")
    assert not is_valid_telegram_bot_token("not-a-token")
    assert is_valid_telegram_bot_token(VALID_TOKEN)


@pytest.mark.asyncio
async def test_invalid_telegram_token_skips_channel_without_traceback(monkeypatch, tmp_path) -> None:
    starts: list[str] = []
    fake_ipc_server = types.ModuleType("connectors.channels.ipc_server")
    fake_telegram_channel = types.ModuleType("connectors.channels.telegram_channel")

    class _IPCServerChannel:
        def __init__(self, bus, socket):
            self.bus = bus
            self.socket = socket

        async def start(self) -> None:
            starts.append("ipc")

    class _TelegramChannel:
        def __init__(self, **kwargs):
            raise AssertionError("TelegramChannel must not be constructed")

    fake_ipc_server.IPCServerChannel = _IPCServerChannel
    fake_telegram_channel.TelegramChannel = _TelegramChannel
    monkeypatch.setitem(sys.modules, "connectors.channels.ipc_server", fake_ipc_server)
    monkeypatch.setitem(sys.modules, "connectors.channels.telegram_channel", fake_telegram_channel)

    class _PushTool:
        def register_channel(self, *args, **kwargs) -> None:
            raise AssertionError("invalid telegram token must not register channel")

    config = Config(
        provider="fake",
        model="fake",
        api_key="",
        system_prompt="",
        channels=ChannelsConfig(
            telegram=TelegramChannelConfig(token="你的 BotFather token"),
            socket=str(tmp_path / "sock"),
        ),
    )
    resources = SharedHttpResources()
    try:
        ipc, tg, qq, qqbot = await start_channels(
            config,
            bus=cast(Any, object()),
            session_manager=cast(Any, object()),
            push_tool=cast(Any, _PushTool()),
            http_resources=resources,
            event_bus=EventBus(),
        )
    finally:
        await resources.aclose()

    assert ipc is not None
    assert tg is None
    assert qq is None
    assert qqbot is None
    assert starts == ["ipc"]


@pytest.mark.asyncio
async def test_valid_telegram_token_starts_channel(monkeypatch, tmp_path) -> None:
    starts: list[str] = []
    fake_ipc_server = types.ModuleType("connectors.channels.ipc_server")
    fake_telegram_channel = types.ModuleType("connectors.channels.telegram_channel")

    class _IPCServerChannel:
        def __init__(self, bus, socket):
            self.bus = bus
            self.socket = socket

        async def start(self) -> None:
            starts.append("ipc")

    class _TelegramChannel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def start(self) -> None:
            starts.append("telegram")

        async def send(self, *args, **kwargs):
            return None

        async def send_stream(self, *args, **kwargs):
            return None

        async def send_file(self, *args, **kwargs):
            return None

        async def send_image(self, *args, **kwargs):
            return None

    fake_ipc_server.IPCServerChannel = _IPCServerChannel
    fake_telegram_channel.TelegramChannel = _TelegramChannel
    monkeypatch.setitem(sys.modules, "connectors.channels.ipc_server", fake_ipc_server)
    monkeypatch.setitem(sys.modules, "connectors.channels.telegram_channel", fake_telegram_channel)

    registrations = []

    class _PushTool:
        def register_channel(self, name: str, **kwargs) -> None:
            registrations.append(name)

    config = Config(
        provider="fake",
        model="fake",
        api_key="",
        system_prompt="",
        channels=ChannelsConfig(
            telegram=TelegramChannelConfig(token=VALID_TOKEN),
            socket=str(tmp_path / "sock"),
        ),
    )
    resources = SharedHttpResources()
    try:
        ipc, tg, _qq, _qqbot = await start_channels(
            config,
            bus=cast(Any, object()),
            session_manager=cast(Any, object()),
            push_tool=cast(Any, _PushTool()),
            http_resources=resources,
            event_bus=EventBus(),
        )
    finally:
        await resources.aclose()

    assert ipc is not None
    assert tg is not None
    assert starts == ["ipc", "telegram"]
    assert registrations == ["telegram"]
