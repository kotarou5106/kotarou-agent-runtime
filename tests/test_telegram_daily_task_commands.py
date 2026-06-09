from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import types

from agent_runtime.config_models import Config, TelegramChannelConfig
from proactive_system.config import (
    ProactiveConfig,
    TelegramAudioCollectorProactiveConfig,
    TelegramDailySummaryProactiveConfig,
    TelegramMorningGreetingProactiveConfig,
    TelegramTargetConfig,
)
from bootstrap.app import AppRuntime


def _load_telegram_channel_class(monkeypatch):
    telegram = types.ModuleType("telegram")
    telegram.Bot = object
    telegram.BotCommand = lambda command, description: (command, description)
    telegram.MessageEntity = lambda **kwargs: SimpleNamespace(**kwargs)
    telegram.Update = object
    constants = types.ModuleType("telegram.constants")
    constants.ChatAction = SimpleNamespace(TYPING="typing")
    error = types.ModuleType("telegram.error")
    for name in ("BadRequest", "Conflict", "NetworkError", "RetryAfter", "TelegramError", "TimedOut"):
        setattr(error, name, type(name, (Exception,), {}))
    ext = types.ModuleType("telegram.ext")
    ext.Application = SimpleNamespace(builder=lambda: SimpleNamespace(token=lambda _token: SimpleNamespace(build=lambda: SimpleNamespace())))
    ext.CommandHandler = lambda *args, **kwargs: ("command", args, kwargs)
    ext.MessageHandler = lambda *args, **kwargs: ("message", args, kwargs)
    ext.ContextTypes = SimpleNamespace(DEFAULT_TYPE=object)
    ext.filters = SimpleNamespace(
        COMMAND=object(),
        TEXT=object(),
        PHOTO=object(),
        Document=SimpleNamespace(ALL=object()),
    )
    monkeypatch.setitem(sys.modules, "telegram", telegram)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants)
    monkeypatch.setitem(sys.modules, "telegram.error", error)
    monkeypatch.setitem(sys.modules, "telegram.ext", ext)
    import connectors.channels.telegram_channel as module

    return module.TelegramChannel


class _Bus:
    def __init__(self) -> None:
        self.inbound = []

    async def publish_inbound(self, message) -> None:
        self.inbound.append(message)


def _channel(monkeypatch, *, runner=None, allow_from=None):
    TelegramChannel = _load_telegram_channel_class(monkeypatch)
    ch = TelegramChannel.__new__(TelegramChannel)
    ch._allow_from = set(allow_from or ["42", "alice"])
    ch._daily_task_runner = runner
    ch._app = SimpleNamespace(bot=object())
    ch._telegram_outbound_limiter = None
    ch._bus = _Bus()
    ch._channel = "telegram"
    ch._stream_response = False
    ch._message_deduper = SimpleNamespace(seen=lambda _key: False)
    return ch


def _update(text: str, *, user_id: int = 42, username: str = "alice", chat_id: int = 100):
    msg = SimpleNamespace(text=text, message_id=1)
    chat = SimpleNamespace(id=chat_id)
    user = SimpleNamespace(id=user_id, username=username)
    return SimpleNamespace(effective_message=msg, effective_chat=chat, effective_user=user)


async def test_daily_summary_command_calls_runner_and_sends_start(monkeypatch) -> None:
    sent = []
    calls = []

    async def fake_send(_bot, chat_id, message, _limiter):
        sent.append((chat_id, message))

    async def runner(task_name, chat_id):
        calls.append((task_name, chat_id))

    ch = _channel(monkeypatch, runner=runner)
    import connectors.channels.telegram_channel as tg_mod

    monkeypatch.setattr(tg_mod, "send_markdown", fake_send)

    await ch._on_daily_task_command(_update("/daily_summary"), SimpleNamespace())

    assert sent == [("100", "已开始执行 telegram_daily_summary，请稍等。")]
    assert calls == [("telegram_daily_summary", "100")]


async def test_morning_greeting_command_calls_runner(monkeypatch) -> None:
    calls = []

    async def fake_send(_bot, chat_id, message, _limiter):
        pass

    async def runner(task_name, chat_id):
        calls.append((task_name, chat_id))

    ch = _channel(monkeypatch, runner=runner)
    import connectors.channels.telegram_channel as tg_mod

    monkeypatch.setattr(tg_mod, "send_markdown", fake_send)

    await ch._on_daily_task_command(_update("/morning_greeting"), SimpleNamespace())

    assert calls == [("morning_greeting", "100")]


async def test_audio_collect_command_calls_runner(monkeypatch) -> None:
    calls = []

    async def fake_send(_bot, chat_id, message, _limiter):
        pass

    async def runner(task_name, chat_id):
        calls.append((task_name, chat_id))

    ch = _channel(monkeypatch, runner=runner)
    import connectors.channels.telegram_channel as tg_mod

    monkeypatch.setattr(tg_mod, "send_markdown", fake_send)

    await ch._on_daily_task_command(_update("/audio_collect"), SimpleNamespace())

    assert calls == [("telegram_audio_collector", "100")]


async def test_daily_tasks_runtime_runs_three_tasks_in_order(monkeypatch, tmp_path: Path) -> None:
    calls = []

    async def fake_summary(**kwargs):
        calls.append("summary")

    async def fake_greeting(**kwargs):
        calls.append("greeting")

    async def fake_audio(**kwargs):
        calls.append("audio")

    monkeypatch.setattr(
        "agent_runtime.background.telegram_daily_tasks.run_telegram_daily_summary_once",
        fake_summary,
    )
    monkeypatch.setattr(
        "agent_runtime.background.telegram_daily_tasks.run_telegram_morning_greeting_once",
        fake_greeting,
    )
    monkeypatch.setattr(
        "agent_runtime.background.telegram_daily_tasks.run_telegram_audio_collector_once",
        fake_audio,
    )

    config = Config(
        provider="fake",
        model="fake",
        api_key="",
        system_prompt="",
        channels=SimpleNamespace(telegram=TelegramChannelConfig(token="x", channel_name="telegram")),
        proactive=ProactiveConfig(
            telegram_daily_summary=TelegramDailySummaryProactiveConfig(
                summary_targets=[TelegramTargetConfig(chat_id="-1001", topic_id=6)]
            ),
            telegram_morning_greeting=TelegramMorningGreetingProactiveConfig(),
            telegram_audio_collector=TelegramAudioCollectorProactiveConfig(
                audio_targets=[TelegramTargetConfig(chat_id="-1002")]
            ),
        ),
    )
    runtime = AppRuntime(config, tmp_path)
    runtime.push_tool = object()
    runtime.agent_loop = object()

    await runtime.run_telegram_daily_task_command("telegram_daily_tasks", "100")

    assert calls == ["summary", "greeting", "audio"]


async def test_daily_task_exception_sends_error(monkeypatch) -> None:
    sent = []

    async def fake_send(_bot, chat_id, message, _limiter):
        sent.append((chat_id, message))

    async def runner(_task_name, _chat_id):
        raise RuntimeError("boom")

    ch = _channel(monkeypatch, runner=runner)
    import connectors.channels.telegram_channel as tg_mod

    monkeypatch.setattr(tg_mod, "send_markdown", fake_send)

    await ch._on_daily_task_command(_update("/daily_summary"), SimpleNamespace())

    assert sent[0] == ("100", "已开始执行 telegram_daily_summary，请稍等。")
    assert sent[1][0] == "100"
    assert "telegram_daily_summary 执行失败：RuntimeError: boom" in sent[1][1]


async def test_ping_command_replies_pong_without_agentloop(monkeypatch) -> None:
    sent = []

    async def fake_send(_bot, chat_id, message, _limiter):
        sent.append((chat_id, message))

    ch = _channel(monkeypatch, runner=None)
    import connectors.channels.telegram_channel as tg_mod

    monkeypatch.setattr(tg_mod, "send_markdown", fake_send)

    await ch._on_message(_update("/ping"), SimpleNamespace())

    assert ch._bus.inbound == []
    assert sent == [("100", "pong")]


async def test_unknown_command_does_not_publish_inbound(monkeypatch) -> None:
    sent = []

    async def fake_send(_bot, chat_id, message, _limiter):
        sent.append((chat_id, message))

    ch = _channel(monkeypatch, runner=None)
    import connectors.channels.telegram_channel as tg_mod

    monkeypatch.setattr(tg_mod, "send_markdown", fake_send)

    await ch._on_command(_update("/help"), SimpleNamespace())

    assert ch._bus.inbound == []
    assert sent == [("100", "未知命令：/help。请一次只发送一个已支持的命令。")]


async def test_morning_greeting_command_reaches_daily_task_runner_without_agentloop(monkeypatch) -> None:
    sent = []
    calls = []

    async def fake_send(_bot, chat_id, message, _limiter):
        sent.append((chat_id, message))

    async def runner(task_name, chat_id):
        calls.append((task_name, chat_id))

    ch = _channel(monkeypatch, runner=runner)
    import connectors.channels.telegram_channel as tg_mod

    monkeypatch.setattr(tg_mod, "send_markdown", fake_send)

    await ch._on_message(_update("/morning_greeting"), SimpleNamespace())

    assert ch._bus.inbound == []
    assert sent[0] == ("100", "已开始执行 morning_greeting，请稍等。")
    assert calls == [("morning_greeting", "100")]


async def test_multiline_slash_command_does_not_publish_inbound(monkeypatch) -> None:
    sent = []

    async def fake_send(_bot, chat_id, message, _limiter):
        sent.append((chat_id, message))

    ch = _channel(monkeypatch, runner=None)
    import connectors.channels.telegram_channel as tg_mod

    monkeypatch.setattr(tg_mod, "send_markdown", fake_send)

    await ch._on_message(_update("/ping\n/morning_greeting"), SimpleNamespace())

    assert ch._bus.inbound == []
    assert sent == [("100", "请一次只发送一个命令。")]


async def test_fast_path_plain_hello_does_not_enter_agentloop(monkeypatch) -> None:
    sent = []

    async def fake_send(_bot, chat_id, message, _limiter):
        sent.append((chat_id, message))

    ch = _channel(monkeypatch, runner=None)
    import connectors.channels.telegram_channel as tg_mod

    monkeypatch.setattr(tg_mod, "send_markdown", fake_send)

    await ch._on_message(_update("你好"), SimpleNamespace())

    assert sent == []
    assert len(ch._bus.inbound) == 1


async def test_complex_plain_message_enters_agentloop(monkeypatch) -> None:
    sent = []

    async def fake_send(_bot, chat_id, message, _limiter):
        sent.append((chat_id, message))

    ch = _channel(monkeypatch, runner=None)
    import connectors.channels.telegram_channel as tg_mod

    monkeypatch.setattr(tg_mod, "send_markdown", fake_send)

    await ch._on_message(_update("请帮我总结这段话的要点"), SimpleNamespace())

    assert sent == []
    assert len(ch._bus.inbound) == 1
