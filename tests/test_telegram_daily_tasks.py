from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from agent_runtime.background import telegram_daily_tasks as daily
from agent_runtime.background.telegram_job_monitor import build_telegram_message_link


def _raw_message(
    *,
    topic_id: int | None = None,
    filename: str = "",
    mime_type: str = "",
    text: str = "",
    size: int = 123,
) -> SimpleNamespace:
    attrs = [SimpleNamespace(file_name=filename)] if filename else []
    document = SimpleNamespace(attributes=attrs, mime_type=mime_type, size=size)
    reply_to = (
        SimpleNamespace(reply_to_top_id=topic_id, reply_to_msg_id=None, forum_topic=True)
        if topic_id
        else None
    )
    return SimpleNamespace(
        id=8207,
        message=text,
        reply_to=reply_to,
        document=document,
        file=SimpleNamespace(name=filename, mime_type=mime_type, size=size),
    )


def _msg(
    *,
    hours_ago: int = 1,
    text: str = "hello",
    topic_id: int | None = 13017,
    raw_message: object | None = None,
    source_link: str = "https://t.me/c/1570628112/1",
) -> daily.TelegramGroupMessage:
    now = datetime(2026, 6, 9, 7, 0, tzinfo=timezone.utc)
    return daily.TelegramGroupMessage(
        chat_id=-1001570628112,
        chat_title="study-group",
        topic_id=topic_id,
        message_id=8207,
        date=now - timedelta(hours=hours_ago),
        sender="alice",
        text=text,
        original_message_link=source_link,
        raw_message=raw_message,
    )


def test_filter_messages_by_time_window_keeps_recent_messages() -> None:
    now = datetime(2026, 6, 9, 7, 0, tzinfo=timezone.utc)
    messages = [_msg(hours_ago=1), _msg(hours_ago=25)]

    filtered = daily.filter_messages_by_time_window(
        messages,
        now=now,
        window_hours=24,
    )

    assert filtered == [messages[0]]


def test_topic_id_inference_and_target_filtering() -> None:
    message = _raw_message(topic_id=13017)

    assert daily.infer_message_topic_id(message) == 13017
    assert daily.target_matches_message(daily.TelegramTarget(chat_id=-1, topic_id=13017), message)
    assert not daily.target_matches_message(daily.TelegramTarget(chat_id=-1, topic_id=265), message)


def test_original_message_link_generation() -> None:
    private_chat = SimpleNamespace(id=-1001570628112, username=None)
    public_chat = SimpleNamespace(id=-1001570628112, username="public_group")
    message = SimpleNamespace(id=8207)

    assert build_telegram_message_link(public_chat, message) == "https://t.me/public_group/8207"
    assert build_telegram_message_link(private_chat, message) == "https://t.me/c/1570628112/8207"


def test_build_daily_summary_prompt_includes_source_links_and_group_counts() -> None:
    message = _msg(
        text="今天讨论了 agent runtime 的 Telegram 摘要。",
        source_link="https://t.me/c/1570628112/8207",
    )

    prompt = daily.build_daily_summary_prompt([message], lookback_hours=24)

    assert "今日消息总量：1" in prompt
    assert "topic_id=13017 count=1" in prompt
    assert "今天讨论了 agent runtime" in prompt
    assert "original_message_link: https://t.me/c/1570628112/8207" in prompt


def test_morning_greeting_style_changes_by_date() -> None:
    pool = ["Foucault", "Derrida", "Kafka"]

    styles = {
        daily.choose_style_for_date(date(2026, 6, day), pool)
        for day in range(1, 8)
    }

    assert len(styles) > 1


def test_generic_error_message_detection() -> None:
    assert daily.is_generic_error_message(daily.GENERIC_ERROR_MESSAGE)
    assert daily.is_generic_error_message(f"  {daily.GENERIC_ERROR_MESSAGE}  ")
    assert not daily.is_generic_error_message(None)
    assert not daily.is_generic_error_message("")
    assert not daily.is_generic_error_message("正常文案")


def test_morning_greeting_recent_similarity_detection() -> None:
    candidate = "早上好。今天不需要成为完整的人，只需要继续行动，让轮廓慢慢出现。"
    recent = ["早上好。今天不需要把自己整理成完整的人，只需要继续行动。"]

    assert daily.greeting_too_similar(candidate, recent)


async def test_run_telegram_daily_summary_uses_llm_and_push(monkeypatch) -> None:
    message = _msg(text="需要总结的消息", topic_id=6631)
    monkeypatch.setattr(daily, "_resolve_target_topic_title_for_config", lambda target: __import__("asyncio").sleep(0, result="吃瓜闲聊-禁发招聘"))

    async def fake_read_messages(**kwargs):
        return [message]

    monkeypatch.setattr(daily, "read_telegram_target_messages", fake_read_messages)
    loop = SimpleNamespace()
    push = SimpleNamespace(sent=[])

    async def process_direct(**kwargs):
        loop.kwargs = kwargs
        return "这是总结"

    async def execute(**kwargs):
        push.sent.append(kwargs)
        return "文本已发送"

    loop.process_direct = process_direct
    push.execute = execute

    result = await daily.run_telegram_daily_summary_once(
        config=daily.TelegramDailySummaryConfig(
            summary_targets=[
                daily.TelegramTarget(
                    chat_id=-1001570628112,
                    topic_id=6631,
                    expected_topic_title="吃瓜闲聊-禁发招聘",
                )
            ],
        ),
        agent_loop=loop,
        push_tool=push,
        send_channel="telegram",
        send_chat_id="12345",
    )

    assert result.sent is True
    assert result.included_messages == 1
    assert "topic_id=6631" in loop.kwargs["content"] or "吃瓜闲聊-禁发招聘" in loop.kwargs["content"]
    assert "需要总结的消息" in loop.kwargs["content"]
    assert loop.kwargs["disabled_tools"] == ["message_push"]
    assert push.sent == [
        {"channel": "telegram", "chat_id": "12345", "message": "这是总结"}
    ]


async def test_daily_summary_filters_to_topic_6631_only(monkeypatch) -> None:
    seen_topics: list[int | None] = []
    monkeypatch.setattr(daily, "_resolve_target_topic_title_for_config", lambda target: __import__("asyncio").sleep(0, result="吃瓜闲聊-禁发招聘"))

    async def fake_read_messages(**kwargs):
        seen_topics.extend(target.topic_id for target in kwargs["targets"])
        return [
            _msg(text="吃瓜闲聊消息", topic_id=6631),
        ]

    monkeypatch.setattr(daily, "read_telegram_target_messages", fake_read_messages)

    captured = {}

    async def process_direct(**kwargs):
        captured["content"] = kwargs["content"]
        return "总结结果"

    push = SimpleNamespace(execute=lambda **kwargs: __import__("asyncio").sleep(0, result="文本已发送"))
    loop = SimpleNamespace(process_direct=process_direct)

    result = await daily.run_telegram_daily_summary_once(
        config=daily.TelegramDailySummaryConfig(
            summary_targets=[
                daily.TelegramTarget(
                    chat_id=-1002535833398,
                    topic_id=6631,
                    expected_topic_title="吃瓜闲聊-禁发招聘",
                )
            ],
        ),
        agent_loop=loop,
        push_tool=push,
        send_channel="telegram",
        send_chat_id="12345",
    )

    assert result.sent is True
    assert seen_topics == [6631]
    assert "topic_id=6631" in captured["content"]


async def test_daily_summary_target_without_topic_is_skipped(monkeypatch) -> None:
    called = False

    async def fake_read_messages(**kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(daily, "read_telegram_target_messages", fake_read_messages)
    monkeypatch.setattr(daily, "_resolve_target_topic_title_for_config", lambda target: __import__("asyncio").sleep(0, result="<unknown>"))

    async def execute(**kwargs):
        return "文本已发送"

    push = SimpleNamespace(execute=execute)
    loop = SimpleNamespace(process_direct=lambda **kwargs: __import__("asyncio").sleep(0, result="总结"))

    result = await daily.run_telegram_daily_summary_once(
        config=daily.TelegramDailySummaryConfig(
            summary_targets=[daily.TelegramTarget(chat_id=-1002535833398)],
        ),
        agent_loop=loop,
        push_tool=push,
        send_channel="telegram",
        send_chat_id="12345",
    )

    assert result.read_messages == 0
    assert called is False
    assert result.sent is False


async def test_daily_summary_topic_title_mismatch_returns_config_error(monkeypatch) -> None:
    async def fake_resolve(target):
        return "招聘求职"

    monkeypatch.setattr(daily, "_resolve_target_topic_title_for_config", fake_resolve)
    push_calls = []

    async def execute(**kwargs):
        push_calls.append(kwargs)
        return "文本已发送"

    push = SimpleNamespace(execute=execute)
    loop = SimpleNamespace(process_direct=lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not summarize")))

    result = await daily.run_telegram_daily_summary_once(
        config=daily.TelegramDailySummaryConfig(
            summary_targets=[
                daily.TelegramTarget(
                    chat_id=-1002535833398,
                    topic_id=6631,
                    expected_topic_title="吃瓜闲聊-禁发招聘",
                )
            ],
        ),
        agent_loop=loop,
        push_tool=push,
        send_channel="telegram",
        send_chat_id="12345",
    )

    assert result.sent is True
    assert push_calls and "配置错误" in push_calls[0]["message"]


async def test_run_telegram_morning_greeting_uses_llm_push_and_history(tmp_path: Path) -> None:
    provider = SimpleNamespace()
    push = SimpleNamespace(sent=[])

    async def chat(**kwargs):
        provider.kwargs = kwargs
        return SimpleNamespace(content="早上好。今天把注意力收回到手边，先完成一件具体的小事。")

    async def execute(**kwargs):
        push.sent.append(kwargs)
        return "文本已发送"

    provider.chat = chat
    push.execute = execute

    result = await daily.run_telegram_morning_greeting_once(
        config=daily.MorningGreetingConfig(
            state_path=tmp_path / "history.json",
            model="deepseek-chat",
        ),
        llm_provider=provider,
        push_tool=push,
        send_channel="telegram",
        send_chat_id="12345",
        today=date(2026, 6, 9),
    )

    assert result.sent is True
    assert result.style
    assert result.success is True
    assert result.fallback is False
    assert result.generation_failed is False
    assert result.error_message is None
    assert "风格参考" in provider.kwargs["messages"][0]["content"]
    assert provider.kwargs["tool_choice"] == "none"
    assert provider.kwargs["disable_thinking"] is True
    assert push.sent[0]["message"] == result.message
    assert result.message == "早上好。今天把注意力收回到手边，先完成一件具体的小事。"


async def test_morning_greeting_llm_timeout_uses_fallback(tmp_path: Path, monkeypatch) -> None:
    provider = SimpleNamespace()
    push = SimpleNamespace(sent=[])
    logs = []

    async def chat(**kwargs):
        raise TimeoutError("LLM timeout")

    async def execute(**kwargs):
        push.sent.append(kwargs)
        return "文本已发送"

    def fake_exception(message, *args, **kwargs):
        logs.append((message, args, kwargs))

    provider.chat = chat
    push.execute = execute
    monkeypatch.setattr(daily.logger, "exception", fake_exception)

    result = await daily.run_telegram_morning_greeting_once(
        config=daily.MorningGreetingConfig(state_path=tmp_path / "history.json", model="deepseek-chat"),
        llm_provider=provider,
        push_tool=push,
        send_channel="telegram",
        send_chat_id="12345",
        today=date(2026, 6, 9),
    )

    assert logs
    assert result.sent is True
    assert result.success is True
    assert result.fallback is True
    assert result.generation_failed is True
    assert result.error_message and "timeout" in result.error_message.lower()
    assert result.message == push.sent[0]["message"]
    assert result.message != daily.GENERIC_ERROR_MESSAGE


async def test_morning_greeting_401_uses_fallback(tmp_path: Path, monkeypatch) -> None:
    provider = SimpleNamespace()
    push = SimpleNamespace(sent=[])

    async def chat(**kwargs):
        raise RuntimeError("401 unauthorized")

    async def execute(**kwargs):
        push.sent.append(kwargs)
        return "文本已发送"

    provider.chat = chat
    push.execute = execute
    monkeypatch.setattr(daily.logger, "exception", lambda *args, **kwargs: None)

    result = await daily.run_telegram_morning_greeting_once(
        config=daily.MorningGreetingConfig(state_path=tmp_path / "history.json", model="deepseek-chat"),
        llm_provider=provider,
        push_tool=push,
        send_channel="telegram",
        send_chat_id="12345",
        today=date(2026, 6, 9),
    )

    assert result.fallback is True
    assert result.success is True
    assert result.generation_failed is True
    assert "401" in (result.error_message or "")
    assert result.message == push.sent[0]["message"]


async def test_morning_greeting_generic_failure_text_uses_fallback(tmp_path: Path) -> None:
    provider = SimpleNamespace()
    push = SimpleNamespace(sent=[])

    async def chat(**kwargs):
        return SimpleNamespace(content=daily.GENERIC_ERROR_MESSAGE)

    async def execute(**kwargs):
        push.sent.append(kwargs)
        return "文本已发送"

    provider.chat = chat
    push.execute = execute

    result = await daily.run_telegram_morning_greeting_once(
        config=daily.MorningGreetingConfig(state_path=tmp_path / "history.json", model="deepseek-chat"),
        llm_provider=provider,
        push_tool=push,
        send_channel="telegram",
        send_chat_id="12345",
        today=date(2026, 6, 9),
    )

    assert result.sent is True
    assert result.success is True
    assert result.fallback is True
    assert result.generation_failed is True
    assert result.message != daily.GENERIC_ERROR_MESSAGE
    assert daily.GENERIC_ERROR_MESSAGE not in push.sent[0]["message"]


async def test_morning_greeting_does_not_use_agent_loop(tmp_path: Path) -> None:
    provider = SimpleNamespace()
    push = SimpleNamespace(sent=[])
    agent_loop = SimpleNamespace(process_direct=lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not call")))

    async def chat(**kwargs):
        return SimpleNamespace(content="早上好。今天先把一件具体的小事做完。")

    async def execute(**kwargs):
        push.sent.append(kwargs)
        return "文本已发送"

    provider.chat = chat
    push.execute = execute

    result = await daily.run_telegram_morning_greeting_once(
        config=daily.MorningGreetingConfig(state_path=tmp_path / "history.json", model="deepseek-chat"),
        agent_loop=agent_loop,
        llm_provider=provider,
        push_tool=push,
        send_channel="telegram",
        send_chat_id="12345",
        today=date(2026, 6, 9),
    )

    assert result.sent is True
    assert push.sent[0]["message"] != daily.GENERIC_ERROR_MESSAGE


async def test_morning_greeting_history_generic_error_is_auto_repaired(tmp_path: Path) -> None:
    history = tmp_path / "history.json"
    history.write_text(
        """[
  {
    "date": "2026-06-09",
    "style": "Kafka",
    "message": "处理消息时出错，请稍后再试。",
    "success": true,
    "fallback": false,
    "error_message": "",
    "generation_failed": true,
    "created_at": "2026-06-09T00:00:00+00:00"
  }
]
""",
        encoding="utf-8",
    )

    content, style, success, fallback, error_message, generation_failed = await daily.generate_unique_greeting(
        config=daily.MorningGreetingConfig(state_path=history),
        today=date(2026, 6, 9),
    )

    assert success is True
    assert fallback is True
    assert generation_failed is True
    assert error_message
    assert not daily.is_generic_error_message(content)
    assert "已使用本地 fallback" not in content
    saved = history.read_text(encoding="utf-8")
    assert daily.GENERIC_ERROR_MESSAGE not in content
    assert "已使用本地 fallback" not in content
    assert "已使用本地 fallback" not in saved
    assert style == "Kafka"


async def test_morning_greeting_exception_logs_and_uses_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    exceptions = []
    provider = SimpleNamespace()
    push = SimpleNamespace(sent=[])

    async def chat(**kwargs):
        raise RuntimeError("DeepSeek 401 unauthorized")

    async def execute(**kwargs):
        push.sent.append(kwargs)
        return "文本已发送"

    def fake_exception(message, *args, **kwargs):
        exceptions.append((message, args, kwargs))

    provider.chat = chat
    push.execute = execute
    monkeypatch.setattr(daily.logger, "exception", fake_exception)

    result = await daily.run_telegram_morning_greeting_once(
        config=daily.MorningGreetingConfig(state_path=tmp_path / "history.json", model="deepseek-chat"),
        llm_provider=provider,
        push_tool=push,
        send_channel="telegram",
        send_chat_id="12345",
        today=date(2026, 6, 9),
    )

    assert exceptions
    assert result.success is True or result.success is False
    assert result.fallback is True
    assert result.error_message is not None
    assert "DeepSeek 401" in result.error_message
    assert result.message == push.sent[0]["message"]
    assert result.message != daily.GENERIC_ERROR_MESSAGE


async def test_morning_greeting_llm_provider_normal_text_success(tmp_path: Path) -> None:
    provider = SimpleNamespace()
    push = SimpleNamespace(sent=[])

    async def chat(**kwargs):
        return SimpleNamespace(content="早上好。今天把意识从噪声里轻轻取回，先完成一件具体的小事。")

    async def execute(**kwargs):
        push.sent.append(kwargs)
        return "文本已发送"

    provider.chat = chat
    push.execute = execute

    result = await daily.run_telegram_morning_greeting_once(
        config=daily.MorningGreetingConfig(
            state_path=tmp_path / "history.json",
            model="deepseek-chat",
        ),
        llm_provider=provider,
        push_tool=push,
        send_channel="telegram",
        send_chat_id="12345",
        today=date(2026, 6, 9),
    )

    assert result.success is True
    assert result.fallback is False
    assert result.error_message is None
    assert push.sent[0]["message"] == result.message
    assert result.message != daily.GENERIC_ERROR_MESSAGE


def test_audio_mime_and_video_exclusion() -> None:
    assert daily.is_audio_message(_raw_message(filename="a.mp3", mime_type="audio/mpeg"))
    assert daily.is_audio_message(_raw_message(filename="a.m4a", mime_type="audio/mp4"))
    assert not daily.is_audio_message(_raw_message(filename="a.mp4", mime_type="audio/mp4"))
    assert not daily.is_audio_message(_raw_message(filename="a.webm", mime_type="video/webm"))


def test_audio_keywords_and_exclude_keywords() -> None:
    raw = _raw_message(filename="妈妈的录音.mp3", mime_type="audio/mpeg")
    message = _msg(text="母子 音频", raw_message=raw)

    ok, keywords = daily.audio_message_matches(
        message,
        keywords=["妈妈", "母子"],
        exclude_keywords=["未成年"],
    )

    assert ok
    assert keywords == ["妈妈", "母子"]

    blocked, _ = daily.audio_message_matches(
        _msg(text="母子 未成年", raw_message=raw),
        keywords=["母子"],
        exclude_keywords=["未成年"],
    )
    assert not blocked


def test_audio_collector_result_structure() -> None:
    raw = _raw_message(filename="母亲.wav", mime_type="audio/wav")
    message = _msg(text="母亲", raw_message=raw)

    item = daily._audio_item_from_message(message, matched=["母亲"], local_download_path="/tmp/a.wav")
    data = daily.audio_item_to_dict(item)

    assert data["chat_title"] == "study-group"
    assert data["topic_id"] == 13017
    assert data["message_id"] == 8207
    assert data["matched_keywords"] == ["母亲"]
    assert data["filename"] == "母亲.wav"
    assert data["mime_type"] == "audio/wav"
    assert data["original_message_link"]
    assert data["local_download_path"] == "/tmp/a.wav"
