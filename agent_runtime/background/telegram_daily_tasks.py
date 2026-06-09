from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agent_runtime.background.telegram_job_monitor import (
    build_telegram_client,
    build_telegram_message_link,
)

logger = logging.getLogger(__name__)

AUDIO_MIME_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/opus",
    "audio/flac",
    "audio/aac",
    "audio/mp4",
}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".opus", ".flac", ".aac", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
DEFAULT_AUDIO_KEYWORDS = ["妈妈", "儿子", "继母", "母子", "小妈", "母亲", "妈咪", "熟母"]
DEFAULT_AUDIO_EXCLUDE_KEYWORDS = [
    "未成年",
    "幼",
    "小学生",
    "初中",
    "高中",
    "偷拍",
    "强迫",
    "迷奸",
    "未同意",
]
DEFAULT_STYLE_POOL = [
    "Foucault",
    "Derrida",
    "Deleuze",
    "Barthes",
    "Blanchot",
    "Kafka",
    "Borges",
    "Calvino",
    "Beckett",
]
GENERIC_ERROR_MESSAGE = "处理消息时出错，请稍后再试。"
GENERIC_AGENT_FAILURE_TEXT = GENERIC_ERROR_MESSAGE


def is_generic_error_message(text: str | None) -> bool:
    if not text:
        return False
    return str(text).strip() == GENERIC_ERROR_MESSAGE


def build_generic_error_message_error() -> str:
    return f"Agent returned generic error message: {GENERIC_ERROR_MESSAGE}"


@dataclass(frozen=True)
class TelegramTarget:
    chat_id: int | str
    chat_title: str = ""
    topic_id: int | None = None
    expected_topic_title: str = ""


@dataclass(frozen=True)
class TelegramGroupMessage:
    chat_id: int | str
    chat_title: str
    topic_id: int | None
    message_id: int
    date: datetime
    sender: str
    text: str
    original_message_link: str
    raw_message: object | None = field(default=None, repr=False, compare=False)
    raw_entity: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class TelegramDailySummaryConfig:
    summary_targets: list[TelegramTarget] = field(default_factory=list)
    lookback_hours: int = 24
    max_messages_per_target: int = 500
    include_original_links: bool = True


@dataclass(frozen=True)
class TelegramDailySummaryResult:
    read_messages: int
    included_messages: int
    sent: bool


@dataclass(frozen=True)
class MorningGreetingConfig:
    style_pool: list[str] = field(default_factory=lambda: list(DEFAULT_STYLE_POOL))
    avoid_recent_days: int = 14
    state_path: Path = Path("data/morning_greeting_history.json")
    model: str = ""
    max_tokens: int = 300


@dataclass(frozen=True)
class TelegramMorningGreetingResult:
    sent: bool
    success: bool
    fallback: bool
    generation_failed: bool
    message: str
    style: str
    error_message: str | None


@dataclass(frozen=True)
class TelegramAudioCollectorConfig:
    audio_targets: list[TelegramTarget] = field(default_factory=list)
    lookback_hours: int = 24
    max_messages_per_target: int = 500
    download_audio: bool = True
    audio_download_dir: Path = Path("data/telegram_audio_collector")
    include_original_links: bool = True
    keywords: list[str] = field(default_factory=lambda: list(DEFAULT_AUDIO_KEYWORDS))
    exclude_keywords: list[str] = field(
        default_factory=lambda: list(DEFAULT_AUDIO_EXCLUDE_KEYWORDS)
    )


@dataclass(frozen=True)
class TelegramAudioCollectorItem:
    chat_title: str
    chat_id: int | str
    topic_id: int | None
    message_id: int
    sender: str
    date: datetime
    matched_keywords: list[str]
    filename: str
    mime_type: str
    file_size: int | None
    original_message_link: str
    local_download_path: str = ""


@dataclass(frozen=True)
class TelegramAudioCollectorResult:
    scanned_messages: int
    matched_audio: int
    sent: bool
    items: list[TelegramAudioCollectorItem]


def parse_telegram_chat_id(value: int | str) -> int | str:
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def normalize_topic_id(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        topic_id = int(value)
    except (TypeError, ValueError):
        return None
    return topic_id if topic_id > 0 else None


def infer_message_topic_id(message: object) -> int | None:
    reply_to = getattr(message, "reply_to", None)
    if reply_to is None:
        return None
    for attr in ("reply_to_top_id", "reply_to_msg_id"):
        value = normalize_topic_id(getattr(reply_to, attr, None))
        if value is not None:
            return value
    return None


def target_matches_message(target: TelegramTarget, message: object) -> bool:
    if target.topic_id is None:
        return True
    return infer_message_topic_id(message) == target.topic_id


def _message_topic_title(message: object) -> str:
    for attr in ("topic_title", "top_message_topic_title"):
        value = str(getattr(message, attr, "") or "").strip()
        if value:
            return value
    reply_to = getattr(message, "reply_to", None)
    for obj in (reply_to, getattr(message, "reply_to_message", None)):
        if obj is None:
            continue
        for attr in ("topic_title", "title"):
            value = str(getattr(obj, attr, "") or "").strip()
            if value:
                return value
    return "<unknown>"


def resolve_message_topic_title(message: object, entity: object | None = None) -> str:
    title = _message_topic_title(message)
    if title != "<unknown>":
        return title
    return "<unknown>"


def _normalize_message_date(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


async def _message_sender_name(message: object) -> str:
    get_sender = getattr(message, "get_sender", None)
    sender = None
    if callable(get_sender):
        try:
            sender = await get_sender()
        except Exception:
            sender = None
    if sender is None:
        sender = getattr(message, "sender", None)

    for attr in ("username", "first_name", "title"):
        value = str(getattr(sender, attr, "") or "").strip()
        if value:
            return value
    sender_id = getattr(message, "sender_id", None)
    return str(sender_id) if sender_id else "unknown"


def _message_text(message: object) -> str:
    return str(getattr(message, "message", "") or "").strip()


def _target_title(target: TelegramTarget, entity: object) -> str:
    return (
        target.chat_title
        or getattr(entity, "title", None)
        or getattr(entity, "username", None)
        or str(target.chat_id)
    )


async def read_telegram_target_messages(
    *,
    targets: list[TelegramTarget],
    lookback_hours: int,
    max_messages_per_target: int,
    include_empty_text: bool = False,
    now: datetime | None = None,
) -> list[TelegramGroupMessage]:
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cutoff = now_utc - timedelta(hours=max(1, lookback_hours))
    client, phone = build_telegram_client()
    messages: list[TelegramGroupMessage] = []
    try:
        await client.start(phone=phone)
        for target in targets:
            entity = await client.get_entity(target.chat_id)
            chat_title = str(_target_title(target, entity))
            async for message in client.iter_messages(
                entity,
                limit=max(1, max_messages_per_target),
            ):
                date_value = _normalize_message_date(getattr(message, "date", None))
                if date_value < cutoff:
                    break
                if not target_matches_message(target, message):
                    continue
                text = _message_text(message)
                if not text and not include_empty_text:
                    continue
                messages.append(
                    TelegramGroupMessage(
                        chat_id=target.chat_id,
                        chat_title=chat_title,
                        topic_id=infer_message_topic_id(message),
                        message_id=int(getattr(message, "id", 0) or 0),
                        date=date_value,
                        sender=await _message_sender_name(message),
                        text=text,
                        original_message_link=build_telegram_message_link(
                            entity,
                            message,
                        ),
                        raw_message=message,
                        raw_entity=entity,
                    )
                )
    finally:
        await client.disconnect()

    return sorted(messages, key=lambda item: (str(item.chat_id), item.topic_id or 0, item.date))


def filter_messages_by_time_window(
    messages: list[TelegramGroupMessage],
    *,
    now: datetime,
    window_hours: int,
) -> list[TelegramGroupMessage]:
    cutoff = now.astimezone(timezone.utc) - timedelta(hours=max(1, window_hours))
    return [
        message
        for message in messages
        if message.date.astimezone(timezone.utc) >= cutoff
    ]


def build_daily_summary_prompt(
    messages: list[TelegramGroupMessage],
    *,
    lookback_hours: int = 24,
    include_original_links: bool = True,
    topic_title: str | None = None,
) -> str:
    grouped: dict[tuple[str, int | str, int | None], list[TelegramGroupMessage]] = {}
    for message in messages:
        key = (message.chat_title, message.chat_id, message.topic_id)
        grouped.setdefault(key, []).append(message)

    title_hint = topic_title or (messages[0].topic_id and f"topic_id={messages[0].topic_id}") or ""
    lines = [
        f"请总结以下配置的 Telegram 群聊/话题过去 {lookback_hours} 小时的消息。",
        "",
        "输出要求：",
        f"- 主题标识：{title_hint}。",
        f"- 今日消息总量：{len(messages)}。",
        "- 按群聊和 topic 分组说明每组消息数量。",
        "- 提取重要招聘/求职信息、重要讨论、需要我注意的风险。",
        "- 不要总结未配置的群，不要把无关闲聊写得太长。",
        "- 只基于给定消息，不要编造。",
        "- 使用中文，整体控制在 1200 字以内。",
    ]
    if include_original_links:
        lines.append("- 重要结论后附 original_message_link。")
    lines.extend(["", "分组消息："])

    total_chars = 0
    for (chat_title, chat_id, topic_id), items in grouped.items():
        lines.append("")
        lines.append(
            f"## {chat_title} chat_id={chat_id} topic_id={topic_id or 'none'} count={len(items)}"
        )
        for message in items:
            text = message.text.replace("\r\n", "\n").replace("\r", "\n")
            if len(text) > 800:
                text = text[:797].rstrip() + "..."
            link = (
                f"\noriginal_message_link: {message.original_message_link}"
                if include_original_links
                else ""
            )
            item = (
                f"[{message.date.isoformat()}] {message.sender} "
                f"message_id={message.message_id}: {text}{link}"
            )
            if total_chars + len(item) > 35_000:
                lines.append("[后续消息因长度限制省略]")
                return "\n".join(lines).strip()
            lines.append(item)
            total_chars += len(item)
    return "\n".join(lines).strip()


def choose_style_for_date(day: date, style_pool: list[str]) -> str:
    pool = style_pool or DEFAULT_STYLE_POOL
    rng = random.Random(day.isoformat())
    return pool[rng.randrange(len(pool))]


def build_morning_greeting_prompt(style: str) -> str:
    return (
        "请用中文生成一条适合早上发送给用户的早安鼓励消息。\n"
        f"风格参考：{style}。\n"
        "要求：80 到 180 字；原创短句；偏冷静、清醒、有文学性；"
        "不要廉价鸡汤；不要长篇引用；不要提作者名字；不要模仿得像翻译腔；"
        "可以带一点哲思、断裂感、凝视感，但要能让人开始今天。\n"
        "只输出正文，不要标题，不要项目符号，不要解释。"
    )


def local_morning_greeting_fallback(style: str) -> str:
    fragments = [
        "早上好。今天不必把自己整理成一个完整的答案；先把注意力放回手边，让一件具体的小事替你打开今天。",
        "早上好。世界未必温柔，但你仍能把散乱收拢成步骤；先动手，轮廓会在行动里慢慢出现。",
        "早上好。不要急着解释自己，也不必急着完成自己；先让清醒落在第一步，剩下的留给时间。",
        "早上好。今天可以只做一件事：把混乱拆开，把声音放低，把真正重要的那一点留在眼前。",
    ]
    rng = random.Random(f"{style}:fallback")
    return fragments[rng.randrange(len(fragments))]


def is_generic_agent_failure_text(text: str) -> bool:
    return is_generic_error_message(text)


def _agent_result_error(result: object) -> str | None:
    for attr in ("error_message", "error", "exception"):
        value = getattr(result, attr, None)
        if value:
            return f"{attr}: {value}"
    if isinstance(result, dict):
        for key in ("error_message", "error", "exception"):
            value = result.get(key)
            if value:
                return f"{key}: {value}"
    return None


def _agent_result_text(result: object) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in ("final_text", "content", "message", "text", "response"):
            value = result.get(key)
            if value:
                return str(value).strip()
    for attr in ("final_text", "content", "message", "text", "response"):
        value = getattr(result, attr, None)
        if value:
            return str(value).strip()
    return str(result or "").strip()


def summarize_llm_error(exc: Exception) -> str:
    text = str(exc) or repr(exc)
    lowered = text.lower()
    if "401" in lowered or "unauthorized" in lowered or "invalid api key" in lowered:
        detail = "DeepSeek 401 / API key 无效或未授权"
    elif "model" in lowered and ("not found" in lowered or "does not exist" in lowered):
        detail = "model not found / 模型名可能不受当前 provider 支持"
    elif "timeout" in lowered or "timed out" in lowered:
        detail = "timeout / LLM 请求超时"
    else:
        detail = f"unknown error / {type(exc).__name__}"
    return f"{detail}: {text[:300]}"


async def _generate_greeting_with_provider(
    *,
    llm_provider: Any,
    model: str,
    day: date,
    style: str,
    max_tokens: int,
) -> str:
    response = await llm_provider.chat(
        messages=[
            {
                "role": "user",
                "content": build_morning_greeting_prompt(style),
            }
        ],
        tools=[],
        model=model,
        max_tokens=max_tokens,
        tool_choice="none",
        disable_thinking=True,
    )
    return str(getattr(response, "content", "") or "").strip()


async def resolve_target_topic_title(target: TelegramTarget) -> str:
    return await _resolve_target_topic_title(target)


async def _resolve_target_topic_title(target: TelegramTarget) -> str:
    client, phone = build_telegram_client()
    try:
        await client.start(phone=phone)
        entity = await client.get_entity(target.chat_id)
        forum_titles = await _load_forum_topic_titles(client, entity)
        if target.topic_id in forum_titles:
            return forum_titles[target.topic_id]
        root_candidates = [target.topic_id] if target.topic_id is not None else []
        if root_candidates:
            try:
                messages = await client.get_messages(entity, ids=root_candidates)
            except Exception:
                messages = []
            if not isinstance(messages, list):
                messages = [messages]
            for message in messages:
                if message is None:
                    continue
                title = _message_topic_title(message)
                if title != "<unknown>":
                    return title
        return "<unknown>"
    finally:
        await client.disconnect()


async def _resolve_target_topic_title_for_config(target: TelegramTarget) -> str:
    client, phone = build_telegram_client()
    try:
        await client.start(phone=phone)
        entity = await client.get_entity(target.chat_id)
        forum_titles = await _load_forum_topic_titles(client, entity)
        if target.topic_id in forum_titles:
            return forum_titles[target.topic_id]
        if target.topic_id is None:
            return "<unknown>"
        try:
            messages = await client.get_messages(entity, ids=[target.topic_id])
        except Exception:
            messages = []
        if not isinstance(messages, list):
            messages = [messages]
        for message in messages:
            if message is None:
                continue
            title = _message_topic_title(message)
            if title != "<unknown>":
                return title
        return "<unknown>"
    finally:
        await client.disconnect()


def _load_greeting_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    history: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        if _history_is_dirty(normalized):
            day = _history_item_day(normalized)
            style = str(normalized.get("style") or choose_style_for_date(day, DEFAULT_STYLE_POOL))
            normalized["message"] = build_morning_greeting_fallback(style, today=day)
            normalized["fallback"] = True
            normalized["success"] = True
            normalized["error_message"] = normalized.get("error_message") or build_generic_error_message_error()
            normalized["generation_failed"] = True
            if normalized.get("message") == GENERIC_ERROR_MESSAGE:
                normalized["message"] = build_morning_greeting_fallback(style, today=day)
        history.append(normalized)
    return history


def _save_greeting_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _history_item_day(item: dict[str, Any]) -> date:
    raw = str(item.get("date") or "").strip()
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return datetime.now(timezone.utc).date()


def _history_is_dirty(item: dict[str, Any]) -> bool:
    message = str(item.get("message") or "").strip()
    success = bool(item.get("success", False))
    return is_generic_error_message(message) or (success and not message)


def _normalize_for_similarity(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    return {normalized[i : i + 2] for i in range(max(0, len(normalized) - 1))}


def greeting_too_similar(candidate: str, recent: list[str], threshold: float = 0.45) -> bool:
    candidate_set = _normalize_for_similarity(candidate)
    if not candidate_set:
        return False
    for item in recent:
        item_set = _normalize_for_similarity(item)
        if not item_set:
            continue
        score = len(candidate_set & item_set) / len(candidate_set | item_set)
        if score >= threshold:
            return True
    return False


async def generate_unique_greeting(
    *,
    llm_provider: Any | None = None,
    config: MorningGreetingConfig,
    send_channel: str | None = None,
    send_chat_id: str | None = None,
    today: date | None = None,
) -> tuple[str, str, bool, bool, str | None, bool]:
    day = today or datetime.now(timezone.utc).date()
    history = _load_greeting_history(config.state_path)
    existing_today = next(
        (item for item in history if item.get("date") == day.isoformat()),
        None,
    )
    style = choose_style_for_date(day, config.style_pool)
    if existing_today and existing_today.get("message") and existing_today.get("style"):
        message = str(existing_today["message"])
        style = str(existing_today["style"])
        fallback = bool(existing_today.get("fallback", False))
        error_message = str(existing_today.get("error_message") or "").strip() or None
        generation_failed = bool(existing_today.get("generation_failed", False))
        if not message or is_generic_error_message(message):
            logger.error(
                "[telegram_morning_greeting] existing greeting is generic failure text; date=%s style=%s",
                day.isoformat(),
                style,
            )
            message = build_morning_greeting_fallback(style)
            fallback = True
            generation_failed = True
            error_message = error_message or build_generic_error_message_error()
        return message, style, True, fallback, error_message, generation_failed

    content = ""
    error_message: str | None = None
    generation_failed = False
    try:
        if llm_provider is None or not config.model:
            raise RuntimeError("morning_greeting requires llm_provider and model")
        content = await _generate_greeting_with_provider(
            llm_provider=llm_provider,
            model=config.model,
            day=day,
            style=style,
            max_tokens=config.max_tokens,
        )
    except Exception as exc:
        logger.exception("[telegram_morning_greeting] LLM generation failed")
        error_message = summarize_llm_error(exc)
        generation_failed = True
        content = ""

    content = str(content or "").strip()
    if not content or is_generic_error_message(content):
        generation_failed = True
        if not error_message:
            error_message = build_generic_error_message_error() if is_generic_error_message(content) else "LLM 生成了空内容"
        fallback_text = build_morning_greeting_fallback(style)
        history.append(
            {
                "date": day.isoformat(),
                "style": style,
                "message": fallback_text,
                "success": True,
                "fallback": True,
                "error_message": error_message,
                "generation_failed": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        keep_after = day - timedelta(days=max(1, config.avoid_recent_days) + 7)
        history = [
            item
            for item in history
            if str(item.get("date", "")) >= keep_after.isoformat()
        ]
        _save_greeting_history(config.state_path, history)
        return fallback_text, style, True, True, error_message, True

    history.append(
        {
            "date": day.isoformat(),
            "style": style,
            "message": content,
            "success": True,
            "fallback": False,
            "error_message": None,
            "generation_failed": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    keep_after = day - timedelta(days=max(1, config.avoid_recent_days) + 7)
    history = [
        item
        for item in history
        if str(item.get("date", "")) >= keep_after.isoformat()
    ]
    _save_greeting_history(config.state_path, history)
    return content, style, True, False, None, False


def _preview_message(text: str, limit: int = 80) -> str:
    clean = str(text or "").replace("\n", " ").strip()
    return clean[:limit] + "..." if len(clean) > limit else clean


def build_morning_greeting_fallback(style: str, today: date | None = None) -> str:
    day = today or datetime.now(timezone.utc).date()
    fragments = [
        "早上好。今天先保持清醒，不急着把一切解释完整；把该做的事往前推进一点，就够了。",
        "早上好。今天先把注意力放回手边，完成一件具体的小事，比空想完整更重要。",
        "早上好。别急着给自己定论，先让今天的第一步落地，剩下的轮廓会慢慢出现。",
        "早上好。保持一点冷静，把混乱拆开，先处理眼前这一步，再看下一步。",
    ]
    rng = random.Random(f"{day.isoformat()}:{style}:fallback")
    return fragments[rng.randrange(len(fragments))]


local_morning_greeting_fallback = build_morning_greeting_fallback


def _document(message: object) -> object | None:
    media = getattr(message, "media", None)
    document = getattr(message, "document", None)
    if document is not None:
        return document
    return getattr(media, "document", None)


def media_filename(message: object) -> str:
    document = _document(message)
    attrs = list(getattr(document, "attributes", []) or [])
    for attr in attrs:
        filename = str(getattr(attr, "file_name", "") or "").strip()
        if filename:
            return filename
    file_obj = getattr(message, "file", None)
    filename = str(getattr(file_obj, "name", "") or "").strip()
    return filename


def media_mime_type(message: object) -> str:
    document = _document(message)
    mime = str(getattr(document, "mime_type", "") or "").strip().lower()
    if mime:
        return mime
    file_obj = getattr(message, "file", None)
    return str(getattr(file_obj, "mime_type", "") or "").strip().lower()


def media_file_size(message: object) -> int | None:
    document = _document(message)
    size = getattr(document, "size", None)
    if isinstance(size, int):
        return size
    file_obj = getattr(message, "file", None)
    size = getattr(file_obj, "size", None)
    return size if isinstance(size, int) else None


def is_audio_message(message: object) -> bool:
    filename = media_filename(message)
    suffix = Path(filename).suffix.lower()
    mime_type = media_mime_type(message)
    if suffix in VIDEO_EXTENSIONS or mime_type.startswith("video/"):
        return False
    if suffix in AUDIO_EXTENSIONS:
        return True
    return mime_type in AUDIO_MIME_TYPES


def _keyword_source(message: TelegramGroupMessage) -> str:
    raw = message.raw_message
    filename = media_filename(raw) if raw is not None else ""
    mime_type = media_mime_type(raw) if raw is not None else ""
    return "\n".join([message.text, filename, mime_type])


def matched_keywords(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in keywords if keyword.lower() in lowered]


def audio_message_matches(
    message: TelegramGroupMessage,
    *,
    keywords: list[str],
    exclude_keywords: list[str],
) -> tuple[bool, list[str]]:
    raw = message.raw_message
    if raw is None or not is_audio_message(raw):
        return False, []
    source = _keyword_source(message)
    includes = matched_keywords(source, keywords)
    excludes = matched_keywords(source, exclude_keywords)
    return bool(includes and not excludes), includes


def _audio_item_from_message(
    message: TelegramGroupMessage,
    *,
    matched: list[str],
    local_download_path: str = "",
) -> TelegramAudioCollectorItem:
    raw = message.raw_message
    return TelegramAudioCollectorItem(
        chat_title=message.chat_title,
        chat_id=message.chat_id,
        topic_id=message.topic_id,
        message_id=message.message_id,
        sender=message.sender,
        date=message.date,
        matched_keywords=matched,
        filename=media_filename(raw) if raw is not None else "",
        mime_type=media_mime_type(raw) if raw is not None else "",
        file_size=media_file_size(raw) if raw is not None else None,
        original_message_link=message.original_message_link,
        local_download_path=local_download_path,
    )


def format_audio_collector_report(items: list[TelegramAudioCollectorItem]) -> str:
    if not items:
        return "Telegram Audio Collector：过去 24 小时内没有找到符合条件的音频文件。"
    lines = [
        "Telegram Audio Collector",
        f"发现 {len(items)} 条符合条件的音频：",
        "",
    ]
    for index, item in enumerate(items, start=1):
        lines.extend(
            [
                f"{index}. {item.chat_title} topic_id={item.topic_id or 'none'}",
                f"message_id: {item.message_id}",
                f"sender: {item.sender}",
                f"date: {item.date.isoformat()}",
                f"matched_keywords: {', '.join(item.matched_keywords)}",
                f"filename: {item.filename}",
                f"mime_type: {item.mime_type}",
                f"file_size: {item.file_size if item.file_size is not None else 'unknown'}",
                f"original_message_link: {item.original_message_link}",
            ]
        )
        if item.local_download_path:
            lines.append(f"local_download_path: {item.local_download_path}")
        lines.append("")
    return "\n".join(lines).strip()


async def run_telegram_daily_summary_once(
    *,
    config: TelegramDailySummaryConfig,
    agent_loop: Any,
    push_tool: Any,
    send_channel: str,
    send_chat_id: str,
) -> TelegramDailySummaryResult:
    if not config.summary_targets:
        logger.warning("[telegram_daily_summary] no summary_targets configured")
        return TelegramDailySummaryResult(0, 0, False)

    validated_targets: list[TelegramTarget] = []
    topic_title_hint: str | None = None
    for target in config.summary_targets:
        logger.info(
            "[telegram_daily_summary] target chat_id=%s topic_id=%s expected_topic_title=%s",
            target.chat_id,
            target.topic_id,
            target.expected_topic_title or "",
        )
        if target.topic_id is None:
            logger.warning(
                "[telegram_daily_summary] skip target without topic_id chat_id=%s",
                target.chat_id,
            )
            continue
        if target.expected_topic_title:
            try:
                resolved_title = await _resolve_target_topic_title_for_config(target)
            except Exception:
                logger.warning(
                    "[telegram_daily_summary] unable to resolve topic title; using topic_id only"
                )
                resolved_title = "<unknown>"
            logger.info("[telegram_daily_summary] resolved topic_title=%s", resolved_title)
            matched = resolved_title == target.expected_topic_title
            logger.info("[telegram_daily_summary] topic_title_matched=%s", matched)
            if not matched:
                error_message = (
                    f"配置错误：topic_id={target.topic_id} 实际对应“{resolved_title}”，"
                    f"不是“{target.expected_topic_title}”。请重新运行 discover 脚本获取正确 topic_id。"
                )
                logger.error("[telegram_daily_summary] %s", error_message)
                result = await push_tool.execute(
                    channel=send_channel,
                    chat_id=send_chat_id,
                    message=error_message,
                )
                sent = "发送失败" not in str(result)
                return TelegramDailySummaryResult(0, 0, sent)
            if resolved_title != "<unknown>":
                topic_title_hint = resolved_title
            else:
                logger.warning(
                    "[telegram_daily_summary] unable to resolve topic title; using topic_id only"
                )
        else:
            logger.warning(
                "[telegram_daily_summary] unable to resolve topic title; using topic_id only"
            )
        validated_targets.append(target)

    if not validated_targets:
        return TelegramDailySummaryResult(0, 0, False)

    messages = await read_telegram_target_messages(
        targets=validated_targets,
        lookback_hours=config.lookback_hours,
        max_messages_per_target=config.max_messages_per_target,
    )
    if messages:
        prompt = build_daily_summary_prompt(
            messages,
            lookback_hours=config.lookback_hours,
            include_original_links=config.include_original_links,
            topic_title=topic_title_hint,
        )
        content = await agent_loop.process_direct(
            content=prompt,
            channel=send_channel,
            chat_id=send_chat_id,
            session_key="telegram_daily_summary",
            omit_user_turn=True,
            skip_post_memory=True,
            disabled_tools=["message_push"],
        )
    else:
        content = f"过去 {config.lookback_hours} 小时内，配置的 Telegram 群聊/话题没有可总结的新文本消息。"

    sent = False
    if content:
        result = await push_tool.execute(
            channel=send_channel,
            chat_id=send_chat_id,
            message=content,
        )
        sent = "发送失败" not in str(result)
    logger.info(
        "[telegram_daily_summary] targets=%s messages=%s sent=%s",
        len(validated_targets),
        len(messages),
        sent,
    )
    return TelegramDailySummaryResult(
        read_messages=len(messages),
        included_messages=len(messages),
        sent=sent,
    )


async def run_telegram_morning_greeting_once(
    *,
    config: MorningGreetingConfig,
    agent_loop: Any | None = None,
    llm_provider: Any | None = None,
    push_tool: Any,
    send_channel: str,
    send_chat_id: str,
    today: date | None = None,
) -> TelegramMorningGreetingResult:
    del agent_loop
    logger.info(
        "[telegram_morning_greeting] entered chat_id=%s history_enabled=%s",
        send_chat_id,
        bool(config.state_path),
    )
    content, style, success, fallback, error_message, generation_failed = await generate_unique_greeting(
        llm_provider=llm_provider,
        config=config,
        send_channel=send_channel,
        send_chat_id=send_chat_id,
        today=today,
    )
    final_message = content
    if not final_message or is_generic_error_message(final_message):
        logger.error("[telegram_morning_greeting] generic failure text blocked; using fallback")
        final_message = build_morning_greeting_fallback(style)
        fallback = True
        generation_failed = True
        success = True
        if not error_message:
            error_message = build_generic_error_message_error()
    source = "fallback" if fallback else "llm"
    sent = False
    try:
        result = await push_tool.execute(
            channel=send_channel,
            chat_id=send_chat_id,
            message=final_message,
        )
        sent = "发送失败" not in str(result)
    except Exception as exc:
        logger.exception("[telegram_morning_greeting] push failed")
        sent = False
        success = False
        generation_failed = True
        error_message = error_message or summarize_llm_error(exc)
    logger.info(
        "[telegram_morning_greeting] source=%s sent=%s success=%s fallback=%s generation_failed=%s error=%s message_preview=%s",
        source,
        sent,
        success,
        fallback,
        generation_failed,
        error_message,
        _preview_message(final_message, 80),
    )
    return TelegramMorningGreetingResult(
        sent=sent,
        success=success,
        fallback=fallback,
        generation_failed=generation_failed,
        message=final_message,
        style=style,
        error_message=error_message,
    )


async def run_telegram_audio_collector_once(
    *,
    config: TelegramAudioCollectorConfig,
    push_tool: Any,
    send_channel: str,
    send_chat_id: str,
    now: datetime | None = None,
) -> TelegramAudioCollectorResult:
    if not config.audio_targets:
        logger.warning("[telegram_audio_collector] no audio_targets configured")
        return TelegramAudioCollectorResult(0, 0, False, [])
    messages = await read_telegram_target_messages(
        targets=config.audio_targets,
        lookback_hours=config.lookback_hours,
        max_messages_per_target=config.max_messages_per_target,
        include_empty_text=True,
        now=now,
    )
    items: list[TelegramAudioCollectorItem] = []
    run_day = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date()
    download_dir = config.audio_download_dir / run_day.isoformat()
    for message in messages:
        ok, keywords = audio_message_matches(
            message,
            keywords=config.keywords,
            exclude_keywords=config.exclude_keywords,
        )
        if not ok:
            continue
        local_path = ""
        if config.download_audio and message.raw_message is not None:
            download_dir.mkdir(parents=True, exist_ok=True)
            filename = Path(media_filename(message.raw_message) or f"{message.message_id}.audio").name
            target_path = download_dir / f"{message.chat_id}_{message.message_id}_{filename}"
            try:
                downloaded = await message.raw_message.download_media(file=str(target_path))
                local_path = str(downloaded or target_path)
            except Exception as exc:
                logger.warning(
                    "[telegram_audio_collector] download failed message_id=%s err=%s",
                    message.message_id,
                    exc,
                )
        items.append(
            _audio_item_from_message(
                message,
                matched=keywords,
                local_download_path=local_path,
            )
        )

    result = await push_tool.execute(
        channel=send_channel,
        chat_id=send_chat_id,
        message=format_audio_collector_report(items),
    )
    sent = "发送失败" not in str(result)
    logger.info(
        "[telegram_audio_collector] scanned=%s matched=%s sent=%s",
        len(messages),
        len(items),
        sent,
    )
    return TelegramAudioCollectorResult(
        scanned_messages=len(messages),
        matched_audio=len(items),
        sent=sent,
        items=items,
    )


def audio_item_to_dict(item: TelegramAudioCollectorItem) -> dict[str, Any]:
    data = asdict(item)
    data["date"] = item.date.isoformat()
    return data
