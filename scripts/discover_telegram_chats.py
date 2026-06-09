from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_runtime.background.telegram_daily_tasks import (  # noqa: E402
    infer_message_topic_id,
    parse_telegram_chat_id,
    resolve_message_topic_title,
)
from agent_runtime.background.telegram_job_monitor import build_telegram_client  # noqa: E402
from telethon.utils import get_peer_id  # noqa: E402


@dataclass
class TopicSample:
    topic_id: int | None
    topic_title: str = "<unknown>"
    count: int = 0
    latest: datetime | None = None
    example: str = ""
    top_message_id: int | None = None
    root_message_id: int | None = None
    recent_texts: list[str] | None = None


def _entity_title(entity: object) -> str:
    return str(
        getattr(entity, "title", None)
        or getattr(entity, "username", None)
        or getattr(entity, "first_name", None)
        or getattr(entity, "id", "")
    )


def _bool_attr(entity: object, name: str) -> bool:
    return bool(getattr(entity, name, False))


def _chat_id(entity: object) -> int | str:
    try:
        return get_peer_id(entity)
    except Exception:
        return getattr(entity, "id", "")


def _safe_text(value: Any, limit: int = 80) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text[:limit]


def _reply_root_message_id(reply_to: object | None, topic_id: int | None) -> int | None:
    if reply_to is None:
        return topic_id
    for attr in ("reply_to_top_id", "reply_to_msg_id"):
        value = getattr(reply_to, attr, None)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return topic_id


def _topic_title_from_message(message: object | None) -> str:
    if message is None:
        return "<unknown>"
    action = getattr(message, "action", None)
    for obj in (action, message):
        for attr in ("title", "topic_title"):
            value = str(getattr(obj, attr, "") or "").strip()
            if value:
                return value
    text = _safe_text(getattr(message, "message", ""), 60)
    return text or "<unknown>"


def _topic_title_from_entity(entity: object | None) -> str:
    if entity is None:
        return "<unknown>"
    for attr in ("title", "username", "first_name"):
        value = str(getattr(entity, attr, "") or "").strip()
        if value:
            return value
    return "<unknown>"


def _topic_matches_keyword(sample: TopicSample, keyword: str) -> bool:
    if not keyword:
        return True
    haystack = " ".join(
        [
            sample.topic_title,
            sample.example,
            *(sample.recent_texts or []),
        ]
    ).lower()
    return keyword.lower() in haystack


def filter_and_sort_topics(
    topics: dict[int | None, TopicSample],
    *,
    min_count: int = 1,
    topic_keyword: str = "",
    show_all_topics: bool = False,
) -> list[TopicSample]:
    samples = [
        sample
        for sample in topics.values()
        if sample.topic_id is not None
        and sample.count >= min_count
        and (show_all_topics or _topic_matches_keyword(sample, topic_keyword))
    ]
    return sorted(
        samples,
        key=lambda sample: (
            -sample.count,
            -(sample.latest.timestamp() if sample.latest else 0),
            sample.topic_id or 0,
        ),
    )


def skipped_topic_debug_lines(
    topics: dict[int | None, TopicSample],
    *,
    min_count: int = 1,
    topic_keyword: str = "",
    show_all_topics: bool = False,
) -> list[str]:
    if show_all_topics or not topic_keyword:
        return []
    lines: list[str] = []
    for sample in filter_and_sort_topics(
        topics,
        min_count=min_count,
        topic_keyword="",
        show_all_topics=True,
    ):
        if not _topic_matches_keyword(sample, topic_keyword):
            lines.append(
                f"skipped topic_id = {sample.topic_id} because keyword not matched"
            )
    return lines


def add_message_to_topic_groups(
    topics: dict[int | None, TopicSample],
    *,
    topic_id: int | None,
    reply_to: object | None,
    text: str,
    msg_date: object,
) -> None:
    if topic_id is None:
        return
    sample = topics.setdefault(topic_id, TopicSample(topic_id=topic_id))
    sample.count += 1
    root_id = _reply_root_message_id(reply_to, topic_id)
    if root_id is not None:
        sample.root_message_id = root_id
        sample.top_message_id = root_id
    if isinstance(msg_date, datetime) and (
        sample.latest is None or msg_date > sample.latest
    ):
        sample.latest = msg_date
    if not sample.example and text:
        sample.example = text
    if text:
        recent_texts = sample.recent_texts or []
        if len(recent_texts) < 8:
            recent_texts.append(text)
        sample.recent_texts = recent_texts


async def _sender_name(message: object) -> str:
    sender = None
    get_sender = getattr(message, "get_sender", None)
    if callable(get_sender):
        try:
            sender = await get_sender()
        except Exception:
            sender = None
    for attr in ("username", "first_name", "title"):
        value = str(getattr(sender, attr, "") or "").strip()
        if value:
            return value
    sender_id = getattr(message, "sender_id", None)
    return str(sender_id or "")


def _print_dialog(entity: object, unread_count: int) -> None:
    print(f"title: {_entity_title(entity)}")
    print(f"username: {getattr(entity, 'username', None) or ''}")
    print(f"chat_id: {_chat_id(entity)}")
    print(f"entity_id: {getattr(entity, 'id', '')}")
    access_hash = getattr(entity, "access_hash", None)
    if access_hash:
        print(f"access_hash: {access_hash}")
    print(f"is_group: {_bool_attr(entity, 'megagroup') or _bool_attr(entity, 'gigagroup')}")
    print(f"is_channel: {_bool_attr(entity, 'broadcast') or entity.__class__.__name__.lower().endswith('channel')}")
    print(f"is_forum: {_bool_attr(entity, 'forum')}")
    print(f"unread_count: {unread_count}")
    print("")


def _config_target_line(title: str, chat_id: int | str, topic_id: int | None) -> str:
    if topic_id is None:
        return f'  {{ chat_title = "{title}", chat_id = {chat_id} }},'
    return f'  {{ chat_title = "{title}", chat_id = {chat_id}, topic_id = {topic_id} }},'


def render_topic_groups(samples: list[TopicSample]) -> str:
    lines = ["Topic groups:"]
    for sample in samples:
        lines.extend(
            [
                f"topic_id: {sample.topic_id}",
                f"topic_title: {sample.topic_title}",
                f"recent_message_count: {sample.count}",
                f"latest_message_time: {sample.latest or ''}",
                f"example_message: {sample.example}",
                f"top_message_id: {sample.top_message_id or ''}",
                f"root_message_id: {sample.root_message_id or ''}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def render_config_example(
    *,
    title: str,
    chat_id: int | str,
    samples: list[TopicSample],
    include_whole_chat: bool = False,
    target_topic_title: str = "",
) -> str:
    config_samples = [sample for sample in samples if sample.topic_id is not None]
    if target_topic_title:
        config_samples = [
            sample
            for sample in config_samples
            if sample.topic_title == target_topic_title
        ]
    if include_whole_chat:
        config_samples = [
            TopicSample(topic_id=None, topic_title="<whole chat>", count=0),
            *config_samples,
        ]

    lines = [
        "config.toml example:",
        "# 带 topic_id：只监听该话题。",
        "# 不带 topic_id：监听整个群，可能很慢且很杂；默认不输出，除非传 --include-whole-chat。",
        "[telegram_targets]",
        "summary_targets = [",
    ]
    for sample in config_samples:
        lines.append(_config_target_line(title, chat_id, sample.topic_id))
    lines.extend(["]", "", "audio_targets = ["])
    for sample in config_samples[:3]:
        lines.append(_config_target_line(title, chat_id, sample.topic_id))
    lines.append("]")
    return "\n".join(lines)


async def _resolve_chat(client, chat: str):
    try:
        return await client.get_entity(parse_telegram_chat_id(chat))
    except Exception:
        dialogs = await client.get_dialogs()
        for dialog in dialogs:
            entity = dialog.entity
            title = _entity_title(entity)
            username = str(getattr(entity, "username", "") or "")
            if chat == title or chat == username or chat.lower() in title.lower():
                return entity
        raise


async def list_dialogs(filter_text: str) -> None:
    client, phone = build_telegram_client()
    try:
        await client.start(phone=phone)
        dialogs = await client.get_dialogs()
        for dialog in dialogs:
            entity = dialog.entity
            title = _entity_title(entity)
            username = str(getattr(entity, "username", "") or "")
            if filter_text and filter_text.lower() not in f"{title} {username}".lower():
                continue
            _print_dialog(entity, int(getattr(dialog, "unread_count", 0) or 0))
    finally:
        await client.disconnect()


async def _load_forum_topic_titles(client, entity: object) -> dict[int, str]:
    titles: dict[int, str] = {}
    try:
        from telethon.tl.functions.channels import GetForumTopicsRequest

        result = await client(
            GetForumTopicsRequest(
                channel=entity,
                offset_date=None,
                offset_id=0,
                offset_topic=0,
                limit=100,
                q="",
            )
        )
        for topic in getattr(result, "topics", []) or []:
            topic_id = getattr(topic, "id", None)
            title = str(getattr(topic, "title", "") or "").strip()
            if topic_id and title:
                titles[int(topic_id)] = title
    except Exception:
        return titles
    return titles


async def _load_topic_title_by_root_message(client, entity: object, topic_id: int) -> str:
    try:
        root = await client.get_messages(entity, ids=[topic_id])
    except Exception:
        return "<unknown>"
    if isinstance(root, list):
        root = root[0] if root else None
    return _topic_title_from_message(root)


async def _resolve_topic_titles(client, entity: object, topics: dict[int | None, TopicSample]) -> None:
    forum_titles = await _load_forum_topic_titles(client, entity)
    for topic_id, title in forum_titles.items():
        if topic_id in topics:
            topics[topic_id].topic_title = title

    for sample in topics.values():
        if sample.topic_id is None:
            continue
        if sample.topic_title != "<unknown>":
            continue
        root_title = await _load_topic_title_by_root_message(client, entity, sample.root_message_id or sample.top_message_id or sample.topic_id)
        if root_title != "<unknown>":
            sample.topic_title = root_title
        elif sample.recent_texts:
            sample.topic_title = sample.recent_texts[0]


async def scan_chat(
    chat: str,
    limit: int,
    *,
    min_count: int = 1,
    include_whole_chat: bool = False,
    topic_keyword: str = "",
    topic_title: str = "",
    output_config_only: bool = False,
    show_all_topics: bool = False,
) -> None:
    client, phone = build_telegram_client()
    topics: dict[int | None, TopicSample] = {}
    try:
        await client.start(phone=phone)
        entity = await _resolve_chat(client, chat)
        title = _entity_title(entity)
        if not output_config_only:
            print(f"Scanning chat: {title}")
            print(f"chat_id: {_chat_id(entity)}")
            print("")
        async for message in client.iter_messages(entity, limit=max(1, limit)):
            reply_to = getattr(message, "reply_to", None)
            topic_id = infer_message_topic_id(message)
            text = _safe_text(getattr(message, "message", ""))
            sender = await _sender_name(message)
            if not output_config_only:
                print(f"message.id: {getattr(message, 'id', '')}")
                print(f"message.date: {getattr(message, 'date', '')}")
                print(f"sender: {sender}")
                print(f"text: {text}")
                print(f"reply_to: {reply_to!r}")
                print(f"reply_to.reply_to_top_id: {getattr(reply_to, 'reply_to_top_id', None)}")
                print(f"reply_to.reply_to_msg_id: {getattr(reply_to, 'reply_to_msg_id', None)}")
                print(f"reply_to.forum_topic: {getattr(reply_to, 'forum_topic', None)}")
                print(f"inferred_topic_id: {topic_id}")
                print("")
            msg_date = getattr(message, "date", None)
            add_message_to_topic_groups(
                topics,
                topic_id=topic_id,
                reply_to=reply_to,
                text=text,
                msg_date=msg_date,
            )

        await _resolve_topic_titles(client, entity, topics)
        samples = filter_and_sort_topics(
            topics,
            min_count=max(1, min_count),
            topic_keyword=topic_title or topic_keyword,
            show_all_topics=show_all_topics,
        )

        chat_id = _chat_id(entity)
        if not output_config_only:
            for line in skipped_topic_debug_lines(
                topics,
                min_count=max(1, min_count),
                topic_keyword=topic_keyword,
                show_all_topics=show_all_topics,
            ):
                print(line)
            if topic_keyword and show_all_topics:
                print("topic keyword filter disabled by --show-all-topics")
            print(render_topic_groups(samples))
            print("")
        print(
            render_config_example(
                title=title,
                chat_id=chat_id,
                samples=samples,
                include_whole_chat=include_whole_chat,
                target_topic_title=topic_title,
            )
        )
    finally:
        await client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover Telegram dialogs and forum topics.")
    parser.add_argument("--filter", default="", help="Filter dialogs by title/username keyword.")
    parser.add_argument("--chat", default="", help="Chat id, username, or title to scan messages.")
    parser.add_argument("--limit", type=int, default=200, help="Recent messages to scan for topics.")
    parser.add_argument("--min-count", type=int, default=1, help="Only show topics with at least N recent messages.")
    parser.add_argument("--include-whole-chat", action="store_true", help="Include whole-chat targets without topic_id in config examples.")
    parser.add_argument("--topic-keyword", default="", help="Filter topics by topic_title or example_message.")
    parser.add_argument("--topic-title", default="", help="Filter topics by exact topic title.")
    parser.add_argument("--output-config-only", action="store_true", help="Only print the copyable config snippet.")
    parser.add_argument("--show-all-topics", action="store_true", help="Ignore --topic-keyword and show all inferred topics.")
    parser.add_argument("--no-topic-keyword-filter-debug", action="store_true", help="Alias for --show-all-topics.")
    args = parser.parse_args()
    if args.chat:
        asyncio.run(
            scan_chat(
                args.chat,
                args.limit,
                min_count=args.min_count,
                include_whole_chat=args.include_whole_chat,
                topic_keyword=args.topic_keyword,
                topic_title=args.topic_title,
                output_config_only=args.output_config_only,
                show_all_topics=args.show_all_topics or args.no_topic_keyword_filter_debug,
            )
        )
    else:
        asyncio.run(list_dialogs(args.filter))


if __name__ == "__main__":
    main()
