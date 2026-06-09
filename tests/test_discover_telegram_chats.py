from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


def _load_discover_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "discover_telegram_chats.py"
    spec = importlib.util.spec_from_file_location("discover_telegram_chats", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample(topic_id, title, count, example=""):
    mod = _load_discover_module()
    return mod.TopicSample(
        topic_id=topic_id,
        topic_title=title,
        count=count,
        latest=datetime(2026, 6, 9, 6, count, tzinfo=timezone.utc),
        example=example or f"{title} example",
        top_message_id=topic_id,
        root_message_id=topic_id,
    )


def test_config_example_does_not_output_whole_chat_by_default() -> None:
    mod = _load_discover_module()
    samples = [
        _sample(None, "<whole chat>", 12, "whole chat example"),
        _sample(13017, "招聘", 8),
    ]

    rendered = mod.render_config_example(
        title="吃瓜闲聊-禁发招聘",
        chat_id=-1002535833398,
        samples=samples,
    )

    assert '{ chat_title = "吃瓜闲聊-禁发招聘", chat_id = -1002535833398 },' not in rendered
    assert "topic_id = 13017" in rendered


def test_config_example_outputs_whole_chat_when_requested() -> None:
    mod = _load_discover_module()

    rendered = mod.render_config_example(
        title="吃瓜闲聊-禁发招聘",
        chat_id=-1002535833398,
        samples=[_sample(13017, "招聘", 8)],
        include_whole_chat=True,
    )

    assert '{ chat_title = "吃瓜闲聊-禁发招聘", chat_id = -1002535833398 },' in rendered
    assert "不带 topic_id：监听整个群，可能很慢且很杂" in rendered


def test_topic_group_sorting_by_recent_message_count() -> None:
    mod = _load_discover_module()
    topics = {
        1: _sample(1, "low", 1),
        2: _sample(2, "high", 9),
        3: _sample(3, "mid", 4),
    }

    sorted_topics = mod.filter_and_sort_topics(topics)

    assert [item.topic_id for item in sorted_topics] == [2, 3, 1]


def test_topic_keyword_filters_title_or_example() -> None:
    mod = _load_discover_module()
    topics = {
        1: _sample(1, "闲聊", 5, "普通聊天"),
        2: _sample(2, "研发类岗位招聘", 4, "Python 工程师"),
        3: _sample(3, "日常", 3, "这里有招聘信息"),
    }

    filtered = mod.filter_and_sort_topics(topics, topic_keyword="招聘")

    assert [item.topic_id for item in filtered] == [2, 3]


def test_config_example_only_contains_topic_id_targets() -> None:
    mod = _load_discover_module()
    samples = mod.filter_and_sort_topics(
        {
            None: _sample(None, "<unknown>", 20),
            265: _sample(265, "研发", 3),
        }
    )

    rendered = mod.render_config_example(
        title="研发类岗位招聘&求职",
        chat_id=-1002535833398,
        samples=samples,
    )

    target_lines = [line for line in rendered.splitlines() if line.strip().startswith("{")]
    assert target_lines
    assert all("topic_id" in line for line in target_lines)


def test_message_with_reply_to_top_id_enters_topic_groups() -> None:
    mod = _load_discover_module()
    topics = {}
    reply_to = SimpleNamespace(reply_to_top_id=6, reply_to_msg_id=None)

    mod.add_message_to_topic_groups(
        topics,
        topic_id=6,
        reply_to=reply_to,
        text="topic message",
        msg_date=datetime(2026, 6, 9, 6, 0, tzinfo=timezone.utc),
    )

    rendered = mod.render_topic_groups(mod.filter_and_sort_topics(topics))
    assert "topic_id: 6" in rendered
    assert "topic_title: <unknown>" in rendered
    assert "recent_message_count: 1" in rendered
    assert "example_message: topic message" in rendered


def test_unknown_topic_title_still_outputs_config() -> None:
    mod = _load_discover_module()
    samples = [
        mod.TopicSample(
            topic_id=6,
            topic_title="<unknown>",
            count=1,
            latest=datetime(2026, 6, 9, 6, 0, tzinfo=timezone.utc),
            example="topic message",
        )
    ]

    rendered = mod.render_config_example(
        title="吃瓜闲聊-禁发招聘",
        chat_id=-1002535833398,
        samples=samples,
    )

    assert '{ chat_title = "吃瓜闲聊-禁发招聘", chat_id = -1002535833398, topic_id = 6 },' in rendered


def test_topic_keyword_matches_recent_texts() -> None:
    mod = _load_discover_module()
    topics = {
        6: mod.TopicSample(
            topic_id=6,
            topic_title="<unknown>",
            count=1,
            example="",
            recent_texts=["这里最近在聊招聘和求职"],
        )
    }

    filtered = mod.filter_and_sort_topics(topics, topic_keyword="招聘")

    assert [item.topic_id for item in filtered] == [6]


def test_topic_keyword_miss_has_skipped_debug_output() -> None:
    mod = _load_discover_module()
    topics = {
        6: mod.TopicSample(
            topic_id=6,
            topic_title="<unknown>",
            count=1,
            example="普通闲聊",
            recent_texts=["没有目标词"],
        )
    }

    lines = mod.skipped_topic_debug_lines(topics, topic_keyword="招聘")

    assert lines == ["skipped topic_id = 6 because keyword not matched"]
