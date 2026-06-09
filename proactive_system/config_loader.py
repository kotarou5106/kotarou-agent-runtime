"""Proactive 配置加载和验证"""

from __future__ import annotations

import copy
import sys
from typing import Any, cast

from proactive_system.config import (
    ProactiveConfig,
    TelegramAudioCollectorProactiveConfig,
    TelegramDailySummaryProactiveConfig,
    TelegramJobMonitorProactiveConfig,
    TelegramMorningGreetingProactiveConfig,
    TelegramTargetConfig,
)
from proactive_system.presets import ALLOWED_OVERRIDE_KEYS, PRESETS, STRATEGY_PARAMS


class ProactiveConfigError(Exception):
    """Proactive 配置错误"""
    pass


def _validate_preset_name(preset: str) -> None:
    """验证预设名称"""
    if preset not in PRESETS:
        raise ProactiveConfigError(
            f"无效的 preset: '{preset}'。"
            f"只允许: {', '.join(PRESETS.keys())}"
        )


def _validate_overrides(overrides: dict[str, Any]) -> None:
    """验证 overrides 只包含白名单键"""
    for category, values in overrides.items():
        if category not in ALLOWED_OVERRIDE_KEYS:
            raise ProactiveConfigError(
                f"overrides 中的非法类别: '{category}'。"
                f"只允许: {', '.join(ALLOWED_OVERRIDE_KEYS.keys())}"
            )

        if not isinstance(values, dict):
            raise ProactiveConfigError(
                f"overrides.{category} 必须是字典，当前类型: {type(values).__name__}"
            )

        allowed = ALLOWED_OVERRIDE_KEYS[category]
        for key in values.keys():
            if key not in allowed:
                raise ProactiveConfigError(
                    f"overrides.{category}.{key} 不在白名单中。"
                    f"允许的键: {', '.join(sorted(allowed))}"
                )


def _validate_ranges(config: dict[str, Any]) -> None:
    """验证参数范围"""
    # 阈值类必须 0~1
    threshold_keys = [
        "score_llm_threshold",
        "judge_send_threshold",
        "context_only_judge_threshold",
        "context_only_judge_threshold_with_evidence",
        "anyaction_probability_min",
        "anyaction_probability_max",
    ]
    for key in threshold_keys:
        if key in config:
            val = config[key]
            if not (0 <= val <= 1):
                raise ProactiveConfigError(
                    f"{key} 必须在 [0, 1] 范围内，当前值: {val}"
                )

    # probability_min <= probability_max
    if "anyaction_probability_min" in config and "anyaction_probability_max" in config:
        pmin = config["anyaction_probability_min"]
        pmax = config["anyaction_probability_max"]
        if pmin > pmax:
            raise ProactiveConfigError(
                f"anyaction_probability_min ({pmin}) 不能大于 "
                f"anyaction_probability_max ({pmax})"
            )

    # tick_interval_s0 >= s1 >= s2 >= s3 >= 1
    intervals = [
        config.get("tick_interval_s0"),
        config.get("tick_interval_s1"),
        config.get("tick_interval_s2"),
        config.get("tick_interval_s3"),
    ]
    if all(x is not None for x in intervals):
        interval_values = [int(cast(int, x)) for x in intervals]
        for i in range(len(interval_values) - 1):
            if interval_values[i] < interval_values[i + 1]:
                raise ProactiveConfigError(
                    f"tick_interval 必须递减: s{i} ({interval_values[i]}) < s{i+1} ({interval_values[i+1]})"
                )
        if interval_values[-1] < 1:
            raise ProactiveConfigError(
                f"tick_interval_s3 必须 >= 1，当前值: {interval_values[-1]}"
            )

    # context_only_judge_threshold_with_evidence <= context_only_judge_threshold
    if "context_only_judge_threshold" in config and "context_only_judge_threshold_with_evidence" in config:
        with_ev = config["context_only_judge_threshold_with_evidence"]
        without_ev = config["context_only_judge_threshold"]
        if with_ev > without_ev:
            raise ProactiveConfigError(
                f"context_only_judge_threshold_with_evidence ({with_ev}) "
                f"不能大于 context_only_judge_threshold ({without_ev})"
            )


def _check_forbidden_keys(p: dict[str, Any]) -> None:
    """检查是否有旧的平铺键直接出现在 proactive 根下"""
    # 允许的根级键
    allowed_root_keys = {
        "enabled",
        "profile",
        "profiles",
        "target",
        "feed",
        "agent",
        "drift",
        "default_channel",
        "default_chat_id",
        "model",
        "preset",
        "features",
        "overrides",
        "feed_poller_interval_seconds",
        # v2 Agent Tick（独立子系统）
        "agent_tick",
        "telegram_job_monitor",
        "telegram_daily_summary",
        "telegram_morning_greeting",
        "telegram_audio_collector",
        "morning_greeting",
    }

    forbidden = set(p.keys()) - allowed_root_keys
    if forbidden:
        raise ProactiveConfigError(
            f"proactive 配置中出现非法的根级键: {', '.join(sorted(forbidden))}。\n"
            "请使用 profile / profiles / target 等分块方式配置。\n"
            f"允许的根级键: {', '.join(sorted(allowed_root_keys))}"
        )


def _validate_feature_keys(features: dict[str, Any]) -> None:
    forbidden = set(features.keys())
    if forbidden:
        raise ProactiveConfigError(
            f"proactive.features 出现非法键: {', '.join(sorted(forbidden))}。"
            "当前已无允许键。"
        )


def _validate_agent_tick_keys(agent_tick: dict[str, Any]) -> None:
    allowed = {
        "model",
        "max_steps",
        "content_limit",
        "web_fetch_max_chars",
        "context_prob",
        "delivery_cooldown_hours",
        "drift_enabled",
        "drift_max_steps",
        "drift_min_interval_hours",
    }
    forbidden = set(agent_tick.keys()) - allowed
    if forbidden:
        raise ProactiveConfigError(
            f"proactive.agent_tick 出现非法键: {', '.join(sorted(forbidden))}。"
            f"允许键: {', '.join(sorted(allowed))}"
        )


def _validate_profiles(profiles: dict[str, Any]) -> None:
    for name, values in profiles.items():
        if not isinstance(values, dict):
            raise ProactiveConfigError(
                f"proactive.profiles.{name} 必须是字典，当前类型: {type(values).__name__}"
            )
        _validate_overrides(values)


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(cast(dict[str, Any], dst[key]), value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def _merge_profiles(user_profiles: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = copy.deepcopy(PRESETS)
    for name, values in user_profiles.items():
        current = merged.get(name)
        if isinstance(values, dict) and isinstance(current, dict):
            _deep_merge(current, values)
        else:
            merged[name] = copy.deepcopy(values)
    return merged


def _validate_agent_keys(agent: dict[str, Any]) -> None:
    allowed = {
        "model",
        "max_steps",
        "content_limit",
        "web_fetch_max_chars",
        "context_prob",
        "delivery_cooldown_hours",
    }
    forbidden = set(agent.keys()) - allowed
    if forbidden:
        raise ProactiveConfigError(
            f"proactive.agent 出现非法键: {', '.join(sorted(forbidden))}。"
            f"允许键: {', '.join(sorted(allowed))}"
        )


def _validate_drift_keys(drift: dict[str, Any]) -> None:
    allowed = {
        "enabled",
        "max_steps",
        "min_interval_hours",
    }
    forbidden = set(drift.keys()) - allowed
    if forbidden:
        raise ProactiveConfigError(
            f"proactive.drift 出现非法键: {', '.join(sorted(forbidden))}。"
            f"允许键: {', '.join(sorted(allowed))}"
        )


def _validate_telegram_job_monitor_keys(config: dict[str, Any]) -> None:
    allowed = {
        "enabled",
        "schedule",
        "timezone",
        "target_chat_id",
        "read_limit",
        "debug",
    }
    forbidden = set(config.keys()) - allowed
    if forbidden:
        raise ProactiveConfigError(
            f"proactive.telegram_job_monitor 出现非法键: {', '.join(sorted(forbidden))}。"
            f"允许键: {', '.join(sorted(allowed))}"
        )


def _validate_telegram_daily_summary_keys(config: dict[str, Any]) -> None:
    allowed = {
        "enabled",
        "time",
        "schedule",
        "timezone",
        "summary_targets",
        "target_groups",
        "lookback_hours",
        "window_hours",
        "max_messages_per_target",
        "read_limit",
        "include_original_links",
        "send_to",
        "target_chat_id",
        "send_channel",
        "send_chat_id",
    }
    forbidden = set(config.keys()) - allowed
    if forbidden:
        raise ProactiveConfigError(
            f"proactive.telegram_daily_summary 出现非法键: {', '.join(sorted(forbidden))}。"
            f"允许键: {', '.join(sorted(allowed))}"
        )


def _validate_telegram_morning_greeting_keys(config: dict[str, Any]) -> None:
    allowed = {
        "enabled",
        "time",
        "schedule",
        "timezone",
        "send_to",
        "target_chat_id",
        "avoid_recent_days",
        "style_pool",
        "send_channel",
        "send_chat_id",
    }
    forbidden = set(config.keys()) - allowed
    if forbidden:
        raise ProactiveConfigError(
            f"proactive.telegram_morning_greeting 出现非法键: {', '.join(sorted(forbidden))}。"
            f"允许键: {', '.join(sorted(allowed))}"
        )


def _validate_telegram_audio_collector_keys(config: dict[str, Any]) -> None:
    allowed = {
        "enabled",
        "time",
        "schedule",
        "timezone",
        "lookback_hours",
        "send_to",
        "target_chat_id",
        "download_audio",
        "audio_download_dir",
        "include_original_links",
        "max_messages_per_target",
        "read_limit",
        "audio_targets",
        "keywords",
        "exclude_keywords",
    }
    forbidden = set(config.keys()) - allowed
    if forbidden:
        raise ProactiveConfigError(
            f"telegram_audio_collector 出现非法键: {', '.join(sorted(forbidden))}。"
            f"允许键: {', '.join(sorted(allowed))}"
        )


def _pick(primary: dict[str, Any], primary_key: str, legacy: dict[str, Any], legacy_key: str):
    if primary_key in primary:
        return primary[primary_key]
    return legacy.get(legacy_key)


def _as_int(value: Any, field_name: str) -> int:
    if value is None:
        raise ProactiveConfigError(f"{field_name} 不能为空")
    return int(value)


def _as_float(value: Any, field_name: str) -> float:
    if value is None:
        raise ProactiveConfigError(f"{field_name} 不能为空")
    return float(value)


def _config_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("${") and text.endswith("}"):
        return ""
    return text


def _load_telegram_targets(raw_groups: Any, field_name: str) -> list[TelegramTargetConfig]:
    if raw_groups is None:
        return []
    if not isinstance(raw_groups, list):
        raise ProactiveConfigError(f"{field_name} 必须是列表")
    targets: list[TelegramTargetConfig] = []
    for index, item in enumerate(raw_groups):
        if isinstance(item, str):
            chat_id = _config_text(item)
            chat_title = ""
            topic_id = None
        elif isinstance(item, dict):
            allowed = {"chat_id", "chat_title", "name", "topic_id", "message_thread_id", "expected_topic_title"}
            forbidden = set(item.keys()) - allowed
            if forbidden:
                raise ProactiveConfigError(
                    f"{field_name}[{index}] 出现非法键: {', '.join(sorted(forbidden))}。"
                    f"允许键: {', '.join(sorted(allowed))}"
                )
            chat_id = _config_text(item.get("chat_id", ""))
            chat_title = str(item.get("chat_title", item.get("name", "")) or "").strip()
            raw_topic_id = item.get("topic_id", item.get("message_thread_id"))
            topic_id = None if raw_topic_id in (None, "") else int(raw_topic_id)
            if topic_id is not None and topic_id <= 0:
                topic_id = None
            expected_topic_title = str(item.get("expected_topic_title", "") or "").strip() or None
        else:
            raise ProactiveConfigError(
                f"{field_name}[{index}] 必须是字符串或字典"
            )
        if chat_id:
            targets.append(
                TelegramTargetConfig(
                    chat_id=chat_id,
                    chat_title=chat_title,
                    topic_id=topic_id,
                    expected_topic_title=expected_topic_title,
                )
            )
    return targets


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ProactiveConfigError("列表配置项必须是数组")
    return [str(item) for item in value if str(item).strip()]


def load_proactive_config(p: dict[str, Any]) -> ProactiveConfig:
    """从配置字典加载 ProactiveConfig

    Args:
        p: proactive 配置字典

    Returns:
        ProactiveConfig 实例

    Raises:
        ProactiveConfigError: 配置错误时抛出
    """
    # 检查是否有非法的根级键
    _check_forbidden_keys(p)

    # 必填字段
    enabled = p.get("enabled", False)
    target = p.get("target", {}) or {}
    if not isinstance(target, dict):
        raise ProactiveConfigError("proactive.target 必须是字典")
    default_channel = str(target.get("channel", p.get("default_channel", "telegram")))
    default_chat_id = str(target.get("chat_id", p.get("default_chat_id", "")))
    model = p.get("model", "")

    # 预设名称（必填）
    preset_name = p.get("profile", p.get("preset"))
    if not preset_name:
        raise ProactiveConfigError(
            "proactive.profile 是必填字段。"
            f"可选值: {', '.join(PRESETS.keys())}"
        )
    user_profiles = p.get("profiles", {}) or {}
    if not isinstance(user_profiles, dict):
        raise ProactiveConfigError("proactive.profiles 必须是字典")
    _validate_profiles(user_profiles)
    profiles = _merge_profiles(user_profiles)
    _validate_preset_name(preset_name) if preset_name in PRESETS else None
    if preset_name not in profiles:
        raise ProactiveConfigError(
            f"无效的 profile: '{preset_name}'。"
            f"只允许: {', '.join(sorted(profiles.keys()))}"
        )
    preset = profiles[preset_name]

    # 功能开关
    features = p.get("features", {})
    _validate_feature_keys(features)
    feed = p.get("feed", {}) or {}
    if not isinstance(feed, dict):
        raise ProactiveConfigError("proactive.feed 必须是字典")
    feed_poller_interval_seconds = feed.get(
        "poll_interval_seconds",
        p.get("feed_poller_interval_seconds", 150),
    )

    # 合并预设和覆盖
    overrides = p.get("overrides", {})
    _validate_overrides(overrides)

    # 构建最终配置
    final_config = {}

    # 1. 应用预设
    for category, values in preset.items():
        final_config.update(values)

    # 2. 应用策略内置参数
    final_config.update(STRATEGY_PARAMS)

    # 3. 应用覆盖
    for category, values in overrides.items():
        final_config.update(values)

    # 验证范围
    _validate_ranges(final_config)

    # 移除已经显式设置的键，避免冲突
    explicit_keys = {
        "enabled",
        "default_channel",
        "default_chat_id",
        "profile",
        "profiles",
        "target",
        "feed",
        "agent",
        "drift",
        "model",
        "feed_poller_interval_seconds",
    }
    for key in explicit_keys:
        final_config.pop(key, None)

    # 构建 ProactiveConfig
    config = ProactiveConfig(
        enabled=enabled,
        default_channel=default_channel,
        default_chat_id=default_chat_id,
        model=model,
        feed_poller_interval_seconds=feed_poller_interval_seconds,
        **final_config,
    )

    # v2 Agent Tick 配置（独立子系统）
    at = p.get("agent_tick") or {}
    _validate_agent_tick_keys(at)
    agent = p.get("agent") or {}
    if not isinstance(agent, dict):
        raise ProactiveConfigError("proactive.agent 必须是字典")
    _validate_agent_keys(agent)
    drift = p.get("drift") or {}
    if not isinstance(drift, dict):
        raise ProactiveConfigError("proactive.drift 必须是字典")
    _validate_drift_keys(drift)

    agent_model = agent.get("model", at.get("model"))
    if agent_model:
        config.agent_tick_model = str(agent_model)
    if "max_steps" in agent or "max_steps" in at:
        config.agent_tick_max_steps = max(
            1,
            _as_int(_pick(agent, "max_steps", at, "max_steps"), "agent_runtime.max_steps"),
        )
    if "content_limit" in agent or "content_limit" in at:
        config.agent_tick_content_limit = max(
            1,
            _as_int(
                _pick(agent, "content_limit", at, "content_limit"),
                "agent_runtime.content_limit",
            ),
        )
    if "web_fetch_max_chars" in agent or "web_fetch_max_chars" in at:
        config.agent_tick_web_fetch_max_chars = max(
            1000,
            _as_int(
                _pick(agent, "web_fetch_max_chars", at, "web_fetch_max_chars"),
                "agent_runtime.web_fetch_max_chars",
            ),
        )
    if "context_prob" in agent or "context_prob" in at:
        config.agent_tick_context_prob = max(
            0.0,
            min(
                1.0,
                _as_float(
                    _pick(agent, "context_prob", at, "context_prob"),
                    "agent_runtime.context_prob",
                ),
            ),
        )
    if "delivery_cooldown_hours" in agent or "delivery_cooldown_hours" in at:
        config.agent_tick_delivery_cooldown_hours = max(
            0,
            int(
                _as_int(
                    _pick(
                        agent,
                        "delivery_cooldown_hours",
                        at,
                        "delivery_cooldown_hours",
                    ),
                    "agent_runtime.delivery_cooldown_hours",
                )
            ),
        )
    if "enabled" in drift or "drift_enabled" in at:
        config.drift_enabled = bool(drift.get("enabled", at.get("drift_enabled")))
    if "max_steps" in drift or "drift_max_steps" in at:
        config.drift_max_steps = max(
            3,
            _as_int(_pick(drift, "max_steps", at, "drift_max_steps"), "drift.max_steps"),
        )
    if "min_interval_hours" in drift or "drift_min_interval_hours" in at:
        config.drift_min_interval_hours = max(
            0,
            _as_int(
                _pick(drift, "min_interval_hours", at, "drift_min_interval_hours"),
                "drift.min_interval_hours",
            ),
        )

    telegram_job_monitor = p.get("telegram_job_monitor") or {}
    if not isinstance(telegram_job_monitor, dict):
        raise ProactiveConfigError("proactive.telegram_job_monitor 必须是字典")
    _validate_telegram_job_monitor_keys(telegram_job_monitor)
    target_chat_id = _config_text(telegram_job_monitor.get("target_chat_id", ""))
    config.telegram_job_monitor = TelegramJobMonitorProactiveConfig(
        enabled=bool(telegram_job_monitor.get("enabled", False)),
        schedule=str(telegram_job_monitor.get("schedule", "06:00")),
        timezone=str(telegram_job_monitor.get("timezone", "Asia/Shanghai")),
        target_chat_id=target_chat_id,
        read_limit=max(1, int(telegram_job_monitor.get("read_limit", 50))),
        debug=bool(telegram_job_monitor.get("debug", False)),
    )

    telegram_daily_summary = p.get("telegram_daily_summary") or {}
    if not isinstance(telegram_daily_summary, dict):
        raise ProactiveConfigError("proactive.telegram_daily_summary 必须是字典")
    _validate_telegram_daily_summary_keys(telegram_daily_summary)
    config.telegram_daily_summary = _load_telegram_daily_summary_config(
        telegram_daily_summary
    )

    telegram_morning_greeting = p.get("morning_greeting") or p.get("telegram_morning_greeting") or {}
    if not isinstance(telegram_morning_greeting, dict):
        raise ProactiveConfigError("morning_greeting 必须是字典")
    _validate_telegram_morning_greeting_keys(telegram_morning_greeting)
    config.telegram_morning_greeting = _load_morning_greeting_config(
        telegram_morning_greeting
    )

    telegram_audio_collector = p.get("telegram_audio_collector") or {}
    if not isinstance(telegram_audio_collector, dict):
        raise ProactiveConfigError("telegram_audio_collector 必须是字典")
    _validate_telegram_audio_collector_keys(telegram_audio_collector)
    config.telegram_audio_collector = _load_telegram_audio_collector_config(
        telegram_audio_collector
    )

    return config


def _load_telegram_daily_summary_config(
    raw: dict[str, Any],
) -> TelegramDailySummaryProactiveConfig:
    targets_raw = raw.get("summary_targets", raw.get("target_groups", []))
    return TelegramDailySummaryProactiveConfig(
        enabled=bool(raw.get("enabled", False)),
        time=str(raw.get("time", raw.get("schedule", "06:00"))),
        timezone=str(raw.get("timezone", "Asia/Shanghai")),
        lookback_hours=max(1, int(raw.get("lookback_hours", raw.get("window_hours", 24)))),
        send_to=_config_text(raw.get("send_to", "")) or "me",
        target_chat_id=_config_text(raw.get("target_chat_id", "")),
        max_messages_per_target=max(
            1,
            int(raw.get("max_messages_per_target", raw.get("read_limit", 500))),
        ),
        include_original_links=bool(raw.get("include_original_links", True)),
        summary_targets=_load_telegram_targets(
            targets_raw,
            "telegram_daily_summary.summary_targets",
        ),
    )


def _load_morning_greeting_config(
    raw: dict[str, Any],
) -> TelegramMorningGreetingProactiveConfig:
    style_pool = _as_str_list(raw.get("style_pool")) or [
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
    return TelegramMorningGreetingProactiveConfig(
        enabled=bool(raw.get("enabled", False)),
        time=str(raw.get("time", raw.get("schedule", "06:05"))),
        timezone=str(raw.get("timezone", "Asia/Shanghai")),
        send_to=_config_text(raw.get("send_to", "")) or "me",
        target_chat_id=_config_text(raw.get("target_chat_id", "")),
        avoid_recent_days=max(1, int(raw.get("avoid_recent_days", 14))),
        style_pool=style_pool,
    )


def _load_telegram_audio_collector_config(
    raw: dict[str, Any],
) -> TelegramAudioCollectorProactiveConfig:
    return TelegramAudioCollectorProactiveConfig(
        enabled=bool(raw.get("enabled", False)),
        time=str(raw.get("time", raw.get("schedule", "06:10"))),
        timezone=str(raw.get("timezone", "Asia/Shanghai")),
        lookback_hours=max(1, int(raw.get("lookback_hours", 24))),
        send_to=_config_text(raw.get("send_to", "")) or "me",
        target_chat_id=_config_text(raw.get("target_chat_id", "")),
        download_audio=bool(raw.get("download_audio", True)),
        audio_download_dir=str(raw.get("audio_download_dir", "data/telegram_audio_collector")),
        include_original_links=bool(raw.get("include_original_links", True)),
        max_messages_per_target=max(
            1,
            int(raw.get("max_messages_per_target", raw.get("read_limit", 500))),
        ),
        audio_targets=_load_telegram_targets(
            raw.get("audio_targets", []),
            "telegram_audio_collector.audio_targets",
        ),
        keywords=_as_str_list(raw.get("keywords")) or [
            "妈妈",
            "儿子",
            "继母",
            "母子",
            "小妈",
            "母亲",
            "妈咪",
            "熟母",
        ],
        exclude_keywords=_as_str_list(raw.get("exclude_keywords")) or [
            "未成年",
            "幼",
            "小学生",
            "初中",
            "高中",
            "偷拍",
            "强迫",
            "迷奸",
            "未同意",
        ],
    )
