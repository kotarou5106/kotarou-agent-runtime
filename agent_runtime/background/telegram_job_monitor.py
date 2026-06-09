from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from telethon import TelegramClient

from agent_runtime.network_proxy import log_proxy_env, select_telethon_proxy

logger = logging.getLogger(__name__)
STATE_PATH = Path.cwd() / "data" / "telegram_job_monitor_state.json"
TelegramProxy = tuple[str, str, int]

JOB_INTENT_KEYWORDS = [
    "招聘",
    "招人",
    "岗位",
    "职位",
    "JD",
    "待招岗位",
    "岗位职责",
    "岗位要求",
    "薪酬福利",
    "实习",
    "全职",
    "兼职",
]

TARGET_KEYWORDS = [
    "Agent",
    "AI Agent",
    "Agent 开发",
    "Agent 工程师",
    "RAG",
    "LLM",
    "量化",
    "量化研究员",
    "量化研究",
    "量化开发",
    "Quant",
    "Quant Research",
    "Quant Developer",
    "策略研究",
    "交易策略",
    "因子",
    "回测",
    "做市",
    "做市商",
    "Market Maker",
    "Market Making",
    "MM",
    "数据分析",
    "数据分析师",
    "Data Analyst",
    "Business Analyst",
    "BI",
]

EXCLUDE_KEYWORDS = [
    "关注频道",
    "Bounty",
    "Referral",
    "推荐奖励",
    "中介佣金",
    "介绍费",
    "资源对接",
    "人脉",
    "机会市场",
    "空投",
    "白嫖",
    "返佣",
    "求购",
    "买家",
    "卖家",
]

IRRELEVANT_ROLE_KEYWORDS = [
    "客服主管",
    "钱包运营",
    "内容运营",
    "活动运营",
    "产品经理",
    "安全工程师",
    "客服",
    "运营",
    "社群",
    "社区",
    "BD",
    "商务",
    "销售",
    "市场",
    "品牌",
    "HR",
    "猎头",
    "产品",
    "运维",
    "SRE",
    "DevOps",
]

DEBUG_WATCH_KEYWORDS = [
    "Gperp",
    "量化研究员",
    "待招岗位",
    "薪酬福利",
    "岗位职责",
]

ENGLISH_KEYWORDS = {
    "Agent",
    "AI Agent",
    "RAG",
    "LLM",
    "Quant",
    "Quant Research",
    "Quant Developer",
    "Market Maker",
    "Market Making",
    "MM",
    "Data Analyst",
    "Business Analyst",
    "BI",
    "JD",
    "Bounty",
    "Referral",
    "BD",
    "HR",
    "SRE",
    "DevOps",
}

FALLBACK_STRONG_TARGET_KEYWORDS = {
    "量化研究员",
    "AI Agent",
    "RAG",
    "LLM",
    "Data Analyst",
    "Market Maker",
}

SECTION_LABELS = [
    "待招岗位",
    "招聘岗位",
    "岗位职责",
    "岗位要求",
    "薪酬福利",
    "合作方式",
    "投递方式",
    "申请方式",
    "联系方式",
    "联系",
    "岗位来源",
    "官网",
    "中文频道",
    "Global Channel",
    "韩语频道",
    "X",
    "客服",
    "职位",
    "岗位",
]

APPLY_LABELS = {"申请方式", "联系方式", "联系", "投递方式"}
APPLY_STOP_LABELS = {
    "岗位来源",
    "官网",
    "中文频道",
    "Global Channel",
    "韩语频道",
    "X",
    "客服",
}
OFFICIAL_LINK_LABELS = {"官网", "中文频道", "Global Channel", "韩语频道", "X", "客服"}


@dataclass(frozen=True)
class JobSections:
    title_or_roles: str = ""
    responsibilities: str = ""
    requirements: str = ""
    compensation: str = ""
    work_mode: str = ""
    apply_method: str = ""
    source: str = ""
    message_link: str = ""
    official_links: list[str] | None = None
    raw_text: str = ""


@dataclass(frozen=True)
class FilterResult:
    sections: JobSections
    matched_job_intent_keywords: list[str]
    matched_target_keywords: list[str]
    matched_title_keywords: list[str]
    matched_responsibility_keywords: list[str]
    matched_requirement_keywords: list[str]
    matched_exclude_keywords: list[str]
    matched_irrelevant_role_keywords: list[str]
    final_decision: str
    skipped_reason: str

    @property
    def matched(self) -> bool:
        return self.final_decision == "matched"

    @property
    def notify_keywords(self) -> list[str]:
        merged: list[str] = []
        for keyword in [
            *self.matched_job_intent_keywords,
            *self.matched_target_keywords,
        ]:
            if keyword not in merged:
                merged.append(keyword)
        return merged


@dataclass(frozen=True)
class TelegramJobMonitorConfig:
    target_chat_id: int | str
    read_limit: int = 50
    debug: bool = False
    state_path: Path = STATE_PATH


@dataclass(frozen=True)
class JobMonitorResult:
    previous_last_message_id: int
    updated_last_message_id: int
    read_messages: int
    new_messages: int
    matches: int
    sent: bool


def _load_target_chat() -> int | str:
    raw = os.getenv("TELEGRAM_TARGET_CHAT_ID", "").strip()
    if not raw:
        raise SystemExit(
            "Missing required environment variable: TELEGRAM_TARGET_CHAT_ID\n"
            "Fill it in .env or export it before running this script."
        )
    try:
        return int(raw)
    except ValueError:
        return raw


def _load_monitor_limit() -> int:
    raw = (
        os.getenv("TELEGRAM_JOB_MONITOR_LIMIT", "").strip()
        or os.getenv("TELEGRAM_READ_LIMIT", "").strip()
    )
    if not raw:
        return 50
    try:
        limit = int(raw)
    except ValueError as exc:
        raise SystemExit("TELEGRAM_JOB_MONITOR_LIMIT must be an integer.") from exc
    return max(1, limit)


def _load_debug_enabled() -> bool:
    raw = os.getenv("TELEGRAM_JOB_MONITOR_DEBUG", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _load_notify_config() -> tuple[str, str] | None:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id_raw = (
        os.getenv("TELEGRAM_NOTIFY_CHAT_ID", "").strip()
        or os.getenv("TELEGRAM_CHAT_ID", "").strip()
    )
    missing = [
        name
        for name, value in (
            ("TELEGRAM_BOT_TOKEN", bot_token),
            ("TELEGRAM_NOTIFY_CHAT_ID or TELEGRAM_CHAT_ID", chat_id_raw),
        )
        if not value
    ]
    if missing:
        print(
            "Telegram notifier is not configured; falling back to terminal output. "
            "Missing: "
            + ", ".join(missing)
        )
        return None
    return bot_token, chat_id_raw


def _load_telegram_user_config() -> tuple[int, str, str, str, TelegramProxy | None]:
    load_dotenv(Path.cwd() / ".env")

    api_id_raw = os.getenv("TELEGRAM_API_ID", "").strip()
    api_hash = os.getenv("TELEGRAM_API_HASH", "").strip()
    phone = os.getenv("TELEGRAM_PHONE", "").strip()
    session_name = os.getenv("TELEGRAM_JOB_SESSION_NAME", "").strip()
    if not session_name:
        session_name = "job_monitor_session"

    missing = [
        name
        for name, value in (
            ("TELEGRAM_API_ID", api_id_raw),
            ("TELEGRAM_API_HASH", api_hash),
            ("TELEGRAM_PHONE", phone),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    try:
        api_id = int(api_id_raw)
    except ValueError as exc:
        raise RuntimeError("TELEGRAM_API_ID must be an integer.") from exc

    return api_id, api_hash, phone, session_name, _load_telegram_proxy()


def _load_telegram_proxy() -> TelegramProxy | None:
    log_proxy_env("telethon", once=True)
    proxy_type = os.getenv("TELEGRAM_PROXY_TYPE", "").strip()
    proxy_host = os.getenv("TELEGRAM_PROXY_HOST", "").strip()
    proxy_port_raw = os.getenv("TELEGRAM_PROXY_PORT", "").strip()

    if not any((proxy_type, proxy_host, proxy_port_raw)):
        proxy, source = select_telethon_proxy()
        if proxy is not None:
            logger.info("[telethon] proxy configured from %s", source)
        else:
            logger.info("[telethon] proxy not configured")
        return proxy
    if not all((proxy_type, proxy_host, proxy_port_raw)):
        raise RuntimeError(
            "TELEGRAM_PROXY_TYPE, TELEGRAM_PROXY_HOST, and TELEGRAM_PROXY_PORT "
            "must all be set to enable Telegram proxy."
        )
    try:
        proxy_port = int(proxy_port_raw)
    except ValueError as exc:
        raise RuntimeError("TELEGRAM_PROXY_PORT must be an integer.") from exc
    logger.info("[telethon] proxy configured from TELEGRAM_PROXY_*")
    return proxy_type, proxy_host, proxy_port


def build_telegram_client() -> tuple[TelegramClient, str]:
    api_id, api_hash, phone, session_name, proxy = _load_telegram_user_config()
    client = TelegramClient(session_name, api_id, api_hash, proxy=proxy)
    return client, phone


def _target_key(target_chat: int | str) -> str:
    return str(target_chat)


def _load_state(target_chat: int | str, state_path: Path = STATE_PATH) -> dict[str, Any]:
    if not state_path.exists():
        return {"target_chat": _target_key(target_chat), "last_message_id": 0}

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read state file: {state_path}\n{exc}") from exc

    if state.get("target_chat") != _target_key(target_chat):
        return {"target_chat": _target_key(target_chat), "last_message_id": 0}

    last_message_id = state.get("last_message_id", 0)
    if not isinstance(last_message_id, int):
        raise RuntimeError(f"Invalid last_message_id in state file: {state_path}")
    return {"target_chat": _target_key(target_chat), "last_message_id": last_message_id}


def _save_state(
    target_chat: int | str,
    last_message_id: int,
    state_path: Path = STATE_PATH,
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "target_chat": _target_key(target_chat),
        "last_message_id": last_message_id,
    }
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _message_text(message: object) -> str:
    return str(getattr(message, "message", "") or "").strip()


def _contains_keyword(text: str, keyword: str) -> bool:
    if keyword == "市场":
        return re.search(r"市场(?!风险)", text) is not None
    if keyword in ENGLISH_KEYWORDS:
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(keyword)}(?![A-Za-z0-9_])"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    return keyword.lower() in text.lower()


def _matched_from(text: str, keywords: list[str]) -> list[str]:
    return [keyword for keyword in keywords if _contains_keyword(text, keyword)]


def parse_job_sections(text: str) -> JobSections:
    lines = [line.strip() for line in text.splitlines()]
    title_or_roles = ""
    compensation = ""
    work_mode = ""
    apply_method_lines: list[str] = []
    source = ""
    official_links: list[str] = []
    responsibilities_lines: list[str] = []
    requirements_lines: list[str] = []
    section = ""

    for line in lines:
        if not line:
            continue
        label, value = _split_labeled_line(line)
        if label in {"待招岗位", "招聘岗位", "职位", "岗位"}:
            title_or_roles = value
            section = ""
            continue
        if label == "薪酬福利":
            compensation = value
            section = ""
            continue
        if label == "合作方式":
            work_mode = value
            section = ""
            continue
        if label in APPLY_LABELS:
            section = "apply_method"
            if value:
                apply_method_lines.append(value)
            continue
        if label == "岗位职责":
            section = "responsibilities"
            if value:
                responsibilities_lines.append(value)
            continue
        if label == "岗位要求":
            section = "requirements"
            if value:
                requirements_lines.append(value)
            continue
        if label == "岗位来源":
            source = value
            section = ""
            continue
        if label in OFFICIAL_LINK_LABELS:
            official_links.append(f"{label}: {value}".strip())
            section = ""
            continue
        if label in APPLY_STOP_LABELS:
            section = ""
            continue
        if label:
            section = ""
            continue
        if section == "responsibilities":
            responsibilities_lines.append(line)
        elif section == "requirements":
            requirements_lines.append(line)
        elif section == "apply_method":
            apply_method_lines.append(line)

    return JobSections(
        title_or_roles=title_or_roles,
        responsibilities="\n".join(responsibilities_lines).strip(),
        requirements="\n".join(requirements_lines).strip(),
        compensation=compensation,
        work_mode=work_mode,
        apply_method="\n".join(apply_method_lines).strip(),
        source=source,
        message_link="",
        official_links=official_links,
        raw_text=text,
    )


def _split_labeled_line(line: str) -> tuple[str, str]:
    for label in SECTION_LABELS:
        match = re.search(rf"{re.escape(label)}\s*[:：]\s*(.*)$", line)
        if match:
            return label, _normalize_section_value(match.group(1))
    return "", ""


def _normalize_section_value(value: str) -> str:
    text = re.sub(r"#(?=\S)", "", value)
    text = text.replace("#", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _filter_result(
    *,
    sections: JobSections,
    matched_intent: list[str],
    matched_target: list[str],
    matched_title: list[str],
    matched_responsibilities: list[str],
    matched_requirements: list[str],
    matched_exclude: list[str],
    matched_irrelevant_role: list[str],
    final_decision: str,
    skipped_reason: str,
) -> FilterResult:
    return FilterResult(
        sections=sections,
        matched_job_intent_keywords=matched_intent,
        matched_target_keywords=matched_target,
        matched_title_keywords=matched_title,
        matched_responsibility_keywords=matched_responsibilities,
        matched_requirement_keywords=matched_requirements,
        matched_exclude_keywords=matched_exclude,
        matched_irrelevant_role_keywords=matched_irrelevant_role,
        final_decision=final_decision,
        skipped_reason=skipped_reason,
    )


def evaluate_job_message(text: str) -> FilterResult:
    sections = parse_job_sections(text)
    matched_exclude = _matched_from(text, EXCLUDE_KEYWORDS)
    matched_intent = _matched_from(text, JOB_INTENT_KEYWORDS)
    matched_title = _matched_from(sections.title_or_roles, TARGET_KEYWORDS)
    matched_responsibilities = _matched_from(sections.responsibilities, TARGET_KEYWORDS)
    matched_requirements = _matched_from(sections.requirements, TARGET_KEYWORDS)
    matched_target = _dedupe_keywords(
        [*matched_title, *matched_responsibilities, *matched_requirements]
    )
    irrelevant_scope = sections.title_or_roles or text
    matched_irrelevant_role = _matched_from(irrelevant_scope, IRRELEVANT_ROLE_KEYWORDS)

    if matched_exclude:
        return _filter_result(
            sections=sections,
            matched_intent=matched_intent,
            matched_target=matched_target,
            matched_title=matched_title,
            matched_responsibilities=matched_responsibilities,
            matched_requirements=matched_requirements,
            matched_exclude=matched_exclude,
            matched_irrelevant_role=matched_irrelevant_role,
            final_decision="skipped",
            skipped_reason="skipped_by_exclude_keywords",
        )
    if sections.title_or_roles and matched_irrelevant_role:
        return _filter_result(
            sections=sections,
            matched_intent=matched_intent,
            matched_target=matched_target,
            matched_title=matched_title,
            matched_responsibilities=matched_responsibilities,
            matched_requirements=matched_requirements,
            matched_exclude=matched_exclude,
            matched_irrelevant_role=matched_irrelevant_role,
            final_decision="skipped",
            skipped_reason="skipped_irrelevant_role",
        )
    if not matched_intent:
        return _filter_result(
            sections=sections,
            matched_intent=matched_intent,
            matched_target=matched_target,
            matched_title=matched_title,
            matched_responsibilities=matched_responsibilities,
            matched_requirements=matched_requirements,
            matched_exclude=matched_exclude,
            matched_irrelevant_role=matched_irrelevant_role,
            final_decision="skipped",
            skipped_reason="skipped_no_job_intent_keywords",
        )
    if sections.title_or_roles:
        if matched_title:
            return _filter_result(
                sections=sections,
                matched_intent=matched_intent,
                matched_target=matched_title,
                matched_title=matched_title,
                matched_responsibilities=matched_responsibilities,
                matched_requirements=matched_requirements,
                matched_exclude=matched_exclude,
                matched_irrelevant_role=matched_irrelevant_role,
                final_decision="matched",
                skipped_reason="",
            )
        return _filter_result(
            sections=sections,
            matched_intent=matched_intent,
            matched_target=matched_target,
            matched_title=matched_title,
            matched_responsibilities=matched_responsibilities,
            matched_requirements=matched_requirements,
            matched_exclude=matched_exclude,
            matched_irrelevant_role=matched_irrelevant_role,
            final_decision="skipped",
            skipped_reason="skipped_no_title_target_keywords",
        )

    fallback_target = _dedupe_keywords([*matched_responsibilities, *matched_requirements])
    strong_fallback = [kw for kw in fallback_target if kw in FALLBACK_STRONG_TARGET_KEYWORDS]
    if len(fallback_target) >= 2 or strong_fallback:
        return _filter_result(
            sections=sections,
            matched_intent=matched_intent,
            matched_target=fallback_target,
            matched_title=matched_title,
            matched_responsibilities=matched_responsibilities,
            matched_requirements=matched_requirements,
            matched_exclude=matched_exclude,
            matched_irrelevant_role=matched_irrelevant_role,
            final_decision="matched",
            skipped_reason="",
        )
    return _filter_result(
        sections=sections,
        matched_intent=matched_intent,
        matched_target=fallback_target,
        matched_title=matched_title,
        matched_responsibilities=matched_responsibilities,
        matched_requirements=matched_requirements,
        matched_exclude=matched_exclude,
        matched_irrelevant_role=matched_irrelevant_role,
        final_decision="skipped",
        skipped_reason="skipped_no_title_target_keywords",
    )


def _dedupe_keywords(keywords: list[str]) -> list[str]:
    result: list[str] = []
    for keyword in keywords:
        if keyword not in result:
            result.append(keyword)
    return result


def _headline(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return _normalize_section_value(line)[:80]
    return "Telegram 招聘消息"


def _notification_title(text: str, sections: JobSections) -> str:
    headline = _headline(text)
    if not sections.title_or_roles:
        return headline
    if sections.title_or_roles in headline:
        return headline
    return f"{headline} / {sections.title_or_roles}"


def _display_keywords(keywords: list[str]) -> list[str]:
    target_keywords = [keyword for keyword in keywords if keyword not in JOB_INTENT_KEYWORDS]
    return target_keywords or keywords


def _strip_item_marker(item: str) -> str:
    return re.sub(
        r"^\s*(?:[0-9]+[.)、:：]|[0-9]\ufe0f?\u20e3|[①②③④⑤⑥⑦⑧⑨⑩]|[-•*])\s*",
        "",
        item,
    ).strip()


def _section_items(text: str, *, max_items: int = 5, max_chars: int = 120) -> list[str]:
    if not text.strip():
        return []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    marker_pattern = (
        r"(?:^|\n)\s*"
        r"(?:[0-9]+[.)、:：]|[0-9]\ufe0f?\u20e3|[①②③④⑤⑥⑦⑧⑨⑩]|[-•*])\s*"
    )
    if re.search(marker_pattern, normalized):
        chunks = re.split(marker_pattern, normalized)
    else:
        chunks = normalized.splitlines()

    items: list[str] = []
    for chunk in chunks:
        item = _strip_item_marker(re.sub(r"\s+", " ", chunk).strip())
        if not item:
            continue
        if len(item) > max_chars:
            item = item[: max_chars - 1].rstrip() + "..."
        items.append(item)
        if len(items) >= max_items:
            break
    return items


def _append_multiline_field(lines: list[str], label: str, value: str) -> None:
    if not value:
        return
    lines.extend(["", f"{label}：", value])


def _append_numbered_section(lines: list[str], label: str, value: str) -> None:
    items = _section_items(value)
    if not items:
        return
    lines.extend(["", f"{label}："])
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. {item}")


def _format_notify_message(
    matches: list[tuple[object, list[str], str, str]],
) -> str:
    lines = [
        "【Telegram Job Monitor】",
        f"发现 {len(matches)} 条可能相关岗位：",
        "",
    ]
    for index, (_message, keywords, text, message_link) in enumerate(matches, start=1):
        sections = parse_job_sections(text)
        title = _notification_title(text, sections)
        display_keywords = _display_keywords(keywords)
        lines.extend(
            [
                f"{index}. {title}",
                f"命中关键词：{'、'.join(display_keywords) if display_keywords else '无'}",
            ]
        )
        if sections.work_mode:
            lines.append(f"合作方式：{sections.work_mode}")
        if sections.compensation:
            lines.append(f"薪酬福利：{sections.compensation}")
        _append_numbered_section(lines, "岗位职责", sections.responsibilities)
        _append_numbered_section(lines, "岗位要求", sections.requirements)
        _append_multiline_field(lines, "申请方式", sections.apply_method)
        _append_multiline_field(lines, "岗位来源", sections.source)
        _append_multiline_field(lines, "原消息", message_link)
        lines.append("")
    return "\n".join(lines).strip()


def build_telegram_message_link(chat: object, message: object) -> str:
    message_id = getattr(message, "id", None)
    username = str(getattr(chat, "username", "") or "").strip().lstrip("@")
    if username and message_id:
        return f"https://t.me/{username}/{message_id}"

    raw_chat_id = getattr(chat, "id", None)
    if raw_chat_id is not None and message_id:
        chat_id = str(raw_chat_id)
        internal_id = chat_id[4:] if chat_id.startswith("-100") else chat_id.lstrip("-")
        if internal_id.isdigit():
            return f"https://t.me/c/{internal_id}/{message_id}"

    return f"message_id: {message_id}" if message_id is not None else "message_id: unknown"


def _contains_debug_watch_keyword(text: str) -> bool:
    return any(_contains_keyword(text, keyword) for keyword in DEBUG_WATCH_KEYWORDS)


def _print_filter_debug(
    message: object,
    chat: object,
    text: str,
    result: FilterResult,
    *,
    last_message_id: int,
) -> None:
    message_id = int(getattr(message, "id", 0) or 0)
    message_link = build_telegram_message_link(chat, message)
    final_message = _format_notify_message(
        [(message, result.notify_keywords, text, message_link)]
    )
    print("", flush=True)
    print("[job-monitor debug]", flush=True)
    print(f"message_id: {getattr(message, 'id', '')}", flush=True)
    print(f"date: {getattr(message, 'date', '')}", flush=True)
    print(f"text_preview: {text[:300] if text else '[non-text message]'}", flush=True)
    print(f"is_new_for_state: {message_id > last_message_id}", flush=True)
    print(f"contains_Gperp: {_contains_keyword(text, 'Gperp')}", flush=True)
    print(f"title_or_roles: {result.sections.title_or_roles}", flush=True)
    print(f"responsibilities_preview: {result.sections.responsibilities[:300]}", flush=True)
    print(f"responsibilities_length: {len(result.sections.responsibilities)}", flush=True)
    print(f"requirements_preview: {result.sections.requirements[:300]}", flush=True)
    print(f"requirements_length: {len(result.sections.requirements)}", flush=True)
    print(f"apply_method: {result.sections.apply_method}", flush=True)
    print(f"apply_method_preview: {result.sections.apply_method[:300]}", flush=True)
    print(f"source: {result.sections.source}", flush=True)
    print(f"message_link: {message_link}", flush=True)
    print(f"matched_job_intent_keywords: {result.matched_job_intent_keywords}", flush=True)
    print(f"matched_title_keywords: {result.matched_title_keywords}", flush=True)
    print(f"matched_responsibility_keywords: {result.matched_responsibility_keywords}", flush=True)
    print(f"matched_requirement_keywords: {result.matched_requirement_keywords}", flush=True)
    print(f"matched_target_keywords: {result.matched_target_keywords}", flush=True)
    print(f"matched_exclude_keywords: {result.matched_exclude_keywords}", flush=True)
    print(f"matched_irrelevant_role_keywords: {result.matched_irrelevant_role_keywords}", flush=True)
    print(f"final_decision: {result.final_decision}", flush=True)
    print(f"skipped_reason: {result.skipped_reason}", flush=True)
    print(f"final_message_length: {len(final_message)}", flush=True)


def _split_plain_text(text: str, limit: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if current and current_len + len(line) > limit:
            chunks.append("".join(current).strip())
            current = []
            current_len = 0
        while len(line) > limit:
            if current:
                chunks.append("".join(current).strip())
                current = []
                current_len = 0
            chunks.append(line[:limit].strip())
            line = line[limit:]
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _split_for_telegram(text: str, limit: int = 3500) -> list[str]:
    if len(text) <= limit:
        return [text] if text else []

    content_limit = max(1, limit - 40)
    chunks = _split_plain_text(text, content_limit)
    if len(chunks) <= 1:
        return chunks

    total = len(chunks)
    return [
        f"第 {index}/{total} 部分\n\n{chunk}"
        for index, chunk in enumerate(chunks, start=1)
    ]


def _post_telegram_message(bot_token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8", errors="replace")
            if response.status >= 400:
                raise RuntimeError(f"Telegram sendMessage failed: HTTP {response.status} {body}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Telegram sendMessage failed: HTTP {exc.code} {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Telegram sendMessage failed: {exc}") from exc


async def _send_notification(matches: list[tuple[object, list[str], str, str]]) -> bool:
    notify_config = _load_notify_config()
    if notify_config is None:
        return False

    bot_token, chat_id = notify_config
    message = _format_notify_message(matches)
    for chunk in _split_for_telegram(message):
        await asyncio.to_thread(_post_telegram_message, bot_token, chat_id, chunk)
    return True


def _print_matches_to_terminal(matches: list[tuple[object, list[str], str, str]]) -> None:
    print(_format_notify_message(matches))


def load_job_monitor_config_from_env(
    *,
    state_path: Path = STATE_PATH,
) -> TelegramJobMonitorConfig:
    load_dotenv(Path.cwd() / ".env")
    return TelegramJobMonitorConfig(
        target_chat_id=_load_target_chat(),
        read_limit=_load_monitor_limit(),
        debug=_load_debug_enabled(),
        state_path=state_path,
    )


async def run_telegram_job_monitor_once(
    config: TelegramJobMonitorConfig,
) -> JobMonitorResult:
    target_chat = config.target_chat_id
    limit = config.read_limit
    debug_enabled = config.debug
    state = _load_state(target_chat, config.state_path)
    last_message_id = int(state["last_message_id"])
    max_seen_id = last_message_id
    logger.info(
        "[telegram_job_monitor] started target_chat_id=%s previous_last_message_id=%s",
        target_chat,
        last_message_id,
    )

    client, phone = build_telegram_client()
    try:
        await client.start(phone=phone)
        entity = await client.get_entity(target_chat)
        title = (
            getattr(entity, "title", None)
            or getattr(entity, "username", None)
            or getattr(entity, "first_name", None)
            or str(target_chat)
        )

        fetched_messages = []
        new_messages = []
        async for message in client.iter_messages(entity, limit=limit):
            fetched_messages.append(message)
            message_id = int(getattr(message, "id", 0) or 0)
            if message_id <= last_message_id:
                continue
            max_seen_id = max(max_seen_id, message_id)
            new_messages.append(message)

        fetched_messages.sort(key=lambda item: int(getattr(item, "id", 0) or 0))
        new_messages.sort(key=lambda item: int(getattr(item, "id", 0) or 0))

        matches = []
        for message in new_messages:
            text = _message_text(message)
            result = evaluate_job_message(text)
            if result.matched:
                matches.append(
                    (
                        message,
                        result.notify_keywords,
                        text,
                        build_telegram_message_link(entity, message),
                    )
                )

        print("Telegram Job Monitor V1")
        print(f"target: {title} ({target_chat})")
        print(f"state_file: {config.state_path}")
        print(f"previous_last_message_id: {last_message_id}")
        print(f"read_messages: {len(fetched_messages)}")
        print(f"new_messages: {len(new_messages)}")
        print(f"matches: {len(matches)}")
        print(f"debug: {debug_enabled}")

        sent = False
        if debug_enabled:
            print("debug_messages: all read_messages", flush=True)
            for message in fetched_messages:
                text = _message_text(message)
                result = evaluate_job_message(text)
                _print_filter_debug(
                    message,
                    entity,
                    text,
                    result,
                    last_message_id=last_message_id,
                )
        else:
            for message in fetched_messages:
                text = _message_text(message)
                if _contains_debug_watch_keyword(text):
                    result = evaluate_job_message(text)
                    _print_filter_debug(
                        message,
                        entity,
                        text,
                        result,
                        last_message_id=last_message_id,
                    )

        if matches:
            sent = await _send_notification(matches)
            if sent:
                print(f"Sent {len(matches)} matched job(s) to Telegram.")
            else:
                _print_matches_to_terminal(matches)
        else:
            print("No new matched jobs.")

        _save_state(target_chat, max_seen_id, config.state_path)
        print("")
        print(f"updated_last_message_id: {max_seen_id}")
        result = JobMonitorResult(
            previous_last_message_id=last_message_id,
            updated_last_message_id=max_seen_id,
            read_messages=len(fetched_messages),
            new_messages=len(new_messages),
            matches=len(matches),
            sent=bool(matches and sent),
        )
        logger.info(
            "[telegram_job_monitor] target_chat_id=%s previous_last_message_id=%s "
            "read_messages=%s new_messages=%s matches=%s %s updated_last_message_id=%s",
            target_chat,
            result.previous_last_message_id,
            result.read_messages,
            result.new_messages,
            result.matches,
            "sent"
            if result.sent
            else ("no matched jobs" if result.matches == 0 else "matched jobs not sent"),
            result.updated_last_message_id,
        )
        return result
    finally:
        await client.disconnect()


async def main() -> None:
    config = load_job_monitor_config_from_env()
    await run_telegram_job_monitor_once(config)


if __name__ == "__main__":
    asyncio.run(main())
