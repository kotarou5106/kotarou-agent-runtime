from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Protocol


class SessionDashboardReader(Protocol):
    def list_sessions_for_dashboard(self, **kwargs: Any) -> tuple[list[dict[str, Any]], int]: ...

    def list_messages_for_dashboard(self, **kwargs: Any) -> tuple[list[dict[str, Any]], int]: ...


class MemoryDashboardReader(Protocol):
    def list_items_for_dashboard(self, **kwargs: Any) -> tuple[list[dict[str, Any]], int]: ...


class ProactiveDailyReader(Protocol):
    def list_tick_logs(self, **kwargs: Any) -> tuple[list[dict[str, Any]], int]: ...

    def list_tick_steps(self, tick_id: str) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class DailyWorkspaceService:
    workspace: Any
    sessions: SessionDashboardReader
    proactive: ProactiveDailyReader
    memory: MemoryDashboardReader
    agent_name: str = "Kotarou"

    def snapshot(self, target_date: date) -> dict[str, Any]:
        start, end = _day_bounds(target_date)
        ticks, tick_total = self.proactive.list_tick_logs(
            started_from=start,
            started_to=end,
            page=1,
            page_size=100,
            sort_by="started_at",
            sort_order="desc",
        )
        sessions, _ = self.sessions.list_sessions_for_dashboard(
            updated_from=start,
            updated_to=end,
            page=1,
            page_size=50,
            sort_by="updated_at",
            sort_order="desc",
        )
        messages, _ = self.sessions.list_messages_for_dashboard(
            page=1,
            page_size=100,
            sort_by="ts",
            sort_order="desc",
        )
        messages = [item for item in messages if _is_on_date(item.get("ts"), target_date)]
        memory_items = self._memory_items_for_date(target_date)
        archive_dates = self.archive_dates(target_date)

        tool_calls = self._tool_calls(ticks)
        missions = self._missions(ticks, messages)
        failures = [item for item in missions if item["status"] == "failed"]
        needs_approval = [
            item for item in missions if item["status"] == "needs_approval"
        ]
        ephemera = self._ephemera(ticks, messages, sessions)
        next_actions = self._next_actions(failures, needs_approval, memory_items)
        activity_count = len(ephemera) + len(tool_calls) + len(memory_items)
        fallback_fields: list[str] = []

        if not missions:
            fallback_fields.append("missions")
        if not ephemera:
            fallback_fields.append("ephemera")
        if not memory_items:
            fallback_fields.append("memory_items")
        if not tool_calls:
            fallback_fields.append("tool_calls")
        if not next_actions:
            fallback_fields.append("next_actions")

        completed = sum(1 for item in missions if item["status"] == "completed")
        perspective = self._perspective(
            ticks,
            memory_items,
            activity_count=activity_count,
        )
        status = _runtime_status(ticks, failures, needs_approval)
        no_real_data = not any([ticks, messages, sessions, memory_items, tool_calls])

        return {
            "date": target_date.isoformat(),
            "archive_dates": archive_dates,
            "agent_name": self.agent_name,
            "status": status,
            "status_text": _status_text(completed, activity_count),
            "perspective": perspective,
            "missions": missions,
            "ephemera": ephemera,
            "memory_items": memory_items,
            "tool_calls": tool_calls,
            "failures": failures,
            "needs_approval": needs_approval,
            "next_actions": next_actions,
            "is_sample": bool(fallback_fields),
            "no_real_data": no_real_data,
            "sample_fallback_fields": fallback_fields,
            "sources": {
                "missions": "proactive tick_log + session messages",
                "ephemera": "proactive tick_log + session messages",
                "memory_items": "memory_admin.list_items_for_dashboard",
                "tool_calls": "proactive tick_step_log",
                "failures": "derived from missions",
                "needs_approval": "derived from missions and skip reasons",
                "next_actions": "derived from failures/approval/memory gaps",
            },
            "totals": {
                "ticks": tick_total,
                "messages": len(messages),
                "sessions": len(sessions),
                "memory_items": len(memory_items),
            },
        }

    def archive_dates(self, today: date) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for offset in range(7):
            current = today - timedelta(days=offset)
            start, end = _day_bounds(current)
            _, tick_total = self.proactive.list_tick_logs(
                started_from=start,
                started_to=end,
                page=1,
                page_size=1,
                sort_by="started_at",
                sort_order="desc",
            )
            sessions, session_total = self.sessions.list_sessions_for_dashboard(
                updated_from=start,
                updated_to=end,
                page=1,
                page_size=1,
                sort_by="updated_at",
                sort_order="desc",
            )
            messages, _ = self.sessions.list_messages_for_dashboard(
                page=1,
                page_size=200,
                sort_by="ts",
                sort_order="desc",
            )
            message_total = sum(1 for item in messages if _is_on_date(item.get("ts"), current))
            memory_total = len(self._memory_items_for_date(current))
            activity_total = tick_total + session_total + message_total + memory_total
            if activity_total:
                result.append({
                    "date": current.isoformat(),
                    "label": current.strftime("%b %d"),
                    "count": activity_total,
                    "has_real_data": True,
                    "sources": {
                        "ticks": tick_total,
                        "sessions": len(sessions) if session_total else 0,
                        "messages": message_total,
                        "memory_items": memory_total,
                    },
                })
        if result:
            return result
        result.append({
            "date": today.isoformat(),
            "label": today.strftime("%b %d"),
            "count": 0,
            "has_real_data": False,
        })
        return result

    def _memory_items_for_date(self, target_date: date) -> list[dict[str, Any]]:
        try:
            items, _ = self.memory.list_items_for_dashboard(
                page=1,
                page_size=100,
                sort_by="updated_at",
                sort_order="desc",
            )
        except Exception:
            return []
        result: list[dict[str, Any]] = []
        for item in items:
            stamp = item.get("updated_at") or item.get("created_at") or item.get("happened_at")
            if not _is_on_date(stamp, target_date):
                continue
            result.append({
                "id": str(item.get("id") or ""),
                "time": _time_part(stamp),
                "title": str(item.get("summary") or item.get("id") or "Memory update"),
                "summary": str(item.get("summary") or ""),
                "type": str(item.get("memory_type") or "memory"),
                "status": str(item.get("status") or "active"),
                "source": "memory_admin",
                "is_sample": False,
            })
        return result[:12]

    def _tool_calls(self, ticks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for tick in ticks:
            tick_id = str(tick.get("tick_id") or "")
            if not tick_id:
                continue
            for step in self.proactive.list_tick_steps(tick_id):
                result_text = str(step.get("tool_result_text") or "")
                status = "failed" if _looks_failed(result_text) else "completed"
                calls.append({
                    "id": f"{tick_id}:{step.get('step_index')}",
                    "tool_name": str(step.get("tool_name") or "tool"),
                    "time": _time_part(tick.get("started_at")),
                    "status": status,
                    "summary": _preview(result_text or step.get("final_message_after"), 140),
                    "phase": str(step.get("phase") or ""),
                    "source": "proactive.tick_step_log",
                    "is_sample": False,
                })
        return calls[:20]

    def _missions(
        self,
        ticks: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        missions: list[dict[str, Any]] = []
        for tick in ticks:
            tick_id = str(tick.get("tick_id") or "")
            status = _mission_status(tick)
            summary = (
                tick.get("final_message")
                or tick.get("skip_reason")
                or tick.get("gate_exit")
                or "Proactive check completed"
            )
            missions.append({
                "id": tick_id,
                "title": _mission_title(tick),
                "status": status,
                "time": _time_part(tick.get("started_at")),
                "summary": _preview(summary, 150),
                "source": "proactive.tick_log",
                "is_sample": False,
            })
        return missions[:12]

    def _ephemera(
        self,
        ticks: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        sessions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for tick in ticks[:8]:
            items.append({
                "time": _time_part(tick.get("started_at")),
                "kind": "handled" if tick.get("terminal_action") else "flow",
                "text": _preview(
                    tick.get("final_message") or tick.get("skip_reason") or _mission_title(tick),
                    130,
                ),
                "source": "proactive.tick_log",
                "is_sample": False,
            })
        for session in sessions[:4]:
            items.append({
                "time": _time_part(session.get("updated_at")),
                "kind": "session",
                "text": f"更新会话 {session.get('key')}，累计 {session.get('message_count', 0)} 条消息",
                "source": "sessions",
                "is_sample": False,
            })
        for message in messages[:4]:
            items.append({
                "time": _time_part(message.get("ts")),
                "kind": str(message.get("role") or "message"),
                "text": _preview(message.get("content"), 120),
                "source": "sessions.messages",
                "is_sample": False,
            })
        return items[:14]

    def _next_actions(
        self,
        failures: list[dict[str, Any]],
        needs_approval: list[dict[str, Any]],
        memory_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for item in needs_approval[:3]:
            actions.append({
                "title": f"确认：{item['title']}",
                "summary": item["summary"],
                "priority": "high",
                "source": "needs_approval",
                "is_sample": False,
            })
        for item in failures[:3]:
            actions.append({
                "title": f"处理失败任务：{item['title']}",
                "summary": item["summary"],
                "priority": "high",
                "source": "failures",
                "is_sample": False,
            })
        if memory_items:
            actions.append({
                "title": "复核今日长期记忆",
                "summary": f"今日有 {len(memory_items)} 条记忆更新，可抽查摘要是否准确。",
                "priority": "medium",
                "source": "memory_admin",
                "is_sample": False,
            })
        return actions[:6]

    def _perspective(
        self,
        ticks: list[dict[str, Any]],
        memory_items: list[dict[str, Any]],
        *,
        activity_count: int,
    ) -> str:
        if ticks:
            completed = sum(1 for tick in ticks if _mission_status(tick) == "completed")
            pending = sum(1 for tick in ticks if _mission_status(tick) in {"pending", "needs_approval"})
            return f"今天主要完成了 {completed} 次主动检查，留下 {pending} 个待确认事项，并更新了 {len(memory_items)} 条记忆。"
        if activity_count:
            return f"今日记录到 {activity_count} 条 Agent 活动，暂无真实任务完成记录。"
        if memory_items:
            return f"今天更新了 {len(memory_items)} 条长期记忆，暂无 proactive tick 记录。"
        return "今日暂无可总结的 Agent 活动。"


def parse_workspace_date(raw: str | None) -> date:
    if not raw:
        return datetime.now().astimezone().date()
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError("date must be YYYY-MM-DD") from exc


def _day_bounds(value: date) -> tuple[str, str]:
    return f"{value.isoformat()}T00:00:00", f"{value.isoformat()}T23:59:59.999999"


def _is_on_date(value: Any, target_date: date) -> bool:
    text = str(value or "")
    return text.startswith(target_date.isoformat())


def _time_part(value: Any) -> str:
    text = str(value or "")
    if "T" in text:
        return text.split("T", 1)[1][:5]
    if " " in text:
        return text.split(" ", 1)[1][:5]
    return text[:5] if text else "--:--"


def _preview(value: Any, limit: int = 120) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _looks_failed(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("error", "failed", "traceback", "exception"))


def _mission_status(tick: dict[str, Any]) -> str:
    final = str(tick.get("final_message") or "")
    skip = str(tick.get("skip_reason") or "")
    gate = str(tick.get("gate_exit") or "")
    if _looks_failed(" ".join([final, skip, gate])):
        return "failed"
    if "approval" in skip.lower() or "confirm" in skip.lower() or "人工" in skip:
        return "needs_approval"
    if tick.get("terminal_action") or tick.get("finished_at"):
        return "completed"
    return "pending"


def _mission_title(tick: dict[str, Any]) -> str:
    action = tick.get("terminal_action") or tick.get("gate_exit")
    flow = "Drift" if tick.get("drift_entered") else "Proactive"
    if action:
        return f"{flow} {action}"
    return f"{flow} check"


def _runtime_status(
    ticks: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    needs_approval: list[dict[str, Any]],
) -> str:
    if failures:
        return "degraded"
    if needs_approval:
        return "idle"
    if ticks:
        return "running"
    return "idle"


def _status_text(completed_missions: int, activity_count: int) -> str:
    if completed_missions:
        return f"今日 Agent 已完成 {completed_missions} 项任务"
    if activity_count:
        return f"今日记录了 {activity_count} 条 Agent 活动"
    return "今日暂无真实任务完成记录"
