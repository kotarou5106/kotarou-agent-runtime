from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.api import create_dashboard_app
from proactive_system.state import ProactiveStateStore
from storage.sessions.store import SessionStore
from tests.memory_fakes import FakeMemoryEngine


@pytest.fixture
def app_client(tmp_path: Path):
    memory = FakeMemoryEngine()
    with TestClient(
        create_dashboard_app(
            tmp_path,
            memory_admin=memory,
            memory_store=None,
        )
    ) as client:
        yield client


def test_daily_workspace_endpoint_returns_sample_fallback(app_client: TestClient) -> None:
    response = app_client.get("/api/dashboard/daily-workspace?date=2026-06-12")

    assert response.status_code == 200
    payload = response.json()
    assert payload["date"] == "2026-06-12"
    assert payload["agent_name"] == "Kotarou"
    assert payload["is_sample"] is True
    assert payload["no_real_data"] is True
    assert "missions" in payload["sample_fallback_fields"]
    assert len(payload["archive_dates"]) == 1
    assert payload["archive_dates"][0]["has_real_data"] is False
    assert payload["missions"] == []
    assert payload["ephemera"] == []
    assert payload["memory_items"] == []
    assert payload["tool_calls"] == []
    assert payload["next_actions"] == []
    assert payload["perspective"] == "今日暂无可总结的 Agent 活动。"
    assert payload["sources"]["tool_calls"] == "proactive tick_step_log"
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "Sample:" not in encoded
    assert "sample fallback" not in encoded
    assert "岗位监控" not in encoded
    assert "Telegram 摘要" not in encoded
    assert "记忆整理" not in encoded
    assert "配置 VPS" not in encoded
    assert "补充 Nango" not in encoded


def test_daily_workspace_uses_proactive_tick_and_step_data(tmp_path: Path) -> None:
    state = ProactiveStateStore(tmp_path / "proactive.db")
    state.record_tick_log_finish(
        tick_id="tick-1",
        session_key="telegram:jobs",
        started_at="2026-06-12T09:00:00+00:00",
        finished_at="2026-06-12T09:01:00+00:00",
        gate_exit=None,
        terminal_action="reply",
        skip_reason="",
        steps_taken=1,
        alert_count=1,
        content_count=3,
        context_count=2,
        interesting_ids=["job-1"],
        discarded_ids=[],
        cited_ids=["job-1"],
        drift_entered=False,
        final_message="整理了 Telegram 岗位频道摘要。",
    )
    state.record_tick_step_log(
        tick_id="tick-1",
        step_index=1,
        phase="proactive",
        tool_name="telegram.search",
        tool_call_id="call-1",
        tool_args={"channel": "jobs"},
        tool_result_text="Found three relevant job posts.",
        terminal_action_after="reply",
        skip_reason_after="",
        interesting_ids_after=["job-1"],
        discarded_ids_after=[],
        cited_ids_after=["job-1"],
        final_message_after="整理了 Telegram 岗位频道摘要。",
    )
    state.close()

    memory = FakeMemoryEngine()
    with TestClient(
        create_dashboard_app(
            tmp_path,
            memory_admin=memory,
            memory_store=None,
        )
    ) as client:
        payload = client.get("/api/dashboard/daily-workspace?date=2026-06-12").json()

    assert payload["status"] == "running"
    assert payload["missions"][0]["source"] == "proactive.tick_log"
    assert payload["missions"][0]["status"] == "completed"
    assert payload["missions"][0]["is_sample"] is False
    assert payload["tool_calls"][0]["tool_name"] == "telegram.search"
    assert payload["tool_calls"][0]["is_sample"] is False
    assert "missions" not in payload["sample_fallback_fields"]
    assert "tool_calls" not in payload["sample_fallback_fields"]


def test_daily_workspace_session_activity_updates_archive_without_missions(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.upsert_session(
        "telegram:6842574455",
        created_at="2026-06-12T08:00:00+00:00",
        updated_at="2026-06-12T08:30:00+00:00",
        last_consolidated=0,
        metadata={},
    )
    store.insert_message(
        "telegram:6842574455",
        role="assistant",
        content="整理了真实会话活动。",
        ts="2026-06-12T08:20:00+00:00",
        seq=0,
    )
    store.upsert_session(
        "telegram_daily_summary",
        created_at="2026-06-12T09:00:00+00:00",
        updated_at="2026-06-12T09:15:00+00:00",
        last_consolidated=0,
        metadata={},
    )
    store.insert_message(
        "telegram_daily_summary",
        role="assistant",
        content="生成了真实日报摘要。",
        ts="2026-06-12T09:10:00+00:00",
        seq=0,
    )
    store.close()

    memory = FakeMemoryEngine()
    with TestClient(
        create_dashboard_app(
            tmp_path,
            memory_admin=memory,
            memory_store=None,
        )
    ) as client:
        payload = client.get("/api/dashboard/daily-workspace?date=2026-06-12").json()

    assert payload["archive_dates"][0]["date"] == "2026-06-12"
    assert payload["archive_dates"][0]["has_real_data"] is True
    assert payload["archive_dates"][0]["count"] >= 2
    assert payload["missions"] == []
    assert payload["failures"] == []
    assert payload["needs_approval"] == []
    assert payload["ephemera"]
    assert "telegram:6842574455" in json.dumps(payload["ephemera"], ensure_ascii=False)
    assert payload["status_text"] == f"今日记录了 {len(payload['ephemera'])} 条 Agent 活动"
    assert "已完成" not in payload["status_text"]
    assert "完成了" not in payload["perspective"]
    assert "暂无真实任务完成记录" in payload["perspective"]


def test_daily_workspace_rejects_invalid_date(app_client: TestClient) -> None:
    response = app_client.get("/api/dashboard/daily-workspace?date=2026-99-99")

    assert response.status_code == 400
    assert response.json()["detail"] == "date must be YYYY-MM-DD"


def test_daily_showcase_routes_serve_dashboard_app(app_client: TestClient) -> None:
    for path in ("/daily/showcase", "/workspace/showcase"):
        response = app_client.get(path)

        assert response.status_code == 200
        assert "Kotarou Dashboard" in response.text
