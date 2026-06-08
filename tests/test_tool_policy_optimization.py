from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from agent_runtime.learning.bandit import BanditPolicy
from agent_runtime.learning.policy_store import ToolPolicyStore
from agent_runtime.learning.reward_signal import ToolRewardSignal
from agent_runtime.learning.tool_policy import (
    ToolSelectionPolicy,
    build_default_tool_selection_policy,
    main as tool_policy_main,
)
from agent_runtime.tools.base import Tool
from agent_runtime.tools.registry import ToolRegistry


class _StubTool(Tool):
    def __init__(self, name: str, description: str) -> None:
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **_: Any) -> str:
        return "ok"


def test_reward_signal_rule_calculation() -> None:
    assert ToolRewardSignal.from_tool_status(status="success").compute_reward() == 1.0
    assert ToolRewardSignal.from_tool_status(status="error").compute_reward() == -1.0
    assert (
        ToolRewardSignal.from_tool_status(
            status="denied",
            error_type="tool_loop_guard",
        ).compute_reward()
        == -0.5
    )
    assert (
        ToolRewardSignal.from_tool_status(
            status="success",
            user_feedback="thumbs_up",
            task_completed=True,
        ).compute_reward()
        == 4.0
    )
    assert (
        ToolRewardSignal.from_tool_status(
            status="success",
            user_feedback="thumbs_down",
        ).compute_reward()
        == -1.0
    )


def test_policy_store_persists_stats(tmp_path: Path) -> None:
    path = tmp_path / "tool_policy.json"
    store = ToolPolicyStore(path)

    store.update("knowledge_search", 1.0, success=True, created_at="2026-01-01T00:00:00+00:00")
    store.update("knowledge_search", -1.0, success=False, created_at="2026-01-02T00:00:00+00:00")

    reloaded = ToolPolicyStore(path)
    stats = reloaded.get_tool_stats("knowledge_search")
    assert stats.count == 2
    assert stats.avg_reward == 0.0
    assert stats.success_rate == 0.5
    assert stats.last_used_at == "2026-01-02T00:00:00+00:00"


def test_bandit_policy_ranks_by_reward_and_keeps_cold_start_explorable(
    tmp_path: Path,
) -> None:
    store = ToolPolicyStore(tmp_path / "policy.json")
    policy = BanditPolicy(store, exploration_weight=0.0, cold_start_score=0.25)
    store.update("bad_tool", -1.0, success=False)
    store.update("good_tool", 1.0, success=True)
    store.update("good_tool", 1.0, success=True)

    ranked = policy.rank(["bad_tool", "new_tool", "good_tool"])

    assert [item.tool_name for item in ranked] == [
        "good_tool",
        "new_tool",
        "bad_tool",
    ]
    assert ranked[1].stats.count == 0


def test_tool_success_and_failure_move_average(tmp_path: Path) -> None:
    store = ToolPolicyStore(tmp_path / "policy.json")

    first = store.update("shell", 1.0, success=True)
    second = store.update("shell", -1.0, success=False)

    assert first.avg_reward == 1.0
    assert second.avg_reward < first.avg_reward
    assert second.success_rate == 0.5


def test_policy_disabled_does_not_change_registry_search_order() -> None:
    registry = _registry_with_tools(selection_policy=None)

    results = registry.search("search", top_k=2)

    assert [item["name"] for item in results] == ["alpha_search", "beta_search"]
    assert "policy_score" not in results[0]


def test_policy_enabled_reranks_candidate_tool_order(tmp_path: Path) -> None:
    store = ToolPolicyStore(tmp_path / "policy.json")
    store.update("beta_search", 1.0, success=True)
    store.update("beta_search", 1.0, success=True)
    store.update("alpha_search", -1.0, success=False)
    registry = _registry_with_tools(
        selection_policy=ToolSelectionPolicy(
            store,
            BanditPolicy(store, exploration_weight=0.0, cold_start_score=0.0),
        )
    )

    results = registry.search("search", top_k=2)

    assert [item["name"] for item in results] == ["beta_search", "alpha_search"]
    assert results[0]["rank_before_policy"] == 2
    assert results[0]["rank_after_policy"] == 1
    assert results[0]["policy_score"] > results[1]["policy_score"]


def test_env_switch_builds_default_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_TOOL_POLICY_OPTIMIZATION", "true")
    monkeypatch.setenv("TOOL_POLICY_STORE_PATH", str(tmp_path / "env_policy.json"))

    policy = build_default_tool_selection_policy()

    assert policy is not None
    policy.update("read_file", ToolRewardSignal.from_tool_status(status="success"))
    assert (tmp_path / "env_policy.json").exists()


def test_cli_stats_and_reset(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    store = ToolPolicyStore(path)
    store.update("knowledge_search", 1.0, success=True)

    out = StringIO()
    with redirect_stdout(out):
        assert tool_policy_main(["--stats", "--store", str(path)]) == 0
    payload = json.loads(out.getvalue())
    assert payload["tools"][0]["tool_name"] == "knowledge_search"

    out = StringIO()
    with redirect_stdout(out):
        assert tool_policy_main(["--reset", "--store", str(path)]) == 0
    assert ToolPolicyStore(path).list_stats() == []


def _registry_with_tools(
    *,
    selection_policy: ToolSelectionPolicy | None,
) -> ToolRegistry:
    registry = ToolRegistry(selection_policy=selection_policy)
    registry.register(_StubTool("alpha_search", "search candidate"))
    registry.register(_StubTool("beta_search", "search candidate"))
    return registry
