from __future__ import annotations

import json
import subprocess
import sys

from memory_system.policy.memory_policy import MemoryPolicy
from memory_system.policy.policy_store import MemoryPolicyStore
from memory_system.policy.reward_signal import (
    MemoryPolicyEvent,
    calculate_memory_reward,
)
from memory_system.store import MemoryStore2


def test_memory_reward_signal_rules_are_explainable():
    assert calculate_memory_reward("retrieved") == 0.05
    assert calculate_memory_reward("injected") == 0.10
    assert calculate_memory_reward("used") == 0.50
    assert calculate_memory_reward("corrected") == -1.00
    assert calculate_memory_reward("superseded") == -0.75
    assert calculate_memory_reward("deleted") == -1.00
    assert calculate_memory_reward("retrieved", user_feedback="thumbs_up") == 2.00
    assert calculate_memory_reward("retrieved", user_feedback="thumbs_down") == -2.00
    assert calculate_memory_reward("retrieved", explicit_reward=1.25) == 1.25


def test_memory_policy_store_persists_stats(tmp_path):
    path = tmp_path / "memory_policy.json"
    store = MemoryPolicyStore(path)
    store.update("mem-a", MemoryPolicyEvent("mem-a", "retrieved"))
    store.update("mem-a", MemoryPolicyEvent("mem-a", "injected"))
    store.update("mem-a", MemoryPolicyEvent("mem-a", "used"))

    reloaded = MemoryPolicyStore(path).get_memory_stats("mem-a")

    assert reloaded.retrieved_count == 1
    assert reloaded.injected_count == 1
    assert reloaded.used_count == 1
    assert reloaded.avg_reward > 0
    assert reloaded.reliability > 0.5
    assert reloaded.retrieval_boost > 0


def test_memory_policy_negative_feedback_lowers_stats(tmp_path):
    store = MemoryPolicyStore(tmp_path / "memory_policy.json")
    store.update("mem-a", MemoryPolicyEvent("mem-a", "used"))
    positive = store.get_memory_stats("mem-a")
    store.update("mem-a", MemoryPolicyEvent("mem-a", "corrected"))
    store.update("mem-a", MemoryPolicyEvent("mem-a", "superseded"))
    negative = store.get_memory_stats("mem-a")

    assert negative.corrected_count == 1
    assert negative.superseded_count == 1
    assert negative.reliability < positive.reliability
    assert negative.retrieval_boost < positive.retrieval_boost


def test_memory_policy_reranks_vector_results_when_enabled(tmp_path):
    policy = MemoryPolicy(MemoryPolicyStore(tmp_path / "policy.json"))
    store = MemoryStore2(tmp_path / "m.db", memory_policy=policy)
    store.upsert_item("preference", "slightly lower semantic but useful", [0.90, 0.436], extra={})
    useful_id = str(store.list_by_type("preference")[0]["id"])
    store.upsert_item("preference", "higher semantic cold item", [0.95, 0.312], extra={})

    baseline = MemoryStore2(tmp_path / "baseline.db")
    baseline.upsert_item("preference", "slightly lower semantic but useful", [0.90, 0.436], extra={})
    baseline.upsert_item("preference", "higher semantic cold item", [0.95, 0.312], extra={})
    assert baseline.vector_search([1.0, 0.0], top_k=2, score_threshold=0.0)[0]["summary"] == "higher semantic cold item"

    for _ in range(4):
        policy.record_event(useful_id, "injected")
        policy.record_event(useful_id, "used")

    results = store.vector_search([1.0, 0.0], top_k=2, score_threshold=0.0)

    assert results[0]["summary"] == "slightly lower semantic but useful"
    assert results[0]["_score_debug"]["memory_policy_boost"] > 0


def test_memory_policy_cold_start_is_neutral(tmp_path):
    policy = MemoryPolicy(MemoryPolicyStore(tmp_path / "policy.json"))
    ranked = policy.rank_results(
        [
            {"id": "a", "summary": "a", "score": 0.8},
            {"id": "b", "summary": "b", "score": 0.7},
        ]
    )

    assert [item["id"] for item in ranked] == ["a", "b"]
    assert ranked[0]["_score_debug"]["memory_policy_boost"] == 0.0


def test_memory_store_records_supersede_and_delete_events(tmp_path):
    policy = MemoryPolicy(MemoryPolicyStore(tmp_path / "policy.json"))
    store = MemoryStore2(tmp_path / "m.db", memory_policy=policy)
    store.upsert_item("procedure", "old rule", [1.0, 0.0], extra={})
    item_id = str(store.list_by_type("procedure")[0]["id"])

    store.mark_superseded(item_id)
    assert policy.store.get_memory_stats(item_id).superseded_count == 1

    store.upsert_item("procedure", "temporary rule", [0.9, 0.1], extra={})
    delete_id = [
        str(item["id"])
        for item in store.list_by_type("procedure")
        if str(item["id"]) != item_id
    ][0]
    assert store.delete_item(delete_id)
    assert policy.store.get_memory_stats(delete_id).deleted_count == 1


def test_memory_policy_disabled_does_not_change_store_ranking(tmp_path, monkeypatch):
    monkeypatch.delenv("ENABLE_MEMORY_POLICY_OPTIMIZATION", raising=False)
    store = MemoryStore2(tmp_path / "m.db")
    store.upsert_item("preference", "lower semantic", [0.90, 0.436], extra={})
    store.upsert_item("preference", "higher semantic", [0.95, 0.312], extra={})

    results = store.vector_search([1.0, 0.0], top_k=2, score_threshold=0.0)

    assert results[0]["summary"] == "higher semantic"
    assert "memory_policy_boost" not in results[0].get("_score_debug", {})


def test_memory_policy_cli_stats_and_reset(tmp_path):
    path = tmp_path / "policy.json"
    store = MemoryPolicyStore(path)
    store.update("mem-a", MemoryPolicyEvent("mem-a", "used"))

    stats_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "memory_system.policy.memory_policy",
            "--store",
            str(path),
            "--stats",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(stats_run.stdout)
    assert payload["memories"][0]["memory_id"] == "mem-a"

    reset_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "memory_system.policy",
            "--store",
            str(path),
            "--reset",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(reset_run.stdout)["reset"] is True
    assert MemoryPolicyStore(path).list_stats() == []
