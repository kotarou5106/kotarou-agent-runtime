from __future__ import annotations

from agent_runtime.config import _load_orchestration_config


def test_orchestration_config_defaults_to_native() -> None:
    cfg = _load_orchestration_config({})

    assert cfg.backend == "native"
    assert cfg.checkpoint_enabled is True
    assert cfg.interrupt_high_risk_tools is True


def test_orchestration_config_accepts_langgraph() -> None:
    cfg = _load_orchestration_config(
        {
            "backend": "langgraph",
            "checkpoint_enabled": False,
            "interrupt_high_risk_tools": False,
        }
    )

    assert cfg.backend == "langgraph"
    assert cfg.checkpoint_enabled is False
    assert cfg.interrupt_high_risk_tools is False
