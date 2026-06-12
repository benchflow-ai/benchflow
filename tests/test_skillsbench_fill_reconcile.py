from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_reconcile_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments" / "skillsbench-fill" / "reconcile.py"
    spec = importlib.util.spec_from_file_location("skillsbench_fill_reconcile", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reconcile_credit_gate_requires_strict_timeout_overlay() -> None:
    """Guards PR #638 follow-up so raw partial timeouts stay uncredited."""
    reconcile = _load_reconcile_module()
    cfg = {
        "environment": "daytona",
        "include_task_skills": False,
        "agent_env": {"LLM_REASONING_EFFORT": "max"},
    }
    result = {
        "error": "Agent timed out after 900s",
        "partial_trajectory": True,
        "rewards": {"reward": 0.0},
        "timing": {"total": 901.0},
        "agent_result": {"total_tokens": 123},
    }

    ok, reason = reconcile.credit_gate(cfg, result, "minimax-m3", "without")
    assert ok is False
    assert reason == "error"

    ok, reason = reconcile.credit_gate(
        cfg,
        result,
        "minimax-m3",
        "without",
        {
            "accepted_normal_timeout": True,
            "timeout_complete_artifacts": True,
            "checks": {"llm_final_response_ok": False},
        },
    )
    assert ok is False
    assert reason == "error"

    ok, reason = reconcile.credit_gate(
        cfg,
        result,
        "minimax-m3",
        "without",
        {
            "accepted_normal_timeout": True,
            "timeout_complete_artifacts": True,
            "checks": {"llm_final_response_ok": True},
        },
    )
    assert ok is True
    assert reason == "accepted_timeout"
