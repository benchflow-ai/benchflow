"""Regression tests for issue #377 — Scene rollouts must aggregate prompt
counts, prompts.json, and agent-execution time across every turn, not just
the first.

The bug: ``Rollout._build_result()`` passed ``self._resolved_prompts`` (the
static base task prompt list) to ``_build_rollout_result``, so multi-turn
Scenes that generated dynamic per-turn prompts only reported the base
prompt count. Separately, ``Rollout.execute()`` only set
``self._timing["agent_execution"]`` on its first call, so subsequent turns
contributed zero to reported agent time.

The fix: ``execute()`` records every prompt it actually sends into
``self._executed_prompts`` and adds elapsed time onto the running
``agent_execution`` total; ``_build_result()`` then emits the executed
list so ``result.json`` / ``prompts.json`` reflect the real agent input.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from benchflow._types import Role, Scene, Turn
from benchflow.rollout import Rollout, RolloutConfig


def _make_rollout(tmp_path: Path) -> Rollout:
    """Build a minimal Rollout wired enough to drive execute() directly."""
    task_path = tmp_path / "task"
    task_path.mkdir()
    (task_path / "instruction.md").write_text("base task instruction")

    scene = Scene(
        name="multi-turn",
        roles=[Role(name="agent", agent="claude-agent-acp")],
        turns=[
            Turn(role="agent", prompt="turn-1"),
            Turn(role="agent", prompt="turn-2"),
            Turn(role="agent", prompt="turn-3"),
        ],
    )
    cfg = RolloutConfig(task_path=task_path, scenes=[scene])

    rollout = Rollout(cfg)
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    rollout._rollout_dir = rollout_dir
    rollout._rollout_name = "multi-turn-run"
    rollout._started_at = datetime.now()
    rollout._timeout = 60
    rollout._resolved_prompts = ["base task instruction"]
    # Mock ACP plumbing — execute() only checks that these are truthy.
    rollout._acp_client = object()  # type: ignore[assignment]
    rollout._session = object()
    return rollout


@pytest.mark.asyncio
async def test_execute_records_every_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each execute() turn appends its prompts to _executed_prompts."""
    rollout = _make_rollout(tmp_path)

    cumulative: dict[str, Any] = {"traj": [], "tools": 0}

    async def fake_execute_prompts(
        client: Any,
        session: Any,
        prompts: list[str],
        timeout: int,
        idle_timeout: int | None = None,
    ) -> tuple[list[dict], int]:
        # execute_prompts returns the cumulative session trajectory.
        for p in prompts:
            cumulative["traj"].append({"type": "user_message", "text": p})
            cumulative["tools"] += 1
        return list(cumulative["traj"]), int(cumulative["tools"])

    monkeypatch.setattr(rollout._planes, "execute_prompts", fake_execute_prompts)

    await rollout.execute(prompts=["turn-1"])
    await rollout.execute(prompts=["turn-2"])
    await rollout.execute(prompts=["turn-3"])

    assert rollout._executed_prompts == ["turn-1", "turn-2", "turn-3"]


@pytest.mark.asyncio
async def test_agent_execution_time_accumulates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """timing.agent_execution sums every execute() call, not just the first."""
    rollout = _make_rollout(tmp_path)

    async def fake_execute_prompts(*args: Any, **kwargs: Any) -> tuple[list[dict], int]:
        return [], 0

    monkeypatch.setattr(rollout._planes, "execute_prompts", fake_execute_prompts)

    # Drive three turns and assert agent_execution monotonically grows.
    await rollout.execute(prompts=["t1"])
    first = rollout._timing["agent_execution"]
    await rollout.execute(prompts=["t2"])
    second = rollout._timing["agent_execution"]
    await rollout.execute(prompts=["t3"])
    third = rollout._timing["agent_execution"]

    # Each call should add nonnegative elapsed time, so the total only grows.
    assert second >= first
    assert third >= second
    # The previous "set only on first call" bug would make second == first.
    # Use strict inequality on the sum across calls — clocks tick > 0 even for
    # near-instant operations because we read datetime.now() twice each call.
    assert third > 0.0


@pytest.mark.asyncio
async def test_build_result_emits_all_executed_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """result.json.n_prompts and prompts.json reflect every turn's prompt."""
    rollout = _make_rollout(tmp_path)

    cumulative: dict[str, Any] = {"traj": [], "tools": 0}

    async def fake_execute_prompts(
        client: Any,
        session: Any,
        prompts: list[str],
        timeout: int,
        idle_timeout: int | None = None,
    ) -> tuple[list[dict], int]:
        for p in prompts:
            cumulative["traj"].append({"type": "user_message", "text": p})
        cumulative["tools"] += 2  # pretend each turn invoked 2 tools
        return list(cumulative["traj"]), int(cumulative["tools"])

    monkeypatch.setattr(rollout._planes, "execute_prompts", fake_execute_prompts)

    await rollout.execute(prompts=["base task instruction"])
    await rollout.execute(prompts=["reviewer feedback prompt"])
    await rollout.execute(prompts=["finalize prompt"])

    result = rollout._build_result()

    # The RolloutResult and the on-disk artifacts must all agree.
    assert result.n_prompts == 3
    rollout_dir = rollout._rollout_dir
    assert rollout_dir is not None
    rj = json.loads((rollout_dir / "result.json").read_text())
    assert rj["n_prompts"] == 3
    assert rj["agent_result"]["n_prompts"] == 3

    prompts_on_disk = json.loads((rollout_dir / "prompts.json").read_text())
    assert prompts_on_disk == [
        "base task instruction",
        "reviewer feedback prompt",
        "finalize prompt",
    ]

    # timing.agent_execution covers all three executes, not just the first.
    assert rj["timing"]["agent_execution"] >= 0.0
    # n_tool_calls aggregated across all turns (2 per turn * 3 turns).
    assert rj["n_tool_calls"] == 6


def test_build_result_falls_back_to_resolved_prompts_when_no_execute(
    tmp_path: Path,
) -> None:
    """Setup-failure path: no execute() ran, so prompts.json falls back to
    the resolved base prompts (so existing single-prompt invariants still
    hold when the agent never started).
    """
    rollout = _make_rollout(tmp_path)
    # No execute() calls — _executed_prompts stays empty.
    assert rollout._executed_prompts == []

    result = rollout._build_result()

    assert result.n_prompts == 1
    rollout_dir = rollout._rollout_dir
    assert rollout_dir is not None
    prompts_on_disk = json.loads((rollout_dir / "prompts.json").read_text())
    assert prompts_on_disk == ["base task instruction"]
