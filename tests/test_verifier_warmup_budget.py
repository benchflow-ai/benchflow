"""The verifier warm-up never bills the agent's turn (PERF-C review M1).

A separate verifier's sandbox warmed up as the agent's turn begins is
charged against ``max_sandbox_seconds`` only from verify() entry, as one
started there was before the warm-up; its seconds during the turn are
recorded as ``verifier_sandbox_warmup``. Trials are SV3's scenarios (a real
Rollout with fake sandboxes, tests/test_sv_overlap.py).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from benchflow._utils import timing_spans
from benchflow.budget import Budget, BudgetGuard
from tests import test_sv_overlap as svo

AGENT_TURN = 1.0


async def _live_spend(
    tmp_path, monkeypatch, *, warm: bool, prepare: str | None = None
) -> float:
    sc = await svo._scenario(tmp_path, monkeypatch, trajectory=[])
    if prepare is not None:  # after the scenario, which clears it
        monkeypatch.setenv("BENCHFLOW_VERIFIER_PREPARE", prepare)
    guard = BudgetGuard(Budget(max_sandbox_seconds=10_000))
    with guard.track_trial("t"):
        if warm:
            sc.rollout._begin_verifier_warmup()
        await asyncio.sleep(AGENT_TURN)  # the agent's turn
        await svo._trial(sc)
    return guard.spent()["sandbox_seconds"]


@pytest.mark.asyncio
async def test_the_warm_up_does_not_charge_the_agents_turn(tmp_path, monkeypatch):
    spent = await _live_spend(tmp_path, monkeypatch, warm=True)
    # Capture and scoring with fake sandboxes take milliseconds; the warm-up
    # charged the whole turn (about AGENT_TURN) before.
    assert spent < AGENT_TURN / 2


@pytest.mark.asyncio
async def test_the_recorded_overlap_leaves_out_the_warm_up(tmp_path, monkeypatch):
    sc = await svo._scenario(tmp_path, monkeypatch, trajectory=[], clock=True)
    sc.capture_waits_for_verifier = True
    sc.seconds = {"verifier.start": 3.0, "agent.stop": 2.0, "verifier.test_sh": 4.0}
    sc.rollout._begin_verifier_warmup()
    await sc.rollout._verifier_warmup.start_task  # created and started
    sc.clock.now += 600.0  # the agent's turn
    await svo._trial(sc)
    trial = sc.rollout._rollout_dir
    (trial / "timing.json").write_text(
        json.dumps(timing_spans.rounded(sc.rollout.timing))
    )
    guard = BudgetGuard(Budget(max_sandbox_seconds=10_000))
    guard.start("t")
    guard.attempt_done("t", SimpleNamespace(rollout_dir=trial))
    # The agent's stop (2 s) after verify() entry, as without the warm-up.
    assert guard.spent()["sandbox_seconds"] == pytest.approx(2.0, abs=0.5)
    assert sc.rollout.timing["verifier_sandbox_warmup"] >= 600.0


@pytest.mark.asyncio
async def test_prepare_entry_spends_as_the_base(tmp_path, monkeypatch):
    base = await _live_spend(tmp_path / "base", monkeypatch, warm=False)
    entry = await _live_spend(
        tmp_path / "entry", monkeypatch, warm=True, prepare="entry"
    )
    assert entry == pytest.approx(base, abs=0.2)
