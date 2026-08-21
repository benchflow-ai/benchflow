"""Regression tests for in-place branch-child result synthesis.

Guards "feat(branch): synthesize a full result.json for in-place branch
children" (docs/rollout-branching-rfc.md §3.4). A fresh-rollout child
(``env-ready``) is a Rollout of its own and leaves the standard artifact set
for free; an in-place child (``pre-verify`` / ``post-verify`` / cursor branch)
continues the parent instance and used to leave only ``provenance.json`` /
``reward.json`` plus the ``mounted/`` archive — so "what happened in this arm"
required cross-reading ``tree.json``. Now every completed in-place child of a
run-dir-bearing rollout leaves its own ``result.json`` / ``timing.json`` /
trajectory, built from the child's OWN state (scoped to zero before it runs),
with no parent bleed-through: a field the child did not produce is null/absent,
never copied from the parent.

Unit tests against fakes — no Docker, Daytona, or API keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.diagnostics import TransportClosedDiagnostic
from benchflow.environment.protocol import StateSnapshot
from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.task.paths import RolloutPaths


class FakeEnvironment:
    """Environment-plane stand-in recording snapshot/restore calls."""

    def __init__(self) -> None:
        self.snapshots: list[StateSnapshot] = []
        self.restored: list[StateSnapshot] = []

    async def snapshot(self) -> StateSnapshot:
        snap = StateSnapshot(id=f"snap-{len(self.snapshots) + 1}", path="/tmp/x")
        self.snapshots.append(snap)
        return snap

    async def restore(self, snap: StateSnapshot) -> None:
        self.restored.append(snap)


def _rollout(tmp_path: Path) -> Rollout:
    rollout = Rollout(
        RolloutConfig(task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy")])
    )
    rollout._environment = FakeEnvironment()
    run_dir = tmp_path / "run"
    rollout._rollout_dir = run_dir
    RolloutPaths(rollout_dir=run_dir).mkdir()
    return rollout


async def _run_parent_step(rollout: Rollout, monkeypatch) -> None:
    """One real execute() on the parent, so the fork has linear history."""

    async def fake_execute_prompts(*_a, **_kw):
        return [{"role": "agent", "text": "child-step"}], 1

    monkeypatch.setattr(rollout._planes, "execute_prompts", fake_execute_prompts)
    rollout._acp_client = object()
    await rollout.execute(["parent-prompt"])


def _fake_child_phases(monkeypatch, verify_outcomes: list[dict | None]) -> None:
    """Class-level connect/disconnect/verify fakes writing child-own state.

    ``verify()`` mutates the instance the way the real phase does — assigning
    ``_rewards`` / ``_verifier_error`` outright and writing into ``_timing`` —
    with per-child values drawn from ``verify_outcomes`` in fork order.
    """
    outcomes = iter(verify_outcomes)

    async def fake_connect(self):
        self._acp_client = object()

    async def fake_disconnect(self):
        # Mirror the real disconnect: the session counters rewind, so a fresh
        # child session's cumulative trajectory is read from its own start.
        self._acp_client = None
        self._session_tool_count = 0
        self._session_traj_count = 0

    async def fake_verify(self):
        rewards = next(outcomes)
        self._timing["verification"] = 99.0
        self._verifier_error = "child verifier: exit 0"
        self._diagnostics.set(
            TransportClosedDiagnostic(raw_message="child", transport_diagnosis="child")
        )
        self._rewards = rewards
        return self._rewards

    monkeypatch.setattr(Rollout, "connect", fake_connect)
    monkeypatch.setattr(Rollout, "disconnect", fake_disconnect)
    monkeypatch.setattr(Rollout, "verify", fake_verify)


def _child_dir(rollout: Rollout, parent, child) -> Path:
    return rollout._rollout_dir / "branches" / parent.id / "children" / child.id


async def test_in_place_children_leave_their_own_result_json(
    tmp_path: Path, monkeypatch
) -> None:
    """Each completed in-place child leaves a result.json of its OWN run.

    The red case this guards: only fresh-rollout (env-ready) children were
    first-class runs; an in-place ablation arm left no result.json at all.
    Reward, verifier error, timing and trajectory must all be the child's own
    — and the fields the child did not produce (the parent's error, the
    parent's timing keys, token usage) must be null/absent, never inherited.
    """
    rollout = _rollout(tmp_path)
    await _run_parent_step(rollout, monkeypatch)

    # The parent's own result-bearing state, as a linear run leaves it. None
    # of it may appear in a child's result.
    rollout._verifier_error = "parent verifier: 1 test failed"
    rollout._error = "parent agent: idle for 600s"
    rollout._timing["agent_setup"] = 41.5
    timing_before = dict(rollout._timing)
    parent = rollout._cursor

    _fake_child_phases(monkeypatch, [{"reward": 1.0}, {"reward": 0.0}])

    assert await rollout.branch(2) == 0.5

    children = parent.children
    assert len(children) == 2
    for child, expected_reward in zip(children, [1.0, 0.0], strict=True):
        child_dir = _child_dir(rollout, parent, child)
        result = json.loads((child_dir / "result.json").read_text())
        # The child's own observations.
        assert result["rewards"] == {"reward": expected_reward}
        assert result["verifier_error"] == "child verifier: exit 0"
        assert result["rollout_name"] == child.id
        assert result["task_name"] == "task"
        # No parent bleed-through: fields the child never produced are null.
        assert result["error"] is None
        assert result["agent_result"]["n_input_tokens"] is None
        assert result["agent_result"]["usage_source"] == "unavailable"
        # result.json names its own lineage, without tree.json.
        assert result["source"]["kind"] == "benchflow-branch"
        assert result["source"]["branch_stage"] == f"cursor:{parent.id}"
        # The child's own timing: its verify wrote 99.0; the parent's
        # agent_setup is the parent's, never copied.
        timing = json.loads((child_dir / "timing.json").read_text())
        assert timing["verification"] == 99.0
        assert "agent_setup" not in timing
        assert timing["total"] >= 0
        # The trajectory is the child's continuation only — one step, not the
        # parent's history relabelled.
        lines = (
            (child_dir / "trajectory" / "acp_trajectory.jsonl")
            .read_text()
            .strip()
            .splitlines()
        )
        assert len(lines) == 1
        # reward.json (the lineage artifact) still stands beside it.
        assert json.loads((child_dir / "reward.json").read_text()) == {
            "reward": expected_reward
        }

    # The parent's reported state is still the parent's own (the isolation
    # invariant this feature must not weaken).
    assert rollout._timing == timing_before
    assert rollout._verifier_error == "parent verifier: 1 test failed"
    assert rollout._error == "parent agent: idle for 600s"


async def test_an_unscored_in_place_child_still_leaves_its_result(
    tmp_path: Path, monkeypatch
) -> None:
    """A child that ran but was never scored is evidenced, not invented.

    Its result.json exists with ``rewards: null`` (a missing score is not a
    zero) and its own verifier error; reward.json is absent, exactly as the
    lineage writer has always kept it.
    """
    rollout = _rollout(tmp_path)
    await _run_parent_step(rollout, monkeypatch)
    parent = rollout._cursor
    _fake_child_phases(monkeypatch, [{}, {}])

    assert await rollout.branch(2) is None

    for child in parent.children:
        child_dir = _child_dir(rollout, parent, child)
        result = json.loads((child_dir / "result.json").read_text())
        assert result["rewards"] is None
        assert result["verifier_error"] == "child verifier: exit 0"
        assert not (child_dir / "reward.json").exists()


async def test_result_synthesis_failure_never_costs_the_reward(
    tmp_path: Path, monkeypatch
) -> None:
    """The writer is best-effort by contract, like every branch artifact.

    A result set that cannot be built (full disk, a field that will not
    serialize) is logged and skipped — the fork still scores, the node still
    carries its reward, and reward.json is still written.
    """
    rollout = _rollout(tmp_path)
    await _run_parent_step(rollout, monkeypatch)
    parent = rollout._cursor
    _fake_child_phases(monkeypatch, [{"reward": 1.0}, {"reward": 1.0}])

    def boom(*_a, **_kw):
        raise OSError("No space left on device")

    monkeypatch.setattr("benchflow.rollout._results._build_rollout_result", boom)

    assert await rollout.branch(2) == 1.0

    for child in parent.children:
        child_dir = _child_dir(rollout, parent, child)
        assert not (child_dir / "result.json").exists()
        assert json.loads((child_dir / "reward.json").read_text()) == {"reward": 1.0}


async def test_a_child_that_raises_hard_leaves_no_result_json(
    tmp_path: Path, monkeypatch
) -> None:
    """A hard child failure ends the fork before a result can be honest.

    The failing child's evidence is its ``mounted/`` archive and its
    provenance (the partial-lineage guarantee); a synthesized result for a
    child that died mid-phase would describe a run that never completed.
    """
    rollout = _rollout(tmp_path)
    await _run_parent_step(rollout, monkeypatch)
    parent = rollout._cursor

    async def run_child(child):
        raise RuntimeError("agent connection lost")

    with pytest.raises(RuntimeError, match="agent connection lost"):
        await rollout.branch(2, run_child=run_child)

    child = parent.children[0]
    child_dir = _child_dir(rollout, parent, child)
    assert (child_dir / "provenance.json").is_file()
    assert not (child_dir / "result.json").exists()


async def test_fresh_rollout_children_are_not_double_synthesized(
    tmp_path: Path, monkeypatch
) -> None:
    """An ``env-ready`` fresh child writes its own result — the engine must
    not overwrite it with a synthesized in-place one."""
    import benchflow.rollout_branch as rollout_branch
    from benchflow.sandbox.protocol import SandboxImage
    from benchflow.trajectories.tree import Step

    class FakeSnapSandbox:
        supports_snapshot = True

        async def snapshot(self, name: str | None = None) -> SandboxImage:
            return SandboxImage(provider="fake", ref="bf-snap-1")

        async def restore(self, image: SandboxImage) -> None:
            return None

    rollout = _rollout(tmp_path)
    rollout._env = FakeSnapSandbox()
    await rollout.mark_stage("env-ready", snapshot_layers={"environment", "sandbox"})
    rollout._cursor = rollout._tree.advance(rollout._cursor, Step(id="s1"))

    synthesized: list[str] = []
    real_write = rollout_branch.write_in_place_child_result

    def recording_write(*args, **kwargs):
        synthesized.append(kwargs["child"].id)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(rollout_branch, "write_in_place_child_result", recording_write)

    def make_runner(rollout_arg, **kwargs):
        async def _runner(child):
            return 1.0

        return _runner

    monkeypatch.setattr(rollout_branch, "make_fresh_child_runner", make_runner)

    assert await rollout.branch_at_stage("env-ready", 2) == 1.0
    assert synthesized == []
