"""A custom runner's verifier reward is recorded as the verifier's.

Regression test: a branched run whose custom runner scored its children
with ``rollout.verify()`` still got ``tree.json`` said
``reward_source: "runner_return"``, the same label as a hard-coded
``return 1.0``. And the "a missing reward is unscored, never zero" rule held
only for the default runner: a custom runner returning ``0.0`` for a missing
verifier reward produced a scored 0.

Now, when a child's runner called ``verify()``: a returned number equal to
the canonical verifier reward is recorded as ``verifier``; a verifier that
produced no canonical reward makes the child unscored whatever the runner
returned. A runner returning ``None`` is unscored too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.branch_lineage import UnscoredChildError
from benchflow.environment.protocol import StateSnapshot
from benchflow.rollout import Rollout, RolloutConfig, Scene


class _Environment:
    async def snapshot(self):
        return StateSnapshot(id="env", path="/tmp/env")

    async def restore(self, _snap):
        return None


def _rollout(tmp_path: Path, monkeypatch, verify_result) -> Rollout:
    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy --agent")]
        )
    )
    rollout._environment = _Environment()
    rollout._rollout_dir = tmp_path / "run"
    rollout._rollout_dir.mkdir()

    async def fake_verify(self):
        # What the real verify() leaves behind: a counted call, and rewards
        # or an error.
        self._verify_calls += 1
        self._rewards = verify_result
        if verify_result is None:
            self._verifier_error = "reward.txt missing"
        return self._rewards

    monkeypatch.setattr(Rollout, "verify", fake_verify)
    return rollout


def _children(rollout: Rollout) -> list[dict]:
    document = json.loads((rollout._rollout_dir / "tree.json").read_text())
    return document["forks"][-1]["children"]


async def test_verifier_reward_from_a_custom_runner_is_labelled_verifier(
    tmp_path, monkeypatch
):
    rollout = _rollout(tmp_path, monkeypatch, {"reward": 0.75})

    async def run_child(node):
        rewards = await rollout.verify()
        return rewards["reward"]

    assert await rollout.branch(2, run_child) == 0.75
    assert [c["reward_source"] for c in _children(rollout)] == ["verifier"] * 2


async def test_transformed_number_stays_a_runner_return(tmp_path, monkeypatch):
    rollout = _rollout(tmp_path, monkeypatch, {"reward": 0.75})

    async def run_child(node):
        await rollout.verify()
        return 0.25

    assert await rollout.branch(2, run_child) == 0.25
    assert [c["reward_source"] for c in _children(rollout)] == ["runner_return"] * 2


async def test_runner_without_verify_is_a_runner_return(tmp_path, monkeypatch):
    rollout = _rollout(tmp_path, monkeypatch, {"reward": 0.75})

    async def run_child(node):
        return 1.0

    await rollout.branch(2, run_child)
    assert [c["reward_source"] for c in _children(rollout)] == ["runner_return"] * 2


@pytest.mark.parametrize("verify_result", [None, {}, {"other": 1.0}])
async def test_zero_for_a_missing_verifier_reward_is_unscored(
    tmp_path, monkeypatch, verify_result
):
    rollout = _rollout(tmp_path, monkeypatch, verify_result)

    async def run_child(node):
        rewards = await rollout.verify()
        return (rewards or {}).get("reward", 0.0)

    with pytest.raises(UnscoredChildError):
        await rollout.branch(2, run_child)
    first = _children(rollout)[0]
    assert first["status"] == "unscored"
    assert first["reward"] is None
    assert "reward" not in rollout.tree.root.children[0].state


async def test_runner_returning_none_is_unscored(tmp_path, monkeypatch):
    rollout = _rollout(tmp_path, monkeypatch, {"reward": 1.0})

    async def run_child(node):
        return None

    with pytest.raises(UnscoredChildError):
        await rollout.branch(2, run_child)
    assert _children(rollout)[0]["status"] == "unscored"
