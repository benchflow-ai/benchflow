"""A transient failure in one branch child no longer sinks the fork.

A sandbox timeout while connecting the agent in one child's restored
sandbox failed the whole in-place fork; later children never started and V
was lost. Now:

- a child whose runner failed *before the agent did anything* (no step
  recorded on its node, verifier not run) is retried once from the snapshot,
  in place or in its own sandbox; the record keeps ``attempts`` and
  ``retried_after`` (the first error's type and code);
- after a child fails for good, in-place siblings still run (isolated ones
  already did); the fork is ``partial`` and the failure is raised at the end.

A failure after the agent acted, an unscored verdict or a cancellation is
never retried. Both are opt-in on ``Rollout.branch`` (``child_retries``,
``continue_after_child_failure``; the defaults keep PR #1046's stop-at-the-
first-failure contract) and on by default in ``bench eval branch`` and
``bf.branch``. Unit tests against fakes.
"""

from __future__ import annotations

import asyncio

import pytest

from benchflow.branch_lineage import UnscoredChildError
from benchflow.trajectories.tree import Step
from tests.test_branch_isolated import IMAGES, IsoRollout, _root, _tree
from tests.test_branch_restore_parent import _fork, _rollout


class Transient(ConnectionError):
    pass


@pytest.fixture(autouse=True)
def _reset():
    IMAGES.clear()
    IsoRollout.all = []


def _step(rollout, node):
    rollout._cursor = rollout._tree.populate(node, Step(id="s", data={}))


async def test_in_place_child_failing_before_work_is_retried_once(tmp_path):
    rollout, sandbox = _rollout(tmp_path)
    calls = {"a": 0}

    async def child(node, *, child):
        calls.setdefault(child.label, 0)
        calls[child.label] += 1
        if child.label == "a" and calls["a"] == 1:
            raise Transient("connect timed out")
        _step(rollout, node)
        return 1.0

    await rollout.branch(
        2,
        child,
        snapshot_layers={"sandbox"},
        child_labels=["a", "b"],
        child_retries=1,
        continue_after_child_failure=True,
    )
    fork = _fork(rollout)
    a, b = fork["children"]
    assert calls == {"a": 2, "b": 1}
    assert (a["status"], a["attempts"], a["retried_after"]) == (
        "scored",
        2,
        {"type": "Transient", "code": "execution_failed"},
    )
    assert b["attempts"] == 1 and "retried_after" not in b
    # The retry started from a freshly restored checkpoint: a, a again, b, parent.
    assert len(sandbox.restored) == 4
    assert fork["status"] == "completed"


async def test_a_failure_after_the_agent_acted_is_not_retried(tmp_path):
    rollout, _sandbox = _rollout(tmp_path)
    calls = []

    async def child(node, *, child):
        calls.append(child.label)
        if child.label == "a":
            _step(rollout, node)
            raise Transient("lost mid-run")
        _step(rollout, node)
        return 1.0

    with pytest.raises(Transient):
        await rollout.branch(
            2,
            child,
            snapshot_layers={"sandbox"},
            child_labels=["a", "b"],
            child_retries=1,
            continue_after_child_failure=True,
        )
    fork = _fork(rollout)
    # Not retried, but the sibling still ran.
    assert calls == ["a", "b"]
    assert [c["status"] for c in fork["children"]] == ["failed", "scored"]
    assert fork["children"][0]["attempts"] == 1
    assert fork["status"] == "partial"


@pytest.mark.parametrize(
    "error", [UnscoredChildError("no reward"), asyncio.CancelledError()]
)
async def test_unscored_and_cancelled_are_never_retried(tmp_path, error):
    rollout, _sandbox = _rollout(tmp_path)
    calls = []

    async def child(node, *, child):
        calls.append(child.label)
        if child.label == "a":
            raise error
        _step(rollout, node)
        return 1.0

    with pytest.raises(type(error)):
        await rollout.branch(
            2, child, snapshot_layers={"sandbox"}, child_labels=["a", "b"]
        )
    assert calls.count("a") == 1
    if isinstance(error, asyncio.CancelledError):
        assert calls == ["a"]  # cancellation stops the fork


async def test_isolated_child_failing_before_work_is_retried_in_a_new_sandbox(tmp_path):
    root = await _root(tmp_path)
    calls = {"a": 0}

    async def run(node, *, child):
        calls["a"] += child.label == "a"
        if child.label == "a" and calls["a"] == 1:
            raise Transient("sandbox connect timed out")
        await child.rollout.connect()
        await child.rollout.execute(["Do it."], node=node)
        return (await child.rollout.verify())["reward"]

    await root.branch(
        2,
        run,
        snapshot_layers={"sandbox"},
        child_labels=["a", "b"],
        isolate_children=True,
        concurrency=2,
        child_retries=1,
    )
    fork = _tree(root)["forks"][0]
    a = fork["children"][0]
    assert (a["status"], a["attempts"], a["retried_after"]["type"]) == (
        "scored",
        2,
        "Transient",
    )
    # Three sub-rollouts: a (failed), a (retry), b; the failed attempt's
    # folder is kept next to the child's.
    assert len(IsoRollout.all) == 1 + 3
    child_dir = root._rollout_dir / a["artifacts"]["path"]
    assert (child_dir / "observation.json").is_file()
    assert child_dir.with_name(child_dir.name + ".attempt-1").is_dir()
    assert fork["status"] == "completed"


async def test_the_defaults_keep_stop_at_the_first_failure(tmp_path):
    rollout, _sandbox = _rollout(tmp_path)
    calls = []

    async def child(node, *, child):
        calls.append(child.label)
        raise Transient("connect timed out")

    with pytest.raises(Transient):
        await rollout.branch(
            2, child, snapshot_layers={"sandbox"}, child_labels=["a", "b"]
        )
    assert calls == ["a"]
    assert [c["status"] for c in _fork(rollout)["children"]] == [
        "failed",
        "not_started",
    ]
