"""tree.json records how long a fork's checkpoint and restores took.

Before this, a fork's checkpoint and restore cost was not recorded. Each fork
now records ``timing_sec`` with ``checkpoint``, one
``child_restore`` entry per child and ``parent_restore`` (None when skipped).

Unit tests against fakes; no Docker, Daytona or credentials.
"""

from __future__ import annotations

from tests.test_branch_restore_parent import _fork, _rollout


async def test_fork_records_checkpoint_and_restore_seconds(tmp_path):
    """A fork's checkpoint and restore cost is measurable from tree.json."""
    rollout, _sandbox = _rollout(tmp_path)

    async def child(_node):
        return 1.0

    await rollout.branch(2, child, snapshot_layers={"sandbox"})
    timing = _fork(rollout)["timing_sec"]
    assert isinstance(timing["checkpoint"], float)
    assert len(timing["child_restore"]) == 2
    assert all(isinstance(value, float) for value in timing["child_restore"])
    assert isinstance(timing["parent_restore"], float)

    (tmp_path / "second").mkdir()
    discarded, _ = _rollout(tmp_path / "second")
    await discarded.branch(2, child, snapshot_layers={"sandbox"}, restore_parent=False)
    assert _fork(discarded)["timing_sec"]["parent_restore"] is None


async def test_fork_records_the_children_phase(tmp_path):
    """The children phase (every child, restores included) is timed on its
    own, so in-place and isolated forks can be compared without the
    snapshot's run-to-run spread."""
    rollout, _sandbox = _rollout(tmp_path)

    async def child(_node):
        return 1.0

    await rollout.branch(2, child, snapshot_layers={"sandbox"})
    assert isinstance(_fork(rollout)["timing_sec"]["children"], float)
