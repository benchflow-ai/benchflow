"""Fork from a snapshot that already exists instead of taking another.

``bench eval branch --checkpoints every-prompt
--checkpoint-after-prompt 1`` took two snapshots of the same state back to
back, the automatic checkpoint ``prompt:1`` and the fork snapshot.
``--from-checkpoint`` started a fresh sandbox,
replaced it with the kept checkpoint, then snapshotted that state again into
a new snapshot it deleted afterwards.

``Rollout.branch(reuse_snapshot=image)`` uses ``image`` (the sandbox's state
at the cursor, sandbox layer only) as the fork's snapshot: no capture, never
deleted by the fork (its owner, the checkpoint policy, keeps or deletes it),
and the sandbox's credential files are associated with it so in-place
restores and the parent restore get them back (``adopt_snapshot``).
"""

from __future__ import annotations

import pytest

from benchflow.sandbox.protocol import SandboxImage
from tests.test_branch_isolated import IMAGES, IsoRollout, _root, _runner, _tree


@pytest.fixture(autouse=True)
def _reset():
    IMAGES.clear()
    IsoRollout.all = []


async def _given(root):
    image = await root._env.snapshot()  # an existing checkpoint of ["draft"]
    root._env.adopted = []

    async def adopt(img):
        root._env.adopted.append(img.ref)

    root._env.adopt_snapshot = adopt
    return image


@pytest.mark.parametrize("isolated", [True, False])
async def test_the_given_snapshot_is_the_fork_snapshot(tmp_path, isolated):
    root = await _root(tmp_path)
    image = await _given(root)
    snapshots_before = len(IMAGES)
    value = await root.branch(
        2,
        _runner({"a": "Do it.", "b": "Other."}),
        snapshot_layers={"sandbox"},
        child_labels=["a", "b"],
        isolate_children=isolated,
        reuse_snapshot=image,
    )
    assert value == 0.5
    assert len(IMAGES) == snapshots_before  # no second snapshot
    assert root._env.deleted == []  # not the fork's to delete
    assert root._env.adopted == [image.ref]
    fork = _tree(root)["forks"][0]
    snap = fork["snapshot"]
    assert snap["captured_layers"] == ["sandbox"]
    assert snap["retention"] == "kept"
    assert snap["reused"] is True
    assert fork["timing_sec"]["checkpoint"] < 1
    # The parent is back at the given state.
    assert root.world == ["draft"]
    if isolated:
        assert all(s._env.restores == [image.ref] for s in IsoRollout.all[1:])


async def test_reuse_needs_the_sandbox_layer_only(tmp_path):
    root = await _root(tmp_path)
    image = SandboxImage(provider="fake", ref="x")
    with pytest.raises(ValueError, match="reuse_snapshot"):
        await root.branch(
            2,
            _runner({"a": "Do it.", "b": "Other."}),
            snapshot_layers={"sandbox", "environment"},
            reuse_snapshot=image,
        )
