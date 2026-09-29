"""``branch(restore_parent=False)`` skips the parent restore after the last child.

Regression test: ``branch(n)`` always restored n+1 times, one per child plus
the parent, even when the caller throws the parent away. On Daytona a restore
of a small sandbox with an agent installed costs tens of seconds, so a caller
that only wants the children's values paid for one restore it never used.

Skipping it leaves the shared world in the last child's state, so the rollout
must refuse to continue: the parent is marked discarded, every later world
operation (connect, execute, verify, branch) raises, and only ``finalize()`` /
``cleanup()`` remain. ``tree.json`` records ``parent_restore: "skipped"``.

Unit tests against fakes; no Docker, Daytona or credentials.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.sandbox.protocol import SandboxImage


class CountingSandbox:
    supports_snapshot = True

    def __init__(self) -> None:
        self.snapshots: list[SandboxImage] = []
        self.restored: list[SandboxImage] = []
        self.deleted: list[SandboxImage] = []
        self.world = "parent"

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        image = SandboxImage(provider="fake", ref=f"bf-snap-{len(self.snapshots)}")
        self.snapshots.append(image)
        return image

    async def restore(self, image: SandboxImage) -> None:
        self.restored.append(image)
        self.world = "parent"

    async def delete_snapshot(self, image: SandboxImage) -> bool:
        self.deleted.append(image)
        return True


def _rollout(tmp_path: Path) -> tuple[Rollout, CountingSandbox]:
    sandbox = CountingSandbox()
    rollout = Rollout(
        RolloutConfig(task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy")])
    )
    rollout._env = sandbox
    rollout._rollout_dir = tmp_path / "run"
    rollout._rollout_dir.mkdir()
    return rollout, sandbox


def _fork(rollout: Rollout) -> dict:
    return json.loads((rollout._rollout_dir / "tree.json").read_text())["forks"][-1]


async def test_default_branch_restores_the_parent_after_the_last_child(tmp_path):
    rollout, sandbox = _rollout(tmp_path)

    async def child(_node):
        sandbox.world = "child"
        return 1.0

    await rollout.branch(3, child, snapshot_layers={"sandbox"})
    assert len(sandbox.restored) == 4
    assert sandbox.world == "parent"
    assert _fork(rollout)["parent_restore"] == "restored"


async def test_restore_parent_false_saves_the_last_restore(tmp_path):
    rollout, sandbox = _rollout(tmp_path)
    seen = []

    async def child(node):
        seen.append(sandbox.world)
        sandbox.world = node.id
        return float(len(seen) - 1)

    value = await rollout.branch(
        3, child, snapshot_layers={"sandbox"}, restore_parent=False
    )
    assert value == 1.0
    # Every child still starts from the checkpoint.
    assert seen == ["parent", "parent", "parent"]
    assert len(sandbox.restored) == 3
    # The world is left in the last child's state, and the record says so.
    assert sandbox.world != "parent"
    fork = _fork(rollout)
    assert fork["status"] == "completed"
    assert fork["parent_restore"] == "skipped"
    assert fork["value"] == 1.0
    # The snapshot is still released.
    assert sandbox.deleted == sandbox.snapshots


@pytest.mark.parametrize("operation", ["connect", "execute", "verify", "branch"])
async def test_a_discarded_parent_refuses_to_continue(tmp_path, operation):
    rollout, _sandbox = _rollout(tmp_path)

    async def child(_node):
        return 1.0

    await rollout.branch(2, child, snapshot_layers={"sandbox"}, restore_parent=False)
    calls = {
        "connect": lambda: rollout.connect(),
        "execute": lambda: rollout.execute(["more"]),
        "verify": lambda: rollout.verify(),
        "branch": lambda: rollout.branch(2, child, snapshot_layers={"sandbox"}),
    }
    with pytest.raises(RuntimeError, match="restore_parent=False"):
        await calls[operation]()


async def test_a_failed_child_with_restore_parent_false_still_discards(tmp_path):
    rollout, sandbox = _rollout(tmp_path)

    async def child(_node):
        raise ValueError("child failed")

    with pytest.raises(ValueError):
        await rollout.branch(
            2, child, snapshot_layers={"sandbox"}, restore_parent=False
        )
    fork = _fork(rollout)
    assert fork["parent_restore"] == "skipped"
    assert len(sandbox.restored) == 1
    with pytest.raises(RuntimeError, match="restore_parent=False"):
        await rollout.connect()
