"""The branch engine deletes the sandbox snapshots it created.

Regression test: before this fix no code path deleted a fork's container
snapshot. A Daytona provider snapshot outlived the run until it was removed by
hand, and a Docker ``bf-snap-*`` image survived unless ``compose down --rmi
all`` happened to remove it. The engine now releases the snapshot when the
fork finishes, including after a child failure or cancellation, unless the
caller passes ``retain_snapshots=True``.

Unit tests against fakes; no Docker, Daytona or credentials.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.sandbox.protocol import SandboxImage


class DeletingSandbox:
    """Snapshot-capable fake that records snapshot deletions."""

    supports_snapshot = True

    def __init__(self, *, delete_result: bool | Exception = True) -> None:
        self.snapshots: list[SandboxImage] = []
        self.deleted: list[SandboxImage] = []
        self.delete_result = delete_result

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        image = SandboxImage(provider="fake", ref=f"bf-snap-{len(self.snapshots)}")
        self.snapshots.append(image)
        return image

    async def restore(self, image: SandboxImage) -> None:
        assert image not in self.deleted, "restored from a deleted snapshot"

    async def delete_snapshot(self, image: SandboxImage) -> bool:
        if isinstance(self.delete_result, Exception):
            raise self.delete_result
        self.deleted.append(image)
        return self.delete_result


def _rollout(tmp_path: Path, sandbox: DeletingSandbox) -> Rollout:
    rollout = Rollout(
        RolloutConfig(task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy")])
    )
    rollout._env = sandbox
    rollout._rollout_dir = tmp_path / "run"
    rollout._rollout_dir.mkdir()
    return rollout


def _fork(rollout: Rollout) -> dict:
    document = json.loads((rollout._rollout_dir / "tree.json").read_text())
    return document["forks"][-1]


async def test_completed_fork_deletes_its_sandbox_snapshot(tmp_path):
    sandbox = DeletingSandbox()
    rollout = _rollout(tmp_path, sandbox)

    async def child(_node):
        return 1.0

    assert await rollout.branch(2, child, snapshot_layers={"sandbox"}) == 1.0
    assert sandbox.deleted == sandbox.snapshots
    snapshot = _fork(rollout)["snapshot"]
    assert snapshot["retention"] == "deleted"
    assert snapshot["restore_available"] is False


@pytest.mark.parametrize(
    "error", [ValueError("child failed"), asyncio.CancelledError()]
)
async def test_failed_or_cancelled_fork_still_deletes_its_snapshot(tmp_path, error):
    sandbox = DeletingSandbox()
    rollout = _rollout(tmp_path, sandbox)

    async def child(_node):
        raise error

    with pytest.raises(type(error)):
        await rollout.branch(2, child, snapshot_layers={"sandbox"})
    assert sandbox.deleted == sandbox.snapshots
    assert _fork(rollout)["snapshot"]["retention"] == "deleted"


async def test_retain_snapshots_keeps_the_snapshot(tmp_path):
    sandbox = DeletingSandbox()
    rollout = _rollout(tmp_path, sandbox)

    async def child(_node):
        return 0.0

    await rollout.branch(2, child, snapshot_layers={"sandbox"}, retain_snapshots=True)
    assert sandbox.deleted == []
    snapshot = _fork(rollout)["snapshot"]
    assert snapshot["retention"] == "kept"
    assert snapshot["restore_available"] is None


async def test_snapshot_still_in_use_is_recorded_as_deferred(tmp_path):
    """Docker cannot remove the image the restored parent container runs from;
    the sandbox removes it when it stops, and the record says so."""
    sandbox = DeletingSandbox(delete_result=False)
    rollout = _rollout(tmp_path, sandbox)

    async def child(_node):
        return 1.0

    await rollout.branch(2, child, snapshot_layers={"sandbox"})
    assert _fork(rollout)["snapshot"]["retention"] == "deferred"


async def test_delete_failure_is_recorded_without_losing_the_value(tmp_path, caplog):
    sandbox = DeletingSandbox(delete_result=RuntimeError("provider unavailable"))
    rollout = _rollout(tmp_path, sandbox)

    async def child(_node):
        return 1.0

    assert await rollout.branch(2, child, snapshot_layers={"sandbox"}) == 1.0
    snapshot = _fork(rollout)["snapshot"]
    assert snapshot["retention"] == "delete_failed"
    assert snapshot["delete_error"] == {
        "type": "RuntimeError",
        "code": "execution_failed",
    }
    assert "bf-snap-0" in caplog.text


async def test_environment_only_fork_has_no_sandbox_retention(tmp_path):
    rollout = _rollout(tmp_path, DeletingSandbox())

    class Environment:
        async def snapshot(self):
            from benchflow.environment.protocol import StateSnapshot

            return StateSnapshot(id="env", path="/tmp/env")

        async def restore(self, _snap):
            return None

    rollout._environment = Environment()

    async def child(_node):
        return 1.0

    await rollout.branch(2, child)
    assert _fork(rollout)["snapshot"]["retention"] is None
