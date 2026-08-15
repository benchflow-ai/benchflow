"""Regression tests for the composed checkpoint layer.

Guards the composed-checkpoint layer ("feat(branch): compose sandbox +
environment checkpoints in the branch engine"; docs/rollout-branching-rfc.md
WS-1; FrontierPhysics#73). PR number to be added on submission.

The composed checkpoint (RFC §3.1) layers the container snapshot
(``Sandbox.snapshot``) with the environment-state snapshot into one
``StageSnapshot``: environment first on checkpoint, sandbox first on restore.
The branch engine requests layers via ``snapshot_layers``; the default
``{"environment"}`` must keep the legacy environment-only path byte-for-byte.

These are unit tests against fakes — no Docker, Daytona, or API keys.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.branch import (
    StageSnapshot,
    checkpoint,
    checkpoint_composed,
    restore,
    restore_composed,
)
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.environment.manifest_env import ManifestEnvironment
from benchflow.environment.protocol import StateSnapshot
from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.sandbox.protocol import SandboxImage, SandboxSnapshotNotSupported
from benchflow.trajectories.tree import RolloutTree


class FakeEnv:
    """Stateful in-memory Environment — snapshot copies state, restore rolls back.

    ``calls`` may be shared with a fake sandbox to record cross-layer order.
    """

    def __init__(self, calls: list[str] | None = None) -> None:
        self.state: dict[str, str] = {"db": "initial"}
        self._saved: dict[str, dict[str, str]] = {}
        self.calls = calls if calls is not None else []
        self.snapshots: list[StateSnapshot] = []
        self.restored: list[StateSnapshot] = []

    async def snapshot(self) -> StateSnapshot:
        snap = StateSnapshot(id=f"env-snap-{len(self.snapshots) + 1}", path="/tmp/x")
        self._saved[snap.id] = dict(self.state)
        self.snapshots.append(snap)
        self.calls.append("env.snapshot")
        return snap

    async def restore(self, snap: StateSnapshot) -> None:
        self.state = dict(self._saved[snap.id])
        self.restored.append(snap)
        self.calls.append("env.restore")


class FakeSnapSandbox:
    """Snapshot-capable Sandbox stand-in with in-memory filesystem state."""

    supports_snapshot = True

    def __init__(self, calls: list[str] | None = None) -> None:
        self.fs: dict[str, str] = {"/workspace": "clean"}
        self._images: dict[str, dict[str, str]] = {}
        self.calls = calls if calls is not None else []
        self.snapshots: list[SandboxImage] = []
        self.restored: list[SandboxImage] = []

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        img = SandboxImage(provider="fake", ref=f"bf-snap-{len(self.snapshots) + 1}")
        self._images[img.ref] = dict(self.fs)
        self.snapshots.append(img)
        self.calls.append("sandbox.snapshot")
        return img

    async def restore(self, image: SandboxImage) -> None:
        self.fs = dict(self._images[image.ref])
        self.restored.append(image)
        self.calls.append("sandbox.restore")


class NoSnapshotSandbox:
    """Sandbox without container snapshot — the capability gate must fail closed."""

    supports_snapshot = False

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        self.calls.append("sandbox.snapshot")
        raise SandboxSnapshotNotSupported(
            "NoSnapshotSandbox does not support snapshot"
        )

    async def restore(self, image: SandboxImage) -> None:
        self.calls.append("sandbox.restore")
        raise SandboxSnapshotNotSupported("NoSnapshotSandbox does not support restore")


# A manifest with no [environment.state] — a *stateless* environment whose
# snapshot()/restore() raise (the manifest_env fail-closed precedent, #387).
_STATELESS_MANIFEST = EnvironmentManifest.model_validate_toml(
    """
[environment]
name           = "chi-bench"
image          = "chi-bench:latest"
ports          = [8020]
owns_lifecycle = true
"""
)


def _rollout(tmp_path: Path) -> Rollout:
    return Rollout(
        RolloutConfig(task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy")])
    )


# 1. Pure ops: composition order, fail-closed recording, both snapshot shapes


async def test_checkpoint_composed_snapshots_environment_before_sandbox():
    """RFC §3.1 order: environment.snapshot() first, then sandbox.snapshot()."""
    tree = RolloutTree()
    calls: list[str] = []
    env, sandbox = FakeEnv(calls), FakeSnapSandbox(calls)

    snap = await checkpoint_composed(
        tree.root, environment=env, sandbox=sandbox, stage="env-ready"
    )

    assert calls == ["env.snapshot", "sandbox.snapshot"]
    assert isinstance(snap, StageSnapshot)
    assert snap.environment_ref is env.snapshots[0]
    assert snap.sandbox_ref is sandbox.snapshots[0]
    assert snap.stage == "env-ready"
    assert tree.root.state["snapshot"] is snap


async def test_restore_composed_restores_sandbox_before_environment():
    """RFC §3.1 restore order is reversed: sandbox first, then environment."""
    tree = RolloutTree()
    calls: list[str] = []
    env, sandbox = FakeEnv(calls), FakeSnapSandbox(calls)
    await checkpoint_composed(tree.root, environment=env, sandbox=sandbox)
    calls.clear()

    await restore_composed(tree.root, environment=env, sandbox=sandbox)

    assert calls == ["sandbox.restore", "env.restore"]


async def test_checkpoint_composed_needs_at_least_one_layer():
    tree = RolloutTree()
    with pytest.raises(ValueError, match="at least one layer"):
        await checkpoint_composed(tree.root)


async def test_checkpoint_composed_failure_records_nothing_on_the_node():
    """A failing layer propagates and never leaves a partial StageSnapshot.

    The environment layer snapshots fine, the sandbox layer raises — the node
    must not carry a half-recorded roll-back point.
    """
    tree = RolloutTree()
    env, sandbox = FakeEnv(), NoSnapshotSandbox()

    with pytest.raises(SandboxSnapshotNotSupported):
        await checkpoint_composed(tree.root, environment=env, sandbox=sandbox)

    assert "snapshot" not in tree.root.state


async def test_restore_composed_accepts_a_legacy_bare_state_snapshot():
    """Back-compat: checkpoint() consumers keep working through restore_composed.

    A node checkpointed by the legacy checkpoint() holds a bare StateSnapshot;
    restore_composed must treat it as an environment-only checkpoint.
    """
    tree = RolloutTree()
    env = FakeEnv()
    snap = await checkpoint(tree.root, env)

    await restore_composed(tree.root, environment=env)

    assert env.restored == [snap]


async def test_restore_composed_without_a_checkpoint_raises():
    tree = RolloutTree()
    with pytest.raises(ValueError, match="no checkpoint"):
        await restore_composed(tree.root, environment=FakeEnv())


# 2. Pure ops: mismatched snapshot / live-object shapes fail closed


async def test_restore_composed_rejects_a_missing_live_object_for_a_layer():
    """A layer present in the checkpoint needs its live object — and no layer
    is restored before the mismatch is caught."""
    tree = RolloutTree()
    env, sandbox = FakeEnv(), FakeSnapSandbox()
    await checkpoint_composed(tree.root, environment=env, sandbox=sandbox)

    with pytest.raises(ValueError, match="no live sandbox"):
        await restore_composed(tree.root, environment=env)
    with pytest.raises(ValueError, match="no live environment"):
        await restore_composed(tree.root, sandbox=sandbox)

    assert sandbox.restored == []
    assert env.restored == []


async def test_restore_composed_rejects_a_live_object_without_a_layer():
    """A live object for a layer the checkpoint never captured is a bug."""
    tree = RolloutTree()
    env, sandbox = FakeEnv(), FakeSnapSandbox()
    await checkpoint_composed(tree.root, sandbox=sandbox)

    with pytest.raises(ValueError, match="no environment layer"):
        await restore_composed(tree.root, environment=env, sandbox=sandbox)

    assert sandbox.restored == []


async def test_legacy_restore_rejects_a_composed_stage_snapshot():
    """A StageSnapshot node fails closed in legacy restore() — use restore_composed.

    A node checkpointed via checkpoint_composed() holds a StageSnapshot, not a
    bare StateSnapshot; passing it to the legacy restore() used to explode deep
    inside the environment (AttributeError on ``.path``). It must be rejected
    with a clear ValueError *before* the environment is touched.
    """
    tree = RolloutTree()
    env, sandbox = FakeEnv(), FakeSnapSandbox()
    await checkpoint_composed(tree.root, environment=env, sandbox=sandbox)

    with pytest.raises(ValueError, match="restore_composed"):
        await restore(tree.root, env)

    assert env.restored == []
    assert env.calls == ["env.snapshot"]  # checkpoint only — no restore call


async def test_restore_composed_rejects_a_sandbox_against_a_legacy_snapshot():
    """A legacy bare StateSnapshot is environment-only — a live sandbox is a
    shape mismatch, not a silent no-op."""
    tree = RolloutTree()
    env, sandbox = FakeEnv(), FakeSnapSandbox()
    await checkpoint(tree.root, env)

    with pytest.raises(ValueError, match="no sandbox layer"):
        await restore_composed(tree.root, environment=env, sandbox=sandbox)

    assert sandbox.restored == []
    assert env.restored == []


# 3. Pure ops: the zero-delta invariant (RFC §5, T2 shrunk to unit scale)


async def test_zero_delta_checkpoint_restore_round_trips_both_layers():
    """checkpoint_composed + restore_composed is lossless on both layers."""
    tree = RolloutTree()
    env, sandbox = FakeEnv(), FakeSnapSandbox()
    env.state["db"] = "checkpointed"
    sandbox.fs["/workspace"] = "checkpointed"
    await checkpoint_composed(tree.root, environment=env, sandbox=sandbox)

    env.state["db"] = "dirty"
    sandbox.fs["/workspace"] = "dirty"
    await restore_composed(tree.root, environment=env, sandbox=sandbox)

    assert env.state == {"db": "checkpointed"}
    assert sandbox.fs == {"/workspace": "checkpointed"}


# 4. Engine: snapshot_layers wiring through Rollout.branch()


async def test_default_snapshot_layers_keep_the_legacy_environment_only_path(
    tmp_path: Path,
):
    """Back-compat: the default never touches the sandbox layer and records
    the legacy bare-StateSnapshot shape on the node."""
    rollout = _rollout(tmp_path)
    env, sandbox = FakeEnv(), FakeSnapSandbox()
    rollout._environment = env
    rollout._env = sandbox
    parent = rollout._cursor

    async def run_child(child):
        return 1.0

    value = await rollout.branch(2, run_child=run_child)

    assert value == 1.0
    assert sandbox.calls == []
    snap = parent.state["snapshot"]
    assert isinstance(snap, StateSnapshot)
    assert not isinstance(snap, StageSnapshot)


async def test_engine_composed_branch_orders_layers_per_rfc(tmp_path: Path):
    """With both layers requested the engine checkpoints env-then-sandbox once
    and restores sandbox-then-env once per child."""
    rollout = _rollout(tmp_path)
    calls: list[str] = []
    env, sandbox = FakeEnv(calls), FakeSnapSandbox(calls)
    rollout._environment = env
    rollout._env = sandbox
    parent = rollout._cursor

    async def run_child(child):
        return 1.0

    value = await rollout.branch(
        2, run_child=run_child, snapshot_layers={"environment", "sandbox"}
    )

    assert value == 1.0
    assert calls == [
        "env.snapshot",
        "sandbox.snapshot",
        "sandbox.restore",
        "env.restore",
        "sandbox.restore",
        "env.restore",
    ]
    snap = parent.state["snapshot"]
    assert isinstance(snap, StageSnapshot)
    assert snap.environment_ref is env.snapshots[0]
    assert snap.sandbox_ref is sandbox.snapshots[0]


async def test_sandbox_layer_requested_but_unsupported_fails_closed(tmp_path: Path):
    """The capability gate fires before anything is snapshotted — no partial
    node.state mutation, the existing #384 diagnostic pattern."""
    rollout = _rollout(tmp_path)
    env = FakeEnv()
    rollout._environment = env
    rollout._env = NoSnapshotSandbox()
    parent = rollout._cursor

    async def run_child(child):
        return 1.0

    with pytest.raises(RuntimeError, match="container-level snapshot/restore"):
        await rollout.branch(
            2, run_child=run_child, snapshot_layers={"environment", "sandbox"}
        )

    assert "snapshot" not in parent.state
    assert env.calls == []


async def test_environment_layer_on_a_stateless_env_propagates_the_typed_error(
    tmp_path: Path,
):
    """A stateless ManifestEnvironment's snapshot() RuntimeError propagates,
    annotated with the requested snapshot_layers; the sandbox layer (ordered
    second) is never snapshotted and the node stays unmutated."""
    rollout = _rollout(tmp_path)
    sandbox = FakeSnapSandbox()
    rollout._environment = ManifestEnvironment(
        _STATELESS_MANIFEST, sandbox=FakeSnapSandbox()
    )
    rollout._env = sandbox
    parent = rollout._cursor

    async def run_child(child):
        return 1.0

    with pytest.raises(RuntimeError, match="stateless") as excinfo:
        await rollout.branch(
            2, run_child=run_child, snapshot_layers={"environment", "sandbox"}
        )

    assert any("snapshot_layers" in note for note in excinfo.value.__notes__)
    assert "snapshot" not in parent.state
    assert sandbox.snapshots == []


async def test_sandbox_only_branching_on_a_stateless_environment(tmp_path: Path):
    """snapshot_layers={"sandbox"} branches a stateless env end-to-end: the
    children restore from the container image and environment
    snapshot/restore are never called (they would raise if they were)."""
    rollout = _rollout(tmp_path)
    sandbox = FakeSnapSandbox()
    rollout._environment = ManifestEnvironment(
        _STATELESS_MANIFEST, sandbox=FakeSnapSandbox()
    )
    rollout._env = sandbox
    parent = rollout._cursor

    returns = iter([1.0, 0.0])

    async def run_child(child):
        return next(returns)

    value = await rollout.branch(
        2, run_child=run_child, snapshot_layers={"sandbox"}
    )

    assert value == 0.5
    # one container checkpoint, one restore per child, all from the same image
    assert len(sandbox.snapshots) == 1
    assert sandbox.restored == [sandbox.snapshots[0], sandbox.snapshots[0]]
    snap = parent.state["snapshot"]
    assert isinstance(snap, StageSnapshot)
    assert snap.environment_ref is None
    assert snap.sandbox_ref is sandbox.snapshots[0]
    assert parent.state["value"] == 0.5


async def test_unknown_snapshot_layer_is_rejected(tmp_path: Path):
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnv()

    async def run_child(child):
        return 0.0

    with pytest.raises(ValueError, match="unknown snapshot_layers"):
        await rollout.branch(
            2, run_child=run_child, snapshot_layers={"environment", "agent-session"}
        )


async def test_empty_snapshot_layers_are_rejected(tmp_path: Path):
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnv()

    async def run_child(child):
        return 0.0

    with pytest.raises(ValueError, match="at least one layer"):
        await rollout.branch(2, run_child=run_child, snapshot_layers=set())
