"""Regression tests for the stage-boundary snapshot policy.

Guards the stage-boundary policy ("feat(branch): stage-boundary snapshot policy
for rollout branching"; docs/rollout-branching-rfc.md §3.2 / WS-4a;
FrontierPhysics#73). PR number to be added on submission.

The four cascade stages pin to existing lifecycle transitions: ``env-ready``
at the end of ``start()`` (before ``install_agent()``), ``post-research`` at an
explicit ``mark_stage()`` inside ``execute()``, ``pre-verify`` immediately
before ``planes.harden_before_verify``, and ``post-verify`` after ``verify()``.
Capture is opt-in via ``RolloutConfig.snapshot_stages`` — the default set is
empty and must keep today's behavior exactly, with no snapshot call anywhere.
``branch_at_stage()`` then forks from a recorded stage instead of
checkpointing at the cursor, and records the stage name as ``branch_stage``.

These are unit tests against fakes — no Docker, Daytona, or API keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.branch import StageSnapshot
from benchflow.branch_delta import BranchDelta
from benchflow.branch_stage import (
    AUTO_STAGES,
    BRANCH_STAGES,
    MARKED_STAGES,
    BranchStageNotCaptured,
)
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.environment.manifest_env import ManifestEnvironment
from benchflow.environment.protocol import StateSnapshot
from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.rollout_branch import BranchDeltaNotSupported
from benchflow.sandbox.protocol import SandboxImage, SandboxSnapshotNotSupported
from benchflow.trajectories.tree import Step


class FakeEnv:
    """Stateful in-memory Environment — snapshot copies state, restore rolls back.

    ``calls`` may be shared with the fake sandbox and the faked lifecycle
    boundaries to record ordering across all of them.
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

    def __init__(self, calls: list[str] | None = None) -> None:
        self.calls = calls if calls is not None else []

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        self.calls.append("sandbox.snapshot")
        raise SandboxSnapshotNotSupported("NoSnapshotSandbox does not support snapshot")

    async def restore(self, image: SandboxImage) -> None:
        self.calls.append("sandbox.restore")
        raise SandboxSnapshotNotSupported("NoSnapshotSandbox does not support restore")


# A manifest with no [environment.state] — a *stateless* environment whose
# snapshot() raises (the manifest_env fail-closed precedent, #387).
_STATELESS_MANIFEST = EnvironmentManifest.model_validate_toml(
    """
[environment]
name           = "chi-bench"
image          = "chi-bench:latest"
ports          = [8020]
owns_lifecycle = true
"""
)


def _rollout(tmp_path: Path, **config) -> Rollout:
    return Rollout(
        RolloutConfig(
            task_path=tmp_path / "task",
            scenes=[Scene.single(agent="dummy")],
            **config,
        )
    )


def _fake_start_boundary(monkeypatch, calls: list[str]) -> None:
    """Fake everything ``start()`` does before the env-ready boundary."""

    async def fake_start_env_and_upload(*_a, **_kw):
        calls.append("start-env")

    async def fake_healthcheck(*_a, **_kw):
        calls.append("healthcheck")

    async def fake_setup_commands(*_a, **_kw):
        calls.append("env-setup-commands")

    monkeypatch.setattr(
        "benchflow.rollout._start_env_and_upload", fake_start_env_and_upload
    )
    monkeypatch.setattr(
        "benchflow.rollout._run_environment_healthcheck", fake_healthcheck
    )
    monkeypatch.setattr(
        "benchflow.rollout._run_environment_setup_commands", fake_setup_commands
    )

    async def fake_install_agent(self):
        calls.append("install_agent")

    monkeypatch.setattr(Rollout, "install_agent", fake_install_agent)


def _fake_verify_boundary(monkeypatch, calls: list[str], tmp_path: Path) -> None:
    """Fake ``verify()``'s I/O; ``_verify_rollout`` hardens first, as in prod."""

    async def fake_publish(*_a, **_kw):
        calls.append("publish-trajectory")

    async def fake_verify_rollout(*_a, **_kw):
        calls.append("harden_before_verify")
        calls.append("verifier")
        return {"reward": 1.0}, None, None

    monkeypatch.setattr(
        "benchflow.rollout._publish_trajectory_for_verifier", fake_publish
    )
    monkeypatch.setattr("benchflow.rollout._verify_rollout", fake_verify_rollout)


def _ready_to_verify(rollout: Rollout, tmp_path: Path) -> None:
    """Minimal state so ``verify()`` reaches its boundaries with fakes in place."""
    rollout._trajectory = [{"role": "agent", "text": "done"}]
    rollout._rollout_paths = SimpleNamespace(agent_dir=tmp_path)


# 1. Taxonomy + config validation


def test_taxonomy_splits_auto_detectable_from_marked_stages():
    """post-research is the one boundary no lifecycle transition can detect."""
    assert BRANCH_STAGES == (
        "env-ready",
        "post-research",
        "pre-verify",
        "post-verify",
    )
    assert sorted(AUTO_STAGES) == ["env-ready", "post-verify", "pre-verify"]
    assert sorted(MARKED_STAGES) == ["post-research"]


def test_default_config_requests_no_stage_snapshots(tmp_path: Path):
    """The default is empty — the policy is strictly opt-in."""
    rollout = _rollout(tmp_path)
    assert rollout._config.snapshot_stages == frozenset()
    assert rollout._config.snapshot_layers == frozenset({"environment"})
    assert rollout.stage_snapshots == {}


def test_unknown_stage_name_is_rejected_at_config_validation(tmp_path: Path):
    """A typo fails closed at construction, not at the boundary it would miss."""
    with pytest.raises(ValueError, match="unknown snapshot_stages entry"):
        _rollout(tmp_path, snapshot_stages={"env-ready", "post-executuion"})


def test_snapshot_stages_rejects_a_bare_string(tmp_path: Path):
    """A string would silently iterate into characters — every one unknown."""
    with pytest.raises(ValueError, match="collection of stage names"):
        _rollout(tmp_path, snapshot_stages="env-ready")


def test_snapshot_stages_normalizes_any_collection(tmp_path: Path):
    rollout = _rollout(tmp_path, snapshot_stages=["env-ready", "pre-verify"])
    assert rollout._config.snapshot_stages == frozenset({"env-ready", "pre-verify"})


async def test_mark_stage_rejects_a_name_outside_the_taxonomy(tmp_path: Path):
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnv()

    with pytest.raises(ValueError, match="unknown stage"):
        await rollout.mark_stage("post-planning")


async def test_branch_at_stage_rejects_a_name_outside_the_taxonomy(tmp_path: Path):
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnv()

    async def run_child(child):
        return 1.0

    with pytest.raises(ValueError, match="unknown stage"):
        await rollout.branch_at_stage("post-planning", 2, run_child=run_child)


# 2. Opt-in capture: the default lifecycle takes zero snapshots


async def test_default_config_takes_zero_snapshots_across_the_lifecycle(
    tmp_path: Path, monkeypatch
):
    """Zero overhead by default: not one snapshot call, not one artifact."""
    calls: list[str] = []
    rollout = _rollout(tmp_path)
    env, sandbox = FakeEnv(calls), FakeSnapSandbox(calls)
    rollout._environment, rollout._env = env, sandbox
    rollout._rollout_dir = tmp_path
    _fake_start_boundary(monkeypatch, calls)
    _fake_verify_boundary(monkeypatch, calls, tmp_path)
    _ready_to_verify(rollout, tmp_path)

    await rollout.start()
    await rollout.install_agent()
    await rollout.verify()

    assert env.snapshots == []
    assert sandbox.snapshots == []
    assert rollout.stage_snapshots == {}
    assert not (tmp_path / "stage_snapshots.json").exists()


# 3. Each auto stage fires at its lifecycle boundary


async def test_env_ready_is_captured_at_the_end_of_start_before_install_agent(
    tmp_path: Path, monkeypatch
):
    """env-ready precedes install_agent() so a child re-runs skill deployment
    from a skill-free world (RFC §3.2)."""
    calls: list[str] = []
    rollout = _rollout(
        tmp_path,
        snapshot_stages={"env-ready"},
        snapshot_layers={"environment", "sandbox"},
    )
    env, sandbox = FakeEnv(calls), FakeSnapSandbox(calls)
    rollout._environment, rollout._env = env, sandbox
    _fake_start_boundary(monkeypatch, calls)

    await rollout.start()
    await rollout.install_agent()

    assert calls == [
        "start-env",
        "healthcheck",
        "env-setup-commands",
        "env.snapshot",
        "sandbox.snapshot",
        "install_agent",
    ]
    snap = rollout.stage_snapshots["env-ready"]
    assert isinstance(snap, StageSnapshot)
    assert snap.stage == "env-ready"
    assert snap.environment_ref is env.snapshots[0]
    assert snap.sandbox_ref is sandbox.snapshots[0]
    # the stage tag rides on the node's own recorded checkpoint
    assert rollout._cursor.state["snapshot"] is snap


async def test_pre_verify_is_captured_before_hardening_and_post_verify_after(
    tmp_path: Path, monkeypatch
):
    """pre-verify lands immediately before planes.harden_before_verify (which
    _verify_rollout runs first), post-verify after the verifier scored."""
    calls: list[str] = []
    rollout = _rollout(tmp_path, snapshot_stages={"pre-verify", "post-verify"})
    env = FakeEnv(calls)
    rollout._environment, rollout._env = env, FakeSnapSandbox(calls)
    _fake_verify_boundary(monkeypatch, calls, tmp_path)
    _ready_to_verify(rollout, tmp_path)

    rewards = await rollout.verify()

    assert rewards == {"reward": 1.0}
    assert calls == [
        "publish-trajectory",
        "env.snapshot",
        "harden_before_verify",
        "verifier",
        "env.snapshot",
    ]
    assert [s.stage for s in rollout.stage_snapshots.values()] == [
        "pre-verify",
        "post-verify",
    ]


async def test_only_the_requested_stages_are_captured(tmp_path: Path, monkeypatch):
    """A stage nobody asked for costs a membership test and nothing else."""
    calls: list[str] = []
    rollout = _rollout(tmp_path, snapshot_stages={"pre-verify"})
    env = FakeEnv(calls)
    rollout._environment, rollout._env = env, FakeSnapSandbox(calls)
    _fake_start_boundary(monkeypatch, calls)
    _fake_verify_boundary(monkeypatch, calls, tmp_path)
    _ready_to_verify(rollout, tmp_path)

    await rollout.start()
    await rollout.verify()

    assert list(rollout.stage_snapshots) == ["pre-verify"]
    assert len(env.snapshots) == 1


# 4. Capability gaps fail closed, and the diagnostic names the stage


async def test_stage_capture_names_the_stage_when_the_sandbox_cannot_snapshot(
    tmp_path: Path, monkeypatch
):
    """The #384 container-snapshot gate, attributed to the boundary that
    needed it — and start() fails closed instead of continuing uncheckpointed."""
    calls: list[str] = []
    rollout = _rollout(
        tmp_path,
        snapshot_stages={"env-ready"},
        snapshot_layers={"environment", "sandbox"},
    )
    env, sandbox = FakeEnv(calls), NoSnapshotSandbox()
    rollout._environment, rollout._env = env, sandbox
    _fake_start_boundary(monkeypatch, calls)

    with pytest.raises(RuntimeError, match="container-level snapshot/restore") as exc:
        await rollout.start()

    assert "env-ready" in str(exc.value)
    assert env.snapshots == []  # the gate fires before any layer is snapshotted
    assert sandbox.calls == []
    assert rollout.stage_snapshots == {}


async def test_stage_capture_names_the_stage_on_a_stateless_environment(
    tmp_path: Path, monkeypatch
):
    """A stateless ManifestEnvironment's typed error propagates, annotated with
    the stage that asked for the environment layer."""
    calls: list[str] = []
    rollout = _rollout(tmp_path, snapshot_stages={"pre-verify"})
    rollout._environment = ManifestEnvironment(
        _STATELESS_MANIFEST, sandbox=FakeSnapSandbox()
    )
    rollout._env = FakeSnapSandbox(calls)
    _fake_verify_boundary(monkeypatch, calls, tmp_path)
    _ready_to_verify(rollout, tmp_path)

    with pytest.raises(RuntimeError, match="stateless") as exc:
        await rollout.verify()

    assert any("pre-verify" in note for note in exc.value.__notes__)
    assert "harden_before_verify" not in calls
    assert rollout.stage_snapshots == {}


async def test_stage_capture_names_the_stage_without_an_environment_plane(
    tmp_path: Path, monkeypatch
):
    calls: list[str] = []
    rollout = _rollout(tmp_path, snapshot_stages={"env-ready"})
    rollout._env = FakeSnapSandbox(calls)
    assert rollout._environment is None
    _fake_start_boundary(monkeypatch, calls)

    with pytest.raises(RuntimeError, match="needs the Environment plane") as exc:
        await rollout.start()

    assert "env-ready" in str(exc.value)


# 5. Manual marking — the post-research boundary


async def test_mark_stage_registers_post_research_at_the_current_point(
    tmp_path: Path,
):
    """The harness marks the mid-execute() boundary it alone can see."""
    rollout = _rollout(tmp_path)
    env = FakeEnv()
    rollout._environment, rollout._env = env, FakeSnapSandbox()
    rollout._cursor = rollout._tree.advance(rollout._cursor, Step(id="s1"))
    research_node = rollout._cursor

    snap = await rollout.mark_stage("post-research")

    assert rollout.stage_snapshots == {"post-research": snap}
    assert snap.stage == "post-research"
    assert snap.environment_ref is env.snapshots[0]
    assert research_node.state["snapshot"] is snap


async def test_mark_stage_takes_the_layers_it_is_given(tmp_path: Path):
    rollout = _rollout(tmp_path)
    env, sandbox = FakeEnv(), FakeSnapSandbox()
    rollout._environment, rollout._env = env, sandbox

    snap = await rollout.mark_stage(
        "post-research", snapshot_layers={"environment", "sandbox"}
    )

    assert snap.sandbox_ref is sandbox.snapshots[0]


# 6. Branching from a recorded stage


async def _capture_env_ready(rollout: Rollout) -> StageSnapshot:
    """Record env-ready at the root, then run past it linearly."""
    snap = await rollout.mark_stage("env-ready")
    rollout._cursor = rollout._tree.advance(rollout._cursor, Step(id="s1"))
    return snap


async def test_branch_at_stage_restores_the_recorded_snapshot(tmp_path: Path):
    """The children restore env-ready's world — no new checkpoint is taken, and
    they fork at the node that stage was captured on, not at the cursor."""
    rollout = _rollout(tmp_path)
    env = FakeEnv()
    rollout._environment, rollout._env = env, FakeSnapSandbox()
    root = rollout._cursor
    snap = await _capture_env_ready(rollout)
    env.state["db"] = "post-agent"
    cursor_before = rollout._cursor

    returns = iter([1.0, 0.0])

    async def run_child(child):
        return next(returns)

    value = await rollout.branch_at_stage("env-ready", 2, run_child=run_child)

    assert value == 0.5
    assert len(env.snapshots) == 1  # the stage snapshot, not one per branch
    assert env.restored == [snap.environment_ref, snap.environment_ref]
    assert env.state == {"db": "initial"}  # rolled back to the env-ready world
    assert [child.parent for child in root.children[1:]] == [root, root]
    assert rollout._cursor is cursor_before


async def test_branch_at_stage_records_the_stage_name_in_provenance(tmp_path: Path):
    """branch_stage is the stage name, never the cursor:<id> fallback."""
    rollout = _rollout(tmp_path)
    rollout._environment, rollout._env = FakeEnv(), FakeSnapSandbox()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rollout._rollout_dir = run_dir
    await _capture_env_ready(rollout)

    async def run_child(child):
        return 1.0

    await rollout.branch_at_stage("env-ready", 2, run_child=run_child)

    children_dir = run_dir / "branches" / "root" / "children"
    provenance = json.loads((children_dir / "n2" / "provenance.json").read_text())
    assert provenance["branch_stage"] == "env-ready"
    assert provenance["snapshot_ref"]["environment"] == "env-snap-1"
    nodes = {
        node["id"]: node
        for node in json.loads((run_dir / "tree.json").read_text())["nodes"]
    }
    assert nodes["root"]["stage"] == "env-ready"
    assert nodes["n2"]["parent"] == "root"


async def test_branch_at_stage_value_covers_only_this_fork(tmp_path: Path):
    """The stage node already carries the linear continuation, which has no
    reward — averaging it in would silently drag V toward zero."""
    rollout = _rollout(tmp_path)
    rollout._environment, rollout._env = FakeEnv(), FakeSnapSandbox()
    root = rollout._cursor
    await _capture_env_ready(rollout)

    async def run_child(child):
        return 1.0

    value = await rollout.branch_at_stage("env-ready", 2, run_child=run_child)

    assert len(root.children) == 3  # linear child + two branch children
    assert value == 1.0
    assert root.state["value"] == 1.0


async def test_branch_at_stage_on_a_stage_that_was_never_captured(tmp_path: Path):
    """Fail closed with the captured stages named — never degrade to a
    checkpoint at the cursor, which would fork a different world."""
    rollout = _rollout(tmp_path)
    env = FakeEnv()
    rollout._environment, rollout._env = env, FakeSnapSandbox()
    await rollout.mark_stage("env-ready")

    async def run_child(child):
        return 1.0

    with pytest.raises(BranchStageNotCaptured, match="post-research") as exc:
        await rollout.branch_at_stage("post-research", 2, run_child=run_child)

    assert "['env-ready']" in str(exc.value)
    assert len(env.snapshots) == 1
    assert env.restored == []


async def test_branch_at_stage_rejects_layers_the_stage_did_not_capture(
    tmp_path: Path,
):
    rollout = _rollout(tmp_path)
    rollout._environment, rollout._env = FakeEnv(), FakeSnapSandbox()
    await rollout.mark_stage("env-ready")

    async def run_child(child):
        return 1.0

    with pytest.raises(ValueError, match="disagrees with the layers"):
        await rollout.branch_at_stage(
            "env-ready",
            2,
            run_child=run_child,
            snapshot_layers={"environment", "sandbox"},
        )


class RefLessEnv(FakeEnv):
    """An Environment plane whose ``snapshot()`` yields no ref.

    A third-party plane, not the shipped ``ManifestEnvironment``. Nothing
    between the protocol and ``checkpoint_composed`` requires the returned
    handle to be non-``None``, so a plane shaped like this produces a
    ``StageSnapshot`` with no layer refs at all.
    """

    async def snapshot(self) -> StateSnapshot:  # type: ignore[override]
        self.calls.append("env.snapshot")
        return None  # type: ignore[return-value]


async def test_branch_at_stage_rejects_a_stage_that_captured_no_layer(
    tmp_path: Path,
):
    """A stage branch enforces the non-empty layer check the cursor path does.

    Guards the fix from "fix(branch): stage branches enforce the non-empty
    layer check". ``_resolve_layers`` — which rejects an empty layer set — ran
    only on the cursor arm. On the ``at_stage`` arm the layers were *derived*
    from the recorded snapshot's refs, and a snapshot carrying neither ref
    derived the empty set: the capability gate then had nothing to check,
    ``restore_composed(environment=None, sandbox=None)`` rolled nothing back
    before each child, and provenance recorded a clean stage fork. Every child
    would have run in the world the previous one left, and the fork would have
    published a V computed across them.

    ``pre-verify`` rather than ``env-ready``: at ``env-ready`` the
    fresh-children gate already demands the container layer, so the hole is
    only reachable at a boundary whose children run in place.
    """
    rollout = _rollout(tmp_path)
    env = RefLessEnv()
    rollout._environment, rollout._env = env, FakeSnapSandbox()
    await rollout.mark_stage("pre-verify")
    recorded = rollout.stage_snapshots["pre-verify"]
    assert recorded.environment_ref is None and recorded.sandbox_ref is None

    ran: list[str] = []

    async def run_child(child):
        ran.append(child.id)
        return 1.0

    with pytest.raises(ValueError, match="at least one layer") as excinfo:
        await rollout.branch_at_stage("pre-verify", 2, run_child=run_child)

    assert "pre-verify" in str(excinfo.value)  # names the fork that has no point
    # fail closed: nothing restored, no child run, no fork recorded
    assert env.restored == []
    assert ran == []
    assert rollout.tree.root.children == []


async def test_branch_at_stage_composes_both_layers_when_the_stage_did(
    tmp_path: Path,
):
    """A stage captured with both layers restores sandbox-then-env per child."""
    calls: list[str] = []
    rollout = _rollout(tmp_path)
    env, sandbox = FakeEnv(calls), FakeSnapSandbox(calls)
    rollout._environment, rollout._env = env, sandbox
    await rollout.mark_stage("env-ready", snapshot_layers={"environment", "sandbox"})
    calls.clear()

    async def run_child(child):
        return 1.0

    await rollout.branch_at_stage("env-ready", 2, run_child=run_child)

    assert calls == [
        "sandbox.restore",
        "env.restore",
        "sandbox.restore",
        "env.restore",
    ]


async def test_branch_at_stage_reuses_the_engine_delta_validation(tmp_path: Path):
    """Stage branching is the ordinary engine path — a not-yet-executable
    delta still fails closed before any child runs."""
    rollout = _rollout(tmp_path)
    env = FakeEnv()
    rollout._environment, rollout._env = env, FakeSnapSandbox()
    await rollout.mark_stage("env-ready")

    async def run_child(child):
        return 1.0

    with pytest.raises(BranchDeltaNotSupported, match="skill_mode"):
        await rollout.branch_at_stage(
            "env-ready",
            2,
            run_child=run_child,
            deltas=[None, BranchDelta(skill_mode="with-skill")],
        )

    assert env.restored == []


async def test_a_childs_own_stage_capture_stays_scoped_to_the_child(
    tmp_path: Path,
):
    """Branch children are isolated sub-rollouts: a child running through its
    own pre-verify boundary must not become the *parent's* pre-verify — a
    later branch there would fork the child's world, not the parent's."""
    rollout = _rollout(tmp_path, snapshot_stages={"pre-verify"})
    env = FakeEnv()
    rollout._environment, rollout._env = env, FakeSnapSandbox()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rollout._rollout_dir = run_dir
    await _capture_env_ready(rollout)

    async def run_child(child):
        await rollout._capture_stage("pre-verify")
        return 1.0

    await rollout.branch_at_stage("env-ready", 2, run_child=run_child)

    assert list(rollout.stage_snapshots) == ["env-ready"]
    recorded = json.loads((run_dir / "stage_snapshots.json").read_text())
    assert list(recorded["stages"]) == ["env-ready"]


# 7. The stage registry artifact


async def test_stage_snapshots_json_is_deterministic(tmp_path: Path):
    """Sorted keys, per-layer refs, the layers each stage captured — and
    rewritten in full on every capture so a partial run still has evidence."""
    rollout = _rollout(tmp_path)
    rollout._environment, rollout._env = FakeEnv(), FakeSnapSandbox()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rollout._rollout_dir = run_dir

    await rollout.mark_stage("pre-verify")
    partial = (run_dir / "stage_snapshots.json").read_text()
    await rollout.mark_stage("env-ready", snapshot_layers={"environment", "sandbox"})

    assert json.loads(partial)["stages"] == {
        "pre-verify": {
            "environment_ref": "env-snap-1",
            "sandbox_ref": None,
            "layers": ["environment"],
        }
    }
    assert (run_dir / "stage_snapshots.json").read_text() == (
        "{\n"
        '  "schema_version": 1,\n'
        '  "stages": {\n'
        '    "env-ready": {\n'
        '      "environment_ref": "env-snap-2",\n'
        '      "layers": [\n'
        '        "environment",\n'
        '        "sandbox"\n'
        "      ],\n"
        '      "sandbox_ref": "bf-snap-1"\n'
        "    },\n"
        '    "pre-verify": {\n'
        '      "environment_ref": "env-snap-1",\n'
        '      "layers": [\n'
        '        "environment"\n'
        "      ],\n"
        '      "sandbox_ref": null\n'
        "    }\n"
        "  }\n"
        "}\n"
    )
