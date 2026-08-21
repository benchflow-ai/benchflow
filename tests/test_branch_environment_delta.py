"""Regression tests for executing service-level environment_ref branch deltas.

Guards "feat(branch): execute service-level environment_ref deltas from
env-ready" (docs/rollout-branching-rfc.md §3.3 — the ``env0@prod`` vs
``env0@outage`` tool-outage perturbation; FrontierPhysics#73). PR number to be
added on submission.

``environment_ref`` was schema- and provenance-stable but fail-closed. The
executable slice is *service topology*: the env-ready snapshot commits the
**parent's** container, and restoring it kills every framework-started service
with it — so for a manifest pair sharing the same image with
``owns_lifecycle = false``, what the fresh child provisions over the restore
IS the child's service set, and swapping the manifest executes the delta
soundly (the RFC's "service stop/start bracketing around state restore",
§3.1/§3.3). Everything outside that slice fails closed before anything is
quiesced: an image-changing manifest breaks the restore-the-parent-container
premise (it would need a rebuild path, which contradicts branching from a
snapshot), and an entrypoint-owned lifecycle starts whatever the image bakes
in, so a recorded service delta would not be enforced.

The provisioning step also covers the control arm: a zero-delta child of a
manifest-bound parent re-provisions the *parent's* manifest, so the baseline
arm of a framework-started environment no longer scores a world whose services
all died with the container restore.

Unit tests against fakes — no Docker, Daytona, or API keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchflow.branch_delta import BranchDelta
from benchflow.branch_skill import (
    BranchEnvironmentImageConflict,
    resolve_environment_ref_delta,
)
from benchflow.environment.manifest import load_manifest
from benchflow.rollout_branch import BranchDeltaNotSupported
from tests.test_branch_skill_delta import (
    FakePlanes,
    _capture_env_ready,
    _fake_verifier,
    _parent,
    _task_dir,
)

IMAGE = "bf-env0:2026.08"

_SERVICE_BLOCKS = {
    "gmail": '[[environment.services]]\nname = "gmail"\ncommand = "claw-gmail --port 8081"\nport = 8081\n',
    "gcal": '[[environment.services]]\nname = "gcal"\ncommand = "claw-gcal --port 8082"\nport = 8082\n',
}


def _manifest_toml(
    *, image: str = IMAGE, services: tuple[str, ...], owns_lifecycle: bool = False
) -> str:
    blocks = "\n".join(_SERVICE_BLOCKS[name] for name in services)
    return (
        "[environment]\n"
        'name = "env0"\n'
        f'image = "{image}"\n'
        f"owns_lifecycle = {str(owns_lifecycle).lower()}\n\n" + blocks
    )


def _registry(tmp_path: Path, monkeypatch) -> Path:
    """A local env registry holding the documented prod/outage pair plus the
    two unbranchable variants the gates exist for."""
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "env0@prod.toml").write_text(_manifest_toml(services=("gmail", "gcal")))
    (registry / "env0@outage.toml").write_text(_manifest_toml(services=("gmail",)))
    (registry / "env0@rebuilt.toml").write_text(
        _manifest_toml(image="bf-env0:rebuilt", services=("gmail",))
    )
    (registry / "env0@baked.toml").write_text(
        _manifest_toml(services=(), owns_lifecycle=True)
    )
    monkeypatch.setenv("BENCHFLOW_ENV_REGISTRY", str(registry))
    return registry


class FakeManifestEnvironment:
    """Environment-plane stand-in recording what was provisioned, and when."""

    ready = True

    def __init__(self, manifest, calls: list[str]) -> None:
        self.manifest = manifest
        self.calls = calls
        self.provision_ctx: Any = None
        self.torn_down = False

    async def provision(self, ctx: Any) -> Any:
        self.provision_ctx = ctx
        self.calls.append(
            f"provision:{','.join(s.name for s in self.manifest.services)}"
        )
        return SimpleNamespace(name=self.manifest.name, endpoints={})

    async def readiness(self) -> Any:
        self.calls.append("readiness")
        return SimpleNamespace(
            ready=self.ready,
            checked=[f"http://localhost:{s.port}" for s in self.manifest.services],
            error=None if self.ready else "gmail never responded",
        )

    async def teardown(self) -> None:
        self.torn_down = True


class EnvPlanes(FakePlanes):
    """FakePlanes with the manifest-environment factory the child provisions
    through, recording every plane it builds and the order of the calls."""

    environment_cls = FakeManifestEnvironment

    def __init__(self) -> None:
        super().__init__()
        self.environments: list[FakeManifestEnvironment] = []
        self.calls: list[str] = []

    def manifest_environment(self, manifest, *, sandbox):
        environment = self.environment_cls(manifest, self.calls)
        self.environments.append(environment)
        return environment

    async def deploy_skills(self, env, task_path, skills_dir, *args, **kwargs):
        self.calls.append("deploy_skills")
        await super().deploy_skills(env, task_path, skills_dir, *args, **kwargs)


def _manifest_parent(tmp_path: Path, monkeypatch, *, planes: EnvPlanes):
    """A manifest-bound parent positioned as if it had run past env-ready."""
    _registry(tmp_path, monkeypatch)
    return _parent(
        _task_dir(tmp_path),
        tmp_path,
        planes=planes,
        environment_manifest=load_manifest("env0@prod"),
    )


# 1. The delta executes: the child provisions the child manifest's service set


async def test_environment_delta_child_provisions_the_child_manifests_services(
    tmp_path: Path, monkeypatch
):
    """The tool-outage ablation, executed for real.

    The control child re-provisions the parent's own manifest (both services);
    the delta child provisions the outage manifest's subset over the same
    restored container — asserted on the provisioned service sets, which is
    the world each arm actually ran in, not a recorded label.
    """
    planes = EnvPlanes()
    rollout = _manifest_parent(tmp_path, monkeypatch, planes=planes)
    _fake_verifier(monkeypatch)
    await _capture_env_ready(rollout)

    value = await rollout.branch_at_stage(
        "env-ready",
        2,
        deltas=[None, BranchDelta(environment_ref="env0@outage")],
    )

    assert value == 1.0
    control_env, delta_env = planes.environments
    assert [s.name for s in control_env.manifest.services] == ["gmail", "gcal"]
    assert [s.name for s in delta_env.manifest.services] == ["gmail"]
    assert control_env.provision_ctx == {"task_id": "task"}
    assert delta_env.provision_ctx == {"task_id": "task"}
    # both children's planes were torn down by their own cleanup()
    assert control_env.torn_down and delta_env.torn_down
    # ... and the parent's own config still binds the prod manifest
    assert [s.name for s in rollout._config.environment_manifest.services] == [
        "gmail",
        "gcal",
    ]


async def test_environment_delta_child_config_json_records_the_swapped_manifest(
    tmp_path: Path, monkeypatch
):
    """The child is a first-class rollout, so its own config.json records the
    manifest it actually provisioned — name, image, and the differing service
    set — while the control child records the parent's."""
    planes = EnvPlanes()
    rollout = _manifest_parent(tmp_path, monkeypatch, planes=planes)
    _fake_verifier(monkeypatch)
    await _capture_env_ready(rollout)

    await rollout.branch_at_stage(
        "env-ready",
        2,
        deltas=[None, BranchDelta(environment_ref="env0@outage")],
    )

    children_dir = rollout._rollout_dir / "branches" / "root" / "children"
    control = json.loads((children_dir / "n2" / "config.json").read_text())
    child = json.loads((children_dir / "n3" / "config.json").read_text())
    assert control["environment_manifest"]["services"] == ["gmail", "gcal"]
    assert child["environment_manifest"]["services"] == ["gmail"]
    assert child["environment_manifest"]["image"] == IMAGE
    assert child["environment_manifest"]["owns_lifecycle"] is False


async def test_child_environment_is_provisioned_after_restore_before_install(
    tmp_path: Path, monkeypatch
):
    """The RFC §3.1 bracket: services come up after the container roll-back
    (which killed the previous child's) and before install_agent(), mirroring
    the linear lifecycle's start() -> install_agent() order."""
    planes = EnvPlanes()
    rollout = _manifest_parent(tmp_path, monkeypatch, planes=planes)
    _fake_verifier(monkeypatch)
    sandbox = rollout._env
    original_restore = sandbox.restore

    async def recording_restore(image):
        planes.calls.append("sandbox.restore")
        await original_restore(image)

    sandbox.restore = recording_restore
    await _capture_env_ready(rollout)

    await rollout.branch_at_stage(
        "env-ready",
        2,
        deltas=[None, BranchDelta(environment_ref="env0@outage")],
    )

    assert planes.calls == [
        "sandbox.restore",
        "provision:gmail,gcal",
        "readiness",
        "deploy_skills",
        "sandbox.restore",
        "provision:gmail",
        "readiness",
        "deploy_skills",
    ]


async def test_zero_delta_children_reprovision_the_parents_manifest(
    tmp_path: Path, monkeypatch
):
    """The control arm's honesty: restoring the container kills every
    framework-started service, so a zero-delta child of a manifest-bound
    parent re-provisions the parent's own service set — without this, the
    baseline arm of an outage comparison scores a dead world and calls it
    prod."""
    planes = EnvPlanes()
    rollout = _manifest_parent(tmp_path, monkeypatch, planes=planes)
    _fake_verifier(monkeypatch)
    await _capture_env_ready(rollout)

    await rollout.branch_at_stage("env-ready", 2)

    assert [[s.name for s in env.manifest.services] for env in planes.environments] == [
        ["gmail", "gcal"],
        ["gmail", "gcal"],
    ]


async def test_a_failed_readiness_gate_fails_the_child_loudly(
    tmp_path: Path, monkeypatch
):
    """A child whose swapped environment never became ready must not score:
    the gate raises (the same contract as start()), the fork ends, and the
    child's own cleanup still tears its plane down."""

    class NotReadyEnvironment(FakeManifestEnvironment):
        ready = False

    class NotReadyPlanes(EnvPlanes):
        environment_cls = NotReadyEnvironment

    planes = NotReadyPlanes()
    rollout = _manifest_parent(tmp_path, monkeypatch, planes=planes)
    _fake_verifier(monkeypatch)
    await _capture_env_ready(rollout)

    with pytest.raises(RuntimeError, match="not ready"):
        await rollout.branch_at_stage(
            "env-ready",
            2,
            deltas=[None, BranchDelta(environment_ref="env0@outage")],
        )

    assert planes.environments[0].torn_down
    assert planes.deployments == []  # readiness gates before install_agent()


# 2. Provenance: the ref is recorded verbatim on the child


async def test_provenance_records_the_environment_ref_verbatim(
    tmp_path: Path, monkeypatch
):
    """Per-child provenance.json and tree.json carry the registry ref exactly
    as requested (it is already content-addressed by the registry), plus the
    fresh-rollout execution marker."""
    planes = EnvPlanes()
    rollout = _manifest_parent(tmp_path, monkeypatch, planes=planes)
    _fake_verifier(monkeypatch)
    await _capture_env_ready(rollout)

    await rollout.branch_at_stage(
        "env-ready",
        2,
        deltas=[None, BranchDelta(environment_ref="env0@outage")],
    )

    children_dir = rollout._rollout_dir / "branches" / "root" / "children"
    control = json.loads((children_dir / "n2" / "provenance.json").read_text())
    child = json.loads((children_dir / "n3" / "provenance.json").read_text())
    assert control["delta"]["environment_ref"] is None
    assert child["delta"]["environment_ref"] == "env0@outage"
    assert child["delta_execution"] == "fresh-rollout"
    nodes = {
        node["id"]: node
        for node in json.loads((rollout._rollout_dir / "tree.json").read_text())[
            "nodes"
        ]
    }
    assert nodes["n3"]["delta"]["environment_ref"] == "env0@outage"


# 3. The boundary: everything outside the services slice fails closed


async def test_an_image_changing_manifest_fails_closed_with_the_typed_error(
    tmp_path: Path, monkeypatch
):
    """The restore-the-parent-container premise: the snapshot commits the
    parent's image, so a manifest naming a different image needs a rebuild
    path — which contradicts branching from a snapshot. The typed error names
    both images and fires before anything is restored or forked."""
    planes = EnvPlanes()
    rollout = _manifest_parent(tmp_path, monkeypatch, planes=planes)
    await _capture_env_ready(rollout)

    with pytest.raises(BranchEnvironmentImageConflict) as excinfo:
        await rollout.branch_at_stage(
            "env-ready",
            2,
            deltas=[None, BranchDelta(environment_ref="env0@rebuilt")],
        )

    message = str(excinfo.value)
    assert IMAGE in message and "bf-env0:rebuilt" in message
    assert "rebuild" in message
    assert "contradicts branching from a snapshot" in message
    assert isinstance(excinfo.value, BranchDeltaNotSupported)
    assert planes.environments == []
    assert rollout._env.restored == []
    assert [node for node in rollout.tree.nodes() if "delta" in node.state] == []


async def test_an_entrypoint_owned_lifecycle_fails_closed(tmp_path: Path, monkeypatch):
    """Same image, but the child manifest hands the lifecycle to the image
    entrypoint: the restored container starts whatever the image bakes in, so
    the framework cannot enforce a manifest-declared service difference — the
    delta would be recorded but not executed. Fail closed instead."""
    planes = EnvPlanes()
    rollout = _manifest_parent(tmp_path, monkeypatch, planes=planes)
    await _capture_env_ready(rollout)

    with pytest.raises(BranchDeltaNotSupported, match="owns_lifecycle"):
        await rollout.branch_at_stage(
            "env-ready",
            2,
            deltas=[None, BranchDelta(environment_ref="env0@baked")],
        )

    assert planes.environments == []
    assert rollout._env.restored == []


async def test_an_unresolvable_ref_fails_closed_naming_the_registry_problem(
    tmp_path: Path, monkeypatch
):
    """A ref the registry cannot resolve is a delta that cannot be recorded
    honestly, let alone executed — fail closed with the resolution error."""
    planes = EnvPlanes()
    rollout = _manifest_parent(tmp_path, monkeypatch, planes=planes)
    await _capture_env_ready(rollout)

    with pytest.raises(BranchDeltaNotSupported, match="does not resolve"):
        await rollout.branch_at_stage(
            "env-ready",
            2,
            deltas=[None, BranchDelta(environment_ref="env0@nonexistent")],
        )

    assert planes.environments == []


async def test_environment_delta_at_a_cursor_branch_names_env_ready(
    tmp_path: Path, monkeypatch
):
    """A cursor branch keeps the parent's provisioned services alive across
    the fork, so the swap could only be recorded, never enforced — fail closed
    naming the boundary that works."""
    planes = EnvPlanes()
    rollout = _manifest_parent(tmp_path, monkeypatch, planes=planes)
    env = rollout._environment

    with pytest.raises(BranchDeltaNotSupported, match="environment_ref") as excinfo:
        await rollout.branch(
            2,
            snapshot_layers={"environment", "sandbox"},
            deltas=[None, BranchDelta(environment_ref="env0@outage")],
        )

    assert "env-ready" in str(excinfo.value)
    assert env.snapshots == []
    assert rollout._cursor.children == []


async def test_environment_delta_with_an_explicit_run_child_is_rejected(
    tmp_path: Path, monkeypatch
):
    """A caller-supplied runner owns the child's execution, so the engine
    cannot provision the child manifest's services."""
    planes = EnvPlanes()
    rollout = _manifest_parent(tmp_path, monkeypatch, planes=planes)
    await _capture_env_ready(rollout)

    async def run_child(child):
        return 1.0

    with pytest.raises(ValueError, match="run_child"):
        await rollout.branch_at_stage(
            "env-ready",
            2,
            run_child=run_child,
            deltas=[None, BranchDelta(environment_ref="env0@outage")],
        )

    assert planes.environments == []


# 4. The resolution helper is the single source of the boundary


def test_resolve_environment_ref_delta_returns_the_outage_manifest(
    tmp_path: Path, monkeypatch
):
    """The happy path of the shared gate (engine validation, the child runner,
    and the ablate pre-flight all call this one function)."""
    _registry(tmp_path, monkeypatch)
    parent = load_manifest("env0@prod")

    child = resolve_environment_ref_delta(parent, "env0@outage")

    assert child.image == parent.image
    assert [s.name for s in child.services] == ["gmail"]
    assert child.owns_lifecycle is False


def test_resolve_environment_ref_delta_requires_a_manifest_bound_parent(
    tmp_path: Path, monkeypatch
):
    """No parent manifest means no Environment plane to swap: the child would
    run a world its snapshot never contained."""
    _registry(tmp_path, monkeypatch)

    with pytest.raises(BranchDeltaNotSupported, match="no environment manifest"):
        resolve_environment_ref_delta(None, "env0@outage", subject="arm 'env:...'")
