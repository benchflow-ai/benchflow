"""Regression tests for executing skill_mode branch deltas.

Guards the skill-delta execution path ("feat(branch): execute skill_mode deltas
as fresh rollouts from the env-ready snapshot"; docs/rollout-branching-rfc.md
§3.3 / WS-4b; FrontierPhysics#73). PR number to be added on submission.

``skill_mode`` was schema-stable but fail-closed: skills are deployed by
``install_agent()``, so a branch at the cursor forks a world that has already
resolved the question. The ``env-ready`` stage snapshot (WS-4a) is taken before
``install_agent()``, so a child restored from it can re-run installation under
the switched mode — as a *fresh* Rollout over the restored sandbox
(``use_prebuilt_env``, #388). These tests pin that the child really varies
skills (the staged build context and the deploy call, not just a recorded
label), that every other branch point still fails closed, and that lineage
records the effective mode and the fresh-rollout execution.

These are unit tests against fakes — no Docker, Daytona, or API keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchflow.branch_delta import BranchDelta
from benchflow.branch_skill import child_skill_config, resolve_child_skill_policy
from benchflow.environment.protocol import StateSnapshot
from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.rollout_branch import (
    _EXECUTABLE_DELTA_FIELDS,
    _UNSUPPORTED_DELTA_FIELDS,
    BranchDeltaNotSupported,
)
from benchflow.sandbox.protocol import SandboxImage
from benchflow.trajectories.tree import Step

SKILL_BODY = "---\nname: demo\n---\n\nUse the demo skill.\n"
DOCKERFILE = "FROM python:3.12-slim\nCOPY skills /skills\nCOPY . /app\n"
# One tool call: a rollout that ends with zero tokens AND zero tool calls is
# classified as a suspected silent API failure and has its reward nulled, so a
# fake agent has to show some activity to be scoreable.
AGENT_EVENTS: list[dict[str, Any]] = [{"type": "tool_call", "tool_name": "bash"}]


def _task_dir(tmp_path: Path, *, bundled_skills: bool = True) -> Path:
    """A minimal real task — the skill policy resolves against this on disk."""
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Solve the task.")
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
    if bundled_skills:
        skill = task / "environment" / "skills" / "demo"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(SKILL_BODY)
    return task


class FakeEnv:
    """Environment-plane stand-in recording snapshot/restore calls."""

    def __init__(self) -> None:
        self.snapshots: list[StateSnapshot] = []
        self.restored: list[StateSnapshot] = []

    async def snapshot(self) -> StateSnapshot:
        snap = StateSnapshot(id=f"env-snap-{len(self.snapshots) + 1}", path="/tmp/x")
        self.snapshots.append(snap)
        return snap

    async def restore(self, snap: StateSnapshot) -> None:
        self.restored.append(snap)


class FakeSandbox:
    """Snapshot-capable Sandbox stand-in; ``exec`` answers the workspace probe."""

    supports_snapshot = True

    def __init__(self) -> None:
        self.snapshots: list[SandboxImage] = []
        self.restored: list[SandboxImage] = []
        self.stopped = False

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        img = SandboxImage(provider="fake", ref=f"bf-snap-{len(self.snapshots) + 1}")
        self.snapshots.append(img)
        return img

    async def restore(self, image: SandboxImage) -> None:
        self.restored.append(image)

    async def exec(self, command: str, **_kw: Any) -> Any:
        return SimpleNamespace(return_code=0, stdout="/app", stderr="")

    async def stop(self, delete: bool = False) -> None:
        self.stopped = True


class FakePlanes:
    """Recording stand-in for the concrete plane bundle (contracts.RolloutPlanes).

    ``deploy_skills`` is the call that actually puts (or does not put) the pack
    into the sandbox, so it records what the rollout resolved *and* what the
    staged build context looked like at that moment — the temp task copy is
    deleted by cleanup(), so the evidence has to be taken while it exists.
    """

    def __init__(self) -> None:
        self.deployments: list[dict[str, Any]] = []
        self.dockerfile_injections: list[Path] = []

    # --- host-side setup -------------------------------------------------
    def install_docker_compat(self) -> None:
        return None

    def resolve_locked_paths(self, sandbox_user, locked_paths):
        return []

    def resolve_agent_env(self, agent, model, agent_env):
        return dict(agent_env or {})

    def agent_launch(self, agent, *, disallow_web_tools):
        return "oracle"

    def inject_skills_into_dockerfile(
        self, task_path, skills_dir, *, sandbox_dir="/skills"
    ):
        self.dockerfile_injections.append(Path(skills_dir))

    def extract_usage(self, runtime):
        return {"usage_source": "unavailable"}

    # --- install_agent (oracle path) -------------------------------------
    async def setup_sandbox_user(
        self, env, sandbox_user, *, workspace, timeout_sec=120
    ):
        return workspace

    async def snapshot_build_config(self, env, *, workspace):
        return None

    async def seed_verifier_workspace(self, env, *, workspace, sandbox_user):
        return None

    async def deploy_skills(self, env, task_path, skills_dir, *args, **kwargs):
        staged = Path(task_path)
        dockerfile = staged / "environment" / "Dockerfile"
        self.deployments.append(
            {
                "skills_dir": Path(skills_dir) if skills_dir is not None else None,
                "skill_files": sorted(
                    p.parent.name for p in Path(skills_dir).glob("*/SKILL.md")
                )
                if skills_dir is not None
                else [],
                "staged_skills_dir_exists": (
                    staged / "environment" / "skills"
                ).is_dir(),
                "dockerfile": dockerfile.read_text() if dockerfile.exists() else "",
            }
        )

    async def lockdown_paths(self, env, locked_paths):
        return None

    # --- connect / execute ------------------------------------------------
    async def ensure_litellm_runtime(self, *args, **kwargs):
        return kwargs.get("agent_env", {}), None

    async def connect_acp(self, *args, **kwargs):
        async def close() -> None:
            return None

        return SimpleNamespace(close=close), SimpleNamespace(), None, "oracle"

    async def execute_prompts(self, client, session, prompts, timeout, **kwargs):
        return list(AGENT_EVENTS), 1


def _fake_verifier(monkeypatch, reward: float = 1.0) -> None:
    """Fake the child's verifier I/O; everything before it stays real."""

    async def fake_publish(*_a, **_kw):
        return None

    async def fake_verify_rollout(*_a, **_kw):
        return {"reward": reward}, None, None

    monkeypatch.setattr(
        "benchflow.rollout._publish_trajectory_for_verifier", fake_publish
    )
    monkeypatch.setattr("benchflow.rollout._verify_rollout", fake_verify_rollout)


def _parent(
    task: Path,
    tmp_path: Path,
    *,
    planes: FakePlanes,
    skill_mode: str = "no-skill",
    **config: Any,
) -> Rollout:
    """A parent rollout positioned as if it had run past env-ready."""
    rollout = Rollout(
        RolloutConfig(
            task_path=task,
            scenes=[Scene.single(agent="oracle")],
            jobs_dir=tmp_path / "jobs",
            skill_mode=skill_mode,
            planes=planes,
            **config,
        )
    )
    rollout._environment, rollout._env = FakeEnv(), FakeSandbox()
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    rollout._rollout_dir = run_dir
    return rollout


async def _capture_env_ready(
    rollout: Rollout, *, layers: set[str] | None = None
) -> None:
    """Record env-ready at the root, then run past it linearly."""
    await rollout.mark_stage(
        "env-ready",
        snapshot_layers={"environment", "sandbox"} if layers is None else layers,
    )
    rollout._cursor = rollout._tree.advance(rollout._cursor, Step(id="s1"))


# 1. The ablation: a skill_mode child runs a fresh rollout with the switched mode


async def test_skill_delta_children_deploy_different_skills_at_env_ready(
    tmp_path: Path, monkeypatch
):
    """The with-skill/no-skill ablation, executed for real.

    Each child re-runs install_agent() as its own Rollout over the restored
    env-ready sandbox, so the evidence is the deploy call itself: the with-skill
    child deploys the task's bundled pack, the no-skill child deploys nothing
    AND its staged build context no longer contains the pack or the COPY line
    that would smuggle it in.
    """
    planes = FakePlanes()
    task = _task_dir(tmp_path)
    rollout = _parent(task, tmp_path, planes=planes)
    _fake_verifier(monkeypatch)
    await _capture_env_ready(rollout)

    value = await rollout.branch_at_stage(
        "env-ready",
        2,
        deltas=[
            BranchDelta(skill_mode="with-skill"),
            BranchDelta(skill_mode="no-skill"),
        ],
    )

    assert value == 1.0
    with_skill, no_skill = planes.deployments
    assert with_skill["skill_files"] == ["demo"]
    assert with_skill["staged_skills_dir_exists"] is True
    assert "COPY skills /skills" in with_skill["dockerfile"]
    assert no_skill["skills_dir"] is None
    assert no_skill["skill_files"] == []
    assert no_skill["staged_skills_dir_exists"] is False
    assert "COPY skills /skills" not in no_skill["dockerfile"]
    # the pack is stripped from the child's staged copy, never from the task
    assert (task / "environment" / "skills" / "demo" / "SKILL.md").exists()


async def test_skill_delta_child_is_a_separate_rollout_over_the_parent_sandbox(
    tmp_path: Path, monkeypatch
):
    """Fresh rollout, same (restored) sandbox: the child owns its own config,
    run directory and result, and must not stop the caller's container (#388)."""
    planes = FakePlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    sandbox = rollout._env
    _fake_verifier(monkeypatch)
    await _capture_env_ready(rollout)
    parent_task_path = rollout._effective_task_path

    await rollout.branch_at_stage(
        "env-ready",
        2,
        deltas=[
            BranchDelta(skill_mode="with-skill"),
            BranchDelta(skill_mode="no-skill"),
        ],
    )

    children_dir = rollout._rollout_dir / "branches" / "root" / "children"
    configs = {
        child.name: json.loads((child / "config.json").read_text())
        for child in sorted(children_dir.iterdir())
    }
    assert [cfg["skill_mode"] for cfg in configs.values()] == [
        "with-skill",
        "no-skill",
    ]
    assert [cfg["include_task_skills"] for cfg in configs.values()] == [True, False]
    assert all((child / "result.json").exists() for child in children_dir.iterdir())
    # the parent instance was never re-entered: its own setup state is untouched
    assert rollout._effective_task_path == parent_task_path
    assert rollout._config.skill_mode == "no-skill"
    assert sandbox.stopped is False
    assert len(sandbox.restored) == 2  # one container roll-back per child


async def test_skill_delta_child_restores_the_snapshot_before_installing(
    tmp_path: Path, monkeypatch
):
    """Order matters: a child that installed before the roll-back would deploy
    into the previous child's container."""
    planes = FakePlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    _fake_verifier(monkeypatch)
    calls: list[str] = []
    sandbox = rollout._env
    original_restore = sandbox.restore

    async def recording_restore(image):
        calls.append("sandbox.restore")
        await original_restore(image)

    sandbox.restore = recording_restore
    original_deploy = planes.deploy_skills

    async def recording_deploy(*args, **kwargs):
        calls.append("deploy_skills")
        await original_deploy(*args, **kwargs)

    planes.deploy_skills = recording_deploy
    await _capture_env_ready(rollout)

    await rollout.branch_at_stage(
        "env-ready",
        2,
        deltas=[
            BranchDelta(skill_mode="no-skill"),
            BranchDelta(skill_mode="with-skill"),
        ],
    )

    assert calls == [
        "sandbox.restore",
        "deploy_skills",
        "sandbox.restore",
        "deploy_skills",
    ]


async def test_skill_delta_child_delivers_an_injected_prompt(
    tmp_path: Path, monkeypatch
):
    """A skill delta that also injects a prompt: the fresh rollout runs the
    injection as its continuation prompt, not the task's base prompt."""
    planes = FakePlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    _fake_verifier(monkeypatch)
    prompts: list[list[str]] = []

    async def recording_execute(client, session, sent, timeout, **kwargs):
        prompts.append(list(sent))
        return list(AGENT_EVENTS), 1

    planes.execute_prompts = recording_execute
    await _capture_env_ready(rollout)

    await rollout.branch_at_stage(
        "env-ready",
        2,
        deltas=[
            BranchDelta(skill_mode="with-skill"),
            BranchDelta(skill_mode="no-skill", injected_prompt="Follow PLAN.md."),
        ],
    )

    assert prompts == [["Solve the task."], ["Follow PLAN.md."]]


# 2. Lineage: the effective mode and the fresh-rollout execution are recorded


async def test_provenance_records_the_effective_mode_and_fresh_rollout(
    tmp_path: Path, monkeypatch
):
    """Per-child provenance.json carries the delta's skill_mode verbatim and
    delta_execution=fresh-rollout; tree.json carries the same on the node."""
    planes = FakePlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    _fake_verifier(monkeypatch)
    await _capture_env_ready(rollout)

    await rollout.branch_at_stage(
        "env-ready",
        2,
        deltas=[
            BranchDelta(skill_mode="with-skill"),
            BranchDelta(skill_mode="no-skill"),
        ],
    )

    run_dir = rollout._rollout_dir
    children_dir = run_dir / "branches" / "root" / "children"
    provenance = [
        json.loads((children_dir / child_id / "provenance.json").read_text())
        for child_id in ("n2", "n3")
    ]
    assert [p["delta"]["skill_mode"] for p in provenance] == [
        "with-skill",
        "no-skill",
    ]
    assert [p["delta_execution"] for p in provenance] == [
        "fresh-rollout",
        "fresh-rollout",
    ]
    assert [p["branch_stage"] for p in provenance] == ["env-ready", "env-ready"]
    assert provenance[0]["snapshot_ref"] == {
        "environment": "env-snap-1",
        "sandbox": "bf-snap-1",
    }
    nodes = {
        node["id"]: node
        for node in json.loads((run_dir / "tree.json").read_text())["nodes"]
    }
    assert nodes["n2"]["delta_execution"] == "fresh-rollout"
    assert "delta_execution" not in nodes["root"]


async def test_the_child_rollout_carries_the_branch_source_provenance(
    tmp_path: Path, monkeypatch
):
    """The child is a first-class rollout, so its own config.json/result.json
    record which rollout it forked from — the seam a continued run uses."""
    planes = FakePlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    _fake_verifier(monkeypatch, reward=0.5)
    await _capture_env_ready(rollout)

    value = await rollout.branch_at_stage(
        "env-ready",
        2,
        deltas=[
            BranchDelta(skill_mode="with-skill"),
            BranchDelta(skill_mode="no-skill"),
        ],
    )

    child_dir = rollout._rollout_dir / "branches" / "root" / "children" / "n2"
    source = json.loads((child_dir / "config.json").read_text())["source"]
    assert source["kind"] == "benchflow-branch"
    assert source["branch_stage"] == "env-ready"
    assert source["delta_execution"] == "fresh-rollout"
    assert json.loads((child_dir / "result.json").read_text())["rewards"] == {
        "reward": 0.5
    }
    assert json.loads((child_dir / "reward.json").read_text()) == {"reward": 0.5}
    assert value == 0.5


# 3. Everywhere else still fails closed


async def test_skill_delta_at_a_cursor_branch_names_env_ready(tmp_path: Path):
    """A cursor branch forks after install_agent(), so the delta could only be
    recorded, never executed — fail closed naming the boundary that works."""
    planes = FakePlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    env = rollout._environment

    with pytest.raises(BranchDeltaNotSupported, match="skill_mode") as excinfo:
        await rollout.branch(
            2,
            snapshot_layers={"environment", "sandbox"},
            deltas=[None, BranchDelta(skill_mode="with-skill")],
        )

    assert "env-ready" in str(excinfo.value)
    assert "install_agent" in str(excinfo.value)
    assert env.snapshots == []  # nothing quiesced, snapshotted, or forked
    assert rollout._cursor.children == []


async def test_skill_delta_at_another_stage_names_env_ready(
    tmp_path: Path, monkeypatch
):
    """pre-verify is a recorded boundary too — and still the wrong one."""
    planes = FakePlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    await rollout.mark_stage("pre-verify", snapshot_layers={"environment", "sandbox"})

    with pytest.raises(BranchDeltaNotSupported, match="'env-ready'") as excinfo:
        await rollout.branch_at_stage(
            "pre-verify",
            2,
            deltas=[
                BranchDelta(skill_mode="with-skill"),
                BranchDelta(skill_mode="no-skill"),
            ],
        )

    assert "'pre-verify'" in str(excinfo.value)
    assert planes.deployments == []


async def test_skill_delta_needs_the_container_layer_in_the_snapshot(
    tmp_path: Path,
):
    """An environment-state-only env-ready snapshot cannot roll the container
    back, so a no-skill child would re-install on top of the parent's pack —
    fail closed instead of measuring nothing."""
    planes = FakePlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    await _capture_env_ready(rollout, layers={"environment"})

    with pytest.raises(BranchDeltaNotSupported, match="sandbox") as excinfo:
        await rollout.branch_at_stage(
            "env-ready",
            2,
            deltas=[
                BranchDelta(skill_mode="with-skill"),
                BranchDelta(skill_mode="no-skill"),
            ],
        )

    assert "container filesystem" in str(excinfo.value)
    assert rollout._environment.restored == []
    assert planes.deployments == []


async def test_skill_delta_with_an_explicit_run_child_is_rejected(tmp_path: Path):
    """A caller-supplied runner owns the child's execution, so the engine
    cannot re-run install_agent() under the switched mode."""
    planes = FakePlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    await _capture_env_ready(rollout)

    async def run_child(child):
        return 1.0

    with pytest.raises(ValueError, match="run_child"):
        await rollout.branch_at_stage(
            "env-ready",
            2,
            run_child=run_child,
            deltas=[None, BranchDelta(skill_mode="no-skill")],
        )

    assert planes.deployments == []


async def test_with_skill_against_a_task_without_skills_fails_before_any_child(
    tmp_path: Path,
):
    """The ablation's precondition is resolved up front: a with-skill delta on
    a task shipping no pack raises the setup() error before the branch restores
    anything, not halfway through the fork."""
    planes = FakePlanes()
    task = _task_dir(tmp_path, bundled_skills=False)
    rollout = _parent(task, tmp_path, planes=planes)
    await _capture_env_ready(rollout)

    with pytest.raises(FileNotFoundError, match="no bundled skills"):
        await rollout.branch_at_stage(
            "env-ready",
            2,
            deltas=[
                BranchDelta(skill_mode="with-skill"),
                BranchDelta(skill_mode="no-skill"),
            ],
        )

    assert rollout._env.restored == []
    assert planes.deployments == []


@pytest.mark.parametrize(
    "delta",
    [
        BranchDelta(environment_ref="env0@outage"),
        BranchDelta(config_override={"agent": {"timeout_sec": 60}}),
    ],
    ids=["environment_ref", "config_override"],
)
async def test_the_other_delta_fields_still_fail_closed(
    tmp_path: Path, delta: BranchDelta
):
    """Only skill_mode graduated — the remaining axes have no execution path
    and still fail closed at the env-ready boundary too."""
    planes = FakePlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    await _capture_env_ready(rollout)

    with pytest.raises(BranchDeltaNotSupported, match="use_prebuilt_env"):
        await rollout.branch_at_stage("env-ready", 2, deltas=[None, delta])

    assert planes.deployments == []


def test_the_executable_delta_set_now_contains_skill_mode():
    """The blocklist is derived from the schema minus the executable set, so
    graduating skill_mode is one edit and every other field stays
    unsupported-by-default."""
    assert set(_EXECUTABLE_DELTA_FIELDS) == {"injected_prompt", "skill_mode"}
    assert set(_UNSUPPORTED_DELTA_FIELDS) == {"config_override", "environment_ref"}


# 4. The derived child config


def test_child_config_replaces_both_recorded_skill_modes(tmp_path: Path):
    """The skill policy resolves from recorded_skill_mode (artifact_skill_mode
    or skill_mode), so writing only one of them would leave a from_legacy-built
    parent's mode in place and the delta would silently do nothing."""
    task = _task_dir(tmp_path)
    parent = RolloutConfig.from_legacy(
        task_path=task,
        agent="oracle",
        skill_mode="with-skill",
        jobs_dir=tmp_path / "jobs",
    )
    assert parent.recorded_skill_mode == "with-skill"

    child = child_skill_config(
        parent,
        skill_mode="no-skill",
        jobs_dir=tmp_path / "jobs",
        job_name="children",
        rollout_name="n2",
    )

    assert child.recorded_skill_mode == "no-skill"
    assert child.skills_dir is None  # a no-skill config cannot carry skills_dir
    assert resolve_child_skill_policy(child, child.skill_mode).host_dir is None
    assert child.task_path == parent.task_path  # nothing else moved
    assert child.snapshot_stages == frozenset()
    assert parent.skill_mode == "with-skill"  # the parent config is untouched


def test_child_config_keeps_a_strategy_user_materializable(tmp_path: Path):
    """A loop strategy materializes a user in __post_init__; carrying both into
    the child config is rejected by RolloutConfig, so the child re-materializes
    the identical user from the same spec."""
    parent = RolloutConfig(
        task_path=_task_dir(tmp_path),
        scenes=[Scene.single(agent="oracle")],
        jobs_dir=tmp_path / "jobs",
        loop_strategy="verify-retry:k=3",
    )
    assert parent.user is not None

    child = child_skill_config(
        parent,
        skill_mode="with-skill",
        jobs_dir=tmp_path / "jobs",
        job_name="children",
        rollout_name="n2",
    )

    assert child.user is not None
    assert type(child.user) is type(parent.user)
    assert child.max_user_rounds == parent.max_user_rounds
