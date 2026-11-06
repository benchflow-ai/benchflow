"""K1-R15-2: an identity-switch task runs only where the agent's container
has no_new_privs from creation.

At 6c7d11fa5 a task with ``x-agent-identity-switch`` let its agent's
commands skip the barrier on native Daytona, Modal, AgentCore and Apple
containers, and the kill before grading cleaned the declared accounts one
after another, then the agent user: a process under an account not yet
cleaned could restart one under an account already cleaned, which survived
into grading. Such a task is now refused on those backends before any
sandbox starts, by the plan gate a run, ``bench tasks check`` and launch
share; a sandbox from an SDK plane is refused by its own property before any
agent code. Docker, remote Docker and Daytona's Docker-in-Docker keep
running it, kept services (K2) included. The kernel side is in
test_identity_switch_respawn_container.py.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox.providers import PROVIDERS_BY_NAME, SANDBOX_PROVIDERS
from benchflow.task import Task, identity_switch, keep_services
from benchflow.task.runtime_capabilities import (
    IDENTITY_SWITCH_ISSUE_PATH,
    UnsupportedTaskFeatureError,
    validate_task_runtime_support,
)

HOLDS = {
    ("docker", False),
    ("docker", True),
    ("remote-docker", False),
    ("remote-docker", True),
    ("daytona", True),
}


def _task(root: Path, *, switch: bool = True, keep: bool = False, compose=False):
    task_dir = root / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("Do it.\n")
    metadata = ""
    if switch:
        metadata += '"x-agent-identity-switch" = ["worker"]\n'
    if keep:
        metadata += '"x-keep-services" = [{ port = 5000, name = "api" }]\n'
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n'
        + (f"[metadata]\n{metadata}" if metadata else "")
        + "[agent]\ntimeout_sec = 1\n[verifier]\ntimeout_sec = 1\n"
    )
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    if compose:
        (task_dir / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main: {}\n"
        )
    return task_dir


def _switch_issues(task_dir: Path, sandbox: str):
    return [
        issue
        for issue in validate_task_runtime_support(
            Task(task_dir).config, sandbox=sandbox, task_dir=task_dir
        )
        if issue.path == IDENTITY_SWITCH_ISSUE_PATH
    ]


@pytest.mark.parametrize("compose", [False, True])
@pytest.mark.parametrize("sandbox", SANDBOX_PROVIDERS)
def test_the_plan_gate_refuses_every_backend_without_the_barrier(
    tmp_path, sandbox, compose
):
    issues = _switch_issues(_task(tmp_path, compose=compose), sandbox)
    if (sandbox, compose) in HOLDS:
        assert issues == []
        return
    [issue] = issues
    # Names the backend, the hazard, and the backend that runs it.
    assert f"the {sandbox} sandbox" in issue.reason
    assert "survive into grading" in issue.reason
    assert "--sandbox docker" in issue.reason
    assert issue.sandbox == sandbox


@pytest.mark.parametrize("sandbox", SANDBOX_PROVIDERS)
def test_a_task_without_the_key_is_never_refused_for_it(tmp_path, sandbox):
    assert _switch_issues(_task(tmp_path, switch=False), sandbox) == []


def test_daytona_without_the_task_folder_cannot_see_a_compose_file(tmp_path):
    config = Task(_task(tmp_path, compose=True)).config
    issues = validate_task_runtime_support(config, sandbox="daytona")
    assert [i.path for i in issues] == [IDENTITY_SWITCH_ISSUE_PATH]


def test_the_registry_matches_each_backends_own_property():
    """The plan gate reads the registry; the sandboxes carry the property
    generic exec and the start-time check read. They must agree."""
    from benchflow.sandbox._base import BaseSandbox
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    assert {p.agent_container_no_new_privs for p in PROVIDERS_BY_NAME.values()} <= {
        "always",
        "compose",
        "never",
    }
    assert DockerSandbox.agent_container_no_new_privileges is True
    assert RemoteDockerSandbox.agent_container_no_new_privileges is True
    assert BaseSandbox.agent_container_no_new_privileges is False
    for name, cls in _native_classes().items():
        assert PROVIDERS_BY_NAME[name].agent_container_no_new_privs != "always"
        assert cls.agent_container_no_new_privileges is False, name
    daytona = _daytona(compose=False)
    assert daytona.agent_container_no_new_privileges is False
    assert _daytona(compose=True).agent_container_no_new_privileges is True


def _native_classes():
    from benchflow.sandbox.agentcore import AgentCoreSandbox
    from benchflow.sandbox.apple_container import AppleContainerSandbox
    from benchflow.sandbox.modal_impl import ModalSandbox

    return {
        "modal": ModalSandbox,
        "agentcore": AgentCoreSandbox,
        "apple-container": AppleContainerSandbox,
    }


def _daytona(*, compose: bool):
    from benchflow.sandbox.daytona import DaytonaSandbox

    sandbox = object.__new__(DaytonaSandbox)
    sandbox._compose_mode = compose
    return sandbox


@pytest.mark.parametrize("sandbox", ["modal", "agentcore", "apple-container"])
def test_launch_refuses_before_the_sandbox_is_made(tmp_path, monkeypatch, sandbox):
    """The launch gate (also the run's plan gate, validate_task_plan) runs
    before any provider object exists."""
    from benchflow.sandbox import setup
    from benchflow.task import RolloutPaths

    made = []
    monkeypatch.setattr(setup, "_make_sandbox", lambda *a, **k: made.append(a))
    task_dir = _task(tmp_path)
    with pytest.raises(UnsupportedTaskFeatureError, match="x-agent-identity-switch"):
        setup._create_sandbox_environment(
            sandbox,
            Task(task_dir),
            task_dir,
            "r",
            RolloutPaths(rollout_dir=tmp_path / "rollout"),
        )
    assert made == []


@pytest.mark.parametrize(
    ("sandbox", "compose", "refused"),
    [
        ("docker", False, False),
        ("remote-docker", False, False),
        ("daytona", True, False),
        ("daytona", False, True),
    ],
)
def test_the_runs_plan_gate_refuses_with_no_sandbox(
    tmp_path, sandbox, compose, refused
):
    from benchflow.task.plan_validation import validate_task_plan

    task_dir = _task(tmp_path, compose=compose)

    def check():
        validate_task_plan(
            task_dir, sandbox=sandbox, agent="oracle", sandbox_user="agent"
        )

    if refused:
        with pytest.raises(UnsupportedTaskFeatureError, match="--sandbox docker"):
            check()
    else:
        check()


@pytest.mark.parametrize(
    "env",
    [
        SimpleNamespace(),
        SimpleNamespace(agent_container_no_new_privileges=False),
        # Only True itself holds, not a truthy stand-in.
        SimpleNamespace(agent_container_no_new_privileges="yes"),
    ],
)
def test_a_started_sandbox_without_the_barrier_is_refused(env):
    """An SDK-supplied plane skips the named-backend gate; install_agent
    asks the sandbox itself before any agent code."""
    with pytest.raises(UnsupportedTaskFeatureError) as caught:
        identity_switch.require_barrier(env)
    assert caught.value.features[0].path == IDENTITY_SWITCH_ISSUE_PATH
    assert "--sandbox docker" in str(caught.value)
    identity_switch.require_barrier(_daytona(compose=True))
    with pytest.raises(UnsupportedTaskFeatureError):
        identity_switch.require_barrier(_daytona(compose=False))


@pytest.mark.asyncio
async def test_install_agent_refuses_before_any_agent_code(monkeypatch, tmp_path):
    """The check sits before the first step that could run agent code."""
    import benchflow.rollout as rollout_mod

    started = []

    async def capture(self):
        started.append("capture")

    async def cwd(env, task):
        return "/app"

    monkeypatch.setattr(rollout_mod, "prepare_capture_runtime", capture)
    monkeypatch.setattr(rollout_mod, "_resolve_agent_cwd", cwd)
    rollout = object.__new__(rollout_mod.Rollout)
    rollout._config = SimpleNamespace()
    rollout._require_rollout_dir = lambda: tmp_path
    rollout._env = SimpleNamespace(agent_container_no_new_privileges=False)
    rollout._task = Task(_task(tmp_path))
    with pytest.raises(UnsupportedTaskFeatureError):
        await rollout.install_agent()
    assert started == []
    assert getattr(rollout._env, "agent_identity_switch", ()) == ()


def test_kept_services_control_on_docker(tmp_path):
    """K2 preservation: a task that keeps services and switches identity
    still passes the gate on the Docker family, and keep_services still
    reads its ports; elsewhere only the identity switch is refused for it
    (the existing kept-service refusals are unchanged)."""
    task_dir = _task(tmp_path, keep=True)
    task = Task(task_dir)
    kept = keep_services.for_task(task)
    assert kept is not None
    assert [s.port for s in kept.services] == [5000]
    for sandbox in ("docker", "remote-docker"):
        assert (
            validate_task_runtime_support(
                task.config, sandbox=sandbox, task_dir=task_dir
            )
            == []
        )
        assert (
            keep_services.provider_refusal(
                task, task_dir, sandbox=sandbox, keep=kept, agent="claude-agent-acp"
            )
            is None
        )
    plain = _task(tmp_path / "plain", switch=False, keep=True)
    assert _switch_issues(task_dir, "modal") and not _switch_issues(plain, "modal")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["modal", "docker", "custom-nnp"])
@pytest.mark.parametrize("started", [False, True])
@pytest.mark.parametrize("capability", [None, False, "yes"])
async def test_sdk_admission_precedes_start_and_preserves_caller_ownership(
    tmp_path, monkeypatch, backend, started, capability
):
    """Guards ff5ee2df6's K1-R15-2 B1: SDK admission must precede start."""
    from unittest.mock import AsyncMock, MagicMock

    import benchflow as bf
    from benchflow.runtime import Agent, Environment, RuntimeConfig

    inner = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    if capability is not None:
        inner.agent_container_no_new_privileges = capability
    task_dir = _task(tmp_path)
    monkeypatch.chdir(tmp_path)
    planes = SimpleNamespace(create_environment=MagicMock(return_value=inner))
    env = Environment.from_task(task_dir, sandbox=backend, planes=planes)
    if started:
        await env.start()
    with pytest.raises(UnsupportedTaskFeatureError, match="survive into grading"):
        await bf.run(
            Agent("oracle", model="none"),
            env,
            RuntimeConfig(jobs_dir=tmp_path / "jobs", usage_tracking="off"),
        )
    assert inner.start.await_count == int(started)
    assert env._started is started
    inner.stop.assert_not_called()
    if started:
        await env.stop()
        inner.stop.assert_awaited_once()


@pytest.mark.parametrize("capability", [None, False, "yes", True])
def test_custom_backend_plan_requires_an_actual_container_barrier(tmp_path, capability):
    """Guards ff5ee2df6's K1-R15-2 B1: custom planes cannot skip admission."""
    from benchflow.task.plan_validation import validate_task_plan

    task_dir = _task(tmp_path)
    sandbox = SimpleNamespace(agent_container_no_new_privileges=capability)

    def check():
        validate_task_plan(
            task_dir,
            sandbox="custom-nnp",
            agent="oracle",
            custom_backend=True,
            sandbox_instance=sandbox,
        )

    if capability is True:
        check()
    else:
        with pytest.raises(UnsupportedTaskFeatureError, match="survive into grading"):
            check()


def test_custom_backend_without_an_instance_refuses_only_identity_switch(tmp_path):
    """Guards ff5ee2df6's K1-R15-2 B1 before a custom factory is invoked."""
    from benchflow.task.plan_validation import validate_task_plan

    with pytest.raises(UnsupportedTaskFeatureError, match="survive into grading"):
        validate_task_plan(
            _task(tmp_path), sandbox="custom-nnp", agent="oracle", custom_backend=True
        )
    validate_task_plan(
        _task(tmp_path / "plain", switch=False),
        sandbox="custom-nnp",
        agent="oracle",
        custom_backend=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("custom_planes", [False, True])
async def test_custom_prebuilt_backend_with_barrier_passes_rollout_admission(
    tmp_path, custom_planes
):
    """Guards K1-R15-2 B1 at ff5ee2df6 without rejecting a capable custom SDK."""
    from unittest.mock import MagicMock

    from benchflow.rollout import Rollout, RolloutConfig

    planes = MagicMock()
    planes.resolve_locked_paths.return_value = []
    planes.resolve_agent_env.return_value = {}
    rollout = Rollout(
        RolloutConfig.from_legacy(
            task_path=_task(tmp_path),
            agent="oracle",
            environment="custom-nnp",
            jobs_dir=tmp_path / "jobs",
            planes=planes if custom_planes else None,
        )
    )
    inner = SimpleNamespace(agent_container_no_new_privileges=True)
    rollout.use_prebuilt_env(inner)
    await rollout.setup()
    assert rollout.env is inner
    planes.create_environment.assert_not_called()


@pytest.mark.asyncio
async def test_sdk_named_backend_refusal_also_precedes_start(tmp_path, monkeypatch):
    """Guards ff5ee2df6's K1-R15-2 B1 even when a plane reports NNP for Modal."""
    from unittest.mock import AsyncMock, MagicMock

    import benchflow as bf
    from benchflow.runtime import Agent, Environment, RuntimeConfig

    inner = SimpleNamespace(
        agent_container_no_new_privileges=True, start=AsyncMock(), stop=AsyncMock()
    )
    task_dir = _task(tmp_path)
    monkeypatch.chdir(tmp_path)
    env = Environment.from_task(
        task_dir,
        sandbox="modal",
        planes=SimpleNamespace(create_environment=MagicMock(return_value=inner)),
    )
    with pytest.raises(UnsupportedTaskFeatureError, match="the modal sandbox"):
        await bf.run(
            Agent("oracle", model="none"),
            env,
            RuntimeConfig(jobs_dir=tmp_path / "jobs", usage_tracking="off"),
        )
    inner.start.assert_not_called()
    inner.stop.assert_not_called()
    assert not env._started


@pytest.mark.asyncio
async def test_sdk_custom_backend_with_barrier_is_admitted(tmp_path, monkeypatch):
    """Guards K1-R15-2 B1 at ff5ee2df6: a capable custom SDK still starts once."""
    from unittest.mock import AsyncMock, MagicMock

    import benchflow as bf
    from benchflow.models import RolloutResult
    from benchflow.rollout import Rollout
    from benchflow.runtime import Agent, Environment, RuntimeConfig

    inner = SimpleNamespace(
        agent_container_no_new_privileges=True, start=AsyncMock(), stop=AsyncMock()
    )
    task_dir = _task(tmp_path)
    monkeypatch.chdir(tmp_path)
    env = Environment.from_task(
        task_dir,
        sandbox="custom-nnp",
        planes=SimpleNamespace(create_environment=MagicMock(return_value=inner)),
    )

    async def admitted_run(self):
        # Exercise the real rollout plan as well as Runtime's admission.
        await self.setup()
        return RolloutResult(task_name="task")

    monkeypatch.setattr(Rollout, "run", admitted_run)
    try:
        result = await bf.run(
            Agent("oracle", model="none"),
            env,
            RuntimeConfig(jobs_dir=tmp_path / "jobs", usage_tracking="off"),
        )
        assert result.error is None
        inner.start.assert_awaited_once()
        assert env._started
        inner.stop.assert_not_called()
    finally:
        await env.stop()
