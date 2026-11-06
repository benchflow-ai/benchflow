"""Guards the DT2 merge of 6b926c14 with integ 7b4907c8."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._exec_output import diagnostic_result
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from benchflow.task.config import SandboxConfig
from benchflow.task.plan_validation import agent_egress_policy


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", [DockerSandbox, RemoteDockerSandbox])
@pytest.mark.parametrize("code", [0, 7])
@pytest.mark.parametrize(
    "operation", ["script", "exec", "direct", "config", "cp", "up"]
)
async def test_compose_script_records_and_ordinary_diagnostics(
    tmp_path, monkeypatch, backend, code, operation
):
    """Guards DT2-B12-1: retain DT2 records and integ's ordinary merged I/O."""
    (tmp_path / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    sandbox = object.__new__(backend)
    DockerSandbox.__init__(
        sandbox,
        environment_dir=tmp_path,
        environment_name="dt2-merge",
        session_id="dt2-merge",
        rollout_paths=None,
        task_env_config=SandboxConfig(),
    )
    # The remote host's connection surface as integ's RD defines it.
    sandbox.docker_host = SimpleNamespace(
        user="dt2-user",
        display="dt2-host",
        redact=lambda text, **kwargs: text.replace("dt2-user", "[redacted]"),
        compose_args=lambda environ=None: [],
        compose_env=lambda env: env,
        compose_transport_env=lambda: {},
    )
    monkeypatch.setattr(sandbox, "_docker_compose_env", lambda: {})
    sandbox.script_container_config = AsyncMock(return_value={"Env": []})
    stdout = b"1:7"
    diagnostic = b"warning for dt2-user\n"

    async def launch(*argv, **kwargs):
        process = AsyncMock(returncode=code)
        streams = (
            (stdout, diagnostic)
            if kwargs["stderr"] == asyncio.subprocess.PIPE
            else (diagnostic + stdout, None)
        )
        process.communicate.return_value = streams
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    if operation == "script":
        result = await sandbox.script_runner_exec("printf 1:7", timeout_sec=10)
    elif operation == "direct":
        result = await sandbox._run_docker_container_command(
            ["exec", "dt2-container", "true"], check=False
        )
    else:
        args = {
            "exec": ["exec", "-T", "main", "true"],
            "config": ["config", "--services"],
            "cp": ["cp", "source", "main:/tmp/dt2-file"],
            "up": ["up", "--detach"],
        }[operation]
        result = await sandbox._run_docker_compose_command(args, check=False)
    warning = (
        "warning for "
        + ("[redacted]" if backend is RemoteDockerSandbox else "dt2-user")
        + "\n"
    )
    assert result.return_code == code
    if operation == "script":
        # A supervision record is read raw (FF-sw-docker-C13); only its
        # display copy carries the remote host's redaction.
        assert result.stdout == "1:7"
        assert result.stderr == diagnostic.decode()
        assert diagnostic_result(result).stderr.render() == warning
    elif operation == "config":
        assert result.stdout == "1:7"
        assert result.stderr == warning
    else:
        assert result.stdout == warning + "1:7"
        assert not result.stderr


@pytest.mark.parametrize("mode", ["denylist", "allowlist"])
@pytest.mark.parametrize("agent", ["script:attempt.sh", "oracle", "nop", "codex"])
def test_shared_plan_policy_keeps_operator_script_egress(mode, agent):
    """Guards DT2-B12-2: D3's shared gate must retain DT2's script policy."""
    task = SimpleNamespace(
        config=SimpleNamespace(
            sandbox=SandboxConfig(
                network_mode=mode,
                blocked_hosts=["blocked.bench.invalid"] if mode == "denylist" else [],
                allowed_hosts=["allowed.bench.invalid"] if mode == "allowlist" else [],
            )
        )
    )
    policy = agent_egress_policy(task, agent=agent, sandbox_user="agent")
    if agent in ("oracle", "nop"):
        assert policy is None
    else:
        assert policy is not None and policy.mode == mode
        with pytest.raises(ValueError, match="requires a sandbox_user"):
            agent_egress_policy(task, agent=agent, sandbox_user=None)


def test_shared_script_policy_refuses_native_model_only(monkeypatch):
    """Guards DT2-B12-2: the merge retains DT2's native-model refusal."""
    from benchflow.sandbox import egress_denylist

    policy = egress_denylist.EgressDenylist((), (), native_claude_model_only=True)
    monkeypatch.setattr(egress_denylist, "egress_denylist_for", lambda _: policy)
    task = SimpleNamespace(config=SimpleNamespace(sandbox=SandboxConfig()))
    with pytest.raises(ValueError, match="script: cannot use a native model-only"):
        agent_egress_policy(task, agent="script:attempt.sh", sandbox_user="agent")
