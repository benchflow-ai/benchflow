"""Guards FF-sw-docker-C12, introduced by quoted-login redaction in fb4da505."""

import asyncio
import json
import os
import sys

import pytest

from benchflow.sandbox.remote_docker import RemoteDockerHost
from benchflow.task.config import SandboxConfig
from tests.test_docker_build_identity import sandbox


def transport(tmp_path, backend, login):
    env = sandbox(
        tmp_path, backend, task_env_config=SandboxConfig(docker_image="trusted:v1")
    )
    if backend == "remote-docker":
        env.docker_host = RemoteDockerHost(f"ssh://{login}@builder.example.test")
    # Only the transport is replaced: no SSH connection or Docker daemon is
    # needed to exercise both production runners and the real JSON parser.
    env._docker_compose_global_args = lambda: []
    env._docker_compose_env = lambda: dict(os.environ)
    env._compose_transport_env = lambda: {}
    env._docker_client_env = lambda: dict(os.environ)
    return env


def emit_process(monkeypatch, output):
    create = asyncio.create_subprocess_exec

    async def emit(*args, **kwargs):
        stdout, stderr, code = output(args)
        return await create(
            sys.executable,
            "-c",
            "import sys; sys.stdout.write(sys.argv[1]); "
            "sys.stderr.write(sys.argv[2]); sys.exit(int(sys.argv[3]))",
            stdout,
            stderr,
            str(code),
            **kwargs,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", emit)


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize(
    "service", ["worker", "worker.dot", "worker-hyphen", "worker_under"]
)
@pytest.mark.parametrize("separator", ["-", "_"])
@pytest.mark.parametrize("prebuilt", [True, False])
async def test_implicit_plan_survives_login_collision(
    tmp_path, monkeypatch, backend, service, separator, prebuilt
):
    """C12 at 15461d52: a login matching an implicit service cannot change its plan."""
    env = transport(tmp_path, backend, service)
    env._use_prebuilt = prebuilt
    config = {"services": {"main": {"image": "trusted:v1"}, service: {"build": "."}}}
    alias = f"trial{separator}{service}"
    bake = {
        "target": {
            "unrelated-key": {
                "labels": {"com.docker.compose.service": service},
                "tags": [alias],
            }
        }
    }
    calls = []

    def output(args):
        calls.append(args)
        payload = bake if "--print" in args else config
        return json.dumps(payload), f"warning from '{service}'\n", 0

    emit_process(monkeypatch, output)
    try:
        await env._prepare_compose_builds()
        assert env._build_compose_path is not None
        document = env._build_compose_path.read_text()
        assert json.dumps(service) in document
        assert "***" not in document
        assert len(env._private_built_image_tags) == 1
        assert any("--print" in call for call in calls)
    finally:
        if env._build_compose_path is not None:
            env._build_compose_path.unlink(missing_ok=True)


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("return_code", [0, 1, 125])
@pytest.mark.parametrize(
    "runner, command",
    [
        ("compose", ["config", "--format", "json"]),
        ("compose", ["config", "--images"]),
        ("compose", ["config", "--services"]),
        ("compose", ["build", "--print", "--", "worker"]),
        ("compose", ["build", "--no-cache", "--print", "worker"]),
        ("compose", ["ps", "-q", "main"]),
        ("compose", ["images", "--format", "json"]),
        ("compose", ["exec", "-T", "main", "sh", "-c", "probe"]),
        ("compose", ["new-data-command"]),
        ("container", ["exec", "container-id", "sh", "-c", "probe"]),
        ("cli", ["inspect", "container-id"]),
        ("cli", ["image", "ls", "-q"]),
        ("cli", ["image", "inspect", "image-id"]),
        ("cli", ["volume", "ls", "-q"]),
        ("cli", ["volume", "inspect", "worker"]),
        ("cli", ["network", "ls", "-q"]),
        ("cli", ["network", "inspect", "network-id"]),
        ("cli", ["version", "--format", "{{.Server.Version}}"]),
        ("cli", ["ps", "-aq"]),
        ("cli", ["new-data-command"]),
    ],
)
async def test_raw_results_are_independent_of_command_spelling(
    tmp_path, monkeypatch, backend, return_code, runner, command
):
    """C12: the consumer selects raw data, including unknown commands and failures."""
    env = transport(tmp_path, backend, "worker")
    payload = '{"worker": "worker", "endpoint": "ssh://worker@builder.example.test"}\n'
    warning = "warning from 'worker' at builder.example.test\n"
    emit_process(monkeypatch, lambda args: (payload, warning, return_code))
    method = {
        "compose": env._run_docker_compose_command,
        "container": env._run_docker_container_command,
        "cli": env._docker_cli,
    }[runner]
    result = await method(command, check=False, raw_output=True)
    assert result.stdout == payload
    assert result.stderr == warning
    assert result.return_code == return_code


@pytest.mark.parametrize("runner", ["compose", "container", "cli"])
@pytest.mark.parametrize("check", [True, False])
@pytest.mark.parametrize("raw_output", [True, False])
async def test_diagnostics_still_redact_at_display_boundary(
    tmp_path, monkeypatch, runner, check, raw_output
):
    """C12: raw parsing cannot disable checked errors or diagnostic result redaction."""
    env = transport(tmp_path, "remote-docker", "worker")
    message = "failed for 'worker' at ssh://worker@builder.example.test"
    emit_process(monkeypatch, lambda args: (message, message, 1))
    method = {
        "compose": env._run_docker_compose_command,
        "container": env._run_docker_container_command,
        "cli": env._docker_cli,
    }[runner]
    if check:
        with pytest.raises(RuntimeError) as caught:
            await method(["build"], raw_output=raw_output)
        assert "worker" not in str(caught.value)
        assert "builder.example.test" not in str(caught.value)
    else:
        result = await method(["build"], check=False, raw_output=raw_output)
        text = result.stdout + (result.stderr or "")
        if raw_output:
            assert message in text
            text = env._diagnostic_text(text)
        assert "worker" not in text
        assert "builder.example.test" not in text


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize(
    "service, container", [("main", None), ("main", "a" * 64), ("worker", "a" * 64)]
)
async def test_exec_raw_reaches_compose_and_direct_container(
    tmp_path, monkeypatch, backend, service, container
):
    """C12: identity/probe parsers use the same raw route on every exec dispatch."""
    env = transport(tmp_path, backend, "worker")
    env._main_container = container
    text = '"worker"\nssh://worker@builder.example.test\n'
    emit_process(monkeypatch, lambda args: (text, text, 0))
    result = await env.exec("probe", service=service, raw_output=True)
    assert result.stdout == text
    assert result.stderr == text


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize(
    "consumer",
    [
        "services",
        "inspect",
        "main-id",
        "learn-id",
        "leftovers",
        "foreign-volumes",
        "credentials",
        "subnets",
    ],
)
async def test_lifecycle_parsers_keep_colliding_data(
    tmp_path, monkeypatch, backend, consumer
):
    """C12: lifecycle consumers select the raw path before interpreting any name."""
    from benchflow.sandbox.docker import _DockerCredentialOps

    env = transport(tmp_path, backend, "worker")
    if backend == "docker":
        env._project_name = env.compose_project_name
    if consumer == "services":
        payload = "worker\nmain\n"
    elif consumer == "inspect":
        payload = json.dumps([{"Name": "worker", "Config": {"User": "worker"}}])
    elif consumer == "foreign-volumes":
        payload = json.dumps(
            [
                {
                    "Name": "worker",
                    "Labels": {"com.docker.compose.project": env.compose_project_name},
                }
            ]
        )
    elif consumer in {"main-id", "learn-id"}:
        payload = "a" * 64 + "\n"
    elif consumer == "credentials":
        payload = "home:1000:1000:/tmp/worker\nfile:1000:1000:/tmp/worker/'worker'\n"
    else:
        payload = '"worker"\n'
    emit_process(monkeypatch, lambda args: (payload, "warning for 'worker'", 0))
    if consumer == "services":
        assert await env.services() == ["worker", "main"]
    elif consumer == "inspect":
        assert (await env._inspect_container("id"))["Config"]["User"] == "worker"
    elif consumer == "main-id":
        assert await env._main_container_id() == "a" * 64
    elif consumer == "learn-id":
        await env._learn_main_container()
        assert env._main_container == "a" * 64
    elif consumer == "leftovers":
        # Remote cleanup owns this consumer; run it over either transport.
        from benchflow.sandbox.remote_docker import RemoteDockerSandbox

        result = await RemoteDockerSandbox._project_leftovers(env)
        assert all(names == ['"worker"'] for names in result.values())
    elif consumer == "foreign-volumes":
        from benchflow.sandbox.remote_docker import RemoteDockerSandbox

        assert await RemoteDockerSandbox._foreign_volumes(env, ["worker"]) == []
    elif consumer == "credentials":
        result = await _DockerCredentialOps(env, "id").run("dummy discovery")
        assert result.stdout == payload
        if backend == "remote-docker":
            assert "worker" not in result.stderr
    else:
        result = await env._docker_cli_unchecked(["network", "inspect", "id"])
        assert result.stdout == payload
        if backend == "remote-docker":
            assert "worker" not in result.stderr
