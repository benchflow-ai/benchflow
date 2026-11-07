"""Guards C9, introduced by a1f73cfc's implicit Compose build-plan read."""

import asyncio
import json
import sys

import pytest

from benchflow.sandbox._compose_runtime import implicit_build_aliases
from tests.test_docker_build_identity import sandbox


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize(
    "command",
    [
        ["config", "--format", "json"],
        ["config", "--images"],
        ["build", "--print", "--", "worker"],
        ["build", "--no-cache", "--print", "worker"],
        ["ps", "-q", "main"],
        ["images", "--format", "json"],
    ],
)
@pytest.mark.parametrize(
    "warning", ["version is obsolete\n", "unrelated warning\n", "{}\n[]\n"]
)
async def test_compose_control_stdout_excludes_all_diagnostics(
    tmp_path, monkeypatch, backend, command, warning
):
    env = sandbox(tmp_path, backend)
    create = asyncio.create_subprocess_exec
    payload = '{"target":{"worker":{"labels":{"com.docker.compose.service":"worker"},"tags":["trial-worker"]}}}'

    async def emit(*args, **kwargs):
        return await create(
            sys.executable,
            "-c",
            "import sys; sys.stderr.write(sys.argv[2]); sys.stdout.write(sys.argv[1])",
            payload,
            warning,
            **kwargs,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", emit)
    result = await env._run_docker_compose_command(command)
    assert result.stdout == payload
    assert result.stderr == warning
    aliases = await implicit_build_aliases(
        {"services": {"worker": {"build": "."}}},
        prebuilt=True,
        compose=env._run_docker_compose_command,
    )
    assert aliases == {"worker": ["trial-worker"]}
    assert json.loads(result.stdout)["target"]


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
async def test_build_failure_keeps_both_streams_in_error(
    tmp_path, monkeypatch, backend
):
    env = sandbox(tmp_path, backend)
    create = asyncio.create_subprocess_exec

    async def emit(*args, **kwargs):
        return await create(
            sys.executable,
            "-c",
            "import sys; print('plan output'); sys.stderr.write('build refused'); sys.exit(1)",
            **kwargs,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", emit)
    with pytest.raises(RuntimeError, match=r"(?s)plan output.*build refused"):
        await env._run_docker_compose_command(["build", "--print"])
