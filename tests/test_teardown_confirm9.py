"""Guards U10-CONFIRM-9 at 9e8fd20e; image mutations use dummy markers."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.remote_docker import (
    RemoteDockerSandbox,
    resolve_remote_docker_host,
)
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths
from tests.test_teardown_confirmation import InventoryDaemon


def sandbox_with_output(tmp_path, monkeypatch, remote, outputs):
    """Exercise the real runner and OS pipes, substituting only the CLI child."""
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM scratch\n")
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    cls = RemoteDockerSandbox if remote else DockerSandbox
    kwargs = (
        {
            "docker_host": resolve_remote_docker_host(
                {"DOCKER_HOST": "ssh://builder.example.test"}
            )
        }
        if remote
        else {}
    )
    sandbox = cls(
        environment_dir=environment,
        environment_name="dummy",
        session_id="dummy__trial",
        rollout_paths=paths,
        task_env_config=SandboxConfig(),
        **kwargs,
    )
    daemon = InventoryDaemon(sandbox.compose_project_name)
    sandbox._docker_cli = daemon.cli
    real_spawn = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        command = tuple(args[args.index("config") :])
        stdout, stderr, code = outputs[command]
        return await real_spawn(
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.stderr.write(sys.argv[2]); sys.stderr.flush(); "
            "sys.stdout.write(sys.argv[1]); sys.exit(int(sys.argv[3]))",
            stdout,
            stderr,
            str(code),
            **kwargs,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    return sandbox, daemon


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("separator", ["-", "_"])
@pytest.mark.parametrize("warn_on", ["json", "images", "both"])
@pytest.mark.parametrize(
    "warning",
    [
        'level=warning msg="the attribute `version` is obsolete"\n',
        'WARN: the "DUMMY" variable is not set.\nWARN: using a blank string.\n',
        "dummy-api\n",
    ],
)
@pytest.mark.parametrize(
    "retained", [None, "operator/dummy:keep", "benchflow-recovery-lease:dummy"]
)
async def test_config_diagnostics_do_not_change_image_ownership(
    tmp_path, monkeypatch, remote, separator, warn_on, warning, retained
):
    """Guards U10-CONFIRM-9 at 9e8fd20e through both config queries/backends."""
    outputs = {}
    sandbox, daemon = sandbox_with_output(tmp_path, monkeypatch, remote, outputs)
    shared, private = (
        f"{sandbox.compose_project_name}{separator}{name}:latest"
        for name in ("api", "worker")
    )
    config = {
        "services": {
            "api": {"build": "./api"},
            "worker": {"build": "./worker"},
            "reader": {"image": shared, "pull_policy": "never"},
        }
    }
    outputs[("config", "--format", "json")] = (
        json.dumps(config),
        warning if warn_on in ("json", "both") else "",
        0,
    )
    outputs[("config", "--images")] = (
        f"{shared}\n{private}\n{shared}\n",
        warning if warn_on in ("images", "both") else "",
        0,
    )
    daemon.built("sha256:shared", shared, "api")
    daemon.built("sha256:private", private, "worker")
    if retained:
        daemon.images["sha256:private"]["tags"].append(retained)

    await sandbox._record_built_image_tags()
    await sandbox._remove_built_images()

    assert sandbox._private_built_image_tags == {private}
    assert private not in daemon.tags()
    assert daemon._find(shared) == "sha256:shared"
    assert ("sha256:private" in daemon.images) == bool(retained)
    assert not retained or retained in daemon.tags()
    assert daemon.parent_present
    assert all(
        "--no-prune" in call and "--force" not in call
        for call in daemon.cli_calls
        if call[:2] == ["image", "rm"]
    )


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize(
    "query", [("config", "--format", "json"), ("config", "--images")]
)
@pytest.mark.parametrize("check", [False, True])
async def test_config_failure_preserves_diagnostics(
    tmp_path, monkeypatch, remote, query, check
):
    """Guards U10-CONFIRM-9 at 9e8fd20e: separating streams retains errors."""
    sandbox, _ = sandbox_with_output(
        tmp_path, monkeypatch, remote, {query: ("partial output", "dummy error", 1)}
    )
    if check:
        with pytest.raises(RuntimeError, match="dummy error"):
            await sandbox._run_docker_compose_command(list(query))
    else:
        result = await sandbox._run_docker_compose_command(list(query), check=False)
        assert result.return_code == 1
        assert result.stdout == "partial output"
        assert result.stderr == "dummy error"


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("broken", ["json", "images", "json-exit", "images-exit"])
async def test_invalid_config_still_retains_images(
    tmp_path, monkeypatch, remote, broken
):
    """Guards U10-CONFIRM-9 at 9e8fd20e: only stdout can establish ownership."""
    outputs = {}
    sandbox, daemon = sandbox_with_output(tmp_path, monkeypatch, remote, outputs)
    private = f"{sandbox.compose_project_name}-worker:latest"
    config = json.dumps({"services": {"worker": {"build": "."}}})
    outputs[("config", "--format", "json")] = (
        "bad JSON" if broken == "json" else config,
        config,
        int(broken == "json-exit"),
    )
    outputs[("config", "--images")] = (
        "bad image line" if broken == "images" else private,
        private,
        int(broken == "images-exit"),
    )
    daemon.built("sha256:private", private, "worker")
    await sandbox._record_built_image_tags()
    await sandbox._remove_built_images()
    assert daemon._find(private) == "sha256:private"
    assert not sandbox._private_built_image_tags


@pytest.mark.parametrize("remote", [False, True])
async def test_service_query_ignores_even_name_shaped_diagnostics(
    tmp_path, monkeypatch, remote
):
    """Guards U10-CONFIRM-9 at 9e8fd20e for the sibling config consumer."""
    sandbox, _ = sandbox_with_output(
        tmp_path,
        monkeypatch,
        remote,
        {("config", "--services"): ("main\nworker\n", "dummy-service\n", 0)},
    )
    assert await sandbox.services() == ["main", "worker"]
