"""Guards S1-CONF-9/10 at 694245fd: replacement startup and receipt encoding."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._base import SYSTEM_SHELL_UNSET_VARS
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.protocol import ExecResult, SandboxImage
from benchflow.task.config import SandboxConfig


@pytest.mark.parametrize("health", [None, {"Test": ["NONE"]}])
async def test_released_main_must_be_running_without_a_healthcheck(health):
    """Guards acaaefb9: a completed bootstrap cannot hide an exited main process."""
    from benchflow.sandbox import _inert_bootstrap as bootstrap

    env = SimpleNamespace(
        _inspect_container=AsyncMock(
            return_value={"State": {"Running": False, "ExitCode": 1}}
        )
    )
    with pytest.raises(RuntimeError, match="exited"):
        await bootstrap.wait_healthy(env, "a" * 64, {"Healthcheck": health})


@pytest.mark.parametrize("exit_code", [0, 1])
async def test_completed_init_services_preserve_their_exit_status(exit_code):
    """Guards acaaefb9: an explicit successful-init dependency can finish."""
    from benchflow.sandbox import _inert_bootstrap as bootstrap

    env = SimpleNamespace(
        _inspect_container=AsyncMock(
            return_value={"State": {"Running": False, "ExitCode": exit_code}}
        )
    )
    if exit_code:
        with pytest.raises(RuntimeError, match="exited"):
            await bootstrap.wait_healthy(env, "a" * 64, {}, allow_completed=True)
    else:
        await bootstrap.wait_healthy(env, "a" * 64, {}, allow_completed=True)


def sandbox(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM trusted")
    return DockerSandbox(tmp_path, "inert", "inert", None, SandboxConfig())


@pytest.mark.parametrize("loader", SYSTEM_SHELL_UNSET_VARS)
@pytest.mark.parametrize("backend", ["docker", "remote"])
async def test_restore_first_process_is_absolute_inert_and_clears_startup(
    tmp_path, loader, backend
):
    """CONF-9: no inherited entrypoint, healthcheck or loader can precede stat."""
    env = sandbox(tmp_path)
    if backend == "remote":
        from benchflow.sandbox.remote_docker import (
            RemoteDockerHost,
            RemoteDockerSandbox,
        )

        env = RemoteDockerSandbox(
            tmp_path,
            "inert",
            "inert",
            None,
            SandboxConfig(),
            docker_host=RemoteDockerHost(url="tcp://worker:2376"),
        )
    env._main_container_id = AsyncMock(return_value="a" * 64)
    env._inspect_container = AsyncMock(
        return_value={
            "HostConfig": {"NetworkMode": "none"},
            "NetworkSettings": {"Networks": {"none": {}}},
            "Mounts": [],
            "Config": {
                "Env": ["PATH=/workspace/.venv/bin:/usr/bin:/bin"],
                "Entrypoint": ["/workspace/start"],
                "Healthcheck": {"Test": ["CMD", "health"]},
            },
        }
    )
    env._snapshot_image_bind_baseline = AsyncMock(return_value={"main": {}})
    env._verify_replacement_bind_targets = AsyncMock()
    env._docker_cli = AsyncMock(
        return_value=ExecResult(return_code=0, stdout="b" * 64, stderr="")
    )
    await env.restore(SandboxImage(provider="docker", ref="snapshot"))
    run = next(
        c.args[0] for c in env._docker_cli.call_args_list if c.args[0][0] == "run"
    )
    assert "--entrypoint" in run, "inherited entrypoint ran before the bind probe"
    assert run[run.index("--entrypoint") + 1] == "/bin/sh"
    assert "--no-healthcheck" in run
    assert loader in run[: run.index("snapshot")]
    assert "sleep" not in run
    assert "/bin/sleep" in run


@pytest.mark.parametrize("kind", ["restart", "recovery", "first"])
async def test_start_prepares_bootstrap_before_up_and_releases_after_proof(
    tmp_path, monkeypatch, kind
):
    """CONF-9: every Docker/remote inherited start follows the same barrier."""
    from benchflow.sandbox._docker_sweep import release_project

    env = sandbox(tmp_path)
    events = []
    if kind == "restart":
        env._compose_up_attempted = True
        env._bind_baseline = {"main": {}}
    if kind == "recovery":
        env._recovery_baseline = SimpleNamespace(
            effective_allow_internet=True, image_id="trusted"
        )
        monkeypatch.setattr("benchflow.sandbox.docker.validate_baseline", AsyncMock())
        env._inspect_container = AsyncMock(return_value={"Image": "trusted"})
        env._main_container_id = AsyncMock(return_value="a" * 64)
    monkeypatch.setattr(env, "freeze_launch", lambda *args: None)
    for name in (
        "_run_pre_compose_hook",
        "_run_docker_compose_build",
        "_start_from_layers",
        "_choose_subnets",
        "_name_trial_volumes",
        "_note_task_image_pull",
        "_learn_main_container",
        "_prepare_log_dirs_after_up",
    ):
        monkeypatch.setattr(env, name, AsyncMock())
    env._absent_task_image = AsyncMock(return_value=None)

    async def compose(args, **kwargs):
        if args[0] == "up":
            events.append("up")
        return ExecResult(return_code=0, stdout="", stderr="")

    env._run_docker_compose_command = compose

    async def prepared():
        events.append("inert")

    async def up():
        events.append("up")

    async def pin():
        events.append("proof")

    async def released():
        events.append("release")

    monkeypatch.setattr(env, "_prepare_inert_bootstrap", prepared, raising=False)
    monkeypatch.setattr(env, "_release_inert_bootstrap", released, raising=False)
    env._run_docker_compose_up = up
    env._pin_or_verify_bind_targets = pin
    try:
        await env.start(False)
    finally:
        release_project(env.compose_project_name)
    assert events == ["inert", "up", "proof", "release"]


async def test_failed_proof_never_releases_replacement(tmp_path):
    """CONF-9: failed identity proof leaves no task process running."""
    env = sandbox(tmp_path)
    env._main_container_id = AsyncMock(return_value="a" * 64)
    env._inspect_container = AsyncMock(
        return_value={
            "HostConfig": {"NetworkMode": "none"},
            "NetworkSettings": {"Networks": {"none": {}}},
            "Mounts": [],
        }
    )
    env._snapshot_image_bind_baseline = AsyncMock(return_value={"main": {}})
    env._verify_replacement_bind_targets = AsyncMock(
        side_effect=RuntimeError("identity changed")
    )
    env._docker_cli = AsyncMock(
        return_value=ExecResult(return_code=0, stdout="b" * 64, stderr="")
    )
    with pytest.raises(RuntimeError, match="identity changed"):
        await env.restore(SandboxImage(provider="docker", ref="snapshot"))
    assert not any(c.args[0][0] == "kill" for c in env._docker_cli.call_args_list)


@pytest.mark.parametrize("source", ["image", "checkpoint"])
async def test_effective_command_is_quarantined_without_interpreting_text(
    tmp_path, source
):
    """CONF-9: Docker's stopped Config owns command/env, including checkpoint reuse."""
    import base64

    from benchflow.sandbox import _inert_bootstrap as bootstrap

    config = {
        "Entrypoint": ["/workspace/entry"],
        "Cmd": ["$X", "quote'", "line\narg"],
        "Env": [
            "PATH=/workspace/bin:/bin",
            "BASH_ENV=/workspace/rc",
            "PYTHONPATH=/workspace/site",
        ],
        "Healthcheck": {"Test": ["CMD", "/workspace/health"]},
        "Labels": {"com.docker.compose.service": "main"},
    }
    original = dict(config)
    if source == "checkpoint":
        config["Labels"][bootstrap.LABEL] = base64.b64encode(
            json.dumps({k: v for k, v in original.items() if k != "Labels"}).encode()
        ).decode()
        config["Entrypoint"] = bootstrap.entrypoint(original)
        config["Cmd"] = []
    env = SimpleNamespace(
        _inspect_container=AsyncMock(
            return_value={
                "State": {"Running": False},
                "Config": config,
                "Mounts": [
                    {"Type": "bind", "Source": "/tmp/task-data", "Destination": "/data"}
                ],
            }
        )
    )

    def compose(args, **kwargs):
        return ExecResult(
            return_code=0,
            stderr="",
            stdout=json.dumps({"services": {"main": {}}})
            if args[0] == "config"
            else "a" * 64,
        )

    result = await bootstrap.prepare(env, AsyncMock(side_effect=compose))
    main = result["services"]["main"]
    assert main["entrypoint"][:3] == ["/bin/sh", "-c", bootstrap.BOOTSTRAP]
    # The env rows in their canonical order, not Config.Env's.
    assert main["entrypoint"][4:] == [
        *bootstrap.canonical_env(original["Env"]),
        *original["Entrypoint"],
        *original["Cmd"],
    ]
    assert main["command"] == [] and main["healthcheck"] == {"disable": True}
    assert all(main["environment"][key] is None for key in SYSTEM_SHELL_UNSET_VARS)
    assert bootstrap.restored_env(env) == {
        "BASH_ENV": "/workspace/rc",
        "PYTHONPATH": "/workspace/site",
    }


@pytest.mark.parametrize("backend", ["docker", "remote", "dind"])
async def test_bootstrap_overlay_cannot_interpolate_dollars(
    tmp_path, monkeypatch, backend
):
    """CONF-9: every backend writes the same non-interpreting private overlay."""
    from benchflow.sandbox import _inert_bootstrap
    from benchflow.sandbox.daytona_dind import _DaytonaDinD
    from benchflow.sandbox.remote_docker import RemoteDockerHost, RemoteDockerSandbox

    (tmp_path / "Dockerfile").write_text("FROM trusted")
    document = {
        "services": {"main": {"entrypoint": ["/bin/sh", "-c", "a$X $@ $$ ${X}"]}}
    }
    monkeypatch.setattr(_inert_bootstrap, "prepare", AsyncMock(return_value=document))
    if backend == "dind":
        env = _DaytonaDinD.__new__(_DaytonaDinD)
        uploaded = []

        async def upload(local, remote):
            uploaded.append(local.read_text())

        env._env = SimpleNamespace(_sdk_upload_file=upload)
        await env._prepare_inert_bootstrap()
        text = uploaded[0]
    else:
        env = (
            sandbox(tmp_path)
            if backend == "docker"
            else RemoteDockerSandbox(
                tmp_path,
                "inert",
                "inert",
                None,
                SandboxConfig(),
                docker_host=RemoteDockerHost(url="ssh://worker"),
            )
        )
        await env._prepare_inert_bootstrap()
        text = env._bootstrap_override.read_text()
        env._bootstrap_folder.cleanup()
    assert text == json.dumps(document).replace("$", "$$")


async def test_healthcheck_only_runs_after_all_services_have_release_handlers():
    """CONF-9: no original healthcheck or command can pass an incomplete barrier."""
    from benchflow.sandbox import _inert_bootstrap as bootstrap

    events = []
    env = SimpleNamespace(
        _bootstrap_configs={
            "main": {"Healthcheck": {"Test": ["CMD", "/workspace/health"]}}
        },
        _bootstrap_project={"services": {"main": {}}},
        _bootstrap_services={"main"},
        _inspect_container=AsyncMock(
            return_value={"Config": {"Labels": {"com.docker.compose.service": "main"}}}
        ),
    )

    def cli(args, **kwargs):
        events.append(args)
        return ExecResult(return_code=0, stdout="", stderr="")

    env._docker_cli = AsyncMock(side_effect=cli)
    await bootstrap.release(
        env,
        AsyncMock(return_value=ExecResult(return_code=0, stdout="a" * 64, stderr="")),
    )
    assert "SigCgt" in events[0][-1]
    assert events[1] == ["kill", "--signal", "USR1", "a" * 64]
    assert events[2][-1].endswith("/usr/bin/env -- /workspace/health")


@pytest.mark.parametrize("key", SYSTEM_SHELL_UNSET_VARS)
def test_bootstrap_removes_startup_controls_instead_of_setting_empty(key):
    """Guards CONF-9 at 694245fd, including presence-sensitive loader controls."""
    from benchflow.sandbox import _inert_bootstrap as bootstrap

    assert bootstrap.EMPTY_ENV[key] is None
    assert bootstrap.client_env({key: "workspace-value", "HOME": "/task"}) == {
        "HOME": "/task"
    }
    argv = bootstrap.run_args()
    assert ["--env", key] == argv[argv.index(key) - 1 : argv.index(key) + 1]


async def test_all_independent_services_start_before_peer_healthchecks():
    """Guards CONF-9 at 694245fd: health cannot wait for an unreleased peer."""
    from benchflow.sandbox import _inert_bootstrap as bootstrap

    released = set()
    containers = {"a" * 64: "main", "b" * 64: "peer"}
    env = SimpleNamespace(
        _bootstrap_configs={
            name: {"Env": [], "Healthcheck": {"Test": ["CMD", "/check"]}}
            for name in containers.values()
        },
        _bootstrap_project={"services": {name: {} for name in containers.values()}},
        _bootstrap_services=set(containers.values()),
    )
    env._inspect_container = AsyncMock(
        side_effect=lambda cid: {
            "Config": {"Labels": {"com.docker.compose.service": containers[cid]}}
        }
    )

    def cli(args, **kwargs):
        if args[0] == "kill":
            released.add(args[-1])
        if args[-1] == "/check":
            assert released == set(containers)
        return ExecResult(return_code=0, stdout="", stderr="")

    env._docker_cli = AsyncMock(side_effect=cli)
    await bootstrap.release(
        env,
        AsyncMock(
            return_value=ExecResult(
                return_code=0, stdout="\n".join(containers), stderr=""
            )
        ),
    )


@pytest.mark.parametrize(
    "fault",
    [
        "prototype_running",
        "duplicate_service",
        "missing_main",
        "post_start",
        "pre_stop",
    ],
)
async def test_unknown_startup_barriers_refuse_before_any_release(fault):
    """Guards CONF-9 at 694245fd: unsupported lifecycle shapes cannot run early."""
    from benchflow.sandbox import _inert_bootstrap as bootstrap

    spec = (
        {fault: [{"command": "/task/hook"}]}
        if fault in {"post_start", "pre_stop"}
        else {}
    )
    project = {"services": {"main": spec, "peer": {}}}
    config = {
        "Cmd": ["/bin/sleep", "infinity"],
        "Labels": {
            "com.docker.compose.service": "peer" if fault == "missing_main" else "main"
        },
    }
    env = SimpleNamespace(
        _inspect_container=AsyncMock(
            return_value={
                "State": {"Running": fault == "prototype_running"},
                "Mounts": [
                    {"Type": "bind", "Source": "/tmp/task-data", "Destination": "/data"}
                ],
                "Config": config,
            }
        ),
        _docker_cli=AsyncMock(),
    )

    def compose(args, **kwargs):
        return ExecResult(
            return_code=0,
            stderr="",
            stdout=json.dumps(project)
            if args[0] == "config"
            else "\n".join(["a" * 64] * (2 if fault == "duplicate_service" else 1)),
        )

    with pytest.raises(RuntimeError):
        await bootstrap.prepare(env, AsyncMock(side_effect=compose))
    env._docker_cli.assert_not_awaited()
