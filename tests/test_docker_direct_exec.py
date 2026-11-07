"""A started Docker sandbox runs its commands with ``docker exec`` and ``docker cp``.

``docker compose exec`` starts the compose plugin and loads the project for
every call (about 0.1 s on the engine bench's VM against 0.05 s for ``docker
exec``), and a trial runs dozens. start() learns the ``main`` container once
and exec() and the copies address it directly, with the same environment,
stdin, merged output and result; any command that can replace or remove the
container sends them back through compose. The first start skips ``down``
for a project nobody else uses, but retains a single ``up --wait`` so an
initialization failure cannot be masked by restarting the same container.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._base import (
    ROOT_EXEC_PATH,
    SYSTEM_SHELL,
    system_shell_env_flags,
    with_system_path,
)
from benchflow.sandbox._docker_sweep import release_project
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths
from tests._trust_admission import admitted

MAIN_ID = "4d95a10d423d" + "0" * 52
OTHER_ID = "a146163f04d2" + "1" * 52
# Every exec as root or as the image's user is plumbing (C8): the loader
# variables empty at the exec, PATH pinned before the command.
FLAGS = system_shell_env_flags()


def _plumbing(command: str) -> str:
    return with_system_path(command, ROOT_EXEC_PATH)


def _without_client_options(argv):
    # Guards the 7b4907c8/S1 merge: the hermetic client config is a global
    # Docker option, before both Compose and raw Docker subcommands.
    if argv[:2] == ["docker", "--config"]:
        return [argv[0], *argv[3:]]
    return argv


def _write_tar(fd: int, source: str) -> None:
    """``docker cp CONTAINER:source -``: a file, or a folder holding reward.json."""
    import io
    import tarfile

    base = Path(source).name
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        name = base
        if not base.endswith(".json"):
            folder = tarfile.TarInfo(base)
            folder.type = tarfile.DIRTYPE
            folder.mode = 0o755
            tar.addfile(folder)
            name = f"{base}/reward.json"
        member = tarfile.TarInfo(name)
        member.size = 2
        member.mode = 0o644
        tar.addfile(member, io.BytesIO(b"{}"))
    os.write(fd, buf.getvalue())


class FakeDocker:
    """``asyncio.create_subprocess_exec`` for docker: records argv and kwargs."""

    def __init__(self, ps_rows: list[str] | None = None) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.procs: list[AsyncMock] = []
        # compose ps --format answer: one tab-separated row per container
        self.ps_rows = [f"main\t{MAIN_ID}\trunning\t"] if ps_rows is None else ps_rows
        self.ps_q = f"{MAIN_ID}\n"
        self.ps_stderr = b""
        self.mounts: list[dict] = []

    async def __call__(self, *argv, **kwargs):
        argv = [str(a) for a in argv]
        self.calls.append((argv, kwargs))
        argv = _without_client_options(argv)
        out = b""
        err = None
        if argv[-3:] == ["config", "--format", "json"]:
            out = b'{"services": {"main": {"image": "fixture-main:latest", "build": {"context": "."}}}}'
        if argv[:2] == ["docker", "compose"] and "ps" in argv:
            out = (
                ("\n".join(self.ps_rows) + "\n").encode()
                if "--format" in argv
                else self.ps_q.encode()
            )
            if kwargs.get("stderr") == asyncio.subprocess.PIPE:
                err = self.ps_stderr
            else:
                out = self.ps_stderr + out
        elif argv[:2] == ["docker", "ps"]:
            # The launch mount inventory lists the project's containers.
            out = f"{MAIN_ID}\n".encode()
        elif argv[:2] == ["docker", "inspect"]:
            out = json.dumps([{"Id": MAIN_ID, "Mounts": self.mounts}]).encode()
        elif argv[:2] == ["docker", "exec"]:
            out = b"out\nerr\n"
        elif argv[:3] == ["docker", "system", "dial-stdio"]:
            metadata = base64.b64encode(json.dumps({"mode": 0o644}).encode())
            out = (
                b"HTTP/1.1 200 OK\r\nX-Docker-Container-Path-Stat: "
                + metadata
                + b"\r\n\r\n"
            )
        elif argv[:2] == ["docker", "cp"] and argv[-1] == "-":
            # A download streams the copied entry as a tar.
            _write_tar(kwargs["stdout"], argv[-2].split(":", 1)[1])
            stderr = asyncio.StreamReader()
            stderr.feed_eof()
            proc = AsyncMock()
            proc.stderr = stderr
            proc.wait = AsyncMock(return_value=0)
            return proc
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(out, err))
        self.procs.append(proc)
        return proc

    def commands(self) -> list[list[str]]:
        """Each call without the compose prefix: ``["compose", <subcommand>, ...]``
        or ``["docker", <subcommand>, ...]``."""
        result = []
        for argv, _ in self.calls:
            argv = _without_client_options(argv)
            if argv[:2] == ["docker", "compose"]:
                rest = argv[2:]
                while rest and rest[0] in (
                    "--project-name",
                    "--project-directory",
                    "-f",
                ):
                    rest = rest[2:]
                result.append(["compose", *rest])
            else:
                result.append(argv)
        return result


def _sandbox(tmp_path: Path, *, compose_file: bool = False) -> DockerSandbox:
    environment = tmp_path / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    if compose_file:
        (environment / "docker-compose.yaml").write_text(
            "services:\n  db:\n    image: postgres:16\n"
        )
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    sandbox = DockerSandbox(
        environment_dir=environment,
        environment_name="direct-exec",
        session_id="direct-exec",
        rollout_paths=paths,
        task_env_config=SandboxConfig(),
    )
    # The command-routing fake has no synchronous Compose configuration client.
    sandbox.freeze_launch = lambda *args: None
    sandbox._probe_verifier_log_mount = AsyncMock()
    # This fake models command routing, not a daemon's mounted filesystem.
    sandbox._pin_or_verify_bind_targets = AsyncMock()
    sandbox._prepare_inert_bootstrap = AsyncMock()
    sandbox._name_trial_volumes = AsyncMock()
    sandbox._release_inert_bootstrap = AsyncMock()
    return sandbox


@pytest.fixture
def docker(monkeypatch) -> FakeDocker:
    fake = FakeDocker()
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake)
    return fake


async def _started(sandbox: DockerSandbox) -> None:
    try:
        await sandbox.start(force_build=False)
    finally:
        release_project(sandbox.compose_project_name)


async def test_a_started_sandbox_execs_on_its_main_container(tmp_path, docker):
    sandbox = _sandbox(tmp_path)
    await _started(sandbox)
    docker.calls.clear()

    result = await sandbox.exec("id", cwd="/work", user="root", env={"K": "v"})

    ((argv, kwargs),) = docker.calls
    argv = _without_client_options(argv)
    n = 8 + len(FLAGS)
    assert argv[:n] == [
        "docker",
        "exec",
        "-i",
        "-w",
        "/work",
        "-u",
        "root",
        *FLAGS,
        MAIN_ID,
    ]
    assert argv[n : n + 2] == ["/bin/sh", "-c"]
    # Env values travel on the exec's stdin, never on argv (b156).
    assert "K=v" not in " ".join(argv) and ". /dev/stdin" in argv[n + 2]
    # compose exec's I/O: the env on stdin, stderr merged into stdout,
    # compose's env.
    assert kwargs["stdin"] == asyncio.subprocess.PIPE
    docker_proc = docker.procs[-1]
    docker_proc.communicate.assert_awaited_once_with(b"export K=v\nset --\n")
    assert kwargs["stderr"] == asyncio.subprocess.STDOUT
    assert kwargs["env"] == sandbox._docker_compose_env()
    assert (result.stdout, result.stderr, result.return_code) == ("out\nerr\n", None, 0)


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("operation", ["exec", "cp"])
async def test_direct_calls_keep_the_checked_docker_connection(
    tmp_path, monkeypatch, docker, frozen, operation
):
    """Guards the 7b4907c8/S1 merge: direct calls stay on Compose's daemon."""
    sandbox = _sandbox(tmp_path)
    flags = ("--config", "/tmp/dummy-config", "-H", "ssh://dummy@remote.invalid")
    monkeypatch.setattr(sandbox, "_docker_compose_global_args", lambda: flags)
    expected_env = sandbox._docker_compose_env()
    if frozen:
        expected_env = {**expected_env, "S1_FROZEN": "dummy-marker"}
        sandbox._launch = SimpleNamespace(docker_args=flags, env=expected_env)
    await sandbox._run_docker_container_command([operation, "dummy-container"])
    ((argv, kwargs),) = docker.calls
    assert argv == ["docker", *flags, operation, "dummy-container"]
    assert kwargs["env"] == expected_env


async def test_start_learns_the_container_before_the_first_exec(tmp_path, docker):
    sandbox = _sandbox(tmp_path)
    await _started(sandbox)
    commands = docker.commands()
    first_exec = next(i for i, c in enumerate(commands) if "exec" in c[:2])
    assert commands[first_exec][: 3 + len(FLAGS)] == ["docker", "exec", *FLAGS, MAIN_ID]
    assert not any(c[:2] == ["compose", "exec"] for c in commands)


async def test_execs_stay_direct_after_the_bootstrap_release(tmp_path, docker):
    """Guards py8/perf-trial: the inert bootstrap's release signals PID 1 with
    ``docker kill --signal USR1``, which forgets the main container like any
    kill. start() learned it before the release, so every exec after start went
    back through ``docker compose exec`` (49 of 52 execs in a TB2 trial)."""
    sandbox = _sandbox(tmp_path)

    async def release() -> None:
        await sandbox._docker_cli(["kill", "--signal", "USR1", MAIN_ID])

    sandbox._release_inert_bootstrap = release
    await _started(sandbox)
    docker.calls.clear()

    await sandbox.exec("id")

    ((argv, _),) = docker.calls
    assert _without_client_options(argv)[:2] == ["docker", "exec"]
    assert MAIN_ID in argv


async def test_copies_use_docker_cp_on_the_main_container(tmp_path, docker):
    sandbox = _sandbox(tmp_path)
    await _started(sandbox)
    docker.calls.clear()

    await sandbox.upload_file(
        tmp_path / "a.txt", "/app/a.txt", admission=admitted(sandbox)
    )
    await sandbox.upload_dir(tmp_path / "tests", "/tests", admission=admitted(sandbox))
    await sandbox.download_file("/logs/x.json", tmp_path / "x.json")
    await sandbox.download_dir("/logs/verifier", tmp_path / "verifier")

    copies = [c for c in docker.commands() if c[1] == "cp"]
    assert copies == [
        ["docker", "cp", str(tmp_path / "a.txt"), f"{MAIN_ID}:/app/a.txt"],
        ["docker", "cp", f"{tmp_path / 'tests'}/.", f"{MAIN_ID}:/tests"],
        # Downloads stream to BenchFlow and are published through held
        # descriptors; docker cp is never given a host path.
        ["docker", "cp", f"{MAIN_ID}:/logs/x.json", "-"],
        ["docker", "cp", f"{MAIN_ID}:/logs/verifier", "-"],
    ]
    assert (tmp_path / "x.json").read_text() == "{}"
    assert (tmp_path / "verifier" / "reward.json").read_text() == "{}"
    metadata_calls = [
        c for c in docker.commands() if c[1:3] == ["system", "dial-stdio"]
    ]
    assert metadata_calls == [["docker", "system", "dial-stdio"]]
    assert not any(c[:2] == ["compose", "exec"] for c in docker.commands())


async def test_other_services_still_go_through_compose(tmp_path, docker):
    sandbox = _sandbox(tmp_path)
    await _started(sandbox)
    docker.calls.clear()

    await sandbox.exec("true", service="db")
    await sandbox.upload_dir(
        tmp_path / "seed", "/seed", service="db", admission=admitted(sandbox, "db")
    )

    # Root's mkdir starts /bin/sh by path, never one on the image's PATH
    # (J1-R3-A-1); it still goes through compose for a service other than main.
    assert docker.commands() == [
        ["compose", "exec", "-T", *FLAGS, "db", SYSTEM_SHELL, "-c", _plumbing("true")],
        [
            "compose",
            "exec",
            "-T",
            "-u",
            "root",
            *system_shell_env_flags(),
            "db",
            SYSTEM_SHELL,
            "-c",
            with_system_path("mkdir -p /seed"),
        ],
        ["compose", "cp", f"{tmp_path / 'seed'}/.", "db:/seed"],
    ]


@pytest.mark.parametrize(
    "lifecycle",
    [
        ["stop", "-t", "5"],
        ["down", "-t", "0"],
        ["up", "--detach"],
        ["restart"],
        ["kill"],
    ],
)
async def test_a_compose_lifecycle_command_sends_exec_back_to_compose(
    tmp_path, docker, lifecycle
):
    sandbox = _sandbox(tmp_path)
    await _started(sandbox)
    await sandbox._run_docker_compose_command(lifecycle, check=False)
    docker.calls.clear()

    await sandbox.exec("true")

    assert docker.commands() == [
        ["compose", "exec", "-T", *FLAGS, "main", SYSTEM_SHELL, "-c", _plumbing("true")]
    ]


async def test_a_replaced_container_sends_exec_back_to_compose(tmp_path, docker):
    """restore() removes the container and runs a new one with the docker CLI."""
    sandbox = _sandbox(tmp_path)
    await _started(sandbox)
    await sandbox._docker_cli(["rm", "-f", MAIN_ID], check=False)
    docker.calls.clear()

    await sandbox.exec("true")

    assert docker.commands() == [
        ["compose", "exec", "-T", *FLAGS, "main", SYSTEM_SHELL, "-c", _plumbing("true")]
    ]


async def test_teardown_sends_exec_back_to_compose(tmp_path, docker):
    sandbox = _sandbox(tmp_path)
    await _started(sandbox)
    await sandbox.stop(delete=True)
    docker.calls.clear()

    await sandbox.exec("true")

    assert docker.commands() == [
        ["compose", "exec", "-T", *FLAGS, "main", SYSTEM_SHELL, "-c", _plumbing("true")]
    ]


@pytest.mark.parametrize(
    "answer",
    [
        "",
        f"{MAIN_ID}\n{OTHER_ID}\n",
        "not-an-id\n",
    ],
    ids=["none", "two", "garbage"],
)
async def test_exec_stays_on_compose_unless_one_container_is_named(
    tmp_path, docker, answer
):
    docker.ps_rows = ["main\t\trunning\t"]  # the fast start cannot name it either
    docker.ps_q = answer
    sandbox = _sandbox(tmp_path)
    await _started(sandbox)
    docker.calls.clear()

    await sandbox.exec("true")

    assert docker.commands() == [
        ["compose", "exec", "-T", *FLAGS, "main", SYSTEM_SHELL, "-c", _plumbing("true")]
    ]


async def test_compose_warnings_around_the_id_are_skipped(tmp_path, docker):
    docker.ps_rows = []
    docker.ps_stderr = b'time="2026" level=warning msg="X is not set"\n'
    sandbox = _sandbox(tmp_path)
    await _started(sandbox)
    docker.calls.clear()

    await sandbox.exec("true")

    assert docker.commands()[0][: 3 + len(FLAGS)] == ["docker", "exec", *FLAGS, MAIN_ID]


async def test_a_first_start_skips_down_but_keeps_a_single_compose_wait(
    tmp_path, docker
):
    sandbox = _sandbox(tmp_path)
    await _started(sandbox)
    lifecycle = [c[:3] for c in docker.commands() if c[0] == "compose"]
    assert lifecycle[:3] == [
        ["compose", "build"],
        ["compose", "up", "--detach"],
        ["compose", "ps", "-q"],
    ]
    assert not any("down" in c for c in lifecycle)
    assert [c for c in docker.commands() if c[:2] == ["compose", "up"]] == [
        ["compose", "up", "--detach", "--wait", "--no-build"]
    ]


@pytest.mark.parametrize(
    "rows",
    [
        [f"main\t{MAIN_ID}\tcreated\t"],
        [f"main\t{MAIN_ID}\trunning\tstarting"],
        [f"main\t{MAIN_ID}\trunning\tunhealthy"],
        [f"main\t{MAIN_ID}\texited\t"],
        [f"main\t{MAIN_ID}\trunning\t", f"db\t{OTHER_ID}\trunning\t"],
    ],
    ids=["created", "starting", "unhealthy", "exited", "two-services"],
)
async def test_anything_but_one_ready_main_waits_as_compose_does(
    tmp_path, docker, rows
):
    docker.ps_rows = rows
    sandbox = _sandbox(tmp_path)
    await _started(sandbox)
    ups = [c for c in docker.commands() if c[:2] == ["compose", "up"]]
    assert ups == [["compose", "up", "--detach", "--wait", "--no-build"]]


async def test_a_healthy_main_is_ready(tmp_path, docker):
    docker.ps_rows = [f"main\t{MAIN_ID}\trunning\thealthy"]
    sandbox = _sandbox(tmp_path)
    await _started(sandbox)
    assert [c for c in docker.commands() if c[:2] == ["compose", "up"]] == [
        ["compose", "up", "--detach", "--wait", "--no-build"]
    ]


async def test_a_task_compose_file_keeps_compose_wait(tmp_path, docker):
    """Task services may be one-shot jobs or slow to become healthy."""
    sandbox = _sandbox(tmp_path, compose_file=True)
    await _started(sandbox)
    ups = [c for c in docker.commands() if c[:2] == ["compose", "up"]]
    assert ups == [["compose", "up", "--detach", "--wait", "--no-build"]]


async def test_start_never_tries_a_detached_up_before_the_wait(tmp_path, monkeypatch):
    fake = FakeDocker()

    async def failing_first_up(*argv, **kwargs):
        proc = await fake(*argv, **kwargs)
        if "up" in argv and "--wait" not in argv:
            proc.returncode = 1
            proc.communicate = AsyncMock(return_value=(b"network race", None))
        return proc

    monkeypatch.setattr("asyncio.create_subprocess_exec", failing_first_up)
    sandbox = _sandbox(tmp_path)
    await _started(sandbox)
    ups = [c for c in fake.commands() if c[:2] == ["compose", "up"]]
    assert ups == [["compose", "up", "--detach", "--wait", "--no-build"]]


async def test_a_second_start_clears_the_first_one_first(tmp_path, docker):
    sandbox = _sandbox(tmp_path)
    await _started(sandbox)
    docker.calls.clear()
    await _started(sandbox)
    lifecycle = [c[:3] for c in docker.commands() if c[0] == "compose"]
    assert ["compose", "down", "--remove-orphans"] in lifecycle
