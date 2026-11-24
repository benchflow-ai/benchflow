"""/logs/verifier is never the agent's to write (#1247).

Red on 0.10.0. The close before test.sh (root's 0755, then the sweep of
other users' names) keeps a planted reward out of the score, but:

1. from sandbox start through the agent's turn the folder is 0777, so any
   uid can create names there (a flood the close must then sweep, a
   descriptor held on it);
2. after test.sh, a mounted folder goes back to the host uid, and the
   agent's processes, alive through grading in audit and off, can make
   names there again when that uid is also the agent's. A planted link
   named reward.json is never seen by the put-back (it lists regular files
   only), so it stays in the trial's verifier/ unflagged.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.protocol import ExecResult
from benchflow.task import RolloutPaths
from benchflow.task import verifier_outputs as vo
from benchflow.task.config import SandboxConfig
from tests.integration.deterministic import harness as h
from tests.test_verifier_script_shell_docker import POSIX_TEST_SH, UBUNTU, _task

SANDBOX, SKIP_REASON = h.select_sandbox()
# Any uid but root and the host's: BenchFlow's sandbox user is one of these.
AS_OTHER = "setpriv --reuid=4242 --regid=4242 --clear-groups"


@pytest.mark.deterministic
async def test_the_verifier_folder_is_closed_to_other_uids_after_start(tmp_path):
    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else "Docker only")
    name = "verifier-dir-closed"
    task_dir = _task(tmp_path / "tasks", name, POSIX_TEST_SH, UBUNTU)
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    sandbox = DockerSandbox(
        environment_dir=task_dir / "environment",
        environment_name=name,
        session_id=name,
        rollout_paths=paths,
        task_env_config=SandboxConfig(cpus=1, memory_mb=512),
    )
    await sandbox.start(force_build=False)
    try:
        shown = await sandbox.exec(
            "stat -c %a /logs/verifier; "
            f"{AS_OTHER} touch /logs/verifier/planted 2>/dev/null "
            "&& echo planted || echo refused",
            user="root",
        )
    finally:
        await sandbox.stop(delete=True)
    mode, outcome = (getattr(shown, "stdout", "") or "").split()
    assert outcome == "refused", "a uid that is not root nor the host's wrote it"
    assert not int(mode, 8) & 0o022, f"/logs/verifier is {mode}"


def test_a_link_planted_after_exit_is_removed_and_flagged(tmp_path: Path):
    trial = tmp_path / "trial"
    (trial / "verifier").mkdir(parents=True)
    (trial / "verifier" / "reward.txt").write_text("0\n")
    (trial / "integrity").mkdir()
    (trial / "integrity" / "verifier-inputs.json").write_text(
        json.dumps({"status": "checked", "flags": [], "sections": {}})
    )
    taken = vo.take(trial)
    # What an agent process writes into the mounted folder once it is the
    # host uid's again: a link to a path on the host.
    (tmp_path / "elsewhere.json").write_text('{"reward": 1}')
    (trial / "verifier" / "reward.json").symlink_to(tmp_path / "elsewhere.json")
    left = trial / "verifier" / "reward.json"
    # While the sandbox may still run, put_back only names what test.sh didn't
    # write (#1247 part 3): the link is flagged, not removed.
    flags = vo.put_back(trial, taken, {})
    assert any(f["path"] == "/logs/verifier/reward.json" for f in flags), flags
    assert left.is_symlink(), (
        "the first pass removed a name while the sandbox may still run"
    )
    # After the sandbox's stop is confirmed, the same names are removed.
    flags = vo.put_back(trial, taken, {}, remove=True)
    assert not left.is_symlink(), "the planted link stays in verifier/ after the stop"
    assert any(f["path"] == "/logs/verifier/reward.json" for f in flags), flags


# -- Every backend: the setup step's own commands, run in a real container --------------------------------------


@pytest.fixture
def provider(monkeypatch):
    """A fake ``modal`` SDK (as ``tests/test_modal_gpu.py``'s): its sandbox's execs are the test's."""
    module = ModuleType("modal")
    module.App = SimpleNamespace(lookup=SimpleNamespace(aio=AsyncMock()))
    module.Image = MagicMock()
    module.Secret = MagicMock()
    module.Volume = MagicMock()
    remote = SimpleNamespace(
        filesystem=SimpleNamespace(make_directory=SimpleNamespace(aio=AsyncMock()))
    )
    module.Sandbox = SimpleNamespace(
        create=SimpleNamespace(aio=AsyncMock(return_value=remote))
    )
    monkeypatch.setitem(sys.modules, "modal", module)
    monkeypatch.setattr(
        "benchflow.sandbox.modal_impl.bfx_delivery.provider_enabled",
        lambda name: False,
    )
    return module


_IMAGE = "python:3.12-slim"
_PROBE = (
    "stat -c '%u %a' /logs/verifier; "
    f"{AS_OTHER} touch /logs/verifier/planted 2>/dev/null && echo planted || echo refused"
)


@contextlib.contextmanager
def _container():
    """A ``python:3.12-slim`` container with root's /logs/agent and /logs/verifier (0755), as a backend's
    ``mkdir`` leaves them, before the setup step runs."""
    if (
        shutil.which("docker") is None
        or subprocess.run(
            ["docker", "info"], capture_output=True, timeout=20
        ).returncode
        != 0
    ):
        pytest.skip("docker required")
    name = "bf-verifier-dir-" + uuid.uuid4().hex[:8]
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--network",
            "none",
            _IMAGE,
            "sleep",
            "600",
        ],
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        pytest.skip(f"cannot start {_IMAGE}: {run.stderr.strip()}")
    try:
        subprocess.run(
            ["docker", "exec", name, "sh", "-c", "mkdir -p /logs/agent /logs/verifier"],
            check=True,
            capture_output=True,
        )
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _runs_in(name: str):
    """A backend ``exec`` that runs each command as root in container *name*."""

    async def exec_(command, *args, **kwargs):
        done = await asyncio.to_thread(
            subprocess.run,
            ["docker", "exec", "-u", "0", name, "sh", "-c", str(command)],
            capture_output=True,
            text=True,
        )
        return ExecResult(
            return_code=done.returncode, stdout=done.stdout, stderr=done.stderr
        )

    return exec_


def _closed(name: str) -> None:
    shown = subprocess.run(
        ["docker", "exec", "-u", "0", name, "sh", "-c", _PROBE],
        capture_output=True,
        text=True,
    ).stdout.split()
    assert len(shown) == 3, shown
    owner, mode, outcome = shown
    assert outcome == "refused", f"uid 4242 created a file in /logs/verifier ({shown})"
    assert owner == "0" and not int(mode, 8) & 0o022, f"/logs/verifier is {shown}"


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [False, True], ids=["modal_impl", "setup-factory"])
async def test_modal_setup_leaves_the_verifier_folder_roots(
    tmp_path, provider, factory
):
    """Modal (both launchers, fake Modal API): the start's own commands, run as root in a container,
    leave /logs/verifier root's 0755; another uid can't create a name there.

    Mutant: put back the ``chmod 777``."""
    from benchflow.task.config import SandboxConfig
    from tests.test_modal_gpu import _sandbox as modal_sandbox

    with _container() as name:
        sandbox = modal_sandbox(
            tmp_path, factory=factory, config=SandboxConfig(cpus=1, memory_mb=512)
        )
        sandbox.exec = _runs_in(name)
        await sandbox.start(force_build=False)
        _closed(name)


@pytest.mark.asyncio
async def test_remote_docker_setup_leaves_the_verifier_folder_roots():
    """Remote Docker: nothing is mounted; the folders are made in the container and /logs/verifier is
    root's 0755.

    Mutant: put back the ``chmod 777``."""
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    with _container() as name:
        sandbox = object.__new__(RemoteDockerSandbox)
        sandbox.exec = _runs_in(name)
        await sandbox._prepare_log_dirs_after_up()
        _closed(name)


@pytest.mark.parametrize(
    "command",
    ["plain", "closed", "host", "bfx"],
)
def test_every_clear_ends_with_the_folder_closed(command):
    """Unit. Every clear of /logs/verifier leaves it 0755, never 0777: root's, or, for the host's own
    mounted folder, the host's.

    Mutant: the clear's last ``chmod 777``."""
    from benchflow.sandbox import lockdown

    text = {
        "plain": lockdown._CLEAR_VERIFIER_DIR_CMD,
        "closed": lockdown._CLEAR_VERIFIER_DIR_CLOSED_CMD,
        "host": lockdown._CLEAR_HOST_VERIFIER_DIR_CMD,
        "bfx": lockdown._clear_verifier_dir_cmd(bfx=True),
    }[command]
    assert "chmod 777" not in text, text
    assert text.count("chmod 755 /logs/verifier") >= 1, text
    if command != "host":
        assert "chown root:root /logs/verifier && chmod 755 /logs/verifier" in text, (
            text
        )


def test_the_clear_leaves_the_folder_roots_in_a_container():
    """Real kernel. The clear, run as root, empties /logs/verifier and leaves it root's 0755."""
    from benchflow.sandbox import lockdown

    with _container() as name:
        subprocess.run(
            [
                "docker",
                "exec",
                name,
                "sh",
                "-c",
                "chmod 777 /logs/verifier && touch /logs/verifier/old",
            ],
            check=True,
            capture_output=True,
        )
        done = subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                "0",
                name,
                "sh",
                "-c",
                lockdown._CLEAR_VERIFIER_DIR_CMD,
            ],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        _closed(name)


# -- A setup whose result isn't shown closed refuses the trial -------------------------------------------------


def _refused(exc: BaseException) -> None:
    from benchflow._utils.scoring import INTEGRITY_UNAVAILABLE, classify_error

    assert classify_error(str(exc)) == INTEGRITY_UNAVAILABLE, str(exc)
    assert "/logs/verifier" in str(exc), str(exc)


def _unshown():
    """An exec whose setup step found /logs/verifier open (the sandbox's own shell judged it)."""
    return AsyncMock(
        return_value=ExecResult(
            return_code=0, stdout="BF verifier-dir-open 1001 777\n", stderr=""
        )
    )


@pytest.mark.asyncio
async def test_docker_refuses_a_verifier_folder_it_cannot_show_closed(tmp_path):
    """Docker. The setup's root exec shows the folder's owner and mode; an answer that doesn't show it
    closed refuses the trial at setup (``integrity_unavailable``), never an open folder.

    Mutant: ignore the shown mode."""
    import logging

    sandbox = object.__new__(DockerSandbox)
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    sandbox.rollout_paths = paths
    sandbox.logger = logging.getLogger("test")
    sandbox.exec = _unshown()
    with pytest.raises(Exception) as caught:
        await sandbox._probe_verifier_log_mount()
    _refused(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [False, True], ids=["modal_impl", "setup-factory"])
async def test_modal_refuses_a_verifier_folder_it_cannot_show_closed(
    tmp_path, provider, factory
):
    """Modal (both launchers).

    Mutant: ignore the shown mode."""
    from benchflow.task.config import SandboxConfig
    from tests.test_modal_gpu import _sandbox as modal_sandbox

    sandbox = modal_sandbox(
        tmp_path, factory=factory, config=SandboxConfig(cpus=1, memory_mb=512)
    )
    sandbox.exec = _unshown()
    with pytest.raises(Exception) as caught:
        await sandbox.start(force_build=False)
    _refused(caught.value)


@pytest.mark.asyncio
async def test_remote_docker_refuses_a_verifier_folder_it_cannot_show_closed():
    """Remote Docker.

    Mutant: ignore the shown mode."""
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    sandbox = object.__new__(RemoteDockerSandbox)
    sandbox.exec = _unshown()
    with pytest.raises(Exception) as caught:
        await sandbox._prepare_log_dirs_after_up()
    _refused(caught.value)


# -- BenchFlow's own sandbox user never takes the host's uid where the logs are the host's mount ----------------


@pytest.mark.asyncio
async def test_the_default_user_skips_the_host_uid_on_a_mounted_sandbox():
    """Remapped. On a sandbox whose /logs/verifier is the host's bind mount (``is_mounted``), BenchFlow's own
    ``agent`` user never gets the uid BenchFlow runs as, even where no file in the sandbox shows that uid (on local
    Docker the mounted log folders do, today, while they are the host's). Every uid from 1000 below the host's is
    taken in the image, so the first free one is the host's. The user still owns its workspace: the grant goes by
    name.

    Mutant: no remap."""
    from benchflow.sandbox import lockdown
    from tests.test_c8_protected_runtime_container import _sandbox

    host = os.getuid()
    if host < 1000:
        pytest.skip(
            "the host's uid is below the range BenchFlow gives its sandbox user"
        )
    with _container() as name:
        taken = " && ".join(
            f"groupadd -g {u} u{u} && useradd -M -u {u} -g {u} u{u}"
            for u in range(1000, host)
        )
        made = subprocess.run(
            [
                "docker",
                "exec",
                name,
                "sh",
                "-c",
                f"{taken + ' && ' if taken else ''}mkdir -p /app && echo old > /app/in.txt",
            ],
            capture_output=True,
            text=True,
        )
        assert made.returncode == 0, made.stderr
        sandbox = _sandbox(name)
        sandbox._logs_are_mounted = True
        await lockdown.setup_sandbox_user(sandbox, "agent", "/app", guard_daemon=False)
        uid = subprocess.run(
            ["docker", "exec", name, "id", "-u", "agent"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert uid and uid != str(host), f"the sandbox user got the host's uid {uid}"
        works = subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                "agent",
                name,
                "sh",
                "-c",
                "echo new >> /app/in.txt && touch /app/made && mkdir /app/d && rm /app/made && echo ok",
            ],
            capture_output=True,
            text=True,
        )
        assert works.stdout.strip() == "ok", works.stderr


@pytest.mark.parametrize(
    ("stdout", "closed"),
    [
        ("", True),
        ("BF daemon-log absent\n", True),
        ("BF verifier-dir-open 0 777\n", False),
        ("BF verifier-dir-open 1000 755\n", False),
        ("BF verifier-dir-open unknown\n", False),
    ],
)
def test_the_setup_check_refuses_a_folder_its_shell_found_open(stdout, closed):
    """The check every backend's setup shares. The setup step judges the folder in the sandbox's own
    shell, in the same exec, and says so when it isn't closed; that refuses the trial.

    Mutant: ignore what the step says."""
    from benchflow.sandbox._verifier_dir import check_verifier_dir

    if closed:
        check_verifier_dir(stdout)
        return
    with pytest.raises(Exception) as caught:
        check_verifier_dir(stdout)
    _refused(caught.value)


@pytest.mark.parametrize(
    ("setup", "host_uid"),
    [
        ("chmod 777 /logs/verifier", None),
        ("chown 4242 /logs/verifier", None),
        ("chown 1001 /logs/verifier", 1001),
        ("chown 1001 /logs/verifier", None),
    ],
)
def test_the_setup_step_closes_or_says_open_in_a_container(setup, host_uid):
    """Real kernel. The setup step, as root, closes whatever it finds (0755; root's, or kept the host's on
    its mount) and prints nothing; as a uid that can't close it, it says the folder is open."""
    from benchflow.sandbox._verifier_dir import close_verifier_dir_cmd

    with _container() as name:
        subprocess.run(["docker", "exec", name, "sh", "-c", setup], check=True)
        done = subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                "0",
                name,
                "sh",
                "-c",
                close_verifier_dir_cmd(host_uid=host_uid),
            ],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0 and done.stdout == "", done
        owner = "1001" if host_uid is not None else "0"
        shown = subprocess.run(
            ["docker", "exec", name, "stat", "-c", "%u %a", "/logs/verifier"],
            capture_output=True,
            text=True,
        ).stdout.split()
        assert shown == [owner, "755"], shown
        subprocess.run(
            [
                "docker",
                "exec",
                name,
                "sh",
                "-c",
                "chown 4242 /logs/verifier && chmod 777 /logs/verifier",
            ],
            check=True,
        )
        as_other = subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                "4242",
                name,
                "sh",
                "-c",
                close_verifier_dir_cmd(),
            ],
            capture_output=True,
            text=True,
        )
    assert as_other.stdout.split() == ["BF", "verifier-dir-open", "4242", "777"], (
        as_other
    )


# -- A non-root image user; Apple Container and Daytona DinD; a folder still unwritable after its chown ----------


@pytest.mark.deterministic
async def test_a_non_root_image_user_gets_the_folder_closed_not_refused(tmp_path):
    """Docker, image ``USER app``. The start's own command runs as the image's default user,
    which can't chmod the host's folder; the close is root's (the mount probe's exec), so the folder is closed and
    the trial is not refused.

    Mutant: put the close in the default user's start command."""
    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else "Docker only")
    name = "verifier-dir-closed-user-app"
    # uid 4343: off the host's (ubuntu:24.04 has uid 1000, so a plain useradd would give 1001, a test VM's host uid,
    # and that user could close the host's folder itself).
    image = UBUNTU + "RUN useradd -m -u 4343 app\nUSER app\n"
    task_dir = _task(tmp_path / "tasks", name, POSIX_TEST_SH, image)
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    sandbox = DockerSandbox(
        environment_dir=task_dir / "environment",
        environment_name=name,
        session_id=name,
        rollout_paths=paths,
        task_env_config=SandboxConfig(cpus=1, memory_mb=512),
    )
    await sandbox.start(force_build=False)
    try:
        shown = await sandbox.exec(
            "id -un; stat -c %a /logs/verifier; "
            f"{AS_OTHER} touch /logs/verifier/planted 2>/dev/null "
            "&& echo planted || echo refused",
            user="root",
        )
        default = await sandbox.exec("id -un")
    finally:
        await sandbox.stop(delete=True)
    _, mode, outcome = (getattr(shown, "stdout", "") or "").split()
    assert (getattr(default, "stdout", "") or "").strip() == "app"
    assert outcome == "refused", "a uid that is not root nor the host's wrote it"
    assert not int(mode, 8) & 0o022, f"/logs/verifier is {mode}"


@pytest.mark.asyncio
async def test_apple_container_mounts_a_closed_verifier_folder(tmp_path):
    """Apple Container (the CLI stubbed). The host's ``verifier`` folder, bind-mounted at
    ``/logs/verifier`` with the rest of the trial folder, is the host user's 0755, not 0777; the agent's and the
    artifacts' folders stay open.

    Mutant: put back the ``0o777``."""
    import stat as stat_mod
    from unittest.mock import patch

    from benchflow.sandbox import apple_container as apple_mod
    from benchflow.sandbox.apple_container import AppleContainerSandbox

    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    paths = RolloutPaths(tmp_path / "rollout")
    with patch.object(AppleContainerSandbox, "preflight"):
        sandbox = AppleContainerSandbox(
            environment_dir=env_dir,
            environment_name="test-task",
            session_id="sess-d1",
            rollout_paths=paths,
            task_env_config=SandboxConfig(cpus=2, memory_mb=1024),
        )
    with (
        patch.object(apple_mod, "_RUN_SLOT", asyncio.Semaphore(1)),
        patch.object(apple_mod, "_require_kalloc_headroom"),
        patch.object(
            sandbox,
            "_resolve_image",
            new_callable=AsyncMock,
            return_value="ubuntu:24.04",
        ),
        patch.object(
            apple_mod,
            "_run_cli",
            new_callable=AsyncMock,
            return_value=ExecResult(stdout="", stderr=None, return_code=0),
        ),
    ):
        await sandbox.start(force_build=False)
    modes = {
        sub: stat_mod.S_IMODE((paths.rollout_dir / sub).stat().st_mode)
        for sub in ("verifier", "agent", "artifacts")
    }
    assert modes == {"verifier": 0o755, "agent": 0o777, "artifacts": 0o777}, modes


def test_daytona_dind_leaves_the_verifier_folder_roots():
    """Daytona with Compose (DinD): the VM's folder that Compose bind-mounts at
    ``/logs/verifier`` is root's 0755 after the log-folder step, run as root in a container; the agent's stays open.

    Mutant: put back the ``chmod 777`` on the verifier's folder."""
    from benchflow.sandbox.daytona_dind import dind_log_dirs_command

    with _container() as name:
        command = dind_log_dirs_command("/runtime/logs", ["/runtime/task/tests"])
        done = subprocess.run(
            ["docker", "exec", "-u", "0", name, "sh", "-c", command],
            capture_output=True,
            text=True,
        )
        from benchflow.sandbox._verifier_dir import check_verifier_dir

        check_verifier_dir(done.stdout)
        shown = subprocess.run(
            [
                "docker", "exec", "-u", "0", name, "sh", "-c",
                "stat -c '%u %a' /runtime/logs/verifier /runtime/logs/agent; "
                f"{AS_OTHER} touch /runtime/logs/verifier/planted 2>/dev/null "
                "&& echo planted || echo refused",
            ],
            capture_output=True,
            text=True,
        ).stdout.split()  # fmt: skip
    assert shown == ["0", "755", "0", "777", "refused"], shown


@pytest.mark.asyncio
async def test_the_folder_is_never_opened_to_every_user_after_test_sh():
    """Once test.sh has exited, a mounted folder the host still can't write after its chown is not
    made 0777 (the agent's processes may be alive in audit and off): the trial stops with the reason.

    Mutant: put back the ``chmod 777`` fallback."""
    from types import SimpleNamespace

    from benchflow.task import TaskConfig
    from benchflow.task.verifier_core import Verifier
    from benchflow.task.verifier_errors import VerifierInfrastructureError

    task = MagicMock()
    task.config = TaskConfig.model_validate({"verifier": {}})
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(
                stdout="", stderr="", return_code=0, exit_code=0
            )
        )
    )
    verifier = Verifier(task=task, rollout_paths=MagicMock(), sandbox=sandbox)
    verifier._closed_to_host = True
    verifier._host_writes_outputs = lambda: False  # type: ignore[method-assign]
    with pytest.raises(
        VerifierInfrastructureError, match="will not open it to every user"
    ):
        await verifier._open_output_dir_to_host("main")
    commands = [str(c.args[0]) for c in sandbox.exec.await_args_list]
    assert not any("777" in c for c in commands), commands


def test_daytona_refuses_a_verifier_folder_its_setup_exec_found_open():
    """Daytona. The root exec after the start closes /logs/verifier and checks it in the shell; a folder it
    can't close ends that exec before the daemon-log step with its reason, and the sandbox is refused
    (``integrity_unavailable``).

    Mutant: ignore the shown mode."""
    from benchflow.sandbox.daytona_strategies import check_daemon_log_result

    with pytest.raises(Exception) as caught:
        check_daemon_log_result(
            ExecResult(
                return_code=1,
                stdout="BF daemon-log verifier-dir-open 0 777\n",
                stderr="",
            )
        )
    _refused(caught.value)
    check_daemon_log_result(
        ExecResult(return_code=0, stdout="BF daemon-log absent\n", stderr="")
    )


def test_daytonas_setup_check_in_the_shell_takes_only_a_closed_folder():
    """Daytona, real kernel. The shell's own check, run as root (the folder closed: nothing printed, the step
    goes on) and as a uid that can't close a folder it was given 0777 (the reason printed, exit 1)."""
    from benchflow.sandbox._verifier_dir import close_verifier_dir_or_exit_cmd

    command = close_verifier_dir_or_exit_cmd("BF daemon-log") + "; echo GOES-ON"
    with _container() as name:
        as_root = subprocess.run(
            ["docker", "exec", "-u", "0", name, "sh", "-c", command],
            capture_output=True,
            text=True,
        )
        assert (as_root.returncode, as_root.stdout.split()) == (0, ["GOES-ON"]), as_root
        subprocess.run(
            [
                "docker",
                "exec",
                name,
                "sh",
                "-c",
                "chown 4242 /logs/verifier && chmod 777 /logs/verifier",
            ],
            check=True,
        )
        as_other = subprocess.run(
            ["docker", "exec", "-u", "4242", name, "sh", "-c", command],
            capture_output=True,
            text=True,
        )
    assert as_other.returncode == 1, as_other
    assert as_other.stdout.split() == [
        "BF",
        "daemon-log",
        "verifier-dir-open",
        "4242",
        "777",
    ], as_other
