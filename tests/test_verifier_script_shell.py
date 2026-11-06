"""test.sh runs under the shell its task was written for, on every sandbox kind.

Guards the fix for verifier scripts whose shebang is not on line 1.
SWE-bench Verified's and SWT-bench's test.sh (as the public task ports ship
them) open with a comment and have ``#!/bin/bash`` on line 2 or 3, then use
``set -o pipefail``. The verifier started test.sh by path from the backend's
shell, so the kernel found no shebang and left the script to that shell: on
Docker, remote Docker, Daytona-in-Docker and Apple Container that is ``sh -c``,
dash on Debian and Ubuntu images, which died on the first bash-only line. Every
trial, the reference solution's included, ended as an unscored verifier crash.
Now a first-line shebang wins, a script without one runs under bash when the
image has bash, and under ``sh`` only when it does not.

Each sandbox kind is driven through its own ``exec`` (the real wrapping: the
``sh -c`` or ``bash -c`` it calls through, its ``timeout``, ``su`` and env-file
layers), with only the transport replaced by a local run of the command it
would send, sandbox paths mapped into a temp folder. The real-Docker tier is
``tests/test_verifier_script_shell_docker.py``.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import socket
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from benchflow.sandbox._base import SYSTEM_SHELL_EMPTY_VARS, ExecResult
from benchflow.sandbox._task_route import as_task_command
from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig
from benchflow.task.verifier_document import VerifierStrategy
from benchflow.task.verifier_errors import RewardFileNotFoundError
from benchflow.task.verifier_script_strategy import _script_strategy_command
from tests._receipt_fakes import receipt_aware
from tests.reachable_fixture import on_disk

# The SWE-bench Verified shape: a comment first, the shebang on line 2, then
# three things dash cannot run (pipefail, an array, ``[[ ]]``).
LATE_SHEBANG_BASH = """# This file runs the tests for the instance
#!/bin/bash
set -uo pipefail
names=(alpha beta)
if [[ ${#names[@]} -eq 2 && -n "${BASH_VERSION:-}" ]]; then
  echo 1 > "$VERIFIER_DIR/reward.txt"
else
  echo 0 > "$VERIFIER_DIR/reward.txt"
fi
"""

# Runs under Python only: as a shell script line 2 is a syntax error.
PYTHON_SHEBANG = """#!/usr/bin/env python3
import os, sys
with open(os.path.join(os.environ["VERIFIER_DIR"], "reward.txt"), "w") as f:
    f.write("1" if sys.version_info[0] == 3 else "0")
"""

# POSIX only and no shebang: scores 1 under a shell that is not bash, 0 under
# bash, so it says which shell ran it.
POSIX_NO_SHEBANG = """set -eu
if [ -z "${BASH_VERSION:-}" ]; then r=1; else r=0; fi
echo "$r" > "$VERIFIER_DIR/reward.txt"
"""

# No shebang at all and bash-only: runs under bash when the image has it.
BASH_NO_SHEBANG = """set -o pipefail
words=(one two)
[[ ${#words[@]} -eq 2 ]] && echo 1 > "$VERIFIER_DIR/reward.txt"
"""

# What the backends call through: ``sh`` on Debian and Ubuntu images is dash.
_DASH = shutil.which("dash")


def _bin_dir(root: Path, *, with_bash: bool) -> Path:
    """A PATH of the host's tools with ``sh`` as dash, and bash only if asked.

    It stands in for the image: ``command -v bash`` finds bash only when the
    image has it.
    """
    bin_dir = root / ("bin-bash" if with_bash else "bin-nobash")
    bin_dir.mkdir()
    for folder in ("/usr/local/bin", "/usr/bin", "/bin"):
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            target = bin_dir / name
            if name in ("bash", "rbash", "sh") or target.exists():
                continue
            source = os.path.join(folder, name)
            if os.path.isfile(source) and os.access(source, os.X_OK):
                target.symlink_to(source)
    sh = _DASH or shutil.which("sh")
    assert sh is not None
    (bin_dir / "sh").symlink_to(sh)
    if with_bash:
        bash = shutil.which("bash")
        assert bash is not None
        (bin_dir / "bash").symlink_to(bash)
    return bin_dir


class LocalImage:
    """Runs what a backend would send into its sandbox, here, paths mapped."""

    def __init__(
        self,
        tmp_path: Path,
        paths: RolloutPaths,
        *,
        with_bash: bool,
        image_env: dict[str, str] | None = None,
    ):
        self.tests = tmp_path / "sandbox-tests"
        self.paths = paths
        self.bin = _bin_dir(tmp_path, with_bash=with_bash)
        # What the image's ENV lines put in every process (BASH_ENV, ...).
        self.image_env = dict(image_env or {})
        self.sent: list[list[str]] = []
        # Root's staging folder (benchflow.sandbox._transport.STAGE_ROOT).
        self.stage = tmp_path / "stage-root"

    def local(self, text: str) -> str:
        # Backends start su by its absolute path (C8); the image's is ours.
        return (
            text.replace("/logs/verifier", str(on_disk(self.paths.verifier_dir)))
            .replace("/tests/", f"{self.tests}/")
            .replace("/bin/su ", f"{self.bin}/su ")
            .replace("/opt/benchflow/run/", f"{self.stage}/")
        )

    def run(
        self,
        argv: list[str],
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        stdin: str | None = None,
    ) -> ExecResult:
        """Run *argv* (a shell and its arguments) as the sandbox would.

        *env* is what the transport itself puts in the process (Modal's
        secrets, Daytona's create-time env), over the image's own; *stdin* is
        what the shell reads its commands from.
        """
        self.sent.append(argv)
        argv = [self.local(part) for part in argv]
        stdin = None if stdin is None else self.local(stdin)
        program = shutil.which(argv[0], path=str(self.bin))
        assert program is not None, argv[0]
        result = subprocess.run(
            [program, *argv[1:]],
            env={
                "PATH": str(self.bin),
                "VERIFIER_DIR": str(on_disk(self.paths.verifier_dir)),
                **self.image_env,
                **(env or {}),
            },
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout or 60,
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )


# --- each sandbox kind, real exec over a local transport ---------------------
#
# A factory returns ``exec(command, timeout_sec)`` that goes through that
# backend's own ``exec``. The user is left out: the transport runs as the test's
# user, so a backend's ``su`` cannot be replayed (the verifier's own user is
# the image's, which adds none).

Exec = Callable[..., Awaitable[ExecResult]]


def _docker(image: LocalImage, cls: Any = None) -> Exec:
    from benchflow.sandbox.docker import DockerSandbox

    sandbox = (cls or DockerSandbox).__new__(cls or DockerSandbox)
    sandbox._persistent_env = {}
    sandbox.default_user = None

    async def compose(
        args, check=False, timeout_sec=None, *, raw_output=False, stdin_data=None
    ):
        # ["exec", "-T", ..., "main", "sh", "-c", command]; the env, if any,
        # on stdin
        stdin = None if stdin_data is None else stdin_data.decode()
        return image.run(args[args.index("main") + 1 :], timeout_sec, stdin=stdin)

    sandbox._run_docker_compose_command = compose

    async def exec_(command, timeout_sec, env=None, **kwargs):
        return await sandbox.exec(command, env=env, timeout_sec=timeout_sec, **kwargs)

    exec_.sandbox = sandbox  # type: ignore[attr-defined]
    return exec_


def _remote_docker(image: LocalImage) -> Exec:
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    return _docker(image, RemoteDockerSandbox)


def _daytona(image: LocalImage, *, compose: bool) -> Exec:
    pytest.importorskip("daytona")
    from benchflow.sandbox.daytona import (
        DaytonaSandbox,
        _DaytonaDinD,
        _DaytonaDirect,
        _load_daytona_sdk,
    )

    _load_daytona_sdk()
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox._persistent_env = {}
    sandbox.default_user = None
    sandbox.logger = logging.getLogger("test-daytona")
    sent: dict[str, str] = {}
    # The env the sandbox was created with, over the image's ENV: what
    # Daytona's toolbox starts each of its shells with.
    toolbox: dict[str, Any] = {"env": {}, "creates": [], "keeps_bash_env": False}

    def toolbox_shell() -> list[str]:
        # Daytona's GetShell: bash when the image has it, else sh, reading
        # its commands from stdin (session/create.go, process/execute.go).
        return ["bash" if (image.bin / "bash").exists() else "sh", "-s"]

    class Process:
        async def create_session(self, session_id):
            return None

        async def execute_session_command(self, session_id, request, timeout=None):
            sent["command"] = request.command
            return SimpleNamespace(cmd_id="cmd")

        async def exec(self, command, cwd=None, env=None, timeout=None):
            # process.exec: a new toolbox shell, the request's envs added.
            result = image.run(
                toolbox_shell(), timeout, {**toolbox["env"], **(env or {})}, command
            )
            return SimpleNamespace(
                exit_code=result.return_code, result=result.stdout + result.stderr
            )

    class FileSystem:
        # The toolbox's file API, as root: paths mapped like commands.
        async def upload_file(self, data, path, timeout=None):
            target = Path(image.local(path))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        async def delete_file(self, path, recursive=False):
            shutil.rmtree(image.local(path))

    async def create_sandbox(params):
        toolbox["creates"].append(params)
        toolbox["env"] = dict(params.env_vars or {})
        if toolbox["keeps_bash_env"]:
            toolbox["env"].pop("BASH_ENV", None)
        sandbox._sandbox = SimpleNamespace(
            process=Process(), fs=FileSystem(), delete=_no_op
        )

    sandbox._create_sandbox = create_sandbox

    async def poll(session_id, command_id, timeout_sec=None):
        # A session is a toolbox shell; the command line is what it reads.
        return image.run(
            toolbox_shell(), timeout_sec, toolbox["env"], sent["command"] + "\n"
        )

    sandbox._poll_response = poll

    async def unproven(route, files):
        # This transport runs as the test's own user, so no staging folder
        # here can be root's: the staged route's proof (run as root in
        # tests/test_exec_transport_contract.py) always passes here.
        return "true"

    sandbox._stage_proof = unproven
    if compose:
        strategy = _DaytonaDinD.__new__(_DaytonaDinD)
        strategy._env = sandbox

        async def compose_exec(
            parts, timeout_sec=None, cleanup_session=False, stdin_data=None
        ):
            # [..., service, "sh", "-c", command] run inside the service; an
            # env too large for one command on stdin.
            stdin = None if stdin_data is None else stdin_data.decode()
            return image.run(parts[-3:], timeout_sec, stdin=stdin)

        strategy._compose_exec = compose_exec
    else:
        strategy = _DaytonaDirect.__new__(_DaytonaDirect)
        strategy._env = sandbox
    sandbox._strategy = strategy

    async def exec_(command, timeout_sec, env=None, user=None, **kwargs):
        if not toolbox["creates"]:
            await _create_daytona(sandbox, compose=compose)
        return await sandbox.exec(
            command, env=env, timeout_sec=timeout_sec, user=user, **kwargs
        )

    exec_.toolbox = toolbox  # type: ignore[attr-defined]
    exec_.sandbox = sandbox  # type: ignore[attr-defined]
    return exec_


async def _no_op(*args, **kwargs):
    return None


async def _create_daytona(sandbox: Any, *, compose: bool) -> None:
    """Create the sandbox the way the strategy's ``start`` does."""
    from benchflow.sandbox.daytona import CreateSandboxFromSnapshotParams

    params = CreateSandboxFromSnapshotParams(snapshot="task-image")
    create = sandbox._create_sandbox
    if not compose:
        # Daytona direct. Without the fix this method is missing and the
        # sandbox keeps the image's env as it is.
        create = getattr(sandbox, "_create_sandbox_without_toolbox_bash_env", create)
    await create(params)


def _modal(image: LocalImage, monkeypatch) -> Exec:
    modal = pytest.importorskip("modal")
    from benchflow.sandbox.modal_impl import ModalSandbox

    # A secret is the env Modal puts in the process; keep it readable here.
    monkeypatch.setattr(modal.Secret, "from_dict", staticmethod(dict))

    sandbox = ModalSandbox.__new__(ModalSandbox)
    sandbox._persistent_env = {}
    sandbox.default_user = None

    def aio(fn):
        return SimpleNamespace(aio=fn)

    async def sandbox_exec(*argv, workdir=None, secrets=(), timeout=None):
        secret_env = {k: v for secret in secrets for k, v in secret.items()}
        result = image.run(list(argv), timeout, secret_env)

        async def stdout():
            return result.stdout

        async def stderr():
            return result.stderr

        async def wait():
            return result.return_code

        return SimpleNamespace(
            stdout=SimpleNamespace(read=aio(stdout)),
            stderr=SimpleNamespace(read=aio(stderr)),
            wait=aio(wait),
        )

    sandbox._sandbox = SimpleNamespace(exec=aio(sandbox_exec))

    async def exec_(command, timeout_sec, env=None, user=None, **kwargs):
        return await sandbox.exec(
            command, env=env, timeout_sec=timeout_sec, user=user, **kwargs
        )

    exec_.sandbox = sandbox  # type: ignore[attr-defined]
    return exec_


def _apple_container(image: LocalImage, monkeypatch) -> Exec:
    from benchflow.sandbox import apple_container
    from benchflow.sandbox.apple_container import AppleContainerSandbox

    sandbox = AppleContainerSandbox.__new__(AppleContainerSandbox)
    sandbox._persistent_env = {}
    sandbox.default_user = None
    sandbox._container_name = "fake"

    async def run_cli(*args, timeout=None, stdin_data=None):
        # ("exec", ..., name, "sh", "-c", command); the env, if any, on stdin
        stdin = None if stdin_data is None else stdin_data.decode()
        return image.run(list(args[args.index("fake") + 1 :]), timeout, stdin=stdin)

    monkeypatch.setattr(apple_container, "_run_cli", run_cli)

    async def exec_(command, timeout_sec, env=None, **kwargs):
        return await sandbox.exec(command, env=env, timeout_sec=timeout_sec, **kwargs)

    exec_.sandbox = sandbox  # type: ignore[attr-defined]
    return exec_


def _agentcore(image: LocalImage, env_dir: Path) -> Exec:
    from benchflow.sandbox.agentcore import AgentCoreSandbox

    sandbox = AgentCoreSandbox.__new__(AgentCoreSandbox)
    sandbox._persistent_env = {}
    sandbox.default_user = None
    sandbox.runtime_arn = "arn:fake"
    sandbox.runtime_session_id = "session"

    def invoke(payload, timeout):
        # The runtime runs the payload line, "/bin/bash -c ...", in a shell.
        return image.run(["sh", "-c", payload])

    sandbox._invoke_command = invoke

    class Sealed:
        async def stage_env(self, env, owner=None):
            path = env_dir / f"env-{len(list(env_dir.iterdir()))}"
            path.write_text("".join(f"{k}={shlex.quote(v)}\n" for k, v in env.items()))
            return str(path)

    sandbox._sealed = Sealed()

    async def exec_(command, timeout_sec, env=None, user=None, **kwargs):
        return await sandbox.exec(
            command, env=env, timeout_sec=timeout_sec, user=user, **kwargs
        )

    exec_.sandbox = sandbox  # type: ignore[attr-defined]
    return exec_


# The kinds whose exec calls through ``sh -c`` (bash optional in the image)
# and those that call through ``bash -c`` (bash required to run anything).
SH_KINDS = ("docker", "remote-docker", "daytona-dind", "apple-container")
BASH_KINDS = ("daytona", "modal", "agentcore")


def _backend(kind: str, image: LocalImage, monkeypatch) -> Exec:
    if kind == "docker":
        return _docker(image)
    if kind == "remote-docker":
        return _remote_docker(image)
    if kind == "daytona":
        return _daytona(image, compose=False)
    if kind == "daytona-dind":
        return _daytona(image, compose=True)
    if kind == "modal":
        return _modal(image, monkeypatch)
    if kind == "apple-container":
        return _apple_container(image, monkeypatch)
    if kind == "agentcore":
        env_dir = image.tests.parent / "agentcore-env"
        env_dir.mkdir(exist_ok=True)
        return _agentcore(image, env_dir)
    raise AssertionError(kind)


class VerifierSandbox:
    """What the Verifier talks to: uploads copied here, execs sent through *kind*."""

    is_mounted = True

    def __init__(self, image: LocalImage, exec_: Exec):
        self.image = image
        self._exec = exec_
        self.commands: list[str] = []

    @property
    def supports_system_shell(self) -> bool:
        return self._exec.sandbox.supports_system_shell  # type: ignore[attr-defined]

    async def upload_dir(self, source_dir, target_dir, **kwargs):
        shutil.copytree(source_dir, self.image.tests)

    async def exec(self, command, env=None, user=None, timeout_sec=None, **kwargs):
        self.commands.append(command)
        extra = {"system_shell": True} if kwargs.get("system_shell") else {}
        return await self._exec(command, timeout_sec, env=env, **extra)


def _task(tmp_path: Path, script: str) -> MagicMock:
    tests_dir = tmp_path / "task" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text(script)
    task = MagicMock()
    task.name = "task"
    task.task_dir = tests_dir.parent
    task.paths.task_dir = tests_dir.parent
    task.paths.tests_dir = tests_dir
    task.paths.test_path = tests_dir / "test.sh"
    task.paths.uses_native_verifier_dir = False
    task.config = TaskConfig.model_validate_toml(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60.0\n'
    )
    return task


async def _verify(
    tmp_path: Path, script: str, kind: str, monkeypatch, *, with_bash: bool = True
) -> tuple[dict | None, VerifierSandbox]:
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image = LocalImage(tmp_path, paths, with_bash=with_bash)
    sandbox = VerifierSandbox(image, _backend(kind, image, monkeypatch))
    result = await Verifier(
        task=_task(tmp_path, script), rollout_paths=paths, sandbox=sandbox
    ).verify()
    return result.rewards, sandbox


def _stdout(tmp_path: Path) -> str:
    return (tmp_path / "rollout" / "verifier" / "test-stdout.txt").read_text()


@pytest.fixture(autouse=True)
def _needs_dash():
    # The bug is dash's: where the host has no dash, ``sh`` may be bash and
    # the base would pass too. Ubuntu (CI, the build VMs) always has dash.
    if _DASH is None:
        pytest.skip("needs dash, the sh of Debian and Ubuntu images")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", SH_KINDS + BASH_KINDS)
async def test_late_shebang_bash_script_runs_under_bash(tmp_path, kind, monkeypatch):
    """A test.sh with ``#!/bin/bash`` on line 2 and bash-only lines is scored.

    Before the fix the ``sh -c`` kinds ran it under dash: ``set: Illegal
    option -o pipefail`` and no reward file, so the trial was a verifier crash.
    """
    rewards, _ = await _verify(tmp_path, LATE_SHEBANG_BASH, kind, monkeypatch)

    assert rewards == {"reward": 1.0}, _stdout(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", SH_KINDS + BASH_KINDS)
async def test_script_without_shebang_runs_under_bash(tmp_path, kind, monkeypatch):
    rewards, _ = await _verify(tmp_path, BASH_NO_SHEBANG, kind, monkeypatch)

    assert rewards == {"reward": 1.0}, _stdout(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", SH_KINDS + BASH_KINDS)
async def test_first_line_interpreter_shebang_still_wins(tmp_path, kind, monkeypatch):
    """``#!/usr/bin/env python3`` on line 1 runs Python, never a shell."""
    if shutil.which("python3", path="/usr/local/bin:/usr/bin:/bin") is None:
        pytest.skip("needs a system python3")
    rewards, _ = await _verify(tmp_path, PYTHON_SHEBANG, kind, monkeypatch)

    assert rewards == {"reward": 1.0}, _stdout(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", SH_KINDS)
async def test_image_without_bash_runs_a_posix_script_under_sh(
    tmp_path, kind, monkeypatch
):
    """No bash in the image: a POSIX test.sh without a shebang runs under sh.

    Only the ``sh -c`` kinds can run an image without bash at all.
    """
    rewards, _ = await _verify(
        tmp_path, POSIX_NO_SHEBANG, kind, monkeypatch, with_bash=False
    )

    assert rewards == {"reward": 1.0}, _stdout(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", SH_KINDS)
async def test_image_with_bash_runs_a_script_without_shebang_under_bash(
    tmp_path, kind, monkeypatch
):
    """The same POSIX script scores 0 when bash ran it: bash is preferred."""
    rewards, _ = await _verify(tmp_path, POSIX_NO_SHEBANG, kind, monkeypatch)

    assert rewards == {"reward": 0.0}, _stdout(tmp_path)


@pytest.mark.asyncio
async def test_bash_script_in_an_image_without_bash_is_still_a_crash(
    tmp_path, monkeypatch
):
    """Without bash nothing can run a bash script: unscored, as before."""
    with pytest.raises(RewardFileNotFoundError):
        await _verify(tmp_path, BASH_NO_SHEBANG, "docker", monkeypatch, with_bash=False)


@pytest.mark.asyncio
async def test_exit_status_and_output_capture_are_kept(tmp_path, monkeypatch):
    """test.sh's own exit status and output reach the verifier as before."""
    script = '#!/bin/sh\necho checked\necho 0 > "$VERIFIER_DIR/reward.txt"\nexit 3\n'
    rewards, sandbox = await _verify(tmp_path, script, "docker", monkeypatch)

    assert rewards == {"reward": 0.0}
    assert _stdout(tmp_path) == "checked\n"
    (command,) = [c for c in sandbox.commands if "test-stdout.txt" in c]
    assert command.startswith("umask 022 && { ")
    assert command.endswith(
        " /tests/test.sh\n}; } > /logs/verifier/test-stdout.txt 2>&1"
    )


# --- the helper on its own ----------------------------------------------------


def _shell(command: str, bin_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["sh", "-c", command, *args],
        env={"PATH": str(bin_dir)},
        capture_output=True,
        text=True,
        timeout=30,
        executable=str(bin_dir / "sh"),
    )


def test_a_failed_cd_never_starts_the_script(tmp_path):
    """``cd dir && <run>`` keeps the script from starting when the cd fails."""
    from benchflow.task.verifier_script_strategy import script_run_command

    bin_dir = _bin_dir(tmp_path, with_bash=True)
    marker = tmp_path / "ran"
    script = tmp_path / "s.sh"
    script.write_text(f"touch {marker}\n")
    command = f"cd {tmp_path / 'missing'} && " + script_run_command(str(script))

    result = _shell(command, bin_dir)

    assert result.returncode != 0
    assert not marker.exists()


def test_arguments_and_the_rest_of_the_command_are_kept(tmp_path):
    from benchflow.task.verifier_script_strategy import script_run_command

    bin_dir = _bin_dir(tmp_path, with_bash=True)
    script = tmp_path / "s.sh"
    script.write_text('printf "%s|" "$@"\n')
    command = script_run_command(str(script), "'a b' c && printf after # note")

    result = _shell(command, bin_dir)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "a b|c|after"


def test_a_command_that_ends_in_a_background_job_still_parses(tmp_path):
    from benchflow.task.verifier_script_strategy import script_run_command

    bin_dir = _bin_dir(tmp_path, with_bash=True)
    script = tmp_path / "s.sh"
    script.write_text(f"touch {tmp_path / 'ran'}\n")
    command = script_run_command(str(script), "&") + " && wait"

    result = _shell(command, bin_dir)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "ran").exists()


def test_a_missing_script_fails_without_a_reward(tmp_path):
    from benchflow.task.verifier_script_strategy import script_run_command

    bin_dir = _bin_dir(tmp_path, with_bash=True)

    result = _shell(script_run_command(str(tmp_path / "absent.sh")), bin_dir)

    assert result.returncode == 127


def _elf_program(folder: Path) -> Path:
    """A copy of the host's ``true`` (an ELF binary), under its own name."""
    source = shutil.which("true", path="/usr/bin:/bin")
    assert source is not None
    program = folder / "true"
    shutil.copy(os.path.realpath(source), program)
    program.chmod(0o755)
    assert program.read_bytes()[:4] == b"\x7fELF"
    return program


def test_a_compiled_program_starts_by_path_not_under_bash(tmp_path):
    """An ELF binary is run by the kernel; ``bash ./grader`` would refuse it."""
    from benchflow.task.verifier_script_strategy import script_run_command

    bin_dir = _bin_dir(tmp_path, with_bash=True)
    folder = tmp_path / "verifier"
    folder.mkdir()
    program = _elf_program(folder)

    result = _shell(script_run_command(str(program)), bin_dir)

    assert result.returncode == 0, result.stderr


def test_strategy_that_starts_a_compiled_program_still_runs_it(tmp_path):
    """A verifier.md ``./true`` strategy whose program is a binary runs it."""
    bin_dir = _bin_dir(tmp_path, with_bash=True)
    folder = tmp_path / "verifier"
    folder.mkdir()
    _elf_program(folder)
    command = _script_strategy_command(_strategy("./true"), PurePosixPath(folder))

    result = _shell(command, bin_dir)

    assert result.returncode == 0, result.stderr


def _strategy(command: str) -> VerifierStrategy:
    return VerifierStrategy(
        name="deterministic", type="script", config={"command": command}
    )


@pytest.mark.parametrize(
    "command",
    ["./test.sh", "./test.sh --fast", "scripts/run.sh a b && echo done"],
)
def test_strategy_that_starts_a_package_script_runs_it_under_its_shell(command):
    """Imported tasks get ``command: ./test.sh``; it starts like the default."""
    from benchflow.task.verifier_script_strategy import script_run_command

    built = _script_strategy_command(_strategy(command), PurePosixPath("/verifier"))
    first, _, rest = command.partition(" ")

    assert built == "cd /verifier && " + script_run_command(first, rest)


@pytest.mark.parametrize(
    "command",
    ["python verify.py", "bash test.sh", "'./test.sh'", "A=b/c ./test.sh"],
)
def test_other_strategy_commands_run_as_written(command):
    built = _script_strategy_command(_strategy(command), PurePosixPath("/verifier"))

    assert built == f"cd /verifier && {command}"


def test_strategy_script_with_late_shebang_runs_under_bash(tmp_path):
    """A verifier.md ``./test.sh`` strategy behaves like the default test.sh."""
    bin_dir = _bin_dir(tmp_path, with_bash=True)
    folder = tmp_path / "verifier"
    folder.mkdir()
    (folder / "test.sh").write_text(LATE_SHEBANG_BASH)
    (folder / "test.sh").chmod(0o755)
    command = _script_strategy_command(_strategy("./test.sh"), PurePosixPath(folder))

    result = subprocess.run(
        [str(bin_dir / "sh"), "-c", command],
        env={"PATH": str(bin_dir), "VERIFIER_DIR": str(folder)},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert (folder / "reward.txt").read_text() == "1\n"


# --- an agent-written BASH_ENV file never runs in the verifier ----------------
#
# Every non-interactive bash runs the file BASH_ENV names before its script.
# An image or task whose BASH_ENV named a file in the agent's workspace let
# the agent's file run as the verifier user (root here), write reward 1 and
# exit before the trusted test.sh ran: a false pass. The verifier env comes
# from lockdown's real ``_build_verifier_env`` (its probes of the image's
# Pythons replaced, they need a container), so the pin is what is tested.

# The trusted test.sh, which scores 0, without a shebang and with the usual one.
ZERO_SCRIPTS = {
    "no-shebang": 'printf "0\\n" > "$VERIFIER_DIR/reward.txt"\n',
    "bash-shebang": '#!/bin/bash\nprintf "0\\n" > "$VERIFIER_DIR/reward.txt"\n',
}


def _agent_startup_file(tmp_path: Path) -> tuple[Path, Path, Path]:
    """The agent's workspace, a startup file it wrote there, and its marker."""
    workspace = tmp_path / "app"
    workspace.mkdir()
    marker = tmp_path / "startup-ran"
    startup = workspace / "verifier-startup.sh"
    startup.write_text(
        f"touch {marker}\nprintf '1\\n' > \"$VERIFIER_DIR/reward.txt\"\nexit 0\n"
    )
    return workspace, startup, marker


def _drop_tools(bin_dir: Path, prefixes: tuple[str, ...]) -> None:
    """Take the tools whose names start with one of *prefixes* off the image."""
    for tool in bin_dir.iterdir():
        if tool.name.startswith(prefixes):
            tool.unlink()


def _without_image_probes(
    monkeypatch, image: LocalImage, path_extras: tuple[str, ...] = ()
) -> None:
    """*path_extras* are folders the verifier's checked PATH keeps first."""
    from benchflow._utils.trusted_path import TrustedPathReport
    from benchflow.sandbox import lockdown

    async def path(env, sandbox_user, workspace, **kwargs):
        return TrustedPathReport(path=":".join([*path_extras, str(image.bin)]))

    async def nothing(*args, **kwargs):
        return {}

    async def empty(*args, **kwargs):
        return ""

    async def no_uid(*args, **kwargs):
        return None

    async def no_plugins(*args, **kwargs):
        return [], []

    async def guard(env, sandbox_user, workspace, flags, **kwargs):
        return "", flags, True, ()

    monkeypatch.setattr(lockdown, "_judge_verifier_path", path)
    monkeypatch.setattr(lockdown, "_trusted_verifier_pythonpath", empty)
    monkeypatch.setattr(lockdown, "_distro_pip_env", nothing)
    monkeypatch.setattr(lockdown, "_isolate_verifier_tool_state", nothing)
    monkeypatch.setattr(lockdown, "_discover_pytest_plugin_flags", empty)
    monkeypatch.setattr(lockdown, "_discover_pytest_plugins", no_plugins)
    monkeypatch.setattr(lockdown, "_verifier_uid", no_uid)
    monkeypatch.setattr(lockdown, "_install_pytest_plugin_guard", guard)
    # The BASH_ENV check takes its tools from the image's system folders.
    monkeypatch.setattr(lockdown, "_BASH_ENV_CHECK_PATH", str(image.bin), raising=False)


async def _verify_hardened(
    tmp_path: Path,
    script: str,
    kind: str,
    monkeypatch,
    *,
    image_env: dict[str, str],
    task_env: dict[str, str] | None = None,
    without: tuple[str, ...] = (),
    path_extras: tuple[str, ...] = (),
) -> tuple[dict | None, dict[str, str]]:
    """*without* lists name prefixes of tools the image does not have."""
    from benchflow.sandbox.lockdown import _build_verifier_env

    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image = LocalImage(tmp_path, paths, with_bash=True, image_env=image_env)
    _drop_tools(image.bin, without)
    sandbox = VerifierSandbox(image, _backend(kind, image, monkeypatch))
    _test_inspector(sandbox, tmp_path / "inspector", image.bin)
    _without_image_probes(monkeypatch, image, path_extras)
    task = _task(tmp_path, script)
    task.config.verifier.env = dict(task_env or {})
    task.config.verifier.env = await _build_verifier_env(
        sandbox, task, "agent", str(tmp_path / "app")
    )
    result = await Verifier(task=task, rollout_paths=paths, sandbox=sandbox).verify()
    return result.rewards, task.config.verifier.env


@pytest.mark.asyncio
@pytest.mark.parametrize("script", sorted(ZERO_SCRIPTS))
@pytest.mark.parametrize("kind", SH_KINDS + BASH_KINDS)
async def test_image_bash_env_in_the_workspace_never_runs(
    tmp_path, kind, script, monkeypatch
):
    """The image's BASH_ENV names the agent's file: test.sh scores 0 anyway."""
    _, startup, marker = _agent_startup_file(tmp_path)

    rewards, env = await _verify_hardened(
        tmp_path,
        ZERO_SCRIPTS[script],
        kind,
        monkeypatch,
        image_env={"BASH_ENV": str(startup)},
    )

    assert rewards == {"reward": 0.0}, _stdout(tmp_path)
    assert not marker.exists()
    assert env["BASH_ENV"] == "" and env["ENV"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("script", sorted(ZERO_SCRIPTS))
@pytest.mark.parametrize("kind", SH_KINDS + BASH_KINDS)
async def test_task_bash_env_in_the_workspace_never_runs(
    tmp_path, kind, script, monkeypatch
):
    """A task's ``[verifier.env]`` BASH_ENV in the workspace is emptied."""
    if shutil.which("python3", path="/usr/local/bin:/usr/bin:/bin") is None:
        pytest.skip("needs a system python3 for the check")
    _, startup, marker = _agent_startup_file(tmp_path)

    rewards, env = await _verify_hardened(
        tmp_path,
        ZERO_SCRIPTS[script],
        kind,
        monkeypatch,
        image_env={},
        task_env={"BASH_ENV": str(startup)},
    )

    assert rewards == {"reward": 0.0}, _stdout(tmp_path)
    assert not marker.exists()
    assert env["BASH_ENV"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", BASH_KINDS)
async def test_bash_backed_exec_never_runs_the_image_bash_env(
    tmp_path, kind, monkeypatch
):
    """Hardening's own root execs (no env) skip the image's BASH_ENV file.

    Daytona, Modal and AgentCore wrap every command in ``bash -c``: the file
    ran as root on each exec after the agent's turn, before verification.
    """
    _, startup, marker = _agent_startup_file(tmp_path)
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image = LocalImage(
        tmp_path, paths, with_bash=True, image_env={"BASH_ENV": str(startup)}
    )
    exec_ = _backend(kind, image, monkeypatch)

    result = await exec_("echo ran", None)

    assert result.return_code == 0, result.stderr
    assert result.stdout == "ran\n"
    assert not marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", SH_KINDS + BASH_KINDS)
async def test_a_bash_env_the_caller_passes_still_reaches_the_command(
    tmp_path, kind, monkeypatch
):
    """The verifier's own BASH_ENV (a trusted file) is still read by bash."""
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image = LocalImage(tmp_path, paths, with_bash=True)
    exec_ = _backend(kind, image, monkeypatch)
    trusted = tmp_path / "activate.sh"
    trusted.write_text("export ACTIVATED=yes\n")

    # The verifier's BASH_ENV reaches test.sh on the task route
    # (verifier_core); root's plumbing takes none from its caller.
    with as_task_command(None):
        result = await exec_(
            'bash -c \'printf "%s\\n" "${ACTIVATED:-no}"\'',
            None,
            env={"BASH_ENV": str(trusted)},
        )

    assert result.return_code == 0, result.stderr
    assert result.stdout == "yes\n"


class _HostShell:
    """The check's exec, run here through ``sh -c`` with the env it is given
    and root's PATH as the exec route pins it (*path*; a caller never sets
    it)."""

    path: str | None = None

    async def exec(self, command, env=None, user=None, timeout_sec=None, **kwargs):
        route_path = self.path or os.environ.get("PATH", "/usr/bin:/bin")
        result = subprocess.run(
            ["sh", "-c", command],
            env={**(env or {}), "PATH": route_path},
            capture_output=True,
            text=True,
            timeout=60,
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )


# The image's tools the check can use: both its POSIX tools and python3, the
# shell tools alone (a shell-only image), python3 alone, or neither.
CHECK_TOOLS = {
    "all": (),
    "no-python": ("python",),
    "python-only": ("readlink", "stat"),
    "neither": ("python", "readlink", "stat"),
}


async def _check_bash_env(
    tmp_path: Path, value: str, blocked: tuple[str, ...] = (), tools: str = "all"
) -> str:
    from benchflow.sandbox import lockdown

    root = tmp_path / f"check-{tools}"
    bin_dir = root / "bin-bash"
    if not bin_dir.exists():
        root.mkdir()
        _drop_tools(_bin_dir(root, with_bash=True), CHECK_TOOLS[tools])
    task = MagicMock()
    task.config.verifier.env = {"BASH_ENV": value}
    task.config.verifier.service = "main"
    shell = _HostShell()
    shell.path = str(bin_dir)
    _test_inspector(shell, root / "inspector", bin_dir)
    return await lockdown._trusted_verifier_bash_env(shell, task, blocked)


def _test_inspector(env, folder: Path, bin_dir: Path) -> None:
    """Model the protected pre-agent copies in the simulated image transport."""
    if not folder.exists():
        folder.mkdir(mode=0o700)
        for name in ("stat", "readlink", "python3"):
            source = bin_dir / name
            if source.exists():
                shutil.copy2(source.resolve(), folder / name)
                (folder / name).chmod(0o700)
    env._benchflow_bash_env_inspector = str(folder)


def _needs_root_owned(path: str) -> None:
    if not os.path.isfile(path) or os.stat(path).st_uid != 0 or os.getuid() == 0:
        pytest.skip(f"needs a root-owned {path} and a non-root test user")


def _needs_python3() -> None:
    if shutil.which("python3", path="/usr/local/bin:/usr/bin:/bin") is None:
        pytest.skip("needs a system python3")


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", ["all", "no-python", "python-only"])
async def test_the_bash_env_check_keeps_only_root_owned_files(tmp_path, tools):
    """Kept: a root-owned file in root-only folders. Emptied: anything else.

    With the image's POSIX tools alone, and with python3 alone.
    """
    _needs_root_owned("/etc/passwd")
    if tools == "python-only":
        _needs_python3()
    own = tmp_path / "startup.sh"
    own.write_text("true\n")

    async def check(value, blocked=()):
        return await _check_bash_env(tmp_path, value, blocked, tools)

    assert await check("/etc/passwd") == "/etc/passwd"
    assert await check("/etc/passwd", ("/etc",)) == ""
    assert await check(str(own)) == ""
    assert await check("/etc/$HOME") == ""
    assert await check("relative.sh") == ""
    assert await check("/nonexistent/startup.sh") == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", ["all", "no-python", "python-only"])
async def test_the_bash_env_check_answers_the_file_a_link_resolves_to(tmp_path, tools):
    """A link is kept as the root-owned file it names, never as the link.

    Bash would resolve the link again from its own working folder, and a
    ``/proc/self/cwd`` link names another file there.
    """
    _needs_root_owned("/etc/passwd")
    if tools == "python-only":
        _needs_python3()
    own = tmp_path / "startup.sh"
    own.write_text("true\n")
    links = tmp_path / "links"
    links.mkdir()
    (links / "plain").symlink_to("/etc/passwd")
    (links / "by-cwd").symlink_to("/proc/self/cwd/etc/passwd")
    (links / "to-own").symlink_to(own)

    for name in ("plain", "by-cwd"):
        value = str(links / name)
        assert await _check_bash_env(tmp_path, value, (), tools) == "/etc/passwd"
    assert await _check_bash_env(tmp_path, str(links / "to-own"), (), tools) == ""


@pytest.mark.asyncio
async def test_the_bash_env_check_fails_closed_without_tools(tmp_path):
    """No readlink or stat and no python3: the value cannot be checked."""
    _needs_root_owned("/etc/passwd")

    assert await _check_bash_env(tmp_path, "/etc/passwd", (), "neither") == ""


# A task's BASH_ENV is a link outside every agent-writable prefix (the review
# used /dev/shm; tmp_path stands for it here) to ``/proc/self/cwd/etc/...``.
# Checked from ``/`` it is the root-owned file; read by the verifier's bash in
# the workspace it was the agent's copy, which scored 1 and exited.
# The verifier now gets the file that was checked.
@pytest.mark.asyncio
@pytest.mark.parametrize("script", sorted(ZERO_SCRIPTS))
@pytest.mark.parametrize("kind", SH_KINDS + BASH_KINDS)
async def test_a_bash_env_link_runs_the_file_that_was_checked(
    tmp_path, kind, script, monkeypatch
):
    from benchflow.sandbox import lockdown

    _needs_root_owned("/etc/bash.bashrc")
    monkeypatch.setattr(lockdown, "_RUNTIME_PATH_PREFIXES", ("/logs", "/testbed"))
    workspace = tmp_path / "app"
    (workspace / "etc").mkdir(parents=True)
    marker = tmp_path / "startup-ran"
    (workspace / "etc" / "bash.bashrc").write_text(
        f"touch {marker}\nprintf '1\\n' > \"$VERIFIER_DIR/reward.txt\"\nexit 0\n"
    )
    link = tmp_path / "shm" / "start.sh"
    link.parent.mkdir()
    link.symlink_to("/proc/self/cwd/etc/bash.bashrc")
    # The verifier's commands start in the workspace, as Docker's WORKDIR.
    monkeypatch.chdir(workspace)

    rewards, env = await _verify_hardened(
        tmp_path,
        ZERO_SCRIPTS[script],
        kind,
        monkeypatch,
        image_env={},
        task_env={"BASH_ENV": str(link)},
    )

    assert rewards == {"reward": 0.0}, _stdout(tmp_path)
    assert env["BASH_ENV"] == "/etc/bash.bashrc"
    assert not marker.exists()


# A shell-only image (bash, no python3) keeps its root-owned activation file:
# the check needs only the shell, ``readlink`` and ``stat``.
ACTIVATED_SCRIPTS = {
    "no-shebang": (
        'if [ "$BASH_ENV" = /etc/bash.bashrc ]; then r=1; else r=0; fi\n'
        'printf "%s\\n" "$r" > "$VERIFIER_DIR/reward.txt"\n'
    ),
}
ACTIVATED_SCRIPTS["bash-shebang"] = "#!/bin/bash\n" + ACTIVATED_SCRIPTS["no-shebang"]


@pytest.mark.asyncio
@pytest.mark.parametrize("script", sorted(ACTIVATED_SCRIPTS))
@pytest.mark.parametrize("kind", SH_KINDS + BASH_KINDS)
async def test_a_root_owned_bash_env_is_kept_in_an_image_without_python(
    tmp_path, kind, script, monkeypatch
):
    _needs_root_owned("/etc/bash.bashrc")

    rewards, env = await _verify_hardened(
        tmp_path,
        ACTIVATED_SCRIPTS[script],
        kind,
        monkeypatch,
        image_env={"BASH_ENV": "/etc/bash.bashrc"},
        without=("python",),
    )

    assert rewards == {"reward": 1.0}, _stdout(tmp_path)
    assert env["BASH_ENV"] == "/etc/bash.bashrc"


# --- the image's BASH_ENV still reaches the command, as on Docker ------------
#
# The wrapping shells skip the image's BASH_ENV file, but the command and
# every bash it starts still read it: an image that activates conda or nvm
# through a root-owned BASH_ENV works on every sandbox kind (V7-R1-A-1).
# Docker's ``sh -c`` never reads the file and passes the variable on; the
# bash-backed kinds hide it from their wrapping shells and give it back to the
# command. The file counts its runs: exactly one, by the command's own bash.

PRINT_ACTIVATION = (
    'bash -c \'printf "%s|%s|%s\\n" "${MYTOOL:-missing}" '
    '"${BASH_ENV:-unset}" "${_BF_IMAGE_BASH_ENV-none}"\''
)


def _image_activation(tmp_path: Path) -> tuple[Path, Path]:
    """A startup file the image's BASH_ENV names, and the file it counts in."""
    count = tmp_path / "activations"
    activate = tmp_path / "image-activate.sh"
    activate.write_text(f"printf 'x\\n' >> {count}\nexport MYTOOL=ready\n")
    return activate, count


def _fake_su(bin_dir: Path) -> None:
    """``su USER -s SHELL -c COMMAND`` as the test's own user, env kept."""
    su = bin_dir / "su"
    if su.is_symlink() or su.exists():
        su.unlink()
    su.write_text(
        '#!/bin/sh\n[ "$2" = -s ] && [ "$4" = -c ] || exit 99\nexec "$3" -c "$5"\n'
    )
    su.chmod(0o755)


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_env", [None, {"FOO": "bar"}], ids=["no-env", "env"])
@pytest.mark.parametrize(
    ("kind", "user"),
    [(kind, None) for kind in SH_KINDS + BASH_KINDS]
    + [(kind, "agent") for kind in BASH_KINDS],
)
async def test_the_image_bash_env_still_reaches_the_command(
    tmp_path, kind, user, caller_env, monkeypatch
):
    """A bash the command starts reads the image's file once; no wrapper does."""
    activate, count = _image_activation(tmp_path)
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image = LocalImage(
        tmp_path, paths, with_bash=True, image_env={"BASH_ENV": str(activate)}
    )
    _fake_su(image.bin)
    exec_ = _backend(kind, image, monkeypatch)
    kwargs = {"user": user} if user else {}

    # A command of the task's own (C8's task route): BenchFlow's root
    # plumbing never reads the image's file (tests/test_c8_loader_routes.py).
    with as_task_command():
        result = await exec_(PRINT_ACTIVATION, None, env=caller_env, **kwargs)

    assert result.return_code == 0, result.stderr
    assert result.stdout == f"ready|{activate}|none\n"
    assert count.read_text() == "x\n"


@pytest.mark.skipif(
    shutil.which("python3", path="/usr/local/bin:/usr/bin:/bin") is None
    or not os.path.isfile("/etc/bash.bashrc"),
    reason="needs a system python3 and a root-owned /etc/bash.bashrc",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("kind", SH_KINDS + BASH_KINDS)
async def test_a_root_owned_image_bash_env_is_kept_for_the_verifier(
    tmp_path, kind, monkeypatch
):
    """The image's BASH_ENV, a root-owned file in root-only folders, is kept.

    As PATH keeps the image's trusted entries (V7-R1-A-2): a ``#!/bin/bash``
    test.sh that needs the env it activates still gets it.
    """
    if os.stat("/etc/bash.bashrc").st_uid != 0:
        pytest.skip("needs a root-owned /etc/bash.bashrc")
    script = (
        '#!/bin/bash\n[ "$BASH_ENV" = /etc/bash.bashrc ] && r=1 || r=0\n'
        'printf "%s\\n" "$r" > "$VERIFIER_DIR/reward.txt"\n'
    )

    rewards, env = await _verify_hardened(
        tmp_path,
        script,
        kind,
        monkeypatch,
        image_env={"BASH_ENV": "/etc/bash.bashrc"},
    )

    assert env["BASH_ENV"] == "/etc/bash.bashrc"
    assert rewards == {"reward": 1.0}, _stdout(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("docker", "daytona"))
async def test_a_task_that_empties_bash_env_keeps_it_empty(tmp_path, kind, monkeypatch):
    """A task's own ``BASH_ENV = ""`` wins over the image's, trusted or not."""
    _, env = await _verify_hardened(
        tmp_path,
        ZERO_SCRIPTS["bash-shebang"],
        kind,
        monkeypatch,
        image_env={"BASH_ENV": "/etc/bash.bashrc"},
        task_env={"BASH_ENV": ""},
    )

    assert env["BASH_ENV"] == ""


# --- Daytona's own shells never read the image's BASH_ENV file --------------
#
# Daytona's toolbox runs each session command, and each ``process.exec``,
# through a shell it starts with the sandbox's env (the image's ENV included),
# and that shell is bash, reading its commands from stdin, when the image has
# bash. Such a bash runs the BASH_ENV file before it reads anything BenchFlow
# sends, so the agent's file ran as root before every hardening exec and
# test.sh on Daytona direct (V7-R2-A-1). The sandbox is now created again
# with BASH_ENV empty and the image's value carried for the commands.


@pytest.mark.asyncio
async def test_daytona_toolbox_shells_never_read_the_image_bash_env(
    tmp_path, monkeypatch
):
    """Neither a session command nor a process.exec runs the agent's file."""
    _, startup, marker = _agent_startup_file(tmp_path)
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image = LocalImage(
        tmp_path, paths, with_bash=True, image_env={"BASH_ENV": str(startup)}
    )
    exec_ = _backend("daytona", image, monkeypatch)

    # The task's own command gets the value back (C8's task route).
    with as_task_command():
        result = await exec_('printf "%s\\n" "$BASH_ENV"', None)
    direct = await exec_.sandbox._sandbox.process.exec("echo direct")

    assert result.return_code == 0, result.stderr
    assert result.stdout == f"{startup}\n"
    assert direct.exit_code == 0 and direct.result == "direct\n"
    assert not marker.exists()
    creates = exec_.toolbox["creates"]
    # Created again with every loader and start-up variable empty (C8).
    assert [params.env_vars for params in creates] == [
        None,
        {
            **dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, ""),
            "_BF_IMAGE_BASH_ENV": str(startup),
        },
    ]
    assert creates[1].snapshot == "task-image"


@pytest.mark.asyncio
async def test_daytona_image_without_bash_env_is_created_once(tmp_path, monkeypatch):
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image = LocalImage(tmp_path, paths, with_bash=True)
    exec_ = _backend("daytona", image, monkeypatch)

    result = await exec_("echo ran", None)

    assert result.stdout == "ran\n"
    assert [params.env_vars for params in exec_.toolbox["creates"]] == [None]


@pytest.mark.asyncio
async def test_daytona_that_keeps_the_image_bash_env_is_not_used(tmp_path, monkeypatch):
    """If the create-time env cannot empty BASH_ENV, start fails, unscored."""
    _, startup, marker = _agent_startup_file(tmp_path)
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image = LocalImage(
        tmp_path, paths, with_bash=True, image_env={"BASH_ENV": str(startup)}
    )
    exec_ = _backend("daytona", image, monkeypatch)
    exec_.toolbox["keeps_bash_env"] = True

    with pytest.raises(RuntimeError, match="did not clear toolbox startup inputs"):
        await exec_("echo ran", None)
    assert not marker.exists()


# --- taskmd's offline check never runs the agent's BASH_ENV file -------------
#
# ``take_offline`` runs as root after the agent's turn in the shared verifier
# sandbox and, without iptables, starts bash to try the network. With the
# image's BASH_ENV naming the agent's file, that bash ran the file as root,
# and a file ending in ``exit 1`` made both connects fail: an online
# container was reported offline and scored with the network on (V7-R2-B-1).


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", SH_KINDS + BASH_KINDS)
async def test_offline_check_never_runs_the_agent_bash_env_file(
    tmp_path, kind, monkeypatch
):
    """Guard V7-R2-B-1 and 4038ec63's refusal of unprotected root shells."""
    from benchflow.taskmd.verify import TaskMdVerifierError, take_offline

    try:
        interfaces = Path("/proc/net/dev").read_text().splitlines()[2:]
    except OSError:
        pytest.skip("needs /proc/net/dev")
    if all(line.split(":")[0].strip() == "lo" for line in interfaces):
        pytest.skip("needs a network interface besides lo")
    workspace = tmp_path / "app"
    workspace.mkdir()
    marker = tmp_path / "startup-ran"
    startup = workspace / "startup.sh"
    startup.write_text(f"touch {marker}\nexit 1\n")
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image = LocalImage(
        tmp_path, paths, with_bash=True, image_env={"BASH_ENV": str(startup)}
    )
    for tool in ("iptables", "ip6tables"):
        (image.bin / tool).unlink(missing_ok=True)
    sandbox = VerifierSandbox(image, _backend(kind, image, monkeypatch))
    # "The internet" here is a port on this host that accepts connections.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        probe = f"127.0.0.1/{listener.getsockname()[1]}"

        expected = "online" if sandbox.supports_system_shell else "protected root shell"
        with pytest.raises(TaskMdVerifierError, match=expected):
            await take_offline(sandbox, (probe, probe))

    assert not marker.exists()
    if not sandbox.supports_system_shell:
        assert image.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "answer", "kept"),
    [
        ("/app/startup.sh", "/app/startup.sh", ""),
        ("/app/../app/startup.sh", "/app/startup.sh", ""),
        ("/etc/profile.d/a.sh", "/etc/profile.d/a.sh\n", "/etc/profile.d/a.sh"),
        ("/opt/env/a.sh", "/opt/conda/etc/a.sh\n", "/opt/conda/etc/a.sh"),
        ("/etc/profile.d/a.sh", "/app/a.sh\n", ""),
        ("/etc/profile.d/a.sh", "/proc/1/cwd/a.sh\n", ""),
        ("/etc/profile.d/a.sh", "/dev/shm/a.sh\n", ""),
        ("/etc/profile.d/a.sh", "/etc/../app/a.sh\n", ""),
        ("/etc/profile.d/a.sh", "/etc/$(id)\n", ""),
        ("/etc/profile.d/a.sh", "/etc/a.sh\n/etc/b.sh\n", ""),
        ("/etc/profile.d/a.sh", '"/etc/profile.d/a.sh"\n', ""),
    ],
)
async def test_a_bash_env_the_agent_could_reach_is_refused_whatever_the_check_says(
    value, answer, kept
):
    """The container check runs the image's tools, which the agent can plant.

    A planted check that calls every value trusted cannot keep one under an
    agent-writable prefix (V7-R2-A-3), nor answer a resolved path there, in
    a pseudo filesystem, not in normal form, or one bash would expand. A JSON
    answer (the older check's) is not a path.
    """
    from benchflow.sandbox.lockdown import _trusted_verifier_bash_env

    class PlantedCheck:
        _benchflow_bash_env_inspector = "/etc/dummy-inspector"

        @receipt_aware
        async def exec(self, command, **kwargs):
            return ExecResult(stdout=answer, stderr="", return_code=0)

    task = MagicMock()
    task.config.verifier.env = {"BASH_ENV": value}
    task.config.verifier.service = "main"

    assert await _trusted_verifier_bash_env(PlantedCheck(), task, ("/app",)) == kept


# --- the check's own tools, and where it runs -------------------
#
# The check believes what ``stat`` says about a file. An image whose PATH has
# an extra folder that is root's (so the verifier's PATH keeps it) but holds
# a world-writable ``stat`` let the agent rewrite that ``stat`` to call any
# file root's: its own startup file passed and ran as root.

# A ``stat`` (or ``ls``, ``python3``) the agent rewrote: it answers that every
# file is root's and that no one else can write it.
LYING_TOOLS = {
    "stat": (
        "#!/bin/sh\n"
        'case "$*" in *"%u %a"*) echo "0 755"; exit 0 ;; esac\n'
        'exec /usr/bin/stat "$@"\n'
    ),
    "ls": (
        "#!/bin/sh\n"
        "for a; do last=$a; done\n"
        'echo "-rwxr-xr-x 1 0 0 100 Jan  1 00:00 $last"\n'
    ),
    "python3": '#!/bin/sh\nfor a; do last=$a; done\nprintf "%s\\n" "$last"\n',
}


def _plant(bin_dir: Path, name: str) -> Path:
    """Put the agent's rewritten *name* in *bin_dir*, writable by anyone."""
    tool = bin_dir / name
    if tool.is_symlink() or tool.exists():
        tool.unlink()
    tool.write_text(LYING_TOOLS[name])
    tool.chmod(0o777)
    return tool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "tools"),
    [("stat", "all"), ("ls", "all"), ("python3", "python-only")],
)
async def test_the_bash_env_check_never_believes_a_tool_anyone_can_write(
    tmp_path, tool, tools
):
    """A tool on the check's own PATH that is not root's alone empties it."""
    _needs_root_owned("/usr/bin/stat")
    if tools == "python-only":
        _needs_python3()
    own = tmp_path / "startup.sh"
    own.write_text("printf '1\\n' > /logs/verifier/reward.txt; exit 0\n")
    await _check_bash_env(tmp_path, "/etc/passwd", (), tools)  # makes the folder
    _plant(tmp_path / f"check-{tools}" / "bin-bash", tool)

    assert await _check_bash_env(tmp_path, str(own), (), tools) == ""
    assert await _check_bash_env(tmp_path, "/etc/passwd", (), tools) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("script", sorted(ZERO_SCRIPTS))
@pytest.mark.parametrize("kind", SH_KINDS + BASH_KINDS)
async def test_a_planted_stat_on_the_verifier_path_never_keeps_the_agent_file(
    tmp_path, kind, script, monkeypatch
):
    """The review's input: the verifier's PATH keeps a root-owned extra folder
    (``/opt/check-tools``; a temp folder stands for it) whose ``stat`` the
    agent rewrote, and BASH_ENV names the agent's file outside every
    agent-writable prefix. test.sh scores 0 and the file never runs."""
    from benchflow.sandbox import lockdown

    _needs_root_owned("/usr/bin/stat")
    monkeypatch.setattr(lockdown, "_RUNTIME_PATH_PREFIXES", ("/logs", "/testbed"))
    marker = tmp_path / "startup-ran"
    startup = tmp_path / "agent-state" / "start.sh"
    startup.parent.mkdir()
    startup.write_text(
        f"touch {marker}\nprintf '1\\n' > \"$VERIFIER_DIR/reward.txt\"\nexit 0\n"
    )
    tools = tmp_path / "check-tools"
    tools.mkdir()
    _plant(tools, "stat")

    rewards, env = await _verify_hardened(
        tmp_path,
        ZERO_SCRIPTS[script],
        kind,
        monkeypatch,
        image_env={},
        task_env={"BASH_ENV": str(startup)},
        path_extras=(str(tools),),
    )

    assert rewards == {"reward": 0.0}, _stdout(tmp_path)
    assert env["BASH_ENV"] == ""
    assert not marker.exists()


# A Compose task whose test.sh runs in another service: a path names another
# file there (a volume the agent writes through main, mounted at another
# place). The check reads and checks BASH_ENV where bash will read it, never
# main's answer by name.


class _TwoServices:
    """main and target, each with its own view of ``/opt/activation``."""

    def __init__(self, views: dict[str, Path], image_bash_env: dict[str, str]):
        self.views = views
        self.image_bash_env = image_bash_env
        self.services: list[str] = []

    async def exec(self, command, env=None, user=None, timeout_sec=None, **kwargs):
        service = kwargs.get("service", "main")
        self.services.append(service)
        if command.startswith("command printf"):
            return ExecResult(
                stdout=self.image_bash_env.get(service, "") + "\n",
                stderr="",
                return_code=0,
            )
        command = command.replace("/opt/activation", str(self.views[service]))
        return await _HostShell().exec(command, env=env)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["task", "image"])
async def test_the_bash_env_check_runs_in_the_verifier_service(tmp_path, source):
    from benchflow.sandbox.lockdown import _trusted_verifier_bash_env

    _needs_root_owned("/etc/passwd")
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "passwd").write_text("printf '1\\n' > /logs/verifier/reward.txt\n")
    value = "/opt/activation/passwd"
    env = _TwoServices(
        {"main": Path("/etc"), "target": shared},
        {"main": value, "target": value},
    )
    task = MagicMock()
    task.config.verifier.service = "target"
    task.config.verifier.env = {"BASH_ENV": value} if source == "task" else {}

    assert await _trusted_verifier_bash_env(env, task, ()) == ""
    assert set(env.services) == ({"target"} if source == "image" else set())


# A probe that fails (a timeout, a transport error) empties BASH_ENV, which
# can change the score, so it is said at warning level.
class _FailingExec:
    _benchflow_bash_env_inspector = "/etc/dummy-inspector"

    async def exec(self, command, **kwargs):
        raise RuntimeError("exec timed out")


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["task", "image"])
async def test_a_failed_bash_env_probe_is_a_warning(source, caplog):
    from benchflow.sandbox.lockdown import _trusted_verifier_bash_env

    task = MagicMock()
    task.config.verifier.service = "main"
    task.config.verifier.env = (
        {"BASH_ENV": "/etc/profile.d/activate.sh"} if source == "task" else {}
    )

    with caplog.at_level(logging.WARNING, logger="benchflow.sandbox.lockdown"):
        assert await _trusted_verifier_bash_env(_FailingExec(), task, ()) == ""

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("BASH_ENV" in m and "exec timed out" in m for m in warnings), warnings
    if source == "task":
        assert any("/etc/profile.d/activate.sh" in m for m in warnings), warnings
