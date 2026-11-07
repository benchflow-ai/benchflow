"""The exec transport contract, the same cases on every backend (DAYTONA-EXEC).

``benchflow.sandbox._transport``: what one exec hands its provider stays
under one bound, a larger command travels as data (the exec's stdin, or a
file only root can reach), and structured output crosses as a receipt.

Each backend's own ``exec`` runs here against a fake of its provider. Every
fake runs the commands for real, in one Docker container (root and an
``agent`` account, real ``su``), and keeps what its provider does to them:

- Docker / remote Docker: ``docker exec`` with the backend's argv and stdin.
- Daytona: the session command is a file (``cmd.sh``) the session's bash
  runs; its logs lose every NUL, drop the SDK's ``\\x01\\x01\\x01`` /
  ``\\x02\\x02\\x02`` markers, decode bytes that are not UTF-8 as U+FFFD and
  end with a newline (measured on real Daytona, DAYTONA-EXEC.md 1.4); one
  argument over 128 KiB fails with E2BIG as the kernel's own; the toolbox
  writes and deletes files as root.
- Daytona DinD: the VM's command is a Daytona session command in the same
  container; ``docker compose exec`` is a shim that runs the service's argv
  there with its own stdin, user and folder.
- Modal: argv, the env as a Secret, the process's stdin.
- AgentCore: one shell string, refused over the service's 64 KB, and a
  sealed upload that writes root's file.
- Apple container: ``container exec`` argv and stdin.

``BENCHFLOW_TRANSPORT_LIVE_DAYTONA=1`` (with ``DAYTONA_API_KEY``) adds the
real Daytona API as one more backend: the cases and the staged file's
configuration (owner, mode, the folders above it) on a real sandbox with
BenchFlow's protected runtime kit.

Docker is needed (the fakes' container); without it the module skips.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
import subprocess
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchflow.sandbox import _transport
from benchflow.sandbox._exec_output import exec_raw
from benchflow.sandbox._exec_receipt import receipt_exec
from tests.integration.deterministic import harness as h

pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("exec_transport")]

IMAGE = "python:3.12-slim"
AGENT_UID = 1000
LIVE = os.environ.get("BENCHFLOW_TRANSPORT_LIVE_DAYTONA") == "1"
FAKES = [
    "docker",
    "remote-docker",
    "daytona",
    "daytona-dind",
    "modal",
    "agentcore",
    "apple-container",
]
BACKENDS = FAKES + (["daytona-live"] if LIVE else [])
# Kernel limit of one argument (MAX_ARG_STRLEN).
ARG_LIMIT = 131072

# `docker compose ... exec [flags] <service> argv...` inside the DinD "VM":
# the service's argv run here, as its -u user, in its -w folder, with its -e
# env and the shim's own stdin (as Compose passes it on).
_COMPOSE_SHIM = r"""#!/usr/local/bin/python3
import os, pwd, sys
a = sys.argv[1:]
i = a.index("exec") + 1
user = cwd = None
env = {}
while a[i].startswith("-"):
    f = a[i]
    if f in ("-T", "--no-TTY", "-i"):
        i += 1
    elif f in ("-u", "--user"):
        user = a[i + 1]; i += 2
    elif f in ("-w", "--workdir"):
        cwd = a[i + 1]; i += 2
    elif f in ("-e", "--env"):
        k, eq, v = a[i + 1].partition("=")
        if eq:
            env[k] = v
        else:
            env.pop(k, None); os.environ.pop(k, None)
        i += 2
    else:
        raise SystemExit("compose shim: unknown flag " + f)
argv = a[i + 1:]
os.environ.update(env)
if cwd:
    os.chdir(cwd)
if user not in (None, "root", "0"):
    entry = pwd.getpwnam(user) if not user.isdigit() else pwd.getpwuid(int(user))
    os.setgid(entry.pw_gid); os.setuid(entry.pw_uid)
os.execvp(argv[0], argv)
"""


async def _run(
    argv: list[str], *, stdin: bytes | None = None, timeout: float | None = None
) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode or 0, out, err


class Box:
    """The container every fake runs its provider's commands in."""

    def __init__(self, cid: str) -> None:
        self.cid = cid

    async def exec(
        self,
        argv: list[str],
        *,
        user: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> tuple[int, bytes, bytes]:
        flags = ["-i"] if stdin is not None else []
        if user is not None:
            flags += ["-u", user]
        if cwd:
            flags += ["-w", cwd]
        for k, v in (env or {}).items():
            flags += ["-e", f"{k}={v}"]
        return await _run(
            ["docker", "exec", *flags, self.cid, *argv], stdin=stdin, timeout=timeout
        )

    async def write(self, path: str, data: bytes, mode: str | None = None) -> None:
        script = 'mkdir -p "$(dirname "$1")" && cat > "$1"'
        if mode:
            script += f' && chmod {mode} "$1"'
        code, _, err = await self.exec(["sh", "-c", script, "sh", path], stdin=data)
        assert code == 0, err


@pytest.fixture(scope="module")
def box() -> Iterator[Box]:
    if not h._docker_available():
        pytest.skip("needs a Docker daemon for the providers' container")
    name = f"dx1-transport-{uuid.uuid4().hex[:10]}"
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--label",
            "benchflow.test=exec-transport",
            IMAGE,
            "sleep",
            "infinity",
        ],
        check=True,
        capture_output=True,
        timeout=600,
    )
    try:
        for command in (
            ["useradd", "-m", "-u", str(AGENT_UID), "agent"],
            [
                "sh",
                "-c",
                "cat > /usr/local/bin/docker && chmod 755 /usr/local/bin/docker",
            ],
        ):
            subprocess.run(
                ["docker", "exec", "-i", name, *command],
                input=_COMPOSE_SHIM.encode() if "docker" in command[-1] else None,
                check=True,
                capture_output=True,
                timeout=120,
            )
        yield Box(name)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=120)


# --- the providers ----------------------------------------------------------


def _daytona_logs(data: bytes) -> str:
    """What Daytona's session logs give back for *data* (DAYTONA-EXEC 1.4)."""
    data = data.replace(b"\0", b"")
    for marker in (b"\x01\x01\x01", b"\x02\x02\x02"):
        data = data.replace(marker, b"")
    text = data.decode("utf-8", "replace")
    return text if not text or text.endswith("\n") else text + "\n"


class FakeDaytona:
    """``sandbox.process`` and ``sandbox.fs`` of a Daytona sandbox."""

    def __init__(self, box: Box) -> None:
        self.box = box
        self.commands: dict[str, SimpleNamespace] = {}
        self.staged: list[str] = []
        self.process = self
        self.fs = SimpleNamespace(
            upload_file=self.upload_file,
            upload_files=self.upload_files,
            delete_file=self.delete_file,
        )

    async def create_session(self, session_id: str) -> None:
        return None

    async def delete_session(self, session_id: str) -> None:
        return None

    async def execute_session_command(self, session_id, request, timeout=None):
        cmd_id = uuid.uuid4().hex
        record = SimpleNamespace(exit_code=None, stdout="", stderr="")
        self.commands[cmd_id] = record
        folder = f"/tmp/.daytona-sessions/{session_id}/{cmd_id}"

        async def run() -> None:
            # The session's bash runs the command from a file, as Daytona does.
            await self.box.write(f"{folder}/cmd.sh", request.command.encode())
            code, out, err = await self.box.exec(["bash", f"{folder}/cmd.sh"])
            record.stdout, record.stderr = _daytona_logs(out), _daytona_logs(err)
            record.exit_code = code

        record.task = asyncio.ensure_future(run())
        return SimpleNamespace(cmd_id=cmd_id)

    async def get_session_command(self, session_id, command_id):
        return SimpleNamespace(
            id=command_id, exit_code=self.commands[command_id].exit_code
        )

    async def get_session_command_logs(self, session_id, command_id):
        record = self.commands[command_id]
        return SimpleNamespace(stdout=record.stdout, stderr=record.stderr)

    async def get_session(self, session_id):
        return SimpleNamespace(commands=[])

    async def exec(self, command, cwd=None, env=None, timeout=None):
        # The toolbox's direct exec: its result is stdout and stderr together.
        code, out, err = await self.box.exec(["bash", "-c", command], env=env)
        return SimpleNamespace(
            exit_code=code, result=(out + err).decode("utf-8", "replace")
        )

    async def upload_file(self, data, path, timeout=None):
        # Bytes, or a local file's path (the SDK takes both).
        if isinstance(data, str):
            data = Path(data).read_bytes()
        self.staged.append(path)
        await self.box.write(path, data)

    async def delete_file(self, path, recursive=False):
        code, _, err = await self.box.exec(
            ["rm", "-r" if recursive else "-f", "--", path]
        )
        assert code == 0, err

    async def upload_files(self, files, timeout=None):
        # As the SDK does (daytona/_async/filesystem.py upload_files): every
        # source is open at once while the request is built.
        streams = [open(f.source, "rb") for f in files]  # noqa: SIM115
        try:
            for upload, stream in zip(files, streams, strict=True):
                await self.box.write(upload.destination, stream.read())
        finally:
            for stream in streams:
                stream.close()

    async def delete(self):
        return None


def _daytona(box: Box, *, dind: bool) -> Any:
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
    sandbox.logger = logging.getLogger("test-transport-daytona")
    sandbox._sandbox = FakeDaytona(box)
    if dind:
        strategy = _DaytonaDinD.__new__(_DaytonaDinD)
        # The VM's compose command, without a project of its own.
        strategy._compose_cmd = lambda parts: shlex.join(["docker", "compose", *parts])  # type: ignore[method-assign]
        strategy._compose_env_vars = dict  # type: ignore[method-assign]
    else:
        strategy = _DaytonaDirect.__new__(_DaytonaDirect)
    strategy._env = sandbox
    sandbox._strategy = strategy
    return sandbox


def _docker(box: Box, *, remote: bool) -> Any:
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    cls = RemoteDockerSandbox if remote else DockerSandbox
    sandbox = cls.__new__(cls)
    sandbox._persistent_env = {}
    sandbox.default_user = None
    sandbox.logger = logging.getLogger("test-transport-docker")
    sandbox.argv_sent = []

    async def compose(
        argv,
        check=False,
        timeout_sec=None,
        *,
        raw_output=False,
        stdin_data=None,
        unset_env=None,
    ):
        # ["exec", "-T", flags..., "main", shell, "-c", command] for the CLI.
        sandbox.argv_sent.append(list(argv))
        args = [a for a in argv[1:] if a != "-T"]
        at = args.index("main")
        flags, command = args[:at], args[at + 1 :]
        if stdin_data is not None:
            flags = ["-i", *flags]
        try:
            code, out, err = await _run(
                ["docker", "exec", *flags, box.cid, *command],
                stdin=stdin_data,
                timeout=timeout_sec,
            )
        except TimeoutError:
            raise RuntimeError(
                f"Command timed out after {timeout_sec} seconds"
            ) from None
        from benchflow.sandbox._base import ExecResult

        return ExecResult(
            stdout=out.decode("utf-8", "replace"),
            stderr=err.decode("utf-8", "replace"),
            return_code=code,
        )

    sandbox._run_docker_compose_command = compose
    return sandbox


class _ModalProcess:
    def __init__(self, box: Box, argv, workdir, env, timeout) -> None:
        self._box, self._argv, self._workdir, self._env = box, argv, workdir, env
        self._timeout = timeout
        self._stdin = b""
        self._eof = False
        self._done: asyncio.Task | None = None
        self.stdin = SimpleNamespace(
            write=self._write,
            write_eof=self._write_eof,
            drain=SimpleNamespace(aio=self._drain),
        )
        self.stdout = SimpleNamespace(read=SimpleNamespace(aio=self._read_out))
        self.stderr = SimpleNamespace(read=SimpleNamespace(aio=self._read_err))
        self.wait = SimpleNamespace(aio=self._wait)

    def _write(self, data: bytes) -> None:
        self._stdin += data

    def _write_eof(self) -> None:
        self._eof = True

    async def _drain(self) -> None:
        return None

    async def _result(self) -> tuple[int, bytes, bytes]:
        if self._done is None:

            async def run() -> tuple[int, bytes, bytes]:
                try:
                    return await self._box.exec(
                        list(self._argv),
                        cwd=self._workdir,
                        env=self._env,
                        stdin=self._stdin if self._eof else None,
                        timeout=self._timeout,
                    )
                except TimeoutError:
                    # Modal ends a process past its timeout; it reports -1.
                    return -1, b"", b""

            self._done = asyncio.ensure_future(run())
        return await self._done

    async def _read_out(self) -> str:
        return (await self._result())[1].decode("utf-8", "replace")

    async def _read_err(self) -> str:
        return (await self._result())[2].decode("utf-8", "replace")

    async def _wait(self) -> int:
        return (await self._result())[0]


def _modal(box: Box, monkeypatch) -> Any:
    modal = pytest.importorskip("modal")
    from benchflow.sandbox.modal_impl import ModalSandbox

    monkeypatch.setattr(modal.Secret, "from_dict", staticmethod(dict))
    sandbox = ModalSandbox.__new__(ModalSandbox)
    sandbox._persistent_env = {}
    sandbox.default_user = None
    sandbox.logger = logging.getLogger("test-transport-modal")

    async def sandbox_exec(*argv, workdir=None, secrets=(), timeout=None):
        env = {k: v for secret in secrets for k, v in secret.items()}
        return _ModalProcess(box, argv, workdir, env, timeout)

    sandbox._sandbox = SimpleNamespace(exec=SimpleNamespace(aio=sandbox_exec))

    async def upload(source, target):
        await box.write(target, Path(source).read_bytes(), mode="600")

    sandbox._exec_env_upload = lambda: upload  # type: ignore[method-assign]
    return sandbox


def _agentcore(box: Box) -> Any:
    from benchflow.sandbox._base import ExecResult
    from benchflow.sandbox.agentcore import AgentCoreSandbox

    sandbox = AgentCoreSandbox.__new__(AgentCoreSandbox)
    sandbox._persistent_env = {}
    sandbox.default_user = None
    sandbox.logger = logging.getLogger("test-transport-agentcore")
    sandbox.runtime_arn = "arn:fake"
    sandbox.runtime_session_id = "session"
    loop = asyncio.get_running_loop()

    def invoke(payload: str, timeout: int) -> ExecResult:
        # The service refuses a body over 64 KB.
        if len(json.dumps({"command": payload, "timeout": timeout}).encode()) > 64_000:
            raise RuntimeError("ValidationException: command payload too large")
        try:
            code, out, err = asyncio.run_coroutine_threadsafe(
                box.exec(["sh", "-c", payload], timeout=timeout), loop
            ).result()
        except TimeoutError:
            return ExecResult(stdout="", stderr="timed out", return_code=124)
        return ExecResult(
            stdout=out.decode("utf-8", "replace"),
            stderr=err.decode("utf-8", "replace"),
            return_code=code,
        )

    sandbox._invoke_command = invoke

    class Sealed:
        # The sealed channel's plaintext ends up as root's file: the cipher
        # is not this contract's.
        async def upload(self, data, *, target, mode=None, owner=None, **kwargs):
            await box.write(target, data, mode=mode or "644")

        async def stage_env(self, env, owner=None):
            path = f"/tmp/.bf_sealed_env_{uuid.uuid4().hex}"
            text = "".join(f"{k}={shlex.quote(v)}\n" for k, v in env.items())
            await box.write(path, text.encode(), mode="600")
            if owner is not None:
                await box.exec(["chown", str(owner), path])
            return path

    sandbox._sealed = Sealed()

    async def upload(source, target):
        await box.write(target, Path(source).read_bytes(), mode="600")

    sandbox._exec_env_upload = lambda: upload  # type: ignore[method-assign]
    return sandbox


def _apple(box: Box, monkeypatch) -> Any:
    from benchflow.sandbox import apple_container
    from benchflow.sandbox._base import ExecResult
    from benchflow.sandbox.apple_container import AppleContainerSandbox

    sandbox = AppleContainerSandbox.__new__(AppleContainerSandbox)
    sandbox._persistent_env = {}
    sandbox.default_user = None
    sandbox.logger = logging.getLogger("test-transport-apple")
    sandbox._container_name = "fake"

    async def run_cli(*args, timeout=None, stdin_data=None):
        # ("exec", ["-i"], [--workdir d], [--user u], [-e N=V...], "fake", argv...)
        args = list(args[1:])
        at = args.index("fake")
        flags = [{"--workdir": "-w", "--user": "-u"}.get(a, a) for a in args[:at]]
        if stdin_data is not None and "-i" not in flags:
            flags.insert(0, "-i")
        code, out, err = await _run(
            ["docker", "exec", *flags, box.cid, *args[at + 1 :]],
            stdin=stdin_data,
            timeout=timeout,
        )
        return ExecResult(
            stdout=out.decode("utf-8", "replace"),
            stderr=err.decode("utf-8", "replace"),
            return_code=code,
        )

    monkeypatch.setattr(apple_container, "_run_cli", run_cli)
    return sandbox


@contextlib.asynccontextmanager
async def _live_daytona() -> AsyncIterator[Any]:
    """A real Daytona sandbox of :data:`IMAGE` with the protected runtime
    kit, through BenchFlow's own exec; its id is printed and it is deleted."""
    pytest.importorskip("daytona")
    from daytona import AsyncDaytona, CreateSandboxFromImageParams

    from benchflow.sandbox.daytona import (
        DaytonaSandbox,
        _DaytonaDirect,
        _load_daytona_sdk,
    )

    _load_daytona_sdk()
    async with AsyncDaytona() as client:
        created = await client.create(
            CreateSandboxFromImageParams(
                image=IMAGE, labels={"benchflow.test": "exec-transport-contract"}
            ),
            timeout=900,
        )
        record = os.environ.get("BENCHFLOW_TRANSPORT_LIVE_IDS")
        if record:
            with open(record, "a") as ids:
                ids.write(
                    json.dumps({"sandbox_id": created.id, "at": time.time()}) + "\n"
                )
        try:
            sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
            sandbox._persistent_env = {}
            sandbox.default_user = None
            sandbox.logger = logging.getLogger("test-transport-daytona-live")
            sandbox._sandbox = created
            strategy = _DaytonaDirect.__new__(_DaytonaDirect)
            strategy._env = sandbox
            sandbox._strategy = strategy
            code = (
                await sandbox.exec(
                    f"id -u agent 2>/dev/null || useradd -m -u {AGENT_UID} agent",
                    user="root",
                )
            ).return_code
            assert code == 0
            yield sandbox
        finally:
            await created.delete()


@pytest.fixture
async def backend(request, monkeypatch) -> AsyncIterator[Any]:
    kind = request.param
    if kind == "daytona-live":
        # The real API only: no Docker on the machine that runs these.
        async with _live_daytona() as sandbox:
            yield sandbox
        return
    box = request.getfixturevalue("box")
    yield {
        "docker": lambda: _docker(box, remote=False),
        "remote-docker": lambda: _docker(box, remote=True),
        "daytona": lambda: _daytona(box, dind=False),
        "daytona-dind": lambda: _daytona(box, dind=True),
        "modal": lambda: _modal(box, monkeypatch),
        "agentcore": lambda: _agentcore(box),
        "apple-container": lambda: _apple(box, monkeypatch),
    }[kind]()


def _each(*kinds: str):
    return pytest.mark.parametrize("backend", list(kinds) or BACKENDS, indirect=True)


@pytest.fixture
def sent_sizes(monkeypatch) -> list[tuple[str, int]]:
    """Every size a backend handed its provider in this test."""
    seen: list[tuple[str, int]] = []
    observe = lambda backend, size: seen.append((backend, size))  # noqa: E731
    _transport._observers.append(observe)
    yield seen
    _transport._observers.remove(observe)


def _script(size: int, tail: str) -> str:
    """A shell script of about *size* bytes that runs *tail* at its end: a
    heredoc-free body of comment lines and quoted words, the shape that
    multiplies under nested quoting."""
    line = ": 'it''s' \"quoted\" $HOME # padding\n"
    return line * (size // len(line)) + tail + "\n"


# --- the cases ---------------------------------------------------------------

MiB = 1024 * 1024


@_each()
@pytest.mark.parametrize("user", ["root", "agent"])
async def test_a_one_mib_script_runs_whole_as_root_and_as_the_agent(
    backend, user, sent_sizes
):
    script = _script(MiB, 'printf "%s %s" "$(id -u)" done')
    result = await exec_raw(backend.exec, script, user=user, timeout_sec=300)
    assert result.return_code == 0, (result.stderr or "")[-500:]
    expected = "0" if user == "root" else str(AGENT_UID)
    assert (result.stdout or "").strip() == f"{expected} done"
    assert sent_sizes and max(n for _, n in sent_sizes) <= _transport.COMMAND_BOUND


@_each()
async def test_a_large_env_reaches_the_command(backend, sent_sizes):
    # One value under the kernel's 128 KiB per environment string, larger
    # than any inline form of it fits.
    value = "v" * 100_000
    result = await exec_raw(
        backend.exec,
        '[ "${#BIG}" -eq 100000 ] && echo env-ok',
        env={"BIG": value},
        user="root",
        timeout_sec=120,
    )
    assert (result.stdout or "").strip() == "env-ok", (result.stderr or "")[-500:]
    assert max(n for _, n in sent_sizes) <= _transport.COMMAND_BOUND


@_each()
async def test_nul_and_cr_cross_exactly_as_a_receipt(backend):
    result = await receipt_exec(
        backend, "printf 'a\\0b\\r\\nc\\0\\001\\001\\001d'", user="root", timeout_sec=60
    )
    assert result.return_code == 0, result.stderr
    assert result.stdout == "a\0b\r\nc\0\x01\x01\x01d"


@_each()
async def test_large_output_crosses_whole(backend):
    result = await receipt_exec(
        backend, "head -c 3145728 /dev/zero | tr '\\0' x", user="root", timeout_sec=120
    )
    assert result.return_code == 0, result.stderr
    assert result.stdout == "x" * 3145728


@_each()
@pytest.mark.parametrize("code", [0, 1, 126, 127])
@pytest.mark.parametrize("staged", [False, True])
async def test_exit_codes_are_the_commands(backend, code, staged):
    command = f"exit {code}" if code != 127 else "/nonexistent/program"
    if staged:
        command = _script(200_000, command)
    result = await exec_raw(backend.exec, command, user="root", timeout_sec=120)
    assert result.return_code == code, (result.stderr or "")[-300:]


@_each()
@pytest.mark.parametrize("staged", [False, True])
async def test_a_command_past_its_timeout_is_not_a_success(backend, staged):
    command = "sleep 60; echo late"
    if staged:
        command = _script(200_000, command)
    started = time.monotonic()
    try:
        result = await exec_raw(backend.exec, command, user="root", timeout_sec=3)
    except Exception as exc:  # each backend's own timeout error
        assert (
            "timed out" in str(exc).lower() or "timeout" in type(exc).__name__.lower()
        ), exc
    else:
        assert result.return_code != 0 and "late" not in (result.stdout or "")
    assert time.monotonic() - started < 45


@_each()
@pytest.mark.parametrize("staged", [False, True])
async def test_stderr_stays_apart_from_stdout(backend, staged):
    command = "echo out; echo err >&2"
    if staged:
        command = _script(200_000, command)
    result = await exec_raw(backend.exec, command, user="agent", timeout_sec=120)
    assert result.return_code == 0
    assert (result.stdout or "").strip() == "out"
    assert "err" in (result.stderr or "") and "err" not in (result.stdout or "")


@_each(
    "daytona",
    "modal",
    "agentcore",
    "apple-container",
    *(["daytona-live"] if LIVE else []),
)
async def test_a_large_agent_command_runs_inside_the_barrier(backend, sent_sizes):
    """K1: the agent's own command is the data; the barrier is rebuilt
    around the reader, which runs as the agent (no_new_privs set)."""
    from benchflow.sandbox.agent_exec import agent_exec_call

    script = _script(
        300_000,
        'grep -q "^NoNewPrivs:[[:space:]]*1" /proc/self/status && echo "barrier $(id -u)"',
    )
    command, user = agent_exec_call(script, "agent", sandbox=None)
    # Apple container: the env's JSON line, then the staged command, on the
    # one stdin (refused at 29b6ea03c, DX-1 review A L3).
    result = await exec_raw(
        backend.exec, command, user=user, env={"K": "v"}, timeout_sec=300
    )
    assert (result.stdout or "").strip() == f"barrier {AGENT_UID}", (
        result.stderr or ""
    )[-500:]
    assert max(n for _, n in sent_sizes) <= _transport.COMMAND_BOUND


@_each("daytona", "daytona-dind", *(["daytona-live"] if LIVE else []))
async def test_the_staged_file_is_roots_alone_up_to_slash(backend, monkeypatch):
    """(c): from the upload to the read no process other than root can
    create, replace or change the staged file or a folder above it. A
    configuration assertion: the staged payload looks at its own file."""
    from benchflow.sandbox import daytona as daytona_module
    from benchflow.sandbox.lockdown import prepare_runtime_tree

    # BenchFlow's own protected runtime kit, as before the agent's first turn:
    # it makes the staging folder and proves the folders above it.
    assert await prepare_runtime_tree(backend, phase="build", timeout_sec=600)
    folder = f"/opt/runtime/run/stage-{uuid.uuid4().hex}"
    monkeypatch.setattr(daytona_module, "stage_folder", lambda: folder)
    probe = (
        f"stat -c '%n %u %a' / /opt /opt/runtime /opt/runtime/run {folder} "
        f"{folder}/payload"
    )
    # Root's shell opens the file before it selects any account, so a
    # staged command as root shows the file every staged command arrives in.
    result = await exec_raw(
        backend.exec, _script(200_000, probe), user="root", timeout_sec=120
    )
    assert result.return_code == 0, result.stderr
    rows = {
        line.split(" ", 1)[0]: line.split(" ", 1)[1].split()
        for line in (result.stdout or "").strip().splitlines()
    }
    for path in ("/", "/opt/runtime"):
        owner, mode = rows[path]
        assert owner == "0" and int(mode, 8) & 0o022 == 0, (path, rows[path])
    owner, mode = rows["/opt"]
    assert owner == "0" and (int(mode, 8) & 0o022 == 0 or int(mode, 8) & 0o1000), mode
    assert rows["/opt/runtime/run"] == ["0", "700"]
    assert rows[folder][0] == "0" and rows[f"{folder}/payload"][0] == "0"
    gone = await exec_raw(
        backend.exec, f"[ ! -e {folder} ] && echo gone", user="root", timeout_sec=60
    )
    assert (gone.stdout or "").strip() == "gone"
    # A staged command as the agent arrives the same way, and the agent
    # cannot even look inside root's staging folder.
    agent = await exec_raw(
        backend.exec, _script(200_000, "id -u"), user="agent", timeout_sec=120
    )
    assert (agent.stdout or "").strip() == str(AGENT_UID), agent.stderr
    denied = await exec_raw(
        backend.exec, "ls /opt/runtime/run", user="agent", timeout_sec=60
    )
    assert denied.return_code != 0


@contextlib.asynccontextmanager
async def _staging_folder_link(box: Box, at: str, target: str) -> AsyncIterator[str]:
    """An image-shipped link *at* (a fixed folder of BenchFlow's runtime
    tree) to *target*, a folder the agent owns or root's own; the box is put
    back as it was after."""
    marker = f"/tmp/dx3-ran-{uuid.uuid4().hex}"
    owner = "agent:agent" if target.startswith("/home/agent") else "0:0"
    code, _, err = await box.exec(
        [
            "sh",
            "-c",
            'mkdir -p "$2" && chown "$3" "$2" && chmod 755 "$2" && '
            'rm -rf "$1" && mkdir -p "$(dirname "$1")" && ln -s "$2" "$1"',
            "sh",
            at,
            target,
            owner,
        ]
    )
    assert code == 0, err
    try:
        yield marker
    finally:
        await box.exec(
            [
                "sh",
                "-c",
                'rm -f "$1" "$3"; rm -rf "$2"; '
                "mkdir -p /opt/runtime/run/home /opt/runtime/verifier && "
                "chmod 700 /opt/runtime/run /opt/runtime/run/home && "
                "chmod 755 /opt/runtime/verifier",
                "sh",
                at,
                target,
                marker,
            ]
        )


@_each("daytona", "daytona-dind")
@pytest.mark.parametrize("target", ["/home/agent/stage", "/srv/bf-stage"])
async def test_a_link_at_the_staging_folder_runs_nothing(backend, box, target, request):
    """(c), DX-1 review A M1: an image link at ``RUN_DIR`` (to a folder the
    agent owns, or to root's own) is refused by the kit's preparation, and a
    staged root command that finds one runs nothing (status 126). At
    29b6ea03c the kit accepted the link and root ran the payload from the
    agent's folder."""
    from benchflow.sandbox.lockdown import prepare_runtime_tree

    dind = "dind" in request.node.name
    async with _staging_folder_link(box, "/opt/runtime/run", target) as marker:
        if not dind:
            try:
                kit = await prepare_runtime_tree(
                    backend, phase="build", timeout_sec=600
                )
            except Exception:
                kit = False
            assert kit is not True
        result = await exec_raw(
            backend.exec,
            _script(200_000, f"touch {marker}"),
            user="root",
            timeout_sec=120,
        )
        assert result.return_code == _transport.READ_FAILED, result.stderr
        assert "not root's alone" in (result.stderr or "")
        code, _, _ = await box.exec(["test", "-e", marker])
        assert code != 0, "the staged command ran"


@_each("daytona")
async def test_an_honest_image_whose_opt_another_user_owns_still_runs_staged(
    backend, box
):
    """The proof makes ``/opt`` safe as the kit does (the kit's own build is
    a staged command): an image whose ``/opt`` the agent account owns and
    may write gets its kit and runs staged commands, as at 29b6ea03c."""
    from benchflow.sandbox.lockdown import prepare_runtime_tree

    code, _, err = await box.exec(
        ["sh", "-c", "chown agent:agent /opt && chmod 775 /opt"]
    )
    assert code == 0, err
    try:
        assert await prepare_runtime_tree(backend, phase="build", timeout_sec=600)
        result = await exec_raw(
            backend.exec,
            _script(200_000, "echo staged-ok"),
            user="root",
            timeout_sec=120,
        )
        assert (result.stdout or "").strip() == "staged-ok", result.stderr
        code, out, _ = await box.exec(["stat", "-c", "%u %a", "/opt"])
        assert out.decode().split()[0] == "0"
    finally:
        await box.exec(["sh", "-c", "chown 0:0 /opt && chmod 755 /opt"])


@_each("daytona")
@pytest.mark.parametrize("at", ["/opt/runtime/run/home", "/opt/runtime/verifier"])
async def test_the_kit_refuses_a_link_inside_its_tree(backend, box, at):
    """The kit's preparation refuses a link at ``$RT/run/home`` and
    ``$RT/verifier`` as at ``$RT`` and ``$KIT`` (accepted at 29b6ea03c)."""
    from benchflow.sandbox.lockdown import prepare_runtime_tree

    async with _staging_folder_link(box, at, "/home/agent/stage"):
        try:
            kit = await prepare_runtime_tree(backend, phase="build", timeout_sec=600)
        except Exception:
            kit = False
        assert kit is not True


@_each("agentcore")
async def test_agentcores_stager_refuses_a_link_at_its_folder(backend, box):
    """AgentCore's staged command (its sealed folder before the kit, the
    staging folder after): a link there runs nothing (DX-1 review A M1)."""
    from benchflow.sandbox.agentcore_sealed import SEAL_DIR

    async with _staging_folder_link(box, SEAL_DIR, "/home/agent/sealed") as marker:
        result = await exec_raw(
            backend.exec,
            _script(200_000, f"touch {marker}"),
            user="root",
            timeout_sec=120,
        )
        assert result.return_code == _transport.READ_FAILED, result.stderr
        code, _, _ = await box.exec(["test", "-e", marker])
        assert code != 0, "the staged command ran"


@_each("daytona-dind")
@pytest.mark.parametrize("size", [5_000, 20_000])
async def test_a_large_compose_env_with_a_staged_container_command(
    backend, size, sent_sizes
):
    """DX-1 review A M2: with a compose env of about 16 KB or more, a staged
    container command raised CommandTooLarge at 29b6ea03c; the VM's own
    command is staged too now (fd 9), every hop within the bound."""
    backend._strategy._compose_env_vars = lambda: {"TASK_PAD": "e" * size}
    small = await exec_raw(backend.exec, "echo small-ok", user="root", timeout_sec=120)
    assert (small.stdout or "").strip() == "small-ok", small.stderr
    result = await exec_raw(
        backend.exec,
        _script(200_000, 'echo "${#TASK_PAD} big-ok"'),
        user="root",
        timeout_sec=120,
    )
    assert (result.stdout or "").strip() == f"{size} big-ok", (result.stderr or "")[
        -500:
    ]
    assert max(n for _, n in sent_sizes) <= _transport.COMMAND_BOUND


@_each("daytona", "daytona-dind")
async def test_a_directory_of_2000_files_moves_with_bounded_open_files(
    backend, box, tmp_path, request
):
    """The transport pack's many-tests: 2,000 verifier files. The SDK's
    ``upload_files`` opened every source at once (EMFILE under ``ulimit -n``
    1024 at 29b6ea03c); one archive now, unpacked by root, with the same
    contents, modes (755 when any execute bit, else 644) and owner (root)."""
    import resource

    source = tmp_path / "many"
    (source / "cases").mkdir(parents=True)
    for i in range(2000):
        case = source / "cases" / f"case-{i}.sh"
        case.write_text(f"echo {i}\n")
        case.chmod(0o750 if i % 10 == 0 else 0o640)
    (source / "check.sh").write_text("echo check\n")
    target = f"/tmp/dx3-many-{uuid.uuid4().hex}"
    kwargs: dict[str, Any] = {}
    if "dind" in request.node.name:
        from benchflow.sandbox.daytona_dind import VM_ROUTE

        kwargs["route"] = VM_ROUTE
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(1024, hard), hard))
    try:
        await backend._sdk_upload_dir(source, target, **kwargs)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    try:
        code, out, err = await box.exec(
            [
                "sh",
                "-c",
                'cd "$1" && find . -type f -printf "%u %m\\n" | sort | uniq -c; '
                "cat cases/case-1999.sh",
                "sh",
                target,
            ]
        )
        assert code == 0, err
        rows = {
            (parts[1], parts[2]): int(parts[0])
            for parts in (line.split() for line in out.decode().splitlines())
            if len(parts) == 3 and parts[0].isdigit()
        }
        assert rows == {("root", "755"): 200, ("root", "644"): 1801}, out
        assert "echo 1999" in out.decode()
        staged = await box.exec(
            ["sh", "-c", "ls /opt/runtime/run | grep -c stage- || true"]
        )
        assert staged[1].decode().strip() == "0"
    finally:
        await box.exec(["rm", "-rf", target])


@_each("docker", "remote-docker", "modal", "apple-container")
async def test_a_pipe_backend_stages_nothing_on_disk(backend):
    """(c) for the stdin backends: the payload is the exec's own pipe."""
    probe = _script(
        200_000,
        "echo staged-ran; ls -A /opt/runtime/run 2>/dev/null | grep -c stage- || true",
    )
    result = await exec_raw(backend.exec, probe, user="root", timeout_sec=120)
    lines = (result.stdout or "").split()
    assert lines[:1] == ["staged-ran"] and lines[1:] in ([], ["0"]), result.stdout


@_each()
async def test_no_argument_of_a_staged_command_reaches_the_kernel_limit(
    backend, monkeypatch
):
    """The fakes start real programs: a staged 2 MiB script would fail with
    E2BIG on any argument over 128 KiB."""
    result = await exec_raw(
        backend.exec, _script(2 * MiB, "echo two-mib"), user="agent", timeout_sec=300
    )
    assert (result.stdout or "").strip() == "two-mib", (result.stderr or "")[-300:]


# --- bash start-up names: one list for every backend (DAYTONA-GAPS r3) -----
#
# An image ENV (or a sandbox env) that defines a bash function, or sets
# SHELLOPTS or BASHOPTS, is refused when the sandbox starts, with a
# ``benchflow:`` line that names it (``_base.startup_name``): bash imports
# them before its first command, as root, and a function overrides every
# builtin. The functions here only print a word (a configuration check).

_FUNCTION_LINE = "BF-IMAGE-FUNCTION-RAN"


def _function(name: str) -> str:
    return f'() {{ echo {_FUNCTION_LINE} uid=$(id -u) >&2; command {name} "$@"; }}'


_START_UP_ENVS = {
    # sed: root's session shell runs it (system_shell_prefix's PYTHON*
    # clean-up), in every plain exec and in the stage proof; not a POSIX
    # special builtin, so bash's POSIX mode imports it as well.
    "function": {"BASH_FUNC_sed%%": _function("sed")},
    # review A M1's own: bash's POSIX mode refuses to start with it.
    "special-builtin-function": {
        "BASH_FUNC_unset%%": f'() {{ echo {_FUNCTION_LINE} >&2; builtin unset "$@"; }}'
    },
    # A name that is no identifier: only bash's default mode imports it.
    "odd-name-function": {"BASH_FUNC_a-b%%": _function("true")},
    "shellopts": {"SHELLOPTS": "xtrace"},
    # An empty value defines nothing: admitted, as an image without it.
    "empty-function": {"BASH_FUNC_sed%%": ""},
}


@pytest.fixture(scope="module")
def start_up_boxes() -> Iterator[dict[str, tuple[Box, str]]]:
    """Per case of :data:`_START_UP_ENVS`: a running container whose env
    is that image env, and a stopped prototype of it (Docker's bootstrap
    reads the prototype's Config.Env)."""
    if not h._docker_available():
        pytest.skip("needs a Docker daemon for the providers' container")
    made: list[str] = []
    boxes = {}
    label = ["--label", "benchflow.test=exec-transport"]
    try:
        for case, env in _START_UP_ENVS.items():
            flags = [arg for k, v in env.items() for arg in ("-e", f"{k}={v}")]
            for argv in (
                ["run", "-d", *label, *flags, IMAGE, "sleep", "infinity"],
                [
                    "create",
                    *label,
                    "--label",
                    "com.docker.compose.service=main",
                    *flags,
                    IMAGE,
                    "sleep",
                    "infinity",
                ],
            ):
                made.append(
                    subprocess.run(
                        ["docker", *argv],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=600,
                    ).stdout.strip()
                )
            boxes[case] = (Box(made[-2]), made[-1])
        yield boxes
    finally:
        for cid in made:
            subprocess.run(
                ["docker", "rm", "-f", cid], capture_output=True, timeout=120
            )


async def _start_up_admission(kind: str, box: Box, prototype: str, monkeypatch) -> Any:
    """*kind*'s own start-time admission of *box*'s environment; returns
    the backend whose execs reach *box* when it admits it."""
    from benchflow.sandbox import _inert_bootstrap

    if kind == "daytona":
        # The create probe, on the toolbox's bash.
        sandbox = _daytona(box, dind=False)
        await sandbox._toolbox_startup_env()
    if kind == "daytona-dind":
        # The VM runs BenchFlow's DinD image; the task's image is the inner
        # containers', which DinD's bootstrap overlay admits (below).
        sandbox = _daytona(box, dind=True)
    if kind in ("docker", "remote-docker", "daytona-dind"):
        # Docker's bootstrap overlay, which DinD prepares on its VM too.
        docker = _docker(box, remote=kind == "remote-docker")

        async def inspect(cid: str) -> dict:
            code, out, err = await _run(["docker", "inspect", cid], timeout=60)
            assert code == 0, err
            return json.loads(out)[0]

        docker._inspect_container = inspect
        await _inert_bootstrap.overlay(docker, {"services": {"main": {}}}, [prototype])
        return sandbox if kind == "daytona-dind" else docker
    if kind == "modal":
        sandbox = _modal(box, monkeypatch)
        await sandbox._refuse_startup_names()
    elif kind == "apple-container":
        sandbox = _apple(box, monkeypatch)
        await sandbox._refuse_startup_names()
    elif kind == "agentcore":
        # The check the image BenchFlow builds runs at build, in exec form.
        from benchflow.sandbox.agentcore_image import _STARTUP_NAMES_CHECK

        line = next(
            row for row in _STARTUP_NAMES_CHECK.splitlines() if row.startswith("RUN ")
        )
        code, _, err = await box.exec(json.loads(line.removeprefix("RUN ")))
        if code:
            raise RuntimeError(err.decode("utf-8", "replace"))
        sandbox = _agentcore(box)
    return sandbox


@pytest.mark.parametrize("kind", FAKES)
@pytest.mark.parametrize("case", list(_START_UP_ENVS))
async def test_bash_start_up_names_in_the_image_env_are_refused_at_start(
    kind, case, start_up_boxes, monkeypatch
):
    box, prototype = start_up_boxes[case]
    if case == "empty-function":
        await _start_up_admission(kind, box, prototype, monkeypatch)
        return
    with pytest.raises(RuntimeError) as refused:
        await _start_up_admission(kind, box, prototype, monkeypatch)
    message = str(refused.value)
    assert "benchflow: " in message and "bash start-up names" in message, message
    assert all(name in message for name in _START_UP_ENVS[case]), message
    # Names only: no value reaches the message.
    assert _FUNCTION_LINE not in message and "xtrace" not in message, message


@pytest.mark.parametrize("kind", ["daytona", "daytona-dind", "docker"])
async def test_no_image_function_runs_in_roots_shells(
    kind, start_up_boxes, monkeypatch
):
    """Review A M1's reproducer, on the fakes: an image function runs in no
    root shell and in no stage proof. At 9d04974bf the image was admitted
    and root's staged command and bash printed the function's line."""
    box, prototype = start_up_boxes["function"]
    try:
        sandbox = await _start_up_admission(kind, box, prototype, monkeypatch)
    except RuntimeError as exc:
        assert "BASH_FUNC_sed%%" in str(exc)
        return
    seen = []
    for command in (
        "echo plumbing-ok",
        _script(200_000, "echo staged-ok"),
        "bash -c 'echo x | sed s/x/y/'",
    ):
        result = await exec_raw(sandbox.exec, command, user="root", timeout_sec=120)
        seen += [
            line
            for line in f"{result.stdout}\n{result.stderr}".splitlines()
            if _FUNCTION_LINE in line
        ]
    # Any root bash the container starts gets its env (review A's Docker
    # check: Config.Env reaches every ``docker exec``).
    _, out, err = await box.exec(["bash", "-c", "echo x | sed s/x/y/"], user="root")
    seen += [
        line for line in (out + err).decode().splitlines() if _FUNCTION_LINE in line
    ]
    assert not seen, seen


def test_a_command_over_the_bound_is_refused_where_it_is_sent():
    from benchflow.sandbox._transport import COMMAND_BOUND, CommandTooLarge, sent

    sent("unit", "x" * COMMAND_BOUND)
    with pytest.raises(CommandTooLarge):
        sent("unit", "x" * (COMMAND_BOUND + 1))
    with pytest.raises(CommandTooLarge):
        sent("unit", ["sh", "-c", "x" * COMMAND_BOUND])


test_a_command_over_the_bound_is_refused_where_it_is_sent = (
    pytest.mark.transport_over_bound(
        test_a_command_over_the_bound_is_refused_where_it_is_sent
    )
)
