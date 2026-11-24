"""An exec's env reaches Daytona and Daytona DinD commands through a file,
never as command text.

Daytona keeps every session command's text in its API and logs, and while a
shell runs, every process in the sandbox can read its ``/proc/<pid>/cmdline``.
So no env value may be in a session command, in what a backend hands its
provider (``_transport.sent``), in the DinD VM's ``docker`` argv, or in the
``ssh`` argv of DinD's ``attach``, in any form: as is, as octal escapes, hex
or base64. Each exec's env is a file root's toolbox uploads once per distinct
env and box, root-only and 0600, opened by root's shell before the command
runs and removed at stop.

The commands run for real, as in ``tests/test_exec_transport_contract.py``:
one Docker container stands in for the sandbox (root and an ``agent``
account, real ``su``), the Daytona fake writes and deletes files as root, and
DinD's ``docker compose exec`` is a shim that runs the service's argv in the
same container, as its ``-u`` user, with the shim's stdin on a pipe that user
owns (as Docker's is: a non-root exec can open ``/dev/stdin``). Values are
placeholders. Docker is needed; without it the module skips.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import shlex
import subprocess
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchflow.sandbox import _transport
from benchflow.sandbox._exec_output import exec_raw
from tests.integration.deterministic import harness as h
from tests.test_exec_transport_contract import (
    AGENT_UID,
    IMAGE,
    Box,
    FakeDaytona,
    _daytona_logs,
    _script,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("daytona_exec_env")]

ENV_FOLDERS = "/opt/runtime/run/env-"
# Where Daytona's daemon keeps each session command's text (root's, in a
# task image).
SESSIONS = "/root/.daytona/sessions"

# Placeholders, one per place an env comes from.
AGENT_VALUE = "placeholder-agent-credential-4e1b"
VERIFIER_VALUE = "placeholder-judge-credential-77c2"
PERSISTENT_VALUE = "placeholder-task-setting-a90d"
COMPOSE_VALUE = "placeholder-compose-credential-0f3e"
TRICKY_VALUE = 'it\'s "quoted" $HOME `x` \\ back\nslash ünï 🙂 %s %d'
VALUES = [AGENT_VALUE, VERIFIER_VALUE, PERSISTENT_VALUE, COMPOSE_VALUE]

# `docker compose ... exec [flags] <service> argv...` inside the DinD "VM",
# as tests/test_exec_transport_contract.py's shim, except that the service's
# stdin is a pipe its own user owns, as Docker gives it. With
# ENV_PROBE=<file> it runs nothing: it writes the names it was started with
# there (what Compose would interpolate), each with a digest of its value.
_COMPOSE_SHIM = r"""#!/usr/local/bin/python3
import hashlib, os, pwd, sys
a = sys.argv[1:]
probe = os.environ.get("ENV_PROBE")
if probe:
    with open(probe, "w") as out:
        for k in sorted(os.environ):
            v = os.environ[k].encode("utf-8", "surrogateescape")
            out.write(k + "=" + hashlib.sha256(v).hexdigest()[:16] + "\n")
    sys.exit(0)
if "exec" not in a:
    sys.exit(0)
i = a.index("exec") + 1
user = cwd = None
env = {}
while a[i].startswith("-"):
    f = a[i]
    if f in ("-T", "--no-TTY", "-i", "-t", "-it", "--interactive", "--tty"):
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
data = sys.stdin.buffer.read()
os.environ.update(env)
if cwd:
    os.chdir(cwd)
if user not in (None, "root", "0"):
    entry = pwd.getpwnam(user) if not user.isdigit() else pwd.getpwuid(int(user))
    os.setgid(entry.pw_gid); os.setuid(entry.pw_uid)
r, w = os.pipe()
if os.fork() == 0:
    os.close(r)
    while data:
        data = data[os.write(w, data):]
    os._exit(0)
os.close(w)
os.dup2(r, 0)
os.close(r)
os.execvp(argv[0], argv)
"""


# --- the scan --------------------------------------------------------------


def _inner(text: str, lead: int, group: int, width: int) -> str:
    """*text*, an encoding of *lead* filler bytes then a value, without the
    characters the filler and the value's last group touch."""
    return text[-(-lead * width // group) + 1 : -width]


def encodings(value: str) -> list[str]:
    """How a value could ride a command: as is; as the octal escapes the old
    Daytona env wrapper wrote (``\\NNN``) and ``printf %b``'s (``\\0NNN``);
    as ``\\xHH`` escapes, hex and ``%HH``, each in either case; as base64
    (standard and url-safe) and base32 at every alignment; and as its
    SHA-256."""
    import hashlib

    raw = value.encode()
    found = [
        value,
        "".join(f"\\{b:03o}" for b in raw),
        "".join(f"\\0{b:03o}" for b in raw),
        hashlib.sha256(raw).hexdigest(),
    ]
    for case in (str.lower, str.upper):
        found += [
            case(raw.hex()),
            "".join(f"\\x{case(f'{b:02x}')}" for b in raw),
            "".join(f"%{case(f'{b:02x}')}" for b in raw),
        ]
    for lead in range(3):
        for encode in (base64.b64encode, base64.urlsafe_b64encode):
            found.append(_inner(encode(b"\0" * lead + raw).decode(), lead, 3, 4))
    for lead in range(5):
        found.append(_inner(base64.b32encode(b"\0" * lead + raw).decode(), lead, 5, 8))
    return [f for f in found if len(f) >= 8]


def leaks(
    texts: list[str], values: list[str], bodies: Sequence[bytes] = ()
) -> list[tuple[str, str]]:
    """Each value of *values* found in a text of *texts* in any form, and
    the SHA-256 of each env file's bytes of *bodies* (a name built from an
    env's content would say what the env is)."""
    import hashlib

    digests = [hashlib.sha256(b).hexdigest() for b in bodies]
    return [
        (value, text[:200])
        for value in values
        for form in encodings(value)
        for text in texts
        if form in text
    ] + [(d, text[:200]) for d in digests for text in texts if d in text]


async def test_the_scan_finds_each_form():
    import hashlib
    import urllib.parse

    from benchflow.sandbox._base import wrap_command_with_env_file

    wrapped = wrap_command_with_env_file({"K": AGENT_VALUE}, "true")
    assert AGENT_VALUE not in wrapped and leaks([wrapped], [AGENT_VALUE])
    raw = AGENT_VALUE.encode()
    forms = [
        f"x {AGENT_VALUE} y",
        raw.hex(),
        raw.hex().upper(),
        "".join(f"\\0{b:03o}" for b in raw),
        "".join(f"\\x{b:02x}" for b in raw),
        "".join(f"\\x{b:02X}" for b in raw),
        urllib.parse.quote(AGENT_VALUE, safe=""),
        "".join(f"%{b:02x}" for b in raw),
        hashlib.sha256(raw).hexdigest(),
    ]
    for prefix in ("", "x", "xy", "xyz", "wxyz"):
        data = f"{prefix}{AGENT_VALUE}tail".encode()
        forms += [
            base64.b64encode(data).decode(),
            base64.urlsafe_b64encode(data).decode(),
            base64.b32encode(data).decode(),
        ]
    for text in forms:
        assert leaks([f"printf '%b' '{text}' | sh"], [AGENT_VALUE]), text
    body = f"export K={AGENT_VALUE}\n".encode()
    assert leaks([f"/x/{hashlib.sha256(body).hexdigest()}"], [], [body])
    assert leaks(["printf ok"], [AGENT_VALUE], [body]) == []


# --- the box and the fakes -------------------------------------------------


@pytest.fixture(scope="module")
def env_box() -> Iterator[Box]:
    if not h._docker_available():
        pytest.skip("needs a Docker daemon for the providers' container")
    name = f"daytona-env-file-{uuid.uuid4().hex[:10]}"
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--label",
            "benchflow.test=daytona-exec-env",
            IMAGE,
            "sleep",
            "infinity",
        ],
        check=True,
        capture_output=True,
        timeout=600,
    )
    try:
        for command, data in (
            (["useradd", "-m", "-u", str(AGENT_UID), "agent"], None),
            (
                [
                    "sh",
                    "-c",
                    "cat > /usr/local/bin/docker && chmod 755 /usr/local/bin/docker",
                ],
                _COMPOSE_SHIM.encode(),
            ),
        ):
            subprocess.run(
                ["docker", "exec", "-i", name, *command],
                input=data,
                check=True,
                capture_output=True,
                timeout=120,
            )
        yield Box(name)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=120)


class Recording(FakeDaytona):
    """The Daytona fake, keeping every session command, upload and delete."""

    def __init__(self, box: Box) -> None:
        super().__init__(box)
        self.session_commands: list[str] = []
        self.uploads: list[tuple[str, bytes]] = []
        self.deleted: list[str] = []
        self.refuse_env_uploads = False
        self.events: list[str] = []
        self.fs.upload_file = self.upload_file
        self.fs.delete_file = self.delete_file
        self.fs.download_file = self.download_file

    async def execute_session_command(self, session_id, request, timeout=None):
        # As Daytona's daemon: the command's text in the session's folder,
        # root's 0600, kept for the sandbox's life; the session's bash runs it.
        self.session_commands.append(request.command)
        self.events.append("exec")
        cmd_id = uuid.uuid4().hex
        record = SimpleNamespace(exit_code=None, stdout="", stderr="")
        self.commands[cmd_id] = record
        folder = f"{SESSIONS}/{session_id}/{cmd_id}"

        async def run() -> None:
            await self.box.write(f"{folder}/cmd.sh", request.command.encode(), "600")
            code, out, err = await self.box.exec(["bash", f"{folder}/cmd.sh"])
            record.stdout, record.stderr = _daytona_logs(out), _daytona_logs(err)
            record.exit_code = code

        record.task = asyncio.ensure_future(run())
        return SimpleNamespace(cmd_id=cmd_id)

    async def delete_session(self, session_id):
        self.events.append(f"delete session {session_id}")
        await self.box.exec(["rm", "-rf", "--", f"{SESSIONS}/{session_id}"])

    async def exec(self, command, cwd=None, env=None, timeout=None):
        # The toolbox's direct exec: no session, nothing in its folder.
        self.events.append(f"toolbox exec {command[:60]}")
        return await super().exec(command, cwd=cwd, env=env, timeout=timeout)

    async def upload_file(self, data, path, timeout=None):
        body = Path(data).read_bytes() if isinstance(data, str) else bytes(data)
        if self.refuse_env_uploads and path.startswith(ENV_FOLDERS):
            raise RuntimeError("the toolbox refused the upload")
        self.uploads.append((path, body))
        self.events.append(f"upload {path}")
        await super().upload_file(body, path, timeout)

    async def delete_file(self, path, recursive=False):
        self.deleted.append(path)
        self.events.append(f"delete {path}")
        await super().delete_file(path, recursive)

    async def download_file(self, source, destination=None, timeout=None):
        code, out, err = await self.box.exec(["cat", "--", source])
        assert code == 0, err
        return out

    async def create_ssh_access(self, **kwargs):
        return SimpleNamespace(token="ssh-token-placeholder")

    def env_uploads(self) -> list[tuple[str, bytes]]:
        return [(p, b) for p, b in self.uploads if p.startswith(ENV_FOLDERS)]


def _sandbox(box: Box, *, dind: bool, compose_env: dict[str, str] | None = None):
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
    sandbox.logger = logging.getLogger("test-daytona-exec-env")
    fake = Recording(box)
    sandbox._sandbox = fake
    if dind:
        strategy = _DaytonaDinD.__new__(_DaytonaDinD)
        strategy._compose_cmd = lambda parts: shlex.join(["docker", "compose", *parts])  # type: ignore[method-assign]
        values = dict(compose_env or {})
        strategy._compose_env_vars = lambda: dict(values)  # type: ignore[method-assign]
        strategy._env = sandbox
    else:
        strategy = _DaytonaDirect(sandbox)
    sandbox._strategy = strategy
    return sandbox, fake


@pytest.fixture
def handed(monkeypatch) -> list[str]:
    """Everything a backend handed its provider in this test (``_transport.sent``)."""
    seen: list[str] = []
    original = _transport.sent

    def record(backend, what, *args, **kwargs):
        seen.append(what if isinstance(what, str) else shlex.join(map(str, what)))
        return original(backend, what, *args, **kwargs)

    monkeypatch.setattr(_transport, "sent", record)
    return seen


# A probe: the names the command sees and a digest of each value (the
# values are placeholders), written to a file read back afterwards.
def _probe(out: str) -> str:
    return (
        "for n in $(env | sed -n 's/^\\([A-Za-z_][A-Za-z0-9_]*\\)=.*/\\1/p' | sort -u); "
        'do eval "v=\\${$n}"; printf "%s=%s\\n" "$n" '
        '"$(printf %s "$v" | sha256sum | cut -c1-16)"; done > ' + shlex.quote(out)
    )


async def _read(box: Box, path: str) -> dict[str, str]:
    code, out, err = await box.exec(["cat", path])
    assert code == 0, err
    rows = {}
    for line in out.decode().splitlines():
        name, _, digest = line.partition("=")
        rows[name] = digest
    return rows


def _digest(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()[:16]


# Names a probe sees that come from the shells and tools around the command,
# not from the exec's env: they differ between two wrappings of one command.
_AMBIENT = {"_", "SHLVL", "PWD", "OLDPWD"}


# --- no value on any command line ----------------------------------------


async def _session_files(box: Box) -> list[str]:
    """The text of every session command Daytona's daemon kept in the box."""
    code, out, err = await box.exec(
        [
            "sh",
            "-c",
            f"[ ! -d {SESSIONS} ] || find {SESSIONS} -name cmd.sh -exec cat {{}} +",
        ]
    )
    assert code == 0, err
    return [out.decode("utf-8", "replace")]


async def _session_folders(box: Box) -> list[str]:
    code, out, _ = await box.exec(["ls", "-A", SESSIONS])
    return out.decode().split() if code == 0 else []


def _logged(caplog) -> list[str]:
    """BenchFlow's own log lines in this test, at every level."""
    return [caplog.text, *(r.getMessage() for r in caplog.records)]


@pytest.mark.parametrize("staged", [False, True], ids=["fits", "over-the-bound"])
async def test_no_env_value_in_a_daytona_command(env_box, handed, caplog, staged):
    """Single-container Daytona: the verifier's env as root, the agent's own
    launch env as the agent (plain and inside K1's barrier), root's plumbing,
    and the persistent env on every exec, each command short or over the
    transport bound: no value in any session command, anything handed to
    the provider or BenchFlow's log, nor any env file's digest; every value
    went up in an upload."""
    from benchflow.sandbox.agent_exec import agent_exec_call

    caplog.set_level(logging.DEBUG)
    sandbox, fake = _sandbox(env_box, dind=False)
    sandbox._persistent_env = {"TASK_SETTING": PERSISTENT_VALUE}
    command = _script(200_000, "true") if staged else "true"
    runs = [
        ("root", {"JUDGE_API_KEY": VERIFIER_VALUE}, {}),
        ("agent", {"AGENT_API_KEY": AGENT_VALUE}, {}),
        ("root", {"JUDGE_API_KEY": VERIFIER_VALUE}, {"system_shell": True}),
        (None, None, {}),
    ]
    for user, env, extra in runs:
        result = await exec_raw(
            sandbox.exec, command, user=user, env=env, timeout_sec=120, **extra
        )
        assert result.return_code == 0, result.stderr
    agent_command, user = agent_exec_call(command, "agent", sandbox=None)
    result = await exec_raw(
        sandbox.exec, agent_command, user=user, env={"AGENT_API_KEY": AGENT_VALUE}
    )
    assert result.return_code == 0, result.stderr
    kept = await _session_files(env_box)
    assert kept[0].strip(), "no session file was kept"
    texts = fake.session_commands + handed + _logged(caplog) + kept
    bodies = [body for _, body in fake.env_uploads()]
    assert fake.session_commands and leaks(texts, VALUES, bodies) == []
    uploaded = b"\n".join(body for _, body in fake.uploads).decode()
    for value in (AGENT_VALUE, VERIFIER_VALUE, PERSISTENT_VALUE):
        assert value in uploaded, value


async def test_no_env_value_in_a_dind_command(env_box, handed, caplog, monkeypatch):
    """DinD: the compose env (the task's env and persistent env) on every VM
    command, a container's env as root and as the agent, a long exec under
    the heartbeat, and a container command over the transport bound: no
    value in any session command, provider argv, the VM's ``docker`` argv or
    BenchFlow's log; every value went up in an upload."""
    from benchflow.sandbox import daytona_dind

    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(daytona_dind, "_LONG_EXEC_HEARTBEAT_MIN_TIMEOUT_SEC", 30)
    sandbox, fake = _sandbox(
        env_box, dind=True, compose_env={"TASK_CREDENTIAL": COMPOSE_VALUE}
    )
    sandbox._persistent_env = {"TASK_SETTING": PERSISTENT_VALUE}
    for user, env, timeout in (
        ("root", {"JUDGE_API_KEY": VERIFIER_VALUE}, 20),
        ("agent", {"AGENT_API_KEY": AGENT_VALUE}, 20),
        ("agent", {"AGENT_API_KEY": AGENT_VALUE}, 60),
    ):
        result = await exec_raw(
            sandbox.exec, "true", user=user, env=env, timeout_sec=timeout
        )
        assert result.return_code == 0, result.stderr
    big = await exec_raw(
        sandbox.exec,
        _script(200_000, "true"),
        user="root",
        env={"JUDGE_API_KEY": VERIFIER_VALUE},
        timeout_sec=120,
    )
    assert big.return_code == 0, big.stderr
    texts = fake.session_commands + handed + _logged(caplog)
    texts += await _session_files(env_box)
    bodies = [body for _, body in fake.env_uploads()]
    assert texts and leaks(texts, VALUES, bodies) == []
    uploaded = b"\n".join(body for _, body in fake.uploads).decode()
    for value in VALUES:
        assert value in uploaded, value


async def test_the_dind_agent_launch_puts_no_env_value_on_the_ssh_argv(
    env_box, handed, caplog, monkeypatch
):
    """An agent launched over SSH on DinD: the compose env (the task's env
    and persistent env) and the agent's own env go in the launch's env file;
    the ssh argv names only it and the agent env's names. The remote
    command, run in the box, gives Compose every compose value."""
    from benchflow.sandbox.process import daytona as process_module
    from tests.test_process import _DaytonaExecHarness

    caplog.set_level(logging.DEBUG)
    compose_env = {"TASK_CREDENTIAL": COMPOSE_VALUE, "TASK_SETTING": TRICKY_VALUE}
    sandbox, _ = _sandbox(env_box, dind=True, compose_env=compose_env)
    harness = _DaytonaExecHarness()
    real = asyncio.create_subprocess_exec

    async def launch(*argv, **kwargs):
        # The launch's ssh is recorded, never run; the box's own docker is.
        if argv[:1] == ("ssh",):
            return await harness.fake_exec(*argv, **kwargs)
        return await real(*argv, **kwargs)

    monkeypatch.setattr(process_module.asyncio, "create_subprocess_exec", launch)
    process = await process_module.DaytonaProcess.from_sandbox_env(sandbox)
    await process.start("true", env={"AGENT_API_KEY": AGENT_VALUE})
    (ssh,) = harness.calls
    assert ssh[0] == "ssh"
    texts = [shlex.join(ssh), *handed, *_logged(caplog)]
    values = [COMPOSE_VALUE, TRICKY_VALUE, AGENT_VALUE]
    assert leaks(texts, values) == [], leaks(texts, values)[:1]
    assert "--env AGENT_API_KEY" in ssh[-1]
    assert "TASK_CREDENTIAL" not in ssh[-1]
    out = f"/tmp/launch-probe-{uuid.uuid4().hex}"
    code, _, err = await env_box.exec(["sh", "-c", ssh[-1]], env={"ENV_PROBE": out})
    assert code == 0, err
    seen = await _read(env_box, out)
    for name, value in {**compose_env, "AGENT_API_KEY": AGENT_VALUE}.items():
        assert seen.get(name) == _digest(value), name


async def test_dind_attach_puts_no_env_value_on_the_ssh_argv(env_box, monkeypatch):
    """DinD's attach: the compose env (the task's env and persistent env) goes
    in a file the remote shell opens, removes and reads; the ssh argv names
    only its path. The remote command, run in the box, gives Compose every
    name and value."""
    import os

    compose_env = {"TASK_CREDENTIAL": COMPOSE_VALUE, "TASK_SETTING": TRICKY_VALUE}
    sandbox, fake = _sandbox(env_box, dind=True, compose_env=compose_env)

    class Execd(Exception):
        pass

    def execvp(program, argv):
        raise Execd(argv)

    monkeypatch.setattr(os, "execvp", execvp)
    with pytest.raises(Execd) as execd:
        await sandbox._strategy.attach()
    argv = execd.value.args[0]
    assert argv[0] == "ssh"
    bodies = [body for _, body in fake.uploads]
    assert leaks([shlex.join(argv)], [COMPOSE_VALUE, TRICKY_VALUE], bodies) == []
    path = next(p for p, body in fake.uploads if COMPOSE_VALUE.encode() in body)
    assert path in argv[-1]
    out = f"/tmp/attach-probe-{uuid.uuid4().hex}"
    code, _, err = await env_box.exec(["sh", "-c", argv[-1]], env={"ENV_PROBE": out})
    assert code == 0, err
    seen = await _read(env_box, out)
    for name, value in compose_env.items():
        assert seen.get(name) == _digest(value), name
    code, _, _ = await env_box.exec(["test", "-e", path.rsplit("/", 1)[0]])
    assert code != 0, "the attach's env folder is still there"


async def test_a_channel_exec_carries_no_env_value(monkeypatch):
    """Root's plumbing goes over the PTY command channel once the kit is in,
    and carries the persistent env: no value in any frame the channel sends
    (the folder's preparation, then the exec), and the exec opens the env
    file root uploaded, on the same descriptor as a session exec does."""
    import json
    from unittest.mock import AsyncMock

    from benchflow.sandbox._base import ENV_FD, KIT_PROBE_ROUTE
    from tests.test_daytona_channel import output
    from tests.test_daytona_channel_contract import _exec_box

    box, _pool, made = _exec_box(monkeypatch)
    uploads: list[tuple[str, bytes]] = []

    async def upload(data, path, timeout=None):
        uploads.append((path, bytes(data)))

    box._sandbox.fs = SimpleNamespace(upload_file=upload, delete_file=AsyncMock())
    frames: list[dict] = []

    async def send(data):
        head = json.loads(data[4:])
        frames.append(head)
        await made.on_data(output(b"S", head["i"]))
        await made.on_data(output(b"X", head["i"], b"0"))

    made.pty.send_input.side_effect = send
    result = await box.exec(
        "true", user="root", env={"TASK_SETTING": PERSISTENT_VALUE}, timeout_sec=5
    )
    assert result.return_code == 0
    assert len(frames) == 2, frames
    assert all(f["a"][:2] == [KIT_PROBE_ROUTE.shell, "-c"] for f in frames)
    ((path, body),) = uploads
    assert PERSISTENT_VALUE.encode() in body
    argvs = [shlex.join(f["a"]) for f in frames]
    assert leaks(argvs, [PERSISTENT_VALUE], [body]) == []
    assert f"{ENV_FD}<{path}" in frames[-1]["a"][-1]
    assert box._sandbox.process.created == []


async def test_an_attach_ssh_never_ran_leaves_no_env_folder_after_the_stop(
    env_box, monkeypatch
):
    """An attach whose ssh never ran the remote command leaves its env
    folder on the VM: DinD's stop removes it with the session files."""
    import os

    sandbox, fake = _sandbox(
        env_box, dind=True, compose_env={"TASK_CREDENTIAL": COMPOSE_VALUE}
    )

    def execvp(program, argv):
        raise OSError("ssh never started")

    monkeypatch.setattr(os, "execvp", execvp)
    with pytest.raises(OSError):
        await sandbox._strategy.attach()
    (folder,) = {p.rsplit("/", 1)[0] for p, _ in fake.uploads if "/attach-" in p}
    code, _, _ = await env_box.exec(["test", "-e", folder])
    assert code == 0, "the attach made no folder"

    async def teardown(timeout_sec=30):
        return None

    async def stop_sandbox(provider):
        code, _, _ = await env_box.exec(["test", "-e", folder])
        assert code != 0, "the attach's folder is there at the delete"

    sandbox._strategy._compose_teardown = teardown
    sandbox._stop_sandbox = stop_sandbox
    await sandbox._strategy.stop(delete=True)


# --- the env still arrives -----------------------------------------------


def _inline_probe(env: dict[str, str], out: str) -> str:
    from benchflow.sandbox._base import wrap_command_with_env_file

    return wrap_command_with_env_file(env, _probe(out))


ENV = {
    "AGENT_API_KEY": AGENT_VALUE,
    "TRICKY": TRICKY_VALUE,
    "EQ": "a=b==c=\n=d",
    "BIG": "v" * 100_000,
    "EMPTY": "",
    "PATH_LIKE": "/opt/x:/usr/bin",
}


def _same_names(got: dict[str, str], want: dict[str, str]) -> None:
    assert {k: v for k, v in got.items() if k not in _AMBIENT} == {
        k: v for k, v in want.items() if k not in _AMBIENT
    }


@pytest.mark.parametrize("staged", [False, True], ids=["fits", "over-the-bound"])
@pytest.mark.parametrize(
    "user,extra",
    [("root", {}), ("root", {"system_shell": True}), ("agent", {}), (None, {})],
    ids=["root", "root-plumbing", "agent", "default"],
)
async def test_a_daytona_exec_sees_its_env(env_box, user, extra, staged):
    """Each exec sees exactly the names and values it saw with the env
    written into its command: same names, same value digests, a command
    short or over the transport bound (its staged payload on descriptor 0,
    the env on its own)."""
    sandbox, _ = _sandbox(env_box, dind=False)
    sandbox._persistent_env = {"TASK_SETTING": PERSISTENT_VALUE}
    via_file = f"/tmp/probe-file-{uuid.uuid4().hex}"
    inline = f"/tmp/probe-inline-{uuid.uuid4().hex}"

    def command(text: str) -> str:
        return _script(200_000, text) if staged else text

    result = await exec_raw(
        sandbox.exec,
        command(_probe(via_file)),
        user=user,
        env=ENV,
        timeout_sec=120,
        **extra,
    )
    assert result.return_code == 0, result.stderr
    merged = {"TASK_SETTING": PERSISTENT_VALUE, **ENV}
    sandbox._persistent_env = {}
    result = await exec_raw(
        sandbox.exec,
        command(_inline_probe(merged, inline)),
        user=user,
        timeout_sec=120,
        **extra,
    )
    assert result.return_code == 0, result.stderr
    got, want = await _read(env_box, via_file), await _read(env_box, inline)
    for name, value in merged.items():
        if not (extra and name == "PATH_LIKE"):
            assert got.get(name) == _digest(value), (name, got.get(name))
    _same_names(got, want)


@pytest.mark.parametrize("user", ["root", "agent"])
@pytest.mark.parametrize("long", [False, True], ids=["short", "heartbeat"])
async def test_a_dind_container_exec_sees_its_env(env_box, user, long, monkeypatch):
    """A container exec in DinD (as root, as the agent, and under the long
    exec's heartbeat) sees exactly the names and values it saw with the env
    written into its command, the compose env the shim passes on included."""
    from benchflow.sandbox import daytona_dind

    monkeypatch.setattr(daytona_dind, "_LONG_EXEC_HEARTBEAT_MIN_TIMEOUT_SEC", 30)
    sandbox, _ = _sandbox(env_box, dind=True, compose_env={"TASK_PAD": COMPOSE_VALUE})
    timeout = 60 if long else 20
    out = f"/tmp/probe-dind-{uuid.uuid4().hex}"
    inline = f"/tmp/probe-dind-inline-{uuid.uuid4().hex}"
    result = await exec_raw(
        sandbox.exec, _probe(out), user=user, env=ENV, timeout_sec=timeout
    )
    assert result.return_code == 0, result.stderr
    result = await exec_raw(
        sandbox.exec, _inline_probe(ENV, inline), user=user, timeout_sec=timeout
    )
    assert result.return_code == 0, result.stderr
    got, want = await _read(env_box, out), await _read(env_box, inline)
    for name, value in {**ENV, "TASK_PAD": COMPOSE_VALUE}.items():
        assert got.get(name) == _digest(value), (name, got.get(name))
    _same_names(got, want)


@pytest.mark.parametrize("user", ["root", "agent"])
async def test_a_dind_container_command_over_the_bound_sees_its_env(env_box, user):
    sandbox, _ = _sandbox(env_box, dind=True)
    out = f"/tmp/probe-dind-big-{uuid.uuid4().hex}"
    inline = f"/tmp/probe-dind-big-inline-{uuid.uuid4().hex}"
    result = await exec_raw(
        sandbox.exec,
        _script(200_000, _probe(out)),
        user=user,
        env=ENV,
        timeout_sec=120,
    )
    assert result.return_code == 0, result.stderr
    result = await exec_raw(
        sandbox.exec,
        _script(200_000, _inline_probe(ENV, inline)),
        user=user,
        timeout_sec=120,
    )
    assert result.return_code == 0, result.stderr
    got, want = await _read(env_box, out), await _read(env_box, inline)
    for name, value in ENV.items():
        assert got.get(name) == _digest(value), (name, got.get(name))
    _same_names(got, want)


# --- one upload per distinct env per box ----------------------------------


async def test_one_upload_per_distinct_env(env_box):
    sandbox, fake = _sandbox(env_box, dind=False)
    for _ in range(4):
        result = await exec_raw(sandbox.exec, "true", user="agent", env={"A": "1"})
        assert result.return_code == 0, result.stderr
    assert len(fake.env_uploads()) == 1
    await exec_raw(sandbox.exec, "true", user="agent", env={"A": "2"})
    assert len(fake.env_uploads()) == 2
    await exec_raw(sandbox.exec, "true", user="root", env={"A": "1"})
    assert len(fake.env_uploads()) == 2
    # Concurrent execs with one new env upload it once.
    results = await asyncio.gather(
        *(
            exec_raw(sandbox.exec, "true", user="agent", env={"B": "x"})
            for _ in range(4)
        )
    )
    assert all(r.return_code == 0 for r in results)
    assert len(fake.env_uploads()) == 3
    # The env files share one folder in root's staging folder.
    folders = {p.rsplit("/", 1)[0] for p, _ in fake.env_uploads()}
    assert len(folders) == 1


async def test_a_new_provider_sandbox_uploads_again(env_box):
    sandbox, fake = _sandbox(env_box, dind=False)
    await exec_raw(sandbox.exec, "true", user="agent", env={"A": "1"})
    replacement = Recording(env_box)
    sandbox._sandbox = replacement
    result = await exec_raw(sandbox.exec, "true", user="agent", env={"A": "1"})
    assert result.return_code == 0, result.stderr
    assert len(fake.env_uploads()) == 1 and len(replacement.env_uploads()) == 1
    assert {p.rsplit("/", 1)[0] for p, _ in fake.env_uploads()} != {
        p.rsplit("/", 1)[0] for p, _ in replacement.env_uploads()
    }


async def test_dind_uploads_the_compose_env_once(env_box):
    sandbox, fake = _sandbox(
        env_box, dind=True, compose_env={"TASK_CREDENTIAL": COMPOSE_VALUE}
    )
    for _ in range(3):
        result = await exec_raw(sandbox.exec, "true", user="agent", env={"A": "1"})
        assert result.return_code == 0, result.stderr
    # The compose env and the container's env: one file each.
    assert len(fake.env_uploads()) == 2


# --- a failed upload refuses ------------------------------------------------


async def test_a_failed_upload_refuses_the_exec(env_box, caplog):
    caplog.set_level(logging.DEBUG)
    sandbox, fake = _sandbox(env_box, dind=False)
    fake.refuse_env_uploads = True
    marker = f"/tmp/refused-{uuid.uuid4().hex}"
    before = len(fake.session_commands)
    with pytest.raises(Exception, match="refused the upload") as refused:
        await sandbox.exec(f"touch {marker}", user="agent", env={"A": AGENT_VALUE})
    # Nothing of the command was sent: at most the staging folder's own
    # preparation, which carries no env.
    sent = fake.session_commands[before:]
    assert all(marker not in c for c in sent), sent
    text = [str(refused.value), repr(refused.value), *_logged(caplog)]
    assert leaks(sent + text, [AGENT_VALUE]) == []
    code, _, _ = await env_box.exec(["test", "-e", marker])
    assert code != 0
    # The next exec uploads again and runs.
    fake.refuse_env_uploads = False
    result = await exec_raw(
        sandbox.exec, f"touch {marker}", user="agent", env={"A": "1"}
    )
    assert result.return_code == 0, result.stderr
    assert len(fake.env_uploads()) == 1


@pytest.mark.parametrize("dind", [False, True], ids=["single-container", "dind"])
async def test_a_value_with_a_nul_is_refused(env_box, dind):
    """No process environment can hold a NUL: an env value with one is
    refused with its name before anything is uploaded or sent, never cut."""
    sandbox, fake = _sandbox(env_box, dind=dind)
    marker = f"/tmp/nul-ran-{uuid.uuid4().hex}"
    uploads = len(fake.uploads)
    with pytest.raises(ValueError, match="NUL_VALUE") as refused:
        await sandbox.exec(f"touch {marker}", user="agent", env={"NUL_VALUE": "a\0b"})
    assert "NUL" in str(refused.value)
    assert len(fake.uploads) == uploads
    assert not [c for c in fake.session_commands if marker in c]
    code, _, _ = await env_box.exec(["test", "-e", marker])
    assert code != 0, "the command ran"


async def test_a_swapped_env_file_runs_nothing_and_the_next_exec_uploads_afresh(
    env_box,
):
    """An env file that is no longer root's alone (a link to a file the agent
    owns) is never opened: the exec raises and nothing ran. Its folder is not
    used again: the next exec uploads into a new one and runs, and stop
    removes both."""
    from unittest.mock import AsyncMock

    sandbox, fake = _sandbox(env_box, dind=False)
    env = {"A": "1"}
    result = await exec_raw(sandbox.exec, "true", user="agent", env=env)
    assert result.return_code == 0, result.stderr
    ((path, _),) = fake.env_uploads()
    planted = f"/home/agent/planted-{uuid.uuid4().hex}"
    marker = f"/tmp/swapped-{uuid.uuid4().hex}"
    code, _, err = await env_box.exec(
        [
            "sh",
            "-c",
            f"printf 'touch {marker}\\nset --\\n' > {planted} && "
            f"chown agent {planted} && ln -sf {planted} {path}",
        ]
    )
    assert code == 0, err
    ran = f"/tmp/swapped-ran-{uuid.uuid4().hex}"
    # Refused, never a status a caller could read on.
    with pytest.raises(Exception, match="not root's alone"):
        await sandbox.exec(f"touch {ran}", user="root", env=env)
    for left in (marker, ran):
        code, _, _ = await env_box.exec(["test", "-e", left])
        assert code != 0, f"{left} exists: something ran"
    result = await exec_raw(sandbox.exec, f"touch {ran}", user="root", env=env)
    assert result.return_code == 0, result.stderr
    folders = [p.rsplit("/", 1)[0] for p, _ in fake.env_uploads()]
    assert len(folders) == 2 and folders[0] != folders[1]
    sandbox._strategy.stop = AsyncMock()
    sandbox._write_exec_counts = lambda: None
    await sandbox.stop(delete=True)
    for folder in folders:
        code, _, _ = await env_box.exec(["test", "-e", folder])
        assert code != 0, folder


# --- root's alone, 0600 ------------------------------------------------------


async def test_the_env_file_is_roots_0600_while_the_command_runs(env_box):
    """Root's 0600 file in root's 700 folder. The command's own shell closed
    the descriptor; the processes around it (su's shell, timeout) still hold
    it, and the agent cannot open it through their /proc entries."""
    sandbox, fake = _sandbox(env_box, dind=False)
    # The command's own shell closed the descriptor once the env was read:
    # reading it (no reopen) fails here too.
    snoop = (
        "{ /bin/cat <&8; } >/dev/null 2>&1 && echo OPEN; "
        "for p in /proc/[0-9]*/fd/8; do "
        '/bin/cat "$p" 2>/dev/null | grep -q AGENT_API_KEY && echo LEAK; done; '
        "echo done"
    )
    result = await exec_raw(
        sandbox.exec,
        snoop,
        user="agent",
        env={"AGENT_API_KEY": AGENT_VALUE},
        timeout_sec=60,
    )
    assert (result.stdout or "").split() == ["done"], result.stdout
    as_root = await exec_raw(
        sandbox.exec,
        "{ /bin/cat <&8; } >/dev/null 2>&1 && echo OPEN; echo done",
        user="root",
        env={"AGENT_API_KEY": AGENT_VALUE},
    )
    assert (as_root.stdout or "").split() == ["done"], as_root.stdout
    ((path, _),) = fake.env_uploads()
    folder = path.rsplit("/", 1)[0]
    code, out, err = await env_box.exec(
        ["stat", "-c", "%n %u %a", "/opt/runtime/run", folder, path]
    )
    assert code == 0, err
    rows = [line.split()[1:] for line in out.decode().splitlines()]
    assert rows == [["0", "700"], ["0", "700"], ["0", "600"]], rows


# --- removed at stop and before a snapshot -----------------------------------


async def test_stop_removes_the_env_files_before_the_sandbox_goes(env_box):
    """Single-container stop: the env folder is gone by the time the
    strategy's stop (which deletes the sandbox) starts."""
    sandbox, fake = _sandbox(env_box, dind=False)
    await exec_raw(sandbox.exec, "true", user="agent", env={"A": "1"})
    ((path, _),) = fake.env_uploads()
    folder = path.rsplit("/", 1)[0]
    present: list[bool] = []

    sessions_left: list[list[str]] = []

    async def strategy_stop(delete):
        code, _, _ = await env_box.exec(["test", "-e", folder])
        present.append(code == 0)
        sessions_left.append(await _session_folders(env_box))
        fake.events.append("sandbox delete")

    assert await _session_folders(env_box), "no session was kept"
    sandbox._strategy.stop = strategy_stop
    sandbox._write_exec_counts = lambda: None
    await sandbox.stop(delete=True)
    assert present == [False]
    assert sessions_left == [[]], sessions_left
    assert fake.events.index(f"delete {folder}") < fake.events.index("sandbox delete")


def _hang():
    """A file-API delete that never answers."""

    async def delete_file(path, recursive=False):
        await asyncio.Event().wait()

    return delete_file


@pytest.mark.timeout(60)
async def test_a_removal_past_its_bound_still_lets_the_stop_delete(
    env_box, monkeypatch
):
    """The removal is best effort: one that hangs past its bound is logged,
    and the strategy's stop (the delete and its left-behind record) runs."""
    import time

    from benchflow.sandbox import daytona as daytona_module

    monkeypatch.setattr(daytona_module, "_EXEC_TRACES_TIMEOUT_SEC", 0.5)
    sandbox, fake = _sandbox(env_box, dind=False)
    await exec_raw(sandbox.exec, "true", user="agent", env={"A": "1"})
    fake.fs.delete_file = _hang()
    stopped: list[bool] = []

    async def strategy_stop(delete):
        stopped.append(delete)

    sandbox._strategy.stop = strategy_stop
    sandbox._write_exec_counts = lambda: None
    started = time.monotonic()
    await sandbox.stop(delete=True)
    assert stopped == [True]
    assert time.monotonic() - started < 10


@pytest.mark.timeout(60)
@pytest.mark.parametrize("dind", [False, True], ids=["single-container", "dind"])
async def test_a_cancel_during_the_removal_still_lets_the_stop_delete(
    env_box, monkeypatch, dind
):
    """A caller's own stop bound that fires while the removal hangs: the
    sandbox's delete still runs (single container: the strategy's stop;
    DinD: its delete and left-behind record), and the caller gets its
    timeout."""
    sandbox, fake = _sandbox(
        env_box, dind=dind, compose_env={"TASK_CREDENTIAL": COMPOSE_VALUE}
    )
    await exec_raw(sandbox.exec, "true", user="agent", env={"A": "1"})
    fake.fs.delete_file = _hang()
    deleted: list[str] = []
    if dind:

        async def teardown(timeout_sec=30):
            return None

        async def stop_sandbox(provider):
            deleted.append("delete")

        sandbox._strategy._compose_teardown = teardown
        sandbox._stop_sandbox = stop_sandbox
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(sandbox._strategy.stop(delete=True), 0.5)
    else:

        async def strategy_stop(delete):
            deleted.append("delete")

        sandbox._strategy.stop = strategy_stop
        sandbox._write_exec_counts = lambda: None
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(sandbox.stop(delete=True), 0.5)
    assert deleted == ["delete"]


@pytest.mark.timeout(60)
async def test_a_restore_removes_the_files_before_the_sandbox_goes(
    env_box, monkeypatch
):
    """A restore deletes the sandbox it replaces: its env and session files go
    first. A cancel during that removal stops the restore before the delete,
    and the sandbox is still the one a stop deletes."""
    from benchflow.sandbox import _path_inspector_state
    from benchflow.sandbox.protocol import SandboxImage
    from benchflow.trust import state as trust_state

    monkeypatch.setattr(_path_inspector_state, "read", lambda image: None)
    monkeypatch.setattr(_path_inspector_state, "restored", lambda s, state: None)
    monkeypatch.setattr(trust_state, "restore_from", lambda s, image: None)
    sandbox, fake = _sandbox(env_box, dind=False)
    await exec_raw(sandbox.exec, "true", user="agent", env={"A": "1"})
    folder = fake.env_uploads()[0][0].rsplit("/", 1)[0]
    seen: list[tuple[bool, list[str]]] = []

    async def strategy_restore(image):
        code, _, _ = await env_box.exec(["test", "-e", folder])
        seen.append((code == 0, await _session_folders(env_box)))

    sandbox._strategy.restore = strategy_restore
    image = SandboxImage(provider="daytona", ref="snap", meta={})
    await sandbox.restore(image)
    assert seen == [(False, [])], seen
    # A cancel while the removal hangs: no restore, the sandbox kept.
    await exec_raw(sandbox.exec, "true", user="agent", env={"A": "1"})
    fake.fs.delete_file = _hang()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(sandbox.restore(image), 0.5)
    assert len(seen) == 1 and sandbox._sandbox is fake


async def test_the_runner_verifiers_env_file_goes_when_its_command_ends(
    tmp_path, monkeypatch
):
    """The criteria runner's command, as test.sh's: it always carries an exec
    env (the runner's paths, and the task's verifier env), so the sandbox's
    env files are removed right after it, also when its exec fails."""
    from unittest.mock import AsyncMock

    from benchflow.task import verifier_core
    from tests.test_verifier_strategies import _criteria_runner_verifier, _criteria_toml

    events: list[str] = []

    async def removed(sandbox):
        events.append("removed")

    monkeypatch.setattr(verifier_core, "remove_exec_env_files", removed)
    for fails in (False, True):
        events.clear()
        verifier, _ = _criteria_runner_verifier(
            tmp_path / str(fails),
            criteria_toml=_criteria_toml("weighted_mean", {"c1": 1.0}),
            reward_payload={"metrics": {"c1": 1.0}},
        )
        verifier._task.config.verifier.env = {"JUDGE_SETTING": VERIFIER_VALUE}
        runner = verifier._sandbox.exec.side_effect

        async def exec_(*args, runner=runner, fails=fails, **kwargs):
            command = kwargs.get("command") or (args[0] if args else "")
            if "criteria_runner/reward.py" in command:
                events.append("runner")
                if fails:
                    raise RuntimeError("the exec failed")
            return await runner(*args, **kwargs)

        verifier._sandbox.exec = AsyncMock(side_effect=exec_)
        try:
            await verifier.verify()
        except Exception:
            assert fails
        assert events == ["runner", "removed"], (fails, events)


async def test_stop_goes_on_when_the_files_cannot_be_removed(env_box, monkeypatch):
    """A removal that cannot run (no root route on a broken sandbox) is
    logged; the strategy's stop, which deletes the sandbox, still runs."""
    from benchflow.sandbox import daytona as daytona_module

    sandbox, _ = _sandbox(env_box, dind=False)
    await exec_raw(sandbox.exec, "true", user="agent", env={"A": "1"})

    async def no_route(*args, **kwargs):
        raise RuntimeError("no root route")

    monkeypatch.setattr(daytona_module, "system_shell_route", no_route)
    stopped: list[bool] = []

    async def strategy_stop(delete):
        stopped.append(delete)

    sandbox._strategy.stop = strategy_stop
    sandbox._write_exec_counts = lambda: None
    await sandbox.stop(delete=True)
    assert stopped == [True]


async def test_dind_stop_removes_the_env_files_after_compose_down_before_the_delete(
    env_box,
):
    sandbox, fake = _sandbox(
        env_box, dind=True, compose_env={"TASK_CREDENTIAL": COMPOSE_VALUE}
    )
    await exec_raw(sandbox.exec, "true", user="agent", env={"A": "1"})
    folder = fake.env_uploads()[0][0].rsplit("/", 1)[0]

    async def teardown(timeout_sec=30):
        # compose down reads the compose env's file too.
        fake.events.append("compose down")
        result = await sandbox._strategy._compose_exec(
            ["down"], timeout_sec=timeout_sec
        )
        assert result.return_code == 0, result.stderr

    async def stop_sandbox(provider):
        code, _, _ = await env_box.exec(["test", "-e", folder])
        assert code != 0, "the env folder is there at the sandbox's delete"
        assert await _session_folders(env_box) == [], "sessions at the delete"
        fake.events.append("sandbox delete")

    sandbox._strategy._compose_teardown = teardown
    sandbox._stop_sandbox = stop_sandbox
    fake.events.clear()
    await sandbox._strategy.stop(delete=True)
    events = fake.events
    removal = events.index(f"delete {folder}")
    assert events.index("compose down") < removal < events.index("sandbox delete")
    assert not any(e.startswith(f"upload {ENV_FOLDERS}") for e in events[removal:])


CREDENTIAL = "/home/agent/.claude/.credentials.json"


async def _snapshotting(sandbox, fake, monkeypatch, at_create):
    """The single-container strategy's own snapshot (its credential
    take-out included) with the provider's create call replaced by
    *at_create*, which sees the box as the provider would."""
    from benchflow.sandbox import _path_inspector_state
    from benchflow.trust import state as trust_state

    sandbox.environment_name = "env-file-snapshot"
    sandbox._toolbox_startup_receipt = {}
    fake._experimental_create_snapshot = at_create
    monkeypatch.setattr(_path_inspector_state, "checkpoint", lambda s, image: image)
    monkeypatch.setattr(trust_state, "stamp_snapshot", lambda s, image: image)


async def test_no_env_file_exists_when_the_snapshot_is_created(env_box, monkeypatch):
    """At the provider's create call, root's staging folder holds no env
    folder, Daytona's session folder holds no session (each earlier exec's
    command text) and the agent's credential file is out (the take-out);
    every exec the take-out ran carried no env, and none uploaded one. After
    it the credential is back, and the next exec with the same env uploads
    it again instead of opening a file the snapshot took out."""
    sandbox, fake = _sandbox(env_box, dind=False)
    sandbox._persistent_env = {"TASK_SETTING": PERSISTENT_VALUE}
    # Other rows' sandboxes share the box and never stop: their folders go
    # first, so the listing below is this sandbox's alone.
    code, _, err = await env_box.exec(
        [
            "sh",
            "-c",
            f"rm -rf {ENV_FOLDERS}* && mkdir -p {Path(CREDENTIAL).parent} && "
            f"printf '{{}}' > {CREDENTIAL} && chown -R agent:agent /home/agent/.claude",
        ]
    )
    assert code == 0, err
    env = {"AGENT_API_KEY": AGENT_VALUE}
    result = await exec_raw(sandbox.exec, "true", user="agent", env=env)
    assert result.return_code == 0, result.stderr
    seen: dict[str, Any] = {}

    async def at_create(name, **kwargs):
        _, listing, _ = await env_box.exec(["ls", "-A", "/opt/runtime/run"])
        seen["listing"] = listing.decode().split()
        seen["sessions"] = await _session_folders(env_box)
        code, _, _ = await env_box.exec(["test", "-e", CREDENTIAL])
        seen["credential"] = code == 0

    take_out_envs: list[Any] = []
    real_sandbox_exec = sandbox._sandbox_exec

    async def sandbox_exec(command, *args, **kwargs):
        take_out_envs.append(kwargs.get("env"))
        return await real_sandbox_exec(command, *args, **kwargs)

    await _snapshotting(sandbox, fake, monkeypatch, at_create)
    uploads_before = len(fake.env_uploads())
    sandbox._sandbox_exec = sandbox_exec
    await sandbox.snapshot("env-file-snap")
    sandbox._sandbox_exec = real_sandbox_exec
    assert seen["credential"] is False
    assert not [n for n in seen["listing"] if n.startswith("env-")], seen["listing"]
    assert seen["sessions"] == [], seen["sessions"]
    assert take_out_envs and all(not env for env in take_out_envs), take_out_envs
    assert len(fake.env_uploads()) == uploads_before
    code, _, _ = await env_box.exec(["test", "-e", CREDENTIAL])
    assert code == 0, "the credential was not put back"
    folder = fake.env_uploads()[0][0].rsplit("/", 1)[0]
    result = await exec_raw(sandbox.exec, "true", user="agent", env=env)
    assert result.return_code == 0, result.stderr
    assert len(fake.env_uploads()) == uploads_before + 1
    assert fake.env_uploads()[-1][0].rsplit("/", 1)[0] != folder
    await env_box.exec(["rm", "-rf", "/home/agent/.claude"])


async def test_an_exec_with_an_env_during_the_snapshot_is_refused(env_box, monkeypatch):
    """Between the env files' removal and the snapshot's end no env file can
    come back: an exec that carries an env is refused and runs nothing."""
    sandbox, fake = _sandbox(env_box, dind=False)
    result = await exec_raw(sandbox.exec, "true", user="agent", env={"A": "1"})
    assert result.return_code == 0, result.stderr
    marker = f"/tmp/during-snapshot-{uuid.uuid4().hex}"
    refused: list[str] = []

    async def at_create(name, **kwargs):
        try:
            await sandbox.exec(f"touch {marker}", user="agent", env={"A": "1"})
        except Exception as exc:
            refused.append(str(exc))

    await _snapshotting(sandbox, fake, monkeypatch, at_create)
    await sandbox.snapshot("env-file-snap")
    assert refused and "snapshot is being taken" in refused[0], refused
    code, _, _ = await env_box.exec(["test", "-e", marker])
    assert code != 0, "the exec ran during the snapshot"
    assert len(fake.env_uploads()) == 1
    # After the snapshot the same exec uploads and runs.
    result = await exec_raw(
        sandbox.exec, f"touch {marker}", user="agent", env={"A": "1"}
    )
    assert result.return_code == 0, result.stderr
    assert len(fake.env_uploads()) == 2


async def test_a_snapshot_is_refused_when_the_env_files_cannot_go(env_box, monkeypatch):
    sandbox, fake = _sandbox(env_box, dind=False)
    result = await exec_raw(sandbox.exec, "true", user="agent", env={"A": "1"})
    assert result.return_code == 0, result.stderr
    created: list[str] = []

    async def at_create(name, **kwargs):
        created.append(name)

    async def refuse(path, recursive=False):
        raise RuntimeError("the toolbox refused the delete")

    await _snapshotting(sandbox, fake, monkeypatch, at_create)
    fake.fs.delete_file = refuse
    with pytest.raises(Exception, match="snapshot is not taken"):
        await sandbox.snapshot("env-file-snap")
    assert created == []


async def test_a_snapshot_is_refused_when_the_session_files_cannot_go(
    env_box, monkeypatch
):
    sandbox, fake = _sandbox(env_box, dind=False)
    result = await exec_raw(sandbox.exec, "true", user="agent", env={"A": "1"})
    assert result.return_code == 0, result.stderr
    created: list[str] = []

    async def at_create(name, **kwargs):
        created.append(name)

    await _snapshotting(sandbox, fake, monkeypatch, at_create)
    toolbox_exec = fake.exec

    async def refuse(command, cwd=None, env=None, timeout=None):
        if SESSIONS in command:
            return SimpleNamespace(exit_code=1, result="refused")
        return await toolbox_exec(command, cwd=cwd, env=env, timeout=timeout)

    fake.process.exec = refuse
    with pytest.raises(Exception, match="session files"):
        await sandbox.snapshot("env-file-snap")
    assert created == []


async def test_a_snapshot_is_refused_when_the_session_folder_is_not_there(
    env_box, monkeypatch
):
    """A daemon that keeps its sessions elsewhere (no folder where Daytona's
    keeps them, which always holds its own entrypoint session) is refused
    before the create, never taken as clean."""
    from benchflow.sandbox import daytona_exec_env

    monkeypatch.setattr(daytona_exec_env, "SESSIONS_DIR", "/root/.daytona/elsewhere")
    sandbox, fake = _sandbox(env_box, dind=False)
    result = await exec_raw(sandbox.exec, "true", user="agent", env={"A": "1"})
    assert result.return_code == 0, result.stderr
    created: list[str] = []

    async def at_create(name, **kwargs):
        created.append(name)

    await _snapshotting(sandbox, fake, monkeypatch, at_create)
    with pytest.raises(Exception, match="session files"):
        await sandbox.snapshot("env-file-snap")
    assert created == []
