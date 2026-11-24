"""Modal's bfx route (rows M1-M5).

Modal starts a program by argv (``Sandbox.exec(*args)``), with no shell, so
bfx on its read-only Volume at /.bfx is started with no program of the image
in between. Before the agent's first byte the delivery is checked:

- the Volume API read-back (``delivery.modal_volume``, outside the sandbox);
- bfx's own report: the pin and x86_64;
- through Modal's own fs tool, never the image's: the exec user is root (uid
  0), and root's bounding set (CapBnd);
- a root write into /.bfx must fail, so the read-only mount took effect.

The live run (MR, 11-18) found ``/.bfx`` a link in the sandbox's own
filesystem to the volume's mount, and root with CAP_SYS_PTRACE: so bfx on
Modal holds against a non-root agent only. Since that run the route is on
(``delivery.PROVEN_LIVE``); ``BENCHFLOW_BFX_DELIVERY=off`` turns it off.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from types import SimpleNamespace

import pytest

from benchflow._bfx import client as bfx_client
from benchflow._bfx import pins
from benchflow._bfx.locate import SANDBOX_PATH, BfxUnavailable
from benchflow.sandbox import _verifier_links

pytestmark = pytest.mark.bfx_refusal

#: /proc/self/status lines as Modal's fs tool would read them: Docker's
#: default bounding set (no SYS_ADMIN, no SYS_PTRACE).
DOCKER_DEFAULT = "00000000a80425fb"
SYS_ADMIN = f"{0xA80425FB | (1 << 21):016x}"
SYS_PTRACE = f"{0xA80425FB | (1 << 19):016x}"


def status(uid: int = 0, capbnd: str | None = DOCKER_DEFAULT) -> bytes:
    lines = [
        "Name:\tmodal-sandbox-fs-tools",
        f"Uid:\t{uid}\t{uid}\t{uid}\t{uid}",
        f"Gid:\t{uid}\t{uid}\t{uid}\t{uid}",
        "CapInh:\t0000000000000000",
    ]
    if capbnd is not None:
        lines.append(f"CapBnd:\t{capbnd}")
    return ("\n".join(lines) + "\n").encode()


class _Aio:
    def __init__(self, fn):
        self.aio = fn


async def _value(v):
    return v


class FakeProcess:
    """A Modal ContainerProcess: stdin written, stdout read, an exit code."""

    def __init__(self, out: str = "", code: int = 0, run=None):
        self.written = bytearray()
        self.eof = False
        self._out = out
        self._code = code
        self._run = run
        self.stdin = SimpleNamespace(
            write=self._write, write_eof=self._write_eof, drain=_Aio(self._drain)
        )
        self.stdout = SimpleNamespace(read=_Aio(self._read_out))
        self.stderr = SimpleNamespace(read=_Aio(lambda: _value("")))
        self.wait = _Aio(self._wait)

    def _write(self, data):
        self.written += data.encode() if isinstance(data, str) else bytes(data)

    def _write_eof(self):
        self.eof = True

    async def _drain(self):
        return None

    async def _read_out(self):
        if self._run is not None:
            self._out, self._code = await self._run(bytes(self.written))
        return self._out

    async def _wait(self):
        return self._code


def modal_box(
    *,
    delivery: str | None = None,
    version: dict | None = None,
    status_bytes: bytes | Exception | None = None,
    write_ok: bool = False,
    answer: dict | None = None,
):
    """A ModalSandbox whose sandbox handle is a fake: exec answers bfx's
    version and one call; the fs tool reads /proc/self/status and refuses
    (or takes) the write probe."""
    from benchflow.sandbox.modal_impl import ModalSandbox

    if status_bytes is None:
        status_bytes = status()
    box = object.__new__(ModalSandbox)
    box.logger = logging.getLogger("modal-test")
    box.bfx_delivery = delivery
    seen = SimpleNamespace(execs=[], processes=[], writes=[], removed=[], reads=[])
    report = {"name": "bfx", "sha256": pins.SHA256["x86_64"], "arch": "x86_64"}
    report.update(version or {})

    async def exec_aio(*argv, **kwargs):
        seen.execs.append((argv, kwargs))
        if argv[1:] == ("version",):
            process = FakeProcess(json.dumps(report) + "\n")
        else:
            body = {
                "ok": True,
                "result": answer if answer is not None else {"entries": []},
            }
            process = FakeProcess(json.dumps(body) + "\n")
        seen.processes.append(process)
        return process

    async def read_bytes(path):
        seen.reads.append(path)
        if isinstance(status_bytes, Exception):
            raise status_bytes
        return status_bytes

    async def write_bytes(data, path):
        seen.writes.append(path)
        if not write_ok:
            raise PermissionError(f"{path}: Read-only file system")

    async def remove(path, recursive=False):
        seen.removed.append(path)

    box._sandbox = SimpleNamespace(
        exec=_Aio(exec_aio),
        filesystem=SimpleNamespace(
            read_bytes=_Aio(read_bytes),
            write_bytes=_Aio(write_bytes),
            remove=_Aio(remove),
        ),
    )
    return box, seen


async def test_m1_bfx_call_starts_bfx_by_argv_with_its_arguments_on_stdin():
    """M1. One bfx command as root: Modal's exec of ``/.bfx/bfx <cmd>
    --json -`` (argv, no shell, no secrets, no env), the arguments as its
    first stdin line, the answer parsed. Without a delivery, nothing runs.

    Mutant: the call routed through the backend's exec (a bash -c)."""
    box, seen = modal_box(delivery="volume", answer={"entries": [{"path": "/x"}]})
    args = {"paths": ["/x"], "acl": False}
    got = await box.bfx_call("scan", args, timeout_sec=7)
    assert got == {"entries": [{"path": "/x"}]}
    ((argv, kwargs),) = seen.execs
    assert argv == (SANDBOX_PATH, "scan", "--json", "-")
    assert set(kwargs) <= {"timeout", "text"}, kwargs
    assert bytes(seen.processes[0].written) == bfx_client.stdin_args_line(args)
    assert seen.processes[0].eof

    none, nothing = modal_box()
    with pytest.raises(BfxUnavailable):
        await none.bfx_call("scan", args, timeout_sec=7)
    assert nothing.execs == []


@pytest.mark.parametrize(
    "case",
    ["ok", "write-allowed", "uid-1000", "arch-aarch64", "status-unreadable"],
)
async def test_m2_the_walker_runs_only_after_every_volume_check(case):
    """M2. Before the agent: the walker can run only when bfx reports the
    pin and x86_64, the exec user is root, and a root write into /.bfx is
    refused. A probe that was written is removed again.

    Mutants: the write probe dropped; the uid check dropped; the arch not
    checked."""
    kwargs = {
        "ok": {},
        "write-allowed": {"write_ok": True},
        "uid-1000": {"status_bytes": status(uid=1000)},
        "arch-aarch64": {"version": {"arch": "aarch64"}},
        "status-unreadable": {"status_bytes": PermissionError("no /proc")},
    }[case]
    box, seen = modal_box(**kwargs)
    await box._check_bfx()
    if case == "ok":
        assert box.bfx_delivery == "volume"
        assert box._bfx_arch == "x86_64"
        assert _verifier_links.walker_unavailable(box) is None
        assert seen.writes and seen.reads == ["/proc/self/status"]
    else:
        assert box.bfx_delivery is None, case
        assert _verifier_links.walker_unavailable(box) is not None
    if case == "write-allowed":
        assert seen.removed == seen.writes


async def test_m3_modal_holds_against_a_non_root_agent_only():
    """M3, revised by the live run MR (11-18). On Modal, ``/.bfx`` is a link
    in the sandbox's own filesystem to the read-only volume's mount
    (``/.bfx -> /__modal/volumes/vo-…``), which root may replace, and root
    there has CAP_SYS_PTRACE (CapBnd a80c05fb). So whatever root's bounding
    set, bfx on Modal holds against a non-root agent only.

    Mutant: root held on Modal."""
    for capbnd in (DOCKER_DEFAULT, SYS_ADMIN, SYS_PTRACE, None):
        box, _ = modal_box(status_bytes=status(capbnd=capbnd))
        await box._check_bfx()
        assert box.bfx_delivery == "volume", capbnd
        assert box.bfx_guarantee == "non-root-agent", (capbnd, box.bfx_guarantee)


async def test_m8_the_sandbox_runs_use_mounts_and_checks_bfx_at_creation(
    monkeypatch, tmp_path
):
    """M8 (found by the live run MR, 11-18). Runs start Modal sandboxes as
    ``BenchFlowModalSandbox`` (sandbox/setup.py), whose own ``start`` never
    reached ModalSandbox.start's bfx volume: with the switch on, no trial
    got bfx. The volume now goes in, and is checked, where every start makes
    the sandbox (``_create_sandbox``).

    Mutant: the volume left out of ``_create_sandbox``."""
    import modal

    from benchflow._bfx import delivery
    from benchflow.sandbox.setup import _create_benchflow_modal_environment_class

    monkeypatch.setenv(delivery.PROVIDER_ENV, "on")
    volume = object()
    monkeypatch.setattr(delivery, "modal_volume", lambda modal, arch="x86_64": volume)
    fake, seen = modal_box()
    made = []

    async def create(**kwargs):
        made.append(kwargs)
        return fake._sandbox

    monkeypatch.setattr(modal.Sandbox, "create", SimpleNamespace(aio=create))
    cls = _create_benchflow_modal_environment_class()
    box = object.__new__(cls)
    box.logger = logging.getLogger("modal-test")
    box.session_id = "m8"
    box._app = box._image = None
    box._sandbox_timeout = 60
    box._sandbox_idle_timeout = None
    box.task_env_config = SimpleNamespace(
        cpus=1, memory_mb=512, allow_internet=True, docker_image=None
    )
    # _create_sandbox reads the image's env from the task's Dockerfile (#1245).
    box.environment_dir = tmp_path
    (tmp_path / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    box._persistent_env = {}
    sandbox = await box._create_sandbox(
        gpu_config=None, secrets_config=[], volumes_config={"/data": "user-volume"}
    )
    assert sandbox is fake._sandbox
    (kwargs,) = made
    assert kwargs["volumes"] == {"/data": "user-volume", "/.bfx": volume}
    assert box.bfx_delivery == "volume", box.bfx_unavailable_why
    assert seen.execs[0][0] == (SANDBOX_PATH, "version")

    monkeypatch.delenv(delivery.PROVIDER_ENV)
    monkeypatch.setenv(delivery.DELIVERY_ENV, "off")
    made.clear()
    off = object.__new__(cls)
    off.__dict__.update({k: v for k, v in box.__dict__.items() if k != "bfx_delivery"})
    await off._create_sandbox(gpu_config=None, secrets_config=[], volumes_config={})
    assert made[0]["volumes"] == {}
    assert off.bfx_delivery is None
    assert off.bfx_unavailable_why is None


def _real_bfx():
    from benchflow._bfx import locate

    try:
        return locate.verified_binary("x86_64")
    except BfxUnavailable as exc:
        pytest.skip(f"no pinned bfx in this checkout: {exc}")


def local_box(prefix: list[str] | None = None):
    """A ModalSandbox whose exec runs argv on this machine: ``/.bfx/bfx`` is
    the pinned binary, started with *prefix* (``[]``, or a gVisor runsc
    ``do``)."""
    from benchflow.sandbox.modal_impl import ModalSandbox

    binary = str(_real_bfx())
    box = object.__new__(ModalSandbox)
    box.logger = logging.getLogger("modal-local")
    box.bfx_delivery = "volume"
    starts = []

    async def exec_aio(*argv, **kwargs):
        assert argv[0] == SANDBOX_PATH
        command = [*(prefix or []), binary, *argv[1:]]
        starts.append(command)

        async def run(stdin: bytes):
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, _err = await process.communicate(stdin)
            return out.decode(), process.returncode

        return FakeProcess(run=run)

    box._sandbox = SimpleNamespace(exec=_Aio(exec_aio))
    return box, starts


async def test_m5_the_pinned_binary_walks_and_removes_through_the_route(tmp_path):
    """M5. The pinned bfx itself, started through ModalSandbox.bfx_call (the
    fake exec runs its argv here): the walk finds a planted link into the
    verifier's folder, and bfx removes the link, not its target.

    Mutant: the remove's root and path swapped."""
    box, starts = local_box()
    (tmp_path / "app").mkdir()
    (tmp_path / "target").write_text("1")
    os.symlink("/logs/verifier/reward.txt", tmp_path / "app" / "grade.log")
    answer = await box.bfx_call(
        "scan",
        {
            "walks": [{"root": str(tmp_path), "depth": 64}],
            "mounts": False,
            "acl": False,
        },
        timeout_sec=30,
    )
    links = {
        e.get("path"): e.get("link")
        for e in answer.get("entries") or ()
        if e.get("type") == "link"
    }
    assert links == {str(tmp_path / "app" / "grade.log"): "/logs/verifier/reward.txt"}
    await box.bfx_call(
        "remove", {"root": str(tmp_path), "path": "app/grade.log"}, timeout_sec=30
    )
    assert not os.path.lexists(tmp_path / "app" / "grade.log")
    assert (tmp_path / "target").read_text() == "1"
    assert all(s[0].endswith("bfx-x86_64") for s in starts)


def test_the_pins_are_the_binary_this_checkout_ships():
    """A control for M5/M6: the binary they start is the pinned one."""
    binary = _real_bfx()
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == pins.SHA256["x86_64"]
