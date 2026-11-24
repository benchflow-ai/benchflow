"""BenchFlow's bfx on Daytona native, and Daytona DinD's default,
through a fake exec stream (row Z).

Daytona's toolbox starts every command through the image's shell
(``sandbox/daytona.py``, the toolbox note), so the route holds against a
non-root agent only. A program of the image's that checks the binary adds
nothing against root, and an image without one failed every call. So these
rows ask for:

- one delivery before the turn into a fresh root-only folder, and another
  once the turn is over;
- each upload read back through the toolbox's file API and hashed on the
  host, then bfx's own report of its hash and arch, and its folder's owner and
  mode from bfx's own look (no link followed);
- any failure of that before the turn leaves no delivery, so the trial is
  refused before its turn (Q4);
- every later call starting the folder's bfx by path, with no check program.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shlex
import types
from pathlib import Path

import pytest

from benchflow._bfx import locate, pins
from benchflow._bfx.locate import BfxUnavailable
from benchflow.sandbox import _verifier_links
from benchflow.sandbox._bounded_exec import end_agent_turn

pytestmark = pytest.mark.bfx_refusal

PINNED = b"\x7fELF pinned stand-in"


@pytest.fixture(autouse=True)
def pinned(tmp_path, monkeypatch):
    binary = tmp_path / "bfx-pinned"
    binary.write_bytes(PINNED)
    monkeypatch.setattr(locate, "verified_binary", lambda arch: binary)
    monkeypatch.setitem(pins.SHA256, "x86_64", hashlib.sha256(PINNED).hexdigest())
    return binary


class Result(types.SimpleNamespace):
    pass


def _segments(command: str) -> list[list[str]]:
    """The commands of a ``&&`` list, each split as the shell would; the
    rc5 per-call check's prefix (``…; <bfx command>``) is dropped."""
    out = []
    for part in command.split(" && "):
        if "sha256sum" in part:
            part = part.rsplit("; ", 1)[-1]
        try:
            out.append(shlex.split(part))
        except ValueError:
            out.append([part])
    return out


def runs_bfx(command: str) -> bool:
    """Whether *command* starts a program named bfx."""
    return any(words and words[0].endswith("/bfx") for words in _segments(command))


class FakeDaytona:
    """A Daytona sandbox's bfx methods on a fake: the root shell runs what
    it is given, as an image would; the toolbox keeps each upload by path and
    reads it back on download."""

    def __init__(
        self,
        *,
        machine: str = "x86_64",
        sha_rc: int = 0,
        version_rc: int = 0,
        version_sha: str | None = None,
        version_arch: str | None = None,
        owner: int = 0,
        file_mode: int = 0o500,
        folder_mode: int = 0o700,
        run_mode: int = 0o755,
        tamper: bytes | None = None,
    ) -> None:
        from benchflow.sandbox.daytona import DaytonaSandbox

        self.logger = logging.getLogger("z")
        self.bfx_delivery = None
        self.machine = machine
        self.sha_rc = sha_rc
        self.version_rc = version_rc
        self.version_sha = version_sha
        self.version_arch = version_arch
        self.owner = owner
        self.file_mode = file_mode
        self.folder_mode = folder_mode
        self.run_mode = run_mode
        self.tamper = tamper
        self.files: dict[str, bytes] = {}
        self.calls: list[str] = []
        self.uploads: list[str] = []
        self.downloads: list[str] = []
        for name, value in vars(DaytonaSandbox).items():
            if "bfx" in name and callable(value) and not isinstance(value, property):
                setattr(self, name, types.MethodType(value, self))

    async def exec(self, command, user=None, timeout_sec=None, **_):
        self.calls.append(command)
        if command == "uname -m":
            return Result(stdout=self.machine + "\n", stderr="", return_code=0)
        if "sha256sum" in command and self.sha_rc:
            return Result(stdout="", stderr="sha256sum: not found", return_code=98)
        if not runs_bfx(command):
            return Result(stdout="", stderr="", return_code=0)
        out = []
        for words in _segments(command):
            if not words or not words[0].endswith("/bfx"):
                continue
            path = words[0]
            if path not in self.files:
                return Result(stdout="", stderr=f"{path}: not found", return_code=127)
            if words[1] == "version":
                if self.version_rc:
                    return Result(
                        stdout="",
                        stderr="Permission denied",
                        return_code=self.version_rc,
                    )
                out.append(
                    json.dumps(
                        {
                            "name": "bfx",
                            "sha256": self.version_sha or pins.SHA256["x86_64"],
                            "arch": self.version_arch or self.machine,
                        }
                    )
                )
            else:
                args = json.loads(words[3])
                out.append(json.dumps({"ok": True, "result": self._answer(path, args)}))
        return Result(stdout="\n".join(out) + "\n", stderr="", return_code=0)

    def _answer(self, path: str, args: dict) -> dict:
        folder = path.rsplit("/", 1)[0]
        asked = args.get("paths") or []
        if path not in asked and folder not in asked:
            return {"entries": []}
        entries = [
            {"path": "/", "type": "dir", "uid": 0, "mode": 0o755, "exists": True},
            {
                "path": "/run",
                "type": "dir",
                "uid": 0,
                "mode": self.run_mode,
                "exists": True,
            },
            {
                "path": folder,
                "type": "dir",
                "uid": self.owner,
                "mode": self.folder_mode,
                "exists": True,
            },
            {
                "path": path,
                "type": "file",
                "uid": self.owner,
                "mode": self.file_mode,
                "exists": True,
            },
        ]
        return {"entries": entries, "errors": []}

    async def upload_file(self, source, target, *, mode=None, admission=None):
        self.uploads.append(target)
        self.files[target] = Path(source).read_bytes()

    async def download_file(self, source, target):
        self.downloads.append(source)
        if source not in self.files:
            raise FileNotFoundError(source)
        data = self.tamper if self.tamper is not None else self.files[source]
        Path(target).write_bytes(data)


async def _scan(box):
    return await box.bfx_call("scan", {"paths": ["/app"]}, timeout_sec=5)


async def test_z0_the_delivery_is_made_and_checked_before_the_turn():
    """Z0. x86_64 with a pinned bfx: the plan delivers it before the turn,
    into a fresh root-only folder, reads it back and asks it for its report;
    the walker can run. An arch without a pinned bfx: nothing is uploaded and
    the trial is refused before its turn.

    Mutant: the plan back to the arch alone (no upload before the turn)."""
    box = FakeDaytona()
    await box.plan_bfx_upload()
    assert box.bfx_delivery == "per-call", box.calls
    assert len(box.uploads) == 1 and box.uploads[0].startswith("/run/benchflow-bfx-")
    assert box.downloads == box.uploads
    assert _verifier_links.walker_unavailable(box) is None
    other = FakeDaytona(machine="riscv64")
    await other.plan_bfx_upload()
    assert other.bfx_delivery is None and other.uploads == []
    assert _verifier_links.walker_unavailable(other) is not None


async def test_z1_no_program_of_the_image_checks_bfx():
    """Z1. No exec names ``sha256sum``; every bfx that starts is the
    folder's own, by its path; a call's exec is that bfx and nothing before
    it.

    Mutant: the rc5 ``sha256sum`` prefix put back."""
    box = FakeDaytona()
    await box.plan_bfx_upload()
    await _scan(box)
    assert box.bfx_delivery == "per-call"
    assert not [c for c in box.calls if "sha256sum" in c], box.calls
    path = box._bfx_path
    runs = [c for c in box.calls if runs_bfx(c)]
    assert runs, box.calls
    started = [w[0] for c in runs for w in _segments(c) if w and w[0].endswith("/bfx")]
    assert started and set(started) == {path}, runs
    assert shlex.split(runs[-1])[:3] == [path, "scan", "--json"], runs[-1]


async def test_z2_a_read_back_that_differs_never_runs_bfx():
    """Z2. The bytes the toolbox reads back are not the pin's: no delivery
    before the turn, and nothing in the folder is started. After the turn the
    fresh copy's read-back differs: the call is refused (bfx unavailable) and
    nothing in the new folder is started.

    Mutants: the read-back skipped; compared with another arch's pin."""
    box = FakeDaytona(tamper=b"not the pinned bytes")
    await box.plan_bfx_upload()
    assert box.downloads, "the upload was never read back"
    assert box.bfx_delivery is None
    assert not [c for c in box.calls if runs_bfx(c)], box.calls

    good = FakeDaytona()
    await good.plan_bfx_upload()
    assert good.bfx_delivery == "per-call"
    end_agent_turn(good)
    good.tamper = b"swapped"
    seen = len(good.calls)
    with pytest.raises(BfxUnavailable):
        await _scan(good)
    assert not [c for c in good.calls[seen:] if runs_bfx(c)], good.calls[seen:]


async def test_z3_a_bfx_that_cannot_start_is_refused_before_the_turn():
    """Z3. The uploaded bfx can't start here (a noexec /run: exit 126): no
    delivery, so the trial is refused before its turn.

    Mutant: the plan back to the arch alone."""
    box = FakeDaytona(version_rc=126)
    await box.plan_bfx_upload()
    assert box.bfx_delivery is None
    assert _verifier_links.walker_unavailable(box) is not None


async def test_z4_an_image_without_sha256sum_still_walks():
    """Z4. An image with no ``sha256sum``: bfx still runs and answers.

    Mutant: as Z1."""
    box = FakeDaytona(sha_rc=98)
    await box.plan_bfx_upload()
    assert box.bfx_delivery == "per-call"
    assert await _scan(box) == {"entries": []}


async def test_z_bfx_is_delivered_afresh_after_the_turn():
    """Z. One copy before the turn (calls reuse it), a fresh folder once the
    turn is over; each copy is read back once.

    Mutant: the folder kept across the turn."""
    box = FakeDaytona()
    await box.plan_bfx_upload()
    await _scan(box)
    await _scan(box)
    assert len(box.uploads) == 1
    end_agent_turn(box)
    await _scan(box)
    assert len(box.uploads) == 2 and box.uploads[1] != box.uploads[0]
    assert box.downloads == box.uploads
    assert shlex.split(box.calls[-1])[0] == box.uploads[1]


@pytest.mark.parametrize(
    "report",
    [{"version_sha": "0" * 64}, {"version_arch": "aarch64"}],
    ids=["other-hash", "other-arch"],
)
async def test_z_a_bfx_that_reports_another_hash_or_arch_is_refused(report):
    """Z. bfx's own report names another hash or arch: no delivery.

    Mutant: the report not checked."""
    box = FakeDaytona(**report)
    await box.plan_bfx_upload()
    assert box.bfx_delivery is None


@pytest.mark.parametrize(
    "folder",
    [
        {"owner": 1000},
        {"file_mode": 0o555},
        {"folder_mode": 0o755},
        {"run_mode": 0o777},
    ],
    ids=["not-root", "file-0555", "folder-0755", "run-writable"],
)
async def test_z_a_bfx_folder_others_could_change_is_refused(folder):
    """Z. bfx's own look at its folder: not root's, not 0700/0500, or a
    folder on the way others may write (no sticky bit): no delivery.

    Mutant: the owner and mode not checked."""
    box = FakeDaytona(**folder)
    await box.plan_bfx_upload()
    assert box.bfx_delivery is None
