"""bfx's helper-volume steps against a fake daemon (M-2, M-3 of the Claude
review of TR-P1).

What must hold:
- No step names a task image. The check and the fill run in containers of a
  helper image made by ``docker import`` from the pinned bytes alone (one
  member, ``bfx``, root 0555), created with ``--pull never``. The filler's
  ``/bfx`` is checked against the pin before it starts. The helper image is
  removed by its label at the end, whatever happened.
- Every step is bounded. A step that runs out is a ``BfxUnavailable`` (which
  the start reports as its own error), and the step's container is removed
  by the name it was given before it was asked for, even when ``create``
  itself never answered.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from benchflow._bfx import delivery, locate, pins
from benchflow._bfx.locate import BfxUnavailable

DATA = b"bfx stand-in x86_64"


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "bfx-x86_64").write_bytes(DATA)
    monkeypatch.setitem(pins.SHA256, "x86_64", hashlib.sha256(DATA).hexdigest())
    monkeypatch.setattr(locate, "BIN_DIR", bindir)
    monkeypatch.setattr(locate, "_checked", {})
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(delivery, "_verified_volumes", set())
    return bindir


class FakeDaemon:
    """A daemon with one helper volume, images made by import, and created
    containers. *hang* names a step (``import``, ``create``, ``cp``,
    ``start``) that never answers."""

    def __init__(self, *, hang: str | None = None, filler: bytes | None = None):
        self.hang = hang
        self.filler = filler
        self.calls: list[list[str]] = []
        self.volume: bytes | None = None
        self.images: dict[str, tuple[bytes, str]] = {}
        self.containers: dict[str, str] = {}

    async def _maybe_hang(self, step: str) -> None:
        if self.hang == step:
            await asyncio.sleep(3600)

    async def run(self, args, *, timeout=120.0):
        self.calls.append(list(args))
        ok = (0, b"", b"")
        if args[:2] == ["volume", "create"]:
            return ok
        if args[:2] == ["volume", "inspect"]:
            label = f'{{"{delivery.VOLUME_LABEL}": "{pins.SHA256["x86_64"]}"}}'
            return 0, f'{{"Driver": "local", "Labels": {label}}}'.encode(), b""
        if args[0] == "create":
            await self._maybe_hang("create")
            name = args[args.index("--name") + 1]
            image = args[args.index("--mount") + 2]
            assert image in self.images, f"create of {image}, not the helper"
            assert args[args.index("--pull") + 1] == "never"
            self.containers[name] = image
            return ok
        if args[:2] == ["start", "-a"]:
            await self._maybe_hang("start")
            self.volume = self.images[self.containers[args[2]]][0]
            return ok
        if args[:2] == ["rm", "-f"]:
            self.containers.pop(args[2], None)
            return ok
        if args[:2] == ["image", "ls"]:
            want = args[-1].split("=", 2)[2]
            found = [i for i, (_, label) in self.images.items() if label == want]
            return 0, "\n".join(found).encode(), b""
        if args[:2] == ["rmi", "--force"]:
            for image in args[2:]:
                self.images.pop(image, None)
            return ok
        raise AssertionError(args)

    async def import_image(self, archive: Path, changes, *, timeout):
        self.calls.append(["import", *changes])
        await self._maybe_hang("import")
        with tarfile.open(fileobj=io.BytesIO(archive.read_bytes())) as tar:
            [member] = tar.getmembers()
            assert (member.name, member.mode, member.uid, member.gid) == (
                "bfx",
                0o555,
                0,
                0,
            )
            data = tar.extractfile(member).read()
        assert 'ENTRYPOINT ["/bfx"]' in changes
        label = next(c.split("=", 1)[1] for c in changes if c.startswith("LABEL "))
        image = "sha256:" + hashlib.sha256(label.encode()).hexdigest()
        self.images[image] = (self.filler or data, label)
        return 0, f"{image}\n".encode(), b""

    async def folder_names(self, container, path):
        await self._maybe_hang("cp")
        return ["bfx"] if self.volume is not None else []

    async def file_digest(self, container, path):
        await self._maybe_hang("cp")
        if path == "/bfx":
            return hashlib.sha256(
                self.images[self.containers[container]][0]
            ).hexdigest()
        return None if self.volume is None else hashlib.sha256(self.volume).hexdigest()


def _ensure(daemon: FakeDaemon) -> str:
    return asyncio.run(
        delivery.ensure_volume(
            [], None, arch="x86_64", daemon_key="fake", transport=daemon
        )
    )


def test_the_fill_uses_only_the_helper_image_and_removes_it(stand_in):
    daemon = FakeDaemon()
    assert _ensure(daemon) == delivery.volume_name("x86_64")
    assert daemon.volume == DATA
    assert daemon.images == {} and daemon.containers == {}
    creates = [c for c in daemon.calls if c[0] == "create"]
    assert len(creates) == 3  # check, fill, check again


def test_a_filler_whose_bfx_is_not_the_pin_is_never_started(stand_in):
    daemon = FakeDaemon(filler=b"other bytes")
    with pytest.raises(BfxUnavailable, match="was not started"):
        _ensure(daemon)
    assert not [c for c in daemon.calls if c[:2] == ["start", "-a"]]
    assert daemon.volume is None and daemon.images == {}


@pytest.mark.parametrize("step", ["import", "create", "cp", "start"])
def test_a_step_that_runs_out_is_bfx_unavailable_and_cleaned_up(
    stand_in, monkeypatch, step
):
    monkeypatch.setattr(delivery, "STEP_TIMEOUT", 0.2)
    daemon = FakeDaemon(hang=step)
    with pytest.raises(BfxUnavailable, match="did not finish within"):
        _ensure(daemon)
    # Every container a step named is removed by that name, including one
    # whose create never answered; the helper image is removed by its label.
    named = [c[c.index("--name") + 1] for c in daemon.calls if c[0] == "create"]
    removed = [c[2] for c in daemon.calls if c[:2] == ["rm", "-f"]]
    assert named == removed
    assert daemon.containers == {} and daemon.images == {}
