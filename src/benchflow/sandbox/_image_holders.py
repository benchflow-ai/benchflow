"""Who is using a Docker image on this machine, so teardown removes it only when nobody is.

Every Docker rollout of a task builds (or pulls) one image, ``bf__<task>`` or
the task's ``docker_image``, and its teardown runs ``compose down --rmi all``.
When several rollouts of one task share a daemon (a GRPO group of eight, its
retries, two evaluations of one task set), the first teardown used to delete
the image while a sibling was between build and ``compose up``, and the
sibling failed at sandbox start.

A rollout now registers as a *holder* of its image before building and, at
teardown, removes the image only when no other live holder is left. Holders
are files under one machine-wide folder (``/tmp/benchflow-image-holders``,
or ``$BENCHFLOW_IMAGE_HOLDER_DIR``), not under ``$HOME``, so processes with
different homes on one daemon still see each other. The last holder marks
the removal with a ``.removing`` file (checked and written under a short
``flock`` per image), so a new holder waits for a removal in progress and
then builds afresh. A holder or remover whose process is gone (:func:`benchflow.sandbox.leases.lease_state`) does
not count. A daemon reached through ``DOCKER_HOST`` gets its own namespace.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import itertools
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

HOLDER_DIR_ENV = "BENCHFLOW_IMAGE_HOLDER_DIR"
_DEFAULT_DIR = Path("/tmp/benchflow-image-holders")
_counter = itertools.count(1)

fcntl: Any
try:  # POSIX only; elsewhere holders are recorded but not locked.
    fcntl = importlib.import_module("fcntl")
except ImportError:  # pragma: no cover - Windows
    fcntl = None


def holder_root() -> Path:
    configured = os.environ.get(HOLDER_DIR_ENV)
    root = Path(configured) if configured else _DEFAULT_DIR
    if not root.is_dir():
        root.mkdir(parents=True, exist_ok=True)
        if not configured:
            # Shared by every user of the daemon on this machine: sticky and
            # world-writable, like /tmp itself.
            with contextlib.suppress(OSError):
                root.chmod(0o1777)
    return root


def _key(image: str) -> str:
    daemon = os.environ.get("DOCKER_HOST") or "local"
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", image)[:80]
    digest = hashlib.sha256(f"{daemon}\0{image}".encode()).hexdigest()[:12]
    return f"{readable}-{digest}"


class ImageHold:
    """One rollout's claim on an image; see the module docstring.

    Every method holds the per-image ``flock`` only for a few file operations,
    never across a ``docker`` command or an ``await``, so the methods are safe
    to call on the event loop and a cancelled caller cannot leave the lock
    held. An image removal in progress is marked by a ``.removing`` file
    naming the remover; new holders wait for it (:meth:`acquire`).
    """

    def __init__(self, image: str) -> None:
        from benchflow.sandbox.leases import lease_token

        self.image = image
        self.folder = holder_root() / _key(image)
        self._token = lease_token()
        self.path = self.folder / f"{self._token.replace(':', '~')}~{next(_counter)}"
        self._marker_owner: str | None = None

    @property
    def _marker(self) -> Path:
        return self.folder / ".removing"

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self.folder.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            self.folder.chmod(0o1777)
        fd = os.open(self.folder / ".lock", os.O_RDONLY | os.O_CREAT, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)  # closing the descriptor releases the flock

    def _removal_in_progress(self) -> bool:
        """A live process is removing the image (call under the lock)."""
        from benchflow.sandbox.leases import lease_state

        try:
            owner = self._marker.read_text().split("\n", 1)[0].strip()
        except OSError:
            return False
        if lease_state(owner) == "gone":
            with contextlib.suppress(OSError):
                self._marker.unlink()
            return False
        return True

    def _other_live_holders(self) -> list[Path]:
        from benchflow.sandbox.leases import lease_state

        others: list[Path] = []
        for path in self.folder.iterdir():
            if path.name.startswith(".") or path == self.path:
                continue
            token = ":".join(path.name.split("~")[:4])
            if lease_state(token) == "gone":
                with contextlib.suppress(OSError):
                    path.unlink()
                continue
            others.append(path)
        return others

    def try_acquire(self) -> bool:
        """Register as a holder unless the image is being removed right now."""
        with self._locked():
            if self._removal_in_progress():
                return False
            self.path.write_text(self.image)
            return True

    async def acquire(self, *, wait_sec: float = 300.0) -> None:
        """Register as a holder, waiting out an image removal in progress.

        After ``wait_sec`` the holder registers anyway: a remover that hangs
        must not stop every rollout of the task from starting.
        """
        import asyncio
        import time

        deadline = time.monotonic() + wait_sec
        while not self.try_acquire():
            if time.monotonic() > deadline:
                with self._locked():
                    self.path.write_text(self.image)
                return
            await asyncio.sleep(1.0)

    def discard(self) -> None:
        """Stop holding the image without removing it."""
        with contextlib.suppress(OSError):
            self.path.unlink()

    def begin_release(self) -> bool:
        """Stop holding the image; True when the caller should remove it.

        True only when no other live process holds the image and nobody is
        already removing it. The caller then owns the ``.removing`` marker
        until :meth:`end_release`, and new holders wait for it.
        """
        with self._locked():
            with contextlib.suppress(OSError):
                self.path.unlink()
            if self._removal_in_progress() or self._other_live_holders():
                return False
            self._marker.write_text(self._token + "\n" + self.image)
            self._marker_owner = self._token
            return True

    def end_release(self) -> None:
        """The removal is over (done or failed): let new holders in."""
        if self._marker_owner is None:
            return
        with self._locked():
            try:
                owner = self._marker.read_text().split("\n", 1)[0].strip()
            except OSError:
                owner = None
            if owner == self._marker_owner:
                with contextlib.suppress(OSError):
                    self._marker.unlink()
        self._marker_owner = None
