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
different homes on one daemon still see each other. Registration and the
last holder's removal happen under one ``flock`` per image, so a new holder
waits for an image removal in progress and then builds afresh. A holder
whose process is gone (:func:`benchflow.sandbox.leases.lease_state`) does
not count. A daemon reached through ``DOCKER_HOST`` gets its own namespace.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import os
import re
from collections.abc import Iterator
from pathlib import Path

HOLDER_DIR_ENV = "BENCHFLOW_IMAGE_HOLDER_DIR"
_DEFAULT_DIR = Path("/tmp/benchflow-image-holders")
_counter = itertools.count(1)

try:  # POSIX only; elsewhere holders are recorded but not locked.
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]


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
    """One rollout's claim on an image; see the module docstring."""

    def __init__(self, image: str) -> None:
        from benchflow.sandbox.leases import lease_token

        self.image = image
        self.folder = holder_root() / _key(image)
        token = lease_token().replace(":", "~")
        self.path = self.folder / f"{token}~{next(_counter)}"
        self._lock_fd: int | None = None

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

    def discard(self) -> None:
        """Stop holding the image without removing it."""
        with contextlib.suppress(OSError):
            self.path.unlink()

    def acquire(self) -> None:
        """Register as a holder (blocks while the last holder removes the image)."""
        with self._locked():
            self.path.write_text(self.image)

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

    def begin_release(self) -> bool:
        """Drop this holder and lock the image; True when nobody else holds it.

        The lock stays held until :meth:`end_release`, so no new holder can
        register (and start building) while the caller removes the image.
        """
        self.folder.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.folder / ".lock", os.O_RDONLY | os.O_CREAT, 0o644)
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        self._lock_fd = fd
        with contextlib.suppress(OSError):
            self.path.unlink()
        return not self._other_live_holders()

    def end_release(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
