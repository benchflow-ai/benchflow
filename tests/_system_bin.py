"""The host's /usr/bin and /bin as links, built once per test worker.

Tests that run a shell script against stub firewall tools put these links
after their stubs on PATH, leaving out the host tools the script would find
first. Built in each test's own folder, the links were about 1,150 files per
test on Ubuntu (more on a GitHub runner image) for about 2,900 tests, and
pytest keeps every test's folder: 72% of the unit suite's files, enough to run
CI's runner out of inodes partway through the suite.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

SOURCES = ("/usr/bin", "/bin")


def system_bin(
    tmp_path: Path,
    hidden: Iterable[str],
    hidden_prefixes: tuple[str, ...],
    *,
    mode: int | None = None,
    sources: tuple[str, ...] = SOURCES,
    executables_only: bool = False,
) -> Path:
    """The links folder for this filter, beside ``tmp_path`` in the worker's
    temporary root, built on first use (in a staging folder, then renamed,
    so an interrupted build is never used). Tests only read it.

    The first source holding a name wins; ``executables_only`` keeps only
    executable files."""
    names = frozenset(hidden)
    spec = repr((sorted(names), hidden_prefixes, mode, sources, executables_only))
    digest = hashlib.sha256(spec.encode()).hexdigest()[:12]
    folder = tmp_path.parent / f"sysbin-shared-{digest}"
    if folder.is_dir():
        return folder
    staging = folder.with_name(f".{folder.name}-building")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    if mode is not None:
        staging.chmod(mode)
    for d in sources:
        for src in Path(d).iterdir():
            name = src.name
            if name in names or name.startswith(hidden_prefixes):
                continue
            # A link to a folder is no tool, and one can loop: Debian's
            # /usr/bin/X11 is a link to "." (GitHub's runner image has it),
            # so a walk that follows links would go X11/X11/... without end.
            if src.is_dir():
                continue
            if executables_only and not (src.is_file() and os.access(src, os.X_OK)):
                continue
            if not (staging / name).is_symlink():
                (staging / name).symlink_to(src)
    staging.rename(folder)
    return folder
