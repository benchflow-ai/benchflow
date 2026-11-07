"""Shared helpers for the benchflow.taskmd tests (task.md draft 2)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "taskmd"
EXAMPLES = FIXTURES / "examples"
GOLDEN = FIXTURES / "golden"
VECTORS = FIXTURES / "vectors" / "judge-prompt-1"


def emulate_kept_tar(root: Path, command: str) -> str | None:
    """Model the bounded capture protocol in folder-backed sandbox stand-ins."""
    from benchflow.taskmd.verify import COPY_TAR

    found = re.search(r"tar -C / -cf - '?\./([^' ]+)'? ", command)
    if found is None:
        return None
    relative = found.group(1)
    archive = root / COPY_TAR.lstrip("/")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w", dereference=False) as output:
        output.add(root / relative, arcname=relative)
    return f"BF copy {archive.stat().st_size} 0\n"


def taskmd_repo() -> Path | None:
    """A task-md checkout at hand (``TASKMD_REPO``), for the live comparisons."""
    raw = os.environ.get("TASKMD_REPO")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if (path / "tools" / "taskmd.py").is_file() else None


def require_taskmd_repo() -> Path:
    repo = taskmd_repo()
    if repo is None:
        pytest.skip(
            "TASKMD_REPO does not point at a task-md checkout (live comparison with the reference tools)"
        )
    return repo


def reference_tool(
    repo: Path, *args: str, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run one of the task-md checkout's own tools with this Python (3.11 or later)."""
    return subprocess.run(
        [sys.executable, str(repo / "tools" / args[0]), *args[1:]],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
        timeout=300,
    )
