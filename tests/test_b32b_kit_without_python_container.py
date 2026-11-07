"""A kit without a Python runs no image Python as root, in a real container
(merge wave batch 32b, review B B32B-B-1).

On ``continuumio/miniconda3`` the only ``python3`` is ``/opt/conda/bin``'s,
outside the folders PT's kit copies a Python from, so the sandbox's kit has
none. V3's judge would accept the conda interpreter; now it is
not judged or run: the purges skip, the hook clean-up says it skipped, and
the judge's answer is the skip hardening logs. A wrapper in place of the
conda ``python3`` records any run. Skipped without a Docker daemon.
"""

from __future__ import annotations

import subprocess
import uuid

import pytest

from benchflow._runtime_tree import runtime_kit_python
from benchflow.sandbox import lockdown
from tests.test_c8_protected_runtime_container import (
    LABEL,
    _docker_ready,
    _dx,
    _sandbox,
)

IMAGE = "continuumio/miniconda3:24.7.1-0"


@pytest.fixture
def conda_box():
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = "PT-b32b-conda-" + uuid.uuid4().hex[:8]
    run = subprocess.run(
        ["docker", "run", "-d", "--name", name, "--label", LABEL,
         "--network", "none", IMAGE, "sleep", "1200"],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start {IMAGE}: {run.stderr.strip()}")
    try:
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.mark.asyncio
async def test_no_image_python_runs_as_root_on_a_kit_without_one(conda_box):
    from benchflow.sandbox.lockdown import setup_sandbox_user

    box = conda_box
    # No root-owned python3 where the kit copies one from: the conda one on
    # PATH is the only Python, so the kit is built without a Python.
    moved = _dx(
        box,
        "mkdir -p /opt/sys-python && for f in /usr/bin/python3* /bin/python3* "
        '/usr/local/bin/python3*; do [ -e "$f" ] && mv "$f" /opt/sys-python/; '
        "done; ls /usr/bin/python3* /bin/python3* /usr/local/bin/python3* 2>/dev/null; "
        "test -x /opt/conda/bin/python3 && echo conda",
    )
    assert moved.stdout.strip() == "conda", moved.stdout
    sandbox = _sandbox(box)
    await setup_sandbox_user(sandbox, "agent", "/app", guard_daemon=False)
    # Any run of the conda interpreter leaves a mark.
    wrapped = _dx(
        box,
        "real=$(readlink -f /opt/conda/bin/python3) && rm /opt/conda/bin/python3 && "
        "printf '#!/bin/sh\\necho ran >> /tmp/conda-python-ran\\nexec %s \"$@\"\\n' "
        '"$real" > /opt/conda/bin/python3 && chmod 755 /opt/conda/bin/python3',
    )
    assert wrapped.returncode == 0, wrapped.stderr
    # An external link the purge would remove, if it ran.
    made = _dx(box, "mkdir -p /app && ln -s /etc/passwd /app/out")
    assert made.returncode == 0, made.stderr

    judged = await lockdown._judge_root_python(sandbox, (), "agent")
    assert runtime_kit_python(sandbox) is None
    assert judged.path is None and judged.kit_without_python
    assert "holds no Python" in judged.missing_text(IMAGE)

    token = lockdown._root_python.set(judged.path)
    try:
        purge = await sandbox.exec(
            lockdown._prep_on_path(lockdown._purge_external_symlinks_cmd("/app")),
            user="root",
            timeout_sec=120,
        )
        cleanup = await sandbox.exec(
            lockdown._prep_on_path(
                lockdown._build_cleanup_cmd({"cleanup_conftests": False}, "/app")
            ),
            user="root",
            timeout_sec=120,
        )
    finally:
        lockdown._root_python.reset(token)
    assert purge.return_code == 0, purge.stderr
    assert cleanup.return_code == 0, cleanup.stderr
    # Skipped, and the hook clean-up says so.
    assert _dx(box, "test -L /app/out").returncode == 0
    assert lockdown.NO_PROTECTED_PYTHON_SKIP in (cleanup.stderr or "")
    # No image Python ran as root.
    assert _dx(box, "cat /tmp/conda-python-ran 2>/dev/null").stdout == ""
