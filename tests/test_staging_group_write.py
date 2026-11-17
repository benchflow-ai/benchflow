"""The private task copy drops group and other write (FP patch 84).

A task checked out under umask 002 has its files 664 and folders 775; under
022, 644 and 755. Git records neither, and the task digest hashes content
only, so one digest gave two builds: twisted-slinky's loader seal refused
``/usr/local/lib/slinky-isolation.py (mode -rw-rw-r--)`` from a umask-002
checkout and ran from a umask-022 one. The staged copy now keeps every mode
bit but group and other write.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from benchflow.task.staging import copy_tree, stage_environment


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


@pytest.fixture
def task(tmp_path: Path) -> Path:
    old = os.umask(0o002)
    try:
        root = tmp_path / "twisted-slinky"
        (root / "environment" / "lib").mkdir(parents=True)
        (root / "environment" / "Dockerfile").write_text("FROM scratch\n")
        (root / "environment" / "lib" / "slinky-isolation.py").write_text("x = 1\n")
        script = root / "environment" / "run.sh"
        script.write_text("#!/bin/sh\n")
        script.chmod(0o775)
        secret = root / "environment" / "key"
        secret.write_text("k\n")
        secret.chmod(0o600)
    finally:
        os.umask(old)
    assert _mode(root / "environment" / "lib") == 0o775
    assert _mode(root / "environment" / "lib" / "slinky-isolation.py") == 0o664
    return root


def test_the_copy_has_the_modes_of_a_umask_022_checkout(task, tmp_path):
    copy = tmp_path / "copy"
    copy_tree(task, copy)
    env = copy / "environment"
    assert _mode(env / "lib") == 0o755
    assert _mode(env / "lib" / "slinky-isolation.py") == 0o644
    assert _mode(env / "run.sh") == 0o755
    assert _mode(env / "key") == 0o600
    assert _mode(env / "Dockerfile") == 0o644


def test_the_staged_environment_has_them_too(task):
    staged = stage_environment(task / "environment")
    try:
        assert _mode(staged.environment_dir / "lib") == 0o755
        assert _mode(staged.environment_dir / "lib" / "slinky-isolation.py") == 0o644
    finally:
        staged.remove()
