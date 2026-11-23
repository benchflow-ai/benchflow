"""A git that stalls never gives one build two identities.

``benchflow_build`` (result.json; test_local_daemon_docker compares warm
records with cold ones) comes from git. With a fixed 10 s bound and null on a
timeout, a git that stalled in a cold call's start burst stamped null, while
the runtime that warm calls inherit from had stamped the commit: one tree,
two records. A fake git on PATH stalls on purpose here.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from benchflow import build_info
from benchflow.build_info import git_identity
from tests.test_benchflow_build import _repo


def _stalling_git(
    tmp_path: Path, monkeypatch, *, stalls: int, stall_sec: float
) -> None:
    """A git first on PATH whose first *stalls* calls sleep *stall_sec*
    before running the real git."""
    real = shutil.which("git")
    assert real is not None
    folder = tmp_path / "bin"
    folder.mkdir()
    left = tmp_path / "stalls-left"
    left.write_text(str(stalls))
    script = folder / "git"
    script.write_text(
        "#!/bin/sh\n"
        f'n=$(cat "{left}")\n'
        f'if [ "$n" -gt 0 ]; then echo $((n - 1)) > "{left}"; sleep {stall_sec}; fi\n'
        f'exec "{real}" "$@"\n'
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ['PATH']}")


def _load(monkeypatch, one_minute: float) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    monkeypatch.setattr(os, "getloadavg", lambda: (one_minute,) * 3)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(build_info, "_GIT_TIMEOUT_SEC", 1)
    return _repo(tmp_path / "repo")


def test_a_cold_read_whose_git_stalled_names_the_same_build(
    repo, tmp_path, monkeypatch
):
    _load(monkeypatch, 1.0)
    warm = git_identity(repo)
    assert warm is not None and warm["commit"]

    _stalling_git(tmp_path, monkeypatch, stalls=1, stall_sec=1.6)
    cold = git_identity(repo)

    assert cold == warm


def test_a_loaded_host_gives_git_longer(repo, tmp_path, monkeypatch):
    _load(monkeypatch, 12.0)  # 3 per CPU: the bound is 3 s
    warm = git_identity(repo)

    _stalling_git(tmp_path, monkeypatch, stalls=2, stall_sec=1.6)
    cold = git_identity(repo)

    assert cold == warm


def test_a_git_that_never_answers_is_unknown_and_bounded(repo, tmp_path, monkeypatch):
    _load(monkeypatch, 1.0)
    _stalling_git(tmp_path, monkeypatch, stalls=99, stall_sec=5)
    started = time.monotonic()

    assert git_identity(repo) is None
    assert time.monotonic() - started < 15
