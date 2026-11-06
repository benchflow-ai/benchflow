"""The dataset cache stays out of the user's git repository (DF-p1-first-run-15).

Run inside a git repository, BenchFlow caches sources at its root
(``.cache/datasets-v2``), and ``git status`` listed ``?? .cache/``: one
``git add .`` from committing datasets (SkillsBench's task folder is about
1.4 GB). The cache now holds a ``.gitignore`` with ``*``, as pytest's and
ruff's caches do.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from benchflow._utils import benchmark_repos as br


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "init.defaultBranch=main", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "PATH": "/usr/bin:/bin",
        },
    ).stdout


@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
def test_the_cache_is_invisible_to_git_status(tmp_path, monkeypatch):
    repo = tmp_path / "myrepo"
    repo.mkdir()
    _git(repo, "init", "-q")
    monkeypatch.setattr(br, "_cache_dir", lambda: repo / ".cache" / "datasets-v2")

    with br._repo_cache_lock("benchflow-ai", "skillsbench"):
        (repo / ".cache" / "datasets-v2" / "benchflow-ai" / "skillsbench").mkdir()
        (
            repo / ".cache" / "datasets-v2" / "benchflow-ai" / "skillsbench" / "t.md"
        ).write_text("x")

    assert (repo / ".cache" / "datasets-v2" / ".gitignore").read_text().splitlines()[
        -1
    ] == "*"
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""


def test_an_existing_gitignore_is_left_alone(tmp_path, monkeypatch):
    cache = tmp_path / "datasets-v2"
    cache.mkdir()
    (cache / ".gitignore").write_text("mine\n")
    monkeypatch.setattr(br, "_cache_dir", lambda: cache)

    with br._repo_cache_lock("org", "repo"):
        pass

    assert (cache / ".gitignore").read_text() == "mine\n"


def test_a_cache_folder_that_already_exists_is_never_written_into(
    tmp_path, monkeypatch
):
    """Review L-3: only a cache folder BenchFlow creates gets the .gitignore;
    one that is already there (an earlier BenchFlow's, or the user's) keeps
    exactly what it holds."""
    cache = tmp_path / "datasets-v2"
    cache.mkdir()
    (cache / "mine.txt").write_text("x")
    monkeypatch.setattr(br, "_cache_dir", lambda: cache)

    with br._repo_cache_lock("org", "repo"):
        pass

    assert not (cache / ".gitignore").exists()
    assert sorted(p.name for p in cache.iterdir()) == ["mine.txt", "org"]
