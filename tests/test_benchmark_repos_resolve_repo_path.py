"""Subpath validation in ``_resolve_repo_path`` (benchmark dataset resolution).

``resolve_source`` clones an ``org/repo`` then resolves an optional subpath via
``_resolve_repo_path(root, path, repo_label)``. The #825 diff dropped the legacy
``TASK_REPOS`` dict and added task.md support but kept the three subpath guards
intact — these pin them so a future refactor can't silently drop them:

* an absolute subpath is rejected (must be repo-relative);
* a non-existent subpath raises with the "Available:" listing;
* a subpath that escapes the repo root (via symlink) is rejected;
* a valid relative subpath resolves to the real directory.

No clone is needed: the function operates on a plain ``root`` Path, so a tmp
directory standing in for a checked-out repo exercises every branch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow._utils.benchmark_repos import _resolve_repo_path


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "tasks").mkdir(parents=True)
    return root


def test_resolve_repo_path_rejects_absolute_subpath(tmp_path: Path):
    root = _repo(tmp_path)
    abs_path = str(tmp_path / "outside" / "abs-target")  # absolute, not repo-relative
    with pytest.raises(ValueError, match="must be relative to repository root"):
        _resolve_repo_path(root, abs_path, "org/repo")


def test_resolve_repo_path_missing_subpath_lists_available(tmp_path: Path):
    root = _repo(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"not found in org/repo\. Available:"):
        _resolve_repo_path(root, "does-not-exist", "org/repo")


def test_resolve_repo_path_rejects_escape_via_symlink(tmp_path: Path):
    root = _repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes repository root"):
        _resolve_repo_path(root, "escape", "org/repo")


def test_resolve_repo_path_returns_relative_subdir(tmp_path: Path):
    root = _repo(tmp_path)
    resolved = _resolve_repo_path(root, "tasks", "org/repo")
    assert resolved == (root / "tasks").resolve(strict=True)
    assert resolved.is_dir()
