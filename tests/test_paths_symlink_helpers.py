"""Tests for the symlink-safe traversal helpers in ``benchflow._paths``.

These cover the primitives (`is_safe_regular_file`, `iter_safe_children`,
`iter_safe_tree`, `ignore_symlinks`) added for PR-B's symlink-defence work.
The higher-level call sites have their own regression files:

* ``test_dashboard_symlink_ingestion.py``  (#390, #416)
* ``test_judge_symlink_ingestion.py``      (#404)
* ``test_sandbox_upload_symlink.py``       (#411)
"""

from __future__ import annotations

from pathlib import Path

from benchflow._paths import (
    ignore_symlinks,
    is_safe_regular_dir,
    is_safe_regular_file,
    iter_safe_children,
    iter_safe_tree,
)


def test_is_safe_regular_file_accepts_regular(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("x")
    assert is_safe_regular_file(f) is True


def test_is_safe_regular_file_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("x")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    assert is_safe_regular_file(link) is False


def test_is_safe_regular_file_rejects_missing(tmp_path: Path) -> None:
    assert is_safe_regular_file(tmp_path / "nope") is False


def test_is_safe_regular_dir_rejects_symlinked_dir(tmp_path: Path) -> None:
    target = tmp_path / "real_dir"
    target.mkdir()
    link = tmp_path / "linked_dir"
    link.symlink_to(target)
    assert is_safe_regular_dir(target) is True
    assert is_safe_regular_dir(link) is False


def test_iter_safe_children_skips_symlinks(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("secret")

    parent = tmp_path / "dir"
    parent.mkdir()
    (parent / "real.txt").write_text("ok")
    (parent / "leak.txt").symlink_to(secret)

    names = {c.name for c in iter_safe_children(parent, context="test")}
    assert names == {"real.txt"}


def test_iter_safe_tree_does_not_descend_symlinked_dir(tmp_path: Path) -> None:
    secret_dir = tmp_path / "secret_dir"
    secret_dir.mkdir()
    (secret_dir / "secret.txt").write_text("secret")

    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "real.txt").write_text("real")
    (root / "linked_dir").symlink_to(secret_dir)
    (root / "linked_file.txt").symlink_to(secret_dir / "secret.txt")

    files = list(iter_safe_tree(root, context="test"))
    names = {f.name for f in files}
    # Only the real, in-tree file.
    assert names == {"real.txt"}


def test_iter_safe_tree_refuses_symlinked_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "f.txt").write_text("x")
    link = tmp_path / "link"
    link.symlink_to(real)
    assert list(iter_safe_tree(link, context="test")) == []


def test_ignore_symlinks_returns_link_names(tmp_path: Path) -> None:
    (tmp_path / "real.txt").write_text("x")
    (tmp_path / "linked.txt").symlink_to(tmp_path / "real.txt")
    skipped = ignore_symlinks(str(tmp_path), ["real.txt", "linked.txt"])
    assert skipped == ["linked.txt"]
