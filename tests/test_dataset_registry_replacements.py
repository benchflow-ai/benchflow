"""Guard P78-CONFIRM-2 at 4cb6f9d4 against replacement objects."""

import subprocess

import pytest

from benchflow._utils.dataset_registry import _task_matches_commit


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(root):
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=BenchFlow Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "fixture",
    )
    return _git(root, "rev-parse", "HEAD")


@pytest.mark.parametrize("object_kind", ["commit", "tree", "blob"])
def test_commit_integrity_ignores_replacement_objects(tmp_path, object_kind):
    """Guards 4cb6f9d4: HEAD C cannot authorize replacement D's verifier."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    verifier = root / "verifier.sh"
    verifier.write_bytes(b"exit 0 # original dummy marker\n")
    first = _commit(root)
    verifier.write_bytes(b"exit 99 # replacement dummy marker\n")
    second = _commit(root)
    suffix = {"commit": "", "tree": "^{tree}", "blob": ":verifier.sh"}[object_kind]
    old = _git(root, "rev-parse", first + suffix)
    new = _git(root, "rev-parse", second + suffix)
    _git(root, "checkout", "-q", "--detach", first)
    _git(root, "replace", old, new)
    verifier.write_bytes(b"exit 99 # replacement dummy marker\n")
    assert _git(root, "rev-parse", "HEAD") == first
    assert not _task_matches_commit(root, first)
    verifier.write_bytes(b"exit 0 # original dummy marker\n")
    assert _task_matches_commit(root, first)
