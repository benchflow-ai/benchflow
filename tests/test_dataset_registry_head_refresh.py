"""A ``HEAD`` registry entry runs the default branch as it is now (P78-R1-A-1).

An unpinned entry (``git_commit_id`` ``HEAD``, as many entries of the public
dataset registry are) was resolved with no ref, and with no ref an existing
source cache was reused without a fetch. So the run used whatever commit the
cache last held: the first clone's commit forever, or the commit an earlier
pinned dataset from the same repository had checked out. These tests run real
git against a local repository served over ``file://``.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from benchflow._utils import benchmark_repos as br
from benchflow._utils.dataset_registry import DatasetResolutionError, resolve_dataset

_IDENTITY = ("-c", "user.name=BenchFlow Test", "-c", "user.email=test@example.com")


def _git_version() -> tuple[int, int]:
    try:
        out = subprocess.run(["git", "version"], capture_output=True, text=True).stdout
    except (OSError, ValueError):
        return (0, 0)
    match = re.search(r"(\d+)\.(\d+)", out)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


# `git init -b` needs 2.28.
pytestmark = pytest.mark.skipif(
    _git_version() < (2, 28), reason="needs git 2.28 or newer"
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_task(root: Path, name: str, instruction: str) -> None:
    task = root / "tasks" / name
    (task / "tests").mkdir(parents=True, exist_ok=True)
    (task / "task.toml").write_text('version = "1.1"\n')
    (task / "instruction.md").write_text(instruction)
    (task / "tests" / "test.sh").write_text("exit 0\n")


def _entry(version: str, commit: str, tasks: tuple[str, ...]) -> dict:
    return {
        "name": "ds",
        "version": version,
        "tasks": [
            {
                "name": task,
                "git_url": "https://github.com/acme/public-bench.git",
                "git_commit_id": commit,
                "path": f"tasks/{task}",
            }
            for task in tasks
        ],
    }


class Origin:
    """A bare repository acme/public-bench whose default branch can move."""

    def __init__(self, tmp_path: Path) -> None:
        self.src = tmp_path / "src"
        self.bare = tmp_path / "remote.git"
        _write_task(self.src, "task-a", "Solve task-a, first edition.\n")
        _write_task(self.src, "task-b", "Solve task-b.\n")
        _git(tmp_path, "init", "-q", "-b", "main", str(self.src))
        self.first = self._commit("one")
        _git(tmp_path, "clone", "-q", "--bare", str(self.src), str(self.bare))
        _git(self.bare, "config", "uploadpack.allowFilter", "true")
        _git(self.bare, "config", "uploadpack.allowAnySHA1InWant", "true")

    def _commit(self, message: str) -> str:
        _git(self.src, "add", "-A")
        _git(self.src, *_IDENTITY, "commit", "-q", "-m", message)
        return _git(self.src, "rev-parse", "HEAD")

    def advance(self) -> str:
        """Change task-a, add task-c, and push to the default branch."""
        _write_task(self.src, "task-a", "Solve task-a, second edition.\n")
        _write_task(self.src, "task-c", "Solve task-c.\n")
        sha = self._commit("two")
        _git(self.src, "push", "-q", str(self.bare), "main")
        return sha


@pytest.fixture
def origin(tmp_path, monkeypatch) -> Origin:
    served = Origin(tmp_path)
    monkeypatch.setattr(br, "_repo_url", lambda org, repo: f"file://{served.bare}")
    monkeypatch.setattr(br, "_cache_dir", lambda: tmp_path / "cache")
    return served


def _registry(tmp_path: Path, entries: list[dict]) -> str:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(entries))
    return str(path)


@pytest.mark.parametrize("change", ["tracked", "untracked", "ignored"])
def test_a_commit_pin_without_digests_rejects_modified_task_bytes(
    tmp_path, origin, change
):
    """Guards the fix at 12f002dc, without discarding local edits."""
    registry = _registry(tmp_path, [_entry("1.0", origin.first, ("task-a",))])
    first = resolve_dataset("ds@1.0", registry=registry)
    task = first.tasks_dir / "task-a"
    if change == "tracked":
        modified = task / "tests/test.sh"
    else:
        modified = task / (
            "extra.sh" if change == "untracked" else "__pycache__/extra.sh"
        )
        modified.parent.mkdir(exist_ok=True)
        if change == "ignored":
            info = br._git_stdout(task, "rev-parse", "--git-path", "info/exclude")
            assert info is not None
            exclude = Path(info)
            if not exclude.is_absolute():
                exclude = task / exclude
            exclude.parent.mkdir(parents=True, exist_ok=True)
            exclude.write_text("__pycache__/\n")
    modified.write_text("echo tampered\n")
    with pytest.raises(DatasetResolutionError, match=r"commit|snapshot"):
        resolve_dataset("ds@1.0", registry=registry)
    assert modified.read_text() == "echo tampered\n"


@pytest.mark.parametrize(
    "suppression", ["assume-unchanged", "skip-worktree", "staged", "clean-filter"]
)
@pytest.mark.parametrize("ref", ["pin", "HEAD"])
def test_no_digest_tasks_are_compared_to_the_tree_not_the_index(
    tmp_path, origin, suppression, ref
):
    """Guards P78-CONFIRM-2 at 78120e49, preserving edits and index flags."""
    registry = _registry(
        tmp_path,
        [_entry("checked", origin.first if ref == "pin" else ref, ("task-a",))],
    )
    first = resolve_dataset("ds@checked", registry=registry)
    task = first.tasks_dir / "task-a"
    modified = task / "tests/test.sh"
    if suppression in {"assume-unchanged", "skip-worktree"}:
        if suppression == "skip-worktree":
            # Sparse status otherwise clears skip-worktree for present files.
            _git(task, "config", "--worktree", "core.sparseCheckout", "false")
        _git(task, "update-index", f"--{suppression}", "tests/test.sh")
    elif suppression == "clean-filter":
        _git(task, "config", "filter.hidden.clean", "printf 'exit 0\\n'")
        attrs = Path(_git(task, "rev-parse", "--git-path", "info/attributes"))
        attrs.parent.mkdir(parents=True, exist_ok=True)
        attrs.write_text("tasks/task-a/tests/test.sh filter=hidden\n")
    modified.write_text("echo tampered\n")
    if suppression in {"staged", "clean-filter"}:
        _git(task, "add", "tests/test.sh")
    before = _git(task, "ls-files", "-v", "--stage")
    if suppression in {"assume-unchanged", "skip-worktree", "clean-filter"}:
        assert not _git(task, "status", "--porcelain", "--", ".")
    with pytest.raises(DatasetResolutionError, match=r"commit|snapshot"):
        resolve_dataset("ds@checked", registry=registry)
    assert modified.read_text() == "echo tampered\n"
    assert _git(task, "ls-files", "-v", "--stage") == before


def test_a_head_entry_fetches_the_moved_default_branch_after_a_pinned_run(
    tmp_path, origin
):
    registry = _registry(
        tmp_path,
        [
            _entry("head", "HEAD", ("task-a", "task-b")),
            _entry("1.0", origin.first, ("task-a", "task-b")),
            _entry("latest", "HEAD", ("task-a", "task-c")),
        ],
    )

    # A fresh clone for a HEAD entry: the default branch, no `--branch HEAD`.
    first = resolve_dataset("ds@head", registry=registry)
    assert first.provenance["resolved_sha"] == origin.first

    second = origin.advance()

    # A pinned run of the older commit leaves the cache detached there.
    pinned = resolve_dataset("ds@1.0", registry=registry)
    assert pinned.provenance["resolved_sha"] == origin.first
    assert "first edition" in (pinned.tasks_dir / "task-a/instruction.md").read_text()

    # The HEAD entry runs the default branch as it is now, not the cache's commit.
    again = resolve_dataset("ds@head", registry=registry)
    assert again.provenance["resolved_sha"] == second
    assert "second edition" in (again.tasks_dir / "task-a/instruction.md").read_text()

    # A task added on the default branch is there, not "missing from snapshot".
    latest = resolve_dataset("ds@latest", registry=registry)
    assert latest.provenance["resolved_sha"] == second
    assert (latest.tasks_dir / "task-c" / "task.toml").is_file()


def test_a_repeated_head_run_does_not_keep_the_first_clones_commit(tmp_path, origin):
    registry = _registry(tmp_path, [_entry("head", "HEAD", ("task-a", "task-b"))])
    assert (
        resolve_dataset("ds@head", registry=registry).provenance["resolved_sha"]
        == origin.first
    )
    second = origin.advance()
    again = resolve_dataset("ds@head", registry=registry)
    assert again.provenance["resolved_sha"] == second
    assert "second edition" in (again.tasks_dir / "task-a/instruction.md").read_text()
