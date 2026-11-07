"""Guard P78-CONFIRM-3 at 4cb6f9d4 with honest transformed checkouts."""

import json
import shlex
import subprocess

import pytest

from benchflow._utils import benchmark_repos as br
from benchflow._utils.dataset_registry import (
    DatasetDigestMismatchError,
    _task_matches_commit,
    resolve_dataset,
)


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.mark.parametrize("conversion", ["crlf", "encoding", "ident"])
@pytest.mark.parametrize("scope", ["root", "parent", "task"])
def test_pinned_attributes_preserve_untouched_checkout_bytes(
    tmp_path, conversion, scope
):
    """Guards 4cb6f9d4: Git's built-in checkout transforms are not edits."""
    root = tmp_path / "repo"
    task = root / "tasks" / "task-a"
    task.mkdir(parents=True)
    _git(root, "init", "-q")
    location = {"root": root, "parent": task.parent, "task": task}[scope]
    pattern = {
        "root": "tasks/task-a/input.txt",
        "parent": "task-a/input.txt",
        "task": "input.txt",
    }[scope]
    attribute = {
        "crlf": "text eol=crlf",
        "encoding": "text working-tree-encoding=UTF-16LE",
        "ident": "ident",
    }[conversion]
    (location / ".gitattributes").write_text(f"{pattern} {attribute}\n")
    input_file = task / "input.txt"
    original = b"line one\nline two\n" if conversion != "ident" else b"$Id$\n"
    input_file.write_bytes(
        original.decode().encode("utf-16le") if conversion == "encoding" else original
    )
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
    sha = _git(root, "rev-parse", "HEAD")
    input_file.unlink()
    _git(root, "checkout", "--", "tasks/task-a/input.txt")
    checkout = input_file.read_bytes()
    assert checkout != original
    before = _git(root, "ls-files", "-v", "--stage")
    assert _task_matches_commit(task, sha)
    input_file.write_bytes(checkout + b"tampered dummy marker")
    _git(root, "update-index", "--assume-unchanged", "tasks/task-a/input.txt")
    flags = _git(root, "ls-files", "-v", "--stage")
    assert not _task_matches_commit(task, sha)
    assert _git(root, "ls-files", "-v", "--stage") == flags
    assert before != flags


@pytest.mark.parametrize("override", ["worktree", "index", "info", "global"])
def test_checkout_conversion_ignores_mutable_attributes_and_filter_drivers(
    tmp_path, monkeypatch, override
):
    """Guards 4cb6f9d4: mutable conversion rules cannot bless changed bytes."""
    root = tmp_path / "repo"
    task = root / "tasks" / "task-a"
    task.mkdir(parents=True)
    _git(root, "init", "-q")
    attributes = root / ".gitattributes"
    attributes.write_text("tasks/task-a/input.txt text eol=crlf filter=dummy\n")
    input_file = task / "input.txt"
    input_file.write_bytes(b"original\n")
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
    sha = _git(root, "rev-parse", "HEAD")
    input_file.unlink()
    _git(root, "checkout", "--", "tasks/task-a/input.txt")
    assert input_file.read_bytes() == b"original\r\n"
    if override in {"worktree", "index"}:
        attributes.write_text("tasks/task-a/input.txt -text -filter\n")
        if override == "index":
            _git(root, "add", ".gitattributes")
    elif override == "info":
        (root / ".git/info/attributes").write_text(
            "tasks/task-a/input.txt -text -filter\n"
        )
    else:
        global_attributes = tmp_path / "global-attributes"
        global_attributes.write_text("* -text -filter\n")
        monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.attributesFile")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(global_attributes))
    marker = tmp_path / "filter-ran"
    _git(
        root,
        "config",
        "filter.dummy.smudge",
        f"echo dummy > {shlex.quote(str(marker))}; cat",
    )
    _git(root, "config", "filter.dummy.clean", "printf 'original\\n'")
    assert _task_matches_commit(task, sha)
    input_file.write_bytes(b"tampered dummy marker\r\n")
    assert not _task_matches_commit(task, sha)
    assert not marker.exists()


@pytest.mark.parametrize("ref", ["pin", "HEAD"])
@pytest.mark.parametrize("conversion", ["crlf", "encoding", "ident"])
def test_fresh_registry_cache_accepts_pinned_checkout_conversions(
    tmp_path, monkeypatch, ref, conversion
):
    """Guards 4cb6f9d4 through fresh sparse cache resolution, not a stub."""
    root = tmp_path / "origin"
    task = root / "tasks" / "task-a"
    task.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    attributes = {
        "crlf": "text eol=crlf",
        "encoding": "text working-tree-encoding=UTF-16LE",
        "ident": "ident",
    }[conversion]
    (root / ".gitattributes").write_text(f"tasks/task-a/input.txt {attributes}\n")
    original = "$Id$\n" if conversion == "ident" else "original\n"
    (task / "input.txt").write_bytes(
        original.encode("utf-16le" if conversion == "encoding" else "utf-8")
    )
    (task / "task.toml").write_text('version = "1.1"\n')
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
    sha = _git(root, "rev-parse", "HEAD")
    remote = tmp_path / "remote.git"
    _git(root, "clone", "-q", "--bare", str(root), str(remote))
    monkeypatch.setattr(br, "_repo_url", lambda *_: remote.as_uri())
    monkeypatch.setattr(br, "_cache_dir", lambda: tmp_path / "cache")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "name": "ds",
                    "version": "1",
                    "tasks": [
                        {
                            "name": "task-a",
                            "path": "tasks/task-a",
                            "git_url": "https://github.com/acme/tasks",
                            "git_commit_id": sha if ref == "pin" else ref,
                        }
                    ],
                }
            ]
        )
    )
    resolved = resolve_dataset("ds@1", registry=str(registry))
    input_file = resolved.tasks_dir / "task-a/input.txt"
    assert input_file.read_bytes() != original.encode()
    assert resolved.provenance["resolved_sha"] == sha
    with input_file.open("ab") as stream:
        stream.write(b"tampered dummy marker")
    with pytest.raises(DatasetDigestMismatchError):
        resolve_dataset("ds@1", registry=str(registry))
