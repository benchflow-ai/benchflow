"""Guard P78-CONFIRM-3/5/6 against regressions in 946349da's conversions."""

import json
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


def _registry(tmp_path, monkeypatch, root, ref):
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
                            "git_commit_id": ref,
                        }
                    ],
                }
            ]
        )
    )
    return str(registry)


@pytest.mark.parametrize("ref", ["pin", "HEAD"])
@pytest.mark.parametrize("conversion", ["crlf", "encoding", "ident"])
@pytest.mark.parametrize("depth", [1, 2])
def test_nested_pinned_attributes_preserve_fresh_registry_tasks(
    tmp_path, monkeypatch, ref, conversion, depth
):
    """Guards 946349da: every file's pinned attribute chain applies."""
    root = tmp_path / "origin"
    task = root / "tasks/task-a"
    directory = task / "environment"
    if depth == 2:
        directory /= "data"
    directory.mkdir(parents=True)
    _git(root, "init", "-q")
    # A nested rule must override the root rule, including for ident.
    (root / ".gitattributes").write_text("* -text -ident\n")
    attribute = {
        "crlf": "text eol=crlf",
        "encoding": "text working-tree-encoding=UTF-16LE",
        "ident": "ident",
    }[conversion]
    (directory / ".gitattributes").write_text(f"input.txt {attribute}\n")
    original = "$Id$\n" if conversion == "ident" else "original\n"
    (directory / "input.txt").write_bytes(
        original.encode("utf-16le" if conversion == "encoding" else "utf-8")
    )
    (task / "task.toml").write_text('version = "1.1"\n')
    sha = _commit(root)
    registry = _registry(tmp_path, monkeypatch, root, sha if ref == "pin" else ref)
    resolved = resolve_dataset("ds@1", registry=registry)
    input_file = (
        resolved.tasks_dir / "task-a" / directory.relative_to(task) / "input.txt"
    )
    checkout = input_file.read_bytes()
    assert checkout != original.encode()
    assert resolved.provenance["resolved_sha"] == sha
    with input_file.open("ab") as stream:
        stream.write(b"tampered dummy marker")
    with pytest.raises(DatasetDigestMismatchError):
        resolve_dataset("ds@1", registry=registry)
    assert input_file.read_bytes() == checkout + b"tampered dummy marker"


@pytest.mark.parametrize("location", ["home", "xdg"])
@pytest.mark.parametrize("pinned", [False, True])
def test_default_user_attributes_cannot_authorize_or_override_task_bytes(
    tmp_path, monkeypatch, location, pinned
):
    """Guards 946349da: HOME/XDG attributes are outside the pinned tree."""
    root = tmp_path / "origin"
    task = root / "tasks/task-a"
    task.mkdir(parents=True)
    _git(root, "init", "-q")
    # Git 2.26's initial sparse checkout needs a root file before the
    # resolver selects tasks/. Keep that separate issue out of this fixture.
    (root / "README").write_text("dummy registry fixture\n")
    if pinned:
        (root / ".gitattributes").write_text("tasks/task-a/input.txt text eol=crlf\n")
    (task / "input.txt").write_bytes(b"original\n")
    sha = _commit(root)
    registry = _registry(tmp_path, monkeypatch, root, sha)
    resolved = resolve_dataset("ds@1", registry=registry)
    cached = resolved.tasks_dir / "task-a"
    input_file = cached / "input.txt"
    honest = input_file.read_bytes()
    home = tmp_path / "dummy-home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    if location == "home":
        config = home / ".config"
    else:
        config = tmp_path / "dummy-xdg"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    attributes = config / "git/attributes"
    attributes.parent.mkdir(parents=True)
    attributes.write_text("tasks/task-a/input.txt working-tree-encoding=UTF-16LE\n")
    # A default user file must neither reject an honest pinned conversion
    # nor authorize an unpinned encoding conversion of edited bytes.
    assert _task_matches_commit(cached, sha)
    input_file.write_bytes(honest.decode().encode("utf-16le"))
    with pytest.raises(DatasetDigestMismatchError):
        resolve_dataset("ds@1", registry=registry)
    assert input_file.read_bytes() == honest.decode().encode("utf-16le")


@pytest.mark.parametrize("linked", [False, True])
def test_checkout_conversion_supports_git_226_object_paths(tmp_path, linked):
    """Guards 946349da on real Git 2.26 and current Git, including worktrees."""
    root = tmp_path / "repo"
    task = root / "tasks/task-a"
    task.mkdir(parents=True)
    _git(root, "init", "-q")
    (root / ".gitattributes").write_text("tasks/task-a/input.txt text eol=crlf\n")
    (task / "input.txt").write_bytes(b"original\n")
    sha = _commit(root)
    if linked:
        checkout = tmp_path / "linked"
        _git(root, "worktree", "add", "--detach", str(checkout), sha)
        task = checkout / "tasks/task-a"
    else:
        (task / "input.txt").unlink()
        _git(root, "checkout", "--", "tasks/task-a/input.txt")
    assert (task / "input.txt").read_bytes() == b"original\r\n"
    assert _task_matches_commit(task, sha)
    assert _task_matches_commit(task, sha)  # private conversion must not change state
    (task / "input.txt").write_bytes(b"tampered dummy marker\r\n")
    assert not _task_matches_commit(task, sha)
