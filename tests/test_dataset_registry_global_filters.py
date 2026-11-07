"""Guard P78-CONFIRM-7, introduced by Git 2.26 compatibility in c2075402."""

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


def _repository(tmp_path, *, crlf):
    root = tmp_path / "origin"
    task = root / "tasks/task-a"
    task.mkdir(parents=True)
    _git(root, "init", "-q")
    conversion = "text eol=crlf" if crlf else "-text"
    (root / ".gitattributes").write_text(
        f"tasks/task-a/verifier.sh filter=demo {conversion}\n"
    )
    (task / "verifier.sh").write_bytes(b"exit 1\n")
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
    return root, _git(root, "rev-parse", "HEAD")


def _global_filter(tmp_path, monkeypatch, source, driver):
    # Write only synthetic config files, never run git config --global.
    home = tmp_path / "dummy-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("GIT_CONFIG_GLOBAL", raising=False)
    if source.startswith("home") and source != "home-xdg":
        config = home / ".gitconfig"
    elif source == "home-xdg":
        config = home / ".config/git/config"
    else:
        xdg = tmp_path / "dummy-xdg"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        config = xdg / "git/config"
    config.parent.mkdir(parents=True, exist_ok=True)
    marker = tmp_path / "filter-ran"
    edited = tmp_path / "edited-verifier"
    edited.write_bytes(b"exit 0\n")
    script = tmp_path / "dummy-filter.sh"
    # A process driver deliberately fails its handshake after the marker:
    # even a failed external driver must never execute during validation.
    script.write_text(
        f"printf dummy > {shlex.quote(str(marker))}\n"
        + (f"cat {shlex.quote(str(edited))}\n" if driver == "smudge" else "exit 1\n")
    )
    settings = f'[filter "demo"]\n\t{driver} = sh {shlex.quote(str(script))}\n'
    if source.endswith("include"):
        included = tmp_path / "included-config"
        included.write_text(settings)
        config.write_text(f'[include]\n\tpath = "{included}"\n')
    else:
        config.write_text(settings)
    return marker


@pytest.mark.parametrize(
    "source", ["home", "home-xdg", "xdg", "home-include", "xdg-include"]
)
@pytest.mark.parametrize("driver", ["smudge", "process"])
@pytest.mark.parametrize("linked", [False, True])
def test_global_filter_cannot_authorize_bytes_or_execute(
    tmp_path, monkeypatch, source, driver, linked
):
    """Guards c2075402 on Git 2.26 and current Git, ordinary and linked trees."""
    root, sha = _repository(tmp_path, crlf=True)
    if linked:
        checkout = tmp_path / "linked"
        _git(root, "worktree", "add", "--detach", str(checkout), sha)
        task = checkout / "tasks/task-a"
    else:
        task = root / "tasks/task-a"
    verifier = task / "verifier.sh"
    verifier.write_bytes(b"exit 1\r\n")
    assert _task_matches_commit(task, sha)
    marker = _global_filter(tmp_path, monkeypatch, source, driver)
    verifier.write_bytes(b"exit 0\r\n")
    matches = _task_matches_commit(task, sha)
    assert (matches, marker.exists()) == (False, False)
    assert verifier.read_bytes() == b"exit 0\r\n"
    # Honest built-in conversions stay accepted despite an ambient driver.
    verifier.write_bytes(b"exit 1\r\n")
    assert _task_matches_commit(task, sha)
    assert not marker.exists()


@pytest.mark.parametrize("source", ["home", "home-xdg", "xdg"])
@pytest.mark.parametrize("ref", ["pin", "HEAD"])
def test_registry_rejects_cached_edits_with_global_smudge(
    tmp_path, monkeypatch, source, ref
):
    """Guards c2075402 through both public no-digest resolution paths."""
    root, sha = _repository(tmp_path, crlf=False)
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
    verifier = resolved.tasks_dir / "task-a/verifier.sh"
    marker = _global_filter(tmp_path, monkeypatch, source, "smudge")
    verifier.write_bytes(b"exit 0\n")
    with pytest.raises(DatasetDigestMismatchError):
        resolve_dataset("ds@1", registry=str(registry))
    assert verifier.read_bytes() == b"exit 0\n"
    assert not marker.exists()
