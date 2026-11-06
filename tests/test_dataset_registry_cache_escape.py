"""A registry entry cannot point the source cache outside it (P78-R2-B-1).

A registry named with ``--registry`` is downloaded, untrusted data. Its
``git_url`` named the cache folder with no check, so
``https://github.com/../..`` made the cache ``<git root>/.cache/datasets/../..``:
the git root of the user's own project. That folder has a ``.git``, so the
refresh ran ``git fetch origin HEAD`` and ``git checkout --detach FETCH_HEAD``
there, detaching the user's branch at their origin's default branch, and ran
their project files as the dataset's tasks. ``git_commit_id`` also went into
``git fetch`` and ``git clone`` as given.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from benchflow._utils import benchmark_repos as br
from benchflow._utils import dataset_registry as dr
from benchflow._utils.dataset_registry import DatasetResolutionError, resolve_dataset

_IDENTITY = ("-c", "user.name=BenchFlow Test", "-c", "user.email=test@example.com")


def _git_version() -> tuple[int, int]:
    try:
        out = subprocess.run(["git", "version"], capture_output=True, text=True).stdout
    except (OSError, ValueError):
        return (0, 0)
    match = re.search(r"(\d+)\.(\d+)", out)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _registry(tmp_path: Path, git_url: str, commit: str = "HEAD") -> str:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "evil",
                    "version": "1.0",
                    "tasks": [
                        {
                            "name": "x",
                            "git_url": git_url,
                            "git_commit_id": commit,
                            "path": "tasks/x",
                        }
                    ],
                }
            ]
        )
    )
    return str(path)


@pytest.mark.skipif(_git_version() < (2, 28), reason="needs git 2.28 or newer")
def test_a_dot_dot_entry_leaves_the_users_own_repository_alone(tmp_path, monkeypatch):
    # An upstream whose default branch is ahead of the user's work.
    upstream = tmp_path / "upstream"
    _git(tmp_path, "init", "-q", "-b", "main", str(upstream))
    (upstream / "a.txt").write_text("one\n")
    _git(upstream, "add", "-A")
    _git(upstream, *_IDENTITY, "commit", "-q", "-m", "upstream-1")
    bare = tmp_path / "upstream.git"
    _git(tmp_path, "clone", "-q", "--bare", str(upstream), str(bare))

    # The user's project: a full clone, on a branch with an unpushed commit.
    project = tmp_path / "project"
    _git(tmp_path, "clone", "-q", str(bare), str(project))
    _git(project, "checkout", "-q", "-b", "mywork")
    (project / "mine.txt").write_text("mine\n")
    _git(project, "add", "-A")
    _git(project, *_IDENTITY, "commit", "-q", "-m", "mine")
    (upstream / "a.txt").write_text("two\n")
    _git(upstream, *_IDENTITY, "commit", "-q", "-am", "upstream-2")
    _git(upstream, "push", "-q", str(bare), "main")
    head = _git(project, "rev-parse", "HEAD")

    registry = _registry(tmp_path, "https://github.com/../..")
    monkeypatch.chdir(project)
    try:
        resolve_dataset("evil@1.0", registry=registry)
    except Exception as exc:  # checked below, after the repository's state
        error: Exception | None = exc
    else:
        error = None

    # The user's repository first: on the branch, at its commit, full history.
    branch = _git(project, "rev-parse", "--abbrev-ref", "HEAD")
    assert branch == "mywork", f"the user's branch was switched to {branch!r}"
    assert _git(project, "rev-parse", "HEAD") == head
    assert not (project / ".git" / "shallow").exists()
    assert (project / "mine.txt").is_file()
    assert not (project / ".cache").exists()
    assert isinstance(error, DatasetResolutionError), error
    assert "Unsupported git_url" in str(error)


@pytest.mark.parametrize(
    "git_url",
    [
        "https://github.com/../..",
        "git@github.com:../..",
        "https://github.com/acme/..",
        "https://github.com/acme/...git",
        "https://github.com/acme/.",
        "https://github.com/./repo",
        "https://github.com/-acme/repo",
        "https://github.com/acme/re po",
        "https://github.com/acme/%2e%2e",
    ],
)
def test_a_git_url_that_is_not_a_plain_github_name_is_refused(git_url):
    with pytest.raises(DatasetResolutionError, match="Unsupported git_url"):
        dr._github_org_repo(git_url)


@pytest.mark.parametrize(
    ("git_url", "repo"),
    [
        ("https://github.com/acme/public-bench.git", "acme/public-bench"),
        ("https://github.com/Factory-AI/legacy-bench.git", "Factory-AI/legacy-bench"),
        ("git@github.com:acme/re.po.git", "acme/re.po"),
        ("https://github.com/acme/repo/", "acme/repo"),
        ("https://github.com/acme/.github", "acme/.github"),
        ("https://github.com/acme/under_score", "acme/under_score"),
    ],
)
def test_real_github_urls_still_resolve(git_url, repo):
    assert dr._github_org_repo(git_url) == repo


def _no_resolve(*args, **kwargs):
    raise AssertionError("the source cache must not be touched")


@pytest.mark.parametrize(
    "commit", ["--upload-pack=touch /tmp/victim-a", "main", "abc1234", "a" * 39]
)
def test_a_git_commit_id_that_is_not_a_commit_id_is_refused(
    tmp_path, monkeypatch, commit
):
    monkeypatch.setattr(br, "resolve_source_with_metadata", _no_resolve)
    registry = _registry(tmp_path, "https://github.com/acme/bench.git", commit)
    with pytest.raises(DatasetResolutionError, match="neither a full commit id"):
        resolve_dataset("evil@1.0", registry=registry)


@pytest.mark.parametrize("commit", ["A" * 40, "b" * 64, "head", "HEAD"])
def test_full_commit_ids_and_head_reach_the_cache(tmp_path, monkeypatch, commit):
    asked: list[str | None] = []

    def capture(repo, path=None, ref=None):
        asked.append(ref)
        raise RuntimeError("stop here")

    monkeypatch.setattr(br, "resolve_source_with_metadata", capture)
    registry = _registry(tmp_path, "https://github.com/acme/bench.git", commit)
    with pytest.raises(RuntimeError, match="stop here"):
        resolve_dataset("evil@1.0", registry=registry)
    assert asked == ["HEAD" if commit.lower() == "head" else commit.lower()]


@pytest.fixture
def no_git(tmp_path, monkeypatch) -> Path:
    """A cache under tmp_path, and any subprocess (git) is a test failure."""

    def refuse(*args, **kwargs):
        raise AssertionError(f"ran {args!r}")

    cache = tmp_path / "cache"
    monkeypatch.setattr(br, "_cache_dir", lambda: cache)
    monkeypatch.setattr(br.subprocess, "run", refuse)
    return cache


@pytest.mark.parametrize("repo", ["../..", "acme/..", "./x", "-x/y", "acme/a/b"])
def test_the_cache_refuses_a_repo_that_is_not_one_folder_name(no_git, repo):
    with pytest.raises(ValueError, match="Invalid repo"):
        br.resolve_source_with_metadata(repo)
    with pytest.raises(ValueError, match="Invalid repo"):
        br.resolve_source(repo)
    assert not no_git.exists()


def test_the_cache_refuses_a_ref_that_git_would_read_as_an_option(no_git):
    with pytest.raises(ValueError, match="cannot start with '-'"):
        br.resolve_source_with_metadata("acme/bench", ref="--upload-pack=x")


def test_the_cache_refuses_a_symlink_that_leads_out_of_it(tmp_path, no_git):
    outside = tmp_path / "victim-a"
    outside.mkdir()
    no_git.mkdir()
    (no_git / "acme").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="resolves outside"):
        br.resolve_source_with_metadata("acme/bench")
    assert list(outside.iterdir()) == []
