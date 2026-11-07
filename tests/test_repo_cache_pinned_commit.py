"""A cached checkout already at a pinned full commit id is not fetched again.

``bench agent list`` reads the agent catalog at a pinned commit
(``benchflow-ai/agents@<sha>``) and used to ``git fetch`` it from GitHub on
every run, about 0.4 s and a network round trip, although a full commit id
names fixed content. Branches, tags and short ids still refresh. Real git on a
local repository; the fetch itself is recorded, never run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from benchflow._utils import benchmark_repos as br

_IDENTITY = ("-c", "user.name=BenchFlow Test", "-c", "user.email=test@example.com")


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def cached(tmp_path, monkeypatch):
    """A cache holding org/repo, checked out at its only commit."""
    cache = tmp_path / "cache"
    repo = cache / "org" / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "README.md").write_text("catalog\n")
    _git(repo, "add", "-A")
    _git(repo, *_IDENTITY, "commit", "-q", "-m", "one")
    monkeypatch.setattr(br, "_cache_dir", lambda: cache)
    fetched: list[str] = []
    monkeypatch.setattr(
        br, "_checkout_fetched_ref", lambda root, ref: fetched.append(ref)
    )
    return repo, _git(repo, "rev-parse", "HEAD"), fetched


def test_pinned_commit_already_checked_out_is_not_fetched(cached):
    repo, head, fetched = cached
    assert br._clone_repo_unlocked("org", "repo", ref=head) == repo
    assert fetched == []


@pytest.mark.parametrize("kind", ["branch", "short id", "other commit"])
def test_other_refs_still_refresh(cached, kind):
    repo, head, fetched = cached
    ref = {"branch": "main", "short id": head[:12], "other commit": "f" * 40}[kind]
    assert br._clone_repo_unlocked("org", "repo", ref=ref) == repo
    assert fetched == [ref]
