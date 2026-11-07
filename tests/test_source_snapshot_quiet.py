"""The per-commit source snapshot is created quietly.

Regression test: `bench eval run
--source-repo ... --source-path ...` clones with `--quiet`, but the snapshot
worktree it then adds for the resolved commit did not, so the console got
"Preparing worktree (detached HEAD …)" and, for SkillsBench (2,876 files),
git's "Updating files: n%" progress as one very long line in logs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from benchflow._utils import benchmark_repos


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_snapshot_worktree_add_prints_nothing(tmp_path, monkeypatch, capfd):
    repo = tmp_path / "repo"
    (repo / "tasks" / "t1").mkdir(parents=True)
    (repo / "tasks" / "t1" / "task.md").write_text("hello\n")
    _git(tmp_path, "init", "-q", str(repo))
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
        "commit",
        "-q",
        "-m",
        "init",
    )
    sha = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(benchmark_repos, "_cache_dir", lambda: tmp_path / "cache")
    capfd.readouterr()

    snapshot = benchmark_repos._snapshot_repo_root(
        repo, org="org", repo_name="repo", resolved_sha=sha
    )

    out, err = capfd.readouterr()
    assert (snapshot / "tasks" / "t1" / "task.md").read_text() == "hello\n"
    assert out == ""
    assert err == ""
