"""An rc becomes final by a version-only commit (tools/release_promote_check.py).

The cadence promotes an rc that has been out seven days with no blocker. Its
gate receipts carry over only when the final's source differs from the rc's
in the version alone: replacing the old version with the new one in each
changed file gives the new file, apart from the changelog and the acceptance
records.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.release_promote_check import main, version_only_paths


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    _git(r, "init", "-q")
    (r / "pyproject.toml").write_text(
        '[project]\nversion = "0.10.0rc6"\nname = "benchflow"\n'
    )
    (r / "uv.lock").write_text(
        '[[package]]\nname = "benchflow"\nversion = "0.10.0rc6"\n'
    )
    (r / "docs").mkdir()
    (r / "docs" / "install.md").write_text(
        "uv tool install 'benchflow==0.10.0rc6'\nother\n"
    )
    (r / "app.py").write_text("x = 1\n")
    (r / "CHANGELOG.md").write_text("# Changelog\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "rc")
    return r


def _bump(repo: Path) -> None:
    for name in ("pyproject.toml", "uv.lock", "docs/install.md"):
        p = repo / name
        p.write_text(p.read_text().replace("0.10.0rc6", "0.10.0"))
    (repo / "CHANGELOG.md").write_text("# Changelog\n\n## 0.10.0\n- final\n")
    rec = repo / ".github" / "release-acceptance" / "0.10.0.json"
    rec.parent.mkdir(parents=True)
    rec.write_text("{}\n")


def test_a_version_bump_is_version_only(repo):
    base = _git(repo, "rev-parse", "HEAD")
    _bump(repo)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "final")
    assert version_only_paths(repo, base, "HEAD") == []
    assert main(["--repo", str(repo), base, "HEAD"]) == 0


def test_any_other_change_is_named(repo):
    base = _git(repo, "rev-parse", "HEAD")
    _bump(repo)
    (repo / "app.py").write_text("x = 2\n")
    (repo / "docs" / "install.md").write_text(
        (repo / "docs" / "install.md").read_text().replace("other", "changed")
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "final plus a change")
    assert version_only_paths(repo, base, "HEAD") == ["app.py", "docs/install.md"]
    assert main(["--repo", str(repo), base, "HEAD"]) == 1
