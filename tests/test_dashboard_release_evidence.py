"""Tests for the dashboard release-evidence freshness gate (#366).

The dashboard is a release-evidence surface, so it must refuse to publish
stale or missing test evidence by default — and must surface the freshness
verdict in ``data.json`` so the UI can warn on it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from dashboard import generate


def _write_junit(path: Path, *, passed: int = 1) -> None:
    """Minimal valid junit.xml with ``passed`` passing testcases."""
    cases = "\n".join(
        f'    <testcase classname="suite_a" name="t{i}" time="0.01"/>'
        for i in range(passed)
    )
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<testsuite name="dashboard" tests="{passed}" failures="0" errors="0" '
        'skipped="0">\n'
        f"{cases}\n"
        "</testsuite>\n"
    )


def _stamp(path: Path, when: datetime) -> None:
    ts = when.timestamp()
    os.utime(path, (ts, ts))


def _make_repo(tmp_path: Path, *, version: str = "0.5.0.dev0") -> Path:
    """Build a minimal repo layout mirroring what generate.py reads."""
    repo = tmp_path / "repo"
    dash = repo / "dashboard"
    dash.mkdir(parents=True)
    pyproject = repo / "pyproject.toml"
    pyproject.write_text(f'[project]\nname = "x"\nversion = "{version}"\n')
    subprocess.run(
        ["git", "init", "-q"], cwd=repo, check=True, stdout=subprocess.DEVNULL
    )
    subprocess.run(
        ["git", "config", "user.email", "t@example.com"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return repo


def _retarget_generate(monkeypatch, repo: Path) -> None:
    dash = repo / "dashboard"
    monkeypatch.setattr(generate, "ROOT", repo)
    monkeypatch.setattr(generate, "DASH", dash)
    monkeypatch.setattr(generate, "JUNIT", dash / "junit.xml")
    monkeypatch.setattr(generate, "OUT", dash / "data.json")
    monkeypatch.setattr(generate, "PYPROJECT_TOML", repo / "pyproject.toml")
    monkeypatch.setattr(generate, "ARCHITECTURE_MD", dash / "architecture.md")
    # The fake repo has no roadmap helper or experiments/labs trees; stub the
    # collectors so the freshness gate is what we are actually exercising.
    monkeypatch.setattr(
        generate,
        "collect_roadmap",
        lambda: {"source": {"kind": "linear-live"}, "milestones": []},
    )
    monkeypatch.setattr(generate, "collect_experiments", lambda: [])
    monkeypatch.setattr(generate, "collect_jobs", lambda: _empty_jobs(dash))


def _empty_jobs(dash: Path) -> dict:
    return {
        "groups": [],
        "total_tasks": 0,
        "total_runs": 0,
        "archived_tasks": 0,
        "archived_runs": 0,
        "source": {
            "path": str(dash / "jobs"),
            "label": "jobs",
            "configured": False,
            "remembered": False,
            "available": False,
            "latest_modified_at": None,
        },
    }


def test_collect_tests_embeds_junit_modified_at(tmp_path: Path, monkeypatch):
    """Tests payload carries junit.xml's mtime so freshness can be checked."""
    repo = _make_repo(tmp_path)
    junit = repo / "dashboard" / "junit.xml"
    _write_junit(junit)
    when = datetime(2026, 5, 22, 12, 0, 0)
    _stamp(junit, when)
    _retarget_generate(monkeypatch, repo)

    tests = generate.collect_tests()

    assert tests["available"] is True
    assert tests["modified_at"] == "2026-05-22 12:00:00"
    assert tests["summary"] == {"passed": 1, "failed": 0, "skipped": 0, "total": 1}


def test_collect_tests_marks_missing_junit_without_modified_at(
    tmp_path: Path, monkeypatch
):
    """Missing junit.xml stays an explicit unavailable state with no mtime."""
    repo = _make_repo(tmp_path)
    _retarget_generate(monkeypatch, repo)

    tests = generate.collect_tests()

    assert tests["available"] is False
    assert tests["modified_at"] is None


def test_release_evidence_flags_missing_junit_as_stale(tmp_path: Path, monkeypatch):
    """Missing test evidence is the strongest stale signal."""
    repo = _make_repo(tmp_path)
    _retarget_generate(monkeypatch, repo)

    evidence = generate.collect_release_evidence(generate.collect_tests())

    assert evidence["fresh"] is False
    assert evidence["junit_modified_at"] is None
    assert any("junit.xml missing" in r for r in evidence["stale_reasons"])


def test_release_evidence_flags_junit_older_than_pyproject(
    tmp_path: Path, monkeypatch
):
    """A version bump after the last suite run leaves the evidence stale."""
    repo = _make_repo(tmp_path)
    junit = repo / "dashboard" / "junit.xml"
    _write_junit(junit)
    _stamp(junit, datetime(2026, 5, 20, 12, 0, 0))
    _stamp(repo / "pyproject.toml", datetime(2026, 5, 22, 12, 0, 0))
    _retarget_generate(monkeypatch, repo)

    evidence = generate.collect_release_evidence(generate.collect_tests())

    assert evidence["fresh"] is False
    assert any("pyproject.toml" in r for r in evidence["stale_reasons"])
    assert evidence["version"] == "0.5.0.dev0"


def test_release_evidence_flags_junit_older_than_head_commit(
    tmp_path: Path, monkeypatch
):
    """Code changes since the last suite run leave the evidence stale."""
    repo = _make_repo(tmp_path)
    junit = repo / "dashboard" / "junit.xml"
    _write_junit(junit)
    # Pin junit and pyproject well before the HEAD commit time so only the
    # commit-vs-junit comparison fails.
    long_ago = datetime.now() - timedelta(days=365)
    _stamp(junit, long_ago)
    _stamp(repo / "pyproject.toml", long_ago)
    _retarget_generate(monkeypatch, repo)

    evidence = generate.collect_release_evidence(generate.collect_tests())

    assert evidence["fresh"] is False
    assert any("HEAD commit" in r for r in evidence["stale_reasons"])


def test_release_evidence_is_fresh_when_junit_newer_than_release_surface(
    tmp_path: Path, monkeypatch
):
    """Junit newer than both pyproject and HEAD passes the gate."""
    repo = _make_repo(tmp_path)
    junit = repo / "dashboard" / "junit.xml"
    _write_junit(junit)
    future = datetime.now() + timedelta(hours=1)
    _stamp(junit, future)
    _retarget_generate(monkeypatch, repo)

    evidence = generate.collect_release_evidence(generate.collect_tests())

    assert evidence["fresh"] is True
    assert evidence["stale_reasons"] == []
    assert evidence["junit_modified_at"] is not None
    assert evidence["pyproject_modified_at"] is not None
    assert evidence["head_committed_at"] is not None


def test_main_refuses_to_publish_when_evidence_is_stale(tmp_path: Path, monkeypatch):
    """``main()`` exits non-zero and does not write data.json when stale."""
    repo = _make_repo(tmp_path)
    _retarget_generate(monkeypatch, repo)
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate.py", "--allow-missing-linear"],
    )
    # No junit.xml — evidence is stale.

    rc = generate.main()

    assert rc == 1
    assert not (repo / "dashboard" / "data.json").exists()


def test_main_publishes_with_allow_stale_evidence_for_local_dev(
    tmp_path: Path, monkeypatch
):
    """``--allow-stale-evidence`` is the documented escape hatch for local UI dev."""
    repo = _make_repo(tmp_path)
    _retarget_generate(monkeypatch, repo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--allow-missing-linear",
            "--allow-stale-evidence",
        ],
    )

    rc = generate.main()

    assert rc == 0
    data = json.loads((repo / "dashboard" / "data.json").read_text())
    assert data["release_evidence"]["fresh"] is False
    assert data["summary"]["release_evidence_fresh"] is False


def test_main_publishes_when_evidence_is_fresh(tmp_path: Path, monkeypatch):
    """Happy path — fresh junit lets ``main()`` write data.json normally."""
    repo = _make_repo(tmp_path)
    junit = repo / "dashboard" / "junit.xml"
    _write_junit(junit)
    _stamp(junit, datetime.now() + timedelta(hours=1))
    _retarget_generate(monkeypatch, repo)
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate.py", "--allow-missing-linear"],
    )

    rc = generate.main()

    assert rc == 0
    data = json.loads((repo / "dashboard" / "data.json").read_text())
    assert data["release_evidence"]["fresh"] is True
    assert data["summary"]["release_evidence_fresh"] is True


def test_release_evidence_handles_missing_pyproject(tmp_path: Path, monkeypatch):
    """A repo with no pyproject.toml falls back to the HEAD signal only."""
    repo = _make_repo(tmp_path)
    junit = repo / "dashboard" / "junit.xml"
    _write_junit(junit)
    _stamp(junit, datetime.now() + timedelta(hours=1))
    pyproject = repo / "pyproject.toml"
    pyproject.unlink()
    _retarget_generate(monkeypatch, repo)

    evidence = generate.collect_release_evidence(generate.collect_tests())

    assert evidence["version"] is None
    assert evidence["pyproject_modified_at"] is None
    # No pyproject signal, junit newer than HEAD ⇒ fresh.
    assert evidence["fresh"] is True


@pytest.mark.parametrize(
    "version_line",
    [
        'version = "0.5.0.dev0"',
        'version  =  "1.2.3"',
        '\tversion = "9.9.9rc1"',
    ],
)
def test_project_version_parses_common_pyproject_formats(
    tmp_path: Path, monkeypatch, version_line: str
):
    """The version regex tolerates whitespace variants seen in real configs."""
    repo = _make_repo(tmp_path)
    (repo / "pyproject.toml").write_text(f"[project]\nname = \"x\"\n{version_line}\n")
    _retarget_generate(monkeypatch, repo)

    expected = version_line.split('"')[1]
    assert generate._project_version() == expected
