"""The gate receipts a release's acceptance record names (schema 2).

An acceptance record binds bytes to a commit; schema 2 also names the gate
runs on that commit, and public-release checks each against GitHub before it
publishes: the run passed, it is the expected workflow, and it ran on the
accepted source (or on a commit whose diff from it changes only the version).
Without the cross-repository token the check is skipped with a loud notice
(the lead's ruling), never silently.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.check_gate_receipts import (
    GATES_REQUIRED_FROM,
    REQUIRED,
    GateError,
    check_gate_receipts,
    main,
)
from tools.release_promote_check import version_only_paths

SHA = "a" * 40


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str]:
    r = tmp_path / "src"
    r.mkdir()
    _git(r, "init", "-q")
    (r / "pyproject.toml").write_text(
        '[project]\nname = "benchflow"\nversion = "0.10.0rc6"\n'
    )
    (r / "CHANGELOG.md").write_text("# Changelog\n")
    (r / "app.py").write_text("print(1)\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "rc6")
    return r, _git(r, "rev-parse", "HEAD")


def _record(source: str, gates: dict | None = None, version: str = "0.10.0rc6") -> dict:
    record = {
        "schema_version": 2,
        "version": version,
        "status": "accepted",
        "source_commit": source,
        "acceptance_receipt_sha256": "b" * 64,
        "artifacts": {},
    }
    if gates is not None:
        record["gates"] = gates
    return record


def _gates(head: str) -> dict:
    return {
        name: {
            "repo": "benchflow-ai/benchflow-private",
            "run_id": 100 + i,
            "head_sha": head,
        }
        for i, name in enumerate(sorted(REQUIRED))
    }


def _runs(head: str, **override) -> dict:
    runs = {}
    for i, name in enumerate(sorted(REQUIRED)):
        runs[100 + i] = {
            "conclusion": "success",
            "path": REQUIRED[name],
            "head_sha": head,
        }
    for run_id, fields in override.items():
        runs[int(run_id)].update(fields)
    return runs


def _fetch(runs):
    return lambda repo, run_id: runs[run_id]


def test_receipts_on_the_accepted_source_pass(repo):
    path, head = repo
    lines = check_gate_receipts(_record(head, _gates(head)), path, _fetch(_runs(head)))
    assert len(lines) == len(REQUIRED)


@pytest.mark.parametrize(
    ("field", "value", "words"),
    [
        ("conclusion", "failure", "did not pass"),
        ("path", ".github/workflows/other.yml", "not the"),
        ("head_sha", SHA, "ran on"),
    ],
)
def test_a_receipt_that_does_not_hold_is_refused(repo, field, value, words):
    path, head = repo
    runs = _runs(head, **{"100": {field: value}})
    with pytest.raises(GateError, match=words):
        check_gate_receipts(_record(head, _gates(head)), path, _fetch(runs))


def test_a_missing_required_gate_is_refused(repo):
    path, head = repo
    gates = _gates(head)
    gates.pop(sorted(REQUIRED)[0])
    with pytest.raises(GateError, match="no receipt"):
        check_gate_receipts(_record(head, gates), path, _fetch(_runs(head)))


def test_a_gate_on_a_commit_that_changed_more_than_the_version_is_refused(repo):
    path, head = repo
    (path / "app.py").write_text("print(2)\n")
    _git(path, "commit", "-qam", "code change")
    later = _git(path, "rev-parse", "HEAD")
    with pytest.raises(GateError, match=r"app\.py"):
        check_gate_receipts(_record(later, _gates(head)), path, _fetch(_runs(head)))


def test_a_gate_on_a_version_only_parent_carries_over(repo):
    path, head = repo
    text = (path / "pyproject.toml").read_text().replace("0.10.0rc6", "0.10.0")
    (path / "pyproject.toml").write_text(text)
    (path / "CHANGELOG.md").write_text("# Changelog\n\n## 0.10.0\n")
    _git(path, "commit", "-qam", "final")
    final = _git(path, "rev-parse", "HEAD")
    assert version_only_paths(path, head, final) == []
    record = _record(final, _gates(head), version="0.10.0")
    assert check_gate_receipts(record, path, _fetch(_runs(head)))


def test_a_new_record_must_name_its_gates(repo):
    path, head = repo
    with pytest.raises(GateError, match="gates"):
        check_gate_receipts(_record(head, None), path, _fetch({}))
    record = _record(head, None, version="0.9.4")
    record["schema_version"] = 1
    assert GATES_REQUIRED_FROM == "0.10.0rc6"
    assert check_gate_receipts(record, path, _fetch({})) == [
        "0.9.4: schema 1 record, released before gate receipts were required"
    ]


def test_without_the_token_the_check_is_skipped_loudly(
    repo, tmp_path, monkeypatch, capsys
):
    path, head = repo
    target = path / ".github" / "release-acceptance" / "0.10.0rc6.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(_record(head, _gates(head))))
    monkeypatch.delenv("GATE_RECEIPTS_TOKEN", raising=False)
    assert main(["--repo", str(path)]) == 0
    out = capsys.readouterr()
    assert "::warning" in out.out + out.err
    assert "NOT checked" in out.out + out.err
