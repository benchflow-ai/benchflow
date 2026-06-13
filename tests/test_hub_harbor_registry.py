import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.hub import harbor_registry
from benchflow.hub.harbor_registry import (
    HarborTaskRef,
    check_harbor_registry,
    dataclass_record,
    harbor_env_uid,
    load_harbor_registry,
    records_from_jsonl,
    records_summary,
    records_to_markdown,
    select_harbor_tasks,
)


def _write_task(root: Path, name: str, *, valid: bool = True) -> Path:
    task = root / "tasks" / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        "[agent]\ntimeout_sec = 300\n[verifier]\ntimeout_sec = 120\n"
    )
    if valid:
        (task / "instruction.md").write_text("Do the task.\n")
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests" / "test.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    return task


def _write_registry(tmp_path: Path) -> Path:
    repo = tmp_path / "source-repo"
    _write_task(repo, "quixbugs-java-bitcount")
    _write_task(repo, "quixbugs-java-breadth_first_search", valid=False)
    _write_task(repo, "quixbugs-java-bucketsort")

    registry = [
        {
            "name": "quixbugs",
            "version": "1.0",
            "description": "Local fixture using real Harbor registry task names",
            "tasks": [
                {
                    "name": "quixbugs-java-bitcount",
                    "git_url": str(repo),
                    "git_commit_id": "HEAD",
                    "path": "tasks/quixbugs-java-bitcount",
                },
                {
                    "name": "quixbugs-java-breadth_first_search",
                    "git_url": str(repo),
                    "git_commit_id": "HEAD",
                    "path": "tasks/quixbugs-java-breadth_first_search",
                },
                {
                    "name": "quixbugs-java-bucketsort",
                    "git_url": str(repo),
                    "git_commit_id": "HEAD",
                    "path": "tasks/quixbugs-java-bucketsort",
                },
            ],
        }
    ]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    return path


def test_select_harbor_tasks_caps_per_dataset(tmp_path: Path) -> None:
    registry_path = _write_registry(tmp_path)
    registry = load_harbor_registry(registry_path)

    selected = select_harbor_tasks(registry, tasks_per_dataset=2)

    assert [ref.task for ref in selected] == [
        "quixbugs-java-bitcount",
        "quixbugs-java-breadth_first_search",
    ]
    assert selected[0].dataset == "quixbugs"
    assert selected[0].version == "1.0"


def test_select_harbor_tasks_skips_malformed_registry_entries() -> None:
    registry = [
        {"tasks": []},
        {"name": "not-a-list", "tasks": "bad"},
        {
            "name": "quixbugs",
            "tasks": [
                "bad",
                {"name": "quixbugs-java-missing-path", "git_url": "/repo"},
                {
                    "name": "quixbugs-java-gcd",
                    "git_url": "/repo",
                    "path": "tasks/quixbugs-java-gcd",
                },
            ],
        },
    ]

    selected = select_harbor_tasks(registry, tasks_per_dataset=3)

    assert [ref.task for ref in selected] == ["quixbugs-java-gcd"]


def test_select_harbor_tasks_rejects_invalid_cap() -> None:
    with pytest.raises(ValueError, match="tasks_per_dataset"):
        select_harbor_tasks([], tasks_per_dataset=0)


def test_load_harbor_registry_rejects_non_list_json(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"tasks": []}))

    with pytest.raises(ValueError, match="JSON list"):
        load_harbor_registry(registry_path)


def test_harbor_registry_inventory_writes_jsonl(tmp_path: Path) -> None:
    registry_path = _write_registry(tmp_path)
    out = tmp_path / "inventory.jsonl"

    records = check_harbor_registry(
        registry_path,
        tasks_per_dataset=2,
        level="inventory",
        out=out,
    )

    assert len(records) == 2
    assert records[0]["badges"] == ["parse"]
    assert records[0]["status"] == "pass"
    assert records[0]["env_uid"] == "harbor:quixbugs/quixbugs-java-bitcount@HEAD"
    assert records[0]["hub_url"] == "https://hub.harborframework.com/"
    assert records_from_jsonl(out) == records


def test_records_from_jsonl_rejects_non_object_lines(tmp_path: Path) -> None:
    report = tmp_path / "bad.jsonl"
    report.write_text("\n[]\n")

    with pytest.raises(ValueError, match="line 2"):
        records_from_jsonl(report)


def test_harbor_registry_check_marks_invalid_task_failed(tmp_path: Path) -> None:
    registry_path = _write_registry(tmp_path)

    records = check_harbor_registry(
        registry_path,
        tasks_per_dataset=2,
        level="check",
        cache_dir=tmp_path / "cache",
    )

    assert records[0]["task"] == "quixbugs-java-bitcount"
    assert records[0]["badges"] == ["parse", "package", "check"]
    assert records[0]["status"] == "pass"
    assert records[1]["task"] == "quixbugs-java-breadth_first_search"
    assert records[1]["badges"] == ["parse", "package"]
    assert records[1]["status"] == "fail"
    assert "Missing required file: instruction.md" in records[1]["notes"]


def test_harbor_registry_check_rejects_unknown_level(tmp_path: Path) -> None:
    registry_path = _write_registry(tmp_path)

    with pytest.raises(ValueError, match="level"):
        check_harbor_registry(registry_path, level="oracle")


def test_harbor_registry_check_marks_remote_clone_error_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = [
        {
            "name": "quixbugs",
            "tasks": [
                {
                    "name": "quixbugs-java-bitcount",
                    "git_url": "https://example.com/missing.git",
                    "git_commit_id": "HEAD",
                    "path": "datasets/quixbugs/quixbugs-java-bitcount",
                }
            ],
        }
    ]
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry))

    def fail_git(args: list[str]) -> None:
        raise RuntimeError(f"git failed: {' '.join(args)}")

    monkeypatch.setattr(harbor_registry, "_run_git", fail_git)

    records = check_harbor_registry(
        registry_path,
        level="check",
        cache_dir=tmp_path / "cache",
    )

    assert records[0]["status"] == "blocked"
    assert "git failed: clone" in records[0]["blocked_reason"]


def test_materialize_harbor_task_supports_file_urls(tmp_path: Path) -> None:
    repo = tmp_path / "source-repo"
    task = _write_task(repo, "quixbugs-java-bitcount")
    ref = HarborTaskRef(
        dataset="quixbugs",
        version=None,
        task="quixbugs-java-bitcount",
        git_url=repo.as_uri(),
        git_commit_id=None,
        path="tasks/quixbugs-java-bitcount",
        index=0,
    )

    assert (
        harbor_registry.materialize_harbor_task(ref, cache_dir=tmp_path / "cache")
        == task
    )


def test_records_summary_markdown_and_dataclass_record() -> None:
    ref = HarborTaskRef(
        dataset="quixbugs",
        version="1",
        task="quixbugs-java-bitcount",
        git_url="https://github.com/example/repo.git",
        git_commit_id="abc",
        path="tasks/quixbugs-java-bitcount",
        index=0,
    )
    records = [
        {
            "dataset": "quixbugs",
            "task": "quixbugs-java-bitcount",
            "status": "pass",
            "badges": ["parse", "check"],
            "notes": [],
        },
        {
            "dataset": "quixbugs",
            "task": "quixbugs-java-breadth_first_search",
            "status": "blocked",
            "badges": ["parse"],
            "blocked_reason": "missing secret",
            "notes": [],
        },
    ]

    assert dataclass_record(ref) == {
        "dataset": "quixbugs",
        "version": "1",
        "task": "quixbugs-java-bitcount",
        "git_url": "https://github.com/example/repo.git",
        "git_commit_id": "abc",
        "path": "tasks/quixbugs-java-bitcount",
        "index": 0,
    }
    assert harbor_env_uid(ref) == "harbor:quixbugs/quixbugs-java-bitcount@abc"
    assert records_summary(records) == {
        "total": 2,
        "pass": 1,
        "fail": 0,
        "blocked": 1,
    }
    markdown = records_to_markdown(records)
    assert "| quixbugs | quixbugs-java-bitcount | pass | parse, check |  |" in markdown
    assert (
        "| quixbugs | quixbugs-java-breadth_first_search | blocked | parse | missing secret |"
        in markdown
    )


def test_run_git_reports_missing_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harbor_registry.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="git is required"):
        harbor_registry._run_git(["status"])


def test_run_git_wraps_subprocess_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harbor_registry.shutil, "which", lambda _: "/usr/bin/git")

    def fail_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=args[0],
            stderr="fatal: nope",
        )

    monkeypatch.setattr(harbor_registry.subprocess, "run", fail_run)

    with pytest.raises(RuntimeError, match="fatal: nope"):
        harbor_registry._run_git(["status"])


def test_harbor_registry_cli_check_outputs_report(tmp_path: Path) -> None:
    registry_path = _write_registry(tmp_path)
    out = tmp_path / "compat.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "hub",
            "check",
            "--registry",
            str(registry_path),
            "--level",
            "check",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    assert "Harbor compatibility:" in result.stdout
    assert "2 task refs" in result.stdout
    records = records_from_jsonl(out)
    assert [record["status"] for record in records] == ["pass", "fail"]
