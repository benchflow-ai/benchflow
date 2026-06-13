"""Evaluation coverage for running foreign tasks through inbound adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.adapters.inbound import UnsupportedInboundTaskError
from benchflow.cli.main import app
from benchflow.evaluation import Evaluation, EvaluationConfig
from benchflow.models import RolloutResult


def _write_browser_use_task(
    root: Path,
    *,
    dirname: str = "browser-use-task",
    task_id: str = "Open Local Page",
    expected_result: str = "browser-use-smoke: ready",
) -> Path:
    task_dir = root / dirname
    task_dir.mkdir()
    (task_dir / "browser-use-task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "benchmark": "browser-use",
                "category": "local-browser",
                "confirmed_task": (
                    "Use the browser fixture to report the page status. "
                    f"Final answer must be exactly: {expected_result}"
                ),
                "ground_truth": f"Final result must be exactly {expected_result}",
                "expected_result": expected_result,
                "url": "file:///app/browser_fixture/index.html",
                "timeout_sec": 120,
            },
            indent=2,
        )
        + "\n"
    )
    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "WORKDIR /app\n"
        "RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts "
        "/app/browser_fixture\n"
        "COPY browser_fixture/ /app/browser_fixture/\n"
    )
    fixture_dir = env_dir / "browser_fixture"
    fixture_dir.mkdir()
    (fixture_dir / "index.html").write_text(f"<p>{expected_result}</p>\n")
    solution_dir = task_dir / "solution"
    solution_dir.mkdir()
    (solution_dir / "solve.sh").write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"printf '{expected_result}\\n' > /app/final_result.txt\n"
    )
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "mkdir -p /logs/verifier /logs/artifacts\n"
        "if [ -f /app/final_result.txt ] && "
        f'[ "$(tr -d \'\\n\' < /app/final_result.txt)" = "{expected_result}" ]; '
        "then reward=1.0; else reward=0.0; fi\n"
        'printf "%s\\n" "$reward" > /logs/verifier/reward.txt\n'
        'printf \'{"reward": %s}\\n\' "$reward" > /logs/verifier/reward.json\n'
    )
    return task_dir


def _write_cookbook_osworld_task(root: Path) -> Path:
    task_dir = root / "use-computer-osworld-task"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Observe the desktop once, then stop.\n")
    (task_dir / "task.toml").write_text(
        """\
version = "1.0"

[task]
name = "osworld/ubuntu-smoke"
description = "OSWorld Ubuntu smoke task"
keywords = ["osworld", "ubuntu", "smoke"]

[metadata]
author_name = "Use.Computer"
difficulty = "smoke"
category = "desktop-automation"
tags = ["osworld", "ubuntu", "gui", "smoke"]

[verifier]
timeout_sec = 180

[agent]
timeout_sec = 180

[environment]
cpus = 4
memory_mb = 8192
allow_internet = true
"""
    )
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "osworld_task.json").write_text(
        json.dumps(
            {
                "id": "smoke__ubuntu-osworld",
                "instruction": "Observe the desktop once, then stop.",
                "config": [
                    {
                        "type": "execute",
                        "parameters": {
                            "command": [
                                "bash",
                                "-lc",
                                "printf 'setup-ok\\n' > /tmp/runner-osworld-setup-ok",
                            ],
                            "shell": False,
                            "until": {"returncode": 0},
                        },
                    }
                ],
                "evaluator": {
                    "func": "exact_match",
                    "result": {
                        "type": "vm_command_line",
                        "command": "cat /tmp/runner-osworld-setup-ok",
                    },
                },
            },
            indent=2,
        )
        + "\n"
    )
    return task_dir


def _write_unsupported_cookbook_task(root: Path) -> Path:
    task_dir = root / "use-computer-cuagym-task"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Take one screenshot, then stop.\n")
    (task_dir / "task.toml").write_text(
        """\
[metadata]
author_name = "Use.Computer"
difficulty = "smoke"
category = "desktop-automation"
tags = ["cuagym", "ubuntu", "smoke"]

[verifier]
timeout_sec = 180

[agent]
timeout_sec = 180

[environment]
cpus = 4
memory_mb = 8192
allow_internet = true
"""
    )
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("#!/bin/bash\necho unsupported\n")
    return task_dir


def test_evaluation_discovers_single_foreign_adapter_task(tmp_path: Path) -> None:
    """Guards BenchFlow 0.7 eval create accepting one adapted computer-use task."""
    task_dir = _write_cookbook_osworld_task(tmp_path)
    evaluation = Evaluation(
        tasks_dir=task_dir,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(agent="computer-use-smoke", environment="cua"),
    )

    task_dirs = evaluation._get_task_dirs()

    assert len(task_dirs) == 1
    native = task_dirs[0]
    assert native != task_dir
    assert native.name == "smoke__ubuntu-osworld"
    assert (native / "task.md").is_file()
    assert (native / "environment" / "Dockerfile").is_file()
    assert (native / "verifier" / "test.sh").is_file()


def test_evaluation_discovers_browser_use_adapter_task(tmp_path: Path) -> None:
    """Guards BenchFlow 0.7 eval create accepting one adapted browser-use task."""
    task_dir = _write_browser_use_task(tmp_path)
    evaluation = Evaluation(
        tasks_dir=task_dir,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(agent="browser-use-cli", environment="docker"),
    )

    task_dirs = evaluation._get_task_dirs()

    assert len(task_dirs) == 1
    native = task_dirs[0]
    assert native != task_dir
    assert native.name == "open-local-page"
    assert (native / "task.md").is_file()
    assert (native / "environment" / "Dockerfile").is_file()
    assert (native / "verifier" / "test.sh").is_file()


def test_evaluation_discovers_foreign_adapter_task_collection(
    tmp_path: Path,
) -> None:
    """Guards raw Browser Use task collections materializing into native dirs."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_browser_use_task(tasks_root, dirname="browser-a")
    _write_browser_use_task(
        tasks_root,
        dirname="browser-b",
        task_id="Open Local Page Two",
        expected_result="browser-use-smoke: second",
    )
    evaluation = Evaluation(
        tasks_dir=tasks_root,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(agent="browser-use-cli", environment="docker"),
    )

    task_dirs = evaluation._get_task_dirs()

    assert [path.name for path in task_dirs] == [
        "open-local-page",
        "open-local-page-two",
    ]
    for native in task_dirs:
        assert native.parent.name.startswith("benchflow-eval-tasks-")
        assert (native / "task.md").is_file()
        assert (native / "environment" / "Dockerfile").is_file()
        assert (native / "verifier" / "test.sh").is_file()


def test_evaluation_disambiguates_duplicate_foreign_adapter_task_ids(
    tmp_path: Path,
) -> None:
    """Duplicate foreign task IDs still produce distinct native run dirs."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_browser_use_task(tasks_root, dirname="browser-a")
    _write_browser_use_task(tasks_root, dirname="browser-b")
    evaluation = Evaluation(
        tasks_dir=tasks_root,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(agent="browser-use-cli", environment="docker"),
    )

    task_dirs = evaluation._get_task_dirs()

    assert [path.name for path in task_dirs] == [
        "open-local-page",
        "open-local-page-2",
    ]
    assert task_dirs[0] != task_dirs[1]
    assert all((path / "task.md").is_file() for path in task_dirs)


def test_evaluation_filters_foreign_task_by_adapter_task_id(tmp_path: Path) -> None:
    """Include/exclude filters can target either source dir or adapted task id."""
    task_dir = _write_cookbook_osworld_task(tmp_path)

    included = Evaluation(
        tasks_dir=task_dir,
        jobs_dir=tmp_path / "jobs-a",
        config=EvaluationConfig(include_tasks={"smoke__ubuntu-osworld"}),
    )
    excluded = Evaluation(
        tasks_dir=task_dir,
        jobs_dir=tmp_path / "jobs-b",
        config=EvaluationConfig(exclude_tasks={"smoke__ubuntu-osworld"}),
    )

    assert [path.name for path in included._get_task_dirs()] == [
        "smoke__ubuntu-osworld"
    ]
    assert excluded._get_task_dirs() == []


def test_evaluation_filters_foreign_collection_by_source_or_adapted_id(
    tmp_path: Path,
) -> None:
    """Collection filters can target source dirs or adapted task IDs."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_browser_use_task(tasks_root, dirname="browser-a")
    _write_browser_use_task(
        tasks_root,
        dirname="browser-b",
        task_id="Open Local Page Two",
        expected_result="browser-use-smoke: second",
    )

    include_by_source = Evaluation(
        tasks_dir=tasks_root,
        jobs_dir=tmp_path / "jobs-a",
        config=EvaluationConfig(include_tasks={"browser-b"}),
    )
    include_by_adapted = Evaluation(
        tasks_dir=tasks_root,
        jobs_dir=tmp_path / "jobs-b",
        config=EvaluationConfig(include_tasks={"open-local-page-two"}),
    )
    exclude_by_source = Evaluation(
        tasks_dir=tasks_root,
        jobs_dir=tmp_path / "jobs-c",
        config=EvaluationConfig(exclude_tasks={"browser-a"}),
    )

    assert [path.name for path in include_by_source._get_task_dirs()] == [
        "open-local-page-two"
    ]
    assert [path.name for path in include_by_adapted._get_task_dirs()] == [
        "open-local-page-two"
    ]
    assert [path.name for path in exclude_by_source._get_task_dirs()] == [
        "open-local-page-two"
    ]


def test_evaluation_reports_unsupported_foreign_adapter_task(tmp_path: Path) -> None:
    """Unsupported recognized foreign tasks fail before rollout launch."""
    task_dir = _write_unsupported_cookbook_task(tmp_path)
    evaluation = Evaluation(tasks_dir=task_dir, jobs_dir=tmp_path / "jobs")

    with pytest.raises(UnsupportedInboundTaskError) as exc:
        evaluation._get_task_dirs()

    assert exc.value.report.source == "use-computer-cookbook"
    assert exc.value.report.dataset == "cuagym"
    assert "provider-honest setup/runtime" in (exc.value.report.reason or "")


def test_eval_create_reports_unsupported_foreign_adapter_task(
    tmp_path: Path,
) -> None:
    """`bench eval create` reports unsupported recognized adapters cleanly."""
    task_dir = _write_unsupported_cookbook_task(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--tasks-dir",
            str(task_dir),
            "--agent",
            "computer-use-smoke",
            "--sandbox",
            "cua",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--concurrency",
            "1",
        ],
    )

    assert result.exit_code == 1
    assert "unsupported adapter task" in result.output
    assert "Adapter: use-computer-cookbook" in result.output
    assert "Dataset: cuagym" in result.output
    assert "CUA-Gym cookbook tasks need provider-honest setup/runtime" in result.output
    assert "Tags: cuagym, ubuntu, smoke" in result.output
    assert "Traceback (most recent call last)" not in result.output


def test_eval_create_json_reports_unsupported_foreign_adapter_task(
    tmp_path: Path,
) -> None:
    """Adapter adoption loops can consume unsupported-task failures as JSON."""
    task_dir = _write_unsupported_cookbook_task(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--tasks-dir",
            str(task_dir),
            "--agent",
            "computer-use-smoke",
            "--sandbox",
            "cua",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--concurrency",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "unsupported-adapter-task"
    assert payload["adapter"] == "use-computer-cookbook"
    assert payload["dataset"] == "cuagym"
    assert "provider-honest setup/runtime" in payload["reason"]
    assert payload["details"]["tags"] == ["cuagym", "ubuntu", "smoke"]
    assert "unsupported adapter task" not in result.stdout
    assert "Traceback (most recent call last)" not in result.output


def test_eval_create_runs_foreign_adapter_task_through_materialized_native_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """`bench eval create --tasks-dir <foreign-task>` reaches a native task dir."""
    task_dir = _write_cookbook_osworld_task(tmp_path)
    seen: list[str] = []

    async def fake_run_single_task(self, native_task: Path, cfg):
        assert native_task.name == "smoke__ubuntu-osworld"
        assert (native_task / "task.md").is_file()
        assert (native_task / "environment" / "Dockerfile").is_file()
        assert (native_task / "verifier" / "test.sh").is_file()
        seen.append(native_task.name)
        return RolloutResult(
            task_name=native_task.name,
            rewards={"reward": 1.0},
            agent=cfg.agent,
        )

    monkeypatch.setattr(
        "benchflow.evaluation.Evaluation._run_single_task",
        fake_run_single_task,
    )

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--tasks-dir",
            str(task_dir),
            "--agent",
            "computer-use-smoke",
            "--sandbox",
            "cua",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--concurrency",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == ["smoke__ubuntu-osworld"]
    assert "Score: 1/1 (100.0%)" in result.output


def test_eval_create_json_reports_foreign_adapter_run_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """`bench eval create --json` exposes the persisted summary for adapters."""
    task_dir = _write_browser_use_task(tmp_path)
    jobs_dir = tmp_path / "jobs"
    seen: list[str] = []

    async def fake_run_single_task(self, native_task: Path, cfg):
        assert native_task.name == "open-local-page"
        seen.append(native_task.name)
        return RolloutResult(
            task_name=native_task.name,
            rewards={"reward": 1.0},
            agent=cfg.agent,
        )

    monkeypatch.setattr(
        "benchflow.evaluation.Evaluation._run_single_task",
        fake_run_single_task,
    )

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--tasks-dir",
            str(task_dir),
            "--agent",
            "browser-use-cli",
            "--sandbox",
            "docker",
            "--jobs-dir",
            str(jobs_dir),
            "--concurrency",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == ["open-local-page"]
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert payload["ok"] is True
    assert payload["jobs_dir"] == str(jobs_dir)
    assert payload["summary_path"] == str(jobs_dir / "summary.json")
    assert payload["result"]["total"] == 1
    assert payload["result"]["passed"] == 1
    assert payload["result"]["score"] == 1.0
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["passed"] == 1
    assert payload["summary"]["environment"] == "docker"
    assert "Score:" not in result.stdout


def test_eval_create_runs_browser_use_task_through_materialized_native_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """`bench eval create --tasks-dir <browser-use-task>` reaches native layout."""
    task_dir = _write_browser_use_task(tmp_path)
    seen: list[str] = []

    async def fake_run_single_task(self, native_task: Path, cfg):
        assert native_task.name == "open-local-page"
        assert (native_task / "task.md").is_file()
        assert (native_task / "environment" / "Dockerfile").is_file()
        assert (native_task / "verifier" / "test.sh").is_file()
        seen.append(native_task.name)
        return RolloutResult(
            task_name=native_task.name,
            rewards={"reward": 1.0},
            agent=cfg.agent,
        )

    monkeypatch.setattr(
        "benchflow.evaluation.Evaluation._run_single_task",
        fake_run_single_task,
    )

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--tasks-dir",
            str(task_dir),
            "--agent",
            "browser-use-cli",
            "--sandbox",
            "docker",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--concurrency",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == ["open-local-page"]
    assert "Score: 1/1 (100.0%)" in result.output


def test_eval_create_json_reports_plan_error(tmp_path: Path) -> None:
    """`bench eval create --json` reports planning failures as JSON."""
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--tasks-dir",
            str(tmp_path / "missing"),
            "--agent",
            "oracle",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload == {
        "status": "error",
        "ok": False,
        "reason": f"--tasks-dir not found: {tmp_path / 'missing'}",
    }
    assert "Traceback (most recent call last)" not in result.output


def test_eval_create_runs_foreign_adapter_task_collection_through_native_dirs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """`bench eval create --tasks-dir <foreign-collection>` adapts each task."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_browser_use_task(tasks_root, dirname="browser-a")
    _write_browser_use_task(
        tasks_root,
        dirname="browser-b",
        task_id="Open Local Page Two",
        expected_result="browser-use-smoke: second",
    )
    seen: list[str] = []

    async def fake_run_single_task(self, native_task: Path, cfg):
        assert native_task.name in {"open-local-page", "open-local-page-two"}
        assert (native_task / "task.md").is_file()
        assert (native_task / "environment" / "Dockerfile").is_file()
        assert (native_task / "verifier" / "test.sh").is_file()
        seen.append(native_task.name)
        return RolloutResult(
            task_name=native_task.name,
            rewards={"reward": 1.0},
            agent=cfg.agent,
        )

    monkeypatch.setattr(
        "benchflow.evaluation.Evaluation._run_single_task",
        fake_run_single_task,
    )

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--tasks-dir",
            str(tasks_root),
            "--agent",
            "browser-use-cli",
            "--sandbox",
            "docker",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--concurrency",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == ["open-local-page", "open-local-page-two"]
    assert "Score: 2/2 (100.0%)" in result.output
