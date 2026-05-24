"""Single-task `bench eval create` must honor --include / --exclude (#401).

Before the fix the single-task branch called ``SDK().run`` directly and
ignored the include/exclude sets that the batch path applied.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.evaluation import Evaluation, EvaluationConfig


def _write_task(task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text('version = "1.0"\n')


def test_evaluation_single_task_dir_respects_include(tmp_path):
    """The single-task layout must drop the task when --include excludes it."""
    task_dir = tmp_path / "pass-json"
    _write_task(task_dir)
    cfg = EvaluationConfig(include_tasks={"no-such-task"})
    job = Evaluation(
        tasks_dir=task_dir, jobs_dir=tmp_path / "jobs", config=cfg
    )
    assert job._get_task_dirs() == []


def test_evaluation_single_task_dir_respects_exclude(tmp_path):
    """The single-task layout must drop the task when --exclude names it."""
    task_dir = tmp_path / "pass-json"
    _write_task(task_dir)
    cfg = EvaluationConfig(exclude_tasks={"pass-json"})
    job = Evaluation(
        tasks_dir=task_dir, jobs_dir=tmp_path / "jobs", config=cfg
    )
    assert job._get_task_dirs() == []


def test_cli_single_task_threads_include_into_evaluation_config(tmp_path):
    """`--include` reaches EvaluationConfig.include_tasks for --tasks-dir <task>."""
    task_dir = tmp_path / "pass-json"
    _write_task(task_dir)
    captured = {}

    async def fake_run(self):
        captured["include"] = self._config.include_tasks
        captured["exclude"] = self._config.exclude_tasks
        captured["tasks_dir"] = self._tasks_dir
        return SimpleNamespace(
            passed=1, total=1, score=1.0, errored=0, verifier_errored=0
        )

    with patch.object(Evaluation, "run", new=fake_run):
        result = CliRunner().invoke(
            app,
            [
                "eval",
                "create",
                "--tasks-dir",
                str(task_dir),
                "--include",
                "pass-json",
                "--exclude",
                "other-task",
                "--agent",
                "oracle",
                "--jobs-dir",
                str(tmp_path / "jobs"),
            ],
        )

    assert result.exit_code == 0, result.stdout
    assert captured["include"] == {"pass-json"}
    assert captured["exclude"] == {"other-task"}
    assert captured["tasks_dir"] == task_dir


def test_cli_single_task_dir_excluded_task_exits_nonzero(tmp_path, monkeypatch):
    """Repro from #401: `--exclude <single-task-name>` must not silently run it.

    Before the fix the single-task branch ignored --exclude and ran the
    task anyway.  Now the include/exclude filter zeroes the selection
    and the zero-task guard (#407) fails fast.
    """
    task_dir = tmp_path / "pass-json"
    _write_task(task_dir)

    # The fix must not even reach the agent — assert it doesn't.
    sdk_calls = {"count": 0}

    async def fake_sdk_run(self, **_kwargs):
        sdk_calls["count"] += 1
        return SimpleNamespace(rewards={"reward": 1.0})

    monkeypatch.setattr("benchflow.sdk.SDK.run", fake_sdk_run)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--tasks-dir",
            str(task_dir),
            "--exclude",
            "pass-json",
            "--agent",
            "oracle",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )

    assert result.exit_code == 1, result.stdout
    assert sdk_calls["count"] == 0, (
        "--exclude on a single-task dir must not run the excluded task (#401)."
    )


def test_cli_single_task_dir_unmatched_include_exits_nonzero(tmp_path, monkeypatch):
    """Repro from #401: `--include no-such-task` over a single-task dir.

    Before the fix BenchFlow ran the task anyway and emitted reward=1.0,
    silently overriding the user's filter contract.
    """
    task_dir = tmp_path / "pass-json"
    _write_task(task_dir)

    sdk_calls = {"count": 0}

    async def fake_sdk_run(self, **_kwargs):
        sdk_calls["count"] += 1
        return SimpleNamespace(rewards={"reward": 1.0})

    monkeypatch.setattr("benchflow.sdk.SDK.run", fake_sdk_run)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--tasks-dir",
            str(task_dir),
            "--include",
            "no-such-task",
            "--agent",
            "oracle",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )

    assert result.exit_code == 1, result.stdout
    assert sdk_calls["count"] == 0, (
        "--include with a missing name must not silently run the task (#401)."
    )
