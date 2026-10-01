"""``bench eval run`` on a spent login: the job's report, then exit 1.

Guards the dx/errors fix: ``Evaluation.run`` raises ``UsageLimitError`` once
its running trials finish; the CLI reports what finished instead of
printing the exception or a traceback.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from benchflow.agents.errors import UsageLimitError
from benchflow.cli.main import app
from benchflow.evaluation import EvaluationConfig, EvaluationResult
from benchflow.models import RolloutResult

HELLO = Path(__file__).parent / "examples" / "hello-world-task"


def _stopped(jobs: Path) -> UsageLimitError:
    err = UsageLimitError(
        "You've hit your weekly limit · resets Oct 2, 4pm (UTC)",
        login="CLAUDE_CODE_OAUTH_TOKEN (environment)",
        window="7-day",
        resets_at=datetime(2026, 10, 2, 16, tzinfo=UTC),
    )
    err.result = EvaluationResult(
        job_name="job",
        config=EvaluationConfig(),
        total=1,
        errored=1,
        job_dir=jobs / "job",
        results={
            "hello-world-task": RolloutResult(
                task_name="hello-world-task",
                error=str(err),
                error_category="usage_limit",
            )
        },
    )
    return err


def test_a_stopped_job_is_reported_and_exits_1(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"

    async def spent(self):
        raise _stopped(jobs)

    monkeypatch.setattr("benchflow.evaluation.Evaluation.run", spent)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(HELLO),
            "--agent",
            "oracle",
            "--jobs-dir",
            str(jobs),
            "--job-name",
            "job",
        ],
    )
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Score: 0/1" in result.output
    assert "Traceback" not in result.output


def test_a_job_that_never_started_prints_the_error(tmp_path, monkeypatch):
    async def spent(self):
        raise UsageLimitError("You've hit your weekly limit · resets Oct 2, 4pm (UTC)")

    monkeypatch.setattr("benchflow.evaluation.Evaluation.run", spent)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(HELLO),
            "--agent",
            "oracle",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )
    assert result.exit_code == 1
    assert "usage limit reached" in result.stderr
    assert "bench eval resume" in result.stderr
