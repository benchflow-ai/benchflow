"""Export and record fixes.

Job.to_csv, the call python-api.md shows,
failed on every job ('dict contains fields not in fieldnames'). Trial had no
agent/model/duration shortcuts. Records lacked the cache-token split, the settings
and a job label, so totals could not be reconciled from a CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path

import benchflow as bf
from tests.test_python_sdk_load_job import _job, _trial


def test_job_to_csv_writes_every_record_column(tmp_path: Path) -> None:
    job = bf.load_job(_job(tmp_path, "a", {"t1": 1.0, "t2": 0.0}))
    rows = list(csv.DictReader(job.to_csv(tmp_path / "trials.csv").open()))
    assert len(rows) == len(job.trials)
    assert set(job.to_records()[0]) == set(rows[0])


def test_trial_has_agent_model_and_duration(tmp_path: Path) -> None:
    t = bf.load_trial(_trial(tmp_path / "job", "hello"))
    assert (t.agent, t.model, t.duration_sec) == (
        "claude-agent-acp",
        "claude-haiku-4-5",
        90.0,
    )


def test_records_reconcile_tokens_and_carry_settings_and_job(tmp_path: Path) -> None:
    job = bf.load_job(_job(tmp_path, "a", {"t1": 1.0}))
    record = next(r for r in job.to_records() if r["task_name"] == "t1")
    for key in (
        "n_cache_read_tokens",
        "n_cache_creation_tokens",
        "job",
        "timeout_sec",
        "environment",
        "task_digest",
        "reasoning_effort",
    ):
        assert key in record, key
    assert record["job"] == str(job.path)
    assert record["timeout_sec"] == 300


def test_best_attempt_keeps_one_trial_per_agent_and_model(tmp_path: Path) -> None:
    """bf.run_batch of an oracle and another agent on one task
    writes both into one job folder, and load_job(attempts='best') silently
    kept only one of them (the key was task + folder).

    A batch folder holds no retries, so every rollout is kept (dx/sdk fix of
    the collapse from bf6e8412); in an Evaluation job, whose retries share its
    folder, attempts collapse, still one trial per agent and model."""
    job = tmp_path / "job"
    _trial(job, "t1", agent="oracle", model=None, suffix="00000001")
    _trial(
        job, "t1", agent="claude-agent-acp", model="claude-haiku-4-5", suffix="00000002"
    )
    _trial(job, "t1", agent="codex-acp", model="gpt-5.5", suffix="00000003")
    _trial(
        job,
        "t1",
        agent="codex-acp",
        model="gpt-5.5",
        reward=None,
        error="x",
        suffix="00000004",
    )
    loaded = bf.load_job(job)
    assert sorted(t.agent for t in loaded.trials) == [
        "claude-agent-acp",
        "codex-acp",
        "codex-acp",
        "oracle",
    ]
    (job / "evaluation.json").write_text("{}")
    loaded = bf.load_job(job)
    assert sorted(t.agent for t in loaded.trials) == [
        "claude-agent-acp",
        "codex-acp",
        "oracle",
    ]
    assert next(t for t in loaded.trials if t.agent == "codex-acp").reward == 1.0


def test_rows_without_a_task_name_are_refused_not_collapsed(tmp_path: Path) -> None:
    """A results.jsonl in another shape (id, slug,
    variant, reward) loaded with empty task names and collapsed to 1 trial
    ('1/1 scored passed') with no warning."""
    import json

    import pytest

    folder = tmp_path / "foreign"
    folder.mkdir()
    (folder / "results.jsonl").write_text(
        "".join(
            json.dumps({"id": i, "slug": f"s{i}", "variant": "oracle", "reward": 1.0})
            + "\n"
            for i in range(5)
        )
    )
    with pytest.raises(ValueError, match=r"5 of 5 rows .* no task name"):
        bf.load_job(folder)


def test_a_json_file_is_refused_with_its_job_folder_named(tmp_path: Path) -> None:
    """load_job('jobs/x/summary.json') printed one 'Skipping
    unreadable row' line per line of the file, then 'no trial under jobs/x'."""
    import pytest

    job = tmp_path / "jobs" / "2026-01-01__12-00-00"
    _trial(job, "t1")
    (tmp_path / "jobs" / "summary.json").write_text('{\n  "total": 1\n}\n')
    with pytest.raises(ValueError, match="pass a job directory"):
        bf.load_job(tmp_path / "jobs" / "summary.json")


def test_evaluation_csv_has_execution_and_assessment(tmp_path: Path) -> None:
    """The CSV could not tell an infra failure from a zero."""
    import csv

    from benchflow import RolloutResult
    from benchflow.batch import Results

    rows = Results(
        [
            RolloutResult(
                "a",
                rewards={"reward": 1.0},
                n_input_tokens=5,
                n_cache_read_tokens=7,
                total_tokens=12,
            ),
            RolloutResult(
                "b",
                rewards=None,
                error="agent timed out after 60s",
                error_category="timeout",
            ),
        ]
    )
    with rows.to_csv(tmp_path / "r.csv").open() as handle:
        got = list(csv.DictReader(handle))
    assert [(r["execution"], r["assessment"]) for r in got] == [
        ("completed", "scored"),
        ("timed_out", "unscored"),
    ]
    assert got[0]["n_cache_read_tokens"] == "7"
    assert rows.to_records()[1]["execution"] == "timed_out"
