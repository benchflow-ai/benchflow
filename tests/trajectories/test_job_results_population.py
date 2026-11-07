"""Job exports retain active attempts without consuming preserved aggregates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from benchflow.trajectories.results import (
    JOB_RESULTS_ERRORS_FILENAME,
    write_job_results_jsonl,
    write_rollout_results_jsonl,
)


def _write_trial(
    trial: Path,
    *,
    model: str,
    purpose: Literal["task", "reviewer"] = "task",
) -> dict:
    return write_rollout_results_jsonl(
        trial,
        task_name="task-a",
        rollout_name=trial.name,
        agent="oracle",
        agent_name="oracle",
        model=model,
        n_tool_calls=0,
        prompts=["Complete the task."],
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        error=None,
        verifier_error=None,
        purpose=purpose,
    )


def test_job_results_excludes_preserved_aggregate_keeps_active_attempts(tmp_path):
    """Guards job exports at commit c6b2fe0d98ca, including repeated exports."""
    job = tmp_path / "job"
    # Results-only trials are supported; no config/result marker is required.
    # A dot-named trial and repeated task/settings must remain separate rows.
    current = [
        _write_trial(job / ".task-a__first", model="model-a"),
        _write_trial(job / "task-a__second", model="model-a"),
        _write_trial(job / "task-a__third", model="model-b"),
    ]
    legacy = {
        "task_id": "task-b",
        "reward": 1.0,
        "info": {"model": "model-a"},
    }
    legacy_trial = job / "task-b__legacy"
    legacy_trial.mkdir()
    (legacy_trial / "results.jsonl").write_text(json.dumps(legacy) + "\n")
    _write_trial(job / "reviewer__one", model="model-b", purpose="reviewer")
    _write_trial(
        job / "task-a__third" / "reviewer__nested",
        model="model-b",
        purpose="reviewer",
    )
    unfinished = job / "task-c__unfinished"
    unfinished.mkdir()
    (unfinished / "config.json").write_text('{"task_name": "task-c"}\n')

    # A prior job aggregate is a saved input, not an additional attempt. One
    # row is identical to an active export; the other has an obsolete score.
    old_rows = [current[0], {**legacy, "reward": 0.0}]
    old_bytes = ("\n".join(json.dumps(row) for row in old_rows) + "\n").encode()
    backup = job / ".pre-0.8" / "results.jsonl"
    backup.parent.mkdir()
    backup.write_bytes(old_bytes)
    backup_stat = backup.stat()
    set_aside = job / ".set-aside" / "previous"
    set_aside.mkdir(parents=True)
    (set_aside / "results.jsonl").write_bytes(old_bytes)

    expected = [{**row, "example_id": 0} for row in current]
    expected.append({**legacy, "example_id": 1})
    # Re-aggregation must preserve the backup and the same active population.
    for _ in range(2):
        out = write_job_results_jsonl(job)
        assert out == job / "results.jsonl"
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert rows == expected
        assert len(rows) == 4
        assert [row["info"]["model"] for row in rows] == [
            "model-a",
            "model-a",
            "model-b",
            "model-a",
        ]
        assert sum(row["reward"] for row in rows) / len(rows) == 1.0
        assert not (job / JOB_RESULTS_ERRORS_FILENAME).exists()
        assert backup.read_bytes() == old_bytes
        assert backup.stat().st_mode == backup_stat.st_mode
        assert backup.stat().st_mtime_ns == backup_stat.st_mtime_ns
        assert (set_aside / "results.jsonl").read_bytes() == old_bytes


def test_job_results_backup_without_trial_exports_is_not_current_results(tmp_path):
    """Guards job exports at commit c6b2fe0d98ca for an unfinished job."""
    backup = tmp_path / ".pre-0.8" / "results.jsonl"
    backup.parent.mkdir()
    old_bytes = b'{"task_id": "task-a", "reward": 1.0}\n'
    backup.write_bytes(old_bytes)
    unfinished = tmp_path / "task-a__unfinished"
    unfinished.mkdir()
    (unfinished / "config.json").write_text('{"task_name": "task-a"}\n')

    assert write_job_results_jsonl(tmp_path) is None
    assert not (tmp_path / "results.jsonl").exists()
    assert not (tmp_path / JOB_RESULTS_ERRORS_FILENAME).exists()
    assert backup.read_bytes() == old_bytes
