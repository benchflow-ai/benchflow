"""Guards FF-sw-evaluation fixes for regressions at 363d815b."""

import json

import pytest

from benchflow.evaluation import Evaluation, EvaluationConfig


@pytest.mark.parametrize("mode", ["sequential-shared", "parallel-independent"])
@pytest.mark.parametrize(
    "error", ["ACP error -32603", "agent timeout", "sandbox startup failed"]
)
def test_resume_retains_terminal_errors_only_for_sequential(tmp_path, mode, error):
    trial = tmp_path / "job" / "task-a__one"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps({"task_name": "task-a", "error": error, "rewards": None})
    )
    job = object.__new__(Evaluation)
    job._jobs_dir, job._job_name = tmp_path, "job"
    job._config = EvaluationConfig(job_mode=mode)
    assert ("task-a" in job._get_completed_tasks()) == (mode == "sequential-shared")
