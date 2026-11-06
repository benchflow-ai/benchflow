"""Guards FF-sw-evaluation fixes for regressions at 363d815b."""

import json
from types import SimpleNamespace

import pytest

from benchflow.evaluation import Evaluation


@pytest.mark.parametrize(
    "existing", [{}, {"timing": {}, "integration_failure_info": {}}]
)
def test_fresh_loop_payload_preserves_persisted_trajectory(tmp_path, existing):
    trial = tmp_path / "job" / "task-a__one"
    trial.mkdir(parents=True)
    loop = {
        "strategy": "verify-retry",
        "first_pass_iteration": 1,
        "iterations_run": 2,
        "stop_reason": "passed_bar",
    }
    (trial / "result.json").write_text(json.dumps({"loop": loop}))
    job = object.__new__(Evaluation)
    job._jobs_dir, job._job_name = tmp_path, "job"
    payload = dict(existing)
    job._enrich_payload_with_persisted_timing(
        payload, SimpleNamespace(rollout_name=trial.name)
    )
    assert payload["loop"] == loop
