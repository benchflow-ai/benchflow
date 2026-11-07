"""Old runs from a task.toml-package runner read as BenchFlow trials.

The fixture is a real job folder that runner wrote for Terminal-Bench 4.1's
interleaved-vigenere (its oracle, reward 1), local paths removed, and a copy
of its trial that ended on an exception instead of a verifier result. Before,
the job's own result.json was read as the one trial and hid the trial folders
below it, and the reward under ``verifier_result`` was not read: the oracle's
1.0 showed as an unscored 0.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchflow._utils.result_paths import is_job_summary
from benchflow.jobs import load_job
from benchflow.metrics import collect_metrics

JOB = Path(__file__).parent / "fixtures" / "task_toml_results" / "job"


def test_the_jobs_own_result_is_not_a_trial():
    assert is_job_summary(JOB / "result.json")
    assert not is_job_summary(JOB / "interleaved-vigenere__pTZ5JVB" / "result.json")


def test_each_trial_folder_is_read_with_its_reward_and_error():
    trials = {t.path.name: t for t in load_job(JOB).trials}
    assert sorted(trials) == [
        "interleaved-vigenere__Errored1",
        "interleaved-vigenere__pTZ5JVB",
    ]
    passed, errored = (
        trials["interleaved-vigenere__pTZ5JVB"],
        trials["interleaved-vigenere__Errored1"],
    )
    assert passed.raw["rewards"] == {"reward": 1.0}
    assert passed.task_name == "interleaved-vigenere"
    assert passed.raw["source_task_name"] == "terminal-bench/interleaved-vigenere"
    assert passed.raw["agent_name"] == "oracle"
    assert passed.raw["verifier_isolation"] == "separate"
    assert errored.raw["error"] == (
        "AgentTimeoutError: Agent execution timed out after 28800.0 seconds"
    )


def test_bench_counts_the_errored_trial_by_its_own_rule():
    """The runner's summary left the errored trial out of its mean; BenchFlow
    counts it as not solved and lists it under the errors."""
    metrics = collect_metrics(str(JOB))
    rows = {t.task_name: t for t in metrics.tasks}
    assert len(metrics.tasks) == 2
    assert sum(1 for t in metrics.tasks if t.reward == 1.0) == 1
    assert any(
        t.error and t.error.startswith("AgentTimeoutError") for t in metrics.tasks
    )
    assert set(rows) == {"interleaved-vigenere"}
    summary = json.loads((JOB / "result.json").read_text())
    assert summary["stats"]["evals"]["oracle__adhoc"]["metrics"] == [{"mean": 1.0}]
