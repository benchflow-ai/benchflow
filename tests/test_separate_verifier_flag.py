"""``--separate-verifier`` is one isolation setting with ``--verifier-sandbox separate``.

Guards the b127 finding: on 3a54f24a the retained flag moved the verifier
after the merged resolver had recorded the trial as shared, and job plan
validation never saw it, so a task the separate sandbox cannot run was not
refused first and ``result.json``/``summary.json`` counted the trial as
shared. ``effective_verifier_sandbox`` now folds the flag in at every reader.
"""

from __future__ import annotations

import pytest


def test_the_separate_verifier_flag_records_a_separate_trial(tmp_path, monkeypatch):
    """Guards the b127 finding (fails on 3a54f24a: recorded shared/task)."""
    from tests.test_sv_switch import _records, _rollout_config, _task, _trial

    task = _task(tmp_path, "shared-task")
    rollout, env, calls = _trial(
        _rollout_config(task, tmp_path / "jobs", separate_verifier=True), monkeypatch
    )
    assert calls == ["separate"]
    assert env.uploads == []
    assert _records(rollout) == ("separate", "separate", "run-flag")


def test_the_separate_verifier_flag_reaches_the_plan_check(tmp_path, monkeypatch):
    """A job with the flag refuses tasks the separate sandbox cannot run, first."""
    import asyncio

    from benchflow.evaluation import Evaluation, EvaluationConfig
    from tests.test_sv_switch import _no_sandbox, _refused_set

    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
    started = _no_sandbox(monkeypatch)
    config = EvaluationConfig(agent="oracle", environment="docker")
    config.separate_verifier = True  # as the CLI sets it, after construction
    job = Evaluation(
        tasks_dir=_refused_set(tmp_path),
        jobs_dir=tmp_path / "jobs",
        config=config,
        job_name="forced",
        preflight=False,
    )
    with pytest.raises(ValueError):
        asyncio.run(job.run())
    assert not (tmp_path / "jobs").exists()
    assert started == []
