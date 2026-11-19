"""Regrade lists the attempt the run kept when it left no folder alone, as
``bf.load_job`` counts it.

Guards the fix for the review of PR #138 (2026-11-13, N3) on this branch's
regrade (``audit-final-attempt-3-regrade-resume``), whose trial list
needs the attempt records.
"""

from __future__ import annotations

import pytest

import benchflow as bf
from benchflow.eval_regrade import find_trials
from tests.test_final_attempt_rows_and_folderless import (
    CRASHED,
    _placeholder,
    _run,
    _written_before_the_fix,
)


@pytest.mark.parametrize("older_job", [False, True], ids=["now", "older-job"])
@pytest.mark.parametrize("agent", ["acp", "script"])
def test_regrade_lists_a_kept_attempt_that_left_no_folder_alone(
    tmp_path, agent, older_job
):
    """Review of PR #138, 2026-11-13 (N3), for this branch's regrade: the
    first run kept attempt A (it errored); a resume ran the task again, and
    its attempt failed before its folder existed and is the trial. Regrade
    listed both: the attempt with no folder named its agent otherwise than
    A (no ``agent_name``; in an older job no ``agent`` either; a script
    agent by its host path), so A was a trial of its own."""
    config = {}
    if agent == "script":
        script = tmp_path / "bin" / "solve.sh"
        script.parent.mkdir()
        script.write_text("#!/bin/bash\nexit 0\n")
        config = {"agent": f"script:{script}", "model": None, "sandbox_user": "agent"}
    _run(tmp_path, {"t1": [("t1__aaaa0001", None, CRASHED)]}, retries=0, **config)
    job_dir = _run(tmp_path, {"t1": [None]}, retries=0, **config)
    kept = _placeholder(job_dir)
    if older_job:
        _written_before_the_fix(kept)

    assert [t.name for t in find_trials(job_dir)] == [kept.name]
    assert [t.path for t in bf.load_job(job_dir).trials] == [kept]
