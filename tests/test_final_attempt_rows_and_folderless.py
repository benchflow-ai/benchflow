"""A job's results.jsonl, and a kept attempt that left no folder, read as the
job's folder does.

Guards the fixes after the re-review of PR #138 (2026-11-13) on branch
``audit-final-attempt-2-selector``:

- N1: ``bf.load_job`` given a job's own ``results.jsonl`` ignored each row's
  ``attempt_decision`` and fell back to "a scored attempt, then the newest".
  An attempt that went idle with reward 1 and was retried, then a retry that
  failed at sandbox start and was kept: ``bench results jobs/job`` said 0/1
  errored, ``bench results jobs/job/results.jsonl`` said 1/1 passed.
- N2: the export labels were worked out among the attempts that had an
  export file only. A kept attempt that failed before its rollout folder
  existed has none, so after a resume the earlier attempt stayed ``final``.
- N3: regrade and the training-signal and embodied exports group attempts by
  task, agent and model; such an attempt's result.json had no agent or model
  (they were only in its config.json), so it was a trial of its own.
- The lead's ruling on the fixes: such an attempt also had no row in the
  job's results.jsonl, so the file said its task had no result (or only the
  attempt the run retried) where the job's folder said errored.
- The re-review of the fixes: a file joining two jobs' rows read as one
  trial per task (the later job's), and a script agent's attempt with no
  folder was named by its host path, where a rollout names its file.

Every job here is made by ``Evaluation.run()`` with a fake trial runner that
writes what a rollout writes (its folder's config.json, result.json and
results.jsonl), or, for a trial that fails before its folder exists, returns
what the rollout returns then and leaves the folder to the job.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import benchflow as bf
from benchflow._utils.attempt_decision import FINAL, RETRIED, SUPERSEDED, read_decision
from benchflow.agents.scripts import script_label
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.models import RolloutResult
from benchflow.trajectories.results import (
    write_job_results_jsonl,
    write_rollout_results_jsonl,
)
from tests.test_dogfood_rc_results import _invoke, _write_task

AGENT = "claude-agent-acp"
REPORTED = "@agentclientprotocol/claude-agent-acp"  # what the agent calls itself
MODEL = "claude-haiku-4-5"
IDLE = "Agent idle for 600s (no output); killed"
SANDBOX = "sandbox startup failed: docker compose up exited 1"
CRASHED = "RuntimeError: the agent crashed"
NO_FOLDER = "Setup failed before trial directory was created"


def _folder(job: Evaluation, task: str, name: str, rewards, error) -> RolloutResult:
    """An attempt with its own folder, as a rollout leaves it: config.json
    names the agent as the run was given it (a script by its host path),
    result.json and the row as the rollout records it (a script by its
    file, which is also the name a script agent reports)."""
    trial = job.job_dir / name
    trial.mkdir(parents=True)
    spec, model = job._config.agent, job._config.model
    agent = script_label(spec)
    agent_name = REPORTED if spec == AGENT else agent
    (trial / "config.json").write_text(
        json.dumps(
            {
                "agent": spec,
                "model": model,
                "task_path": task,
                "benchflow_version": bf.__version__,
            }
        )
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": task,
                "rollout_name": name,
                "agent": agent,
                "agent_name": agent_name,
                "model": model,
                "rewards": rewards,
                "error": error,
            }
        )
    )
    write_rollout_results_jsonl(
        trial,
        task_name=task,
        rollout_name=name,
        agent=agent,
        agent_name=agent_name,
        model=model,
        n_tool_calls=0,
        prompts=[],
        trajectory=[],
        partial_trajectory=False,
        rewards=rewards,
        error=error,
        verifier_error=None,
    )
    trainer = trial / "trainer"
    trainer.mkdir()
    (trainer / "verifiers.jsonl").write_text(json.dumps({"rollout": name}) + "\n")
    return RolloutResult(
        task_name=task,
        rollout_name=name,
        rewards=rewards,
        error=error,
        agent=agent,
        agent_name=agent_name,
        model=model,
        rollout_dir=trial,
    )


def _run(
    tmp_path: Path,
    attempts: dict[str, list[tuple]],
    *,
    retries: int,
    job_name: str = "job",
    **config,
) -> Path:
    """One ``Evaluation.run()`` of the job ``jobs/<job_name>`` (a second one
    resumes it). *attempts* lists, per task, what each attempt the run starts
    does: ``(name, rewards, error)`` for one with a folder, ``None`` (or its
    error text) for one that fails before its folder exists. *config* adds to
    or replaces the run's settings."""
    tasks = tmp_path / "tasks"
    for task in attempts:
        _write_task(tasks / task)
    job = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        job_name=job_name,
        preflight=False,
        config=EvaluationConfig(
            **{
                "agent": AGENT,
                "model": MODEL,
                "concurrency": 1,
                "retry": RetryConfig(max_retries=retries, min_wait_sec=0),
                **config,
            }
        ),
    )
    queues = {task: list(planned) for task, planned in attempts.items()}

    async def run_trial(task_path, _cfg):
        attempt = queues[task_path.name].pop(0)
        if attempt is None or isinstance(attempt, str):
            # What the rollout returns when it made no folder.
            return RolloutResult(task_name=task_path.name, error=attempt or NO_FOLDER)
        return _folder(job, task_path.name, *attempt)

    async def no_sweep():
        return None

    job._run_single_task = AsyncMock(side_effect=run_trial)
    job._sweep_docker = no_sweep
    asyncio.run(job.run())
    assert not any(queues.values()), queues
    return job.job_dir


def _placeholder(job_dir: Path) -> Path:
    [folder] = [
        p
        for p in job_dir.glob("t1__*")
        if json.loads((p / "result.json").read_text()).get("no_rollout_folder")
    ]
    return folder


def _written_before_the_fix(folder: Path) -> None:
    """The placeholder's result.json as the job wrote it before it recorded
    the agent and model there (they were in its config.json only)."""
    result = json.loads((folder / "result.json").read_text())
    result.pop("agent", None)
    result.pop("model", None)
    (folder / "result.json").write_text(json.dumps(result))


def _labels(path: Path) -> dict[str | None, str | None]:
    """Each row's attempt_decision, by the rollout it names (None for a
    trainer row of a trial with no folder, which names none)."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {
        (row.get("info") or {}).get("rollout_name") or row.get("rollout"): row.get(
            "attempt_decision"
        )
        for row in rows
    }


def _counts(job: bf.Job) -> tuple[int, int, int]:
    d = job.denominators()
    return d.attempted, d.passed, d.execution_errors


def _line(output: str) -> str:
    [line] = [line for line in output.splitlines() if "attempted passed" in line]
    return line


def _both_ways(job_dir: Path) -> tuple[bf.Job, bf.Job, str, str]:
    by_folder = _line(_invoke("results", str(job_dir)).output)
    by_rows = _line(_invoke("results", str(job_dir / "results.jsonl")).output)
    return (
        bf.load_job(job_dir),
        bf.load_job(job_dir / "results.jsonl"),
        by_folder,
        by_rows,
    )


# --- N1: a job's results.jsonl read on its own ------------------------------


def test_a_jobs_results_jsonl_counts_the_attempt_the_run_kept(tmp_path):
    """Review of PR #138, 2026-11-13 (N1), the reviewer's case: attempt 1
    goes idle with reward 1 and is retried, attempt 2 fails at sandbox start
    and is kept. The job's folder and its results.jsonl give one count."""
    job_dir = _run(
        tmp_path,
        {
            "t0": [("t0__cccc0003", {"reward": 1.0}, None)],
            "t1": [
                ("t1__aaaa0001", {"reward": 1.0}, IDLE),
                ("t1__bbbb0002", None, SANDBOX),
            ],
        },
        retries=1,
    )
    assert _labels(job_dir / "results.jsonl") == {
        "t0__cccc0003": FINAL,
        "t1__aaaa0001": RETRIED,
        "t1__bbbb0002": FINAL,
    }

    folder = bf.load_job(job_dir)
    rows = bf.load_job(job_dir / "results.jsonl")

    names = ["t0__cccc0003", "t1__bbbb0002"]
    assert [t.result.rollout_name for t in folder.trials] == names
    assert [t.result.rollout_name for t in rows.trials] == names
    assert _counts(folder) == _counts(rows) == (2, 1, 1)
    assert rows.unfinished() == folder.unfinished() == ([], [])
    by_folder = _line(_invoke("results", str(job_dir)).output)
    by_rows = _line(_invoke("results", str(job_dir / "results.jsonl")).output)
    assert "1/2 attempted passed" in by_folder
    assert by_rows == by_folder


def test_a_superseded_row_is_never_the_trial(tmp_path):
    """Review of PR #138, 2026-11-13 (N1): a resume ran the task again, so
    two of its attempts are recorded ``final``; the export marks the earlier
    one ``superseded``, and reading the file keeps the resume's attempt even
    where the earlier rule (the newest row) would take the other."""
    _run(tmp_path, {"t1": [("t1__zzzz0001", None, CRASHED)]}, retries=0)
    job_dir = _run(tmp_path, {"t1": [("t1__aaaa0002", None, SANDBOX)]}, retries=0)
    assert _labels(job_dir / "results.jsonl") == {
        "t1__aaaa0002": FINAL,
        "t1__zzzz0001": SUPERSEDED,
    }

    folder = bf.load_job(job_dir)
    rows = bf.load_job(job_dir / "results.jsonl")

    assert [t.result.rollout_name for t in folder.trials] == ["t1__aaaa0002"]
    assert [t.result.rollout_name for t in rows.trials] == ["t1__aaaa0002"]
    assert rows.trials[0].result.error == SANDBOX


def test_a_copied_out_results_file_keeps_the_kept_attempt(tmp_path):
    """Review of PR #138, 2026-11-13 (N1): a results.jsonl copied away from
    its job folder has no job records beside it; its labelled rows are still
    attempts of one trial, not independent samples."""
    job_dir = _run(
        tmp_path,
        {
            "t1": [
                ("t1__aaaa0001", {"reward": 1.0}, IDLE),
                ("t1__bbbb0002", None, SANDBOX),
            ]
        },
        retries=1,
    )
    copied = tmp_path / "elsewhere" / "results.jsonl"
    copied.parent.mkdir()
    copied.write_text((job_dir / "results.jsonl").read_text())

    rows = bf.load_job(copied)

    assert [t.result.rollout_name for t in rows.trials] == ["t1__bbbb0002"]
    assert _counts(rows) == (1, 0, 1)


def _joined(tmp_path: Path, *job_dirs: Path) -> Path:
    """The jobs' results.jsonl files joined into one, as ``cat`` joins them,
    in a folder that is no job's."""
    joined = tmp_path / "joined" / "results.jsonl"
    joined.parent.mkdir()
    joined.write_text("".join((d / "results.jsonl").read_text() for d in job_dirs))
    return joined


def test_a_file_joining_two_jobs_keeps_each_jobs_trial(tmp_path):
    """Re-review of the fixes for PR #138, 2026-11-13: rows that carry a
    decision were taken as attempts of one trial per file, so a file joining
    two jobs' rows kept one trial per task: job one's pass was dropped and
    the file read 0/1 where it holds 1 of 2. Each job's own retried attempt
    is still no trial."""
    one = _run(
        tmp_path,
        {"t1": [("t1__aaaa0001", {"reward": 1.0}, None)]},
        retries=0,
        job_name="one",
    )
    two = _run(
        tmp_path,
        {
            "t1": [
                ("t1__bbbb0002", {"reward": 1.0}, IDLE),
                ("t1__cccc0003", None, SANDBOX),
            ]
        },
        retries=1,
        job_name="two",
    )
    joined = _joined(tmp_path, one, two)

    rows = bf.load_job(joined)

    assert [t.result.rollout_name for t in rows.trials] == [
        "t1__aaaa0001",
        "t1__cccc0003",
    ]
    assert _counts(rows) == (2, 1, 1)
    assert sum(len(t.attempts) for t in rows.trials) == 3
    assert rows.unfinished() == ([], [])
    assert rows.to_json_dict()["interrupted"] == []
    assert "1/2 attempted passed" in _line(_invoke("results", str(joined)).output)

    # A row that names no rollout folder is a trial of its own: rows of
    # unknown jobs are never joined, and a retried one is still no trial.
    lines = [json.loads(line) for line in joined.read_text().splitlines()]
    for line in lines:
        del line["info"]["rollout_dir"]
    joined.write_text("".join(json.dumps(line) + "\n" for line in lines))
    rows = bf.load_job(joined)
    assert [t.result.rollout_name for t in rows.trials] == [
        "t1__aaaa0001",
        "t1__cccc0003",
    ]


def test_a_kept_attempt_with_no_folder_is_errored_in_the_file_too(tmp_path):
    """The lead's ruling on the review of PR #138, 2026-11-13: attempt 1 goes
    idle with reward 1 and is retried; attempt 2 fails before its folder
    exists and is kept. The job's results.jsonl had no row for it, so read
    on its own the file said 1/1 passed with t1 interrupted; the job's folder
    said 1/2, t1 errored. (The re-review of the fixes found the earlier test
    here accepted that.)"""
    job_dir = _run(
        tmp_path,
        {
            "t0": [("t0__cccc0003", {"reward": 1.0}, None)],
            "t1": [("t1__aaaa0001", {"reward": 1.0}, IDLE), None],
        },
        retries=1,
    )
    kept = _placeholder(job_dir)
    assert _labels(job_dir / "results.jsonl") == {
        "t0__cccc0003": FINAL,
        "t1__aaaa0001": RETRIED,
        kept.name: FINAL,
    }

    folder, rows, by_folder, by_rows = _both_ways(job_dir)

    names = ["t0__cccc0003", kept.name]
    assert [t.result.rollout_name for t in folder.trials] == names
    assert [t.result.rollout_name for t in rows.trials] == names
    assert [t.execution for t in rows.trials] == ["completed", "errored"]
    assert _counts(folder) == _counts(rows) == (2, 1, 1)
    assert folder.unfinished() == rows.unfinished() == ([], [])
    assert rows.retried == []
    assert rows.to_json_dict()["interrupted"] == []
    assert "1/2 attempted passed" in by_folder and "1 errored" in by_folder
    assert by_rows == by_folder


def test_a_file_without_the_kept_row_leaves_the_other_tasks_finished(tmp_path):
    """Review of PR #138, 2026-11-13 (N1): a file that lacks the row of t1's
    kept attempt (its write failed) has only the attempt the run retried for
    t1. Reading the file never counts that row; t0 stays finished (every row
    shares the file's path, so the retried row made every task unfinished),
    and the export lists t1 as ``unfinished()`` does."""
    job_dir = _run(
        tmp_path,
        {
            "t0": [("t0__cccc0003", {"reward": 1.0}, None)],
            "t1": [("t1__aaaa0001", {"reward": 1.0}, IDLE), None],
        },
        retries=1,
    )
    kept = _placeholder(job_dir)
    for row_file in (kept / "results.jsonl", kept / "trainer" / "verifiers.jsonl"):
        row_file.unlink(missing_ok=True)
    write_job_results_jsonl(job_dir)
    assert _labels(job_dir / "results.jsonl") == {
        "t0__cccc0003": FINAL,
        "t1__aaaa0001": RETRIED,
    }

    rows = bf.load_job(job_dir / "results.jsonl")

    assert [t.result.rollout_name for t in rows.trials] == ["t0__cccc0003"]
    assert [t.result.rollout_name for t in rows.retried] == ["t1__aaaa0001"]
    assert rows.unfinished() == (["t1"], [])
    exported = rows.to_json_dict()["interrupted"]
    assert [(e["task_name"], e["path"]) for e in exported] == [
        ("t1", str(job_dir / "results.jsonl"))
    ]
    assert [t.task_name for t in bf.load_job(job_dir).trials] == ["t0", "t1"]


# --- N2: the export labels, with a kept attempt that has no row -------------


def test_a_kept_attempt_with_no_folder_supersedes_the_earlier_final_row(tmp_path):
    """Review of PR #138, 2026-11-13 (N2): the first run kept an attempt that
    errored; the resume's attempt failed before its folder existed and was
    kept. The earlier attempt's rows said ``final``."""
    _run(tmp_path, {"t1": [("t1__aaaa0001", None, CRASHED)]}, retries=0)
    job_dir = _run(tmp_path, {"t1": [None]}, retries=0)
    kept = _placeholder(job_dir)
    assert read_decision(kept)["decision"] == FINAL
    assert read_decision(kept)["sequence"] == 2

    assert _labels(job_dir / "results.jsonl") == {
        "t1__aaaa0001": SUPERSEDED,
        kept.name: FINAL,
    }
    assert _labels(job_dir / "verifiers.jsonl") == {
        "t1__aaaa0001": SUPERSEDED,
        None: FINAL,
    }
    folder, rows, by_folder, by_rows = _both_ways(job_dir)

    assert [t.path for t in folder.trials] == [kept]
    assert [t.result.rollout_name for t in rows.trials] == [kept.name]
    assert [t.execution for t in rows.trials] == ["errored"]
    assert _counts(folder) == _counts(rows) == (1, 0, 1)
    assert folder.unfinished() == rows.unfinished() == ([], [])
    assert "0/1 attempted passed" in by_folder and "1 errored" in by_folder
    assert by_rows == by_folder


# --- a kept attempt that left no folder has its row --------------------------


def test_a_single_attempt_that_left_no_folder_is_errored_in_the_file_too(tmp_path):
    """The lead's ruling on the review of PR #138, 2026-11-13: the one attempt
    of the task failed before its folder existed. The job wrote no
    results.jsonl at all."""
    job_dir = _run(tmp_path, {"t1": [None]}, retries=0)
    kept = _placeholder(job_dir)

    folder, rows, by_folder, by_rows = _both_ways(job_dir)

    assert [t.result.rollout_name for t in folder.trials] == [kept.name]
    assert [t.result.rollout_name for t in rows.trials] == [kept.name]
    assert [t.execution for t in rows.trials] == ["errored"]
    assert _counts(folder) == _counts(rows) == (1, 0, 1)
    assert folder.unfinished() == rows.unfinished() == ([], [])
    assert "0/1 attempted passed" in by_folder and "1 errored" in by_folder
    assert by_rows == by_folder


def _keys(value: object, prefix: str = "") -> set[str]:
    """Every key path of a row; a timing block's phases are the trial's."""
    if not isinstance(value, dict) or prefix == "timing.":
        return set()
    paths = set()
    for key, item in value.items():
        paths.add(prefix + key)
        paths |= _keys(item, f"{prefix}{key}.")
    return paths


def test_the_row_of_a_trial_with_no_folder_is_an_errored_trials_row(tmp_path):
    """The lead's ruling on the review of PR #138, 2026-11-13: the row has the
    fields an errored rollout's row has and no other, its error masked as
    that row's is, and the agent and model its result.json records. The
    comparison is a row the rollout's own writer makes for a trial that
    errored before its agent ran (a sandbox that did not start)."""
    from datetime import datetime

    from benchflow.rollout._results import _build_rollout_result
    from benchflow.trajectories.types import redact_trajectory_text

    secret = "FAKEtoken0123456789abcd"  # made up
    error = f"Unexpected: upstream refused Authorization: Bearer {secret}"
    job_dir = _run(tmp_path, {"t1": [error]}, retries=0)
    kept = _placeholder(job_dir)
    errored = tmp_path / "rollout" / "t1__dddd0004"
    errored.mkdir(parents=True)
    _build_rollout_result(
        errored,
        task_name="t1",
        rollout_name=errored.name,
        agent=AGENT,
        agent_name="",
        model=MODEL,
        n_tool_calls=0,
        prompts=["Fix the bug."],
        error=error,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards=None,
        started_at=datetime.now(),
        timing={},
    )

    [row] = [
        json.loads(line) for line in (kept / "results.jsonl").read_text().splitlines()
    ]
    [job_row] = [
        json.loads(line)
        for line in (job_dir / "results.jsonl").read_text().splitlines()
    ]
    reference = json.loads((errored / "results.jsonl").read_text())
    assert _keys(row) == _keys(reference)
    assert row["error"] == reference["error"]
    assert row["error"]["error_chain_str"] == redact_trajectory_text(error)
    assert secret not in (job_dir / "results.jsonl").read_text()
    result = json.loads((kept / "result.json").read_text())
    assert (row["info"]["agent"], row["info"]["model"]) == (
        result["agent"],
        result["model"],
    )
    assert (row["reward"], row["is_completed"]) == (None, False)
    assert job_row == {**row, "attempt_decision": FINAL}
    trainer = json.loads((kept / "trainer" / "verifiers.jsonl").read_text())
    reference = json.loads((errored / "trainer" / "verifiers.jsonl").read_text())
    assert _keys(trainer) == _keys(reference)


# --- N3: a trial that left no folder groups with its task's attempts ---------


def test_a_trial_that_left_no_folder_records_its_agent_and_model(tmp_path):
    """Review of PR #138, 2026-11-13 (N3): its result.json had neither."""
    job_dir = _run(tmp_path, {"t1": [None]}, retries=0)
    folder = _placeholder(job_dir)

    result = json.loads((folder / "result.json").read_text())
    config = json.loads((folder / "config.json").read_text())
    assert (result["agent"], result["model"]) == (AGENT, MODEL)
    assert (config["agent"], config["model"]) == (AGENT, MODEL)


@pytest.mark.parametrize("older_job", [False, True], ids=["now", "older-job"])
def test_regrade_lists_the_resumes_attempt_alone(tmp_path, older_job):
    """Review of PR #138, 2026-11-13 (N3): the first run's attempt failed
    before its folder existed; the resume ran the task again and it passed.
    Regrade listed both as trials: the placeholder's result.json had no agent
    (in an older job) and no ``agent_name``, which regrade grouped by."""
    from benchflow.eval_regrade import find_trials

    _run(tmp_path, {"t1": [None]}, retries=0)
    job_dir = _run(
        tmp_path, {"t1": [("t1__bbbb0002", {"reward": 1.0}, None)]}, retries=0
    )
    if older_job:
        _written_before_the_fix(_placeholder(job_dir))

    assert [t.name for t in find_trials(job_dir)] == ["t1__bbbb0002"]
    assert [t.path.name for t in bf.load_job(job_dir).trials] == ["t1__bbbb0002"]


@pytest.mark.parametrize("older_job", [False, True], ids=["now", "older-job"])
def test_the_exports_leave_out_the_attempt_a_placeholder_replaced(
    tmp_path, monkeypatch, older_job
):
    """Review of PR #138, 2026-11-13 (N3): the first run kept attempt A
    (errored); the resume's attempt failed before its folder existed and is
    the trial. The training signal kept A and the embodied export wrote it."""
    import benchflow.embodied.export as embodied_export
    from benchflow.eval_regrade import _attempt_key
    from benchflow.trajectories.training_signal import replaced_attempts

    _run(tmp_path, {"t1": [("t1__aaaa0001", None, CRASHED)]}, retries=0)
    job_dir = _run(tmp_path, {"t1": [None]}, retries=0)
    a, kept = job_dir / "t1__aaaa0001", _placeholder(job_dir)
    if older_job:
        _written_before_the_fix(kept)
    results = [(d, json.loads((d / "result.json").read_text())) for d in (a, kept)]

    assert replaced_attempts(results) == {a}
    # Regrade puts both in one trial, whose kept attempt is the placeholder.
    assert _attempt_key(a) == _attempt_key(kept)

    steps = a / embodied_export.STEPS_RELPATH
    steps.write_text(json.dumps({"trial": a.name}) + "\n")
    monkeypatch.setattr(
        embodied_export,
        "write_rollout_embodied",
        lambda trial, **kwargs: (
            {"video_index": {}}
            if (trial / embodied_export.STEPS_RELPATH).is_file()
            else None
        ),
    )
    done = embodied_export.export_job(job_dir, tmp_path / "out")

    assert (tmp_path / "out" / "episodes.jsonl").read_text() == ""
    assert (done["episodes"], done["steps"]) == (0, 0)


@pytest.mark.parametrize("older_job", [False, True], ids=["now", "older-job"])
def test_every_reader_takes_a_script_agents_attempt_with_no_folder(
    tmp_path, monkeypatch, older_job
):
    """Re-review of the fixes for PR #138, 2026-11-13: a script agent's trial
    that left no folder was named ``script:/<host path>/solve.sh``, where a
    rollout records ``script:solve.sh``, so it was no attempt of its task's
    other trial. Run one's attempt crashes; the resume's leaves no folder
    and is the trial."""
    import benchflow.embodied.export as embodied_export
    from benchflow.eval_regrade import _attempt_key
    from benchflow.trajectories.training_signal import replaced_attempts

    script = tmp_path / "bin" / "solve.sh"
    script.parent.mkdir()
    script.write_text("#!/bin/bash\nexit 0\n")
    agent = {"agent": f"script:{script}", "model": None, "sandbox_user": "agent"}
    _run(tmp_path, {"t1": [("t1__aaaa0001", None, CRASHED)]}, retries=0, **agent)
    job_dir = _run(tmp_path, {"t1": [None]}, retries=0, **agent)
    a, kept = job_dir / "t1__aaaa0001", _placeholder(job_dir)
    for trial in (a, kept):
        result = json.loads((trial / "result.json").read_text())
        [row] = [json.loads(line) for line in (trial / "results.jsonl").open()]
        assert result["agent"] == row["info"]["agent"] == "script:solve.sh"
    assert json.loads((kept / "config.json").read_text())["agent"] == f"script:{script}"
    if older_job:
        _written_before_the_fix(kept)

    folder, rows, by_folder, by_rows = _both_ways(job_dir)
    assert [t.path for t in folder.trials] == [kept]
    assert [t.result.rollout_name for t in rows.trials] == [kept.name]
    assert [t.agent for t in folder.trials] == ["script:solve.sh"]
    assert _counts(folder) == _counts(rows) == (1, 0, 1)
    assert by_rows == by_folder
    results = [(d, json.loads((d / "result.json").read_text())) for d in (a, kept)]
    assert replaced_attempts(results) == {a}
    assert _attempt_key(a) == _attempt_key(kept)
    (a / embodied_export.STEPS_RELPATH).write_text(json.dumps({"trial": a.name}) + "\n")
    monkeypatch.setattr(
        embodied_export,
        "write_rollout_embodied",
        lambda trial, **kwargs: (
            {"video_index": {}}
            if (trial / embodied_export.STEPS_RELPATH).is_file()
            else None
        ),
    )
    assert embodied_export.export_job(job_dir, tmp_path / "out")["episodes"] == 0
