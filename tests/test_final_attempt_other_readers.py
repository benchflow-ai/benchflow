"""The other readers of a job take the attempt the run kept, like load_job.

Guards the routing added on branch ``audit-final-attempt-2-selector``
after the independent review of PR #138 (2026-11-13): the training-signal
export, the embodied reducer, ``bench eval lift``, the resume's solver
checkpoints, the pending-review pick and the rollout stream each still picked
"a scored attempt, then the newest" (or guessed from the retry policy), and
the job-level export rows did not say what the run did with their attempt.

The case in every test: attempt A went idle with reward 1 and the run retried
it; attempt B failed at sandbox start and is the attempt the run kept.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from benchflow._utils.attempt_decision import FINAL, RETRIED, record_decision
from benchflow.evaluation import Evaluation, EvaluationConfig
from tests.trajectories.test_training_signal import _rollout

SANDBOX = "sandbox startup failed: docker compose up exited 1"


def _retried_then_kept(job: Path) -> tuple[Path, Path]:
    """A (retried, reward 1, older) and B (kept, unscored, newer) of one task."""
    a = _rollout(job, "t1__aaaa0001", reward=1.0, task="t1", error="idle timeout")
    b = _rollout(job, "t1__bbbb0002", reward=None, task="t1", error=SANDBOX)
    (job / "evaluation.json").write_text(json.dumps({"schema_version": 1}))
    os.utime(a / "result.json", (1_800_000_000, 1_800_000_000))
    os.utime(b / "result.json", (1_800_000_060, 1_800_000_060))
    record_decision(a, RETRIED, task="t1", attempt=1)
    record_decision(b, FINAL, task="t1", attempt=2)
    return a, b


def _result(folder: Path) -> dict:
    return json.loads((folder / "result.json").read_text())


def test_the_training_signal_replaces_the_retried_attempt(tmp_path):
    from benchflow.trajectories.training_signal import replaced_attempts

    a, b = _retried_then_kept(tmp_path / "job")

    assert replaced_attempts([(a, _result(a)), (b, _result(b))]) == {a}


def test_a_lone_retried_attempt_is_no_trial_result_for_training(tmp_path):
    from benchflow.trajectories.training_signal import replaced_attempts

    a, _ = _retried_then_kept(tmp_path / "job")

    assert replaced_attempts([(a, _result(a))]) == {a}


def test_the_embodied_reducer_reads_the_kept_attempt(tmp_path):
    from benchflow.embodied.rollouts import _latest_results

    _, b = _retried_then_kept(tmp_path / "job")

    assert [path for path, _ in _latest_results(tmp_path / "job")] == [
        b / "result.json"
    ]


def test_lift_never_selects_the_retried_attempt(tmp_path):
    from benchflow.eval_lift import _load_rollouts

    _, b = _retried_then_kept(tmp_path / "job")

    assert [r.rollout_dir for r in _load_rollouts(tmp_path / "job")] == [b]


def test_a_resume_does_not_reuse_a_retried_attempts_solver_checkpoint(tmp_path):
    job = tmp_path / "jobs" / "job"
    (tmp_path / "tasks" / "t1").mkdir(parents=True)
    for name, decision in (("t1__aaaa0001", RETRIED), ("t2__cccc0003", None)):
        folder = job / name
        folder.mkdir(parents=True)
        task = name.split("__")[0]
        (folder / "solver-complete.json").write_text(
            json.dumps({"task_name": task, "rewards": {"reward": 1.0}})
        )
        if decision:
            record_decision(folder, decision, task=task, attempt=1)
    evaluation = Evaluation(
        tasks_dir=tmp_path / "tasks",
        jobs_dir=tmp_path / "jobs",
        job_name="job",
        config=EvaluationConfig(agent="oracle"),
    )

    # t2's checkpoint has no record and is reused, as in an older job.
    assert set(evaluation._get_completed_tasks()) == {"t2"}


def test_a_pending_review_resumes_on_the_kept_attempt(tmp_path, monkeypatch):
    import benchflow.review.resume as review_resume

    job = tmp_path / "job"
    folders = {}
    for name, decision, mtime in (
        ("t1__aaaa0001", RETRIED, 1_800_000_060),  # the newer snapshot
        ("t1__bbbb0002", FINAL, 1_800_000_000),
    ):
        folder = folders[decision] = job / name
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps({"review": {}}))
        (folder / "solver.json").write_text(json.dumps({"task_name": "t1"}))
        os.utime(folder / "solver.json", (mtime, mtime))
        record_decision(folder, decision, task="t1", attempt=1)
    resumed = []

    async def resume_review(path, **kwargs):
        resumed.append(path)

    monkeypatch.setattr(review_resume, "resume_review", resume_review)

    asyncio.run(
        review_resume.resume_pending_reviews(
            job,
            tasks_root=tmp_path,
            reviewer=None,
            task_names={"t1"},
            budget_guard=None,
        )
    )

    assert resumed == [folders[FINAL]]


def test_the_stream_reads_what_the_run_did_not_what_its_policy_would(tmp_path):
    from benchflow.trajectories.rollout_stream import _Scanner

    a, b = _retried_then_kept(tmp_path / "job")

    records = {
        Path(r.rollout_path).name: r
        for r in _Scanner(tmp_path / "job", None, lambda message: None).scan()
    }

    # The policy would retry B's sandbox failure; the run kept it (its last try).
    assert records[a.name]._retryable is True
    assert records[b.name]._retryable is False


def test_job_level_export_rows_say_what_the_run_did_with_their_attempt(tmp_path):
    from benchflow.trajectories._export_common import aggregate_rollout_jsonl
    from benchflow.trajectories.results import write_job_results_jsonl

    job = tmp_path / "job"
    a, b = _retried_then_kept(job)
    c = _rollout(job, "t2__cccc0003", reward=1.0, task="t2")  # no record
    for folder, task in ((a, "t1"), (b, "t1"), (c, "t2")):
        row = json.dumps({"task_name": task, "rollout": folder.name}) + "\n"
        (folder / "results.jsonl").write_text(row)
        (folder / "trainer").mkdir()
        (folder / "trainer" / "verifiers.jsonl").write_text(row)

    def decisions(path: Path) -> dict[str, str | None]:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return {row["rollout"]: row.get("attempt_decision") for row in rows}

    expected = {a.name: RETRIED, b.name: FINAL, c.name: None}
    assert decisions(write_job_results_jsonl(job)) == expected
    out = aggregate_rollout_jsonl(
        job, rollout_relpath="trainer/verifiers.jsonl", out_filename="verifiers.jsonl"
    )
    assert decisions(out) == expected
    assert "attempt_decision" not in out.read_text().splitlines()[2]


def _resumed(job: Path) -> tuple[Path, Path, Path]:
    """A (retried), B (kept by the run, errored), C (kept by a resume that
    ran the task again): three attempts of one task, two of them marked
    final by their own folder's record."""
    a, b = _retried_then_kept(job)
    c = _rollout(job, "t1__cccc0003", reward=1.0, task="t1")
    os.utime(c / "result.json", (1_800_000_120, 1_800_000_120))
    record_decision(c, FINAL, task="t1", attempt=1)
    return a, b, c


def test_export_rows_of_a_resumed_task_name_one_final_attempt(tmp_path):
    """Re-review of PR #138 (2026-11-13): each row carried its own folder's
    record, so after a resume one task's rows read retried, final, final."""
    from benchflow.trajectories._export_common import aggregate_rollout_jsonl
    from benchflow.trajectories.results import write_job_results_jsonl

    job = tmp_path / "job"
    a, b, c = _resumed(job)
    for folder in (a, b, c):
        row = json.dumps({"task_name": "t1", "rollout": folder.name}) + "\n"
        (folder / "results.jsonl").write_text(row)
        (folder / "trainer").mkdir()
        (folder / "trainer" / "verifiers.jsonl").write_text(row)

    def decisions(path: Path) -> dict[str, str | None]:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return {row["rollout"]: row.get("attempt_decision") for row in rows}

    expected = {a.name: RETRIED, b.name: "superseded", c.name: FINAL}
    assert decisions(write_job_results_jsonl(job)) == expected
    out = aggregate_rollout_jsonl(
        job, rollout_relpath="trainer/verifiers.jsonl", out_filename="verifiers.jsonl"
    )
    assert decisions(out) == expected


def test_the_sft_export_converts_only_the_kept_attempt(tmp_path):
    """Re-review of PR #138: without group advantages every attempt was
    converted, so a retried attempt with reward 1 passed --min-reward."""
    from benchflow.trajectories.export_prime_sft import _rollout_dirs_for_groups

    a, b = _retried_then_kept(tmp_path / "job")

    assert _rollout_dirs_for_groups(tmp_path / "job", None, None) == [b]
    # With group advantages every attempt comes; its group marks A retried.
    assert set(_rollout_dirs_for_groups(tmp_path / "job", None, "grpo")) == {a, b}


def test_the_embodied_export_writes_no_episode_for_a_retried_attempt(
    tmp_path, monkeypatch
):
    """Re-review of PR #138: export_job wrote every attempt, untagged."""
    import benchflow.embodied.export as embodied_export

    a, b = _retried_then_kept(tmp_path / "job")
    for folder in (a, b):
        steps = folder / embodied_export.STEPS_RELPATH
        steps.parent.mkdir(parents=True, exist_ok=True)
        steps.write_text(json.dumps({"trial": folder.name}) + "\n")
    monkeypatch.setattr(
        embodied_export,
        "write_rollout_embodied",
        lambda trial, **kwargs: {"video_index": {}},
    )

    done = embodied_export.export_job(tmp_path / "job", tmp_path / "out")

    episodes = (tmp_path / "out" / "episodes.jsonl").read_text().splitlines()
    assert [json.loads(line)["trial"] for line in episodes] == [b.name]
    assert (done["episodes"], done["steps"]) == (1, 1)
