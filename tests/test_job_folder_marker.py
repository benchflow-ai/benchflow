"""Only folders that carry BenchFlow's own marker are resumed.

A job folder is BenchFlow's when it has a marker BenchFlow writes as the job
starts or later: the job record (``evaluation.json``, or the ``summary.json``
of a BenchFlow 0.7 job) or a trial BenchFlow started: a config.json naming the BenchFlow that ran it
(``benchflow_version``/``harness_mode``, which ``bf.run(job_name=...)``
folders have instead of a record) or holding every key a 0.7 trial's has, or
a result.json that is a BenchFlow rollout result (``rollout_name``). Any
other folder that holds files is a folder BenchFlow did not make: an unnamed
run never picks it, a named run refuses it, and nothing in it moves. The
decision reads BenchFlow's markers only, never another program's file names
or contents.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from benchflow.evaluation import (
    JOB_LOCK,
    Evaluation,
    EvaluationConfig,
    ResumeMismatchError,
    _acquire_job_lock,
    _check_resume_mismatch,
    _is_benchflow_job,
    _is_external_job,
    _job_folders,
)

STAMP = "2026-10-19__01-17-57"


def _trial(job: Path, name: str, config: dict) -> Path:
    trial = job / name
    trial.mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps(config))
    (trial / "result.json").write_text(json.dumps({"task_name": name}))
    return trial


def _foreign_job(jobs: Path, name: str = STAMP) -> Path:
    """Another runner's job: same timestamp names, its own files."""
    job = jobs / name
    _trial(job, "hello__abc123", {"task": {"path": "hello"}, "trial_name": "x"})
    (job / "config.json").write_text(json.dumps({"job_name": name}))
    (job / "result.json").write_text(json.dumps({"stats": {}}))
    return job


def test_a_folder_with_files_but_no_benchflow_marker_is_foreign(tmp_path):
    job = _foreign_job(tmp_path / "jobs")
    assert not _is_benchflow_job(job)
    assert _is_external_job(job)
    assert _job_folders(tmp_path / "jobs") == []


@pytest.mark.parametrize(
    "lock",
    [{}, {"anything": 1}, {"framework": "other"}],
    ids=["empty", "unrelated", "named-framework"],
)
def test_no_lock_file_content_makes_a_folder_benchflows(tmp_path, lock):
    job = _foreign_job(tmp_path / "jobs")
    (job / "lock.json").write_text(json.dumps(lock))
    assert _is_external_job(job)


def _record(job: Path) -> dict:
    """The keys of the evaluation.json every 0.8 job writes as it starts."""
    return {"schema_version": 1, "tasks_dir": "/t", "job_name": job.name, "config": {}}


def test_benchflow_job_record_is_resumed(tmp_path):
    jobs = tmp_path / "jobs"
    job = _foreign_job(jobs)
    (job / "evaluation.json").write_text(json.dumps(_record(job)))
    assert _is_benchflow_job(job)
    assert not _is_external_job(job)
    assert _job_folders(jobs) == [job]


@pytest.mark.parametrize("body", ["{}", '{"tool": "other"}', "[]", "{"])
def test_an_evaluation_json_benchflow_did_not_write_is_no_marker(tmp_path, body):
    job = _foreign_job(tmp_path / "jobs")
    (job / "evaluation.json").write_text(body)
    assert _is_external_job(job)


@pytest.mark.parametrize("body", ['{"tool": "other"}', "{}", "{"])
def test_a_summary_json_alone_is_no_marker(tmp_path, body):
    """Many programs write a summary.json (review REL r4 H2)."""
    job = tmp_path / "jobs" / "foreign2"
    job.mkdir(parents=True)
    (job / "summary.json").write_text(body)
    assert not _is_benchflow_job(job)
    assert _is_external_job(job)
    assert _job_folders(tmp_path / "jobs") == []


def _snapshot(folder: Path) -> dict[str, tuple]:
    """Every entry under (and including) ``folder``: its bytes and its times."""
    entries = {".": folder}
    entries.update({str(p.relative_to(folder)): p for p in folder.rglob("*")})
    return {
        name: (
            p.read_bytes() if p.is_file() else None,
            p.stat().st_mtime_ns,
            p.stat().st_mode,
        )
        for name, p in entries.items()
    }


def test_a_named_run_leaves_a_foreign_summary_folder_as_it_was(tmp_path):
    """The reviewer's repro: `bench run ... --job foreign2` into a folder
    holding only another program's summary.json. Refused before anything is
    written (the job lock included), every byte and time unchanged."""
    from tests.test_upgrade_mean_and_resume_dx import _evaluation

    tasks = tmp_path / "tasks" / "ok"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"\n')
    jobs = tmp_path / "jobs"
    job = jobs / "foreign2"
    job.mkdir(parents=True)
    (job / "summary.json").write_text('{"tool":"other"}\n')
    import os

    os.utime(job / "summary.json", ns=(1_000_000_000, 1_000_000_000))
    os.utime(job, ns=(2_000_000_000, 2_000_000_000))
    before = _snapshot(job)
    jobs_before = sorted(p.name for p in jobs.iterdir())

    run = _evaluation(tasks.parent, job, preflight=False)
    with pytest.raises(ResumeMismatchError, match="a folder BenchFlow did not make"):
        asyncio.run(run.run())
    assert _snapshot(job) == before
    assert sorted(p.name for p in jobs.iterdir()) == jobs_before
    run._run_single_task.assert_not_called()


def test_the_refusal_names_the_current_job_flag(tmp_path):
    job = _foreign_job(tmp_path / "jobs")
    with pytest.raises(ResumeMismatchError) as refused:
        _check_resume_mismatch(job, EvaluationConfig(agent="oracle"))
    assert refused.value.hint == "pick another --job or another --jobs-dir"


@pytest.mark.parametrize("key", ["benchflow_version", "harness_mode"])
def test_a_trial_naming_benchflow_marks_its_folder(tmp_path, key):
    """bf.run(job_name=...) writes no evaluation.json; its trials say who ran them."""
    job = tmp_path / "jobs" / "my-run"
    _trial(job, "hello__1", {"agent": "oracle", key: "0.8.3"})
    assert _is_benchflow_job(job)
    assert not _is_external_job(job)
    assert _job_folders(tmp_path / "jobs") == [job]


def test_an_empty_folder_is_not_foreign(tmp_path):
    job = tmp_path / "jobs" / STAMP
    (job / ".set-aside").mkdir(parents=True)
    assert not _is_external_job(job)
    assert _job_folders(tmp_path / "jobs") == [job]


def test_an_unnamed_run_starts_a_new_job_beside_a_foreign_one(tmp_path):
    jobs = tmp_path / "jobs"
    foreign = _foreign_job(jobs)
    name = Evaluation._resolve_job_name(jobs, EvaluationConfig(agent="oracle"))
    assert name != foreign.name


def test_a_named_run_refuses_a_foreign_folder_and_moves_nothing(tmp_path):
    jobs = tmp_path / "jobs"
    foreign = _foreign_job(jobs)
    before = sorted(p.relative_to(foreign) for p in foreign.rglob("*"))
    with pytest.raises(ResumeMismatchError, match="a folder BenchFlow did not make"):
        _check_resume_mismatch(foreign, EvaluationConfig(agent="oracle"))
    assert sorted(p.relative_to(foreign) for p in foreign.rglob("*")) == before


def test_an_interrupted_0_7_job_is_known_by_its_rollout_results(tmp_path):
    """A 0.7 job stopped before its summary.json: its trials' result.json are
    BenchFlow rollout results."""
    job = tmp_path / "jobs" / STAMP
    trial = _trial(job, "hello__1234abcd", {"task_path": "hello", "agent": "oracle"})
    assert _is_external_job(job)
    (trial / "result.json").write_text(
        json.dumps({"task_name": "hello", "rollout_name": trial.name, "rewards": None})
    )
    assert _is_benchflow_job(job)
    assert _job_folders(tmp_path / "jobs") == [job]


FIXTURES = Path(__file__).parent / "fixtures" / "job_folders_0_7_11"
REAL_07_JOB = "2026-10-22__06-28-47"


def _real_0_7_11_job_stopped_at_start(jobs: Path) -> Path:
    """A real BenchFlow 0.7.11 job (``bench eval run``, oracle, Docker) stopped
    while its only trial ran: the trial's config.json and its empty agent/,
    artifacts/, trajectory/ and verifier/ folders, no result.json and no
    summary.json. The config.json is the file 0.7.11 wrote, unchanged."""
    job = jobs / REAL_07_JOB
    trial = job / "slow__b22bd4f5"
    trial.mkdir(parents=True)
    (trial / "config.json").write_text(
        (FIXTURES / REAL_07_JOB / trial.name / "config.json").read_text()
    )
    for name in ("agent", "artifacts", "trajectory", "verifier"):
        (trial / name).mkdir()
    return job


def test_a_0_7_job_stopped_before_any_trial_finished_is_benchflows(tmp_path):
    jobs = tmp_path / "jobs"
    job = _real_0_7_11_job_stopped_at_start(jobs)
    assert _is_benchflow_job(job)
    assert not _is_external_job(job)
    assert _job_folders(jobs) == [job]


def test_a_named_run_does_not_call_a_stopped_0_7_job_foreign(tmp_path):
    job = _real_0_7_11_job_stopped_at_start(tmp_path / "jobs")
    try:
        _check_resume_mismatch(job, EvaluationConfig(agent="oracle"))
    except ResumeMismatchError as error:
        assert "did not make" not in str(error)


@pytest.mark.parametrize(
    "missing", ["acp_transport", "sandbox_locked_paths", "agent_idle_timeout_sec"]
)
def test_part_of_the_0_7_trial_keys_does_not_mark_a_folder(tmp_path, missing):
    job = _real_0_7_11_job_stopped_at_start(tmp_path / "jobs")
    config_path = job / "slow__b22bd4f5" / "config.json"
    config = json.loads(config_path.read_text())
    del config[missing]
    config_path.write_text(json.dumps(config))
    assert _is_external_job(job)


def test_the_job_lock_this_run_takes_is_no_marker(tmp_path):
    """A run takes the job lock in the folder before it checks the folder, so
    the lock cannot make a foreign folder BenchFlow's."""
    jobs = tmp_path / "jobs"
    foreign = _foreign_job(jobs)
    stopped_07 = _real_0_7_11_job_stopped_at_start(tmp_path / "jobs07")
    for job, external in ((foreign, True), (stopped_07, False)):
        lock = _acquire_job_lock(job)
        try:
            assert (job / JOB_LOCK).is_file()
            assert _is_external_job(job) is external
        finally:
            lock.release()


def test_a_folder_holding_only_empty_folders_is_foreign(tmp_path):
    """FIX-49 r4 L1: c278f22a6 wrote into unrecognised empty directory trees."""
    job = tmp_path / "jobs" / STAMP
    for name in ("agent", "artifacts"):
        (job / "slow__1" / name).mkdir(parents=True)
    assert _is_external_job(job)
    (job / "slow__1" / "agent" / "notes.txt").write_text("x")
    assert _is_external_job(job)


def test_a_named_run_leaves_empty_foreign_subfolders_unchanged(tmp_path):
    """FIX-49 r4 L1: the review's lost+found and data/sub mount-root shape."""
    from tests.test_upgrade_mean_and_resume_dx import _evaluation

    tasks = tmp_path / "tasks" / "ok"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"\n')
    job = tmp_path / "jobs" / "foreign"
    (job / "lost+found").mkdir(parents=True)
    (job / "data/sub").mkdir(parents=True)
    before = _snapshot(job)
    run = _evaluation(tasks.parent, job, preflight=False)
    with pytest.raises(ResumeMismatchError, match="a folder BenchFlow did not make"):
        asyncio.run(run.run())
    assert _snapshot(job) == before
    run._run_single_task.assert_not_called()


def test_a_foreign_summary_beside_0_7_trials_is_no_0_7_job(tmp_path):
    """A 0.7 trial config.json marks the folder only when the summary.json
    beside it, if any, is the one 0.7 writes."""
    job = _real_0_7_11_job_stopped_at_start(tmp_path / "jobs")
    (job / "summary.json").write_text('{"tool": "other"}')
    assert _is_external_job(job)


FINISHED_07_JOB = "2026-10-22__11-51-51"


def _real_0_7_11_finished_job(jobs: Path) -> Path:
    """A real BenchFlow 0.7.11 job that finished (``bench eval run``, oracle,
    Docker): the summary.json and the trial's config.json 0.7.11 wrote,
    unchanged; its rollout result is left out, so the summary and the trial's
    config.json are what mark it."""
    job = jobs / FINISHED_07_JOB
    trial = job / "fast__9c30cd34"
    trial.mkdir(parents=True)
    source = FIXTURES / FINISHED_07_JOB
    (job / "summary.json").write_bytes((source / "summary.json").read_bytes())
    (trial / "config.json").write_bytes(
        (source / trial.name / "config.json").read_bytes()
    )
    return job


def test_a_finished_0_7_11_job_is_benchflows(tmp_path):
    job = _real_0_7_11_finished_job(tmp_path / "jobs")
    assert _is_benchflow_job(job)
    assert _job_folders(tmp_path / "jobs") == [job]


@pytest.mark.parametrize("missing", ["score_ratio", "memory_scores", "job_name"])
def test_a_summary_without_every_0_7_key_is_no_0_7_job(tmp_path, missing):
    job = _real_0_7_11_finished_job(tmp_path / "jobs")
    summary = json.loads((job / "summary.json").read_text())
    del summary[missing]
    (job / "summary.json").write_text(json.dumps(summary))
    assert _is_external_job(job)


def test_resuming_a_finished_0_7_11_job_keeps_its_files_first(tmp_path, caplog):
    import logging

    from tests.test_upgrade_mean_and_resume_dx import _evaluation

    tasks = tmp_path / "tasks" / "fast"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"\n')
    job = _real_0_7_11_finished_job(tmp_path / "jobs")
    original = (job / "summary.json").read_bytes()
    with caplog.at_level(logging.WARNING):
        asyncio.run(_evaluation(tasks.parent, job, preflight=False).run())
    assert (job / ".pre-0.8" / "summary.json").read_bytes() == original
    assert (job / "summary.json").read_bytes() != original
    assert "made by BenchFlow 0.7" in caplog.text


# --- FIX-49 r3 L3: the review's shapes A and C --------------------------------


@pytest.mark.parametrize(
    "record",
    [
        # The review's shape A: the four keys, another tool's values.
        {"schema_version": "x", "tasks_dir": 7, "job_name": None, "config": "other"},
        {"schema_version": True, "tasks_dir": "/t", "job_name": "j", "config": {}},
        {"schema_version": 1, "tasks_dir": None, "job_name": "j", "config": {}},
        {"schema_version": 1, "tasks_dir": "/t", "job_name": 3, "config": {}},
        {"schema_version": 1, "tasks_dir": "/t", "job_name": "j", "config": []},
    ],
    ids=["review-A", "bool-version", "no-tasks-dir", "int-name", "list-config"],
)
def test_a_record_with_the_four_keys_but_other_types_is_no_marker(tmp_path, record):
    # 6c7426dcb: any evaluation.json holding the four keys marked a 0.8 job.
    job = tmp_path / "jobs" / "A"
    job.mkdir(parents=True)
    (job / "evaluation.json").write_text(json.dumps(record))
    (job / "notes.txt").write_text("mine\n")
    assert not _is_benchflow_job(job)
    assert _is_external_job(job)


def test_a_record_of_every_0_8_shape_is_still_a_marker(tmp_path):
    job = tmp_path / "jobs" / "named"
    job.mkdir(parents=True)
    record = {**_record(job), "benchflow_version": "0.8.2", "planned_tasks": ["a"]}
    (job / "evaluation.json").write_text(json.dumps(record))
    assert _is_benchflow_job(job)


@pytest.mark.parametrize(
    "entries",
    [
        {".git/HEAD": "ref: refs/heads/main\n", ".env": "SECRET=1\n"},  # review's C
        {".env": "SECRET=1\n"},
        {".notes": "mine\n"},
        {".cache/x": "1"},
    ],
    ids=["review-C", "env", "dot-file", "dot-folder"],
)
def test_a_folder_of_only_foreign_dot_entries_is_foreign(tmp_path, entries):
    # 6c7426dcb skipped every dot-entry, so these counted as empty folders
    # and a job was written into them.
    job = tmp_path / "jobs" / "C"
    for name, text in entries.items():
        (job / name).parent.mkdir(parents=True, exist_ok=True)
        (job / name).write_text(text)
    assert _is_external_job(job)
    with pytest.raises(ResumeMismatchError, match="a folder BenchFlow did not make"):
        _check_resume_mismatch(job, EvaluationConfig(agent="oracle"))


def test_a_named_run_leaves_a_dot_only_folder_as_it_was(tmp_path):
    """The review's h2 shape C through a run: refused before anything is
    written, every byte and time unchanged."""
    from tests.test_upgrade_mean_and_resume_dx import _evaluation

    tasks = tmp_path / "tasks" / "ok"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"\n')
    job = tmp_path / "jobs" / "C"
    (job / ".git").mkdir(parents=True)
    (job / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (job / ".env").write_text("SECRET=1\n")
    before = _snapshot(job)

    run = _evaluation(tasks.parent, job, preflight=False)
    with pytest.raises(ResumeMismatchError, match="a folder BenchFlow did not make"):
        asyncio.run(run.run())
    assert _snapshot(job) == before
    run._run_single_task.assert_not_called()


@pytest.mark.parametrize(
    "names",
    [
        [JOB_LOCK],
        [".benchflow-prepared"],
        [".pre-0.8/summary.json", ".pre-0.8.tmp-0123abcd/x"],
        [".set-aside/2026-10-22T00-00-00Z/set-aside.json"],
        [".budget-spend/t.json"],
        [".summary.json.k2j3h4_x", ".evaluation.json.abc", ".results.jsonl.q1"],
    ],
    ids=["lock", "prepared", "pre-0.8", "set-aside", "budget", "atomic-temp"],
)
def test_benchflows_own_dot_entries_leave_a_folder_empty(tmp_path, names):
    job = tmp_path / "jobs" / "own"
    for name in names:
        (job / name).parent.mkdir(parents=True, exist_ok=True)
        (job / name).write_text("{}")
    assert not _is_external_job(job)


def test_finder_metadata_also_makes_an_unrecognised_folder_foreign(tmp_path):
    """FIX-49 r4 L1: any non-BenchFlow entry counts, including Finder's file."""
    (tmp_path / ".DS_Store").touch()
    assert _is_external_job(tmp_path)
