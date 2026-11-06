"""DX-e round 2: the review of 82913086.

M1: the python3-less outside-writes listing failed open (a find that refused
its arguments exited 0 through ``head``). M2: the ``.pre-0.8/`` copy was not
atomic. DX-E-P4-TWO-MEANS: a task that failed before its trial folder existed
counted in summary.json (0.40 over 5) but not in ``bf.load_job`` or ``bench
eval metrics`` (0.50 over 4). L1-L6: the lows.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import shutil
import warnings
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.models import RolloutResult
from benchflow.sandbox._base import ExecResult
from tests.test_upgrade_mean_and_resume_dx import _evaluation, _pre_08_job

# --- M1: a listing that cannot run records nothing ----------------------------


class _NoPythonBrokenFind:
    """python3 is missing and find refuses its arguments (BusyBox without
    -newermt, before the fallback), printing only to stderr."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec(self, cmd: str, *, user: str = "root", timeout_sec: int = 30):
        self.commands.append(cmd)
        if cmd.startswith("python3"):
            return ExecResult(
                return_code=127, stdout="", stderr="sh: python3: not found\n"
            )
        return ExecResult(
            return_code=1, stdout="", stderr="find: unrecognized: -newer\n"
        )


@pytest.mark.asyncio
async def test_a_find_that_fails_records_none_and_warns(caplog) -> None:
    from benchflow.rollout._separate_verifier import scan_outside_workspace_writes

    env = _NoPythonBrokenFind()
    with caplog.at_level(logging.WARNING):
        report = await scan_outside_workspace_writes(
            env, since=1792170000.0, keep=["/app"]
        )

    assert report is None
    assert "Could not list files outside the workspace" in caplog.text
    assert "find exited 1" in caplog.text
    # The find runs without a pipe that could hide its status.
    assert "| head" not in env.commands[-1]


def test_the_listing_counts_a_name_with_a_newline_once() -> None:
    from benchflow.rollout._separate_verifier import _outside_writes_report

    report = _outside_writes_report("/", "/root/a\nb.txt\0/srv/c.csv\0")
    assert report["count"] == 2 and not report["truncated"]
    assert report["sample"] == ["/root/a\nb.txt", "/srv/c.csv"]
    cut = _outside_writes_report("/", "/root/a\0/srv/par")
    assert cut["count"] == 1 and cut["truncated"]


def test_the_listing_script_falls_back_to_newer_with_a_stamp() -> None:
    from benchflow.rollout._separate_verifier import _outside_writes_find

    script = _outside_writes_find("/", 1792170000.5, ["/app"], [".cache/pip"])
    assert "find / -maxdepth 0 -newermt @0" in script
    assert 'touch -d @1792169999 "$ref"' in script
    assert '-newer "$ref" -print0' in script
    assert '[ "$rc" -eq 0 ] || exit "$rc"' in script


# --- M2: the copy is whole or absent ------------------------------------------


def _files(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir()) if p.is_file()}


def test_a_failed_copy_changes_nothing_and_the_retry_keeps_every_file(
    tmp_path, monkeypatch
) -> None:
    from benchflow.errors import UserError

    tasks, job = _pre_08_job(tmp_path)
    (job / "verifiers.jsonl").write_text('{"task": "verifier-crash"}\n')
    originals = _files(job)
    real_copy = shutil.copy2
    calls = {"n": 0}

    def full_disk(src, dst, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            Path(dst).write_bytes(Path(src).read_bytes()[:3])  # a partial file
            raise OSError(28, "No space left on device")
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", full_disk)
    with pytest.raises(UserError) as caught:
        asyncio.run(_evaluation(tasks, job, preflight=False).run())
    message = str(caught.value)
    assert f"Could not copy the BenchFlow 0.7 files of job {job.name}" in message
    assert "No space left on device" in message
    assert "nothing in the job was changed" in message
    assert "free space" in (caught.value.hint or "")
    # Nothing rewritten, and no copy that looks complete.
    assert _files(job) == originals
    assert not (job / ".pre-0.8").exists()
    assert not list(job.glob(".pre-0.8.tmp-*"))
    assert not (job / "evaluation.json").exists()

    monkeypatch.setattr(shutil, "copy2", real_copy)
    asyncio.run(_evaluation(tasks, job, preflight=False).run())
    assert _files(job / ".pre-0.8") == originals


def test_a_partial_copy_from_an_interrupted_run_is_made_again(tmp_path) -> None:
    tasks, job = _pre_08_job(tmp_path)
    originals = _files(job)
    # What an older, non-atomic copy left after an interruption.
    (job / ".pre-0.8").mkdir()
    (job / ".pre-0.8" / "summary.json").write_text("{")
    (job / ".pre-0.8.tmp-deadbeef").mkdir()

    asyncio.run(_evaluation(tasks, job, preflight=False).run())

    assert _files(job / ".pre-0.8") == originals
    assert not list(job.glob(".pre-0.8.tmp-*"))


# --- DX-E-P4-TWO-MEANS: one every-trial mean ----------------------------------


def _tasks(tmp_path: Path, names: tuple[str, ...]) -> Path:
    tasks = tmp_path / "tasks"
    for name in names:
        (tasks / name).mkdir(parents=True, exist_ok=True)
        (tasks / name / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    return tasks


# Not a control run: bf.load_job(...).denominators() leaves controls out.
AGENT = "claude-agent-acp"


def _job_with_a_folderless_error(tmp_path: Path):
    from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig

    tasks = _tasks(tmp_path, ("ok", "typo"))
    job = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            agent=AGENT, concurrency=1, retry=RetryConfig(max_retries=0)
        ),
        job_name="j",
        preflight=False,
    )

    async def fake_run(task_path, _cfg):
        if task_path.name == "typo":
            # A task.toml typo: refused before any rollout folder exists.
            return RolloutResult(
                task_name="typo",
                error="TaskConfigKeyError: unknown task.toml key 'agent.timout_sec'",
                agent=AGENT,
            )
        trial = job.job_dir / "ok__aaaa1111"
        trial.mkdir(parents=True, exist_ok=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "ok",
                    "rollout_name": trial.name,
                    "rewards": {"reward": 1.0},
                    "error": None,
                    "verifier_error": None,
                    "agent": AGENT,
                }
            )
        )
        (trial / "config.json").write_text(json.dumps({"agent": AGENT}))
        return RolloutResult(
            task_name="ok",
            rollout_name=trial.name,
            rewards={"reward": 1.0},
            agent=AGENT,
            rollout_dir=trial,
        )

    job._run_single_task = AsyncMock(side_effect=fake_run)
    return job


def test_every_reader_counts_the_trial_that_failed_before_its_folder(
    tmp_path,
) -> None:
    import benchflow as bf
    from benchflow.metrics import collect_metrics

    job = _job_with_a_folderless_error(tmp_path)
    result = asyncio.run(job.run())

    summary = json.loads((job.job_dir / "summary.json").read_text())
    loaded = bf.load_job(job.job_dir).denominators()
    metrics = collect_metrics(job.job_dir)
    # One every-trial mean, over one stated n, in every place.
    assert summary["mean_reward"] == pytest.approx(0.5)
    assert summary["mean_reward_n"] == 2
    assert (loaded.attempted, loaded.mean_reward) == (2, pytest.approx(0.5))
    assert (metrics.total, metrics.mean_reward) == (2, pytest.approx(0.5))
    assert result.mean_reward == pytest.approx(0.5) and result.mean_reward_n == 2
    # Its folder holds the error, and a resume still runs the task again.
    folders = [p for p in job.job_dir.iterdir() if p.name.startswith("typo__")]
    assert len(folders) == 1
    recorded = json.loads((folders[0] / "result.json").read_text())
    assert recorded["error"].startswith("TaskConfigKeyError")
    again = _job_with_a_folderless_error(tmp_path)
    second = asyncio.run(again.run())
    ran = [c.args[0].name for c in again._run_single_task.call_args_list]
    assert ran == ["typo"] and second.reused == 1


# --- L1: the file and the SDK say which mean they hold ------------------------


def test_summary_and_the_sdk_result_name_the_mean(tmp_path) -> None:
    job = _job_with_a_folderless_error(tmp_path)
    result = asyncio.run(job.run())

    summary = json.loads((job.job_dir / "summary.json").read_text())
    assert summary["mean_reward_rule"].startswith("every trial, an unscored one as 0")
    assert "BenchFlow 0.7's mean_reward" in summary["mean_reward_rule"]
    assert summary["mean_reward_scored_only"] == pytest.approx(1.0)
    assert summary["mean_reward_scored_only_n"] == 1
    assert result.mean_reward_scored_only_n == 1
    text = repr(result)
    assert "mean_reward=0.5 (over 2 trials, unscored as 0)" in text
    assert "mean_reward_scored_only=1.0 (over 1)" in text


# --- L2, L3: whose job it is ---------------------------------------------------


def _batch_folder(tmp_path: Path) -> tuple[Path, Path]:
    """A folder 0.8's bf.run_batch writes: trials, no evaluation.json."""
    tasks = _tasks(tmp_path, ("ok",))
    job = tmp_path / "jobs" / "exp1"
    trial = job / "ok__bbbb2222"
    trial.mkdir(parents=True)
    (trial / "config.json").write_text(
        json.dumps({"agent": "oracle", "harness_mode": "acp"})
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "ok",
                "rollout_name": trial.name,
                "rewards": {"reward": 1.0},
                "error": None,
                "verifier_error": None,
                "agent": "oracle",
            }
        )
    )
    return tasks, job


def test_a_run_batch_folder_is_not_called_a_0_7_job(tmp_path, caplog) -> None:
    from benchflow.evaluation import Evaluation, NoEvaluationRecordError

    tasks, job = _batch_folder(tmp_path)
    with caplog.at_level(logging.WARNING):
        asyncio.run(_evaluation(tasks, job, preflight=False).run())
    assert "made by BenchFlow 0.7" not in caplog.text
    assert not (job / ".pre-0.8").exists()
    record = json.loads((job / "evaluation.json").read_text())
    assert record["created_by_benchflow_version"] == "unknown"

    _, other = _batch_folder(tmp_path / "second")
    with pytest.raises(NoEvaluationRecordError) as caught:
        Evaluation.resume(other)
    assert "made without a run record (by bf.run or bf.run_batch)" in str(caught.value)
    assert "0.7" not in str(caught.value)


def test_an_older_record_without_the_key_is_not_claimed(tmp_path) -> None:
    tasks, job = _batch_folder(tmp_path)
    (job / "evaluation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tasks_dir": str(tasks),
                "job_name": job.name,
                "config": {},
            }
        )
    )
    asyncio.run(_evaluation(tasks, job, preflight=False).run())
    record = json.loads((job / "evaluation.json").read_text())
    assert record["created_by_benchflow_version"] == "unknown"


# --- L4: no link in the folder compared with is followed -----------------------


def test_relative_to_does_not_follow_a_link_at_the_folder(tmp_path) -> None:
    from benchflow._utils.reachable_write import ReachablePath

    trial = tmp_path / "trial"
    (trial / "verifier").mkdir(parents=True)
    (trial / "agent").mkdir()
    (trial / "agent" / "up").symlink_to("..")
    reward = ReachablePath.rollout(trial, "verifier") / "reward.txt"
    assert not reward.is_relative_to(trial / "agent" / "up")
    assert reward.is_relative_to(trial)


# --- L5: a copy of a RuntimeConfig is silent ----------------------------------


def test_replace_and_asdict_copies_do_not_warn() -> None:
    import benchflow as bf

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = bf.RuntimeConfig()
        dataclasses.replace(cfg, timeout=60)
        bf.RuntimeConfig(
            **{
                k: v
                for k, v in dataclasses.asdict(cfg).items()
                if k not in ("reviewer", "codex_apps_policy")
            }
        )
    assert not [w for w in caught if w.category is DeprecationWarning]
    assert cfg.reward_stream is True and cfg.max_rounds == 10


# --- L6: the rebuilt command says what it does not carry ----------------------


def test_the_rebuilt_resume_command_names_the_flags_it_cannot_carry(
    tmp_path,
) -> None:
    from benchflow.evaluation import Evaluation, NoEvaluationRecordError

    _, job = _pre_08_job(tmp_path)
    with pytest.raises(NoEvaluationRecordError) as caught:
        Evaluation.resume(job)
    assert "plus any other flags it had (--agent-env, --sandbox-user" in (
        caught.value.hint
    )
