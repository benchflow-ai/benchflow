"""A task on BenchFlow's in-package refusal list is refused before the turn.

seta-env 1243's reference solution fails under BenchFlow's always-on
root-exec limit, so a 0 for it would be BenchFlow's doing. The lead's ruling:
a listed task is refused before the turn, unscored, for every agent; the list
is keyed by benchmark, task id and BenchFlow's task digest and lives in the
package; the reason names the cause in plain words.
"""

from __future__ import annotations

import asyncio
import json
import logging
from importlib import resources
from pathlib import Path

import pytest

REASON = (
    "this task's reference solution fails under BenchFlow's root-exec limit; not scored"
)
SETA_1243 = "sha256:4723dae6c5cadd8300021e72fbf7d29a29227ba63542c171580ef7e6ddd4600b"


def _task(root: Path, name: str = "1243") -> Path:
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )
    (task / "instruction.md").write_text("Deliver the mail.\n")
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    return task


@pytest.fixture
def listed(monkeypatch):
    """Put the given task folder's digest on the list in place of the package's."""
    from benchflow._utils.task_authoring import task_digest
    from benchflow.task import refused_tasks as module

    def put(task: Path) -> str:
        digest = task_digest(task)
        entry = module.RefusedTask("seta-env", task.name, digest, REASON)
        monkeypatch.setattr(module, "refused_tasks", lambda: (entry,))
        return digest

    return put


def test_the_package_list_names_seta_env_1243():
    from benchflow.task.refused_tasks import RefusedTask, refused_tasks

    listing = resources.files("benchflow.task").joinpath("refused_tasks.json")
    assert listing.is_file()
    assert RefusedTask("seta-env", "1243", SETA_1243, REASON) in refused_tasks()


def test_a_listed_task_is_refused_with_the_reason(tmp_path, listed):
    from benchflow._utils.scoring import TASK_REFUSED, classify_error
    from benchflow.task.refused_tasks import TaskRefusedError, refuse_listed_task

    task = _task(tmp_path)
    listed(task)
    with pytest.raises(TaskRefusedError) as caught:
        refuse_listed_task(task)
    message = str(caught.value)
    assert message == f"task refused before the turn: seta-env 1243: {REASON}"
    assert classify_error(f"TaskRefusedError: {message}") == TASK_REFUSED


def test_one_changed_byte_is_another_task_and_runs(tmp_path, listed):
    from benchflow.task.refused_tasks import refusal_for, refuse_listed_task

    task = _task(tmp_path)
    digest = listed(task)
    assert refusal_for(task) is not None
    with (task / "instruction.md").open("a") as stream:
        stream.write("!")
    assert refusal_for(task) is None
    refuse_listed_task(task)
    from benchflow._utils.task_authoring import task_digest

    assert task_digest(task) != digest


def test_another_task_id_with_the_same_files_is_not_refused(tmp_path, listed):
    from benchflow.task.refused_tasks import refusal_for

    listed(_task(tmp_path / "a"))
    assert refusal_for(_task(tmp_path / "b", name="1244")) is None


class _PastTheList(Exception):
    """Raised by the first step of setup after the list's check."""


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["oracle", "nop", "claude-agent-acp"])
async def test_setup_refuses_before_any_sandbox_for_every_agent(
    tmp_path, listed, monkeypatch, agent
):
    import benchflow.rollout as rollout_module
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.task.refused_tasks import TaskRefusedError

    task = _task(tmp_path)
    listed(task)

    def past(*_a, **_k):
        raise _PastTheList

    monkeypatch.setattr(rollout_module, "prepare_terminal_review", past)
    rollout = await Rollout.create(
        RolloutConfig(
            task_path=task,
            agent=agent,
            model=None if agent in ("oracle", "nop") else "claude-sonnet-4-6",
            jobs_dir=tmp_path / "jobs",
        )
    )
    with pytest.raises(TaskRefusedError, match="not scored"):
        await rollout.setup()
    assert getattr(rollout, "_env", None) is None

    with (task / "instruction.md").open("a") as stream:
        stream.write("!")
    edited = await Rollout.create(
        RolloutConfig(
            task_path=task,
            agent=agent,
            model=None if agent in ("oracle", "nop") else "claude-sonnet-4-6",
            jobs_dir=tmp_path / "jobs2",
        )
    )
    with pytest.raises(_PastTheList):
        await edited.setup()


def test_the_job_summary_counts_it_unscored(tmp_path, listed, caplog):
    from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig

    tasks_dir = tmp_path / "tasks"
    listed(_task(tasks_dir))
    job = Evaluation(
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            agent="oracle", concurrency=1, retry=RetryConfig(max_retries=2)
        ),
        job_name="j",
    )
    with caplog.at_level(logging.INFO):
        result = asyncio.run(job.run())

    # In the task count, unscored: not a 0, not a failure, never retried.
    assert (result.total, result.passed, result.failed, result.errored) == (1, 0, 0, 1)
    summary = json.loads((tmp_path / "jobs" / "j" / "summary.json").read_text())
    assert summary["error_categories"] == {"task_refused": 1}
    (row,) = summary["refused_before_turn"]["trials"]
    assert row["task"] == "1243"
    assert REASON in row["reason"]
    (trial,) = (tmp_path / "jobs" / "j").glob("1243__*/result.json")
    record = json.loads(trial.read_text())
    assert record["rewards"] is None
    assert record["error_category"] == "task_refused"
    assert REASON in record["error"]
    assert len(list((tmp_path / "jobs" / "j").glob("1243__*"))) == 1
    assert "1243: refused before the turn, unscored: " in " ".join(caplog.messages)
