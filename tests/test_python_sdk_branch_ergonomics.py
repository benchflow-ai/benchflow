"""bf.branch ergonomics for training workflows.

``await bf.abranch(...)`` used to print nothing until it finished;
``BranchChildResult`` had no ``advantage``; ``BranchPlanError`` messages
named CLI flags (``--resume-session needs…``, ``--checkpoint prompt:9…``).
Branching from automatic checkpoint ``prompt:1`` was ``fork="prompt:1"``
(found only in the signature), and ``agent=`` and the task path were still
required with ``from_checkpoint`` (the CLI takes them from the source trial).

Now: ``child.advantage`` (reward − its fork's V), ``on_event=`` receives
progress events (trial started, checkpoint reached, each child finished,
fork finished, trial finished), errors name Python keywords, ``checkpoint=``
is the name for ``fork=``, and ``task_path``/``agent``/``model`` default to
the checkpoint's trial.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchflow as bf
from benchflow.branch_run import run_branch_trial
from tests.test_branch_run import ScriptedRollout, _plan
from tests.test_python_sdk_branch import TASK, fake_trial  # noqa: F401


def test_children_carry_their_advantage(fake_trial, tmp_path: Path) -> None:  # noqa: F811
    result = bf.branch(
        TASK, agent="oracle", children={"a": None, "b": None}, jobs_dir=tmp_path
    )
    assert result.child("a").advantage == 0.5
    assert result.child("b").advantage == -0.5
    assert result.to_records()[0]["advantage"] == 0.5


def test_on_event_reports_the_trial(fake_trial, tmp_path: Path) -> None:  # noqa: F811
    events: list[dict] = []
    bf.branch(
        TASK,
        agent="oracle",
        children={"a": None, "b": None},
        jobs_dir=tmp_path,
        on_event=events.append,
    )
    kinds = [e["event"] for e in events]
    assert kinds[0] == "trial_started" and kinds[-1] == "trial_finished"
    assert events[-1]["value"] == 0.5


async def test_run_branch_trial_reports_children_as_they_finish(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("Do it.")
    events: list[dict] = []
    plan = _plan(tmp_path, task_paths=[task], checkpoint_after=1)
    await run_branch_trial(
        plan, task, rollout_factory=ScriptedRollout, on_event=events.append
    )
    kinds = [e["event"] for e in events]
    assert kinds == [
        "checkpoint_reached",
        "child_finished",
        "child_finished",
        "fork_finished",
        "parent_finished",
    ]
    finished = [e for e in events if e["event"] == "child_finished"]
    assert [e["label"] for e in finished] == ["baseline", "hint"]
    assert all("reward" in e for e in finished)


def test_errors_name_python_keywords(fake_trial) -> None:  # noqa: F811
    with pytest.raises(bf.BranchPlanError) as info:
        bf.branch(
            TASK,
            agent="oracle",
            children={"a": None, "b": None},
            resume_session=True,
        )
    message = str(info.value)
    assert "resume_session" in message
    assert "--resume-session" not in message
    assert "--checkpoint-after-prompt" not in message


def _source(tmp_path: Path) -> Path:
    trial = tmp_path / "old" / "hello-world-task__old"
    trial.mkdir(parents=True)
    (trial / "config.json").write_text(
        json.dumps({"task_path": str(TASK), "agent": "oracle", "model": None})
    )
    (trial / "checkpoints.json").write_text(
        json.dumps(
            {
                "kind": "benchflow-checkpoints",
                "checkpoints": [
                    {
                        "id": "prompt:1",
                        "after_prompt": 1,
                        "status": "kept",
                        "provider": "docker",
                        "ref": "bf-snap-kept",
                    }
                ],
            }
        )
    )
    return trial


def test_checkpoint_keyword_and_defaults_from_the_source(
    fake_trial,  # noqa: F811
    tmp_path: Path,
) -> None:
    trial = _source(tmp_path)
    bf.branch(
        from_checkpoint=trial,
        checkpoint="prompt:1",
        children={"a": None, "b": None},
        jobs_dir=tmp_path / "jobs",
    )
    [plan] = fake_trial
    assert plan.task_paths == [TASK]
    assert plan.agent == "oracle"
    assert plan.source.fork_id == "prompt:1"
    with pytest.raises(bf.BranchPlanError, match="checkpoint= or fork="):
        bf.branch(
            from_checkpoint=trial,
            checkpoint="prompt:1",
            fork="prompt:1",
            children={"a": None, "b": None},
        )
    with pytest.raises(bf.BranchPlanError, match="task_path"):
        bf.branch(agent="oracle", children={"a": None, "b": None})
