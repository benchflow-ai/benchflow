"""A named job is not resumed with other settings.

``Evaluation(..., job_name=<existing job>)``, ``Evaluation.resume`` and their
command lines (``bench run --job``, ``bench eval run --job-name``, ``bench eval
resume``) refuse a job whose earlier trials ran with another value of a
setting that changes what a trial measures, or with an earlier version of a
task. The refusal comes before anything in the job is copied, moved or run,
names each change with its old and new value (never a secret's), and says how
many finished trials it concerns; the command line exits 2. ``restart_changed
=True`` (``--restart-changed``) sets those trials aside and runs them again,
and the job's summary.json and run-card.json record the restart
(``restarted_after_change``). Unnamed runs are unchanged: other settings start
a new job.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from benchflow._utils.result_paths import SET_ASIDE_DIR, load_task_results
from benchflow.budget import Budget
from benchflow.evaluation import (
    _SHOWN_CHANGES,
    Evaluation,
    ResumeMismatchError,
    _changed_resume_refusal,
    _restart_record,
    _ResumeChange,
)
from tests.test_resume_settings_match import (
    _config,
    _evaluation,
    _fake_runner,
    _first_job,
    _mark_reviewed,
    _run,
    _tasks,
    _tree,
)

SECRET = "sk-test-not-a-real-key-1234567890"


def _resumed(job_dir: Path, **overrides) -> Evaluation:
    job = Evaluation.resume(job_dir, **overrides)
    job._run_single_task = AsyncMock(side_effect=_fake_runner(job))
    return job


def _refused(job: Evaluation) -> ResumeMismatchError:
    with pytest.raises(ResumeMismatchError) as refused:
        _run(job)
    job._run_single_task.assert_not_called()
    return refused.value


_MODEL = "model='m1' (this run: 'm2')"
_EFFORT = "reasoning_effort=None (this run: 'high')"
_ENV = "agent_env MY_FLAG changed"


@pytest.mark.parametrize(
    ("overrides", "named"),
    [
        ({"model": "m2"}, [_MODEL]),
        ({"reasoning_effort": "high"}, [_EFFORT]),
        ({"agent_env": {"MY_FLAG": "off"}}, [_ENV]),
        (
            {
                "model": "m2",
                "reasoning_effort": "high",
                "agent_env": {"MY_FLAG": "off"},
            },
            [_MODEL, _EFFORT, _ENV],
        ),
    ],
    ids=["model", "effort", "agent-env", "all-three"],
)
def test_a_resume_with_changed_settings_is_refused_naming_each(
    tmp_path, overrides, named
) -> None:
    old = _first_job(tmp_path, job_name="named", agent_env={"MY_FLAG": "on"})
    files = _tree(old)
    error = _refused(_resumed(old, **overrides))
    assert error.changed == named
    assert error.finished_trials == 2 and error.exit_code == 2
    message = str(error)
    assert message.startswith("Refusing to resume job named: 2 finished trials ")
    assert "ran " + "; ".join(named) + "." in message
    assert "nothing in the job was moved or run" in message
    # The next step: the job's own settings, or the override.
    hint = error.hint or ""
    assert f"`bench eval resume {shlex.quote(str(old))}`" in hint
    assert "--restart-changed (in Python, restart_changed=True)" in hint
    assert f"{shlex.quote(str(old))}/.set-aside/" in hint
    # agent_env values are named, never shown.
    assert "'off'" not in message and "MY_FLAG=off" not in message
    # Nothing in the job was written, copied or moved.
    assert _tree(old) == files


def test_a_named_rerun_with_changed_settings_is_refused(tmp_path) -> None:
    """The same refusal for Evaluation(job_name=...) as for Evaluation.resume."""
    old = _first_job(tmp_path, job_name="named")
    files = _tree(old)
    job = _evaluation(tmp_path, _config(model="m2"), job_name="named")
    error = _refused(job)
    assert error.changed == [_MODEL] and error.exit_code == 2
    assert _tree(old) == files


def test_restart_changed_sets_the_trials_aside_and_records_it(tmp_path) -> None:
    old = _first_job(tmp_path, job_name="named", agent_env={"MY_FLAG": "on"})
    job = _resumed(
        old,
        model="m2",
        reasoning_effort="high",
        agent_env={"MY_FLAG": "off"},
        restart_changed=True,
    )
    result = _run(job)
    assert (result.reused, result.ran) == (0, 2)
    assert {r["model"] for r in load_task_results(old).values()} == {"m2"}
    (dest,) = (old / SET_ASIDE_DIR).iterdir()
    assert len(load_task_results(dest)) == 2
    kept = json.loads((dest / "set-aside.json").read_text())
    assert kept["changed"] == [_MODEL, _EFFORT, _ENV]
    assert (kept["trials"], kept["finished_trials"]) == (2, 2)
    expected = {
        "at": kept["set_aside_at"],
        "changed": [_MODEL, _EFFORT, _ENV],
        "set_aside": f"{SET_ASIDE_DIR}/{dest.name}",
        "trials": 2,
        "finished_trials": 2,
    }
    summary = json.loads((old / "summary.json").read_text())
    assert summary["restarted_after_change"] == expected
    card = json.loads((old / "run-card.json").read_text())
    assert card["restarted_after_change"] == expected
    # The record names this run's settings, so the next resume resumes.
    assert _run(_resumed(old)).reused == 2
    assert list((old / SET_ASIDE_DIR).iterdir()) == [dest]
    # And the restart stays on the job's records after that resume.
    summary = json.loads((old / "summary.json").read_text())
    assert summary["restarted_after_change"] == expected
    card = json.loads((old / "run-card.json").read_text())
    assert card["restarted_after_change"] == expected


def test_an_unchanged_resume_still_resumes_with_or_without_the_override(
    tmp_path,
) -> None:
    old = _first_job(tmp_path, job_name="named")
    for restart_changed in (False, True):
        job = _resumed(old, restart_changed=restart_changed)
        result = _run(job)
        assert (result.reused, result.ran) == (2, 0)
    assert not (old / SET_ASIDE_DIR).exists()
    assert "restarted_after_change" not in json.loads(
        (old / "summary.json").read_text()
    )


def test_an_unnamed_rerun_with_changed_settings_still_starts_a_new_job(
    tmp_path,
) -> None:
    old = _first_job(tmp_path)
    files = _tree(old)
    job = _evaluation(tmp_path, _config(model="m2"))
    result = _run(job)
    assert job.job_dir != old
    assert (result.reused, result.ran) == (0, 2)
    assert _tree(old) == files
    assert not (job.job_dir / SET_ASIDE_DIR).exists()


@pytest.mark.parametrize(
    "case", ["raised-budget", "rotated-key", "reviewer-of-unreviewed-job"]
)
def test_what_is_no_change_resumes_a_named_job(tmp_path, case) -> None:
    """A bigger cap after a budget stop, a rotated key and the reviewer of a
    job none of whose trials was reviewed are not changes: no refusal."""
    if case == "raised-budget":
        old = _first_job(tmp_path, job_name="named", budget=Budget(max_rollouts=2))
        summary = json.loads((old / "summary.json").read_text())
        summary["budget"] = {**summary.get("budget", {}), "stopped": True}
        (old / "summary.json").write_text(json.dumps(summary))
        rerun = _config(budget=Budget(max_rollouts=10))
    elif case == "rotated-key":
        old = _first_job(
            tmp_path, job_name="named", agent_env={"OPENAI_API_KEY": SECRET}
        )
        rerun = _config(agent_env={"OPENAI_API_KEY": "sk-test-rotated-not-real-0987"})
    else:
        old = _first_job(tmp_path, job_name="named")
        rerun = _config(reviewer={"model": "judge-2"})
    result = _run(_evaluation(tmp_path, rerun, job_name="named"))
    assert (result.reused, result.ran) == (2, 0)
    assert not (old / SET_ASIDE_DIR).exists()


def test_a_reviewer_change_of_a_reviewed_job_is_refused(tmp_path) -> None:
    old = _first_job(tmp_path, job_name="named")
    _mark_reviewed(old)
    job = _evaluation(tmp_path, _config(reviewer={"model": "judge-2"}), "named")
    error = _refused(job)
    assert "reviewer.model=None (this run: 'judge-2')" in str(error)
    # The reviewed trial has no result.json: it counts as unfinished.
    assert "2 finished trials (and 1 unfinished) in it ran" in str(error)


def test_a_refusal_counts_the_changes_it_does_not_name(tmp_path) -> None:
    """A dataset update can change hundreds of task digests: the message
    names the first ones and counts the rest; ``changed`` has every one."""
    job_dir = _first_job(tmp_path, job_name="named")
    changed = [
        f"task 'task-{i}' at task_digest x (this run: y; the task's files "
        "changed since)"
        for i in range(_SHOWN_CHANGES + 3)
    ]
    error = _changed_resume_refusal(
        job_dir, _ResumeChange([], "r", changed, whole_job=False)
    )
    assert f"task 'task-{_SHOWN_CHANGES - 1}'" in str(error)
    assert f"task 'task-{_SHOWN_CHANGES}'" not in str(error)
    assert "; and 3 more." in str(error)
    assert error.changed == changed
    assert "put the changed tasks' files back" in (error.hint or "")


def test_the_restart_record_reads_a_set_aside_from_before_the_list(tmp_path) -> None:
    """A set-aside.json BenchFlow 0.9 wrote has a reason and no changed list."""
    job_dir = tmp_path / "job"
    kept = job_dir / SET_ASIDE_DIR / "2026-10-01T00-00-00Z"
    (kept / "task-0__1").mkdir(parents=True)
    (kept / "task-0__1" / "result.json").write_text("{}")
    (kept / "task-1__1").mkdir()
    (kept / "set-aside.json").write_text(
        json.dumps(
            {
                "set_aside_at": "2026-10-01T00:00:00+00:00",
                "reason": "model='m1' (this run: 'm2')",
                "moved": ["task-0__1", "task-1__1", "summary.json"],
            }
        )
    )
    earlier = job_dir / SET_ASIDE_DIR / "2026-09-01T00-00-00Z"
    earlier.mkdir()
    (earlier / "set-aside.json").write_text(
        json.dumps({"set_aside_at": "2026-09-01T00:00:00+00:00", "reason": "x"})
    )
    assert _restart_record(job_dir) == {
        "at": "2026-10-01T00:00:00+00:00",
        "changed": ["model='m1' (this run: 'm2')"],
        "set_aside": f"{SET_ASIDE_DIR}/2026-10-01T00-00-00Z",
        "trials": 2,
        "finished_trials": 1,
    }
    assert _restart_record(tmp_path / "never-set-aside") is None


# --- the command line -----------------------------------------------------------


def _fake_rollouts(monkeypatch) -> list[str]:
    """Trials without sandboxes: each writes a scored result and its config."""
    from benchflow.loop_strategies import loop_block
    from benchflow.models import RolloutResult

    ran: list[str] = []

    async def fake(self, task_path, cfg):
        ran.append(task_path.name)
        trial = self.job_dir / f"{task_path.name}__{len(ran):08d}"
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task_path.name,
                    "rollout_name": trial.name,
                    "rewards": {"reward": 1.0},
                    "agent": cfg.agent,
                }
            )
        )
        (trial / "config.json").write_text(
            json.dumps({"agent": cfg.agent, "loop": loop_block(cfg.loop_strategy)})
        )
        return RolloutResult(
            task_path.name,
            rollout_name=trial.name,
            rewards={"reward": 1.0},
            agent=cfg.agent,
            rollout_dir=trial,
        )

    monkeypatch.setattr(Evaluation, "_run_single_task", fake)
    return ran


def _bench(*args: str):
    from benchflow.cli.main import app

    return CliRunner().invoke(app, list(args), terminal_width=200)


def _squash(text: str) -> str:
    return " ".join(text.split())


def test_bench_run_job_name_with_other_settings_exits_2(tmp_path, monkeypatch) -> None:
    ran = _fake_rollouts(monkeypatch)
    tasks, jobs = _tasks(tmp_path), tmp_path / "jobs"
    line = ("run", str(tasks), "--jobs-dir", str(jobs), "--job", "exp1")
    first = _bench(*line, "--extra-instruction", "Be brief.")
    assert first.exit_code == 0, first.output
    assert len(ran) == 2
    files = _tree(jobs / "exp1")

    refused = _bench(*line, "--extra-instruction", "Answer in French.")
    assert refused.exit_code == 2, refused.output
    out = _squash(refused.output)
    assert (
        "Refusing to resume job exp1: 2 finished trials in it ran "
        "extra_instruction='Be brief.' (this run: 'Answer in French.')"
    ) in out
    assert "--restart-changed" in out and "bench eval resume" in out
    assert len(ran) == 2
    assert _tree(jobs / "exp1") == files

    restarted = _bench(
        *line, "--extra-instruction", "Answer in French.", "--restart-changed"
    )
    assert restarted.exit_code == 0, restarted.output
    assert len(ran) == 4
    summary = json.loads((jobs / "exp1" / "summary.json").read_text())
    assert summary["restarted_after_change"]["changed"] == [
        "extra_instruction='Be brief.' (this run: 'Answer in French.')"
    ]


def test_bench_eval_resume_with_another_env_value_exits_2(
    tmp_path, monkeypatch
) -> None:
    ran = _fake_rollouts(monkeypatch)
    tasks, jobs = _tasks(tmp_path), tmp_path / "jobs"
    first = _bench(
        "eval", "run", "--tasks-dir", str(tasks), "--agent", "oracle",
        "--jobs-dir", str(jobs), "--job-name", "j1", "--agent-env", "MY_FLAG=on",
    )  # fmt: skip
    assert first.exit_code == 0, first.output
    job = jobs / "j1"
    files = _tree(job)

    refused = _bench("eval", "resume", str(job), "--agent-env", "MY_FLAG=off")
    assert refused.exit_code == 2, refused.output
    out = _squash(refused.output)
    assert (
        "Refusing to resume job j1: 2 finished trials in it ran agent_env "
        "MY_FLAG changed"
    ) in out
    assert "next:" in out and "--restart-changed" in out
    assert "MY_FLAG=off" not in out and "MY_FLAG=on" not in out
    assert len(ran) == 2
    assert _tree(job) == files

    # The job's own settings resume it.
    same = _bench("eval", "resume", str(job))
    assert same.exit_code == 0, same.output
    assert len(ran) == 2

    restarted = _bench(
        "eval", "resume", str(job), "--agent-env", "MY_FLAG=off", "--restart-changed"
    )
    assert restarted.exit_code == 0, restarted.output
    assert len(ran) == 4
    (dest,) = (job / SET_ASIDE_DIR).iterdir()
    assert len(load_task_results(dest)) == 2


def test_restart_changed_with_fresh_is_a_usage_error(tmp_path) -> None:
    tasks = _tasks(tmp_path)
    result = _bench(
        "run", str(tasks), "--jobs-dir", str(tmp_path / "jobs"), "--fresh",
        "--restart-changed",
    )  # fmt: skip
    assert result.exit_code == 2, result.output
    assert "--restart-changed restarts a named job's earlier trials" in _squash(
        result.output
    )


def test_a_resume_with_a_changed_task_digest_is_refused(tmp_path) -> None:
    """A trial that recorded another version of its task is one change, and
    restart_changed runs only that task again."""
    old = _first_job(tmp_path, job_name="named")
    trial = next(p for p in old.iterdir() if p.name.startswith("task-0__"))
    config = json.loads((trial / "config.json").read_text())
    config["task_digest"] = "sha256:" + "0" * 64
    (trial / "config.json").write_text(json.dumps(config))
    files = _tree(old)
    error = _refused(_resumed(old))
    assert "task 'task-0' at task_digest sha256:000000000000" in str(error)
    assert error.finished_trials == 1
    assert _tree(old) == files
    assert asyncio.run(_resumed(old, restart_changed=True).run()).ran == 1
