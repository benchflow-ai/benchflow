"""A job resumes only when every setting that changes what a trial measures matches.

``_recorded_run_differs`` used to compare only the agent, the model and the
tasks folder, and on a named job ``_check_resume_mismatch`` refused only an
agent change, so a rerun with a new model, timeout or extra instruction reused
the old trials and published one score over two runs. Now every setting in
``_RESUME_SETTINGS`` is compared: an unnamed rerun that differs starts a new
job, and a named one is refused before anything in the job moves, naming each
setting; with ``restart_changed=True`` it moves the earlier trials to
``<job>/.set-aside/`` (never deleted, never read as the job's trials) instead,
with a line naming the setting.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import yaml

import benchflow as bf
from benchflow._utils.result_paths import (
    SET_ASIDE_DIR,
    iter_task_result_paths,
    load_task_results,
)
from benchflow.budget import Budget
from benchflow.evaluation import (
    _RESUME_SETTINGS,
    Evaluation,
    EvaluationConfig,
    ResumeMismatchError,
    RetryConfig,
    _check_resume_mismatch,
    _recorded_run_differs,
)
from benchflow.models import RolloutResult, RunResult
from benchflow.trial_budget import TrialBudget

SECRET = "sk-test-not-a-real-key-1234567890"


def _tasks(tmp_path: Path, n: int = 2) -> Path:
    tasks_dir = tmp_path / "tasks"
    for i in range(n):
        (tasks_dir / f"task-{i}").mkdir(parents=True, exist_ok=True)
        (tasks_dir / f"task-{i}" / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    return tasks_dir


def _config(**overrides) -> EvaluationConfig:
    base: dict[str, Any] = {
        "agent": "oracle",
        "model": "m1",
        "concurrency": 1,
        "retry": RetryConfig(max_retries=0),
    }
    base.update(overrides)
    return EvaluationConfig(**base)


def _fake_runner(job: Evaluation):
    async def fake_run(task_path, _cfg):
        rollout_dir = job._jobs_dir / job._job_name / f"{task_path.name}__abcd1234"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_name": task_path.name,
            "rollout_name": rollout_dir.name,
            "rewards": {"reward": 1.0},
            "error": None,
            "verifier_error": None,
            "agent": job._config.agent,
            "model": job._config.model,
        }
        (rollout_dir / "result.json").write_text(json.dumps(payload))
        (rollout_dir / "config.json").write_text(
            json.dumps({"agent": job._config.agent, "model": job._config.model})
        )
        return RolloutResult(
            task_name=task_path.name,
            rollout_name=rollout_dir.name,
            rewards={"reward": 1.0},
            agent=job._config.agent,
            rollout_dir=rollout_dir,
        )

    return fake_run


def _evaluation(
    tmp_path: Path, config: EvaluationConfig, job_name=None, *, restart_changed=False
) -> Evaluation:
    job = Evaluation(
        tasks_dir=_tasks(tmp_path),
        jobs_dir=tmp_path / "jobs",
        config=config,
        job_name=job_name,
        restart_changed=restart_changed,
    )
    job._run_single_task = AsyncMock(side_effect=_fake_runner(job))
    return job


def _run(job: Evaluation):
    return asyncio.run(job.run())


def _mark_reviewed(job_dir: Path) -> None:
    """Give the job a trial with a reviewer plan, as a rubric task's trial
    records it in config.json (the reviewer is compared only then)."""
    trial = job_dir / "task-0__review00"
    trial.mkdir(parents=True, exist_ok=True)
    (trial / "config.json").write_text(
        json.dumps({"agent": "oracle", "review": {"rubric_sha256": "0" * 64}})
    )


def _first_job(tmp_path: Path, job_name=None, **overrides) -> Path:
    """Run a two-task job to the end and return its folder."""
    job = _evaluation(tmp_path, _config(**overrides), job_name)
    result = _run(job)
    assert result.ran == 2
    return job.job_dir


# One changed value per setting in _RESUME_SETTINGS, each from the base config.
_CHANGED = {
    "agent": {"agent": "nop", "model": None},
    "model": {"model": "m2"},
    "reasoning_effort": {"reasoning_effort": "high"},
    "harness": {"harness": "native"},
    "codex_apps_policy": {"codex_apps_policy": "disabled"},
    "prompts": {"prompts": [None, "Check your work."]},
    "extra_instruction": {"extra_instruction": "Answer in French."},
    "timeout_multiplier": {"timeout_multiplier": 2.0},
    "agent_idle_timeout": {"agent_idle_timeout": 300},
    "sandbox_setup_timeout": {"sandbox_setup_timeout": 600},
    "budget": {"budget": Budget(max_rollouts=50)},
    "trial_budget": {"trial_budget": TrialBudget(turns=20)},
    "agent_env": {"agent_env": {"MY_BASE_URL": "http://127.0.0.1:9"}},
    "environment": {"environment": "daytona"},
    "sandbox_user": {"sandbox_user": None},
    "sandbox_locked_paths": {"sandbox_locked_paths": ["/opt/answers"]},
    "allow_cloud_metadata": {"allow_cloud_metadata": True},
    "verifier_sandbox": {"verifier_sandbox": "separate"},
    "base_image_override": {"base_image_override": "python:3.12-slim"},
    "separate_verifier": {"separate_verifier": True},
    "separate_verifier_include": {
        "separate_verifier": True,
        "separate_verifier_include": ("/root",),
    },
    "context_root": {"context_root": "/tmp/ctx-a"},
    "skip_agent_install": {"skip_agent_install": True},
    "environment_manifest": None,  # set in the test (needs a manifest object)
    "config_override": {"config_override": {"agent": {"timeout_sec": 5}}},
    "skill_mode": {"skill_mode": "with-skill"},
    "skills_dir": {"skill_mode": "with-skill", "skills_dir": "/tmp/skills-a"},
    "skill_creator_dir": {"skill_creator_dir": "/tmp/creator-a"},
    "self_gen_no_internet": {"self_gen_no_internet": True},
    "job_mode": {"job_mode": "sequential-shared"},
    "loop_strategy": {"loop_strategy": "verify-retry:k=2"},
    "checkpoints": {"checkpoints": "every-prompt"},
    "retry_from_checkpoint": {
        "checkpoints": "every-prompt",
        "retry_from_checkpoint": "on-failure",
    },
    "retry_prompt": {"retry_prompt": "Try again."},
    "retry_resume_session": {"retry_resume_session": True},
    "reviewer": {"reviewer": {"model": "judge-2"}},
    "pre_agent_hooks": None,  # set in the test (a callable)
    "dataset_name": {"dataset_name": "suite-a"},
    "dataset_version": {"dataset_name": "suite-a", "dataset_version": "2.0"},
}


async def _hook(_sandbox) -> None:
    return None


def _changed(setting: str) -> dict:
    if setting == "pre_agent_hooks":
        return {"pre_agent_hooks": [_hook]}
    if setting == "environment_manifest":
        from benchflow.environment.manifest import EnvironmentManifest

        return {
            "environment_manifest": EnvironmentManifest.model_validate(
                {"name": "world-a", "image": "world-a:1"}
            )
        }
    return _CHANGED[setting]


def test_every_compared_setting_has_a_case() -> None:
    assert set(_CHANGED) == set(_RESUME_SETTINGS)


@pytest.mark.parametrize("setting", _RESUME_SETTINGS)
def test_each_recorded_setting_is_compared(tmp_path: Path, setting: str) -> None:
    """Every setting in _RESUME_SETTINGS, changed alone, is named as a difference."""
    tasks_dir = _tasks(tmp_path)
    job = Evaluation(tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=_config())
    job._write_evaluation_record()
    _mark_reviewed(job.job_dir)
    differs = _recorded_run_differs(
        job.job_dir, _config(**_changed(setting)), tasks_dir
    )
    assert differs is not None and setting in differs
    assert _recorded_run_differs(job.job_dir, _config(), tasks_dir) is None


# --- the settings the Done-when names: one test each, unnamed and named ----


def _assert_new_job(tmp_path: Path, old: Path, caplog, words: str, **changed) -> None:
    """An unnamed rerun with ``changed`` starts a new job naming ``words``."""
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        job = _evaluation(tmp_path, _config(**changed))
    assert job.job_dir != old
    assert f"Starting a new job {job._job_name}" in caplog.text
    assert words in caplog.text
    assert old.name in caplog.text
    # The old job is untouched.
    assert len(load_task_results(old)) == 2
    assert not (old / SET_ASIDE_DIR).exists()


def _tree(folder: Path) -> dict[str, bytes]:
    """Every file under ``folder``, by its relative path, with its bytes."""
    return {
        str(p.relative_to(folder)): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def _assert_refused_then_set_aside(
    tmp_path: Path, caplog, words: str, **changed
) -> None:
    """A named rerun with ``changed`` is refused, naming ``words``, and moves
    nothing; with restart_changed it sets the earlier trials aside."""
    old = _first_job(tmp_path, job_name="named")
    (old / "results.jsonl").write_text("{}\n")  # the job-level aggregate
    (old / "results.errors.json").write_text("[]\n")  # and its skipped rows
    before = sorted(p.name for p in old.iterdir() if p.is_dir())
    files = _tree(old)
    with pytest.raises(ResumeMismatchError) as refused:
        _check_resume_mismatch(old, _config(**changed))
    assert words in str(refused.value)
    assert "2 finished trials in it ran" in str(refused.value)
    assert (refused.value.exit_code, refused.value.finished_trials) == (2, 2)
    assert "--restart-changed" in (refused.value.hint or "")
    assert _tree(old) == files
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        dest = _check_resume_mismatch(old, _config(**changed), restart_changed=True)
    assert dest is not None and dest.parent == old / SET_ASIDE_DIR
    assert sorted(p.name for p in dest.iterdir() if p.is_dir()) == before
    assert not [p for p in old.iterdir() if p.is_dir() and p.name != SET_ASIDE_DIR]
    record = json.loads((dest / "set-aside.json").read_text())
    assert words in record["reason"]
    assert {"summary.json", "results.jsonl", "results.errors.json"} <= set(
        record["moved"]
    )
    assert not (old / "summary.json").exists()
    assert not (old / "results.jsonl").exists()
    assert not (old / "results.errors.json").exists()
    assert (dest / "evaluation.json").is_file()
    assert "Setting aside 2 earlier trial(s) of job named" in caplog.text
    assert words in caplog.text
    # Nothing is deleted, and the job's readers no longer see the trials.
    assert len(load_task_results(dest)) == 2
    assert load_task_results(old) == {}


def test_model_change_starts_a_new_job(tmp_path, caplog) -> None:
    old = _first_job(tmp_path)
    _assert_new_job(tmp_path, old, caplog, "model='m1' (this run: 'm2')", model="m2")


def test_model_change_refuses_a_named_job(tmp_path, caplog) -> None:
    _assert_refused_then_set_aside(
        tmp_path, caplog, "model='m1' (this run: 'm2')", model="m2"
    )


@pytest.mark.parametrize(
    ("changed", "words"),
    [
        ({"timeout_multiplier": 2.0}, "timeout_multiplier=None (this run: 2.0)"),
        ({"agent_idle_timeout": 300}, "agent_idle_timeout=600 (this run: 300)"),
        ({"sandbox_setup_timeout": 600}, "sandbox_setup_timeout=120 (this run: 600)"),
    ],
)
def test_timeout_change_starts_a_new_job(tmp_path, caplog, changed, words) -> None:
    old = _first_job(tmp_path)
    _assert_new_job(tmp_path, old, caplog, words, **changed)


@pytest.mark.parametrize(
    ("changed", "words"),
    [
        ({"timeout_multiplier": 2.0}, "timeout_multiplier=None (this run: 2.0)"),
        ({"agent_idle_timeout": 300}, "agent_idle_timeout=600 (this run: 300)"),
        ({"sandbox_setup_timeout": 600}, "sandbox_setup_timeout=120 (this run: 600)"),
    ],
)
def test_timeout_change_refuses_a_named_job(tmp_path, caplog, changed, words) -> None:
    _assert_refused_then_set_aside(tmp_path, caplog, words, **changed)


def test_extra_instruction_change_starts_a_new_job(tmp_path, caplog) -> None:
    old = _first_job(tmp_path, extra_instruction="Be brief.")
    _assert_new_job(
        tmp_path,
        old,
        caplog,
        "extra_instruction='Be brief.' (this run: 'Answer in French.')",
        extra_instruction="Answer in French.",
    )


def test_extra_instruction_change_refuses_a_named_job(tmp_path, caplog) -> None:
    _assert_refused_then_set_aside(
        tmp_path,
        caplog,
        "extra_instruction=None (this run: 'Answer in French.')",
        extra_instruction="Answer in French.",
    )


def test_budget_change_starts_a_new_job(tmp_path, caplog) -> None:
    old = _first_job(tmp_path, budget=Budget(max_rollouts=50))
    _assert_new_job(
        tmp_path,
        old,
        caplog,
        "budget.max_rollouts=50 (this run: 10)",
        budget=Budget(max_rollouts=10),
    )


def test_budget_change_refuses_a_named_job(tmp_path, caplog) -> None:
    _assert_refused_then_set_aside(
        tmp_path,
        caplog,
        "budget=None (this run: {'max_cost_usd': 5.0})",
        budget=Budget(max_cost_usd=5),
    )


def test_agent_env_change_starts_a_new_job(tmp_path, caplog) -> None:
    old = _first_job(tmp_path, agent_env={"MY_BASE_URL": "http://127.0.0.1:8"})
    _assert_new_job(
        tmp_path,
        old,
        caplog,
        "agent_env MY_BASE_URL changed",
        agent_env={"MY_BASE_URL": "http://127.0.0.1:9"},
    )
    # The line names the variable, never its value.
    assert "127.0.0.1:9" not in caplog.text


def test_agent_env_change_refuses_a_named_job(tmp_path, caplog) -> None:
    _assert_refused_then_set_aside(
        tmp_path,
        caplog,
        "agent_env MY_FLAG added",
        agent_env={"MY_FLAG": "on"},
    )


def test_sandbox_change_starts_a_new_job(tmp_path, caplog) -> None:
    old = _first_job(tmp_path)
    _assert_new_job(
        tmp_path,
        old,
        caplog,
        "environment='docker' (this run: 'daytona')",
        environment="daytona",
    )


def test_sandbox_change_refuses_a_named_job(tmp_path, caplog) -> None:
    _assert_refused_then_set_aside(
        tmp_path,
        caplog,
        "environment='docker' (this run: 'daytona')",
        environment="daytona",
    )


# --- what still resumes -----------------------------------------------------


def test_an_unchanged_unnamed_rerun_still_resumes(tmp_path) -> None:
    old = _first_job(tmp_path)
    job = _evaluation(tmp_path, _config())
    assert job.job_dir == old
    result = _run(job)
    assert (result.reused, result.ran) == (2, 0)
    assert not (old / SET_ASIDE_DIR).exists()


def test_an_unchanged_named_rerun_still_resumes(tmp_path) -> None:
    old = _first_job(tmp_path, job_name="named")
    job = _evaluation(tmp_path, _config(), job_name="named")
    result = _run(job)
    assert (result.reused, result.ran) == (2, 0)
    assert not (old / SET_ASIDE_DIR).exists()


def test_settings_that_do_not_change_a_trial_still_resume(tmp_path) -> None:
    """Concurrency, retries and a rotated key are not compared, and the
    selection is checked by run() (a narrower one: test below)."""
    old = _first_job(tmp_path, agent_env={"OPENAI_API_KEY": SECRET})
    job = _evaluation(
        tmp_path,
        _config(
            concurrency=8,
            include_tasks={"task-0"},
            retry=RetryConfig(max_retries=3),
            freeze_workspace=True,
            agent_env={"OPENAI_API_KEY": "sk-test-rotated-not-real-0987654321"},
        ),
    )
    assert job.job_dir == old


def test_more_seeds_is_a_wider_selection_not_a_change(tmp_path) -> None:
    """Each seed runs as its own task (<task>--seed-<n>), so a rerun that adds
    seeds resumes the seeds that ran and runs the new ones, as a wider
    --include does."""
    tasks_dir = _tasks(tmp_path)
    job = Evaluation(
        tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=_config(seeds=[0, 1])
    )
    job._write_evaluation_record()
    for seeds in ([0, 1, 2], None):
        differs = _recorded_run_differs(job.job_dir, _config(seeds=seeds), tasks_dir)
        assert differs is None


def _selected_job(tmp_path: Path, include: set[str], job_name=None) -> Path:
    """Run the job on ``include`` alone and return its folder."""
    job = _evaluation(tmp_path, _config(include_tasks=include), job_name)
    assert _run(job).ran == len(include)
    return job.job_dir


def test_another_selection_starts_a_new_job(tmp_path, caplog) -> None:
    """DFR-p5-4: `-i fix-git` then `-i regex-log` in one --jobs-dir scored 2/2.
    The second run now scores only the task it selected, in a new job."""
    old = _selected_job(tmp_path, {"task-0"})
    job = _evaluation(tmp_path, _config(include_tasks={"task-1"}))
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        result = _run(job)
    assert job.job_dir != old
    assert (result.reused, result.ran, result.passed, result.total) == (0, 1, 1, 1)
    assert f"Starting a new job {job._job_name}" in caplog.text
    assert "holds results of 1 task(s) this run does not select (task-0)" in caplog.text
    assert "--include, --exclude, --limit" in caplog.text
    assert "Resuming into existing job directory" not in caplog.text
    assert set(load_task_results(old)) == {"task-0"}
    assert set(load_task_results(job.job_dir)) == {"task-1"}


def test_a_narrower_selection_starts_a_new_job(tmp_path) -> None:
    old = _first_job(tmp_path)
    job = _evaluation(tmp_path, _config(include_tasks={"task-1"}))
    result = _run(job)
    assert job.job_dir != old
    assert (result.reused, result.ran, result.total) == (0, 1, 1)


def test_the_same_or_a_wider_selection_resumes(tmp_path, caplog) -> None:
    """The job's results are all selected again, so its score is the selection's."""
    old = _selected_job(tmp_path, {"task-0"})
    same = _evaluation(tmp_path, _config(include_tasks={"task-0"}))
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        assert (_run(same).reused, same.job_dir) == (1, old)
    assert f"Resuming into existing job directory: {old.name}" in caplog.text
    wider = _evaluation(tmp_path, _config())
    result = _run(wider)
    assert wider.job_dir == old
    assert (result.reused, result.ran, result.total) == (1, 1, 2)


def test_a_named_job_with_another_selection_warns(tmp_path, caplog) -> None:
    """A named job keeps the tasks it holds (behaviour unchanged) and says so."""
    old = _selected_job(tmp_path, {"task-0"}, job_name="named")
    job = _evaluation(tmp_path, _config(include_tasks={"task-1"}), job_name="named")
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        result = _run(job)
    assert job.job_dir == old
    assert (result.reused, result.ran) == (1, 1)
    assert (
        "Job named also holds results of 1 task(s) this run does not select "
        "(task-0); its score counts them" in caplog.text
    )


def test_eval_resume_of_a_selection_is_unchanged(tmp_path, caplog) -> None:
    old = _selected_job(tmp_path, {"task-0"})
    job = Evaluation.resume(old)
    job._run_single_task = AsyncMock(side_effect=_fake_runner(job))
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        result = _run(job)
    assert (job.job_dir, result.reused, result.ran) == (old, 1, 0)
    assert "does not select" not in caplog.text


def test_a_record_without_a_setting_does_not_count_it_as_changed(tmp_path) -> None:
    """An older evaluation.json lacks newer keys: unknown, not different."""
    tasks_dir = _tasks(tmp_path)
    job_dir = tmp_path / "jobs" / "old"
    job_dir.mkdir(parents=True)
    (job_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "tasks_dir": str(tasks_dir),
                "config": {"agent": "oracle", "model": "m1"},
            }
        )
    )
    assert (
        _recorded_run_differs(job_dir, _config(timeout_multiplier=3), tasks_dir) is None
    )
    differs = _recorded_run_differs(job_dir, _config(model="m2"), tasks_dir)
    assert differs == "model='m1' (this run: 'm2')"


def test_a_raised_cap_on_a_budget_stopped_job_resumes(tmp_path) -> None:
    """A rerun with a bigger cap is how a budget-stopped job runs the rest."""
    old = _first_job(tmp_path, budget=Budget(max_rollouts=2))
    summary = json.loads((old / "summary.json").read_text())
    summary["budget"] = {**summary.get("budget", {}), "stopped": True}
    (old / "summary.json").write_text(json.dumps(summary))
    assert _evaluation(tmp_path, _config(budget=Budget(max_rollouts=10))).job_dir == old
    assert _evaluation(tmp_path, _config(budget=None)).job_dir == old
    # A lower cap, or a new kind of cap, is still a change.
    assert _evaluation(tmp_path, _config(budget=Budget(max_rollouts=1))).job_dir != old
    assert (
        _evaluation(
            tmp_path, _config(budget=Budget(max_rollouts=10, max_tokens=5))
        ).job_dir
        != old
    )


def test_a_raised_cap_on_a_job_the_budget_did_not_stop_is_a_change(tmp_path) -> None:
    old = _first_job(tmp_path, budget=Budget(max_rollouts=2))
    assert _evaluation(tmp_path, _config(budget=Budget(max_rollouts=10))).job_dir != old


# --- the whole run ------------------------------------------------------------


def test_a_named_rerun_with_another_model_runs_every_task_again(tmp_path) -> None:
    old = _first_job(tmp_path, job_name="named")
    job = _evaluation(
        tmp_path, _config(model="m2"), job_name="named", restart_changed=True
    )
    result = _run(job)
    assert (result.reused, result.ran, result.passed) == (0, 2, 2)
    loaded = bf.load_job(old)
    assert len(loaded.trials) == 2
    assert not loaded.interrupted
    assert {r["model"] for r in load_task_results(old).values()} == {"m2"}
    summary = json.loads((old / "summary.json").read_text())
    assert summary["total"] == 2
    record = json.loads((old / "evaluation.json").read_text())
    assert record["config"]["model"] == "m2"
    (dest,) = (old / SET_ASIDE_DIR).iterdir()
    old_record = json.loads((dest / "evaluation.json").read_text())
    assert old_record["config"]["model"] == "m1"


def test_a_setting_changed_after_construction_still_starts_a_new_job(
    tmp_path, caplog
) -> None:
    """The CLI's --config path sets flags after Evaluation() picked the name."""
    old = _first_job(tmp_path)
    job = _evaluation(tmp_path, _config())
    assert job.job_dir == old
    job._config.extra_instruction = "Answer in French."
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        result = _run(job)
    assert job.job_dir != old
    assert (result.reused, result.ran) == (0, 2)
    assert "extra_instruction=None (this run: 'Answer in French.')" in caplog.text
    assert len(load_task_results(old)) == 2
    assert not (old / SET_ASIDE_DIR).exists()


def test_a_setting_changed_after_construction_to_match_resumes(
    tmp_path, caplog
) -> None:
    """The other way round: Evaluation() saw the YAML's settings, which differ
    from the latest job, and the CLI's --config path then set the flag
    (--agent-env here) that makes them match it. That unchanged rerun resumes
    the job instead of starting a new one each time."""
    old = _first_job(tmp_path, agent_env={"MY_FLAG": "on"})
    job = _evaluation(tmp_path, _config())
    assert job.job_dir != old
    job._config.agent_env = {"MY_FLAG": "on"}
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        result = _run(job)
    assert job.job_dir == old
    assert (result.reused, result.ran) == (2, 0)
    assert [p.name for p in (tmp_path / "jobs").iterdir() if p.is_dir()] == [old.name]
    assert f"Resuming into existing job directory: {old.name}" in caplog.text
    assert not (old / SET_ASIDE_DIR).exists()


def test_a_trial_of_another_agent_is_set_aside_without_a_record(tmp_path) -> None:
    """A job from before evaluation.json is compared by its trials' config.json."""
    job_dir = tmp_path / "jobs" / "legacy"
    for name, agent in (("task-0__1", "oracle"), ("task-1__1", "codex-acp")):
        (job_dir / name).mkdir(parents=True)
        (job_dir / name / "config.json").write_text(json.dumps({"agent": agent}))
        (job_dir / name / "result.json").write_text(
            json.dumps(
                {
                    "task_name": name[:6],
                    "rollout_name": name,
                    "rewards": {"reward": 1.0},
                }
            )
        )
    with pytest.raises(ResumeMismatchError) as refused:
        _check_resume_mismatch(job_dir, _config())
    assert "1 finished trial in it ran agent='codex-acp'" in str(refused.value)
    # No evaluation.json: `bench eval resume` cannot rebuild it.
    assert "the command that started it" in (refused.value.hint or "")
    dest = _check_resume_mismatch(job_dir, _config(), restart_changed=True)
    assert dest is not None
    assert [p.name for p in dest.iterdir() if p.is_dir()] == ["task-1__1"]
    assert (job_dir / "task-0__1").is_dir()
    record = json.loads((dest / "set-aside.json").read_text())
    assert record["reason"] == "agent='codex-acp' (this run: 'oracle')"


def test_a_failed_move_refuses_instead_of_blending(tmp_path, monkeypatch) -> None:
    old = _first_job(tmp_path, job_name="named")

    def no_rename(self, target):
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "rename", no_rename)
    with pytest.raises(
        ResumeMismatchError, match="model='m1' \\(this run: 'm2'\\)"
    ) as e:
        _check_resume_mismatch(old, _config(model="m2"), restart_changed=True)
    assert "setting them aside" in str(e.value) and e.value.exit_code == 1


# --- readers and resume -----------------------------------------------------


def test_readers_skip_set_aside_trials(tmp_path) -> None:
    old = _first_job(tmp_path, job_name="named")
    attempt = old / "task-1__dead0000"
    attempt.mkdir()
    (attempt / "config.json").write_text(json.dumps({"agent": "oracle"}))
    dest = _check_resume_mismatch(old, _config(model="m2"), restart_changed=True)
    assert dest is not None and (dest / "task-1__dead0000").is_dir()
    assert iter_task_result_paths(old) == []
    from benchflow.jobs import _interrupted

    assert _interrupted(old) == []
    # A reader pointed at the set-aside folder itself reads it.
    assert len(iter_task_result_paths(dest)) == 2
    assert _interrupted(dest) == [dest / "task-1__dead0000"]


def test_load_job_does_not_read_a_set_aside_results_file(tmp_path) -> None:
    """A job started over has no trial yet: load_job's results.jsonl fallback
    must not read the set-aside run's file as this job's trials."""
    from benchflow.jobs import load_job

    old = _first_job(tmp_path, job_name="named")
    (old / "results.jsonl").write_text(
        "".join(
            json.dumps({"reward": 1.0, "info": {"task_name": f"task-{i}"}}) + "\n"
            for i in range(2)
        )
    )
    dest = _check_resume_mismatch(old, _config(model="m2"), restart_changed=True)
    assert dest is not None and (dest / "results.jsonl").is_file()
    with pytest.raises(FileNotFoundError):
        load_job(old)
    # Pointed at the set-aside folder itself, it reads the old run.
    assert len(load_job(dest).trials) == 2


def test_resume_restores_recorded_env_and_ignores_a_missing_key(tmp_path) -> None:
    """bench eval resume passes no env: the recorded values come back, a
    secret one (never stored) is not a change, so the job resumes."""
    old = _first_job(
        tmp_path,
        job_name="named",
        agent_env={"MY_BASE_URL": "http://127.0.0.1:8", "OPENAI_API_KEY": SECRET},
    )
    text = (old / "evaluation.json").read_text()
    assert SECRET not in text
    resumed = Evaluation.resume(old)
    assert resumed._config.agent_env == {"MY_BASE_URL": "http://127.0.0.1:8"}
    resumed._run_single_task = AsyncMock(side_effect=_fake_runner(resumed))
    result = _run(resumed)
    assert (result.reused, result.ran) == (2, 0)
    assert not (old / SET_ASIDE_DIR).exists()


def test_resume_with_a_changed_setting_is_refused(tmp_path) -> None:
    old = _first_job(tmp_path, job_name="named")
    files = _tree(old)
    resumed = Evaluation.resume(old, timeout_multiplier=2.0)
    resumed._run_single_task = AsyncMock(side_effect=_fake_runner(resumed))
    with pytest.raises(ResumeMismatchError, match="timeout_multiplier=None"):
        _run(resumed)
    resumed._run_single_task.assert_not_called()
    assert _tree(old) == files


def test_resume_with_a_changed_setting_and_restart_changed_sets_the_trials_aside(
    tmp_path,
) -> None:
    old = _first_job(tmp_path, job_name="named")
    resumed = Evaluation.resume(old, timeout_multiplier=2.0, restart_changed=True)
    resumed._run_single_task = AsyncMock(side_effect=_fake_runner(resumed))
    result = _run(resumed)
    assert (result.reused, result.ran) == (0, 2)
    (dest,) = (old / SET_ASIDE_DIR).iterdir()
    assert (
        "timeout_multiplier=None (this run: 2.0)"
        in json.loads((dest / "set-aside.json").read_text())["reason"]
    )


# --- the run's own checks come before any trial moves ------------------------


def _review_a_trial(job_dir: Path) -> None:
    """Give one finished trial the reviewer plan a rubric task's trial
    records in its config.json (the reviewer is compared only then)."""
    trial = sorted(p for p in job_dir.iterdir() if p.is_dir() and p.name[0] != ".")[0]
    path = trial / "config.json"
    plan = {"review": {"rubric_sha256": "0" * 64}}
    path.write_text(json.dumps({**json.loads(path.read_text()), **plan}))


def test_a_refused_reviewer_leaves_the_trials_for_the_old_settings(
    tmp_path, monkeypatch
) -> None:
    """A restarted rerun with another reviewer model whose key is missing is
    refused before any trial moves, so rerunning with the old settings reuses
    them."""
    import benchflow.review.automatic as automatic

    old = _first_job(tmp_path, job_name="named", reviewer={"model": "judge-1"})
    _review_a_trial(old)
    real = automatic.prepare_review

    def no_key_for_judge_2(task_path, reviewer):
        if reviewer.model == "judge-2":
            raise ValueError("Reviewer preflight: no key for judge-2")
        return real(task_path, reviewer)

    monkeypatch.setattr(automatic, "prepare_review", no_key_for_judge_2)
    judge_2 = _config(reviewer={"model": "judge-2"})
    with pytest.raises(ValueError, match="no key for judge-2"):
        _run(_evaluation(tmp_path, judge_2, "named", restart_changed=True))
    assert not (old / SET_ASIDE_DIR).exists()
    assert len(load_task_results(old)) == 2
    record = json.loads((old / "evaluation.json").read_text())["config"]
    assert record["reviewer"]["model"] == "judge-1"

    again = _evaluation(tmp_path, _config(reviewer={"model": "judge-1"}), "named")
    result = _run(again)
    assert (result.reused, result.ran) == (2, 0)
    assert again._run_single_task.await_count == 0


def test_the_record_follows_a_set_aside_when_the_run_then_stops(
    tmp_path, monkeypatch
) -> None:
    """Once the trials are set aside, evaluation.json names this run's
    settings, so the record always describes what the job folder holds."""
    import benchflow.review.resume as resume

    old = _first_job(tmp_path, job_name="named")
    stop = AsyncMock(side_effect=RuntimeError("stopped"))
    monkeypatch.setattr(resume, "resume_pending_reviews", stop)
    with pytest.raises(RuntimeError, match="stopped"):
        _run(_evaluation(tmp_path, _config(model="m2"), "named", restart_changed=True))
    assert (old / SET_ASIDE_DIR).is_dir()
    record = json.loads((old / "evaluation.json").read_text())["config"]
    assert record["model"] == "m2"
    # An unnamed rerun with the old settings no longer picks the emptied job.
    assert _evaluation(tmp_path, _config()).job_dir != old


def _sequential_job(
    tmp_path: Path, model: str, *, restart_changed: bool = False
) -> Evaluation:
    job = Evaluation(
        tasks_dir=_tasks(tmp_path),
        jobs_dir=tmp_path / "jobs",
        config=_config(job_mode="sequential-shared", model=model),
        job_name="named",
        restart_changed=restart_changed,
    )

    async def fake_run(task_path, *args, **kwargs):
        rollout_dir = job.job_dir / f"{task_path.name}__abcd1234"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        (rollout_dir / "result.json").write_text(
            json.dumps({"task_name": task_path.name, "rewards": {"reward": 1.0}})
        )
        return RunResult(
            task_name=task_path.name,
            rewards={"reward": 1.0},
            evolved_skills={"k": "v"},
        )

    job._run_task = fake_run  # type: ignore[method-assign]
    return job


def test_a_sequential_shared_job_sets_its_learner_store_aside(tmp_path) -> None:
    """The store holds every earlier trial's lessons: it moves with them and
    the rerun learns from generation 0 again."""
    _run(_sequential_job(tmp_path, "m1"))
    old = tmp_path / "jobs" / "named"
    assert json.loads((old / "learner_store.json").read_text())["generation"] == 2
    job = _sequential_job(tmp_path, "m2", restart_changed=True)
    assert job.learner_store is not None and job.learner_store.generation == 2
    result = _run(job)
    assert (result.reused, result.ran) == (0, 2)
    (dest,) = (old / SET_ASIDE_DIR).iterdir()
    assert json.loads((dest / "learner_store.json").read_text())["generation"] == 2
    assert json.loads((old / "learner_store.json").read_text())["generation"] == 2
    assert job.learner_store.generation == 2


def test_the_record_keeps_only_values_that_do_not_look_secret(tmp_path) -> None:
    job = Evaluation(
        tasks_dir=_tasks(tmp_path),
        jobs_dir=tmp_path / "jobs",
        config=_config(agent_env={"OPENAI_API_KEY": SECRET, "MY_FLAG": "on"}),
    )
    job._write_evaluation_record()
    record = json.loads((job.job_dir / "evaluation.json").read_text())
    assert record["config"]["agent_env"] == {"MY_FLAG": "on"}
    assert record["config"]["agent_env_keys"] == ["MY_FLAG", "OPENAI_API_KEY"]
    assert SECRET not in json.dumps(record)


def test_dataclass_fields_are_all_sorted_into_compared_or_not() -> None:
    """A new EvaluationConfig field must be placed: compared, or listed as not."""
    not_compared = {
        "concurrency",
        "build_concurrency",
        "warm_pool",
        "retry",
        "exclude_tasks",
        "include_tasks",
        "n_tasks",
        "sample_seed",
        "seeds",
        "source_provenance",
        "dataset_task_digests",
        "usage_tracking",
        "checkpoint_keep",
        "freeze_workspace",
        "keep_sandbox",
        "accept_deviations",
        "planned_trials",
        "trial_number",
        "model_given",
        "agent_idle_timeout_given",
        "integrity",
        "agent_defaulted",
        "fresh_install",
    }
    fields = {f.name for f in dataclasses.fields(EvaluationConfig)}
    unsorted = fields - not_compared - set(_RESUME_SETTINGS)
    assert not unsorted, (
        f"EvaluationConfig field(s) {sorted(unsorted)}: add each to "
        "evaluation._RESUME_SETTINGS if it changes what a trial measures, "
        "else to not_compared here (and to the comment above _RESUME_SETTINGS)"
    )
    assert set(_RESUME_SETTINGS) <= fields


def test_a_reviewer_key_a_resume_cannot_pass_back_is_not_a_change(tmp_path) -> None:
    """The reviewer's secret-looking values are never stored either."""
    tasks_dir = _tasks(tmp_path)
    job = Evaluation(
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
        config=_config(
            reviewer={"model": "judge-1", "agent_env": {"AZURE_API_KEY": SECRET}}
        ),
        job_name="named",
    )
    job._write_evaluation_record()
    _mark_reviewed(job.job_dir)
    resumed = Evaluation.resume(job.job_dir)
    assert resumed._config.reviewer.agent_env == {}
    assert _recorded_run_differs(job.job_dir, resumed._config, tasks_dir) is None
    other = dataclasses.replace(
        resumed._config, reviewer={"model": "judge-2", "agent_env": {}}
    )
    differs = _recorded_run_differs(job.job_dir, other, tasks_dir)
    assert differs == "reviewer.model='judge-1' (this run: 'judge-2')"


def test_a_field_an_older_record_lacks_is_not_a_change(tmp_path) -> None:
    """An upgrade that adds a reviewer or budget field must not set aside
    every earlier trial on the next resume."""
    tasks_dir = _tasks(tmp_path)
    job = Evaluation(
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
        config=_config(),
        job_name="named",
    )
    job._write_evaluation_record()
    path = job.job_dir / "evaluation.json"
    record = json.loads(path.read_text())
    del record["config"]["reviewer"]["open_network"]
    path.write_text(json.dumps(record))
    assert _recorded_run_differs(job.job_dir, _config(), tasks_dir) is None


def test_a_named_job_resumes_from_a_moved_tasks_folder(tmp_path) -> None:
    """Naming the job is how a moved tasks folder is resumed; an unnamed
    rerun on another tasks folder still starts a new job."""
    old = _first_job(tmp_path, job_name="named")
    moved = tmp_path / "moved"
    (tmp_path / "tasks").rename(moved)
    unnamed = Evaluation(tasks_dir=moved, jobs_dir=tmp_path / "jobs", config=_config())
    assert unnamed.job_dir != old
    resumed = Evaluation.resume(old, tasks_dir=moved)
    resumed._run_single_task = AsyncMock(side_effect=_fake_runner(resumed))
    result = _run(resumed)
    assert (result.reused, result.ran) == (2, 0)
    assert not (old / SET_ASIDE_DIR).exists()


# --- what the comparison keeps out of logs and leaves out ---------------------

OLD_HF = "hf_" + "a1B2c3D4" * 4
NEW_HF = "hf_" + "z9Y8x7W6" * 4


def _override(token: str, flag: str) -> dict:
    """A config_override with a key under a secret name, one under a harmless
    name (token-shaped value) and a plain value."""
    return {"sandbox": {"env": {"HF_TOKEN": token, "HUB": token, "MY_FLAG": flag}}}


def _refusal(job_dir: Path, config: EvaluationConfig) -> str:
    """The refusal to resume ``job_dir`` with ``config``: message and next step."""
    with pytest.raises(ResumeMismatchError) as refused:
        _check_resume_mismatch(job_dir, config)
    return f"{refused.value}\n{refused.value.hint}"


def test_config_override_values_stay_out_of_logs_and_set_aside_json(
    tmp_path, caplog
) -> None:
    old = _first_job(
        tmp_path, job_name="named", config_override=_override(OLD_HF, "flag-one")
    )
    changed = _config(config_override=_override(NEW_HF, "flag-two"))
    refusal = _refusal(old, changed)
    assert "config_override sandbox.env.MY_FLAG changed" in refusal
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        dest = _check_resume_mismatch(old, changed, restart_changed=True)
    assert dest is not None
    text = (dest / "set-aside.json").read_text()
    reason = json.loads(text)["reason"]
    assert reason == "config_override sandbox.env.MY_FLAG changed"
    assert reason in caplog.text
    for value in (OLD_HF, NEW_HF, "flag-one", "flag-two"):
        assert value not in caplog.text
        assert value not in text
        assert value not in refusal


def test_a_rotated_key_inside_config_override_is_not_a_change(tmp_path) -> None:
    """As for agent_env: a string that looks secret, by its name or its shape,
    is not compared, so rotating it resumes the job."""
    old = _first_job(
        tmp_path, job_name="named", config_override=_override(OLD_HF, "flag-one")
    )
    rotated = _config(config_override=_override(NEW_HF, "flag-one"))
    assert _check_resume_mismatch(old, rotated) is None
    assert _evaluation(tmp_path, rotated).job_dir == old


def test_a_token_shaped_value_of_another_setting_is_not_printed(tmp_path) -> None:
    tasks_dir = _tasks(tmp_path)
    job = Evaluation(
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
        config=_config(extra_instruction=f"Use {OLD_HF}."),
    )
    job._write_evaluation_record()
    differs = _recorded_run_differs(
        job.job_dir, _config(extra_instruction=f"Use {NEW_HF}."), tasks_dir
    )
    assert differs == "extra_instruction changed"


# --- a secret-looking string outside an env mapping is compared by digest ----

_LOGIN_FIX = "Log in with password: admin123, then fix the login bug."
_LOGIN_LIMIT = "Log in with password: admin123, then add rate limiting."


def test_a_prompt_with_a_password_in_it_is_compared(tmp_path, caplog) -> None:
    """A login task's prompt often holds a password: changing the rest of the
    prompt is still a change, named without the prompt's text."""
    old = _first_job(tmp_path, prompts=[_LOGIN_FIX])
    _assert_new_job(tmp_path, old, caplog, "prompts changed", prompts=[_LOGIN_LIMIT])
    assert "admin123" not in caplog.text


def test_a_prompt_with_a_password_in_it_refuses_a_named_job(tmp_path, caplog) -> None:
    old = _first_job(tmp_path, job_name="named", prompts=[_LOGIN_FIX])
    assert _check_resume_mismatch(old, _config(prompts=[_LOGIN_FIX])) is None
    refusal = _refusal(old, _config(prompts=[_LOGIN_LIMIT]))
    assert "ran prompts changed" in refusal
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        dest = _check_resume_mismatch(
            old, _config(prompts=[_LOGIN_LIMIT]), restart_changed=True
        )
    assert dest is not None
    text = (dest / "set-aside.json").read_text()
    assert json.loads(text)["reason"] == "prompts changed"
    assert "admin123" not in caplog.text
    assert "admin123" not in text
    assert "admin123" not in refusal


def test_removing_a_prompt_with_a_token_in_it_is_a_change(tmp_path) -> None:
    tasks_dir = _tasks(tmp_path)
    both = ["Use token=abc123 for the API.", "Check your work."]
    job = Evaluation(
        tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=_config(prompts=both)
    )
    job._write_evaluation_record()
    assert _recorded_run_differs(job.job_dir, _config(prompts=both), tasks_dir) is None
    differs = _recorded_run_differs(
        job.job_dir, _config(prompts=["Check your work."]), tasks_dir
    )
    assert differs == "prompts changed"


def test_a_password_in_a_config_override_field_is_compared(tmp_path) -> None:
    """Outside an env or headers mapping a field's name is no credential's:
    a secret-looking value there is compared, and named by its path only."""
    tasks_dir = _tasks(tmp_path)
    job = Evaluation(
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
        config=_config(config_override={"agent": {"note": _LOGIN_FIX}}),
    )
    job._write_evaluation_record()
    differs = _recorded_run_differs(
        job.job_dir,
        _config(config_override={"agent": {"note": _LOGIN_LIMIT}}),
        tasks_dir,
    )
    assert differs == "config_override agent.note changed"


def test_reviewer_concurrency_alone_still_resumes(tmp_path) -> None:
    """How many reviews run at once changes no score, even in a reviewed job."""
    old = _first_job(tmp_path, job_name="named", reviewer={"model": "judge-1"})
    _mark_reviewed(old)
    faster = _config(reviewer={"model": "judge-1", "concurrency": 8})
    assert _evaluation(tmp_path, faster).job_dir == old
    assert _check_resume_mismatch(old, faster) is None
    assert not (old / SET_ASIDE_DIR).exists()


def test_the_reviewer_counts_only_for_a_job_with_a_reviewed_trial(tmp_path) -> None:
    """The reviewer block (with its defaults: image, timeouts) is recorded for
    every job; a change, or an upgrade that bumps a default, sets nothing
    aside in a job none of whose trials was reviewed."""
    old = _first_job(tmp_path, job_name="named")
    changed = _config(
        reviewer={"model": "judge-2", "image": "python:3.13-slim", "timeout_sec": 60}
    )
    assert _check_resume_mismatch(old, changed) is None
    _mark_reviewed(old)
    assert "reviewer.model=None (this run: 'judge-2')" in _refusal(old, changed)
    dest = _check_resume_mismatch(old, changed, restart_changed=True)
    assert dest is not None
    reason = json.loads((dest / "set-aside.json").read_text())["reason"]
    assert "reviewer.model=None (this run: 'judge-2')" in reason
    assert "reviewer.image=" in reason


# --- a secret-looking name with a value that is no key is still compared -----


def test_a_token_count_in_agent_env_starts_a_new_job(tmp_path, caplog) -> None:
    """MAX_THINKING_TOKENS names a count, not a key: 0 then 32000 is two runs."""
    old = _first_job(tmp_path, agent_env={"MAX_THINKING_TOKENS": "0"})
    _assert_new_job(
        tmp_path,
        old,
        caplog,
        "agent_env MAX_THINKING_TOKENS changed",
        agent_env={"MAX_THINKING_TOKENS": "32000"},
    )
    assert "32000" not in caplog.text


def test_a_token_count_in_agent_env_refuses_a_named_job(tmp_path, caplog) -> None:
    old = _first_job(
        tmp_path, job_name="named", agent_env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "1024"}
    )
    changed = _config(agent_env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000"})
    refusal = _refusal(old, changed)
    assert "agent_env CLAUDE_CODE_MAX_OUTPUT_TOKENS changed" in refusal
    assert "32000" not in refusal and "1024" not in refusal
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        dest = _check_resume_mismatch(old, changed, restart_changed=True)
    assert dest is not None
    reason = json.loads((dest / "set-aside.json").read_text())["reason"]
    assert reason == "agent_env CLAUDE_CODE_MAX_OUTPUT_TOKENS changed"


def test_the_record_keeps_plain_values_but_no_password(tmp_path) -> None:
    """A plain number or switch word is recorded and comes back on resume;
    a credential name keeps even a numeric value out, and keys stay out."""
    job = Evaluation(
        tasks_dir=_tasks(tmp_path),
        jobs_dir=tmp_path / "jobs",
        config=_config(
            agent_env={
                "MAX_THINKING_TOKENS": "32000",
                "AUTH_ENABLED": "false",
                "DB_PASSWORD": "123456",
                "MY_TOKEN": "abc-not-a-number",
                "OPENAI_API_KEY": SECRET,
            },
            reviewer={"model": "judge-1", "agent_env": {"MAX_TOKENS": "4000"}},
        ),
        job_name="named",
    )
    job._write_evaluation_record()
    record = json.loads((job.job_dir / "evaluation.json").read_text())["config"]
    assert record["agent_env"] == {
        "MAX_THINKING_TOKENS": "32000",
        "AUTH_ENABLED": "false",
    }
    assert record["reviewer"]["agent_env"] == {"MAX_TOKENS": "4000"}
    resumed = Evaluation.resume(job.job_dir)
    assert resumed._config.agent_env == record["agent_env"]
    assert resumed._config.reviewer.agent_env == {"MAX_TOKENS": "4000"}


def test_a_reviewer_token_count_is_compared(tmp_path) -> None:
    tasks_dir = _tasks(tmp_path)
    job = Evaluation(
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
        config=_config(reviewer={"model": "j", "agent_env": {"MAX_TOKENS": "1000"}}),
    )
    job._write_evaluation_record()
    _mark_reviewed(job.job_dir)
    other = _config(reviewer={"model": "j", "agent_env": {"MAX_TOKENS": "4000"}})
    differs = _recorded_run_differs(job.job_dir, other, tasks_dir)
    assert differs == "reviewer.agent_env MAX_TOKENS changed"


def test_an_older_record_without_a_plain_value_is_not_a_change(tmp_path) -> None:
    """A record from before plain values were stored lists the name only:
    its value is unknown there, so the same rerun still resumes."""
    tasks_dir = _tasks(tmp_path)
    config = _config(agent_env={"MAX_THINKING_TOKENS": "32000"})
    job = Evaluation(tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=config)
    job._write_evaluation_record()
    path = job.job_dir / "evaluation.json"
    record = json.loads(path.read_text())
    record["config"]["agent_env"] = {}
    path.write_text(json.dumps(record))
    assert _recorded_run_differs(job.job_dir, config, tasks_dir) is None


@pytest.mark.parametrize(
    "name", ["MAX_THINKING_TOKENS", "CLAUDE_CODE_MAX_OUTPUT_TOKENS", "SESSION_TIMEOUT"]
)
def test_a_token_count_in_config_override_env_is_a_change(
    tmp_path, caplog, name
) -> None:
    def override(value: str) -> dict:
        return {"sandbox": {"env": {name: value, "HF_TOKEN": OLD_HF}}}

    old = _first_job(tmp_path, job_name="named", config_override=override("1024"))
    changed = _config(config_override=override("32000"))
    refusal = _refusal(old, changed)
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        dest = _check_resume_mismatch(old, changed, restart_changed=True)
    assert dest is not None
    text = (dest / "set-aside.json").read_text()
    assert json.loads(text)["reason"] == f"config_override sandbox.env.{name} changed"
    assert f"config_override sandbox.env.{name} changed" in refusal
    for value in ("32000", OLD_HF):
        assert value not in caplog.text
        assert value not in text
        assert value not in refusal


def test_a_schema_field_named_key_is_compared(tmp_path) -> None:
    """environment_manifest.task_selection.key is the variable that carries
    the task id: a field name, not a credential, so a fix to it is a change."""
    from benchflow.environment.manifest import EnvironmentManifest, TaskSelection

    def manifest(key: str) -> EnvironmentManifest:
        return EnvironmentManifest(
            name="svc", image="svc:1", task_selection=TaskSelection(key=key)
        )

    tasks_dir = _tasks(tmp_path)
    job = Evaluation(
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
        config=_config(environment_manifest=manifest("TASK")),
    )
    job._write_evaluation_record()
    differs = _recorded_run_differs(
        job.job_dir, _config(environment_manifest=manifest("CHI_TASK_ID")), tasks_dir
    )
    assert differs == (
        "environment_manifest.task_selection.key='TASK' (this run: 'CHI_TASK_ID')"
    )


# --- agent_env values a YAML or JSON config gives as numbers or switches ------


def _yaml_config(tmp_path: Path, agent_env: dict) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tasks_dir": str(_tasks(tmp_path)),
                "jobs_dir": str(tmp_path / "jobs"),
                "agent": "oracle",
                "concurrency": 1,
                "agent_env": agent_env,
            }
        )
    )
    return path


_YAML_ENV = {
    "MAX_THINKING_TOKENS": 32000,
    "MY_RATIO": 0.5,
    "MY_FLAG": True,
    "AUTH_ENABLED": False,
    "MY_COUNT": 0,
    "MY_NAME": "x",
}
_YAML_ENV_AS_STRINGS = {
    "MAX_THINKING_TOKENS": "32000",
    "MY_RATIO": "0.5",
    "MY_FLAG": "true",
    "AUTH_ENABLED": "false",
    "MY_COUNT": "0",
    "MY_NAME": "x",
}


def test_yaml_numbers_and_switches_in_agent_env_are_strings(tmp_path) -> None:
    """MAX_THINKING_TOKENS: 32000 in a config used to reach the record (and
    the sandbox's env file) as an int and raise TypeError before any trial."""
    job = Evaluation.from_yaml(_yaml_config(tmp_path, _YAML_ENV))
    assert job._config.agent_env == _YAML_ENV_AS_STRINGS
    job._write_evaluation_record()
    record = json.loads((job.job_dir / "evaluation.json").read_text())["config"]
    assert record["agent_env"] == _YAML_ENV_AS_STRINGS
    from_dict = Evaluation.from_dict(
        yaml.safe_load(_yaml_config(tmp_path, _YAML_ENV).read_text())
    )
    assert from_dict._config.agent_env == _YAML_ENV_AS_STRINGS


def test_a_job_run_with_a_number_resumes_with_its_string(tmp_path, caplog) -> None:
    old = _first_job(
        tmp_path, job_name="named", agent_env={"MAX_THINKING_TOKENS": 32000}
    )
    record = json.loads((old / "evaluation.json").read_text())["config"]
    assert record["agent_env"] == {"MAX_THINKING_TOKENS": "32000"}
    for same in ("32000", 32000):
        config = _config(agent_env={"MAX_THINKING_TOKENS": same})
        assert _recorded_run_differs(old, config, None) is None
        assert _check_resume_mismatch(old, config) is None
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        dest = _check_resume_mismatch(
            old, _config(agent_env={"MAX_THINKING_TOKENS": 64000}), restart_changed=True
        )
    assert dest is not None
    reason = json.loads((dest / "set-aside.json").read_text())["reason"]
    assert reason == "agent_env MAX_THINKING_TOKENS changed"
    assert "64000" not in caplog.text


def test_a_number_set_after_construction_is_recorded_as_its_string(
    tmp_path,
) -> None:
    """The --config path sets agent_env after EvaluationConfig() ran."""
    job = _evaluation(tmp_path, _config(), job_name="named")
    job._config.agent_env = {"MAX_THINKING_TOKENS": 32000, "MY_COUNT": 7}
    job._write_evaluation_record()
    record = json.loads((job.job_dir / "evaluation.json").read_text())["config"]
    assert record["agent_env"] == {"MAX_THINKING_TOKENS": "32000", "MY_COUNT": "7"}


@pytest.mark.parametrize("value", [None, [1, 2], {"a": 1}])
def test_an_env_value_with_no_string_form_is_refused_by_name(tmp_path, value) -> None:
    with pytest.raises(ValueError, match="agent_env MY_VAR is"):
        _config(agent_env={"MY_VAR": value})
    with pytest.raises(ValueError, match="agent_env MY_VAR is"):
        Evaluation.from_yaml(_yaml_config(tmp_path, {"MY_VAR": value}))


def test_the_redaction_helpers_take_a_number() -> None:
    from benchflow._utils.config_redaction import (
        _is_secret_env_value,
        _should_compare_env_entry,
        _should_record_env_entry,
    )

    assert _should_compare_env_entry("MAX_THINKING_TOKENS", 32000)
    assert not _should_record_env_entry("MAX_THINKING_TOKENS", 32000)
    assert _should_record_env_entry("MY_COUNT", 7)
    assert _should_record_env_entry("MY_FLAG", True)
    assert not _should_compare_env_entry("DB_PASSWORD", 123456)
    assert not _is_secret_env_value("MY_RATIO", 0.5)


def test_a_sharded_worker_gets_strings() -> None:
    from benchflow.eval_worker import _evaluation_config

    config = _evaluation_config({"agent": "oracle", "agent_env": _YAML_ENV})
    assert config.agent_env == _YAML_ENV_AS_STRINGS


def test_cli_config_numbers_in_agent_env_reach_the_record_as_strings(
    tmp_path,
) -> None:
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    path = _yaml_config(tmp_path, {"MAX_THINKING_TOKENS": 32000, "MY_FLAG": True})
    seen = []

    async def record_only(self):
        self._write_evaluation_record()
        seen.append(dict(self._config.agent_env))
        record = json.loads((self.job_dir / "evaluation.json").read_text())
        seen.append(record["config"]["agent_env"])
        return SimpleNamespace(
            passed=1, total=1, score=1.0, errored=0, verifier_errored=0
        )

    with patch.object(Evaluation, "run", new=record_only):
        result = CliRunner().invoke(
            app,
            ["eval", "run", "--config", str(path), "--agent-env", "MY_COUNT=1"],
        )
    assert result.exit_code == 0, result.output
    expected = {"MAX_THINKING_TOKENS": "32000", "MY_FLAG": "true", "MY_COUNT": "1"}
    assert seen == [expected, expected]


@pytest.mark.parametrize("recorded", [True, False])
def test_sweep_dot_prefixed_trials_are_checked(tmp_path, recorded) -> None:
    """Hidden task names are valid trials."""
    tasks_dir = _tasks(tmp_path, n=1)
    (tasks_dir / "task-0").rename(tasks_dir / ".case")
    job = Evaluation(
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
        job_name="named",
        config=_config(),
    )
    job._run_single_task = AsyncMock(side_effect=_fake_runner(job))
    assert _run(job).ran == 1
    if not recorded:
        (job.job_dir / "evaluation.json").unlink()
    changed = _config(agent="nop")
    assert _recorded_run_differs(job.job_dir, changed, tasks_dir) is not None
    dest = _check_resume_mismatch(job.job_dir, changed, restart_changed=True)
    assert dest is not None
    assert len(load_task_results(dest)) == 1
    assert load_task_results(job.job_dir) == {}
    # A later scan must not descend into the reserved archive.
    assert _check_resume_mismatch(job.job_dir, changed) is None


@pytest.mark.parametrize(
    "name", ["SSH_KEY_PASSPHRASE", "API_KEY", "TOKEN", "AUTH", "SESSION"]
)
@pytest.mark.parametrize("value", ["24681357", "false"])
def test_sweep_numeric_credentials_are_not_recorded(tmp_path, name, value) -> None:
    """Guards the fix at 8f7d1c62 for both execution environments."""
    job = _evaluation(
        tmp_path,
        _config(agent_env={name: value}, reviewer={"agent_env": {name: value}}),
    )
    job._write_evaluation_record()
    record = json.loads((job.job_dir / "evaluation.json").read_text())["config"]
    assert record["agent_env"] == {}
    assert record["reviewer"]["agent_env"] == {}
    resumed = Evaluation.resume(job.job_dir)
    assert resumed._config.agent_env == {}
    assert resumed._config.reviewer.agent_env == {}


@pytest.mark.parametrize("value", ["24681357", 24681357, False])
def test_sweep_override_numeric_credentials_are_not_compared(tmp_path, value) -> None:
    """Shape cannot make auth public."""
    job = _evaluation(
        tmp_path, _config(config_override={"sandbox": {"env": {"API_KEY": value}}})
    )
    job._write_evaluation_record()
    changed = _config(config_override={"sandbox": {"env": {"API_KEY": "13572468"}}})
    assert _recorded_run_differs(job.job_dir, changed, None) is None


@pytest.mark.parametrize(
    "flag,value,field,expected",
    [
        ("--timeout-multiplier", "2", "timeout_multiplier", 2.0),
        (
            "--extra-instruction",
            "Check edge cases.",
            "extra_instruction",
            "Check edge cases.",
        ),
    ],
)
def test_sweep_config_file_applies_cli_settings_before_resume(
    tmp_path, flag, value, field, expected
) -> None:
    """Guards the fix at 8f7d1c62 through the real CLI and resume guard."""
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    path = _yaml_config(tmp_path, {})
    jobs = []
    configs = []

    async def run(self):
        self._run_single_task = AsyncMock(side_effect=_fake_runner(self))
        result = await original_run(self)
        jobs.append(self.job_dir)
        configs.append(getattr(self._config, field))
        return result

    original_run = Evaluation.run
    with patch.object(Evaluation, "run", new=run):
        runner = CliRunner()
        for extra in ([], [flag, value], [flag, value]):
            result = runner.invoke(app, ["eval", "run", "--config", str(path), *extra])
            assert result.exit_code == 0, result.output
    assert configs[1:] == [expected, expected]
    assert jobs[0] != jobs[1]
    assert jobs[1] == jobs[2]


@pytest.mark.parametrize("mode", ["parallel-independent", "sequential-shared"])
def test_sweep_pinned_scene_agent_resumes(tmp_path, mode) -> None:
    """Guards the fix at 8f7d1c62 against repeated scene reruns."""
    from benchflow.rollout import RolloutConfig

    tasks = _tasks(tmp_path)
    (tasks / "task-1" / "task.toml").unlink()
    (tasks / "task-1" / "task.md").write_text(
        "---\nsandbox:\n  docker_image: python:3.12-slim\nagents:\n  roles:\n    solver:\n      agent: codex\n"
        "scenes:\n  - name: solve\n    roles: [solver]\n---\n## prompt\nSolve it.\n"
    )
    (tasks / "task-1" / "environment").mkdir()
    (tasks / "task-1" / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
    )
    (tasks / "task-1" / "verifier").mkdir()
    (tasks / "task-1" / "verifier" / "test.sh").write_text("#!/bin/sh\nexit 0\n")

    def job(restart_changed=False):
        evaluation = Evaluation(
            tasks_dir=tasks,
            jobs_dir=tmp_path / "jobs",
            config=_config(job_mode=mode),
            job_name="named",
            restart_changed=restart_changed,
        )
        fake = _fake_runner(evaluation)

        async def run_task(task_path, config):
            effective = RolloutConfig.from_legacy(
                task_path=task_path, agent=config.agent, model=config.model
            )
            result = await fake(task_path, config)
            assert result.rollout_dir is not None
            path = result.rollout_dir / "config.json"
            path.write_text(json.dumps({"agent": effective.primary_agent}))
            return result

        # Exercise the same durable trial records in either scheduling mode.
        evaluation._run_single_task = AsyncMock(side_effect=run_task)
        return evaluation

    first = job()
    assert _run(first).ran == 2
    for _ in range(2):
        resumed = job()
        result = _run(resumed)
        assert (result.reused, result.ran) == (2, 0)
        assert not (resumed.job_dir / SET_ASIDE_DIR).exists()
    narrower = job()
    narrower._config.include_tasks = {"task-0"}
    assert _run(narrower).ran == 0
    assert not (narrower.job_dir / SET_ASIDE_DIR).exists()
    # Explicit prompts suppress pinned scenes, just as on a fresh rollout.
    changed = job(restart_changed=True)
    changed._config.prompts = ["Use the default agent."]
    assert _run(changed).ran == 2


@pytest.mark.parametrize("owner", ["solver", "reviewer", "override"])
@pytest.mark.parametrize("name", ["CODEX_CONFIG", "OPENCODE_CONFIG_CONTENT"])
def test_sweep_inline_settings_survive_credential_redaction(
    tmp_path, caplog, owner, name
) -> None:
    """Guards the fix at 8f7d1c62."""

    def config(enabled, credential, *, pretty=False):
        inline = {
            "web_search": "live" if enabled else "disabled",
            "tools": {"webfetch": enabled},
            "provider": {"options": {"apiKey": credential}},
            "mcp_servers": {
                "docs": {"http_headers": {"Authorization": f"Bearer {credential}"}}
            },
        }
        env = {name: json.dumps(inline, indent=2 if pretty else None, sort_keys=pretty)}
        if owner == "solver":
            return _config(agent_env=env)
        if owner == "reviewer":
            return _config(reviewer={"agent_env": env})
        return _config(config_override={"sandbox": {"env": env}})

    job = _evaluation(tmp_path, config(False, OLD_HF), job_name="named")
    assert _run(job).ran == 2
    _mark_reviewed(job.job_dir)
    if owner != "override":
        text = (job.job_dir / "evaluation.json").read_text()
        assert OLD_HF not in text
        # A fingerprint is metadata, never a value replayed as agent env.
        resumed = Evaluation.resume(job.job_dir)
        env = (
            resumed._config.agent_env
            if owner == "solver"
            else resumed._config.reviewer.agent_env
        )
        assert name not in env
    # Formatting and credential rotation alone preserve the protocol.
    rotated = config(False, NEW_HF, pretty=True)
    assert _recorded_run_differs(job.job_dir, rotated, None) is None
    changed = config(True, NEW_HF)
    assert _recorded_run_differs(job.job_dir, changed, None) is not None
    assert _evaluation(tmp_path, changed).job_dir != job.job_dir
    refusal = _refusal(job.job_dir, changed)
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        dest = _check_resume_mismatch(job.job_dir, changed, restart_changed=True)
    assert dest is not None
    reason = (dest / "set-aside.json").read_text()
    assert name in reason and name in refusal
    for secret in (OLD_HF, NEW_HF):
        assert secret not in reason
        assert secret not in caplog.text
        assert secret not in refusal


def test_sweep_inline_credentials_are_removed_before_fingerprinting(tmp_path) -> None:
    """Even short, unshaped credentials stay out."""
    from benchflow._utils.config_redaction import _inline_config_fingerprint

    def inline(credential):
        return json.dumps(
            {"tools": {"webfetch": False}, "provider": {"apiKey": credential}}
        )

    raw = inline("24681357")
    job = _evaluation(tmp_path, _config(agent_env={"OPENCODE_CONFIG_CONTENT": raw}))
    job._write_evaluation_record()
    text = (job.job_dir / "evaluation.json").read_text()
    assert "24681357" not in text
    fingerprint = _inline_config_fingerprint("OPENCODE_CONFIG_CONTENT", raw)
    assert fingerprint == _inline_config_fingerprint(
        "OPENCODE_CONFIG_CONTENT", inline("13572468")
    )
    assert fingerprint == _inline_config_fingerprint(
        "OPENCODE_CONFIG_CONTENT",
        json.dumps({"tools": {"webfetch": False}, "provider": {}}),
    )


@pytest.mark.parametrize("owner", ["solver", "reviewer", "override"])
@pytest.mark.parametrize(
    "name,path,old,new",
    [
        ("CODEX_CONFIG", ("tool_output_token_limit",), 1024, 8192),
        ("CODEX_CONFIG", ("profiles", "small", "model_max_output_tokens"), 1024, 8192),
        ("CODEX_CONFIG", ("model_providers", "local", "env_key"), "KEY_A", "KEY_B"),
        (
            "CODEX_CONFIG",
            ("mcp_servers", "local", "env_http_headers", "Authorization"),
            "HEADER_A",
            "HEADER_B",
        ),
        (
            "CODEX_CONFIG",
            ("mcp_servers", "local", "bearer_token_env_var"),
            "TOKEN_A",
            "TOKEN_B",
        ),
        ("OPENCODE_CONFIG_CONTENT", ("keybinds", "session_export"), "a", "b"),
        (
            "OPENCODE_CONFIG_CONTENT",
            ("provider", "local", "options", "maxTokens"),
            1024,
            8192,
        ),
    ],
)
def test_confirm_inline_execution_fields_are_compared(
    tmp_path, owner, name, path, old, new
) -> None:
    """Guards P2-CONFIRM-1 at 7ab60011: JSON fields are not env names."""

    def config(value):
        node = value
        for key in reversed(path):
            node = {key: node}
        env = {name: json.dumps(node)}
        if owner == "solver":
            return _config(agent_env=env)
        if owner == "reviewer":
            return _config(reviewer={"agent_env": env})
        return _config(config_override={"sandbox": {"env": env}})

    first = _evaluation(tmp_path, config(old), job_name="named")
    assert _run(first).ran == 2
    _mark_reviewed(first.job_dir)
    assert _recorded_run_differs(first.job_dir, config(old), None) is None
    assert _recorded_run_differs(first.job_dir, config(new), None) is not None
    assert _evaluation(tmp_path, config(new)).job_dir != first.job_dir
    with pytest.raises(ResumeMismatchError):
        _check_resume_mismatch(first.job_dir, config(new))


def test_confirm_inline_credential_paths_and_public_fields() -> None:
    """Guards P2-CONFIRM-1 at 7ab60011, including credential siblings."""
    from benchflow._utils.config_redaction import (
        _inline_config_fingerprint,
        _inline_config_structure,
        _should_record_env_entry,
    )

    public = {
        "tool_output_token_limit": 1024,
        "model_providers": {"local": {"env_key": "OPENAI_API_KEY"}},
        "mcp_servers": {
            "local": {
                "env": {"MAX_TOKENS": "1024"},
                "env_http_headers": {"Authorization": "HEADER_A"},
            }
        },
    }
    credential = {
        **public,
        "provider": {"options": {"apiKey": "24681357"}},
        "mcp_servers": {
            "local": {
                "env": {"MAX_TOKENS": "1024", "SERVICE_SESSION": "24681357"},
                "http_headers": {"Authorization": "24681357"},
            }
        },
    }
    for name in ("CODEX_CONFIG", "OPENCODE_CONFIG_CONTENT"):
        raw = json.dumps(public)
        assert _inline_config_structure(name, raw) == (public, public)
        assert _should_record_env_entry(name, raw)
        raw = json.dumps(credential)
        assert not _should_record_env_entry(name, raw)
        _, stripped = _inline_config_structure(name, raw)
        assert "24681357" not in json.dumps(stripped)
        assert stripped["tool_output_token_limit"] == 1024
        assert stripped["model_providers"] == public["model_providers"]
        assert stripped["mcp_servers"]["local"]["env"] == {"MAX_TOKENS": "1024"}
        assert _inline_config_fingerprint(name, raw) == _inline_config_fingerprint(
            name, raw.replace("24681357", "13572468")
        )
        for field in (
            "apiKey",
            "access_key_id",
            "secretAccessKey",
            "private_key",
            "sessionToken",
            "client_secret",
            "experimental_bearer_token",
        ):
            raw = json.dumps({"provider": {"options": {field: "24681357"}}})
            assert not _should_record_env_entry(name, raw)
            _, stripped = _inline_config_structure(name, raw)
            assert "24681357" not in json.dumps(stripped)


@pytest.mark.parametrize("mode", ["parallel-independent", "sequential-shared"])
@pytest.mark.parametrize("recorded", [True, False])
def test_confirm_narrowed_draft2_resume_materializes_retained_tasks(
    tmp_path, monkeypatch, mode, recorded
) -> None:
    """Guards P2-CONFIRM-2 at 7ab60011 through both resume entry points."""
    import shutil

    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "cache"))
    tasks = tmp_path / "tasks"
    examples = Path(__file__).parent / "fixtures" / "taskmd" / "examples"
    for name in ("hello-world", "regex-log"):
        shutil.copytree(examples / name, tasks / name)

    def job():
        evaluation = Evaluation(
            tasks_dir=tasks,
            jobs_dir=tmp_path / "jobs",
            job_name="named",
            config=_config(job_mode=mode),
        )
        evaluation._run_single_task = AsyncMock(side_effect=_fake_runner(evaluation))
        return evaluation

    first = job()
    assert _run(first).ran == 2
    if not recorded:
        (first.job_dir / "evaluation.json").unlink()
    # Constructor-time legacy matching also resolves raw format sources.
    assert _recorded_run_differs(first.job_dir, first._config, tasks) is None
    resumed = job()
    resumed._config.include_tasks = {"hello-world"}
    result = _run(resumed)
    assert result.ran == 0
    assert result.reused >= 1
    resumed._run_single_task.assert_not_called()
    assert not (resumed.job_dir / SET_ASIDE_DIR).exists()
    # The unselected raw source stays in its original format.
    assert not (tasks / "regex-log" / "task.md").read_text().startswith("---")


@pytest.mark.parametrize("owner", ["solver", "reviewer", "override"])
@pytest.mark.parametrize(
    "path,old,new",
    [
        (("permission", "grep", "password"), "deny", "allow"),
        (("agent", "build", "permission", "grep", "password"), "deny", "allow"),
        (("mode", "build", "permission", "grep", "password"), "deny", "allow"),
        (("permission", "password"), "deny", "allow"),
        (("tools", "password"), False, True),
        (("agent", "build", "tools", "password"), False, True),
        (("mode", "build", "tools", "password"), False, True),
        (("keybinds", "password"), "ctrl+a", "ctrl+b"),
        (
            ("permission", "grep"),
            {"*": "deny", "password": "allow"},
            {"password": "allow", "*": "deny"},
        ),
        (
            ("agent", "build", "permission"),
            {"*": "deny", "grep": "allow"},
            {"grep": "allow", "*": "deny"},
        ),
    ],
)
def test_confirm5_public_mapping_changes_prevent_resume(
    tmp_path, caplog, owner, path, old, new
) -> None:
    """Guards P2-CONFIRM-1 at 8f239906: permission patterns are public keys."""

    def config(value, credential):
        node = value
        for key in reversed(path):
            node = {key: node}
        node["provider"] = {"local": {"options": {"apiKey": credential}}}
        env = {"OPENCODE_CONFIG_CONTENT": json.dumps(node)}
        if owner == "solver":
            return _config(agent_env=env)
        if owner == "reviewer":
            return _config(reviewer={"agent_env": env})
        return _config(config_override={"sandbox": {"env": env}})

    first = _evaluation(tmp_path, config(old, "24681357"), job_name="named")
    assert _run(first).ran == 2
    _mark_reviewed(first.job_dir)
    assert _recorded_run_differs(first.job_dir, config(old, "13572468"), None) is None
    changed = config(new, "13572468")
    assert _recorded_run_differs(first.job_dir, changed, None) is not None
    assert _evaluation(tmp_path, changed).job_dir != first.job_dir
    refusal = _refusal(first.job_dir, changed)
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        dest = _check_resume_mismatch(first.job_dir, changed, restart_changed=True)
    assert dest is not None
    reason = (dest / "set-aside.json").read_text()
    for marker in ("24681357", "13572468"):
        assert marker not in reason
        assert marker not in caplog.text
        assert marker not in refusal


@pytest.mark.parametrize("scope", [(), ("agent", "headers"), ("mode", "env")])
@pytest.mark.parametrize("tool", ["grep", "headers", "environment", "env_http_headers"])
def test_confirm5_permission_names_are_not_credential_fields(scope, tool) -> None:
    """Guards P2-CONFIRM-1 at 8f239906 across public schema-map contexts."""
    from benchflow._utils.config_redaction import (
        _inline_config_structure,
        _should_record_env_entry,
    )

    patterns = (
        "password",
        "api_key",
        "token",
        "credentials",
        "auth",
        "authorization",
        "cookie",
        "session",
        "env",
        "headers",
        "http_headers",
        "environment",
        "env_http_headers",
    )
    public = {"permission": {tool: dict.fromkeys(patterns, "deny")}}
    for key in reversed(scope):
        public = {key: public}
    raw = json.dumps(public)
    assert _inline_config_structure("OPENCODE_CONFIG_CONTENT", raw) == (public, public)
    assert _should_record_env_entry("OPENCODE_CONFIG_CONTENT", raw)


@pytest.mark.parametrize("name", ["CODEX_CONFIG", "OPENCODE_CONFIG_CONTENT"])
def test_confirm5_public_names_do_not_exempt_credential_locations(name) -> None:
    """Guards P2-CONFIRM-1 at 8f239906 without exempting lookalike auth paths."""
    from benchflow._utils.config_redaction import (
        _inline_config_fingerprint,
        _inline_config_structure,
        _should_record_env_entry,
    )

    def raw(marker):
        return json.dumps(
            {
                "provider": {
                    "local": {
                        "options": {
                            "permission": {"password": marker},
                            "tools": {"token": marker},
                            "headers": {"Authorization": marker},
                        }
                    }
                },
                "mcp_servers": {"permission": {"env": {"SERVICE_SESSION": marker}}},
            }
        )

    assert not _should_record_env_entry(name, raw("24681357"))
    assert "24681357" not in json.dumps(
        _inline_config_structure(name, raw("24681357"))[1]
    )
    assert _inline_config_fingerprint(
        name, raw("24681357")
    ) == _inline_config_fingerprint(name, raw("13572468"))


@pytest.mark.parametrize("owner", ["solver", "reviewer", "override"])
@pytest.mark.parametrize(
    "scope",
    [(), ("agent", "build"), ("mode", "build"), ("agent", "headers"), ("mode", "env")],
)
@pytest.mark.parametrize(
    "switches",
    [
        {"write": True, "edit": False},
        {"write": True, "patch": False},
        {"edit": True, "patch": False},
        {"*": False, "edit": True},
        {"*": False, "mcp_*": True},
        {"mcp_*": False, "mcp_read": True},
        {"mcp_?ead": False, "mcp_read": True},
    ],
)
@pytest.mark.parametrize("record_kind", ["public", "secret", "legacy-public"])
def test_confirm7_tool_switch_order_prevents_resume(
    tmp_path, caplog, owner, scope, switches, record_kind
) -> None:
    """Guards P2-CONFIRM-3 at d80576e9 across aliases, wildcards and scopes."""

    def config(value, marker, *, pretty=False):
        node = {"tools": value}
        for key in reversed(scope):
            node = {key: node}
        if record_kind == "secret":
            node["provider"] = {"local": {"options": {"apiKey": marker}}}
        env = {
            "OPENCODE_CONFIG_CONTENT": json.dumps(node, indent=2 if pretty else None)
        }
        if owner == "solver":
            return _config(agent_env=env)
        if owner == "reviewer":
            return _config(reviewer={"agent_env": env})
        return _config(config_override={"sandbox": {"env": env}})

    first = _evaluation(tmp_path, config(switches, "24681357"), job_name="named")
    assert _run(first).ran == 2
    _mark_reviewed(first.job_dir)
    if record_kind == "legacy-public":
        record_path = first.job_dir / "evaluation.json"
        record = json.loads(record_path.read_text())
        record["config"].pop("agent_env_fingerprints", None)
        record["config"]["reviewer"].pop("agent_env_fingerprints", None)
        record_path.write_text(json.dumps(record))
    # Whitespace and credential rotation do not change the ordered policy.
    same = config(switches, "13572468", pretty=True)
    assert _recorded_run_differs(first.job_dir, same, None) is None
    assert _evaluation(tmp_path, same).job_dir == first.job_dir
    assert _check_resume_mismatch(first.job_dir, same) is None
    changed = config(dict(reversed(switches.items())), "13572468")
    assert _recorded_run_differs(first.job_dir, changed, None) is not None
    assert _evaluation(tmp_path, changed).job_dir != first.job_dir
    refusal = _refusal(first.job_dir, changed)
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        dest = _check_resume_mismatch(first.job_dir, changed, restart_changed=True)
    assert dest is not None
    reason = (dest / "set-aside.json").read_text()
    assert "OPENCODE_CONFIG_CONTENT" in reason and "OPENCODE_CONFIG_CONTENT" in refusal
    for marker in ("24681357", "13572468"):
        assert marker not in reason
        assert marker not in caplog.text
        assert marker not in refusal


@pytest.mark.parametrize("name", ["CODEX_CONFIG", "OPENCODE_CONFIG_CONTENT"])
def test_confirm7_unordered_mapping_controls(name) -> None:
    """Guards P2-CONFIRM-3 at d80576e9: only schema policy maps carry order."""
    from benchflow._utils.config_redaction import _inline_config_fingerprint

    before = {
        "keybinds": {"session_new": "ctrl+b", "session_export": "ctrl+a"},
        "provider": {"local": {"options": {"tools": {"last": False, "first": True}}}},
    }
    # Codex has no OpenCode policy-map semantics.
    if name == "CODEX_CONFIG":
        before["tools"] = {"write": True, "edit": False}
    raw = json.dumps(before)
    assert _inline_config_fingerprint(name, raw) == _inline_config_fingerprint(
        name, json.dumps(before, sort_keys=True, indent=2)
    )
