"""Guards IU-R1-1 at 7bc81e6ba: known 0.7.11 settings must prevent mixed reuse.

The fixture follows 0d0ccc67d7ec7fc694a29f2120117a6bf12b3f79's
rollout/_results.py writer, without evaluation.json or a trial version stamp.
It is authored directly, not produced by the new writer with a deleted record.
Only trial execution is scripted; selection, preservation and reuse are real.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.result_paths import SET_ASIDE_DIR, load_task_results
from benchflow._utils.task_authoring import task_digest
from benchflow.evaluation import (
    Evaluation,
    EvaluationConfig,
    ResumeMismatchError,
    RetryConfig,
)
from benchflow.jobs import load_job
from benchflow.learner_skills import capture_skills
from benchflow.models import RolloutResult


@pytest.fixture(autouse=True)
def offline_execution(monkeypatch):
    """Keep host maintenance out of these scripted job runs."""
    monkeypatch.setattr(Evaluation, "_sweep_docker", AsyncMock())
    monkeypatch.setattr(
        "benchflow.sandbox.docker.prefetch_local_subnet_allocator", lambda: None
    )
    monkeypatch.setenv("BENCHFLOW_DAYTONA_AUTO_REAP", "0")
    monkeypatch.setenv("BENCHFLOW_PREPULL", "0")


def _config(**overrides):
    return EvaluationConfig(
        **{
            "agent": "codex-acp",
            "model": "openai/fixture-old",
            "reasoning_effort": "low",
            "environment": "docker",
            "sandbox_user": "agent",
            "concurrency": 1,
            "warm_pool": 0,
            "retry": RetryConfig(max_retries=0),
            **overrides,
        }
    )


def _legacy_job(tmp_path, *, interrupted=False):
    tasks = tmp_path / "tasks"
    old = tmp_path / "jobs" / "2026-01-01__00-00-00"
    for i in range(2):
        name = f"task-{i}"
        task = tasks / name
        task.mkdir(parents=True)
        (task / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
        trial = old / f"{name}__old00000"
        trial.mkdir(parents=True)
        (trial / "config.json").write_text(
            json.dumps(
                {
                    "task_path": name,
                    "agent": "codex-acp",
                    "model": "openai/fixture-old",
                    "reasoning_effort": "low",
                    "environment": "docker",
                    "acp_transport": "docker-stdio",
                    "environment_manifest": None,
                    "skill_mode": "no-skill",
                    "skill_source": "none",
                    "requested_skills_dir": None,
                    "effective_skills_dir": None,
                    "skills_sandbox_dir": None,
                    "include_task_skills": False,
                    "sandbox_user": "agent",
                    "sandbox_locked_paths": [],
                    "sandbox_setup_timeout": 120,
                    "skip_install": False,
                    "agent_install_timeout": 900,
                    "context_root": None,
                    "base_image_override": None,
                    "verifier_sandbox_mode": "shared",
                    "verifier_include": [],
                    "timeout_sec": 60,
                    "concurrency": 1,
                    "agent_idle_timeout_sec": 600,
                    "started_at": "2026-01-01 00:00:00",
                    "agent_env": {},
                    "scenes": [
                        {
                            "name": "default",
                            "skills_dir": None,
                            "roles": [
                                {
                                    "name": "agent",
                                    "agent": "codex-acp",
                                    "model": "openai/fixture-old",
                                    "reasoning_effort": "low",
                                    "timeout_sec": None,
                                    "idle_timeout_sec": None,
                                    "skills_dir": None,
                                    "capabilities": None,
                                    "env_keys": [],
                                }
                            ],
                            "turns": [{"role": "agent", "has_prompt": False}],
                        }
                    ],
                    "loop": {"strategy": "single-shot"},
                }
            )
        )
        if not interrupted or i == 0:
            (trial / "result.json").write_text(
                json.dumps(
                    {
                        "task_name": name,
                        "rollout_name": trial.name,
                        "agent": "codex-acp",
                        "model": "openai/fixture-old",
                        "rewards": {"reward": 1.0},
                        "error": None,
                        "verifier_error": None,
                    }
                )
            )
    (old / "summary.json").write_text(
        json.dumps({"total": 2, "passed": 1 if interrupted else 2, "mean_reward": 1.0})
    )
    (old / "results.jsonl").write_text('{"task": "task-0", "reward": 1.0}\n')
    return tasks, old


def _edit_records(old, patch, *, omit=()):
    for path in old.glob("*/config.json"):
        record = json.loads(path.read_text())
        record.update(patch)
        role = record["scenes"][0]["roles"][0]
        role.update({k: v for k, v in patch.items() if k in role})
        for key in omit:
            record.pop(key, None)
            role.pop(key, None)
        path.write_text(json.dumps(record))
        result_path = path.with_name("result.json")
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            result.update({k: v for k, v in patch.items() if k in result})
            for key in omit:
                result.pop(key, None)
            result_path.write_text(json.dumps(result))


def _files(folder):
    return {
        p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()
    }


def _assert_trial_bytes(original, folder):
    for path, content in original.items():
        if path.parts[0].startswith("task-"):
            assert (folder / path).read_bytes() == content


def _evaluation(tasks, old, config, selection):
    job = Evaluation(
        tasks_dir=tasks,
        jobs_dir=old.parent,
        job_name=old.name if selection == "named" else None,
        config=config,
        preflight=False,
    )

    async def run_task(task_path, cfg):
        # Fresh rollout resolution, independent of the resume comparison.
        effective = job._rollout_config(task_path, cfg)
        trial = job.job_dir / f"{task_path.name}__new00000"
        trial.mkdir(parents=True)
        (trial / "config.json").write_text(
            json.dumps(
                {
                    "task_path": task_path.name,
                    "agent": effective.primary_agent,
                    "model": effective.primary_model,
                    "reasoning_effort": effective.primary_reasoning_effort,
                    "environment": effective.environment,
                    "sandbox_user": effective.sandbox_user,
                }
            )
        )
        payload = {
            "task_name": task_path.name,
            "rollout_name": trial.name,
            "agent": effective.primary_agent,
            "model": effective.primary_model,
            "rewards": {"reward": 1.0},
            "error": None,
            "verifier_error": None,
        }
        (trial / "result.json").write_text(json.dumps(payload))
        return RolloutResult(**payload, rollout_dir=trial)

    job._run_single_task = AsyncMock(side_effect=run_task)
    return job


@pytest.mark.parametrize("selection", ["auto", "named"])
@pytest.mark.parametrize("interrupted", [False, True], ids=["complete", "interrupted"])
@pytest.mark.parametrize(
    "changed",
    [
        {},
        {"model": "openai/fixture-new"},
        {"reasoning_effort": "high"},
        {"environment": "remote-docker"},
        {"sandbox_user": None},
    ],
    ids=["unchanged", "model", "effort", "backend", "user"],
)
def test_legacy_settings_control_reuse(tmp_path, selection, interrupted, changed):
    """Guards IU-R1-1 at 7bc81e6ba: both entry paths keep or rerun real old rows."""
    tasks, old = _legacy_job(tmp_path, interrupted=interrupted)
    original = _files(old)
    completed = 1 if interrupted else 2
    assert len(load_job(old).trials) == completed
    assert _files(old) == original  # Reading an old job does not migrate it.
    job = _evaluation(tasks, old, _config(**changed), selection)
    result = asyncio.run(job.run())
    expected = (0, 2) if changed else (completed, 2 - completed)
    assert (result.reused, result.ran) == expected
    assert job._run_single_task.await_count == expected[1]
    assert len(load_task_results(job.job_dir)) == 2
    record = json.loads((job.job_dir / "evaluation.json").read_text())
    assert record["config"]["model"] == job._config.model
    assert record["config"]["reasoning_effort"] == job._config.reasoning_effort
    if changed:
        assert {r["model"] for r in load_task_results(job.job_dir).values()} == {
            job._config.model
        }
        if selection == "auto":
            assert job.job_dir != old
            assert _files(old) == original
        else:
            assert job.job_dir == old
            (kept,) = (old / SET_ASIDE_DIR).iterdir()
            for path, content in original.items():
                assert (kept / path).read_bytes() == content
    else:
        assert job.job_dir == old
        assert not (old / SET_ASIDE_DIR).exists()
        _assert_trial_bytes(original, old)
    if job.job_dir == old:
        for name in ("summary.json", "results.jsonl"):
            assert (old / ".pre-0.8" / name).read_bytes() == original[Path(name)]
        backup = _files(old / ".pre-0.8")
        again = _evaluation(tasks, old, _config(**changed), "named")
        resumed = asyncio.run(again.run())
        assert (resumed.reused, resumed.ran) == (2, 0)
        assert _files(old / ".pre-0.8") == backup


@pytest.mark.parametrize("selection", ["auto", "named"])
@pytest.mark.parametrize(
    "saved,omitted,requested,rerun",
    [
        ({}, ("model",), {"model": "openai/fixture-new"}, False),
        ({}, ("reasoning_effort",), {"reasoning_effort": "high"}, False),
        ({}, ("environment",), {"environment": "remote-docker"}, False),
        ({}, ("sandbox_user",), {"sandbox_user": None}, False),
        ({}, ("model",), {"reasoning_effort": "high"}, True),
        ({"model": None}, (), {"model": None}, False),
        ({"model": None}, (), {"model": "openai/fixture-new"}, True),
        ({"reasoning_effort": None}, (), {"reasoning_effort": None}, False),
        ({"reasoning_effort": None}, (), {"reasoning_effort": "none"}, True),
        ({"reasoning_effort": " HIGH "}, (), {"reasoning_effort": "high"}, False),
        ({"sandbox_user": "root"}, (), {"sandbox_user": None}, False),
        ({"sandbox_user": "0"}, (), {"sandbox_user": "null"}, False),
    ],
    ids=[
        "missing-model",
        "missing-effort",
        "missing-backend",
        "missing-user",
        "missing-one-known-change",
        "null-model",
        "null-model-changed",
        "null-effort",
        "explicit-no-effort",
        "normalized-effort",
        "root-user",
        "root-uid",
    ],
)
def test_legacy_unknown_and_unset_settings(
    tmp_path, selection, saved, omitted, requested, rerun
):
    """Guards IU-R1-1 at 7bc81e6ba: absence is unknown; null has known meaning."""
    tasks, old = _legacy_job(tmp_path)
    _edit_records(old, saved, omit=omitted)
    original = _files(old)
    job = _evaluation(tasks, old, _config(**requested), selection)
    result = asyncio.run(job.run())
    assert (result.reused, result.ran) == ((0, 2) if rerun else (2, 0))
    kept = old
    if rerun and selection == "named":
        (kept,) = (old / SET_ASIDE_DIR).iterdir()
    _assert_trial_bytes(original, kept)


@pytest.mark.parametrize("selection", ["auto", "named"])
@pytest.mark.parametrize("prompts", [None, ["Use the requested settings."]])
@pytest.mark.parametrize("pins", ["mixed", "all-pinned", "all-unset"])
def test_legacy_scene_overrides_and_explicit_prompts(
    tmp_path, selection, prompts, pins
):
    """Guards IU-R1-1 at 7bc81e6ba: compare scene values, not raw job settings."""
    tasks, old = _legacy_job(tmp_path)
    pinned_model = None if pins == "all-unset" else "openai/fixture-old"
    for task in sorted(tasks.iterdir()):
        if pins == "mixed" and task.name == "task-0":
            continue
        settings = (
            ""
            if pins == "all-unset"
            else "      model: openai/fixture-old\n      reasoning_effort: low\n"
        )
        # task.md is the entrypoint even beside task.toml, as on fresh rollouts.
        (task / "task.md").write_text(
            "---\nsandbox:\n  docker_image: python:3.12-slim\nagents:\n  roles:\n"
            "    solver:\n      agent: codex\n"
            f"{settings}scenes:\n  - name: solve\n    roles: [solver]\n"
            "---\n## prompt\nSolve it.\n"
        )
        (task / "verifier").mkdir()
        (task / "verifier" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
        trial = old / f"{task.name}__old00000"
        path = trial / "config.json"
        saved = json.loads(path.read_text())
        saved.update(
            model=pinned_model,
            reasoning_effort=None if pins == "all-unset" else "low",
        )
        scene = saved["scenes"][0]
        scene["name"] = "solve"
        scene["roles"][0].update(
            name="solver",
            model=pinned_model,
            reasoning_effort=saved["reasoning_effort"],
        )
        scene["turns"] = [{"role": "solver", "has_prompt": True}]
        path.write_text(json.dumps(saved))
        path = trial / "result.json"
        path.write_text(
            json.dumps({**json.loads(path.read_text()), "model": pinned_model})
        )
    original = _files(old)
    config = _config(
        model="openai/fixture-new", reasoning_effort="high", prompts=prompts
    )
    job = _evaluation(tasks, old, config, selection)
    result = asyncio.run(job.run())
    reused = 0
    if prompts is None:
        reused = 2 if pins != "mixed" else int(selection == "named")
    assert (result.reused, result.ran) == (reused, 2 - reused)
    expected_models = {
        "task-0": pinned_model
        if pins != "mixed" and prompts is None
        else "openai/fixture-new",
        "task-1": pinned_model if prompts is None else "openai/fixture-new",
    }
    assert {
        n: r["model"] for n, r in load_task_results(job.job_dir).items()
    } == expected_models
    if reused == 2:
        assert job.job_dir == old
        assert not (old / SET_ASIDE_DIR).exists()
        _assert_trial_bytes(original, old)
    elif selection == "named":
        (kept,) = (old / SET_ASIDE_DIR).iterdir()
        for path, content in original.items():
            if path.parts[0].startswith("task-"):
                root = (
                    old if reused == 1 and path.parts[0].startswith("task-1") else kept
                )
                assert (root / path).read_bytes() == content
    else:
        assert _files(old) == original


def test_legacy_setting_failure_preserves_rows_and_redacts_reason(
    tmp_path, monkeypatch, caplog
):
    """Guards IU-R1-1 at 7bc81e6ba: refusal never discards or exposes old inputs."""
    tasks, old = _legacy_job(tmp_path)
    _edit_records(old, {"model": "password: fictional-old-value"})
    original = _files(old)
    job = _evaluation(tasks, old, _config(), "named")
    rename = Path.rename

    def refuse_trial_move(path, target):
        if path.parent == old and path.name.startswith("task-"):
            raise OSError("scripted move refusal")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", refuse_trial_move)
    with caplog.at_level(logging.WARNING), pytest.raises(ResumeMismatchError) as caught:
        asyncio.run(job.run())
    assert job._run_single_task.await_count == 0
    assert "model changed" in str(caught.value)
    assert "fictional-old-value" not in str(caught.value) + caplog.text
    assert not (old / "evaluation.json").exists()
    for path, content in original.items():
        assert (old / path).read_bytes() == content


def test_legacy_stored_task_path_is_not_followed(tmp_path):
    """Guards IU-R1-1 at 7bc81e6ba: current tasks, never saved absolute paths."""
    tasks, old = _legacy_job(tmp_path)
    outside = tmp_path / "previous-location" / "task-0"
    outside.mkdir(parents=True)
    (outside / "task.md").write_text("---\ninvalid: [\n---\n")
    path = old / "task-0__old00000" / "config.json"
    saved = json.loads(path.read_text())
    path.write_text(json.dumps({**saved, "task_path": str(outside)}))
    original = _files(old)
    job = _evaluation(tasks, old, _config(model="openai/fixture-new"), "named")
    result = asyncio.run(job.run())
    assert (result.reused, result.ran) == (0, 2)
    (kept,) = (old / SET_ASIDE_DIR).iterdir()
    _assert_trial_bytes(original, kept)
    assert (outside / "task.md").read_text() == "---\ninvalid: [\n---\n"


@pytest.mark.parametrize("selection", ["auto", "named"])
def test_job_record_comparison_remains_authoritative(tmp_path, selection):
    """Guards IU-R1-1 at 7bc81e6ba: the new-schema prompt comparison still runs."""
    tasks, old = _legacy_job(tmp_path)
    first = _evaluation(tasks, old, _config(), "named")
    assert asyncio.run(first.run()).reused == 2
    job = _evaluation(tasks, old, _config(prompts=["A different prompt."]), selection)
    result = asyncio.run(job.run())
    assert (result.reused, result.ran) == (0, 2)
    kept = old
    if selection == "named":
        (kept,) = (old / SET_ASIDE_DIR).iterdir()
    assert len(load_task_results(kept)) == 2


def _legacy_self_gen_job(tmp_path, *, job_mode, interrupted):
    """Author the final primary settings/scenes and learner snapshot from 0.7.11."""
    tasks, old = _legacy_job(tmp_path, interrupted=interrupted)
    sequential = job_mode == "sequential-shared"
    _edit_records(old, {"reasoning_effort": None})
    creator = tmp_path / "creator-root" / "skill-creator"
    creator.mkdir(parents=True)
    (creator / "SKILL.md").write_text(
        "---\nname: skill-creator\ndescription: Create skills\n---\nCreate skills.\n"
    )
    for task in sorted(tasks.iterdir()):
        (task / "instruction.md").write_text("Solve the task.\n")
        path = old / f"{task.name}__old00000" / "config.json"
        saved = json.loads(path.read_text())
        role = saved["scenes"][0]["roles"][0]
        if sequential:
            # Only task-1 pins its model: a flat model change affects task-0
            # alone, but must set aside BOTH trials and their learner store.
            learner_path = f"/tmp/legacy-learner/{task.name}-skills"
            saved.update(
                skill_mode="with-skill",
                skill_source="custom_runtime",
                requested_skills_dir=learner_path,
                effective_skills_dir=learner_path,
                skills_sandbox_dir="/skills",
            )
            if task.name == "task-1":
                (task / "task.md").write_text(
                    "---\nsandbox:\n  docker_image: python:3.12-slim\n"
                    "agents:\n  roles:\n    solver:\n      agent: codex\n"
                    "      model: openai/fixture-pinned\n"
                    "scenes:\n  - name: solve\n    roles: [solver]\n"
                    "---\n## prompt\nSolve the task.\n"
                )
                (task / "verifier").mkdir()
                (task / "verifier" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
                saved["model"] = "openai/fixture-pinned"
                role.update(name="solver", model=saved["model"])
                saved["scenes"][0].update(
                    name="solve", turns=[{"role": "solver", "has_prompt": True}]
                )
        else:
            # The old self-gen writer recorded the final roles' null effort,
            # even when the command requested high. Do not synthesize these
            # records through the candidate's resolver or rollout writer.
            saved.update(skill_mode="self-gen", skill_source="self_generated")
            saved["scenes"] = [
                {
                    "name": "self-gen-creator",
                    "skills_dir": "/tmp/legacy-self-gen/creator-skills",
                    "roles": [{**role, "name": "skill_creator"}],
                    "turns": [{"role": "skill_creator", "has_prompt": True}],
                },
                {
                    "name": "self-gen-solver",
                    "skills_dir": "/root/generated-skills",
                    "roles": [{**role, "name": "solver"}],
                    "turns": [{"role": "solver", "has_prompt": False}],
                },
            ]
        # The 0.7.11 Evaluation path also recorded a digest of task files.
        # Keep it matching so a model-only change cannot hide behind a digest.
        saved["task_digest"] = task_digest(task)
        path.write_text(json.dumps(saved))
        result_path = path.with_name("result.json")
        if result_path.exists():
            row = json.loads(result_path.read_text())
            row["model"] = saved["model"]
            result_path.write_text(json.dumps(row))
    if sequential:
        completed = 1 if interrupted else 2
        # The version-1 snapshot shape is shared with the 0.7.11 LearnerStore.
        history = [{"number": 0, "metric": None, "memory": {}, "skills": {}}]
        history.extend(
            {
                "number": n,
                "metric": 1.0,
                "memory": {"legacy": "retained lesson"},
                "skills": {"seed": "retained skill"},
            }
            for n in range(1, completed + 1)
        )
        (old / "learner_store.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "generation": completed,
                    "next_number": completed + 1,
                    "history": history,
                }
            )
        )
    return (
        tasks,
        old,
        _config(
            job_mode=job_mode,
            skill_mode="self-gen",
            skill_creator_dir=str(creator.parent),
            reasoning_effort=None if sequential else "high",
        ),
    )


@pytest.fixture
def final_rollouts(monkeypatch):
    """Observe Rollout.create AFTER real scheduling and self-gen transformations."""
    seen = []

    async def create(config):
        inherited = capture_skills(config.skills_dir) if config.skills_dir else {}
        seen.append((config, inherited))

        async def run():
            task_name = config.task_path.name
            trial = Path(config.jobs_dir) / config.job_name / f"{task_name}__new00000"
            trial.mkdir(parents=True)
            (trial / "config.json").write_text(
                json.dumps(
                    {
                        "task_path": task_name,
                        "agent": config.primary_agent,
                        "model": config.primary_model,
                        "reasoning_effort": config.primary_reasoning_effort,
                        "environment": config.environment,
                        "sandbox_user": config.sandbox_user,
                    }
                )
            )
            payload = {
                "task_name": task_name,
                "rollout_name": trial.name,
                "agent": config.primary_agent,
                "model": config.primary_model,
                "rewards": {"reward": 1.0},
                "error": None,
                "verifier_error": None,
            }
            (trial / "result.json").write_text(json.dumps(payload))
            return RolloutResult(
                **payload,
                rollout_dir=trial,
                evolved_skills={**inherited, f"from-{task_name}": "scripted update"},
            )

        return SimpleNamespace(run=run)

    # self_gen imports this same class; patch its factory, not run_self_gen or
    # Evaluation._run_single_task, so the final scene transformation is real.
    monkeypatch.setattr("benchflow.rollout.Rollout.create", create)
    return seen


@pytest.mark.parametrize("selection", ["auto", "named"])
@pytest.mark.parametrize("interrupted", [False, True], ids=["complete", "interrupted"])
@pytest.mark.parametrize("override", ["unchanged", "model", "explicit-prompts"])
@pytest.mark.parametrize("job_mode", ["parallel-independent", "sequential-shared"])
def test_legacy_self_gen_final_rollout_resume(
    tmp_path, selection, interrupted, override, job_mode, final_rollouts
):
    """Guards LR-R1-1/2 at 5e5043dac and IU-R1-1 at 7bc81e6ba via final configs."""
    tasks, old, config = _legacy_self_gen_job(
        tmp_path, job_mode=job_mode, interrupted=interrupted
    )
    original = _files(old)
    completed = 1 if interrupted else 2
    assert not (old / "evaluation.json").exists()
    assert len(load_job(old).trials) == completed
    assert _files(old) == original
    if override == "model":
        config.model = "openai/fixture-new"
    elif override == "explicit-prompts":
        # An explicitly supplied empty list suppresses document scenes too;
        # nonempty prompt overrides are covered by the ordinary scene cases.
        config.prompts = []
    job = Evaluation(
        tasks_dir=tasks,
        jobs_dir=old.parent,
        job_name=old.name if selection == "named" else None,
        config=config,
        preflight=False,
    )
    result = asyncio.run(job.run())
    sequential = job_mode == "sequential-shared"
    # Explicit prompts remove a sequential task's scene pin. Independent
    # self-gen still records the same primary values; legacy prompt text is
    # unknown and is not certified equal by this comparison.
    changed = override == "model" or (sequential and override == "explicit-prompts")
    expected = (0, 2) if changed else (completed, 2 - completed)
    assert (result.reused, result.ran) == expected
    assert len(final_rollouts) == expected[1]
    assert result.passed == 2
    models = {
        "task-0": config.model,
        "task-1": "openai/fixture-pinned"
        if sequential and config.prompts is None
        else config.model,
    }
    rows = load_task_results(job.job_dir)
    assert {name: row["model"] for name, row in rows.items()} == models
    for final, _ in final_rollouts:
        assert final.prompts == config.prompts
        assert final.primary_agent == "codex-acp"
        assert final.primary_model == models[final.task_path.name]
        assert final.primary_reasoning_effort is None
        if sequential:
            assert final.skill_mode == "with-skill"
            assert final.skills_agent_derived
        else:
            assert final.reasoning_effort == "high"
            assert final.skill_mode == "no-skill"
            assert [s.name for s in final.scenes] == [
                "self-gen-creator",
                "self-gen-solver",
            ]
            assert all(
                r.reasoning_effort is None for s in final.scenes for r in s.roles
            )

    kept = old
    if changed and selection == "named":
        assert job.job_dir == old
        (kept,) = (old / SET_ASIDE_DIR).iterdir()
    elif changed:
        assert job.job_dir != old
        assert _files(old) == original
    else:
        assert job.job_dir == old
        assert not (old / SET_ASIDE_DIR).exists()
    _assert_trial_bytes(original, kept)
    if job.job_dir == old:
        for name in ("summary.json", "results.jsonl"):
            assert (old / ".pre-0.8" / name).read_bytes() == original[Path(name)]

    if sequential:
        assert job.learner_store is not None
        store = job.learner_store.to_dict()
        assert store["generation"] == 2
        assert store["next_number"] == 3
        assert json.loads((job.job_dir / "learner_store.json").read_text()) == store
        if changed:
            assert (kept / "learner_store.json").read_bytes() == original[
                Path("learner_store.json")
            ]
            assert [skills for _, skills in final_rollouts] == [
                {},
                {"from-task-0": "scripted update"},
            ]
            assert job.learner_store.current().memory == {}
            assert job.learner_store.current().skills == {
                "from-task-0": "scripted update",
                "from-task-1": "scripted update",
            }
        else:
            saved = json.loads(original[Path("learner_store.json")])
            assert store["history"][: completed + 1] == saved["history"]
            assert job.learner_store.current().memory == {"legacy": "retained lesson"}
            skills = {"seed": "retained skill"}
            if interrupted:
                assert final_rollouts[0][1] == skills
                assert rows["task-1"]["learner_generation"] == {
                    "inherited_from": 1,
                    "produced": 2,
                    "committed": True,
                }
                skills["from-task-1"] = "scripted update"
            else:
                assert (old / "learner_store.json").read_bytes() == original[
                    Path("learner_store.json")
                ]
            assert job.learner_store.current().skills == skills


@pytest.mark.parametrize("selection", ["auto", "named"])
def test_self_gen_job_record_keeps_requested_effort_authority(
    tmp_path, selection, final_rollouts
):
    """Guards IU-R1-1/LR-R1-1 at 5e5043dac: known job inputs keep authority."""
    tasks, old, config = _legacy_self_gen_job(
        tmp_path, job_mode="parallel-independent", interrupted=False
    )
    first = Evaluation(
        tasks_dir=tasks,
        jobs_dir=old.parent,
        job_name=old.name,
        config=config,
        preflight=False,
    )
    first_result = asyncio.run(first.run())
    assert (first_result.reused, first_result.ran) == (2, 0)
    assert not final_rollouts
    recorded = json.loads((old / "evaluation.json").read_text())
    assert recorded["config"]["reasoning_effort"] == "high"
    original = _files(old)
    # The final self-gen role still has null effort, but evaluation.json knows
    # the requested effort. Its existing requested-setting contract wins.
    config.reasoning_effort = "low"
    job = Evaluation(
        tasks_dir=tasks,
        jobs_dir=old.parent,
        job_name=old.name if selection == "named" else None,
        config=config,
        preflight=False,
    )
    result = asyncio.run(job.run())
    assert (result.reused, result.ran) == (0, 2)
    assert result.passed == 2
    assert len(final_rollouts) == 2
    assert all(final.primary_reasoning_effort is None for final, _ in final_rollouts)
    kept = old
    if selection == "named":
        assert job.job_dir == old
        (kept,) = (old / SET_ASIDE_DIR).iterdir()
    else:
        assert job.job_dir != old
        assert _files(old) == original
    _assert_trial_bytes(original, kept)
