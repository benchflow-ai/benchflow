"""Strict self-generated skill orchestration tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.job import Job, JobConfig, RetryConfig
from benchflow.models import RunResult
from benchflow.runtime import run as runtime_run
from benchflow.sdk import SDK
from benchflow.trial import Trial, TrialConfig


def _make_task(tmp_path: Path) -> Path:
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('schema_version = "1.1"\n')
    (task / "instruction.md").write_text("solve the task\n")
    task_skills = task / "environment" / "skills" / "task-bundled"
    task_skills.mkdir(parents=True)
    (task_skills / "SKILL.md").write_text("# Task bundled\n")
    return task


def _make_skill_creator_root(tmp_path: Path) -> Path:
    root = tmp_path / "creator-root"
    skill_creator = root / "skill-creator"
    skill_creator.mkdir(parents=True)
    (skill_creator / "SKILL.md").write_text(
        "---\nname: skill-creator\ndescription: Create skills\n---\n# Skill Creator\n"
    )
    extra = root / "not-for-creator"
    extra.mkdir()
    (extra / "SKILL.md").write_text("# Must not be mounted\n")
    return root


def _skill_dir_names(root: Path) -> set[str]:
    return {child.name for child in root.iterdir() if child.is_dir()}


@pytest.mark.asyncio
async def test_sdk_self_gen_runs_creator_then_solver_in_fresh_trials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Self-gen creates skills in one trial and solves in a fresh with-skills trial."""
    task = _make_task(tmp_path)
    skill_creator_root = _make_skill_creator_root(tmp_path)
    original_skills = tmp_path / "original-skills"
    original_skills.mkdir()
    (original_skills / "original" / "SKILL.md").parent.mkdir()
    (original_skills / "original" / "SKILL.md").write_text("# Original\n")

    seen_configs = []
    solver_result = RunResult(
        task_name="task",
        trial_name="solver",
        rewards={"reward": 1.0},
        agent="opencode",
        model="google/gemini-3.1-pro-preview",
        n_tool_calls=7,
    )

    async def fake_create(config):
        seen_configs.append(config)
        trial = AsyncMock()
        if getattr(config, "skip_verify", False):

            async def creator_run():
                export_root = Path(config.export_generated_skills_to)
                (export_root / "skill-a").mkdir(parents=True)
                (export_root / "skill-a" / "SKILL.md").write_text("# Skill A\n")
                (export_root / "skill-b").mkdir()
                (export_root / "skill-b" / "SKILL.md").write_text("# Skill B\n")
                (export_root / "notes.txt").write_text("not a skill\n")
                (export_root / "not-a-skill").mkdir()
                (export_root / "skill-creator").mkdir()
                (export_root / "skill-creator" / "SKILL.md").write_text(
                    "# Must not reach solver\n"
                )
                return RunResult(task_name="task", error="ignored creator error")

            trial.run = AsyncMock(side_effect=creator_run)
        else:
            trial.run = AsyncMock(return_value=solver_result)
        return trial

    monkeypatch.setattr("benchflow.trial.Trial.create", fake_create)

    result = await SDK().run(
        task_path=task,
        agent="opencode",
        model="google/gemini-3.1-pro-preview",
        agent_env={"GOOGLE_API_KEY": "secret"},
        job_name="job-1",
        trial_name="trial-1",
        jobs_dir=tmp_path / "jobs",
        environment="daytona",
        skills_dir=original_skills,
        sandbox_user="worker",
        sandbox_locked_paths=["/solution"],
        sandbox_setup_timeout=321,
        context_root=tmp_path,
        skill_mode="self-gen",
        skill_creator_dir=skill_creator_root,
        self_gen_no_internet=True,
    )

    assert result is solver_result
    assert len(seen_configs) == 2

    creator_cfg, solver_cfg = seen_configs
    assert creator_cfg.task_path == task
    assert creator_cfg.agent == "opencode"
    assert creator_cfg.model == "google/gemini-3.1-pro-preview"
    assert creator_cfg.environment == "daytona"
    assert creator_cfg.agent_env == {"GOOGLE_API_KEY": "secret"}
    assert creator_cfg.sandbox_user == "worker"
    assert creator_cfg.sandbox_locked_paths == ["/solution"]
    assert creator_cfg.sandbox_setup_timeout == 321
    assert creator_cfg.context_root == tmp_path
    assert creator_cfg.trial_name == "trial-1__skill-gen"
    assert creator_cfg.self_gen_no_internet is True
    assert creator_cfg.skip_verify is True
    assert creator_cfg.include_task_skills is False
    assert creator_cfg.skill_mode == "default"
    assert creator_cfg.skills_dir != original_skills
    assert _skill_dir_names(Path(creator_cfg.skills_dir)) == {"skill-creator"}
    assert creator_cfg.prompts and creator_cfg.prompts[0] is not None
    assert "skill-creator" in creator_cfg.prompts[0]
    assert str(original_skills) not in creator_cfg.prompts[0]

    assert solver_cfg.task_path == task
    assert solver_cfg.agent == "opencode"
    assert solver_cfg.model == "google/gemini-3.1-pro-preview"
    assert solver_cfg.environment == "daytona"
    assert solver_cfg.agent_env == {"GOOGLE_API_KEY": "secret"}
    assert solver_cfg.sandbox_user == "worker"
    assert solver_cfg.sandbox_locked_paths == ["/solution"]
    assert solver_cfg.sandbox_setup_timeout == 321
    assert solver_cfg.context_root == tmp_path
    assert solver_cfg.trial_name == "trial-1"
    assert solver_cfg.skill_mode == "default"
    assert solver_cfg.skill_creator_dir is None
    assert solver_cfg.self_gen_no_internet is False
    assert solver_cfg.skip_verify is False
    assert solver_cfg.include_task_skills is False
    assert solver_cfg.prompts is None
    assert _skill_dir_names(Path(solver_cfg.skills_dir)) == {"skill-a", "skill-b"}
    assert not (Path(solver_cfg.skills_dir) / "not-a-skill").exists()
    assert not (Path(solver_cfg.skills_dir) / "skill-creator").exists()


@pytest.mark.asyncio
async def test_job_self_gen_uses_strict_orchestration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Batch jobs route self-gen through the same two-trial orchestration."""
    task = _make_task(tmp_path)
    skill_creator_root = _make_skill_creator_root(tmp_path)
    job = Job(
        tasks_dir=task.parent,
        jobs_dir=tmp_path / "jobs",
        config=JobConfig(
            agent="opencode",
            model="google/gemini-3.1-pro-preview",
            environment="docker",
            retry=RetryConfig(max_retries=0),
            skill_mode="self-gen",
            skill_creator_dir=str(skill_creator_root),
        ),
    )

    seen_configs = []
    solver_result = RunResult(task_name="task", rewards={"reward": 1.0})

    async def fake_create(config):
        seen_configs.append(config)
        trial = AsyncMock()
        if getattr(config, "skip_verify", False):

            async def creator_run():
                export_root = Path(config.export_generated_skills_to)
                (export_root / "generated").mkdir(parents=True)
                (export_root / "generated" / "SKILL.md").write_text("# Generated\n")
                return RunResult(task_name="task")

            trial.run = AsyncMock(side_effect=creator_run)
        else:
            trial.run = AsyncMock(return_value=solver_result)
        return trial

    monkeypatch.setattr("benchflow.trial.Trial.create", fake_create)

    result = await job._run_single_task(task, job._config)

    assert result is solver_result
    assert len(seen_configs) == 2
    assert seen_configs[0].skip_verify is True
    assert seen_configs[0].include_task_skills is False
    assert seen_configs[1].skip_verify is False
    assert seen_configs[1].include_task_skills is False
    assert Path(seen_configs[1].skills_dir).is_dir()


@pytest.mark.asyncio
async def test_self_gen_requires_at_least_one_generated_skill_md(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Creator success only requires SKILL.md, but at least one is mandatory."""
    task = _make_task(tmp_path)
    skill_creator_root = _make_skill_creator_root(tmp_path)
    seen_configs = []

    async def fake_create(config):
        seen_configs.append(config)
        trial = AsyncMock()

        async def creator_run():
            export_root = Path(config.export_generated_skills_to)
            (export_root / "notes-only").mkdir(parents=True)
            (export_root / "notes-only" / "README.md").write_text("not enough\n")
            return RunResult(task_name="task")

        trial.run = AsyncMock(side_effect=creator_run)
        return trial

    monkeypatch.setattr("benchflow.trial.Trial.create", fake_create)

    result = await SDK().run(
        task_path=task,
        agent="opencode",
        model="google/gemini-3.1-pro-preview",
        jobs_dir=tmp_path / "jobs",
        skill_mode="self-gen",
        skill_creator_dir=skill_creator_root,
    )

    assert len(seen_configs) == 1
    assert result.rewards is None
    assert result.error is not None
    assert "SKILL.md" in result.error


@pytest.mark.asyncio
async def test_runtime_trial_config_self_gen_routes_to_orchestrator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """bf.run(TrialConfig(... self-gen ...)) also uses strict orchestration."""
    task = _make_task(tmp_path)
    expected = RunResult(task_name="task", rewards={"reward": 1.0})
    captured = {}

    async def fake_run_self_gen(config):
        captured["config"] = config
        return expected

    monkeypatch.setattr("benchflow.self_gen.run_self_gen", fake_run_self_gen)
    config = TrialConfig.from_legacy(
        task_path=task,
        agent="opencode",
        model="google/gemini-3.1-pro-preview",
        skill_mode="self-gen",
    )

    result = await runtime_run(config)

    assert result is expected
    assert captured["config"] is config


@pytest.mark.asyncio
async def test_trial_create_rejects_self_gen_without_orchestrator(
    tmp_path: Path,
) -> None:
    """Direct Trial execution cannot silently collapse self-gen into one sandbox."""
    task = _make_task(tmp_path)
    config = TrialConfig.from_legacy(
        task_path=task,
        agent="opencode",
        model="google/gemini-3.1-pro-preview",
        skill_mode="self-gen",
    )

    with pytest.raises(ValueError, match="two-phase orchestrator"):
        await Trial.create(config)

    with pytest.raises(ValueError, match="two-phase orchestrator"):
        await Trial(config).run()
