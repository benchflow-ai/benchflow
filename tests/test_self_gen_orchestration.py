"""Strict self-generated skill orchestration tests."""

from __future__ import annotations

from pathlib import Path

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
async def test_sdk_self_gen_runs_creator_then_solver_in_one_trial_with_isolated_contexts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Guards PR #233: self-gen is one BYOS-style trial with two scenes."""
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
    run_configs = []

    class FakeTrial:
        def __init__(self, config):
            self._config = config

        @classmethod
        async def create(cls, config):
            seen_configs.append(config)
            return cls(config)

        async def run(self):
            run_configs.append(self._config)
            return solver_result

    monkeypatch.setattr("benchflow.self_gen.Trial", FakeTrial)

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
    assert len(seen_configs) == 1
    assert run_configs == seen_configs

    trial_cfg = seen_configs[0]
    assert trial_cfg.task_path == task
    assert trial_cfg.agent == "opencode"
    assert trial_cfg.model == "google/gemini-3.1-pro-preview"
    assert trial_cfg.environment == "daytona"
    assert trial_cfg.agent_env == {"GOOGLE_API_KEY": "secret"}
    assert trial_cfg.sandbox_user == "worker"
    assert trial_cfg.sandbox_locked_paths == ["/solution"]
    assert trial_cfg.sandbox_setup_timeout == 321
    assert trial_cfg.context_root == tmp_path
    assert trial_cfg.trial_name == "trial-1"
    assert trial_cfg.skill_mode == "default"
    assert trial_cfg.skill_creator_dir is None
    assert trial_cfg.include_task_skills is False
    assert trial_cfg.skills_dir is None
    assert trial_cfg.prompts is None
    assert trial_cfg.self_gen_no_internet is True
    assert trial_cfg.export_generated_skills_to is None

    assert [scene.name for scene in trial_cfg.scenes] == [
        "self-gen-creator",
        "self-gen-solver",
    ]
    creator_scene = trial_cfg.scenes[0]
    solver_scene = trial_cfg.scenes[1]

    assert creator_scene.name == "self-gen-creator"
    assert _skill_dir_names(Path(creator_scene.skills_dir)) == {"skill-creator"}
    assert creator_scene.turns[0].prompt is not None
    assert "skill-creator" in creator_scene.turns[0].prompt
    assert str(original_skills) not in creator_scene.turns[0].prompt

    assert solver_scene.name == "self-gen-solver"
    assert solver_scene.skills_dir == "/app/generated-skills"
    assert solver_scene.turns[0].prompt is None


@pytest.mark.asyncio
async def test_job_self_gen_uses_strict_orchestration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Guards PR #233: batch self-gen delegates to the BYOS scene trial."""
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
    run_configs = []
    solver_result = RunResult(task_name="task", rewards={"reward": 1.0})

    class FakeTrial:
        def __init__(self, config):
            self._config = config

        @classmethod
        async def create(cls, config):
            seen_configs.append(config)
            return cls(config)

        async def run(self):
            run_configs.append(self._config)
            return solver_result

    monkeypatch.setattr("benchflow.self_gen.Trial", FakeTrial)

    result = await job._run_single_task(task, job._config)

    assert result is solver_result
    assert len(seen_configs) == 1
    assert run_configs == seen_configs
    assert seen_configs[0].skip_verify is False
    assert seen_configs[0].include_task_skills is False
    assert seen_configs[0].skills_dir is None
    assert [scene.name for scene in seen_configs[0].scenes] == [
        "self-gen-creator",
        "self-gen-solver",
    ]


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

    with pytest.raises(ValueError, match="runtime orchestrator"):
        await Trial.create(config)

    with pytest.raises(ValueError, match="runtime orchestrator"):
        await Trial(config).run()
