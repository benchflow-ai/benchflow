"""Tests for self-generated skill mode CLI wiring."""

from pathlib import Path
from unittest.mock import patch

from benchflow.models import RunResult


def _make_task(tmp_path: Path) -> Path:
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('schema_version = "1.1"\n')
    (task / "instruction.md").write_text("solve\n")
    return task


def test_eval_create_single_task_self_gen_passes_trial_config(tmp_path: Path):
    """`bench eval create --mode self-gen` reaches strict self-gen orchestration."""
    import asyncio

    from benchflow.cli.main import eval_create

    task = _make_task(tmp_path)
    skill_creator = tmp_path / "skills" / "skill-creator"
    skill_creator.mkdir(parents=True)
    (skill_creator / "SKILL.md").write_text(
        "---\nname: skill-creator\ndescription: Create skills\n---\n# Skill Creator\n"
    )
    provided_skills = tmp_path / "provided-skills"
    provided_skills.mkdir()
    captured = {}

    async def fake_run_self_gen(config):
        captured["config"] = config
        return RunResult(
            task_name="task",
            agent_name="claude-agent-acp",
            rewards={"reward": 1.0},
            n_tool_calls=0,
        )

    try:
        with patch("benchflow.self_gen.run_self_gen", new=fake_run_self_gen):
            eval_create(
                config_file=None,
                tasks_dir=task,
                agent="claude-agent-acp",
                model="claude-haiku-4-5-20251001",
                environment="docker",
                concurrency=1,
                jobs_dir=str(tmp_path / "jobs"),
                sandbox_user="agent",
                sandbox_setup_timeout=120,
                skills_dir=provided_skills,
                skill_mode="self-gen",
                skill_creator_dir=skill_creator,
                self_gen_no_internet=True,
            )
    finally:
        asyncio.set_event_loop(asyncio.new_event_loop())

    cfg = captured["config"]
    assert cfg.skill_mode == "self-gen"
    assert cfg.skills_dir == str(provided_skills)
    assert cfg.skill_creator_dir == str(skill_creator)
    assert cfg.self_gen_no_internet is True
