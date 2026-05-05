"""Single-trial orchestration for self-generated skills."""

from __future__ import annotations

import shlex
import shutil
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from benchflow.trial import (
    GENERATED_SKILLS_ROOT,
    SKILL_MODE_DEFAULT,
    Role,
    Scene,
    Trial,
    TrialConfig,
    Turn,
    _resolve_skill_creator_root,
    _safe_skill_name,
    _self_gen_prompt,
    _skill_frontmatter_name,
)


def _find_skill_creator_dir(skills_root: Path, skill_name: str) -> Path:
    """Find the one skill-creator directory inside a resolved skills root."""
    direct = skills_root / "skill-creator"
    if (direct / "SKILL.md").exists():
        return direct

    skill_dirs = [
        child
        for child in skills_root.iterdir()
        if child.is_dir() and (child / "SKILL.md").exists()
    ]
    for skill_dir in skill_dirs:
        if _skill_frontmatter_name(skill_dir) == skill_name:
            return skill_dir
    if len(skill_dirs) == 1:
        return skill_dirs[0]
    raise FileNotFoundError(
        f"Could not identify skill-creator under resolved skills root: {skills_root}"
    )


def _copy_single_skill(skill_dir: Path, dest_root: Path) -> Path:
    """Copy only one skill directory into a fresh skills root."""
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True)
    shutil.copytree(skill_dir, dest_root / skill_dir.name)
    return dest_root


def _self_gen_artifact_root(config: TrialConfig) -> Path:
    return (
        Path(config.jobs_dir)
        / "_self_gen"
        / f"{_safe_skill_name(config.task_path.name)}-{uuid4().hex[:8]}"
    )


def _creator_scene(
    config: TrialConfig, creator_skills_root: Path, skill_creator_name: str
) -> Scene:
    return Scene(
        name="self-gen-creator",
        roles=[Role("skill_creator", config.agent, config.model)],
        turns=[
            Turn(
                "skill_creator",
                _self_gen_prompt(
                    config.task_path,
                    config.generated_skills_root or GENERATED_SKILLS_ROOT,
                    skill_creator_name,
                ),
            )
        ],
        skills_dir=creator_skills_root,
    )


def _solver_scene(config: TrialConfig) -> Scene:
    return Scene(
        name="self-gen-solver",
        roles=[Role("solver", config.agent, config.model)],
        turns=[
            Turn("solver", prompt)
            for prompt in (
                config.prompts if config.prompts is not None else [None]
            )
        ],
        skills_dir=config.generated_skills_root or GENERATED_SKILLS_ROOT,
    )


def _ensure_generated_skills_hook(config: TrialConfig):
    generated_skills_root = config.generated_skills_root or GENERATED_SKILLS_ROOT

    async def ensure_generated_skills(env):
        q_root = shlex.quote(generated_skills_root)
        result = await env.exec(f"mkdir -p {q_root} && chmod 777 {q_root}", timeout_sec=10)
        if result.return_code != 0:
            raise RuntimeError(
                "Failed to create self-gen generated skills directory: "
                f"{result.stderr or result.stdout}"
            )

    return ensure_generated_skills


def _single_trial_config(
    config: TrialConfig, creator_skills_root: Path, skill_creator_name: str
) -> TrialConfig:
    return replace(
        config,
        scenes=[
            _creator_scene(config, creator_skills_root, skill_creator_name),
            _solver_scene(config),
        ],
        skills_dir=None,
        skill_mode=SKILL_MODE_DEFAULT,
        skill_creator_dir=None,
        pre_agent_hooks=[
            *(config.pre_agent_hooks or []),
            _ensure_generated_skills_hook(config),
        ],
        include_task_skills=False,
    )


async def run_self_gen(config: TrialConfig):
    """Run creator and solver as normal BYOS-style scenes in one trial.

    The creator scene gets only skill-creator. The solver scene gets only the
    generated skills root through the existing scene-local skills_dir mechanism.
    """
    skill_creator_root, skill_creator_name = _resolve_skill_creator_root(
        config.skill_creator_dir
    )
    skill_creator_dir = _find_skill_creator_dir(skill_creator_root, skill_creator_name)

    artifact_root = _self_gen_artifact_root(config)
    creator_skills_root = _copy_single_skill(
        skill_creator_dir, artifact_root / "creator-skills"
    )
    trial = await Trial.create(
        _single_trial_config(config, creator_skills_root, skill_creator_name)
    )
    return await trial.run()
