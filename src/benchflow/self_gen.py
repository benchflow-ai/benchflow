"""Strict two-phase orchestration for self-generated skills."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from benchflow.models import RunResult
from benchflow.trial import (
    GENERATED_SKILLS_ROOT,
    SKILL_MODE_DEFAULT,
    Trial,
    TrialConfig,
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


def _bundle_generated_skills(
    export_root: Path, dest_root: Path, excluded_names: set[str] | None = None
) -> list[Path]:
    """Copy only generated child skill dirs that contain SKILL.md."""
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True)

    generated: list[Path] = []
    excluded_names = excluded_names or set()
    if not export_root.is_dir():
        return generated

    for child in sorted(export_root.iterdir()):
        if not child.is_dir() or not (child / "SKILL.md").exists():
            continue
        if child.name in excluded_names:
            continue
        target = dest_root / child.name
        shutil.copytree(child, target)
        generated.append(target)
    return generated


def _self_gen_artifact_root(config: TrialConfig) -> Path:
    return (
        Path(config.jobs_dir)
        / "_self_gen"
        / f"{_safe_skill_name(config.task_path.name)}-{uuid4().hex[:8]}"
    )


def _creator_trial_name(config: TrialConfig) -> str | None:
    if config.trial_name is None:
        return None
    return f"{config.trial_name}__skill-gen"


async def run_self_gen(config: TrialConfig) -> RunResult:
    """Run creator and solver as separate isolated trials.

    The creator trial succeeds only by producing at least one child directory
    with a SKILL.md. The solver result is the only successful final result.
    """
    skill_creator_root, skill_creator_name = _resolve_skill_creator_root(
        config.skill_creator_dir
    )
    skill_creator_dir = _find_skill_creator_dir(skill_creator_root, skill_creator_name)

    artifact_root = _self_gen_artifact_root(config)
    creator_skills_root = _copy_single_skill(
        skill_creator_dir, artifact_root / "creator-skills"
    )
    creator_export_root = artifact_root / "creator-export"
    solver_skills_root = artifact_root / "generated-skills"

    creator_config = replace(
        config,
        scenes=[],
        prompts=[
            _self_gen_prompt(
                config.task_path,
                config.generated_skills_root or GENERATED_SKILLS_ROOT,
                skill_creator_name,
            )
        ],
        skills_dir=creator_skills_root,
        skill_mode=SKILL_MODE_DEFAULT,
        trial_name=_creator_trial_name(config),
        skill_creator_dir=None,
        include_task_skills=False,
        skip_verify=True,
        export_generated_skills_to=creator_export_root,
    )
    creator_trial = await Trial.create(creator_config)
    await creator_trial.run()

    generated = _bundle_generated_skills(
        creator_export_root,
        solver_skills_root,
        excluded_names={"skill-creator", skill_creator_dir.name},
    )
    if not generated:
        return RunResult(
            task_name=config.task_path.name,
            error=(
                "Self-generated skill creator did not produce any skill "
                "directories containing SKILL.md"
            ),
        )

    solver_config = replace(
        config,
        scenes=[],
        prompts=config.prompts,
        skills_dir=solver_skills_root,
        skill_mode=SKILL_MODE_DEFAULT,
        skill_creator_dir=None,
        self_gen_no_internet=False,
        include_task_skills=False,
        skip_verify=False,
        export_generated_skills_to=None,
    )
    solver_trial = await Trial.create(solver_config)
    return await solver_trial.run()
