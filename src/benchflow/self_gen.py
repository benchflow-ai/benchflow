"""Single-trial orchestration for self-generated skills."""

from __future__ import annotations

import contextlib
import shlex
import shutil
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from benchflow.models import RunResult
from benchflow.trial import (
    GENERATED_SKILLS_ROOT,
    SKILL_MODE_DEFAULT,
    Role,
    Scene,
    Trial,
    TrialConfig,
    Turn,
    _ensure_sandbox_dir,
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


def _single_trial_config(config: TrialConfig) -> TrialConfig:
    return replace(
        config,
        scenes=[],
        skills_dir=None,
        skill_mode=SKILL_MODE_DEFAULT,
        skill_creator_dir=None,
        include_task_skills=False,
        export_generated_skills_to=None,
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


async def _remove_remote_skill_dirs(
    env, generated_skills_root: str, excluded_names: set[str]
) -> None:
    if not excluded_names:
        return
    root = generated_skills_root.rstrip("/")
    targets = " ".join(shlex.quote(f"{root}/{name}") for name in sorted(excluded_names))
    result = await env.exec(f"rm -rf {targets}", timeout_sec=10)
    if result.return_code != 0:
        raise RuntimeError(
            "Failed to remove non-generated self-gen skill directories: "
            f"{result.stderr or result.stdout}"
        )


async def _remote_generated_skill_md_paths(env, generated_skills_root: str) -> list[str]:
    q_root = shlex.quote(generated_skills_root)
    result = await env.exec(
        f"find {q_root} -mindepth 2 -maxdepth 2 -type f -name SKILL.md | sort",
        timeout_sec=10,
    )
    if result.return_code != 0:
        raise RuntimeError(
            "Failed to inspect self-generated skills: "
            f"{result.stderr or result.stdout}"
        )
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def _self_gen_error(config: TrialConfig, error: str) -> RunResult:
    return RunResult(
        task_name=config.task_path.name,
        trial_name=config.trial_name or "",
        agent=config.agent,
        model=config.model or "",
        error=error,
    )


async def run_self_gen(config: TrialConfig) -> RunResult:
    """Run creator and solver in one trial with isolated ACP contexts.

    The creator context gets only skill-creator. The solver context gets only
    generated child directories containing SKILL.md, linked through the normal
    scene-local skills mechanism.
    """
    skill_creator_root, skill_creator_name = _resolve_skill_creator_root(
        config.skill_creator_dir
    )
    skill_creator_dir = _find_skill_creator_dir(skill_creator_root, skill_creator_name)

    artifact_root = _self_gen_artifact_root(config)
    creator_skills_root = _copy_single_skill(
        skill_creator_dir, artifact_root / "creator-skills"
    )
    generated_skills_root = config.generated_skills_root or GENERATED_SKILLS_ROOT
    excluded_names = {"skill-creator", skill_creator_dir.name}
    result: RunResult | None = None
    trial: Trial | None = None

    try:
        trial = await Trial.create(_single_trial_config(config))
        await trial.setup()
        await trial.start()
        await trial.install_agent()
        await _ensure_sandbox_dir(trial._env, generated_skills_root, config.sandbox_user)

        await trial._run_scene(
            _creator_scene(config, creator_skills_root, skill_creator_name)
        )
        await _remove_remote_skill_dirs(trial._env, generated_skills_root, excluded_names)
        generated = await _remote_generated_skill_md_paths(
            trial._env, generated_skills_root
        )
        if not generated:
            result = _self_gen_error(
                config,
                "Self-generated skill creator did not produce any skill "
                "directories containing SKILL.md",
            )
        else:
            await trial._run_scene(_solver_scene(config))
            if not config.skip_verify:
                await trial.verify()
    except Exception as exc:
        if trial is not None:
            with contextlib.suppress(Exception):
                trial._error = str(exc)
        else:
            result = _self_gen_error(config, str(exc))
    finally:
        if trial is not None:
            await trial.cleanup()

    if result is not None:
        return result
    if trial is None or trial._trial_dir is None:
        return _self_gen_error(
            config, "Self-gen failed before trial directory was created"
        )
    return trial._build_result()
