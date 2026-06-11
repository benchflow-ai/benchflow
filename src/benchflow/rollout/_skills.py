"""Skill-name and self-gen prompt helpers for :mod:`benchflow.rollout`.

Pure helpers that resolve the mounted skill-creator root, normalise generated
skill directory names, read a skill's frontmatter name, and build the self-gen
creator prompt. Split out of ``rollout.py`` for cohesion; re-exported from
:mod:`benchflow.rollout` so ``from benchflow.rollout import _self_gen_prompt``
(used by ``self_gen.py``) keeps resolving.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def _safe_skill_name(value: str) -> str:
    """Return an AgentSkills-compatible generated skill directory name."""
    name = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    name = re.sub(r"-+", "-", name).strip("-")
    return name or "generated-task"


def _skill_frontmatter_name(skill_dir: Path) -> str:
    """Read a skill's frontmatter name, falling back to the directory name."""
    from benchflow.skills import parse_skill

    info = parse_skill(skill_dir / "SKILL.md")
    return info.name if info and info.name else skill_dir.name


def _resolve_skill_creator_root(path: str | Path | None) -> tuple[Path, str]:
    """Resolve a skills root containing the official skill-creator directory.

    BenchFlow mounts skills as a root directory whose children are individual
    skill directories. If the caller points directly at skill-creator, use its
    parent as the mounted root and leave the skill pack contents unchanged.
    """
    candidates: list[Path] = []
    scan_single_skill_roots: set[Path] = set()
    if path:
        explicit_path = Path(path).expanduser()
        candidates.append(explicit_path)
        scan_single_skill_roots.add(explicit_path)
    env_path = os.environ.get("BENCHFLOW_SKILL_CREATOR_DIR")
    if env_path:
        env_candidate = Path(env_path).expanduser()
        candidates.append(env_candidate)
        scan_single_skill_roots.add(env_candidate)
    # ``__file__`` is ``src/benchflow/rollout/_skills.py``; ``parents[3]`` is the
    # repo root — matching the original ``rollout.py`` resolution which used
    # ``parents[2]`` from one directory shallower.
    repo_skill_creator = (
        Path(__file__).resolve().parents[3] / ".claude" / "skills" / "skill-creator"
    )
    cwd_skill_creator = Path.cwd() / ".claude" / "skills" / "skill-creator"
    candidates.append(repo_skill_creator)
    if cwd_skill_creator != repo_skill_creator:
        candidates.append(cwd_skill_creator)
    candidates.extend(
        [
            Path.home() / ".claude" / "skills" / "skill-creator",
            Path.home() / ".codex" / "skills" / ".system" / "skill-creator",
            Path.home() / ".agents" / "skills" / "skill-creator",
        ]
    )

    for candidate in candidates:
        if (candidate / "SKILL.md").exists():
            return candidate.parent, _skill_frontmatter_name(candidate)
        if (candidate / "skill-creator" / "SKILL.md").exists():
            skill_dir = candidate / "skill-creator"
            return candidate, _skill_frontmatter_name(skill_dir)
        if candidate in scan_single_skill_roots and candidate.is_dir():
            skill_dirs = [
                child
                for child in candidate.iterdir()
                if child.is_dir() and (child / "SKILL.md").exists()
            ]
            if len(skill_dirs) == 1:
                skill_dir = skill_dirs[0]
                return candidate, _skill_frontmatter_name(skill_dir)

    checked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find skill-creator. Pass --skill-creator-dir or set "
        f"BENCHFLOW_SKILL_CREATOR_DIR. Checked: {checked}"
    )


def _self_gen_prompt(
    task_path: Path, generated_skills_root: str, skill_creator_name: str
) -> str:
    """Prompt the clean creator agent to use the mounted skill-creator skill."""
    skill_dir_name = f"{_safe_skill_name(task_path.name)}-skill"
    target_dir = f"{generated_skills_root}/{skill_dir_name}"
    return f"""Use the {skill_creator_name} skill exactly as provided.

Read /instruction.md and inspect the task environment only as needed to understand the reusable workflow. Do not solve the task directly.

Create one or more complete Anthropic-standard skill packs as immediate child directories under:

{generated_skills_root}

Use this suggested path if one skill is enough:

{target_dir}

Each generated skill pack path must look like {generated_skills_root}/<skill-name>/SKILL.md. It may include scripts/, references/, assets/, examples, or other bundled resources when they help a fresh solver avoid repeated work.

The solver context will start with a clean agent session and only the generated skill packs mounted. Make the skills useful for solving this task type from the same sandbox environment."""
