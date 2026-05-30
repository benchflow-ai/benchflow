"""Skill injection policy for rollout setup.

Task-local ``environment/skills`` packs are agent capabilities, not ordinary
environment assets. This module owns the small bit of policy needed to decide
whether those packs are part of a rollout and whether the task build context
must be stripped so a no-skills run cannot pick them up through ``COPY .``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskSkillPolicy:
    bundled_dir: Path
    has_bundled_dir: bool
    use_bundled_dir: bool

    @property
    def prompt_bundled_dir(self) -> Path | None:
        return self.bundled_dir if self.use_bundled_dir else None

    @property
    def needs_task_copy(self) -> bool:
        return self.has_bundled_dir

    @property
    def strip_bundled_dir_from_copy(self) -> bool:
        return self.has_bundled_dir and not self.use_bundled_dir


def task_bundled_skills_dir(task_path: Path) -> Path:
    return task_path / "environment" / "skills"


def resolve_task_skill_policy(
    *,
    task_path: Path,
    runtime_skills_dir: str | Path | None,
    declared_sandbox_skills_dir: str | None,
    include_task_skills: bool,
) -> TaskSkillPolicy:
    bundled = task_bundled_skills_dir(task_path)
    has_bundled = bundled.is_dir()
    use_bundled = has_bundled and (
        _same_path(runtime_skills_dir, bundled)
        or (include_task_skills and bool(declared_sandbox_skills_dir))
    )
    return TaskSkillPolicy(
        bundled_dir=bundled,
        has_bundled_dir=has_bundled,
        use_bundled_dir=use_bundled,
    )


def strip_task_bundled_skills(task_path: Path) -> None:
    bundled = task_bundled_skills_dir(task_path)
    if bundled.exists():
        shutil.rmtree(bundled)


def _same_path(a: str | Path | None, b: Path) -> bool:
    if not a:
        return False
    try:
        return Path(a).resolve() == b.resolve()
    except OSError:
        return False
