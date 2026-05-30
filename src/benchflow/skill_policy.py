"""Skill injection policy for rollout setup.

Task-local ``environment/skills`` packs are agent capabilities, not ordinary
environment assets. This module owns the small bit of policy needed to decide
whether those packs are part of a rollout and whether the task build context
must be stripped so a no-skills run cannot pick them up through ``COPY .``.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

_CONTAINER_MOUNT_PATH_RE = re.compile(r"^/[A-Za-z0-9._/-]+$")


@dataclass(frozen=True)
class TaskSkillPolicy:
    bundled_dir: Path
    has_bundled_dir: bool
    enabled: bool
    host_dir: Path | None
    sandbox_dir: str | None
    strip_bundled_dir_from_copy: bool

    @property
    def prompt_dir(self) -> Path | None:
        return self.host_dir

    @property
    def needs_task_copy(self) -> bool:
        return self.has_bundled_dir or self.host_dir is not None

    @property
    def host_dir_is_bundled(self) -> bool:
        return self.host_dir is not None and _same_path(self.host_dir, self.bundled_dir)


def task_bundled_skills_dir(task_path: Path) -> Path:
    return task_path / "environment" / "skills"


def resolve_runtime_skills_dir(
    task_path: Path,
    skills_dir: str | Path | None,
) -> Path | None:
    if skills_dir is None:
        return None
    if str(skills_dir) == "auto":
        bundled = task_bundled_skills_dir(task_path)
        return bundled if bundled.is_dir() else None
    return skills_dir if isinstance(skills_dir, Path) else Path(skills_dir)


def validate_container_mount_path(value: object, field: str = "sandbox_dir") -> str:
    """Validate a simple absolute POSIX path inside a sandbox container."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if value != value.strip():
        raise ValueError(f"{field} must be a simple absolute container path")

    path = value.rstrip("/")
    if path == "" or path == "/" or not _CONTAINER_MOUNT_PATH_RE.fullmatch(path):
        raise ValueError(f"{field} must be a simple absolute container path")
    parts = path.split("/")[1:]
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{field} must be a simple absolute container path")
    return path


def resolve_task_skill_policy(
    *,
    task_path: Path,
    runtime_skills_dir: str | Path | None,
    declared_sandbox_skills_dir: str | None,
    include_task_skills: bool,
) -> TaskSkillPolicy:
    bundled = task_bundled_skills_dir(task_path)
    has_bundled = bundled.is_dir()
    resolved_runtime = resolve_runtime_skills_dir(task_path, runtime_skills_dir)

    enabled = False
    host_dir: Path | None = None
    sandbox_dir: str | None = None
    strip_bundled = has_bundled

    if resolved_runtime is not None:
        enabled = True
        host_dir = resolved_runtime
        sandbox_dir = validate_container_mount_path("/skills")
        strip_bundled = has_bundled and not _same_path(resolved_runtime, bundled)
    elif include_task_skills and has_bundled:
        enabled = True
        host_dir = bundled
        sandbox_dir = validate_container_mount_path(
            declared_sandbox_skills_dir or "/skills",
            "environment.skills_dir",
        )
        strip_bundled = False

    return TaskSkillPolicy(
        bundled_dir=bundled,
        has_bundled_dir=has_bundled,
        enabled=enabled,
        host_dir=host_dir,
        sandbox_dir=sandbox_dir,
        strip_bundled_dir_from_copy=strip_bundled,
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
