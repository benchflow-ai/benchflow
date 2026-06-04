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

SKILL_MODE_NO_SKILL = "no-skill"
SKILL_MODE_WITH_SKILL = "with-skill"
SKILL_MODE_SELF_GEN = "self-gen"
SKILL_MODES = frozenset(
    {
        SKILL_MODE_NO_SKILL,
        SKILL_MODE_WITH_SKILL,
        SKILL_MODE_SELF_GEN,
    }
)

SKILL_SOURCE_NONE = "none"
SKILL_SOURCE_TASK_BUNDLED = "task_bundled"
SKILL_SOURCE_CUSTOM_RUNTIME = "custom_runtime"
SKILL_SOURCE_SELF_GENERATED = "self_generated"


@dataclass(frozen=True)
class TaskSkillPolicy:
    mode: str
    source: str
    bundled_dir: Path
    has_bundled_dir: bool
    enabled: bool
    requested_dir: str | None
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

    @property
    def include_task_skills(self) -> bool:
        return self.source == SKILL_SOURCE_TASK_BUNDLED and self.enabled

    def config_metadata(self) -> dict[str, str | bool | None]:
        return {
            "skill_mode": self.mode,
            "skill_source": self.source,
            "requested_skills_dir": self.requested_dir,
            "effective_skills_dir": str(self.host_dir) if self.host_dir else None,
            "skills_sandbox_dir": self.sandbox_dir,
            "include_task_skills": self.include_task_skills,
        }


def task_bundled_skills_dir(task_path: Path) -> Path:
    return task_path / "environment" / "skills"


def normalize_skill_mode(value: str | None) -> str:
    if value is None:
        return SKILL_MODE_NO_SKILL
    text = value.strip()
    if text in SKILL_MODES:
        return text
    expected = ", ".join(sorted(SKILL_MODES))
    raise ValueError(f"skill_mode must be one of: {expected}")


def resolve_runtime_skills_dir(skills_dir: str | Path | None) -> Path | None:
    if skills_dir is None:
        return None
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
    skill_mode: str,
    runtime_skills_dir: str | Path | None,
    declared_sandbox_skills_dir: str | None,
) -> TaskSkillPolicy:
    mode = normalize_skill_mode(skill_mode)
    bundled = task_bundled_skills_dir(task_path)
    has_bundled = bundled.is_dir()
    resolved_runtime = resolve_runtime_skills_dir(runtime_skills_dir)
    requested = str(runtime_skills_dir) if runtime_skills_dir is not None else None

    enabled = False
    source = SKILL_SOURCE_NONE
    host_dir: Path | None = None
    sandbox_dir: str | None = None
    strip_bundled = has_bundled

    if mode == SKILL_MODE_NO_SKILL:
        if runtime_skills_dir is not None:
            raise ValueError("no-skill mode cannot be combined with skills_dir")
    elif mode == SKILL_MODE_SELF_GEN:
        if runtime_skills_dir is not None:
            raise ValueError("self-gen mode cannot be combined with skills_dir")
        source = SKILL_SOURCE_SELF_GENERATED
    elif resolved_runtime is not None:
        if not resolved_runtime.is_dir():
            raise FileNotFoundError(f"skills_dir not found: {resolved_runtime}")
        enabled = True
        source = (
            SKILL_SOURCE_TASK_BUNDLED
            if _same_path(resolved_runtime, bundled)
            else SKILL_SOURCE_CUSTOM_RUNTIME
        )
        host_dir = resolved_runtime
        if source == SKILL_SOURCE_TASK_BUNDLED:
            sandbox_dir = validate_container_mount_path(
                declared_sandbox_skills_dir or "/skills",
                "environment.skills_dir",
            )
        else:
            sandbox_dir = validate_container_mount_path("/skills")
        strip_bundled = has_bundled and not _same_path(resolved_runtime, bundled)
    elif mode == SKILL_MODE_WITH_SKILL:
        if not has_bundled:
            raise FileNotFoundError(f"task has no bundled skills: {bundled}")
        enabled = True
        source = SKILL_SOURCE_TASK_BUNDLED
        host_dir = bundled
        sandbox_dir = validate_container_mount_path(
            declared_sandbox_skills_dir or "/skills",
            "environment.skills_dir",
        )
        strip_bundled = False

    return TaskSkillPolicy(
        mode=mode,
        source=source,
        bundled_dir=bundled,
        has_bundled_dir=has_bundled,
        enabled=enabled,
        requested_dir=requested,
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
