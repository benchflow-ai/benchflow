"""Environment setup utilities: Dockerfile preprocessing, skills injection, environment creation."""
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

from benchflow.task import Task
from benchflow.contracts.paths import TrialPaths
from benchflow.agents.registry import AGENTS

logger = logging.getLogger(__name__)

# Daytona's per-sandbox cap on the default tier is 4 CPU / 8 GB. Tasks declaring
# more fail at sandbox creation. Clamp here so tasks degrade gracefully (slower
# build) instead of erroring out. Override via env if running on a paid tier.
_DAYTONA_MAX_CPUS = int(os.environ.get("BENCHFLOW_DAYTONA_MAX_CPUS", "4"))
_DAYTONA_MAX_MEMORY_MB = int(os.environ.get("BENCHFLOW_DAYTONA_MAX_MEMORY_MB", "8192"))

# Directories to ignore when copying deps
_IGNORE_DIRS = {
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".git",
    ".mypy_cache",
    ".ruff_cache",
}


def _get_agent_skill_paths() -> list[str]:
    """Derive Dockerfile skill symlink targets from all agents' skill_paths.

    Returns deduplicated list of absolute paths (with $HOME resolved to /root).
    Only includes $HOME-based paths; $WORKSPACE paths are runtime-only.
    """
    paths: list[str] = []
    seen: set[str] = set()
    for cfg in AGENTS.values():
        for sp in cfg.skill_paths:
            if sp.startswith("$HOME/"):
                resolved = sp.replace("$HOME", "/root")
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append(resolved)
    return paths


def _dep_local_name(src_path: str) -> str:
    """Compute a short unique local name for a dependency path.

    packages/environments/claw-gmail  -> claw-gmail
    tasks/email-foo/environment/skills -> skills
    tasks/email-foo/data              -> email-foo__data
    """
    parts = Path(src_path).parts
    if len(parts) == 1:
        return parts[0]
    basename = parts[-1]
    if basename in ("data", "config", "src", "lib", "skills", "environment"):
        return f"{parts[-2]}__{basename}"
    return basename


def stage_dockerfile_deps(
    task_path: Path,
    context_root: Path,
) -> None:
    """Copy Dockerfile COPY sources into environment/_deps/ and rewrite paths.

    When a Dockerfile references files relative to the repo root (e.g.
    `COPY packages/environments/claw-gmail /app`), the Docker build context
    (set to environment/) won't find them. This function:

    1. Scans the Dockerfile for COPY instructions
    2. Copies each source from context_root into environment/_deps/
    3. Rewrites the COPY instruction to use the local _deps/ path

    Args:
        task_path: Path to the task directory (contains environment/Dockerfile)
        context_root: Path to the repo root where COPY sources are relative to
    """
    env_dir = task_path / "environment"
    dockerfile_path = env_dir / "Dockerfile"
    if not dockerfile_path.exists():
        return

    content = dockerfile_path.read_text()
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        copy_match = re.match(r"^(\s*COPY\s+(?:--\S+\s+)*)(\S+)\s+(\S+)\s*$", line)
        if copy_match:
            prefix = copy_match.group(1)
            src_path = copy_match.group(2)
            dst_path = copy_match.group(3)

            # Skip sources already relative to env dir, absolute, or using build args
            if src_path.startswith("/") or src_path.startswith("$") or src_path == ".":
                new_lines.append(line)
                continue

            abs_src = context_root / src_path
            if abs_src.exists():
                dep_name = _dep_local_name(src_path)
                local_dest = env_dir / "_deps" / dep_name

                if abs_src.is_dir():
                    if local_dest.exists():
                        shutil.rmtree(local_dest)
                    shutil.copytree(
                        abs_src,
                        local_dest,
                        ignore=shutil.ignore_patterns(*_IGNORE_DIRS),
                    )
                else:
                    local_dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(abs_src, local_dest)

                new_lines.append(f"{prefix}_deps/{dep_name} {dst_path}")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    dockerfile_path.write_text("\n".join(new_lines))


def _inject_skills_into_dockerfile(task_path: Path, skills_dir: Path) -> None:
    """Inject skills into the task's Dockerfile (baked into image).

    Copies skills_dir into environment/_deps/skills/ and appends COPY + symlink
    lines to the Dockerfile. This is more reliable than runtime upload since
    skills are part of the image.
    """
    env_dir = task_path / "environment"
    dockerfile_path = env_dir / "Dockerfile"
    if not dockerfile_path.exists() or not skills_dir.is_dir():
        return

    dest = env_dir / "_deps" / "skills"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(skills_dir, dest, ignore=shutil.ignore_patterns(*_IGNORE_DIRS))

    lines = [
        "",
        "# Skills directory (injected by benchflow --skills-dir)",
        "COPY _deps/skills /skills/",
    ]
    for agent_path in _get_agent_skill_paths():
        parent = str(Path(agent_path).parent)
        lines.append(f"RUN mkdir -p {parent} && ln -sf /skills {agent_path}")

    content = dockerfile_path.read_text()
    dockerfile_path.write_text(content + "\n".join(lines) + "\n")
    logger.info(
        f"Skills injected into Dockerfile: {len(list(skills_dir.iterdir()))} items"
    )


def _create_environment(
    environment_type: str,
    task: Task,
    task_path: Path,
    trial_name: str,
    trial_paths: TrialPaths,
) -> Any:
    """Create a benchflow sandbox (Docker or Daytona)."""
    if environment_type == "docker":
        from benchflow.sandbox.docker import DockerSandbox

        return DockerSandbox(
            environment_dir=task.paths.environment_dir,
            environment_name=task_path.name,
            session_id=trial_name,
            trial_paths=trial_paths,
            task_env_config=task.config.environment,
        )
    elif environment_type == "daytona":
        from benchflow.sandbox.daytona import DaytonaSandbox

        env_config = task.config.environment
        if env_config.cpus > _DAYTONA_MAX_CPUS:
            logger.warning(
                "Clamping cpus %d -> %d for Daytona (override with BENCHFLOW_DAYTONA_MAX_CPUS)",
                env_config.cpus,
                _DAYTONA_MAX_CPUS,
            )
            env_config.cpus = _DAYTONA_MAX_CPUS
        if env_config.memory_mb > _DAYTONA_MAX_MEMORY_MB:
            logger.warning(
                "Clamping memory_mb %d -> %d for Daytona (override with BENCHFLOW_DAYTONA_MAX_MEMORY_MB)",
                env_config.memory_mb,
                _DAYTONA_MAX_MEMORY_MB,
            )
            env_config.memory_mb = _DAYTONA_MAX_MEMORY_MB

        return DaytonaSandbox(
            environment_dir=task.paths.environment_dir,
            environment_name=task_path.name,
            session_id=trial_name,
            trial_paths=trial_paths,
            task_env_config=env_config,
            auto_stop_interval_mins=1440,
            auto_delete_interval_mins=1440,
        )
    else:
        raise ValueError(
            f"Unknown environment_type: {environment_type!r} (use 'docker' or 'daytona')"
        )
