"""Environment setup utilities: Dockerfile preprocessing, DinD patching, environment creation."""

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths

from benchflow.agents.registry import AGENTS

logger = logging.getLogger(__name__)

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


def _detect_dind_mount() -> tuple[str, str] | None:
    """Detect Docker-in-Docker host path translation.

    When running inside a devcontainer that shares the host Docker socket,
    bind mount paths must be translated from container paths to host paths.

    Returns (host_source, container_dest) tuple, or None if not in DinD.
    """
    if not Path("/.dockerenv").exists():
        return None
    import subprocess as _sp

    try:
        hostname = _sp.check_output(["hostname"], text=True).strip()
        result = _sp.run(
            ["docker", "inspect", hostname],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        cwd = str(Path.cwd())
        best = None
        for mount in data[0].get("Mounts", []):
            if mount.get("Type") != "bind":
                continue
            dest = mount.get("Destination", "")
            if cwd.startswith(dest) and (best is None or len(dest) > len(best[1])):
                best = (mount["Source"], dest)
        return best
    except Exception:
        logger.debug("DinD mount detection failed", exc_info=True)
        return None


def _patch_harbor_dind() -> None:
    """Monkey-patch Harbor's DockerEnvironmentEnvVars for DinD path translation.

    When running inside a devcontainer, HOST_*_PATH env vars need to use
    host filesystem paths, not container paths. Applied once at import time.
    """
    dind_mount = _detect_dind_mount()
    if not dind_mount:
        return

    host_source, container_dest = dind_mount
    logger.info(f"DinD detected: {container_dest} → {host_source}")

    try:
        from harbor.environments.docker.docker import DockerEnvironmentEnvVars
    except ImportError:
        return

    _original = DockerEnvironmentEnvVars.to_env_dict

    def _patched(self, include_os_env=True):
        env = _original(self, include_os_env=include_os_env)
        for key in (
            "HOST_VERIFIER_LOGS_PATH",
            "HOST_AGENT_LOGS_PATH",
            "HOST_ARTIFACTS_PATH",
        ):
            val = env.get(key, "")
            if val.startswith(container_dest):
                env[key] = host_source + val[len(container_dest) :]
        return env

    # Monkey-patch Harbor's DockerEnvironmentEnvVars to rewrite host paths
    # for DinD nesting. ty flags this as an implicit signature shadowing.
    DockerEnvironmentEnvVars.to_env_dict = _patched  # ty: ignore[invalid-assignment]


def _create_environment(
    environment_type: str,
    task: Task,
    task_path: Path,
    trial_name: str,
    trial_paths: TrialPaths,
) -> Any:
    """Create a Harbor environment (Docker or Daytona)."""
    if environment_type == "docker":
        from harbor.environments.docker.docker import DockerEnvironment

        return DockerEnvironment(
            environment_dir=task.paths.environment_dir,
            environment_name=task_path.name,
            session_id=trial_name,
            trial_paths=trial_paths,
            task_env_config=task.config.environment,
        )
    elif environment_type == "daytona":
        from harbor.environments.daytona import DaytonaEnvironment

        return DaytonaEnvironment(
            environment_dir=task.paths.environment_dir,
            environment_name=task_path.name,
            session_id=trial_name,
            trial_paths=trial_paths,
            task_env_config=task.config.environment,
            auto_stop_interval_mins=1440,
            auto_delete_interval_mins=1440,
        )
    else:
        raise ValueError(
            f"Unknown environment_type: {environment_type!r} (use 'docker' or 'daytona')"
        )
