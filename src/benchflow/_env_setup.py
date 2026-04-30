"""Environment setup utilities: Dockerfile preprocessing, DinD patching, environment creation."""

import base64
import json
import logging
import os
import re
import shlex
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths

from benchflow.agents.registry import AGENTS

logger = logging.getLogger(__name__)

# Daytona's per-sandbox cap on the default tier is 4 CPU / 8 GB. Tasks declaring
# more fail at sandbox creation. Clamp here so tasks degrade gracefully (slower
# build) instead of erroring out. Override via env if running on a paid tier.
_DAYTONA_MAX_CPUS = int(os.environ.get("BENCHFLOW_DAYTONA_MAX_CPUS", "4"))
_DAYTONA_MAX_MEMORY_MB = int(os.environ.get("BENCHFLOW_DAYTONA_MAX_MEMORY_MB", "8192"))
_DAYTONA_MAX_STORAGE_MB = int(
    os.environ.get("BENCHFLOW_DAYTONA_MAX_STORAGE_MB", "10240")
)

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


_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?([A-Za-z0-9_.-]+)['\"]?")
_MODAL_PYTHON_SYMLINK = (
    "RUN if command -v python3 >/dev/null 2>&1 "
    "&& ! command -v python >/dev/null 2>&1; then "
    'ln -sf "$(command -v python3)" /usr/local/bin/python || true; fi'
)


def _modal_dockerfile_uses_python_base(dockerfile_path: Path) -> bool:
    """Return True if any Dockerfile stage is based on a Python image."""
    try:
        lines = dockerfile_path.read_text().splitlines()
    except OSError:
        return False

    for line in lines:
        parts = line.strip().split()
        if not parts or parts[0].upper() != "FROM":
            continue
        image_parts = [part for part in parts[1:] if not part.startswith("--")]
        if not image_parts:
            continue
        image = image_parts[0].lower()
        if image.startswith("python:") or "/python:" in image:
            return True
    return False


def _modal_dockerfile_installs_python(dockerfile_path: Path) -> bool:
    try:
        text = dockerfile_path.read_text().lower()
    except OSError:
        return False
    return any(
        token in text
        for token in (
            "python3-pip",
            "python-is-python3",
            "pip3 install",
            "python3 -m pip",
            "pip install",
        )
    )


def _modal_add_python_version(dockerfile_path: Path) -> str | None:
    """Python version Modal should add to Dockerfile images that lack Python."""
    configured = os.environ.get("BENCHFLOW_MODAL_ADD_PYTHON")
    if configured is None and (
        _modal_dockerfile_uses_python_base(dockerfile_path)
        or _modal_dockerfile_installs_python(dockerfile_path)
    ):
        return None
    value = (configured if configured is not None else "3.12").strip()
    return value or None


def _modal_heredoc_decoder(body: str) -> str:
    encoded = base64.b64encode(body.encode()).decode()
    code = f"import base64,sys;sys.stdout.buffer.write(base64.b64decode({encoded!r}))"
    return f"python3 -c {shlex.quote(code)}"


def _modal_rewrite_heredoc_line(line: str, match: re.Match[str], body: str) -> str:
    before = line[: match.start()].rstrip()
    decoder = _modal_heredoc_decoder(body)

    cat_match = re.search(r"cat\s*>\s*(?P<target>\S+)\s*$", before)
    if cat_match:
        target = cat_match.group("target")
        return f"{before[: cat_match.start()]}{decoder} > {target}"

    stripped = before.strip()
    if stripped.upper() == "RUN":
        shell = "/bin/bash" if "pipefail" in body else "/bin/sh"
        return f"RUN {decoder} | {shell}"
    if stripped.upper().startswith("RUN "):
        command = stripped[3:].strip()
        if command.endswith("-"):
            command = command[:-1].rstrip()
        return f"RUN {decoder} | {command}"
    return line


def _modal_rewrite_dockerfile_heredocs(text: str) -> str:
    """Rewrite Dockerfile heredocs into syntax accepted by Modal's parser."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        match = _HEREDOC_RE.search(line)
        if not match:
            out.append(line)
            i += 1
            continue

        delimiter = match.group(1)
        body_lines: list[str] = []
        i += 1
        while i < len(lines) and lines[i].strip() != delimiter:
            body_lines.append(lines[i])
            i += 1

        if i == len(lines):
            out.append(line)
            out.extend(body_lines)
            break

        body = "\n".join(body_lines) + "\n"
        out.append(_modal_rewrite_heredoc_line(line, match, body))
        i += 1

    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(out) + suffix


def _modal_builder_dockerfile(
    dockerfile_path: Path,
) -> tuple[Path, Callable[[], None]]:
    text = dockerfile_path.read_text()
    rewritten = _modal_rewrite_dockerfile_heredocs(text)
    if (
        not _modal_dockerfile_uses_python_base(dockerfile_path)
        and _modal_dockerfile_installs_python(dockerfile_path)
        and _MODAL_PYTHON_SYMLINK not in rewritten
    ):
        rewritten = rewritten.rstrip() + "\n\n" + _MODAL_PYTHON_SYMLINK + "\n"
    if rewritten == text:
        return dockerfile_path, lambda: None

    tmpdir = tempfile.TemporaryDirectory(prefix="benchflow-modal-dockerfile-")
    tmp_path = Path(tmpdir.name) / "Dockerfile"
    tmp_path.write_text(rewritten)
    return tmp_path, tmpdir.cleanup


def _create_benchflow_modal_environment_class():
    """Create a ModalEnvironment subclass with BenchFlow's image-build defaults."""
    from harbor.environments.modal import ModalEnvironment

    class BenchFlowModalEnvironment(ModalEnvironment):
        async def start(self, force_build: bool) -> None:
            """Starts the Modal sandbox, adding Python for plain Linux images."""
            from harbor.models.trial.paths import EnvironmentPaths
            from modal import App, Image, Secret, Volume

            def noop_cleanup_dockerfile() -> None:
                return None

            cleanup_dockerfile: Callable[[], None] = noop_cleanup_dockerfile
            docker_image = self.task_env_config.docker_image

            if docker_image:
                registry_secret = (
                    Secret.from_name(self._registry_secret)
                    if self._registry_secret
                    else None
                )
                if ".dkr.ecr." in docker_image:
                    self._image = Image.from_aws_ecr(
                        docker_image,
                        secret=registry_secret,
                    )
                else:
                    self._image = Image.from_registry(
                        docker_image,
                        secret=registry_secret,
                    )
            else:
                dockerfile_path = self._environment_definition_path
                modal_dockerfile_path, cleanup_dockerfile = _modal_builder_dockerfile(
                    dockerfile_path
                )
                self._image = Image.from_dockerfile(
                    modal_dockerfile_path,
                    force_build=force_build,
                    context_dir=self.environment_dir,
                    add_python=_modal_add_python_version(dockerfile_path),
                )

            self._app = await App.lookup.aio(
                name="__harbor__",
                create_if_missing=True,
            )

            gpu_config = None
            gpu_type = "any"

            if self.task_env_config.gpus > 0:
                if self.task_env_config.gpu_types:
                    if len(self.task_env_config.gpu_types) > 1:
                        self.logger.debug(
                            "Multiple GPU types specified but Modal only supports one GPU "
                            "type. Using the first GPU type."
                        )
                    gpu_type = self.task_env_config.gpu_types[0]

                gpu_config = f"{gpu_type}:{self.task_env_config.gpus}"

            secrets_config = [Secret.from_name(secret) for secret in self._secrets]
            volumes_config = {
                mount_path: Volume.from_name(volume_name)
                for mount_path, volume_name in self._volumes.items()
            }

            try:
                self._sandbox = await self._create_sandbox(
                    gpu_config=gpu_config,
                    secrets_config=secrets_config,
                    volumes_config=volumes_config,
                )
            finally:
                cleanup_dockerfile()

            await self._sandbox.mkdir.aio(
                str(EnvironmentPaths.agent_dir),
                parents=True,
            )
            await self._sandbox.mkdir.aio(
                str(EnvironmentPaths.verifier_dir),
                parents=True,
            )

            # Make log directories world-writable so non-root agents/verifiers can write to them.
            await self.exec(
                f"chmod 777 {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir}"
            )

    return BenchFlowModalEnvironment


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
    preserve_agent_network: bool = False,
) -> Any:
    """Create a Harbor environment (Docker, Daytona, or Modal)."""
    env_config = task.config.environment
    environment_dir = task_path / "environment"
    if not environment_dir.exists():
        environment_dir = task.paths.environment_dir
    if preserve_agent_network and env_config.allow_internet is False:
        # LLM agents run inside the sandbox and need outbound network for model
        # APIs and first-run agent installation. BenchFlow enforces the task's
        # no-web policy at the agent layer instead of applying Harbor's container
        # network block for these runs.
        env_config = env_config.model_copy(deep=True)
        env_config.allow_internet = True

    if environment_type == "docker":
        from harbor.environments.docker.docker import DockerEnvironment

        return DockerEnvironment(
            environment_dir=environment_dir,
            environment_name=task_path.name,
            session_id=trial_name,
            trial_paths=trial_paths,
            task_env_config=env_config,
        )
    elif environment_type == "daytona":
        from harbor.environments.daytona import DaytonaEnvironment

        from benchflow._daytona_patches import apply as _apply_daytona_patches

        _apply_daytona_patches()

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
        if env_config.storage_mb > _DAYTONA_MAX_STORAGE_MB:
            logger.warning(
                "Clamping storage_mb %d -> %d for Daytona (override with BENCHFLOW_DAYTONA_MAX_STORAGE_MB)",
                env_config.storage_mb,
                _DAYTONA_MAX_STORAGE_MB,
            )
            env_config.storage_mb = _DAYTONA_MAX_STORAGE_MB

        return DaytonaEnvironment(
            environment_dir=environment_dir,
            environment_name=task_path.name,
            session_id=trial_name,
            trial_paths=trial_paths,
            task_env_config=env_config,
            auto_stop_interval_mins=1440,
            auto_delete_interval_mins=1440,
        )
    elif environment_type == "modal":
        modal_environment_class = _create_benchflow_modal_environment_class()
        modal_environment_class.preflight()

        return modal_environment_class(
            environment_dir=environment_dir,
            environment_name=task_path.name,
            session_id=trial_name,
            trial_paths=trial_paths,
            task_env_config=env_config,
        )
    else:
        raise ValueError(
            f"Unknown environment_type: {environment_type!r} (use 'docker', 'daytona', or 'modal')"
        )
