"""Sandbox setup utilities: Dockerfile preprocessing, DinD patching, sandbox creation."""

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
from typing import Any, NoReturn, cast

from benchflow._paths import ignore_symlinks, is_safe_regular_file
from benchflow.agents.registry import AGENTS
from benchflow.sandbox.registry import (
    SandboxBuildContext,
    register_sandbox,
    resolve_sandbox,
)
from benchflow.task import RolloutPaths, Task

logger = logging.getLogger(__name__)

# Daytona's per-sandbox caps are 4 CPU / 16 GB RAM / 10 GB disk. Tasks declaring
# more fail at sandbox creation, so clamp here to degrade gracefully (slower
# build) instead of erroring out. Override via env if you have higher limits.
_DAYTONA_MAX_CPUS = int(os.environ.get("BENCHFLOW_DAYTONA_MAX_CPUS", "4"))
_DAYTONA_MAX_MEMORY_MB = int(os.environ.get("BENCHFLOW_DAYTONA_MAX_MEMORY_MB", "16384"))
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


def _stage_ignore(directory: str, contents: list[str]) -> list[str]:
    """``shutil.copytree`` ignore callback: drop noise dirs *and* every symlink.

    Composes :func:`shutil.ignore_patterns` for ``_IGNORE_DIRS`` with
    :func:`benchflow._paths.ignore_symlinks` so a task-controlled symlink
    cannot smuggle host files into the Docker build context (#411).
    """
    pattern_skip = set(shutil.ignore_patterns(*_IGNORE_DIRS)(directory, contents))
    link_skip = set(ignore_symlinks(directory, contents))
    return sorted(pattern_skip | link_skip)


_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?([A-Za-z0-9_.-]+)['\"]?")


_OPTIONAL_SANDBOX_EXTRAS = {
    "daytona": "sandbox-daytona",
    "modal": "sandbox-modal",
}


def _raise_missing_optional_sandbox_dependency(
    sandbox_type: str,
    exc: ModuleNotFoundError,
) -> NoReturn:
    extra = _OPTIONAL_SANDBOX_EXTRAS[sandbox_type]
    raise RuntimeError(
        f"Missing optional dependency for {sandbox_type!r} sandbox. "
        f"Install it with `uv sync --extra {extra}` for local development, "
        f"or `pip install 'benchflow[{extra}]'` for a packaged install."
    ) from exc


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


def _modal_add_python_version(dockerfile_path: Path) -> str | None:
    """Python version Modal should add to Dockerfile images that lack Python."""
    configured = os.environ.get("BENCHFLOW_MODAL_ADD_PYTHON")
    if configured is None and _modal_dockerfile_uses_python_base(dockerfile_path):
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
    if rewritten == text:
        return dockerfile_path, lambda: None

    tmpdir = tempfile.TemporaryDirectory(prefix="benchflow-modal-dockerfile-")
    tmp_path = Path(tmpdir.name) / "Dockerfile"
    tmp_path.write_text(rewritten)
    return tmp_path, tmpdir.cleanup


def _create_benchflow_modal_environment_class():
    """Create a ModalSandbox subclass with BenchFlow's image-build defaults."""
    from benchflow.sandbox.modal_impl import ModalSandbox

    class BenchFlowModalSandbox(ModalSandbox):
        async def start(self, force_build: bool) -> None:
            """Starts the Modal sandbox, adding Python for plain Linux images."""
            from modal import App, Image, Secret, Volume

            from benchflow.task import SandboxPaths

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
                name="__benchflow__",
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

            sandbox = cast(Any, self._sandbox)
            await sandbox.mkdir.aio(
                str(SandboxPaths.agent_dir),
                parents=True,
            )
            await sandbox.mkdir.aio(
                str(SandboxPaths.verifier_dir),
                parents=True,
            )

            # Make log directories world-writable so non-root agents/verifiers can write to them.
            await self.exec(
                f"chmod 777 {SandboxPaths.agent_dir} {SandboxPaths.verifier_dir}"
            )

    return BenchFlowModalSandbox


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
        rewritten = _rewrite_copy_line(line, env_dir, context_root)
        new_lines.append(rewritten if rewritten is not None else line)

    dockerfile_path.write_text("\n".join(new_lines))


def _stage_copy_source(src_path: str, env_dir: Path, context_root: Path) -> str:
    """Stage a single COPY source into ``_deps/`` and return its rewritten path.

    Returns the original ``src_path`` unchanged when it cannot or should not be
    staged (absolute path, build arg, ``.``, glob, or a source that does not
    exist under ``context_root``).

    Sources that resolve outside ``context_root`` (e.g. ``../outside-secret``)
    are rejected: with a permissive SDK ``context_root`` a malicious Dockerfile
    could otherwise stage arbitrary host files into ``environment/_deps/``
    where they then enter the build context (issue #363).
    """
    # Skip sources already relative to env dir, absolute, or using build args.
    if src_path.startswith("/") or src_path.startswith("$") or src_path == ".":
        return src_path
    # Globs can't be resolved to a single staged path — leave them alone.
    if any(ch in src_path for ch in "*?["):
        return src_path

    abs_src = context_root / src_path
    # Reject sources that escape ``context_root`` via ``..`` or symlinks.
    # ``resolve(strict=False)`` collapses ``..`` segments without requiring the
    # file to exist yet (Dockerfile lint can run before sources are present).
    try:
        resolved_src = abs_src.resolve(strict=False)
        resolved_root = context_root.resolve(strict=False)
    except (OSError, RuntimeError):
        # Unresolvable paths (e.g. symlink loops) — refuse to stage them.
        logger.warning(
            "stage_dockerfile_deps: refusing to stage COPY source %r "
            "(path could not be resolved)",
            src_path,
        )
        return src_path
    if not resolved_src.is_relative_to(resolved_root):
        logger.warning(
            "stage_dockerfile_deps: refusing to stage COPY source %r — "
            "resolves outside context_root %s",
            src_path,
            resolved_root,
        )
        return src_path
    if not abs_src.exists():
        return src_path

    # Reject symlinked sources outright (#411). Following a symlink here
    # would bake the link target into the Docker build context, which is an
    # exfiltration sink. ``abs_src.is_symlink()`` does *not* follow the
    # link, so this is checked before any read.
    if abs_src.is_symlink():
        logger.warning(
            "stage_dockerfile_deps: refusing to stage symlinked source %s",
            abs_src,
        )
        return src_path

    dep_name = _dep_local_name(src_path)
    local_dest = env_dir / "_deps" / dep_name

    if abs_src.is_dir():
        if local_dest.exists():
            shutil.rmtree(local_dest)
        # ``symlinks=False`` is the default but we re-state it for
        # readability; the composed ignore drops both noise dirs and any
        # symlink found inside the tree.
        shutil.copytree(
            abs_src,
            local_dest,
            symlinks=False,
            ignore=_stage_ignore,
        )
    elif is_safe_regular_file(abs_src):
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(abs_src, local_dest)
    else:
        logger.warning("stage_dockerfile_deps: refusing non-regular source %s", abs_src)
        return src_path

    return f"_deps/{dep_name}"


def _rewrite_copy_line(line: str, env_dir: Path, context_root: Path) -> str | None:
    """Rewrite a single Dockerfile ``COPY`` line, staging its sources.

    Handles both the shell form (``COPY src... dst``, including multiple
    sources) and the JSON/exec form (``COPY ["src", "dst"]``). Every source
    argument is staged into ``_deps/``. Returns the rewritten line, or
    ``None`` if the line is not a ``COPY`` that needs staging.

    A ``COPY`` line that looks like it has sources but cannot be parsed
    emits a warning rather than being silently skipped.
    """
    # Shell form: COPY [--flags...] src... dst
    shell_match = re.match(r"^(\s*COPY\s+(?:--\S+\s+)*)(\S.*\S|\S)\s*$", line)
    if shell_match:
        prefix = shell_match.group(1)
        args_str = shell_match.group(2)

        # JSON/exec form: COPY ["src", ..., "dst"]
        if args_str.startswith("["):
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                logger.warning(
                    "stage_dockerfile_deps: could not parse JSON-form COPY, "
                    "leaving unchanged: %s",
                    line.strip(),
                )
                return None
            if not isinstance(args, list) or len(args) < 2:
                logger.warning(
                    "stage_dockerfile_deps: malformed JSON-form COPY, "
                    "leaving unchanged: %s",
                    line.strip(),
                )
                return None
            *sources, dst = [str(a) for a in args]
            new_sources = [
                _stage_copy_source(s, env_dir, context_root) for s in sources
            ]
            if new_sources == sources:
                return None  # nothing staged
            return f"{prefix}{json.dumps([*new_sources, dst])}"

        # Shell form: split into whitespace-separated args. The last arg is
        # the destination; everything before it is a source (>= 1 source).
        try:
            args = shlex.split(args_str)
        except ValueError:
            logger.warning(
                "stage_dockerfile_deps: could not parse COPY arguments, "
                "leaving unchanged: %s",
                line.strip(),
            )
            return None
        if len(args) < 2:
            logger.warning(
                "stage_dockerfile_deps: COPY has fewer than 2 arguments, "
                "leaving unchanged: %s",
                line.strip(),
            )
            return None
        *sources, dst = args
        new_sources = [_stage_copy_source(s, env_dir, context_root) for s in sources]
        if new_sources == sources:
            return None  # nothing staged
        return f"{prefix}{' '.join(new_sources)} {dst}"

    return None


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
    if not any(skills_dir.iterdir()):
        logger.info("Skills injection skipped: skills directory is empty")
        return

    dest = env_dir / "_deps" / "skills"
    if dest.exists():
        shutil.rmtree(dest)
    # Refuse to follow symlinks under skills_dir (#411). A symlink baked
    # into the image would otherwise serve attacker-chosen content to every
    # agent for the lifetime of the build.
    shutil.copytree(skills_dir, dest, symlinks=False, ignore=_stage_ignore)

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


_DIND_PATCH_APPLIED = False


def _patch_docker_dind() -> None:
    """Monkey-patch DockerSandboxEnvVars for DinD path translation.

    When running inside a devcontainer, HOST_*_PATH env vars need to use
    host filesystem paths, not container paths.

    Idempotent: safe to call repeatedly; only applies the wrapper once.
    Called from ``Rollout.__init__`` so importing ``benchflow.rollout`` has
    no side effects on sandbox/provider behavior — only constructing a
    rollout activates the DinD compatibility shim.
    """
    global _DIND_PATCH_APPLIED
    if _DIND_PATCH_APPLIED:
        return

    dind_mount = _detect_dind_mount()
    if not dind_mount:
        return

    host_source, container_dest = dind_mount
    logger.info(f"DinD detected: {container_dest} → {host_source}")

    from benchflow.sandbox.docker import DockerSandboxEnvVars

    _original = DockerSandboxEnvVars.to_env_dict

    def _patched(self, include_os_env=True):  # type: ignore[override]
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

    DockerSandboxEnvVars.to_env_dict = _patched  # type: ignore[assignment, ty:invalid-assignment]
    _DIND_PATCH_APPLIED = True


# ── Built-in sandbox provider factories ────────────────────────────────────
# Each factory does only its own provider-specific construction; the shared,
# provider-independent preamble runs once in ``_create_sandbox_environment``
# and is handed in via the ``SandboxBuildContext``. Registered into the
# provider registry at import time (bottom of module) so the kernel resolves
# sandboxes by name (``contracts``/registry seam) rather than via this if/elif.


def _build_docker_sandbox(ctx: SandboxBuildContext) -> Any:
    from benchflow.sandbox.docker import DockerSandbox

    return DockerSandbox(
        environment_dir=ctx.environment_dir,
        environment_name=ctx.environment_name,
        session_id=ctx.session_id,
        rollout_paths=ctx.rollout_paths,
        task_env_config=ctx.task_env_config,
        persistent_env=ctx.persistent_env,
    )


def _build_daytona_sandbox(ctx: SandboxBuildContext) -> Any:
    try:
        from benchflow.sandbox._sdk_ops import apply as _apply_daytona_patches
        from benchflow.sandbox.daytona import DaytonaSandbox
    except ModuleNotFoundError as exc:
        _raise_missing_optional_sandbox_dependency("daytona", exc)

    _apply_daytona_patches()

    env_config = ctx.task_env_config
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

    return DaytonaSandbox(
        environment_dir=ctx.environment_dir,
        environment_name=ctx.environment_name,
        session_id=ctx.session_id,
        rollout_paths=ctx.rollout_paths,
        task_env_config=env_config,
        auto_stop_interval_mins=1440,
        auto_delete_interval_mins=1440,
        persistent_env=ctx.persistent_env,
    )


def _build_modal_sandbox(ctx: SandboxBuildContext) -> Any:
    try:
        modal_environment_class = _create_benchflow_modal_environment_class()
    except ModuleNotFoundError as exc:
        _raise_missing_optional_sandbox_dependency("modal", exc)
    modal_environment_class.preflight()

    return modal_environment_class(
        environment_dir=ctx.environment_dir,
        environment_name=ctx.environment_name,
        session_id=ctx.session_id,
        rollout_paths=ctx.rollout_paths,
        task_env_config=ctx.task_env_config,
        persistent_env=ctx.persistent_env,
    )


def _create_sandbox_environment(
    sandbox_type: str,
    task: Task,
    task_path: Path,
    rollout_name: str,
    rollout_paths: RolloutPaths,
    preserve_agent_network: bool = False,
    environment_manifest: Any = None,
) -> Any:
    """Create a sandbox (Docker, Daytona, Modal, or any BYO provider).

    Runs the shared, provider-independent preamble — network-policy
    adjustment and manifest image/env resolution — then resolves the named
    provider through :func:`benchflow.sandbox.registry.resolve_sandbox`.

    When ``environment_manifest`` is provided, its declared controls take
    effect at sandbox-construction time: the manifest's runnable ``image``
    overrides ``task.config.environment.docker_image`` (so the manifest —
    not the task's local Dockerfile — drives image selection), and the
    manifest's ``task_selection`` + ``forward_env`` are resolved into a
    persistent env overlay so the values reach the container's entrypoint
    via compose and every subsequent ``sandbox.exec`` call.
    """
    env_config = task.config.environment
    environment_dir = task_path / "environment"
    if not environment_dir.exists():
        environment_dir = task.paths.environment_dir
    if preserve_agent_network and env_config.allow_internet is False:
        # LLM agents run inside the sandbox and need outbound network for model
        # APIs and first-run agent installation. BenchFlow enforces the task's
        # no-web policy at the agent layer instead of applying the container
        # network block for these runs.
        env_config = env_config.model_copy(deep=True)
        env_config.allow_internet = True

    manifest_env: dict[str, str] = {}
    if environment_manifest is not None:
        from benchflow.environment.manifest import (
            resolve_manifest_image,
            resolve_manifest_runtime_env,
        )

        manifest_image = resolve_manifest_image(environment_manifest)
        if manifest_image:
            # Image control point — manifest's run target wins over task.toml's
            # docker_image so a benchmark author can pin the runtime image
            # from the manifest without editing every task.toml.
            if env_config is task.config.environment:
                env_config = env_config.model_copy(deep=True)
            env_config.docker_image = manifest_image
        manifest_env = resolve_manifest_runtime_env(
            environment_manifest, task_id=task_path.name
        )

    ctx = SandboxBuildContext(
        environment_dir=environment_dir,
        environment_name=task_path.name,
        session_id=rollout_name,
        rollout_paths=rollout_paths,
        task_env_config=env_config,
        persistent_env=manifest_env or None,
    )
    return resolve_sandbox(sandbox_type, ctx)


# Register the built-in providers so the kernel resolves them by name.
register_sandbox("docker", _build_docker_sandbox)
register_sandbox("daytona", _build_daytona_sandbox)
register_sandbox("modal", _build_modal_sandbox)

# Backward compatibility alias
_create_environment = _create_sandbox_environment
