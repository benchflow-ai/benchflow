"""Native DockerSandbox — internalized from Harbor with RL-first terminology.

Uses docker-compose for container orchestration on local Docker.
"""

from __future__ import annotations

import asyncio
import asyncio.subprocess
import base64
import contextlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from benchflow.sandbox._base import (
    BaseSandbox,
    ExecResult,
    _filter_compose_service_names,
)
from benchflow.sandbox._compose import (
    COMPOSE_BASE_PATH,
    COMPOSE_BUILD_PATH,
    COMPOSE_NO_NETWORK_PATH,
    COMPOSE_PREBUILT_PATH,
)
from benchflow.task.config import SandboxConfig
from benchflow.task.env import resolve_env_vars
from benchflow.task.paths import RolloutPaths, SandboxPaths

logger = logging.getLogger("benchflow")


_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def _env_flag_enabled(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE_ENV_VALUES


def _sanitize_docker_image_name(name: str) -> str:
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    name = re.sub(r"[^a-z0-9._-]", "-", name)
    return name


def _sanitize_docker_compose_project_name(name: str) -> str:
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    name = re.sub(r"[^a-z0-9_-]", "-", name)
    return name


class DockerSandboxEnvVars(BaseModel):
    main_image_name: str
    context_dir: str
    host_verifier_logs_path: str
    host_agent_logs_path: str
    host_artifacts_path: str
    env_verifier_logs_path: str
    env_agent_logs_path: str
    env_artifacts_path: str
    prebuilt_image_name: str | None = None
    cpus: int = 1
    memory: str = "1G"

    def to_env_dict(self, include_os_env: bool = True) -> dict[str, str]:
        env_dict: dict[str, str] = {} if not include_os_env else dict(os.environ)

        for field_name, value in self.model_dump(exclude_none=True).items():
            if value is None:
                continue
            env_dict[field_name.upper()] = str(value)

        return env_dict


class DockerSandbox(BaseSandbox):
    _DOCKER_COMPOSE_BASE_PATH = COMPOSE_BASE_PATH
    _DOCKER_COMPOSE_BUILD_PATH = COMPOSE_BUILD_PATH
    _DOCKER_COMPOSE_PREBUILT_PATH = COMPOSE_PREBUILT_PATH
    _DOCKER_COMPOSE_NO_NETWORK_PATH = COMPOSE_NO_NETWORK_PATH

    _image_build_locks: ClassVar[dict[str, asyncio.Lock]] = {}

    @classmethod
    def preflight(cls) -> None:
        if not shutil.which("docker"):
            raise SystemExit(
                "Docker is not installed or not on PATH. "
                "Please install Docker and try again."
            )
        try:
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise SystemExit(
                "Docker daemon is not running. Please start Docker and try again."
            ) from exc

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        rollout_paths: RolloutPaths | None,
        task_env_config: SandboxConfig,
        keep_containers: bool = False,
        mounts_json: list[dict[str, str]] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            rollout_paths=rollout_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

        self._keep_containers = keep_containers
        self._mounts_json = mounts_json
        self._mounts_compose_path: Path | None = None

        verifier_dir = (
            str(rollout_paths.verifier_dir.resolve().absolute())
            if rollout_paths
            else "/tmp/verifier"
        )
        agent_dir = (
            str(rollout_paths.agent_dir.resolve().absolute())
            if rollout_paths
            else "/tmp/agent"
        )
        artifacts_dir = (
            str(rollout_paths.artifacts_dir.resolve().absolute())
            if rollout_paths
            else "/tmp/artifacts"
        )

        self._env_vars = DockerSandboxEnvVars(
            main_image_name=_sanitize_docker_image_name(f"bf__{environment_name}"),
            context_dir=str(self.environment_dir.resolve().absolute()),
            host_verifier_logs_path=verifier_dir,
            host_agent_logs_path=agent_dir,
            host_artifacts_path=artifacts_dir,
            env_verifier_logs_path=str(SandboxPaths.verifier_dir),
            env_agent_logs_path=str(SandboxPaths.agent_dir),
            env_artifacts_path=str(SandboxPaths.artifacts_dir),
            prebuilt_image_name=task_env_config.docker_image,
            cpus=task_env_config.cpus,
            memory=f"{task_env_config.memory_mb}M",
        )
        self._use_prebuilt = False

        self._compose_task_env: dict[str, str] = {}
        if task_env_config.env and self._uses_compose:
            self._compose_task_env = resolve_env_vars(task_env_config.env)

        resolved_task_keys = set(self._compose_task_env.keys()) | set(
            self._persistent_env.keys()
        )
        if resolved_task_keys:
            benchflow_keys = set(
                self._env_vars.to_env_dict(include_os_env=False).keys()
            )
            collisions = benchflow_keys & resolved_task_keys
            if collisions:
                self.logger.warning(
                    "Environment vars override BenchFlow compose variable(s): %s",
                    ", ".join(sorted(collisions)),
                )

    @property
    def _uses_compose(self) -> bool:
        return self._environment_docker_compose_path.exists()

    @property
    def is_mounted(self) -> bool:
        return _env_flag_enabled("BENCHFLOW_DOCKER_LOGS_HOST_MOUNTED", default=True)

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @property
    def _environment_docker_compose_path(self) -> Path:
        return self.environment_dir / "docker-compose.yaml"

    @property
    def _docker_compose_paths(self) -> list[Path]:
        build_or_prebuilt = (
            self._DOCKER_COMPOSE_PREBUILT_PATH
            if self._use_prebuilt
            else self._DOCKER_COMPOSE_BUILD_PATH
        )

        if self._environment_docker_compose_path.exists():
            paths = [
                self._DOCKER_COMPOSE_BASE_PATH,
                build_or_prebuilt,
                self._environment_docker_compose_path,
            ]
        else:
            paths = [self._DOCKER_COMPOSE_BASE_PATH, build_or_prebuilt]

        if self._mounts_compose_path:
            paths.append(self._mounts_compose_path)

        if not self.task_env_config.allow_internet:
            paths.append(self._DOCKER_COMPOSE_NO_NETWORK_PATH)

        return paths

    def _write_mounts_compose_file(self) -> Path:
        compose = {"services": {"main": {"volumes": self._mounts_json}}}
        assert self.rollout_paths is not None
        path = self.rollout_paths.rollout_dir / "docker-compose-mounts.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(compose, indent=2))
        return path

    def _validate_definition(self) -> None:
        if (
            not self._dockerfile_path.exists()
            and not self._environment_docker_compose_path.exists()
        ):
            raise FileNotFoundError(
                f"{self._dockerfile_path} and {self._environment_docker_compose_path} "
                "not found. Please ensure at least one of these files exist."
            )

    async def _run_docker_compose_command(
        self, command: list[str], check: bool = True, timeout_sec: int | None = None
    ) -> ExecResult:
        full_command = [
            "docker",
            "compose",
            "--project-name",
            _sanitize_docker_compose_project_name(self.session_id),
            "--project-directory",
            str(self.environment_dir.resolve().absolute()),
        ]
        for path in self._docker_compose_paths:
            full_command.extend(["-f", str(path.resolve().absolute())])
        full_command.extend(command)

        env = self._env_vars.to_env_dict(include_os_env=True)
        if self._compose_task_env:
            env.update(self._compose_task_env)
        if self._persistent_env:
            env.update(self._persistent_env)

        process = await asyncio.create_subprocess_exec(
            *full_command,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        try:
            if timeout_sec:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_sec
                )
            else:
                stdout_bytes, stderr_bytes = await process.communicate()
        except TimeoutError:
            process.terminate()
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=5
                )
            except TimeoutError:
                process.kill()
                stdout_bytes, stderr_bytes = await process.communicate()
            raise RuntimeError(
                f"Command timed out after {timeout_sec} seconds"
            ) from None

        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else None
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else None

        result = ExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=process.returncode or 0,
        )

        if check and result.return_code != 0:
            raise RuntimeError(
                f"Docker compose command failed for environment {self.environment_name}. "
                f"Command: {' '.join(full_command)}. "
                f"Return code: {result.return_code}. "
                f"Stdout: {result.stdout}. "
                f"Stderr: {result.stderr}. "
            )

        return result

    async def start(self, force_build: bool) -> None:
        if self._mounts_json:
            self._mounts_compose_path = self._write_mounts_compose_file()

        self._use_prebuilt = not force_build and bool(self.task_env_config.docker_image)

        if not self._use_prebuilt:
            lock = self._image_build_locks.setdefault(
                self.environment_name, asyncio.Lock()
            )
            async with lock:
                await self._run_docker_compose_command(["build"])

        with contextlib.suppress(RuntimeError):
            await self._run_docker_compose_command(["down", "--remove-orphans"])

        await self._run_docker_compose_command(["up", "--detach", "--wait"])

        await self.exec(
            f"chmod 777 {SandboxPaths.agent_dir} {SandboxPaths.verifier_dir}"
        )

    async def stop(self, delete: bool) -> None:
        try:
            await self._chown_to_host_user(str(SandboxPaths.logs_dir), recursive=True)
        except Exception as e:
            self.logger.warning(f"Failed to chown logs directory: {e}")

        if self._keep_containers and delete:
            self.logger.warning(
                "Both `keep_containers` and `--delete` option are set. "
                "keep_containers takes precedence."
            )
        if self._keep_containers:
            try:
                await self._run_docker_compose_command(["stop"])
            except Exception as e:
                self.logger.warning(f"Docker compose stop failed: {e}")
        elif delete:
            try:
                down_command = ["down", "--volumes", "--remove-orphans"]
                if not self._use_prebuilt:
                    down_command = [
                        "down",
                        "--rmi",
                        "all",
                        "--volumes",
                        "--remove-orphans",
                    ]
                await self._run_docker_compose_command(down_command)
            except Exception as e:
                self.logger.warning(f"Docker compose down failed: {e}")
        else:
            try:
                await self._run_docker_compose_command(["down"])
            except Exception as e:
                self.logger.warning(f"Docker compose down failed: {e}")

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        target_parent = str(Path(target_path).parent)
        if target_parent not in {"", "."}:
            await self.exec(f"mkdir -p {shlex.quote(target_parent)}", user="root")
        await self._run_docker_compose_command(
            ["cp", str(source_path), f"main:{target_path}"],
            check=True,
        )

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        """Upload a directory into a compose service container.

        ``service`` defaults to ``"main"``; pass a target service to land the
        directory in an additional vulhub-style container (#248).
        """
        await self.exec(
            f"mkdir -p {shlex.quote(target_dir)}", user="root", service=service
        )
        await self._run_docker_compose_command(
            ["cp", f"{source_dir}/.", f"{service}:{target_dir}"],
            check=True,
        )
        if sys.platform == "win32":
            await self._run_docker_compose_command(
                [
                    "exec",
                    service,
                    "bash",
                    "-c",
                    f"find {target_dir} -type f \\( -name '*.sh' -o -name '*.py' \\) "
                    "-exec sed -i 's/\\r$//' {} \\;",
                ],
                check=False,
            )

    async def _chown_to_host_user(
        self, path: str, recursive: bool = False, service: str = "main"
    ) -> None:
        if not hasattr(os, "getuid"):
            return
        flag = "-R " if recursive else ""
        await self.exec(
            f"chown {flag}{os.getuid()}:{os.getgid()} {shlex.quote(path)}",
            user="root",
            service=service,
        )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._chown_to_host_user(source_path)
        await self._run_docker_compose_command(
            ["cp", f"main:{source_path}", str(target_path)],
            check=True,
        )

    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        """Download a directory from a compose service container.

        ``service`` defaults to ``"main"``; pass a target service to fetch
        target-side verifier output from a vulhub-style container (#248).
        """
        await self._chown_to_host_user(source_dir, recursive=True, service=service)
        await self._run_docker_compose_command(
            ["cp", f"{service}:{source_dir}/.", str(target_dir)],
            check=True,
        )

    async def services(self) -> list[str]:
        """List compose service names defined for this sandbox.

        Includes BenchFlow's own ``main`` service plus any additional
        services the task declares in its ``docker-compose.yaml``
        (vulhub-style target/database containers — see #248).

        ``_run_docker_compose_command`` merges stderr into stdout, so the
        output is filtered to lines that match the Docker Compose service
        naming grammar — a stray warning line cannot become a spurious
        "service".
        """
        result = await self._run_docker_compose_command(
            ["config", "--services"], check=True
        )
        return _filter_compose_service_names(result.stdout or "")

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
    ) -> ExecResult:
        """Run a command in a compose service container.

        ``service`` defaults to ``"main"`` (the agent container). Pass a
        different service name to target an additional container declared
        in the task's ``docker-compose.yaml`` — e.g. to inject a flag into
        a vulnerable target before the agent runs, or to verify exploit
        success by inspecting target-side state afterwards (#248).
        """
        user = self._resolve_user(user)
        env = self._merge_env(env)

        exec_command: list[str] = ["exec"]

        if cwd:
            exec_command.extend(["-w", cwd])

        if user is not None:
            exec_command.extend(["-u", str(user)])

        exec_command.append(service)

        # Env vars are written to a file inside the container and sourced,
        # rather than passed as `-e KEY=VALUE` flags. The flags are visible in
        # `ps aux` on the host, which would leak secrets — e.g. the verifier's
        # [verifier.env] LLM-judge API keys. DockerProcess/DaytonaProcess avoid
        # `-e` for the same reason; this keeps `exec` consistent with them.
        if env:
            command = self._wrap_command_with_env_file(env, command)

        # Use POSIX ``sh`` rather than ``bash``: with multi-service support
        # (#248), ``exec(..., service=...)`` can target arbitrary task
        # containers — Alpine/distroless/minimal DB images frequently ship no
        # ``/bin/bash``. The wrapped command (env-file sourcing, ``trap``,
        # ``base64 -d``, ``set -a``/``. file``) uses only POSIX constructs.
        exec_command.extend(["sh", "-c", command])

        return await self._run_docker_compose_command(
            exec_command, check=False, timeout_sec=timeout_sec
        )

    # A POSIX shell identifier: a name the shell can `export`. Keys outside
    # this grammar (e.g. containing `.` or `-`) are valid process env keys but
    # cannot be assigned via `export NAME=...`; sourcing such a line aborts the
    # whole `. {env_path}` step, so the user command would never run (PR #323).
    _SHELL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    @classmethod
    def _wrap_command_with_env_file(cls, env: dict[str, str], command: str) -> str:
        """Return *command* prefixed to materialize *env* from a file.

        The env vars are base64-encoded into the command string (not visible
        as individual ``KEY=VALUE`` args in ``ps aux``), decoded to a
        mode-0600 file inside the container, and sourced before the real
        command runs. A ``trap ... EXIT`` deletes the temp file
        unconditionally — even if the decode/source step fails — so a failed
        ``&&`` chain can never leave the env file behind.

        Only keys that are valid POSIX shell identifiers are emitted as
        ``export`` lines. Keys that are not (e.g. containing ``.`` or ``-``)
        cannot be assigned by the shell — emitting them would make
        ``. {env_path}`` fail and the user command would never run — so they
        are skipped with a warning rather than silently breaking exec
        (regression guarded — PR #323).

        The ``umask 077`` that protects the env file is scoped to a subshell
        so it does not leak into the user's command — otherwise files the
        command creates would get mode-0600 unexpectedly (PR #323).
        """
        exportable: dict[str, str] = {}
        skipped: list[str] = []
        for k, v in env.items():
            if cls._SHELL_IDENTIFIER_RE.match(k):
                exportable[k] = v
            else:
                skipped.append(k)
        if skipped:
            logger.warning(
                "Skipping env var(s) with non-identifier names (cannot be "
                "exported by the shell): %s",
                ", ".join(sorted(skipped)),
            )

        env_body = "".join(
            f"export {k}={shlex.quote(v)}\n" for k, v in exportable.items()
        )
        encoded = base64.b64encode(env_body.encode()).decode()
        # Unique suffix so concurrent exec() calls in one container can't
        # clobber each other's env file.
        env_path = f"/tmp/.benchflow_exec_env_{uuid.uuid4().hex[:16]}"
        return (
            f"trap 'rm -f {env_path}' EXIT && "
            f"(umask 077 && printf %s {shlex.quote(encoded)} | base64 -d > "
            f"{env_path}) && set -a && . {env_path} && set +a && "
            f"{command}"
        )

    async def attach(self) -> None:
        variables = " ".join(
            f"export {k}={shlex.quote(str(v))}"
            for k, v in self._env_vars.to_env_dict(include_os_env=False).items()
        )

        compose_file_args: list[str] = []
        for path in self._docker_compose_paths:
            compose_file_args.extend(
                ["-f", shlex.quote(str(path.resolve().absolute()))]
            )

        project_name = _sanitize_docker_compose_project_name(self.session_id)
        compose_base = [
            "docker",
            "compose",
            "--project-name",
            project_name,
            *compose_file_args,
        ]

        os.execvp(
            "bash",
            [
                "bash",
                "-c",
                f"{variables}; "
                + " ".join([*compose_base, "exec", "-it", "main", "bash"])
                + "; "
                + " ".join([*compose_base, "down"]),
            ],
        )
