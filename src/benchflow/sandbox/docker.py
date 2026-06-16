"""Native DockerSandbox — internalized from Harbor with RL-first terminology.

Uses docker-compose for container orchestration on local Docker.
"""

from __future__ import annotations

import asyncio
import asyncio.subprocess
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
    wrap_command_with_env_file,
)
from benchflow.sandbox._compose import (
    COMPOSE_BASE_PATH,
    COMPOSE_BUILD_PATH,
    COMPOSE_NO_NETWORK_PATH,
    COMPOSE_PREBUILT_PATH,
    COMPOSE_UP_RETRY_DELAYS_SEC,
    is_compose_up_network_race_error,
)
from benchflow.sandbox.protocol import SandboxImage
from benchflow.task.config import SandboxConfig
from benchflow.task.env import resolve_env_vars
from benchflow.task.paths import RolloutPaths, SandboxPaths

logger = logging.getLogger("benchflow")

_DOCKER_BUILD_RETRY_DELAYS_SEC = (2.0, 5.0)
_DOCKER_BUILD_RETRYABLE_ERRORS = (
    re.compile(r"at least one invalid signature was encountered", re.IGNORECASE),
    re.compile(r"the repository '.+' is not signed", re.IGNORECASE),
    re.compile(r"no space left on device", re.IGNORECASE),
    re.compile(r"readtimeouterror", re.IGNORECASE),
    re.compile(r"read timed out", re.IGNORECASE),
    re.compile(r"connection (?:timed out|reset by peer)", re.IGNORECASE),
)

# Compose-up network-race retry config lives in _compose so the host docker
# path and the Daytona DinD path share the exact same race detection + back-off.
_COMPOSE_UP_RETRY_DELAYS_SEC = COMPOSE_UP_RETRY_DELAYS_SEC


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


def _is_retryable_docker_build_error(message: str) -> bool:
    return any(pattern.search(message) for pattern in _DOCKER_BUILD_RETRYABLE_ERRORS)


def _is_compose_up_network_race_error(message: str) -> bool:
    return is_compose_up_network_race_error(message)


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
    _build_semaphore: ClassVar[asyncio.Semaphore | None] = None

    @classmethod
    def set_build_concurrency(cls, n: int) -> None:
        """Limit how many sandboxes go through the docker startup phase
        (build + compose down --remove-orphans + compose up --wait) in parallel.

        Default is unlimited. Setting this is critical when --concurrency is
        high (e.g. 60): otherwise N tasks all hammer the docker daemon at once,
        causing build/network creation races and `docker container prune`
        timeouts. Agent execution after the container is up is NOT gated.
        """
        cls._build_semaphore = asyncio.Semaphore(n)

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
        # Install-before-lockdown: the restrictive network policy is applied by
        # relock_network() AFTER the agent installs, not at sandbox start.
        self._network_locked = False
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
        return True

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

        paths.extend(self._network_policy_compose_paths())
        return paths

    def _network_policy_compose_paths(self) -> list[Path]:
        """Compose overrides enforcing the task's resolved network policy.

        ``no-network`` detaches the container network; ``allowlist`` confines
        egress to ``allowed_hosts`` via an internal network + proxy sidecar
        (see ``_egress.build_egress_override``); ``public`` adds nothing. An
        allowlist with no writable rollout dir fails closed to no-network.
        """
        from benchflow.sandbox._egress import build_egress_override
        from benchflow.sandbox.network_policy import (
            EffectivePolicy,
            resolve_network_decision,
        )

        if not self._network_locked:
            # Stay open during the install phase; relock_network() applies the
            # restrictive policy once the agent has been installed.
            return []
        decision = resolve_network_decision(self.task_env_config, "docker")
        if decision.policy is EffectivePolicy.OPEN:
            return []
        lane = None
        if decision.model_lane:
            from benchflow.providers.litellm_runtime import _docker_host_address

            lane = _docker_host_address()
        # An allowlist, or a no-network run that keeps only the model lane open, is
        # enforced by the egress sidecar; both need a writable rollout dir to stage
        # the proxy compose override.
        if self.rollout_paths and (
            decision.policy is EffectivePolicy.ALLOWLIST or lane
        ):
            hosts = (
                decision.allowed_hosts
                if decision.policy is EffectivePolicy.ALLOWLIST
                else ()
            )
            override = build_egress_override(
                hosts,
                out_dir=self.rollout_paths.rollout_dir,
                model_lane=lane,
            )
            return [override]
        # BLOCK_ALL with no lane, or nowhere to stage the proxy → fail closed.
        return [self._DOCKER_COMPOSE_NO_NETWORK_PATH]

    async def relock_network(self) -> dict[str, str]:
        """Apply the task's restrictive network policy to the running container.

        The container came up open so the agent could install (install-before-
        lockdown); now drop it off the public bridge. For allowlist / model-lane
        runs, start the egress sidecar and move the container onto the internal-
        only network, returning the HTTP(S)_PROXY env the agent must use. ``public``
        is a no-op. The ``main`` container is never recreated, so the install
        survives. Returns the proxy env to merge into the agent launch env.
        """
        from benchflow.sandbox._egress import (
            _EGRESS_INTERNAL_NET,
            _EGRESS_PORT,
            _EGRESS_SERVICE,
        )
        from benchflow.sandbox.network_policy import (
            EffectivePolicy,
            resolve_network_decision,
        )

        decision = resolve_network_decision(self.task_env_config, "docker")
        if decision.policy is EffectivePolicy.OPEN:
            return {}

        # Gate _network_policy_compose_paths to emit the real override now.
        self._network_locked = True
        cid = await self._main_container_id()
        if not cid:
            self.logger.warning("relock_network: no 'main' container; skipping")
            return {}

        project = _sanitize_docker_compose_project_name(self.session_id)
        paths = self._network_policy_compose_paths()
        use_sidecar = bool(paths and paths[0] != self._DOCKER_COMPOSE_NO_NETWORK_PATH)

        if use_sidecar:
            # Bring up ONLY the egress sidecar (creates the bf_egress_* networks);
            # --no-deps leaves the already-running 'main' container in place.
            await self._run_docker_compose_command(
                ["up", "--detach", "--no-deps", _EGRESS_SERVICE]
            )
            await self._docker_cli(
                ["network", "connect", f"{project}_{_EGRESS_INTERNAL_NET}", cid],
                check=False,
            )
        # Lockdown: detach the container from the public bridge.
        await self._docker_cli(
            ["network", "disconnect", f"{project}_default", cid], check=False
        )
        self.logger.info(
            "relock_network: %s applied (sidecar=%s)", decision.policy.name, use_sidecar
        )
        if use_sidecar:
            proxy = f"http://{_EGRESS_SERVICE}:{_EGRESS_PORT}"
            return {
                "HTTP_PROXY": proxy,
                "HTTPS_PROXY": proxy,
                "http_proxy": proxy,
                "https_proxy": proxy,
                "NO_PROXY": "localhost,127.0.0.1",
                "no_proxy": "localhost,127.0.0.1",
            }
        return {}

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

    async def _run_docker_compose_build(self) -> None:
        max_attempts = len(_DOCKER_BUILD_RETRY_DELAYS_SEC) + 1
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_docker_compose_command(["build"])
                return
            except RuntimeError as exc:
                if attempt == max_attempts or not _is_retryable_docker_build_error(
                    str(exc)
                ):
                    raise

                delay = _DOCKER_BUILD_RETRY_DELAYS_SEC[attempt - 1]
                self.logger.warning(
                    "Retrying Docker build for %s after transient failure "
                    "(attempt %s/%s, retrying in %.1fs): %s",
                    self.environment_name,
                    attempt,
                    max_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

    async def _run_docker_compose_up(self) -> None:
        max_attempts = len(_COMPOSE_UP_RETRY_DELAYS_SEC) + 1
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_docker_compose_command(["up", "--detach", "--wait"])
                return
            except RuntimeError as exc:
                if attempt == max_attempts or not _is_compose_up_network_race_error(
                    str(exc)
                ):
                    raise

                delay = _COMPOSE_UP_RETRY_DELAYS_SEC[attempt - 1]
                self.logger.warning(
                    "Retrying docker compose up for %s after network "
                    "create/attach race (attempt %s/%s, retrying in %.1fs): %s",
                    self.environment_name,
                    attempt,
                    max_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

    async def start(self, force_build: bool) -> None:
        if self._mounts_json:
            self._mounts_compose_path = self._write_mounts_compose_file()

        self._use_prebuilt = not force_build and bool(self.task_env_config.docker_image)

        # Gate the entire startup phase (build + down + up) — not just build.
        # When images are cached, build is a no-op so a build-only semaphore
        # has no effect, but the simultaneous `compose up` calls still flood
        # the docker daemon (network creation races, prune timeouts).
        build_sem = self._build_semaphore
        if build_sem is not None:
            await build_sem.acquire()
        try:
            if not self._use_prebuilt:
                lock = self._image_build_locks.setdefault(
                    self.environment_name, asyncio.Lock()
                )
                async with lock:
                    await self._run_docker_compose_build()

            with contextlib.suppress(RuntimeError):
                await self._run_docker_compose_command(["down", "--remove-orphans"])

            await self._run_docker_compose_up()
        finally:
            if build_sem is not None:
                build_sem.release()

        await self.exec(
            f"chmod 777 {SandboxPaths.agent_dir} {SandboxPaths.verifier_dir}"
        )

    async def stop(self, delete: bool) -> None:
        # Bounded chown: a hung agent container will make `docker exec` block
        # forever. We don't need the chown to succeed for correctness — it just
        # makes host-side log reading nicer. Time out fast and continue to the
        # actual teardown.
        try:
            await asyncio.wait_for(
                self._chown_to_host_user(str(SandboxPaths.logs_dir), recursive=True),
                timeout=30,
            )
        except TimeoutError:
            self.logger.warning("Chown logs directory timed out; continuing teardown.")
        except Exception as e:
            self.logger.warning(f"Failed to chown logs directory: {e}")

        if self._keep_containers and delete:
            self.logger.warning(
                "Both `keep_containers` and `--delete` option are set. "
                "keep_containers takes precedence."
            )
        # Pass `-t 5` so unresponsive containers are SIGKILLed quickly rather
        # than waiting the default 10s per container, and wrap each call in a
        # hard 90s deadline. If the daemon is wedged we fall through to a
        # force-kill by compose project label so the rollout's gather() can
        # advance instead of stalling the entire batch.
        try:
            if self._keep_containers:
                await self._run_docker_compose_command(
                    ["stop", "-t", "5"], timeout_sec=90
                )
            elif delete:
                await self._run_docker_compose_command(
                    [
                        "down",
                        "--rmi",
                        "all",
                        "--volumes",
                        "--remove-orphans",
                        "-t",
                        "5",
                    ],
                    timeout_sec=120,
                )
            else:
                await self._run_docker_compose_command(
                    ["down", "-t", "5"], timeout_sec=90
                )
        except Exception as e:
            self.logger.warning(
                f"Docker compose down hung/failed ({e}); force-killing project."
            )
            await self._force_kill_project()

    async def _force_kill_project(self) -> None:
        """Last-resort cleanup when `compose down` hangs or fails.

        Lists containers by ``com.docker.compose.project`` label and `docker
        rm -f`s them, then prunes the matching network. We don't propagate
        errors — by the time we're here, the batch just needs to move on.
        """
        project = _sanitize_docker_compose_project_name(self.session_id)
        label = f"label=com.docker.compose.project={project}"
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "ps",
                "-aq",
                "--filter",
                label,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            cids = stdout.decode().split()
            for cid in cids:
                rm_proc = await asyncio.create_subprocess_exec(
                    "docker",
                    "rm",
                    "-f",
                    "-v",
                    cid,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(rm_proc.wait(), timeout=10)
            net_proc = await asyncio.create_subprocess_exec(
                "docker",
                "network",
                "prune",
                "-f",
                "--filter",
                label,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(net_proc.wait(), timeout=10)
        except Exception as e:
            self.logger.warning(f"Force-kill of compose project {project} failed: {e}")

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

    # Container snapshot/restore (Branch substrate)
    #
    # ``docker commit`` captures the ``main`` container's filesystem into a
    # local image; restore re-creates ``main`` from that image. Snapshots
    # are container-level only — they include the filesystem the agent has
    # mutated, but **not** mounted host volumes (rollout dir, verifier dir)
    # and **not** sibling compose services. The Branch lifecycle composes
    # this with the Environment-state snapshot (DB dump) that captures the
    # mounted/sidecar state separately (#384).

    @property
    def supports_snapshot(self) -> bool:
        return True

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        """Commit the current ``main`` container into a re-usable image.

        Uses ``docker commit`` so the snapshot lives in the local Docker
        image store. Returns a :class:`SandboxImage` whose ``ref`` is the
        committed image tag — pass it back to :meth:`restore` to roll the
        ``main`` container back to this checkpoint.
        """
        from benchflow.sandbox.protocol import SandboxSnapshotNotSupported

        container_id = await self._main_container_id()
        if not container_id:
            raise SandboxSnapshotNotSupported(
                "DockerSandbox.snapshot requires the main container to be "
                "running; call start() before snapshot()."
            )
        suffix = name or uuid.uuid4().hex[:12]
        tag = _sanitize_docker_image_name(f"bf-snap-{self.environment_name}-{suffix}")
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "commit",
            container_id,
            tag,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                "docker commit failed: "
                f"{(stderr_bytes or stdout_bytes or b'').decode(errors='replace')}"
            )
        digest = (stdout_bytes or b"").decode(errors="replace").strip()
        self.logger.info(f"Snapshot created: {tag} ({digest})")
        return SandboxImage(
            provider="docker",
            ref=tag,
            meta={"container_id": container_id, "digest": digest},
        )

    async def restore(self, image: SandboxImage) -> None:
        """Restore the ``main`` container from a previously committed image.

        Stops and removes the current ``main`` container, then ``docker
        run``s a replacement from ``image.ref``. Sibling compose services
        are untouched — they keep running as before, matching the
        documented container-only scope of the Sandbox layer.
        """
        from benchflow.sandbox.protocol import SandboxSnapshotNotSupported

        if image.provider != "docker":
            raise SandboxSnapshotNotSupported(
                f"DockerSandbox.restore cannot consume a {image.provider!r} "
                f"snapshot (got ref={image.ref!r}); snapshots are not portable "
                "across providers."
            )

        container_id = await self._main_container_id()
        if container_id:
            await self._docker_cli(["stop", container_id])
            await self._docker_cli(["rm", "-f", container_id])

        project_name = _sanitize_docker_compose_project_name(self.session_id)
        new_name = f"{project_name}-main-restored-{uuid.uuid4().hex[:8]}"

        run_cmd = [
            "run",
            "--detach",
            "--name",
            new_name,
            "--network",
            f"{project_name}_default",
            "--label",
            f"com.docker.compose.project={project_name}",
            "--label",
            "com.docker.compose.service=main",
            image.ref,
            "sleep",
            "infinity",
        ]
        result = await self._docker_cli(run_cmd, check=False)
        if result.return_code != 0:
            raise RuntimeError(
                f"docker run from snapshot {image.ref!r} failed: "
                f"{result.stderr or result.stdout}"
            )
        self.logger.info(f"Snapshot restored: {image.ref} -> {new_name}")

    async def _main_container_id(self) -> str | None:
        """Return the container id of the ``main`` compose service, or None."""
        try:
            result = await self._run_docker_compose_command(
                ["ps", "-q", "main"], check=False
            )
        except RuntimeError:
            return None
        cid = (result.stdout or "").strip().splitlines()
        return cid[0] if cid else None

    async def _docker_cli(self, args: list[str], check: bool = True) -> ExecResult:
        """Run a raw ``docker`` CLI command — bypasses compose."""
        proc = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        result = ExecResult(
            stdout=(stdout_bytes or b"").decode(errors="replace"),
            stderr=(stderr_bytes or b"").decode(errors="replace"),
            return_code=proc.returncode or 0,
        )
        if check and result.return_code != 0:
            raise RuntimeError(
                f"docker {' '.join(args)} failed: {result.stderr or result.stdout}"
            )
        return result

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

        exec_command: list[str] = ["exec", "-T"]

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

    # Prefix for the decoded env file inside the container. A unique 16-hex
    # suffix is appended by the shared wrapper so concurrent exec() calls in one
    # container can't clobber each other's env file.
    _ENV_FILE_PREFIX = "/tmp/.benchflow_exec_env_"

    @classmethod
    def _wrap_command_with_env_file(cls, env: dict[str, str], command: str) -> str:
        """Return *command* prefixed to materialize *env* from a file.

        Thin wrapper over the canonical :func:`wrap_command_with_env_file` so
        the secret-redaction logic lives in exactly one place (shared with the
        Daytona backend). See that function for the full contract — base64
        argv-hiding, mode-0600 file, ``trap ... EXIT`` cleanup, non-identifier
        key skipping (PR #323), and subshell-scoped ``umask 077`` (PR #323).
        """
        return wrap_command_with_env_file(
            env, command, env_path_prefix=cls._ENV_FILE_PREFIX
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
