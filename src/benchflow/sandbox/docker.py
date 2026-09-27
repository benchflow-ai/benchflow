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
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn

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
    COMPOSE_NET_ADMIN_PATH,
    COMPOSE_NO_NETWORK_PATH,
    COMPOSE_PREBUILT_PATH,
    COMPOSE_UP_RETRY_DELAYS_SEC,
    compose_needs_net_admin,
    docker_daemon_unreachable_reason,
    is_compose_up_network_race_error,
)
from benchflow.sandbox._recovery_baseline import (
    DockerRecoveryBaseline,
    capture_baseline,
    validate_baseline,
)
from benchflow.sandbox._snapshot_credentials import (
    CredentialScrubError,
    StashedCredential,
    put_back_credentials,
    scrub_credentials,
)
from benchflow.sandbox.protocol import (
    SandboxImage,
    SandboxRestoreHostConfigUnavailable,
    SandboxSnapshotNotSupported,
)
from benchflow.task.config import NetworkMode, SandboxConfig
from benchflow.task.env import resolve_env_vars
from benchflow.task.paths import RolloutPaths, SandboxPaths

if TYPE_CHECKING:
    from benchflow.sandbox.process import LiveProcess

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

# On macOS the daemon always runs in a Linux VM (Colima, Docker Desktop, ...).
_DOCKER_RUNS_IN_VM = sys.platform == "darwin"


def _docker_unavailable_reason(exc: BaseException) -> str | None:
    """Why a docker command could not reach Docker at all, or None.

    Either the ``docker`` CLI is missing or the daemon is unreachable; any
    other failure (a timeout, a daemon error) returns None.
    """
    if isinstance(exc, FileNotFoundError) and exc.filename == "docker":
        return "docker CLI not found on PATH"
    return docker_daemon_unreachable_reason(str(exc))


def _vm_does_not_share(host_path: Path) -> bool:
    """Whether the Docker VM is expected not to see ``host_path``.

    A Docker VM sees only the host directories it shares. Colima, and Lima
    under it, share only ``$HOME`` by default, so a rollout directory elsewhere
    (for example under ``/tmp``) is bind-mounted as an empty VM-local
    directory. Under ``$HOME``, or on a host where the daemon runs natively,
    the daemon should see the path.
    """
    if not _DOCKER_RUNS_IN_VM:
        return False
    try:
        host_path.resolve().relative_to(Path.home().resolve())
    except ValueError:
        return True
    return False


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


# The benchflow.owned label the leak sweeper filters on (evaluation.py).
_BENCHFLOW_OWNED_LABEL = "benchflow.owned"

# HostConfig keys restore reproduces below. Anything else must hold an unset
# value, so a setting restore cannot rebuild -- including one a newer Docker
# adds -- fails closed instead of vanishing (e.g. a gVisor runtime silently
# restored onto runc).
_REPLAYED_HOST_CONFIG = frozenset(
    {
        "NetworkMode",
        "Tmpfs",
        "RestartPolicy",
        "IpcMode",
        "CapAdd",
        "CapDrop",
        "SecurityOpt",
        "ReadonlyRootfs",
        "NanoCpus",
        "Memory",
        "MemorySwap",
        "MemoryReservation",
        "MemorySwappiness",
        "PidsLimit",
        "ShmSize",
        "Dns",
        "DnsSearch",
        "DnsOptions",
        "Runtime",
        "Init",
        "GroupAdd",
        "OomScoreAdj",
        "CgroupnsMode",
        "LogConfig",
    }
)
# Keys with no setting of their own: mounts are replayed from the container's
# Mounts list, the daemon recomputes masked paths from Privileged/SecurityOpt,
# and the rest only affect the client that created the container.
_DERIVED_HOST_CONFIG = frozenset(
    {
        "Binds",
        "Mounts",
        "MaskedPaths",
        "ReadonlyPaths",
        "ContainerIDFile",
        "ConsoleSize",
    }
)
_UNSET_VALUES: tuple[Any, ...] = (None, False, 0, "", [], {})


def _replayed_run_args(container: dict[str, Any], *, default_network: str) -> list[str]:
    """Replay supported host configuration, rejecting lossy restore before removal.

    Adapted from JeremyJC67's PR #1046. Docker commit preserves
    image configuration, but not mounts, networking, security or resource caps.
    Mounted contents and running processes are not captured by this operation.
    """

    def reject(detail: str) -> NoReturn:
        raise SandboxRestoreHostConfigUnavailable(
            f"Docker snapshot restore cannot preserve {detail}; container left intact"
        )

    host = container.get("HostConfig")
    mounts = container.get("Mounts")
    if not isinstance(host, dict) or not isinstance(mounts, list):
        reject("missing HostConfig or Mounts")
    for key, value in host.items():
        if (
            key not in _REPLAYED_HOST_CONFIG
            and key not in _DERIVED_HOST_CONFIG
            and value not in _UNSET_VALUES
        ):
            reject(f"HostConfig.{key}")
    for mount in host.get("Mounts") or []:
        # Only bind propagation is replayed from the container's Mounts list.
        options = {
            **(mount.get("BindOptions") or {}),
            **(mount.get("VolumeOptions") or {}),
            **(mount.get("TmpfsOptions") or {}),
        }
        options.pop("Propagation", None)
        if any(value not in _UNSET_VALUES for value in options.values()):
            reject(f"mount options for {mount.get('Target')!r}")
    if (host.get("RestartPolicy") or {}).get("Name") not in (None, "", "no"):
        reject("RestartPolicy")
    network = host.get("NetworkMode")
    if (
        not network
        or network in {"host", "default", "bridge"}
        or network.startswith("container:")
    ):
        reject(f"network mode {network!r}")
    networks = (container.get("NetworkSettings") or {}).get("Networks") or {}
    if len(networks) > 1:
        reject("multiple network attachments")
    if network != "none" and network != default_network:
        reject(f"network {network!r} outside the compose project")
    args = ["--network", network]
    for settings in networks.values():
        if settings.get("IPAMConfig"):
            reject("static network addressing")
        for alias in settings.get("Aliases") or []:
            args += ["--network-alias", alias]
    # Docker lists --tmpfs mounts only in HostConfig.Tmpfs, not in Mounts.
    tmpfs = host.get("Tmpfs") or {}
    for destination, options in tmpfs.items():
        args += ["--tmpfs", destination + (f":{options}" if options else "")]
    for mount in mounts:
        kind, destination = mount.get("Type"), mount.get("Destination")
        if not destination:
            reject("a mount without a destination")
        if kind == "tmpfs":
            if destination not in tmpfs:
                reject("tmpfs options absent from HostConfig")
            continue
        source = mount.get("Name") if kind == "volume" else mount.get("Source")
        if kind not in {"bind", "volume"} or not source:
            reject(f"mount type {kind!r} or missing source")
        if "," in source or "," in destination:
            reject("mount paths containing commas")
        spec = f"type={kind},src={source},dst={destination}"
        if mount.get("RW") is False:
            spec += ",readonly"
        if kind == "bind" and mount.get("Propagation"):
            spec += f",bind-propagation={mount['Propagation']}"
        args += ["--mount", spec]
    for key, flag in (
        ("CapAdd", "--cap-add"),
        ("CapDrop", "--cap-drop"),
        ("SecurityOpt", "--security-opt"),
        ("GroupAdd", "--group-add"),
    ):
        for value in host.get(key) or []:
            args += [flag, value]
    if host.get("ReadonlyRootfs"):
        args += ["--read-only"]
    if host.get("Init") is not None:
        args += ["--init" if host["Init"] else "--init=false"]
    if host.get("NanoCpus"):
        args += ["--cpus", f"{host['NanoCpus'] / 1_000_000_000:g}"]
    for key, flag in (
        ("Runtime", "--runtime"),
        ("CgroupnsMode", "--cgroupns"),
        ("OomScoreAdj", "--oom-score-adj"),
        ("Memory", "--memory"),
        ("MemorySwap", "--memory-swap"),
        ("MemoryReservation", "--memory-reservation"),
        ("PidsLimit", "--pids-limit"),
        ("ShmSize", "--shm-size"),
    ):
        if host.get(key):
            args += [flag, str(host[key])]
    if host.get("MemorySwappiness") is not None:
        args += ["--memory-swappiness", str(host["MemorySwappiness"])]
    log_config = host.get("LogConfig") or {}
    if log_config.get("Type"):
        args += ["--log-driver", log_config["Type"]]
    for name, value in (log_config.get("Config") or {}).items():
        args += ["--log-opt", f"{name}={value}"]
    if host.get("IpcMode") not in (None, "", "private"):
        reject("non-private IPC")
    for key, flag in (
        ("Dns", "--dns"),
        ("DnsSearch", "--dns-search"),
        ("DnsOptions", "--dns-option"),
    ):
        for value in host.get(key) or []:
            args += [flag, value]
    return args


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
    _DOCKER_COMPOSE_NET_ADMIN_PATH = COMPOSE_NET_ADMIN_PATH

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
        self._logs_are_mounted = True

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
        # Set just before `compose up`: from then on containers may exist and
        # stop() must tear them down; before it, nothing was started.
        self._compose_up_attempted = False
        self._recovery_baseline: DockerRecoveryBaseline | None = None
        # Branch snapshot images still used by a container when their fork
        # finished; stop() removes them once the containers are gone.
        self._deferred_snapshot_refs: set[str] = set()
        # Credential files scrubbed out of each snapshot image, held in host
        # memory so restore() can put them back (see _snapshot_credentials).
        self._snapshot_credentials: dict[str, list[StashedCredential]] = {}

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
        return self._logs_are_mounted

    async def _probe_verifier_log_mount(self) -> None:
        """Confirm that the daemon can see the host verifier-log bind mount.

        Docker Desktop normally translates WSL paths before they reach its VM.
        A client connected directly to the underlying daemon can bypass that
        translation: Compose still accepts the bind mount, but the container
        and Benchflow then write to different directories.  Treat that case as
        non-mounted so the verifier clears and downloads its remote outputs.

        The gap is logged at debug level where it is expected: a Docker VM
        that does not share the directory (Colima on a Mac with the job
        outside ``$HOME``). Anywhere else it stays a warning.
        """
        if self.rollout_paths is None:
            return

        probe_name = f".benchflow-mount-probe-{uuid.uuid4().hex}"
        host_probe = self.rollout_paths.verifier_dir / probe_name
        sandbox_probe = SandboxPaths.verifier_dir / probe_name
        host_probe.parent.mkdir(parents=True, exist_ok=True)
        host_probe.touch()
        try:
            result = await self.exec(
                f"test -f {shlex.quote(str(sandbox_probe))}",
                user="root",
                timeout_sec=10,
            )
            self._logs_are_mounted = result.return_code == 0
        finally:
            host_probe.unlink(missing_ok=True)

        if self._logs_are_mounted:
            return
        verifier_dir = self.rollout_paths.verifier_dir
        if _vm_does_not_share(verifier_dir):
            # Expected (Colima on a Mac, jobs outside $HOME) and harmless.
            self.logger.debug(
                "Verifier-log directory %s is outside $HOME, which the Docker VM "
                "(Colima by default) does not share; verifier outputs will be "
                "copied back from the container instead.",
                verifier_dir,
            )
            return
        self.logger.warning(
            "Docker verifier-log bind mount is not visible inside the "
            "container; verifier outputs will be copied back explicitly. The "
            "daemon cannot see %s, so it may be remote or reached without path "
            "translation (check DOCKER_HOST and `docker context ls`).",
            verifier_dir,
        )

    def _docker_client_env(self) -> dict[str, str] | None:
        """Environment for raw ``docker`` subprocesses; None inherits the caller's.

        The remote provider returns an environment pointed at its host.
        """
        return None

    def _is_retryable_build_error(self, message: str) -> bool:
        return _is_retryable_docker_build_error(message)

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @property
    def _environment_docker_compose_path(self) -> Path:
        return self.environment_dir / "docker-compose.yaml"

    @property
    def _pre_compose_hook_path(self) -> Path:
        return self.environment_dir / "benchflow-pre-compose.sh"

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

        if compose_needs_net_admin(self.task_env_config):
            paths.append(self._DOCKER_COMPOSE_NET_ADMIN_PATH)

        return paths

    def _docker_compose_env(self) -> dict[str, str]:
        env = self._env_vars.to_env_dict(include_os_env=True)
        if self._compose_task_env:
            env.update(self._compose_task_env)
        if self._persistent_env:
            env.update(self._persistent_env)
        return env

    def _write_mounts_compose_file(self) -> Path:
        compose = {"services": {"main": {"volumes": self._mounts_json}}}
        assert self.rollout_paths is not None
        path = self.rollout_paths.rollout_dir / "docker-compose-mounts.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(compose, indent=2))
        return path

    def _validate_definition(self) -> None:
        if self.task_env_config.docker_image:
            # Prebuilt-image task: compose references the image directly and
            # the build step is skipped (matches modal/agentcore/apple).
            return
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

        env = self._docker_compose_env()

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

    async def _run_pre_compose_hook(self) -> None:
        hook = self._pre_compose_hook_path
        if not hook.is_file():
            return

        timeout_sec = max(120, round(self.task_env_config.build_timeout_sec))
        process = await asyncio.create_subprocess_exec(
            "sh",
            str(hook.resolve().absolute()),
            cwd=str(self.environment_dir.resolve().absolute()),
            env=self._docker_compose_env(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout_bytes, _ = await asyncio.wait_for(
                process.communicate(), timeout=timeout_sec
            )
        except TimeoutError:
            process.terminate()
            try:
                stdout_bytes, _ = await asyncio.wait_for(
                    process.communicate(), timeout=5
                )
            except TimeoutError:
                process.kill()
                stdout_bytes, _ = await process.communicate()
            output = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
            raise RuntimeError(
                f"Pre-compose hook timed out after {timeout_sec} seconds for "
                f"environment {self.environment_name}. Output: {output}"
            ) from None

        output = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
        if process.returncode:
            raise RuntimeError(
                f"Pre-compose hook failed for environment {self.environment_name}. "
                f"Return code: {process.returncode}. Output: {output}"
            )

    async def _run_docker_compose_build(self) -> None:
        max_attempts = len(_DOCKER_BUILD_RETRY_DELAYS_SEC) + 1
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_docker_compose_command(["build"])
                return
            except RuntimeError as exc:
                if attempt == max_attempts or not self._is_retryable_build_error(
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

    async def capture_recovery_baseline(
        self, task_digest: str, effective_config_digest: str
    ) -> DockerRecoveryBaseline:
        return await capture_baseline(self, task_digest, effective_config_digest)

    def use_recovery_baseline(self, baseline: DockerRecoveryBaseline) -> None:
        if (
            self._uses_compose
            or self._mounts_json
            or self._pre_compose_hook_path.exists()
        ):
            raise ValueError(
                "Custom runtime topology cannot use a recovery image lease"
            )
        self._recovery_baseline = baseline

    async def start(self, force_build: bool) -> None:
        if self._mounts_json:
            self._mounts_compose_path = self._write_mounts_compose_file()

        if self._recovery_baseline is not None:
            if force_build:
                raise ValueError("Recovery must never rebuild its original baseline")
            effective = self.task_env_config.model_copy(deep=True)
            effective.allow_internet = self._recovery_baseline.effective_allow_internet
            await validate_baseline(
                self, self._recovery_baseline, sandbox_config=effective
            )
            self.task_env_config = effective
            self._env_vars.prebuilt_image_name = self._recovery_baseline.image_id
        self._use_prebuilt = self._recovery_baseline is not None or (
            not force_build and bool(self.task_env_config.docker_image)
        )

        # Gate the entire startup phase (build + down + up) — not just build.
        # When images are cached, build is a no-op so a build-only semaphore
        # has no effect, but the simultaneous `compose up` calls still flood
        # the docker daemon (network creation races, prune timeouts).
        build_sem = self._build_semaphore
        if build_sem is not None:
            await build_sem.acquire()
        try:
            if self._recovery_baseline is None:
                await self._run_pre_compose_hook()

            if not self._use_prebuilt:
                lock = self._image_build_locks.setdefault(
                    self.environment_name, asyncio.Lock()
                )
                async with lock:
                    await self._run_docker_compose_build()

            with contextlib.suppress(RuntimeError):
                await self._run_docker_compose_command(["down", "--remove-orphans"])

            self._compose_up_attempted = True
            if self._recovery_baseline is None:
                await self._run_docker_compose_up()
            else:
                await self._run_docker_compose_command(
                    ["up", "--detach", "--wait", "--no-build", "--pull", "never"]
                )
                container_id = await self._main_container_id()
                if (
                    not container_id
                    or (await self._inspect_container(container_id)).get("Image")
                    != self._recovery_baseline.image_id
                ):
                    raise RuntimeError(
                        "Recovery container did not use the original image"
                    )
        finally:
            if build_sem is not None:
                build_sem.release()

        await self._prepare_log_dirs_after_up()

    async def _prepare_log_dirs_after_up(self) -> None:
        """Make the mounted log folders writable and check the mount is shared."""
        await self.exec(
            f"chmod 777 {SandboxPaths.agent_dir} {SandboxPaths.verifier_dir}"
        )
        await self._probe_verifier_log_mount()

    def _down_command(self, delete: bool) -> list[str]:
        """The ``compose down`` arguments teardown runs (unless containers are kept)."""
        if delete and self._recovery_baseline is None:
            return ["down", "--rmi", "all", "--volumes", "--remove-orphans", "-t", "5"]
        return ["down", "-t", "5"]

    async def stop(self, delete: bool) -> None:
        # Bounded chown: a hung agent container will make `docker exec` block
        # forever. We don't need the chown to succeed for correctness — it just
        # makes host-side log reading nicer. Time out fast and continue to the
        # actual teardown. Before `compose up` there is no container to chown.
        try:
            if self._compose_up_attempted:
                await asyncio.wait_for(
                    self._chown_to_host_user(
                        str(SandboxPaths.logs_dir), recursive=True
                    ),
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
            else:
                down = self._down_command(delete)
                await self._run_docker_compose_command(
                    down, timeout_sec=120 if "--rmi" in down else 90
                )
        except Exception as e:
            unavailable = _docker_unavailable_reason(e)
            if unavailable is not None and not self._compose_up_attempted:
                # Startup failed because Docker itself is unavailable, and the
                # rollout already reported that; no container was started.
                self.logger.debug(
                    "Skipped teardown of %s: Docker is unavailable (%s) and no "
                    "container was started.",
                    _sanitize_docker_compose_project_name(self.session_id),
                    unavailable,
                )
            else:
                self.logger.warning(
                    f"Docker compose down hung/failed ({e}); force-killing project."
                )
                await self._force_kill_project()
        self._snapshot_credentials.clear()
        await self._delete_deferred_snapshots()

    async def _delete_deferred_snapshots(self) -> None:
        """Remove branch snapshot images that were in use when their fork ended."""
        for ref in sorted(self._deferred_snapshot_refs):
            try:
                gone = await self._remove_snapshot_image(ref)
            except Exception as e:
                gone = False
                self.logger.warning(f"Snapshot image {ref} removal failed: {e}")
            if gone:
                self._deferred_snapshot_refs.discard(ref)
            else:
                self.logger.warning(
                    f"Snapshot image {ref} is still in use (kept container?); "
                    f"remove it with `docker image rm {ref}` when done"
                )

    async def _remove_snapshot_image(self, ref: str) -> bool:
        """``docker image rm`` without force; True when the image is gone."""
        result = await self._docker_cli(["image", "rm", ref], check=False)
        if result.return_code == 0:
            return True
        return "no such image" in f"{result.stderr} {result.stdout}".lower()

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
                env=self._docker_client_env(),
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
                    env=self._docker_client_env(),
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
                env=self._docker_client_env(),
            )
            await asyncio.wait_for(net_proc.wait(), timeout=10)
        except Exception as e:
            self.logger.warning(f"Force-kill of compose project {project} failed: {e}")

    async def upload_file(
        self, source_path: Path | str, target_path: str, *, mode: str | None = None
    ) -> None:
        target_parent = str(Path(target_path).parent)
        if target_parent not in {"", "."}:
            await self.exec(f"mkdir -p {shlex.quote(target_parent)}", user="root")
        await self._run_docker_compose_command(
            ["cp", str(source_path), f"main:{target_path}"],
            check=True,
        )
        await self._apply_upload_mode(target_path, mode)

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
        container_id = await self._main_container_id()
        if not container_id:
            raise SandboxSnapshotNotSupported(
                "DockerSandbox.snapshot requires the main container to be "
                "running; call start() before snapshot()."
            )
        suffix = name or uuid.uuid4().hex[:12]
        tag = _sanitize_docker_image_name(f"bf-snap-{self.environment_name}-{suffix}")
        # Agent credential files stay out of the image (_snapshot_credentials).
        ops = self._credential_ops(container_id)
        stash = await scrub_credentials(ops)
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "commit",
                # docker commit copies the container's compose labels; with
                # them, `compose down --rmi all` at teardown deletes the
                # snapshot as one of the project's images (Compose 5.5), and
                # a kept checkpoint is gone before a retry can use it.
                "--change",
                "LABEL com.docker.compose.project=",
                "--change",
                "LABEL com.docker.compose.service=",
                container_id,
                tag,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._docker_client_env(),
            )
            stdout_bytes, stderr_bytes = await proc.communicate()
        finally:
            if stash:
                await put_back_credentials(ops, stash)
        if proc.returncode != 0:
            raise RuntimeError(
                "docker commit failed: "
                f"{(stderr_bytes or stdout_bytes or b'').decode(errors='replace')}"
            )
        digest = (stdout_bytes or b"").decode(errors="replace").strip()
        if stash:
            self._snapshot_credentials[tag] = stash
        self.logger.info(
            f"Snapshot created: {tag} ({digest}); "
            f"{len(stash)} credential file(s) kept out of it"
        )
        return SandboxImage(
            provider="docker",
            ref=tag,
            meta={"container_id": container_id, "digest": digest},
        )

    async def restore(self, image: SandboxImage) -> None:
        """Restore the ``main`` container from a previously committed image.

        Inspects the live ``main`` container, stops and removes it, then
        ``docker run``s a replacement from ``image.ref`` with the inspected host
        configuration. The bind mounts matter most: without them the rollout's
        ``verifier``/``agent``/``artifacts`` output stays inside the container
        and a verifier reward is silently lost. Sibling compose services are
        untouched, matching the container-only scope of the Sandbox layer.

        Raises :class:`~benchflow.sandbox.protocol.SandboxRestoreHostConfigUnavailable`
        before removing anything when the live container cannot be resolved or
        inspected, or its host configuration cannot be replayed.
        """
        if image.provider != "docker":
            raise SandboxSnapshotNotSupported(
                f"DockerSandbox.restore cannot consume a {image.provider!r} "
                f"snapshot (got ref={image.ref!r}); snapshots are not portable "
                "across providers."
            )

        project_name = _sanitize_docker_compose_project_name(self.session_id)
        default_network = f"{project_name}_default"

        container_id = await self._main_container_id()
        if not container_id:
            raise SandboxRestoreHostConfigUnavailable(
                f"DockerSandbox.restore({image.ref!r}) cannot resolve the "
                f"'main' container of compose project {project_name!r}, so its "
                "bind mounts, network and resource limits cannot be replayed"
            )
        # Inspected *before* removal — the host config only exists while
        # the container does.
        try:
            inspected = await self._inspect_container(container_id)
        except RuntimeError as exc:
            raise SandboxRestoreHostConfigUnavailable(
                f"DockerSandbox.restore({image.ref!r}) cannot read the host "
                f"config of the 'main' container {container_id!r}, so the "
                f"replacement cannot be made equivalent to it: {exc}"
            ) from exc
        if self.task_env_config.network_mode in (
            NetworkMode.DENYLIST,
            NetworkMode.ALLOWLIST,
        ):
            raise SandboxRestoreHostConfigUnavailable(
                "Docker snapshot restore cannot preserve live "
                f"{self.task_env_config.network_mode.value.upper()} firewall rules; "
                "container left intact"
            )
        replayed = _replayed_run_args(inspected, default_network=default_network)
        owned = ((inspected.get("Config") or {}).get("Labels") or {}).get(
            _BENCHFLOW_OWNED_LABEL
        )
        owned_label = (
            ["--label", f"{_BENCHFLOW_OWNED_LABEL}={owned}"]
            if owned is not None
            else []
        )
        await self._docker_cli(["stop", container_id])
        await self._docker_cli(["rm", "-f", container_id])

        new_name = f"{project_name}-main-restored-{uuid.uuid4().hex[:8]}"

        run_cmd = [
            "run",
            "--detach",
            "--name",
            new_name,
            "--label",
            f"com.docker.compose.project={project_name}",
            "--label",
            "com.docker.compose.service=main",
            *owned_label,
            *replayed,
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
        self.logger.info(
            "Snapshot restored: %s -> %s (replayed host config: %s)",
            image.ref,
            new_name,
            " ".join(replayed) or "none",
        )
        stash = self._snapshot_credentials.get(image.ref)
        if stash:
            await put_back_credentials(self._credential_ops(new_name), stash)

    async def adopt_snapshot(self, image: SandboxImage) -> None:
        """Remember the live container's credential files for ``image``, a
        snapshot of this state taken elsewhere (a kept checkpoint), so a
        restore from it puts them back. The live files are left as they are."""
        ops = self._credential_ops(await self._main_container_id())
        stash = await scrub_credentials(ops)
        if stash:
            await put_back_credentials(ops, stash)
            self._snapshot_credentials[image.ref] = stash

    def _credential_ops(self, container: str) -> _DockerCredentialOps:
        return _DockerCredentialOps(self, container)

    async def delete_snapshot(self, image: SandboxImage) -> bool:
        """Remove a committed snapshot image.

        The image a restored container runs from cannot be removed without
        force (and forcing only untags it, leaving the layers behind), so an
        image still in use is removed by :meth:`stop` after the containers
        are gone. Returns True when the image is gone now.
        """
        if image.provider != "docker":
            raise SandboxSnapshotNotSupported(
                f"DockerSandbox.delete_snapshot cannot delete a {image.provider!r} "
                f"snapshot (got ref={image.ref!r})"
            )
        self._snapshot_credentials.pop(image.ref, None)
        if await self._remove_snapshot_image(image.ref):
            self._deferred_snapshot_refs.discard(image.ref)
            return True
        self._deferred_snapshot_refs.add(image.ref)
        self.logger.info(
            f"Snapshot image {image.ref} is still in use; removing it at stop()"
        )
        return False

    async def _inspect_container(self, container_id: str) -> dict[str, Any]:
        """``docker inspect`` one container as a dict — raises if it cannot."""
        result = await self._docker_cli(["inspect", container_id], check=False)
        if result.return_code != 0:
            raise RuntimeError(
                f"docker inspect {container_id!r} failed: "
                f"{result.stderr or result.stdout}"
            )
        try:
            payload = json.loads(result.stdout or "")
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"docker inspect {container_id!r} returned unparseable JSON: {exc}"
            ) from exc
        if (
            not isinstance(payload, list)
            or len(payload) != 1
            or not isinstance(payload[0], dict)
        ):
            raise RuntimeError(
                f"docker inspect {container_id!r} returned no container object"
            )
        return payload[0]

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
            env=self._docker_client_env(),
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

    async def live_process(self, *, agent: str | None = None) -> LiveProcess:
        from benchflow.sandbox.process import DockerProcess

        return DockerProcess.from_sandbox_env(self)

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

        client_env = self._docker_client_env()
        argv = [
            "bash",
            "-c",
            f"{variables}; "
            + " ".join([*compose_base, "exec", "-it", "main", "bash"])
            + "; "
            + " ".join([*compose_base, "down"]),
        ]
        if client_env is None:
            os.execvp("bash", argv)
        else:
            os.execvpe("bash", argv, client_env)


class _DockerCredentialOps:
    """Root access to one container for the snapshot credential scrub.

    File contents travel over ``docker exec`` stdin/stdout as base64, never as
    command-line arguments; Docker keeps no copy of exec output in the
    container filesystem.
    """

    def __init__(self, sandbox: DockerSandbox, container: str) -> None:
        self._sandbox = sandbox
        self._container = container

    async def run(self, command: str) -> ExecResult:
        return await self._sandbox._docker_cli(
            ["exec", "-u", "0", self._container, "sh", "-c", command], check=False
        )

    async def read(self, path: str) -> bytes:
        result = await self.run(f"base64 < {shlex.quote(path)}")
        if result.return_code != 0:
            raise CredentialScrubError(f"could not read {path} before the snapshot")
        return base64.b64decode("".join((result.stdout or "").split()))

    async def write(self, path: str, content: bytes) -> None:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            "-u",
            "0",
            self._container,
            "sh",
            "-c",
            f"umask 077 && base64 -d > {shlex.quote(path)}",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._sandbox._docker_client_env(),
        )
        _, stderr = await proc.communicate(base64.b64encode(content))
        if proc.returncode != 0:
            raise CredentialScrubError(
                f"could not write {path} back: "
                f"{(stderr or b'').decode(errors='replace').strip()[:300]}"
            )
