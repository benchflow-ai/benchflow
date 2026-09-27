"""Sandboxes on a Docker host the user controls (``--sandbox remote-docker``).

The local Docker provider (:class:`~benchflow.sandbox.docker.DockerSandbox`)
with the daemon somewhere else, reached over ``ssh://user@host[:port]`` or
``tcp://host:port`` with TLS. Compose files, sandbox user, verifier
hardening, network modes and snapshots are the local provider's. What
differs:

- Every ``docker`` and ``docker compose`` call gets ``DOCKER_HOST`` set
  explicitly and ``DOCKER_CONTEXT`` removed, so the caller's docker context
  never changes the target.
- No host path is bind-mounted: the remote daemon cannot see the rollout
  folders and would create empty ones on its own disk. Logs, verifier output
  and artifacts are copied back (``is_mounted`` is False).
- The host is checked (``docker info``) before anything starts: an
  unreachable host or one with fewer CPUs or less memory than the task asks
  for is a :class:`~benchflow.sandbox.protocol.SandboxStartupError`.
- Teardown always removes volumes and orphans, then lists what is left under
  the rollout's compose project label and removes it.
- The ssh user in the URL (often an access token) is redacted from every
  message and log line.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._compose import COMPOSE_REMOTE_BASE_PATH
from benchflow.sandbox.docker import (
    DockerSandbox,
    _is_retryable_docker_build_error,
    _sanitize_docker_compose_project_name,
)
from benchflow.sandbox.protocol import SandboxStartupError
from benchflow.task.paths import SandboxPaths

REMOTE_DOCKER_HOST_ENV = "BENCHFLOW_REMOTE_DOCKER_HOST"
_PROBE_TIMEOUT_SEC = 20
_REASON_LIMIT = 300
_TLS_FILES = ("ca.pem", "cert.pem", "key.pem")
_DISK_FULL = re.compile(r"no space left on device", re.IGNORECASE)
_LIST_TIMEOUT_SEC = 30


class RemoteDockerConfigError(ValueError):
    """The remote Docker host is not configured, or configured unsafely."""


@dataclass(frozen=True)
class RemoteDockerHost:
    """One remote daemon endpoint and the client settings that reach it."""

    url: str
    user: str | None = None
    cert_path: str | None = None

    @property
    def display(self) -> str:
        """The URL with the ssh user (often a secret token) replaced."""
        if not self.user:
            return self.url
        parts = urlsplit(self.url)
        netloc = parts.netloc.rsplit("@", 1)[-1]
        return f"{parts.scheme}://***@{netloc}"

    def redact(self, text: str) -> str:
        """``text`` with every occurrence of the ssh user replaced."""
        if not text or not self.user:
            return text
        text = text.replace(self.url, self.display)
        user = re.escape(self.user)
        text = re.sub(rf"(?<![\w.-]){user}(?=[@:])", "***", text)
        return re.sub(rf"(-l\s+){user}(?![\w.-])", r"\1***", text)

    def apply(self, env: dict[str, str]) -> dict[str, str]:
        """Point a docker CLI environment at this host (in place; returned)."""
        env.pop("DOCKER_CONTEXT", None)
        env["DOCKER_HOST"] = self.url
        if self.cert_path is None:
            env.pop("DOCKER_TLS_VERIFY", None)
            env.pop("DOCKER_CERT_PATH", None)
        else:
            env["DOCKER_TLS_VERIFY"] = "1"
            env["DOCKER_CERT_PATH"] = self.cert_path
        return env

    def client_env(self) -> dict[str, str]:
        """The caller's environment, pointed at this host."""
        return self.apply(dict(os.environ))


def resolve_remote_docker_host(
    environ: dict[str, str] | None = None,
) -> RemoteDockerHost:
    """Read and validate the host from ``BENCHFLOW_REMOTE_DOCKER_HOST`` or ``DOCKER_HOST``."""
    env = os.environ if environ is None else environ
    url = (env.get(REMOTE_DOCKER_HOST_ENV) or env.get("DOCKER_HOST") or "").strip()
    if not url:
        raise RemoteDockerConfigError(
            f"remote-docker needs {REMOTE_DOCKER_HOST_ENV} or DOCKER_HOST: "
            "ssh://user@host[:port] or tcp://host:2376 with DOCKER_TLS_VERIFY=1"
        )
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme in ("unix", "npipe", "fd"):
        raise RemoteDockerConfigError(
            f"{url} is a local Docker socket; use --sandbox docker for a local "
            "daemon, or set an ssh:// or tcp:// host for remote-docker"
        )
    if scheme not in ("ssh", "tcp") or not parts.hostname:
        raise RemoteDockerConfigError(
            f"unsupported remote Docker host {_safe(url)!r}: use "
            "ssh://user@host[:port] or tcp://host:port with TLS"
        )
    if scheme == "ssh":
        return RemoteDockerHost(url=url, user=parts.username)
    verify = env.get("DOCKER_TLS_VERIFY", "").strip()
    if not verify or verify == "0":
        raise RemoteDockerConfigError(
            f"tcp host {url} needs TLS: set DOCKER_TLS_VERIFY=1 and DOCKER_CERT_PATH "
            "(ca.pem, cert.pem, key.pem); a plain tcp daemon gives root on the host "
            "to anyone who can reach the port, so it is refused"
        )
    cert_path = env.get("DOCKER_CERT_PATH") or str(Path.home() / ".docker")
    missing = [n for n in _TLS_FILES if not (Path(cert_path) / n).is_file()]
    if missing:
        raise RemoteDockerConfigError(
            f"tcp host {url} with TLS: {', '.join(missing)} not found in {cert_path} "
            "(DOCKER_CERT_PATH)"
        )
    return RemoteDockerHost(url=url, cert_path=cert_path)


def _safe(url: str) -> str:
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    return f"{parts.scheme}://***@{parts.netloc.rsplit('@', 1)[-1]}"


@dataclass(frozen=True)
class HostCapacity:
    """What ``docker info`` reports for the remote host.

    On a daemon inside a VM or container, these are the numbers the daemon
    sees, which can be the physical machine's rather than a quota's, so the
    capacity check is a lower bound.
    """

    cpus: int
    memory_mb: int
    server_version: str


def unreachable_reason(text: str) -> str:
    """The line of docker/ssh output that says why the host could not be reached."""
    marker = text.find("stderr=")
    if marker >= 0:
        reason = text[marker + len("stderr=") :].strip().splitlines()
        line = reason[0] if reason else ""
    else:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        line = next(
            (
                ln
                for ln in lines
                if any(
                    key in ln.lower()
                    for key in (
                        "cannot connect",
                        "error during connect",
                        "tls",
                        "unable to resolve docker endpoint",
                        "timed out",
                    )
                )
            ),
            lines[-1] if lines else "no output",
        )
    if len(line) > _REASON_LIMIT:
        line = line[: _REASON_LIMIT - 3] + "..."
    return line


def probe_remote_docker(
    host: RemoteDockerHost, *, timeout_sec: int = _PROBE_TIMEOUT_SEC
) -> HostCapacity:
    """Run ``docker info`` on ``host``; raise :class:`SandboxStartupError` if it cannot."""
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=host.client_env(),
        )
    except FileNotFoundError as exc:
        raise SandboxStartupError(
            "Remote Docker host unreachable: docker CLI not found on PATH "
            "(remote-docker runs the local docker CLI against the remote daemon)"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SandboxStartupError(
            f"Remote Docker host unreachable: {host.display}: no answer to "
            f"`docker info` within {timeout_sec}s"
        ) from exc
    output = f"{proc.stderr or ''}\n{proc.stdout or ''}"
    info: dict[str, Any] = {}
    if proc.returncode == 0:
        try:
            info = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            info = {}
    if proc.returncode != 0 or not info.get("ServerVersion"):
        reason = host.redact(unreachable_reason(output))
        raise SandboxStartupError(
            f"Remote Docker host unreachable: {host.display}: {reason}"
        )
    return HostCapacity(
        cpus=int(info.get("NCPU") or 0),
        memory_mb=int(info.get("MemTotal") or 0) // (1024 * 1024),
        server_version=str(info.get("ServerVersion")),
    )


def check_capacity(
    capacity: HostCapacity, *, cpus: int, memory_mb: int, host_display: str
) -> None:
    """Refuse a task that asks for more CPUs or memory than the host has."""
    problems = []
    if capacity.cpus and cpus > capacity.cpus:
        problems.append(f"task needs {cpus} CPUs; {host_display} has {capacity.cpus}")
    if capacity.memory_mb and memory_mb > capacity.memory_mb:
        problems.append(
            f"task needs {memory_mb} MB of memory; {host_display} has "
            f"{capacity.memory_mb} MB"
        )
    if problems:
        raise SandboxStartupError(
            "Remote Docker host lacks resources: " + "; ".join(problems)
        )


class RemoteDockerSandbox(DockerSandbox):
    """:class:`DockerSandbox` on a remote daemon. See the module docstring."""

    _DOCKER_COMPOSE_BASE_PATH = COMPOSE_REMOTE_BASE_PATH

    def __init__(
        self,
        *args: Any,
        docker_host: RemoteDockerHost | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("mounts_json"):
            raise RemoteDockerConfigError(
                "remote-docker cannot use host mounts: the paths are on this "
                "machine, not on the remote Docker host"
            )
        self.docker_host = docker_host or resolve_remote_docker_host()
        super().__init__(*args, **kwargs)
        self._logs_are_mounted = False
        self._host_checked = False

    @property
    def _project_name(self) -> str:
        return _sanitize_docker_compose_project_name(self.session_id)

    # --- where docker calls go ------------------------------------------------

    def _docker_client_env(self) -> dict[str, str] | None:
        return self.docker_host.client_env()

    def _docker_compose_env(self) -> dict[str, str]:
        return self.docker_host.apply(super()._docker_compose_env())

    def _remote_error(self, exc: RuntimeError) -> RuntimeError:
        message = self.docker_host.redact(str(exc))
        if _DISK_FULL.search(message):
            message = f"remote Docker host is out of disk space ({self.docker_host.display}): {message}"
        if message == str(exc):
            return exc
        return RuntimeError(message)

    async def _run_docker_compose_command(
        self, command: list[str], check: bool = True, timeout_sec: int | None = None
    ) -> ExecResult:
        try:
            result = await super()._run_docker_compose_command(
                command, check=check, timeout_sec=timeout_sec
            )
        except RuntimeError as exc:
            redacted = self._remote_error(exc)
            if redacted is exc:
                raise
            raise redacted from None
        return self._redact_result(result)

    async def _docker_cli(self, args: list[str], check: bool = True) -> ExecResult:
        try:
            result = await super()._docker_cli(args, check=check)
        except RuntimeError as exc:
            redacted = self._remote_error(exc)
            if redacted is exc:
                raise
            raise redacted from None
        return self._redact_result(result)

    def _redact_result(self, result: ExecResult) -> ExecResult:
        if self.docker_host.user is None:
            return result
        return ExecResult(
            stdout=self.docker_host.redact(result.stdout or "") or result.stdout,
            stderr=self.docker_host.redact(result.stderr or "") or result.stderr,
            return_code=result.return_code,
        )

    def _is_retryable_build_error(self, message: str) -> bool:
        # A full remote disk stays full across a retry seconds later.
        if _DISK_FULL.search(message):
            return False
        return _is_retryable_docker_build_error(message)

    # --- lifecycle --------------------------------------------------------------

    async def start(self, force_build: bool) -> None:
        capacity = await asyncio.to_thread(probe_remote_docker, self.docker_host)
        check_capacity(
            capacity,
            cpus=self.task_env_config.cpus,
            memory_mb=self.task_env_config.memory_mb,
            host_display=self.docker_host.display,
        )
        self._host_checked = True
        self.logger.info(
            "Remote Docker host %s: Docker %s, %d CPUs, %d MB",
            self.docker_host.display,
            capacity.server_version,
            capacity.cpus,
            capacity.memory_mb,
        )
        await super().start(force_build)

    async def _prepare_log_dirs_after_up(self) -> None:
        """Create the log folders in the container; nothing is mounted."""
        dirs = " ".join(
            str(p)
            for p in (
                SandboxPaths.agent_dir,
                SandboxPaths.verifier_dir,
                SandboxPaths.artifacts_dir,
            )
        )
        result = await self.exec(f"mkdir -p {dirs} && chmod 777 {dirs}", user="root")
        if result.return_code != 0:
            raise RuntimeError(
                f"could not create the log folders in the remote container: "
                f"{result.stdout or result.stderr}"
            )
        self._logs_are_mounted = False

    async def _chown_to_host_user(
        self, path: str, recursive: bool = False, service: str = "main"
    ) -> None:
        # ``docker cp`` writes copied files as the caller; owner ids from this
        # machine mean nothing inside a container on another host.
        return None

    def _down_command(self, delete: bool) -> list[str]:
        command = ["down", "--volumes", "--remove-orphans", "-t", "5"]
        if delete and self._recovery_baseline is None:
            command[1:1] = ["--rmi", "all"]
        return command

    async def stop(self, delete: bool) -> None:
        if not self._host_checked:
            # The host check failed or start() never ran: nothing exists.
            self._snapshot_credentials.clear()
            return
        await super().stop(delete)
        if self._keep_containers:
            return
        leftovers = await self._project_leftovers()
        if leftovers and any(leftovers.values()):
            await self._remove_leftovers(leftovers)
            leftovers = await self._project_leftovers()
        if leftovers is None:
            self.logger.warning(
                "Could not confirm cleanup of compose project %s on %s (host "
                "unreachable). Remove it when the host is back: DOCKER_HOST=%s "
                "docker compose -p %s down --volumes --remove-orphans",
                self._project_name,
                self.docker_host.display,
                self.docker_host.display,
                self._project_name,
            )
        elif any(leftovers.values()):
            self.logger.warning(
                "Compose project %s left on %s after teardown: %s",
                self._project_name,
                self.docker_host.display,
                ", ".join(
                    f"{kind} {' '.join(ids)}" for kind, ids in leftovers.items() if ids
                ),
            )

    async def _force_kill_project(self) -> None:
        leftovers = await self._project_leftovers()
        if leftovers:
            await self._remove_leftovers(leftovers)

    async def _project_leftovers(self) -> dict[str, list[str]] | None:
        """Containers, networks and volumes with this project's label; None if unreachable."""
        label = f"label=com.docker.compose.project={self._project_name}"
        found: dict[str, list[str]] = {}
        for kind, args in (
            ("containers", ["ps", "-aq", "--filter", label]),
            ("networks", ["network", "ls", "-q", "--filter", label]),
            ("volumes", ["volume", "ls", "-q", "--filter", label]),
        ):
            try:
                result = await asyncio.wait_for(
                    self._docker_cli(args, check=False), timeout=_LIST_TIMEOUT_SEC
                )
            except (TimeoutError, OSError):
                return None
            if result.return_code != 0:
                return None
            found[kind] = str(result.stdout or "").split()
        return found

    async def _remove_leftovers(self, leftovers: dict[str, list[str]]) -> None:
        for kind, args in (
            ("containers", ["rm", "-f", "-v"]),
            ("networks", ["network", "rm"]),
            ("volumes", ["volume", "rm", "-f"]),
        ):
            ids = leftovers.get(kind) or []
            if not ids:
                continue
            try:
                await asyncio.wait_for(
                    self._docker_cli([*args, *ids], check=False),
                    timeout=_LIST_TIMEOUT_SEC,
                )
            except (TimeoutError, OSError) as exc:
                self.logger.warning(
                    "Removing %s of compose project %s failed: %s",
                    kind,
                    self._project_name,
                    exc,
                )
