"""Native DaytonaSandbox — internalized from Harbor with RL-first terminology.

Supports two strategies:
- Direct: single-container sandbox (Dockerfile only)
- DinD: Docker-in-Docker with compose (docker-compose.yaml present)
"""

from __future__ import annotations

import asyncio
import atexit
import importlib
import logging
import os
import shlex
from abc import abstractmethod
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
except ImportError:  # base install without ``sandbox-daytona`` extras (#358)
    # ``tenacity`` ships under the ``sandbox-daytona`` extra. The fallbacks below
    # let this module import cleanly in a base install — ``DaytonaSandbox()`` is
    # what requires the real SDK (and tenacity along with it).

    def retry(*_args: Any, **_kwargs: Any) -> Any:
        def _decorator(fn: Any) -> Any:
            return fn

        return _decorator

    def stop_after_attempt(*_args: Any, **_kwargs: Any) -> Any:
        return None

    def wait_exponential(*_args: Any, **_kwargs: Any) -> Any:
        return None

    def retry_if_exception_type(*_args: Any, **_kwargs: Any) -> Any:
        return None


from benchflow._paths import iter_safe_tree
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
    compose_cp_destination,
    compose_mkdir_p_command,
    compose_parent_mkdir_p_command,
)
from benchflow.sandbox.metadata import persist_sandbox_info
from benchflow.sandbox.protocol import (
    SandboxImage,
    SandboxSnapshotNotSupported,
    SandboxStartupError,
)
from benchflow.task.config import SandboxConfig
from benchflow.task.env import resolve_env_vars
from benchflow.task.paths import RolloutPaths, SandboxPaths

# ``SandboxStartupError`` used to live in this module. It now lives in
# ``benchflow.sandbox.protocol`` so a base install without the
# ``sandbox-daytona`` extra can still import ``benchflow.rollout`` (issue #358).
# Re-export here for backward compatibility — existing imports of
# ``benchflow.sandbox.daytona.SandboxStartupError`` keep working.
__all__ = ["DaytonaSandbox", "SandboxStartupError"]


def _ensure_daytona_anyio_compat() -> None:
    """Patch the anyio symbol that Daytona 0.176 imports on newer anyio."""
    try:
        import anyio
    except ImportError:
        return

    if hasattr(anyio, "AsyncContextManagerMixin"):
        return

    class _AsyncContextManagerMixin:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            aclose = getattr(self, "aclose", None)
            if aclose is not None:
                await aclose()

    anyio.AsyncContextManagerMixin = _AsyncContextManagerMixin  # type: ignore[attr-defined]


# Module-level handles for the Daytona SDK. The SDK is shipped under the
# ``sandbox-daytona`` extra and is pulled in lazily — see ``_load_daytona_sdk``
# — so importing this module in a base install does not require it (#358).
_DAYTONA_SDK_LOADED = False
AsyncDaytona: Any = None
AsyncSandbox: Any = None
CreateSandboxFromImageParams: Any = None
CreateSandboxFromSnapshotParams: Any = None
DaytonaNotFoundError: Any = None
FileDownloadRequest: Any = None
FileUpload: Any = None
Image: Any = None
Resources: Any = None
SessionExecuteRequest: Any = None
SnapshotState: Any = None


def _load_daytona_sdk() -> None:
    """Import the optional Daytona SDK on first use.

    The Daytona Python SDK is shipped under the ``sandbox-daytona`` extra; a
    base install of ``benchflow`` must not require it. This helper materializes
    the module-level handles the strategy classes consume, and is idempotent so
    it is cheap to call at the top of each entry-point method.
    """
    global _DAYTONA_SDK_LOADED
    global AsyncDaytona, AsyncSandbox
    global CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams
    global DaytonaNotFoundError, FileDownloadRequest, FileUpload
    global Image, Resources, SessionExecuteRequest, SnapshotState

    if _DAYTONA_SDK_LOADED:
        return

    _ensure_daytona_anyio_compat()
    try:
        _daytona = importlib.import_module("daytona")
        _snapshot = importlib.import_module("daytona._async.snapshot")
    except ImportError as e:
        raise ImportError(
            "The Daytona sandbox requires the 'sandbox-daytona' extra. "
            "Install it with: pip install 'benchflow[sandbox-daytona]'"
        ) from e

    AsyncDaytona = _daytona.AsyncDaytona
    AsyncSandbox = _daytona.AsyncSandbox
    CreateSandboxFromImageParams = _daytona.CreateSandboxFromImageParams
    CreateSandboxFromSnapshotParams = _daytona.CreateSandboxFromSnapshotParams
    DaytonaNotFoundError = _daytona.DaytonaNotFoundError
    FileDownloadRequest = _daytona.FileDownloadRequest
    FileUpload = _daytona.FileUpload
    Image = _daytona.Image
    Resources = _daytona.Resources
    SessionExecuteRequest = _daytona.SessionExecuteRequest
    SnapshotState = _snapshot.SnapshotState
    _DAYTONA_SDK_LOADED = True


def build_sync_client(api_key: str | None = None) -> Any:
    """Return a synchronous ``daytona.Daytona`` client with anyio compat applied.

    Canonical entry point for the *sync* SDK client (the async strategy classes
    use :func:`_load_daytona_sdk`). Applies :func:`_ensure_daytona_anyio_compat`
    first — the SDK's sync client imports ``anyio.AsyncContextManagerMixin`` at
    import time, which the pinned anyio may not expose — then builds the client
    with an explicit key (no ``os.environ`` mutation) when one is given,
    otherwise letting the SDK read ``DAYTONA_API_KEY`` itself.
    """
    _ensure_daytona_anyio_compat()
    from daytona import Daytona

    if not api_key:
        return Daytona()
    from daytona import DaytonaConfig

    return Daytona(DaytonaConfig(api_key=api_key))


logger = logging.getLogger("benchflow")

# ``_SandboxParams`` was previously a top-level union of two SDK types. The
# SDK types are now loaded lazily (issue #358), so concrete sandbox params are
# typed as ``Any`` here — callers build them inside methods that have already
# called ``_load_daytona_sdk()``.
_SandboxParams = Any
_DAYTONA_COMMAND_POLL_INTERVAL_SEC = 1.0
_STARTUP_HARD_TIMEOUT_BUFFER_SEC = 120

# Safety-net ceiling for ``_poll_response`` when the caller passes no
# ``timeout_sec``. A Daytona *session* command only reports its ``exit_code``
# once it completes, and Daytona treats the command as still-running while any
# child holds the session's stdout/stderr stream open. So a backgrounded daemon
# launched without redirecting its std fds (``mysvc &`` with no
# ``</dev/null >log 2>&1``) keeps that stream open, ``exit_code`` never arrives,
# and an unbounded poll loop would wedge ``exec`` forever (BF-6). This cap is
# deliberately sized *well above* any legitimately long-running command (an
# agent rollout can run for many minutes) so it never trips for real work — it
# exists purely so a never-completing session command cannot spin the poll loop
# indefinitely. It applies ONLY when no positive ``timeout_sec`` is supplied
# (``None``, or a non-positive value — both previously meant "no deadline");
# an explicit positive ``timeout_sec`` is honored byte-for-byte as before.
_DAYTONA_EXEC_HARD_CAP_SEC = 3600

# Shared tenacity policy for the idempotent Daytona SDK calls — session-command
# polling and filesystem up/download. Three attempts with exponential backoff,
# re-raising the final failure. ``_create_sandbox`` and ``_stop_sandbox`` keep
# their own policies (different attempt counts and backoff bounds), so they are
# intentionally not folded in here. Reusing one ``retry(...)`` decorator across
# methods is safe: tenacity builds a fresh controller per decorated function.
_SDK_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


_REAP_DEFAULT_MAX_AGE_MIN = 1440
_REAP_FAILED_MAX_AGE_MIN = 120
_REAP_FAILED_STATE_MARKERS = ("FAILED", "ERROR")

# Ownership scoping for the auto-reaper. Every sandbox benchflow creates is
# stamped with this label so :func:`reap_stale_sandboxes` can restrict its
# age-based deletion to benchflow's *own* sandboxes. Without it, a reap run
# against a ``DAYTONA_API_KEY`` shared across an org (or with other tools) would
# delete unrelated sandboxes by age alone — irreversible, cross-tenant data
# loss. Foreign / unlabeled sandboxes are therefore never touched, regardless
# of age. The dotted key follows the Docker/Daytona label convention.
_BENCHFLOW_MANAGED_LABEL = "benchflow.managed"
_BENCHFLOW_MANAGED_VALUE = "1"
# The dotted prefix every benchflow-stamped label key shares. Used only by the
# read-only orphan-leak guard below to recognize a sandbox that *looks* like
# benchflow created it (carries the namespace) yet fails the strict ownership
# check — i.e. its ownership label drifted off the exact key/value the reaper
# keys on. ``_BENCHFLOW_MANAGED_LABEL`` itself lives under this namespace.
_BENCHFLOW_LABEL_NAMESPACE = "benchflow."


def _benchflow_owned_labels() -> dict[str, str]:
    """Return a fresh ownership-label dict for one sandbox-creation call.

    A new dict per call is required: the Daytona SDK mutates ``params.labels``
    in place (it injects the language label), so a shared dict would leak that
    mutation across creation sites.
    """
    return {_BENCHFLOW_MANAGED_LABEL: _BENCHFLOW_MANAGED_VALUE}


def _is_benchflow_owned(sb: Any) -> bool:
    """Return whether *sb* carries benchflow's exact ownership label.

    This scope check is the only thing standing between the age-based reaper
    and other people's sandboxes when the API key is shared. Anything missing
    the exact key/value pair — including sandboxes with no labels at all, or a
    ``labels`` attribute that is not a mapping — is treated as foreign and left
    untouched.
    """
    labels = getattr(sb, "labels", None)
    if not isinstance(labels, dict):
        return False
    return labels.get(_BENCHFLOW_MANAGED_LABEL) == _BENCHFLOW_MANAGED_VALUE


def _is_benchflow_label_orphan(sb: Any) -> bool:
    """Return whether *sb* looks benchflow-created but lacks the ownership label.

    True only when the sandbox carries at least one ``benchflow.``-namespaced
    label key yet fails :func:`_is_benchflow_owned` (the exact
    ``benchflow.managed=1`` pair is absent or has drifted to another value).
    Such a sandbox is almost certainly one benchflow created whose ownership
    label was lost — the age-based reaper's scope gate will now skip it forever,
    so it leaks. This is a *detection-only* predicate: the missing/altered label
    means ownership cannot be proven strongly enough to delete on a shared API
    key, so the reaper only warns. A correctly-labeled (owned) sandbox is never
    an orphan, and a purely foreign sandbox (no benchflow namespace) is ignored.
    """
    if _is_benchflow_owned(sb):
        return False
    labels = getattr(sb, "labels", None)
    if not isinstance(labels, dict):
        return False
    return any(
        isinstance(key, str) and key.startswith(_BENCHFLOW_LABEL_NAMESPACE)
        for key in labels
    )


def reap_stale_sandboxes(
    client: Any | None = None,
    *,
    max_age_minutes: int = _REAP_DEFAULT_MAX_AGE_MIN,
    failed_max_age_minutes: int = _REAP_FAILED_MAX_AGE_MIN,
    dry_run: bool = False,
    on_decision: Any | None = None,
) -> dict[str, int]:
    """Delete orphaned Daytona sandboxes past their TTL.

    Ownership-scoped: only sandboxes benchflow created — those carrying the
    ``benchflow.managed`` label (see :func:`_is_benchflow_owned`) — are ever
    considered. Foreign / unlabeled sandboxes are skipped before any age check,
    so a ``DAYTONA_API_KEY`` shared across an org or with other tools cannot be
    used to destroy unrelated sandboxes by age alone.

    Two tiers: sandboxes whose state contains a failure marker (e.g.
    ``BUILD_FAILED``) are reaped after *failed_max_age_minutes*; everything
    else after *max_age_minutes*. Defaults are deliberately conservative so
    concurrent live runs are never touched — only multi-hour orphans from
    crashed or interrupted sessions.

    *on_decision* (sandbox, age_minutes, will_delete) is called per *owned*
    sandbox when provided — the CLI uses it for per-row display; foreign
    sandboxes are never surfaced as reap candidates. Returns counts:
    ``{"found", "deleted", "skipped", "failed"}`` (``found`` counts every
    sandbox listed; foreign ones fall into ``skipped``).
    """
    from datetime import UTC, datetime

    if client is None:
        client = build_sync_client()
    now = datetime.now(UTC)
    counts = {"found": 0, "deleted": 0, "skipped": 0, "failed": 0}
    for sb in client.list():
        counts["found"] += 1
        if not _is_benchflow_owned(sb):
            # Scope guard: never touch a sandbox benchflow did not create.
            # This is the load-bearing safety check on a shared API key.
            if _is_benchflow_label_orphan(sb):
                # Read-only orphan-leak guard: a sandbox carrying the benchflow
                # namespace but missing the exact ownership label will never be
                # reaped by age. Surface it so an operator can reclaim it by
                # hand; we deliberately do not delete unlabeled sandboxes.
                logger.warning(
                    "Daytona sandbox %s carries a benchflow label namespace but "
                    "is missing the %s=%s ownership label; the age-based reaper "
                    "will never reclaim it (possible orphan leak). Not deleting — "
                    "verify and remove it manually if it is stale.",
                    getattr(sb, "id", "?"),
                    _BENCHFLOW_MANAGED_LABEL,
                    _BENCHFLOW_MANAGED_VALUE,
                )
            counts["skipped"] += 1
            continue
        if not getattr(sb, "created_at", None):
            counts["skipped"] += 1
            continue
        created_at = datetime.fromisoformat(sb.created_at.replace("Z", "+00:00"))
        age_minutes = (now - created_at).total_seconds() / 60
        state = str(getattr(sb, "state", "") or "").upper()
        is_failed = any(marker in state for marker in _REAP_FAILED_STATE_MARKERS)
        ttl = failed_max_age_minutes if is_failed else max_age_minutes
        will_delete = age_minutes >= ttl
        if on_decision is not None:
            on_decision(sb, age_minutes, will_delete)
        if not will_delete:
            counts["skipped"] += 1
            continue
        if dry_run:
            counts["deleted"] += 1
            continue
        try:
            client.delete(sb)
            counts["deleted"] += 1
        except Exception:
            logger.warning("Failed to delete sandbox %s", getattr(sb, "id", "?"))
            counts["failed"] += 1
    return counts


# Prefix for the decoded env file inside the Daytona sandbox. A unique 16-hex
# suffix is appended by the shared wrapper so concurrent exec() calls can't
# clobber each other's env file.
_DAYTONA_ENV_FILE_PREFIX = "/tmp/.benchflow_daytona_env_"


def _wrap_daytona_command_with_env_file(env: dict[str, str], command: str) -> str:
    """Return *command* prefixed to materialize *env* from a file.

    Thin wrapper over the canonical
    :func:`benchflow.sandbox._base.wrap_command_with_env_file` so the
    secret-redaction logic lives in exactly one place (shared with the Docker
    backend). See that function for the full contract: secrets never reach the
    remote process argv (visible via ``ps``, Daytona audit logs, or any
    provider-side command logging) — they are base64-encoded into the command
    string, decoded to a mode-0600 file inside the sandbox, sourced, and
    unconditionally removed via ``trap ... EXIT``.

    Issue #412: previously this used ``env K=V ...`` argv, which placed raw
    secret values into the remote command line.
    """
    return wrap_command_with_env_file(
        env, command, env_path_prefix=_DAYTONA_ENV_FILE_PREFIX
    )


def _exec_failure_output(result: ExecResult) -> str:
    output = " ".join(
        text.strip()
        for text in (result.stdout or "", result.stderr or "")
        if text and text.strip()
    )
    return output[:4000]


def _reject_non_main_service(service: str) -> None:
    """Raise ``ValueError`` for a non-``main`` service on the direct strategy.

    The direct (single-container) Daytona sandbox cannot target additional
    compose services; multi-container (vulhub-style) tasks require a
    ``docker-compose.yaml`` (#248). Centralizes the identical guard that
    ``_DaytonaDirect.exec``/``upload_dir``/``download_dir`` each raised inline.
    """
    if service != "main":
        raise ValueError(
            f"Direct (non-compose) Daytona sandbox is single-container "
            f"and cannot target service {service!r}. Multi-container "
            "(vulhub-style) tasks require a docker-compose.yaml (#248)."
        )


def _daytona_preflight() -> None:
    if not os.environ.get("DAYTONA_API_KEY"):
        raise SystemExit(
            "Daytona requires DAYTONA_API_KEY to be set. "
            "Please set this environment variable and try again."
        )


class DaytonaClientManager:
    """Singleton manager for the AsyncDaytona client."""

    _instance: DaytonaClientManager | None = None
    _lock = asyncio.Lock()

    def __init__(self) -> None:
        self._client: AsyncDaytona | None = None
        self._client_lock = asyncio.Lock()
        self._logger = logger.getChild("DaytonaClientManager")
        self._cleanup_registered = False

    @classmethod
    async def get_instance(cls) -> DaytonaClientManager:
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        assert cls._instance is not None
        return cls._instance

    async def get_client(self) -> AsyncDaytona:
        async with self._client_lock:
            if self._client is None:
                self._logger.debug("Creating new AsyncDaytona client")
                self._client = AsyncDaytona()
                if not self._cleanup_registered:
                    atexit.register(self._cleanup_sync)
                    self._cleanup_registered = True
            return self._client

    def _cleanup_sync(self) -> None:
        try:
            asyncio.run(self._cleanup())
        except Exception as e:
            print(f"Error during Daytona client cleanup: {e}")

    async def _cleanup(self) -> None:
        async with self._client_lock:
            if self._client is not None:
                try:
                    self._logger.debug("Closing AsyncDaytona client at program exit")
                    await self._client.close()
                except Exception as e:
                    self._logger.error(f"Error closing AsyncDaytona client: {e}")
                finally:
                    self._client = None


# Strategy pattern


class _DaytonaStrategy:
    """Base for Daytona implementation strategies."""

    # Strategies declare whether they can satisfy container-level snapshots;
    # the DaytonaSandbox wrapper forwards the question to its strategy so the
    # capability tracks the active mode (direct vs. DinD/compose) — see #384.
    supports_snapshot: bool = False

    def __init__(self, env: DaytonaSandbox) -> None:
        self._env = env

    @abstractmethod
    async def start(self, force_build: bool) -> None: ...

    @abstractmethod
    async def stop(self, delete: bool) -> None: ...

    @abstractmethod
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
    ) -> ExecResult: ...

    @abstractmethod
    async def upload_file(self, source_path: Path | str, target_path: str) -> None: ...

    @abstractmethod
    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None: ...

    @abstractmethod
    async def download_file(
        self, source_path: str, target_path: Path | str
    ) -> None: ...

    @abstractmethod
    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None: ...

    @abstractmethod
    async def is_dir(self, path: str, user: str | int | None = None) -> bool: ...

    @abstractmethod
    async def is_file(self, path: str, user: str | int | None = None) -> bool: ...

    @abstractmethod
    async def services(self) -> list[str]: ...

    @abstractmethod
    async def attach(self) -> None: ...

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        """Capture a provider-level snapshot. Default: not supported."""
        raise SandboxSnapshotNotSupported(
            f"{type(self).__name__} does not support container-level snapshots."
        )

    async def restore(self, image: SandboxImage) -> None:
        """Restore from a provider-level snapshot. Default: not supported."""
        raise SandboxSnapshotNotSupported(
            f"{type(self).__name__} does not support container-level restore."
        )


class _DaytonaDirect(_DaytonaStrategy):
    """Direct sandbox strategy — single-container behavior."""

    # Daytona ships a native sandbox-snapshot API on AsyncSandbox; the direct
    # strategy uses it for the container layer of Branch (#384).
    supports_snapshot: bool = True

    async def start(self, force_build: bool) -> None:
        env = self._env
        resources = Resources(
            cpu=env.task_env_config.cpus,
            memory=env.task_env_config.memory_mb // 1024,
            disk=env.task_env_config.storage_mb // 1024,
        )

        env._client_manager = await DaytonaClientManager.get_instance()
        daytona = await env._client_manager.get_client()

        snapshot_name: str | None = None
        snapshot_exists = False

        if env._snapshot_template_name:
            snapshot_name = env._snapshot_template_name.format(
                name=env.environment_name
            )
            try:
                snapshot = await daytona.snapshot.get(snapshot_name)
                if snapshot.state == SnapshotState.ACTIVE:
                    snapshot_exists = True
            except Exception:
                snapshot_exists = False

        if snapshot_exists and force_build:
            env.logger.warning(
                "Snapshot template specified but force_build is True. "
                "Snapshot will be used instead of building from scratch."
            )

        params: _SandboxParams

        if snapshot_exists and snapshot_name:
            env.logger.debug(f"Using snapshot: {snapshot_name}")
            params = CreateSandboxFromSnapshotParams(
                auto_delete_interval=env._auto_delete_interval,
                auto_stop_interval=env._auto_stop_interval,
                snapshot=snapshot_name,
                network_block_all=env._network_block_all,
                labels=_benchflow_owned_labels(),
            )
        else:
            # Both non-snapshot paths build identical CreateSandboxFromImageParams
            # (including the benchflow.managed ownership label); only the image
            # source and log line differ between build-from-Dockerfile and
            # prebuilt-image.
            if force_build or not env.task_env_config.docker_image:
                env.logger.debug(f"Building environment from {env._dockerfile_path}")
                image = Image.from_dockerfile(env._dockerfile_path)
            else:
                env.logger.debug(
                    f"Using prebuilt image: {env.task_env_config.docker_image}"
                )
                image = Image.base(env.task_env_config.docker_image)
            params = CreateSandboxFromImageParams(
                image=image,
                auto_delete_interval=env._auto_delete_interval,
                auto_stop_interval=env._auto_stop_interval,
                resources=resources,
                network_block_all=env._network_block_all,
                labels=_benchflow_owned_labels(),
            )

        try:
            await env._create_sandbox(params=params)
        except (TimeoutError, RuntimeError, Exception) as e:
            sandbox_id = getattr(env._sandbox, "id", None) if env._sandbox else None
            raise SandboxStartupError(
                f"Sandbox creation failed after retries: {e}",
                sandbox_id=sandbox_id,
                sandbox_state="error",
                attempts=3,
                build_timeout_sec=env.task_env_config.build_timeout_sec,
            ) from e

        await env._sandbox_exec(
            f"mkdir -p {SandboxPaths.agent_dir} {SandboxPaths.verifier_dir} && "
            f"chmod 777 {SandboxPaths.agent_dir} {SandboxPaths.verifier_dir}"
        )

    async def stop(self, delete: bool) -> None:
        env = self._env
        if not delete:
            env.logger.info(
                "Daytona sandboxes are ephemeral and will be deleted after use, "
                "regardless of delete=False."
            )

        try:
            if not env._sandbox:
                env.logger.warning(
                    "Sandbox not found. Please build the environment first."
                )
            else:
                try:
                    await env._stop_sandbox()
                except Exception as e:
                    env.logger.error(f"Error stopping sandbox {env._sandbox.id}: {e}")
                finally:
                    env._sandbox = None
        finally:
            env._client_manager = None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
    ) -> ExecResult:
        _reject_non_main_service(service)
        return await self._env._sandbox_exec(
            command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._env._sdk_upload_file(source_path, target_path)

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        _reject_non_main_service(service)
        prep_result = await self._env._sandbox_exec(
            f"mkdir -p {shlex.quote(target_dir)}",
            timeout_sec=30,
            user="root",
        )
        if prep_result.return_code != 0:
            raise RuntimeError(
                "Daytona direct upload_dir destination prep failed: "
                f"{_exec_failure_output(prep_result)}"
            )
        await self._env._sdk_upload_dir(source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._env._sdk_download_file(source_path, target_path)

    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        _reject_non_main_service(service)
        await self._env._sdk_download_dir(source_dir, target_dir)

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        sandbox = self._env._require_sandbox()
        file_info = await sandbox.fs.get_file_info(path)
        return file_info.is_dir

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        sandbox = self._env._require_sandbox()
        file_info = await sandbox.fs.get_file_info(path)
        return not file_info.is_dir

    async def services(self) -> list[str]:
        raise NotImplementedError(
            "Direct (non-compose) Daytona sandbox is single-container and has "
            "no compose topology. services() requires a multi-container "
            "docker-compose task (#248)."
        )

    async def attach(self) -> None:
        env = self._env
        if not env._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        ssh_access = await env._sandbox.create_ssh_access()
        os.execvp(
            "ssh",
            ["ssh", f"{ssh_access.token}@ssh.app.daytona.io"],
        )

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        """Create a Daytona snapshot from the current sandbox state.

        Wraps ``AsyncSandbox._experimental_create_snapshot``; the snapshot
        name is the ``ref`` other Daytona sandboxes can be created from via
        :class:`CreateSandboxFromSnapshotParams`.
        """
        env = self._env
        if not env._sandbox:
            raise SandboxSnapshotNotSupported(
                "DaytonaSandbox.snapshot requires a started sandbox; call "
                "start() before snapshot()."
            )
        snap_name = name or f"bf-snap-{env.environment_name}-{uuid4().hex[:12]}"
        # Daytona names: lowercase, dash-separated, ascii — sanitize defensively.
        snap_name = snap_name.lower().replace("_", "-")
        await env._sandbox._experimental_create_snapshot(snap_name)
        env.logger.info(f"Snapshot created: {snap_name}")
        return SandboxImage(
            provider="daytona",
            ref=snap_name,
            meta={"sandbox_id": getattr(env._sandbox, "id", "") or ""},
        )

    async def restore(self, image: SandboxImage) -> None:
        """Replace the current Daytona sandbox with one from ``image``.

        Daytona snapshots are immutable — restore is implemented by deleting
        the current sandbox and creating a fresh one from the snapshot,
        matching the provider's native semantics.
        """
        env = self._env
        if image.provider != "daytona":
            raise SandboxSnapshotNotSupported(
                f"DaytonaSandbox.restore cannot consume a {image.provider!r} "
                f"snapshot (got ref={image.ref!r}); snapshots are not portable "
                "across providers."
            )
        if env._sandbox is not None:
            try:
                await env._sandbox.delete()
            except Exception as e:
                env.logger.warning(f"Failed to delete sandbox before restore: {e}")
            env._sandbox = None

        params = CreateSandboxFromSnapshotParams(
            auto_delete_interval=env._auto_delete_interval,
            auto_stop_interval=env._auto_stop_interval,
            snapshot=image.ref,
            network_block_all=env._network_block_all,
            labels=_benchflow_owned_labels(),
        )
        await env._create_sandbox(params=params)
        env.logger.info(f"Snapshot restored: {image.ref}")


class _DaytonaDinD(_DaytonaStrategy):
    """Docker-in-Docker compose strategy for multi-container tasks.

    Topology:
        Local machine (benchflow CLI)
          +-- Daytona Sandbox (DinD VM, docker:28.3.3-dind)
                +-- dockerd (Docker daemon)
                +-- docker compose
                      +-- main        <- agent runs here
                      +-- mcp-server  <- sidecar services
                      +-- ...
    """

    _DOCKER_DAEMON_TIMEOUT_SEC = 60
    _COMPOSE_DIR = "/benchflow/compose"
    _ENVIRONMENT_DIR = "/benchflow/environment"
    _LOGS_DIR = "/benchflow/logs"

    def __init__(self, env: DaytonaSandbox) -> None:
        super().__init__(env)
        self._use_prebuilt = False

        self._resolved_task_env: dict[str, str] = {}
        benchflow_keys = set(self._compose_env_vars().keys()) - set(
            self._env._persistent_env.keys()
        )
        if self._env.task_env_config.env:
            self._resolved_task_env = resolve_env_vars(self._env.task_env_config.env)

        resolved_task_keys = set(self._resolved_task_env.keys()) | set(
            self._env._persistent_env.keys()
        )
        if resolved_task_keys:
            collisions = benchflow_keys & resolved_task_keys
            if collisions:
                self._env.logger.warning(
                    "Environment vars override BenchFlow compose variable(s): %s",
                    ", ".join(sorted(collisions)),
                )

    async def _vm_exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        """Run a command on the DinD sandbox VM using sh (Alpine-compatible)."""
        return await self._env._sandbox_exec(
            command, cwd=cwd, env=env, timeout_sec=timeout_sec, shell="sh -c"
        )

    def _compose_env_vars(self) -> dict[str, str]:
        env_vars: dict[str, str] = {
            "CONTEXT_DIR": self._ENVIRONMENT_DIR,
            "MAIN_IMAGE_NAME": f"bf__{self._env.environment_name}",
            "HOST_VERIFIER_LOGS_PATH": f"{self._LOGS_DIR}/verifier",
            "HOST_AGENT_LOGS_PATH": f"{self._LOGS_DIR}/agent",
            "HOST_ARTIFACTS_PATH": f"{self._LOGS_DIR}/artifacts",
            "ENV_VERIFIER_LOGS_PATH": str(SandboxPaths.verifier_dir),
            "ENV_AGENT_LOGS_PATH": str(SandboxPaths.agent_dir),
            "ENV_ARTIFACTS_PATH": str(SandboxPaths.artifacts_dir),
            "CPUS": str(self._env.task_env_config.cpus),
            "MEMORY": f"{self._env.task_env_config.memory_mb}M",
        }
        if self._use_prebuilt and self._env.task_env_config.docker_image:
            env_vars["PREBUILT_IMAGE_NAME"] = self._env.task_env_config.docker_image
        if self._resolved_task_env:
            env_vars.update(self._resolved_task_env)
        if self._env._persistent_env:
            env_vars.update(self._env._persistent_env)
        return env_vars

    def _compose_file_flags(self) -> list[str]:
        build_or_prebuilt = (
            "docker-compose-prebuilt.yaml"
            if self._use_prebuilt
            else "docker-compose-build.yaml"
        )
        files = [
            f"{self._COMPOSE_DIR}/docker-compose-base.yaml",
            f"{self._COMPOSE_DIR}/{build_or_prebuilt}",
            f"{self._ENVIRONMENT_DIR}/docker-compose.yaml",
        ]
        if not self._env.task_env_config.allow_internet:
            files.append(f"{self._COMPOSE_DIR}/docker-compose-no-network.yaml")

        flags: list[str] = []
        for f in files:
            flags.extend(["-f", f])
        return flags

    @property
    def _project_name(self) -> str:
        return self._env.session_id.lower().replace(".", "-")

    def _compose_cmd(self, subcommand: list[str]) -> str:
        parts = [
            "docker",
            "compose",
            "-p",
            self._project_name,
            "--project-directory",
            self._ENVIRONMENT_DIR,
            *self._compose_file_flags(),
            *subcommand,
        ]
        return shlex.join(parts)

    async def _compose_exec(
        self,
        subcommand: list[str],
        timeout_sec: int | None = None,
    ) -> ExecResult:
        return await self._vm_exec(
            self._compose_cmd(subcommand),
            env=self._compose_env_vars(),
            timeout_sec=timeout_sec,
        )

    async def _wait_for_docker_daemon(self) -> None:
        self._env.logger.debug("Waiting for Docker daemon inside DinD sandbox...")
        last_output = ""
        for _ in range(self._DOCKER_DAEMON_TIMEOUT_SEC // 2):
            result = await self._vm_exec("docker info", timeout_sec=10)
            if result.return_code == 0:
                self._env.logger.debug("Docker daemon is ready")
                return
            last_output = (result.stdout or "") + (result.stderr or "")
            await asyncio.sleep(2)
        raise RuntimeError(
            f"Docker daemon not ready after {self._DOCKER_DAEMON_TIMEOUT_SEC}s. "
            f"Last output: {last_output}"
        )

    async def _wait_for_main_container(self, timeout_sec: int = 60) -> None:
        self._env.logger.debug("Waiting for main container to be running...")
        for _ in range(timeout_sec // 2):
            result = await self._compose_exec(
                ["exec", "-T", "main", "true"], timeout_sec=10
            )
            if result.return_code == 0:
                self._env.logger.debug("Main container is running")
                return
            await asyncio.sleep(2)
        raise RuntimeError(f"Main container not running after {timeout_sec}s")

    async def start(self, force_build: bool) -> None:
        env = self._env

        resources = Resources(
            cpu=env.task_env_config.cpus,
            memory=env.task_env_config.memory_mb // 1024,
            disk=env.task_env_config.storage_mb // 1024,
        )

        env._client_manager = await DaytonaClientManager.get_instance()

        dind_image: str = env._kwargs.get("dind_image", "docker:28.3.3-dind")
        dind_snapshot: str | None = env._kwargs.get("dind_snapshot")

        params: _SandboxParams
        if dind_snapshot:
            params = CreateSandboxFromSnapshotParams(
                snapshot=dind_snapshot,
                auto_delete_interval=env._auto_delete_interval,
                auto_stop_interval=env._auto_stop_interval,
                network_block_all=False,
                labels=_benchflow_owned_labels(),
            )
        else:
            image = Image.base(dind_image)
            params = CreateSandboxFromImageParams(
                image=image,
                auto_delete_interval=env._auto_delete_interval,
                auto_stop_interval=env._auto_stop_interval,
                resources=resources,
                network_block_all=False,
                labels=_benchflow_owned_labels(),
            )

        try:
            await env._create_sandbox(params=params)
        except (TimeoutError, RuntimeError, Exception) as e:
            sandbox_id = getattr(env._sandbox, "id", None) if env._sandbox else None
            raise SandboxStartupError(
                f"Sandbox creation failed after retries: {e}",
                sandbox_id=sandbox_id,
                sandbox_state="error",
                attempts=3,
                build_timeout_sec=env.task_env_config.build_timeout_sec,
            ) from e

        env.logger.debug("Starting Docker daemon inside DinD sandbox...")
        await self._vm_exec(
            "dockerd-entrypoint.sh dockerd > /var/log/dockerd.log 2>&1 &",
            timeout_sec=10,
        )

        await self._wait_for_docker_daemon()

        # Upload BenchFlow compose files to the sandbox
        for path in (
            COMPOSE_BASE_PATH,
            COMPOSE_BUILD_PATH,
            COMPOSE_PREBUILT_PATH,
            COMPOSE_NO_NETWORK_PATH,
        ):
            await env._sdk_upload_file(path, f"{self._COMPOSE_DIR}/{path.name}")

        # Upload task environment directory
        await env._sdk_upload_dir(env.environment_dir, self._ENVIRONMENT_DIR)

        # Create log directories
        await self._vm_exec(
            f"mkdir -p {self._LOGS_DIR}/verifier {self._LOGS_DIR}/agent "
            f"{self._LOGS_DIR}/artifacts && "
            f"chmod 777 {self._LOGS_DIR}/verifier {self._LOGS_DIR}/agent "
            f"{self._LOGS_DIR}/artifacts"
        )

        # Build and start compose services
        self._use_prebuilt = not force_build and bool(env.task_env_config.docker_image)

        env.logger.debug("Building compose services inside DinD sandbox...")
        result = await self._compose_exec(
            ["build"],
            timeout_sec=round(env.task_env_config.build_timeout_sec),
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"docker compose build failed: {result.stdout} {result.stderr}"
            )

        env.logger.debug("Starting compose services inside DinD sandbox...")
        result = await self._compose_exec(["up", "-d"], timeout_sec=120)
        if result.return_code != 0:
            raise RuntimeError(
                f"docker compose up failed: {result.stdout} {result.stderr}"
            )

        await self._wait_for_main_container()

    async def stop(self, delete: bool) -> None:
        env = self._env
        if not delete:
            env.logger.info(
                "Daytona sandboxes are ephemeral and will be deleted after use, "
                "regardless of delete=False."
            )

        if env._sandbox:
            try:
                await self._compose_exec(["down", "--remove-orphans"], timeout_sec=30)
            except Exception as e:
                env.logger.warning(f"docker compose down failed: {e}")

        try:
            if not env._sandbox:
                env.logger.warning(
                    "Sandbox not found. Please build the environment first."
                )
            else:
                try:
                    await env._stop_sandbox()
                except Exception as e:
                    env.logger.error(f"Error stopping sandbox {env._sandbox.id}: {e}")
                finally:
                    env._sandbox = None
        finally:
            env._client_manager = None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
    ) -> ExecResult:
        """Run a command in a compose service inside the DinD VM.

        ``service`` defaults to ``"main"`` (the agent container); pass
        another name to target an additional container declared in the
        task's ``docker-compose.yaml`` (vulhub-style targets — see #248).
        """
        parts: list[str] = ["exec", "-T"]
        if cwd:
            parts.extend(["-w", cwd])
        if user is not None:
            parts.extend(["-u", str(user)])
        # Env vars are materialized inside the target container via a
        # mode-0600 file and sourced, rather than passed as ``-e KEY=VALUE``
        # flags. The flags would land in the DinD VM's process list (visible
        # to ``ps`` and any Daytona session audit log) and leak verifier
        # API keys / agent secrets on every multi-service task (#412
        # follow-up). Matches the wrapping that ``DaytonaSandbox._sandbox_exec``
        # already applies to the outer VM-side command.
        if env:
            command = _wrap_daytona_command_with_env_file(env, command)
        # Use POSIX ``sh`` rather than ``bash``: with multi-service support
        # (#248), ``service`` can target arbitrary task containers, and
        # Alpine/distroless/minimal images often ship no ``/bin/bash``. The
        # wrapped command uses only POSIX constructs (``trap``, ``base64 -d``,
        # ``set -a``/``. file``).
        parts.extend([service, "sh", "-c", command])

        return await self._compose_exec(parts, timeout_sec=timeout_sec)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        temp = f"/tmp/benchflow_{uuid4().hex}"
        try:
            await self._env._sdk_upload_file(source_path, temp)
            target_parent_cmd = compose_parent_mkdir_p_command(target_path)
            if target_parent_cmd is not None:
                prep_result = await self.exec(
                    target_parent_cmd,
                    timeout_sec=30,
                    user="root",
                    service="main",
                )
                if prep_result.return_code != 0:
                    raise RuntimeError(
                        "docker compose upload_file destination prep failed: "
                        f"{_exec_failure_output(prep_result)}"
                    )
            result = await self._compose_exec(
                ["cp", temp, compose_cp_destination("main", target_path)],
                timeout_sec=60,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose cp failed: {result.stdout} {result.stderr}"
                )
        finally:
            await self._vm_exec(f"rm -f {shlex.quote(temp)}", timeout_sec=10)

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        """Upload a directory into a compose service inside the DinD VM.

        ``service`` defaults to ``"main"``; pass a target service to land the
        directory in an additional vulhub-style container (#248).
        """
        temp = f"/tmp/benchflow_{uuid4().hex}"
        try:
            await self._env._sdk_upload_dir(source_dir, temp)
            prep_result = await self.exec(
                compose_mkdir_p_command(target_dir),
                timeout_sec=30,
                user="root",
                service=service,
            )
            if prep_result.return_code != 0:
                raise RuntimeError(
                    "docker compose upload_dir destination prep failed: "
                    f"{_exec_failure_output(prep_result)}"
                )
            result = await self._compose_exec(
                ["cp", f"{temp}/.", compose_cp_destination(service, target_dir)],
                timeout_sec=120,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose cp failed: {result.stdout} {result.stderr}"
                )
        finally:
            await self._vm_exec(f"rm -rf {shlex.quote(temp)}", timeout_sec=10)

    def _sandbox_log_path(self, container_path: str) -> str | None:
        mappings = {
            str(SandboxPaths.verifier_dir): f"{self._LOGS_DIR}/verifier",
            str(SandboxPaths.agent_dir): f"{self._LOGS_DIR}/agent",
            str(SandboxPaths.artifacts_dir): f"{self._LOGS_DIR}/artifacts",
        }
        for env_prefix, sandbox_prefix in mappings.items():
            if container_path == env_prefix or container_path.startswith(
                env_prefix + "/"
            ):
                return container_path.replace(env_prefix, sandbox_prefix, 1)
        return None

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        sandbox_path = self._sandbox_log_path(source_path)
        if sandbox_path:
            try:
                await self._env._sdk_download_file(sandbox_path, target_path)
            except Exception as host_error:
                self._env.logger.warning(
                    "Daytona host log download_file failed for %s; falling back "
                    "to docker compose cp: %s",
                    source_path,
                    host_error,
                )
                try:
                    await self._compose_download_file(source_path, target_path)
                except Exception as compose_error:
                    raise RuntimeError(
                        "Daytona log download_file failed via host path and "
                        "compose fallback"
                    ) from compose_error
            return

        await self._compose_download_file(source_path, target_path)

    async def _compose_download_file(
        self, source_path: str, target_path: Path | str
    ) -> None:
        temp = f"/tmp/benchflow_{uuid4().hex}"
        try:
            result = await self._compose_exec(
                ["cp", f"main:{source_path}", temp], timeout_sec=60
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose cp failed: {result.stdout} {result.stderr}"
                )
            await self._env._sdk_download_file(temp, target_path)
        finally:
            await self._vm_exec(f"rm -f {shlex.quote(temp)}", timeout_sec=10)

    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        """Download a directory from a compose service inside the DinD VM.

        ``service`` defaults to ``"main"``; pass a target service to fetch
        target-side verifier output from a vulhub-style container (#248).
        The host-mounted-log fast path only applies to ``main`` — target
        services have no host bind mount, so their contents are always
        copied out via ``docker compose cp``.
        """
        if service == "main":
            sandbox_path = self._sandbox_log_path(source_dir)
            if sandbox_path:
                try:
                    await self._env._sdk_download_dir(sandbox_path, target_dir)
                except Exception as host_error:
                    self._env.logger.warning(
                        "Daytona host log download_dir failed for %s; falling "
                        "back to docker compose cp: %s",
                        source_dir,
                        host_error,
                    )
                    try:
                        await self._compose_download_dir(
                            source_dir, target_dir, service=service
                        )
                    except Exception as compose_error:
                        raise RuntimeError(
                            "Daytona log download_dir failed via host path and "
                            "compose fallback"
                        ) from compose_error
                return

        await self._compose_download_dir(source_dir, target_dir, service=service)

    async def _compose_download_dir(
        self, source_dir: str, target_dir: Path | str, service: str
    ) -> None:
        temp = f"/tmp/benchflow_{uuid4().hex}"
        try:
            await self._vm_exec(f"mkdir -p {shlex.quote(temp)}", timeout_sec=10)
            result = await self._compose_exec(
                ["cp", f"{service}:{source_dir}/.", temp], timeout_sec=120
            )
            if result.return_code != 0:
                self._env.logger.error(
                    f"download_dir: docker compose cp failed: {result.stdout} {result.stderr}"
                )
                raise RuntimeError(
                    f"download_dir: docker compose cp failed: {result.stdout} {result.stderr}"
                )
            await self._env._sdk_download_dir(temp, target_dir)
        finally:
            await self._vm_exec(f"rm -rf {shlex.quote(temp)}", timeout_sec=10)

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(
            f"test -d {shlex.quote(path)}", timeout_sec=10, user=user
        )
        return result.return_code == 0

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(
            f"test -f {shlex.quote(path)}", timeout_sec=10, user=user
        )
        return result.return_code == 0

    async def services(self) -> list[str]:
        """List compose service names defined inside the DinD VM.

        Runs ``docker compose config --services`` against the task's compose
        stack — includes BenchFlow's ``main`` service plus any vulhub-style
        target/database containers the task declares (#248). The output is
        filtered to the compose service-name grammar so stray warning lines
        cannot be mistaken for services.
        """
        result = await self._compose_exec(["config", "--services"], timeout_sec=30)
        if result.return_code != 0:
            raise RuntimeError(
                f"docker compose config --services failed: "
                f"{result.stdout} {result.stderr}"
            )
        return _filter_compose_service_names(result.stdout or "")

    async def attach(self) -> None:
        env = self._env
        if not env._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        ssh_access = await env._sandbox.create_ssh_access()

        compose_cmd = self._compose_cmd(["exec", "-it", "main", "bash"])
        compose_env = " ".join(
            f"{k}={shlex.quote(v)}" for k, v in self._compose_env_vars().items()
        )
        remote_cmd = f"{compose_env} {compose_cmd}"

        os.execvp(
            "ssh",
            [
                "ssh",
                "-t",
                f"{ssh_access.token}@ssh.app.daytona.io",
                remote_cmd,
            ],
        )


class DaytonaSandbox(BaseSandbox):
    @classmethod
    def preflight(cls) -> None:
        _daytona_preflight()

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        rollout_paths: RolloutPaths | None,
        task_env_config: SandboxConfig,
        snapshot_template_name: str | None = None,
        network_block_all: bool | None = None,
        auto_stop_interval_mins: int = 0,
        auto_delete_interval_mins: int = 0,
        **kwargs: object,
    ) -> None:
        # Materialize the optional Daytona SDK on first DaytonaSandbox
        # instantiation. Importing this module is now free of the SDK
        # dependency (issue #358); the SDK is required only at construction
        # time of a real Daytona sandbox.
        _load_daytona_sdk()
        # Detect compose mode before super().__init__ calls _validate_definition
        self._compose_mode = (environment_dir / "docker-compose.yaml").exists()
        self._kwargs = kwargs

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            rollout_paths=rollout_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

        self._auto_stop_interval = auto_stop_interval_mins
        self._auto_delete_interval = auto_delete_interval_mins
        self._snapshot_template_name = snapshot_template_name
        if network_block_all is not None:
            self._network_block_all = network_block_all
            expected = not task_env_config.allow_internet
            if network_block_all != expected:
                self.logger.warning(
                    f"network_block_all={network_block_all} overrides task config "
                    f"allow_internet={task_env_config.allow_internet}"
                )
        else:
            self._network_block_all = not task_env_config.allow_internet

        self._sandbox: AsyncSandbox | None = None  # pyright: ignore[reportInvalidTypeForm]
        self._client_manager: DaytonaClientManager | None = None

        self._strategy: _DaytonaStrategy = (
            _DaytonaDinD(self) if self._compose_mode else _DaytonaDirect(self)
        )
        self.logger.debug(f"Selected strategy: {self._strategy.__class__.__name__}")

    @property
    def _uses_compose(self) -> bool:
        return self._compose_mode

    @property
    def sandbox_id(self) -> str | None:
        return self._sandbox.id if self._sandbox else None

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @property
    def _environment_docker_compose_path(self) -> Path:
        return self.environment_dir / "docker-compose.yaml"

    def _validate_definition(self) -> None:
        if self._compose_mode:
            path = self._environment_docker_compose_path
        else:
            path = self._dockerfile_path
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Please ensure the file exists.")

    # Shared helpers used by both strategies

    def _require_sandbox(self) -> AsyncSandbox:  # pyright: ignore[reportInvalidTypeForm]
        """Return the started sandbox, raising if ``start()`` has not run.

        Centralizes the ``if not self._sandbox`` precondition repeated across the
        SDK helpers (and the direct strategy's ``is_dir``/``is_file``) so the type
        checker narrows ``AsyncSandbox | None`` to ``AsyncSandbox`` at each call
        site. Uses the same falsiness check and message as the inlined guards.
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please build the environment first.")
        return self._sandbox

    def _on_sandbox_created(self) -> None:
        """Persist sandbox.json the instant the Daytona sandbox id is known.

        Called from ``_create_sandbox`` the moment ``self._sandbox`` is assigned
        — before ``start()`` does its remaining (and, for DinD, long) work:
        launching dockerd, polling ``_wait_for_docker_daemon`` for tens of
        seconds, mkdir/chmod, compose build/up. If the run is interrupted
        (CancelledError/SIGINT/timeout) anywhere in that stretch, the Daytona
        sandbox already exists server-side; persisting here means there is still
        a ``sandbox.json`` to audit and clean up (#554/#563).

        Best-effort and self-contained: a missing ``rollout_paths`` (e.g. a
        snapshot/branch sandbox built outside a rollout dir) is a no-op, and the
        underlying write swallows-and-logs its own failures. The rollout-layer
        ``on_started`` callback still runs after ``start()`` returns as an
        idempotent fallback.
        """
        if self.rollout_paths is None:
            return
        persist_sandbox_info(self, self.rollout_paths.rollout_dir)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    async def _create_sandbox(
        self,
        params: _SandboxParams,
    ) -> None:
        if not self._client_manager:
            raise RuntimeError(
                "Client manager not initialized. This should never happen."
            )

        # Clean up any previous failed sandbox before retry
        if self._sandbox is not None:
            try:
                self.logger.warning("Cleaning up previous sandbox before retry")
                await self._sandbox.delete()
            except Exception as cleanup_err:
                self.logger.debug(f"Cleanup of previous sandbox failed: {cleanup_err}")
            finally:
                self._sandbox = None

        daytona = await self._client_manager.get_client()
        build_timeout = round(self.task_env_config.build_timeout_sec)
        hard_timeout = build_timeout + _STARTUP_HARD_TIMEOUT_BUFFER_SEC

        create_task = asyncio.ensure_future(
            daytona.create(
                params=params,
                timeout=build_timeout,
            )
        )
        try:
            self._sandbox = await asyncio.wait_for(
                asyncio.shield(create_task), timeout=hard_timeout
            )
            # Persist the id the moment the sandbox exists — before start()'s
            # remaining (DinD: long) work — so an interrupt mid-start() still
            # leaves a sandbox.json to audit/clean up (#554/#563).
            self._on_sandbox_created()
        except TimeoutError:
            self.logger.error(
                f"Sandbox creation timed out after {hard_timeout}s "
                f"(build_timeout={build_timeout}s + buffer={_STARTUP_HARD_TIMEOUT_BUFFER_SEC}s)"
            )
            create_task.cancel()
            raise
        except asyncio.CancelledError:
            try:
                self._sandbox = await asyncio.wait_for(create_task, timeout=30)
                # Sandbox came back even though we were cancelled — persist its
                # id before re-raising so it is not orphaned (#554/#563).
                self._on_sandbox_created()
            except (TimeoutError, asyncio.CancelledError, Exception):
                create_task.cancel()
            raise

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _stop_sandbox(self) -> None:
        if self._sandbox:
            await self._sandbox.delete()

    @_SDK_RETRY
    async def _get_session_command_with_retry(
        self, session_id: str, command_id: str
    ) -> object:
        sandbox = self._require_sandbox()
        return await sandbox.process.get_session_command(session_id, command_id)

    @_SDK_RETRY
    async def _get_session_command_logs_with_retry(
        self, session_id: str, command_id: str
    ) -> object:
        sandbox = self._require_sandbox()
        return await sandbox.process.get_session_command_logs(session_id, command_id)

    @staticmethod
    def _poll_timeout_error(
        timeout_sec: int | float | None, capped: bool
    ) -> RuntimeError:
        """Build the ``RuntimeError`` raised when a poll deadline is exceeded.

        The bounded path (explicit positive ``timeout_sec``) keeps the exact
        legacy message so its semantics are byte-for-byte unchanged. The
        safety-net path (no positive ``timeout_sec``, falling back to
        :data:`_DAYTONA_EXEC_HARD_CAP_SEC`) raises the *same* ``RuntimeError``
        type for consistent error handling, but with a message that points at
        the most likely cause — a backgrounded command still holding the Daytona
        session's stdout/stderr stream open.
        """
        if capped:
            return RuntimeError(
                f"Command timed out after {_DAYTONA_EXEC_HARD_CAP_SEC} seconds "
                "(no positive timeout_sec given; hit the Daytona exec "
                "safety-net cap). "
                "A backgrounded command is likely holding the session "
                "stdout/stderr stream open — redirect its std fds, e.g. "
                "`nohup CMD </dev/null >log 2>&1 &`."
            )
        return RuntimeError(f"Command timed out after {timeout_sec} seconds")

    async def _poll_response(
        self,
        session_id: str,
        command_id: str,
        timeout_sec: int | float | None = None,
    ) -> ExecResult:
        self._require_sandbox()

        loop = asyncio.get_running_loop()
        # When the caller passes an explicit, positive ``timeout_sec`` we honor
        # it exactly as before. When it is ``None`` (or non-positive) we fall
        # back to a generous safety-net ceiling so ``deadline`` is *never*
        # ``None`` — a Daytona session command whose ``exit_code`` never resolves
        # (e.g. a backgrounded child still holding the session stdout/stderr
        # stream open) can therefore never spin this loop forever (BF-6). The
        # cap is sized well above any real command, so normal behavior on the
        # unbounded path is unchanged for everything except the wedge case.
        if timeout_sec is not None and timeout_sec > 0:
            deadline = loop.time() + float(timeout_sec)
            capped = False
        else:
            deadline = loop.time() + float(_DAYTONA_EXEC_HARD_CAP_SEC)
            capped = True

        response = await self._get_session_command_with_retry(session_id, command_id)

        while response.exit_code is None:  # type: ignore[union-attr]
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise self._poll_timeout_error(timeout_sec, capped)
            await asyncio.sleep(min(_DAYTONA_COMMAND_POLL_INTERVAL_SEC, remaining))
            if loop.time() >= deadline:
                raise self._poll_timeout_error(timeout_sec, capped)
            response = await self._get_session_command_with_retry(
                session_id,
                response.id,  # type: ignore[union-attr]
            )

        logs = await self._get_session_command_logs_with_retry(session_id, command_id)

        return ExecResult(
            stdout=logs.stdout,  # type: ignore[union-attr]
            stderr=logs.stderr,  # type: ignore[union-attr]
            return_code=int(response.exit_code),  # type: ignore[union-attr]
        )

    async def _sandbox_exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        shell: str = "bash -c",
        user: str | int | None = None,
    ) -> ExecResult:
        """Run ``command`` as a Daytona session command and poll to completion.

        ``timeout_sec=None`` falls back to the
        :data:`_DAYTONA_EXEC_HARD_CAP_SEC` safety-net deadline in
        :meth:`_poll_response`, so a backgrounded command that never releases the
        session stdout/stderr stream cannot wedge ``exec`` forever (BF-6).
        Backgrounded daemons must redirect all std fds
        (``</dev/null >log 2>&1``) — see :meth:`exec`.
        """
        sandbox = self._require_sandbox()

        session_id = str(uuid4())
        await sandbox.process.create_session(session_id)

        # Env vars are written to a temp file inside the sandbox and
        # sourced rather than passed as ``env KEY=value ...`` argv. The
        # argv form would leak verifier API keys / agent secrets into the
        # remote process list and any provider-side command audit log
        # (#412). The wrapping must happen before the ``timeout``/``su``
        # prefixes so they too see the exported vars; the wrapper itself
        # runs under ``sh``-compatible POSIX constructs so the surrounding
        # ``bash -c`` / ``su -s /bin/bash -c`` shells handle it fine.
        if env:
            command = f"{shell} {shlex.quote(_wrap_daytona_command_with_env_file(env, command))}"
        else:
            command = f"{shell} {shlex.quote(command)}"

        if timeout_sec:
            command = f"timeout {timeout_sec} {command}"

        if cwd:
            command = f"cd {cwd} && {command}"

        if user is not None:
            if isinstance(user, int):
                user_arg = f"$(getent passwd {user} | cut -d: -f1)"
            else:
                user_arg = shlex.quote(user)
            command = f"su {user_arg} -s /bin/bash -c {shlex.quote(command)}"

        response = await sandbox.process.execute_session_command(
            session_id,
            SessionExecuteRequest(
                command=command,
                run_async=True,
            ),
            timeout=timeout_sec,
        )

        if response.cmd_id is None:
            raise RuntimeError("Cannot find command ID.")

        # Don't delete session; Daytona kills child processes
        return await self._poll_response(
            session_id,
            response.cmd_id,
            timeout_sec=timeout_sec,
        )

    @_SDK_RETRY
    async def _sdk_upload_file(self, source_path: Path | str, target_path: str) -> None:
        sandbox = self._require_sandbox()
        await sandbox.fs.upload_file(str(source_path), target_path)

    @_SDK_RETRY
    async def _sdk_upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        sandbox = self._require_sandbox()

        file_uploads = []
        source_dir = Path(source_dir)

        # Walk with followlinks=False and skip symlinks (#411): a task or
        # workspace symlink under source_dir must not exfiltrate host files
        # into the remote Daytona sandbox.
        for file_path in iter_safe_tree(
            source_dir, context=f"daytona upload_dir {source_dir}"
        ):
            relative_path = file_path.relative_to(source_dir).as_posix()
            destination_path = f"{target_dir}/{relative_path}"
            file_uploads.append(
                FileUpload(
                    source=str(file_path),
                    destination=destination_path,
                )
            )

        if file_uploads:
            await sandbox.fs.upload_files(files=file_uploads)

    @_SDK_RETRY
    async def _sdk_download_file(
        self, source_path: str, target_path: Path | str
    ) -> None:
        sandbox = self._require_sandbox()
        await sandbox.fs.download_file(source_path, str(target_path))

    @_SDK_RETRY
    async def _sdk_download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        sandbox = self._require_sandbox()

        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        search_result = await sandbox.fs.search_files(source_dir, "*")

        file_downloads = []
        for file_path in search_result.files:
            try:
                file_info = await sandbox.fs.get_file_info(file_path)
            except DaytonaNotFoundError:
                self.logger.debug(
                    f"Skipping file not found during download_dir: {file_path}"
                )
                continue
            except Exception as exc:
                self.logger.warning(
                    "Could not read Daytona file info for %s; treating search "
                    "hit as a file: %s",
                    file_path,
                    exc,
                )
                file_info = None

            if file_info is None or not file_info.is_dir:
                path_obj = Path(file_path)
                relative_path = path_obj.relative_to(Path(source_dir))
                local_file_path = target_dir / relative_path
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                file_downloads.append(
                    FileDownloadRequest(
                        source=file_path,
                        destination=str(local_file_path),
                    )
                )

        if file_downloads:
            try:
                await sandbox.fs.download_files(files=file_downloads)
            except Exception as batch_error:
                self.logger.warning(
                    "Daytona batch download_dir failed for %s (%d files); "
                    "retrying files individually: %s",
                    source_dir,
                    len(file_downloads),
                    batch_error,
                )
                await self._download_files_individually(file_downloads)

    async def _download_files_individually(self, file_downloads: list[Any]) -> None:
        sandbox = self._require_sandbox()

        failures: list[str] = []
        downloaded = 0
        for request in file_downloads:
            source = request.source
            destination = request.destination
            try:
                await sandbox.fs.download_file(source, destination)
                downloaded += 1
            except DaytonaNotFoundError:
                self.logger.debug(
                    "Skipping file not found during individual download_dir: %s",
                    source,
                )
            except Exception as exc:
                failures.append(f"{source}: {exc}")

        if failures:
            preview = "; ".join(failures[:3])
            if len(failures) > 3:
                preview += f"; ... {len(failures) - 3} more"
            raise RuntimeError(f"Daytona individual download_dir failed: {preview}")
        if downloaded == 0:
            raise RuntimeError("Daytona individual download_dir recovered no files")

    # Public interface — delegates to strategy

    async def start(self, force_build: bool) -> None:
        return await self._strategy.start(force_build)

    async def stop(self, delete: bool) -> None:
        return await self._strategy.stop(delete)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
    ) -> ExecResult:
        """Run ``command`` to completion in the sandbox and return its result.

        Daytona runs every command as a *session* command and reports its
        ``exit_code`` only once the command completes; it treats the command as
        still-running while any child keeps the session's stdout/stderr stream
        open. A backgrounded process that inherits those fds (``mysvc &`` with no
        redirection) therefore holds ``exec`` open until ``timeout_sec`` elapses
        — or, when ``timeout_sec`` is ``None``, until the safety-net cap
        (:data:`_DAYTONA_EXEC_HARD_CAP_SEC`, currently 3600s) is reached, at
        which point a "Command timed out" :class:`RuntimeError` is raised. To run
        a daemon in the background, fully detach its std fds so it does not hold
        the session stream open, e.g.
        ``nohup CMD </dev/null >log 2>&1 &`` (redirection alone severs the
        inherited session stream; ``disown`` is bash/zsh-only and absent from a
        plain ``sh`` hook shell).

        This is a Daytona-specific asymmetry: ``DockerSandbox.exec`` returns as
        soon as the foreground ``sh -c`` exits regardless of orphaned background
        children, so a ``mysvc &`` setup hook can pass under Docker yet wedge on
        Daytona without the redirection above.
        """
        user = self._resolve_user(user)
        env = self._merge_env(env)
        return await self._strategy.exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
            service=service,
        )

    async def services(self) -> list[str]:
        return await self._strategy.services()

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        return await self._strategy.upload_file(source_path, target_path)

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        return await self._strategy.upload_dir(source_dir, target_dir, service=service)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        return await self._strategy.download_file(source_path, target_path)

    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        return await self._strategy.download_dir(
            source_dir, target_dir, service=service
        )

    async def is_dir(
        self, path: str, user: str | int | None = None, service: str = "main"
    ) -> bool:
        # The strategy fast-path only knows the agent's `main` container; a
        # non-main service must route through the generic `test -d` exec path
        # so the compose `service` selector is honored.
        if service != "main":
            return await super().is_dir(path, user=user, service=service)
        return await self._strategy.is_dir(path, user=self._resolve_user(user))

    async def is_file(
        self, path: str, user: str | int | None = None, service: str = "main"
    ) -> bool:
        # The strategy fast-path only knows the agent's `main` container; a
        # non-main service must route through the generic `test -f` exec path
        # so the compose `service` selector is honored.
        if service != "main":
            return await super().is_file(path, user=user, service=service)
        return await self._strategy.is_file(path, user=self._resolve_user(user))

    async def attach(self) -> None:
        return await self._strategy.attach()

    # Container snapshot/restore — delegates to active strategy (#384)

    @property
    def supports_snapshot(self) -> bool:
        return self._strategy.supports_snapshot

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        return await self._strategy.snapshot(name)

    async def restore(self, image: SandboxImage) -> None:
        return await self._strategy.restore(image)
