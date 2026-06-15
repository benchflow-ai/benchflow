"""Native DaytonaSandbox with RL-first terminology.

Supports two strategies:
- Direct: single-container sandbox (Dockerfile only)
- DinD: Docker-in-Docker with compose (docker-compose.yaml present)
"""

from __future__ import annotations

import asyncio
import atexit
import importlib
import logging
import shlex
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from tenacity import (
        retry,
        retry_if_exception,
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

    def retry_if_exception(*_args: Any, **_kwargs: Any) -> None:
        return None

    def stop_after_attempt(*_args: Any, **_kwargs: Any) -> Any:
        return None

    def wait_exponential(*_args: Any, **_kwargs: Any) -> Any:
        return None


from benchflow._paths import iter_safe_tree
from benchflow.sandbox._base import BaseSandbox, ExecResult
from benchflow.sandbox.daytona_dind import _DaytonaDinD

# Re-export the extracted command-wrapping helpers so existing imports of
# ``benchflow.sandbox.daytona._wrap_daytona_command_with_env_file`` (and the
# sibling underscore helpers) keep resolving from this module path unchanged.
# The unused names below are intentional façade re-exports, not dead imports.
from benchflow.sandbox.daytona_pty import (
    _DAYTONA_ENV_FILE_PREFIX,  # noqa: F401
    _daytona_preflight,
    _exec_failure_output,  # noqa: F401
    _reject_non_main_service,  # noqa: F401
    _wrap_daytona_command_with_env_file,
)

# Re-export the extracted ownership labels + auto-reaper so existing imports of
# ``benchflow.sandbox.daytona.reap_stale_sandboxes`` (and the private ownership
# helpers/constants) keep resolving from this module path unchanged.
from benchflow.sandbox.daytona_reaper import (
    _BENCHFLOW_LABEL_NAMESPACE,  # noqa: F401
    _BENCHFLOW_MANAGED_LABEL,  # noqa: F401
    _BENCHFLOW_MANAGED_VALUE,  # noqa: F401
    _REAP_DEFAULT_MAX_AGE_MIN,  # noqa: F401
    _REAP_FAILED_MAX_AGE_MIN,  # noqa: F401
    _REAP_FAILED_STATE_MARKERS,  # noqa: F401
    _benchflow_owned_labels,  # noqa: F401
    _is_benchflow_label_orphan,  # noqa: F401
    _is_benchflow_owned,  # noqa: F401
    reap_leaked_snapshots,  # noqa: F401
    reap_stale_sandboxes,  # noqa: F401
)
from benchflow.sandbox.daytona_strategies import _DaytonaDirect, _DaytonaStrategy
from benchflow.sandbox.metadata import persist_sandbox_info, record_snapshot_leak
from benchflow.sandbox.protocol import (
    SandboxImage,
    SandboxStartupError,
)
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths

# ``SandboxStartupError`` used to live in this module. It now lives in
# ``benchflow.sandbox.protocol`` so a base install without the
# ``sandbox-daytona`` extra can still import ``benchflow.rollout`` (issue #358).
# Re-export here for backward compatibility — existing imports of
# ``benchflow.sandbox.daytona.SandboxStartupError`` keep working.
#
# The reaper (``reap_stale_sandboxes`` + ownership labels), the command-wrapping
# helpers (``_wrap_daytona_command_with_env_file`` …) and the strategy classes
# (``_DaytonaDirect`` / ``_DaytonaDinD``) now live in sibling ``daytona_*``
# modules; they are re-exported above so every name previously importable from
# ``benchflow.sandbox.daytona`` keeps resolving from this path unchanged.
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
_DAYTONA_TRANSIENT_RETRY_CLASS_NAMES = frozenset(
    {
        "DaytonaConnectionError",
        "DaytonaRateLimitError",
        "DaytonaTimeoutError",
    }
)


def _is_daytona_transient_retry_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    exc_type = type(exc)
    return (
        exc_type.__module__.startswith("daytona.")
        and exc_type.__name__ in _DAYTONA_TRANSIENT_RETRY_CLASS_NAMES
    )


_DAYTONA_TRANSIENT_RETRY: Any = retry_if_exception(_is_daytona_transient_retry_error)

# Retry-attempt budgets for transient Daytona failures, named here so the
# repeated ``3`` cannot silently drift between the idempotent-SDK policy and
# sandbox creation (#532). ``_stop_sandbox`` deliberately uses a smaller budget.
_DAYTONA_RETRY_ATTEMPTS = 3
_DAYTONA_STOP_RETRY_ATTEMPTS = 2

# Shared tenacity policy for the idempotent Daytona SDK calls — session-command
# polling and filesystem up/download. Three attempts with exponential backoff,
# re-raising the final failure. ``_create_sandbox`` and ``_stop_sandbox`` keep
# their own policies (different backoff bounds, and ``_stop_sandbox`` a smaller
# attempt budget), so they are intentionally not folded in here. Reusing one
# ``retry(...)`` decorator across methods is safe: tenacity builds a fresh
# controller per decorated function.
_SDK_RETRY = retry(
    stop=stop_after_attempt(_DAYTONA_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=_DAYTONA_TRANSIENT_RETRY,
    reraise=True,
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
        stop=stop_after_attempt(_DAYTONA_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=_DAYTONA_TRANSIENT_RETRY,
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
        stop=stop_after_attempt(_DAYTONA_STOP_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=_DAYTONA_TRANSIENT_RETRY,
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

    def record_snapshot_leak(self, snapshot_names: list[str]) -> None:
        """Record snapshots that could not be deleted on teardown.

        Called by the strategy when ``snapshot.delete`` fails: the cloud
        snapshot is a real cost/quota leak we can't reclaim here, so persist its
        name for a post-mortem reaper / operator (see ``record_snapshot_leak``
        in :mod:`benchflow.sandbox.metadata`).
        """
        rollout_dir = self.rollout_paths.rollout_dir if self.rollout_paths else None
        record_snapshot_leak(snapshot_names, type(self).__name__, rollout_dir)
