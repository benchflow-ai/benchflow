from __future__ import annotations

import asyncio
import importlib
import shlex
from pathlib import Path
from typing import Any
from uuid import uuid4

from tenacity import retry, stop_after_attempt, wait_exponential

from benchflow.sandbox._base import ExecResult

_DAYTONA_IMPORT_ERROR_MESSAGE = (
    "Daytona support requires the optional 'daytona' package. "
    "Install it with '.venv/bin/uv pip install daytona>=0.121'."
)


class DaytonaSDKImportError(RuntimeError):
    """Raised when Daytona SDK support is requested but not installed."""


class SandboxUnavailableError(RuntimeError):
    """Raised when a Daytona sandbox operation is attempted before creation."""


class _SDKNamespace:
    def __init__(self, module: Any, snapshot_module: Any) -> None:
        self.AsyncDaytona = module.AsyncDaytona
        self.CreateSandboxFromImageParams = module.CreateSandboxFromImageParams
        self.CreateSandboxFromSnapshotParams = module.CreateSandboxFromSnapshotParams
        self.DaytonaNotFoundError = module.DaytonaNotFoundError
        self.FileDownloadRequest = module.FileDownloadRequest
        self.FileUpload = module.FileUpload
        self.Image = module.Image
        self.Resources = module.Resources
        self.SessionExecuteRequest = module.SessionExecuteRequest
        self.SnapshotState = snapshot_module.SnapshotState


def import_daytona_sdk() -> _SDKNamespace:
    try:
        module = importlib.import_module("daytona")
        snapshot_module = importlib.import_module("daytona._async.snapshot")
    except ModuleNotFoundError as exc:
        raise DaytonaSDKImportError(_DAYTONA_IMPORT_ERROR_MESSAGE) from exc
    return _SDKNamespace(module, snapshot_module)


def _require_sandbox(sandbox: Any | None) -> Any:
    if sandbox is None:
        raise SandboxUnavailableError(
            "Sandbox not found. Please build the environment first."
        )
    return sandbox


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def create_sandbox(
    *, client_manager: Any, params: Any, build_timeout_sec: float
) -> Any:
    daytona = await client_manager.get_client()
    create_task = asyncio.ensure_future(
        daytona.create(params=params, timeout=round(build_timeout_sec))
    )
    try:
        return await asyncio.shield(create_task)
    except asyncio.CancelledError:
        try:
            return await asyncio.wait_for(create_task, timeout=30)
        except (asyncio.CancelledError, TimeoutError, Exception):
            create_task.cancel()
        raise


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def stop_sandbox(sandbox: Any | None) -> None:
    await _require_sandbox(sandbox).delete()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def _get_session_command_with_retry(
    sandbox: Any | None, session_id: str, command_id: str
) -> Any:
    return await _require_sandbox(sandbox).process.get_session_command(
        session_id, command_id
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def _get_session_command_logs_with_retry(
    sandbox: Any | None, session_id: str, command_id: str
) -> Any:
    return await _require_sandbox(sandbox).process.get_session_command_logs(
        session_id, command_id
    )


async def sandbox_exec(
    sandbox: Any | None,
    command: str,
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout_sec: int | None = None,
    shell: str = "bash -c",
    user: str | int | None = None,
) -> ExecResult:
    sdk = import_daytona_sdk()
    sandbox = _require_sandbox(sandbox)
    session_id = str(uuid4())
    await sandbox.process.create_session(session_id)

    command = f"{shell} {shlex.quote(command)}"
    if env:
        env_args = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        command = f"env {env_args} {command}"
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
        sdk.SessionExecuteRequest(command=command, run_async=True),
        timeout=timeout_sec,
    )
    if response.cmd_id is None:
        raise RuntimeError("Cannot find command ID.")

    result = await _get_session_command_with_retry(sandbox, session_id, response.cmd_id)
    while result.exit_code is None:
        await asyncio.sleep(1)
        result = await _get_session_command_with_retry(sandbox, session_id, result.id)

    logs = await _get_session_command_logs_with_retry(
        sandbox, session_id, response.cmd_id
    )
    return ExecResult(
        stdout=logs.stdout,
        stderr=logs.stderr,
        return_code=int(result.exit_code),
    )


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def sdk_upload_file(
    sandbox: Any | None, source_path: Path | str, target_path: str
) -> None:
    await _require_sandbox(sandbox).fs.upload_file(str(source_path), target_path)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def sdk_upload_dir(
    sandbox: Any | None, source_dir: Path | str, target_dir: str
) -> None:
    sdk = import_daytona_sdk()
    sandbox = _require_sandbox(sandbox)
    uploads: list[Any] = []
    source_dir = Path(source_dir)
    for file_path in source_dir.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(source_dir).as_posix()
            uploads.append(
                sdk.FileUpload(
                    source=str(file_path),
                    destination=f"{target_dir}/{relative_path}",
                )
            )
    if uploads:
        await sandbox.fs.upload_files(files=uploads)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def sdk_download_file(
    sandbox: Any | None, source_path: str, target_path: Path | str
) -> None:
    await _require_sandbox(sandbox).fs.download_file(source_path, str(target_path))


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def sdk_download_dir(
    sandbox: Any | None, source_dir: str, target_dir: Path | str
) -> None:
    sdk = import_daytona_sdk()
    sandbox = _require_sandbox(sandbox)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    search_result = await sandbox.fs.search_files(source_dir, "*")
    downloads: list[Any] = []
    for file_path in search_result.files:
        try:
            file_info = await sandbox.fs.get_file_info(file_path)
        except sdk.DaytonaNotFoundError:
            continue
        if not file_info.is_dir:
            path_obj = Path(file_path)
            relative_path = path_obj.relative_to(Path(source_dir))
            local_file_path = target_dir / relative_path
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            downloads.append(
                sdk.FileDownloadRequest(
                    source=file_path,
                    destination=str(local_file_path),
                )
            )
    if downloads:
        await sandbox.fs.download_files(files=downloads)
