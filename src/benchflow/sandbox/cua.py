"""Cua-backed desktop sandbox provider.

This backend is the first BenchFlow bridge to Cua desktop environments. It
implements the existing Sandbox surface with Cua's async ``Sandbox`` API:

* ``start`` creates or connects to a Cua sandbox.
* ``exec`` runs commands through ``sb.shell.run``.
* file transfer is bootstrapped through shell/base64 so BenchFlow's verifier
  and agent setup paths can run before we depend on a richer Cua file API.

The provider is intentionally capability-first rather than OSWorld-specific.
OSWorld/CUA adapters can ask for ``--sandbox cua`` and later tighten the image
selection/capability checks without changing the rollout kernel.
"""

from __future__ import annotations

import asyncio
import base64
import math
import os
import re
import shlex
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from benchflow.sandbox._base import (
    BaseSandbox,
    ExecResult,
    wrap_command_with_env_file,
)
from benchflow.sandbox.process import _BUFFER_LIMIT, LiveProcess
from benchflow.sandbox.protocol import SandboxStartupError
from benchflow.task.config import TaskOS

_CUA_NAME_INVALID = re.compile(r"[^a-z0-9-]+")


def _load_cua() -> tuple[Any, Any]:
    """Import the optional Cua SDK with an actionable dependency error."""
    try:
        from cua_sandbox import Image, Sandbox

        return Image, Sandbox
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        from cua import Image, Sandbox

        return Image, Sandbox
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Missing optional dependency for 'cua' sandbox. "
            "Install it with `uv sync --extra sandbox-cua` for local "
            "development, or `pip install 'benchflow[sandbox-cua]'` for a "
            "packaged install."
        ) from exc


def _env_bool(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number of seconds") from exc


def _cua_api_key() -> str | None:
    return os.environ.get("CUA_API_KEY") or None


def _cua_local() -> bool:
    return _env_bool("BENCHFLOW_CUA_LOCAL", default=False)


def _cua_name_part(value: str) -> str:
    slug = _CUA_NAME_INVALID.sub("-", value.strip().lower()).strip("-")
    return slug or "task"


async def list_cua_sandboxes() -> list[Any]:
    """List Cua sandboxes visible to the configured SDK credentials."""
    _Image, Sandbox = _load_cua()
    return await Sandbox.list(local=_cua_local(), api_key=_cua_api_key())


async def delete_cua_sandbox(name: str) -> None:
    """Delete one Cua sandbox by name using the configured SDK credentials."""
    _Image, Sandbox = _load_cua()
    await Sandbox.delete(name, local=_cua_local(), api_key=_cua_api_key())


def _command_result_value(result: Any, *names: str, default: Any = "") -> Any:
    for name in names:
        if hasattr(result, name):
            return getattr(result, name)
    return default


def _normalize_cua_result(result: Any) -> ExecResult:
    """Map Cua CommandResult-like objects to BenchFlow's ExecResult."""
    return ExecResult(
        stdout=str(_command_result_value(result, "stdout", default="") or ""),
        stderr=str(_command_result_value(result, "stderr", default="") or ""),
        return_code=int(
            _command_result_value(result, "returncode", "return_code", default=0) or 0
        ),
    )


def _safe_extract_tar(tar: tarfile.TarFile, dest: Path) -> None:
    root = dest.resolve()
    for member in tar.getmembers():
        target = (root / member.name).resolve()
        if not target.is_relative_to(root):
            raise RuntimeError(f"Refusing unsafe archive member: {member.name}")
    tar.extractall(root, filter="data")


class CuaSandbox(BaseSandbox):
    """Sandbox backend for Cua cloud/local desktop environments."""

    _ENV_FILE_PREFIX = "/tmp/.benchflow_cua_exec_env_"

    @classmethod
    def preflight(cls) -> None:
        _load_cua()
        if os.environ.get("BENCHFLOW_CUA_REQUIRE_API_KEY") in {
            "1",
            "true",
            "yes",
        } and not os.environ.get("CUA_API_KEY"):
            raise SystemExit(
                "Cua cloud mode requires CUA_API_KEY. Set it in the "
                "environment or run Cua's auth flow before using --sandbox cua."
            )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._sandbox: Any | None = None
        self._cua_name = os.environ.get("BENCHFLOW_CUA_SANDBOX_NAME") or os.environ.get(
            "CUA_SANDBOX_NAME"
        )
        self._created = False
        super().__init__(*args, **kwargs)

    @property
    def sandbox_id(self) -> str | None:
        if self._sandbox is None:
            return None
        return self._cua_name or self.environment_name

    @property
    def host(self) -> str:
        return "localhost"

    @property
    def expose_ports(self) -> list[int]:
        return []

    @property
    def process(self) -> LiveProcess:
        if not _cua_local():
            raise RuntimeError(
                "Cua cloud ACP agents require a streaming process API; "
                "use BENCHFLOW_CUA_LOCAL=1 for local Cua ACP dogfood."
            )
        if self._cua_name is None:
            raise RuntimeError("Cua sandbox is not started")
        return CuaLocalDockerProcess(self._cua_name)

    def _validate_definition(self) -> None:
        if self.task_env_config.os not in {
            TaskOS.LINUX,
            TaskOS.WINDOWS,
            TaskOS.MACOS,
            TaskOS.ANDROID,
        }:
            raise ValueError(
                f"Cua sandbox does not support task OS {self.task_env_config.os!r}"
            )

    def _image(self) -> Any:
        Image, _Sandbox = _load_cua()
        image = self.task_env_config.docker_image
        if image:
            return Image.from_registry(image)

        if self.task_env_config.os == TaskOS.WINDOWS:
            return Image.windows(os.environ.get("BENCHFLOW_CUA_WINDOWS_VERSION", "11"))

        # macOS and Android are always VM-backed (no container kind); on Apple
        # Silicon they run locally via Hypervisor.framework.
        if self.task_env_config.os == TaskOS.MACOS:
            return Image.macos(
                version=os.environ.get("BENCHFLOW_CUA_MACOS_VERSION", "26"),
                kind="vm",
            )
        if self.task_env_config.os == TaskOS.ANDROID:
            return Image.android(
                version=os.environ.get("BENCHFLOW_CUA_ANDROID_VERSION", "14"),
                kind="vm",
            )

        linux_kind = os.environ.get("BENCHFLOW_CUA_LINUX_KIND", "vm")
        return Image.linux(
            distro=os.environ.get("BENCHFLOW_CUA_LINUX_DISTRO", "ubuntu"),
            version=os.environ.get("BENCHFLOW_CUA_LINUX_VERSION", "24.04"),
            kind=linux_kind,
        )

    def _cua_kwargs(self) -> dict[str, Any]:
        storage_mb = self.task_env_config.storage_mb
        kwargs = {
            "api_key": _cua_api_key(),
            "local": _cua_local(),
            "cpu": self.task_env_config.cpus or None,
            "memory_mb": self.task_env_config.memory_mb or None,
            "disk_gb": math.ceil(storage_mb / 1024) if storage_mb else None,
            "region": os.environ.get("BENCHFLOW_CUA_REGION", "us-east-1"),
            "telemetry_enabled": _env_bool("CUA_TELEMETRY_ENABLED", default=False),
        }
        time_to_start = _env_float("BENCHFLOW_CUA_TIME_TO_START_SEC")
        request_timeout = _env_float("BENCHFLOW_CUA_REQUEST_TIMEOUT_SEC")
        if time_to_start is not None:
            kwargs["time_to_start"] = time_to_start
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        return kwargs

    def _create_name(self) -> str:
        explicit = os.environ.get("BENCHFLOW_CUA_CREATE_NAME")
        if explicit and explicit.strip():
            return explicit.strip()
        prefix = os.environ.get("BENCHFLOW_CUA_NAME_PREFIX", "benchflow-")
        name = (
            f"{prefix}{_cua_name_part(self.environment_name)}-"
            f"{_cua_name_part(self.session_id)}"
        )
        return name[:63].strip("-") or "benchflow-task"

    def _should_pass_create_name(self) -> bool:
        # cua-sandbox 0.1.16 treats Sandbox.create(name=...) on cloud as
        # "connect to this existing VM", which 404s before creation. Local
        # runtimes do support named creates, and cloud named creates can be
        # re-enabled when the SDK/API behavior is fixed.
        return _cua_local() or _env_bool("BENCHFLOW_CUA_NAMED_CREATE", default=False)

    async def start(self, force_build: bool = False) -> None:
        _Image, Sandbox = _load_cua()
        kwargs = self._cua_kwargs()
        try:
            if self._cua_name:
                self._sandbox = await Sandbox.connect(self._cua_name, **kwargs)
                self._created = False
            else:
                desired_name = self._create_name()
                if self._should_pass_create_name():
                    kwargs["name"] = desired_name
                self._sandbox = await Sandbox.create(self._image(), **kwargs)
                self._cua_name = (
                    getattr(self._sandbox, "name", None)
                    or kwargs.get("name")
                    or desired_name
                )
                self._created = True
        except Exception as exc:
            self._sandbox = None
            hint = (
                "Cua sandbox failed to become ready. If the Cua cloud VM is "
                "running but its command endpoint returns 404, retry the run "
                "or attach to a known-good VM with BENCHFLOW_CUA_SANDBOX_NAME. "
                "Use BENCHFLOW_CUA_TIME_TO_START_SEC to tune the startup wait."
            )
            raise SandboxStartupError(hint) from exc
        try:
            init_result = await self.exec(
                "mkdir -p /logs/agent /logs/verifier /logs/artifacts /app /tmp",
                user="root",
            )
            if init_result.return_code != 0:
                raise SandboxStartupError(
                    "Cua sandbox failed to initialize BenchFlow runtime directories"
                )
            chmod_result = await self.exec(
                "chmod 755 /logs && chmod 777 /logs/agent /logs/verifier "
                "/logs/artifacts /tmp",
                user="root",
            )
            if chmod_result.return_code != 0:
                raise SandboxStartupError(
                    "Cua sandbox failed to initialize BenchFlow runtime permissions"
                )
        except SandboxStartupError:
            raise
        except Exception as exc:
            raise SandboxStartupError(
                "Cua sandbox failed to initialize BenchFlow runtime directories"
            ) from exc

    async def stop(self, delete: bool = True) -> None:
        if self._sandbox is None:
            return
        sandbox = self._sandbox
        self._sandbox = None
        if delete and self._created:
            for method_name in ("destroy", "delete", "kill"):
                method = getattr(sandbox, method_name, None)
                if method is not None:
                    await method()
                    return
        disconnect = getattr(sandbox, "disconnect", None)
        if disconnect is not None:
            await disconnect()

    def _require_sandbox(self) -> Any:
        if self._sandbox is None:
            raise RuntimeError("Cua sandbox is not started")
        return self._sandbox

    @staticmethod
    def _reject_non_main(service: str) -> None:
        if service != "main":
            raise RuntimeError(
                "Cua sandbox does not support compose service selection; "
                "only service='main' is available."
            )

    @classmethod
    def _wrap_command_with_env_file(cls, env: dict[str, str], command: str) -> str:
        return wrap_command_with_env_file(
            env, command, env_path_prefix=cls._ENV_FILE_PREFIX
        )

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
    ) -> ExecResult:
        self._reject_non_main(service)
        sandbox = self._require_sandbox()
        command_env = self._merge_env(env)
        if cwd:
            command = f"cd {shlex.quote(cwd)} && {command}"
        if command_env:
            command = self._wrap_command_with_env_file(command_env, command)
        if user is not None and str(user) not in {""}:
            if str(user) in {"root", "0"}:
                quoted = shlex.quote(command)
                command = (
                    'if [ "$(id -u)" = 0 ]; then '
                    f"/bin/sh -c {quoted}; "
                    "elif command -v sudo >/dev/null 2>&1; then "
                    f"sudo -n /bin/sh -c {quoted}; "
                    "else "
                    "echo 'root execution requested but sudo is unavailable' >&2; "
                    "exit 126; "
                    "fi"
                )
            else:
                command = (
                    f"su -s /bin/sh {shlex.quote(str(user))} -c {shlex.quote(command)}"
                )
        timeout = timeout_sec or 30
        try:
            result = await asyncio.wait_for(
                sandbox.shell.run(command, timeout=timeout),
                timeout=timeout,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"Cua shell command timed out after {timeout} seconds"
            ) from exc
        return _normalize_cua_result(result)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        data = base64.b64encode(source.read_bytes()).decode()
        parent = str(Path(target_path).parent)
        script = (
            "python3 - <<'PY'\n"
            "import base64, pathlib\n"
            f"path = pathlib.Path({target_path!r})\n"
            "path.parent.mkdir(parents=True, exist_ok=True)\n"
            f"path.write_bytes(base64.b64decode({data!r}))\n"
            "PY"
        )
        if parent and parent != ".":
            await self.exec(f"mkdir -p {shlex.quote(parent)}", user="root")
        result = await self.exec(script, user="root", timeout_sec=120)
        if result.return_code != 0:
            raise RuntimeError(
                f"Cua upload_file failed: {result.stderr or result.stdout}"
            )

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        self._reject_non_main(service)
        source = Path(source_dir)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
            with tarfile.open(archive.name, "w:gz") as tar:
                for child in sorted(p for p in source.rglob("*") if p.is_file()):
                    if child.is_symlink():
                        continue
                    tar.add(child, arcname=child.relative_to(source).as_posix())
            archive.seek(0)
            data = base64.b64encode(archive.read()).decode()
        script = (
            "python3 - <<'PY'\n"
            "import base64, io, pathlib, tarfile\n"
            f"target = pathlib.Path({target_dir!r})\n"
            "target.mkdir(parents=True, exist_ok=True)\n"
            f"payload = base64.b64decode({data!r})\n"
            "with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as tar:\n"
            "    tar.extractall(target)\n"
            "PY"
        )
        result = await self.exec(script, user="root", timeout_sec=300)
        if result.return_code != 0:
            raise RuntimeError(
                f"Cua upload_dir failed: {result.stderr or result.stdout}"
            )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        command = (
            "python3 - <<'PY'\n"
            "import base64, pathlib\n"
            f"print(base64.b64encode(pathlib.Path({source_path!r}).read_bytes()).decode())\n"
            "PY"
        )
        result = await self.exec(command, user="root", timeout_sec=120)
        if result.return_code != 0:
            raise RuntimeError(
                f"Cua download_file failed: {result.stderr or result.stdout}"
            )
        stdout = result.stdout or ""
        Path(target_path).write_bytes(base64.b64decode(stdout.strip()))

    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        self._reject_non_main(service)
        command = (
            "python3 - <<'PY'\n"
            "import base64, io, pathlib, tarfile\n"
            f"source = pathlib.Path({source_dir!r})\n"
            "buf = io.BytesIO()\n"
            "with tarfile.open(fileobj=buf, mode='w:gz') as tar:\n"
            "    for child in sorted(p for p in source.rglob('*') if p.is_file()):\n"
            "        tar.add(child, arcname=child.relative_to(source).as_posix())\n"
            "print(base64.b64encode(buf.getvalue()).decode())\n"
            "PY"
        )
        result = await self.exec(command, user="root", timeout_sec=300)
        if result.return_code != 0:
            raise RuntimeError(
                f"Cua download_dir failed: {result.stderr or result.stdout}"
            )
        dest = Path(target_dir)
        dest.mkdir(parents=True, exist_ok=True)
        stdout = result.stdout or ""
        payload = base64.b64decode(stdout.strip())
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
            archive.write(payload)
            archive.flush()
            with tarfile.open(archive.name, "r:gz") as tar:
                _safe_extract_tar(tar, dest)

    async def read_file(self, path: str) -> bytes:
        with tempfile.NamedTemporaryFile() as tmp:
            await self.download_file(path, tmp.name)
            return Path(tmp.name).read_bytes()

    async def write_file(self, path: str, content: bytes) -> None:
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(content)
            tmp.flush()
            await self.upload_file(tmp.name, path)

    async def screenshot(
        self,
        text: str | None = None,
        *,
        format: str = "png",
        quality: int = 95,
    ) -> bytes:
        sandbox = self._require_sandbox()
        method = getattr(sandbox, "screenshot", None)
        if method is None:
            raise RuntimeError("Cua sandbox does not expose screenshot()")
        return await method(text=text, format=format, quality=quality)

    async def screenshot_base64(
        self,
        text: str | None = None,
        *,
        format: str = "png",
        quality: int = 95,
    ) -> str:
        sandbox = self._require_sandbox()
        method = getattr(sandbox, "screenshot_base64", None)
        if method is not None:
            return str(await method(text=text, format=format, quality=quality))
        return base64.b64encode(
            await self.screenshot(text=text, format=format, quality=quality)
        ).decode()

    async def get_dimensions(self) -> tuple[int, int]:
        sandbox = self._require_sandbox()
        method = getattr(sandbox, "get_dimensions", None)
        if method is None:
            raise RuntimeError("Cua sandbox does not expose get_dimensions()")
        width, height = await method()
        return int(width), int(height)

    async def get_display_url(self, *, share: bool = False) -> str:
        sandbox = self._require_sandbox()
        method = getattr(sandbox, "get_display_url", None)
        if method is None:
            raise RuntimeError("Cua sandbox does not expose get_display_url()")
        return str(await method(share=share))


class CuaLocalDockerProcess(LiveProcess):
    """Live ACP stdio bridge for local Cua Docker-backed sandboxes."""

    _ENV_PATH = "/tmp/.benchflow_cua_process_env"

    def __init__(self, container_name: str) -> None:
        self._container_name = container_name

    async def _write_env_to_container(self, env: dict[str, str]) -> None:
        lines = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in env.items())
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            "-u",
            "root",
            self._container_name,
            "bash",
            "-c",
            f"cat > {self._ENV_PATH} && chmod 600 {self._ENV_PATH}",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        _stdout, stderr = await asyncio.wait_for(proc.communicate(lines.encode()), 30)
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to write env file in Cua local container "
                f"(rc={proc.returncode}): {stderr.decode(errors='replace')[:500]}"
            )

    async def start(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        if env:
            await self._write_env_to_container(env)
            command = f". {self._ENV_PATH} && rm -f {self._ENV_PATH} && {command}"

        cmd = ["docker", "exec", "-i", "-u", "root"]
        if cwd:
            cmd.extend(["-w", cwd])
        cmd.extend([self._container_name, "bash", "-c", command])
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
            limit=_BUFFER_LIMIT,
        )
