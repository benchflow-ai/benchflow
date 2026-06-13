"""Cua sandbox provider unit tests with a fake SDK module."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchflow.sandbox.cua import CuaLocalDockerProcess, CuaSandbox
from benchflow.sandbox.protocol import SandboxStartupError
from benchflow.task.config import SandboxConfig, TaskOS


class _FakeShell:
    def __init__(self) -> None:
        self.commands: list[tuple[str, int]] = []

    async def run(self, command: str, timeout: int = 30) -> Any:
        self.commands.append((command, timeout))
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)


class _HangingShell:
    async def run(self, command: str, timeout: int = 30) -> Any:
        await asyncio.sleep(3600)


class _FakeCuaSandboxInstance:
    def __init__(self) -> None:
        self.shell = _FakeShell()
        self.destroyed = False
        self.disconnected = False

    async def screenshot(
        self,
        text: str | None = None,
        *,
        format: str = "png",
        quality: int = 95,
    ) -> bytes:
        return f"{text}:{format}:{quality}".encode()

    async def screenshot_base64(
        self,
        text: str | None = None,
        *,
        format: str = "png",
        quality: int = 95,
    ) -> str:
        return f"base64:{text}:{format}:{quality}"

    async def get_dimensions(self) -> tuple[int, int]:
        return (1024, 768)

    async def get_display_url(self, *, share: bool = False) -> str:
        return f"https://display.example/{int(share)}"

    async def destroy(self) -> None:
        self.destroyed = True

    async def disconnect(self) -> None:
        self.disconnected = True


class _FakeImage:
    @classmethod
    def linux(cls, distro: str, version: str, kind: str) -> tuple[str, str, str, str]:
        return ("linux", distro, version, kind)

    @classmethod
    def windows(cls, version: str) -> tuple[str, str]:
        return ("windows", version)

    @classmethod
    def macos(cls, version: str, kind: str) -> tuple[str, str, str]:
        return ("macos", version, kind)

    @classmethod
    def android(cls, version: str, kind: str) -> tuple[str, str, str]:
        return ("android", version, kind)

    @classmethod
    def from_registry(cls, ref: str) -> tuple[str, str]:
        return ("registry", ref)


class _FakeSandbox:
    created_image: Any = None
    created_kwargs: dict[str, Any] | None = None
    instance: _FakeCuaSandboxInstance | None = None
    create_error: Exception | None = None

    @classmethod
    async def create(cls, image: Any, **kwargs: Any) -> _FakeCuaSandboxInstance:
        if cls.create_error is not None:
            raise cls.create_error
        cls.created_image = image
        cls.created_kwargs = kwargs
        cls.instance = _FakeCuaSandboxInstance()
        return cls.instance

    @classmethod
    async def connect(cls, name: str, **kwargs: Any) -> _FakeCuaSandboxInstance:
        cls.created_image = ("connect", name)
        cls.created_kwargs = kwargs
        cls.instance = _FakeCuaSandboxInstance()
        return cls.instance


@pytest.fixture
def fake_cua(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSandbox]:
    _FakeSandbox.created_image = None
    _FakeSandbox.created_kwargs = None
    _FakeSandbox.instance = None
    _FakeSandbox.create_error = None
    monkeypatch.setitem(
        sys.modules,
        "cua_sandbox",
        SimpleNamespace(Image=_FakeImage, Sandbox=_FakeSandbox),
    )
    return _FakeSandbox


def _sandbox(tmp_path: Path, config: SandboxConfig | None = None) -> CuaSandbox:
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    return CuaSandbox(
        environment_dir=env_dir,
        environment_name="task",
        session_id="rollout",
        rollout_paths=None,
        task_env_config=config or SandboxConfig(),
    )


def test_cua_preflight_imports_optional_sdk(fake_cua) -> None:
    CuaSandbox.preflight()


@pytest.mark.asyncio
async def test_cua_start_creates_default_linux_vm(
    fake_cua: type[_FakeSandbox],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUA_API_KEY", "test-key")
    monkeypatch.setenv("BENCHFLOW_CUA_REGION", "us-west-2")
    sandbox = _sandbox(tmp_path)

    await sandbox.start()

    assert fake_cua.created_image == ("linux", "ubuntu", "24.04", "vm")
    assert fake_cua.created_kwargs == {
        "api_key": "test-key",
        "local": False,
        "cpu": 1,
        "memory_mb": 2048,
        "disk_gb": 10,
        "region": "us-west-2",
        "telemetry_enabled": False,
    }
    assert sandbox.sandbox_id == "benchflow-task-rollout"
    assert fake_cua.instance is not None
    init_cmd = fake_cua.instance.shell.commands[0][0]
    assert "mkdir -p /logs/agent /logs/verifier /logs/artifacts /app /tmp" in init_cmd
    assert 'if [ "$(id -u)" = 0 ]' in init_cmd
    chmod_cmd = fake_cua.instance.shell.commands[1][0]
    assert "chmod 755 /logs" in chmod_cmd


@pytest.mark.asyncio
async def test_cua_start_passes_create_name_for_local_runtime(
    fake_cua: type[_FakeSandbox],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCHFLOW_CUA_LOCAL", "1")
    sandbox = _sandbox(tmp_path)

    await sandbox.start()

    assert fake_cua.created_kwargs is not None
    assert fake_cua.created_kwargs["name"] == "benchflow-task-rollout"
    assert fake_cua.created_kwargs["local"] is True


@pytest.mark.asyncio
async def test_cua_start_can_opt_into_cloud_named_create(
    fake_cua: type[_FakeSandbox],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCHFLOW_CUA_NAMED_CREATE", "1")
    sandbox = _sandbox(tmp_path)

    await sandbox.start()

    assert fake_cua.created_kwargs is not None
    assert fake_cua.created_kwargs["name"] == "benchflow-task-rollout"


@pytest.mark.asyncio
async def test_cua_start_forwards_timeout_overrides(
    fake_cua: type[_FakeSandbox],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCHFLOW_CUA_TIME_TO_START_SEC", "45")
    monkeypatch.setenv("BENCHFLOW_CUA_REQUEST_TIMEOUT_SEC", "9.5")
    sandbox = _sandbox(tmp_path)

    await sandbox.start()

    assert fake_cua.created_kwargs is not None
    assert fake_cua.created_kwargs["time_to_start"] == 45.0
    assert fake_cua.created_kwargs["request_timeout"] == 9.5


@pytest.mark.asyncio
async def test_cua_start_wraps_sdk_startup_errors(
    fake_cua: type[_FakeSandbox],
    tmp_path: Path,
) -> None:
    fake_cua.create_error = TimeoutError("command endpoint not reachable")
    sandbox = _sandbox(tmp_path)

    with pytest.raises(SandboxStartupError, match="Cua sandbox failed to become ready"):
        await sandbox.start()


@pytest.mark.asyncio
async def test_cua_exec_wraps_cwd_env_and_user(fake_cua, tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    await sandbox.start()

    result = await sandbox.exec("echo hi", cwd="/app", env={"FOO": "bar"}, user="agent")

    assert result.return_code == 0
    assert fake_cua.instance is not None
    command = fake_cua.instance.shell.commands[-1][0]
    assert "su -s /bin/sh agent -c" in command
    assert "benchflow_cua_exec_env" in command
    assert "cd /app" in command


@pytest.mark.asyncio
async def test_cua_exec_hard_bounds_sdk_shell_run(fake_cua, tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    await sandbox.start()
    assert fake_cua.instance is not None
    fake_cua.instance.shell = _HangingShell()

    with pytest.raises(TimeoutError, match="Cua shell command timed out after 1"):
        await sandbox.exec("printf never", timeout_sec=1)


@pytest.mark.asyncio
async def test_cua_exec_uses_sudo_for_root_when_needed(
    fake_cua, tmp_path: Path
) -> None:
    sandbox = _sandbox(tmp_path)
    await sandbox.start()

    await sandbox.exec("touch /instruction.md", user="root")

    assert fake_cua.instance is not None
    command = fake_cua.instance.shell.commands[-1][0]
    assert 'if [ "$(id -u)" = 0 ]' in command
    assert "sudo -n /bin/sh -c" in command
    assert "root execution requested but sudo is unavailable" in command


@pytest.mark.asyncio
async def test_cua_desktop_observation_helpers_delegate_to_sdk(
    fake_cua, tmp_path: Path
) -> None:
    sandbox = _sandbox(tmp_path)
    await sandbox.start()

    assert await sandbox.screenshot(text="hello", format="jpeg", quality=70) == (
        b"hello:jpeg:70"
    )
    assert await sandbox.screenshot_base64(text="hello") == "base64:hello:png:95"
    assert await sandbox.get_dimensions() == (1024, 768)
    assert await sandbox.get_display_url(share=True) == "https://display.example/1"


@pytest.mark.asyncio
async def test_cua_local_exposes_live_process_bridge(
    fake_cua,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCHFLOW_CUA_LOCAL", "1")
    sandbox = _sandbox(tmp_path)
    await sandbox.start()

    assert isinstance(sandbox.process, CuaLocalDockerProcess)


@pytest.mark.asyncio
async def test_cua_cloud_process_bridge_fails_explicitly(
    fake_cua, tmp_path: Path
) -> None:
    sandbox = _sandbox(tmp_path)
    await sandbox.start()

    with pytest.raises(RuntimeError, match="streaming process API"):
        _ = sandbox.process


def test_cua_image_selects_macos_vm_with_version_knob(
    fake_cua: type[_FakeSandbox],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCHFLOW_CUA_MACOS_VERSION", "15")
    sandbox = _sandbox(tmp_path, SandboxConfig(os=TaskOS.MACOS))

    assert sandbox._image() == ("macos", "15", "vm")


def test_cua_image_macos_defaults_to_26(
    fake_cua: type[_FakeSandbox], tmp_path: Path
) -> None:
    sandbox = _sandbox(tmp_path, SandboxConfig(os=TaskOS.MACOS))

    assert sandbox._image() == ("macos", "26", "vm")


def test_cua_image_selects_android_vm_with_version_knob(
    fake_cua: type[_FakeSandbox],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCHFLOW_CUA_ANDROID_VERSION", "13")
    sandbox = _sandbox(tmp_path, SandboxConfig(os=TaskOS.ANDROID))

    assert sandbox._image() == ("android", "13", "vm")


def test_cua_image_android_defaults_to_14(
    fake_cua: type[_FakeSandbox], tmp_path: Path
) -> None:
    sandbox = _sandbox(tmp_path, SandboxConfig(os=TaskOS.ANDROID))

    assert sandbox._image() == ("android", "14", "vm")


@pytest.mark.parametrize("os_value", [TaskOS.MACOS, TaskOS.ANDROID])
def test_cua_os_gate_accepts_desktop_targets(
    fake_cua: type[_FakeSandbox], tmp_path: Path, os_value: TaskOS
) -> None:
    # _validate_definition runs during construction; it must not raise.
    sandbox = _sandbox(tmp_path, SandboxConfig(os=os_value))

    assert sandbox.task_env_config.os == os_value


@pytest.mark.asyncio
async def test_cua_stop_destroys_created_sandbox(fake_cua, tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    await sandbox.start()
    instance = fake_cua.instance

    await sandbox.stop(delete=True)

    assert instance is not None
    assert instance.destroyed is True


@pytest.mark.parametrize("ctor", ["macos", "android"])
def test_cua_sdk_reports_local_support_for_desktop_images(ctor: str) -> None:
    """The installed Cua SDK can describe local support without booting a VM.

    This exercises the real SDK (not the fake module). ``local_support()`` is a
    pure host-capability probe; it must not touch the network or start a VM.
    """
    cua_sandbox = pytest.importorskip("cua_sandbox")
    image = getattr(cua_sandbox.Image, ctor)()

    support = image.local_support()

    # ``supported`` is the contract we rely on; assert it is a real boolean
    # rather than the specific value, which is host-dependent.
    assert isinstance(support.supported, bool)
