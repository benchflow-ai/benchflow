"""macOS iOS Simulator sandbox provider tests.

Two layers:

* Unit tests stub the single ``simctl`` subprocess chokepoint (``_simctl``)
  and the host-capability probe, so preflight error paths and the
  create -> boot -> exec -> shutdown -> delete command shape are exercised
  without touching a real simulator.
* One live dogfood test, gated on a real iOS runtime being installed, boots an
  actual simulator device, runs a trivial command, tears it down, and asserts
  no ``benchflow-ios-`` device leaks — the provider's real lifecycle contract.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any

import pytest

from benchflow.sandbox import macos_ios_simulator as ios_sim
from benchflow.sandbox.macos_ios_simulator import (
    MacosIosSimulatorSandbox,
    detect_ios_simulator_capabilities,
    device_type_id,
    runtime_id,
)
from benchflow.sandbox.protocol import ExecResult, SandboxStartupError
from benchflow.task.config import SandboxConfig, TaskOS

_DEVICE_PREFIX = "benchflow-ios-"


# --------------------------------------------------------------------------
# Unit tests: stubbed simctl + probe
# --------------------------------------------------------------------------


class _FakeSimctl:
    """Records ``_simctl`` calls and returns scripted ExecResults.

    Keyed on the first simctl subcommand (``create``/``boot``/...); a default
    success result is returned for any subcommand not explicitly scripted.
    """

    def __init__(self, udid: str = "ABCD-1234") -> None:
        self.calls: list[tuple[str, ...]] = []
        self.udid = udid
        self._booted = False

    async def __call__(
        self, *args: str, timeout_sec: float | None = None, check: bool = False
    ) -> ExecResult:
        self.calls.append(args)
        sub = args[0] if args else ""
        if sub == "create":
            return ExecResult(stdout=self.udid + "\n", return_code=0)
        if sub == "boot":
            self._booted = True
            return ExecResult(return_code=0)
        if sub == "list" and "devices" in args:
            state = "Booted" if self._booted else "Shutdown"
            line = f"    iPhone 16 Pro ({self.udid}) ({state})\n"
            return ExecResult(stdout=line, return_code=0)
        return ExecResult(stdout="ok\n", return_code=0)

    def subcommands(self) -> list[str]:
        return [call[0] for call in self.calls if call]


def _sandbox(
    tmp_path: Path, config: SandboxConfig | None = None
) -> MacosIosSimulatorSandbox:
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    return MacosIosSimulatorSandbox(
        environment_dir=env_dir,
        environment_name="clock-001",
        session_id="rollout",
        rollout_paths=None,
        task_env_config=config or SandboxConfig(os=TaskOS.MACOS),
    )


def test_runtime_and_device_type_defaults_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BENCHFLOW_IOS_RUNTIME", raising=False)
    monkeypatch.delenv("BENCHFLOW_IOS_DEVICE_TYPE", raising=False)
    assert runtime_id() == "com.apple.CoreSimulator.SimRuntime.iOS-26-3"
    assert device_type_id() == "com.apple.CoreSimulator.SimDeviceType.iPhone-16-Pro"

    monkeypatch.setenv("BENCHFLOW_IOS_RUNTIME", "com.example.iOS-99")
    monkeypatch.setenv("BENCHFLOW_IOS_DEVICE_TYPE", "com.example.iPhone-99")
    assert runtime_id() == "com.example.iOS-99"
    assert device_type_id() == "com.example.iPhone-99"


def test_preflight_rejects_non_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    with pytest.raises(SystemExit, match="requires macOS"):
        MacosIosSimulatorSandbox.preflight()


def test_preflight_reports_missing_simctl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(ios_sim, "_run_host", lambda *a, **k: (127, "", ""))
    with pytest.raises(SystemExit, match="xcrun simctl"):
        MacosIosSimulatorSandbox.preflight()


def test_preflight_reports_missing_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(ios_sim, "_run_host", lambda *a, **k: (0, "", ""))
    monkeypatch.setattr(
        ios_sim,
        "detect_ios_simulator_capabilities",
        lambda: {
            "macos": True,
            "xcode-26": True,
            "ios-26-simulator-runtime": False,
            "appium-xcuitest": True,
        },
    )
    with pytest.raises(SystemExit, match="iOS Simulator runtime"):
        MacosIosSimulatorSandbox.preflight()


def test_preflight_reports_missing_appium(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(ios_sim, "_run_host", lambda *a, **k: (0, "", ""))
    monkeypatch.setattr(
        ios_sim,
        "detect_ios_simulator_capabilities",
        lambda: {
            "macos": True,
            "xcode-26": True,
            "ios-26-simulator-runtime": True,
            "appium-xcuitest": False,
        },
    )
    with pytest.raises(SystemExit, match="xcuitest driver"):
        MacosIosSimulatorSandbox.preflight()


def test_force_unsupported_env_zeros_all_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCHFLOW_IOS_FORCE_UNSUPPORTED", "1")
    caps = detect_ios_simulator_capabilities()
    assert caps == {
        "macos": False,
        "xcode-26": False,
        "ios-26-simulator-runtime": False,
        "appium-xcuitest": False,
    }


def test_validate_definition_rejects_non_host_os(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="macOS host"):
        _sandbox(tmp_path, SandboxConfig(os=TaskOS.WINDOWS))


async def test_start_creates_boots_and_tracks_udid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCHFLOW_IOS_RUNTIME", "RUNTIME-X")
    monkeypatch.setenv("BENCHFLOW_IOS_DEVICE_TYPE", "DEVICE-X")
    sandbox = _sandbox(tmp_path)
    fake = _FakeSimctl(udid="UDID-42")
    monkeypatch.setattr(sandbox, "_simctl", fake)

    await sandbox.start()

    assert sandbox.udid == "UDID-42"
    assert sandbox.sandbox_id == "UDID-42"
    assert fake.subcommands()[:2] == ["create", "boot"]
    # The create call names a benchflow-owned device and passes device-type
    # then runtime in the validated order.
    create = fake.calls[0]
    assert create[0] == "create"
    assert create[1].startswith(_DEVICE_PREFIX)
    assert create[2] == "DEVICE-X"
    assert create[3] == "RUNTIME-X"
    # Boot is confirmed via a `list devices` poll.
    assert "list" in fake.subcommands()


async def test_start_raises_structured_error_on_create_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = _sandbox(tmp_path)

    async def _fail(*args: str, **kwargs: Any) -> ExecResult:
        if args and args[0] == "create":
            return ExecResult(stderr="device type unavailable", return_code=1)
        return ExecResult(return_code=0)

    monkeypatch.setattr(sandbox, "_simctl", _fail)

    with pytest.raises(SandboxStartupError, match="Failed to create iOS Simulator"):
        await sandbox.start()
    # No UDID tracked -> nothing to leak.
    assert sandbox.udid is None


async def test_exec_runs_via_simctl_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = _sandbox(tmp_path)
    fake = _FakeSimctl(udid="UDID-7")
    monkeypatch.setattr(sandbox, "_simctl", fake)
    await sandbox.start()

    result = await sandbox.exec("echo hi", cwd="/tmp", env={"FOO": "a b"})

    assert result.return_code == 0
    spawn = fake.calls[-1]
    assert spawn[0] == "spawn"
    assert spawn[1] == "UDID-7"
    assert spawn[2:4] == ("/bin/sh", "-c")
    shell = spawn[4]
    # Env exports are shell-quoted and prefixed before the cwd change.
    assert "export FOO='a b'" in shell
    assert shell.index("export FOO=") < shell.index("cd /tmp")
    assert "cd /tmp && echo hi" in shell


async def test_exec_before_start_raises(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    with pytest.raises(RuntimeError, match="not started"):
        await sandbox.exec("true")


async def test_exec_rejects_non_main_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = _sandbox(tmp_path)
    monkeypatch.setattr(sandbox, "_simctl", _FakeSimctl())
    await sandbox.start()
    with pytest.raises(RuntimeError, match="no compose topology"):
        await sandbox.exec("true", service="target")


async def test_stop_shuts_down_and_deletes_no_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = _sandbox(tmp_path)
    fake = _FakeSimctl(udid="UDID-9")
    monkeypatch.setattr(sandbox, "_simctl", fake)
    await sandbox.start()

    await sandbox.stop(delete=True)

    tail = fake.subcommands()[-2:]
    assert tail == ["shutdown", "delete"]
    # The shutdown and delete both target the created UDID.
    assert ("shutdown", "UDID-9") in fake.calls
    assert ("delete", "UDID-9") in fake.calls
    # State cleared -> no device tracked -> no leak.
    assert sandbox.udid is None


async def test_stop_without_delete_keeps_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = _sandbox(tmp_path)
    fake = _FakeSimctl(udid="UDID-keep")
    monkeypatch.setattr(sandbox, "_simctl", fake)
    await sandbox.start()

    await sandbox.stop(delete=False)

    assert "shutdown" in fake.subcommands()
    assert "delete" not in fake.subcommands()


async def test_stop_is_idempotent_when_not_started(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    # No _simctl stub installed: stop() must not call simctl when there is no
    # device, so this would raise if it tried.
    await sandbox.stop(delete=True)
    assert sandbox.udid is None


async def test_install_app_uses_simctl_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = _sandbox(tmp_path)
    fake = _FakeSimctl(udid="UDID-app")
    monkeypatch.setattr(sandbox, "_simctl", fake)
    await sandbox.start()
    app_bundle = tmp_path / "Clock.app"
    app_bundle.mkdir()

    await sandbox.install_app(app_bundle)

    install = fake.calls[-1]
    assert install[0] == "install"
    assert install[1] == "UDID-app"
    assert install[2] == str(app_bundle)


async def test_install_app_missing_bundle_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = _sandbox(tmp_path)
    monkeypatch.setattr(sandbox, "_simctl", _FakeSimctl())
    await sandbox.start()
    with pytest.raises(FileNotFoundError):
        await sandbox.install_app(tmp_path / "missing.app")


async def test_add_media_uses_simctl_addmedia(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = _sandbox(tmp_path)
    fake = _FakeSimctl(udid="UDID-media")
    monkeypatch.setattr(sandbox, "_simctl", fake)
    await sandbox.start()
    photo = tmp_path / "photo.png"
    photo.write_bytes(b"\x89PNG")

    await sandbox.add_media(photo)

    call = fake.calls[-1]
    assert call[0] == "addmedia"
    assert call[1] == "UDID-media"
    assert call[2] == str(photo)


# --------------------------------------------------------------------------
# File transfer: simctl exposes no general in-guest copy, so transfer is over
# host-resolvable paths (e.g. the get_app_container host path); arbitrary
# in-guest paths raise a structured NotImplementedError rather than faking it.
# --------------------------------------------------------------------------


async def test_file_round_trip_over_host_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = _sandbox(tmp_path)
    monkeypatch.setattr(sandbox, "_simctl", _FakeSimctl())
    await sandbox.start()
    # Simulate a host-resolvable app data-container subtree.
    container = tmp_path / "container"
    container.mkdir()

    await sandbox.write_file(str(container / "state.json"), b'{"ok": true}')
    assert await sandbox.read_file(str(container / "state.json")) == b'{"ok": true}'

    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    await sandbox.upload_file(src, str(container / "up.bin"))
    assert (container / "up.bin").read_bytes() == b"payload"

    out = tmp_path / "down.bin"
    await sandbox.download_file(str(container / "up.bin"), out)
    assert out.read_bytes() == b"payload"


async def test_file_transfer_rejects_unmapped_in_guest_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = _sandbox(tmp_path)
    monkeypatch.setattr(sandbox, "_simctl", _FakeSimctl())
    await sandbox.start()
    # A path whose parent does not exist on the host is not host-resolvable.
    unmapped = "/var/mobile/Containers/Data/Application/NOPE/Documents/x.txt"
    with pytest.raises(NotImplementedError, match="get_app_container"):
        await sandbox.write_file(unmapped, b"x")
    with pytest.raises(NotImplementedError, match="not a host-resolvable path"):
        await sandbox.read_file(unmapped)


async def test_get_app_container_returns_host_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = _sandbox(tmp_path)

    class _ContainerSimctl(_FakeSimctl):
        async def __call__(
            self, *args: str, timeout_sec: float | None = None, check: bool = False
        ) -> ExecResult:
            if args and args[0] == "get_app_container":
                return ExecResult(stdout="/host/path/to/Data\n", return_code=0)
            return await super().__call__(*args, timeout_sec=timeout_sec, check=check)

    monkeypatch.setattr(sandbox, "_simctl", _ContainerSimctl(udid="UDID-c"))
    await sandbox.start()

    path = await sandbox.get_app_container("com.iosworld.clock", "data")
    assert path == "/host/path/to/Data"


# --------------------------------------------------------------------------
# Live dogfood: real simulator lifecycle, gated on an installed iOS runtime
# --------------------------------------------------------------------------


def _ios_runtime_present() -> bool:
    if platform.system() != "Darwin":
        return False
    try:
        completed = subprocess.run(
            ["xcrun", "simctl", "list", "runtimes"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and "ios" in completed.stdout.lower()


def _list_benchflow_ios_devices() -> list[str]:
    completed = subprocess.run(
        ["xcrun", "simctl", "list", "devices"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return [
        line.strip() for line in completed.stdout.splitlines() if _DEVICE_PREFIX in line
    ]


@pytest.mark.skipif(
    not _ios_runtime_present(),
    reason="no iOS Simulator runtime installed (live dogfood)",
)
async def test_live_simulator_lifecycle_no_leak(tmp_path: Path) -> None:
    """Real create -> boot -> exec -> shutdown -> delete with no leaked device."""
    before = set(_list_benchflow_ios_devices())
    sandbox = _sandbox(tmp_path)
    try:
        await sandbox.start()
        assert sandbox.udid is not None
        result = await sandbox.exec("echo benchflow-ios-live", timeout_sec=60)
        assert result.return_code == 0
        assert "benchflow-ios-live" in (result.stdout or "")
    finally:
        # Always tear down, even on assertion failure, so a failed run leaves
        # no device behind.
        await sandbox.stop(delete=True)

    after = set(_list_benchflow_ios_devices())
    leaked = after - before
    assert not leaked, f"leaked iOS Simulator device(s): {sorted(leaked)}"
