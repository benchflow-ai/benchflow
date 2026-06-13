"""macOS iOS Simulator sandbox provider.

This backend runs an agent against a real iOS Simulator device booted on a
macOS host via Apple's ``simctl`` (``xcrun simctl``). It is the substrate the
iOSWorld adapter targets: tasks drive 26 SwiftUI apps over Appium/XCUITest,
and BenchFlow owns the simulator-device lifecycle (create -> boot -> exec ->
shutdown -> delete) so no device leaks between rollouts.

Like :class:`~benchflow.sandbox.cua.CuaSandbox`, this is a non-Docker provider:
there is no image build, no compose topology, and no remote provider SDK. It
shells out to the host's ``xcrun simctl`` and uses the host's Appium/Xcode
toolchain that the adapter's capability probe detects.

Scope of this provider (the *infra*): device lifecycle, command exec via
``simctl spawn``, the file-transfer subset ``simctl`` actually supports, an
app-install helper for later iosworld-app bootstrap, and a host-capability
probe shared with the adapter. The agent-driven Appium loop and the build of
the iOSWorld SwiftUI apps are deliberately out of scope here.

File-transfer limitation (verified on the host): ``simctl spawn`` runs inside
the iOS *RuntimeRoot* filesystem, whose ``PATH`` has none of the usual Unix
tools (``cat``/``base64``/``mkdir``/``find``), so it is not a viable transfer
channel. simctl's real file primitives are host-resolvable paths — the
``get_app_container`` host path for app data and ``addmedia`` for media — so
file transfer operates on host-visible paths and raises a structured
``NotImplementedError`` for any in-guest path with no host mapping, rather than
pretending an unsupported copy succeeded.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchflow.sandbox._base import BaseSandbox
from benchflow.sandbox.protocol import ExecResult, SandboxStartupError
from benchflow.task.config import TaskOS

# The validated host configuration (iOS 26.3.1 runtime + iPhone 16 Pro device
# type on Xcode 26). Both are overridable so the provider tracks newer runtimes
# without a code change.
_DEFAULT_RUNTIME = "com.apple.CoreSimulator.SimRuntime.iOS-26-3"
_DEFAULT_DEVICE_TYPE = "com.apple.CoreSimulator.SimDeviceType.iPhone-16-Pro"

# BenchFlow owns every device it creates under this name prefix so post-run
# cleanup/audit can find (and never leak) them.
_DEVICE_NAME_PREFIX = "benchflow-ios-"

# simctl device names are free-form, but we keep ours to a tidy slug so they
# are greppable and shell-safe in ``simctl list devices`` output.
_NAME_INVALID = re.compile(r"[^a-z0-9-]+")

# The four host capabilities the iOSWorld adapter gates on. ``iosworld-app-
# bootstrap`` is intentionally *not* probed here — building/installing the 26
# SwiftUI apps is a follow-up milestone, so the adapter treats it as an
# always-pending capability rather than a host fact this provider can detect.
CAP_MACOS = "macos"
CAP_XCODE_26 = "xcode-26"
CAP_IOS_RUNTIME = "ios-26-simulator-runtime"
CAP_APPIUM_XCUITEST = "appium-xcuitest"


def runtime_id() -> str:
    """Return the iOS runtime id this provider boots (env-overridable)."""
    value = os.environ.get("BENCHFLOW_IOS_RUNTIME")
    return value.strip() if value and value.strip() else _DEFAULT_RUNTIME


def device_type_id() -> str:
    """Return the simulator device type to create (env-overridable)."""
    value = os.environ.get("BENCHFLOW_IOS_DEVICE_TYPE")
    return value.strip() if value and value.strip() else _DEFAULT_DEVICE_TYPE


def _name_part(value: str) -> str:
    slug = _NAME_INVALID.sub("-", value.strip().lower()).strip("-")
    return slug or "task"


def _run_host(
    program: str, *args: str, timeout: float | None = None
) -> tuple[int, str, str]:
    """Run a host command synchronously; return ``(returncode, stdout, stderr)``.

    Used by the capability probe and ``preflight`` — both run before any event
    loop exists, so they cannot use the async exec path.
    """
    import subprocess

    try:
        completed = subprocess.run(
            [program, *args],
            capture_output=True,
            text=True,
            timeout=timeout if timeout is not None else 30,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return (127, "", "")
    return (completed.returncode, completed.stdout, completed.stderr)


def _appium_available() -> bool:
    """Whether an ``appium`` CLI with the xcuitest driver is reachable.

    Honors ``BENCHFLOW_APPIUM_BIN`` for a pinned binary (the validated host
    installs it under an nvm path that is not always on ``PATH``).
    """
    appium_bin = os.environ.get("BENCHFLOW_APPIUM_BIN", "appium")
    code, out, err = _run_host(appium_bin, "driver", "list", "--installed")
    if code != 0:
        return False
    # ``appium driver list`` writes the human-readable table (with ANSI color
    # codes) to stderr, not stdout — check both streams for the driver name.
    return "xcuitest" in out.lower() or "xcuitest" in err.lower()


def detect_ios_simulator_capabilities() -> dict[str, bool]:
    """Probe the host for the iOS-Simulator capabilities the adapter gates on.

    Returns a capability -> present mapping. This is the single source of
    truth shared by :meth:`MacosIosSimulatorSandbox.preflight` and the
    iOSWorld adapter's support report, so the sandbox and the adapter never
    disagree about whether a host can run iOSWorld.

    Set ``BENCHFLOW_IOS_FORCE_UNSUPPORTED=1`` to force every capability to
    ``False`` regardless of host — used by tests (and operators) that want the
    provider-honest *unsupported* path on a host that happens to be capable.
    """
    if os.environ.get("BENCHFLOW_IOS_FORCE_UNSUPPORTED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return {
            CAP_MACOS: False,
            CAP_XCODE_26: False,
            CAP_IOS_RUNTIME: False,
            CAP_APPIUM_XCUITEST: False,
        }

    import platform

    is_macos = platform.system() == "Darwin"

    simctl_ok = is_macos and _run_host("xcrun", "simctl", "help")[0] == 0

    xcode_ok = False
    if is_macos:
        code, out, _err = _run_host("xcodebuild", "-version")
        if code == 0:
            match = re.search(r"Xcode\s+(\d+)", out)
            xcode_ok = bool(match and int(match.group(1)) >= 26)

    runtime_ok = False
    if simctl_ok:
        code, out, _err = _run_host("xcrun", "simctl", "list", "runtimes")
        if code == 0:
            runtime_ok = "ios" in out.lower()

    return {
        CAP_MACOS: is_macos,
        CAP_XCODE_26: xcode_ok,
        CAP_IOS_RUNTIME: runtime_ok,
        CAP_APPIUM_XCUITEST: _appium_available() if is_macos else False,
    }


class MacosIosSimulatorSandbox(BaseSandbox):
    """Sandbox backend that boots a real iOS Simulator device via ``simctl``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The created device's UDID — set on start(), cleared on stop(). When
        # None there is no device to operate on or to leak.
        self._udid: str | None = None
        self._device_name: str | None = None
        super().__init__(*args, **kwargs)

    @classmethod
    def preflight(cls) -> None:
        """Fail with an actionable error when the host cannot run iOSWorld."""
        import platform

        if platform.system() != "Darwin":
            raise SystemExit(
                "The macos-ios-simulator sandbox requires macOS (this host is "
                f"{platform.system()!r}). iOS Simulators only run on a Mac with "
                "Xcode installed."
            )
        if _run_host("xcrun", "simctl", "help")[0] != 0:
            raise SystemExit(
                "`xcrun simctl` is not available. Install Xcode and the command "
                "line tools (`xcode-select --install`), then accept the Xcode "
                "license (`sudo xcodebuild -license accept`)."
            )
        caps = detect_ios_simulator_capabilities()
        if not caps[CAP_IOS_RUNTIME]:
            raise SystemExit(
                "No iOS Simulator runtime is installed. Install one via Xcode "
                "(Settings -> Components) or `xcodebuild -downloadPlatform iOS`. "
                f"Expected runtime id {runtime_id()!r} (override with "
                "BENCHFLOW_IOS_RUNTIME)."
            )
        if not caps[CAP_APPIUM_XCUITEST]:
            raise SystemExit(
                "Appium with the xcuitest driver was not found. Install Appium "
                "(`npm i -g appium`) and the driver "
                "(`appium driver install xcuitest`), or point "
                "BENCHFLOW_APPIUM_BIN at the appium binary."
            )

    def _validate_definition(self) -> None:
        # The simulator runs on the macOS host; the task's declared OS is the
        # host OS. Accept MACOS (and the permissive default LINUX, which is the
        # SandboxConfig default for adapters that do not set os) rather than
        # forcing every iOSWorld task to carry a non-default os field.
        if self.task_env_config.os not in {TaskOS.MACOS, TaskOS.LINUX}:
            raise ValueError(
                "macos-ios-simulator sandbox runs on a macOS host; task OS "
                f"{self.task_env_config.os!r} is not supported."
            )

    @property
    def sandbox_id(self) -> str | None:
        """The booted device UDID once started (the provider-side identifier)."""
        return self._udid

    @property
    def udid(self) -> str | None:
        """The created simulator device UDID, or ``None`` before start()."""
        return self._udid

    def _create_name(self) -> str:
        return (
            f"{_DEVICE_NAME_PREFIX}{_name_part(self.environment_name)}-"
            f"{uuid4().hex[:8]}"
        )

    async def _simctl(
        self, *args: str, timeout_sec: float | None = None, check: bool = False
    ) -> ExecResult:
        """Run ``xcrun simctl <args...>`` and return the unified ExecResult."""
        process = await asyncio.create_subprocess_exec(
            "xcrun",
            "simctl",
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            if timeout_sec:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_sec
                )
            else:
                stdout_bytes, stderr_bytes = await process.communicate()
        except TimeoutError:
            process.kill()
            await process.communicate()
            raise TimeoutError(
                f"simctl {args[0] if args else ''} timed out after "
                f"{timeout_sec} seconds"
            ) from None

        result = ExecResult(
            stdout=stdout_bytes.decode(errors="replace") if stdout_bytes else None,
            stderr=stderr_bytes.decode(errors="replace") if stderr_bytes else None,
            return_code=process.returncode or 0,
        )
        if check and result.return_code != 0:
            raise SandboxStartupError(
                f"simctl {' '.join(args)} failed (rc={result.return_code}): "
                f"{(result.stderr or result.stdout or '').strip()[:500]}"
            )
        return result

    async def _wait_for_booted(self, *, timeout_sec: float, poll_sec: float) -> None:
        """Poll ``simctl list devices`` until our device reports ``Booted``."""
        deadline = asyncio.get_event_loop().time() + timeout_sec
        assert self._udid is not None
        while True:
            result = await self._simctl("list", "devices", timeout_sec=30)
            for line in (result.stdout or "").splitlines():
                if self._udid in line and "Booted" in line:
                    return
            if asyncio.get_event_loop().time() >= deadline:
                raise SandboxStartupError(
                    f"iOS Simulator {self._udid} did not reach 'Booted' within "
                    f"{timeout_sec}s",
                    sandbox_id=self._udid,
                    sandbox_state="boot-timeout",
                )
            await asyncio.sleep(poll_sec)

    async def start(self, force_build: bool = False) -> None:
        name = self._create_name()
        create = await self._simctl(
            "create", name, device_type_id(), runtime_id(), timeout_sec=120
        )
        if create.return_code != 0:
            raise SandboxStartupError(
                f"Failed to create iOS Simulator device {name!r} "
                f"(device_type={device_type_id()!r}, runtime={runtime_id()!r}): "
                f"{(create.stderr or create.stdout or '').strip()[:500]}"
            )
        udid = (create.stdout or "").strip()
        if not udid:
            raise SandboxStartupError(
                "simctl create returned no UDID for the new iOS Simulator device"
            )
        # Track the UDID *before* boot so a boot failure still cleans up the
        # created (but un-booted) device in stop() — no leak on partial start.
        self._udid = udid
        self._device_name = name

        boot = await self._simctl("boot", udid, timeout_sec=120)
        # `simctl boot` on an already-booted device exits non-zero; tolerate
        # that specific state and let the poll confirm readiness.
        if boot.return_code != 0 and "current state: Booted" not in (
            boot.stderr or boot.stdout or ""
        ):
            raise SandboxStartupError(
                f"Failed to boot iOS Simulator {udid}: "
                f"{(boot.stderr or boot.stdout or '').strip()[:500]}",
                sandbox_id=udid,
            )
        await self._wait_for_booted(timeout_sec=120, poll_sec=2.0)

    async def stop(self, delete: bool = True) -> None:
        udid = self._udid
        if udid is None:
            return
        # Clear state first so a failure mid-teardown does not strand the
        # instance pointing at a half-deleted device.
        self._udid = None
        self._device_name = None
        # Both steps are best-effort/idempotent: a missing or already-shutdown
        # device must not raise during cleanup.
        await self._simctl("shutdown", udid, timeout_sec=60)
        if delete:
            await self._simctl("delete", udid, timeout_sec=60)

    def _require_device(self) -> str:
        if self._udid is None:
            raise RuntimeError("iOS Simulator sandbox is not started")
        return self._udid

    @staticmethod
    def _reject_non_main(service: str) -> None:
        if service != "main":
            raise RuntimeError(
                "macos-ios-simulator sandbox has no compose topology; only "
                "service='main' is available."
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
        """Run a command inside the booted simulator via ``simctl spawn``.

        IMPORTANT — this is *not* a general-purpose host shell. ``simctl spawn``
        runs the process inside the simulator's iOS *RuntimeRoot* filesystem,
        whose ``PATH`` contains only the iOS runtime's own binaries. Common
        Unix tools (``cat``, ``ls``, ``mkdir``, ``base64``, ``find``) are *not*
        present there — only shell builtins (``echo``, ``printf``) and the
        runtime's binaries resolve. This method is kept as an honest thin
        wrapper for those cases and for diagnostics; it does **not** provide a
        Docker-style in-guest shell. App control for iOSWorld is done over
        Appium/XCUITest, not here, and file movement goes through
        :meth:`get_app_container` / :meth:`add_media` (see the file-transfer
        section below).

        ``cwd``/``env`` are folded into the ``/bin/sh -c`` command. ``user`` is
        ignored (simulator processes run as the host user; there is no in-guest
        user switch) and non-``main`` services are rejected.
        """
        self._reject_non_main(service)
        udid = self._require_device()
        merged_env = self._merge_env(env)
        shell_command = command
        if cwd:
            shell_command = f"cd {shlex.quote(cwd)} && {shell_command}"
        if merged_env:
            prefix = "".join(
                f"export {k}={shlex.quote(v)}; " for k, v in merged_env.items()
            )
            shell_command = f"{prefix}{shell_command}"
        return await self._simctl(
            "spawn",
            udid,
            "/bin/sh",
            "-c",
            shell_command,
            timeout_sec=timeout_sec or 60,
        )

    async def get_app_container(self, bundle_id: str, kind: str = "data") -> str:
        """Return a host path to an installed app's container (``simctl``).

        ``kind`` is one of simctl's container kinds (``app``, ``data``,
        ``groups``, ...). Raises if the app is not installed.
        """
        udid = self._require_device()
        result = await self._simctl(
            "get_app_container", udid, bundle_id, kind, timeout_sec=30
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"simctl get_app_container failed for {bundle_id!r} ({kind}): "
                f"{(result.stderr or result.stdout or '').strip()[:500]}"
            )
        return (result.stdout or "").strip()

    async def install_app(self, app_path: Path | str) -> None:
        """Install a built ``.app`` bundle into the simulator (``simctl install``).

        The mechanism the later iosworld-app-bootstrap step uses to load the 26
        SwiftUI apps. The bundle must be a simulator (not device) build.
        """
        udid = self._require_device()
        path = Path(app_path)
        if not path.exists():
            raise FileNotFoundError(f"App bundle does not exist: {path}")
        result = await self._simctl("install", udid, str(path), timeout_sec=300)
        if result.return_code != 0:
            raise RuntimeError(
                f"simctl install failed for {path}: "
                f"{(result.stderr or result.stdout or '').strip()[:500]}"
            )

    async def add_media(self, *media_paths: Path | str) -> None:
        """Add photos/videos to the simulator's photo library (``simctl addmedia``).

        The one ``simctl`` primitive for pushing user files (images, videos)
        into the device. For app data, write through the host path returned by
        :meth:`get_app_container` instead.
        """
        udid = self._require_device()
        paths = [str(Path(p)) for p in media_paths]
        for p in paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"Media path does not exist: {p}")
        result = await self._simctl("addmedia", udid, *paths, timeout_sec=120)
        if result.return_code != 0:
            raise RuntimeError(
                f"simctl addmedia failed: "
                f"{(result.stderr or result.stdout or '').strip()[:500]}"
            )

    # ---- file transfer ----------------------------------------------------
    #
    # DESIGN / LIMITATION (verified on the host, iOS 26.3 runtime):
    #
    # simctl has *no* general "push an arbitrary host file to an arbitrary
    # in-guest device path" primitive, and ``simctl spawn`` is not a usable
    # transfer channel: it runs inside the iOS RuntimeRoot, whose PATH lacks
    # ``cat``/``base64``/``mkdir``/``find`` — so a base64-over-shell copy (the
    # Docker/Cua trick) simply fails with "command not found".
    #
    # What simctl *does* expose is host-resolvable paths:
    #   * ``get_app_container <udid> <bundle> data`` returns a **host** path to
    #     the app's data container; the host can read/write it with normal file
    #     I/O. ``upload_file``/``download_file`` therefore operate on
    #     host-visible (typically container-derived) paths via ``shutil``.
    #   * ``addmedia`` (see :meth:`add_media`) ingests photos/videos.
    #
    # For any path that is *not* host-resolvable (an in-guest device path with
    # no host mapping), these methods raise a structured ``NotImplementedError``
    # naming the supported mechanisms rather than silently failing.

    @staticmethod
    def _require_host_path(path: str, *, op: str) -> Path:
        candidate = Path(path)
        if candidate.exists() or candidate.parent.exists():
            return candidate
        raise NotImplementedError(
            f"macos-ios-simulator {op}: {path!r} is not a host-resolvable path. "
            "simctl has no arbitrary in-guest file-transfer primitive and "
            "`simctl spawn` runs in the iOS RuntimeRoot (no cat/base64/mkdir). "
            "Use get_app_container() to obtain the app's host data-container "
            "path and read/write there, or add_media() for photos/videos."
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        import shutil

        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"upload_file source is not a file: {source}")
        dest = self._require_host_path(target_path, op="upload_file")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        import shutil

        src = self._require_host_path(source_path, op="download_file")
        if not src.is_file():
            raise FileNotFoundError(
                f"download_file source is not a host-visible file: {src}"
            )
        dest = Path(target_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        import shutil

        self._reject_non_main(service)
        source = Path(source_dir)
        if not source.is_dir():
            raise FileNotFoundError(f"upload_dir source is not a directory: {source}")
        dest = self._require_host_path(target_dir, op="upload_dir")
        shutil.copytree(source, dest, dirs_exist_ok=True)

    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        import shutil

        self._reject_non_main(service)
        src = self._require_host_path(source_dir, op="download_dir")
        if not src.is_dir():
            raise FileNotFoundError(
                f"download_dir source is not a host-visible directory: {src}"
            )
        shutil.copytree(src, Path(target_dir), dirs_exist_ok=True)

    async def read_file(self, path: str) -> bytes:
        src = self._require_host_path(path, op="read_file")
        if not src.is_file():
            raise FileNotFoundError(
                f"read_file source is not a host-visible file: {src}"
            )
        return src.read_bytes()

    async def write_file(self, path: str, content: bytes) -> None:
        dest = self._require_host_path(path, op="write_file")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
