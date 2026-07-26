"""Tests for the Apple Container sandbox backend.

Unit tests exercise argv and lifecycle contracts without requiring macOS. The
integration test at the bottom is gated on a real Apple Container installation.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.sandbox import apple_container as apple_mod
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.apple_container import (
    AppleContainerSandbox,
    _container_cli_version,
    _kalloc_headroom,
    _parse_version,
)
from benchflow.task.config import NetworkMode, SandboxConfig
from benchflow.task.paths import RolloutPaths


@pytest.fixture(autouse=True)
def isolated_run_slot(monkeypatch):
    """Give each test an unclaimed process-local lifecycle slot."""

    monkeypatch.setattr(apple_mod, "_RUN_SLOT", asyncio.Semaphore(1))


@pytest.fixture
def make_sandbox(tmp_path):
    sandboxes: list[AppleContainerSandbox] = []

    def make(
        *,
        image: str | None = None,
        skills_dir: str | None = None,
        allow_internet: bool = True,
        network_mode: NetworkMode = NetworkMode.PUBLIC,
        session_id: str = "sess-001",
    ) -> AppleContainerSandbox:
        index = len(sandboxes)
        env_dir = tmp_path / f"environment-{index}"
        env_dir.mkdir()
        (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\nRUN echo hi\n")
        rollout_dir = tmp_path / f"rollout-{index}"
        paths = RolloutPaths(rollout_dir)
        config = SandboxConfig(
            cpus=2,
            memory_mb=1024,
            docker_image=image,
            skills_dir=skills_dir,
            build_timeout_sec=60,
            allow_internet=allow_internet,
            network_mode=network_mode,
        )
        with patch.object(AppleContainerSandbox, "preflight"):
            sandbox = AppleContainerSandbox(
                environment_dir=env_dir,
                environment_name="test-task",
                session_id=session_id,
                rollout_paths=paths,
                task_env_config=config,
            )
        sandboxes.append(sandbox)
        return sandbox

    return make


def _success(stdout: str = "") -> ExecResult:
    return ExecResult(stdout=stdout, stderr=None, return_code=0)


def _started(make_sandbox) -> AppleContainerSandbox:
    sandbox = make_sandbox()
    sandbox._container_name = "bf_sess-001"
    return sandbox


class TestVersionAndPreflight:
    def test_parse_version_accepts_release_text(self):
        """Guards PR #936 against rejecting Apple Container release output."""

        assert _parse_version("container CLI version 1.1.0") == (1, 1, 0)
        assert _parse_version("invalid") is None

    @pytest.mark.parametrize(
        "payload",
        [
            {"appName": "container", "version": "1.1.0"},
            [
                {"appName": "container-apiserver", "version": "1.1.0"},
                {"appName": "container CLI", "version": "1.1.0"},
            ],
        ],
    )
    def test_cli_version_handles_old_and_current_json_shapes(self, payload):
        """Guards PR #936 across Apple Container version output shape changes."""

        completed = MagicMock(returncode=0, stdout=json.dumps(payload))
        with patch("subprocess.run", return_value=completed):
            assert _container_cli_version() == (1, 1, 0)

    def test_kalloc_parser_reads_live_cur_inuse_column(self):
        """Guards PR #936 against selecting cur-size instead of cur-inuse."""

        output = (
            "data.kalloc.1024 1024 0K 0K 0 0 1934 0K 0\n"
            "data.kalloc.2048 2048 0K 0K 0 0 99 0K 0\n"
        )
        completed = MagicMock(returncode=0, stdout=output)
        with patch("subprocess.run", return_value=completed) as run:
            current, headroom = _kalloc_headroom()
        assert current == 1934
        assert headroom == 3_000_000 - 1934
        run.assert_called_once_with(
            ["zprint", "-H", "-L", "data.kalloc.1024"],
            capture_output=True,
            text=True,
            timeout=5,
        )

    def test_kalloc_parser_fails_closed_on_unknown_shape(self):
        """Guards PR #936 against trusting an unknown zprint schema."""

        completed = MagicMock(returncode=0, stdout="unrecognized output\n")
        with patch("subprocess.run", return_value=completed):
            assert _kalloc_headroom() == (-1, -1)

    def test_preflight_rejects_non_darwin(self, monkeypatch):
        """Guards PR #936 against selecting Apple Container off macOS."""

        monkeypatch.setattr(apple_mod.sys, "platform", "linux")
        with pytest.raises(RuntimeError, match="requires macOS"):
            AppleContainerSandbox.preflight()

    def test_preflight_rejects_non_apple_silicon(self, monkeypatch):
        """Guards PR #936 against selecting Apple Container on Intel Macs."""

        monkeypatch.setattr(apple_mod.sys, "platform", "darwin")
        monkeypatch.setattr(apple_mod.platform, "machine", lambda: "x86_64")
        with pytest.raises(RuntimeError, match="requires Apple silicon"):
            AppleContainerSandbox.preflight()

    def test_preflight_rejects_old_cli(self, monkeypatch):
        """Guards PR #936 against using unsupported native CLI behavior."""

        monkeypatch.setattr(apple_mod.sys, "platform", "darwin")
        monkeypatch.setattr(apple_mod.platform, "machine", lambda: "arm64")
        monkeypatch.setattr(apple_mod.shutil, "which", lambda _name: "/bin/container")
        monkeypatch.setattr(apple_mod, "_container_cli_version", lambda: (1, 0, 0))
        with pytest.raises(RuntimeError, match=r"1\.1\.0\+ is required"):
            AppleContainerSandbox.preflight()

    def test_preflight_fails_closed_when_kalloc_is_unreadable(self, monkeypatch):
        """Guards PR #936 against launching when VM leak headroom is unknown."""

        monkeypatch.setattr(apple_mod.sys, "platform", "darwin")
        monkeypatch.setattr(apple_mod.platform, "machine", lambda: "arm64")
        monkeypatch.setattr(apple_mod.shutil, "which", lambda _name: "/bin/container")
        monkeypatch.setattr(apple_mod, "_container_cli_version", lambda: (1, 1, 0))
        monkeypatch.setattr(
            apple_mod.subprocess,
            "run",
            lambda *_args, **_kwargs: MagicMock(returncode=0, stderr=""),
        )
        monkeypatch.setattr(apple_mod, "_kalloc_headroom", lambda: (-1, -1))
        with pytest.raises(RuntimeError, match="cannot be verified"):
            AppleContainerSandbox.preflight()

    def test_preflight_rejects_low_kalloc_headroom(self, monkeypatch):
        """Guards PR #936 against crossing Apple's documented crash region."""

        monkeypatch.setattr(apple_mod.sys, "platform", "darwin")
        monkeypatch.setattr(apple_mod.platform, "machine", lambda: "arm64")
        monkeypatch.setattr(apple_mod.shutil, "which", lambda _name: "/bin/container")
        monkeypatch.setattr(apple_mod, "_container_cli_version", lambda: (1, 1, 0))
        monkeypatch.setattr(
            apple_mod.subprocess,
            "run",
            lambda *_args, **_kwargs: MagicMock(returncode=0, stderr=""),
        )
        monkeypatch.setattr(apple_mod, "_kalloc_headroom", lambda: (2_850_000, 150_000))
        with pytest.raises(RuntimeError, match="Reboot your Mac"):
            AppleContainerSandbox.preflight()


class TestDefinitionAndBuild:
    def test_rejects_missing_dockerfile_and_image(self, make_sandbox, tmp_path):
        """Guards PR #936 against launching without an image definition."""

        sandbox = make_sandbox(image="ubuntu:24.04")
        sandbox.environment_dir.joinpath("Dockerfile").unlink()
        sandbox.task_env_config.docker_image = None
        with pytest.raises(ValueError, match="No Dockerfile"):
            sandbox._validate_definition()

    def test_rejects_no_network(self, make_sandbox):
        """Guards PR #936 against claiming unenforced network isolation."""

        with pytest.raises(ValueError, match="does not currently enforce no-network"):
            make_sandbox(
                allow_internet=False,
                network_mode=NetworkMode.NO_NETWORK,
            )

    @pytest.mark.asyncio
    async def test_prebuilt_image_wins_without_force_build(self, make_sandbox):
        """Guards PR #936 against rebuilding an explicitly configured image."""

        sandbox = make_sandbox(image="ubuntu:24.04")
        with patch.object(apple_mod, "_run_cli", new_callable=AsyncMock) as run:
            assert await sandbox._resolve_image(force_build=False) == "ubuntu:24.04"
        run.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("force_build", [False, True])
    async def test_build_is_native_arm64_and_cache_policy_is_explicit(
        self, make_sandbox, force_build
    ):
        """Guards PR #936 against forced cache misses and amd64 VM failures."""

        sandbox = make_sandbox()
        with patch.object(
            apple_mod, "_run_cli", new_callable=AsyncMock, return_value=_success()
        ) as run:
            image = await sandbox._resolve_image(force_build=force_build)
        assert image == "bf__test-task"
        call = run.await_args
        assert call is not None
        args = call.args
        assert args[0] == "build"
        assert args[args.index("--platform") + 1] == "linux/arm64"
        assert ("--no-cache" in args) is force_build


class TestStartAndLifecycle:
    @pytest.mark.asyncio
    async def test_start_uses_detach_and_only_mounts_logs(self, make_sandbox):
        """Guards PR #936 against task-source and skills host mutation."""

        sandbox = make_sandbox(skills_dir="/skills")
        with (
            patch.object(apple_mod, "_require_kalloc_headroom"),
            patch.object(
                sandbox,
                "_resolve_image",
                new_callable=AsyncMock,
                return_value="ubuntu:24.04",
            ),
            patch.object(
                apple_mod, "_run_cli", new_callable=AsyncMock, return_value=_success()
            ) as run,
        ):
            await sandbox.start(force_build=False)

        launch = run.await_args_list[0].args
        assert launch[:2] == ("run", "--detach")
        joined = "\n".join(launch)
        assert "target=/logs" in joined
        assert "target=/app" not in joined
        assert "/skills" not in joined
        assert "--platform\nlinux/arm64" in joined

    @pytest.mark.asyncio
    async def test_start_does_not_put_provider_secrets_on_run_argv(self, make_sandbox):
        """Guards PR #936 against exposing model credentials in host process argv."""

        sandbox = make_sandbox()
        sandbox._persistent_env = {"API_KEY": "sk-secret-123"}
        with (
            patch.object(apple_mod, "_require_kalloc_headroom"),
            patch.object(
                sandbox,
                "_resolve_image",
                new_callable=AsyncMock,
                return_value="ubuntu:24.04",
            ),
            patch.object(
                apple_mod, "_run_cli", new_callable=AsyncMock, return_value=_success()
            ) as run,
        ):
            await sandbox.start(force_build=False)
        launch = run.await_args_list[0].args
        assert "sk-secret-123" not in "\n".join(launch)
        assert "-e" not in launch

    @pytest.mark.asyncio
    async def test_start_failure_releases_process_slot(self, make_sandbox):
        """Guards PR #936 against permanently wedging later Apple rollouts."""

        sandbox = make_sandbox()
        with (
            patch.object(apple_mod, "_require_kalloc_headroom"),
            patch.object(
                sandbox,
                "_resolve_image",
                new_callable=AsyncMock,
                return_value="ubuntu:24.04",
            ),
            patch.object(
                apple_mod,
                "_run_cli",
                new_callable=AsyncMock,
                return_value=ExecResult(
                    stdout=None, stderr="launch failed", return_code=1
                ),
            ),
            pytest.raises(RuntimeError, match="launch failed"),
        ):
            await sandbox.start(force_build=False)
        assert not apple_mod._RUN_SLOT.locked()
        assert sandbox.sandbox_id is None

    @pytest.mark.asyncio
    async def test_only_one_sandbox_is_active_per_process(self, make_sandbox):
        """Guards PR #936 by making the kalloc safety limit enforceable."""

        first = make_sandbox(session_id="first")
        second = make_sandbox(session_id="second")
        with (
            patch.object(apple_mod, "_require_kalloc_headroom"),
            patch.object(
                AppleContainerSandbox,
                "_resolve_image",
                new_callable=AsyncMock,
                return_value="ubuntu:24.04",
            ),
            patch.object(
                apple_mod, "_run_cli", new_callable=AsyncMock, return_value=_success()
            ),
        ):
            await first.start(force_build=False)
            second_start = asyncio.create_task(second.start(force_build=False))
            await asyncio.sleep(0)
            assert not second_start.done()
            await first.stop(delete=True)
            await asyncio.wait_for(second_start, timeout=1)
            await second.stop(delete=True)
        assert not apple_mod._RUN_SLOT.locked()

    @pytest.mark.asyncio
    async def test_stop_uses_native_stop_and_force_remove(self, make_sandbox):
        """Guards PR #936 against leaking VMs or stopping global BuildKit."""

        sandbox = _started(make_sandbox)
        sandbox._holds_run_slot = True
        await apple_mod._RUN_SLOT.acquire()
        with patch.object(
            apple_mod, "_run_cli", new_callable=AsyncMock, return_value=_success()
        ) as run:
            await sandbox.stop(delete=True)
        calls = [call.args for call in run.await_args_list]
        assert ("stop", "--time", "5", "bf_sess-001") in calls
        assert ("rm", "--force", "bf_sess-001") in calls
        assert all("buildkit" not in call for call in calls)
        assert sandbox.sandbox_id is None
        assert not apple_mod._RUN_SLOT.locked()


class TestExec:
    @pytest.mark.asyncio
    async def test_exec_uses_native_workdir_and_user_flags(self, make_sandbox):
        """Guards PR #936 against shell-injected cwd/user and missing native flags."""

        sandbox = _started(make_sandbox)
        hostile_user = "worker; touch /tmp/pwned"
        hostile_cwd = "/app/a path; touch /tmp/pwned"
        with patch.object(
            apple_mod, "_run_cli", new_callable=AsyncMock, return_value=_success()
        ) as run:
            await sandbox.exec("printf ok", cwd=hostile_cwd, user=hostile_user)
        run.assert_awaited_once_with(
            "exec",
            "--workdir",
            hostile_cwd,
            "--user",
            hostile_user,
            "bf_sess-001",
            "sh",
            "-c",
            "printf ok",
            timeout=None,
        )

    @pytest.mark.asyncio
    async def test_exec_preserves_numeric_user(self, make_sandbox):
        """Guards PR #936 against dropping valid numeric user identities."""

        sandbox = _started(make_sandbox)
        with patch.object(
            apple_mod, "_run_cli", new_callable=AsyncMock, return_value=_success()
        ) as run:
            await sandbox.exec("id", user=1000)
        call = run.await_args
        assert call is not None
        assert call.args[1:3] == ("--user", "1000")

    @pytest.mark.asyncio
    async def test_exec_redacts_environment_secret(self, make_sandbox):
        """Guards PR #936 against exposing exec secrets in host argv."""

        sandbox = _started(make_sandbox)
        with patch.object(
            apple_mod, "_run_cli", new_callable=AsyncMock, return_value=_success()
        ) as run:
            await sandbox.exec("run.sh", env={"API_KEY": "sk-secret-123"})
        call = run.await_args
        assert call is not None
        argv = call.args
        assert "sk-secret-123" not in "\n".join(argv)
        assert "base64 -d" in argv[-1]

    @pytest.mark.asyncio
    async def test_exec_rejects_non_main_service(self, make_sandbox):
        sandbox = _started(make_sandbox)
        with pytest.raises(ValueError, match="single-container"):
            await sandbox.exec("true", service="target")

    @pytest.mark.asyncio
    async def test_exec_timeout_forces_cleanup(self, make_sandbox):
        """Guards PR #936 against retaining a VM after an exec timeout."""

        sandbox = _started(make_sandbox)
        with (
            patch.object(
                apple_mod, "_run_cli", new_callable=AsyncMock, side_effect=TimeoutError
            ),
            patch.object(sandbox, "_force_cleanup", new_callable=AsyncMock) as cleanup,
            pytest.raises(RuntimeError, match="timed out"),
        ):
            await sandbox.exec("sleep 999", timeout_sec=5)
        cleanup.assert_awaited_once()


class TestFileTransfer:
    @pytest.mark.asyncio
    async def test_mounted_upload_is_host_copy(self, make_sandbox, tmp_path):
        """Guards PR #936 by retaining the established /logs fast path."""

        sandbox = _started(make_sandbox)
        source = tmp_path / "source.txt"
        source.write_text("data")
        with patch.object(apple_mod.shutil, "copy2") as copy:
            await sandbox.upload_file(source, "/logs/verifier/out.txt")
        assert sandbox.rollout_paths is not None
        copy.assert_called_once_with(
            source, sandbox.rollout_paths.rollout_dir / "verifier" / "out.txt"
        )

    @pytest.mark.asyncio
    async def test_unmounted_upload_uses_native_copy(self, make_sandbox, tmp_path):
        """Guards PR #936 against base64 file transfer and host task mutation."""

        sandbox = _started(make_sandbox)
        source = tmp_path / "source.txt"
        source.write_text("data")
        with patch.object(
            apple_mod, "_run_cli", new_callable=AsyncMock, return_value=_success()
        ) as run:
            await sandbox.upload_file(source, "/app/out.txt")
        calls = [call.args for call in run.await_args_list]
        assert calls[-1] == ("cp", str(source), "bf_sess-001:/app/out.txt")
        assert all("base64" not in "\n".join(call) for call in calls)

    @pytest.mark.asyncio
    async def test_upload_dir_uses_native_copy(self, make_sandbox, tmp_path):
        """Guards PR #936 against nesting the host directory under its target."""

        sandbox = _started(make_sandbox)
        source = tmp_path / "source"
        source.mkdir()
        source.joinpath("file.txt").write_text("data")
        source.joinpath("nested").mkdir()
        source.joinpath("nested", "child.txt").write_text("nested data")
        with patch.object(
            apple_mod, "_run_cli", new_callable=AsyncMock, return_value=_success()
        ) as run:
            await sandbox.upload_dir(source, "/skills")
        copy_calls = [call.args for call in run.await_args_list if call.args[0] == "cp"]
        assert copy_calls == [
            ("cp", str(source / "file.txt"), "bf_sess-001:/skills/"),
            ("cp", str(source / "nested"), "bf_sess-001:/skills/"),
        ]

    @pytest.mark.asyncio
    async def test_unmounted_download_uses_native_copy(self, make_sandbox, tmp_path):
        """Guards PR #936 against shell-encoded downloads from the VM."""

        sandbox = _started(make_sandbox)
        target = tmp_path / "out.bin"
        with patch.object(
            apple_mod, "_run_cli", new_callable=AsyncMock, return_value=_success()
        ) as run:
            await sandbox.download_file("/opt/data.bin", target)
        run.assert_awaited_once_with(
            "cp",
            "bf_sess-001:/opt/data.bin",
            str(target),
            timeout=120,
        )

    def test_mounted_path_rejects_parent_traversal(self, make_sandbox):
        """Guards PR #936 against escaping the rollout directory through /logs."""

        sandbox = _started(make_sandbox)
        with pytest.raises(ValueError, match="Unsafe mounted path"):
            sandbox._mounted_host_path("/logs/../../outside")


class TestProperties:
    def test_backend_properties(self, make_sandbox):
        """Guards PR #936 by preserving the common sandbox capability contract."""

        sandbox = _started(make_sandbox)
        assert sandbox.is_mounted is True
        assert sandbox.sandbox_id == "bf_sess-001"
        assert sandbox.supports_snapshot is False


@pytest.mark.skipif(
    sys.platform != "darwin" or not shutil.which("container"),
    reason="Requires macOS with Apple Container 1.1+",
)
@pytest.mark.asyncio
async def test_real_apple_container_lifecycle_and_copy(tmp_path):
    """Guards PR #936 with a real detached lifecycle, exec, and native copy."""

    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    environment_dir.joinpath("Dockerfile").write_text("FROM ubuntu:24.04\n")
    rollout_dir = tmp_path / "rollout"
    paths = RolloutPaths(rollout_dir)
    config = SandboxConfig(
        cpus=1,
        memory_mb=512,
        docker_image="ubuntu:24.04",
        build_timeout_sec=120,
        allow_internet=True,
        network_mode=NetworkMode.PUBLIC,
    )

    AppleContainerSandbox.preflight()
    sandbox = AppleContainerSandbox(
        environment_dir=environment_dir,
        environment_name="integration-test",
        session_id="integration-936",
        rollout_paths=paths,
        task_env_config=config,
    )
    try:
        await sandbox.start(force_build=False)
        result = await sandbox.exec("printf hello", timeout_sec=10)
        assert result == ExecResult(stdout="hello", stderr=None, return_code=0)

        source = tmp_path / "upload.txt"
        source.write_text("uploaded content")
        await sandbox.upload_file(source, "/tmp/upload.txt")
        downloaded = tmp_path / "downloaded.txt"
        await sandbox.download_file("/tmp/upload.txt", downloaded)
        assert downloaded.read_text() == "uploaded content"

        source_dir = tmp_path / "upload-dir"
        source_dir.mkdir()
        source_dir.joinpath("nested").mkdir()
        source_dir.joinpath("nested", "child.txt").write_text("nested content")
        await sandbox.upload_dir(source_dir, "/tmp/dir-target")
        result = await sandbox.exec("cat /tmp/dir-target/nested/child.txt")
        assert result.stdout == "nested content"
    finally:
        await sandbox.stop(delete=True)
