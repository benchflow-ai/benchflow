"""Unit tests for the Apple Container sandbox backend.

All tests mock asyncio.create_subprocess_exec so they run on any platform
(no macOS or container CLI required). Integration tests requiring a real
container runtime are gated at the bottom.
"""

from __future__ import annotations

import asyncio
import base64
import shutil
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.apple_container import (
    AppleContainerSandbox,
    _dockerfile_is_arm64_clean,
    _kalloc_headroom,
)

# --- Fixtures ---


@pytest.fixture
def mock_rollout_paths(tmp_path):
    paths = MagicMock()
    paths.rollout_dir = tmp_path / "rollout"
    paths.verifier_dir = tmp_path / "rollout" / "verifier"
    paths.agent_dir = tmp_path / "rollout" / "agent"
    paths.artifacts_dir = tmp_path / "rollout" / "artifacts"
    return paths


@pytest.fixture
def mock_env_config():
    config = MagicMock()
    config.cpus = 2
    config.memory_mb = 1024
    config.docker_image = None
    config.env = None
    config.skills_dir = None
    config.build_timeout_sec = 60
    config.allow_internet = True
    config.network_mode = "public"
    config.storage_mb = None
    config.gpus = None
    return config


@pytest.fixture
def sandbox(tmp_path, mock_rollout_paths, mock_env_config):
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\nRUN echo hi\n")
    with patch.object(AppleContainerSandbox, "preflight"):
        sb = AppleContainerSandbox(
            environment_dir=env_dir,
            environment_name="test-task",
            session_id="sess-001",
            rollout_paths=mock_rollout_paths,
            task_env_config=mock_env_config,
        )
    sb._container_name = "bf_sess-001"
    return sb


def _mock_proc(returncode=0, stdout=b"", stderr=b""):
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.kill = MagicMock()
    proc.terminate = MagicMock()
    proc.wait = AsyncMock(return_value=returncode)
    return proc


# --- kalloc parsing ---


class TestKallocHeadroom:
    def test_parses_zprint_output(self):
        zprint_output = (
            "Zone                          cur      alloc       size     elems  fl  maxelts\n"
            "data.kalloc.1024          5000000    5000000       1024   5000000   0  8000000\n"
            "data.kalloc.2048          1000000    1000000       2048   1000000   0  4000000\n"
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=zprint_output)
            elts, headroom = _kalloc_headroom()
        assert elts == 5000000
        assert headroom == 3000000

    def test_healthy_headroom(self):
        zprint_output = "data.kalloc.1024          3000000    3000000       1024   3000000   0  3000000\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=zprint_output)
            elts, headroom = _kalloc_headroom()
        assert elts == 3000000
        assert headroom == 5000000

    def test_zprint_failure_returns_negative(self):
        with patch("subprocess.run", side_effect=OSError("no zprint")):
            elts, headroom = _kalloc_headroom()
        assert elts == -1
        assert headroom == -1


# --- Dockerfile arm64 validation ---


class TestDockerfileArm64:
    def test_clean_dockerfile(self, tmp_path):
        df = tmp_path / "Dockerfile"
        df.write_text("FROM ubuntu:24.04\nRUN apt-get install -y curl\n")
        assert _dockerfile_is_arm64_clean(df) is True

    def test_rejects_platform_amd64(self, tmp_path):
        df = tmp_path / "Dockerfile"
        df.write_text("FROM --platform=linux/amd64 ubuntu:24.04\n")
        assert _dockerfile_is_arm64_clean(df) is False

    def test_rejects_x86_64_reference(self, tmp_path):
        df = tmp_path / "Dockerfile"
        df.write_text("FROM ubuntu:24.04\nRUN dpkg --add-architecture x86_64\n")
        assert _dockerfile_is_arm64_clean(df) is False

    def test_missing_file_is_clean(self, tmp_path):
        assert _dockerfile_is_arm64_clean(tmp_path / "nonexistent") is True


# --- Preflight ---


class TestPreflight:
    def test_rejects_non_darwin(self):
        with patch("benchflow.sandbox.apple_container.sys") as mock_sys:
            mock_sys.platform = "linux"
            with pytest.raises(RuntimeError, match="requires macOS"):
                AppleContainerSandbox.preflight()

    def test_rejects_missing_container_cli(self):
        with (
            patch("benchflow.sandbox.apple_container.sys") as mock_sys,
            patch("benchflow.sandbox.apple_container.shutil.which", return_value=None),
        ):
            mock_sys.platform = "darwin"
            with pytest.raises(RuntimeError, match="container CLI not found"):
                AppleContainerSandbox.preflight()

    def test_rejects_exhausted_kalloc(self):
        with (
            patch("benchflow.sandbox.apple_container.sys") as mock_sys,
            patch(
                "benchflow.sandbox.apple_container.shutil.which",
                return_value="/usr/bin/container",
            ),
            patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="")),
            patch(
                "benchflow.sandbox.apple_container._kalloc_headroom",
                return_value=(7_900_000, 100_000),
            ),
            patch("benchflow.sandbox.apple_container._disk_free_gb", return_value=50.0),
        ):
            mock_sys.platform = "darwin"
            with pytest.raises(RuntimeError, match="kalloc zone nearly exhausted"):
                AppleContainerSandbox.preflight()


# --- Validation ---


class TestValidateDefinition:
    def test_rejects_amd64_dockerfile(
        self, tmp_path, mock_rollout_paths, mock_env_config
    ):
        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "Dockerfile").write_text(
            "FROM --platform=linux/amd64 ubuntu:24.04\n"
        )
        with (
            patch.object(AppleContainerSandbox, "preflight"),
            pytest.raises(ValueError, match="arm64"),
        ):
            AppleContainerSandbox(
                environment_dir=env_dir,
                environment_name="bad-task",
                session_id="s1",
                rollout_paths=mock_rollout_paths,
                task_env_config=mock_env_config,
            )

    def test_rejects_no_dockerfile_no_image(
        self, tmp_path, mock_rollout_paths, mock_env_config
    ):
        env_dir = tmp_path / "env"
        env_dir.mkdir()
        mock_env_config.docker_image = None
        with (
            patch.object(AppleContainerSandbox, "preflight"),
            pytest.raises(ValueError, match="No Dockerfile"),
        ):
            AppleContainerSandbox(
                environment_dir=env_dir,
                environment_name="empty-task",
                session_id="s1",
                rollout_paths=mock_rollout_paths,
                task_env_config=mock_env_config,
            )

    def test_rejects_no_network_config(
        self, tmp_path, mock_rollout_paths, mock_env_config
    ):
        """Guards PR #936 against silently running no-network tasks with public egress."""
        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        mock_env_config.allow_internet = False
        mock_env_config.network_mode = "no-network"
        with (
            patch.object(AppleContainerSandbox, "preflight"),
            pytest.raises(ValueError, match="does not currently enforce no-network"),
        ):
            AppleContainerSandbox(
                environment_dir=env_dir,
                environment_name="no-network-task",
                session_id="s1",
                rollout_paths=mock_rollout_paths,
                task_env_config=mock_env_config,
            )


# --- Start/lifecycle contract ---


class TestStart:
    @pytest.mark.asyncio
    async def test_start_mounts_logs_but_not_app(self, sandbox):
        """Guards PR #936 against mounting task sources over /app."""
        with (
            patch.object(sandbox, "_build_image", new_callable=AsyncMock) as mock_build,
            patch(
                "benchflow.sandbox.apple_container._kalloc_headroom",
                return_value=(3_000_000, 5_000_000),
            ),
            patch("benchflow.sandbox.apple_container._run_cli") as mock_cli,
            patch(
                "benchflow.sandbox.apple_container.asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
            ) as mock_subproc,
        ):
            mock_build.return_value = "ubuntu:24.04"
            mock_cli.return_value = ExecResult(stdout="", stderr=None, return_code=0)
            mock_subproc.return_value = _mock_proc()

            await sandbox.start(force_build=False)

        launched_args = mock_subproc.call_args.args
        assert launched_args[:2] == ("container", "run")
        joined = "\n".join(str(arg) for arg in launched_args)
        assert "target=/logs" in joined
        assert "target=/app" not in joined


# --- Exec argv construction ---


class TestExec:
    @pytest.mark.asyncio
    async def test_basic_exec_argv(self, sandbox):
        with patch("benchflow.sandbox.apple_container._run_cli") as mock_cli:
            mock_cli.return_value = ExecResult(
                stdout="ok\n", stderr=None, return_code=0
            )
            result = await sandbox.exec("echo hello")
        mock_cli.assert_called_once_with(
            "exec", "bf_sess-001", "sh", "-c", "echo hello", timeout=None
        )
        assert result.stdout == "ok\n"
        assert result.return_code == 0

    @pytest.mark.asyncio
    async def test_exec_with_cwd(self, sandbox):
        with patch("benchflow.sandbox.apple_container._run_cli") as mock_cli:
            mock_cli.return_value = ExecResult(
                stdout="/app\n", stderr=None, return_code=0
            )
            await sandbox.exec("pwd", cwd="/app")
        args = mock_cli.call_args[0]
        assert "cd /app && pwd" in args[4]

    @pytest.mark.asyncio
    async def test_exec_with_user(self, sandbox):
        with patch("benchflow.sandbox.apple_container._run_cli") as mock_cli:
            mock_cli.return_value = ExecResult(
                stdout="uid=1000\n", stderr=None, return_code=0
            )
            await sandbox.exec("id", user="testuser")
        args = mock_cli.call_args[0]
        assert "su testuser -s /bin/sh -c" in args[4]

    @pytest.mark.asyncio
    async def test_exec_with_env_redacts_secrets(self, sandbox):
        with patch("benchflow.sandbox.apple_container._run_cli") as mock_cli:
            mock_cli.return_value = ExecResult(stdout="", stderr=None, return_code=0)
            await sandbox.exec("run.sh", env={"API_KEY": "sk-secret-123"})
        args = mock_cli.call_args[0]
        cmd = args[4]
        assert "sk-secret-123" not in cmd
        assert "base64 -d" in cmd

    @pytest.mark.asyncio
    async def test_exec_rejects_non_main_service(self, sandbox):
        with pytest.raises(ValueError, match="single-container"):
            await sandbox.exec("echo hi", service="target")

    @pytest.mark.asyncio
    async def test_exec_timeout_triggers_cleanup(self, sandbox):
        with (
            patch(
                "benchflow.sandbox.apple_container._run_cli",
                side_effect=asyncio.TimeoutError,
            ),
            patch.object(
                sandbox, "_force_cleanup", new_callable=AsyncMock
            ) as mock_cleanup,
            pytest.raises(RuntimeError, match="timed out"),
        ):
            await sandbox.exec("sleep 999", timeout_sec=5)
        mock_cleanup.assert_called_once()


# --- Upload/Download routing ---


class TestFileTransfer:
    @pytest.mark.asyncio
    async def test_upload_to_mounted_path_is_host_copy(self, sandbox, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("data")
        with patch("shutil.copy2") as mock_copy:
            await sandbox.upload_file(src, "/logs/verifier/out.txt")
        mock_copy.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_to_app_uses_exec_not_host_copy(self, sandbox, tmp_path):
        """Guards PR #936 against writing agent files back into the task source tree."""
        src = tmp_path / "src.txt"
        src.write_text("data")
        with (
            patch("shutil.copy2") as mock_copy,
            patch("benchflow.sandbox.apple_container._run_cli") as mock_cli,
        ):
            mock_cli.return_value = ExecResult(stdout="", stderr=None, return_code=0)
            await sandbox.upload_file(src, "/app/out.txt")

        mock_copy.assert_not_called()
        args = mock_cli.call_args[0]
        assert args[:3] == ("exec", "-i", "bf_sess-001")

    @pytest.mark.asyncio
    async def test_upload_to_unmounted_uses_exec_i(self, sandbox, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("hello")
        with patch("benchflow.sandbox.apple_container._run_cli") as mock_cli:
            mock_cli.return_value = ExecResult(stdout="", stderr=None, return_code=0)
            await sandbox.upload_file(src, "/opt/data.txt")
        args = mock_cli.call_args[0]
        assert args[0] == "exec"
        assert args[1] == "-i"
        assert "base64 -d" in args[5]
        assert mock_cli.call_args[1]["stdin_data"] is not None

    @pytest.mark.asyncio
    async def test_download_from_mounted_path_is_host_copy(self, sandbox, tmp_path):
        host_file = sandbox.rollout_paths.verifier_dir / "reward.txt"
        host_file.parent.mkdir(parents=True, exist_ok=True)
        host_file.write_text("1.0")
        target = tmp_path / "downloaded.txt"
        with patch("shutil.copy2") as mock_copy:
            await sandbox.download_file("/logs/verifier/reward.txt", target)
        mock_copy.assert_called_once()

    def test_mounted_path_rejects_parent_traversal(self, sandbox):
        """Guards PR #936 against /logs host-path escape through .. segments."""
        with pytest.raises(ValueError, match="Unsafe mounted path escapes /logs"):
            sandbox._mounted_host_path("/logs/../../outside")

    @pytest.mark.asyncio
    async def test_download_from_unmounted_uses_base64(self, sandbox, tmp_path):
        content = b"binary content"
        encoded = base64.b64encode(content).decode()
        target = tmp_path / "out.bin"
        with patch("benchflow.sandbox.apple_container._run_cli") as mock_cli:
            mock_cli.return_value = ExecResult(
                stdout=encoded, stderr=None, return_code=0
            )
            await sandbox.download_file("/opt/data.bin", target)
        args = mock_cli.call_args[0]
        assert args[0] == "exec"
        assert args[1] == "bf_sess-001"
        assert args[2] == "base64"
        assert target.read_bytes() == content


# --- Stop ---


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_calls_container_stop_and_rm(self, sandbox):
        sandbox._bg_proc = _mock_proc()
        with patch("benchflow.sandbox.apple_container._run_cli") as mock_cli:
            mock_cli.return_value = ExecResult(stdout="", stderr=None, return_code=0)
            await sandbox.stop(delete=True)
        calls = [c[0] for c in mock_cli.call_args_list]
        assert ("stop", "bf_sess-001") in calls
        assert ("rm", "bf_sess-001") in calls
        assert sandbox._container_name is None

    @pytest.mark.asyncio
    async def test_stop_without_delete_keeps_name(self, sandbox):
        sandbox._bg_proc = _mock_proc()
        with patch("benchflow.sandbox.apple_container._run_cli") as mock_cli:
            mock_cli.return_value = ExecResult(stdout="", stderr=None, return_code=0)
            await sandbox.stop(delete=False)
        calls = [c[0] for c in mock_cli.call_args_list]
        assert ("stop", "bf_sess-001") in calls
        assert ("rm", "bf_sess-001") not in calls
        assert sandbox._container_name == "bf_sess-001"


# --- Properties ---


class TestProperties:
    def test_is_mounted(self, sandbox):
        assert sandbox.is_mounted is True

    def test_sandbox_id(self, sandbox):
        assert sandbox.sandbox_id == "bf_sess-001"

    def test_supports_snapshot_false(self, sandbox):
        assert sandbox.supports_snapshot is False


# --- Integration test (macOS only) ---


@pytest.mark.skipif(
    sys.platform != "darwin" or not shutil.which("container"),
    reason="Requires macOS with container CLI",
)
@pytest.mark.asyncio
class TestIntegrationLifecycle:
    """Full lifecycle test against a real container. Gated to macOS."""

    async def test_full_lifecycle(self, tmp_path):
        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")

        config = MagicMock()
        config.cpus = 1
        config.memory_mb = 512
        config.docker_image = "ubuntu:24.04"
        config.env = None
        config.skills_dir = None
        config.build_timeout_sec = 60
        config.storage_mb = None
        config.gpus = None

        paths = MagicMock()
        paths.rollout_dir = tmp_path / "rollout"
        paths.verifier_dir = tmp_path / "rollout" / "verifier"
        paths.agent_dir = tmp_path / "rollout" / "agent"
        paths.artifacts_dir = tmp_path / "rollout" / "artifacts"

        AppleContainerSandbox.preflight()
        sb = AppleContainerSandbox(
            environment_dir=env_dir,
            environment_name="integration-test",
            session_id="integ-001",
            rollout_paths=paths,
            task_env_config=config,
        )
        try:
            await sb.start(force_build=False)
            result = await sb.exec("echo hello", timeout_sec=10)
            assert result.return_code == 0
            assert "hello" in (result.stdout or "")

            # Upload via exec -i
            src = tmp_path / "upload.txt"
            src.write_text("uploaded content")
            await sb.upload_file(src, "/tmp/upload.txt")
            result = await sb.exec("cat /tmp/upload.txt", timeout_sec=10)
            assert "uploaded content" in (result.stdout or "")
        finally:
            await sb.stop(delete=True)
