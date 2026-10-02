"""Contract tests for the local Apptainer sandbox provider."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.apptainer import ApptainerSandbox
from benchflow.sandbox.apptainer_image import ApptainerImage
from benchflow.sandbox.process.apptainer import ApptainerProcess
from benchflow.task import RolloutPaths
from benchflow.task.config import NetworkMode, SandboxConfig


def _sandbox(tmp_path: Path, *, network_mode: NetworkMode = NetworkMode.PUBLIC):
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /workspace\n")
    rollout_paths = RolloutPaths(tmp_path / "rollout")
    builder = AsyncMock()
    builder.resolve.return_value = ApptainerImage(
        tmp_path / "image.sif", "digest", "/workspace"
    )
    sandbox = ApptainerSandbox(
        environment_dir=environment,
        environment_name="task",
        session_id="trial",
        rollout_paths=rollout_paths,
        task_env_config=SandboxConfig(
            network_mode=network_mode,
            storage_mb=64,
        ),
        image_builder=builder,
    )
    return sandbox, builder, rollout_paths


@pytest.mark.asyncio
async def test_start_creates_overlay_mounts_logs_and_enforces_no_network(
    tmp_path: Path,
) -> None:
    """Express rollout isolation and network policy in Apptainer arguments."""
    sandbox, builder, rollout_paths = _sandbox(
        tmp_path, network_mode=NetworkMode.NO_NETWORK
    )
    success = ExecResult(stdout="", stderr="", return_code=0)

    with patch(
        "benchflow.sandbox.apptainer._run_apptainer",
        new=AsyncMock(return_value=success),
    ) as run:
        await sandbox.start(force_build=False)
        calls = [call.args for call in run.await_args_list]
        start = next(args for args in calls if args[:2] == ("instance", "start"))

        assert "--fakeroot" in start
        assert "--overlay" in start
        assert "--sparse" in calls[0]
        assert "--containall" in start
        assert "--no-home" in start
        assert start[start.index("--net") : start.index("--net") + 3] == (
            "--net",
            "--network",
            "none",
        )
        assert f"{rollout_paths.rollout_dir}:/logs" in start
        assert sandbox.sandbox_id
        assert sandbox.is_mounted
        builder.resolve.assert_awaited_once()

        await sandbox.stop(delete=True)


@pytest.mark.asyncio
async def test_exec_uses_image_workdir_and_merges_persistent_env(
    tmp_path: Path,
) -> None:
    """Preserve working-directory and environment semantics during execution."""
    sandbox, _, _ = _sandbox(tmp_path)
    sandbox._instance_name = "bf-task"
    sandbox._staging_dir = tmp_path / "staging"
    sandbox._staging_dir.mkdir()
    sandbox._default_cwd = "/workspace"
    sandbox._persistent_env = {"TOKEN": "secret"}
    success = ExecResult(stdout="ok", stderr="", return_code=0)

    with patch(
        "benchflow.sandbox.apptainer._run_apptainer",
        new=AsyncMock(return_value=success),
    ) as run:
        result = await sandbox.exec("pwd", env={"MODE": "test"})

    assert result.stdout == "ok"
    args = run.await_args.args
    assert args[:3] == ("exec", "--cleanenv", "--cwd")
    assert args[3] == "/workspace"
    wrapped = args[-1]
    assert "TOKEN" not in wrapped
    assert "secret" not in wrapped
    assert "base64 -d" in wrapped


@pytest.mark.asyncio
async def test_logs_are_transferred_through_the_host_mount(tmp_path: Path) -> None:
    """Transfer mounted rollout artifacts without an extra container copy."""
    sandbox, _, rollout_paths = _sandbox(tmp_path)
    sandbox._instance_name = "bf-task"
    sandbox._staging_dir = tmp_path / "staging"
    sandbox._staging_dir.mkdir()
    rollout_paths.mkdir()
    source = tmp_path / "reward.txt"
    source.write_text("1")

    await sandbox.upload_file(source, "/logs/verifier/reward.txt", mode="600")
    target = tmp_path / "downloaded.txt"
    await sandbox.download_file("/logs/verifier/reward.txt", target)

    assert target.read_text() == "1"
    assert (rollout_paths.verifier_dir / "reward.txt").stat().st_mode & 0o777 == 0o600


def test_preflight_requires_apptainer_cli() -> None:
    """Fail preflight before a rollout when the Apptainer CLI is unavailable."""
    with (
        patch(
            "benchflow.sandbox.apptainer.require_apptainer",
            side_effect=RuntimeError("missing"),
        ),
        pytest.raises(RuntimeError, match="missing"),
    ):
        ApptainerSandbox.preflight()


@pytest.mark.asyncio
async def test_live_process_inherits_instance_and_workdir(tmp_path: Path) -> None:
    """Use the Apptainer live-process transport for ACP agents."""
    sandbox, _, _ = _sandbox(tmp_path)
    sandbox._instance_name = "bf-task"
    sandbox._staging_dir = tmp_path / "staging"
    sandbox._staging_dir.mkdir()
    sandbox._default_cwd = "/workspace"

    process = await sandbox.live_process()

    assert process._instance_name == "bf-task"
    assert process._default_cwd == "/workspace"


@pytest.mark.asyncio
async def test_live_process_stages_env_without_exposing_values_in_argv() -> None:
    """Stage ACP environment values without exposing them in process arguments."""
    staged = MagicMock(returncode=0)
    staged.communicate = AsyncMock(return_value=(b"", b""))
    running = MagicMock(pid=123, stderr=None)

    with patch(
        "benchflow.sandbox.process.apptainer.asyncio.create_subprocess_exec",
        new=AsyncMock(side_effect=[staged, running]),
    ) as spawn:
        process = ApptainerProcess("bf-task", "/workspace")
        await process.start("agent --stdio", env={"TOKEN": "top secret"})

    stage_args = spawn.await_args_list[0].args
    exec_args = spawn.await_args_list[1].args
    assert "top secret" not in " ".join(str(arg) for arg in stage_args + exec_args)
    assert exec_args[:3] == ("apptainer", "exec", "--cleanenv")
    assert exec_args[3:5] == ("--cwd", "/workspace")
    assert exec_args[5] == "instance://bf-task"
    assert ". /tmp/.benchflow_agent_env_" in exec_args[-1]
    assert "rm -f /tmp/.benchflow_agent_env_" in exec_args[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type", [TimeoutError, asyncio.CancelledError], ids=["timeout", "cancelled"]
)
async def test_overlay_creation_interruption_removes_the_runtime_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    """Guard PR #1073: reclaim storage when overlay creation is interrupted."""
    sandbox, _, _ = _sandbox(tmp_path)
    runtime_root = tmp_path / "runtime-root"
    runtime_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(runtime_root))

    with (
        patch(
            "benchflow.sandbox.apptainer._run_apptainer",
            new=AsyncMock(side_effect=error_type()),
        ),
        pytest.raises(error_type),
    ):
        await sandbox.start(force_build=False)

    assert sandbox._runtime_dir is None
    assert sandbox._instance_name is None
    assert list(runtime_root.iterdir()) == []


@pytest.mark.asyncio
async def test_failed_start_without_an_instance_removes_runtime_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard PR #1073: a failed start without an instance is fully cleaned."""
    sandbox, _, _ = _sandbox(tmp_path)
    runtime_root = tmp_path / "runtime-root"
    runtime_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(runtime_root))
    success = ExecResult(stdout="", stderr="", return_code=0)
    failure = ExecResult(stdout="", stderr="start failed", return_code=1)
    absent = ExecResult(stdout='{"instances": []}', stderr="", return_code=0)

    with (
        patch(
            "benchflow.sandbox.apptainer._run_apptainer",
            new=AsyncMock(side_effect=[success, failure, failure, absent]),
        ),
        pytest.raises(RuntimeError, match="Could not start Apptainer instance"),
    ):
        await sandbox.start(force_build=False)

    assert sandbox._instance_name is None
    assert sandbox._runtime_dir is None
    assert list(runtime_root.iterdir()) == []


@pytest.mark.asyncio
async def test_graceful_stop_timeout_still_forces_the_instance_down(
    tmp_path: Path,
) -> None:
    """Guard PR #1073: force-stop an instance after graceful-stop timeout."""
    sandbox, _, _ = _sandbox(tmp_path)
    sandbox._instance_name = "bf-task"
    sandbox._runtime_dir = tmp_path / "runtime"
    sandbox._runtime_dir.mkdir()
    success = ExecResult(stdout="", stderr="", return_code=0)

    with patch(
        "benchflow.sandbox.apptainer._run_apptainer",
        new=AsyncMock(side_effect=[TimeoutError(), success]),
    ) as run:
        await sandbox.stop(delete=True)

    assert run.await_args_list[-1].args == (
        "instance",
        "stop",
        "--force",
        "bf-task",
    )
    assert sandbox._instance_name is None
    assert sandbox._runtime_dir is None


@pytest.mark.asyncio
async def test_unstoppable_instance_keeps_its_overlay_for_retry(
    tmp_path: Path,
) -> None:
    """Guard PR #1073: retain storage while an instance remains live."""
    sandbox, _, _ = _sandbox(tmp_path)
    sandbox._instance_name = "bf-task"
    sandbox._runtime_dir = tmp_path / "runtime"
    sandbox._runtime_dir.mkdir()
    failure = ExecResult(stdout="", stderr="still running", return_code=1)
    running = ExecResult(
        stdout='{"instances": [{"instance": "bf-task"}]}',
        stderr="",
        return_code=0,
    )

    with patch(
        "benchflow.sandbox.apptainer._run_apptainer",
        new=AsyncMock(side_effect=[failure, running]),
    ):
        await sandbox._force_cleanup()

    assert sandbox._instance_name == "bf-task"
    assert sandbox._runtime_dir is not None
    assert sandbox._runtime_dir.is_dir()

    with (
        patch(
            "benchflow.sandbox.apptainer._run_apptainer",
            new=AsyncMock(side_effect=[failure, failure, running]),
        ),
        pytest.raises(RuntimeError, match="Could not stop Apptainer instance"),
    ):
        await sandbox.stop(delete=True)
