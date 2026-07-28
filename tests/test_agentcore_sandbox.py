"""AgentCore sandbox behaviour, driven against a fake boto3 command plane.

The expectations encoded here were measured against the live service during
the design of this backend (issue #935), not inferred from documentation:

* ``contentDelta`` carries ``stdout`` and ``stderr`` as separate fields, so
  ``ExecResult`` must keep them separate.
* ``contentStop.exitCode`` is the command's real exit status and must not be
  flattened to 0.
* ``contentStop.status == "TIMED_OUT"`` is how a timeout is reported.
* Throttling and quota exhaustion arrive as typed members of the response
  stream, and are infrastructure failures rather than failed commands.

The live end-to-end path is exercised separately by
``test_real_agentcore_lifecycle``, which is skipped unless
``BENCHFLOW_AGENTCORE_LIVE_TEST=1`` and AWS credentials are present.
"""

from __future__ import annotations

import base64
import contextlib
import os
import re
import shlex
import tarfile
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from benchflow.sandbox.agentcore import AgentCoreSandbox
from benchflow.sandbox.protocol import SandboxStartupError
from benchflow.task.config import SandboxConfig


def _stream(*, stdout="", stderr="", exit_code=0, status="COMPLETED"):
    chunks = []
    if stdout or stderr:
        chunks.append({"chunk": {"contentDelta": {"stdout": stdout, "stderr": stderr}}})
    chunks.append({"chunk": {"contentStop": {"exitCode": exit_code, "status": status}}})
    return {"stream": chunks}


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    env = AgentCoreSandbox(
        environment_dir=tmp_path,
        environment_name="demo-task",
        session_id="run-1",
        rollout_paths=None,
        task_env_config=SandboxConfig(),
    )
    env.runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:1:runtime/demo"
    env.runtime_session_id = "s" * 40
    return env


class TestExecSemantics:
    @pytest.mark.asyncio
    async def test_stdout_and_stderr_stay_separate(self, sandbox):
        """The service splits the streams; ExecResult must not merge them."""
        client = MagicMock()
        client.invoke_agent_runtime_command.return_value = _stream(
            stdout="out", stderr="err", exit_code=0
        )
        with patch.object(sandbox, "_client", return_value=client):
            result = await sandbox.exec("echo hi")

        assert result.stdout == "out"
        assert result.stderr == "err"
        assert result.return_code == 0

    @pytest.mark.asyncio
    async def test_nonzero_exit_code_is_preserved(self, sandbox):
        client = MagicMock()
        client.invoke_agent_runtime_command.return_value = _stream(exit_code=7)
        with patch.object(sandbox, "_client", return_value=client):
            result = await sandbox.exec("exit 7")

        assert result.return_code == 7

    @pytest.mark.asyncio
    async def test_missing_content_stop_is_not_reported_as_success(self, sandbox):
        """A truncated stream has no exit status — inventing 0 would hide it."""
        client = MagicMock()
        client.invoke_agent_runtime_command.return_value = {
            "stream": [{"chunk": {"contentDelta": {"stdout": "partial"}}}]
        }
        with patch.object(sandbox, "_client", return_value=client):
            result = await sandbox.exec("something")

        assert result.return_code != 0

    @pytest.mark.asyncio
    async def test_timed_out_status_raises(self, sandbox):
        client = MagicMock()
        client.invoke_agent_runtime_command.return_value = _stream(
            exit_code=None, status="TIMED_OUT"
        )
        with (
            patch.object(sandbox, "_client", return_value=client),
            pytest.raises(TimeoutError),
        ):
            await sandbox.exec("sleep 999", timeout_sec=5)

    @pytest.mark.asyncio
    async def test_command_is_wrapped_for_bash(self, sandbox):
        """AgentCore does not run a shell for you; commands need bash -c."""
        client = MagicMock()
        client.invoke_agent_runtime_command.return_value = _stream()
        with patch.object(sandbox, "_client", return_value=client):
            await sandbox.exec("echo hi")

        body = client.invoke_agent_runtime_command.call_args.kwargs["body"]
        assert body["command"].startswith("/bin/bash -c ")

    @pytest.mark.asyncio
    async def test_timeout_is_clamped_to_service_range(self, sandbox):
        """The service rejects timeouts outside 1..3600."""
        client = MagicMock()
        client.invoke_agent_runtime_command.return_value = _stream()
        with patch.object(sandbox, "_client", return_value=client):
            await sandbox.exec("echo hi", timeout_sec=99999)

        body = client.invoke_agent_runtime_command.call_args.kwargs["body"]
        assert body["timeout"] == 3600

    @pytest.mark.asyncio
    async def test_secrets_are_not_passed_as_literal_argv(self, sandbox):
        """Env values must go through the base64 env-file wrapper (#412)."""
        client = MagicMock()
        client.invoke_agent_runtime_command.return_value = _stream()
        with patch.object(sandbox, "_client", return_value=client):
            await sandbox.exec("run", env={"SECRET_TOKEN": "hunter2"})

        command = client.invoke_agent_runtime_command.call_args.kwargs["body"][
            "command"
        ]
        assert "hunter2" not in command
        assert "base64 -d" in command

    @pytest.mark.asyncio
    async def test_non_main_service_is_rejected(self, sandbox):
        with pytest.raises(ValueError, match="single-container"):
            await sandbox.exec("echo hi", service="target")


class TestInfrastructureAttribution:
    @pytest.mark.parametrize(
        "error_key",
        ["throttlingException", "serviceQuotaExceededException"],
    )
    @pytest.mark.asyncio
    async def test_capacity_errors_are_infra_not_agent_failure(
        self, sandbox, error_key
    ):
        """Throttling must not be recorded as a command that failed.

        Attributing a quota error to the agent is how an infrastructure
        problem silently becomes a scored 0 in result.json.
        """
        client = MagicMock()
        client.invoke_agent_runtime_command.return_value = {
            "stream": [{error_key: {"message": "slow down"}}]
        }
        with (
            patch.object(sandbox, "_client", return_value=client),
            pytest.raises(SandboxStartupError),
        ):
            await sandbox.exec("echo hi")


class TestSessionWarmup:
    @pytest.mark.asyncio
    async def test_cold_start_500_is_retried_not_fatal(self, sandbox, monkeypatch):
        """A cold image 500s on the first command; that is not a task failure.

        READY describes the runtime definition, not a running session — the
        first command pulls the image and starts the container. Observed live:
        a never-pulled image fails the first command and succeeds moments
        later. Failing closed here would score a rollout 0 for a sandbox that
        was merely still booting.
        """
        monkeypatch.setattr(
            "benchflow.sandbox.agentcore._SESSION_WARMUP_BACKOFF_SEC", 0.0
        )
        calls = {"n": 0}

        async def _exec(command, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("Received error (500) from runtime")
            return MagicMock(return_code=0, stdout="", stderr="")

        with patch.object(sandbox, "exec", side_effect=_exec):
            await sandbox._warm_session()

        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_persistent_failure_still_reports_startup_error(
        self, sandbox, monkeypatch
    ):
        monkeypatch.setattr(
            "benchflow.sandbox.agentcore._SESSION_WARMUP_BACKOFF_SEC", 0.0
        )

        async def _exec(command, **kwargs):
            raise RuntimeError("Received error (500) from runtime")

        with (
            patch.object(sandbox, "exec", side_effect=_exec),
            pytest.raises(SandboxStartupError, match="did not accept commands"),
        ):
            await sandbox._warm_session()

    @pytest.mark.asyncio
    async def test_throttling_during_warmup_is_not_retried(self, sandbox):
        """Quota errors are infra and terminal; retrying only wastes time."""

        async def _exec(command, **kwargs):
            raise SandboxStartupError("AgentCore throttlingException: slow down")

        with (
            patch.object(sandbox, "exec", side_effect=_exec) as mock_exec,
            pytest.raises(SandboxStartupError, match="throttling"),
        ):
            await sandbox._warm_session()

        assert mock_exec.call_count == 1


class TestFileTransfer:
    @pytest.mark.asyncio
    async def test_upload_dir_ships_one_archive_not_one_command_per_file(
        self, sandbox, tmp_path
    ):
        """A round trip per file would be unusably slow for a task tree."""
        source = tmp_path / "payload"
        (source / "nested").mkdir(parents=True)
        for i in range(12):
            (source / f"file{i}.txt").write_text(f"contents {i}")
        (source / "nested" / "deep.txt").write_text("deep")

        commands: list[str] = []

        async def _record(command, **kwargs):
            commands.append(command)
            return MagicMock(return_code=0, stdout="", stderr="")

        with patch.object(sandbox, "exec", side_effect=_record):
            await sandbox.upload_dir(source, "/workspace")

        # mkdir + staged chunk(s) + one extract, not 13 uploads.
        assert len(commands) < 13
        assert any("tar -xzf" in c for c in commands)

    @staticmethod
    def _extract_staged_archive(commands: list[str]) -> tarfile.TarFile:
        """Rebuild the tar that upload_dir actually streamed to the sandbox.

        ``_upload_via_tar`` base64-encodes the archive into ``printf`` commands,
        so asserting against the raw command text can never observe the
        archive's contents — a regression that started following symlinks would
        have passed. Decode the staged chunks back into the real tar instead.
        """
        encoded = "".join(
            match.group(1)
            for command in commands
            if (match := re.search(r"printf %s (\S+) >>? /tmp/", command))
        )
        # shlex.quote may wrap the chunk; strip the quoting before decoding.
        encoded = "".join(shlex.split(encoded)) if encoded else ""
        return tarfile.open(fileobj=BytesIO(base64.b64decode(encoded)), mode="r:gz")

    @pytest.mark.asyncio
    async def test_upload_dir_skips_symlinks(self, sandbox, tmp_path):
        """Guards #411: a task symlink must not exfiltrate host files."""
        source = tmp_path / "payload"
        source.mkdir()
        (source / "real.txt").write_text("ok")
        secret = tmp_path / "host-secret.txt"
        secret.write_text("do not ship me")
        (source / "link.txt").symlink_to(secret)

        staged: list[str] = []

        async def _record(command, **kwargs):
            staged.append(command)
            return MagicMock(return_code=0, stdout="", stderr="")

        with patch.object(sandbox, "exec", side_effect=_record):
            await sandbox.upload_dir(source, "/workspace")

        with self._extract_staged_archive(staged) as tar:
            names = tar.getnames()
            payloads = {
                name: (tar.extractfile(name) or BytesIO()).read() for name in names
            }

        # Positive control: the archive really was inspected, not empty.
        assert any(name.endswith("real.txt") for name in names)
        assert not any(name.endswith("link.txt") for name in names)
        assert all(b"do not ship me" not in blob for blob in payloads.values())

    @pytest.mark.asyncio
    async def test_download_dir_extracts_the_returned_archive(self, sandbox, tmp_path):
        payload = tmp_path / "src"
        payload.mkdir()
        (payload / "result.json").write_text('{"reward": 1.0}')
        archive = tmp_path / "a.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(payload / "result.json", arcname="result.json")
        encoded = base64.b64encode(archive.read_bytes()).decode()

        async def _exec(command, **kwargs):
            return MagicMock(return_code=0, stdout=encoded, stderr="")

        target = tmp_path / "out"
        with patch.object(sandbox, "exec", side_effect=_exec):
            await sandbox.download_dir("/logs", target)

        assert (target / "result.json").read_text() == '{"reward": 1.0}'

    @pytest.mark.asyncio
    async def test_oversized_inline_write_is_refused(self, sandbox):
        """Silently truncating past the 64 KB command cap would corrupt files."""
        with pytest.raises(ValueError, match="Refusing to inline"):
            await sandbox.write_text_file("/tmp/big", "x" * 200_000)


class TestImagePreparation:
    def _request(self, sandbox):
        from benchflow.sandbox import agentcore_builder as builders
        from benchflow.sandbox.agentcore import _PING_SHIM

        return builders.BuildRequest(
            context_dir=sandbox.environment_dir,
            dockerfile_text=sandbox._generated_dockerfile_text(),
            shim_text=_PING_SHIM,
            image_uri="reg/repo:tag",
            registry="reg",
            region="us-west-2",
            force_build=False,
            timeout_sec=None,
        )

    def test_generated_dockerfile_adds_the_ping_shim(self, sandbox):
        """Without a /ping responder the microVM 500s on every command."""
        text = sandbox._generated_dockerfile_text()

        assert "FROM python:3.12-slim" in text
        assert "benchflow_agentcore_shim.py" in text
        assert "EXPOSE 8080" in text

    def test_task_dockerfile_is_not_modified_in_place(self, sandbox, tmp_path):
        from benchflow.sandbox import agentcore_builder as builders

        original = (tmp_path / "Dockerfile").read_text()
        with builders.materialized(self._request(sandbox)) as dockerfile:
            assert dockerfile.exists()
            assert (tmp_path / "Dockerfile").read_text() == original

    def test_generated_files_never_survive_the_build(self, sandbox, tmp_path):
        """The build context is the task's own directory; don't litter it."""
        from benchflow.sandbox import agentcore_builder as builders

        before = {p.name for p in tmp_path.iterdir()}
        with builders.materialized(self._request(sandbox)):
            assert {p.name for p in tmp_path.iterdir()} != before

        assert {p.name for p in tmp_path.iterdir()} == before

    def test_generated_files_are_cleaned_up_after_a_failed_build(
        self, sandbox, tmp_path
    ):
        """A build that raises must not leave scaffolding behind either."""
        from benchflow.sandbox import agentcore_builder as builders

        before = {p.name for p in tmp_path.iterdir()}
        with contextlib.suppress(RuntimeError):  # noqa: SIM117
            with builders.materialized(self._request(sandbox)):
                raise RuntimeError("build blew up")

        assert {p.name for p in tmp_path.iterdir()} == before

    def test_docker_image_config_is_honoured(self, tmp_path):
        env = AgentCoreSandbox(
            environment_dir=tmp_path,
            environment_name="img-task",
            session_id="run-1",
            rollout_paths=None,
            task_env_config=SandboxConfig(docker_image="python:3.12-slim"),
        )

        assert "FROM python:3.12-slim" in env._generated_dockerfile_text()


class TestCapabilityGating:
    def test_snapshots_are_unsupported(self, sandbox):
        """AgentCore has no container checkpoint primitive."""
        assert sandbox.supports_snapshot is False

    def test_no_network_tasks_are_refused(self, tmp_path):
        """networkMode offers only PUBLIC/VPC, so isolation cannot be honoured."""
        from benchflow.task.config import TaskConfig
        from benchflow.task.runtime_capabilities import validate_task_runtime_support

        config = TaskConfig.model_validate(
            {"environment": {"network_mode": "no-network"}}
        )
        issues = validate_task_runtime_support(config, sandbox="agentcore")

        assert any(
            "no-network' is not enforced by agentcore" in issue.reason
            for issue in issues
        )


@pytest.mark.skipif(
    os.environ.get("BENCHFLOW_AGENTCORE_LIVE_TEST") != "1",
    reason="live AWS test; set BENCHFLOW_AGENTCORE_LIVE_TEST=1 to run",
)
@pytest.mark.asyncio
async def test_real_agentcore_lifecycle(tmp_path):
    """End-to-end against the live service: build, run, transfer, tear down.

    Requires AWS credentials with bedrock-agentcore and ECR access, a docker
    daemon, and BENCHFLOW_AGENTCORE_ROLE_ARN pointing at the runtime execution
    role.
    """
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.12-slim\nRUN echo baked > /baked.txt\n"
    )
    env = AgentCoreSandbox(
        environment_dir=tmp_path,
        environment_name="live-canary",
        session_id="live-1",
        rollout_paths=None,
        task_env_config=SandboxConfig(),
    )
    AgentCoreSandbox.preflight()
    await env.start(force_build=False)
    try:
        baked = await env.exec("cat /baked.txt")
        assert baked.return_code == 0
        assert "baked" in (baked.stdout or "")

        streams = await env.exec("echo o; echo e 1>&2; exit 3")
        assert streams.return_code == 3
        assert "o" in (streams.stdout or "")
        assert "e" in (streams.stderr or "")

        payload = tmp_path / "payload"
        payload.mkdir()
        (payload / "hello.txt").write_text("from-host")
        await env.upload_dir(payload, "/workspace/payload")
        echoed = await env.exec("cat /workspace/payload/hello.txt")
        assert (echoed.stdout or "").strip() == "from-host"

        out = tmp_path / "out"
        await env.download_dir("/workspace/payload", out)
        assert (out / "hello.txt").read_text() == "from-host"
    finally:
        await env.stop(delete=True)
