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
    async def test_secrets_never_reach_any_command_body(self, sandbox):
        """Env must not appear in *any* command, encoded or not (#412, #942).

        This platform records command bodies permanently, so the pre-#942
        base64 env wrapper was reversible from the log. The environment is
        now staged over the sealed channel and merely sourced by path.
        """
        _seeded_seal_keypair(sandbox)
        client = MagicMock()
        client.invoke_agent_runtime_command.return_value = _stream()
        with patch.object(sandbox, "_client", return_value=client):
            await sandbox.exec("run", env={"SECRET_TOKEN": "hunter2"})

        bodies = [
            call.kwargs["body"]["command"]
            for call in client.invoke_agent_runtime_command.call_args_list
        ]
        assert bodies
        for command in bodies:
            assert "hunter2" not in command
            assert "SECRET_TOKEN" not in command
            assert base64.b64encode(b"hunter2").decode() not in command
        # The final command sources a staged env file by path only, and
        # that file lives outside the root-only seal directory (behavior:
        # a non-root exec user must be able to read it).
        from benchflow.sandbox.agentcore_sealed import SEAL_DIR

        sourced = [c for c in bodies if "set -a; . " in c]
        assert sourced
        assert all(f". {SEAL_DIR}/" not in c for c in sourced)

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


def _seeded_seal_keypair(sandbox):
    """Give the sandbox's sealed channel a real public key."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    sandbox._sealed._public_key = (
        private.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return private


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

        _seeded_seal_keypair(sandbox)
        with (
            patch.object(sandbox, "exec", side_effect=_record),
            patch.object(sandbox._sealed, "_exec_raw", side_effect=_record),
        ):
            await sandbox.upload_dir(source, "/workspace")

        # mkdir + sealed chunk(s) + one decrypt-and-extract, not 13 uploads
        # (PR #942 moved transfers to the sealed transport).
        assert len(commands) < 13
        assert any("tar -xzf" in c for c in commands)
        extract = next(c for c in commands if "tar -xzf" in c)
        assert "/tmp/.bf_sealed/" in extract
        assert "openssl pkeyutl -decrypt" in extract
        assert "set -o pipefail" in extract

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

        sealed: list[bytes] = []

        async def _capture(data, **kwargs):
            sealed.append(data)

        async def _exec(command, **kwargs):
            # Only the target-dir mkdir may run outside the sealed transport.
            assert command.startswith("mkdir -p "), command
            return MagicMock(return_code=0, stdout="", stderr="")

        with (
            patch.object(sandbox, "exec", side_effect=_exec),
            patch.object(sandbox._sealed, "upload", side_effect=_capture),
        ):
            await sandbox.upload_dir(source, "/workspace")

        # The archive is inspected pre-seal: link exclusion happens while the
        # tar is built; sealed-command confidentiality is covered separately
        # (PR #942).
        import tarfile as _tarfile

        with _tarfile.open(fileobj=BytesIO(sealed[-1]), mode="r:gz") as tar:
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
    async def test_oversized_write_is_chunked_under_the_command_cap(self, sandbox):
        """Large bodies stage as multiple sealed chunks, none oversized.

        Replaces the pre-PR #942 refusal: the sealed transport chunks
        ciphertext instead, so no single command exceeds the service cap and
        nothing is silently truncated.
        """
        from benchflow.sandbox.agentcore import _MAX_INLINE_UPLOAD_BYTES

        _seeded_seal_keypair(sandbox)
        commands: list[str] = []

        async def _record(command, **kwargs):
            commands.append(command)
            return MagicMock(return_code=0, stdout="", stderr="")

        with (
            patch.object(sandbox, "exec", side_effect=_record),
            patch.object(sandbox._sealed, "_exec_raw", side_effect=_record),
        ):
            assert await sandbox.write_text_file("/tmp/big", "x" * 200_000)

        staging = [
            c
            for c in commands
            if ">> /tmp/.bf_sealed/" in c or "> /tmp/.bf_sealed/s_" in c
        ]
        assert len(staging) > 1  # genuinely chunked
        for command in commands:
            assert len(command) < _MAX_INLINE_UPLOAD_BYTES + 4096
        assert "x" * 64 not in " ".join(commands)  # plaintext never appears

    @pytest.mark.asyncio
    async def test_oversized_file_download_is_refused_before_writing(
        self, sandbox, tmp_path, monkeypatch
    ):
        """Guards PR #937: file downloads need the same memory cap as dirs."""
        monkeypatch.setattr("benchflow.sandbox.agentcore._MAX_DOWNLOAD_BYTES", 8)
        encoded = base64.b64encode(b"012345678").decode()

        async def _exec(command, **kwargs):
            return MagicMock(return_code=0, stdout=encoded, stderr="")

        target = tmp_path / "too-large.bin"
        with (
            patch.object(sandbox, "exec", side_effect=_exec),
            pytest.raises(RuntimeError, match="8 byte cap"),
        ):
            await sandbox.download_file("/remote/large", target)

        assert not target.exists()

    def test_invalid_download_base64_fails_closed(self, sandbox):
        """Guards PR #937: corrupted provider output must not become a file."""
        with pytest.raises(RuntimeError, match="invalid base64"):
            sandbox._decode_download_payload("not@base64", kind="file")


class TestImagePreparation:
    def _request(self, sandbox):
        from benchflow.sandbox import agentcore_builder as builders
        from benchflow.sandbox.agentcore_image import PING_SHIM

        return builders.BuildRequest(
            context_dir=sandbox.environment_dir,
            dockerfile_text=sandbox._images.generated_dockerfile_text(),
            shim_text=PING_SHIM,
            image_uri="reg/repo:tag",
            registry="reg",
            region="us-west-2",
            force_build=False,
            timeout_sec=None,
        )

    def test_generated_dockerfile_adds_the_ping_shim(self, sandbox):
        """Without a /ping responder the microVM 500s on every command."""
        text = sandbox._images.generated_dockerfile_text()

        assert "FROM python:3.12-slim" in text
        assert "benchflow_agentcore_shim.py" in text
        assert "EXPOSE 8080" in text

    def test_task_dockerfile_is_not_modified_in_place(self, sandbox, tmp_path):
        from benchflow.sandbox import agentcore_builder as builders

        original = (tmp_path / "Dockerfile").read_text()
        with builders.materialized(self._request(sandbox)) as dockerfile:
            assert dockerfile.exists()
            assert dockerfile.parent != tmp_path
            assert (tmp_path / "Dockerfile").read_text() == original

    def test_generated_files_never_survive_the_build(self, sandbox, tmp_path):
        """Guards PR #937: staging never mutates the caller's task directory."""
        from benchflow.sandbox import agentcore_builder as builders

        before = {p.name for p in tmp_path.iterdir()}
        with builders.materialized(self._request(sandbox)) as dockerfile:
            assert {p.name for p in tmp_path.iterdir()} == before
            assert (dockerfile.parent / ".benchflow_agentcore_shim.py").is_file()

        assert {p.name for p in tmp_path.iterdir()} == before

    def test_generated_files_are_cleaned_up_after_a_failed_build(
        self, sandbox, tmp_path
    ):
        """A build that raises must not leave scaffolding behind either."""
        from benchflow.sandbox import agentcore_builder as builders

        before = {p.name for p in tmp_path.iterdir()}
        staged = None
        with contextlib.suppress(RuntimeError):  # noqa: SIM117
            with builders.materialized(self._request(sandbox)) as dockerfile:
                staged = dockerfile.parent
                raise RuntimeError("build blew up")

        assert {p.name for p in tmp_path.iterdir()} == before
        assert staged is not None and not staged.exists()

    def test_docker_image_config_is_honoured(self, tmp_path):
        env = AgentCoreSandbox(
            environment_dir=tmp_path,
            environment_name="img-task",
            session_id="run-1",
            rollout_paths=None,
            task_env_config=SandboxConfig(docker_image="python:3.12-slim"),
        )

        assert "FROM python:3.12-slim" in env._images.generated_dockerfile_text()


class TestCapabilityGating:
    def test_snapshots_are_unsupported(self, sandbox):
        """AgentCore has no container checkpoint primitive."""
        assert sandbox.supports_snapshot is False

    def test_no_network_tasks_are_refused(self, tmp_path):
        """networkMode offers only PUBLIC/VPC, so isolation cannot be honoured."""
        from benchflow.task.config import TaskConfig
        from benchflow.task.runtime_capabilities import validate_task_runtime_support

        config = TaskConfig.model_validate({"sandbox": {"network_mode": "no-network"}})
        issues = validate_task_runtime_support(config, sandbox="agentcore")

        assert any(
            "no-network' is not enforced by agentcore" in issue.reason
            for issue in issues
        )


class TestNetworkConfiguration:
    """Opting a runtime out of the public internet.

    ``networkMode`` accepts ``PUBLIC`` or ``VPC``. Every runtime was
    registered ``PUBLIC``, so an agent under test was reachable from the
    internet with no way to opt out. These pin the opt-in and, more
    importantly, the ways it can fail open.
    """

    _MODE = "BENCHFLOW_AGENTCORE_NETWORK_MODE"
    _SUBNETS = "BENCHFLOW_AGENTCORE_SUBNETS"
    _SECURITY_GROUPS = "BENCHFLOW_AGENTCORE_SECURITY_GROUPS"

    def _vpc_env(self, monkeypatch):
        monkeypatch.setenv(self._MODE, "VPC")
        monkeypatch.setenv(self._SUBNETS, "subnet-0a1b2c3d4e5f67890")
        monkeypatch.setenv(self._SECURITY_GROUPS, "sg-0a1b2c3d4e5f67890")

    def test_the_default_is_public(self, sandbox, monkeypatch):
        """Unset must stay unchanged, or every existing deployment moves."""
        for name in (self._MODE, self._SUBNETS, self._SECURITY_GROUPS):
            monkeypatch.delenv(name, raising=False)

        assert sandbox._network_configuration() == {"networkMode": "PUBLIC"}

    def test_vpc_mode_carries_the_subnets_and_security_groups(
        self, sandbox, monkeypatch
    ):
        """The runtime shape nests these under ``networkModeConfig``.

        Browsers and code interpreters take the same ``VpcConfig`` under
        ``vpcConfig`` instead; sending that key here is a ValidationException.
        """
        monkeypatch.setenv(self._MODE, "VPC")
        monkeypatch.setenv(self._SUBNETS, "subnet-0a1b2c3d4e5f67890, subnet-01234567")
        monkeypatch.setenv(self._SECURITY_GROUPS, "sg-0a1b2c3d4e5f67890")

        assert sandbox._network_configuration() == {
            "networkMode": "VPC",
            "networkModeConfig": {
                "subnets": ["subnet-0a1b2c3d4e5f67890", "subnet-01234567"],
                "securityGroups": ["sg-0a1b2c3d4e5f67890"],
            },
        }

    @pytest.mark.parametrize(
        "missing",
        ["BENCHFLOW_AGENTCORE_SUBNETS", "BENCHFLOW_AGENTCORE_SECURITY_GROUPS"],
    )
    def test_vpc_mode_requires_both_lists(self, sandbox, monkeypatch, missing):
        """VPC with only half the config is rejected by the service."""
        self._vpc_env(monkeypatch)
        monkeypatch.delenv(missing)

        with pytest.raises(ValueError, match=missing):
            sandbox._network_configuration()

    def test_vpc_resources_without_vpc_mode_are_refused(self, sandbox, monkeypatch):
        """Ignoring them would leave the runtime on the public internet.

        The operator who sets subnets but forgets the mode believes the agent
        is isolated. Failing open there is the whole bug this guards.
        """
        monkeypatch.delenv(self._MODE, raising=False)
        monkeypatch.setenv(self._SUBNETS, "subnet-0a1b2c3d4e5f67890")
        monkeypatch.setenv(self._SECURITY_GROUPS, "sg-0a1b2c3d4e5f67890")

        with pytest.raises(ValueError, match=self._MODE):
            sandbox._network_configuration()

    def test_an_unknown_mode_is_refused(self, sandbox, monkeypatch):
        """``no-network`` reads as isolation but is not a mode the service has."""
        monkeypatch.setenv(self._MODE, "ISOLATED")

        with pytest.raises(ValueError, match="PUBLIC"):
            sandbox._network_configuration()

    def test_a_malformed_subnet_id_is_refused(self, sandbox, monkeypatch):
        """Otherwise a pasted VPC id surfaces as an opaque ValidationException."""
        self._vpc_env(monkeypatch)
        monkeypatch.setenv(self._SUBNETS, "vpc-0a1b2c3d4e5f67890")

        with pytest.raises(ValueError, match="subnet-"):
            sandbox._network_configuration()

    def test_a_malformed_security_group_id_is_refused(self, sandbox, monkeypatch):
        self._vpc_env(monkeypatch)
        monkeypatch.setenv(self._SECURITY_GROUPS, "0a1b2c3d4e5f67890")

        with pytest.raises(ValueError, match="sg-"):
            sandbox._network_configuration()


@pytest.mark.skipif(
    os.environ.get("BENCHFLOW_AGENTCORE_LIVE_TEST") != "1",
    reason="live AWS test; set BENCHFLOW_AGENTCORE_LIVE_TEST=1 to run",
)
@pytest.mark.asyncio
async def test_real_agentcore_lifecycle(tmp_path):
    """End-to-end against the live service: build, run, transfer, tear down.

    Requires AWS credentials with bedrock-agentcore and ECR access,
    BENCHFLOW_AGENTCORE_ROLE_ARN pointing at the runtime execution role, and
    either Docker or BENCHFLOW_AGENTCORE_CODEBUILD_ROLE_ARN.
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
