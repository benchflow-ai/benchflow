"""Image build strategy selection for the AgentCore backend.

AgentCore only runs images that already exist in ECR, so something must build
one. Requiring a local Docker daemon defeats the point of a cloud backend on a
machine that has no room to run containers, so the builder falls back to AWS
CodeBuild on a Graviton worker when no daemon is reachable.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import threading
import time
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.sandbox import agentcore_builder as builders


def _request(tmp_path, **overrides):
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    defaults = dict(
        context_dir=tmp_path,
        dockerfile_text="FROM python:3.12-slim\n",
        shim_text="print('shim')\n",
        image_uri="1.dkr.ecr.us-west-2.amazonaws.com/repo:tag",
        registry="1.dkr.ecr.us-west-2.amazonaws.com",
        region="us-west-2",
        force_build=False,
        timeout_sec=None,
    )
    defaults.update(overrides)
    return builders.BuildRequest(**defaults)


def _materialize_in_child(request, barrier, results):
    """Hold one staged context open so another OS process overlaps it."""
    with builders.materialized(request) as dockerfile:
        results.put(
            (
                str(dockerfile.parent),
                (dockerfile.parent / ".benchflow_agentcore_shim.py").read_text(),
            )
        )
        barrier.wait(timeout=10)


class TestBuilderSelection:
    def test_prefers_local_docker_when_the_daemon_is_up(self):
        with patch.object(builders, "docker_available", return_value=True):
            builder = builders.select_builder(
                MagicMock(), account_id="1", region="us-west-2"
            )

        assert isinstance(builder, builders.LocalDockerBuilder)

    def test_falls_back_to_codebuild_without_a_daemon(self):
        """The whole point: usable on a machine with no container runtime."""
        with patch.object(builders, "docker_available", return_value=False):
            builder = builders.select_builder(
                MagicMock(), account_id="1", region="us-west-2"
            )

        assert isinstance(builder, builders.CodeBuildBuilder)

    def test_explicit_preference_overrides_detection(self):
        with patch.object(builders, "docker_available", return_value=True):
            builder = builders.select_builder(
                MagicMock(), account_id="1", region="us-west-2", preference="codebuild"
            )

        assert isinstance(builder, builders.CodeBuildBuilder)

    def test_invalid_preference_is_rejected(self):
        with pytest.raises(ValueError, match="auto, docker, or codebuild"):
            builders.select_builder(
                MagicMock(), account_id="1", region="us-west-2", preference="podman"
            )

    def test_a_stopped_daemon_counts_as_unavailable(self):
        """An installed CLI with a dead daemon is the common laptop case.

        Discovering that at build time would waste the whole provisioning path.
        """
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch.object(builders, "_run", return_value=MagicMock(returncode=1)),
        ):
            assert builders.docker_available() is False

    def test_codebuild_without_a_role_says_what_to_set(self, monkeypatch):
        monkeypatch.delenv(builders.ENV_CODEBUILD_ROLE, raising=False)
        builder = builders.CodeBuildBuilder(MagicMock(), "1", "us-west-2")

        with pytest.raises(RuntimeError) as excinfo:
            builder._role_arn()

        assert builders.ENV_CODEBUILD_ROLE in str(excinfo.value)
        assert "codebuild.amazonaws.com" in str(excinfo.value)


class TestCodeBuildPackaging:
    def test_archive_carries_the_generated_dockerfile_and_shim(self, tmp_path):
        """CodeBuild only sees the archive, so the scaffolding must be inside."""
        (tmp_path / "data.txt").write_text("payload")
        builder = builders.CodeBuildBuilder(MagicMock(), "1", "us-west-2")

        blob = builder._package(_request(tmp_path))

        with zipfile.ZipFile(BytesIO(blob)) as archive:
            names = set(archive.namelist())
        from benchflow.sandbox import agentcore_provisioning as provisioning

        assert provisioning.GENERATED_DOCKERFILE in names
        assert provisioning.GENERATED_SHIM in names
        assert "Dockerfile" in names
        assert "data.txt" in names

    def test_packaging_leaves_no_scaffolding_behind(self, tmp_path):
        builder = builders.CodeBuildBuilder(MagicMock(), "1", "us-west-2")
        request = _request(tmp_path)
        before = {p.name for p in tmp_path.iterdir()}

        builder._package(request)

        assert {p.name for p in tmp_path.iterdir()} == before

    def test_symlinks_are_not_packaged(self, tmp_path):
        """Guards #411 — a task symlink must not ship host files to AWS."""
        secret = tmp_path.parent / "host-secret.txt"
        secret.write_text("do not upload me")
        (tmp_path / "link.txt").symlink_to(secret)
        builder = builders.CodeBuildBuilder(MagicMock(), "1", "us-west-2")

        blob = builder._package(_request(tmp_path))

        with zipfile.ZipFile(BytesIO(blob)) as archive:
            assert "link.txt" not in set(archive.namelist())

    def test_empty_directories_are_packaged_and_change_identity(self, tmp_path):
        """Guards PR #937: empty directories are real Docker context entries."""
        from benchflow.sandbox import agentcore_provisioning as provisioning

        request = _request(tmp_path)
        before = provisioning.build_context_digest(
            tmp_path, request.dockerfile_text, request.shim_text
        )
        (tmp_path / "empty").mkdir()
        after = provisioning.build_context_digest(
            tmp_path, request.dockerfile_text, request.shim_text
        )

        blob = builders.CodeBuildBuilder(MagicMock(), "1", "us-west-2")._package(
            request
        )
        with zipfile.ZipFile(BytesIO(blob)) as archive:
            assert "empty/" in archive.namelist()
        assert after != before

    def test_parallel_processes_use_distinct_staging_contexts(self, tmp_path):
        """Guards PR #937 against cross-process scaffold clobbering."""
        if "fork" not in multiprocessing.get_all_start_methods():
            pytest.skip("requires fork to overlap local staging contexts")
        context = multiprocessing.get_context("fork")
        barrier = context.Barrier(2)
        results = context.Queue()
        first = _request(tmp_path, shim_text="first")
        second = _request(tmp_path, shim_text="second")
        processes = [
            context.Process(
                target=_materialize_in_child,
                args=(request, barrier, results),
            )
            for request in (first, second)
        ]

        for process in processes:
            process.start()
        observed = [results.get(timeout=10) for _ in processes]
        for process in processes:
            process.join(timeout=10)

        assert all(process.exitcode == 0 for process in processes)
        assert len({path for path, _shim in observed}) == 2
        assert {shim for _path, shim in observed} == {"first", "second"}
        assert all(not Path(path).exists() for path, _shim in observed)
        assert not (tmp_path / ".benchflow_agentcore_shim.py").exists()

    def test_partial_scaffold_write_cleans_the_staging_context(
        self, tmp_path, monkeypatch
    ):
        """Guards PR #937: a second generated-file write may fail."""
        request = _request(tmp_path)
        original = Path.write_text
        staged: list[Path] = []

        def _write(path, data, *args, **kwargs):
            if path.name == builders.provisioning.GENERATED_SHIM:
                staged.append(path.parent)
            if path.name == builders.provisioning.GENERATED_DOCKERFILE:
                raise OSError("disk full")
            return original(path, data, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", _write)

        with (
            pytest.raises(OSError, match="disk full"),
            builders.materialized(request),
        ):
            pass

        assert staged and all(not path.exists() for path in staged)

    def test_read_only_directories_stage_with_their_original_mode(self, tmp_path):
        """Guards PR #937: applying 0555 before copying children breaks staging."""
        request = _request(tmp_path)
        readonly = tmp_path / "readonly"
        readonly.mkdir()
        (readonly / "payload.txt").write_text("payload")
        os.chmod(readonly, 0o555)

        try:
            with builders.materialized(request) as dockerfile:
                staged = dockerfile.parent / "readonly"
                assert (staged / "payload.txt").read_text() == "payload"
                assert staged.stat().st_mode & 0o777 == 0o555
        finally:
            os.chmod(readonly, 0o755)

    def test_generated_file_modes_do_not_depend_on_host_umask(self, tmp_path):
        """Guards PR #937: generated shim mode is part of the built image."""
        with builders.materialized(_request(tmp_path)) as dockerfile:
            shim = dockerfile.parent / ".benchflow_agentcore_shim.py"
            assert dockerfile.stat().st_mode & 0o777 == 0o644
            assert shim.stat().st_mode & 0o777 == 0o644

    def test_buildspec_enforces_the_image_size_cap_remotely(self):
        """The 2 GB cap must be caught on the worker, before the push."""
        commands = " ".join(builders._BUILDSPEC["phases"]["build"]["commands"])

        assert "BENCHFLOW_IMAGE_TOO_LARGE" in commands
        # The push must come after the gate, or an oversized image still lands.
        assert commands.index("BENCHFLOW_IMAGE_TOO_LARGE") < commands.index(
            "docker push"
        )

    def test_buildspec_gate_compares_bytes_not_floored_megabytes(self):
        """Cap + 1 byte floors to the cap and would slip a megabyte compare.

        The remote gate must reject exactly what image_size_error() rejects.
        """
        from benchflow.sandbox import agentcore_provisioning as provisioning

        cap_bytes = provisioning.MAX_IMAGE_MB * 1024 * 1024
        commands = " ".join(builders._BUILDSPEC["phases"]["build"]["commands"])

        assert f'"$SIZE" -gt {cap_bytes}' in commands
        # Local and remote gates must agree on the boundary.
        assert provisioning.image_size_error(cap_bytes, "img") is None
        assert provisioning.image_size_error(cap_bytes + 1, "img") is not None

    def test_buildspec_builds_arm64(self):
        """AgentCore runs arm64 only; an x86 image would fail opaquely."""
        commands = " ".join(builders._BUILDSPEC["phases"]["build"]["commands"])

        assert "--platform linux/arm64" in commands

    def test_oversized_remote_build_is_reported_as_a_size_problem(self):
        """A generic 'build failed' would send the user hunting the wrong bug."""
        from benchflow.sandbox.protocol import SandboxStartupError

        builder = builders.CodeBuildBuilder(MagicMock(), "1", "us-west-2")
        build = {"id": "b:1", "buildStatus": "FAILED", "phases": []}

        with patch.object(
            builders.CodeBuildBuilder,
            "_log_tail",
            return_value="BENCHFLOW_IMAGE_TOO_LARGE 3200",
        ):
            error = builder._build_failure(build, "repo:tag")

        assert isinstance(error, SandboxStartupError)
        assert "2048" in str(error)

    def test_context_is_zipped_not_tarred(self, tmp_path):
        """CodeBuild's S3 source only unpacks ZIP.

        A tar.gz is downloaded verbatim, leaving the build directory holding
        the archive itself — which surfaces as a confusing "Dockerfile ... no
        such file or directory" from docker build.
        """
        builder = builders.CodeBuildBuilder(MagicMock(), "1", "us-west-2")

        blob = builder._package(_request(tmp_path))

        assert zipfile.is_zipfile(BytesIO(blob))


class TestDockerIgnoreParity:
    """The remote path must see the same context Docker would.

    The local daemon honors .dockerignore natively; the CodeBuild path builds
    its own file list. When those diverged, ignored files — including secrets —
    were zipped and uploaded to S3, and an ignored file also perturbed the
    image identity.
    """

    def _context(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
        (tmp_path / "keep.txt").write_text("keep me")
        (tmp_path / "secret.env").write_text("SUPER_SECRET=hunter2")
        (tmp_path / "cache").mkdir()
        (tmp_path / "cache" / "big.bin").write_text("x" * 64)
        (tmp_path / ".dockerignore").write_text("secret.env\ncache/\n")
        return tmp_path

    def test_ignored_files_are_not_uploaded(self, tmp_path):
        context = self._context(tmp_path)
        builder = builders.CodeBuildBuilder(MagicMock(), "1", "us-west-2")

        blob = builder._package(_request(context))

        with zipfile.ZipFile(BytesIO(blob)) as archive:
            names = set(archive.namelist())
            payloads = b"".join(archive.read(n) for n in names)

        assert "secret.env" not in names
        assert not any(n.startswith("cache/") for n in names)
        assert b"hunter2" not in payloads
        # Positive control: the archive is real and non-ignored files survive.
        assert "keep.txt" in names

    def test_ignored_files_do_not_change_image_identity(self, tmp_path):
        """An ignored file cannot affect the build, so it must not affect the tag."""
        from benchflow.sandbox import agentcore_provisioning as provisioning

        context = self._context(tmp_path)
        before = provisioning.build_context_digest(context, "FROM python:3.12-slim\n")

        (context / "secret.env").write_text("SUPER_SECRET=rotated")

        assert (
            provisioning.build_context_digest(context, "FROM python:3.12-slim\n")
            == before
        )

    def test_negation_re_includes_a_file(self, tmp_path):
        """`!pattern` is Docker's re-include rule; last match wins."""
        from benchflow.sandbox import agentcore_provisioning as provisioning

        (tmp_path / "Dockerfile").write_text("FROM scratch\n")
        (tmp_path / "a.log").write_text("drop")
        (tmp_path / "keep.log").write_text("keep")
        (tmp_path / ".dockerignore").write_text("*.log\n!keep.log\n")

        relatives = {rel for _p, rel in provisioning.iter_context_files(tmp_path)}

        assert "a.log" not in relatives
        assert "keep.log" in relatives

    def test_character_classes_are_honoured(self, tmp_path):
        """`secret[0-9].pem` is a valid Docker pattern; missing it leaks to S3.

        Guards PR #937.
        """
        from benchflow.sandbox import agentcore_provisioning as provisioning

        (tmp_path / "Dockerfile").write_text("FROM scratch\n")
        for name in ("secret1.pem", "secret9.pem", "secretX.pem"):
            (tmp_path / name).write_text("blob")
        (tmp_path / ".dockerignore").write_text("secret[0-9].pem\n")

        relatives = {rel for _p, rel in provisioning.iter_context_files(tmp_path)}

        assert "secret1.pem" not in relatives
        assert "secret9.pem" not in relatives
        # Positive control: the class must not over-match.
        assert "secretX.pem" in relatives

    @pytest.mark.parametrize(
        ("pattern", "secret_name"),
        [
            ("foo/../secret.env", "secret.env"),
            (r"\!secret.env", "!secret.env"),
        ],
    )
    def test_docker_path_cleaning_and_escapes_do_not_leak(
        self, tmp_path, pattern, secret_name
    ):
        """Guards PR #937 with patterns verified against Docker 29.3.0."""
        (tmp_path / "Dockerfile").write_text("FROM scratch\n")
        (tmp_path / ".dockerignore").write_text(pattern + "\n")
        (tmp_path / secret_name).write_text("DO_NOT_UPLOAD")

        blob = builders.CodeBuildBuilder(MagicMock(), "1", "us-west-2")._package(
            _request(tmp_path)
        )

        with zipfile.ZipFile(BytesIO(blob)) as archive:
            assert secret_name not in archive.namelist()
            assert b"DO_NOT_UPLOAD" not in b"".join(
                archive.read(name)
                for name in archive.namelist()
                if not name.endswith("/")
            )

    def test_generated_dockerfile_specific_ignore_takes_precedence(self, tmp_path):
        """Guards PR #937: Dockerfile-specific ignore overrides the root file."""
        from benchflow.sandbox import agentcore_provisioning as provisioning

        (tmp_path / "Dockerfile").write_text("FROM scratch\n")
        (tmp_path / "secret.env").write_text("DO_NOT_UPLOAD")
        (tmp_path / ".dockerignore").write_text("!secret.env\n")
        (tmp_path / f"{provisioning.GENERATED_DOCKERFILE}.dockerignore").write_text(
            "secret.env\n"
        )

        blob = builders.CodeBuildBuilder(MagicMock(), "1", "us-west-2")._package(
            _request(tmp_path)
        )

        with zipfile.ZipFile(BytesIO(blob)) as archive:
            assert "secret.env" not in archive.namelist()

    def test_permission_bits_change_image_identity(self, tmp_path):
        """0600 and 0644 build different containers from identical bytes.

        Guards PR #937.
        """
        import os

        from benchflow.sandbox import agentcore_provisioning as provisioning

        (tmp_path / "Dockerfile").write_text("FROM scratch\n")
        script = tmp_path / "entrypoint.sh"
        script.write_text("#!/bin/sh\n")

        os.chmod(script, 0o644)
        readable = provisioning.build_context_digest(tmp_path, "DF", "SHIM")
        os.chmod(script, 0o600)
        private = provisioning.build_context_digest(tmp_path, "DF", "SHIM")
        os.chmod(script, 0o755)
        executable = provisioning.build_context_digest(tmp_path, "DF", "SHIM")

        assert len({readable, private, executable}) == 3

    def test_shim_bytes_change_image_identity(self, tmp_path):
        """The shim is the image entrypoint; an upgrade must not reuse the old.

        Guards PR #937.
        """
        from benchflow.sandbox import agentcore_provisioning as provisioning

        (tmp_path / "Dockerfile").write_text("FROM scratch\n")

        assert provisioning.build_context_digest(
            tmp_path, "DF", "shim-v1"
        ) != provisioning.build_context_digest(tmp_path, "DF", "shim-v2")

    def test_identity_fields_cannot_be_confused(self, tmp_path):
        """Length-prefixed framing kills a real digest collision.

        Guards PR #937. The superseded scheme concatenated the Dockerfile, a
        fixed ``\0shim\0`` separator, and the shim with no lengths, so a shim
        containing that separator could be re-split as a different
        (dockerfile, shim) pair with an identical digest — two different images
        sharing one identity, and so one runtime.

        The first assertion pins that the chosen inputs genuinely collided
        before; without it this test could pass against the old scheme too.
        """
        import hashlib

        from benchflow.sandbox import agentcore_provisioning as provisioning

        def superseded_framing(dockerfile: str, shim: str) -> str:
            digest = hashlib.sha256()
            digest.update(dockerfile.encode())
            digest.update(b"\0shim\0")
            digest.update(shim.encode())
            return digest.hexdigest()

        first = ("A", "B\0shim\0C")
        second = ("A\0shim\0B", "C")

        assert superseded_framing(*first) == superseded_framing(*second)
        assert provisioning.build_context_digest(
            tmp_path, *first
        ) != provisioning.build_context_digest(tmp_path, *second)


class TestBuildLifecycle:
    """Remote and local builders fail closed and clean up after themselves."""

    def test_s3_control_conflict_is_retried(self):
        """Guards PR #937: parallel builders hit live AWS OperationAborted."""
        from botocore.exceptions import ClientError

        operation = MagicMock(
            side_effect=[
                ClientError(
                    {
                        "Error": {
                            "Code": "OperationAborted",
                            "Message": "conflicting conditional operation",
                        }
                    },
                    "PutBucketLifecycleConfiguration",
                ),
                None,
            ]
        )

        with (
            patch.object(
                builders._S3_RETRY_JITTER,
                "uniform",
                return_value=builders._S3_CONTROL_RETRY_DELAYS_SEC[0],
            ),
            patch.object(builders.time, "sleep") as sleep,
        ):
            builders.CodeBuildBuilder._retry_s3_mutation(operation, Bucket="build")

        assert operation.call_count == 2
        sleep.assert_called_once_with(builders._S3_CONTROL_RETRY_DELAYS_SEC[0])

    def test_bucket_hardening_preserves_existing_lifecycle_rules(self):
        """Guards PR #937: hardening a shared bucket must not erase its policies."""
        s3 = MagicMock()
        s3.get_bucket_lifecycle_configuration.return_value = {
            "Rules": [
                {
                    "ID": "operator-archive",
                    "Status": "Enabled",
                    "Filter": {"Prefix": "history/"},
                    "Transition": {"Days": 30, "StorageClass": "GLACIER"},
                },
                {
                    "ID": "benchflow-expire-build-contexts",
                    "Status": "Disabled",
                    "Filter": {"Prefix": "old/"},
                    "Expiration": {"Days": 90},
                },
            ]
        }
        builder = builders.CodeBuildBuilder(lambda service: s3, "1", "us-west-2")

        builder._ensure_bucket()

        lifecycle = s3.put_bucket_lifecycle_configuration.call_args.kwargs[
            "LifecycleConfiguration"
        ]
        assert [rule["ID"] for rule in lifecycle["Rules"]] == [
            "operator-archive",
            "benchflow-expire-build-contexts",
        ]
        assert lifecycle["Rules"][0]["Transition"]["StorageClass"] == "GLACIER"
        assert lifecycle["Rules"][1]["Expiration"] == {"Days": 1}

    def test_existing_codebuild_project_is_reconciled(self, monkeypatch):
        """Guards PR #937: stale shared project settings must not run a wrong build."""
        from botocore.exceptions import ClientError

        monkeypatch.setenv(builders.ENV_CODEBUILD_ROLE, "arn:aws:iam::1:role/build")
        codebuild = MagicMock()
        codebuild.create_project.side_effect = ClientError(
            {"Error": {"Code": "ResourceAlreadyExistsException", "Message": "exists"}},
            "CreateProject",
        )
        builder = builders.CodeBuildBuilder(lambda service: codebuild, "1", "us-west-2")

        builder._ensure_project()

        config = codebuild.update_project.call_args.kwargs
        assert config["name"] == builders.CODEBUILD_PROJECT
        assert config["environment"]["type"] == "ARM_CONTAINER"
        assert config["environment"]["privilegedMode"] is True
        assert config["serviceRole"] == "arn:aws:iam::1:role/build"

    @pytest.mark.asyncio
    async def test_unmeasurable_image_is_not_pushed(self, tmp_path):
        """Failing open pushes an image that may exceed the hard 2 GB cap.

        Guards PR #937.
        """
        from benchflow.sandbox.protocol import SandboxStartupError

        builder = builders.LocalDockerBuilder(MagicMock())

        with (
            patch.object(
                builders,
                "_run",
                return_value=MagicMock(returncode=1, stdout="", stderr="boom"),
            ),
            pytest.raises(SandboxStartupError, match="Could not measure"),
        ):
            await builder._reject_oversized("repo:tag")

    @pytest.mark.asyncio
    async def test_failed_context_cleanup_is_warned_not_hidden(self, tmp_path, caplog):
        """Guards PR #937: a retained context can hold source and credentials.

        Bucket hardening and the one-day lifecycle bound the exposure, so this
        must not fail the build — but at DEBUG the retained object is invisible
        for that entire day.
        """
        import logging

        builder = builders.CodeBuildBuilder(MagicMock(), "1", "us-west-2")
        s3 = MagicMock()
        s3.delete_object.side_effect = RuntimeError("AccessDenied")

        with (
            patch.object(builder, "_client", return_value=s3),
            patch.object(builder, "_ensure_bucket"),
            patch.object(builder, "_ensure_project"),
            patch.object(builder, "_run_build", new=AsyncMock()),
            caplog.at_level(logging.WARNING, logger="benchflow.agentcore-builder"),
        ):
            await builder.build_and_push(_request(tmp_path))

        assert any(
            "Could not delete uploaded build context" in r.message
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_failure_diagnostics_do_not_block_the_event_loop(self, tmp_path):
        """Guards PR #937: one failed build must not stall other provisioning.

        `_build_failure` fetches CloudWatch logs; running it inline on the loop
        froze every concurrent coroutine for the duration of that fetch.
        """
        builder = builders.CodeBuildBuilder(MagicMock(), "1", "us-west-2")
        codebuild = MagicMock()
        codebuild.start_build.return_value = {"build": {"id": "b:1"}}
        codebuild.batch_get_builds.return_value = {
            "builds": [{"id": "b:1", "buildStatus": "FAILED", "phases": []}]
        }

        def _slow_log_tail(_self, _build):
            time.sleep(0.25)
            return "boom"

        ticks = 0

        async def _ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        with (
            patch.object(builder, "_client", return_value=codebuild),
            patch.object(builders, "_CODEBUILD_POLL_SEC", 0),
            patch.object(builders.CodeBuildBuilder, "_log_tail", _slow_log_tail),
        ):
            beat = asyncio.create_task(_ticker())
            with pytest.raises(RuntimeError, match="CodeBuild"):
                await builder._run_build(_request(tmp_path), "contexts/x.zip")
            beat.cancel()

        # A blocking diagnostic starved the ticker to ~1 tick.
        assert ticks > 5

    @pytest.mark.asyncio
    async def test_ambiguous_upload_is_still_deleted(self, tmp_path):
        """Guards PR #937 when S3 commits the upload but loses its response."""
        builder = builders.CodeBuildBuilder(MagicMock(), "1", "us-west-2")
        s3 = MagicMock()
        s3.put_object.side_effect = TimeoutError("response lost")

        with (
            patch.object(builder, "_client", return_value=s3),
            patch.object(builder, "_ensure_bucket"),
            pytest.raises(TimeoutError, match="response lost"),
        ):
            await builder.build_and_push(_request(tmp_path))

        s3.delete_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancelled_upload_finishes_before_context_deletion(self, tmp_path):
        """Guards PR #937 against to_thread upload/delete reordering."""
        builder = builders.CodeBuildBuilder(MagicMock(), "1", "us-west-2")
        s3 = MagicMock()
        entered = threading.Event()
        release = threading.Event()
        calls: list[str] = []

        def _put(**_kwargs):
            entered.set()
            release.wait(timeout=10)
            calls.append("put")

        s3.put_object.side_effect = _put
        s3.delete_object.side_effect = lambda **_kwargs: calls.append("delete")

        with (
            patch.object(builder, "_client", return_value=s3),
            patch.object(builder, "_ensure_bucket"),
        ):
            task = asyncio.create_task(builder.build_and_push(_request(tmp_path)))
            await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert calls == ["put", "delete"]

    @pytest.mark.asyncio
    async def test_timeout_stops_the_remote_build(self, tmp_path):
        """Guards PR #937: a timed-out build must not keep running and push."""
        builder = builders.CodeBuildBuilder(MagicMock(), "1", "us-west-2")
        codebuild = MagicMock()
        codebuild.start_build.return_value = {"build": {"id": "b:timeout"}}
        request = _request(tmp_path, timeout_sec=1)

        with (
            patch.object(builder, "_client", return_value=codebuild),
            patch.object(
                builders,
                "time",
                MagicMock(monotonic=MagicMock(side_effect=[0.0, 2.0])),
            ),
            pytest.raises(TimeoutError, match="did not finish"),
        ):
            await builder._run_build(request, "contexts/x.zip")

        codebuild.stop_build.assert_called_once_with(id="b:timeout")

    @pytest.mark.asyncio
    async def test_cancellation_stops_build_before_deleting_context(self, tmp_path):
        """Guards PR #937: stop the consumer before deleting its S3 source."""
        builder = builders.CodeBuildBuilder(MagicMock(), "1", "us-west-2")
        s3 = MagicMock()
        codebuild = MagicMock()
        codebuild.start_build.return_value = {"build": {"id": "b:cancel"}}
        codebuild.batch_get_builds.return_value = {
            "builds": [{"id": "b:cancel", "buildStatus": "IN_PROGRESS"}]
        }
        calls: list[str] = []
        codebuild.stop_build.side_effect = lambda **_kwargs: calls.append("stop")
        s3.delete_object.side_effect = lambda **_kwargs: calls.append("delete")

        def _client(service):
            return codebuild if service == "codebuild" else s3

        with (
            patch.object(builder, "_client", side_effect=_client),
            patch.object(builder, "_ensure_bucket"),
            patch.object(builder, "_ensure_project"),
            patch.object(builders, "_CODEBUILD_POLL_SEC", 0),
        ):
            task = asyncio.create_task(builder.build_and_push(_request(tmp_path)))
            while not codebuild.batch_get_builds.called:
                await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert calls == ["stop", "delete"]

    @pytest.mark.asyncio
    async def test_cancellation_during_start_recovers_id_and_stops_build(
        self, tmp_path
    ):
        """Guards PR #937 when StartBuild succeeds after local cancellation."""
        builder = builders.CodeBuildBuilder(MagicMock(), "1", "us-west-2")
        codebuild = MagicMock()
        entered = threading.Event()
        release = threading.Event()

        def _start(**_kwargs):
            entered.set()
            release.wait(timeout=10)
            return {"build": {"id": "b:start-race"}}

        codebuild.start_build.side_effect = _start

        with patch.object(builder, "_client", return_value=codebuild):
            task = asyncio.create_task(
                builder._run_build(_request(tmp_path), "contexts/x.zip")
            )
            await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        codebuild.stop_build.assert_called_once_with(id="b:start-race")
