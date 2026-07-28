"""Image build strategy selection for the AgentCore backend.

AgentCore only runs images that already exist in ECR, so something must build
one. Requiring a local Docker daemon defeats the point of a cloud backend on a
machine that has no room to run containers, so the builder falls back to AWS
CodeBuild on a Graviton worker when no daemon is reachable.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from unittest.mock import MagicMock, patch

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
