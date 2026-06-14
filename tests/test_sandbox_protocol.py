"""Tests for benchflow.sandbox — protocol types and Sandbox conformance.

Validates that sandbox implementations satisfy the Sandbox protocol.
Since DockerSandbox and DaytonaSandbox are now full implementations
(not thin adapter wrappers), protocol conformance is tested via
structural subtyping checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.sandbox.protocol import ExecResult, ImageConfig, ImageRef

# Dataclass tests


class TestExecResult:
    def test_fields(self):
        r = ExecResult(return_code=0, stdout="ok", stderr="")
        assert r.return_code == 0
        assert r.stdout == "ok"
        assert r.stderr == ""

    def test_stdout_stderr_optional(self):
        """``stdout``/``stderr`` default to ``None`` — backends decode empty
        process output to ``None`` rather than ``""``, so the public contract
        type must accept that."""
        r = ExecResult(return_code=0)
        assert r.stdout is None
        assert r.stderr is None

    def test_is_the_backend_exec_result(self):
        """The public ``ExecResult`` is the same object the backends return —
        a single contract, not two incompatible types."""
        from benchflow.sandbox._base import ExecResult as BaseExecResult

        assert ExecResult is BaseExecResult


class TestImageRef:
    def test_defaults(self):
        ref = ImageRef(tag="latest")
        assert ref.tag == "latest"
        assert ref.digest is None

    def test_with_digest(self):
        ref = ImageRef(tag="v1", digest="sha256:abc")
        assert ref.digest == "sha256:abc"

    def test_frozen(self):
        ref = ImageRef(tag="v1")
        with pytest.raises(AttributeError):
            ref.tag = "v2"  # type: ignore[misc]


class TestImageConfig:
    def test_required_fields(self):
        cfg = ImageConfig(dockerfile=Path("Dockerfile"), context_dir=Path("."))
        assert cfg.dockerfile == Path("Dockerfile")
        assert cfg.context_dir == Path(".")
        assert cfg.build_args is None
        assert cfg.cache_key is None

    def test_optional_fields(self):
        cfg = ImageConfig(
            dockerfile=Path("Dockerfile"),
            context_dir=Path("."),
            build_args={"FOO": "bar"},
            cache_key="my-key",
        )
        assert cfg.build_args == {"FOO": "bar"}
        assert cfg.cache_key == "my-key"

    def test_mutable(self):
        cfg = ImageConfig(dockerfile=Path("Dockerfile"), context_dir=Path("."))
        cfg.cache_key = "new-key"
        assert cfg.cache_key == "new-key"


# Protocol conformance


class TestDockerSandboxProtocol:
    """Verify DockerSandbox is a structural subtype of Sandbox."""

    def test_has_sandbox_interface(self):
        from benchflow.sandbox.docker import DockerSandbox

        for attr in (
            "exec",
            "start",
            "stop",
            "upload_file",
            "download_file",
            "upload_dir",
            "download_dir",
        ):
            assert hasattr(DockerSandbox, attr), f"DockerSandbox missing {attr}"

    def test_has_required_methods(self):
        from benchflow.sandbox.docker import DockerSandbox

        assert hasattr(DockerSandbox, "exec")
        assert hasattr(DockerSandbox, "start")
        assert hasattr(DockerSandbox, "stop")
        assert hasattr(DockerSandbox, "upload_file")
        assert hasattr(DockerSandbox, "download_file")

    def test_exec_is_async(self):
        import asyncio

        from benchflow.sandbox.docker import DockerSandbox

        assert asyncio.iscoroutinefunction(DockerSandbox.exec)

    def test_start_is_async(self):
        import asyncio

        from benchflow.sandbox.docker import DockerSandbox

        assert asyncio.iscoroutinefunction(DockerSandbox.start)

    def test_stop_is_async(self):
        import asyncio

        from benchflow.sandbox.docker import DockerSandbox

        assert asyncio.iscoroutinefunction(DockerSandbox.stop)

    def test_upload_file_is_async(self):
        import asyncio

        from benchflow.sandbox.docker import DockerSandbox

        assert asyncio.iscoroutinefunction(DockerSandbox.upload_file)

    def test_inherits_base_sandbox(self):
        from benchflow.sandbox._base import BaseSandbox
        from benchflow.sandbox.docker import DockerSandbox

        assert issubclass(DockerSandbox, BaseSandbox)


_daytona_available = True
try:
    import daytona as _daytona_mod  # noqa: F401
except ImportError:
    _daytona_available = False


@pytest.mark.skipif(not _daytona_available, reason="daytona not installed")
class TestDaytonaSandboxProtocol:
    """Verify DaytonaSandbox is a structural subtype of Sandbox."""

    def test_has_sandbox_interface(self):
        from benchflow.sandbox.daytona import DaytonaSandbox

        for attr in (
            "exec",
            "start",
            "stop",
            "upload_file",
            "upload_dir",
            "download_file",
            "download_dir",
        ):
            assert hasattr(DaytonaSandbox, attr), f"DaytonaSandbox missing {attr}"

    def test_has_required_methods(self):
        from benchflow.sandbox.daytona import DaytonaSandbox

        assert hasattr(DaytonaSandbox, "exec")
        assert hasattr(DaytonaSandbox, "start")
        assert hasattr(DaytonaSandbox, "stop")
        assert hasattr(DaytonaSandbox, "upload_file")

    def test_exec_is_async(self):
        import asyncio

        from benchflow.sandbox.daytona import DaytonaSandbox

        assert asyncio.iscoroutinefunction(DaytonaSandbox.exec)

    def test_start_is_async(self):
        import asyncio

        from benchflow.sandbox.daytona import DaytonaSandbox

        assert asyncio.iscoroutinefunction(DaytonaSandbox.start)

    def test_stop_is_async(self):
        import asyncio

        from benchflow.sandbox.daytona import DaytonaSandbox

        assert asyncio.iscoroutinefunction(DaytonaSandbox.stop)

    def test_inherits_base_sandbox(self):
        from benchflow.sandbox._base import BaseSandbox
        from benchflow.sandbox.daytona import DaytonaSandbox

        assert issubclass(DaytonaSandbox, BaseSandbox)


# ExecResult edge cases


class TestNoneToEmptyString:
    def test_exec_result_with_empty_fields(self):
        """ExecResult with empty string fields."""
        r = ExecResult(return_code=0, stdout="", stderr="")
        assert r.stdout == ""
        assert r.stderr == ""

    def test_exec_result_with_content(self):
        r = ExecResult(return_code=0, stdout="hello", stderr="")
        assert r.stdout == "hello"


class TestExecResultReturnCode:
    def test_exec_result_failure_code(self):
        """ExecResult with non-zero return_code indicates failure."""
        r = ExecResult(return_code=1, stdout="", stderr="No such file")
        assert r.return_code != 0
        assert "No such file" in r.stderr

    def test_exec_result_success_code(self):
        r = ExecResult(return_code=0, stdout="file contents", stderr="")
        assert r.return_code == 0
