"""Importing ``benchflow.rollout`` must not mutate Docker sandbox behavior.

Regression test for #421: a module-level monkey patch caused
``DockerSandboxEnvVars.to_env_dict`` to be replaced as a side effect of
``import benchflow.rollout``. The shim is now activated only when a
``Rollout`` is constructed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a fresh subprocess.

    Each call gets a clean module-import context — no pollution from the
    pytest process where ``benchflow.rollout`` may already be loaded.
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_importing_rollout_does_not_patch_docker_sandbox_env() -> None:
    """``import benchflow.rollout`` must leave ``to_env_dict`` untouched."""
    result = _run_python(
        """
        from benchflow.sandbox.docker import DockerSandboxEnvVars

        before = DockerSandboxEnvVars.to_env_dict

        import benchflow.rollout  # noqa: F401

        after = DockerSandboxEnvVars.to_env_dict

        assert before is after, (
            "benchflow.rollout import mutated "
            "DockerSandboxEnvVars.to_env_dict — module-level side effect leaked"
        )
        print("OK")
        """
    )
    assert result.returncode == 0, (
        f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "OK" in result.stdout


def test_importing_rollout_does_not_flag_dind_patch_applied() -> None:
    """The setup-module flag should remain False after a bare import."""
    result = _run_python(
        """
        import benchflow.rollout  # noqa: F401
        from benchflow.sandbox import setup

        assert setup._DIND_PATCH_APPLIED is False, (
            "rollout import set _DIND_PATCH_APPLIED — patch leaked at import time"
        )
        print("OK")
        """
    )
    assert result.returncode == 0, (
        f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "OK" in result.stdout


def test_install_docker_compat_is_idempotent_and_explicit() -> None:
    """Calling the explicit compat hook twice must not double-wrap."""
    result = _run_python(
        """
        from unittest.mock import patch

        from benchflow.sandbox.docker import DockerSandboxEnvVars
        from benchflow.sandbox import setup as sandbox_setup
        from benchflow.rollout import _install_docker_compat

        original = DockerSandboxEnvVars.to_env_dict

        # Force the DinD-detection branch so the patch actually applies.
        with patch.object(
            sandbox_setup,
            "_detect_dind_mount",
            return_value=("/host/work", "/workspaces"),
        ):
            _install_docker_compat()
            after_first = DockerSandboxEnvVars.to_env_dict
            _install_docker_compat()
            after_second = DockerSandboxEnvVars.to_env_dict

        assert after_first is not original, "first call should install the wrapper"
        assert after_second is after_first, "second call must be a no-op (idempotent)"
        assert sandbox_setup._DIND_PATCH_APPLIED is True
        print("OK")
        """
    )
    assert result.returncode == 0, (
        f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "OK" in result.stdout


def test_without_dind_environment_patch_does_not_apply() -> None:
    """When ``_detect_dind_mount`` returns None, ``to_env_dict`` stays original."""
    result = _run_python(
        """
        from unittest.mock import patch

        from benchflow.sandbox.docker import DockerSandboxEnvVars
        from benchflow.sandbox import setup as sandbox_setup
        from benchflow.rollout import _install_docker_compat

        original = DockerSandboxEnvVars.to_env_dict

        with patch.object(
            sandbox_setup, "_detect_dind_mount", return_value=None
        ):
            _install_docker_compat()

        assert DockerSandboxEnvVars.to_env_dict is original, (
            "no DinD mount detected — patch should not be installed"
        )
        assert sandbox_setup._DIND_PATCH_APPLIED is False
        print("OK")
        """
    )
    assert result.returncode == 0, (
        f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "OK" in result.stdout
