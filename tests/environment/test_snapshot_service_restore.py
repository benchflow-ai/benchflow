"""Service lifecycle regressions for the selective composed PR #1046 port."""

from unittest.mock import AsyncMock

import pytest

from benchflow.environment.manifest import EnvironmentManifest
from benchflow.environment.manifest_env import ManifestEnvironment
from benchflow.sandbox.protocol import ExecResult


def environment():
    sandbox = type(
        "Sandbox",
        (),
        {
            "exec": AsyncMock(
                return_value=ExecResult(return_code=0, stdout="", stderr="")
            )
        },
    )()
    manifest = EnvironmentManifest.model_validate(
        {
            "name": "service",
            "image": "test",
            "owns_lifecycle": False,
            "services": [{"name": "api", "command": "api-server", "port": 9000}],
            "readiness": {"timeout_sec": 1},
        }
    )
    return ManifestEnvironment(manifest, sandbox=sandbox), sandbox


async def test_container_restore_restarts_services_then_checks_readiness():
    """PR #1046 integration must restart processes that Docker commit cannot save."""
    env, sandbox = environment()
    old_handle = await env.provision(None)
    baseline = env._baseline
    sandbox.exec.reset_mock()
    await env.resume_after_sandbox_restore()
    commands = [call.args[0] for call in sandbox.exec.call_args_list]
    assert "api-server" in commands[0]
    assert "curl -sf" in commands[1]
    assert env._handle is not old_handle
    assert env._baseline is baseline


async def test_readiness_failure_invalidates_handle_and_can_retry():
    """PR #1046: an unreachable restarted service must not permit child execution."""
    env, sandbox = environment()
    await env.provision(None)
    sandbox.exec.return_value = ExecResult(return_code=1, stdout="", stderr="")
    with pytest.raises(RuntimeError, match="not ready"):
        await env.resume_after_sandbox_restore()
    assert env._handle is None
    sandbox.exec.return_value = ExecResult(return_code=0, stdout="", stderr="")
    await env.resume_after_sandbox_restore()
    assert env._handle is not None


def test_unprovisioned_framework_services_rejected():
    """PR #1046: unknown started-service set must not pass readiness vacuously."""
    env, _ = environment()
    with pytest.raises(RuntimeError, match="provisioned"):
        env.validate_sandbox_restore()


async def test_prepare_invalidates_handle_before_restore_can_fail():
    """PR #1046: failed database copying must not retain a stale process handle."""
    env, _ = environment()
    await env.provision(None)
    env.prepare_sandbox_restore()
    assert env._handle is None
    assert env._started  # Retain the service restart recipe, not live state.
