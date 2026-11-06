"""Guard FF-install-C9: cd4d492d rejected honest /opt workspaces."""

from pathlib import PurePosixPath

import pytest

from benchflow import _uv_pin
from benchflow.sandbox import env_layers, lockdown
from tests.test_root_only_tools import _RecordingEnv


def test_privileged_tool_storage_is_outside_normal_opt_workspaces():
    # Every privileged tool is in the reserved runtime tree, which an /opt
    # workspace's grant leaves root's (PT's workspace grant).
    from benchflow import _runtime_tree

    root = PurePosixPath(_runtime_tree.RUNTIME_ROOT)
    for path in (
        _uv_pin.UV_DIR,
        _uv_pin.PYTHON_INSTALL_DIR,
        _uv_pin.UV_CACHE_DIR,
        env_layers.PROXY_LAYER_VENV,
        env_layers.PROXY_LAYER_STAMP,
    ):
        assert PurePosixPath(path).is_relative_to(root)
    assert _runtime_tree.workspace_grant_problem("/opt") is None
    grant = _runtime_tree.workspace_grant_command("/opt", "agent")
    assert f"! -path {root}" in grant and "chown -R agent:agent /opt" not in grant


@pytest.mark.asyncio
async def test_oracle_workspace_needs_no_unused_runtime_reservation():
    env = _RecordingEnv()
    await lockdown.setup_sandbox_user(env, "agent", "/opt/project", guard_daemon=False)
    assert any("chown -R agent:agent /opt/project" in c for c in env.commands)
