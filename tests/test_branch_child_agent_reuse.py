"""An isolated branch child reuses the agent its snapshot already contains.

An isolated branch child reinstalled the agent (npm install) although its
sandbox was created from a snapshot that already contained it. A sub-rollout now carries the parent's installed
agent config and skips only the binary install; sandbox user, credential
files, skills and lockdown still run (credential files are scrubbed from
snapshots, so they must be written again).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.test_branch_snapshot_start import _probe_env
from tests.test_trial_install_agent_timeout import _make_trial


@pytest.mark.asyncio
async def test_a_child_with_the_parents_agent_skips_only_the_binary_install(
    tmp_path, monkeypatch
):
    trial = _make_trial(tmp_path, agent="claude-agent-acp", sandbox_setup_timeout=41)
    parent_cfg = MagicMock(name="parent agent config")
    parent_cfg.launch_cmd = "/opt/benchflow/bin/claude-agent-acp"
    trial._installed_agent_cfg = parent_cfg
    # The reuse is verified: the child must come from a
    # branch snapshot and the binary must be present (test_branch_snapshot_start).
    trial._from_branch_snapshot = True
    trial._env = _probe_env(agent=True, baseline=True)
    install = AsyncMock()
    credentials = AsyncMock()
    setup_user = AsyncMock(return_value="/home/agent")
    monkeypatch.setattr(trial._planes, "install_agent", install)
    monkeypatch.setattr(trial._planes, "write_credential_files", credentials)
    monkeypatch.setattr(trial._planes, "setup_sandbox_user", setup_user)
    for name in (
        "upload_subscription_auth",
        "snapshot_build_config",
        "seed_verifier_workspace",
        "deploy_skills",
        "lockdown_paths",
        "prepare_log_dirs",
    ):
        monkeypatch.setattr(trial._planes, name, AsyncMock())

    await trial.install_agent()

    install.assert_not_awaited()
    assert trial._agent_cfg is parent_cfg
    credentials.assert_awaited_once()
    setup_user.assert_awaited_once()


async def test_isolated_children_carry_the_parents_agent_config(tmp_path):
    from tests.test_branch_isolated import IMAGES, IsoRollout, _root, _runner

    IMAGES.clear()
    IsoRollout.all = []
    root = await _root(tmp_path)
    root._agent_cfg = MagicMock(name="installed")
    await root.branch(
        2,
        _runner({"a": "Do it.", "b": "Do it."}),
        snapshot_layers={"sandbox"},
        child_labels=["a", "b"],
        isolate_children=True,
    )
    subs = IsoRollout.all[1:]
    assert [sub._installed_agent_cfg for sub in subs] == [root._agent_cfg] * 2
