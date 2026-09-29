"""A rollout started from a branch snapshot keeps what the snapshot holds.

An isolated branch child spent most of install_agent in
``snapshot_build_config`` and ``seed_verifier_workspace``. Both re-capture the verifier's *pre-agent* baseline (build-config files and the
/testbed_verify copy). In a child those already exist in the snapshot, taken
before the parent's agent ran; re-capturing them records the parent agent's
post-fork workspace as "pre-agent", so a build file the parent agent
tampered with before the fork would be restored as legitimate at
verification. In-place children never re-run install_agent and keep the
original baseline; isolated children and --from-checkpoint trials now do the
same. Likewise the task's setup_commands already ran before the snapshot and
are not re-run on the checkpoint state.

The agent binary is reused only after checking it is present; a missing
binary or baseline falls back to installing/re-capturing, and the child's
record says which happened (``snapshot_start``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox.protocol import ExecResult
from tests.test_trial_install_agent_timeout import _make_trial


def _probe_env(*, agent: bool, baseline: bool) -> MagicMock:
    env = MagicMock()

    async def exec_(cmd, **_):
        if "benchflow-snapshot-start" in cmd:
            lines = (["agent"] if agent else []) + (["baseline"] if baseline else [])
            return ExecResult(stdout="\n".join(lines), stderr="", return_code=0)
        return ExecResult(stdout="/workspace\n", stderr="", return_code=0)

    env.exec = AsyncMock(side_effect=exec_)
    return env


def _patch_planes(trial, monkeypatch) -> dict[str, AsyncMock]:
    mocks = {}
    for name in (
        "install_agent",
        "write_credential_files",
        "upload_subscription_auth",
        "snapshot_build_config",
        "seed_verifier_workspace",
        "prepare_log_dirs",
        "deploy_skills",
        "lockdown_paths",
        "apply_web_tool_policy",
    ):
        mocks[name] = AsyncMock()
        monkeypatch.setattr(trial._planes, name, mocks[name])
    mocks["setup_sandbox_user"] = AsyncMock(return_value="/home/agent")
    monkeypatch.setattr(
        trial._planes, "setup_sandbox_user", mocks["setup_sandbox_user"]
    )
    return mocks


def _child(tmp_path, *, agent: bool, baseline: bool, agent_name="claude-agent-acp"):
    trial = _make_trial(tmp_path, agent=agent_name, sandbox_setup_timeout=41)
    trial._env = _probe_env(agent=agent, baseline=baseline)
    trial._from_branch_snapshot = True
    trial._installed_agent_cfg = SimpleNamespace(
        name=agent_name, launch_cmd="/opt/benchflow/bin/claude-agent-acp"
    )
    return trial


@pytest.mark.asyncio
async def test_everything_present_is_reused(tmp_path, monkeypatch):
    trial = _child(tmp_path, agent=True, baseline=True)
    mocks = _patch_planes(trial, monkeypatch)
    await trial.install_agent()
    mocks["install_agent"].assert_not_awaited()
    mocks["snapshot_build_config"].assert_not_awaited()
    mocks["seed_verifier_workspace"].assert_not_awaited()
    # Still per sandbox: log dir ownership, credentials (scrubbed from
    # snapshots), sandbox user, lockdown.
    mocks["prepare_log_dirs"].assert_awaited_once()
    mocks["write_credential_files"].assert_awaited_once()
    mocks["setup_sandbox_user"].assert_awaited_once()
    mocks["lockdown_paths"].assert_awaited_once()
    assert trial._agent_cfg is trial._installed_agent_cfg
    assert trial._snapshot_start == {
        "agent": "reused",
        "verifier_baseline": "inherited",
        "setup_commands": "skipped",
    }


@pytest.mark.asyncio
async def test_a_missing_binary_is_installed(tmp_path, monkeypatch):
    trial = _child(tmp_path, agent=False, baseline=True)
    mocks = _patch_planes(trial, monkeypatch)
    await trial.install_agent()
    mocks["install_agent"].assert_awaited_once()
    assert trial._snapshot_start["agent"] == "installed"


@pytest.mark.asyncio
async def test_a_missing_baseline_is_recaptured_and_reported(tmp_path, monkeypatch):
    trial = _child(tmp_path, agent=True, baseline=False)
    mocks = _patch_planes(trial, monkeypatch)
    await trial.install_agent()
    mocks["snapshot_build_config"].assert_awaited_once()
    mocks["seed_verifier_workspace"].assert_awaited_once()
    assert trial._snapshot_start["verifier_baseline"] == "recaptured"


@pytest.mark.asyncio
async def test_oracle_child_keeps_the_baseline_too(tmp_path, monkeypatch):
    trial = _child(tmp_path, agent=False, baseline=True, agent_name="oracle")
    trial._installed_agent_cfg = None
    mocks = _patch_planes(trial, monkeypatch)
    await trial.install_agent()
    mocks["snapshot_build_config"].assert_not_awaited()
    mocks["seed_verifier_workspace"].assert_not_awaited()
    mocks["prepare_log_dirs"].assert_awaited_once()
    assert trial._snapshot_start["verifier_baseline"] == "inherited"


@pytest.mark.asyncio
async def test_a_normal_rollout_is_unchanged(tmp_path, monkeypatch):
    trial = _make_trial(tmp_path, agent="claude-agent-acp", sandbox_setup_timeout=41)
    mocks = _patch_planes(trial, monkeypatch)
    await trial.install_agent()
    mocks["install_agent"].assert_awaited_once()
    mocks["snapshot_build_config"].assert_awaited_once()
    mocks["seed_verifier_workspace"].assert_awaited_once()
    mocks["prepare_log_dirs"].assert_not_awaited()
    assert trial._snapshot_start is None


@pytest.mark.asyncio
async def test_setup_commands_do_not_rerun_on_a_snapshot(tmp_path, monkeypatch):
    import benchflow.rollout as rollout_mod

    ran = []

    async def fake_setup(env, task):
        ran.append(task)

    monkeypatch.setattr(rollout_mod, "_run_environment_setup_commands", fake_setup)
    monkeypatch.setattr(rollout_mod, "_start_env_and_upload", AsyncMock())
    monkeypatch.setattr(rollout_mod, "_run_environment_healthcheck", AsyncMock())
    normal = _make_trial(tmp_path, agent="oracle", sandbox_setup_timeout=1)
    await normal.start()
    assert len(ran) == 1
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child = _make_trial(child_dir, agent="oracle", sandbox_setup_timeout=1)
    child._from_branch_snapshot = True
    await child.start()
    assert len(ran) == 1


@pytest.mark.asyncio
async def test_isolated_children_start_from_the_snapshot_and_record_it(tmp_path):
    from tests.test_branch_isolated import IMAGES, IsoRollout, _root, _runner, _tree

    IMAGES.clear()
    IsoRollout.all = []
    root = await _root(tmp_path)
    await root.branch(
        2,
        _runner({"a": "Do it.", "b": "Do it."}),
        snapshot_layers={"sandbox"},
        child_labels=["a", "b"],
        isolate_children=True,
    )
    assert all(sub._from_branch_snapshot for sub in IsoRollout.all[1:])
    assert not root._from_branch_snapshot
    for child in _tree(root)["forks"][0]["children"]:
        assert "snapshot_start" in child


@pytest.mark.asyncio
async def test_from_checkpoint_trial_is_marked_and_records_it(tmp_path):
    from benchflow.branch_run import load_checkpoint_source, run_branch_trial
    from tests.test_branch_run import ScriptedRollout, _kept_trial, _plan

    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("Do it.")
    source = load_checkpoint_source(_kept_trial(tmp_path), None)
    plan = _plan(
        tmp_path,
        task_paths=[task],
        checkpoint_after=0,
        parent_mode="discard",
        source=source,
    )
    seen = {}

    class Marked(ScriptedRollout):
        async def install_agent(self):
            seen["flag"] = self._from_branch_snapshot
            self._snapshot_start = {"agent": "installed"}
            await super().install_agent()

    await run_branch_trial(plan, task, rollout_factory=Marked)
    assert seen["flag"] is True
    import json

    recorded = json.loads(
        (ScriptedRollout.last._rollout_dir / "checkpoint_source.json").read_text()
    )
    assert recorded["snapshot_start"] == {"agent": "installed"}


@pytest.mark.asyncio
async def test_task_files_are_not_uploaded_again_into_a_snapshot(tmp_path, monkeypatch):
    """A child sandbox created from a branch snapshot already holds
    /instruction.md, the solution and any uploads; re-uploading them only
    slowed the child's start."""
    import benchflow.rollout as rollout_mod

    calls = []

    async def fake_start(env, task_path, timing, **kwargs):
        calls.append(kwargs.get("upload_task_files", True))

    monkeypatch.setattr(rollout_mod, "_start_env_and_upload", fake_start)
    monkeypatch.setattr(rollout_mod, "_run_environment_setup_commands", AsyncMock())
    monkeypatch.setattr(rollout_mod, "_run_environment_healthcheck", AsyncMock())
    normal = _make_trial(tmp_path, agent="oracle", sandbox_setup_timeout=1)
    await normal.start()
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child = _make_trial(child_dir, agent="oracle", sandbox_setup_timeout=1)
    child._from_branch_snapshot = True
    await child.start()
    assert calls == [True, False]


@pytest.mark.asyncio
async def test_start_env_without_uploads_only_starts(tmp_path):
    from benchflow.rollout._setup import _start_env_and_upload

    env = MagicMock()
    env.start = AsyncMock()
    env.upload_file = AsyncMock()
    env.upload_dir = AsyncMock()
    task = tmp_path / "task"
    (task / "solution").mkdir(parents=True)
    (task / "instruction.md").write_text("Do it.")
    await _start_env_and_upload(env, task, {}, upload_task_files=False)
    env.start.assert_awaited_once()
    env.upload_file.assert_not_awaited()
    env.upload_dir.assert_not_awaited()
