"""Parent evidence and state regressions for selective PR #1046."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import benchflow.rollout_branch as engine
from benchflow.branch_artifacts import MountedArtifacts
from benchflow.branch_result import RESULT_STATE_FIELDS
from benchflow.environment.protocol import StateSnapshot
from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.task.paths import RolloutPaths


def rollout_for(tmp_path):
    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy --agent")]
        )
    )
    rollout._rollout_dir = tmp_path / "parent"
    rollout._rollout_paths = RolloutPaths(rollout._rollout_dir)
    rollout._rollout_paths.mkdir()
    rollout._environment = type(
        "Environment",
        (),
        {
            "snapshot": AsyncMock(
                return_value=StateSnapshot(id="parent", path="/tmp/snapshot")
            ),
            "restore": AsyncMock(),
        },
    )()
    rollout.disconnect = AsyncMock()
    return rollout


@pytest.mark.parametrize("child_fails", [False, True])
@pytest.mark.parametrize("restore_fails", [False, True])
async def test_parent_state_evidence_and_fork_unique_bundles(
    tmp_path, child_fails, restore_fails
):
    """PR #1046: failed children/restores preserve parent and retain child evidence."""
    rollout = rollout_for(tmp_path)
    root = rollout._rollout_dir
    parent_paths = rollout._rollout_paths
    roots = [
        parent_paths.agent_dir,
        parent_paths.artifacts_dir,
        parent_paths.verifier_dir,
    ]
    for path in roots:
        (path / "same.txt").write_text("parent")
    inodes = [path.stat().st_ino for path in roots]
    (root / "trajectory").mkdir()
    (root / "trajectory/acp_trajectory.jsonl").write_text("parent trajectory")
    rollout._timing = {"agent_execution": 8}
    rollout._trajectory = [{"nested": {"parent": 1}}]
    rollout._native_usage_metrics = {"input_tokens": 7}
    rollout._solver_execution_complete = True
    rollout._solver_completion_result = {"parent": [1]}
    rollout._rewards = {"reward": 0.8}
    before = {
        name: copy.deepcopy(getattr(rollout, name))
        for name in RESULT_STATE_FIELDS
        if hasattr(rollout, name)
    }
    restore_count = 0

    async def restore(snap):
        nonlocal restore_count
        restore_count += 1
        if restore_fails and restore_count == (2 if child_fails else 3):
            raise OSError("restore failure")

    rollout._environment.restore = restore
    seen = []

    async def child(node):
        seen.append(node.id)
        assert rollout._rollout_paths is parent_paths
        for path in roots:
            assert not list(path.iterdir())
            (path / "same.txt").write_text(node.id)
        assert rollout._rollout_dir != root
        (rollout._rollout_dir / "trajectory").mkdir()
        (rollout._rollout_dir / "trajectory/acp_trajectory.jsonl").write_text(node.id)
        rollout._trajectory[0]["nested"]["parent"] = 99
        rollout._trajectory.append({"child": node.id})
        rollout._native_usage_metrics["input_tokens"] = 3
        rollout._solver_execution_complete = True
        rollout._solver_completion_result = {"child": [9]}
        rollout._verifier_error = "child diagnostic"
        rollout._timing["agent_execution"] = 2
        if child_fails:
            raise ValueError("child failure")
        return 1

    if child_fails or restore_fails:
        with pytest.raises(BaseException) as caught:
            await rollout.branch(2, child)

        def messages(exc):
            return [str(exc)] + [
                s for child in getattr(exc, "exceptions", ()) for s in messages(child)
            ]

        text = " ".join(messages(caught.value))
        if child_fails:
            assert "child failure" in text
        if restore_fails:
            assert "restore failure" in text
    else:
        assert await rollout.branch(2, child) == 1
        assert await rollout.branch(2, child) == 1
        assert len(list((root / "branches").iterdir())) == 2
    assert rollout._rollout_dir == root
    assert rollout._rollout_paths is parent_paths
    for path, inode in zip(roots, inodes, strict=True):
        assert path.stat().st_ino == inode
        assert (path / "same.txt").read_text() == "parent"
    assert (root / "trajectory/acp_trajectory.jsonl").read_text() == "parent trajectory"
    for name, value in before.items():
        if name == "_diagnostics":
            assert rollout._diagnostics.to_result_fields() == value.to_result_fields()
        else:
            assert getattr(rollout, name) == value, name
    assert rollout._trajectory == [{"nested": {"parent": 1}}]
    assert rollout._rewards == {"reward": 0.8}
    bundles = list((root / "branches").glob("*/children/*"))
    assert len(bundles) == len(seen)
    for bundle in bundles:
        assert (bundle / "mounted/verifier/same.txt").read_text() == bundle.name
        observation = json.loads((bundle / "observation.json").read_text())
        assert observation["native_usage"]["input_tokens"] == 3
        assert observation["trajectory"] == [{"child": bundle.name}]
        assert observation["reward"] == (None if child_fails else 1.0)


async def test_active_provider_rejected_before_disconnect(tmp_path):
    """PR #1046 port rejects live background writers without a fork contract."""
    rollout = rollout_for(tmp_path)
    runtime = object()
    rollout._usage_runtime = runtime
    with pytest.raises(RuntimeError, match="runtime fork contract"):
        await rollout.branch(2, AsyncMock(return_value=1))
    rollout.disconnect.assert_not_called()
    assert rollout._usage_runtime is runtime


def test_partial_hold_failure_restores_already_moved_parent_entries(
    tmp_path, monkeypatch
):
    """PR #1046: partial custody failure must not hide unheld parent evidence."""
    paths = RolloutPaths(tmp_path / "parent")
    paths.mkdir()
    for root in (paths.agent_dir, paths.artifacts_dir, paths.verifier_dir):
        (root / "parent").write_text(root.name)
    original = Path.rename

    def rename(path, target):
        if path == paths.artifacts_dir / "parent":
            raise OSError("hold failed")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(OSError, match="hold failed"):
        MountedArtifacts.hold(paths, tmp_path / "fork")
    for root in (paths.agent_dir, paths.artifacts_dir, paths.verifier_dir):
        assert (root / "parent").read_text() == root.name


async def test_real_acp_host_writer_is_scoped_to_each_child(tmp_path):
    """PR #1046: actual streaming trajectory callbacks must not overwrite parent."""
    rollout = rollout_for(tmp_path)
    root = rollout._rollout_dir
    parent_trajectory = root / "trajectory/acp_trajectory.jsonl"
    parent_trajectory.parent.mkdir()
    parent_trajectory.write_text("parent stream")

    async def child(node):
        session = SimpleNamespace(steps=[{"type": "agent_message", "text": node.id}])
        rollout._session = session
        rollout._attach_trajectory_writer(rollout._rollout_dir)
        session.on_change(session)
        rollout._session = None
        return 1

    await rollout.branch(2, child)
    assert parent_trajectory.read_text() == "parent stream"
    for directory in (root / "branches").glob("*/children/*"):
        records = [
            json.loads(line)
            for line in (directory / "trajectory/acp_trajectory.jsonl")
            .read_text()
            .splitlines()
        ]
        assert records == [{"type": "agent_message", "text": directory.name}]


async def test_handoff_failure_keeps_original_exception_and_parent_evidence(
    tmp_path, monkeypatch
):
    """PR #1046: custody failures retain both diagnosis and unclaimed child output."""
    rollout = rollout_for(tmp_path)
    paths = rollout._rollout_paths
    (paths.verifier_dir / "same").write_text("parent")
    root = rollout._rollout_dir

    def fail_handoff(self, child_dir):
        raise OSError("handoff failure")

    monkeypatch.setattr(MountedArtifacts, "hand_off", fail_handoff)

    async def child(node):
        (paths.verifier_dir / "same").write_text("child")
        raise ValueError("original failure")

    with pytest.raises(BaseExceptionGroup) as caught:
        await rollout.branch(2, child)
    assert [str(exc) for exc in caught.value.exceptions] == [
        "original failure",
        "handoff failure",
    ]
    assert (paths.verifier_dir / "same").read_text() == "parent"
    assert (
        next((root / "branches").glob("*/unclaimed/verifier/same")).read_text()
        == "child"
    )
    assert rollout._branch_child_active is False


@pytest.mark.parametrize("child_fails", [False, True])
async def test_custom_session_is_closed_and_native_usage_captured_before_custody(
    tmp_path, monkeypatch, child_fails
):
    """PR #1046: engine closes custom-runner sessions before handing off evidence."""
    rollout = rollout_for(tmp_path)
    del rollout.disconnect  # Exercise the actual Rollout.disconnect implementation.
    rollout._usage_metrics = {"total_tokens": 999, "usage_source": "parent"}
    rollout._native_usage_metrics = {"total_tokens": 17}
    paths = rollout._rollout_paths
    current = None
    original = MountedArtifacts.hand_off

    def handoff(holder, directory):
        assert current.closed
        assert rollout._session is None
        assert rollout._acp_client is None
        original(holder, directory)

    monkeypatch.setattr(MountedArtifacts, "hand_off", handoff)

    async def child(node):
        nonlocal current
        assert rollout._usage_metrics.get("total_tokens") != 999
        session = SimpleNamespace(
            steps=[{"type": "agent_message", "text": node.id}],
            latest_usage_totals=lambda: {
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
            },
        )

        class Client:
            closed = False

            async def close(self):
                session.on_change(session)
                (paths.agent_dir / "closed").write_text(node.id)
                self.closed = True

        current = Client()
        rollout._session = session
        rollout._acp_client = current
        rollout._is_session_factory = True
        rollout._attach_trajectory_writer(rollout._rollout_dir)
        if child_fails:
            raise ValueError("custom failure")
        return 1

    if child_fails:
        with pytest.raises(ValueError, match="custom failure"):
            await rollout.branch(2, child)
    else:
        await rollout.branch(2, child)
    assert rollout._usage_metrics == {"total_tokens": 999, "usage_source": "parent"}
    assert rollout._native_usage_metrics == {"total_tokens": 17}
    for directory in (rollout._rollout_dir / "branches").glob("*/children/*"):
        observation = json.loads((directory / "observation.json").read_text())
        assert observation["native_usage"]["total_tokens"] == 5
        assert observation["native_usage"]["n_input_tokens"] == 3
        assert (directory / "mounted/agent/closed").read_text() == directory.name


async def test_unquiesced_child_keeps_parent_evidence_held(tmp_path, monkeypatch):
    """PR #1046: cancellation-resistant cleanup is bounded and cannot expose parent files."""
    rollout = rollout_for(tmp_path)
    paths = rollout._rollout_paths
    (paths.agent_dir / "parent").write_text("preserved")
    release = asyncio.Event()
    completed = asyncio.Event()
    calls = 0

    async def disconnect():
        nonlocal calls
        calls += 1
        if calls == 1:
            return
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
        finally:
            completed.set()

    rollout.disconnect = disconnect
    monkeypatch.setattr(engine, "_CHILD_CLEANUP_TIMEOUT", 0.01)
    monkeypatch.setattr(engine, "_CHILD_CANCEL_GRACE", 0.01)
    try:
        with pytest.raises(BaseExceptionGroup):
            await asyncio.wait_for(rollout.branch(2, AsyncMock(return_value=1)), 1)
        held = next((rollout._rollout_dir / "branches").glob("*/parent/agent/parent"))
        assert held.read_text() == "preserved"
        assert not (paths.agent_dir / "parent").exists()
        assert rollout._branch_child_active is False
    finally:
        release.set()
        await asyncio.wait_for(completed.wait(), 1)


async def test_default_runner_unresolved_disconnect_cannot_be_erased_by_retry(
    tmp_path, monkeypatch
):
    """PR #1046: default cleanup is bounded and a fast retry cannot release held evidence."""
    rollout = rollout_for(tmp_path)
    paths = rollout._rollout_paths
    (paths.verifier_dir / "parent").write_text("preserved")
    release = asyncio.Event()
    completed = asyncio.Event()
    calls = 0
    rollout.connect = AsyncMock()
    rollout.execute = AsyncMock(return_value=([], 0))
    rollout.verify = AsyncMock(return_value={"reward": 1})

    async def disconnect():
        nonlocal calls
        calls += 1
        if calls != 2:
            return  # Initial disconnect and engine retry both finish immediately.
        rollout._session = None
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()  # First cleanup is still alive despite retry success.
        finally:
            completed.set()

    rollout.disconnect = disconnect
    monkeypatch.setattr(
        MountedArtifacts,
        "hand_off",
        lambda *args: pytest.fail("unresolved cleanup must not hand off"),
    )
    monkeypatch.setattr(engine, "_CHILD_CLEANUP_TIMEOUT", 0.01)
    monkeypatch.setattr(engine, "_CHILD_CANCEL_GRACE", 0.01)
    try:
        with pytest.raises(BaseExceptionGroup) as caught:
            await asyncio.wait_for(rollout.branch(2), 1)
        assert isinstance(caught.value.exceptions[0], TimeoutError)
        assert calls == 3
        assert not completed.is_set()
        held = next(
            (rollout._rollout_dir / "branches").glob("*/parent/verifier/parent")
        )
        assert held.read_text() == "preserved"
        assert not (paths.verifier_dir / "parent").exists()
        fork = json.loads((rollout._rollout_dir / "tree.json").read_text())["forks"][0]
        assert fork["children"][0]["cleanup_error"]["code"] == "unresolved_cleanup"
        assert fork["children"][0]["artifacts"]["status"] == "unavailable"
        assert rollout._branch_cleanup_unquiesced
        assert fork["parent_restore"] == "deferred"
    finally:
        release.set()
        await asyncio.wait_for(completed.wait(), 1)


async def test_pending_writer_never_runs_over_restored_parent_world(
    tmp_path, monkeypatch
):
    """PR #1046: unresolved cleanup defers rollback and permanently blocks unsafe continuation."""
    rollout = rollout_for(tmp_path)
    world = {"value": "parent"}
    restored_values = []
    release, finished = asyncio.Event(), asyncio.Event()
    calls = 0

    async def restore(snapshot):
        world["value"] = "parent"
        restored_values.append("parent")

    rollout._environment.restore = restore

    async def disconnect():
        nonlocal calls
        calls += 1
        if calls == 1:
            return
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
            world["value"] = "late child write"
        finally:
            finished.set()

    rollout.disconnect = disconnect
    monkeypatch.setattr(engine, "_CHILD_CLEANUP_TIMEOUT", 0.01)
    monkeypatch.setattr(engine, "_CHILD_CANCEL_GRACE", 0.01)

    async def child(node):
        world["value"] = "child"
        return 1

    try:
        with pytest.raises(BaseExceptionGroup):
            await rollout.branch(2, child)
        assert restored_values == ["parent"]  # Only the pre-child restore ran.
        assert world["value"] == "child"
        assert rollout._branch_world_unsafe
        assert rollout._branch_cleanup_unquiesced
        for operation in (
            rollout.connect(),
            rollout.execute(),
            rollout.verify(),
            rollout.branch(2, child),
        ):
            with pytest.raises(RuntimeError, match="Branch world is unsafe"):
                await operation
        release.set()
        await asyncio.wait_for(finished.wait(), 1)
        assert world["value"] == "late child write"
        assert restored_values == ["parent"]
        # Late cleanup completion is not proof that a parent checkpoint was restored.
        with pytest.raises(RuntimeError, match="Branch world is unsafe"):
            await rollout.connect()
        fork = json.loads((rollout._rollout_dir / "tree.json").read_text())["forks"][0]
        assert fork["parent_restore"] == "deferred"
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), 1)
