"""Branch -> Rollout engine integration.

A ``Rollout`` builds a ``RolloutTree`` as it executes (a linear rollout is a
degree-1 tree) and can ``branch`` at the cursor: checkpoint the Environment,
fork N children, run each child continuation from the env checkpoint with a
fresh agent session, then aggregate the children's returns into V(parent).

These are unit tests against fakes — no Docker, Daytona, or API keys. The
live e2e is a separate task.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from benchflow.environment.manifest import EnvironmentManifest
from benchflow.environment.manifest_env import ManifestEnvironment
from benchflow.environment.protocol import StateSnapshot
from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.sandbox.protocol import ExecResult
from benchflow.trajectories.tree import RolloutTree, branch_points, trajectory


class FakeEnvironment:
    """Environment-plane stand-in recording snapshot/restore calls."""

    def __init__(self) -> None:
        self.snapshots: list[StateSnapshot] = []
        self.restored: list[StateSnapshot] = []

    async def snapshot(self) -> StateSnapshot:
        snap = StateSnapshot(id=f"snap-{len(self.snapshots) + 1}", path="/tmp/x")
        self.snapshots.append(snap)
        return snap

    async def restore(self, snap: StateSnapshot) -> None:
        self.restored.append(snap)


def _rollout(tmp_path: Path) -> Rollout:
    return Rollout(
        RolloutConfig(task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy")])
    )


_STATEFUL_MANIFEST = EnvironmentManifest.model_validate_toml(
    """
[environment]
name           = "clawsbench"
base_image     = "x:latest"
owns_lifecycle = false

[[environment.services]]
name    = "gmail"
command = "claw-gmail --db /data/gmail.db serve --port 9001"
port    = 9001

[environment.state]
kind  = "sqlite"
paths = ["/data/gmail.db"]
"""
)


class FakeSandbox:
    """Sandbox stand-in recording exec calls — every command succeeds."""

    def __init__(self) -> None:
        self.exec_calls: list[str] = []

    async def exec(
        self, cmd: str, *, user: str = "root", timeout_sec: int = 30
    ) -> ExecResult:
        self.exec_calls.append(cmd)
        return ExecResult(return_code=0, stdout="", stderr="")


def test_fresh_rollout_exposes_a_degree_one_tree(tmp_path: Path):
    """A new Rollout has a RolloutTree with only the root node — no branches."""
    rollout = _rollout(tmp_path)
    assert isinstance(rollout.tree, RolloutTree)
    assert rollout.tree.root.children == []
    assert branch_points(rollout.tree) == []


async def test_execute_grows_the_tree_by_one_step(tmp_path: Path, monkeypatch):
    """Each execute() call advances the cursor down a degree-1 chain.

    A linear rollout's tree stays a chain — every node has at most one child,
    so there are no branch points.
    """
    rollout = _rollout(tmp_path)

    async def fake_execute_prompts(*_a, **_kw):
        return [{"role": "agent", "text": "hi"}], 1

    monkeypatch.setattr(rollout._planes, "execute_prompts", fake_execute_prompts)
    rollout._acp_client = object()  # execute() only needs this non-None

    root = rollout.tree.root
    await rollout.execute(["first"])
    after_first = rollout._cursor
    assert after_first is not root
    assert after_first.parent is root
    assert root.children == [after_first]

    await rollout.execute(["second"])
    after_second = rollout._cursor
    assert after_second.parent is after_first
    assert branch_points(rollout.tree) == []  # still linear


async def test_branch_checkpoints_forks_and_aggregates(tmp_path: Path):
    """branch() runs the full Branch lifecycle and returns V(parent).

    checkpoint the env once, fork N children, run each child continuation
    from the checkpoint, score each, average the returns into V(parent).
    """
    rollout = _rollout(tmp_path)
    env = FakeEnvironment()
    rollout._environment = env

    parent = rollout._cursor
    run_order: list[str] = []
    seen = []

    async def run_child(child):
        run_order.append(child.id)
        seen.append(child)
        return float(len(run_order) - 1)  # child 0 -> 0.0, child 1 -> 1.0

    value = await rollout.branch(2, run_child=run_child)

    # checkpoint happened exactly once, at the parent
    assert len(env.snapshots) == 1
    assert parent.state["snapshot"] is env.snapshots[0]
    # two children forked, parent is now a branch point
    assert parent in branch_points(rollout.tree)
    assert len(parent.children) == 2
    # each child ran, each restored to the checkpoint first
    assert len(run_order) == 2
    assert env.restored == [env.snapshots[0]] * 3  # children, then parent
    # returns recorded on the children and aggregated into V(parent)
    assert [c.state["reward"] for c in parent.children] == [0.0, 1.0]
    assert value == 0.5
    assert parent.state["value"] == 0.5


async def test_branch_rejects_fewer_than_two_children(tmp_path: Path):
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()

    async def run_child(child):
        return 0.0

    with pytest.raises(ValueError, match=">= 2"):
        await rollout.branch(1, run_child=run_child)


async def test_branch_without_environment_raises(tmp_path: Path):
    """Branching needs the Environment plane — there is no world to snapshot."""
    rollout = _rollout(tmp_path)
    assert rollout._environment is None

    async def run_child(child):
        return 0.0

    with pytest.raises(RuntimeError, match="Environment"):
        await rollout.branch(2, run_child=run_child)


async def test_branch_does_not_corrupt_the_parent_rollout(tmp_path: Path, monkeypatch):
    """MUST-FIX 1: a branch child runs as an isolated sub-rollout.

    After branch() returns, the parent's linear state — cursor, trajectory,
    rewards, phase, n_tool_calls — must be exactly what it was before. The
    children run *real* (non-stubbed) execute/verify, so this proves the
    children's mutations are scoped, not re-entrant on the shared instance.
    """
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()

    # Drive two real linear execute() calls so the parent has non-trivial state.
    async def fake_execute_prompts(*_a, **_kw):
        return [{"role": "agent", "text": "parent-step"}], 2

    monkeypatch.setattr(rollout._planes, "execute_prompts", fake_execute_prompts)
    rollout._acp_client = object()
    await rollout.execute(["p1"])
    await rollout.execute(["p2"])

    # Snapshot the parent's linear state before branching. (_phase is
    # excluded: branch() legitimately sets it to "branched" — see the
    # dedicated phase test.)
    cursor_before = rollout._cursor
    trajectory_before = list(rollout._trajectory)
    n_tool_calls_before = rollout._n_tool_calls
    rewards_before = rollout._rewards

    # Children run REAL execute()/verify() — non-stubbed — through the engine's
    # default per-child runner. connect/verify are faked at the boundary only.
    async def fake_connect_inner(self):
        self._acp_client = object()

    async def fake_disconnect_inner(self):
        self._acp_client = None

    async def fake_verify_inner(self):
        self._rewards = {"reward": 1.0}
        self._phase = "verified"
        return self._rewards

    monkeypatch.setattr(Rollout, "connect", fake_connect_inner)
    monkeypatch.setattr(Rollout, "disconnect", fake_disconnect_inner)
    monkeypatch.setattr(Rollout, "verify", fake_verify_inner)

    value = await rollout.branch(2)

    # The children ran and aggregated.
    assert value == 1.0
    # The parent's linear state is byte-for-byte intact.
    assert rollout._cursor is cursor_before
    assert rollout._trajectory == trajectory_before
    assert rollout._n_tool_calls == n_tool_calls_before
    assert rollout._rewards == rewards_before


async def test_post_branch_execute_grows_off_the_parent_cursor(
    tmp_path: Path, monkeypatch
):
    """After branch(), a linear execute() continues off the parent — not a child.

    The 'tree is additive / no-regression' invariant: branch() must leave the
    cursor where it found it, so a post-branch execute() grows the right node.
    """
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()

    async def fake_execute_prompts(*_a, **_kw):
        return [{"role": "agent", "text": "x"}], 1

    monkeypatch.setattr(rollout._planes, "execute_prompts", fake_execute_prompts)
    rollout._acp_client = object()
    await rollout.execute(["p1"])
    parent = rollout._cursor

    async def run_child(child):
        return 1.0

    await rollout.branch(2, run_child=run_child)

    # branch left the cursor at the parent
    assert rollout._cursor is parent
    # a post-branch execute grows a NEW child off the parent
    rollout._acp_client = object()
    await rollout.execute(["after"])
    after = rollout._cursor
    assert after.parent is parent
    # parent now has 3 children: 2 branch children + 1 linear continuation
    assert len(parent.children) == 3


async def test_branch_child_continuation_attaches_to_the_child_node(
    tmp_path: Path, monkeypatch
):
    """MUST-FIX 4: a child's continuation Steps attach to the child node itself.

    No content-free placeholder Step: trajectory(leaf) through a branch child
    must not contain an empty Step, and the reward must land on the real leaf.
    """
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()

    async def fake_execute_prompts(*_a, **_kw):
        return [{"role": "agent", "text": "child-work"}], 1

    monkeypatch.setattr(rollout._planes, "execute_prompts", fake_execute_prompts)

    async def fake_connect_inner(self):
        self._acp_client = object()

    async def fake_disconnect_inner(self):
        self._acp_client = None

    async def fake_verify_inner(self):
        self._rewards = {"reward": 0.7}
        return self._rewards

    monkeypatch.setattr(Rollout, "connect", fake_connect_inner)
    monkeypatch.setattr(Rollout, "disconnect", fake_disconnect_inner)
    monkeypatch.setattr(Rollout, "verify", fake_verify_inner)

    parent = rollout._cursor
    await rollout.branch(2)

    for child in parent.children:
        # the child node carries the reward — not a descendant placeholder
        assert child.state["reward"] == 0.7
        # every Step on the root->child path has real content (no empty Step)
        steps = trajectory(child)
        assert steps, "child has at least one continuation Step"
        assert all(s.data for s in steps), "no content-free placeholder Step"
        # the child IS a leaf — its work did not hang off a descendant
        assert child.children == []


async def test_branch_uses_a_real_branched_phase(tmp_path: Path):
    """SHOULD-FIX 6: branch() sets a 'branched' phase, not a thrashed one."""
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()

    async def run_child(child):
        return 1.0

    await rollout.branch(2, run_child=run_child)
    assert rollout._phase == "branched"


async def test_default_runner_uses_fresh_agent_per_child(tmp_path: Path, monkeypatch):
    """SHOULD-FIX 10: the default per-child runner restarts the agent.

    Each child re-runs from the env checkpoint with a fresh agent session,
    and the previous child's agent is disconnected before the next connects.
    """
    rollout = _rollout(tmp_path)
    env = FakeEnvironment()
    rollout._environment = env
    calls: list[str] = []

    async def fake_connect(self):
        calls.append("connect")
        self._acp_client = object()

    async def fake_disconnect(self):
        calls.append("disconnect")
        self._acp_client = None

    async def fake_execute(self, prompts=None, *, node=None):
        calls.append("execute")
        return [], 0

    async def fake_verify(self):
        calls.append("verify")
        return {"reward": 1.0}

    monkeypatch.setattr(Rollout, "connect", fake_connect)
    monkeypatch.setattr(Rollout, "disconnect", fake_disconnect)
    monkeypatch.setattr(Rollout, "execute", fake_execute)
    monkeypatch.setattr(Rollout, "verify", fake_verify)

    value = await rollout.branch(2)

    # a fresh agent per child: connect happens once per child
    assert calls.count("connect") == 2
    assert calls.count("verify") == 2
    # each connect is preceded by a disconnect — no agent overlap between children
    for i, c in enumerate(calls):
        if c == "connect" and i > 0:
            assert "disconnect" in calls[:i]
    assert value == 1.0


@pytest.mark.parametrize("verify_result", [None, {}, {"other": 1}])
async def test_default_runner_missing_reward_is_unscored(
    tmp_path: Path, monkeypatch, verify_result
):
    """PR #1046 custody port: missing observations must not become measured zeros."""
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()

    async def fake_connect(self):
        self._acp_client = object()

    async def fake_disconnect(self):
        self._acp_client = None

    async def fake_execute(self, prompts=None, *, node=None):
        return [], 0

    async def fake_verify(self):
        return verify_result

    monkeypatch.setattr(Rollout, "connect", fake_connect)
    monkeypatch.setattr(Rollout, "disconnect", fake_disconnect)
    monkeypatch.setattr(Rollout, "execute", fake_execute)
    monkeypatch.setattr(Rollout, "verify", fake_verify)

    with pytest.raises(RuntimeError, match="unscored"):
        await rollout.branch(2)
    assert "reward" not in rollout.tree.root.children[0].state
    assert "value" not in rollout.tree.root.state
    assert rollout._acp_client is None


async def test_linear_rollout_run_never_branches(tmp_path: Path, monkeypatch):
    """No-regression: a normal run() grows only a degree-1 tree, no branches.

    The tree/branch path is dead code unless branch() is explicitly called.
    """
    rollout = _rollout(tmp_path)
    rollout._rollout_dir = tmp_path / "trial"
    rollout._rollout_dir.mkdir()
    rollout._rollout_name = "trial-1"

    async def noop(*_a, **_kw):
        return None

    async def fake_setup(*_a, **_kw):
        from datetime import datetime

        rollout._started_at = datetime.now()

    monkeypatch.setattr(rollout, "setup", fake_setup)
    monkeypatch.setattr(rollout, "start", noop)
    monkeypatch.setattr(rollout, "install_agent", noop)
    monkeypatch.setattr(rollout, "_run_steps", noop)
    monkeypatch.setattr(rollout, "verify", noop)
    monkeypatch.setattr(rollout, "cleanup", noop)

    await rollout.run()

    assert branch_points(rollout.tree) == []


async def test_branch_drives_manifest_environment_snapshot_restore(tmp_path: Path):
    """branch() works against the real ManifestEnvironment over a fake sandbox.

    Exercises the real Environment-plane snapshot/restore — the SQLite
    .backup / cp commands — without Docker.
    """
    rollout = _rollout(tmp_path)
    sandbox = FakeSandbox()
    rollout._environment = ManifestEnvironment(_STATEFUL_MANIFEST, sandbox=sandbox)

    async def run_child(child):
        return 1.0

    value = await rollout.branch(2, run_child=run_child)

    assert value == 1.0
    # the real snapshot path ran: one SQLite .backup
    assert any(".backup" in c for c in sandbox.exec_calls)
    # The real restore path ran for each child and the parent continuation.
    restore_cmds = [c for c in sandbox.exec_calls if c.startswith("cp ")]
    assert len(restore_cmds) == 3


class MutableEnvironment(FakeEnvironment):
    """Stateful fake exposing leaks between branches and parent continuations."""

    def __init__(self) -> None:
        super().__init__()
        self.value = "parent"
        self.fail_restore = False

    async def restore(self, snap: StateSnapshot) -> None:
        if self.fail_restore:
            raise RuntimeError("restore failed")
        await super().restore(snap)
        self.value = "parent"


async def test_branch_restores_parent_environment(tmp_path: Path):
    """Guards parent environment leakage present at commit 6b99a10e."""
    rollout = _rollout(tmp_path)
    env = MutableEnvironment()
    rollout._environment = env
    seen = []

    async def run_child(child):
        seen.append(env.value)
        env.value = child.id
        return 1.0

    assert await rollout.branch(2, run_child=run_child) == 1.0
    assert seen == ["parent", "parent"]
    assert env.value == "parent"


@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_branch_failure_restores_parent(tmp_path: Path, error_type):
    """Guards missing failure/cancellation cleanup at commit 6b99a10e."""
    rollout = _rollout(tmp_path)
    env = MutableEnvironment()
    rollout._environment = env
    parent = rollout._cursor
    failure = error_type("child failed")

    async def run_child(child):
        env.value = "child"
        rollout._trajectory.append({"role": "agent", "text": "child"})
        rollout._phase = "executed"
        raise failure

    with pytest.raises(error_type) as raised:
        await rollout.branch(2, run_child=run_child)
    assert raised.value is failure
    assert env.value == "parent"
    assert rollout._cursor is parent
    assert rollout._trajectory == []
    # branch() intentionally disconnects the parent before capturing state.
    assert rollout._phase == "installed"
    assert "value" not in parent.state


@pytest.mark.parametrize("child_fails", [False, True])
async def test_parent_restore_failure_preserves_errors(tmp_path: Path, child_fails):
    """Guards cleanup failures hiding branch errors at commit 6b99a10e."""
    rollout = _rollout(tmp_path)
    env = MutableEnvironment()
    rollout._environment = env
    parent = rollout._cursor
    child_failure = ValueError("child failed")
    calls = 0

    async def run_child(child):
        nonlocal calls
        calls += 1
        rollout._trajectory.append({"role": "agent", "text": "child"})
        if child_fails:
            env.fail_restore = True
            raise child_failure
        if calls == 2:
            env.fail_restore = True
        return 1.0

    if child_fails:
        with pytest.raises(ExceptionGroup) as raised:
            await rollout.branch(2, run_child=run_child)
        assert raised.value.exceptions[0] is child_failure
        assert str(raised.value.exceptions[1]) == "restore failed"
    else:
        with pytest.raises(RuntimeError, match="restore failed"):
            await rollout.branch(2, run_child=run_child)
    assert rollout._cursor is parent
    assert rollout._trajectory == []
    assert "value" not in parent.state


@pytest.mark.parametrize("failure_stage", ["connect", "execute", "verify", "cancel"])
async def test_default_runner_quiesces_failed_child_before_restore(
    tmp_path: Path, monkeypatch, failure_stage
):
    """Guards default-runner agent leakage present at commit 6b99a10e."""
    rollout = _rollout(tmp_path)
    events = []
    entered_execute = asyncio.Event()
    failure = RuntimeError(f"{failure_stage} failed")

    class OrderedEnvironment(MutableEnvironment):
        async def restore(self, snap):
            # Real disconnect must clear all live session references first.
            assert rollout._acp_client is None
            assert rollout._session is None
            assert rollout._session_adapter is None
            events.append("restore")
            await super().restore(snap)

    class Client:
        async def close(self):
            events.append("close")

    env = OrderedEnvironment()
    rollout._environment = env
    parent = rollout._cursor

    async def connect(self):
        self._acp_client = Client()
        self._session = object()
        self._session_adapter = object()
        if failure_stage == "connect":
            raise failure

    async def execute(self, prompts=None, *, node=None):
        env.value = "child"
        if failure_stage == "execute":
            raise failure
        if failure_stage == "cancel":
            entered_execute.set()
            await asyncio.Event().wait()
        return [], 0

    async def verify(self):
        raise failure

    monkeypatch.setattr(Rollout, "connect", connect)
    monkeypatch.setattr(Rollout, "execute", execute)
    monkeypatch.setattr(Rollout, "verify", verify)
    # Exercise the real default runner and real Rollout.disconnect().
    task = asyncio.create_task(rollout.branch(2))
    if failure_stage == "cancel":
        await asyncio.wait_for(entered_execute.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(RuntimeError) as raised:
            await task
        assert raised.value is failure
    assert events == ["restore", "close", "restore"]
    assert env.value == "parent"
    assert rollout._cursor is parent
    assert rollout._acp_client is None
    assert rollout._session is None
    assert rollout._session_adapter is None


async def test_default_runner_preserves_execution_and_disconnect_errors(
    tmp_path: Path, monkeypatch
):
    """Guards default-runner cleanup error masking at commit 6b99a10e."""
    rollout = _rollout(tmp_path)
    rollout._environment = MutableEnvironment()
    primary = ValueError("execute failed")
    cleanup = RuntimeError("disconnect failed")
    disconnect_calls = 0

    async def connect(self):
        pass

    async def execute(self, prompts=None, *, node=None):
        raise primary

    async def disconnect(self):
        nonlocal disconnect_calls
        disconnect_calls += 1
        if disconnect_calls == 2:
            raise cleanup

    monkeypatch.setattr(Rollout, "connect", connect)
    monkeypatch.setattr(Rollout, "execute", execute)
    monkeypatch.setattr(Rollout, "disconnect", disconnect)
    with pytest.raises(ExceptionGroup) as raised:
        await rollout.branch(2)
    assert raised.value.exceptions == (primary, cleanup)


async def test_cancelled_child_survives_parent_restore_failure(tmp_path: Path, caplog):
    """Guards cancellation masking by the parent-restore group from the parent-restore change.

    A cancelled branch must end as a bare ``CancelledError`` so asyncio marks
    the task cancelled; the restore failure is logged, noted and recorded.
    """
    rollout = _rollout(tmp_path)
    env = MutableEnvironment()
    rollout._environment = env
    entered = asyncio.Event()

    async def run_child(child):
        env.fail_restore = True
        entered.set()
        await asyncio.Event().wait()
        return 1.0

    task = asyncio.create_task(rollout.branch(2, run_child=run_child))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with (
        caplog.at_level("WARNING", logger="benchflow.rollout_branch"),
        pytest.raises(asyncio.CancelledError),
    ):
        await task
    assert task.cancelled()
    [fork] = rollout._branch_forks
    assert fork["status"] == "cancelled"
    assert fork["parent_restore"] == "failed"
    assert rollout._branch_world_unsafe is True
    assert any(
        record.exc_info and str(record.exc_info[1]) == "restore failed"
        for record in caplog.records
    )


async def test_direct_child_cancellation_notes_parent_restore_failure(
    tmp_path: Path,
):
    """Guards cancellation masking by the parent-restore group from the parent-restore change."""
    rollout = _rollout(tmp_path)
    env = MutableEnvironment()
    rollout._environment = env
    cancellation = asyncio.CancelledError("child cancelled")

    async def run_child(child):
        env.fail_restore = True
        raise cancellation

    with pytest.raises(asyncio.CancelledError) as raised:
        await rollout.branch(2, run_child=run_child)
    assert raised.value is cancellation
    assert any("restore failed" in note for note in raised.value.__notes__)


async def test_default_runner_cancellation_survives_disconnect_failure(
    tmp_path: Path, monkeypatch
):
    """Guards cancellation masking by the disconnect group from the parent-restore change."""
    rollout = _rollout(tmp_path)
    rollout._environment = MutableEnvironment()
    entered = asyncio.Event()
    disconnect_calls = 0

    async def connect(self):
        pass

    async def execute(self, prompts=None, *, node=None):
        entered.set()
        await asyncio.Event().wait()

    async def disconnect(self):
        nonlocal disconnect_calls
        disconnect_calls += 1
        if disconnect_calls == 2:
            raise RuntimeError("disconnect failed")

    monkeypatch.setattr(Rollout, "connect", connect)
    monkeypatch.setattr(Rollout, "execute", execute)
    monkeypatch.setattr(Rollout, "disconnect", disconnect)
    task = asyncio.create_task(rollout.branch(2))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    [fork] = rollout._branch_forks
    assert fork["status"] == "cancelled"
    assert fork["children"][0]["status"] == "cancelled"


async def test_checkpoint_cancellation_survives_lineage_failure(
    tmp_path: Path, monkeypatch
):
    """Guards cancellation masking by the checkpoint group from solver-evidence preservation."""
    from benchflow.branch_lineage import ForkRecord

    rollout = _rollout(tmp_path)
    rollout._rollout_dir = tmp_path / "run"
    cancellation = asyncio.CancelledError("checkpoint cancelled")
    persists = 0

    class CancelledSnapshotEnvironment(FakeEnvironment):
        async def snapshot(self) -> StateSnapshot:
            raise cancellation

    def persist(self):
        nonlocal persists
        persists += 1
        if persists == 2:
            raise OSError("lineage write failed")

    rollout._environment = CancelledSnapshotEnvironment()
    monkeypatch.setattr(ForkRecord, "persist", persist)
    with pytest.raises(asyncio.CancelledError) as raised:
        await rollout.branch(2, run_child=lambda child: asyncio.sleep(0, 1.0))
    assert raised.value is cancellation
    assert any("lineage write failed" in note for note in raised.value.__notes__)


async def test_plain_task_branch_error_names_the_sandbox_layer(tmp_path: Path):
    """Regression test: on a task without an environment
    manifest, the default layer error told users to write a manifest. For a
    plain task the fix is snapshot_layers={"sandbox"}; the error must say so
    and say what that layer captures."""
    rollout = _rollout(tmp_path)

    async def run_child(child):
        return 0.0

    with pytest.raises(RuntimeError) as excinfo:
        await rollout.branch(2, run_child=run_child)
    message = str(excinfo.value)
    assert "snapshot_layers={'sandbox'}" in message
    assert "container filesystem" in message


async def test_execute_without_node_fills_the_pending_branch_child(
    tmp_path: Path, monkeypatch
):
    """Regression test: a custom runner that called
    rollout.execute(prompts) without node=node hung the child's Steps under
    a grandchild and left the child node pending (step_id null) while
    tree.json still reported it scored. execute() now fills the pending
    branch-child cursor itself."""
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()

    async def fake_execute_prompts(*_a, **_kw):
        return [{"role": "agent", "text": "child-work"}], 1

    monkeypatch.setattr(rollout._planes, "execute_prompts", fake_execute_prompts)
    parent = rollout._cursor

    async def run_child(node):
        rollout._acp_client = object()
        try:
            await rollout.execute(["child prompt"])
        finally:
            rollout._acp_client = None
        return 1.0

    await rollout.branch(2, run_child)
    for child in parent.children:
        assert child.step_in is not None
        assert child.children == []
