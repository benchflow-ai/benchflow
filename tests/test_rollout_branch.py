"""Branch -> Rollout engine integration.

A ``Rollout`` builds a ``RolloutTree`` as it executes (a linear rollout is a
degree-1 tree) and can ``branch`` at the cursor: checkpoint the Environment,
fork N children, run each child continuation from the env checkpoint with a
fresh agent session, then aggregate the children's returns into V(parent).

These are unit tests against fakes — no Docker, Daytona, or API keys. The
live e2e is a separate task.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.environment.manifest import EnvironmentManifest
from benchflow.environment.manifest_env import ManifestEnvironment
from benchflow.environment.protocol import StateSnapshot
from benchflow.rewards.protocol import VerifyResult
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

    monkeypatch.setattr("benchflow.rollout.execute_prompts", fake_execute_prompts)
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
    assert env.restored == [env.snapshots[0], env.snapshots[0]]
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

    monkeypatch.setattr("benchflow.rollout.execute_prompts", fake_execute_prompts)
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

    monkeypatch.setattr("benchflow.rollout.execute_prompts", fake_execute_prompts)
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

    monkeypatch.setattr("benchflow.rollout.execute_prompts", fake_execute_prompts)

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


async def test_default_runner_empty_reward_falls_back_to_zero(
    tmp_path: Path, monkeypatch
):
    """SHOULD-FIX 9: the default runner returns 0.0 when verify() yields nothing.

    verify() can return None or an empty dict; the per-child runner must
    treat both as a 0.0 return rather than crashing.
    """
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()

    async def fake_connect(self):
        self._acp_client = object()

    async def fake_disconnect(self):
        self._acp_client = None

    async def fake_execute(self, prompts=None, *, node=None):
        return [], 0

    verify_returns = [None, {}]

    async def fake_verify(self):
        return verify_returns.pop(0)

    monkeypatch.setattr(Rollout, "connect", fake_connect)
    monkeypatch.setattr(Rollout, "disconnect", fake_disconnect)
    monkeypatch.setattr(Rollout, "execute", fake_execute)
    monkeypatch.setattr(Rollout, "verify", fake_verify)

    value = await rollout.branch(2)

    # both children scored 0.0 (None and {} both fall back), V(parent) = 0.0
    assert value == 0.0


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
    monkeypatch.setattr(rollout, "_run_scene", noop)
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
    # the real restore path ran once per child: cp from the snapshot dir
    restore_cmds = [c for c in sandbox.exec_calls if c.startswith("cp ")]
    assert len(restore_cmds) == 2


@pytest.mark.asyncio
async def test_branch_aggregates_into_a_node_scored_verify_result(tmp_path: Path):
    """PR-1b: branch() composes child VerifyResults into a node-scored
    parent.state['verify_result'], while keeping the float mirror for back-compat."""
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()
    parent = rollout._cursor

    seq = iter([0.0, 1.0])

    async def run_child(child):
        return next(seq)  # bare floats — branch() lifts them to VerifyResults

    value = await rollout.branch(2, run_child=run_child)

    # Back-compat scalar still holds.
    assert value == 0.5
    assert parent.state["value"] == 0.5
    # Canonical VerifyResult composed on the parent.
    vr = parent.state["verify_result"]
    assert isinstance(vr, VerifyResult)
    assert vr.reward == 0.5
    assert vr.space == "output" and vr.granularity == "terminal"
    # Each child carries its own VerifyResult.
    assert all(isinstance(c.state["verify_result"], VerifyResult) for c in parent.children)
    assert {c.state["verify_result"].reward for c in parent.children} == {0.0, 1.0}


@pytest.mark.asyncio
async def test_branch_runner_back_compat_accepts_bare_float(tmp_path: Path):
    """A custom run_child returning a bare float still yields a composed
    VerifyResult on the parent (the runner contract tolerates float|VerifyResult)."""
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()
    parent = rollout._cursor

    async def run_child(child):
        return 1.0

    await rollout.branch(2, run_child=run_child)
    vr = parent.state["verify_result"]
    assert isinstance(vr, VerifyResult)
    assert vr.reward == 1.0


@pytest.mark.asyncio
async def test_linear_state_isolates_verify_result(tmp_path: Path):
    """_LinearState round-trips rollout._verify_result so a child's scoring does
    not leak onto the parent after branch() returns (extends the isolation
    invariant to the Phase 1a field)."""
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()
    sentinel = VerifyResult(reward=0.42)
    rollout._verify_result = sentinel

    async def run_child(child):
        rollout._verify_result = VerifyResult(reward=0.99)  # a child's scoring mutates it
        return 0.99

    await rollout.branch(2, run_child=run_child)
    # The parent's pre-branch verify_result is restored — not a child's value.
    assert rollout._verify_result is sentinel
