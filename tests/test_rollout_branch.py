"""Branch -> Rollout engine integration.

A ``Rollout`` builds a ``RolloutTree`` as it executes (a linear rollout is a
degree-1 tree) and can ``branch`` at a chosen node: checkpoint the Environment,
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
from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.sandbox.protocol import ExecResult
from benchflow.trajectories.tree import RolloutTree, branch_points


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
    return Rollout(RolloutConfig(task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy")]))


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

    async def run_child(child, idx):
        run_order.append(child.id)
        return float(idx)  # child 0 -> 0.0, child 1 -> 1.0

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

    async def run_child(child, idx):
        return 0.0

    with pytest.raises(ValueError, match=">= 2"):
        await rollout.branch(1, run_child=run_child)


async def test_branch_without_environment_raises(tmp_path: Path):
    """Branching needs the Environment plane — there is no world to snapshot."""
    rollout = _rollout(tmp_path)
    assert rollout._environment is None

    async def run_child(child, idx):
        return 0.0

    with pytest.raises(RuntimeError, match="Environment"):
        await rollout.branch(2, run_child=run_child)


async def test_default_runner_uses_fresh_agent_per_child(tmp_path: Path):
    """The default per-child runner restarts the agent — connect, execute, verify.

    Sidesteps agent-session snapshot: each child re-runs from the env
    checkpoint with a fresh agent session, not a restored one.
    """
    rollout = _rollout(tmp_path)
    env = FakeEnvironment()
    rollout._environment = env
    calls: list[str] = []

    async def fake_connect():
        calls.append("connect")

    async def fake_execute(prompts=None):
        calls.append(f"execute:{prompts}")
        return [], 0

    async def fake_verify():
        calls.append("verify")
        return {"reward": 1.0}

    rollout.connect = fake_connect  # type: ignore[method-assign]
    rollout.execute = fake_execute  # type: ignore[method-assign]
    rollout.verify = fake_verify  # type: ignore[method-assign]

    value = await rollout.branch(
        2, continuations=[["child-a prompt"], ["child-b prompt"]]
    )

    # a fresh agent per child: connect happens once per child
    assert calls.count("connect") == 2
    # each child ran its own continuation prompt
    assert "execute:['child-a prompt']" in calls
    assert "execute:['child-b prompt']" in calls
    assert calls.count("verify") == 2
    assert value == 1.0


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

    async def run_child(child, idx):
        return 1.0

    value = await rollout.branch(2, run_child=run_child)

    assert value == 1.0
    # the real snapshot path ran: one SQLite .backup
    assert any(".backup" in c for c in sandbox.exec_calls)
    # the real restore path ran once per child: cp from the snapshot dir
    restore_cmds = [c for c in sandbox.exec_calls if c.startswith("cp ")]
    assert len(restore_cmds) == 2
