"""Branch -> Rollout engine integration.

A ``Rollout`` builds a ``RolloutTree`` as it executes (a linear rollout is a
degree-1 tree) and can ``branch`` at the cursor: checkpoint the Environment,
fork N children, run each child continuation from the env checkpoint with a
fresh agent session, then aggregate the children's returns into V(parent).

These are unit tests against fakes — no Docker, Daytona, or API keys. The
live e2e is a separate task.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.branch import UNSCORED_KEY
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


async def test_default_runner_empty_reward_is_unscored_not_zero(
    tmp_path: Path, monkeypatch
):
    """A child whose verify() yielded nothing is unscored — never a 0.0.

    Supersedes the original SHOULD-FIX 9 expectation (both shapes fell back to
    a 0.0 return). verify() returning ``None`` or ``{}`` still must not crash
    the branch, but a fabricated zero is indistinguishable from a real failing
    score — a lost reward would read as evidence. The child records why it is
    unscored, carries no ``reward``, and V(parent) is undefined.
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

    # neither child was scored (None and {} are both "no reward"), so there is
    # nothing to average: V(parent) is undefined, not 0.0.
    assert value is None
    children = list(rollout.tree.root.children)
    assert len(children) == 2
    assert all("reward" not in child.state for child in children)
    assert all(
        "produced no verifier reward" in child.state[UNSCORED_KEY] for child in children
    )


async def test_a_second_fork_with_an_unscored_child_clears_the_first_forks_value(
    tmp_path: Path,
):
    """An undefined V must not leave the previous fork's V behind.

    Guards the fix from "fix(branch): clear a stale branch value when V is
    undefined". A node can be branched more than once. The unscored path set
    the local ``value = None`` and skipped *writing* ``parent.state["value"]``
    — but never removed the one the earlier fork wrote, so ``branch()``
    returned ``None`` while ``tree.json`` still published the first fork's
    number as the value of a fork that has no value. The stale number is the
    dangerous half: the API's ``None`` is at least honest.
    """
    from benchflow.branch import UnscoredChildError
    from benchflow.branch_lineage import serialize_tree

    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()
    parent = rollout._cursor

    async def scored(child):
        return 1.0

    async def unscored(child):
        raise UnscoredChildError("the verifier never reported a reward")

    assert await rollout.branch(2, run_child=scored) == 1.0
    assert parent.state["value"] == 1.0

    assert await rollout.branch(2, run_child=unscored) is None

    assert "value" not in parent.state
    run_dir = tmp_path / "lineage"
    run_dir.mkdir()
    serialize_tree(rollout.tree, run_dir=run_dir)
    nodes = {
        node["id"]: node
        for node in json.loads((run_dir / "tree.json").read_text())["nodes"]
    }
    assert "value" not in nodes[parent.id]


async def test_a_rescored_fork_still_records_its_own_value(tmp_path: Path):
    """The counterpart: clearing an undefined V must not clear a defined one.

    Guards the fix from "fix(branch): clear a stale branch value when V is
    undefined" against being written as an unconditional ``pop`` — a second
    fork whose children *were* scored publishes its own V, overwriting the
    first fork's.
    """
    from benchflow.branch import UnscoredChildError

    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()
    parent = rollout._cursor

    async def unscored(child):
        raise UnscoredChildError("the verifier never reported a reward")

    async def scored(child):
        return 0.5

    assert await rollout.branch(2, run_child=unscored) is None
    assert "value" not in parent.state

    assert await rollout.branch(2, run_child=scored) == 0.5
    assert parent.state["value"] == 0.5


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
    # the real restore path ran once per child: cp from the snapshot dir
    restore_cmds = [c for c in sandbox.exec_calls if c.startswith("cp ")]
    assert len(restore_cmds) == 2


# The parent's on-disk evidence across a fork
#
# Children share the parent's sandbox, and the sandbox bind-mounts the parent
# rollout's agent/artifacts/verifier directories into the container
# (``DockerSandbox.__init__``: host_*_path -> /logs/*), with ``restore()``
# deliberately replaying those mounts. So a child's writes to /logs/verifier
# land in the *parent's* verifier directory on the host. These tests stand in
# for that at the unit layer: the child runner writes to the same host paths a
# mounted container would.


def _mounted_rollout(tmp_path: Path) -> Rollout:
    """A rollout whose run directory exists, with the parent's own evidence."""
    from benchflow.task.paths import RolloutPaths

    rollout = _rollout(tmp_path)
    run_dir = tmp_path / "run"
    rollout._rollout_dir = run_dir
    paths = RolloutPaths(rollout_dir=run_dir)
    paths.mkdir()
    (paths.verifier_dir / "reward.txt").write_text("parent-1.0\n")
    (paths.verifier_dir / "test-stdout.txt").write_text("parent verifier output\n")
    (paths.agent_dir / "acp_trajectory.jsonl").write_text('{"parent": true}\n')
    (paths.artifacts_dir / "answer.md").write_text("the parent's answer\n")
    return rollout


def _mounted_writer(rollout: Rollout, marks: list[str]):
    """A child runner that writes through the shared mounts, as a child does."""
    from benchflow.task.paths import RolloutPaths

    paths = RolloutPaths(rollout_dir=rollout._rollout_dir)
    order = iter(marks)

    async def run_child(child):
        mark = next(order)
        # What a real child does first: clear_verifier_output_dir wipes
        # /logs/verifier — which is the parent's directory — then the verifier
        # writes the child's own files there.
        for entry in sorted(paths.verifier_dir.iterdir()):
            entry.unlink()
        (paths.verifier_dir / "reward.txt").write_text(f"{mark}\n")
        (paths.verifier_dir / f"{mark}-only.txt").write_text("child-scoped\n")
        (paths.agent_dir / "acp_trajectory.jsonl").write_text(f'{{"child": "{mark}"}}\n')
        (paths.artifacts_dir / "answer.md").write_text(f"{mark} answer\n")
        return 1.0

    return run_child


async def test_branch_children_do_not_clobber_the_parents_verifier_evidence(
    tmp_path: Path,
):
    """The parent's own scoring evidence must survive its children.

    Without the fix the run ends with the last child's ``reward.txt`` sitting
    where the parent's belongs — and, because a child clears /logs/verifier
    before writing, the parent's other files are gone entirely. No reported
    number is wrong, but the parent's row in the ablation table can no longer
    be audited against anything on disk.
    """
    from benchflow.task.paths import RolloutPaths

    rollout = _mounted_rollout(tmp_path)
    rollout._environment = FakeEnvironment()
    paths = RolloutPaths(rollout_dir=rollout._rollout_dir)

    value = await rollout.branch(
        2, run_child=_mounted_writer(rollout, ["child-a", "child-b"])
    )

    assert value == 1.0
    assert (paths.verifier_dir / "reward.txt").read_text() == "parent-1.0\n"
    assert (
        paths.verifier_dir / "test-stdout.txt"
    ).read_text() == "parent verifier output\n"
    assert (paths.agent_dir / "acp_trajectory.jsonl").read_text() == '{"parent": true}\n'
    assert (paths.artifacts_dir / "answer.md").read_text() == "the parent's answer\n"
    # nothing of the children's is left mixed into the parent's directories
    assert not (paths.verifier_dir / "child-a-only.txt").exists()
    assert not (paths.verifier_dir / "child-b-only.txt").exists()


async def test_each_child_keeps_the_output_it_wrote_to_the_shared_mounts(
    tmp_path: Path,
):
    """Preserving the parent must not throw the children's evidence away.

    Each child's mounted output is moved into its own artifact directory under
    ``mounted/``, keyed by node id — so the two arms of an ablation each have
    their verifier output, and neither is the other's.
    """
    rollout = _mounted_rollout(tmp_path)
    rollout._environment = FakeEnvironment()

    await rollout.branch(2, run_child=_mounted_writer(rollout, ["child-a", "child-b"]))

    children = rollout._rollout_dir / "branches" / "root" / "children"
    first = children / "n1" / "mounted"
    second = children / "n2" / "mounted"
    assert (first / "verifier" / "reward.txt").read_text() == "child-a\n"
    assert (second / "verifier" / "reward.txt").read_text() == "child-b\n"
    assert (first / "agent" / "acp_trajectory.jsonl").read_text() == '{"child": "child-a"}\n'
    assert (second / "artifacts" / "answer.md").read_text() == "child-b answer\n"
    # each child's directory holds only its own files
    assert not (first / "verifier" / "child-b-only.txt").exists()
    assert not (second / "verifier" / "child-a-only.txt").exists()
    # and the parent's entries are back at their canonical paths, with the
    # transient hold directory gone (its presence would mean the fork died)
    assert not (rollout._rollout_dir / "branches" / "root" / "parent").exists()


async def test_a_child_that_raises_still_leaves_its_mounted_output(tmp_path: Path):
    """A crashed child's verifier output is often the only evidence of why —
    so the hand-off happens in the same finally that records its wall clock,
    and the parent's evidence is restored even though branch() propagates."""
    from benchflow.task.paths import RolloutPaths

    rollout = _mounted_rollout(tmp_path)
    rollout._environment = FakeEnvironment()
    paths = RolloutPaths(rollout_dir=rollout._rollout_dir)

    async def run_child(child):
        (paths.verifier_dir / "test-stdout.txt").write_text("boom: exit 137\n")
        raise RuntimeError("verifier crashed")

    with pytest.raises(RuntimeError, match="verifier crashed"):
        await rollout.branch(2, run_child=run_child)

    child_mounted = (
        rollout._rollout_dir / "branches" / "root" / "children" / "n1" / "mounted"
    )
    assert (child_mounted / "verifier" / "test-stdout.txt").read_text() == (
        "boom: exit 137\n"
    )
    assert (paths.verifier_dir / "reward.txt").read_text() == "parent-1.0\n"


async def test_a_rollout_without_a_run_directory_still_branches(tmp_path: Path):
    """The custody is a no-op when there is nowhere to keep anything — a
    branch-first rollout that never ran setup() must not start failing."""
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()

    async def run_child(child):
        return 0.5

    assert await rollout.branch(2, run_child=run_child) == 0.5
