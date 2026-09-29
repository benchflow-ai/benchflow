"""Isolated (and parallel) branch children, and nested forks from a child.

A fork's children used to run one after another in the parent's own sandbox (restore, run, restore,
...), so a fork of n agent children took n agent runs of wall-clock time.
``branch(isolate_children=True, concurrency=k)`` runs each child as its own
sub-rollout in its own sandbox created from the snapshot, at most k at once.
The parent's sandbox is not touched by the children; its restore semantics
are unchanged. Each child is a full trial folder at
``branches/<fork>/children/<node>`` and shares the root rollout's tree, so a
child can branch again and ``tree.json`` records the nested fork.

Unit tests against a scripted rollout whose "sandbox" is a list of prompts;
no Docker, Daytona or credentials.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import ClassVar

import pytest

from benchflow.rollout import Rollout, RolloutConfig
from benchflow.rollout_branch import BranchChild, require_safe_branch_world
from benchflow.sandbox.protocol import SandboxImage
from benchflow.trajectories.tree import Step

# Provider storage shared by every sandbox: snapshot ref -> world.
IMAGES: dict[str, list[str]] = {}


class WorldSandbox:
    supports_snapshot = True

    def __init__(self, owner: IsoRollout, *, fast_start: bool = False) -> None:
        self.owner = owner
        self.fast_start = fast_start
        self.restores: list[str] = []
        self.deleted: list[str] = []
        self.start_ref: str | None = None

    async def snapshot(self, name=None) -> SandboxImage:
        ref = f"bf-snap-{len(IMAGES)}"
        IMAGES[ref] = list(self.owner.world)
        return SandboxImage(provider="fake", ref=ref)

    async def restore(self, image: SandboxImage) -> None:
        self.restores.append(image.ref)
        self.owner.world = list(IMAGES[image.ref])

    async def delete_snapshot(self, image: SandboxImage) -> bool:
        self.deleted.append(image.ref)
        return True


class FastStartSandbox(WorldSandbox):
    """A provider that can create the sandbox straight from a snapshot."""

    def start_from_snapshot(self, image: SandboxImage) -> bool:
        self.start_ref = image.ref
        return True


class IsoRollout(Rollout):
    all: ClassVar[list[IsoRollout]] = []
    sandbox_type: ClassVar[type[WorldSandbox]] = WorldSandbox
    running = 0
    peak = 0
    gate: asyncio.Event | None = None

    def __init__(self, config) -> None:
        super().__init__(config)
        self.world: list[str] = []
        self.events: list[str] = []
        IsoRollout.all.append(self)

    async def setup(self) -> None:
        cfg = self._config
        self._rollout_name = cfg.rollout_name or "root"
        self._rollout_dir = Path(cfg.jobs_dir) / cfg.job_name / self._rollout_name
        self._rollout_dir.mkdir(parents=True)
        self._resolved_prompts = ["Do it."]
        self._env = IsoRollout.sandbox_type(self)
        self.events.append("setup")

    async def start(self) -> None:
        if self._env.start_ref is not None:
            self.world = list(IMAGES[self._env.start_ref])
        self.events.append("start")

    async def install_agent(self) -> None:
        self.events.append("install_agent")

    async def connect(self) -> None:
        self._acp_client = object()

    async def disconnect(self) -> None:
        self._acp_client = None

    async def execute(self, prompts=None, *, node=None):
        IsoRollout.running += 1
        IsoRollout.peak = max(IsoRollout.peak, IsoRollout.running)
        try:
            if IsoRollout.gate is not None:
                await asyncio.wait_for(IsoRollout.gate.wait(), 0.2)
        except TimeoutError:
            pass
        finally:
            IsoRollout.running -= 1
        self.events.append(f"execute:{prompts}:{self.world}")
        self.world.extend(prompts)
        step = Step(id=f"s{len(self.world)}", data={"event": None})
        if node is not None:
            self._cursor = self._tree.populate(node, step)
        else:
            self._cursor = self._tree.advance(self._cursor, step)
        return [], 0

    async def verify(self):
        self._verify_calls += 1
        self._rewards = {"reward": 1.0 if self.world[-1:] == ["Do it."] else 0.0}
        return self._rewards

    async def finalize(self):
        self.events.append("finalize")
        (self._rollout_dir / "result.json").write_text(
            json.dumps({"rewards": self._rewards})
        )

    async def cleanup(self):
        self.events.append("cleanup")


@pytest.fixture(autouse=True)
def _reset():
    def reset() -> None:
        IMAGES.clear()
        IsoRollout.all = []
        IsoRollout.sandbox_type = WorldSandbox
        IsoRollout.running = IsoRollout.peak = 0
        IsoRollout.gate = None

    reset()
    yield
    # Other modules (test_branch_reuse_snapshot) import IsoRollout; leave it
    # as found so a test that swaps the sandbox type cannot leak into them.
    reset()


async def _root(tmp_path: Path) -> IsoRollout:
    rollout = IsoRollout(
        RolloutConfig(
            task_path=tmp_path / "task",
            agent="dummy",
            jobs_dir=tmp_path / "jobs",
            job_name="job",
            rollout_name="task__root",
        )
    )
    await rollout.setup()
    rollout.world = ["draft"]
    return rollout


def _runner(prompts: dict[str | None, str]):
    async def run(node, *, child: BranchChild):
        sub = child.rollout
        await sub.connect()
        await sub.execute([prompts[child.label]], node=node)
        return (await sub.verify())["reward"]

    return run


def _tree(rollout) -> dict:
    return json.loads((rollout._rollout_dir / "tree.json").read_text())


async def test_isolated_children_run_in_their_own_sandboxes(tmp_path):
    root = await _root(tmp_path)
    value = await root.branch(
        2,
        _runner({"a": "Do it.", "b": "Other."}),
        snapshot_layers={"sandbox"},
        child_labels=["a", "b"],
        isolate_children=True,
    )
    assert value == 0.5
    subs = IsoRollout.all[1:]
    assert len(subs) == 2
    # Each child: its own sandbox, restored from the checkpoint, then run.
    for sub in subs:
        assert sub._env is not root._env
        assert sub._env.restores == ["bf-snap-0"]
        assert sub.events[:3] == ["setup", "start", "install_agent"]
        assert sub.events[3].endswith(":['draft']")  # started at the checkpoint
        assert sub.events[-1] == "finalize"
    # The parent's sandbox was restored once (parent restore) and never ran a child.
    assert root._env.restores == ["bf-snap-0"]
    assert root.world == ["draft"]
    fork = _tree(root)["forks"][0]
    assert fork["parent_restore"] == "restored"
    assert fork["children_mode"] == {
        "isolated": True,
        "concurrency": 1,
        "prewarm": 1,
        "child_retries": 0,
        "continue_after_child_failure": True,
    }
    for child, sub in zip(fork["children"], subs, strict=True):
        path = child["artifacts"]["path"]
        assert path == f"branches/{fork['id']}/children/{child['node_id']}"
        assert sub._rollout_dir == root._rollout_dir / path
        assert (root._rollout_dir / path / "observation.json").is_file()
        assert (root._rollout_dir / path / "result.json").is_file()
        assert child["reward_source"] == "verifier"
    assert root._env.deleted == ["bf-snap-0"]


async def test_children_steps_land_in_the_root_tree(tmp_path):
    root = await _root(tmp_path)
    parent = root._cursor
    await root.branch(
        2,
        _runner({"a": "Do it.", "b": "Other."}),
        snapshot_layers={"sandbox"},
        child_labels=["a", "b"],
        isolate_children=True,
    )
    assert [c.step_in is not None for c in parent.children] == [True, True]
    ids = {node["id"] for node in _tree(root)["nodes"]}
    assert {c.id for c in parent.children} <= ids
    assert root._cursor is parent


@pytest.mark.parametrize(("concurrency", "peak"), [(1, 1), (2, 2), (4, 3)])
async def test_concurrency_bounds_simultaneous_children(tmp_path, concurrency, peak):
    root = await _root(tmp_path)
    IsoRollout.gate = asyncio.Event()
    labels = ["a", "b", "c"]
    await root.branch(
        3,
        _runner(dict.fromkeys(labels, "Do it.")),
        snapshot_layers={"sandbox"},
        child_labels=labels,
        isolate_children=True,
        concurrency=concurrency,
    )
    assert IsoRollout.peak == peak


async def test_a_fast_start_provider_skips_the_restore(tmp_path):
    IsoRollout.sandbox_type = FastStartSandbox
    root = await _root(tmp_path)
    await root.branch(
        2,
        _runner({"a": "Do it.", "b": "Do it."}),
        snapshot_layers={"sandbox"},
        child_labels=["a", "b"],
        isolate_children=True,
    )
    for sub in IsoRollout.all[1:]:
        assert sub._env.start_ref == "bf-snap-0"
        assert sub._env.restores == []
        assert sub.events[3].endswith(":['draft']")


async def test_one_failing_child_does_not_stop_its_siblings(tmp_path):
    root = await _root(tmp_path)

    async def run(node, *, child: BranchChild):
        if child.label == "bad":
            raise ValueError("child failed")
        await child.rollout.connect()
        await child.rollout.execute(["Do it."], node=node)
        return (await child.rollout.verify())["reward"]

    with pytest.raises(ValueError):
        await root.branch(
            2,
            run,
            snapshot_layers={"sandbox"},
            child_labels=["bad", "good"],
            isolate_children=True,
            concurrency=2,
        )
    fork = _tree(root)["forks"][0]
    assert [c["status"] for c in fork["children"]] == ["failed", "scored"]
    assert fork["status"] == "partial"
    # Every child's sandbox was torn down, the parent restored, the snapshot freed.
    assert all(sub.events[-1] == "finalize" for sub in IsoRollout.all[1:])
    assert fork["parent_restore"] == "restored"
    assert root._env.deleted == ["bf-snap-0"]


async def test_restore_parent_false_still_skips_the_parent_restore(tmp_path):
    root = await _root(tmp_path)
    await root.branch(
        2,
        _runner({"a": "Do it.", "b": "Do it."}),
        snapshot_layers={"sandbox"},
        child_labels=["a", "b"],
        isolate_children=True,
        restore_parent=False,
    )
    assert root._env.restores == []
    assert _tree(root)["forks"][0]["parent_restore"] == "skipped"
    with pytest.raises(RuntimeError, match="restore_parent=False"):
        require_safe_branch_world(root)


async def test_a_child_can_branch_again(tmp_path):
    root = await _root(tmp_path)

    async def run(node, *, child: BranchChild):
        sub = child.rollout
        await sub.connect()
        await sub.execute(["step-" + child.label], node=node)
        if child.label == "a":
            # Nested fork from this child's own state.
            return await sub.branch(
                2,
                _runner({"a1": "Do it.", "a2": "Other."}),
                snapshot_layers={"sandbox"},
                child_labels=["a1", "a2"],
                isolate_children=True,
                concurrency=2,
            )
        return (await sub.verify())["reward"]

    await root.branch(
        2,
        run,
        snapshot_layers={"sandbox"},
        child_labels=["a", "b"],
        isolate_children=True,
    )
    tree = _tree(root)
    assert len(tree["forks"]) == 2
    outer = next(f for f in tree["forks"] if f["parent_node"] == "root")
    inner = next(f for f in tree["forks"] if f is not outer)
    child_a = outer["children"][0]["node_id"]
    parents = {n["id"]: n["parent"] for n in tree["nodes"]}
    # The inner fork hangs below child a's node: depth > 1 in one tree.
    node = inner["parent_node"]
    chain = []
    while node is not None:
        chain.append(node)
        node = parents[node]
    assert child_a in chain and chain[-1] == "root"
    assert inner["value"] == 0.5
    # Grandchildren archive under the root trial, with root-relative paths.
    for grandchild in inner["children"]:
        path = root._rollout_dir / grandchild["artifacts"]["path"]
        assert (path / "observation.json").is_file()
        lineage = json.loads((path / "observation.json").read_text())["lineage"]
        assert lineage["fork_id"] == inner["id"]
        assert lineage["parent_rollout"] == child_a
    # Grandchildren started from child a's state, not from the root checkpoint.
    grandkids = [
        sub
        for sub in IsoRollout.all
        if sub._rollout_name in {c["node_id"] for c in inner["children"]}
    ]
    assert all(g.events[3].endswith(":['draft', 'step-a']") for g in grandkids)
    # The child's result.json lists only its own fork; the root's lists both.
    assert outer["children"][0]["reward"] == 0.5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"snapshot_layers": {"environment", "sandbox"}}, "sandbox layer only"),
        ({"concurrency": 0}, "concurrency"),
        ({"isolate_children": False, "concurrency": 2}, "isolate_children"),
    ],
)
async def test_isolated_mode_is_validated_before_anything_runs(
    tmp_path, kwargs, message
):
    root = await _root(tmp_path)
    root._environment = object()
    options = {
        "snapshot_layers": {"sandbox"},
        "isolate_children": True,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        await root.branch(2, _runner({}), **options)
    assert IMAGES == {}


async def test_isolated_children_record_their_overhead_phases(tmp_path):
    """Where an isolated child's time goes beyond the agent and verifier:
    creating its sandbox from the snapshot, reinstalling the agent, and
    finalizing (result files, sandbox teardown)."""
    root = await _root(tmp_path)
    await root.branch(
        2,
        _runner({"a": "Do it.", "b": "Do it."}),
        snapshot_layers={"sandbox"},
        child_labels=["a", "b"],
        isolate_children=True,
    )
    for child in _tree(root)["forks"][0]["children"]:
        timing = child["timing_sec"]
        for phase in ("sandbox_from_snapshot", "install_agent", "finalize"):
            assert isinstance(timing[phase], float), phase


async def test_the_next_children_are_prepared_while_earlier_ones_run(tmp_path):
    """Pre-warm: with more children than --concurrency, the next
    child's sandbox is created and set up while the current one runs, at most
    `concurrency` prepared ahead, so sandboxes alive stay within 2 x K."""
    root = await _root(tmp_path)
    alive = {"now": 0, "peak": 0}
    order: list[str] = []

    class Tracking(WorldSandbox):
        def __init__(self, owner, **kw):
            super().__init__(owner, **kw)
            alive["now"] += 1
            alive["peak"] = max(alive["peak"], alive["now"])

    IsoRollout.sandbox_type = Tracking

    async def run(node, *, child):
        order.append(f"run:{child.label}")
        await asyncio.sleep(0.05)
        return 1.0

    original_finalize = IsoRollout.finalize

    async def finalize(self):
        await original_finalize(self)
        if self is not root:
            alive["now"] -= 1
            order.append(f"done:{self._rollout_name}")

    IsoRollout.finalize = finalize
    original_setup = IsoRollout.setup

    async def setup(self):
        await original_setup(self)
        order.append(f"setup:{self._rollout_name}")

    IsoRollout.setup = setup
    try:
        await root.branch(
            3,
            run,
            snapshot_layers={"sandbox"},
            child_labels=["a", "b", "c"],
            isolate_children=True,
            concurrency=1,
        )
    finally:
        IsoRollout.finalize = original_finalize
        IsoRollout.setup = original_setup
    fork = _tree(root)["forks"][0]
    nodes = [c["node_id"] for c in fork["children"]]
    # b is set up before a has finished: prepared while a runs.
    assert order.index(f"setup:{nodes[1]}") < order.index(f"done:{nodes[0]}")
    # ... but never more than K running and K prepared (+ the parent's sandbox).
    assert alive["peak"] <= 1 + 2 * 1
    assert fork["children_mode"]["prewarm"] == 1
    assert [c["status"] for c in fork["children"]] == ["scored"] * 3


class ServicePlane:
    """An environment plane whose services live in the sandbox's processes."""

    def __init__(self, owner: IsoRollout) -> None:
        self.owner = owner

    def validate_sandbox_restore(self) -> None:
        pass

    def prepare_sandbox_restore(self) -> None:
        self.owner.events.append("services:prepare")

    async def resume_after_sandbox_restore(self) -> None:
        self.owner.events.append("services:restarted")


class RestoreLoggingSandbox(WorldSandbox):
    async def restore(self, image: SandboxImage) -> None:
        self.owner.events.append("restore")
        await super().restore(image)


async def test_a_restored_child_sandbox_restarts_its_environment_services(tmp_path):
    """On a provider without
    start_from_snapshot (Docker) a child's start() provisions the environment
    plane's services, then restore() replaces the container and kills them;
    the child must restart them the way an in-place restore does
    (prepare_sandbox_restore / resume_after_sandbox_restore). Otherwise the
    in-sandbox fake model of the deterministic tier refuses connections in
    parallel children and checkpoint retries on Docker only."""
    IsoRollout.sandbox_type = RestoreLoggingSandbox
    original_start = IsoRollout.start

    async def start_with_plane(self) -> None:
        await original_start(self)
        self._environment = ServicePlane(self)

    IsoRollout.start = start_with_plane
    try:
        root = await _root(tmp_path)
        root._environment = ServicePlane(root)
        await root.branch(
            2,
            _runner({"a": "Do it.", "b": "Other."}),
            snapshot_layers={"sandbox"},
            child_labels=["a", "b"],
            isolate_children=True,
        )
    finally:
        IsoRollout.start = original_start
    for sub in IsoRollout.all[1:]:
        assert sub.events[:6] == [
            "setup",
            "start",
            "services:prepare",
            "restore",
            "services:restarted",
            "install_agent",
        ]


async def test_a_fast_start_child_needs_no_service_restart(tmp_path):
    IsoRollout.sandbox_type = FastStartSandbox
    root = await _root(tmp_path)
    root._environment = ServicePlane(root)
    await root.branch(
        2,
        _runner({"a": "Do it.", "b": "Do it."}),
        snapshot_layers={"sandbox"},
        child_labels=["a", "b"],
        isolate_children=True,
    )
    for sub in IsoRollout.all[1:]:
        assert "services:restarted" not in sub.events
