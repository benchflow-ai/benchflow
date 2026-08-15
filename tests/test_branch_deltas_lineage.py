"""Regression tests for branch deltas + lineage artifacts (rollout-branching RFC, FrontierPhysics#73).

A branch child's delta (RFC §3.3) is the recorded exactly-one-controlled-change
it runs under: v1 executes ``injected_prompt`` (the child's user-visible first
message), while ``environment_ref`` / ``config_override`` / ``skill_mode`` are
schema-and-provenance-stable but fail closed until the child-as-fresh-rollout
follow-on. Lineage (RFC §3.4) makes a branch leave evidence: a deterministic
``tree.json`` plus per-child ``provenance.json`` / ``reward.json``, with
artifact-write failures isolated from the branch result.

These are unit tests against fakes — no Docker, Daytona, or API keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow._utils.content_address import sha256_prefixed
from benchflow.branch import StageSnapshot
from benchflow.branch_delta import BranchDelta
from benchflow.branch_lineage import child_provenance, serialize_tree
from benchflow.environment.protocol import StateSnapshot
from benchflow.rollout import _TERMINAL_PHASES, Rollout, RolloutConfig, Scene
from benchflow.rollout_branch import BranchDeltaNotSupported
from benchflow.sandbox.protocol import SandboxImage
from benchflow.trajectories.tree import RolloutTree


class FakeEnvironment:
    """Environment-plane stand-in recording snapshot/restore calls."""

    def __init__(self) -> None:
        self.snapshots: list[StateSnapshot] = []
        self.restored: list[StateSnapshot] = []

    async def snapshot(self) -> StateSnapshot:
        snap = StateSnapshot(id=f"env-snap-{len(self.snapshots) + 1}", path="/tmp/x")
        self.snapshots.append(snap)
        return snap

    async def restore(self, snap: StateSnapshot) -> None:
        self.restored.append(snap)


class FakeSnapSandbox:
    """Snapshot-capable Sandbox stand-in."""

    supports_snapshot = True

    def __init__(self) -> None:
        self.snapshots: list[SandboxImage] = []
        self.restored: list[SandboxImage] = []

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        img = SandboxImage(provider="fake", ref=f"bf-snap-{len(self.snapshots) + 1}")
        self.snapshots.append(img)
        return img

    async def restore(self, image: SandboxImage) -> None:
        self.restored.append(image)


def _rollout(tmp_path: Path) -> Rollout:
    return Rollout(
        RolloutConfig(task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy")])
    )


def _fake_agent_boundary(monkeypatch, received_prompts: list[list[str] | None]):
    """Fake connect/execute/verify at the class boundary, recording prompts."""

    async def fake_connect(self):
        self._acp_client = object()

    async def fake_disconnect(self):
        self._acp_client = None

    async def fake_execute(self, prompts=None, *, node=None):
        received_prompts.append(prompts)
        return [], 0

    async def fake_verify(self):
        return {"reward": 1.0}

    monkeypatch.setattr(Rollout, "connect", fake_connect)
    monkeypatch.setattr(Rollout, "disconnect", fake_disconnect)
    monkeypatch.setattr(Rollout, "execute", fake_execute)
    monkeypatch.setattr(Rollout, "verify", fake_verify)


# 1. BranchDelta schema: validation and content-addressed provenance


def test_branch_delta_rejects_an_unknown_skill_mode():
    """skill_mode outside {no-skill, with-skill} fails at construction."""
    with pytest.raises(ValueError, match="skill_mode"):
        BranchDelta(skill_mode="self-gen")
    with pytest.raises(ValueError, match="skill_mode"):
        BranchDelta(skill_mode="skills-on")


def test_branch_delta_is_empty_only_when_every_field_is_unset():
    assert BranchDelta().is_empty
    assert not BranchDelta(injected_prompt="plan").is_empty
    assert not BranchDelta(skill_mode="no-skill").is_empty


def test_provenance_hashes_are_content_addressed_and_key_order_stable():
    """Same config_override content -> same sha; different content -> different.

    The hash is over canonical JSON (sort_keys=True), so key order must not
    change it — the run-level overlay's exact hashing (#790).
    """
    a = BranchDelta(config_override={"agent": {"timeout_sec": 60}, "metadata": {}})
    b = BranchDelta(config_override={"metadata": {}, "agent": {"timeout_sec": 60}})
    c = BranchDelta(config_override={"agent": {"timeout_sec": 61}})

    sha_a = a.provenance_dict()["config_override_sha256"]
    sha_b = b.provenance_dict()["config_override_sha256"]
    sha_c = c.provenance_dict()["config_override_sha256"]

    assert sha_a == sha_b
    assert sha_a != sha_c
    assert sha_a.startswith("sha256:")


def test_provenance_records_the_prompt_hash_never_the_prompt_text():
    """No raw prompt content in provenance — only its sha256 digest."""
    prompt = "SECRET ORACLE PLAN: mine the sqlite db first."
    delta = BranchDelta(injected_prompt=prompt, skill_mode="with-skill")

    prov = delta.provenance_dict()

    assert prov["injected_prompt_sha256"] == sha256_prefixed(prompt.encode())
    assert prov["skill_mode"] == "with-skill"
    assert prov["environment_ref"] is None
    assert prov["config_override_sha256"] is None
    assert prompt not in json.dumps(prov)


# 2. Engine: delta validation fails closed before anything runs


async def test_deltas_length_must_match_n(tmp_path: Path):
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()

    async def run_child(child):
        return 1.0

    with pytest.raises(ValueError, match="one entry per child"):
        await rollout.branch(2, run_child=run_child, deltas=[None])


@pytest.mark.parametrize(
    "delta",
    [
        BranchDelta(environment_ref="env0@outage"),
        BranchDelta(config_override={"agent": {"timeout_sec": 60}}),
        BranchDelta(skill_mode="with-skill"),
    ],
    ids=["environment_ref", "config_override", "skill_mode"],
)
async def test_unsupported_delta_fields_fail_closed_before_any_child_runs(
    tmp_path: Path, delta: BranchDelta
):
    """environment_ref/config_override/skill_mode raise BranchDeltaNotSupported.

    The typed error names the field, points at the RFC follow-on, and fires
    before quiesce/checkpoint — no snapshot taken, no child forked or run.
    """
    rollout = _rollout(tmp_path)
    env = FakeEnvironment()
    rollout._environment = env
    parent = rollout._cursor
    ran: list[str] = []

    async def run_child(child):
        ran.append(child.id)
        return 1.0

    field_name = next(
        name
        for name in ("environment_ref", "config_override", "skill_mode")
        if getattr(delta, name) is not None
    )
    with pytest.raises(BranchDeltaNotSupported, match=field_name) as excinfo:
        await rollout.branch(2, run_child=run_child, deltas=[None, delta])

    assert "use_prebuilt_env" in str(excinfo.value)
    assert isinstance(excinfo.value, NotImplementedError)
    # fail closed BEFORE any child ran: nothing snapshotted, nothing forked
    assert ran == []
    assert env.snapshots == []
    assert parent.children == []


async def test_injected_prompt_with_an_explicit_run_child_is_rejected(
    tmp_path: Path,
):
    """An injected_prompt delta needs the default runner to deliver it — a
    caller-supplied run_child owns the child's prompts, so the combination
    fails closed instead of silently not delivering the injection."""
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()

    async def run_child(child):
        return 1.0

    with pytest.raises(ValueError, match="injected_prompt"):
        await rollout.branch(
            2,
            run_child=run_child,
            deltas=[BranchDelta(injected_prompt="go"), None],
        )


# 3. Engine: injected_prompt executes and is provenance-recorded


async def test_injected_prompt_becomes_the_childs_continuation_prompt(
    tmp_path: Path, monkeypatch
):
    """The child runs with the injected prompt as its user-visible message;
    the zero-delta sibling keeps the rollout's resolved prompts (None here)."""
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()
    received: list[list[str] | None] = []
    _fake_agent_boundary(monkeypatch, received)

    prompt = "Execute PLAN.md verbatim; do not re-research."
    value = await rollout.branch(
        2, deltas=[None, BranchDelta(injected_prompt=prompt)]
    )

    assert value == 1.0
    assert received == [None, [prompt]]


async def test_injected_prompt_provenance_records_the_hash_not_the_text(
    tmp_path: Path, monkeypatch
):
    """children/<i>/provenance.json carries injected_prompt_sha256 only."""
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rollout._rollout_dir = run_dir
    received: list[list[str] | None] = []
    _fake_agent_boundary(monkeypatch, received)

    prompt = "Execute PLAN.md verbatim; do not re-research."
    await rollout.branch(2, deltas=[None, BranchDelta(injected_prompt=prompt)])

    prov = json.loads((run_dir / "children" / "1" / "provenance.json").read_text())
    assert prov["delta"]["injected_prompt_sha256"] == sha256_prefixed(prompt.encode())
    assert prompt not in (run_dir / "children" / "1" / "provenance.json").read_text()
    zero = json.loads((run_dir / "children" / "0" / "provenance.json").read_text())
    assert zero["delta"]["injected_prompt_sha256"] is None


# 4. Lineage: tree.json structure, determinism, and failure isolation


async def test_tree_json_records_nodes_snapshot_refs_rewards_and_deltas(
    tmp_path: Path, monkeypatch
):
    """tree.json after a composed branch: schema_version 1, one entry per
    node, the branch point carrying both layers' snapshot refs and V(parent),
    each child carrying its reward and delta provenance."""
    rollout = _rollout(tmp_path)
    env, sandbox = FakeEnvironment(), FakeSnapSandbox()
    rollout._environment = env
    rollout._env = sandbox
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rollout._rollout_dir = run_dir
    received: list[list[str] | None] = []
    _fake_agent_boundary(monkeypatch, received)

    prompt = "Use the oracle plan."
    await rollout.branch(
        2,
        snapshot_layers={"environment", "sandbox"},
        deltas=[None, BranchDelta(injected_prompt=prompt)],
    )

    payload = json.loads((run_dir / "tree.json").read_text())
    assert payload["schema_version"] == 1
    assert payload["cut_point"] is None

    nodes = {node["id"]: node for node in payload["nodes"]}
    root = nodes["root"]
    assert root["parent"] is None
    assert root["snapshot"] == {"environment": "env-snap-1", "sandbox": "bf-snap-1"}
    assert root["value"] == 1.0

    children = [node for node in payload["nodes"] if node["parent"] == "root"]
    assert len(children) == 2
    assert [child["reward"] for child in children] == [1.0, 1.0]
    assert children[0]["delta"]["injected_prompt_sha256"] is None
    assert children[1]["delta"]["injected_prompt_sha256"] == sha256_prefixed(
        prompt.encode()
    )
    assert prompt not in (run_dir / "tree.json").read_text()


def test_serialize_tree_is_deterministic_across_calls(tmp_path: Path):
    """Two serializations of the same tree are byte-identical — sorted keys,
    trailing newline, no wall-clock timestamps."""
    tree = RolloutTree()
    snap = StageSnapshot(
        environment_ref=StateSnapshot(id="env-snap-1", path="/tmp/x"),
        sandbox_ref=SandboxImage(provider="fake", ref="bf-snap-1"),
        stage="env-ready",
    )
    tree.root.state["snapshot"] = snap
    tree.root.state["value"] = 0.5
    for reward in (0.0, 1.0):
        child = tree.attach(tree.root)
        child.state["reward"] = reward
    deltas = [None, BranchDelta(injected_prompt="plan")]

    first_dir, second_dir = tmp_path / "a", tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    first = serialize_tree(tree, run_dir=first_dir, snapshot=snap, deltas=deltas)
    second = serialize_tree(tree, run_dir=second_dir, snapshot=snap, deltas=deltas)

    text = first.read_text()
    assert text == second.read_text()
    assert text.endswith("\n")
    # the stage tag comes from the StageSnapshot itself
    root_entry = next(
        node for node in json.loads(text)["nodes"] if node["id"] == "root"
    )
    assert root_entry["stage"] == "env-ready"


async def test_artifact_write_failure_never_corrupts_the_branch_result(
    tmp_path: Path,
):
    """An unwritable run dir (here: a plain file) is logged and swallowed —
    the branch still returns V(parent) and the tree still grew."""
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()
    not_a_dir = tmp_path / "run"
    not_a_dir.write_text("occupied")
    rollout._rollout_dir = not_a_dir
    parent = rollout._cursor

    returns = iter([0.0, 1.0])

    async def run_child(child):
        return next(returns)

    value = await rollout.branch(2, run_child=run_child)

    assert value == 0.5
    assert len(parent.children) == 2
    assert not_a_dir.read_text() == "occupied"


async def test_children_artifacts_are_written_and_parseable(tmp_path: Path):
    """children/<index>/ gets provenance.json (kind benchflow-branch, legacy
    env-only snapshot ref, cursor-tagged stage) and reward.json."""
    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rollout._rollout_dir = run_dir

    returns = iter([0.25, 0.75])

    async def run_child(child):
        return next(returns)

    await rollout.branch(2, run_child=run_child)

    for index, expected in enumerate([0.25, 0.75]):
        child_dir = run_dir / "children" / str(index)
        prov = json.loads((child_dir / "provenance.json").read_text())
        assert prov["kind"] == "benchflow-branch"
        assert prov["parent_rollout"] == str(run_dir)
        assert prov["branch_stage"] == "cursor:root"
        assert prov["snapshot_ref"] == {"environment": "env-snap-1", "sandbox": None}
        assert prov["cut_point"] is None
        assert prov["delta"] == BranchDelta().provenance_dict()
        assert json.loads((child_dir / "reward.json").read_text()) == {
            "reward": expected
        }


def test_child_provenance_shape_matches_the_rfc():
    """child_provenance builds the RFC §3.4 dict for a composed snapshot."""
    snap = StageSnapshot(
        environment_ref=StateSnapshot(id="env-snap-9", path="/tmp/x"),
        sandbox_ref=SandboxImage(provider="fake", ref="bf-snap-9"),
        stage="env-ready",
    )
    prov = child_provenance(
        "/runs/parent-1",
        branch_stage="env-ready",
        snapshot=snap,
        delta=BranchDelta(skill_mode="no-skill"),
    )
    assert prov == {
        "kind": "benchflow-branch",
        "parent_rollout": "/runs/parent-1",
        "branch_stage": "env-ready",
        "snapshot_ref": {"sandbox": "bf-snap-9", "environment": "env-snap-9"},
        "cut_point": None,
        "delta": {
            "environment_ref": None,
            "config_override_sha256": None,
            "skill_mode": "no-skill",
            "injected_prompt_sha256": None,
        },
    }


# 5. "branched" is a terminal phase


async def test_branched_phase_is_terminal_and_result_is_reachable(
    tmp_path: Path, monkeypatch
):
    """Rollout.result no longer returns None after a branch-first workflow.

    Guards the terminal-phase change: "branched" joined _TERMINAL_PHASES and
    the result gate, so a branch-first run can read its result without a
    linear verify()/cleanup() pass.
    """
    assert "branched" in _TERMINAL_PHASES

    rollout = _rollout(tmp_path)
    rollout._environment = FakeEnvironment()

    async def run_child(child):
        return 1.0

    assert rollout.result is None  # pre-branch: no terminal phase yet
    await rollout.branch(2, run_child=run_child)
    assert rollout._phase == "branched"

    sentinel = object()
    monkeypatch.setattr(rollout, "_build_result", lambda: sentinel)
    assert rollout.result is sentinel
