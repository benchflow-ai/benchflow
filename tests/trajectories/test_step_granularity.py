"""Step-level granularity in the rollout tree (closes #414).

Each ACP event (tool_call, agent_message, agent_thought, user_message)
inside a single ``execute()`` call must produce its own Step node — not
get collapsed into one. Branching, process rewards, and value estimation
all target individual Steps, so an execute() that emits N events must
walk the cursor down N nodes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.trajectories.tree import trajectory


def _rollout(tmp_path: Path) -> Rollout:
    return Rollout(
        RolloutConfig(task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy")])
    )


def _multi_event_session() -> tuple[list[dict], int]:
    """An execute_prompts return: 3 events, 1 tool_call among them."""
    events = [
        {"type": "agent_thought", "text": "let me look"},
        {
            "type": "tool_call",
            "tool_call_id": "tc1",
            "kind": "execute",
            "title": "ls -la",
            "status": "completed",
            "content": [],
        },
        {"type": "agent_message", "text": "done"},
    ]
    return events, 1


@pytest.mark.asyncio
async def test_n_events_in_one_execute_produce_n_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3 ACP events from one execute() => 3 Steps on the linear path."""
    rollout = _rollout(tmp_path)

    async def fake_execute_prompts(*_a, **_kw):
        return _multi_event_session()

    monkeypatch.setattr(rollout._planes, "execute_prompts", fake_execute_prompts)
    rollout._acp_client = object()  # only needs to be non-None

    root = rollout.tree.root
    await rollout.execute(["go"])

    leaf = rollout._cursor
    steps = trajectory(leaf)
    assert len(steps) == 3, f"expected 3 steps, got {len(steps)}"

    # Each Step carries a single event with the right type
    assert [s.data["event_type"] for s in steps] == [
        "agent_thought",
        "tool_call",
        "agent_message",
    ]
    # Only the tool_call step counts a tool call
    assert [s.data["n_tool_calls"] for s in steps] == [0, 1, 0]
    # The chain is degree-1: root -> n1 -> n2 -> leaf
    chain = [leaf, leaf.parent, leaf.parent.parent, leaf.parent.parent.parent]
    assert chain[-1] is root
    assert all(node.parent is not None or node is root for node in chain)


@pytest.mark.asyncio
async def test_single_event_execute_still_produces_one_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The single-event case is preserved — execute() still grows by 1."""
    rollout = _rollout(tmp_path)

    async def fake_execute_prompts(*_a, **_kw):
        return [{"type": "agent_message", "text": "hi"}], 0

    monkeypatch.setattr(rollout._planes, "execute_prompts", fake_execute_prompts)
    rollout._acp_client = object()

    root = rollout.tree.root
    await rollout.execute(["go"])

    leaf = rollout._cursor
    steps = trajectory(leaf)
    assert len(steps) == 1
    assert leaf.parent is root  # one new node on the chain
    assert steps[0].data["event_type"] == "agent_message"


@pytest.mark.asyncio
async def test_zero_event_execute_still_emits_one_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A no-event execute() still advances the cursor (one empty Step).

    Required so a branch child's pending node always gets populated and
    the post-execute phase still has a non-root cursor — see
    test_rollout_branch.test_branch_child_continuation_attaches_to_the_child_node.
    """
    rollout = _rollout(tmp_path)

    async def fake_execute_prompts(*_a, **_kw):
        return [], 0

    monkeypatch.setattr(rollout._planes, "execute_prompts", fake_execute_prompts)
    rollout._acp_client = object()

    root = rollout.tree.root
    await rollout.execute(["go"])

    leaf = rollout._cursor
    assert leaf is not root
    assert leaf.parent is root
    steps = trajectory(leaf)
    assert len(steps) == 1
    # The empty-event Step still has truthy data so consumers can iterate
    assert steps[0].data, "empty-event Step must carry recognizable data"


@pytest.mark.asyncio
async def test_branch_child_pending_node_populated_by_first_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``node=`` is given (a pending branch child), the first event
    populates the pending node in place; remaining events extend down the
    chain. The child's real work lands on the child node itself — no
    content-free placeholder Step."""
    rollout = _rollout(tmp_path)

    async def fake_execute_prompts(*_a, **_kw):
        return _multi_event_session()

    monkeypatch.setattr(rollout._planes, "execute_prompts", fake_execute_prompts)
    rollout._acp_client = object()

    pending = rollout.tree.attach(rollout.tree.root)
    assert pending.step_in is None  # pending — no Step yet

    await rollout.execute(["branch-child"], node=pending)

    # The first event populated the pending child in place
    assert pending.step_in is not None
    assert pending.step_in.data["event_type"] == "agent_thought"
    # The remaining events extended off the child — 3 events total, so
    # the chain from the pending child to the cursor has 3 Steps.
    steps = trajectory(rollout._cursor)
    assert len(steps) == 3
    assert [s.data["event_type"] for s in steps] == [
        "agent_thought",
        "tool_call",
        "agent_message",
    ]


@pytest.mark.asyncio
async def test_tool_call_count_invariant_across_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sum of per-Step ``n_tool_calls`` across the batch matches the
    ``new_tools`` value reported by execute_prompts — even when the agent
    shim under-counts tool_call events vs. what handle_update saw."""
    rollout = _rollout(tmp_path)

    async def fake_execute_prompts(*_a, **_kw):
        events = [
            {"type": "agent_message", "text": "thinking"},
            {
                "type": "tool_call",
                "tool_call_id": "tc1",
                "kind": "execute",
                "title": "ls",
                "status": "completed",
                "content": [],
            },
            {
                "type": "tool_call",
                "tool_call_id": "tc2",
                "kind": "execute",
                "title": "cat x",
                "status": "completed",
                "content": [],
            },
        ]
        return events, 2

    monkeypatch.setattr(rollout._planes, "execute_prompts", fake_execute_prompts)
    rollout._acp_client = object()

    await rollout.execute(["go"])
    steps = trajectory(rollout._cursor)
    assert sum(s.data["n_tool_calls"] for s in steps) == 2
