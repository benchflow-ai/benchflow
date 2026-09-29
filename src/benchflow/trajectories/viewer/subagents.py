"""Nest subagent steps under the tool call that spawned them.

Claude captures attribute a child agent's events to its spawning tool call
through ``parent_tool_call_id``. The scope and depth rules mirror the ATIF
export's ``_SubagentPlan`` (``export_atif.py``) so the viewer and the export
agree on what nests where.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Any

from benchflow.trajectories._export_common import is_attributed_child_event
from benchflow.trajectories.export_atif import MAX_SUBAGENT_DEPTH

from .models import (
    Step,
    SubagentGroupReason,
    SubagentGroupStep,
    SubagentTrace,
    ToolStep,
)


def _spawn_text(event: dict[str, Any] | None, key: str) -> str | None:
    """A non-empty string from a spawning tool call's raw input.

    Claude's Task/Agent tool input carries ``subagent_type`` and
    ``description``; other harnesses simply leave these unset.
    """
    raw = event.get("raw_input") if event is not None else None
    value = raw.get(key) if isinstance(raw, dict) else None
    return value if isinstance(value, str) and value else None


def nest_subagent_steps(
    steps: list[Step], sources: list[dict[str, Any] | None]
) -> list[Step]:
    """Nest attributed subagent steps under the tool call that spawned them.

    Scope assignment and reachability follow the ATIF export's
    ``_SubagentPlan`` (``export_atif.py``) so both views agree: an event
    belongs to the scope named by its ``parent_tool_call_id``; a scope hangs
    under the scope holding the first captured tool call with that id, found
    breadth-first from the main agent, at most ``MAX_SUBAGENT_DEPTH`` levels
    deep; a scope whose spawning call was never captured hangs under the main
    agent, here as a labelled group. Unlike the export, unreachable scopes
    (cyclic attribution, or deeper than the cap) are not dropped: they render
    as labelled groups too, so every recorded event stays visible.
    """
    scope_of: list[str | None] = []
    scopes: dict[str | None, list[int]] = {None: []}
    call_ids: dict[int, str] = {}
    spawns: dict[str, int] = {}
    for index, (step, event) in enumerate(zip(steps, sources, strict=True)):
        scope = (
            str(event["parent_tool_call_id"])
            if event is not None and is_attributed_child_event(event)
            else None
        )
        scope_of.append(scope)
        scopes.setdefault(scope, []).append(index)
        if isinstance(step, ToolStep) and event is not None:
            call_id = str(event.get("tool_call_id") or "")
            if call_id:
                call_ids[index] = call_id
                spawns.setdefault(call_id, index)
    if len(scopes) == 1:
        return steps

    depth: dict[str | None, int] = {None: 0}
    children: dict[str | None, list[str]] = {}
    queue: deque[str | None] = deque([None])
    while queue:
        scope = queue.popleft()
        found = [
            call_ids[index]
            for index in scopes[scope]
            if index in call_ids
            and call_ids[index] in scopes
            and spawns[call_ids[index]] == index
        ]
        if scope is None:
            found += [
                child for child in scopes if child is not None and child not in spawns
            ]
        for child in found:
            if child in depth or depth[scope] >= MAX_SUBAGENT_DEPTH:
                continue
            depth[child] = depth[scope] + 1
            children.setdefault(scope, []).append(child)
            queue.append(child)

    def unreachable_reason(scope: str) -> SubagentGroupReason:
        seen: set[str] = set()
        current: str | None = scope
        while current is not None and current not in depth:
            if current not in spawns:
                return "parent_not_captured"
            if current in seen:
                return "cyclic"
            seen.add(current)
            current = scope_of[spawns[current]]
        return "too_deep"

    def trace(scope: str, nested: list[Step]) -> SubagentTrace:
        spawn_index = spawns.get(scope)
        spawn = sources[spawn_index] if spawn_index is not None else None
        return SubagentTrace(
            parent_tool_call_id=scope,
            depth=depth.get(scope, 0),
            steps=nested,
            subagent_type=_spawn_text(spawn, "subagent_type"),
            description=_spawn_text(spawn, "description"),
        )

    groups = 0

    def group(scope: str, reason: SubagentGroupReason, nested: list[Step]) -> Step:
        nonlocal groups
        groups += 1
        return SubagentGroupStep(
            gid=f"g{groups}", reason=reason, subagent=trace(scope, nested)
        )

    def build(scope: str | None) -> list[Step]:
        spawned = set(children.get(scope, []))
        placed: list[tuple[int, Step]] = []
        for index in scopes[scope]:
            step = steps[index]
            call_id = call_ids.get(index)
            if (
                isinstance(step, ToolStep)
                and call_id in spawned
                and spawns[call_id] == index
            ):
                step = replace(step, subagent=trace(call_id, build(call_id)))
            placed.append((index, step))
        if scope is None:
            # Groups sit at the capture position of their first event.
            for child in children.get(None, []):
                if child not in spawns:
                    placed.append(
                        (
                            scopes[child][0],
                            group(child, "parent_not_captured", build(child)),
                        )
                    )
            for orphan, indexes in scopes.items():
                if orphan is not None and orphan not in depth:
                    placed.append(
                        (
                            indexes[0],
                            group(
                                orphan,
                                unreachable_reason(orphan),
                                [steps[index] for index in indexes],
                            ),
                        )
                    )
        placed.sort(key=lambda item: item[0])
        return [step for _, step in placed]

    return build(None)


def count_subagents(steps: list[Step]) -> int:
    count = 0
    pending: list[list[Step]] = [steps]
    while pending:
        for step in pending.pop():
            trace = (
                step.subagent
                if isinstance(step, (ToolStep, SubagentGroupStep))
                else None
            )
            if trace is not None:
                count += 1
                pending.append(trace.steps)
    return count
