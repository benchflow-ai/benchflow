"""Metrics derived from structured trajectory events."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def count_skill_invocations(trajectory: list[dict[str, Any]]) -> int:
    """Count ACP skill invocations from structured tool-call events.

    BenchFlow records ACP tool calls as dict events with ``type`` and ``kind``.
    A skill invocation is only counted when the event is explicitly a
    ``tool_call`` whose structured ``kind`` is ``"skill"``. Display text such as
    titles, messages, or tool names is intentionally ignored.
    """
    return sum(
        1
        for event in trajectory
        if isinstance(event, dict)
        and event.get("type") == "tool_call"
        and event.get("kind") == "skill"
    )


def result_skill_invocations(result: Mapping[str, Any]) -> int:
    """Return a result artifact's skill invocation count.

    New artifacts expose ``n_skill_invocations`` at the top level and inside
    ``agent_result``. Older artifacts may have neither; they are treated as
    zero so aggregate readers remain backward compatible.
    """
    value = result.get("n_skill_invocations")
    if value is None:
        agent_result = result.get("agent_result")
        if isinstance(agent_result, Mapping):
            value = agent_result.get("n_skill_invocations")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
