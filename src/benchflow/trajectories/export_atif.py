"""Export captured trajectories in ATIF (Agent Trajectory Interchange Format).

ATIF is the trajectory interchange format used by the Harbor evaluation
framework (the terminal-bench lineage). A BenchFlow rollout's ACP-style
trajectory events become one ATIF trajectory document that Harbor tooling
(trajectory validator, viewer, SFT/RL pipelines) ingests directly.

The record shape is pinned against the Harbor pydantic models
(``harbor-framework/harbor``, ``src/harbor/models/trajectories/*.py``) and
the ATIF RFC (``rfcs/0001-trajectory-format.md``), schema version
``ATIF-v1.7``. Spec constraints honoured here:

- ``steps`` is required with at least one entry; ``step_id`` is sequential
  starting from 1.
- ``reasoning_content``, ``tool_calls``, and ``metrics`` are only valid on
  ``source: "agent"`` steps.
- Every ``observation.results[].source_call_id`` must reference a
  ``tool_call_id`` in the *same* step's ``tool_calls``.
- Every embedded subagent carries a ``trajectory_id`` unique within its
  parent's ``subagent_trajectories``, and every ``subagent_trajectory_ref``
  resolves against that array.

Mapping from BenchFlow ACP trajectory events (see
:mod:`benchflow.trajectories._capture`):

- *prompts* and ``user_message`` events → ``user`` steps;
- ``agent_message`` → an ``agent`` step;
- ``agent_thought`` → ``reasoning_content`` on the next agent step, or a
  standalone agent step with an empty message when no agent event follows;
- ``tool_call`` → an ``agent`` step whose single ``tool_calls`` entry carries
  the ACP ``kind`` as ``function_name``. ACP updates carry no structured
  arguments, so ``arguments`` is ``{}`` and the ACP ``title``/``status`` ride
  in the tool call's ``extra`` (added in ATIF-v1.7) rather than being passed
  off as arguments. Captured output text becomes the step's ``observation``;
- ``oracle`` → an ``agent`` step rendering the command, mirroring
  :func:`benchflow.trajectories.export.acp_events_to_messages`.

Subagents (ATIF-v1.7 ``subagent_trajectories``): events the capture attributed
to a child via ``parent_tool_call_id`` never enter the parent's steps. Each
child's events become one embedded trajectory in the parent's
``subagent_trajectories`` (``trajectory_id`` ``subagent:<parent tool call
id>``), referenced from that tool call's observation result through
``subagent_trajectory_ref``. A child that spawns its own child nests the same
way, up to :data:`MAX_SUBAGENT_DEPTH` levels. Children whose spawning tool
call was not captured are embedded at the root without a reference. Children
beyond the depth cap, unreachable from the root (cyclic attribution), or with
no step to show are left out. The record's ``extra.acp_projection`` accounts
for every attributed event, and the raw ACP trajectory keeps them all.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from benchflow._utils.json_safe import dumps_finite
from benchflow.trajectories._export_common import (
    ThoughtBuffer,
    acp_projection_coverage,
    content_blocks_to_text,
    is_attributed_child_event,
)
from benchflow.trajectories.types import redact_trajectory_obj

ATIF_SCHEMA_VERSION = "ATIF-v1.7"

# Canonical artifact location, sibling of ``trainer/verifiers.jsonl``.
ROLLOUT_ATIF_RELPATH = "trainer/atif.json"

# Nesting cap for embedded subagent trajectories. Claude Code subagents cannot
# spawn subagents today, so real rollouts nest one level; the cap bounds the
# document depth that redaction and serialisation recurse through when the
# agent-supplied attribution is malformed.
MAX_SUBAGENT_DEPTH = 8


def acp_events_to_atif_steps(
    events: list[dict[str, Any]],
    prompts: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert BenchFlow ACP trajectory events to the root agent's ATIF steps.

    *prompts* are the user-facing prompts handed to the agent before any
    ACP event is captured; they become leading ``user`` steps. Consecutive
    ``agent_thought`` events are joined and attached as the next agent
    step's ``reasoning_content`` — flushed as a standalone agent step when
    a user step or the end of the trajectory would otherwise drop them.
    Attributed child events are skipped here; :func:`trajectory_to_atif_record`
    exports them as subagent trajectories.
    """
    root = [
        event
        for event in events
        if isinstance(event, dict) and not is_attributed_child_event(event)
    ]
    return _scope_events_to_steps(root, prompts)


def _scope_events_to_steps(
    events: list[dict[str, Any]],
    prompts: list[str] | None = None,
    subagent_refs: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Walk one agent's events into ATIF steps.

    *subagent_refs* maps a captured ``tool_call_id`` to the ``trajectory_id``
    of the subagent that tool call spawned; the call's observation result
    then carries the ``subagent_trajectory_ref``.
    """
    refs = subagent_refs or {}
    steps: list[dict[str, Any]] = []
    thoughts = ThoughtBuffer()

    def append_step(source: str, body: dict[str, Any]) -> None:
        steps.append({"step_id": len(steps) + 1, "source": source, **body})

    def flush_thoughts() -> None:
        reasoning = thoughts.take()
        if reasoning:
            append_step("agent", {"message": "", "reasoning_content": reasoning})

    for prompt in prompts or []:
        if prompt:
            append_step("user", {"message": str(prompt)})

    for event in events:
        etype = event.get("type")
        if etype == "user_message":
            text = str(event.get("text") or "")
            if text:
                flush_thoughts()
                append_step("user", {"message": text})
        elif etype == "agent_thought":
            text = str(event.get("text") or "")
            if text:
                thoughts.push(text)
        elif etype == "agent_message":
            text = str(event.get("text") or "")
            if text:
                body: dict[str, Any] = {"message": text}
                reasoning = thoughts.take()
                if reasoning:
                    body["reasoning_content"] = reasoning
                append_step("agent", body)
        elif etype == "tool_call":
            # The fallback id needs no trajectory-wide dedupe: ATIF resolves
            # source_call_id within the same step's tool_calls only (unlike
            # ADP, whose claim_call_id must keep ids unique per trajectory).
            captured_id = str(event.get("tool_call_id") or "")
            call_id = captured_id or f"call_{len(steps) + 1}"
            tool_call: dict[str, Any] = {
                "tool_call_id": call_id,
                "function_name": str(event.get("kind") or "tool"),
                "arguments": {},
            }
            extra = {
                key: str(event[key]) for key in ("title", "status") if event.get(key)
            }
            if extra:
                tool_call["extra"] = extra
            body = cast(dict[str, Any], {"message": "", "tool_calls": [tool_call]})
            reasoning = thoughts.take()
            if reasoning:
                body["reasoning_content"] = reasoning
            result: dict[str, Any] = {"source_call_id": call_id}
            result_text = content_blocks_to_text(event.get("content"))
            if result_text:
                result["content"] = result_text
            if captured_id in refs:
                result["subagent_trajectory_ref"] = [
                    {"trajectory_id": refs[captured_id]}
                ]
            if len(result) > 1:
                body["observation"] = {"results": [result]}
            append_step("agent", body)
        elif etype == "oracle":
            cmd = str(event.get("command") or "oracle")
            append_step("agent", {"message": f"[oracle: {cmd}]"})

    flush_thoughts()
    return steps


def _subagent_trajectory_id(parent_tool_call_id: str) -> str:
    return f"subagent:{parent_tool_call_id}"


def _spawn_input(event: dict[str, Any] | None, key: str) -> str | None:
    """Read a non-empty string field from a spawning tool call's raw input."""
    raw = event.get("raw_input") if event is not None else None
    value = raw.get(key) if isinstance(raw, dict) else None
    return value if isinstance(value, str) and value else None


class _SubagentPlan:
    """Split a rollout's events into the root agent and its embedded subagents.

    Each attributed child event belongs to the scope named by its
    ``parent_tool_call_id``; unattributed events belong to the root (scope
    ``None``). A child scope hangs under the scope holding the first captured
    tool call with that id, found breadth-first from the root, so cyclic or
    self-referencing attribution can never recurse. Child scopes whose
    spawning call was never captured hang, unreferenced, under the root.
    Trajectories are built deepest-first, so a parent only references
    children that produced at least one step.
    """

    def __init__(
        self, events: list[dict[str, Any]], *, agent_name: str, agent_version: str
    ) -> None:
        self.scopes: dict[str | None, list[dict[str, Any]]] = {None: []}
        self._spawns: dict[str, dict[str, Any]] = {}
        for event in events:
            if not isinstance(event, dict):
                continue
            scope = (
                str(event["parent_tool_call_id"])
                if is_attributed_child_event(event)
                else None
            )
            self.scopes.setdefault(scope, []).append(event)
            if event.get("type") == "tool_call":
                call_id = str(event.get("tool_call_id") or "")
                if call_id:
                    self._spawns.setdefault(call_id, event)

        self._children: dict[str | None, list[str]] = {}
        depth: dict[str | None, int] = {None: 0}
        order: list[str] = []
        queue: deque[str | None] = deque([None])
        while queue:
            scope = queue.popleft()
            found = [
                call_id
                for event in self.scopes[scope]
                if event.get("type") == "tool_call"
                and (call_id := str(event.get("tool_call_id") or "")) in self.scopes
                and self._spawns[call_id] is event
            ]
            if scope is None:
                found += [
                    child
                    for child in self.scopes
                    if child is not None and child not in self._spawns
                ]
            for child in found:
                if child in depth or depth[scope] >= MAX_SUBAGENT_DEPTH:
                    continue
                depth[child] = depth[scope] + 1
                self._children.setdefault(scope, []).append(child)
                order.append(child)
                queue.append(child)

        self.built: dict[str, dict[str, Any]] = {}
        for scope in reversed(order):
            self._build(scope, agent_name=agent_name, agent_version=agent_version)

    def _build(self, scope: str, *, agent_name: str, agent_version: str) -> None:
        spawn = self._spawns.get(scope)
        # Claude's Task tool input carries the prompt the subagent starts from;
        # no ACP event replays it inside the child's own stream.
        prompt = _spawn_input(spawn, "prompt")
        steps = self.steps(scope, [prompt] if prompt else None)
        if not steps:
            return
        agent: dict[str, Any] = {
            "name": agent_name or "unknown",
            "version": agent_version,
        }
        subagent_type = _spawn_input(spawn, "subagent_type")
        if subagent_type:
            agent["extra"] = {"subagent_type": subagent_type}
        extra: dict[str, Any] = {"parent_tool_call_id": scope}
        if spawn is None:
            extra["parent_tool_call_captured"] = False
        elif prompt:
            extra["prompt_source"] = "parent_tool_call.raw_input.prompt"
        trajectory: dict[str, Any] = {
            "schema_version": ATIF_SCHEMA_VERSION,
            "trajectory_id": _subagent_trajectory_id(scope),
            "agent": agent,
            "steps": steps,
            "final_metrics": {"total_steps": len(steps)},
            "extra": extra,
        }
        embedded = self.embedded(scope)
        if embedded:
            trajectory["subagent_trajectories"] = embedded
        self.built[scope] = trajectory

    def _built_children(self, scope: str | None) -> list[str]:
        return [c for c in self._children.get(scope, []) if c in self.built]

    def embedded(self, scope: str | None) -> list[dict[str, Any]]:
        """Built subagent trajectories hanging directly under *scope*."""
        return [self.built[child] for child in self._built_children(scope)]

    def steps(
        self, scope: str | None, prompts: list[str] | None
    ) -> list[dict[str, Any]]:
        """*scope*'s ATIF steps, referencing its built children by spawn call."""
        refs = {
            child: _subagent_trajectory_id(child)
            for child in self._built_children(scope)
            if child in self._spawns
        }
        return _scope_events_to_steps(self.scopes[scope], prompts, refs)

    def coverage(self, events: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Account for every attributed child event against the embedded tree."""
        root_only = acp_projection_coverage(events)
        if root_only is None:
            return None
        attributed = root_only.pop("excluded_attributed_child_events")
        del root_only["scope"]
        embedded_events = sum(len(self.scopes[scope]) for scope in self.built)
        return {
            "scope": "root_with_embedded_subagents",
            "attributed_child_events": attributed,
            "embedded_subagent_trajectories": len(self.built),
            "unlinked_subagent_trajectories": sum(
                scope not in self._spawns for scope in self.built
            ),
            "excluded_attributed_child_events": attributed - embedded_events,
            "max_subagent_depth": MAX_SUBAGENT_DEPTH,
            **root_only,
        }


def trajectory_to_atif_record(
    *,
    session_id: str,
    agent_name: str,
    events: list[dict[str, Any]],
    prompts: list[str] | None = None,
    agent_version: str = "unknown",
    model: str | None = None,
    total_prompt_tokens: int | None = None,
    total_completion_tokens: int | None = None,
    total_cached_tokens: int | None = None,
    total_cost_usd: float | None = None,
) -> dict[str, Any]:
    """Build one ATIF trajectory document from a captured rollout.

    ``agent`` requires both ``name`` and ``version`` in ATIF; BenchFlow
    does not track agent binary versions, so *agent_version* defaults to
    ``"unknown"`` rather than fabricating one. Token totals come from the
    raw LLM-traffic capture (``Trajectory.total_*`` in
    :mod:`benchflow.trajectories.types`) when the caller has it; ATIF's
    per-step ``metrics`` are omitted because ACP events carry no usage.
    Those totals cover every LLM call of the rollout, subagents included;
    embedded subagent trajectories carry only their own ``total_steps``.

    Raises ``ValueError`` for an empty trajectory — ATIF requires at least
    one step, so there is no valid empty document to emit.
    """
    plan = _SubagentPlan(events, agent_name=agent_name, agent_version=agent_version)
    steps = plan.steps(None, prompts)
    if not steps:
        raise ValueError("ATIF requires at least one step; trajectory is empty")
    agent: dict[str, Any] = {
        "name": agent_name or "unknown",
        "version": agent_version,
    }
    if model:
        agent["model_name"] = model
    final_metrics: dict[str, Any] = {"total_steps": len(steps)}
    for key, value in (
        ("total_prompt_tokens", total_prompt_tokens),
        ("total_completion_tokens", total_completion_tokens),
        ("total_cached_tokens", total_cached_tokens),
        ("total_cost_usd", total_cost_usd),
    ):
        if value is not None:
            final_metrics[key] = value
    record: dict[str, Any] = {"schema_version": ATIF_SCHEMA_VERSION}
    if session_id:
        record["session_id"] = session_id
    record["agent"] = agent
    record["steps"] = steps
    record["final_metrics"] = final_metrics
    coverage = plan.coverage(events)
    if coverage is not None:
        record["extra"] = {"acp_projection": coverage}
    subagents = plan.embedded(None)
    if subagents:
        record["subagent_trajectories"] = subagents
    return record


def _record_to_redacted_json(record: dict[str, Any]) -> str:
    return dumps_finite(redact_trajectory_obj(record), default=str, indent=2)


def write_rollout_atif_json(
    rollout_dir: str | Path,
    *,
    session_id: str,
    agent_name: str,
    prompts: list[str] | None,
    trajectory: list[dict[str, Any]],
    agent_version: str = "unknown",
    model: str | None = None,
    total_prompt_tokens: int | None = None,
    total_completion_tokens: int | None = None,
    total_cached_tokens: int | None = None,
    total_cost_usd: float | None = None,
) -> dict[str, Any] | None:
    """Write one rollout's ATIF document to ``rollout_dir/trainer/atif.json``.

    ATIF is a single JSON document per trajectory (not JSONL). Returns the
    redacted record as written, or ``None`` when the trajectory is empty —
    an empty ATIF document would be schema-invalid, so no artifact is
    produced in that case.
    """
    out = Path(rollout_dir) / ROLLOUT_ATIF_RELPATH
    coverage_path = out.with_name("atif.coverage.json")
    coverage = acp_projection_coverage(trajectory)
    if coverage is None:
        coverage_path.unlink(missing_ok=True)
    try:
        record = trajectory_to_atif_record(
            session_id=session_id,
            agent_name=agent_name,
            events=trajectory,
            prompts=prompts,
            agent_version=agent_version,
            model=model,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_cached_tokens=total_cached_tokens,
            total_cost_usd=total_cost_usd,
        )
    except ValueError:
        if coverage is not None:
            # ATIF requires a real step. Keep an explicit coverage receipt when
            # the root projection is empty instead of inventing a system step.
            out.parent.mkdir(parents=True, exist_ok=True)
            coverage_path.write_text(
                _record_to_redacted_json(
                    {
                        "artifact_status": "omitted_empty_root_projection",
                        "acp_projection": coverage,
                    }
                )
                + "\n"
            )
        out.unlink(missing_ok=True)
        return None
    coverage_path.unlink(missing_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    redacted = _record_to_redacted_json(record)
    out.write_text(redacted + "\n")
    return cast(dict[str, Any], json.loads(redacted))
