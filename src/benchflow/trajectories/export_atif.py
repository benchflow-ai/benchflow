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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from benchflow._utils.json_safe import dumps_finite
from benchflow.trajectories.types import redact_trajectory_text

ATIF_SCHEMA_VERSION = "ATIF-v1.7"

# Canonical artifact location, sibling of ``trainer/verifiers.jsonl``.
ROLLOUT_ATIF_RELPATH = "trainer/atif.json"


def content_blocks_to_text(content: Any) -> str:
    """Render ACP tool-call content blocks to plain text.

    Handles both the flat shape (``{"text": ...}`` / ``{"content": "..."}``)
    and the nested ACP shape (``{"type": "content", "content": {"type":
    "text", "text": ...}}``). Non-text blocks are skipped.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        inner = item.get("content")
        if isinstance(inner, dict):
            inner = inner.get("text")
        text = item.get("text") or inner
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def acp_events_to_atif_steps(
    events: list[dict[str, Any]],
    prompts: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert BenchFlow ACP trajectory events to ATIF step objects.

    *prompts* are the user-facing prompts handed to the agent before any
    ACP event is captured; they become leading ``user`` steps. Consecutive
    ``agent_thought`` events are joined and attached as the next agent
    step's ``reasoning_content`` — flushed as a standalone agent step when
    a user step or the end of the trajectory would otherwise drop them.
    """
    steps: list[dict[str, Any]] = []
    pending_thoughts: list[str] = []

    def take_reasoning() -> str | None:
        if not pending_thoughts:
            return None
        joined = "\n\n".join(pending_thoughts)
        pending_thoughts.clear()
        return joined

    def append_step(source: str, body: dict[str, Any]) -> None:
        steps.append({"step_id": len(steps) + 1, "source": source, **body})

    def flush_thoughts() -> None:
        reasoning = take_reasoning()
        if reasoning:
            append_step("agent", {"message": "", "reasoning_content": reasoning})

    for prompt in prompts or []:
        if prompt:
            append_step("user", {"message": str(prompt)})

    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "user_message":
            text = str(event.get("text") or "")
            if text:
                flush_thoughts()
                append_step("user", {"message": text})
        elif etype == "agent_thought":
            text = str(event.get("text") or "")
            if text:
                pending_thoughts.append(text)
        elif etype == "agent_message":
            text = str(event.get("text") or "")
            if text:
                body: dict[str, Any] = {"message": text}
                reasoning = take_reasoning()
                if reasoning:
                    body["reasoning_content"] = reasoning
                append_step("agent", body)
        elif etype == "tool_call":
            call_id = str(event.get("tool_call_id") or "") or f"call_{len(steps) + 1}"
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
            reasoning = take_reasoning()
            if reasoning:
                body["reasoning_content"] = reasoning
            result_text = content_blocks_to_text(event.get("content"))
            if result_text:
                body["observation"] = {
                    "results": [{"source_call_id": call_id, "content": result_text}]
                }
            append_step("agent", body)
        elif etype == "oracle":
            cmd = str(event.get("command") or "oracle")
            append_step("agent", {"message": f"[oracle: {cmd}]"})

    flush_thoughts()
    return steps


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

    Raises ``ValueError`` for an empty trajectory — ATIF requires at least
    one step, so there is no valid empty document to emit.
    """
    steps = acp_events_to_atif_steps(events, prompts)
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
    return record


def _record_to_redacted_json(record: dict[str, Any]) -> str:
    raw = dumps_finite(record, default=str, indent=2)
    return redact_trajectory_text(raw)


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
    if not acp_events_to_atif_steps(trajectory, prompts):
        return None
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
    out = Path(rollout_dir) / ROLLOUT_ATIF_RELPATH
    out.parent.mkdir(parents=True, exist_ok=True)
    redacted = _record_to_redacted_json(record)
    out.write_text(redacted + "\n")
    return cast(dict[str, Any], json.loads(redacted))
