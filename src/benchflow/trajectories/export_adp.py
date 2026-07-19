"""Export captured trajectories in ADP (Agent Data Protocol) format.

ADP is the standardized agent-trajectory schema used by the OpenHands
training pipeline lineage. A BenchFlow rollout's ACP-style trajectory
events become one ADP ``Trajectory`` record — JSONL datasets of these
records feed ADP's SFT converters directly.

The record shape is pinned against the ADP pydantic schemas
(``neulab/agent-data-protocol``, ``schema/trajectory.py``,
``schema/action/*.py``, ``schema/observation/*.py``) and
``schema/SCHEMA.md``, schema version ``1.3.1``. Spec constraints honoured
here:

- every tool action (``api_action``) carries a ``tool_call_id`` and is
  followed by exactly one matching ``text_observation`` with the same id —
  ADP validation rejects unmatched or duplicate ids;
- ``tool_call_id``s are unique within a trajectory; missing or colliding
  ACP ids are replaced with synthesized ``call_NNNNNN`` ids;
- tool results use ``source: "environment"`` (``source: "user"`` is
  reserved for actual user input and is invalid on tool results);
- ``message_action`` never carries a ``tool_call_id``.

Mapping from BenchFlow ACP trajectory events (see
:mod:`benchflow.trajectories._capture`):

- *prompts* and ``user_message`` events → ``text_observation`` with
  ``source: "user"``;
- ``agent_message`` → ``message_action``;
- ``agent_thought`` → ``reasoning_content`` on the next action, or a
  standalone empty ``message_action`` when no action follows;
- ``tool_call`` → an ``api_action``/``text_observation`` pair. ACP updates
  carry no structured arguments, so ``kwargs`` is ``{}`` and the ACP
  ``title`` becomes the action's ``description``; the ACP ``status`` has no
  ADP slot (``extra`` fields are forbidden) and is dropped — failures
  remain visible in the observation content;
- ``oracle`` → a ``message_action`` rendering the command, mirroring
  :func:`benchflow.trajectories.export.acp_events_to_messages`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

from benchflow._utils.json_safe import dumps_finite
from benchflow.trajectories._export_common import (
    ThoughtBuffer,
    aggregate_rollout_jsonl,
    content_blocks_to_text,
)
from benchflow.trajectories.types import redact_trajectory_obj

logger = logging.getLogger(__name__)

ADP_SCHEMA_VERSION = "1.3.1"

# Canonical artifact locations, siblings of the verifiers.jsonl seam.
ROLLOUT_ADP_RELPATH = "trainer/adp.jsonl"
JOB_ADP_FILENAME = "adp.jsonl"

_ACTION_CLASSES = ("api_action", "code_action", "message_action")


def acp_events_to_adp_content(
    events: list[dict[str, Any]],
    prompts: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert BenchFlow ACP trajectory events to ADP content items.

    *prompts* are the user-facing prompts handed to the agent before any
    ACP event is captured; they become leading user ``text_observation``
    items. Consecutive ``agent_thought`` events are joined and attached as
    the next action's ``reasoning_content`` — flushed as a standalone
    ``message_action`` when a user observation or the end of the
    trajectory would otherwise drop them.
    """
    content: list[dict[str, Any]] = []
    thoughts = ThoughtBuffer()
    used_ids: set[str] = set()
    synth_count = 0

    def claim_call_id(raw: Any) -> str:
        nonlocal synth_count
        candidate = str(raw or "")
        while not candidate or candidate in used_ids:
            synth_count += 1
            candidate = f"call_{synth_count:06d}"
        used_ids.add(candidate)
        return candidate

    def flush_thoughts() -> None:
        reasoning = thoughts.take()
        if reasoning:
            content.append(
                {
                    "class_": "message_action",
                    "content": "",
                    "reasoning_content": reasoning,
                }
            )

    for prompt in prompts or []:
        if prompt:
            content.append(
                {
                    "class_": "text_observation",
                    "content": str(prompt),
                    "source": "user",
                }
            )

    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "user_message":
            text = str(event.get("text") or "")
            if text:
                flush_thoughts()
                content.append(
                    {"class_": "text_observation", "content": text, "source": "user"}
                )
        elif etype == "agent_thought":
            text = str(event.get("text") or "")
            if text:
                thoughts.push(text)
        elif etype == "agent_message":
            text = str(event.get("text") or "")
            if text:
                action: dict[str, Any] = {"class_": "message_action", "content": text}
                reasoning = thoughts.take()
                if reasoning:
                    action["reasoning_content"] = reasoning
                content.append(action)
        elif etype == "tool_call":
            call_id = claim_call_id(event.get("tool_call_id"))
            action = {
                "class_": "api_action",
                "tool_call_id": call_id,
                "function": str(event.get("kind") or "tool"),
                "kwargs": {},
            }
            title = event.get("title")
            if title:
                action["description"] = str(title)
            reasoning = thoughts.take()
            if reasoning:
                action["reasoning_content"] = reasoning
            content.append(action)
            content.append(
                {
                    "class_": "text_observation",
                    "tool_call_id": call_id,
                    "content": content_blocks_to_text(event.get("content")),
                    "source": "environment",
                }
            )
        elif etype == "oracle":
            cmd = str(event.get("command") or "oracle")
            content.append({"class_": "message_action", "content": f"[oracle: {cmd}]"})

    flush_thoughts()
    return content


def trajectory_to_adp_record(
    *,
    trajectory_id: str,
    events: list[dict[str, Any]],
    prompts: list[str] | None = None,
    reward: float | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one ADP ``Trajectory`` record from a captured rollout.

    *reward*, when given, is attached to the trajectory's final action as
    ADP's per-step ``reward`` field — the terminal-reward convention for
    RL training data. When the trajectory has no action to carry it (e.g. an
    agent that crashed after consuming the prompt but before acting), the
    reward is surfaced as ``details['terminal_reward']`` and a warning is
    logged rather than silently dropped. *details* passes through as the
    dataset-specific metadata dict. ``available_apis`` is omitted: BenchFlow
    does not know the agent's full tool universe, and the field is optional.
    """
    content = acp_events_to_adp_content(events, prompts)
    out_details = dict(details or {})
    if reward is not None:
        placed = False
        for item in reversed(content):
            if item["class_"] in _ACTION_CLASSES:
                item["reward"] = reward
                placed = True
                break
        if not placed:
            out_details["terminal_reward"] = reward
            logger.warning(
                "ADP trajectory %s has terminal reward %r but no action to "
                "carry it; recorded under details.terminal_reward",
                trajectory_id,
                reward,
            )
    return {
        "schema_version": ADP_SCHEMA_VERSION,
        "id": trajectory_id,
        "content": content,
        "details": out_details,
    }


def _record_to_redacted_json_line(record: dict[str, Any]) -> str:
    return dumps_finite(redact_trajectory_obj(record), default=str)


def write_rollout_adp_jsonl(
    rollout_dir: str | Path,
    *,
    trajectory_id: str,
    task_id: str,
    prompts: list[str] | None,
    trajectory: list[dict[str, Any]],
    model: str | None,
    environment: str,
    reward: float | None = None,
) -> dict[str, Any] | None:
    """Write one rollout's ADP record to ``rollout_dir/trainer/adp.jsonl``.

    One record per line, matching the verifiers.jsonl seam. Returns the
    redacted record so callers can aggregate across a job, or ``None`` when an
    empty-content trajectory carries no score — mirroring ATIF's empty-record
    contract so a meaningless ``content == []`` line never reaches the job
    aggregate. An empty-content trajectory that still has a terminal reward is
    written (the score lands in ``details.terminal_reward``) so a scored crash
    rollout is not lost.
    """
    record = trajectory_to_adp_record(
        trajectory_id=trajectory_id,
        events=trajectory,
        prompts=prompts,
        reward=reward,
        details={"task_id": task_id, "environment": environment, "model": model or ""},
    )
    if not record["content"] and "terminal_reward" not in record["details"]:
        return None
    out = Path(rollout_dir) / ROLLOUT_ADP_RELPATH
    out.parent.mkdir(parents=True, exist_ok=True)
    redacted = _record_to_redacted_json_line(record)
    out.write_text(redacted + "\n")
    return cast(dict[str, Any], json.loads(redacted))


def write_job_adp_jsonl(job_dir: str | Path) -> Path | None:
    """Aggregate per-rollout ADP JSONLs into ``job_dir/adp.jsonl``.

    Scans ``job_dir/*/trainer/adp.jsonl`` and concatenates their lines
    into one job-level dataset. Returns the artifact path, or ``None``
    when no rollouts have emitted records yet.
    """
    return aggregate_rollout_jsonl(
        job_dir,
        rollout_relpath=ROLLOUT_ADP_RELPATH,
        out_filename=JOB_ADP_FILENAME,
    )
