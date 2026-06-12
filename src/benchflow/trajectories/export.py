"""Export scored trajectories in the Verifiers / ORS dataset format.

The trajectory is the seam to the trainer (architecture.md, "The edges").
A BenchFlow scored rollout becomes a JSONL dataset that prime-rl /
Verifiers ingests directly — one record per scored rollout.

The record shape is pinned against the Verifiers ``RolloutOutput`` type
(``willccbb/verifiers``, ``verifiers/types.py``): ``prompt``, ``completion``,
``reward``, ``metrics``, ``is_completed``, ``is_truncated``, ``example_id``,
``info``. The reward conversion reuses the existing ``ORSAdapter`` so the
reward is validated and JSON-safe.

This module also provides the live wiring used by ``Rollout`` and
``Evaluation`` to emit trainer-ready ``verifiers.jsonl`` artifacts:

- :func:`acp_events_to_messages` — convert BenchFlow's ACP-style trajectory
  events to the ``[{"role", "content"}, ...]`` shape the record expects.
- :func:`reward_map_to_verify_result` — adapt the canonical verifier reward
  ``dict`` produced by ``Rollout.verify()`` back to a :class:`VerifyResult`.
- :func:`write_rollout_verifiers_jsonl` — write one rollout's record to
  ``rollout_dir/trainer/verifiers.jsonl``.
- :func:`write_job_verifiers_jsonl` — concatenate per-rollout records into a
  single ``job_dir/verifiers.jsonl`` dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from benchflow._utils.json_safe import scrub_non_finite
from benchflow.adapters.ors import ORSAdapter
from benchflow.rewards.protocol import VerifyResult
from benchflow.trajectories._export_common import aggregate_rollout_jsonl
from benchflow.trajectories.types import redact_trajectory_text

# Canonical artifact locations (see issue #385).
ROLLOUT_ARTIFACT_RELPATH = "trainer/verifiers.jsonl"
JOB_ARTIFACT_FILENAME = "verifiers.jsonl"


def _split_prompt_completion(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a message list into (prompt, completion).

    Prompt = the leading user/system messages; completion = everything from
    the first assistant message onward — the Verifiers convention.
    """
    first_assistant = next(
        (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
        None,
    )
    if first_assistant is None:
        return list(messages), []
    return messages[:first_assistant], messages[first_assistant:]


def trajectory_to_verifiers_record(
    *,
    task_id: str,
    messages: list[dict[str, Any]],
    verify_result: VerifyResult,
    model: str,
    environment: str,
    example_id: int = 0,
    is_completed: bool = True,
    is_truncated: bool = False,
) -> dict[str, Any]:
    """Build one Verifiers / ORS dataset record from a scored trajectory."""
    prompt, completion = _split_prompt_completion(messages)
    ors = ORSAdapter.verify_result_to_ors(verify_result)
    return {
        "example_id": example_id,
        "prompt": prompt,
        "completion": completion,
        # ors["reward"] is validated + clamped (NaN/out-of-range -> 0.0).
        "reward": ors["reward"],
        # ors metadata items are already JSON-safe floats.
        "metrics": ors["metadata"]["items"],
        "is_completed": is_completed,
        "is_truncated": is_truncated,
        "info": {
            "task_id": task_id,
            "environment": environment,
            "model": model,
            "reward_valid": ors["is_valid"],
            "reward_metadata": ors["metadata"],
        },
    }


def _record_to_redacted_json_line(record: dict[str, Any]) -> str:
    clean = scrub_non_finite(record)
    raw = json.dumps(clean, default=str, allow_nan=False)
    return redact_trajectory_text(raw)


def _redact_verifiers_record(record: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(_record_to_redacted_json_line(record)))


def export_trajectories_to_jsonl(
    records: list[dict[str, Any]], path: str | Path
) -> None:
    """Write Verifiers records to a JSONL file — one JSON object per line.

    Non-finite floats (``NaN``, ``±Infinity``) anywhere in a record are
    normalized to ``null`` before serialization, and ``allow_nan=False`` is
    set as defense-in-depth so any future regression that lets a non-finite
    slip through raises ``ValueError`` instead of producing JSONL that
    strict parsers reject.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in records:
            f.write(_record_to_redacted_json_line(rec) + "\n")


def _tool_call_to_content(event: dict[str, Any]) -> str:
    """Render an ACP ``tool_call`` event as a single text block.

    Verifiers' record uses message-format strings, so the tool call's title
    and any text-bearing content are summarised into one assistant string.
    """
    title = str(event.get("title") or event.get("kind") or "tool_call")
    parts: list[str] = [f"[tool_call: {title}]"]
    content = event.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
    elif isinstance(content, str) and content:
        parts.append(content)
    return "\n".join(parts)


def acp_events_to_messages(
    events: list[dict[str, Any]],
    prompts: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert BenchFlow ACP trajectory events to message format.

    The live rollout trajectory is a list of typed events
    (``user_message``, ``agent_message``, ``agent_thought``, ``tool_call``)
    captured in :mod:`benchflow.trajectories._capture`. The Verifiers /
    ORS record expects ``[{"role", "content"}, ...]`` — typical OpenAI
    chat message shape.

    *prompts* are the user-facing prompts handed to the agent before any
    ACP event is captured (instruction.md, scene turns, etc.). They are
    materialised as leading ``user`` messages so the record's ``prompt``
    half is non-empty when ``trajectory_to_verifiers_record`` splits the
    sequence at the first ``assistant`` message.

    Tool calls are rendered into a single assistant text block via
    :func:`_tool_call_to_content` — this keeps the record's shape stable
    without coupling to a specific tool-call schema. ``agent_thought``
    events fold into the prior or following assistant message under the
    same role to avoid breaking the user→assistant→... ordering.
    """
    messages: list[dict[str, Any]] = []
    for prompt in prompts or []:
        if prompt:
            messages.append({"role": "user", "content": str(prompt)})

    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "user_message":
            text = str(event.get("text") or "")
            if text:
                messages.append({"role": "user", "content": text})
        elif etype in ("agent_message", "agent_thought"):
            text = str(event.get("text") or "")
            if text:
                messages.append({"role": "assistant", "content": text})
        elif etype == "tool_call":
            messages.append(
                {"role": "assistant", "content": _tool_call_to_content(event)}
            )
        elif etype == "oracle":
            # Oracle rollouts have no agent dialogue; surface the command so
            # the record still carries a non-empty completion.
            cmd = str(event.get("command") or "oracle")
            messages.append({"role": "assistant", "content": f"[oracle: {cmd}]"})
    return messages


def reward_map_to_verify_result(
    rewards: dict[str, Any] | None,
    *,
    error: str | None = None,
) -> VerifyResult:
    """Adapt a canonical verifier reward ``dict`` to :class:`VerifyResult`.

    ``Rollout.verify()`` produces a validated reward map with the shape
    ``{"reward": float, "rubric": [...], ...other scalars...}``. The
    Verifiers record helper expects a :class:`VerifyResult`, so we lift the
    headline ``reward`` and any per-item scalars into ``VerifyResult.items``.

    A ``None`` rewards map (verifier crashed/timed out) yields a 0.0 reward
    with ``error`` populated so the exported record's ``reward_valid`` flag
    is ``False``.
    """
    if rewards is None:
        return VerifyResult(reward=0.0, items={}, error=error or "no rewards")

    reward = rewards.get("reward")
    headline = float(reward) if isinstance(reward, (int, float)) else 0.0

    items: dict[str, float] = {}
    for key, value in rewards.items():
        if key in ("reward", "rubric"):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            items[str(key)] = float(value)

    rubric = rewards.get("rubric")
    if isinstance(rubric, list):
        for i, item in enumerate(rubric):
            if not isinstance(item, dict):
                continue
            rubric_item = cast(dict[str, Any], item)
            score = rubric_item.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                continue
            name = str(rubric_item.get("name") or f"rubric_{i}")
            items[name] = float(score)

    return VerifyResult(reward=headline, items=items, error=error)


def write_rollout_verifiers_jsonl(
    rollout_dir: str | Path,
    *,
    task_id: str,
    prompts: list[str] | None,
    trajectory: list[dict[str, Any]],
    rewards: dict[str, Any] | None,
    model: str | None,
    environment: str,
    example_id: int = 0,
    is_completed: bool = True,
    is_truncated: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    """Write one trainer-ready Verifiers/ORS record into the rollout dir.

    Output path is ``rollout_dir/trainer/verifiers.jsonl`` — the
    architecture's trainer seam (issue #385). Returns the written record so
    callers can aggregate across a job.
    """
    messages = acp_events_to_messages(trajectory, prompts)
    verify_result = reward_map_to_verify_result(rewards, error=error)
    record = trajectory_to_verifiers_record(
        task_id=task_id,
        messages=messages,
        verify_result=verify_result,
        model=model or "",
        environment=environment,
        example_id=example_id,
        is_completed=is_completed,
        is_truncated=is_truncated,
    )
    out = Path(rollout_dir) / ROLLOUT_ARTIFACT_RELPATH
    redacted_record = _redact_verifiers_record(record)
    export_trajectories_to_jsonl([redacted_record], out)
    return redacted_record


def write_job_verifiers_jsonl(job_dir: str | Path) -> Path | None:
    """Aggregate per-rollout trainer JSONLs into ``job_dir/verifiers.jsonl``.

    Scans ``job_dir/*/trainer/verifiers.jsonl`` (rollouts produced by the
    train-mode seam) and concatenates their lines into one job-level
    dataset. Returns the artifact path, or ``None`` when no rollouts have
    emitted records yet.
    """
    return aggregate_rollout_jsonl(
        job_dir,
        rollout_relpath=ROLLOUT_ARTIFACT_RELPATH,
        out_filename=JOB_ARTIFACT_FILENAME,
    )
