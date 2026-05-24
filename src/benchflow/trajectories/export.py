"""Export scored trajectories in the Verifiers / ORS dataset format.

The trajectory is the seam to the trainer (architecture.md, "The edges").
A BenchFlow scored rollout becomes a JSONL dataset that prime-rl /
Verifiers ingests directly — one record per scored rollout.

The record shape is pinned against the Verifiers ``RolloutOutput`` type
(``willccbb/verifiers``, ``verifiers/types.py``): ``prompt``, ``completion``,
``reward``, ``metrics``, ``is_completed``, ``is_truncated``, ``example_id``,
``info``. The reward conversion reuses the existing ``ORSAdapter`` so the
reward is validated and JSON-safe.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from benchflow.adapters.ors import ORSAdapter
from benchflow.rewards.protocol import VerifyResult


def _scrub_non_finite(value: Any) -> Any:
    """Replace ``NaN`` / ``Infinity`` floats with ``None`` recursively.

    Plain ``json.dumps`` emits non-finite floats as the bare tokens ``NaN``,
    ``Infinity``, ``-Infinity`` — valid Python but rejected by strict JSON
    parsers (jq, serde, Node ``JSON.parse``). We normalize to ``null`` so
    downstream trainer ingestion never sees an invalid JSONL line.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _scrub_non_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub_non_finite(v) for v in value]
    return value


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
            clean = _scrub_non_finite(rec)
            f.write(json.dumps(clean, default=str, allow_nan=False) + "\n")
