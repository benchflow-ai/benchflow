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
from pathlib import Path
from typing import Any

from benchflow.adapters.ors import ORSAdapter
from benchflow.rewards.protocol import VerifyResult


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
    """Write Verifiers records to a JSONL file — one JSON object per line."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")
