"""Adapter to expose BenchFlow rewards in ORS (OpenReward) format.

This module provides thin format converters — no ORS SDK dependency is
required.  ``VerifyResult`` and ``RewardEvent`` instances are mapped to
plain dicts matching the ORS reward-response schema so consumers can
forward them to an ORS-compatible endpoint or file.

Extend by subclassing ``ORSAdapter`` and overriding the static methods for
custom field mappings.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchflow.rewards.events import RewardEvent
    from benchflow.rewards.protocol import VerifyResult


def _json_safe_float(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _event_to_dict(event: RewardEvent) -> dict[str, Any]:
    # space/granularity are the architecture's mandatory ``(space, granularity,
    # value)`` reward-record tag (docs/architecture.md, "Every reward record is
    # tagged"). Preserve them on the outbound ORS seam so memory/action/
    # reasoning process events stay distinguishable downstream.
    return {
        "type": event.type,
        "reward": _json_safe_float(event.reward),
        "source": event.source,
        "step": event.step,
        "space": event.space,
        "granularity": event.granularity,
        "timestamp": event.ts,
    }


class ORSAdapter:
    """Converts BenchFlow reward types to ORS-compatible format."""

    @staticmethod
    def verify_result_to_ors(result: VerifyResult) -> dict[str, Any]:
        """Convert ``VerifyResult`` to ORS reward response format.

        Returns::

            {
                "reward": float,
                "is_valid": bool,
                "metadata": {
                    "items": dict[str, float],
                    "events": list[dict],
                    "error": str | None,
                },
            }
        """
        reward_is_valid = math.isfinite(result.reward) and 0.0 <= result.reward <= 1.0
        metadata_error = result.error
        items = {name: _json_safe_float(score) for name, score in result.items.items()}
        metadata: dict[str, Any] = {
            "items": items,
            "events": [_event_to_dict(e) for e in result.events],
            "error": metadata_error,
            # Headline ``(space, granularity)`` tag of the aggregate reward —
            # mirrors the per-event tags above so downstream trainers know
            # which evaluation space the top-level ``reward`` belongs to.
            "space": result.space,
            "granularity": result.granularity,
        }
        if not reward_is_valid:
            metadata["raw_reward"] = repr(result.reward)
            metadata["error"] = metadata_error or f"invalid reward: {result.reward!r}"
        raw_invalid_items = {
            name: repr(score)
            for name, score in result.items.items()
            if not math.isfinite(score)
        }
        if raw_invalid_items:
            metadata["raw_items"] = raw_invalid_items

        return {
            "reward": result.reward if reward_is_valid else 0.0,
            "is_valid": result.error is None and reward_is_valid,
            "metadata": metadata,
        }

    @staticmethod
    def reward_event_to_ors(event: RewardEvent) -> dict[str, Any]:
        """Convert a single ``RewardEvent`` to ORS event format."""
        return _event_to_dict(event)

    @staticmethod
    def tool_outputs_to_reward_events(
        outputs: list[dict[str, Any]],
        *,
        source: str = "ors-tool-output",
    ) -> list[dict[str, Any]]:
        """Normalize ORS-style tool outputs into verifier evidence records.

        OpenReward/ORS environments can emit a reward on each tool/action and
        a terminal ``finished`` marker. BenchFlow's verifier-side
        ``ors-episode`` strategy already consumes JSONL reward-event records
        from ``trajectory/ors-rewards.jsonl``; this method is the runtime-side
        contract for producing that artifact from local or hosted ORS session
        output.
        """

        records: list[dict[str, Any]] = []
        for index, output in enumerate(outputs, start=1):
            if not isinstance(output, dict):
                raise ValueError("ORS tool output records must be JSON objects")
            reward = _bounded_reward(_reward_value(output), path=f"outputs[{index}]")
            finished = bool(
                output.get("finished") is True or output.get("done") is True
            )
            record_type = str(
                output.get("type") or ("terminal" if finished else "dense")
            )
            record: dict[str, Any] = {
                "type": record_type,
                "reward": reward,
                "source": str(
                    output.get("source")
                    or output.get("tool_name")
                    or output.get("tool")
                    or source
                ),
                "step": _step_value(output, fallback=index),
                "space": str(
                    output.get("space") or ("output" if finished else "action")
                ),
                "granularity": str(
                    output.get("granularity") or ("terminal" if finished else "step")
                ),
            }
            timestamp = output.get("timestamp") or output.get("ts")
            if timestamp is not None:
                record["timestamp"] = str(timestamp)
            tool_call_id = output.get("tool_call_id") or output.get("toolCallId")
            if tool_call_id is not None:
                record["tool_call_id"] = str(tool_call_id)
            if finished:
                record["finished"] = True
            records.append(record)
        return records

    @staticmethod
    def write_tool_outputs_jsonl(
        outputs: list[dict[str, Any]],
        path: str | Path,
        *,
        source: str = "ors-tool-output",
    ) -> list[dict[str, Any]]:
        """Write ORS tool-output rewards to ``trajectory/ors-rewards.jsonl``."""

        records = ORSAdapter.tool_outputs_to_reward_events(outputs, source=source)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for record in records:
                f.write(json.dumps(record, allow_nan=False) + "\n")
        return records


def to_ors_reward(result: VerifyResult) -> dict[str, Any]:
    """Convenience function to convert ``VerifyResult`` to ORS format."""
    return ORSAdapter.verify_result_to_ors(result)


def ors_tool_outputs_to_reward_events(
    outputs: list[dict[str, Any]],
    *,
    source: str = "ors-tool-output",
) -> list[dict[str, Any]]:
    """Convenience wrapper for ORS tool-output reward event normalization."""

    return ORSAdapter.tool_outputs_to_reward_events(outputs, source=source)


def write_ors_tool_outputs_jsonl(
    outputs: list[dict[str, Any]],
    path: str | Path,
    *,
    source: str = "ors-tool-output",
) -> list[dict[str, Any]]:
    """Convenience wrapper for writing ORS reward JSONL evidence."""

    return ORSAdapter.write_tool_outputs_jsonl(outputs, path, source=source)


def _reward_value(output: dict[str, Any]) -> Any:
    value = output.get("reward")
    if isinstance(value, dict):
        return value.get("reward")
    if value is not None:
        return value
    for key in ("tool_output", "output", "result"):
        nested = output.get(key)
        if isinstance(nested, dict) and "reward" in nested:
            nested_reward = nested["reward"]
            if isinstance(nested_reward, dict):
                return nested_reward.get("reward")
            return nested_reward
    return None


def _bounded_reward(value: Any, *, path: str) -> float:
    try:
        reward = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}.reward must be numeric") from exc
    if not math.isfinite(reward) or reward < 0.0 or reward > 1.0:
        raise ValueError(f"{path}.reward must be finite and between 0.0 and 1.0")
    return reward


def _step_value(output: dict[str, Any], *, fallback: int) -> int:
    value = output.get("step")
    if value is None:
        return fallback
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("ORS tool output step must be an integer")
    return value
