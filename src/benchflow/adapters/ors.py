"""Adapter to expose BenchFlow rewards in ORS (OpenReward) format.

This module provides thin format converters — no ORS SDK dependency is
required.  ``VerifyResult`` and ``RewardEvent`` instances are mapped to
plain dicts matching the ORS reward-response schema so consumers can
forward them to an ORS-compatible endpoint or file.

Extend by subclassing ``ORSAdapter`` and overriding the static methods for
custom field mappings.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchflow.rewards.events import RewardEvent
    from benchflow.rewards.protocol import VerifyResult


def _json_safe_float(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _event_to_dict(event: RewardEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "reward": _json_safe_float(event.reward),
        "source": event.source,
        "step": event.step,
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


def to_ors_reward(result: VerifyResult) -> dict[str, Any]:
    """Convenience function to convert ``VerifyResult`` to ORS format."""
    return ORSAdapter.verify_result_to_ors(result)
