"""Adapter to expose BenchFlow rewards in ORS (OpenReward) format.

This module provides thin format converters — no ORS SDK dependency is
required.  ``VerifyResult`` and ``RewardEvent`` instances are mapped to
plain dicts matching the ORS reward-response schema so consumers can
forward them to an ORS-compatible endpoint or file.

Extend by subclassing ``ORSAdapter`` and overriding the static methods for
custom field mappings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchflow.rewards.events import RewardEvent
    from benchflow.rewards.protocol import VerifyResult


def _event_to_dict(event: RewardEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "reward": event.reward,
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
        return {
            "reward": result.reward,
            "is_valid": result.error is None,
            "metadata": {
                "items": result.items,
                "events": [_event_to_dict(e) for e in result.events],
                "error": result.error,
            },
        }

    @staticmethod
    def reward_event_to_ors(event: RewardEvent) -> dict[str, Any]:
        """Convert a single ``RewardEvent`` to ORS event format."""
        return _event_to_dict(event)


def to_ors_reward(result: VerifyResult) -> dict[str, Any]:
    """Convenience function to convert ``VerifyResult`` to ORS format."""
    return ORSAdapter.verify_result_to_ors(result)
