"""Rollout lifecycle entry point.

This module is the v0.4 home for the execution lifecycle. During the migration
it delegates to the existing decomposed lifecycle implementation; follow-up
commits will remove the legacy module name once all callsites move here.
"""

from __future__ import annotations

from benchflow.rollouts.config import RolloutConfig
from benchflow.trial import Trial


class Rollout(Trial):
    """One attempt on one task in one sandbox."""

    def __init__(self, config: RolloutConfig) -> None:
        super().__init__(config)

    @classmethod
    async def create(cls, config: RolloutConfig) -> Rollout:
        """Create a rollout lifecycle instance."""

        if config.skill_mode == "self-gen":
            raise ValueError(
                "self-gen requires the runtime orchestrator. Use the high-level "
                "run path instead of Rollout.create()."
            )
        return cls(config)
