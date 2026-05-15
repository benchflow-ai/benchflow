"""Rollout configuration primitives.

The v0.4 architecture uses these dataclasses as the canonical definitions for
roles, turns, and scenes. Legacy modules may temporarily re-export them during
the migration, but new code should import from ``benchflow.rollouts``.
"""

from benchflow.rollouts.config import (
    GENERATED_SKILLS_ROOT,
    SKILL_MODE_DEFAULT,
    SKILL_MODE_SELF_GEN,
    Role,
    RolloutConfig,
    Scene,
    Turn,
)
from benchflow.rollouts.result import RolloutResult, TrajectorySource

__all__ = [
    "GENERATED_SKILLS_ROOT",
    "Role",
    "RolloutConfig",
    "RolloutResult",
    "SKILL_MODE_DEFAULT",
    "SKILL_MODE_SELF_GEN",
    "Scene",
    "TrajectorySource",
    "Turn",
]
