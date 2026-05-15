"""Rollout configuration primitives.

The v0.4 architecture uses these dataclasses as the canonical definitions for
roles, turns, and scenes. Legacy modules may temporarily re-export them during
the migration, but new code should import from ``benchflow.rollouts``.
"""

from benchflow.rollouts.config import Role, Scene, Turn

__all__ = ["Role", "Scene", "Turn"]
