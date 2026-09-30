"""Embodied rollouts: the low-level layer for robot and simulator tasks. See docs/embodied.md.

  spec        Embodiment: sensors, action groups, skills, budgets (one schema for every robot kind);
              the restore boundary (branching, checkpoint restores and replay of embodied tasks)
  backend     SimBackend: the contract a simulator implements
  protocol    the episode wire protocol (JSON over a Unix socket)
  robo        the agent-facing `robo` command (standard library only; copied into agent images)
  server      EpisodeServer: budgets, recording, judging, roles, per-step trace
  skills      built-in arm and gripper controllers
  serve       `python -m benchflow.embodied.serve --factory module:function ...`
  sidecar     EmbodiedTaskFormat: materializes tasks as native packages with a simulator sidecar
  verifier    the physical verifier (runs in the simulator service)
  export      training export of episode records
  rollouts    seeded rollouts: pass@k, variance, reset reproducibility
  trials      the restore boundary of task folders and finished trials (host side)

This package imports nothing else from BenchFlow, so simulator images can vendor it (sidecar.vendor_embodied).
"""

from benchflow.embodied.backend import SimBackend, StepResult
from benchflow.embodied.spec import (
    ActionGroup,
    Budgets,
    Camera,
    Embodiment,
    Field,
    RestoreBoundary,
    RestoreRefused,
    RewardSpec,
    Sensors,
    Skill,
    SkillArg,
    SpecError,
    restore_boundary,
    task_restore_boundary,
)

__all__ = [
    "ActionGroup",
    "Budgets",
    "Camera",
    "Embodiment",
    "Field",
    "RestoreBoundary",
    "RestoreRefused",
    "RewardSpec",
    "Sensors",
    "SimBackend",
    "Skill",
    "SkillArg",
    "SpecError",
    "StepResult",
    "restore_boundary",
    "task_restore_boundary",
]
