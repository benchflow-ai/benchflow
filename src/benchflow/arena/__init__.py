"""Inter-agent concurrent (arena) runtime — an OPT-IN scaffold.

Runs N agent "seats" concurrently against ONE shared environment service, the
genuinely-missing axis of multi-agent (the deferred ``arena-concurrent`` mode).
This package is additive and self-contained: it does not modify the sequential
scene path (``scenes.compile_scenes_to_steps``) or the scalar reward path, so no
existing scored benchmark is affected.

  - :mod:`~benchflow.arena.protocol` — the turn-poll contract (Seam 3).
  - :func:`~benchflow.arena.runtime.run_arena` — concurrent seat driver (Seam 1
    + eval glue: worker cap, wall-clock deadline, straggler reaper).
  - :class:`~benchflow.arena.reward.SharedEnvReward` — per-seat reward vector
    (Seam 4).

The co-tenant environment topology (Seam 2 — provision one service per scene and
attach K seats) is intentionally left to the caller / a future
``SharedManifestEnvironment``; this scaffold is driven by any ``SeatClient``.
"""

from __future__ import annotations

from benchflow.arena.protocol import (
    Observation,
    SeatClient,
    SeatPolicy,
    SeatStatus,
)
from benchflow.arena.reward import FloorMode, SharedEnvReward
from benchflow.arena.runtime import run_arena

__all__ = [
    "FloorMode",
    "Observation",
    "SeatClient",
    "SeatPolicy",
    "SeatStatus",
    "SharedEnvReward",
    "run_arena",
]
