"""Rollout run helpers."""

from __future__ import annotations

from benchflow.rollouts.config import RolloutConfig
from benchflow.rollouts.result import RolloutResult
from benchflow.rollouts.rollout import Rollout


async def run(config: RolloutConfig) -> RolloutResult:
    """Execute a rollout config and return its result."""

    if config.skill_mode == "self-gen":
        from benchflow.self_gen import run_self_gen

        return await run_self_gen(config)
    rollout = await Rollout.create(config)
    return await rollout.run()
