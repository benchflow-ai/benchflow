"""Single entry point: ``bf.run(RolloutConfig) → RolloutResult``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchflow.models import RolloutResult

if TYPE_CHECKING:
    from benchflow.rollout import RolloutConfig


async def run(config: RolloutConfig | None = None, **kwargs: Any) -> RolloutResult:
    """Execute a single rollout and return the result.

    This is the canonical entry point for running an agent on a task::

        import benchflow as bf
        result = await bf.run(RolloutConfig(task_path=..., agent=...))

    If *config* is None, a :class:`RolloutConfig` is built from *kwargs*
    via :meth:`RolloutConfig.from_legacy`.
    """
    from benchflow.rollout import SKILL_MODE_SELF_GEN, Rollout
    from benchflow.rollout import RolloutConfig as _RolloutConfig

    if config is None:
        config = _RolloutConfig.from_legacy(**kwargs)
    if config.skill_mode == SKILL_MODE_SELF_GEN:
        from benchflow.self_gen import run_self_gen

        return await run_self_gen(config)
    rollout = await Rollout.create(config)
    return await rollout.run()
