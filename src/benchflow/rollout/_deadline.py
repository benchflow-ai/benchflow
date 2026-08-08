"""Host-side hard deadline for one rollout attempt.

Every phase inside a rollout carries its own timeout (install, agent idle +
wall-clock, verifier, PTY readline), but an await stuck BELOW that
instrumentation — a Daytona PTY kill on a dead websocket, a wedged session
exec in the post-verify export path — can block forever and freeze the caller
(observed 2026-08-07: one hung bike-rebalance rollout wedged a 25-task eval
for 11+ hours after its verifier had already finished). The hard deadline is a
backstop, not a budget: it is derived from the sum of every phase budget plus
a generous fixed margin, so it can only fire when some await is stuck outside
all phase-level timeouts.

Enforcement lives here too (:func:`enforce_hard_deadline`), and deliberately
does NOT use a bare ``asyncio.wait_for``: cancelling ``Rollout.run()`` runs
its ``finally: cleanup()``, and when the *teardown* is the wedged path,
``wait_for`` blocks past its own deadline waiting for that cleanup to finish
— the exact hang this backstop exists to break. Instead the lifecycle runs as
a task, cancellation gets its own bounded grace period, and a still-wedged
task is abandoned (the sandbox provider's GC reaps the leaked resources).

Override with ``BENCHFLOW_ROLLOUT_HARD_DEADLINE`` (seconds; ``off``/``none``/
``0`` or any non-positive number disables the backstop entirely).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any

from benchflow._utils.scoring import INFRA_ERROR
from benchflow.models import RolloutResult

if TYPE_CHECKING:
    from benchflow.rollout._config import RolloutConfig

logger = logging.getLogger(__name__)

HARD_DEADLINE_ENV = "BENCHFLOW_ROLLOUT_HARD_DEADLINE"
_MARGIN_SEC = 1800.0
_FALLBACK_SEC = 3 * 3600.0
# Grace period for the cancelled lifecycle's own cleanup() before the task is
# abandoned outright.
ABANDON_CLEANUP_BOUND_SEC = 120.0


def hard_deadline_sec(cfg: RolloutConfig) -> float | None:
    """Compute the hard backstop deadline for one rollout attempt.

    Returns ``None`` when the operator disabled the backstop. On any failure
    to read the task's budgets, falls back to a conservative constant rather
    than running unbounded.
    """
    raw = os.environ.get(HARD_DEADLINE_ENV, "").strip().lower()
    if raw in {"off", "none"}:
        return None
    if raw:
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                f"{HARD_DEADLINE_ENV}={raw!r} is not a number; using computed deadline"
            )
        else:
            return value if value > 0 else None
    try:
        return _computed_deadline_sec(cfg)
    except Exception as e:
        logger.debug(f"hard-deadline: could not read {cfg.task_path.name} budgets: {e}")
        return _FALLBACK_SEC


def _computed_deadline_sec(cfg: RolloutConfig) -> float:
    """Sum every phase budget the rollout could legitimately spend.

    Uses the same sources the phases themselves enforce: the caller wall-clock
    override (``cfg.timeout``) or the task's agent budget per turn, per-role
    ``timeout_sec`` overrides, user-loop rounds (each round runs one prompt
    plus a soft verify), and :func:`effective_install_timeout` — the single
    source of truth for the install budget — per distinct agent.
    """
    from benchflow.agents.install import effective_install_timeout
    from benchflow.task import Task

    tcfg = Task(cfg.task_path).config
    agent_default = float(cfg.timeout or tcfg.agent.timeout_sec or 900.0)
    verifier_sec = float(tcfg.verifier.timeout_sec or 900.0)
    build_sec = float(tcfg.sandbox.build_timeout_sec or 600.0)

    scenes = cfg.effective_scenes
    role_budget = {
        role.name: float(role.timeout_sec)
        for scene in scenes
        for role in scene.roles
        if role.timeout_sec
    }
    agent_total = sum(
        role_budget.get(turn.role, agent_default)
        for scene in scenes
        for turn in scene.turns
    )
    agent_total = max(agent_total, agent_default)
    if cfg.user is not None:
        agent_total = max(
            agent_total, cfg.max_user_rounds * (agent_default + verifier_sec)
        )

    agents = {role.agent for scene in scenes for role in scene.roles} or {cfg.agent}
    install_total = sum(
        float(effective_install_timeout(agent, cfg.sandbox_setup_timeout) or 0.0)
        for agent in agents
    )
    return agent_total + verifier_sec + build_sec + install_total + _MARGIN_SEC


async def enforce_hard_deadline(
    lifecycle: Coroutine[Any, Any, RolloutResult],
    *,
    config: RolloutConfig,
) -> RolloutResult:
    """Run one rollout lifecycle under the host-side hard deadline.

    A deadline trip cancels the lifecycle (which runs its own ``finally``
    cleanup), bounds that cleanup with :data:`ABANDON_CLEANUP_BOUND_SEC`, and
    returns a normal infra-retryable error result. External cancellation of
    the caller is forwarded to the lifecycle under the same cleanup bound.
    """
    deadline = hard_deadline_sec(config)
    if deadline is None:
        return await lifecycle
    inner = asyncio.ensure_future(lifecycle)
    try:
        done, _ = await asyncio.wait({inner}, timeout=deadline)
    except asyncio.CancelledError:
        await _cancel_and_abandon(inner)
        raise
    if inner in done:
        return inner.result()
    task_name = config.task_path.name
    logger.error(
        f"[HARD-DEADLINE] {task_name}: rollout exceeded host deadline "
        f"({deadline:.0f}s); abandoning sandbox"
    )
    await _cancel_and_abandon(inner)
    return RolloutResult(
        task_name=task_name,
        error=(
            f"Rollout exceeded host hard deadline ({deadline:.0f}s) — "
            "transport or teardown wedged below the idle/wall-clock "
            "watchdogs; sandbox abandoned"
        ),
        error_category=INFRA_ERROR,
    )


async def _cancel_and_abandon(inner: asyncio.Task) -> None:
    """Cancel the lifecycle; give its cleanup a bounded grace, then abandon.

    Cancellation runs the lifecycle's ``finally: cleanup()``. When that
    teardown is itself the wedged path, waiting for it would re-freeze the
    caller — so after the bound the task is left running detached, with its
    eventual outcome swallowed to silence the never-retrieved warning.
    """
    inner.cancel()
    done, _ = await asyncio.wait({inner}, timeout=ABANDON_CLEANUP_BOUND_SEC)
    if inner in done:
        return
    logger.error(
        "[HARD-DEADLINE] rollout cleanup wedged too; abandoning the task outright"
    )
    inner.add_done_callback(_swallow_abandoned_outcome)


def _swallow_abandoned_outcome(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()
