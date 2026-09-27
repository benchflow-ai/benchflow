"""Host-side hard deadline for one rollout attempt.

Every phase inside a rollout carries its own timeout (install, agent idle +
wall-clock, verifier, PTY readline), but an await stuck BELOW that
instrumentation — a Daytona PTY kill on a dead websocket, a wedged session
exec in the post-verify export path — can block forever and freeze the caller
(for example, one hung rollout can block a whole multi-task evaluation long
after its verifier has finished). The hard deadline is a
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

A deadline that fires while the agent is still active is an agent timeout,
not an infrastructure failure (#1134): only the agent phase is stopped, the
lifecycle verifies and tears down within :func:`post_agent_grace_sec`, and the
result keeps the verifier's reward with the ``timeout`` category. The
whole-lifecycle kill (``INFRA_ERROR``) remains the backstop when the agent
phase had already ended or the post-agent stages outlive that grace.

Override with ``BENCHFLOW_ROLLOUT_HARD_DEADLINE`` (seconds; ``off``/``none``/
``0`` or any non-positive number disables the backstop entirely).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Coroutine, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchflow._utils.config_override import apply_config_override
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
# Uploaded evidence has no phase budget of its own; allow it the same ~1 MiB/s
# transfer floor publish/broker.py grants slow uplinks.
TRANSFER_BYTES_PER_SEC = 1024**2


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
    plus a soft verify), :func:`effective_install_timeout` — the single
    source of truth for the install budget — per distinct agent, and each
    setup command's own timeout. ``cfg.uploads`` (a reviewer's copy of the
    trial and its workspace archives) adds transfer time for its size, so a
    reviewer's deadline scales with the workspace it reviews (#1134).
    """
    from benchflow.agents.install import effective_install_timeout
    from benchflow.task import Task

    # Match setup() before reading any phase budget: per-run overlays can
    # raise both agent and sandbox build timeouts above the on-disk defaults.
    tcfg = apply_config_override(Task(cfg.task_path).config, cfg.config_override)
    agent_default = float(cfg.timeout or tcfg.agent.timeout_sec or 900.0)
    verifier_sec = float(tcfg.verifier.timeout_sec or 900.0)
    build_sec = float(tcfg.sandbox.build_timeout_sec or 600.0)
    setup_sec = sum(
        float(command.timeout_sec) for command in tcfg.sandbox.setup_commands
    )
    transfer_sec = _upload_bytes(cfg.uploads) / TRANSFER_BYTES_PER_SEC

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
    # A completed solver may need a fresh sandbox for verifier-only recovery
    # (#1136). Include both evidence capture/upload budgets (600s each), a
    # second sandbox build, and one verifier budget without replaying the agent.
    recovery_sec = (
        build_sec + verifier_sec + 1200.0
        if not cfg.skip_verify and tcfg.verifier.workspace_recovery
        else 0.0
    )
    return (
        agent_total
        + verifier_sec
        + build_sec
        + install_total
        + setup_sec
        + transfer_sec
        + recovery_sec
        + _MARGIN_SEC
    )


def _upload_bytes(sources: Iterable[str]) -> int:
    """Total size of the host files and directory trees a rollout uploads."""
    total = 0
    for source in sources:
        path = Path(source)
        try:
            if not path.is_dir():
                total += path.stat().st_size
                continue
            for root, _, files in os.walk(path):
                total += sum(
                    os.lstat(os.path.join(root, name)).st_size for name in files
                )
        except OSError as e:
            logger.debug(f"hard-deadline: could not size upload {source}: {e}")
    return total


def post_agent_grace_sec(cfg: RolloutConfig) -> float:
    """Budget for verification and teardown after the deadline stopped the agent.

    One verifier budget plus the fixed margin, which also covers workspace
    evidence capture and cleanup. Verifier-only recovery and review run after
    the lifecycle returns, so they need no share of this grace.
    """
    from benchflow.task import Task

    try:
        tcfg = apply_config_override(Task(cfg.task_path).config, cfg.config_override)
        verifier_sec = float(tcfg.verifier.timeout_sec or 900.0)
    except Exception as e:
        logger.debug(f"hard-deadline: could not read {cfg.task_path.name} budgets: {e}")
        verifier_sec = 900.0
    return verifier_sec + _MARGIN_SEC


async def enforce_hard_deadline(
    lifecycle: Coroutine[Any, Any, RolloutResult],
    *,
    config: RolloutConfig,
    stop_agent: Callable[[str], bool] | None = None,
) -> RolloutResult:
    """Run one rollout lifecycle under the host-side hard deadline.

    At the deadline, ``stop_agent(reason)`` is offered the trip first: when it
    reports that it stopped a still-active agent phase, the lifecycle records
    ``reason`` as the agent timeout and gets :func:`post_agent_grace_sec` to
    verify and clean up, and its own result is returned. Otherwise — or when
    that grace also runs out — the lifecycle is cancelled (running its own
    ``finally`` cleanup, bounded by :data:`ABANDON_CLEANUP_BOUND_SEC`) and a
    normal infra-retryable error result is returned. External cancellation of
    the caller is forwarded to the lifecycle under the same cleanup bound.
    """
    deadline = hard_deadline_sec(config)
    if deadline is None:
        return await lifecycle
    inner = asyncio.ensure_future(lifecycle)
    if await _finished_within(inner, deadline):
        return inner.result()
    task_name = config.task_path.name
    reason = (
        f"Agent timed out: still running at the host hard deadline "
        f"({deadline:.0f}s); agent stopped so the verifier could run"
    )
    wedged = "transport or teardown wedged below the idle/wall-clock watchdogs"
    if stop_agent is not None and stop_agent(reason):
        grace = post_agent_grace_sec(config)
        logger.error(
            f"[HARD-DEADLINE] {task_name}: agent still running at host deadline "
            f"({deadline:.0f}s); stopped it, verifying within {grace:.0f}s"
        )
        if await _finished_within(inner, grace):
            return inner.result()
        wedged = (
            f"agent stopped at the deadline, then verification or teardown "
            f"exceeded its {grace:.0f}s grace"
        )
    logger.error(
        f"[HARD-DEADLINE] {task_name}: rollout exceeded host deadline "
        f"({deadline:.0f}s); abandoning sandbox"
    )
    await _cancel_and_abandon(inner)
    return RolloutResult(
        task_name=task_name,
        error=(
            f"Rollout exceeded host hard deadline ({deadline:.0f}s) — "
            f"{wedged}; sandbox abandoned"
        ),
        error_category=INFRA_ERROR,
    )


async def _finished_within(inner: asyncio.Task, timeout: float) -> bool:
    """Wait for the lifecycle; forward caller cancellation under the bound."""
    try:
        done, _ = await asyncio.wait({inner}, timeout=timeout)
    except asyncio.CancelledError:
        await _cancel_and_abandon(inner)
        raise
    return inner in done


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
