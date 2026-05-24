"""Monitor mode — score a rollout in production.

Architecture (``docs/architecture.md``) names three first-class modes on the
single ``scored rollout`` engine:

- **Eval**    — score it and stop.
- **Train**   — score it and hand the trajectory to a trainer.
- **Monitor** — score it in production.

Eval and Train have runtime code paths; Monitor does not yet. This module is
the **API surface stub** for #386 so callers, the CLI, and downstream tooling
have a stable import target while the runtime is built out. Every entry point
raises :class:`NotImplementedError` with a pointer to the next step.

Why a stub now (versus deleting Monitor from the architecture doc):
    The reward plane, trajectory schema, and ``RewardEvent`` tagging
    (``space``/``granularity``) were designed so that Eval, Train, and Monitor
    share one signal pipeline. Deferring the API forever risks Eval and Train
    drifting into shapes that the future Monitor cannot adopt without a
    breaking change. The stubs let us *fail closed* with a clear message
    today and grow the real implementation behind the same surface.

Next steps (tracked under #386):
    1. Input source: persisted-rollout replay vs. live trace ingestion
       (webhook/polling) — pick one for the MVP.
    2. Output schema: how :class:`MonitorResult` differs from
       :class:`benchflow.runtime.RuntimeResult` (alert metadata, incident
       tags), or whether it just reuses the existing trajectory record with
       a ``mode="monitor"`` discriminator.
    3. Failure taxonomy: production incident vs. verifier error vs. infra
       error — these must be distinguishable at the dashboard level.
    4. Artifact path: ``jobs/monitor/<run>/`` parallel to ``jobs/eval/`` so
       monitor evidence does not pollute release-eval evidence.
    5. CLI wiring: ``bench monitor run``, ``bench monitor replay``,
       ``bench monitor watch`` — exact verbs land with the MVP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "MonitorConfig",
    "MonitorResult",
    "Monitor",
    "MonitorNotImplementedError",
    "not_implemented_message",
]


class MonitorNotImplementedError(NotImplementedError):
    """Raised by Monitor stubs until the runtime lands (#386).

    A dedicated subclass so callers and tests can distinguish "this whole
    mode isn't built yet" from incidental NotImplementedErrors elsewhere.
    """


_NOT_IMPLEMENTED_MSG = (
    "Monitor mode is not yet implemented — only the API surface is scaffolded. "
    "Track progress on https://github.com/benchflow-ai/benchflow/issues/386 ; "
    "see docstring on benchflow.monitor for the planned next steps."
)


def not_implemented_message() -> str:
    """Canonical not-implemented message for Monitor mode.

    Exported so the CLI (and other surfaces) emit consistent wording
    without reaching into module-private constants.
    """
    return _NOT_IMPLEMENTED_MSG


@dataclass
class MonitorConfig:
    """Configuration for a monitor run.

    Mirrors the shape of :class:`benchflow.rollout.RolloutConfig` so that
    Eval/Train/Monitor share one mental model: a scored rollout with a
    pluggable source. Fields are intentionally minimal until #386's MVP
    pins the input source contract.

    Attributes:
        source:
            Where the monitored trajectory comes from. Until the runtime
            lands this is opaque — planned values are a path to a
            persisted rollout directory (replay mode) or a URI for live
            ingestion (watch mode).
        rubric_path:
            Optional path to a rubric/verifier definition to score the
            trajectory against. ``None`` means "use the same verifier the
            trajectory was originally produced with."
        jobs_dir:
            Output root for monitor artifacts. Defaults to
            ``jobs/monitor`` so monitor evidence is separable from eval
            release evidence (see ``docs/architecture.md`` §Lifecycles).
        run_name:
            Optional human-readable name for the monitor run; defaults
            to a timestamped id.
        metadata:
            Free-form tags (incident id, deployment id, region…) carried
            into :class:`MonitorResult` for downstream alerting.
    """

    source: str | Path
    rubric_path: str | Path | None = None
    jobs_dir: str | Path = "jobs/monitor"
    run_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorResult:
    """Result of a monitor run.

    Shape mirrors :class:`benchflow.runtime.RuntimeResult` deliberately so
    that the reward/trajectory contract is identical across Eval, Train,
    and Monitor (the architecture's "eval = monitoring = reward" promise).

    Attributes:
        run_name:        Stable id for the monitor run.
        source:          Origin of the scored trajectory (as supplied in
                         :class:`MonitorConfig`).
        rewards:         Same shape as ``RolloutResult.rewards`` — a
                         dict keyed by reward name, plus the canonical
                         ``"reward"`` key for the terminal scalar.
        trajectory:      Full trajectory under the monitor lens. Same
                         schema as eval/train trajectories.
        error:           Infra/transport failure (could not score).
        verifier_error:  Verifier crashed while scoring (failure of the
                         monitor itself, *not* of the production trace).
        alert:           True iff this run should page a human — distinct
                         from ``passed`` so production incidents do not
                         look like eval regressions on dashboards.
        artifact_dir:    Where artifacts are written under ``jobs_dir``.
        metadata:        Echoed from :class:`MonitorConfig`.
        started_at/finished_at: Run window for the monitor itself, not
                         for the underlying production event.
    """

    run_name: str
    source: str
    rewards: dict[str, Any] | None = None
    trajectory: list[dict] = field(default_factory=list)
    error: str | None = None
    verifier_error: str | None = None
    alert: bool = False
    artifact_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class Monitor:
    """Score a rollout in production (Mode 3 of the architecture).

    API surface only — all methods raise :class:`MonitorNotImplementedError`
    pointing at #386. The shape is fixed so callers and the CLI can import
    against it today without committing to runtime details.

    Usage (target — not yet functional)::

        from benchflow.monitor import Monitor, MonitorConfig

        config = MonitorConfig(source="jobs/prod-trace/abc123")
        monitor = Monitor(config)
        result = await monitor.run()       # -> MonitorResult
        # or, for a live stream of production events:
        async for result in monitor.watch():
            ...
    """

    def __init__(self, config: MonitorConfig) -> None:
        self.config = config

    async def run(self) -> MonitorResult:
        """Score a single persisted trajectory against the monitor rubric.

        Not implemented — see :class:`Monitor` and :mod:`benchflow.monitor`.
        """
        logger.warning(_NOT_IMPLEMENTED_MSG)
        raise MonitorNotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def replay(self, trajectory_path: str | Path) -> MonitorResult:
        """Re-score a persisted rollout under monitor semantics.

        Not implemented — see :class:`Monitor` and :mod:`benchflow.monitor`.
        """
        logger.warning(_NOT_IMPLEMENTED_MSG)
        raise MonitorNotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def watch(self):  # type: ignore[no-untyped-def]
        """Stream-score live production events.

        Not implemented — see :class:`Monitor` and :mod:`benchflow.monitor`.

        Will be an async iterator yielding :class:`MonitorResult` per event
        once the input source contract is pinned (step 1 of the next steps
        in :mod:`benchflow.monitor`).
        """
        logger.warning(_NOT_IMPLEMENTED_MSG)
        raise MonitorNotImplementedError(_NOT_IMPLEMENTED_MSG)
