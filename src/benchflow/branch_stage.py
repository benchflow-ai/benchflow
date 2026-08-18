"""The branch stage taxonomy — the four cascade boundaries (RFC §3.2).

The FrontierPhysics failure cascade names four stages a research-style agent
run can die in; each pins to an existing transition of the rollout lifecycle,
so this is a *naming* layer over ``rollout/__init__.py``, not a new phase
system:

===================  ==================================================
``env-ready``        end of ``start()`` — sandbox up, environment plane
                     provisioned, readiness gate passed, **before**
                     ``install_agent()`` (so a child re-runs agent/skill
                     installation from a skill-free world)
``post-research``    a mid-``execute()`` cut point that cannot be
                     auto-detected — recorded by an explicit
                     ``Rollout.mark_stage()`` from the caller/harness that
                     knows when planning ended
``pre-verify``       agent quiesced, immediately **before**
                     ``planes.harden_before_verify``
``post-verify``      after ``verify()`` completes
===================  ==================================================

This module is a dependency-free leaf on purpose: ``RolloutConfig`` validates
``snapshot_stages`` at construction time and must not drag the branch engine's
snapshot/environment/sandbox import chain into config validation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

STAGE_ENV_READY = "env-ready"
STAGE_POST_RESEARCH = "post-research"
STAGE_PRE_VERIFY = "pre-verify"
STAGE_POST_VERIFY = "post-verify"

#: Every stage, in lifecycle order — the order diagnostics and artifacts use.
BRANCH_STAGES: tuple[str, ...] = (
    STAGE_ENV_READY,
    STAGE_POST_RESEARCH,
    STAGE_PRE_VERIFY,
    STAGE_POST_VERIFY,
)

#: Stages the rollout captures by itself when they are requested.
AUTO_STAGES = frozenset({STAGE_ENV_READY, STAGE_PRE_VERIFY, STAGE_POST_VERIFY})

#: Stages only an explicit ``Rollout.mark_stage()`` can record.
MARKED_STAGES = frozenset(BRANCH_STAGES) - AUTO_STAGES

_KNOWN_STAGES = frozenset(BRANCH_STAGES)


class BranchStageNotCaptured(LookupError):
    """A branch was requested at a stage this rollout never snapshotted.

    Fail-closed counterpart of the capability errors: branching from a stage
    that was never captured cannot silently degrade into a checkpoint at the
    cursor — that would fork a *different* world state than the caller asked
    for and mislabel it in provenance.
    """


def validate_stage(stage: str, *, field: str = "stage") -> str:
    """Return ``stage`` if it names a taxonomy stage, else raise ``ValueError``."""
    if stage not in _KNOWN_STAGES:
        raise ValueError(
            f"unknown {field} {stage!r} — the branch stage taxonomy is "
            f"{list(BRANCH_STAGES)!r}"
        )
    return stage


def normalize_stages(stages: Iterable[str] | None) -> frozenset[str]:
    """Coerce a requested stage set to a validated ``frozenset``.

    ``None`` and an empty iterable both mean *no stage snapshots* — today's
    behavior exactly, with no snapshot call anywhere in the lifecycle. A
    single string is rejected rather than silently iterated into characters.
    """
    if stages is None:
        return frozenset()
    if isinstance(stages, str):
        raise ValueError(
            f"snapshot_stages must be a collection of stage names, got the "
            f"string {stages!r} — pass e.g. {{'{STAGE_ENV_READY}'}}"
        )
    resolved = frozenset(stages)
    for stage in sorted(resolved):
        validate_stage(stage, field="snapshot_stages entry")
    return resolved


def captured_stages(registry: Mapping[str, Any]) -> list[str]:
    """The recorded stages of ``registry``, in lifecycle order (deterministic)."""
    return [stage for stage in BRANCH_STAGES if stage in registry]
