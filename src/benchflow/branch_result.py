"""A first-class ``result.json`` for in-place branch children (RFC §3.4).

A branch child forked from ``env-ready`` runs as its own Rollout, so it leaves
the standard run artifacts — ``config.json`` / ``result.json`` /
``timing.json`` / trajectory — in its child directory for free. An *in-place*
child (a ``pre-verify`` / ``post-verify`` stage branch, or a cursor branch)
continues the **parent** Rollout instance instead: until this module, its
directory held only ``provenance.json`` / ``reward.json`` plus the ``mounted/``
archive, and "what happened in this arm" could only be answered by
cross-reading ``tree.json``. This module closes that gap so every child
directory is self-describing, whichever way the engine ran the child.

The mechanism leans on the isolation the engine already provides. The parent's
linear and result-bearing state is captured before the fork and restored
around every child (:class:`~benchflow.rollout_branch._LinearState`), so while
a child runs, the shared instance *is* that child's state. Two functions split
the work:

* :func:`scope_child_result_state` — called by the engine after it restores
  the parent's state onto the instance and before the child runs. It zeroes
  the result-bearing fields the child's own phases write (``_timing``,
  ``_rewards``, ``_verifier_error``, the diagnostics collector, …), so that
  what those fields hold *after* the child is exactly the child's own — never
  the parent's value showing through a field the child happened not to write.
  The parent's values come back with the engine's next restore, exactly as
  they always have.
* :func:`write_in_place_child_result` — called after the child completed
  (scored or unscored; a child that raised ends the fork and is evidenced by
  its ``mounted/`` archive instead). It builds the child's result through the
  same :func:`~benchflow.rollout._results._build_rollout_result` the linear
  run and the fresh-rollout child use — one result writer, no branch-specific
  copy — from the child's own slice of the shared state: the trajectory steps
  and prompts *appended since the fork baseline*, the scoped timing dict, the
  child's own rewards and verifier error.

The honesty rule is the module's contract: a field an in-place child does not
genuinely produce is ``null``/absent, **never copied from the parent**. That
is why token usage reports ``usage_source="unavailable"`` (per-child usage is
not attributable on a shared instance), why ``error`` is ``None`` unless the
child itself recorded one, and why the child's ``rewards`` fall back to the
reward the engine recorded on the child node — an observation the child did
produce — rather than to the parent's rewards dict.

Failure-isolated like every branch artifact writer: a result that cannot be
built is logged and skipped, and never costs the reward the fork was run to
measure.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

from benchflow.branch_lineage import branch_child_dir, child_provenance

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from benchflow.rollout import Rollout
    from benchflow.trajectories.tree import RolloutNode

logger = logging.getLogger(__name__)

#: The result-bearing fields :func:`scope_child_result_state` zeroes to
#: ``None``. Each is either assigned outright by a child phase (``_rewards`` /
#: ``_verifier_error`` by ``verify()``) or never written by a child at all
#: (``_error`` / ``_export_error`` / ``_evolved_skills`` belong to ``run()``
#: and ``cleanup()``, which no in-place child re-runs) — so after the child,
#: a non-``None`` value is the child's own and ``None`` is the honest
#: "not produced by this child".
_SCOPED_NONE_FIELDS: tuple[str, ...] = (
    "_rewards",
    "_verifier_error",
    "_error",
    "_export_error",
    "_evolved_skills",
)


def scope_child_result_state(rollout: Rollout) -> None:
    """Zero the result-bearing state an in-place child reports through.

    Called between the engine's per-child ``restore_onto`` (which put the
    *parent's* values back) and the child runner. Only attributes the instance
    already has are touched — the ``_LinearState`` convention: what is absent
    stays absent — and the parent's own values are restored by the engine
    before the next child and at the end of the fork, so nothing here outlives
    the child it scopes.
    """
    for name in _SCOPED_NONE_FIELDS:
        if hasattr(rollout, name):
            setattr(rollout, name, None)
    if hasattr(rollout, "_timing"):
        rollout._timing = {}
    if hasattr(rollout, "_diagnostics"):
        from benchflow.diagnostics import RolloutDiagnostics

        rollout._diagnostics = RolloutDiagnostics()


def write_in_place_child_result(
    rollout: Rollout,
    *,
    parent: RolloutNode,
    child: RolloutNode,
    run_dir: Path,
    started_at: datetime,
    base_trajectory_len: int,
    base_prompt_count: int,
    base_tool_calls: int,
) -> None:
    """Write the completed in-place child's own result artifacts, best-effort.

    The ``base_*`` arguments are the fork baseline — the lengths/counts of the
    parent's captured linear state — so the child's trajectory, prompts and
    tool-call count are the *delta* its continuation appended, never the
    parent's history re-labelled as the child's. Everything else is read off
    the instance, which :func:`scope_child_result_state` made the child's own.

    Never raises: an artifact failure is logged and the child's reward (already
    recorded on its node) stands.
    """
    try:
        _write_result(
            rollout,
            parent=parent,
            child=child,
            run_dir=run_dir,
            started_at=started_at,
            base_trajectory_len=base_trajectory_len,
            base_prompt_count=base_prompt_count,
            base_tool_calls=base_tool_calls,
        )
    except Exception:
        logger.warning(
            "in-place branch child %s result artifacts failed to build — the "
            "child's reward is unaffected",
            child.id,
            exc_info=True,
        )


def _write_result(
    rollout: Rollout,
    *,
    parent: RolloutNode,
    child: RolloutNode,
    run_dir: Path,
    started_at: datetime,
    base_trajectory_len: int,
    base_prompt_count: int,
    base_tool_calls: int,
) -> None:
    """Build and write the child's result set — see the module docstring."""
    config = getattr(rollout, "_config", None)
    if config is None:
        # A harness-built stand-in with no config cannot honestly describe a
        # run; there is nothing to synthesize a result *about*.
        logger.debug(
            "in-place branch child %s has no rollout config; skipping result synthesis",
            child.id,
        )
        return
    from benchflow.rollout._results import _build_rollout_result
    from benchflow.skill_policy import resolve_task_skill_policy

    child_dir = branch_child_dir(run_dir, parent.id, child.id)

    # The child's own continuation: everything appended past the fork baseline.
    trajectory = list(getattr(rollout, "_trajectory", []) or [])[base_trajectory_len:]
    prompts = list(getattr(rollout, "_executed_prompts", []) or [])[base_prompt_count:]
    n_tool_calls = max(
        0, getattr(rollout, "_n_tool_calls", base_tool_calls) - base_tool_calls
    )

    # The child's own score. ``_rewards`` was scoped to None before the child
    # ran, so a dict here is what *its* verify() produced; the fallback is the
    # scalar the engine recorded on the child node (a caller-run child may
    # score without writing ``_rewards``). A child the engine recorded as
    # *unscored* publishes ``None`` outright — the node's unscored marker is
    # the engine's verdict that no score was observed, and a leftover ``{}``
    # from a verifier that returned nothing must not read as an observation.
    from benchflow.branch import UNSCORED_KEY

    if UNSCORED_KEY in child.state:
        rewards = None
    else:
        rewards = copy.deepcopy(getattr(rollout, "_rewards", None))
        if rewards is None and "reward" in child.state:
            rewards = {"reward": float(child.state["reward"])}

    # The same branch provenance the child's provenance.json carries, so
    # result.json names its parent, stage and snapshot refs on its own.
    snapshot = parent.state.get("snapshot")
    stage = getattr(snapshot, "stage", None)
    provenance = child_provenance(
        str(run_dir),
        branch_stage=stage if stage is not None else f"cursor:{parent.id}",
        snapshot=snapshot,
        delta=child.state.get("delta"),
        delta_execution=child.state.get("delta_execution"),
    )

    # The world the in-place child genuinely ran in is the parent's: shared
    # sandbox, shared skill deployment. The parent's resolved policy (or the
    # same resolution _build_result would run) is therefore the child's own.
    skill_policy = getattr(rollout, "_task_skill_policy", None)
    if skill_policy is None:
        skill_policy = resolve_task_skill_policy(
            task_path=config.task_path,
            skill_mode=config.recorded_skill_mode,
            runtime_skills_dir=config.skills_dir,
            declared_sandbox_skills_dir=None,
        )
    sandbox_id_fn = getattr(rollout, "_current_sandbox_id", None)
    _build_rollout_result(
        child_dir,
        task_name=config.task_path.name,
        rollout_name=child.id,
        agent=config.primary_agent,
        agent_name=getattr(rollout, "_agent_name", "") or "",
        model=config.primary_model,
        n_tool_calls=n_tool_calls,
        prompts=prompts,
        error=getattr(rollout, "_error", None),
        verifier_error=getattr(rollout, "_verifier_error", None),
        export_error=getattr(rollout, "_export_error", None),
        trajectory=trajectory,
        partial_trajectory=bool(getattr(rollout, "_partial_trajectory", False)),
        trajectory_source=getattr(rollout, "_trajectory_source", None),
        rewards=rewards,
        started_at=started_at,
        timing=dict(getattr(rollout, "_timing", {}) or {}),
        scenes=config.effective_scenes,
        evolved_skills=getattr(rollout, "_evolved_skills", None),
        source_provenance=provenance,
        dataset=config.dataset,
        task_digest=config.task_digest,
        diagnostics=getattr(rollout, "_diagnostics", None),
        skill_policy=skill_policy,
        sandbox_id=sandbox_id_fn() if callable(sandbox_id_fn) else None,
        # No usage kwargs on purpose: per-child token usage is not
        # attributable on the shared instance, so the result honestly reports
        # usage_source="unavailable" instead of inheriting the parent's.
    )
