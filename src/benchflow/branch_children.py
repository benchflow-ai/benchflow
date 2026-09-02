"""How a branch child executes its delta — the runner side of the engine.

One module answers "given this child's delta and this fork's boundary, what
actually runs?" (RFC §3.3). There are exactly two execution paths, and the
boundary — not the delta — picks between them:

* **in place** — the ordinary child at a post-installation boundary: a fresh
  agent session over the restored world, driven on the parent Rollout instance
  (:func:`make_default_runner`). An ``injected_prompt`` delta binds here as the
  child's user-visible continuation prompt.
* **fresh rollout** — every engine-run child of ``env-ready``: that boundary
  precedes ``install_agent()``, so the restored world has no agent to connect
  to and the child re-runs installation as its own Rollout over the restored
  sandbox. The implementation lives in :mod:`benchflow.branch_skill` (its
  original reason to exist — the skills ablation — named the module, and six
  test files plus the ``run_fresh_child`` monkeypatch seam pin that import
  path); this module re-exports its API so delta-execution callers have one
  home to import from.

:func:`select_child_runner` is the one place the choice is made — the branch
transaction loop asks it per child, so the boundary rule ("fresh at env-ready,
in place elsewhere, caller-supplied runners are left alone") cannot fork
between call sites.

The gates deciding whether a delta may execute at all live in
:mod:`benchflow.branch_policy` and fail closed before anything is quiesced.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from benchflow.branch import UnscoredChildError

# The fresh-rollout execution path (module of record: benchflow.branch_skill).
from benchflow.branch_skill import (  # noqa: F401
    EXECUTION_FRESH_ROLLOUT,
    FRESH_CHILD_LAYER,
    FRESH_CHILD_STAGE,
    SKILL_DELTA_LAYER,
    SKILL_DELTA_STAGE,
    BranchEnvironmentImageConflict,
    child_skill_config,
    fresh_child_skill_mode,
    make_fresh_child_runner,
    provision_child_environment,
    resolve_child_skill_policy,
    resolve_environment_ref_delta,
    run_fresh_child,
)

if TYPE_CHECKING:
    from pathlib import Path

    from benchflow.branch_delta import BranchDelta
    from benchflow.rollout import Rollout
    from benchflow.trajectories.tree import RolloutNode

# The per-child runner: given the child's branch node, run its continuation and
# return the scalar return. No ``int`` index — a caller that needs per-child
# prompts binds them into a closure (see ``select_child_runner``).
ChildRunner = Callable[["RolloutNode"], Awaitable[float]]


def make_default_runner(
    rollout: Rollout, *, prompts: list[str] | None = None
) -> ChildRunner:
    """Build the default per-child runner bound to ``rollout``.

    The default runner re-runs the child from the parent's env checkpoint with
    a *fresh agent session* — agent-session snapshot is the unsolved hard part
    (``docs/architecture.md``, "The hard part"), so the agent restarts per
    child. Each child connects a fresh agent and disconnects it at the end, so
    no two children's agents overlap (the next child connects only after the
    previous one disconnected). ``verify()`` returning ``None``, an empty dict,
    or a dict with a ``None`` reward raises
    :class:`~benchflow.branch.UnscoredChildError`: the child ran but was never
    scored, and a fabricated ``0.0`` would read as a real failing score.

    ``prompts`` — the child's continuation prompts; ``None`` keeps the
    rollout's resolved prompts. An ``injected_prompt`` delta (RFC §3.3) binds
    here, so the injection is the child's user-visible first message — never
    silently merged into other prompt content (#908).
    """

    async def _runner(child: RolloutNode) -> float:
        await rollout.connect()
        # Fill the pending branch-child node in place — the continuation Step
        # lands on `child` itself, no content-free placeholder.
        await rollout.execute(prompts, node=child)
        rewards = await rollout.verify()
        await rollout.disconnect()
        if not rewards or rewards.get("reward") is None:
            raise UnscoredChildError(
                f"branch child {child.id} produced no verifier reward "
                f"(verify() returned {rewards!r})" + _verifier_error_suffix(rollout)
            )
        return float(rewards["reward"])

    return _runner


def select_child_runner(
    rollout: Rollout,
    *,
    delta: BranchDelta | None,
    default: ChildRunner,
    run_child: ChildRunner | None,
    fresh_children: bool,
    parent: RolloutNode,
    run_dir: Path | None,
    fresh_runner_factory: Callable[..., ChildRunner] = make_fresh_child_runner,
) -> ChildRunner:
    """Pick the runner one child executes under — the boundary rule, per child.

    Every case was validated by :mod:`benchflow.branch_policy` to not combine
    with an explicit ``run_child``. A child forked from ``env-ready`` runs as
    a fresh Rollout over the just-restored sandbox — that boundary precedes
    ``install_agent()``, so there is no agent in the restored world to connect
    to and no installed skills to keep; the child re-runs installation for
    itself, under the delta's skill_mode when it has one and under the
    parent's own recorded mode when it does not. Everywhere else the agent is
    already installed and the child runs in place, with an injected-prompt
    delta binding the child's continuation prompt into a per-child default
    runner — the formalized version of the caller's per-child-prompt closure.

    ``fresh_runner_factory`` defaults to :func:`make_fresh_child_runner`; the
    orchestrator passes its own module global through so
    ``benchflow.rollout_branch`` remains the patch seam for faking the
    fresh-child path.
    """
    if fresh_children:
        return fresh_runner_factory(
            rollout,
            delta=delta,
            parent=parent,
            branch_stage=FRESH_CHILD_STAGE,
            run_dir=run_dir,
        )
    if run_child is None and delta is not None and delta.injected_prompt is not None:
        return make_default_runner(rollout, prompts=[delta.injected_prompt])
    return default


def _verifier_error_suffix(rollout: Rollout) -> str:
    """`` — <verifier error>`` when the rollout recorded one, else ``''``.

    The verifier's own diagnostic is the difference between "the agent scored
    nothing" and "the score never reached the host", so it is carried into the
    unscored reason verbatim rather than left in the log.
    """
    error = getattr(rollout, "_verifier_error", None)
    return f" — {error}" if error else ""
