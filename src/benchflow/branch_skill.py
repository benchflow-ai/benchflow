"""Executing an ``env-ready`` branch child as a fresh child rollout (RFC §3.3).

Skills are deployed by ``install_agent()``, so a branch taken at the cursor —
after installation — cannot vary them: the container already carries the pack
(or already lacks it), and restoring environment *state* cannot put a
filesystem back. The ``env-ready`` stage snapshot (RFC §3.2) is the seam that
makes the ablation honest: it is taken at the end of ``start()``, **before**
``install_agent()``, so restoring it yields a world with no agent and no skills
in it yet.

That seam cuts both ways, and it is a property of the **stage**, not of the
delta: a world restored to ``env-ready`` has no agent binary, no sandbox user,
no seeded verifier workspace, no path lockdown and no skill pack, because none
of them exist until ``install_agent()`` runs. An *in-place* child forked there
— the default runner's ``connect()`` → ``execute()`` → ``verify()`` on the
parent instance — therefore either dies launching an agent the restore just
deleted, or (when the agent survives in the base image) runs and scores inside
a world missing everything installation deploys, and reports that as an
ordinary child carrying one recorded delta. So **every** engine-run child of
``env-ready`` runs through this module, whatever its delta: with a
``skill_mode`` delta the mode is switched, without one the parent's own
recorded mode is re-installed unchanged.

This module owns the child side of that path. It derives the child's
:class:`~benchflow.rollout.RolloutConfig` from the parent's with the skill mode
set — the existing skill-policy resolution then does the real work
(``no-skill`` strips the task-bundled pack out of the staged build context and
rewrites the Dockerfile ``COPY`` lines, ``with-skill`` mounts it) — and runs
the child as a first-class Rollout over the already-restored sandbox through
the ``use_prebuilt_env`` seam (#388): ``setup`` → ``install_agent`` →
``connect`` → ``execute`` → ``verify`` → ``cleanup``. ``start()`` is
deliberately skipped: the sandbox is up and the environment plane provisioned,
which is exactly what ``env-ready`` means, and most backends' ``start()`` is
not idempotent.

The gates deciding whether such a child may run at all — the stage, the
snapshot layer it requires, the caller-supplied-runner conflict — belong to the
branch engine (:mod:`benchflow.rollout_branch`), which fails closed before
anything is quiesced, restored, or run.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchflow.branch import UnscoredChildError
from benchflow.branch_lineage import branch_child_dir, child_provenance
from benchflow.branch_stage import STAGE_ENV_READY
from benchflow.skill_policy import (
    SKILL_MODE_WITH_SKILL,
    TaskSkillPolicy,
    resolve_task_skill_policy,
)

if TYPE_CHECKING:
    from benchflow.branch_delta import BranchDelta
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.trajectories.tree import RolloutNode

logger = logging.getLogger(__name__)

#: The branch point whose children must run as fresh rollouts: it is the one
#: recorded boundary that precedes ``install_agent()``, so a child restored to
#: it has no agent and no skills and must re-run installation for itself.
#: Also the only branch point a ``skill_mode`` delta can execute from.
FRESH_CHILD_STAGE = STAGE_ENV_READY

#: The checkpoint layer that snapshot must carry. Everything
#: ``install_agent()`` deploys — the agent binary, the sandbox user, the skill
#: pack, the path lockdown — lives in the container filesystem, so an
#: environment-state-only checkpoint cannot roll it back: a child restored
#: without this layer would re-install on top of the parent's install (and a
#: ``no-skill`` child would still find the parent's pack mounted), quietly
#: measuring nothing.
FRESH_CHILD_LAYER = "sandbox"

#: Back-compat aliases. Skills were the original reason this boundary exists,
#: and the skill-specific diagnostics still name it that way.
SKILL_DELTA_STAGE = FRESH_CHILD_STAGE
SKILL_DELTA_LAYER = FRESH_CHILD_LAYER

#: ``delta_execution`` in lineage: the child was run as its own Rollout rather
#: than in place on the parent instance. Absent = the ordinary in-place child.
EXECUTION_FRESH_ROLLOUT = "fresh-rollout"


def fresh_child_skill_mode(config: RolloutConfig, skill_mode: str | None) -> str:
    """The skill mode a fresh ``env-ready`` child installs under.

    A ``skill_mode`` delta names it; every other child re-installs the mode the
    parent itself recorded (``artifact_skill_mode or skill_mode``). Falling
    back to the *recorded* mode rather than the raw field is what keeps a
    non-skill child a genuine zero-delta-on-skills child: a
    ``from_legacy``-built parent carries its effective mode in
    ``artifact_skill_mode``, and re-installing from ``skill_mode`` alone would
    hand the child a different world than the parent ran in and label it "no
    delta".
    """
    return skill_mode if skill_mode is not None else config.recorded_skill_mode


def resolve_child_skill_policy(
    config: RolloutConfig, skill_mode: str
) -> TaskSkillPolicy:
    """Resolve the child's skill policy — the ablation's precondition.

    Called at delta-validation time, before anything is quiesced, so that a
    ``with-skill`` delta against a task shipping no bundled pack (or a runtime
    ``skills_dir`` that has since gone missing) fails closed with the same
    typed error ``setup()`` would raise later — after the branch had already
    restored a snapshot and started running children.
    """
    return resolve_task_skill_policy(
        task_path=config.task_path,
        skill_mode=skill_mode,
        runtime_skills_dir=(
            config.skills_dir if skill_mode == SKILL_MODE_WITH_SKILL else None
        ),
        declared_sandbox_skills_dir=None,
    )


def child_run_location(
    config: RolloutConfig,
    *,
    run_dir: Path | None,
    parent_id: str,
    child_id: str,
) -> tuple[Path, str, str]:
    """Where the child rollout writes its own artifacts, as ``_init_rollout`` args.

    A branch child is a first-class rollout (RFC §3.4), so its run directory
    *is* the per-child artifact directory the lineage writer already owns
    (:func:`~benchflow.branch_lineage.branch_child_dir`): the standard
    ``config.json`` / ``result.json`` / trajectory files land beside the
    branch's own ``provenance.json``. ``_init_rollout`` composes the run
    directory as ``jobs_dir / job_name / rollout_name``, so the target
    directory is split back into exactly those three parts. A parent with no
    run directory (a branch-first rollout that never ran ``setup()``) falls
    back to its configured jobs directory under a node-derived name.
    """
    target = (
        branch_child_dir(Path(run_dir), parent_id, child_id)
        if run_dir is not None
        else Path(config.jobs_dir)
        / (config.job_name or "branches")
        / f"branch-{child_id}"
    )
    return target.parent.parent, target.parent.name, target.name


def child_skill_config(
    config: RolloutConfig,
    *,
    skill_mode: str,
    jobs_dir: Path,
    job_name: str,
    rollout_name: str,
    source_provenance: dict[str, Any] | None = None,
) -> RolloutConfig:
    """The parent's config with the skill mode replaced — nothing else moved.

    Both ``skill_mode`` and ``artifact_skill_mode`` are written: the skill
    policy resolves from ``recorded_skill_mode`` (``artifact_skill_mode or
    skill_mode``), so setting only the former would leave a
    ``from_legacy``-built parent's recorded mode in place and the delta would
    silently do nothing. ``skills_dir`` is a with-skill-only field and is
    dropped for a ``no-skill`` child; ``snapshot_stages`` is cleared because a
    child forked *from* a stage does not re-checkpoint the parent's boundaries
    (its environment plane belongs to the parent).

    The user is re-materialized from ``loop_strategy`` when the parent's was
    strategy-built — ``RolloutConfig`` rejects carrying both, and the strategy
    rebuilds an identical user — and left alone otherwise.
    """
    return dataclasses.replace(
        config,
        skill_mode=skill_mode,
        artifact_skill_mode=skill_mode,
        skills_dir=(config.skills_dir if skill_mode == SKILL_MODE_WITH_SKILL else None),
        snapshot_stages=frozenset(),
        jobs_dir=jobs_dir,
        job_name=job_name,
        rollout_name=rollout_name,
        user=None if config.loop_strategy_spec is not None else config.user,
        source_provenance=source_provenance,
    )


async def run_fresh_child(
    rollout: Rollout,
    config: RolloutConfig,
    *,
    prompts: list[str] | None = None,
) -> float:
    """Run ``config`` as a fresh Rollout over ``rollout``'s restored sandbox.

    The branch engine has already rolled the container and environment back to
    the ``env-ready`` snapshot, so the child adopts that sandbox
    (``use_prebuilt_env``) and re-runs installation from it: this is the call
    that installs the agent binary the restore removed and redeploys — or
    refuses to deploy — the skill pack. ``start()`` is skipped (see the module
    docstring).

    ``cleanup()`` always runs, including on failure: it releases the child's own
    resources (its LiteLLM runtime, its staged task copy) and, because the
    sandbox is caller-owned, leaves the parent's container up for the next
    child. Exceptions propagate — a child that could not run is not a child
    that scored zero.

    The child is a first-class rollout, so its terminal
    :class:`~benchflow.models.RolloutResult` is materialized after cleanup —
    that write is what leaves ``result.json`` / ``timing.json`` / ``prompts.json``
    beside its ``config.json`` (RFC §3.4). Artifact failure is isolated and
    logged, exactly as the branch engine isolates lineage writes: the child's
    reward stands either way.

    A child that *ran* but was never scored raises
    :class:`~benchflow.branch.UnscoredChildError`. That is the same principle
    one step further: a child whose reward does not exist is not a child that
    scored zero either, and the engine records it as unscored.
    """
    from benchflow.rollout import Rollout as _Rollout

    child_rollout = _Rollout(config)
    child_rollout.use_prebuilt_env(rollout.env)
    rewards: dict | None = None
    try:
        await child_rollout.setup()
        await child_rollout.install_agent()
        await child_rollout.connect()
        await child_rollout.execute(prompts)
        rewards = await child_rollout.verify()
    finally:
        await child_rollout.cleanup()

    result = None
    try:
        result = child_rollout.result
    except Exception:
        logger.warning(
            "branch child result artifacts (%s) failed to build — the child's "
            "reward is unaffected",
            config.rollout_name,
            exc_info=True,
        )
    scored = (result.rewards if result is not None else None) or rewards
    if not scored or scored.get("reward") is None:
        # The child ran but was never scored — its verifier crashed, or its
        # output never reached the host. Reporting 0.0 here is how a lost
        # reward became a real-looking observation in a live ablation; the
        # engine records this as an unscored child instead.
        verifier_error = getattr(child_rollout, "_verifier_error", None)
        raise UnscoredChildError(
            f"branch child {config.rollout_name} produced no verifier reward "
            f"(verify() returned {rewards!r})"
            + (f" — {verifier_error}" if verifier_error else "")
        )
    return float(scored["reward"])


def make_fresh_child_runner(
    rollout: Rollout,
    *,
    delta: BranchDelta | None,
    parent: RolloutNode,
    branch_stage: str,
    run_dir: Path | None,
) -> Callable[[RolloutNode], Awaitable[float]]:
    """Build the per-child runner for a child forked from ``env-ready``.

    Every engine-run child of that boundary uses this, whatever its delta —
    the restored world has no agent installed, so there is nothing for an
    in-place child to connect to (see the module docstring). ``delta`` may be
    ``None`` (the zero delta): the child then re-installs the parent's own
    recorded skill mode and runs the rollout's resolved prompts, which is what
    makes it a genuine control arm rather than a differently-provisioned one.

    The child carries the branch provenance as its *own* ``source_provenance``
    — the seam a continued run already uses — so its ``config.json`` and
    ``result.json`` record which rollout it forked from, at which stage, from
    which snapshot refs, and that it ran as a fresh rollout. An
    ``injected_prompt`` on the delta is delivered as the child's continuation
    prompt, exactly as the default runner delivers it.
    """
    skill_mode = fresh_child_skill_mode(
        rollout._config, delta.skill_mode if delta is not None else None
    )
    injected_prompt = delta.injected_prompt if delta is not None else None

    async def _runner(child: RolloutNode) -> float:
        jobs_dir, job_name, rollout_name = child_run_location(
            rollout._config,
            run_dir=run_dir,
            parent_id=parent.id,
            child_id=child.id,
        )
        config = child_skill_config(
            rollout._config,
            skill_mode=skill_mode,
            jobs_dir=jobs_dir,
            job_name=job_name,
            rollout_name=rollout_name,
            source_provenance=child_provenance(
                str(run_dir) if run_dir is not None else str(rollout._config.task_path),
                branch_stage=branch_stage,
                snapshot=parent.state.get("snapshot"),
                delta=child.state.get("delta"),
                delta_execution=EXECUTION_FRESH_ROLLOUT,
            ),
        )
        logger.info(
            "branch child %s runs as a fresh rollout with skill_mode=%s",
            child.id,
            skill_mode,
        )
        return await run_fresh_child(
            rollout,
            config,
            prompts=([injected_prompt] if injected_prompt is not None else None),
        )

    return _runner
