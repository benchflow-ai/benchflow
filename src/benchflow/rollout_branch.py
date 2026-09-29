"""The Branch -> Rollout engine wiring.

The pure Branch primitives live in :mod:`benchflow.branch` — ``checkpoint``,
``restore``, ``aggregate`` operate on a ``RolloutTree`` node and an
``Environment`` with no I/O beyond the env contract. This module is the
*engine*: it drives those primitives against a live
:class:`~benchflow.rollout.Rollout` — quiescing the agent, running each forked
child as an **isolated sub-rollout**, and restoring the parent's linear state
afterward.

Why a separate module: ``rollout.py`` is the 5-phase lifecycle; the Branch
path is a distinct, optional capability. Keeping it here holds ``rollout.py``
under the size threshold and keeps the branch logic independently testable.

The engine functions are free functions taking a ``Rollout`` as their first
argument — ``Rollout.branch`` is a thin one-line entry point that delegates
here.

Isolation invariant (the architecture's "tree is additive / no-regression"):
after :func:`branch` returns, the parent Rollout's linear state — ``_cursor``,
``_trajectory``, ``_rewards``, ``_phase``, ``_n_tool_calls`` (and the session
bookkeeping) — is *exactly* what it was before. A branch child never
re-entrantly mutates the shared instance: it runs against a scoped snapshot of
that state, captured before and restored after each child, and its real
continuation Steps attach to a *pending* branch-child node so the reward and
value land on the right node.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import inspect
import logging
import math
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from benchflow.branch import (
    _SNAPSHOT_KEY,
    StageSnapshot,
    checkpoint_composed,
    restore_composed,
)
from benchflow.branch import checkpoint as _checkpoint_branch
from benchflow.branch import restore as _restore_branch
from benchflow.branch_artifacts import MountedArtifacts
from benchflow.branch_lineage import (
    ForkRecord,
    NonfiniteChildReward,
    UnscoredChildError,
    child_cost,
    child_status,
    error_record,
    fork_cost,
)
from benchflow.branch_result import (
    RESULT_STATE_FIELDS,
    scope_child_result_state,
    write_child_observation,
)
from benchflow.embodiment import require_world_restore, task_embodiment
from benchflow.models import TrajectorySource
from benchflow.trajectories.tree import RolloutNode

if TYPE_CHECKING:
    from benchflow.rollout import Rollout
    from benchflow.sandbox.protocol import SandboxImage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BranchChild:
    """Which child of a fork a runner call is for.

    ``index`` is the child's position (0-based) in this fork, ``label`` the
    matching ``child_labels`` entry (None when no labels were given), ``node``
    the pending tree node the child's Steps attach to (pass it to
    ``Rollout.execute(..., node=child.node)``), and ``fork_id`` the fork's id
    in ``tree.json``.
    """

    index: int
    label: str | None
    node: RolloutNode
    fork_id: str
    # The rollout to drive for this child: the parent itself for in-place
    # children, the child's own sub-rollout with isolate_children=True.
    rollout: Any = None


# The per-child runner: given the child's branch node, run its continuation and
# return the scalar return. The original form takes the node only.
ChildRunner = Callable[[RolloutNode], Awaitable[float]]


class IdentifiedChildRunner(Protocol):
    """A runner that also wants the child's identity.

    Declare a keyword-only ``child`` parameter (or ``**kwargs``) and the
    engine passes a :class:`BranchChild`; a runner that takes only the node
    is called as before.
    """

    def __call__(
        self, node: RolloutNode, *, child: BranchChild
    ) -> Awaitable[float]: ...


def _accepts_child_identity(runner: Any) -> bool:
    """Whether ``runner`` declares a keyword-only ``child`` or ``**kwargs``.

    A positional parameter that happens to be named ``child`` does not count:
    existing runners (``async def run_child(child): ...``) receive the node.
    """
    try:
        parameters = inspect.signature(runner).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        param.kind is param.VAR_KEYWORD
        or (param.kind is param.KEYWORD_ONLY and param.name == "child")
        for param in parameters
    )


@dataclass
class _LinearState:
    """A scoped snapshot of a Rollout's linear (non-tree) execution state.

    Captured before a branch child runs and restored after — this is what
    makes a branch child an *isolated sub-rollout* rather than a re-entrant
    mutation of the shared Rollout instance.
    """

    cursor: RolloutNode
    trajectory: list[dict]
    n_tool_calls: int
    phase: str
    rewards: dict | None
    trajectory_source: TrajectorySource | None
    partial_trajectory: bool
    session_tool_count: int
    session_traj_count: int
    executed_prompts: list[str]
    result_state: dict[str, Any]

    @classmethod
    def capture(cls, rollout: Rollout) -> _LinearState:
        """Snapshot ``rollout``'s linear state, deep-copying mutable values."""
        return cls(
            cursor=rollout._cursor,
            trajectory=copy.deepcopy(rollout._trajectory),
            n_tool_calls=rollout._n_tool_calls,
            phase=rollout._phase,
            rewards=copy.deepcopy(rollout._rewards),
            trajectory_source=rollout._trajectory_source,
            partial_trajectory=rollout._partial_trajectory,
            session_tool_count=rollout._session_tool_count,
            session_traj_count=rollout._session_traj_count,
            executed_prompts=list(rollout._executed_prompts),
            result_state={
                name: copy.deepcopy(getattr(rollout, name))
                for name in RESULT_STATE_FIELDS
                if hasattr(rollout, name)
            },
        )

    def restore_onto(self, rollout: Rollout) -> None:
        """Write this snapshot back onto ``rollout`` — undoing a child's mutations."""
        rollout._cursor = self.cursor
        rollout._trajectory = copy.deepcopy(self.trajectory)
        rollout._n_tool_calls = self.n_tool_calls
        rollout._phase = self.phase
        rollout._rewards = copy.deepcopy(self.rewards)
        rollout._trajectory_source = self.trajectory_source
        rollout._partial_trajectory = self.partial_trajectory
        rollout._session_tool_count = self.session_tool_count
        rollout._session_traj_count = self.session_traj_count
        rollout._executed_prompts = list(self.executed_prompts)
        for name in RESULT_STATE_FIELDS:
            if name in self.result_state:
                setattr(rollout, name, copy.deepcopy(self.result_state[name]))
            elif hasattr(rollout, name):
                delattr(rollout, name)


async def branch(
    rollout: Rollout,
    n: int,
    run_child: ChildRunner | IdentifiedChildRunner | None = None,
    *,
    require_sandbox_snapshot: bool = False,
    snapshot_layers: frozenset[str] | set[str] | None = None,
    child_labels: list[str | None] | None = None,
    retain_snapshots: bool = False,
    restore_parent: bool = True,
    child_requests: list[str | None] | None = None,
    isolate_children: bool = False,
    concurrency: int = 1,
    resume_session: bool = False,
    child_retries: int = 0,
    continue_after_child_failure: bool = False,
    resume_session_id: str | None = None,
    reuse_snapshot: SandboxImage | None = None,
) -> float:
    """Branch ``rollout`` at its cursor into ``n`` child continuations.

    The Branch lifecycle (``docs/architecture.md``; usage in
    ``docs/composed-checkpoints.md``):

    1. ``quiesce``: disconnect the agent.
    2. ``checkpoint``: capture the requested ``snapshot_layers``, environment
       first (declared database state), then sandbox (the container
       filesystem). Agent credential files are kept out of a sandbox snapshot.
    3. ``run children``: for each child, restore the checkpoint (sandbox
       first) and run the runner as an **isolated sub-rollout**: its own
       scoped linear state and a fresh agent session. Its Steps attach to a
       pending branch node, so the reward lands on the real leaf.
    4. ``score / aggregate``: each child's reward goes to
       ``child.state["reward"]``; their mean is V(parent), stored on
       ``parent.state["value"]`` and returned.
    5. restore the parent's world and linear state, then delete the fork's
       container snapshot unless ``retain_snapshots`` (recorded as
       ``snapshot.retention`` in ``tree.json``).

    ``child_requests`` describes, per child, what the caller's runner does
    differently (for example "own prompt, 212 characters"); it is recorded as
    ``intervention.requested`` with ``execution: "runner"``. At most 200
    characters each; the engine does not act on it.

    ``isolate_children=True`` runs each child as its own sub-rollout in its
    own sandbox created from the snapshot (sandbox layer only), at most
    ``concurrency`` at once; see :func:`_run_isolated_children`. The parent's
    sandbox is not touched by the children, and step 5 is unchanged. The
    runner then drives ``child.rollout`` (a :class:`BranchChild` field), and
    a child can call ``child.rollout.branch(...)`` to fork again from its own
    state; the nested fork is recorded in the same ``tree.json``.

    ``resume_session=True`` hands each child the parent's ACP session id, so
    the child's ``connect()`` resumes the parent's conversation with
    ``session/load`` (the agent must advertise ``loadSession`` and keep its
    session on disk, where the snapshot captures it, as Claude Code does).
    ``tree.json`` then records ``snapshot.agent_session: "resumed"``.

    ``child_retries=N`` retries a child up to N times when it failed before
    its agent did anything (nothing recorded on its node, verifier not run:
    a provider hiccup such as a connect timeout), from the checkpoint again;
    the child record keeps ``attempts`` and ``retried_after``.
    ``continue_after_child_failure=True`` lets in-place siblings run after a
    child failed for good (isolated children always do); the failures are
    raised together at the end and the fork is ``partial``.

    ``restore_parent=False`` skips the world restore in step 5 for a caller
    that finishes after the fork: ``n`` restores instead of ``n + 1``. The
    world then holds the last child's state, so the rollout is marked
    discarded (``parent_restore: "skipped"``) and refuses every later
    setup, connect, execute, verify or branch; only ``finalize()`` and
    ``cleanup()`` remain. The linear state is still restored, so the parent's
    ``result.json`` describes the parent up to the fork.

    The branch point is always the current cursor. ``run_child(node)`` runs
    one child; a runner declaring a keyword-only ``child`` (or ``**kwargs``)
    also receives a :class:`BranchChild`. The default runner
    (:func:`make_default_runner`) connects a fresh agent, runs the task's
    prompts, verifies and disconnects. See :func:`_score_child_outcome` for
    how a runner's return becomes a reward and its ``reward_source``.

    ``snapshot_layers`` defaults to environment-only; a task without an
    environment manifest needs ``{"sandbox"}``. ``require_sandbox_snapshot``
    retains its legacy capability-check-only behavior. No layer captures
    agent-session memory or external/mounted state.

    After this returns, ``rollout``'s linear state is exactly what it was
    before: the tree gained ``n`` children at the cursor, nothing else moved,
    and the agent is disconnected.

    A task whose embodiment cannot be restored by software (every physical
    embodiment) raises :class:`~benchflow.embodiment.PhysicalRestoreRefused`
    before the agent is quiesced or anything is checkpointed.
    """
    require_safe_branch_world(rollout)
    require_world_restore(task_embodiment(rollout._task), "branch")
    # Layer selection adapted from JeremyJC67's PR #1046.
    layers = frozenset({"environment"} if snapshot_layers is None else snapshot_layers)
    unknown = layers - {"environment", "sandbox"}
    if unknown:
        raise ValueError(f"unknown snapshot_layers: {sorted(unknown)!r}")
    if not layers:
        raise ValueError("branch needs at least one layer in snapshot_layers")
    if reuse_snapshot is not None and layers != frozenset({"sandbox"}):
        raise ValueError(
            "reuse_snapshot stands for the sandbox layer only; "
            f"snapshot_layers is {sorted(layers)!r}"
        )
    # An explicit id resumes a session this rollout did not open itself (a
    # trial started from a kept checkpoint that recorded its session).
    parent_session_id = resume_session_id or getattr(
        getattr(rollout, "_session", None), "session_id", None
    )
    if resume_session and not parent_session_id:
        raise ValueError(
            "resume_session=True, but the rollout has no agent session to "
            "resume: branch after at least one prompt (connect and execute)"
        )
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    if child_retries < 0:
        raise ValueError(f"child_retries must be 0 or more, got {child_retries}")
    if concurrency > 1 and not isolate_children:
        raise ValueError(
            "concurrency > 1 needs isolate_children=True: in-place children "
            "share the parent's sandbox and must run one at a time"
        )
    if isolate_children and layers != frozenset({"sandbox"}):
        raise ValueError(
            "isolate_children supports the sandbox layer only: declared "
            "environment state is restored inside the parent's own sandbox "
            f"and cannot be copied to a new one (got {sorted(layers)!r})"
        )
    if "environment" in layers and rollout._environment is None:
        raise RuntimeError(
            "branch() with snapshot_layers={'environment'} (the default) needs the "
            "Environment plane, and this rollout has none: the task declares no "
            "environment manifest, so there is no declared state to snapshot. "
            "For a plain task pass snapshot_layers={'sandbox'}, which checkpoints "
            "the container filesystem (Docker, or Daytona in direct mode); for "
            "declared database state, set RolloutConfig(environment_manifest=...)."
        )
    if child_labels is not None and (
        len(child_labels) != n
        or any(
            label is not None and (not isinstance(label, str) or len(label) > 200)
            for label in child_labels
        )
    ):
        raise ValueError(
            "child_labels must contain one string (at most 200 characters) or null per child"
        )
    if child_requests is not None and (
        len(child_requests) != n
        or any(
            request is not None and (not isinstance(request, str) or len(request) > 200)
            for request in child_requests
        )
    ):
        raise ValueError(
            "child_requests must contain one string (at most 200 characters) or null per child"
        )
    if n < 2:
        raise ValueError(f"a branch forks into >= 2 children, got n={n}")

    if require_sandbox_snapshot or "sandbox" in layers:
        sandbox = rollout._env
        supports = getattr(sandbox, "supports_snapshot", False)
        if not supports:
            sandbox_name = type(sandbox).__name__ if sandbox else "<none>"
            raise RuntimeError(
                f"branch cannot run with sandbox snapshot required: the active "
                f"sandbox {sandbox_name!r} does not implement container-level "
                "snapshot/restore. Use a provider whose Sandbox satisfies the "
                "checkpoint contract (DockerSandbox or DaytonaSandbox in direct "
                "mode), or request only environment state when that captures "
                "the task's mutable world."
            )

    if "sandbox" in layers:
        validate_restore = getattr(
            rollout._environment, "validate_sandbox_restore", None
        )
        if validate_restore is not None:
            validate_restore()

    if rollout._branch_child_active:
        raise RuntimeError(
            "Nested shared-rollout branches are not supported; fork with "
            "isolate_children=True and branch the child's own rollout"
        )
    if rollout._usage_runtime is not None:
        raise RuntimeError(
            "Branching an active provider runtime needs a runtime fork contract; "
            "use a native-subscription or provider-free rollout"
        )

    parent = rollout._cursor
    # Any: the two runner forms are dispatched on pass_identity below.
    runner = cast(
        Any, run_child if run_child is not None else make_default_runner(rollout)
    )
    pass_identity = _accepts_child_identity(runner)

    run_dir = getattr(rollout, "_rollout_dir", None)
    # tree.json and branches/ live in the root trial; a sub-rollout that
    # forks again (isolate_children) records there, not in its own folder.
    lineage_dir = getattr(rollout, "_lineage_dir", None) or run_dir
    if isolate_children and lineage_dir is None:
        raise ValueError("isolate_children needs a rollout directory (call setup())")
    fork_id = uuid.uuid4().hex
    event = ForkRecord(
        rollout, lineage_dir, fork_id, parent, layers, n, child_labels, child_requests
    )
    event.record["children_mode"] = {
        "isolated": isolate_children,
        "concurrency": concurrency,
        # Isolated children prepared ahead while others run (sandboxes alive
        # stay within 2 x concurrency); 0 for in-place children.
        "prewarm": concurrency if isolate_children else 0,
        "child_retries": child_retries,
        "continue_after_child_failure": continue_after_child_failure
        or isolate_children,
    }
    if resume_session:
        event.record["snapshot"]["agent_session"] = "resumed"
        event.record["snapshot"]["excluded"].remove("agent_session")
    resume_id = parent_session_id if resume_session else None
    try:
        event.persist()
    except BaseException as exc:
        event.record["status"] = "failed"
        event.record["artifact_error"] = error_record(exc)
        raise

    # checkpoint — snapshot the env at the parent; the roll-back point.
    composed = layers != frozenset({"environment"})
    snap_env = rollout._environment if "environment" in layers else None
    snap_sandbox = rollout._env if "sandbox" in layers else None
    loop = asyncio.get_running_loop()
    capture_started = loop.time()
    try:
        await rollout.disconnect()
        if reuse_snapshot is not None:
            # The sandbox's state at the cursor is already an image (an
            # automatic checkpoint, or the kept checkpoint the trial started
            # from): use it. Its credential files go with it for restores.
            adopt = getattr(snap_sandbox, "adopt_snapshot", None)
            if adopt is not None:
                await adopt(reuse_snapshot)
            parent.state[_SNAPSHOT_KEY] = StageSnapshot(
                environment_ref=None, sandbox_ref=reuse_snapshot
            )
            event.record["snapshot"]["reused"] = True
        elif composed:
            await checkpoint_composed(
                parent, environment=snap_env, sandbox=snap_sandbox
            )
        else:
            await _checkpoint_branch(parent, rollout._environment)
    except BaseException as exc:
        event.record["status"] = (
            "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        )
        event.record["error"] = error_record(exc)
        try:
            event.persist()
        except BaseException as artifact_error:
            raise _combined_failure(
                [exc, artifact_error], "Checkpoint and lineage publication failed"
            ) from None
        exc.add_note(f"branch checkpoint requested snapshot_layers={sorted(layers)!r}")
        raise
    event.timing["checkpoint"] = round(loop.time() - capture_started, 3)
    event.captured(parent.state["snapshot"])
    captured = parent.state["snapshot"]
    sandbox_image = getattr(captured, "sandbox_ref", None)

    async def restore_world() -> None:
        if composed:
            prepare = getattr(rollout._environment, "prepare_sandbox_restore", None)
            if prepare is not None:
                prepare()
            await restore_composed(parent, environment=snap_env, sandbox=snap_sandbox)
            resume = getattr(rollout._environment, "resume_after_sandbox_restore", None)
            if resume is not None:
                await resume()
        else:
            await _restore_branch(parent, rollout._environment)

    # The parent's linear state, captured once. Each child runs against a fresh
    # restore of this; the parent is restored to it at the end.
    saved = _LinearState.capture(rollout)

    parent_rollout = saved.result_state.get("_rollout_name") or (
        run_dir.name if run_dir is not None else None
    )
    mounted_paths = getattr(rollout, "_rollout_paths", None)
    fork_dir = lineage_dir / "branches" / fork_id if lineage_dir is not None else None
    holder = None
    custody_quiesced = True
    failures: list[BaseException] = []
    children_started: float | None = None
    try:
        event.persist()
        if fork_dir is not None:
            fork_dir.mkdir(parents=True, exist_ok=False)
            if mounted_paths is not None and not isolate_children:
                holder = MountedArtifacts.hold(mounted_paths, fork_dir)
        children_started = loop.time()
        sibling_failures: list[BaseException] = []
        if isolate_children:
            assert fork_dir is not None and sandbox_image is not None
            await _run_isolated_children(
                rollout,
                parent,
                sandbox_image,
                event,
                fork_dir,
                run_child,
                labels=child_labels,
                concurrency=concurrency,
                resume_session_id=resume_id,
                child_retries=child_retries,
            )
        for index in range(0 if isolate_children else n):
            child = rollout._tree.attach(parent)
            observation = event.record["children"][index]
            observation["node_id"] = child.id
            child_dir = (
                fork_dir / "children" / child.id if fork_dir is not None else None
            )
            restore_started = child_started = loop.time()
            await restore_world()
            event.timing["child_restore"].append(
                round(loop.time() - restore_started, 3)
            )
            saved.restore_onto(rollout)
            rollout._cursor = child
            scope_child_result_state(rollout)
            rollout._branch_child_active = True
            if child_dir is not None:
                child_dir.mkdir(parents=True, exist_ok=False)
                # Host writers follow this path; sandbox bind mounts stay attached
                # to their original root, whose entries are separately in custody.
                rollout._rollout_dir = child_dir

            child_failures: list[BaseException] = []
            observation["status"] = "running"
            event.persist()
            verify_calls = rollout._verify_calls
            rollout._resume_session_id = resume_id
            try:
                attempt = 1
                while True:
                    try:
                        if pass_identity:
                            identity = BranchChild(
                                index=index,
                                label=child_labels[index] if child_labels else None,
                                node=child,
                                fork_id=fork_id,
                                rollout=rollout,
                            )
                            outcome = await runner(child, child=identity)
                        else:
                            outcome = await runner(child)
                        break
                    except Exception as exc:
                        if attempt > child_retries or not _retry_before_work(
                            exc, child, rollout._verify_calls != verify_calls
                        ):
                            raise
                        # Failed before the agent did anything (a provider
                        # hiccup): run the child once more from the checkpoint.
                        attempt += 1
                        observation["retried_after"] = error_record(exc)
                        logger.warning(
                            "Branch child %s failed before doing anything (%s: %s); "
                            "retrying it once from the checkpoint",
                            child.id,
                            type(exc).__name__,
                            exc,
                        )
                        await _bounded_child_cleanup(
                            rollout.disconnect(), rollout=rollout
                        )
                        await restore_world()
                        saved.restore_onto(rollout)
                        rollout._cursor = child
                        scope_child_result_state(rollout)
                        rollout._branch_child_active = True
                        if child_dir is not None:
                            rollout._rollout_dir = child_dir
                    finally:
                        observation["attempts"] = attempt
                ret, source = _score_child_outcome(
                    rollout,
                    outcome,
                    default_runner=run_child is None,
                    verified=rollout._verify_calls != verify_calls,
                )
                child.state["reward"] = ret
                observation.update(status="scored", reward=ret, reward_source=source)
            except BaseException as exc:
                child_failures.append(exc)
                observation.update(status=child_status(exc), error=error_record(exc))
            finally:
                rollout._resume_session_id = None
                try:
                    await _bounded_child_cleanup(rollout.disconnect(), rollout=rollout)
                except BaseException as exc:
                    custody_quiesced = False
                    observation["cleanup_error"] = error_record(exc)
                    child_failures.append(exc)
                runtime = rollout._usage_runtime
                if runtime is not None:
                    try:
                        await _bounded_child_cleanup(
                            rollout._planes.stop_provider_runtime(runtime),
                            rollout=rollout,
                        )
                    except BaseException as exc:
                        custody_quiesced = False
                        observation["cleanup_error"] = error_record(exc)
                        child_failures.append(exc)
                    finally:
                        rollout._usage_runtime = None
                if rollout._branch_cleanup_unquiesced:
                    custody_quiesced = False
                    observation["cleanup_error"] = {
                        "type": "TimeoutError",
                        "code": "unresolved_cleanup",
                    }
                if child_dir is not None:
                    evidence_ok = custody_quiesced
                    try:
                        write_child_observation(
                            rollout,
                            child,
                            child_dir,
                            error=child_failures[0] if child_failures else None,
                            trajectory_start=len(saved.trajectory),
                            lineage={
                                "parent_rollout": parent_rollout,
                                "fork_id": fork_id,
                                "parent_node": parent.id,
                                "index": index,
                                "label": child_labels[index] if child_labels else None,
                            },
                        )
                    except BaseException as exc:
                        evidence_ok = False
                        event.record["artifact_error"] = error_record(exc)
                        child_failures.append(exc)
                    if holder is not None and custody_quiesced:
                        try:
                            holder.hand_off(child_dir)
                        except BaseException as exc:
                            evidence_ok = False
                            event.record["artifact_error"] = error_record(exc)
                            child_failures.append(exc)
                    if evidence_ok:
                        event.artifacts(index, child_dir)
                        child.state["artifact_dir"] = str(
                            child_dir.relative_to(lineage_dir)
                        )
                observation.update(
                    child_cost(
                        rollout._native_usage_metrics,
                        rollout._timing,
                        # In place: the child held the parent's sandbox.
                        sandbox_seconds=loop.time() - child_started,
                    )
                )
                try:
                    event.persist()
                except BaseException as exc:
                    event.record["artifact_error"] = error_record(exc)
                    child_failures.append(exc)
                rollout._rollout_dir = run_dir
                saved.restore_onto(rollout)
            if child_failures:
                if (
                    continue_after_child_failure
                    and custody_quiesced
                    and not any(
                        isinstance(exc, asyncio.CancelledError)
                        for exc in child_failures
                    )
                ):
                    # The world is restored before the next child, so one
                    # failed child does not stop its siblings.
                    sibling_failures.extend(child_failures)
                    continue
                raise _combined_failure(
                    child_failures, "Branch child and evidence cleanup failed"
                )
        if sibling_failures:
            raise _combined_failure(sibling_failures, "Branch children failed")
    except BaseException as exc:
        failures.append(exc)
        event.record["error"] = error_record(exc)
        for observation in event.record["children"]:
            if observation["status"] == "running":
                observation.update(status=child_status(exc), error=error_record(exc))
    finally:
        if children_started is not None:
            event.timing["children"] = round(loop.time() - children_started, 3)
        rollout._rollout_dir = run_dir
        try:
            if not restore_parent:
                # The caller finishes after the fork: the world keeps the last
                # child's state and the rollout refuses to continue.
                event.record["parent_restore"] = "skipped"
                rollout._branch_parent_discarded = True
            elif custody_quiesced:
                restore_started = loop.time()
                await restore_world()
                event.timing["parent_restore"] = round(loop.time() - restore_started, 3)
                event.record["parent_restore"] = "restored"
            else:
                event.record["parent_restore"] = "deferred"
                event.record["parent_restore_error"] = {
                    "type": "RuntimeError",
                    "code": "child_not_quiesced",
                }
        except BaseException as exc:
            event.record["parent_restore"] = "failed"
            event.record["parent_restore_error"] = error_record(exc)
            failures.append(exc)
        finally:
            if holder is not None and custody_quiesced:
                try:
                    holder.release()
                except BaseException as exc:
                    event.record["artifact_error"] = error_record(exc)
                    failures.append(exc)
            if holder is not None and not custody_quiesced:
                event.record["artifact_error"] = {
                    "type": "ArtifactCustodyError",
                    "code": "child_not_quiesced",
                }
                failures.append(
                    RuntimeError(
                        f"Child did not quiesce; parent evidence remains at {holder.fork_dir / 'parent'}"
                    )
                )
            saved.restore_onto(rollout)
            if not custody_quiesced or event.record["parent_restore"] == "failed":
                rollout._branch_world_unsafe = True
            if not custody_quiesced:
                rollout._branch_cleanup_unquiesced = True
            if sandbox_image is not None:
                interrupted = await _release_sandbox_snapshot(
                    snap_sandbox,
                    sandbox_image,
                    event.record,
                    # A reused image belongs to whoever made it.
                    retain=retain_snapshots or reuse_snapshot is not None,
                )
                if interrupted is not None:
                    failures.append(interrupted)
    event.record["cost"] = fork_cost(
        event.record,
        wall_seconds=loop.time() - capture_started,
        isolated=isolate_children,
    )
    statuses = [child["status"] for child in event.record["children"]]
    if failures:
        event.record["status"] = (
            "cancelled"
            if "cancelled" in statuses
            or any(isinstance(exc, asyncio.CancelledError) for exc in failures)
            else "partial"
            if any(status != "not_started" for status in statuses)
            else "failed"
        )
    else:
        event.record["status"] = "completed"
        event.record["value"] = math.fsum(
            child["reward"] / n for child in event.record["children"]
        )
        event.record["value_stderr"] = _stderr(
            [child["reward"] for child in event.record["children"]]
        )
    try:
        event.persist()
    except BaseException as exc:
        event.record["artifact_error"] = error_record(exc)
        event.record["status"] = "partial"
        event.record["value"] = None
        failures.append(exc)
    if failures:
        raise _combined_failure(
            failures, "Branch failed and parent restoration also failed"
        )

    # aggregate — per-child return -> V(parent).
    value = event.record["value"]
    assert value is not None
    parent.state["value"] = value
    rollout._phase = "branched"
    return value


async def restore_sandbox_with_services(rollout: Any, image: Any) -> None:
    """Replace ``rollout``'s sandbox with ``image`` and restart its services.

    A sandbox restore replaces the container, so the environment plane's
    framework-started services (running processes, never part of a snapshot)
    stop with it. The in-place branch restore already brackets the restore
    with ``prepare_sandbox_restore`` / ``resume_after_sandbox_restore``; a
    sandbox that ``start()`` provisioned and that is then restored from a
    snapshot (an isolated child without ``start_from_snapshot``, a trial
    started ``--from-checkpoint``) needs the same.
    """
    environment = getattr(rollout, "_environment", None)
    prepare = getattr(environment, "prepare_sandbox_restore", None)
    if prepare is not None:
        prepare()
    await rollout._env.restore(image)
    resume = getattr(environment, "resume_after_sandbox_restore", None)
    if resume is not None:
        await resume()


def _make_sub_rollout(
    rollout: Rollout,
    fork_dir: Any,
    node: RolloutNode,
    resume_session_id: str | None = None,
) -> Any:
    """A child's own rollout: same task and agent, its folder at
    ``branches/<fork>/children/<node>``, steps in the root rollout's tree."""
    config = dataclasses.replace(
        rollout._config,
        jobs_dir=fork_dir,
        job_name="children",
        rollout_name=node.id,
    )
    sub = type(rollout)(config)
    sub._tree = rollout._tree
    sub._cursor = node
    sub._resume_session_id = resume_session_id
    # The snapshot was taken after the parent installed its agent.
    sub._installed_agent_cfg = getattr(rollout, "_agent_cfg", None)
    sub._from_branch_snapshot = True
    root_dir = getattr(rollout, "_lineage_dir", None) or rollout._rollout_dir
    sub._lineage_dir = root_dir
    sub._lineage_forks = getattr(rollout, "_lineage_forks", None) or getattr(
        rollout, "_branch_forks", None
    )
    return sub


async def _run_isolated_children(
    rollout: Rollout,
    parent: RolloutNode,
    image: Any,
    event: ForkRecord,
    fork_dir: Any,
    run_child: Any,
    *,
    labels: list[str | None] | None,
    concurrency: int,
    resume_session_id: str | None = None,
    child_retries: int = 0,
) -> None:
    """Run every child as its own sub-rollout, at most ``concurrency`` at once.

    Each child: ``setup`` (its folder), ``start`` straight from the snapshot
    when the sandbox offers ``start_from_snapshot`` (Daytona), else start from
    the task image and ``restore`` the snapshot into it (Docker); then
    ``install_agent`` (credential files never enter a snapshot), the runner,
    ``observation.json`` and ``finalize`` (its own result.json; the child's
    sandbox is deleted). Children are independent: one failing does not stop
    the others. The failures are raised together once all have finished.
    """
    loop = asyncio.get_running_loop()
    n = len(event.record["children"])
    nodes = [rollout._tree.attach(parent) for _ in range(n)]
    for index, node in enumerate(nodes):
        event.record["children"][index]["node_id"] = node.id
    event.timing["child_restore"] = [None] * n
    gate = asyncio.Semaphore(concurrency)
    prewarm = asyncio.Semaphore(concurrency)
    parent_name = getattr(rollout, "_rollout_name", None) or (
        rollout._rollout_dir.name if rollout._rollout_dir is not None else None
    )

    async def one(index: int, node: RolloutNode) -> None:
        observation = event.record["children"][index]
        # Pre-warm: up to `concurrency` children prepare (sandbox from the
        # snapshot, agent setup) while up to `concurrency` others run.
        await prewarm.acquire()
        prewarm_held = True
        try:
            for attempt in range(1, child_retries + 2):
                observation["status"] = "running"
                event.persist()
                sub_started = loop.time()
                sub = _make_sub_rollout(rollout, fork_dir, node, resume_session_id)
                # Isolated-only overhead phases, added to the child's timing_sec.
                phases: dict[str, float] = {}
                failure: BaseException | None = None
                running = False
                try:
                    await sub.setup()
                    started = loop.time()
                    start_from = getattr(sub._env, "start_from_snapshot", None)
                    fast = bool(start_from is not None and start_from(image))
                    await sub.start()
                    if not fast:
                        await restore_sandbox_with_services(sub, image)
                    event.timing["child_restore"][index] = round(
                        loop.time() - started, 3
                    )
                    phases["sandbox_from_snapshot"] = event.timing["child_restore"][
                        index
                    ]
                    started = loop.time()
                    await sub.install_agent()
                    phases["install_agent"] = round(loop.time() - started, 3)
                    await gate.acquire()
                    running = True
                    if prewarm_held:
                        prewarm.release()
                        prewarm_held = False
                    runner = cast(
                        Any,
                        make_default_runner(sub) if run_child is None else run_child,
                    )
                    identity = BranchChild(
                        index=index,
                        label=labels[index] if labels else None,
                        node=node,
                        fork_id=event.record["id"],
                        rollout=sub,
                    )
                    if run_child is None or not _accepts_child_identity(runner):
                        outcome = await runner(node)
                    else:
                        outcome = await runner(node, child=identity)
                    ret, source = _score_child_outcome(
                        sub,
                        outcome,
                        default_runner=run_child is None,
                        verified=getattr(sub, "_verify_calls", 0) > 0,
                    )
                    node.state["reward"] = ret
                    observation.update(
                        status="scored", reward=ret, reward_source=source
                    )
                except BaseException as exc:
                    failure = exc
                    observation.update(
                        status=child_status(exc), error=error_record(exc)
                    )
                finally:
                    child_dir = getattr(sub, "_rollout_dir", None)
                    evidence_ok = child_dir is not None
                    try:
                        await _bounded_child_cleanup(sub.disconnect(), rollout=sub)
                    except BaseException as exc:
                        observation["cleanup_error"] = error_record(exc)
                    if child_dir is not None:
                        try:
                            write_child_observation(
                                sub,
                                node,
                                child_dir,
                                error=failure,
                                trajectory_start=0,
                                lineage={
                                    "parent_rollout": parent_name,
                                    "fork_id": event.record["id"],
                                    "parent_node": parent.id,
                                    "index": index,
                                    "label": labels[index] if labels else None,
                                },
                            )
                        except BaseException as exc:
                            evidence_ok = False
                            event.record["artifact_error"] = error_record(exc)
                    observation.update(
                        child_cost(
                            getattr(sub, "_native_usage_metrics", None),
                            getattr(sub, "_timing", None),
                        )
                    )
                    started = loop.time()
                    try:
                        if child_dir is not None:
                            await sub.finalize()
                        else:
                            await sub.cleanup()
                    except BaseException as exc:
                        observation["cleanup_error"] = error_record(exc)
                        failure = failure or exc
                    phases["finalize"] = round(loop.time() - started, 3)
                    # The child's own sandbox, from setup to teardown.
                    observation["cost"]["sandbox_seconds"] = round(
                        loop.time() - sub_started, 3
                    )
                    observation["timing_sec"].update(phases)
                    # What the child's sandbox reused from the snapshot.
                    observation["snapshot_start"] = getattr(
                        sub, "_snapshot_start", None
                    )
                    if evidence_ok and child_dir is not None:
                        event.artifacts(index, child_dir)
                        node.state["artifact_dir"] = observation["artifacts"]["path"]
                    event.persist()
                    if running:
                        gate.release()
                observation["attempts"] = attempt
                if (
                    failure is not None
                    and attempt <= child_retries
                    and _retry_before_work(
                        failure, node, getattr(sub, "_verify_calls", 0) > 0
                    )
                ):
                    # Failed before the agent did anything: keep that
                    # attempt's folder aside and run the child once more in a
                    # new sandbox from the snapshot.
                    observation["retried_after"] = error_record(failure)
                    observation.update(status="running", error=None, cleanup_error=None)
                    logger.warning(
                        "Branch child %s failed before doing anything (%s: %s); "
                        "retrying it once in a new sandbox",
                        node.id,
                        type(failure).__name__,
                        failure,
                    )
                    if child_dir is not None and child_dir.exists():
                        child_dir.rename(
                            child_dir.with_name(f"{child_dir.name}.attempt-{attempt}")
                        )
                    continue
                break
            if failure is not None:
                raise failure
        finally:
            if prewarm_held:
                prewarm.release()

    results = await asyncio.gather(
        *(one(index, node) for index, node in enumerate(nodes)),
        return_exceptions=True,
    )
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise _combined_failure(failures, "Isolated branch children failed")


def _stderr(rewards: list[float]) -> float | None:
    """Standard error of the mean reward (sample std / sqrt(n)); None below 2."""
    n = len(rewards)
    if n < 2:
        return None
    mean = math.fsum(rewards) / n
    variance = math.fsum((r - mean) ** 2 for r in rewards) / (n - 1)
    return round(math.sqrt(variance / n), 6)


def _retry_before_work(exc: BaseException, node: RolloutNode, verified: bool) -> bool:
    """A child is retried once when it failed before its agent did anything:
    nothing recorded on its node and the verifier not run (a provider hiccup
    such as a connect timeout). Never an unscored verdict or a cancellation."""
    return (
        isinstance(exc, Exception)
        and not isinstance(exc, UnscoredChildError | NonfiniteChildReward)
        and node.step_in is None
        and not verified
    )


def _score_child_outcome(
    rollout: Rollout, outcome: Any, *, default_runner: bool, verified: bool
) -> tuple[float, str]:
    """Turn a runner's return into (reward, reward_source), or refuse it.

    A missing reward is unscored, never zero, for custom runners too: when
    the child's runner called ``verify()`` (``verified``: the rollout's verify
    counter moved during the child) and the verifier produced no canonical
    reward, the child is unscored whatever number the runner returned. A
    number equal to the canonical verifier reward is recorded as
    ``verifier``; any other number as ``runner_return``.
    """
    if outcome is None:
        raise UnscoredChildError("Branch child is unscored: the runner returned None")
    ret = float(outcome)
    if not math.isfinite(ret):
        raise NonfiniteChildReward("Branch child returned a non-finite reward")
    if default_runner:
        return ret, "verifier"
    if not verified:
        return ret, "runner_return"
    rewards = rollout._rewards
    canonical = rewards.get("reward") if isinstance(rewards, dict) else None
    if canonical is None:
        raise UnscoredChildError(
            "Branch child is unscored: its verifier returned no canonical reward "
            f"({rollout._verifier_error or 'no reward key'}); the runner's "
            f"{ret!r} is not a score"
        )
    return ret, "verifier" if float(canonical) == ret else "runner_return"


async def _release_sandbox_snapshot(
    sandbox: Any, image: Any, record: dict[str, Any], *, retain: bool
) -> asyncio.CancelledError | None:
    """Delete the fork's container snapshot unless the caller keeps it.

    Runs once the parent world is restored (or its restore was given up), on
    every exit path: completed, partial, failed and cancelled forks alike. A
    provider snapshot holds the full container filesystem at rest, so leaving
    it behind is a leak, not a cache. ``delete_snapshot`` returns False when
    the provider still uses the snapshot (Docker: the restored parent container
    runs from the image); the sandbox then deletes it when it stops.

    The deletion is shielded, so a second cancellation cannot abandon it
    half-way; that cancellation is returned for the caller to raise after the
    record is written. Any other failure is logged and recorded, never raised
    over the fork's own outcome.
    """
    target = record["snapshot"]
    if retain:
        target["retention"] = "kept"
        return None
    delete = getattr(sandbox, "delete_snapshot", None)
    try:
        if delete is None:
            raise NotImplementedError(
                f"{type(sandbox).__name__} cannot delete snapshots"
            )
        deleted_now = await asyncio.shield(asyncio.ensure_future(delete(image)))
    except BaseException as exc:
        target["retention"] = "delete_failed"
        target["delete_error"] = error_record(exc)
        logger.warning(
            "Branch snapshot %s was not deleted (%s: %s); remove it by hand or "
            "with `bench sandbox cleanup`",
            image.ref,
            type(exc).__name__,
            exc,
        )
        return exc if isinstance(exc, asyncio.CancelledError) else None
    target["retention"] = "deleted" if deleted_now is not False else "deferred"
    target["restore_available"] = False
    return None


def _combined_failure(failures: list[BaseException], message: str) -> BaseException:
    """Choose what to raise for ``failures``, never hiding a cancellation.

    asyncio marks a task cancelled only when it ends with ``CancelledError``;
    a group containing one reads as an ordinary crash. The other failures are
    logged and attached to the cancellation as notes instead.
    """
    cancellation = next(
        (exc for exc in failures if isinstance(exc, asyncio.CancelledError)), None
    )
    if cancellation is None:
        if len(failures) == 1:
            return failures[0]
        return BaseExceptionGroup(message, failures)
    for other in failures:
        if other is not cancellation:
            logger.warning("%s during cancellation", message, exc_info=other)
            cancellation.add_note(f"{message}: {type(other).__name__}: {other}")
    return cancellation


def make_default_runner(rollout: Rollout) -> ChildRunner:
    """Build the default per-child runner bound to ``rollout``.

    The default runner re-runs the child from the parent's env checkpoint. Its
    ``connect()`` starts a fresh agent session, or resumes the parent's
    conversation when :func:`branch` set ``resume_session=True``; the agent
    process itself is never snapshotted. Each child connects a fresh agent and disconnects it at the end, so
    no two children's agents overlap (the next child connects only after the
    previous one disconnected). A missing canonical reward fails explicitly;
    it is never invented as zero.
    """

    async def _runner(child: RolloutNode) -> float:
        child_error: BaseException | None = None
        try:
            await rollout.connect()
            # Fill the pending node with its real continuation Step.
            await rollout.execute(node=child)
            rewards = await rollout.verify()
        except BaseException as exc:
            child_error = exc
            raise
        finally:
            # Quiesce the child before the outer engine restores the world,
            # including when connection only partially succeeded.
            try:
                await _bounded_child_cleanup(rollout.disconnect(), rollout=rollout)
            except BaseException as disconnect_error:
                if child_error is not None:
                    raise _combined_failure(
                        [child_error, disconnect_error],
                        "Branch child failed and agent disconnect also failed",
                    ) from None
                raise
        if not rewards or "reward" not in rewards:
            raise UnscoredChildError(
                "Branch child is unscored: "
                + (rollout._verifier_error or "verifier returned no canonical reward")
            )
        return float(rewards["reward"])

    return _runner


_CHILD_CLEANUP_TIMEOUT = 30.0
_CHILD_CANCEL_GRACE = 5.0


async def _bounded_child_cleanup(
    operation: Awaitable[Any], *, rollout: Rollout
) -> None:
    """Bound even a cleanup coroutine that ignores cancellation."""
    # Deferred: benchflow.rollout imports this module.
    from benchflow.rollout._deadline import _swallow_abandoned_outcome

    task = asyncio.ensure_future(operation)
    try:
        done, _ = await asyncio.wait({task}, timeout=_CHILD_CLEANUP_TIMEOUT)
        if task in done:
            await task
            return
        raise TimeoutError("Branch child cleanup timed out")
    finally:
        if not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout=_CHILD_CANCEL_GRACE)
        if task.done():
            _swallow_abandoned_outcome(task)
        else:
            rollout._branch_cleanup_unquiesced = True
            task.add_done_callback(_swallow_abandoned_outcome)


def require_safe_branch_world(rollout: Any) -> None:
    """Do not continue a shared world whose branch rollback was not established."""
    if getattr(rollout, "_branch_parent_discarded", False):
        raise RuntimeError(
            "This rollout was branched with restore_parent=False: its world "
            "holds the last child's state, not the parent's, so it cannot "
            "continue. Call finalize() or cleanup()."
        )
    if getattr(rollout, "_branch_world_unsafe", False) or getattr(
        rollout, "_branch_cleanup_unquiesced", False
    ):
        raise RuntimeError(
            "Branch world is unsafe: child cleanup or parent restoration did not complete; dispose of this rollout and recreate it"
        )
