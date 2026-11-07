"""Step- and user-driven execution drivers for :class:`benchflow.rollout.Rollout`.

These are the orchestration loops that step the live ``Rollout`` instance
through scene steps and the user-driven progressive-disclosure loop, plus the
generated-skill export hook. They are free functions taking the ``Rollout`` as
their first argument — mirroring the ``rollout_branch.py`` engine convention —
so the lifecycle file stays under its size threshold while the driver logic
stays independently testable.

``Rollout`` keeps thin one-line methods (``_run_steps``, ``_run_user_loop``,
``_activate_step_skills``, ``_export_generated_skills``) that delegate here, so
instance-level patching (``monkeypatch.setattr(rollout, "_run_steps", ...)``)
and unbound calls (``Rollout._export_generated_skills(rollout)``) keep working
exactly as before.

Note: the two driver loops in :func:`_run_user_loop` — the scene-step loop and
the free-round loop — are *deliberately* kept distinct. They differ in prompt
source (scene prompt vs. plain instruction), role selection (per-step role vs.
sticky last role), scene metadata, and the ``use_scene_prompts`` branch, so
they are not provably equivalent and are not collapsed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchflow._types import Role
from benchflow.checkpoints import after_prompt
from benchflow.contracts import RoundResult
from benchflow.loop_strategies import LoopStrategyUser
from benchflow.rollout._results import (
    _compose_scene_user_prompt,
    _is_document_user,
    _user_handoff_kind,
)
from benchflow.rollout._setup import _ensure_sandbox_dir
from benchflow.rollout._skills import _safe_skill_name
from benchflow.scenes import (
    compile_scenes_to_steps,
    scene_step_prompt,
    scene_step_role,
    scene_step_skills_dir,
)
from benchflow.skill_policy import SKILL_MODE_SELF_GEN
from benchflow.trajectories.tree import Step
from benchflow.trust.state import IngressKind, admit
from benchflow.usage_tracking import is_token_usage_available

if TYPE_CHECKING:
    from benchflow._utils.reachable_write import ReachablePath
    from benchflow.rollout import Rollout

logger = logging.getLogger(__name__)


def _round_tokens(rollout: Rollout) -> int | None:
    """Cumulative native-ACP tokens spent so far, or ``None`` if untrusted.

    ``_native_usage_metrics`` is initialized to a zeroed ``usage_unavailable()``
    payload (``usage_source="unavailable"``, ``total_tokens=0``), so a bare read
    can't tell "spent 0 tokens" apart from "no usage captured". Gating on the
    trusted-source check keeps uninstrumented paths reporting ``None`` rather
    than a spurious ``0`` — preserving the cost-curve's best-effort contract.
    """
    metrics = rollout._native_usage_metrics
    if is_token_usage_available(metrics):
        return metrics.get("total_tokens")
    return None


def _host_solution(task: Any) -> str | None:
    """The task's own solve.sh as text, read on the host, or None."""
    path = getattr(getattr(task, "paths", None), "solve_path", None)
    try:
        text = path.read_text(encoding="utf-8") if path is not None else ""
    except (OSError, UnicodeDecodeError):
        return None
    return text.strip() or None


def skill_export_root(target: Path) -> ReachablePath:
    """Hold the user-selected skill export folder's parent as a destination root."""
    from benchflow._utils.reachable_write import pin_destination

    absolute = Path(os.path.abspath(target))
    return pin_destination(absolute.parent) / absolute.name


async def _export_generated_skills(rollout: Rollout) -> None:
    """Download creator-produced skills before sandbox cleanup.

    Also captures the exported skill packs into ``self._evolved_skills``
    — the ``name -> body`` dict a continual-learning Job commits to its
    persistent LearnerStore (capability 5).

    Retries transient download failures up to 3 times.
    """
    export_target = rollout._config.export_generated_skills_to
    if export_target is None:
        return
    # TASKMD-R14 B-M2: the export's own root is held before the agent ran
    # (Rollout setup); a direct caller pins it here. Sandboxes still receive
    # the plain path (their download contract); private_download selects the
    # held root from it, and the read-back goes through the capability.
    held = getattr(rollout, "_skill_export_root", None) or skill_export_root(
        Path(export_target)
    )
    held.mkdir(parents=True, exist_ok=True)
    target = Path(os.path.abspath(export_target))

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            await rollout._env.download_dir(
                rollout._config.generated_skills_root, target
            )
            break
        except Exception as e:
            last_err = e
            if attempt < 2:
                delay = 2 ** (attempt + 1)
                logger.warning(
                    f"Skill export attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
    else:
        raise RuntimeError(
            f"Skill export failed after 3 attempts: {last_err}"
        ) from last_err

    from benchflow.learner_skills import capture_skills

    rollout._evolved_skills = capture_skills(held)
    if (
        rollout._config.recorded_skill_mode == SKILL_MODE_SELF_GEN
        and not rollout._evolved_skills
    ):
        raise RuntimeError(
            "self-gen creator produced no generated skills; aborting empty "
            "self-gen result"
        )


async def _activate_step_skills(rollout: Rollout, step: Step) -> None:
    """Activate scene-local skills attached by the Scene desugaring pass."""
    skills_dir = scene_step_skills_dir(step)
    if not skills_dir:
        return
    if rollout._env is None:
        raise RuntimeError("Environment is not started")

    scene_name = str(step.data.get("scene") or "scene")
    role = scene_step_role(step)
    source = str(skills_dir)
    local_source = Path(source).expanduser()
    expected_skill_names: tuple[str, ...] = ()
    # Upload whenever skills_dir resolves to a real directory on the
    # orchestrator host, regardless of whether it arrived as a str or a
    # PathLike. The SkillsBench entrypoint passes an absolute *str* host
    # path (EvaluationConfig.skills_dir is typed str), so gating the upload
    # on isinstance(..., os.PathLike) silently skipped it: the path then
    # fell through to the "already inside the sandbox" branch below and
    # _link_skill_paths produced a dangling symlink, so deployed task skills
    # never reached the agent. An absolute path that does NOT exist on the
    # host is a sandbox path produced by an earlier scene and is linked
    # as-is.
    if local_source.is_dir():
        expected_skill_names = tuple(
            sorted(path.parent.name for path in local_source.glob("*/SKILL.md"))
        )
        remote_source = f"/skills/{_safe_skill_name(scene_name)}"
        admission = admit(rollout._env, "main", IngressKind.USER_CONTEXT)
        await _ensure_sandbox_dir(rollout._env, Path(remote_source).parent)
        await rollout._env.upload_dir(local_source, remote_source, admission=admission)
    elif source.startswith("/"):
        remote_source = source
    else:
        raise FileNotFoundError(f"Scene skills_dir not found: {skills_dir}")

    home = (
        f"/home/{rollout._config.sandbox_user}"
        if rollout._config.sandbox_user
        else "/root"
    )
    agent_cfg = rollout._planes.agent_config(role.agent)
    if not agent_cfg or not agent_cfg.skill_paths:
        return
    await rollout._planes.link_skill_paths(
        rollout._env,
        remote_source,
        agent_cfg.skill_paths,
        home,
        rollout._agent_cwd,
        rollout._config.sandbox_user,
        expected_skill_names=expected_skill_names,
    )


async def _run_steps(rollout: Rollout, steps: list[Step]) -> None:
    """Execute already-compiled rollout Steps in declaration order."""
    current_role_key: tuple[Any, ...] | None = None
    try:
        for number, step in enumerate(steps, start=1):
            role = scene_step_role(step)
            role_key = (
                step.data.get("scene_index"),
                role.name,
                role.agent,
                role.model,
                role.reasoning_effort,
                role.timeout_sec,
                role.idle_timeout_sec,
                tuple(sorted(role.env.items())),
            )
            logger.info(
                "[Step] %s scene=%s role=%s",
                step.id,
                step.data.get("scene"),
                role.name,
            )
            await rollout._activate_step_skills(step)
            if current_role_key != role_key:
                if current_role_key is not None:
                    await rollout.disconnect()
                await rollout.connect_as(role)
                current_role_key = role_key
            await rollout.execute(prompts=[scene_step_prompt(step)])
            # Opt-in retained snapshot (RolloutConfig.checkpoints).
            await after_prompt(rollout, number)
    finally:
        if current_role_key is not None:
            await rollout.disconnect()


# Per-iteration view of a round-log entry, persisted to loop/iterations.jsonl
# for loop-strategy runs.
_ITERATION_RECORD_KEYS = (
    "round",
    "rewards",
    "verifier_error",
    "feedback_level",
    "wall_sec",
    "tokens",
)


_ROUND_OUTPUT_MAX_BYTES = 1024 * 1024


def _soft_check_stdout(rollout: Rollout) -> str | None:
    """The last MB of test.sh's output from the soft check that just ran
    (``verifier/test-stdout.txt``, which the next check overwrites).

    Code the tests ran could have left a link or a FIFO there: only a regular
    file is read, through the trial folder's capability, as the verifier's
    own scan reads it (rollout/_setup.py)."""
    import stat

    from benchflow._utils.reachable_write import destination

    path = getattr(getattr(rollout, "_rollout_paths", None), "test_stdout_path", None)
    if path is None or rollout._rollout_dir is None:
        return None
    try:
        # The trial folder is BenchFlow's own; below it nothing is followed.
        with destination(path, root=rollout._rollout_dir).open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode):
                return None
            stream.seek(max(0, opened.st_size - _ROUND_OUTPUT_MAX_BYTES))
            data = stream.read(_ROUND_OUTPUT_MAX_BYTES)
    except (OSError, ValueError):
        return None
    return data.decode("utf-8", errors="replace") or None


def _keep_round_output(
    rollout: Rollout, round_num: int, output: str | None, error: str | None
) -> str | None:
    """Write a round's soft-check output beside the trial and return its path
    relative to the trial folder: test.sh's output (or, without it, the
    verifier's logs) and any error. The next check overwrites them, and the
    round log keeps only the reward."""
    if rollout._rollout_dir is None:
        return None
    stdout = _soft_check_stdout(rollout)
    text = "\n".join(part.rstrip("\n") for part in (error, stdout or output) if part)
    if not text:
        return None
    relative = Path("rounds") / f"round-{round_num}-check.txt"
    try:
        path = rollout._rollout_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    except OSError as exc:
        logger.warning(f"Could not keep round {round_num}'s check output: {exc}")
        return None
    return relative.as_posix()


def _persist_round_logs(
    rollout: Rollout, rounds_log: list[dict], *, loop_active: bool
) -> None:
    """Write user_rounds.jsonl (and loop/iterations.jsonl for strategy runs).

    Called from the user-loop ``finally`` so rounds completed before a
    mid-loop crash (agent timeout, ACP error, isolation failure) still land
    on disk — the round log is also what the result.json ``loop`` block is
    derived from at build time.
    """
    if not rounds_log or rollout._rollout_dir is None:
        return
    log_path = rollout._rollout_dir / "user_rounds.jsonl"
    with log_path.open("w") as f:
        for entry in rounds_log:
            f.write(json.dumps(entry) + "\n")
    # A multi-step task's steps.json says the same per step (task/steps.py).
    steps_user = getattr(rollout, "_active_steps_user", None)
    (logger.debug if callable(steps_user) and steps_user() else logger.info)(
        f"[User] {len(rounds_log)} rounds → {log_path}"
    )
    if not loop_active:
        return
    loop_dir = rollout._rollout_dir / "loop"
    loop_dir.mkdir(parents=True, exist_ok=True)
    iterations_path = loop_dir / "iterations.jsonl"
    with iterations_path.open("w") as f:
        for entry in rounds_log:
            f.write(
                json.dumps({key: entry.get(key) for key in _ITERATION_RECORD_KEYS})
                + "\n"
            )
    logger.info(f"[Loop] {len(rounds_log)} iterations → {iterations_path}")


async def _run_user_loop(rollout: Rollout) -> None:
    """Execute a user-driven progressive-disclosure loop.

    Each round: user.run() → connect → agent.execute() → disconnect →
    soft_verify() → build RoundResult → repeat. Stops when user.run()
    returns None or max_user_rounds is reached.
    """
    cfg = rollout._config
    user = cfg.user
    assert user is not None

    scenes = cfg.effective_scenes
    allow_team_handoff = (
        _is_document_user(user) and _user_handoff_kind(user) == "sequential-shared"
    )
    for scene in scenes:
        if len(scene.roles) != 1:
            if not allow_team_handoff:
                raise ValueError(
                    "User-driven loops require each scene to have exactly one "
                    f"role. Scene {scene.name!r} has {len(scene.roles)} roles."
                )
            if not scene.turns:
                raise ValueError(
                    "Sequential team handoff user loops require explicit turns "
                    f"for multi-role scene {scene.name!r}."
                )
            continue
        scene_role = scene.roles[0].name
        if any(turn.role != scene_role for turn in scene.turns):
            raise ValueError(
                "User-driven loops require every turn in a scene to use "
                f"that scene's single role. Scene {scene.name!r} uses role "
                f"{scene_role!r}."
            )

    steps = compile_scenes_to_steps(
        scenes,
        default_prompt=(
            rollout._resolved_prompts[0] if rollout._resolved_prompts else None
        ),
    )
    if not steps:
        raise ValueError(
            "User-driven loops require at least one single-role scene turn."
        )
    if len(steps) > cfg.max_user_rounds:
        raise ValueError(
            "User-driven loops require max_user_rounds to cover every "
            f"scene turn. Got {len(steps)} turns and "
            f"max_user_rounds={cfg.max_user_rounds}."
        )
    installed_confirmation_handler = rollout._install_document_confirmation_handler(
        user
    )

    # The engine's authoritative per-round log. Each round is appended as
    # soon as its soft verify completes — before anything that can raise —
    # and the list is aliased onto the rollout so _build_result() can derive
    # the result.json ``loop`` block from whatever rounds finished, on every
    # path including a mid-loop timeout or ACP error.
    rounds_log: list[dict] = []
    rollout._user_rounds_log = rounds_log

    try:
        instruction = (
            rollout._resolved_prompts[0]
            if rollout._resolved_prompts
            else ("Solve the task described in /app/instruction.md")
        )

        # Oracle access: the user reads the task's solve.sh on the host. The
        # sandbox never holds the solution while an agent runs, under any
        # name (/oracle_backup included): an agent run as root could read it
        # (GAP-ORACLE-READABLE).
        solution: str | None = None
        if cfg.oracle_access:
            solution = _host_solution(rollout._task)

        await user.setup(instruction, solution)

        round_result: RoundResult | None = None
        last_role = scene_step_role(steps[0])
        use_scene_prompts = _is_document_user(user)

        async def run_round(
            *,
            round_num: int,
            role: Role,
            prompt: str,
            scene_name: str | None,
            handoff_from: str | None = None,
        ) -> RoundResult:
            round_started = time.monotonic()
            # A multi-step task's round is one of its steps: that step's
            # tests score it, under its own agent timeout (task/steps.py).
            task_for_round = getattr(user, "task_for_round", None)
            step_view = task_for_round(round_num) if callable(task_for_round) else None
            shown = f"{prompt[:80]!r}..." if len(prompt) > 80 else repr(prompt)
            step_label = getattr(step_view, "step_label", None)
            logger.info(
                f"{step_label[0].upper()}{step_label[1:]}: prompt={shown}"
                if step_label
                else f"[User] round {round_num}: prompt={shown}"
            )
            saved_timeout = rollout._timeout
            if step_view is not None:
                rollout._verify_task = step_view
                if step_view.config.agent.timeout_sec:
                    rollout._timeout = int(step_view.config.agent.timeout_sec)

            # Fresh ACP session each round — agent starts clean but sees
            # its previous workspace changes in the shared sandbox.
            traj_before = len(rollout._trajectory)
            try:
                await rollout.connect_as(role)
                await rollout.execute(prompts=[prompt])
            finally:
                rollout._timeout = saved_timeout
                await rollout.disconnect()

            round_trajectory = rollout._trajectory[traj_before:]
            round_tools = sum(
                1
                for e in round_trajectory
                if isinstance(e, dict) and e.get("type") == "tool_call"
            )

            # Soft verify: run tests after agent disconnected but before
            # next round. Verifiers never read the solution, so nothing is
            # restored for them while the agent's processes are alive.
            rewards, verifier_output, verifier_error = await rollout.soft_verify()
            check_output = _keep_round_output(
                rollout, round_num, verifier_output, verifier_error
            )

            handoff_from_role = (
                handoff_from if handoff_from and handoff_from != role.name else None
            )
            handoff_to_role = role.name if handoff_from_role else None

            # Logged BEFORE the isolation re-lock below: everything after
            # this point can raise (sandbox exec, the next round's agent),
            # and a completed round must survive into user_rounds.jsonl and
            # the result.json loop block regardless.
            entry: dict[str, Any] = {
                "round": round_num,
                "scene": scene_name,
                "role": role.name,
                "handoff_from": handoff_from_role,
                "handoff_to": handoff_to_role,
                "prompt": prompt,
                "rewards": rewards,
                "verifier_error": verifier_error,
                "check_output": check_output,
                "n_tool_calls": round_tools,
                "n_trajectory_events": len(round_trajectory),
                "wall_sec": round(time.monotonic() - round_started, 1),
                # Cumulative native-ACP tokens spent through this round — the
                # cost-curve x-axis. None (NOT 0) when no trusted usage was
                # captured: _native_usage_metrics defaults to a zeroed
                # usage_unavailable() payload, so a bare total_tokens read would
                # report a spurious 0 on uninstrumented paths (e.g. a
                # LiteLLM-proxied run that never surfaces native ACP usage).
                "tokens": _round_tokens(rollout),
            }
            if isinstance(user, LoopStrategyUser):
                entry["feedback_level"] = user.feedback_level.value
            rounds_log.append(entry)

            # Mid-loop verifier isolation. soft_verify() uploads the verifier
            # tests after the install-time lockdown ran and leaves its raw
            # stdout under the world-writable /logs/verifier — both readable
            # by the agent on the next round, bypassing the feedback filter.
            # Re-lock and clear now that verifier_output is captured, before
            # the next connect_as. Both are no-ops without a sandbox user
            # (setup() already warned loudly about that configuration).
            #
            # BEST-EFFORT: mid-loop isolation is defense-in-depth for the NEXT
            # round's feedback. The final verify() ALWAYS runs harden_before_verify
            # and is the only result that scores, so a mid-loop isolation failure
            # (e.g. a single-container task whose verifier service is no longer
            # running mid-loop — "service main is not running") must NOT crash the
            # whole run. Warn loudly and continue; the worst case is the
            # pre-existing, less-isolated next-round feedback, never a mis-scored
            # result.
            try:
                if rollout._effective_locked:
                    await rollout._planes.lockdown_paths(
                        rollout._env, rollout._effective_locked
                    )
                if cfg.sandbox_user:
                    await rollout._planes.clear_verifier_output_dir(
                        rollout._env,
                        "User loop isolation failed: clearing verifier output directory",
                        user="root",
                        timeout_sec=10,
                    )
            except Exception as iso_err:
                logger.warning(
                    "Mid-loop verifier isolation skipped at round %d: %s. "
                    "Final verify() still hardens before scoring; next-round "
                    "feedback may be less isolated.",
                    round_num,
                    iso_err,
                )

            result = RoundResult(
                round=round_num,
                trajectory=round_trajectory,
                rewards=rewards,
                verifier_output=verifier_output,
                verifier_error=verifier_error,
                n_tool_calls=round_tools,
                scene=scene_name,
                role=role.name,
                handoff_from=handoff_from_role,
                handoff_to=handoff_to_role,
            )

            # A step's line comes from StepsUser.record, with its reward.
            (logger.debug if step_label else logger.info)(
                f"[User] round {round_num} done: rewards={rewards}, tools={round_tools}"
            )
            return result

        round_num = 0
        # Tracks whether the scene-step loop terminated because the user
        # stopped (returned None) or raised. When set, the free-round loop
        # below must NOT run: re-calling user.run() would resurrect a
        # stopped user or retry one that already errored, while self._error
        # stays set — producing a half-script rollout reported as errored.
        loop_terminated = False
        for step in steps:
            scene_prompt = scene_step_prompt(step)
            try:
                user_prompt = await user.run(
                    round_num,
                    scene_prompt,
                    round_result,
                )
            except Exception as e:
                rollout._error = f"user.run() failed at round {round_num}: {e}"
                logger.error(rollout._error, exc_info=True)
                loop_terminated = True
                break

            if use_scene_prompts:
                prompt = _compose_scene_user_prompt(scene_prompt, user_prompt)
            else:
                if user_prompt is None:
                    logger.info(f"[User] stopped at round {round_num}")
                    loop_terminated = True
                    break
                prompt = user_prompt
            next_role = scene_step_role(step)
            round_result = await run_round(
                round_num=round_num,
                role=next_role,
                prompt=prompt,
                scene_name=str(step.data.get("scene") or "") or None,
                handoff_from=round_result.role if round_result else None,
            )
            last_role = next_role
            round_num += 1

        while not loop_terminated and round_num < cfg.max_user_rounds:
            try:
                prompt = await user.run(round_num, instruction, round_result)
            except Exception as e:
                rollout._error = f"user.run() failed at round {round_num}: {e}"
                logger.error(rollout._error, exc_info=True)
                break

            if prompt is None:
                (
                    logger.debug
                    if getattr(user, "task_for_round", None)
                    else logger.info
                )(f"[User] stopped at round {round_num}")
                break

            round_result = await run_round(
                round_num=round_num,
                role=last_role,
                prompt=prompt,
                scene_name=None,
                handoff_from=round_result.role if round_result else None,
            )
            round_num += 1

    finally:
        _persist_round_logs(
            rollout, rounds_log, loop_active=cfg.loop_strategy_spec is not None
        )
        if installed_confirmation_handler:
            rollout.on_ask_user(None)
