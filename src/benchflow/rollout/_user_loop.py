"""Step- and user-driven execution drivers for :class:`benchflow.rollout.Rollout`.

These are the orchestration loops that step the live ``Rollout`` instance
through scene steps and the user-driven progressive-disclosure loop, plus the
generated-skill export hook. They are free functions taking the ``Rollout`` as
their first argument — mirroring the ``rollout_branch.py`` engine convention —
so the lifecycle file stays under its size threshold while the driver logic
stays independently testable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchflow._types import Role
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
from benchflow.trajectories.multiagent import RealAgentTraceRecorder
from benchflow.trajectories.tree import Step
from benchflow.usage_tracking import is_token_usage_available

if TYPE_CHECKING:
    from benchflow.rollout import Rollout

logger = logging.getLogger(__name__)


def _round_tokens(rollout: Rollout) -> int | None:
    """Cumulative native-ACP tokens spent so far, or ``None`` if untrusted."""

    metrics = getattr(rollout, "_native_usage_metrics", None)
    if metrics and is_token_usage_available(metrics):
        return metrics.get("total_tokens")
    return None


async def _export_generated_skills(rollout: Rollout) -> None:
    """Download creator-produced skills before sandbox cleanup."""

    export_target = rollout._config.export_generated_skills_to
    if export_target is None:
        return
    target = Path(export_target)
    target.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            await rollout._env.download_dir(
                rollout._config.generated_skills_root,
                target,
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

    rollout._evolved_skills = capture_skills(target)
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
    if local_source.is_dir():
        remote_source = f"/skills/{_safe_skill_name(scene_name)}"
        await _ensure_sandbox_dir(rollout._env, Path(remote_source).parent)
        await rollout._env.upload_dir(local_source, remote_source)
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
    )


async def _run_steps(rollout: Rollout, steps: list[Step]) -> None:
    """Execute already-compiled rollout Steps in declaration order.

    Every concrete role connection is recorded as a real agent session when the
    rollout directory is available. The existing merged ACP trajectory is left
    untouched; this adds per-session source views and relationship artifacts.
    """

    recorder = RealAgentTraceRecorder.for_rollout(rollout)
    current_role_key: tuple[Any, ...] | None = None
    current_session_id: str | None = None
    current_session_start = 0
    current_session_error: str | None = None

    def start_recording_session(step: Step, role: Role) -> None:
        nonlocal current_session_id, current_session_start
        if recorder is None:
            return
        driver = (
            "session-factory"
            if getattr(rollout, "_is_session_factory", False)
            else "acp"
        )
        current_session_id = recorder.start_session(
            agent_id=role.name,
            agent_type=role.agent,
            model=role.model,
            driver=driver,
            scene=str(step.data.get("scene") or "") or None,
            scene_index=step.data.get("scene_index"),
            turn_index=step.data.get("turn_index"),
        )
        current_session_start = len(getattr(rollout, "_trajectory", []) or [])

    def finish_recording_session(error: str | None = None) -> None:
        nonlocal current_session_id, current_session_start, current_session_error
        if recorder is None or current_session_id is None:
            current_session_error = None
            return
        trajectory = (getattr(rollout, "_trajectory", []) or [])[current_session_start:]
        recorder.finish_session(current_session_id, trajectory, error=error)
        current_session_id = None
        current_session_error = None

    try:
        for step in steps:
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
                    try:
                        await rollout.disconnect()
                    finally:
                        finish_recording_session(current_session_error)
                    current_role_key = None
                await rollout.connect_as(role)
                current_role_key = role_key
                current_session_error = None
                start_recording_session(step, role)
            try:
                await rollout.execute(prompts=[scene_step_prompt(step)])
            except Exception as exc:
                current_session_error = str(exc)
                raise
    finally:
        if current_role_key is not None:
            try:
                await rollout.disconnect()
            finally:
                finish_recording_session(current_session_error)


_ITERATION_RECORD_KEYS = (
    "round",
    "rewards",
    "verifier_error",
    "feedback_level",
    "wall_sec",
    "tokens",
)


def _persist_round_logs(
    rollout: Rollout, rounds_log: list[dict], *, loop_active: bool
) -> None:
    """Write user_rounds.jsonl and loop/iterations.jsonl when relevant."""

    if not rounds_log or rollout._rollout_dir is None:
        return
    log_path = rollout._rollout_dir / "user_rounds.jsonl"
    with log_path.open("w") as f:
        for entry in rounds_log:
            f.write(json.dumps(entry) + "\n")
    logger.info(f"[User] {len(rounds_log)} rounds → {log_path}")
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
    """Execute a user-driven progressive-disclosure loop."""

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
    rounds_log: list[dict] = []
    rollout._user_rounds_log = rounds_log
    recorder = RealAgentTraceRecorder.for_rollout(rollout)

    try:
        instruction = (
            rollout._resolved_prompts[0]
            if rollout._resolved_prompts
            else ("Solve the task described in /app/instruction.md")
        )

        solution: str | None = None
        if cfg.oracle_access:
            cat = await rollout._env.exec(
                "cat /oracle/solve.sh 2>/dev/null || cat /solution/solve.sh 2>/dev/null || true",
                user="root",
                timeout_sec=10,
            )
            solution = (cat.stdout or "").strip() or None

        await user.setup(instruction, solution)

        if cfg.oracle_access:
            await rollout._env.exec(
                "mv /oracle /oracle_backup 2>/dev/null || true; "
                "mv /solution /solution_oracle_backup 2>/dev/null || true",
                user="root",
                timeout_sec=10,
            )

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
            logger.info(
                f"[User] round {round_num}: prompt={prompt[:80]!r}..."
                if len(prompt) > 80
                else f"[User] round {round_num}: prompt={prompt!r}"
            )

            traj_before = len(rollout._trajectory)
            session_id: str | None = None
            session_error: str | None = None
            try:
                await rollout.connect_as(role)
                if recorder is not None:
                    driver = (
                        "session-factory"
                        if getattr(rollout, "_is_session_factory", False)
                        else "acp"
                    )
                    session_id = recorder.start_session(
                        agent_id=role.name,
                        agent_type=role.agent,
                        model=role.model,
                        driver=driver,
                        scene=scene_name,
                        scene_index=None,
                        turn_index=round_num,
                    )
                    traj_before = len(rollout._trajectory)
                try:
                    await rollout.execute(prompts=[prompt])
                except Exception as exc:
                    session_error = str(exc)
                    raise
            finally:
                try:
                    await rollout.disconnect()
                finally:
                    if recorder is not None and session_id is not None:
                        recorder.finish_session(
                            session_id,
                            rollout._trajectory[traj_before:],
                            error=session_error,
                        )

            round_trajectory = rollout._trajectory[traj_before:]
            round_tools = sum(
                1
                for e in round_trajectory
                if isinstance(e, dict) and e.get("type") == "tool_call"
            )

            if cfg.oracle_access:
                await rollout._env.exec(
                    "mv /oracle_backup /oracle 2>/dev/null || true; "
                    "mv /solution_oracle_backup /solution 2>/dev/null || true",
                    user="root",
                    timeout_sec=10,
                )
            try:
                rewards, verifier_output, verifier_error = await rollout.soft_verify()
            finally:
                if cfg.oracle_access:
                    await rollout._env.exec(
                        "mv /oracle /oracle_backup 2>/dev/null || true; "
                        "mv /solution /solution_oracle_backup 2>/dev/null || true",
                        user="root",
                        timeout_sec=10,
                    )

            handoff_from_role = (
                handoff_from if handoff_from and handoff_from != role.name else None
            )
            handoff_to_role = role.name if handoff_from_role else None
            entry: dict[str, Any] = {
                "round": round_num,
                "scene": scene_name,
                "role": role.name,
                "handoff_from": handoff_from_role,
                "handoff_to": handoff_to_role,
                "prompt": prompt,
                "rewards": rewards,
                "verifier_error": verifier_error,
                "n_tool_calls": round_tools,
                "n_trajectory_events": len(round_trajectory),
                "wall_sec": round(time.monotonic() - round_started, 1),
                "tokens": _round_tokens(rollout),
            }
            if isinstance(user, LoopStrategyUser):
                entry["feedback_level"] = user.feedback_level.value
            rounds_log.append(entry)

            try:
                if rollout._effective_locked:
                    await rollout._planes.lockdown_paths(
                        rollout._env,
                        rollout._effective_locked,
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
            logger.info(
                f"[User] round {round_num} done: rewards={rewards}, tools={round_tools}"
            )
            return result

        round_num = 0
        loop_terminated = False
        for step in steps:
            scene_prompt = scene_step_prompt(step)
            try:
                user_prompt = await user.run(round_num, scene_prompt, round_result)
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
                logger.info(f"[User] stopped at round {round_num}")
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
