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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchflow._types import Role
from benchflow.contracts import RoundResult
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
from benchflow.trajectories.tree import Step

if TYPE_CHECKING:
    from benchflow.rollout import Rollout

logger = logging.getLogger(__name__)


async def _export_generated_skills(rollout: Rollout) -> None:
    """Download creator-produced skills before sandbox cleanup.

    Also captures the exported skill packs into ``self._evolved_skills``
    — the ``name -> body`` dict a continual-learning Job commits to its
    persistent LearnerStore (capability 5).

    Retries transient download failures up to 3 times (guards ENG-147).
    """
    export_target = rollout._config.export_generated_skills_to
    if export_target is None:
        return
    target = Path(export_target)
    target.mkdir(parents=True, exist_ok=True)

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

    rollout._evolved_skills = capture_skills(target)


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
    """Execute already-compiled rollout Steps in declaration order."""
    current_role_key: tuple[Any, ...] | None = None
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
                    await rollout.disconnect()
                await rollout.connect_as(role)
                current_role_key = role_key
            await rollout.execute(prompts=[scene_step_prompt(step)])
    finally:
        if current_role_key is not None:
            await rollout.disconnect()


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

    try:
        instruction = (
            rollout._resolved_prompts[0]
            if rollout._resolved_prompts
            else ("Solve the task described in /app/instruction.md")
        )

        # Oracle access: read /solution before the agent runs, then remove it
        solution: str | None = None
        if cfg.oracle_access:
            cat = await rollout._env.exec(
                "cat /oracle/solve.sh 2>/dev/null || cat /solution/solve.sh 2>/dev/null || true",
                user="root",
                timeout_sec=10,
            )
            solution = (cat.stdout or "").strip() or None

        await user.setup(instruction, solution)

        # Hide oracle files from agent — move rather than delete so the
        # final verify() can still access them if the verifier needs them.
        if cfg.oracle_access:
            await rollout._env.exec(
                "mv /oracle /oracle_backup 2>/dev/null || true; "
                "mv /solution /solution_oracle_backup 2>/dev/null || true",
                user="root",
                timeout_sec=10,
            )

        round_result: RoundResult | None = None
        rounds_log: list[dict] = []
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
            nonlocal rounds_log

            logger.info(
                f"[User] round {round_num}: prompt={prompt[:80]!r}..."
                if len(prompt) > 80
                else f"[User] round {round_num}: prompt={prompt!r}"
            )

            # Fresh ACP session each round — agent starts clean but sees
            # its previous workspace changes in the shared sandbox.
            traj_before = len(rollout._trajectory)
            try:
                await rollout.connect_as(role)
                await rollout.execute(prompts=[prompt])
            finally:
                await rollout.disconnect()

            round_trajectory = rollout._trajectory[traj_before:]
            round_tools = sum(
                1
                for e in round_trajectory
                if isinstance(e, dict) and e.get("type") == "tool_call"
            )

            # Soft verify: run tests after agent disconnected but before
            # next round. Temporarily restore /solution so the verifier can
            # access it, then re-hide before the next agent round.
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

            rounds_log.append(
                {
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
                }
            )

            logger.info(
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

        # Persist round log
        if rounds_log and rollout._rollout_dir:
            log_path = rollout._rollout_dir / "user_rounds.jsonl"
            with log_path.open("w") as f:
                for entry in rounds_log:
                    f.write(json.dumps(entry) + "\n")
            logger.info(f"[User] {len(rounds_log)} rounds → {log_path}")
    finally:
        if installed_confirmation_handler:
            rollout.on_ask_user(None)
