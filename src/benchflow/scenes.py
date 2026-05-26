"""Scene desugaring.

Scenes are authoring sugar only: they declare role/skill attribution for a
sequence of turns, then compile away to Rollout tree ``Step`` objects before
execution. Runtime scheduling, message routing, and scene lifecycles belong
outside this module.
"""

from __future__ import annotations

from pathlib import Path

from benchflow._types import Role, Scene
from benchflow.trajectories.tree import Step

DEFAULT_SCENE_PROMPT = "Solve the task described in /app/instruction.md"


def compile_scenes_to_steps(
    scenes: list[Scene],
    *,
    default_prompt: str | None = None,
) -> list[Step]:
    """Lower declarative scenes into explicit rollout Steps.

    Each turn becomes one ``Step`` with the role, prompt, and scene-local
    skill metadata the rollout executor needs. The compiled list is linear;
    branching remains the job of the tree/branch engine, not Scene.
    """

    fallback_prompt = default_prompt or DEFAULT_SCENE_PROMPT
    steps: list[Step] = []
    for scene_index, scene in enumerate(scenes):
        role_map = {role.name: role for role in scene.roles}
        for turn_index, turn in enumerate(scene.turns):
            role = role_map.get(turn.role)
            if role is None:
                raise ValueError(f"Turn references unknown role {turn.role!r}")
            prompt = turn.prompt if turn.prompt is not None else fallback_prompt
            steps.append(_scene_step(scene, scene_index, turn_index, role, prompt))
    return steps


def _scene_step(
    scene: Scene,
    scene_index: int,
    turn_index: int,
    role: Role,
    prompt: str,
) -> Step:
    return Step(
        id=f"scene-{scene_index}-turn-{turn_index}-{role.name}",
        data={
            "type": "scene_turn",
            "scene": scene.name,
            "scene_index": scene_index,
            "turn_index": turn_index,
            "role": role,
            "prompt": prompt,
            "skills_dir": scene.skills_dir,
            "role_skills_dir": role.skills_dir,
        },
    )


def scene_step_role(step: Step) -> Role:
    """Return the role attached by :func:`compile_scenes_to_steps`."""

    role = step.data.get("role")
    if not isinstance(role, Role):
        raise TypeError(f"Step {step.id!r} is missing Scene role metadata")
    return role


def scene_step_prompt(step: Step) -> str:
    """Return the prompt attached by :func:`compile_scenes_to_steps`."""

    prompt = step.data.get("prompt")
    if not isinstance(prompt, str):
        raise TypeError(f"Step {step.id!r} is missing Scene prompt metadata")
    return prompt


def scene_step_skills_dir(step: Step) -> str | Path | None:
    """Return the scene-local skills root attached to *step*."""

    skills_dir = step.data.get("skills_dir")
    if skills_dir is None or isinstance(skills_dir, (str, Path)):
        return skills_dir
    raise TypeError(f"Step {step.id!r} has invalid Scene skills_dir metadata")


__all__ = [
    "DEFAULT_SCENE_PROMPT",
    "compile_scenes_to_steps",
    "scene_step_prompt",
    "scene_step_role",
    "scene_step_skills_dir",
]
