"""Built-in skill controllers (``impl: builtin``), driven through the backend's action interface.

A built-in skill needs ``ee_position(arm)`` from the backend. It commands either the backend's
``ee_command(arm, delta, grip)`` (when the backend has one) or the action groups named in the skill's ``binds``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .spec import Skill

if TYPE_CHECKING:
    from .server import EpisodeServer

REACH_GAIN = 10.0  # proportional gain: group units per metre of error
REACH_LIMIT = 1.0  # per-component clip of the servo command, in group units
STALL_WINDOW = 8
STALL_EPS = 0.002


def _arm_name(skill: Skill) -> str | None:
    g = skill.binds.get("arm")
    return g.split(".", 1)[0] if g else None


def _servo_action(
    ep: EpisodeServer, skill: Skill, delta: list[float], grip: float | None
) -> list[float]:
    cmd = getattr(ep.backend, "ee_command", None)
    if callable(cmd):
        return [float(x) for x in cmd(_arm_name(skill), delta, grip)]
    groups: dict[str, Any] = {}
    if "arm" in skill.binds:
        groups[skill.binds["arm"]] = delta
    if grip is not None and "gripper" in skill.binds:
        groups[skill.binds["gripper"]] = [grip]
    return ep.pack(groups)


def reach(ep: EpisodeServer, skill: Skill, kw: dict) -> dict:
    """Proportional servo of the end effector to (x, y, z)."""
    pos_fn = getattr(ep.backend, "ee_position", None)
    if not callable(pos_fn):
        return {
            "ok": False,
            "error": f"{skill.name} needs end-effector feedback, which this simulator does not provide",
        }
    arm = _arm_name(skill)
    target = [float(kw["x"]), float(kw["y"]), float(kw["z"])]
    grip = kw.get("grip")
    grip = None if grip is None else max(-1.0, min(1.0, float(grip)))
    tol = float(kw.get("tol", 0.01))
    max_steps = int(kw.get("max_steps", 100))
    n, stalled = 0, False
    recent: list[list[float]] = []
    for _ in range(max(1, min(ep.config.max_skill_steps, max_steps))):
        if ep.steps >= ep.config.max_steps or ep.finished:
            break
        hp = [float(x) for x in pos_fn(arm)]
        d = [t - h for t, h in zip(target, hp, strict=True)]
        if _norm(d) < tol:
            break
        recent.append(hp)
        if (
            len(recent) > STALL_WINDOW
            and _norm(
                [
                    a - b
                    for a, b in zip(recent[-1], recent[-1 - STALL_WINDOW], strict=True)
                ]
            )
            < STALL_EPS
        ):
            stalled = True  # the end effector stopped moving: the target is out of reach or blocked
            break
        delta = [max(-REACH_LIMIT, min(REACH_LIMIT, x * REACH_GAIN)) for x in d]
        ep.env_step(_servo_action(ep, skill, delta, grip), op=f"skill:{skill.name}")
        n += 1
    dist = _norm([t - float(h) for t, h in zip(target, pos_fn(arm), strict=True)])
    res: dict[str, Any] = {
        "reached": dist < tol * 1.5,
        "distance": round(dist, 4),
        "executed_steps": n,
        "state": ep.state(),
        "steps_used": ep.steps,
    }
    if stalled:
        res["stalled"] = (
            "the hand stopped moving before reaching the target (out of reach or blocked)"
        )
    return {"ok": True, "result": res}


def grip(ep: EpisodeServer, skill: Skill, kw: dict) -> dict:
    """Hold the arm and command the gripper for a number of steps."""
    value = max(-1.0, min(1.0, float(kw["value"])))
    steps = int(kw.get("steps", 15))
    n = 0
    for _ in range(max(1, min(ep.config.max_repeat, steps))):
        if ep.steps >= ep.config.max_steps or ep.finished:
            break
        ep.env_step(
            _servo_action(ep, skill, [0.0, 0.0, 0.0], value), op=f"skill:{skill.name}"
        )
        n += 1
    return {
        "ok": True,
        "result": {"executed_steps": n, "state": ep.state(), "steps_used": ep.steps},
    }


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


# by the last part of the skill name: `arm.move_to`, `gripper.set`, and `arm.grasp` (a hand's grasp preset,
# commanded through the backend's ee_command)
BUILTINS: dict[str, Callable[[EpisodeServer, Skill, dict], dict]] = {
    "move_to": reach,
    "set": grip,
    "grasp": grip,
}


def builtin_for(skill: Skill) -> Callable[[EpisodeServer, Skill, dict], dict] | None:
    return BUILTINS.get(skill.name.rsplit(".", 1)[-1])
