"""Reference (oracle) policy for the Metal simulator BenchFlow tasks.

Contract (see the task prompt): ``act(observation) -> 7 joint targets in degrees``. The
observation dict carries ``joint_pos`` (deg), ``eef_pos`` and ``target_pos`` (m), ``goal``
("reach" or "pick"), ``grasped``, ``step`` and ``labels``. This oracle uses the package's
forward kinematics and damped-least-squares IK; a policy may also be written from scratch.
"""

from __future__ import annotations

import numpy as np
from inspect_robots_metal_sim.config import MetalSimConfig
from inspect_robots_metal_sim.kinematics import MetalKinematics

_CFG = MetalSimConfig()
_KIN = MetalKinematics(tip_offset_m=_CFG.tip_offset_m)
_LOW, _HIGH = _CFG.low, _CFG.high
_MAX_STEP = np.asarray(_CFG.max_step_deg)
_GRIP = 6
_OPEN, _CLOSE = 100.0, 20.0
_ABOVE, _LIFT = 0.06, 0.14
_state = {"phase": "approach", "hold": 0}


def reset(scene: dict) -> None:
    _state.update(phase="approach", hold=0)


def act(obs: dict) -> list[float]:
    q = np.asarray(obs["joint_pos"], dtype=float)
    tip = np.asarray(obs["eef_pos"], dtype=float)
    cube = np.asarray(obs["target_pos"], dtype=float)
    above = cube + np.array([0.0, 0.0, _ABOVE])
    grip = float(q[_GRIP])
    goal = obs.get("goal") or ("pick" if "pick" in (obs.get("instruction") or "").lower() else "reach")

    if goal == "reach":
        if _state["phase"] == "approach" and np.linalg.norm(tip - above) <= 0.02:
            _state["phase"] = "descend"
        waypoint = above if _state["phase"] == "approach" else cube
        target_grip = grip
    else:
        phase = _state["phase"]
        if phase == "approach":
            waypoint, target_grip = above, _OPEN
            if np.linalg.norm(tip - above) <= 0.02 and grip >= _OPEN - 5:
                _state["phase"] = "descend"
        elif phase == "descend":
            waypoint, target_grip = cube, _OPEN
            if np.linalg.norm(tip - cube) <= 0.012:
                _state["phase"] = "close"
        elif phase == "close":
            waypoint, target_grip = cube, _CLOSE
            if grip <= _CLOSE + 1:
                _state["hold"] += 1
                if _state["hold"] >= 2:
                    _state["phase"] = "lift"
        else:
            waypoint, target_grip = cube + np.array([0.0, 0.0, _LIFT]), _CLOSE
            if not obs.get("grasped") and np.linalg.norm(tip - cube) > 0.05:
                _state.update(phase="approach", hold=0)

    q_star, _ = _KIN.solve_ik(waypoint, q, low=_LOW, high=_HIGH, iters=60)
    q_star[_GRIP] = target_grip
    step = np.clip(q_star - q, -_MAX_STEP, _MAX_STEP)
    return [float(v) for v in np.clip(q + step, _LOW, _HIGH)]
