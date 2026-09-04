"""Policies for the Metal simulator that need no LLM.

* ``metal_ik`` — an oracle: damped-least-squares IK drives the fingertip to the cube (with an
  approach from above), and for ``pick`` goals opens, descends, closes, and lifts. It doubles as
  the BenchFlow oracle solution and as a smoke test for the embodiment.
* ``metal_pyfile`` — loads ``act(observation) -> [7 floats]`` from a Python file. This is the
  code-as-policy contract the BenchFlow tasks ask an agent to fulfil: the agent writes the file,
  the verifier runs it through the simulator.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import ObservationSpace
from inspect_robots.types import Action, ActionChunk, Observation

from inspect_robots_metal_sim.config import EEF_KEY, STATE_KEY, TARGET_KEY, MetalSimConfig, action_box
from inspect_robots_metal_sim.kinematics import JOINT_NAMES, N_DIM, MetalKinematics

Vec = npt.NDArray[np.float64]
GRIPPER = JOINT_NAMES.index("gripper")
_OBS = ObservationSpace(state_keys=frozenset({STATE_KEY, EEF_KEY, TARGET_KEY}))


def _goal_from_instruction(text: str | None) -> str:
    text = (text or "").lower()
    return "pick" if any(w in text for w in ("pick", "lift", "grasp", "grab")) else "reach"


class IKPolicy(PolicyBase):
    """Oracle: move the fingertip to the cube, picking it up when the instruction asks."""

    def __init__(
        self,
        *,
        approach_height_m: float = 0.06,
        open_deg: float = 100.0,
        close_deg: float = 20.0,
        lift_m: float = 0.14,
        goal: str | None = None,
        **_: Any,
    ) -> None:
        cfg = MetalSimConfig()
        self._kin = MetalKinematics(tip_offset_m=cfg.tip_offset_m)
        box = action_box(cfg)
        self._low, self._high = np.asarray(box.low, dtype=np.float64), np.asarray(box.high, dtype=np.float64)
        self._max_step = np.asarray(box.semantics.max_step, dtype=np.float64)
        self._goal_override = goal
        self.approach_height_m = float(approach_height_m)
        self.open_deg, self.close_deg, self.lift_m = float(open_deg), float(close_deg), float(lift_m)
        self.info = PolicyInfo(name="metal_ik", action_space=box, observation_space=_OBS)
        self.config = PolicyConfig(action_horizon=1)
        self.num_inferences = 0
        self._phase = "approach"
        self._goal = "reach"
        self._hold = 0

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        """Adopt the embodiment's declared box so limits and ramp match exactly."""
        box = embodiment_info.action_space
        if box.low is not None and box.high is not None:
            self._low = np.asarray(box.low, dtype=np.float64)
            self._high = np.asarray(box.high, dtype=np.float64)
        if box.semantics is not None and box.semantics.max_step is not None:
            self._max_step = np.asarray([m if m is not None else np.inf for m in box.semantics.max_step], dtype=np.float64)

    def reset(self, scene: Scene) -> None:
        self.num_inferences = 0
        self._goal = self._goal_override or (scene.metadata or {}).get("goal") or _goal_from_instruction(scene.instruction)
        self._phase = "approach"
        self._hold = 0

    def act(self, observation: Observation) -> ActionChunk:
        self.num_inferences += 1
        q = np.asarray(observation.state[STATE_KEY], dtype=np.float64)
        tip = np.asarray(observation.state[EEF_KEY], dtype=np.float64)
        cube = np.asarray(observation.state[TARGET_KEY], dtype=np.float64)
        above = cube + np.array([0.0, 0.0, self.approach_height_m])
        gripper = float(q[GRIPPER])

        if self._goal == "reach":
            waypoint = above if (self._phase == "approach" and np.linalg.norm(tip - above) > 0.02) else cube
            if self._phase == "approach" and np.linalg.norm(tip - above) <= 0.02:
                self._phase = "descend"
            target_grip = gripper
        else:
            if self._phase == "approach":
                waypoint, target_grip = above, self.open_deg
                if np.linalg.norm(tip - above) <= 0.02 and gripper >= self.open_deg - 5:
                    self._phase = "descend"
            elif self._phase == "descend":
                waypoint, target_grip = cube, self.open_deg
                if np.linalg.norm(tip - cube) <= 0.012:
                    self._phase = "close"
            elif self._phase == "close":
                waypoint, target_grip = cube, self.close_deg
                if gripper <= self.close_deg + 1:
                    self._hold += 1
                    if self._hold >= 2:
                        self._phase = "lift"
            else:
                waypoint, target_grip = cube + np.array([0.0, 0.0, self.lift_m]), self.close_deg
                if self._phase == "lift" and observation.extra.get("grasped") is False and np.linalg.norm(tip - cube) > 0.05:
                    self._phase = "approach"  # dropped it: start over

        q_star, _ = self._kin.solve_ik(waypoint, q, low=self._low, high=self._high, iters=60)
        q_star[GRIPPER] = target_grip
        step = np.clip(q_star - q, -self._max_step, self._max_step)
        return ActionChunk(actions=[Action(data=np.clip(q + step, self._low, self._high))])


def ik_policy(**kwargs: Any) -> IKPolicy:
    """Registry factory for ``metal_ik``."""
    return IKPolicy(**kwargs)


class PyFilePolicy(PolicyBase):
    """Run ``act(observation: dict) -> sequence[7 floats]`` from a user Python file.

    The observation dict has ``joint_pos``, ``eef_pos``, ``target_pos`` (lists), ``instruction``,
    ``step``, ``labels`` (joint names), ``grasped`` and ``goal``. An optional
    ``reset(scene: dict)`` in the file is called at the start of every episode.
    """

    def __init__(self, *, path: str, **_: Any) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"policy file not found: {self.path}")
        spec = importlib.util.spec_from_file_location(f"metal_policy_{abs(hash(str(self.path)))}", self.path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import policy file {self.path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if not callable(getattr(module, "act", None)):
            raise AttributeError(f"{self.path} must define act(observation) -> 7 joint targets in degrees")
        self._module = module
        self.info = PolicyInfo(name="metal_pyfile", action_space=action_box(MetalSimConfig()), observation_space=_OBS)
        self.config = PolicyConfig(action_horizon=1)
        self.num_inferences = 0
        self._step = 0
        self._instruction: str | None = None

    def reset(self, scene: Scene) -> None:
        self.num_inferences = 0
        self._step = 0
        self._instruction = scene.instruction
        hook = getattr(self._module, "reset", None)
        if callable(hook):
            hook({"id": scene.id, "instruction": scene.instruction, "metadata": dict(scene.metadata or {})})

    def act(self, observation: Observation) -> ActionChunk:
        self.num_inferences += 1
        obs = {
            "joint_pos": [float(v) for v in observation.state[STATE_KEY]],
            "eef_pos": [float(v) for v in observation.state[EEF_KEY]],
            "target_pos": [float(v) for v in observation.state[TARGET_KEY]],
            "instruction": self._instruction,
            "step": self._step,
            "labels": list(JOINT_NAMES),
            "grasped": bool(observation.extra.get("grasped", False)),
            "goal": observation.extra.get("goal"),
        }
        self._step += 1
        out = np.asarray(self._module.act(obs), dtype=np.float64).reshape(-1)
        if out.shape != (N_DIM,):
            raise ValueError(f"act() returned shape {out.shape}, expected ({N_DIM},)")
        return ActionChunk(actions=[Action(data=out)])


def pyfile_policy(**kwargs: Any) -> PyFilePolicy:
    """Registry factory for ``metal_pyfile`` (``-P path=/path/to/policy.py``)."""
    return PyFilePolicy(**kwargs)
