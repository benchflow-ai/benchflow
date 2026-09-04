"""``MetalSimEmbodiment``: a kinematic simulator of the MakerMods Metal arm.

It mirrors the real ``metal_arm`` plugin's contract — seven absolute joint targets in degrees,
the same soft limits, the same per-step travel cap — but replaces the CAN bus with the URDF
forward kinematics. A red cube is placed on the table at a seeded position; the episode ends
with ``info["success"]`` when the fingertip point reaches it (``reach`` goal) or when it has been
grasped and lifted (``pick`` goal). ``info["distance"]`` is the fingertip-to-cube distance, so the
built-in ``success_at_end`` and ``min_distance_to_goal`` scorers work unchanged.

The simulator writes one JSONL trace per episode under ``trace_dir`` (joint positions, fingertip,
cube, distance, success per step); the ``web/`` viewer replays and live-streams those traces.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import numpy as np
import numpy.typing as npt
from inspect_robots.conformance import NumberSlot, OptionSlot
from inspect_robots.embodiment import (
    AUTO_RESET,
    PRIVILEGED_SUCCESS,
    RENDERABLE,
    RESETTABLE,
    SEEDABLE,
    EmbodimentBase,
    EmbodimentInfo,
)
from inspect_robots.errors import EmbodimentFault
from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult

from inspect_robots_metal_sim.config import (
    CAMERA_NAME,
    EEF_KEY,
    STATE_KEY,
    TARGET_KEY,
    MetalSimConfig,
    action_box,
    observation_space,
)
from inspect_robots_metal_sim.kinematics import ARM_JOINTS, JOINT_NAMES, N_DIM, MetalKinematics
from inspect_robots_metal_sim.render import FrontCamera

Vec = npt.NDArray[np.float64]
GOALS = ("reach", "pick")
GRIPPER = JOINT_NAMES.index("gripper")
#: Hysteresis above ``grasp_close_deg`` before a held cube is released.
RELEASE_MARGIN_DEG = 10.0


def infer_goal(scene: Scene) -> str:
    """Goal kind for a scene: explicit metadata/target wins, else keywords in the instruction."""
    meta_goal = scene.metadata.get("goal") if scene.metadata else None
    if meta_goal is None and scene.target is not None:
        meta_goal = scene.target.spec.get("goal")
    if meta_goal is not None:
        goal = str(meta_goal).lower()
        if goal not in GOALS:
            raise EmbodimentFault(f"unknown goal {goal!r}; expected one of {GOALS}")
        return goal
    text = (scene.instruction or "").lower()
    return "pick" if any(word in text for word in ("pick", "lift", "grasp", "grab")) else "reach"


def operating_notes(cfg: MetalSimConfig, kin: MetalKinematics) -> str:
    """Markdown notes injected into LLM policy prompts; motion directions come from the model."""
    probe = np.array([0.0, -120.0, 40.0, 70.0, 0.0, 0.0, 60.0])
    directions = kin.motion_directions(probe)
    zero_tip = kin.tip(np.zeros(N_DIM))
    rows = "\n".join(
        f"- {name}: [{lo:+.0f}, {hi:+.0f}] deg" for name, lo, hi in zip(JOINT_NAMES, cfg.low, cfg.high, strict=True)
    )
    dir_rows = "\n".join(f"- +1 deg on {j}: tip moves {directions[j]}" for j in ARM_JOINTS)
    return (
        "Simulated MakerMods Metal arm (kinematic model from the vendor URDF): 6 revolute joints plus "
        "a gripper. Actions are absolute joint targets in DEGREES, identical to the real arm. Joint "
        "order from base to tip: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, "
        "wrist_roll, gripper. The gripper is a motor angle: 0 = fully closed, larger = wider open.\n\n"
        "The zero pose (all joints 0) is the vendor's parked pose: the upper arm lies horizontal "
        "pointing backward (-x) and the forearm folds forward over it, so the fingertips start at "
        f"about ({zero_tip[0]:.2f}, {zero_tip[1]:.2f}, {zero_tip[2]:.2f}) m pointing forward. "
        "shoulder_lift: 0 = upper arm horizontal backward, -90 = vertical, -180 = horizontal "
        "forward (it only goes negative). elbow_flex: 0 = fully folded, 180 = straight (it only "
        "goes positive). To reach the table in front of the base you typically lean the shoulder "
        "past vertical (around -110 to -140) and partly unfold the elbow (around 40 to 90), then "
        "pitch the wrist down.\n\n"
        "World frame: the base sits at the origin on a table at z = 0; +x is forward (away from "
        f"the base), +y is left, +z is up, all in metres. state[{EEF_KEY}] is the point between "
        f"the fingertips and state[{TARGET_KEY}] is the centre of a red cube "
        f"({cfg.cube_size_m * 100:.0f} cm) resting on the table. Reaching means bringing the "
        f"fingertip point within {cfg.goal_radius_m * 100:.0f} cm of the cube centre. Picking "
        f"means: open the gripper, reach the cube, close it to {cfg.grasp_close_deg:.0f} deg or "
        f"less (it then travels with the fingertips), and lift it to at least "
        f"{cfg.lift_height_m * 100:.0f} cm. Opening the gripper again drops the cube.\n\n"
        f"The controller clamps to the soft limits and caps per-step travel to "
        f"{cfg.max_velocity_deg_s / cfg.control_hz:.1f} deg per step "
        f"({cfg.max_velocity_deg_s:.0f} deg/s); joints move exactly to the capped command.\n\n"
        "Approximate tip motion directions at a typical reaching pose "
        "(shoulder_lift -120, elbow_flex 40, wrist_flex 70):\n" + dir_rows + "\n\n"
        "Commandable ranges:\n" + rows
    )


class MetalSimEmbodiment(EmbodimentBase):
    """Kinematic Metal arm plus a table-top cube; seeded, resettable, with privileged success."""

    OPTION_SLOTS = (
        OptionSlot(arg="camera", label="Render the synthetic front camera", default=True),
        OptionSlot(arg="realtime", label="Pace steps to control_hz (for the live viewer)", default=False),
    )
    NUMBER_SLOTS = (
        NumberSlot(arg="max_velocity_deg_s", label="Joint speed ceiling (deg/s)", default=30, minimum=1, maximum=180),
        NumberSlot(arg="control_hz", label="Step rate (Hz)", default=10, minimum=1, maximum=60),
        NumberSlot(arg="goal_radius_m", label="Reach tolerance (m)", default=0.03, minimum=0.005, maximum=0.2),
    )

    def __init__(
        self,
        config: MetalSimConfig | None = None,
        *,
        sleep_fn: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else MetalSimConfig.from_kwargs(**flat)
        self._sleep = sleep_fn or time.sleep
        self._clock = clock or time.perf_counter
        self._kin = MetalKinematics(tip_offset_m=self._cfg.tip_offset_m)
        self._camera = FrontCamera(self._cfg.cam_width, self._cfg.cam_height) if self._cfg.camera else None
        self._low, self._high = self._cfg.low, self._cfg.high
        self._max_step: Vec = np.asarray(self._cfg.max_step_deg, dtype=np.float64)

        self._q: Vec = np.zeros(N_DIM)
        self._last_cmd: Vec | None = None
        self._cube: Vec = np.array([0.25, 0.0, self._cfg.cube_size_m / 2])
        self._grasped = False
        self._success = False
        self._goal = "reach"
        self._instruction: str | None = None
        self._scene_id: str | None = None
        self._seed: int | None = None
        self._task_name: str | None = None
        self._epochs: dict[str, int] = {}
        self._t_last = 0.0
        self.num_steps = 0

        self._run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
        self._trace: IO[str] | None = None
        self._trace_path: Path | None = None

        self.info = EmbodimentInfo(
            name="metal_sim",
            action_space=action_box(self._cfg),
            observation_space=observation_space(self._cfg),
            control_hz=self._cfg.control_hz,
            is_simulated=True,
            capabilities=frozenset({SEEDABLE, RESETTABLE, AUTO_RESET, PRIVILEGED_SUCCESS, RENDERABLE}),
            docs=operating_notes(self._cfg, self._kin),
        )

    # -- framework hooks ------------------------------------------------------

    def bind_task(self, envelope: Any) -> None:
        """Remember the task name for the trace header."""
        self._task_name = getattr(envelope, "name", None)

    @property
    def kinematics(self) -> MetalKinematics:
        return self._kin

    @property
    def config(self) -> MetalSimConfig:
        return self._cfg

    @property
    def cube_pos(self) -> Vec:
        return self._cube.copy()

    @property
    def trace_path(self) -> Path | None:
        return self._trace_path

    # -- lifecycle ------------------------------------------------------------

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Home the arm, place the cube from the seed (or ``scene.metadata['cube_pos']``)."""
        self._end_trace()
        self._goal = infer_goal(scene)
        self._instruction = scene.instruction
        self._scene_id = scene.id
        self._seed = seed if seed is not None else scene.init_seed
        rng = np.random.RandomState(self._seed if self._seed is not None else 0)
        override = scene.metadata.get("cube_pos") if scene.metadata else None
        if override is not None:
            self._cube = np.asarray([float(v) for v in override], dtype=np.float64).reshape(3)
        else:
            x = rng.uniform(*self._cfg.x_range)
            y = rng.uniform(*self._cfg.y_range)
            self._cube = np.array([x, y, self._cfg.cube_size_m / 2], dtype=np.float64)
        home = scene.metadata.get("home_pose") if scene.metadata else None
        self._q = (
            np.clip(np.asarray([float(v) for v in home], dtype=np.float64).reshape(N_DIM), self._low, self._high)
            if home is not None
            else np.zeros(N_DIM)
        )
        self._last_cmd = self._q.copy()
        self._grasped = False
        self._success = False
        self.num_steps = 0
        self._t_last = self._clock()
        self._epochs[scene.id] = self._epochs.get(scene.id, -1) + 1
        self._start_trace(scene)
        return self._observe()

    def step(self, action: Action) -> StepResult:
        """Clamp, ramp, move the kinematic model, update the cube, and report success."""
        if self._scene_id is None:
            raise EmbodimentFault("step() called before reset()")
        cmd = np.asarray(action.data, dtype=np.float64).reshape(-1)
        if cmd.shape != (N_DIM,):
            raise EmbodimentFault(f"action has shape {cmd.shape}, expected ({N_DIM},)")
        if not np.all(np.isfinite(cmd)):
            raise EmbodimentFault("action contains non-finite values")
        clamped = np.clip(cmd, self._low, self._high)
        if self._last_cmd is not None:
            clamped = np.clip(clamped, self._last_cmd - self._max_step, self._last_cmd + self._max_step)
        self._last_cmd = clamped.copy()
        self._q = clamped.copy()
        if self._cfg.realtime:
            self._pace()
        self.num_steps += 1

        tip = self._kin.tip(self._q)
        gripper = float(self._q[GRIPPER])
        if self._grasped:
            if gripper > self._cfg.grasp_close_deg + RELEASE_MARGIN_DEG:
                self._grasped = False
                self._cube = np.array([tip[0], tip[1], self._cfg.cube_size_m / 2])
            else:
                self._cube = tip.copy()
        distance = float(np.linalg.norm(tip - self._cube))
        if not self._grasped and self._goal == "pick" and distance <= self._cfg.goal_radius_m and gripper <= self._cfg.grasp_close_deg:
            self._grasped = True
            self._cube = tip.copy()
            distance = 0.0
        if self._goal == "reach":
            success = distance <= self._cfg.goal_radius_m
        else:
            success = self._grasped and float(self._cube[2]) >= self._cfg.lift_height_m
        self._success = success
        info = {
            "success": success,
            "distance": distance,
            "grasped": self._grasped,
            "goal": self._goal,
            "cube_pos": self._cube.tolist(),
            "tip_pos": tip.tolist(),
        }
        obs = self._observe(tip=tip, distance=distance)
        self._trace_step(clamped, tip, distance, success)
        return StepResult(
            observation=obs,
            reward=-distance,
            terminated=success,
            termination_reason="success" if success else None,
            truncated=False,
            info=info,
        )

    def close(self) -> None:
        """Flush the episode trace; the world itself is in memory."""
        self._end_trace()

    # -- internals ------------------------------------------------------------

    def _pace(self) -> None:
        period = 1.0 / self._cfg.control_hz
        elapsed = self._clock() - self._t_last
        self._sleep(max(0.0, period - elapsed))
        self._t_last = self._clock()

    def _observe(self, *, tip: Vec | None = None, distance: float | None = None) -> Observation:
        if tip is None:
            tip = self._kin.tip(self._q)
        if distance is None:
            distance = float(np.linalg.norm(tip - self._cube))
        images: dict[str, npt.NDArray[np.uint8]] = {}
        image_times: dict[str, float] = {}
        now = self._clock()
        if self._camera is not None:
            images[CAMERA_NAME] = self._camera.render(
                self._kin.joint_positions(self._q), self._cube, self._cfg.cube_size_m, grasped=self._grasped, success=self._success
            )
            image_times[CAMERA_NAME] = now
        return Observation(
            images=images,
            state={STATE_KEY: self._q.copy(), EEF_KEY: tip.astype(np.float64), TARGET_KEY: self._cube.copy()},
            instruction=self._instruction,
            image_times=image_times,
            state_time=now,
            extra={"distance_m": distance, "grasped": self._grasped, "goal": self._goal},
        )

    # -- traces ---------------------------------------------------------------

    def _start_trace(self, scene: Scene) -> None:
        if self._cfg.trace_dir is None:
            return
        run_dir = Path(self._cfg.trace_dir) / self._run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        epoch = self._epochs[scene.id]
        self._trace_path = run_dir / f"{_slug(scene.id)}-e{epoch}.jsonl"
        self._trace = self._trace_path.open("w")
        header = {
            "kind": "header",
            "run_id": self._run_id,
            "task": self._task_name,
            "scene_id": scene.id,
            "epoch": epoch,
            "seed": self._seed,
            "goal": self._goal,
            "instruction": scene.instruction,
            "cube_pos": self._cube.tolist(),
            "home_pose": self._q.tolist(),
            "labels": list(JOINT_NAMES),
            "control_hz": self._cfg.control_hz,
            "goal_radius_m": self._cfg.goal_radius_m,
            "cube_size_m": self._cfg.cube_size_m,
            "started_at": time.time(),
        }
        self._trace.write(json.dumps(header) + "\n")
        self._trace.flush()

    def _trace_step(self, cmd: Vec, tip: Vec, distance: float, success: bool) -> None:
        if self._trace is None:
            return
        line = {
            "kind": "step",
            "t": self.num_steps,
            "time": time.time(),
            "joint_pos": [round(float(v), 4) for v in self._q],
            "action": [round(float(v), 4) for v in cmd],
            "tip": [round(float(v), 5) for v in tip],
            "cube": [round(float(v), 5) for v in self._cube],
            "distance": round(distance, 5),
            "grasped": self._grasped,
            "success": success,
        }
        self._trace.write(json.dumps(line) + "\n")
        self._trace.flush()

    def _end_trace(self) -> None:
        if self._trace is None:
            return
        end = {"kind": "end", "steps": self.num_steps, "success": self._success, "ended_at": time.time()}
        self._trace.write(json.dumps(end) + "\n")
        self._trace.close()
        self._trace = None


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text) or "scene"
