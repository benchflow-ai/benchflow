"""Configuration and declared spaces for the Metal arm simulator.

The action space is deliberately identical to the real ``metal_arm`` plugin: seven absolute joint
targets in degrees with the vendor's soft limits, so a policy written against the simulator runs
unchanged on the hardware embodiment.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace, StateField, StateSpec

from inspect_robots_metal_sim.kinematics import JOINT_NAMES, N_DIM

STATE_KEY = "joint_pos"
EEF_KEY = "eef_pos"
TARGET_KEY = "target_pos"
CAMERA_NAME = "front"

#: Vendor soft limits in degrees (copied from LeRobot's ``MetalFollowerConfigBase.joint_limits``
#: and the ``metal_arm`` plugin so the two embodiments declare the same box). Gripper 0 = closed.
VENDOR_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-160.0, 160.0),
    "shoulder_lift": (-180.0, 0.0),
    "elbow_flex": (0.0, 180.0),
    "wrist_flex": (-123.0, 81.0),
    "wrist_yaw": (-85.0, 85.0),
    "wrist_roll": (-145.0, 145.0),
    "gripper": (0.0, 137.5),
}

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})
_NONE = frozenset({"", "none", "null"})


def _coerce(name: str, value: Any, annotation: str) -> Any:
    optional = "None" in annotation
    base = annotation.replace("| None", "").replace("None |", "").strip()
    if value is None:
        return None
    if isinstance(value, str) and optional and value.strip().lower() in _NONE:
        return None
    if base == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ValueError(f"{name}: expected a boolean, got {value!r}")
    if base == "int":
        return int(value)
    if base == "float":
        return float(value)
    if base == "str":
        return str(value)
    return value


@dataclass(frozen=True)
class MetalSimConfig:
    """Scalar settings; every field is settable with ``-E key=value``."""

    control_hz: float = 10.0  # declared step rate (steps are not paced unless realtime=true)
    realtime: bool = False  # sleep so that steps happen at control_hz (nice for the live viewer)
    max_velocity_deg_s: float = 30.0  # joint speed ceiling; per-step cap = this / control_hz
    gripper_max_velocity_deg_s: float = 90.0
    gripper_max_deg: float = 115.0
    limit_margin_deg: float = 1.0
    goal_radius_m: float = 0.03  # tip-to-cube-centre distance that counts as reached
    grasp_close_deg: float = 45.0  # gripper at or below this while at the cube = grasped
    lift_height_m: float = 0.08  # cube centre height that completes a pick
    cube_size_m: float = 0.03
    tip_offset_m: float = 0.09  # fingertip point beyond the gripper mount, along the tool axis
    workspace_x: str = "0.22,0.38"  # seeded cube x range (m), forward of the base
    workspace_y: str = "-0.15,0.15"  # seeded cube y range (m)
    camera: bool = True  # render the synthetic front camera
    cam_width: int = 320
    cam_height: int = 240
    trace_dir: str | None = "logs/metal_sim_traces"  # per-episode JSONL traces for the web viewer

    def __post_init__(self) -> None:
        if not self.control_hz > 0:
            raise ValueError("control_hz must be positive")
        if not self.max_velocity_deg_s > 0 or not self.gripper_max_velocity_deg_s > 0:
            raise ValueError("velocity ceilings must be positive")
        if not 0 < self.gripper_max_deg <= VENDOR_LIMITS_DEG["gripper"][1]:
            raise ValueError("gripper_max_deg must be in (0, 137.5]")
        if self.limit_margin_deg < 0:
            raise ValueError("limit_margin_deg must be non-negative")
        if self.goal_radius_m <= 0 or self.cube_size_m <= 0 or self.lift_height_m <= 0:
            raise ValueError("goal_radius_m, cube_size_m and lift_height_m must be positive")
        if self.cam_width <= 0 or self.cam_height <= 0:
            raise ValueError("camera size must be positive")
        self.x_range  # validate
        self.y_range

    @classmethod
    def from_kwargs(cls, **flat: Any) -> MetalSimConfig:
        fields = {f.name: f for f in dataclasses.fields(cls)}
        unknown = set(flat) - set(fields)
        if unknown:
            raise TypeError(f"MetalSimConfig got unexpected config keys: {sorted(unknown)}")
        return cls(**{k: _coerce(k, v, str(fields[k].type)) for k, v in flat.items()})

    @staticmethod
    def _parse_range(text: str, name: str) -> tuple[float, float]:
        parts = [float(p) for p in text.split(",")]
        if len(parts) != 2 or parts[0] > parts[1]:
            raise ValueError(f"{name} must be 'lo,hi' with lo <= hi, got {text!r}")
        return parts[0], parts[1]

    @property
    def x_range(self) -> tuple[float, float]:
        return self._parse_range(self.workspace_x, "workspace_x")

    @property
    def y_range(self) -> tuple[float, float]:
        return self._parse_range(self.workspace_y, "workspace_y")

    @property
    def joint_limits(self) -> dict[str, tuple[float, float]]:
        limits = dict(VENDOR_LIMITS_DEG)
        limits["gripper"] = (0.0, self.gripper_max_deg)
        return limits

    @property
    def max_step_deg(self) -> tuple[float, ...]:
        joint = self.max_velocity_deg_s / self.control_hz
        gripper = self.gripper_max_velocity_deg_s / self.control_hz
        return tuple(gripper if name == "gripper" else joint for name in JOINT_NAMES)

    @property
    def low(self) -> npt.NDArray[np.float64]:
        limits = self.joint_limits
        return np.asarray(
            [limits[n][0] + (0.0 if n == "gripper" else self.limit_margin_deg) for n in JOINT_NAMES], dtype=np.float64
        )

    @property
    def high(self) -> npt.NDArray[np.float64]:
        limits = self.joint_limits
        return np.asarray(
            [limits[n][1] - (0.0 if n == "gripper" else self.limit_margin_deg) for n in JOINT_NAMES], dtype=np.float64
        )

    @property
    def cameras(self) -> tuple[str, ...]:
        return (CAMERA_NAME,) if self.camera else ()


def action_box(cfg: MetalSimConfig) -> Box:
    """The 7-D absolute joint-position action space in degrees (identical to ``metal_arm``)."""
    return Box(
        shape=(N_DIM,),
        low=cfg.low,
        high=cfg.high,
        semantics=ActionSemantics(
            control_mode="joint_pos",
            rotation_repr="none",
            gripper="continuous",
            frame="base",
            dim_labels=JOINT_NAMES,
            max_step=cfg.max_step_deg,
        ),
    )


def observation_space(cfg: MetalSimConfig) -> ObservationSpace:
    """Joint positions (deg), fingertip and cube positions (m), plus the optional camera."""
    cameras = tuple(CameraSpec(name=n, height=cfg.cam_height, width=cfg.cam_width, channels=3) for n in cfg.cameras)
    return ObservationSpace(
        cameras=cameras,
        state=StateSpec(
            fields=(
                StateField(key=STATE_KEY, shape=(N_DIM,), unit="deg"),
                StateField(key=EEF_KEY, shape=(3,), unit="m"),
                StateField(key=TARGET_KEY, shape=(3,), unit="m"),
            )
        ),
    )
