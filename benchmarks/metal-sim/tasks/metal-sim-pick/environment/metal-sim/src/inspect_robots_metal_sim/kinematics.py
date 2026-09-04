"""Forward kinematics of the MakerMods Metal arm from the vendor's canonical URDF.

The chain (joint origins, axes, limits) is loaded from ``data/metal_chain.json``, composed by
``scripts/urdf_to_chain.py`` from the vendor's kinematics URDF (``metal_sdk/example/urdf/
metal_with_gripper.urdf`` in makermods-robotics/metal-python-ros @ ef4181f, joints 1-6) plus the
gripper mount and finger joints of the ``metal_description`` URDF (visualisation only).

Joint conventions
-----------------
Motor degrees map to the URDF joints **one-to-one**: ``rad(q)`` on ``joint1..joint6``, no offsets
and no sign flips. This is the mapping the vendor SDK, the gravity-compensated LeRobot leader and
the teammate handoff use, and it reproduces the vendor's recorded ``end_pose`` values (the
``Link6`` origin) to within a millimetre (see ``tests/test_sim.py``).

Consequently the calibrated zero is the vendor's parked pose, not an "upright" arm: at q = 0 the
upper arm is horizontal pointing backward (-x) and the forearm is folded forward over it, with
the gripper pointing +x at about 19 cm height. ``shoulder_lift`` 0 = horizontal backward, -90 =
vertical, -180 = horizontal forward; ``elbow_flex`` 0 = fully folded, 180 = straight. The gripper
motor angle (0 = closed) drives the two prismatic finger joints, 0.05 m of travel each at the
vendor's 137.5 deg full stroke.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]

JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)
ARM_JOINTS = JOINT_NAMES[:6]
N_DIM = len(JOINT_NAMES)
GRIPPER_FULL_STROKE_DEG = 137.5
FINGER_TRAVEL_M = 0.05

#: Driver joint -> URDF joint name.
URDF_JOINT_FOR: dict[str, str] = {
    "shoulder_pan": "joint1",
    "shoulder_lift": "joint2",
    "elbow_flex": "joint3",
    "wrist_flex": "joint4",
    "wrist_yaw": "joint5",
    "wrist_roll": "joint6",
}


def load_chain() -> dict[str, Any]:
    """Return the parsed ``metal_chain.json`` shipped with the package."""
    with resources.files("inspect_robots_metal_sim.data").joinpath("metal_chain.json").open() as fh:
        return json.load(fh)


def rpy_matrix(r: float, p: float, y: float) -> Vec:
    """URDF roll-pitch-yaw (extrinsic x-y-z) to a rotation matrix."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def axis_angle(axis: Vec, theta: float) -> Vec:
    """Rodrigues rotation about a unit axis."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def driver_to_urdf(q_deg: Vec) -> dict[str, float]:
    """Map the 7 driver joint values (degrees) to URDF joint values (rad / m)."""
    q = np.asarray(q_deg, dtype=np.float64).reshape(N_DIM)
    rad = np.deg2rad(q[:6])
    finger = float(np.clip(q[6], 0.0, GRIPPER_FULL_STROKE_DEG)) / GRIPPER_FULL_STROKE_DEG * FINGER_TRAVEL_M
    return {
        "joint1": float(rad[0]),
        "joint2": float(rad[1]),
        "joint3": float(rad[2]),
        "joint4": float(rad[3]),
        "joint5": float(rad[4]),
        "joint6": float(rad[5]),
        "joint7": -finger,
        "joint8": finger,
    }


@dataclass(frozen=True)
class _Joint:
    name: str
    type: str
    parent: str
    child: str
    xyz: Vec
    rot: Vec
    axis: Vec | None


class MetalKinematics:
    """Forward kinematics and a small numeric IK for the Metal arm."""

    def __init__(self, *, tip_offset_m: float = 0.09) -> None:
        chain = load_chain()
        self.joints: list[_Joint] = [
            _Joint(
                name=j["name"],
                type=j["type"],
                parent=j["parent"],
                child=j["child"],
                xyz=np.asarray(j["xyz"], dtype=np.float64),
                rot=rpy_matrix(*j["rpy"]),
                axis=None if j["axis"] is None else np.asarray(j["axis"], dtype=np.float64),
            )
            for j in chain["joints"]
        ]
        self.links: tuple[str, ...] = tuple(link["name"] for link in chain["links"])
        self.tip_offset_m = float(tip_offset_m)
        # The gripper is mounted at +x of link6; the tool (approach) axis continues along it.
        gripper_mount = next(j for j in self.joints if j.name == "gripper_base")
        self._tool_dir_link6: Vec = gripper_mount.xyz / np.linalg.norm(gripper_mount.xyz)
        self._tool_len: float = float(np.linalg.norm(gripper_mount.xyz))

    # -- forward kinematics ---------------------------------------------------

    def link_transforms(self, q_deg: Vec) -> dict[str, Vec]:
        """World 4x4 transform of every link frame for the given driver joint values."""
        values = driver_to_urdf(q_deg)
        frames: dict[str, Vec] = {"world": np.eye(4)}
        for j in self.joints:
            parent = frames[j.parent]
            local = np.eye(4)
            local[:3, :3] = j.rot
            local[:3, 3] = j.xyz
            motion = np.eye(4)
            if j.axis is not None:
                value = values.get(j.name, 0.0)
                if j.type == "revolute":
                    motion[:3, :3] = axis_angle(j.axis, value)
                elif j.type == "prismatic":
                    motion[:3, 3] = j.axis / np.linalg.norm(j.axis) * value
            frames[j.child] = parent @ local @ motion
        return frames

    def joint_positions(self, q_deg: Vec) -> Vec:
        """World positions of base, the six joint origins, the gripper mount, and the tip (9, 3)."""
        frames = self.link_transforms(q_deg)
        pts = [frames[name][:3, 3] for name in ("base_link", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_base")]
        pts.append(self.tip_from_frames(frames))
        return np.asarray(pts, dtype=np.float64)

    def tip_from_frames(self, frames: dict[str, Vec]) -> Vec:
        link6 = frames["link6"]
        local = self._tool_dir_link6 * (self._tool_len + self.tip_offset_m)
        return link6[:3, :3] @ local + link6[:3, 3]

    def tip(self, q_deg: Vec) -> Vec:
        """World position (m) of the point between the fingertips."""
        return self.tip_from_frames(self.link_transforms(q_deg))

    def flange(self, q_deg: Vec) -> Vec:
        """World position (m) of the ``link6`` origin: the vendor SDK's ``end_pose`` point."""
        return self.link_transforms(q_deg)["link6"][:3, 3].copy()

    def tool_axis(self, q_deg: Vec) -> Vec:
        """Unit approach direction of the gripper in world coordinates."""
        link6 = self.link_transforms(q_deg)["link6"]
        return link6[:3, :3] @ self._tool_dir_link6

    # -- inverse kinematics ---------------------------------------------------

    def jacobian(self, q_deg: Vec, eps_deg: float = 0.05) -> Vec:
        """Numeric (3 x 6) Jacobian of the tip position w.r.t. the six arm joints, per degree."""
        q = np.asarray(q_deg, dtype=np.float64).copy()
        base = self.tip(q)
        jac = np.zeros((3, 6), dtype=np.float64)
        for i in range(6):
            dq = q.copy()
            dq[i] += eps_deg
            jac[:, i] = (self.tip(dq) - base) / eps_deg
        return jac

    #: Extra seed poses (deg) tried when the caller's start pose stalls in a joint limit; they cover
    #: the reaching postures used on the hardware (shoulder past vertical, elbow partly unfolded).
    IK_SEEDS: tuple[tuple[float, ...], ...] = (
        (0.0, -120.0, 40.0, 70.0, 0.0, 0.0),
        (0.0, -130.0, 60.0, 50.0, 0.0, 0.0),
        (0.0, -100.0, 90.0, 20.0, 0.0, 0.0),
        (0.0, -150.0, 90.0, 60.0, 0.0, 0.0),
    )

    def solve_ik(
        self,
        target: Vec,
        q0_deg: Vec,
        *,
        low: Vec,
        high: Vec,
        iters: int = 200,
        damping: float = 0.02,
        step_deg: float = 6.0,
        tol_m: float = 1e-3,
        restarts: bool = True,
    ) -> tuple[Vec, float]:
        """Damped-least-squares IK for the tip position; returns (q_deg, residual_m).

        Runs from ``q0_deg`` first; if that stalls above ``tol_m`` and ``restarts`` is set, retries
        from ``IK_SEEDS`` (with the pan joint pointed at the target) and keeps the best solution.
        The gripper dimension is passed through unchanged. Joint limits are enforced by clipping.
        """
        target = np.asarray(target, dtype=np.float64).reshape(3)
        q0 = np.asarray(q0_deg, dtype=np.float64).copy()
        best_q, best_err = self._dls(target, q0, low, high, iters, damping, step_deg, tol_m)
        if best_err > tol_m and restarts:
            pan = float(np.rad2deg(np.arctan2(target[1], target[0])))
            for seed in self.IK_SEEDS:
                q_seed = q0.copy()
                q_seed[:6] = seed
                q_seed[0] = np.clip(pan, low[0], high[0])
                q, err = self._dls(target, q_seed, low, high, iters, damping, step_deg, tol_m)
                if err < best_err:
                    best_q, best_err = q, err
                if best_err <= tol_m:
                    break
        return best_q, best_err

    def _dls(self, target: Vec, q0: Vec, low: Vec, high: Vec, iters: int, damping: float, step_deg: float, tol_m: float) -> tuple[Vec, float]:
        q = q0.copy()
        q[:6] = np.clip(q[:6], low[:6], high[:6])
        best_q, best_err = q.copy(), float("inf")
        for _ in range(iters):
            err = target - self.tip(q)
            dist = float(np.linalg.norm(err))
            if dist < best_err:
                best_q, best_err = q.copy(), dist
            if dist < tol_m:
                break
            jac = self.jacobian(q)
            jjt = jac @ jac.T + (damping**2) * np.eye(3)
            dq = jac.T @ np.linalg.solve(jjt, err)
            norm = float(np.linalg.norm(dq))
            if norm > step_deg:
                dq *= step_deg / norm
            q[:6] = np.clip(q[:6] + dq, low[:6], high[:6])
        return best_q, best_err

    # -- documentation helpers -------------------------------------------------

    def motion_directions(self, q_deg: Vec) -> dict[str, str]:
        """Human-readable direction the tip moves for +1 deg on each arm joint at a pose."""
        jac = self.jacobian(q_deg)
        names = {0: "+x (forward)", 1: "+y (left)", 2: "+z (up)"}
        out: dict[str, str] = {}
        for i, joint in enumerate(ARM_JOINTS):
            col = jac[:, i]
            k = int(np.argmax(np.abs(col)))
            if np.abs(col[k]) < 1e-6:
                out[joint] = "no tip motion at this pose"
                continue
            sign = "" if col[k] > 0 else "-"
            label = names[k].replace("+", sign) if sign else names[k]
            if sign:
                label = label.replace("forward", "backward").replace("left", "right").replace("up", "down")
            out[joint] = f"{label} ({col[k] * 1000:+.1f} mm per deg)"
        return out
