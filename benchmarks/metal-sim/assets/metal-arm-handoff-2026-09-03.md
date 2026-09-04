# MakerMods Metal arm — engineering handoff

Everything needed to understand, wire up, and drive the Metal arm.
Assembled 2026-09-03.

## What the arm is

Metal is MakerMods' 7-DOF manipulator: **6 revolute joints plus a permanent
(non-optional) gripper**, all Damiao motors on classic CAN at 1 Mbps (not
CAN FD), driven in MIT position control. All positions are **degrees**.

| Joint | Damiao variant | send / recv CAN id | Soft limits (deg) |
|---|---|---|---|
| shoulder_pan  | metal_jlo | 0x01 / 0x11 | -160 .. 160 |
| shoulder_lift | metal_j2  | 0x02 / 0x12 | -180 .. 0   |
| elbow_flex    | metal_jlo | 0x03 / 0x13 | 0 .. 180    |
| wrist_flex    | metal_jlo | 0x04 / 0x14 | -123 .. 81  |
| wrist_yaw     | metal_jhi | 0x05 / 0x15 | -85 .. 85   |
| wrist_roll    | metal_jhi | 0x06 / 0x16 | -145 .. 145 |
| gripper       | metal_jhi | 0x07 / 0x17 | 0 .. 137.5  |

The per-variant MIT parameter ranges are flashed into the motors and mirrored
in `lerobot-integration/motors-damiao/tables.py`. Limits are the vendor's
`position_min/max` from `motor_config.cpp`.

## Conventions that matter

- **Zero pose**: arm standing fully upright, all joints 0, gripper closed.
  Calibration (LeRobot flow) only confirms this pose and zeroes each motor's
  absolute encoder — no range-of-motion recording is needed.
- **Gripper**: raw motor degrees, `0 = closed`; the vendor stroke table
  documents jaw opening up to 116.4 deg (the range above that is past the
  widest measured opening). Physical stroke is 0–80 mm; no stroke-to-angle
  conversion is applied anywhere in the LeRobot path.
- **shoulder_lift** is 0 at upright and goes *negative* to lean the arm over.
- **URDF q ↔ motor degrees is the identity mapping**: calibrated degrees →
  radians, joints in URDF order `JOINT1..JOINT6`, no offsets and no sign
  flips. This is proven on hardware — the gravity-compensated leader feeds
  exactly this q into Pinocchio and the arm floats correctly.
- Both arms of a bimanual pair ship with the same CAN ids, so each arm needs
  its own bus/adapter.

## URDF (`urdf/`)

- `metal_with_gripper.urdf` — the canonical dynamics/kinematics model:
  **6 revolute joints (nq=6), gripper mass lumped into Link6**. This is the
  variant every downstream consumer should use.
  - Provenance: `makermods-robotics/metal-python-ros` @ commit
    `ef4181f1305cbcfc63431d3bcfb96f5fb7f72763`, path
    `metal_sdk/example/urdf/metal_with_gripper.urdf`.
  - sha256: `faac0ba624b28cf531834be0bf4eb90595ae78648dbfe12058b8ec656b65f7ef`
- `metal_no_gripper.urdf` — the bare-arm variant.
- Do **not** use the `metal_ros2/src/metal_description` URDF for dynamics: it
  models the gripper's two prismatic jaws (nq=8) and breaks any consumer
  expecting the 6-joint model.
- `JOINT6` (wrist roll) axis is the gripper's pointing direction — extend a
  tool point along its local x to model the gripper tip.

## SDK (`sdk-metal-python-ros/`)

Full checkout of `makermods-robotics/metal-python-ros` (minus git/build):

- `metal_sdk/` — the C++ SDK with a pybind11 Python binding
  (`MetalSDKInterface`). Recommended mode for model inference is
  `ControlMode.NRT_JOINT_POSITION`; `SetArmJointPosition(list, speed_ratio)`
  takes radians for J1–J6 with a 1–10 speed ratio;
  `SetGripperStroke(mm, speed_ratio)` takes 0–80 mm. Feedback getters:
  `GetJointPosition/Velocity/Effort`, `GetArmEndPose`. See
  `metal_sdk/example/single_arm_control.py`. Needs a native build
  (`build_metal_sdk.sh`, `build_sdk.md`); CAN via SocketCAN interface names.
- `metal_ros2/` — ROS 2 packages including `metal_description`.
- `sdk底层库/` — vendored low-level libraries the native SDK links against.
- `docker/` — build environment.

## LeRobot integration (`lerobot-integration/`)

Pure-Python alternative to the C++ SDK — no native build, works on macOS —
from branch **`arm/makermods-metal`** of
`https://github.com/makermods-robotics/lerobot` (see
`branch-commit-history.txt` for the commit trail; `docs/metal.mdx` is the
full user guide and the best single document in this package).

- `motors-damiao/` — `DamiaoMotorsBus`: classic-CAN Damiao driver with two
  transports. `slcan` (default; pyserial straight to the USB-CAN adapter's
  serial port, works on Linux/macOS/Windows, no sudo) and `socketcan`
  (Linux; `sudo slcand -o -f -s8 /dev/ttyACM0 can0 && sudo ip link set up
  can0 && sudo ip link set can0 txqueuelen 1000`). Measured cost of a full
  7-motor tick (7 state requests + 7 MIT writes + replies) over slcan on a
  CANable: **~5 ms**, so a 60 Hz loop has ample margin.
- `robots/metal_follower/` — the follower: soft-limit clamping on every
  action, firm follow gains written at connect (kp up to 390 on
  shoulder_lift; the bus default kp=10 cannot hold the arm against gravity),
  a slow startup sync (1 deg/step until within 3 deg of the leader, with
  per-joint stall release), an optional `max_relative_target` per-step travel
  cap, and filtered velocity feedforward (`velocity_ff_max_deg_s=120`,
  alpha 0.08) sent through per-frame MIT writes.
- `teleoperators/metal_leader/` — the gravity-compensated leader: Pinocchio
  gravity + Coriolis + viscous-friction feedforward streamed as MIT torque
  with kp=0, so the arm feels weightless in the operator's hand. Vendor
  per-joint scaling coefficients from `kdl_solver.cpp` are baked in. Keep
  `gravity_hz ≤ 100` over slcan (a tick is ~4.2 ms p50). The gripper is left
  fully backdrivable with a small friction feedforward.
- `teleoperators/config_rebot_102_leader_metal.py` — joint-mapping preset to
  drive a Metal follower from a reBot 102 leader (`joint_directions` /
  `joint_ranges` overridable per joint; fractional direction values rescale a
  leader joint onto a differently-ranged follower joint).
- `bi_metal_follower/`, `bi_metal_leader/` — bimanual wrappers (one bus per
  arm, same CAN ids on both).
- `tests/` — the branch's test suite, including packet-level MIT frame
  decoding checks.

## Wiring quick reference

macOS (or any OS), no privileges: plug the USB-CAN adapter in and use
`can_interface="slcan"` with `port="/dev/cu.usbmodemXXXX"` (Linux:
`/dev/ttyACM0`, Windows: `COM5`). Linux kernel stack instead: bring up
`can0` with the slcand commands above and use `can_interface="socketcan"`,
`port="can0"`.

## Known gaps

- Gripper angle→jaw-opening (mm) curve beyond the vendor stroke table is not
  measured; the gripper is driven in raw degrees everywhere.
- Per-joint positive Cartesian directions have not been independently
  verified against a physical build outside of the gravity-compensation
  evidence above; probe with small motions before trusting sign-sensitive
  math on a new unit.
