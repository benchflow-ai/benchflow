# Vendor assets for the MakerMods Metal arm

| File | Source | Use |
|---|---|---|
| `metal_with_gripper.canonical.urdf` | makermods-robotics/metal-python-ros @ `ef4181f1305cbcfc63431d3bcfb96f5fb7f72763`, `metal_sdk/example/urdf/metal_with_gripper.urdf` (sha256 `faac0ba6…65f7ef`) | **Kinematics and dynamics.** 6 revolute joints `JOINT1..JOINT6`, gripper mass lumped into `Link6`. Motor degrees map to these joints one-to-one (radians, no offsets): verified against the vendor's recorded end poses (0 mm) and the gravity-compensated LeRobot leader. |
| `metal_no_gripper.canonical.urdf` | same commit, `metal_sdk/example/urdf/` | bare-arm variant |
| `metal_with_gripper.description.urdf` | makermods-robotics/metal-hardware `docs/urdf/` (same as `metal_ros2/src/metal_description/urdf/`) | **Visualisation only.** Adds `gripper_base` and the two prismatic finger joints (`joint7`, `joint8`); nq = 8, so do not feed it to a dynamics library. |
| `meshes/*.STL` | metal-python-ros `metal_ros2/src/metal_description/meshes/` | link meshes in metres, one per link frame |

`metal-sim/scripts/urdf_to_chain.py` composes the simulator's chain from the canonical arm plus the description's gripper.

Joint conventions (from the teammate handoff of 2026-09-03, `docs/metal-arm-handoff-2026-09-03.md`): the calibrated zero is the vendor's parked pose, which in the URDF is the upper arm horizontal pointing backward (-x) with the forearm folded forward over it. `shoulder_lift` 0 = horizontal backward, -90 = vertical, -180 = horizontal forward. `elbow_flex` 0 = fully folded, 180 = straight. `JOINT6`'s local x is the gripper's pointing direction.
