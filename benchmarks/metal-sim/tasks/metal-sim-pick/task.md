---
schema_version: "1.3"
task:
  name: metal-sim/metal-sim-pick
  description: Write a joint-space controller that makes a simulated MakerMods Metal arm pick a red cube on the table
metadata:
  author_name: benchflow
  difficulty: medium
  category: robotics
  tags: [robotics, simulation, inspect-robots, code-as-policy, metal-arm]
agent:
  timeout_sec: 1800
verifier:
  timeout_sec: 900
sandbox:
  cpus: 2
  memory_mb: 4096
---
# Pick up the cube with a simulated Metal arm

## prompt

A simulated MakerMods Metal arm (6 revolute joints plus a gripper) stands on a table with a red 3 cm cube in front of it. Your job is to write the controller in `/app/policy.py` so that the arm picks the cube from any of the held-out cube positions the verifier will sample.

The file must define `act(observation) -> list[float]`: seven absolute joint targets in degrees (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, gripper). It is called once per control step at 10 Hz. `observation` is a dict with `joint_pos` (deg), `eef_pos` (fingertip point, metres; base at the origin, +x forward, +y left, +z up), `target_pos` (cube centre, metres), `goal`, `grasped`, `instruction`, `step` and `labels`. An optional `reset(scene)` runs at the start of each episode. The starter file in `/app/policy.py` documents the contract.

Facts about the arm: the zero pose (all joints 0) is the vendor's parked pose, with the upper arm horizontal pointing backward, the forearm folded forward over it, and the gripper pointing forward about 19 cm above the table. `shoulder_lift` is 0 there, -90 with the upper arm vertical and -180 with it horizontal forward (it only goes negative); `elbow_flex` is 0 fully folded and 180 straight (it only goes positive). Reaching the table in front of the base typically means a shoulder around -110 to -140, an elbow around 40 to 90, and the wrist pitched down. Commands are clamped to the soft limits and to 3 deg of travel per step (9 deg for the gripper). The gripper is a motor angle (0 = closed, 115 = fully open). The cube is grasped when the fingertip point is within 3 cm of the cube centre and the gripper is at or below 45 deg; it then travels with the fingertips. Opening past 55 deg drops it. The episode succeeds when the held cube's centre is at least 8 cm above the table.

The Python package `inspect_robots_metal_sim` is installed: `MetalKinematics` gives forward kinematics of the fingertip (`tip(q_deg)`) and a numeric IK (`solve_ik`), and the `front` camera is rendered if you want images. Test your controller inside this environment before finishing:

```bash
metal-sim-eval --task metal-sim-pick --policy-file /app/policy.py -T num_scenes=4 -T max_steps=200 --no-frames
```

Success is the fraction of held-out scenes (different seeds from the ones above) in which the episode ends in success within 200 steps. Leave the final controller at `/app/policy.py`.
