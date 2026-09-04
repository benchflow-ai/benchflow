# Metal Sim: robot-arm control on a simulated MakerMods Metal arm

## Overview

Two hand-authored BenchFlow tasks in which the agent writes a joint-space controller for a 7-DOF robot arm and a simulator judges it. The simulator (`plugin/`, an [Inspect Robots](https://github.com/makermods-robotics/makermods-inspect-robots) embodiment named `metal_sim`) is a kinematic model of the [MakerMods Metal arm](https://www.makermods.ai/metal-arm) built from the vendor's canonical URDF. It declares exactly the action space of the real arm (seven absolute joint targets in degrees, vendor soft limits, 30 deg/s ramp), so a controller developed against it runs unchanged on the hardware embodiment.

| Task | Goal | Steps | Reward |
|---|---|---|---|
| `metal-sim-reach` | bring the fingertip point within 3 cm of a seeded red cube on the table | 120 | scenes succeeded / 6 held-out scenes |
| `metal-sim-pick` | open the gripper, grasp the cube (gripper at or below 45 deg while at it), lift it 8 cm | 200 | same |

This is the **code-as-policy** mode: any BenchFlow agent (Claude Code, Codex, Gemini, ...) works in the sandbox, tests its controller with `metal-sim-eval`, and leaves it at `/app/policy.py`. The verifier runs that file through the simulator on six held-out cube positions (eval seed 4242) and writes the success rate to `/logs/verifier/reward.txt`. `oracle/policy.py` is a damped-least-squares IK controller that scores 1.0.

## Layout

```
benchmark.yaml            descriptor
metal-sim-oracle.yaml     job config: oracle agent (expect 1.0 on both tasks)
metal-sim-claude.yaml     job config: claude agent, code as policy
tasks/metal-sim-reach/    task.md, environment/ (Dockerfile + vendored plugin + starter policy), verifier/, oracle/
tasks/metal-sim-pick/     same layout
plugin/                   the simulator package (source of truth; vendored into each task by sync_plugin.sh)
shared/                   Dockerfile, starter policy, reward script, oracle controller (copied into tasks by sync_plugin.sh)
assets/                   vendor URDFs with provenance and the hardware handoff notes
```

## Running

```bash
bench tasks check benchmarks/metal-sim/tasks/metal-sim-reach
bench eval run --config benchmarks/metal-sim/metal-sim-oracle.yaml      # sanity: 1.0 / 1.0
bench eval run --config benchmarks/metal-sim/metal-sim-claude.yaml
bench eval run --tasks-dir benchmarks/metal-sim/tasks --agent codex --model gpt-5.5 --sandbox daytona
```

Any sandbox backend works (Docker, Daytona, ...); the environment image is `python:3.12-slim` plus `inspect-robots==0.58.0` and the vendored plugin, no GPU, no ROS.

After editing `plugin/` or `shared/`, re-vendor with `benchmarks/metal-sim/sync_plugin.sh` and commit the copies: task packages must stay self-contained because a task's Docker build context is its own `environment/` directory.

## Verifier without a sandbox

The scripts honour `APP_DIR` and `LOG_ROOT`, so the verifier and oracle can be exercised on a workstation with the plugin installed (`pip install -e benchmarks/metal-sim/plugin`):

```bash
mkdir -p /tmp/bf/app /tmp/bf/logs
APP_DIR=/tmp/bf/app benchmarks/metal-sim/tasks/metal-sim-pick/oracle/solve.sh
APP_DIR=/tmp/bf/app LOG_ROOT=/tmp/bf/logs benchmarks/metal-sim/tasks/metal-sim-pick/verifier/test.sh   # reward 1.0000
```

## Simulator facts an agent needs

* World frame: base at the origin on a table at z = 0, +x forward, +y left, +z up, metres. Observation fields: `joint_pos` (deg), `eef_pos` (fingertip point), `target_pos` (cube centre), `goal`, `grasped`.
* Zero pose (all joints 0) is the vendor's parked pose: upper arm horizontal pointing backward, forearm folded forward over it, gripper pointing forward about 19 cm up. `shoulder_lift` 0 = horizontal backward, -90 = vertical, -180 = horizontal forward; `elbow_flex` 0 = folded, 180 = straight. Motor degrees map one-to-one to the canonical URDF joints, verified against the vendor's recorded end poses (0 mm); see `assets/README.md`.
* Commands are clamped to the soft limits and to 3 deg of travel per step (9 deg for the gripper) at 10 Hz.

## Difficulty knobs

The prompt tells the agent that `MetalKinematics.solve_ik` exists, which makes reach easy. Remove that sentence (and the import) for a variant where the agent derives kinematics itself. Widen `workspace_x` / `workspace_y` (`-E` embodiment args in `verifier/test.sh`) or add clutter for harder variants.

## Provenance

Developed alongside the physical arm in the `robot` workspace (LeRobot driver, hardware embodiment `metal_arm`, live web viewer). The LLM-as-policy loop (`inspect-robots run --policy agent --embodiment metal_sim`) runs outside BenchFlow today; wrapping it as an ACP agent is the natural next step.
