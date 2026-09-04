# inspect-robots-metal-sim

A kinematic simulator of the MakerMods Metal arm packaged as an [Inspect Robots](https://github.com/makermods-robotics/makermods-inspect-robots) embodiment. It exposes the **same action space as the real `metal_arm` plugin** (seven absolute joint targets in degrees, vendor soft limits, 30 deg/s ramp), so a policy developed here runs unchanged on the hardware.

The kinematics come from the vendor URDF (`metal_with_gripper.urdf` in makermods-robotics/metal-hardware), converted to `data/metal_chain.json` by `scripts/urdf_to_chain.py`. No physics engine: joints move exactly to the capped command, the red cube sits on the table (z = 0) and rides with the fingertips once grasped.

## Components

| Kind | Name | What it does |
|---|---|---|
| embodiment | `metal_sim` | The simulator. State: `joint_pos` (deg), `eef_pos`, `target_pos` (m). Camera `front` (320x240, software rendered). Privileged `info["success"]` / `info["distance"]`. Writes JSONL traces under `trace_dir` for the web viewer. |
| task | `metal-sim-reach` | Bring the fingertips within 3 cm of a seeded cube (`-T num_scenes=4 -T max_steps=120`). |
| task | `metal-sim-pick` | Open, grasp (gripper at or below 45 deg while at the cube), lift the cube to 8 cm. |
| policy | `metal_ik` | Oracle: damped-least-squares IK with an approach-from-above; solves both tasks. |
| policy | `metal_pyfile` | Loads `act(observation) -> [7 floats]` from `-P path=policy.py` (code-as-policy contract). |
| CLI | `metal-sim-eval` | Runs a task and prints/writes a JSON summary with `success_rate`; used by the BenchFlow verifiers. |

## Quick start

```bash
source .venv/bin/activate
inspect-robots doctor --embodiment metal_sim
inspect-robots run --task metal-sim-reach --policy metal_ik --embodiment metal_sim --no-prompt
inspect-robots run --task metal-sim-pick --policy metal_ik --embodiment metal_sim --no-prompt
inspect-robots view logs/<log>.json          # self-contained HTML report

# LLM as policy (needs ANTHROPIC_API_KEY; model/images come from ~/.config/inspect-robots/config.ini)
inspect-robots run --task metal-sim-reach --policy agent --embodiment metal_sim --no-prompt
inspect-robots "reach the red cube" --sim   # once config.ini has sim_embodiment = metal_sim
```

Every embodiment field is a `-E key=value` flag: `-E realtime=true` paces steps at `control_hz` (nice with the live web viewer), `-E camera=false` skips rendering, `-E max_velocity_deg_s=60` speeds the arm up, `-E trace_dir=none` disables traces.

## Conventions worth knowing

* World frame: base at the origin, +x forward, +y left, +z up, metres. Motor degrees map to the canonical URDF joints one-to-one (no offsets), so the zero pose is the vendor's parked pose: upper arm horizontal pointing backward, forearm folded forward over it, gripper pointing forward about 19 cm up. `shoulder_lift` 0 = horizontal backward, -90 = vertical, -180 = horizontal forward; `elbow_flex` 0 = folded, 180 = straight. Verified against the vendor's recorded end poses (0 mm) in `tests/test_sim.py`.
* The wrist joint signs follow the URDF and have not been checked against the physical unit.
* Gripper: 0 = closed, `grasp_close_deg` (45) or below while within `goal_radius_m` of the cube grasps it; opening past 55 deg drops it.

## Tests

```bash
.venv/bin/python -m pytest metal-sim/tests -q
```
