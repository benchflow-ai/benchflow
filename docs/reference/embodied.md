# Embodied reference

This page is for authors of simulator backends and embodied task formats. To run embodied tasks, start with [Embodied tasks](../tasks/embodied.md).

The code is in `benchflow.embodied`:

| Module | Holds |
|---|---|
| `spec.py` | `Embodiment`: sensors, action groups, skills, budgets, safety |
| `robo.py` | the agent's `robo` client (standard library only, copied into the agent image) |
| `protocol.py` | the JSON-over-Unix-socket requests, older forms included |
| `server.py` | the episode server: budgets, recording, judging, oracle token, roles |
| `skills.py` | built-in arm and gripper controllers |
| `sidecar.py` | the task package: compose file, agent image, verifier, oracle wrapper |
| `verifier.py` | the physical verifier, which runs in the simulator container |
| `export.py`, `rollouts.py` | training export, seeded rollouts and pass@k |

## The embodiment spec

A backend describes its robot with one `Embodiment` object. `robo info` prints it, the episode server checks every action against it, and exports save it next to the trace. The schema covers arms, bimanual rigs, hands, mobile robots, legged robots, drones, vehicles and skill-only robots.

The spec for a Sawyer arm with a two-finger gripper:

```json
{
  "spec_version": "1",
  "name": "sawyer-2f",
  "kind": "arm",
  "step_s": 0.0125,
  "action_groups": [
    {"name": "arm.ee_delta", "components": ["dx", "dy", "dz"], "low": [-1, -1, -1], "high": [1, 1, 1],
     "units": "x 1 cm per step", "mode": "ee_delta_pos", "frame": "world", "hold": "zero"},
    {"name": "gripper", "components": ["grip"], "low": [-1], "high": [1], "units": "normalized",
     "mode": "gripper", "hold": "last", "initial": [-1]}
  ],
  "sensors": {
    "cameras": [{"name": "corner", "mount": "world", "width": 320, "height": 320, "calibrated": true}],
    "proprioception": [{"name": "hand_pos", "shape": [3], "units": "m", "frame": "world"},
                       {"name": "gripper_open", "shape": [], "units": "fraction"}],
    "state": [{"name": "obj1_pos", "shape": [3], "units": "m", "privileged": true}]
  },
  "skills": [
    {"name": "arm.move_to", "aliases": ["move_to", "move-to"], "impl": "builtin",
     "binds": {"arm": "arm.ee_delta", "gripper": "gripper"},
     "args": [{"name": "x", "type": "float", "units": "m"}, {"name": "y", "type": "float", "units": "m"},
              {"name": "z", "type": "float", "units": "m"},
              {"name": "grip", "type": "float", "optional": true},
              {"name": "max_steps", "type": "int", "optional": true, "default": 100},
              {"name": "tol", "type": "float", "units": "m", "optional": true, "default": 0.01}],
     "preconditions": ["skills are enabled for the task"],
     "doc": "Servo the end effector toward a point; stops when reached, stalled or out of budget."}
  ],
  "budgets": {"max_steps": 500, "max_repeat": 50, "max_skill_steps": 150, "settle_steps": 10},
  "reward": {"dense": "shaped", "success_mode": "first"}
}
```

Check a spec, or a saved `robo info --json` response:

```bash
bench embodied check-spec spec.json
```

### Sensors

| Sensor | Holds |
|---|---|
| **Cameras** | a name, a mount (`world`, `wrist`, `head` or `body`) and an image size |
| **Proprioception** | the robot's own state: joint angles, end-effector pose, base pose, velocities. Always visible. |
| **State** | object and world state |

A calibrated camera (`calibrated: true`) also gives intrinsics (`fx`, `fy`, `cx`, `cy`, `fovy_deg`), extrinsics (`position`, and `rotation` from camera to world) and a 3x4 `projection` matrix in the pixels of the saved images. A camera that moves with the robot, or that a task keeps uncalibrated, has `calibrated: false` and no matrices. `robo info` computes calibration when it runs. For MuJoCo, use `benchflow.embodied.cameras.mujoco_camera`.

In vision mode (`observation_mode: vision`), the server hides state fields marked `privileged: true`. The agent sees proprioception, the task's `visible_fields` and camera images. The reference solution gets a per-episode oracle token and sees everything. In state mode, `observe` can return fields beyond the ones the spec lists.

### Action groups

An action group is a named control channel. Its size is the number of its `components`. Each group also has `name`, `low` and `high` per component, `units`, a control `mode`, a `frame`, and an `initial` value used before the first command.

`hold` says what the channel does on a step where the agent does not command it:

| `hold` | Sends | Use for |
|---|---|---|
| `zero` | zeros | velocity and delta channels |
| `last` | the last command | grippers, position targets |
| `value` | `hold_value` | a car whose pedals mean "brake" when idle |

Modes in use (the list is open): `ee_delta_pos`, `ee_delta_rot` (a rotation vector), `ee_pose`, `joint_pos`, `joint_delta`, `joint_vel`, `joint_torque`, `gripper`, `base_twist` (vx, vy, yaw rate), `velocity_setpoint`, `rotor_thrust`, `select` (a discrete selector, such as which arm the skills drive) and `wait`.

The server packs groups into the backend's flat action vector in the order they are declared. A backend keeps its native action vector and only names its slices:

| Robot | Groups |
|---|---|
| Single arm with gripper (Meta-World, tabletop, Fetch, LIBERO, robosuite) | `arm.ee_delta[3]` (+ `arm.ee_rot_delta[3]` for 7-D controllers), `gripper[1]` |
| Bimanual (ALOHA in Menagerie) | `left.ee_delta[3]`, `left.gripper[1]`, `right.ee_delta[3]`, `right.gripper[1]` |
| Dexterous hand on an arm (DexJoCo) | `arm.ee_delta[3]`, `arm.ee_rot_delta[3]`, `hand.joints[16]`; bimanual: `right.*`, `left.*` |
| Mobile manipulator | `base.twist[3]`, `arm.ee_delta[3]`, `gripper[1]` |
| Quadruped or humanoid | `base.twist[3]` (velocity command) or `joints.pos[N]` |
| Drone (Skydio X2) | `body.velocity_setpoint[3]`, `body.yaw_rate[1]` (+ `payload.release[1]`) |
| Skill-only (BEHAVIOR) | `base.wait[1]`; everything else is a skill |

### Skills

A skill is a named, typed operation that can run many simulator steps, such as "move the gripper to x, y, z". A skill has `name`, `aliases`, `args`, `preconditions` (human-readable), `doc` and `impl`. Each arg has a `name` and a `type` (`float`, `int`, `str`, `bool`, `object` or `enum`), plus optional `units`, `choices`, `optional` and `default`.

| `impl` | Runs |
|---|---|
| `builtin` | a controller in `benchflow.embodied.skills` that drives the action groups |
| `backend` | the backend's own `start_skill` or `run_skill` |

A backend's `start_skill(name, args)` is a generator. It yields one full action per control step and returns a result dict. The server runs each yielded action as an ordinary step, with the budget check, a trace line `skill:<name>`, a video frame and a success check. So a skill can never move the robot in a way `act` could not.

Every simulator step a skill runs counts against the step budget. A skill that runs outside the simulator (`run_skill`, such as BEHAVIOR's symbolic actions) counts as one step.

Built-in controllers need one method from the backend, `ee_position(arm)`. They command the action groups named in the skill's `binds`. If the backend has `ee_command(arm, delta, grip)`, its native end-effector command, they use that instead. Robo Use does this to keep its original skill behaviour.

| Built-in skill | Alias | Does |
|---|---|---|
| `arm.move_to(x, y, z, [grip])`; `left.move_to`, `right.move_to` on bimanual rigs | `move_to`, `move-to` | servoes the arm's `*.ee_delta` group; stops within `tol`, when the end effector stalls, or at `max_steps` |
| `gripper.set(value, [steps])` | `grip` | holds the arm and commands the gripper group for `steps` steps |
| `arm.grasp(value, [steps])` | | the same controller for a hand without a gripper channel (DexJoCo's grasp preset, through `ee_command`) |

Backend skills include `base.navigate_to(obj)`, `grasp(obj)`, `place_on_top(obj)` and `toggle_on(obj)` (BEHAVIOR). The drone's `body.move_to(x, y, z)` is the built-in reach controller on its velocity setpoint, with the predicted stopping point as `ee_position`.

### Budgets

| Budget | Limits |
|---|---|
| `max_steps` | simulator steps, skill steps included |
| `max_wall_s` | wall-clock seconds from simulator start. In the simulator container it is the agent timeout plus a margin, because the clock starts before the agent is installed. |
| `max_repeat` | steps one `robo act --repeat` may run |
| `max_skill_steps` | steps one skill may run |
| `settle_steps` | steps held still after `robo done`, before judging |

### Safety

A `real` or `hil-mock` robot must declare a `safety` block. Its backend enforces it outside the agent's container; the spec reports it so agents, graders and reports can see the limits.

| Field | Meaning |
|---|---|
| `joint_limits` | `{"low": [...], "high": [...]}` per joint, in the joint action group's units |
| `max_joint_speed` | the speed cap, in those units per second |
| `workspace` | table height and the allowed box |
| `estop` | how the e-stop is engaged |
| `attended` | a human operator arms the robot and resets the scene |
| `operator_channel` | `tty`, `file` or `script` (mock only) |
| `temp_stop_c` | the motor-temperature stop, in °C |

```json
{"name": "MakerMods Metal arm", "kind": "arm", "mode": "hil-mock",
 "safety": {"joint_limits": {"low": [-159, -179, 1, -122, -84, -144, 0], "high": [159, -1, 179, 80, 84, 144, 115]},
            "max_joint_speed": 15, "workspace": {"table_z": -0.0088, "x": [0.05, 0.62], "y_abs_max": 0.35, "z_max": 0.55},
            "estop": "latched file (robouse estop)", "attended": true, "operator_channel": "script", "temp_stop_c": 65}}
```

Robo Use's real-arm backend (Metal, SO-101, Piper) is the reference implementation.

## The `robo` command

The agent container holds one file, `robo` (`benchflow/embodied/robo.py`, standard library only). It sends one JSON request per connection to the episode socket. [Embodied tasks](../tasks/embodied.md#the-agents-commands) lists the commands. These older forms still work, so published task instructions do not change:

```
robo act V1 V2 ... [--repeat N]                      the whole flat action vector
robo move-to X Y Z [--grip G] [--max-steps N] [--tol M]   same as robo skill arm.move_to
robo grip G [--steps N]                              same as robo skill gripper.set
```

`robo info` also still returns the older keys (`action.names/low/high/doc`, `skills` as names, `steps_used`, `max_steps`, `observation_mode`, and `cameras` in vision mode) next to `embodiment`.

| Variable | Meaning |
|---|---|
| `ROBO_SOCKET` (or the older `ROBOUSE_SOCKET`) | the episode socket |
| `ROBO_ROLE` | the caller's role in a scene |
| `ROBO_ORACLE_TOKEN` | the oracle token; reference solutions only |
| `ROBO_TIMEOUT_S` | the client timeout in seconds (default 120) |
| `ROBOUSE_TEXT_ONLY` | set it for a model that cannot read images; `--image` and `--camera` are then ignored |

To find the standalone `robo` script on your machine:

```bash
bench embodied robo-path
```

## Wire protocol (version 2)

| Request | Fields |
|---|---|
| `{"op": "info"}` | |
| `{"op": "status"}` | |
| `{"op": "observe", ...}` | `image` (bool), `camera` (string or null) |
| `{"op": "act", ...}` | `groups` (`{name: [numbers]}`), or the older `action` (`[numbers]`); `repeat` |
| `{"op": "skill", ...}` | `name`, `args` (a list or an object) |
| `{"op": "done"}`, `{"op": "give_up"}` | `text` |
| `{"op": "shutdown"}` | the verifier's: ends a running episode as `agent_exited` |

Any request may carry `role` (the caller's `ROBO_ROLE`) and `token` (the oracle token, never written to the trace). The older ops `move_to` and `grip` map onto the built-in skills; arguments the skill does not declare are dropped.

A response is `{"ok": true, "result": {...}}` or `{"ok": false, "error": "..."}`. A step that ended the episode carries `result.episode = "finished: <why>"`.

## Task format and package

An embodied task format (a subclass of `benchflow.embodied.sidecar.EmbodiedTaskFormat`) writes this folder for each task:

```
<out_root>/runtime-<hash>/            simulator image build context: the benchmark's Dockerfile and code,
                                      plus a copy of benchflow/embodied under benchflow_embodied/
<out_root>/tasks/<key>/<task-id>/
  task.md                             schema 1.3; the format block moves to metadata.<format>;
                                      verifier.service: simulator (the verifier runs in the simulator container)
  environment/Dockerfile              agent image: python, numpy and the robo client (curl/xz for agent installs)
  environment/robo                    benchflow/embodied/robo.py
  environment/docker-compose.yaml     main + the simulator (network none, episode volume, socket volume)
  verifier/{verifier.md,test.sh}      the physical verifier
  oracle/solve.sh (+ oracle/vendor/)  the reference solution, or the noop control
```

A format that implements `materialize_variant(task_dir, out_root, seed=...)` writes one package per seed for `--seeds`.

The sidecar format keeps a source task's own `metadata.embodied` and adds its variant fields: `format`, `base_task`, `seed`, `seed_override` and `noop`. The older key `metadata.embodiment` is not read: BenchFlow refuses a task that still uses it (`benchflow.embodied.spec.restore_boundary`). A finished trial's episode record (`verifier/episode/episode.json` or `trainer/embodied_episode.json`) is checked too (`benchflow.embodied.trials.trial_restore_boundary`).

### Who does what

| Part | Module | Job |
|---|---|---|
| Episode server | `benchflow.embodied.server` | Owns one simulator through the `SimBackend` protocol, answers requests, enforces budgets, records video and the per-step trace, and judges success. Success comes from the simulator state after the settle period, or at the first success signal with `success_mode: first`. A backend `judge(outcome, text)` hook covers tasks where the right ending is a refusal. |
| Simulator container | `benchflow.embodied.sidecar` | `main` mounts the socket volume read-only and has no simulator, task source or episode record. `simulator` runs the episode server with `network_mode: none`, keeps the record on a volume only it mounts, and runs the verifier. BenchFlow does not mount `/logs/verifier` into `main`. |
| Physical verifier | `benchflow.embodied.verifier` | Closes the episode if the agent exited without `robo done` (outcome `agent_exited`, judged after the settle period). Waits for the record, copies it to `/logs/verifier/episode/`, and writes `reward.txt`, `reward.json` (`reward`, `success_ever`, `budget_used`) and `reward-details.json`. Copies the video and the camera images the agent saw to the trial's `artifacts/`. A missing episode result is an infrastructure error, not a zero. |

The simulator image does not install BenchFlow. The modules it runs (`serve`, `server`, `skills`, `spec`, `protocol`, `backend`, `verifier`, `cameras`) import only the standard library and numpy. They use Pillow and imageio for images and video when installed, and a standard-library PNG writer otherwise. `sidecar.vendor_embodied(dst)` copies the package into the build context under a stub `benchflow/__init__.py`. The host-side modules (`sidecar`, `rollouts`, `export`) also use PyYAML and BenchFlow's result helpers.

## Backend contract

`benchflow.embodied.backend.SimBackend`:

| Method | Required | Returns or does |
|---|---|---|
| `embodiment()` | yes | the `Embodiment` |
| `reset(seed)` | yes | starts an episode from the seed |
| `step(flat_action)` | yes | `StepResult(success, reward, info)` |
| `observe()` | yes | a state dict |
| `render(camera=None)` | yes | an HxWx3 uint8 image |
| `success()` | yes | bool |
| `ee_position(arm)` | no | the end-effector position, for built-in arm skills |
| `ee_command(arm, delta, grip)` | no | the flat action for a native end-effector command; built-in skills use it instead of the groups |
| `start_skill(name, args)` | no | a generator of actions: a closed-loop skill |
| `run_skill(name, args)` | no | a skill that runs outside the simulator (one step) |
| `pop_frames()` | no | frames a backend skill rendered |
| `hold_action()` | no | the "hold still" action, instead of the groups' hold rules |
| `judge(outcome, text)` | no | the score. A judge that raises scores 0 and is recorded as `judge_error` in `result.json`; a `last_judge` attribute is recorded as `judge_detail`. |
| `camera_calibration(name, width, height)` | no | a camera's calibration |
| `privileged_state()` | no | extra state only the oracle sees, merged into `observe` |
| `workspace_images()` | no | images given to the agent at the start |
| `close()` | no | releases the simulator |

## Episode record

The episode server writes these files; the verifier copies them to the trial's `verifier/episode/`:

| File | Content |
|---|---|
| `result.json` | `success`, `outcome`, `steps_used`, `max_steps`, `seed`, `return` (the sum of per-step rewards), `success_ever`, `initial_state_sha256`, `agent_text` |
| `episode.json` | header: protocol, task, seed, the spec, observation mode, visible fields, recording camera and fps, the initial state |
| `steps.jsonl` | one line per simulator step (fields below) |
| `trace.jsonl` | every request and response |
| `frames.jsonl`, `video_index.json`, `recording.mp4` | the video and, per frame, wall time, budget steps and simulator steps done when it was rendered |

`outcome` is one of `done`, `gave_up`, `success_reached`, `budget_exhausted`, `wall_time_exhausted` or `agent_exited`.

Each `steps.jsonl` line has:

- `i`: the simulator-step index;
- `step`: budget steps used so far (settle lines repeat the last);
- `t`, `op` (`act`, `skill:<name>` or `settle`) and `role`;
- `action` by group (a backend skill logs `{"skill": [args]}`);
- `reward`, `success`, `info` (the backend's step info) and `state` after the step;
- `frame`: the video frame index, when one was recorded.

**Dense rewards.** Each step's `reward` is the backend's shaped reward when it has one (`reward.dense: shaped`; Meta-World, Gymnasium-Robotics and robosuite do). Otherwise it is 1 or 0 for success on that step (`reward.dense: sparse`). The trial's final reward in `reward.txt` is still the judged success.

**Training export.** `trainer/embodied_steps.jsonl` has one row per simulator step: `(obs, action, reward, success, done, truncated, op, role, frame, t)`. `obs` is the state before the step, filtered to what the agent could see in vision mode. The next observation is the next row's `obs`; the last row carries `final_obs`. `trainer/embodied_episode.json` holds the spec, the video index (the path to `verifier/episode/recording.mp4`, fps, and the frame-to-step map), the return, the outcome and the ATIF session id. That id joins a transition to the `robo` call that produced it.

## Next steps

- [Embodied tasks](../tasks/embodied.md): run embodied tasks with seeds and read the report.
- [Task formats](../tasks/task-formats.md): the task format interface.
- [Branching](../jobs/branching.md): why embodied tasks cannot branch yet.
