# Embodied rollouts (`benchflow.embodied`)

BenchFlow owns the low-level layer for embodied tasks: the robot's self-description (the **embodiment spec**), the agent-facing wire protocol and `robo` command, the trusted **episode server** (budgets, recording, judging, per-step trace), the **simulator sidecar** wiring, the **physical verifier**, and the **rollout extensions** (seeded rollouts, pass@k, dense per-step reward traces, training export). A robotics benchmark such as [Robo Use](https://github.com/benchflow-ai/robouse) keeps only what is specific to it: simulator backends, tasks, reference solutions and adapters.

```
                    BenchFlow                                           benchmark package (e.g. robouse)
 ┌──────────────────────────────────────────────────────────┐     ┌──────────────────────────────────┐
 │ benchflow.embodied                                        │     │ simulator backends (MuJoCo, ...) │
 │   spec.py      Embodiment: sensors, action groups,        │◄────│   each declares an Embodiment    │
 │                skills, budgets                            │     │ tasks (task.md + oracle)         │
 │   robo.py      agent CLI (stdlib only, copied into main)  │     │ task format: EmbodiedTaskFormat  │
 │   protocol.py  JSON-over-Unix-socket ops, legacy aliases  │     │   subclass + simulator image     │
 │   server.py    EpisodeServer: budgets, recording, judge,  │     └──────────────────────────────────┘
 │                oracle token, roles, steps.jsonl           │
 │   skills.py    built-in arm/gripper controllers           │
 │   sidecar.py   native package: compose (main + simulator),│
 │                agent image, verifier, oracle wrapper      │
 │   verifier.py  physical verifier (runs in the simulator)  │
 │   export.py / rollouts.py                                 │
 │                training export of the per-step trace and  │
 │                video index; seeded rollouts, pass@k       │
 └──────────────────────────────────────────────────────────┘
```

The rest of BenchFlow is unchanged: an embodied task is a task format that materializes an ordinary native package (see [task-formats.md](./task-formats.md)), and it runs with BenchFlow's agents, sandboxes, verifier, trial layout and trainer exports.

## 1. The embodiment spec

A backend describes the robot it simulates with one `Embodiment` object. `robo info` prints it, the episode server validates every action against it, and exports record it next to the trace. The same schema covers single arms, bimanual rigs, dexterous hands, mobile manipulators, quadrupeds, humanoids, drones, vehicles and skill-only robots.

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

### Sensors

- **Cameras**: name, mount (`world`, `wrist`, `head`, `body`), image size, and, when `calibrated`, intrinsics (`fx`, `fy`, `cx`, `cy`, `fovy_deg`), extrinsics (`position`, `rotation` world-from-camera) and a 3x4 `projection` in saved-image pixels. A camera that moves with the robot or whose calibration a task withholds has `calibrated: false` and no matrices. Calibration is computed when `robo info` is called (MuJoCo helper: `benchflow.embodied.cameras.mujoco_camera`).
- **Proprioception**: the robot's own state (joint angles, end-effector pose, base pose, velocities). Always visible.
- **State**: object and world state. Fields marked `privileged: true` are hidden in vision mode (`observation_mode: vision`): the agent sees proprioception, the task's `visible_fields` and camera images only. Requests that carry the per-episode oracle token (given only to the reference solution) still see everything. The spec lists the documented fields; `observe` may return more in state mode.

### Action groups

An action group is a named channel: `name`, `components` (so its dimension is `len(components)`), `low` / `high` per component, `units`, control `mode`, `frame`, and a `hold` policy that says what the channel does in a step where the agent does not command it: `zero` (send zeros, e.g. velocity or delta channels), `last` (repeat the last command, e.g. a gripper or a position target) or `value` (send `hold_value`, e.g. a car whose pedals mean "brake" when not commanded). `initial` is the value before the first command.

Modes (the vocabulary is open; these are the ones in use): `ee_delta_pos`, `ee_delta_rot` (rotation vector), `ee_pose`, `joint_pos`, `joint_delta`, `joint_vel`, `joint_torque`, `gripper`, `base_twist` (vx, vy, yaw rate), `velocity_setpoint`, `rotor_thrust`, `select` (a discrete selector, e.g. which arm the skills drive), `wait`.

Groups are packed into the backend's flat action vector in declaration order, so a backend keeps its native action vector and only names its slices. Examples:

| Robot | Groups |
|---|---|
| Single arm with gripper (Meta-World, tabletop, Fetch, LIBERO, robosuite) | `arm.ee_delta[3]` (+ `arm.ee_rot_delta[3]` for 7-D controllers), `gripper[1]` |
| Bimanual (ALOHA in Menagerie) | `left.ee_delta[3]`, `left.gripper[1]`, `right.ee_delta[3]`, `right.gripper[1]` |
| Dexterous hand on an arm (DexJoCo) | `arm.ee_delta[3]`, `arm.ee_rot_delta[3]`, `hand.joints[16]`; bimanual: `right.*`, `left.*` |
| Mobile manipulator | `base.twist[3]`, `arm.ee_delta[3]`, `gripper[1]` |
| Quadruped / humanoid | `base.twist[3]` (velocity command) or `joints.pos[N]` |
| Drone (Skydio X2) | `body.velocity_setpoint[3]`, `body.yaw_rate[1]` (+ `payload.release[1]`) |
| Skill-only (BEHAVIOR) | `base.wait[1]`; everything else is a skill |

### Skills

A skill is a named, typed operation that may run many simulator steps: `name`, `aliases`, `args` (each `name`, `type` in `float`, `int`, `str`, `bool`, `object`, `enum`, with `units`, `choices`, `optional`, `default`), `preconditions` (human-readable), `doc`, and `impl`: `builtin` (a controller in `benchflow.embodied.skills`, driven through the action groups) or `backend` (the backend's `start_skill` or `run_skill`). A backend with `start_skill(name, args)` writes each skill as a closed-loop controller: a generator that yields one full action per control step and returns a result dict; the server runs every yielded action as an ordinary step (budget, trace line `skill:<name>`, video frame, success check), so a skill never moves the robot in a way `act` could not. Every simulator step a skill runs counts against the step budget; a backend skill that runs off-simulator (`run_skill`, e.g. BEHAVIOR's symbolic primitives) counts as one step.

Built-in controllers need one capability from the backend, `ee_position(arm)`, and command either the action groups named in the skill's `binds` or, when the backend has one, its `ee_command(arm, delta, grip)` (a backend-native end-effector command; Robo Use uses it to keep its skills' exact protocol-1 behaviour):

- `arm.move_to(x, y, z, [grip])` (alias `move_to` / `move-to`), and per-arm `left.move_to`, `right.move_to` for bimanual rigs. Proportional servo on the arm's `*.ee_delta` group; stops when within `tol`, when the end effector stalls, or at `max_steps`.
- `gripper.set(value, [steps])` (alias `grip`): hold the arm, command the gripper group for `steps` steps. A hand without a gripper channel exposes the same controller as `arm.grasp(value, [steps])` (DexJoCo's grasp preset, commanded through the backend's `ee_command`).

Examples of backend skills: `base.navigate_to(obj)`, `grasp(obj)`, `place_on_top(obj)`, `toggle_on(obj)` (BEHAVIOR), and, on the built-in side, `body.move_to(x, y, z)` for the drone (the reach controller on its velocity setpoint, with the predicted stopping point as `ee_position`).

### Budgets

`max_steps` (simulator steps, skills included), `max_wall_s` (wall clock from simulator start; on the sidecar it is the agent timeout plus a margin, since the clock starts before the harness is installed), `max_repeat` (the most steps one `act --repeat` may run), `max_skill_steps`, `settle_steps` (steps held still after `robo done` before judging). Budgets are enforced by the episode server, never by the agent. When the step budget runs out the episode ends as `budget_exhausted`, which scores 0 in `final` mode (a `robo done` is required) and scores `success_ever` in `first` mode.

### Sim, real and hardware-in-the-loop

`mode` places an embodiment on the sim/real axis: `sim` (a simulator, the default), `real` (live hardware) or `hil-mock` (a hardware-in-the-loop mock that replays sessions recorded on the real robot behind the vendor SDK). A `real` or `hil-mock` embodiment must declare its `safety` envelope, which the backend enforces on the trusted side and the spec reports: `joint_limits`, `max_joint_speed`, `workspace` (table height and box), `estop` (how the e-stop is engaged), `attended` (a human operator gates arming and scene resets), `operator_channel` and `temp_stop_c`. The episode record carries `embodiment_mode`, and the seed report groups pass rates by it. Robo Use's real-arm backend (Metal, SO-101, Piper) is the reference implementation.

```json
{"name": "MakerMods Metal arm", "kind": "arm", "mode": "hil-mock",
 "safety": {"joint_limits": {"low": [-159, -179, 1, -122, -84, -144, 0], "high": [159, -1, 179, 80, 84, 144, 115]},
            "max_joint_speed": 15, "workspace": {"table_z": -0.0088, "x": [0.05, 0.62], "y_abs_max": 0.35, "z_max": 0.55},
            "estop": "latched file (robouse estop)", "attended": true, "operator_channel": "script", "temp_stop_c": 65}}
```

## 2. The agent protocol and the `robo` command

The agent container holds one file, `robo` (`benchflow/embodied/robo.py`, standard library only). It sends one JSON request per connection to the episode socket (`$ROBO_SOCKET`, or the legacy `$ROBOUSE_SOCKET`).

```
robo info                                   the embodiment spec, budgets, observation mode, camera calibration
robo status                                 finished?, outcome, steps used
robo observe [--image] [--camera C|all]     state; --image saves camera PNGs and prints their paths
robo act GROUP=V1,V2,... [GROUP=...] [--repeat N]
                                            one step (or N) commanding one or more action groups;
                                            groups not named follow their hold policy
robo act V1 V2 ... [--repeat N]             the whole flat action vector (legacy form)
robo skill NAME [ARG ...] [key=value ...]   a named skill from `robo info`
robo done ["message"]                       end the episode and ask for scoring
robo give-up ["message"]                    end the episode without claiming success
robo move-to X Y Z [--grip G] [--max-steps N] [--tol M]
                                            alias of `robo skill arm.move_to` (kept for published tasks)
robo grip G [--steps N]                     alias of `robo skill gripper.set`
```

Every command takes `--json` for the raw response. Environment: `ROBO_SOCKET` (or `ROBOUSE_SOCKET`), `ROBO_ROLE`, `ROBO_ORACLE_TOKEN` (reference solutions only), `ROBO_TIMEOUT_S` (client timeout, default 120), `ROBOUSE_TEXT_ONLY` (a harness whose model cannot read images: `--image` is ignored).

Examples: `robo act arm.ee_delta=0.2,0,-0.1 gripper=1 --repeat 5`, `robo act left.ee_delta=0,0.3,0 right.gripper=-1`, `robo act hand.joints=0,0.2,0.2,0.2,0,0.2,0.2,0.2,0,0.2,0.2,0.2,0.5,0,0,0`, `robo skill base.navigate_to fridge_1`, `robo skill arm.move_to 0.1 0.6 0.2 grip=1`.

Wire protocol (version 2). Requests: `{"op": "info"}`, `{"op": "status"}`, `{"op": "observe", "image": bool, "camera": str|null}`, `{"op": "act", "groups": {name: [numbers]}, "repeat": n}` or legacy `{"op": "act", "action": [numbers], "repeat": n}`, `{"op": "skill", "name": str, "args": [..] | {..}}`, `{"op": "done"|"give_up", "text": str}`. Legacy ops `move_to` and `grip` map onto the skills above (arguments the skill does not declare, such as `max_steps` on a skill without one, are dropped). `{"op": "shutdown"}` is the verifier's: it ends a running episode as `agent_exited`, the same as the agent stopping without `robo done`. Optional request fields: `role` (the `ROBO_ROLE` of the caller, see scenes below) and `token` (the oracle token; never written to the trace). Responses are `{"ok": true, "result": {...}}` or `{"ok": false, "error": "..."}`; a response to a step-taking request that ended the episode carries `result.episode = "finished: <why>"`.

Backward compatibility: every command and request of the protocol-1 `robo` still works (flat `robo act`, `move-to`, `grip`, `skill`), and `robo info` still returns the protocol-1 keys (`action.names/low/high/doc`, `skills` as names, `steps_used`, `max_steps`, `observation_mode`, `cameras` in vision mode) next to the new `embodiment` key. Published task instructions do not change.

## 3. What runs where

A materialized embodied package (written by an `EmbodiedTaskFormat` subclass):

```
<out_root>/runtime-<hash>/            simulator image build context (benchmark-specific Dockerfile + its code
                                      + a vendored copy of benchflow/embodied under benchflow_embodied/)
<out_root>/tasks/<key>/<task-id>/
  task.md                             schema 1.3; the format block moves to metadata.<format>;
                                      verifier.service: simulator
  environment/Dockerfile              agent image: python + numpy + the `robo` client (and curl/xz for harness installs)
  environment/robo                    benchflow/embodied/robo.py
  environment/docker-compose.yaml     main + trusted simulator (network none, episode volume, socket volume)
  verifier/{verifier.md,test.sh}      the physical verifier; runs in simulator
  oracle/solve.sh (+ oracle/vendor/)  reference solution, or the noop control
```

- **Episode server** (`benchflow.embodied.server`): owns one simulator instance through the `SimBackend` protocol, answers requests, enforces budgets, records video and the per-step trace, and judges success itself when the episode ends. It never trusts the agent: success is computed from the simulator state after a settle period (or at the first success signal, `success_mode: first`), and a backend `judge(outcome, text)` hook covers tasks where the right ending is a refusal.
- **Simulator sidecar** (`benchflow.embodied.sidecar`): the compose topology. `main` (agent) mounts the socket volume read-only and has no simulator, task source or episode record; `simulator` runs the episode server with `network_mode: none`, keeps the record on a volume only it mounts, and is where BenchFlow runs the verifier (`verifier.service: simulator`). BenchFlow's host mount of `/logs/verifier` is removed from `main`, so the agent cannot pre-write rewards.
- **Physical verifier** (`benchflow.embodied.verifier`): closes the episode if the agent exited without `robo done` (outcome `agent_exited`, judged after the settle period), waits for the record, copies it to `/logs/verifier/episode/`, writes `reward.txt`, `reward.json` (`reward`, `success_ever`, `budget_used`) and `reward-details.json`, and copies the video and the camera images the agent saw to the trial's `artifacts/`. A missing episode result is an infrastructure error, not a zero.

The simulator image does not install BenchFlow: the modules it runs (`serve`, `server`, `skills`, `spec`, `protocol`, `backend`, `verifier`, `cameras`) import only the standard library and numpy (Pillow and imageio for images and video when installed; a stdlib PNG writer otherwise), and `sidecar.vendor_embodied(dst)` copies the package into the build context under a stub `benchflow/__init__.py`. The host-side modules (`sidecar`, `rollouts`, `export`) also use PyYAML and BenchFlow's result helpers.

### Backend contract (`benchflow.embodied.backend.SimBackend`)

Required: `embodiment() -> Embodiment`, `reset(seed)`, `step(flat_action) -> StepResult(success, reward, info)`, `observe() -> dict`, `render(camera=None) -> HxWx3 uint8`, `success() -> bool`. Optional: `ee_position(arm)` (built-in arm skills), `start_skill(name, args)` (closed-loop skills, see above), `run_skill(name, args)`, `pop_frames()` (frames a backend skill rendered), `hold_action()` (the "hold still" action, instead of the group hold policies), `judge(outcome, text)` (a judge that raises scores 0 and is recorded as `judge_error` in result.json; a backend attribute `last_judge` is recorded as `judge_detail`), `camera_calibration(name, width, height)`, `workspace_images()`, `close()`.

## 4. Episode record and rollout extensions

Files the episode server writes (the verifier copies them to `verifier/episode/` of the trial):

| File | Content |
|---|---|
| `result.json` | `success`, `outcome` (`done`, `gave_up`, `success_reached`, `budget_exhausted`, `wall_time_exhausted`, `agent_exited`), `steps_used`, `max_steps`, `seed`, `return` (sum of per-step rewards), `success_ever`, `initial_state_sha256`, `agent_text` |
| `episode.json` | header: protocol, task, seed, the embodiment spec, observation mode, visible fields, recording camera and fps, the initial state |
| `steps.jsonl` | one line per simulator step: `i` (simulator-step index), `step` (budget steps used so far; settle lines repeat the last), `t`, `op` (`act`, `skill:<name>`, `settle`), `role`, `action` by group (a backend skill logs `{"skill": [args]}`), `reward`, `success`, `info` (the backend's step info), `state` after the step, `frame` (video frame index, when one was recorded) |
| `trace.jsonl` | every request and response |
| `frames.jsonl`, `video_index.json`, `recording.mp4` | the video and, per frame, wall time, budget steps and simulator steps completed when it was rendered |

**Dense rewards.** `reward` per step is the backend's shaped reward when it has one (`reward.dense: shaped`, e.g. Meta-World, Gymnasium-Robotics, robosuite), otherwise the sparse success indicator (`reward.dense: sparse`), probed every step. The final reward (`reward.txt`) stays the judged success.

**Seeded rollouts.** `bench eval run --seeds 0-4` (or `0,3,7`) runs every task once per seed (`--include` / `--exclude` match the source folder names; `--seeds` cannot be combined with `--matrix` or `--worker-concurrency`). A task format that implements `materialize_variant(task_dir, out_root, seed=...)` writes one package per seed, named `<task>--seed-<n>`; the seed reaches the simulator through the package, so a rollout is reproducible from its folder alone, and `result.json` records `initial_state_sha256` so two rollouts with the same seed can be checked for identical resets. `summary.json` then carries `seeded`: per base task the rewards by seed, mean, standard deviation, unbiased pass@k for k = 1..n, a 95% Wilson interval on the pass rate and median / max / min reducers; over tasks the mean reward with a 95% bootstrap interval, the pass rate with its Wilson interval, and pass rates by `embodiment_mode` (sim / real / hil-mock). `bench embodied report JOB_DIR` prints the same table for any job.

**Training export.** For every trial with an episode record, BenchFlow writes next to its other trainer artifacts (`trainer/atif.json`, `trainer/verifiers.jsonl`, `trainer/adp.jsonl`):

- `trainer/embodied_steps.jsonl`: one row per simulator step, `(obs, action, reward, success, done, truncated, op, role, frame, t)`, where `obs` is the state before the step, filtered to what the agent could see in vision mode. The next observation is the next row's `obs`; the last row carries `final_obs`. Settle steps (`op: settle`, the hold after `robo done`) are included and marked, so they can be dropped for behaviour cloning;
- `trainer/embodied_episode.json`: the embodiment spec, the video index (relative path to `verifier/episode/recording.mp4`, fps, frame to step map), the return, the outcome and the ATIF session id, so a transition can be joined to the agent step (the `robo` call) that produced it.

`bench embodied export JOB_DIR --out DIR` collects them into `DIR/steps.jsonl` + `DIR/episodes.jsonl` for a whole job. `bench embodied check-spec FILE` validates an embodiment spec (or a saved `robo info --json` response), and `bench embodied robo-path` prints the path of the standalone `robo` script.

## 5. Scenes: several roles on one robot

BenchFlow scenes run their roles in the same sandbox, so every role reaches the same simulator sidecar and the episode persists across turns. A role names itself to the episode server with `ROBO_ROLE` (set it in `Role.env`); the server records the role on every trace line and step, and a task may restrict what each role can do:

```yaml
robouse:               # the format block of the task
  roles:
    planner: {allow: [info, observe, status]}
    operator: {allow: all}
```

```python
Scene(name="plan-then-act",
      roles=[Role(name="planner", agent="codex", model="gpt-6-astra", env={"ROBO_ROLE": "planner"}),
             Role(name="operator", agent="codex", model="gpt-6-astra", env={"ROBO_ROLE": "operator"})],
      turns=[Turn(role="planner", prompt="Inspect the scene with robo observe --image and write a plan to /app/plan.md. Do not move the robot."),
             Turn(role="operator", prompt="Carry out /app/plan.md with robo, then call robo done.")])
```

Role restrictions are a guard rail and an attribution record, not a security boundary: roles share a container, so a role can change its own environment.

## 6. Status and limits

- **Snapshot / restore** of simulator state (for `Rollout.branch()`) is not implemented. The episode server has what it needs (a backend `get_state` / `set_state` pair on MuJoCo is small), but branching an embodied rollout also has to fork the video and trace.
- The simulator runs on the Docker sandbox only (the verifier-in-sidecar needs `verifier.service`).
- Simulators that need a GPU or a remote worker (BEHAVIOR on Isaac Sim) keep their worker outside the sidecar; the sidecar holds the thin client.
