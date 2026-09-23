# Physical Metal arm benchmark

Docker trials use a fresh BenchFlow sandbox and ACP session. Host trials
(`--backend host`) launch a fresh native Codex/Claude Code CLI session in a
new temporary workspace using the operator's existing host login.
Codex/Claude Code directly observe and command the real arm. A host sidecar
records both cameras, and a per-trial bridge routes end-effector poses into
the commissioned Metal harness IK solver. No ROS or simulator is involved.

Branch: `physical-robot-benchmark`, based on local `main` at `a0b16985`.
The existing SDK already supports both agents and Docker; this extension is
`benchflow.robotics`, with no changes to the core agent loop.

## Trial contract

1. Operator resets the physical setup and assigns a unique reset ID. A new
   container cannot reset real cups/blocks. Use marked positions or a jig,
   fixed lighting/calibration, and matched setups for both agents.
2. Host acquires one lease per arm; starts both camera recordings; verifies
   initial status and captures both views. Do not use another operator CLI
   on the same arm during a trial. The lease excludes benchmark runners,
   not a human with direct access to the original harness socket.
3. BenchFlow builds a fresh container with the prompt, a small command
   client, and public setup facts. Host logs, past trials, conversation
   history, successful grasp poses and grading files are never mounted.
4. Agent receives a temporary bridge capability. Only `tip`, wrist roll,
   gripper, rest, observation and encoder-monitor commands are accepted.
   Images must be inspected after every motion. The host harness retains
   all calibration, floor, speed, force, and trajectory checks.
5. On completion/error/timeout, the bridge closes before final host
   observation. Recordings finalize independently; raw MJPEG is retained
   for recovery even if MP4 export or the runner fails.
6. Host reviewer scores camera evidence after the agent loses access.
   Agent self-reported success never creates reward. Native BenchFlow
   verification is deliberately deferred (`skip_verify=True`); its original
   result remains unchanged. Physical `assessment.json`/`reward.txt` and
   the combined report are the benchmark's authoritative outcome.

This prototype has no automated physical reset, automatic visual judge,
or hardware emergency stop. Creating a trial `STOP` file rejects subsequent
commands. It cannot interrupt a command already executing on the arm; use
the bench's physical stop/power control for that. A timed-out transport is
ambiguous and freezes control rather than repeating a possible motion.

## Setup and commands

To run everything on the host, append `--backend host` to `probe` or `run`.
Docker and a Docker-reachable bind address are then unnecessary; the bridge
uses loopback. Native CLI JSONL, stderr, final response, timings and reported
usage are kept with the same camera/encoder artifacts. Host sessions are
fresh but do not isolate the host filesystem. Analyze them separately from
container trials (`backend` is recorded). Codex subscription runs may report
tokens without a dollar cost; this remains unknown, not zero. Claude host
mode requires `claude auth login` or working provider credentials.

```bash
uv run python -m benchflow.robotics probe --backend host \
  --setup benchmarks/metal-real/setup.example.json \
  --task benchmarks/metal-real/tasks/sort-blue-left \
  --output /home/lapis/benchflow/physical-trials \
  --reset-id host-vision-only --operator operator-name \
  --agent codex --model gpt-6-astra --reasoning-effort max
```

After a photographed physical reset, use `run --backend host --execute`
with the same arguments and a fresh reset ID to enable manipulation.

Host dependencies: this checkout (`uv sync --extra dev --locked`), Docker
Engine/Desktop with Compose v2, ffmpeg with libx264, and the already-running
Metal operator service and its two HTTP JPEG feeds. The host Python adapter
uses the standard library and does not open USB devices. Docker containers
are Linux; Docker Desktop supplies these on macOS/Windows, but real hardware
operation on those hosts has not been validated here.

```bash
uv run python -m benchflow.robotics init benchmarks/metal-real/tasks
uv run bench tasks check benchmarks/metal-real/tasks/sort-blue-left
uv run python -m benchflow.robotics doctor --env-file /private/path/keys.env

# Standalone footage during direct control, independent of BenchFlow/Docker:
# Run in a separate terminal; Ctrl-C finalizes both MP4s. Raw frames survive
# a recorder interruption. The output directory must not already exist.
uv run python -m benchflow.robotics record \
  --setup benchmarks/metal-real/setup.example.json \
  --output /home/lapis/benchflow/direct-control-recording

# Read-only actual-arm smoke: exercises HTTP bridge, both cameras and video
# finalization. No model request and no motion, including no gripper change.
uv run python -m benchflow.robotics smoke \
  --setup benchmarks/metal-real/setup.example.json \
  --task benchmarks/metal-real/tasks/sort-blue-left \
  --output /home/lapis/benchflow/physical-trials \
  --reset-id connectivity-only --operator operator-name

# Fresh-container model/vision probe with motion disabled by the host:
uv run python -m benchflow.robotics probe \
  --setup benchmarks/metal-real/setup.example.json \
  --task benchmarks/metal-real/tasks/sort-blue-left \
  --output /home/lapis/benchflow/physical-trials \
  --reset-id vision-only --operator operator-name \
  --agent codex --model gpt-6-astra --reasoning-effort max \
  --env-file /private/path/keys.env --bind DOCKER_REACHABLE_HOST_IP

# After the operator has physically reset and checked the scene:
uv run python -m benchflow.robotics run --execute \
  --setup /private/path/setup.json \
  --task benchmarks/metal-real/tasks/sort-blue-left \
  --output /home/lapis/benchflow/physical-trials \
  --reset-id setup1-repeat1-codex --operator operator-name \
  --agent codex --model gpt-6-astra --reasoning-effort max \
  --env-file /private/path/keys.env --bind DOCKER_REACHABLE_HOST_IP

uv run python -m benchflow.robotics score /path/to/trial \
  --placement blue=left --placement green=right --placement yellow=right \
  --cups-upright yes --interventions 0 --reviewer reviewer-name \
  --evidence 'cameras/wrist.mp4 05:20–05:40; cameras/side.mp4 05:20–05:40'
uv run python -m benchflow.robotics report /home/lapis/benchflow/physical-trials --csv
```

The bridge defaults to loopback for smoke checks. Real Docker trials require
an explicitly configured Docker-reachable bind address; the client uses
`host.docker.internal` (Compose adds Linux's host-gateway mapping). Use a
private interface reachable only by the trial container; the endpoint also
requires an unpredictable per-trial token and closes at trial end. No USB,
host home directory, operator socket or Docker socket is mounted into the
agent container. A container with network access is not a security boundary
against every other host service: isolate benchmark networking before
running untrusted agents. Remote/cloud bridge exposure is not implemented.

Copy and verify `setup.example.json`; saved hardware paths are machine-local
examples, not universal calibration. Provider credentials load from the
operator environment or the specified dotenv file, never a checked-in file.
The runner passes them through BenchFlow's existing provider/telemetry
handling. Exact-model provider authentication probes must pass before a
campaign. A subscription login does not establish API access or reliable
cost telemetry. The exact comparison model IDs in `models.example.json` are
`gpt-6-astra` and `claude-fable-5-1`; account access still needs a live probe.

## Measurement and artifacts

Each uniquely named trial directory contains:

- `manifest.json`, frozen `task.md`, task/setup hashes, requested model and
  effort, operator/reset IDs and host trial timing.
- `initial-state.json`, `final-state.json`, `observations/*.jpg`, durable
  `commands.jsonl`, and the bounded slice of harness `encoders.jsonl`.
- `cameras/wrist.mp4`, `side.mp4`, original `.mjpeg` streams, per-frame
  timestamp/offset/SHA256 JSONL, recorder heartbeat and summary/error counts.
- `benchflow/`: native result, phase timing, prompts, ACP/LLM trajectories,
  sandbox identity and provider usage evidence.
- `metrics.json`: provider input/output/cache token counts, tool calls,
  cost estimate and price provenance. Unknown cost is null. Requested
  model/effort are not proof of applied settings: inspect provider trajectory.
- `assessment.json`: observed object outcomes, fraction correct, full task
  success, autonomous success, intervention count, reviewer and evidence.
  Invalid/incomplete experiments retain observations but receive no reward.

`execution_wall_s` includes container setup, agent time and final observation,
but excludes video transcoding. Native BenchFlow phase timing supplies agent
execution separately. Reset/setup labor is outside this timer; reset IDs
identify that boundary. Command log timestamps isolate control/physical
action latency. Videos play at the configured 5 fps; frame timestamps are
authoritative if capture jitter or dropped frames changes playback timing.
Neither 5 fps footage nor encoder sampling proves absence of fast vibration.

## Experiment plan

Start with one container connectivity/vision probe per exact model, with
motion disabled, then a supervised pilot on the same marked setup. Inspect
usage, model/effort actually applied, both videos and reset evidence before
collecting benchmark trials. Pilot/debugging trials are not benchmark scores.

Use paired setup/task/repetition blocks with randomized model order. Begin
with three repetitions per model/task/setup; expand for uncertainty estimates.
`plan` creates a reproducible schedule, not an automatic physical reset loop:

```bash
uv run python -m benchflow.robotics plan \
  --models /private/path/confirmed-models.json \
  --tasks sort-blue-left sort-green-left pick-yellow \
  --setups metal-two-cups-v1 --repetitions 3 --seed 42 > schedule.json
```

Report per-model autonomous success and per-object accuracy with trial
counts, intervention rate, time, cost and tokens. Separate successful-trial
latency from all-attempt cost. Report infrastructure/footage failures rather
than silently omitting them or counting them as manipulation mistakes.
No automatic reruns on real hardware. Keep container image IDs, harness
revision, calibration and model settings fixed per campaign. Current task
image uses a version tag; pin its resolved digest for a published campaign.
Artifacts remain local until a separate publishing decision is made.

Codex supports noninteractive execution and structured output; BenchFlow
uses its existing ACP integration rather than a replacement agent loop:
https://learn.chatgpt.com/docs/non-interactive-mode
Requested Astra model reference: https://developers.openai.com/api/docs/models/gpt-6-astra
Requested Fable model reference: https://platform.claude.com/docs/en/models/fable-5-1/overview
