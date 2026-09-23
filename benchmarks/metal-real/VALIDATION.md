# Initial implementation validation — 2026-09-05 Pacific

## Host execution follow-up

The operator requested host execution instead of Docker. `--backend host`
now launches fresh native CLI sessions with the same camera sidecar and IK
bridge. Docker remains optional; host mode does not isolate the filesystem.
39 adapter tests pass, including native event accounting and exclusion of
externally interrupted trials.

- Codex/Astra host probe `20260906T045953Z-f093b7502e` completed: 42.86 s
  agent time, 91,248 reported input/output tokens, both videos complete.
- Codex/Astra physical pilot `20260906T050224Z-a6ac8805df` stopped after
  observing a hand carrying a box enter the table work area. All three
  blocks remained on the table, both cups upright, gripper empty. The arm
  remains powered at the last approach pose. Agent time 512.69 s, reported
  tokens 2,851,717, including 2,608,384 cached input tokens. Both cameras
  captured 2,570 frames at 5 fps without errors; both MP4s exported.
  `assessment.json` excludes this interrupted pilot from comparison scoring.
- At the operator's request, Fable was tested next in host Claude Code:
  `20260906T051455Z-e110a26b46`. It failed authentication before any model
  request: `Not logged in · Please run /login`. No agent motion command
  occurred; both recordings completed and both MP4s decoded successfully.

All trial directories are under `/home/lapis/benchflow/physical-trials`.
Codex's native stream reports aggregate tokens but no dollar cost and may
omit some tool event types. `host-derived-metrics.json` corrects the initial
default tool-count field to unknown, supplies the observed event count and
token total, and records a Standard API-equivalent cost reference range.
Original events and metrics remain preserved; these estimates are not
subscription charges. Actual Codex cost remains unknown. Claude Code needs
host login (`claude auth login`) before its physical pilot can run.

Branch: `physical-robot-benchmark`, worktree `/home/lapis/benchflow/benchflow-physical`.
This is an SDK extension prototype, not a completed model comparison.

## Software checks

- 34 tests pass across `test_robotics_bridge.py`, `test_robotics_recording.py`,
  and `test_robotics_sdk.py`. Tests use mock hardware/providers; local HTTP
  sockets require execution outside this host's filesystem-only sandbox.
- Ruff and `ty check src/benchflow/robotics` pass.
- All three generated native tasks pass `bench tasks check`.
- Both requested model/agent configurations construct the real SDK config
  and exercise the pre-agent upload hook in tests. Mock providers preserve
  token counts and unknown cost. Probe mode rejects mock motion commands.
- No full repository test suite or real Docker/agent rollout has run.

## Actual resting-arm observation and recording

Successful trial:
`/home/lapis/benchflow/physical-trials/20260906T042100Z-e49c9abaf7`.

- Confirmed Metal service identity; initial, HTTP bridge, and final observations.
- No motion or gripper command was sent. Final images show the arm at rest.
- Both cameras: 21 frames, 1920×1080, 5 fps, zero capture errors.
- Maximum frame gaps: wrist 0.218 s, side 0.213 s (rounded upward).
- Raw MJPEG, per-frame timestamps/hashes, observations, command log and
  harness episode slice retained with the trial manifest.
- Both H.264 MP4s: 4.2 seconds; fully decoded with ffmpeg successfully.
- Capture summary and manifest mark footage and MP4 exports complete.
- Sidecar and trial bridge stopped after finalization.

The earlier trial `20260906T041433Z-353834d240` exposed camera JPEG zero
padding incorrectly rejected as corrupt frames. Its footage is incomplete
and must not be used as a successful recording check, despite its original
`smoke_passed` status. The recorder now strips trailing zero padding only;
truncated/nonzero-trailing data remains rejected. Smoke completion now
requires complete footage, successful MP4 exports, and final observations.
Any capture error also fails the recording health gate for subsequent control.

## Runtime prerequisites still outstanding

Docker is absent on this host. System installation requires a sudo password
unavailable to this session. No provider API credential variables were found
in the inspected environment/configuration. Existing Codex subscription
login does not establish container API authentication or cost telemetry.
Credential values have not been printed or stored in experiment manifests.

Once Docker and an operator-supplied credential file are available:

1. Verify each exact model with a small direct provider request.
2. Run `probe` for Codex/Astra and Claude Code/Fable with motion disabled.
3. Inspect applied model/effort, token/cost provenance, agent image viewing,
   container connectivity and both recordings.
4. Physically reset and photograph the marked scene for a supervised pilot.
5. Only then collect the paired schedule; every trial needs a fresh reset.

The standalone `record` command is available without Docker or model keys
for direct-control footage. It does not connect to or command the arm.
It was tested separately at `/home/lapis/benchflow/direct-record-smoke-20260905`:
both cameras captured 16 frames with zero errors, both MP4 exports succeeded,
and creating the recording's `STOP` file finalized the CLI with exit code 0.
