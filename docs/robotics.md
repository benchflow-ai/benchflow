# Physical robot trials

BenchFlow can let an agent control a real robot arm (`benchflow.robotics`). Each such run is a *physical trial*. A physical trial is not scored when it ends: a person reviews the recorded evidence and records an assessment. This page explains how to read a saved trial and how to record that assessment.

The commands below read or score saved trials. They never contact the robot. Run them with the Python environment that has BenchFlow installed; from a checkout, prefix them with `uv run`. The module's other subcommands (`init`, `plan`, `doctor`, `smoke`, `probe`, `run`, `record`) prepare, check and run trials on a commissioned arm and are not covered here.

## The trial directory and `trial-record.json`

Each trial writes one directory under the run's output root, named `<UTC start time>-<10 hex digits>`, for example `physical-trials/20260101T000000Z-0123456789`. What it holds depends on the backend and on how far the trial got:

- `manifest.json`: what the runner recorded: task, setup, agent and model, the time budget (`agent_timeout_s`), start and finish times, the recorded `status`, and the task's `expected` outcome.
- `commands.jsonl`: bridge events for each command the agent sent: the request, the bridge's rejection or dispatch, and the harness receipt.
- `cameras/`: per-camera frame indexes, MJPEG and MP4 video. `observations/`: the images returned to the agent.
- `encoders.jsonl` (`encoders-<arm>.jsonl` for each other arm): arm telemetry.
- The agent's trajectory: `benchflow/<job>/…/trajectory/acp_trajectory.jsonl` (Docker backend), or `agent-events.jsonl` and `host-agent-summary.json` (host backend).
- `metrics.json`: token usage, the agent's error if any, and where the usage came from.
- `assessment.json` and `reward.txt`: written by `score` (below).
- Rerun recordings (`*.rrd`, in the trial root or in `recordings/`) are indexed when present. BenchFlow does not write them.

`trial-record.json` summarizes the directory. It holds:

- `provenance`: which runtime drove the arm, the arms, the camera mapping and the controller, as declared (never guessed from names).
- `outcome.execution`: how the run ended (below). Its `evidence` object copies raw values without interpreting them: `host_agent` has the host agent's `exit_code` and `agent_wall_s` from `host-agent-summary.json` (null when the trial has none), and `agent_timeout_s` is the manifest's time budget.
- `outcome.assessment`: the reviewer's verdict (below).
- `streams`: every recorded stream with its path, digest, record count, first and last time on one UTC clock, and a `status` (`complete`, `partial`, `empty`, `untimed` or `missing`).
- `actions`: counts of requested, rejected, dispatched and receipted commands, including dispatches with no receipt and outcomes marked uncertain.
- `intact`: false when an expected stream is missing or cut short. `synchronized`: false when a present stream has no usable timestamps, and null when no stream is present.

For a scored trial (manifest `kind: physical_trial`), `result.json` beside it is the ordinary BenchFlow result, so BenchFlow's metrics and summaries see one result per trial. It carries a reward only after the trial has been assessed.

The runner writes both files when a trial ends. For a trial saved before they existed, create them with `index --write`.

## Commands

### `index`: read one trial

```
python -m benchflow.robotics index physical-trials/20260101T000000Z-0123456789
```

Prints the trial record as JSON and changes nothing in the directory. Add `--write` to write `trial-record.json` (and `result.json` for a scored trial) into it.

For the example trial, a host trial made by an older runner, the execution part reads:

```json
{
  "status": "agent_error",
  "recorded_status": "agent_error",
  "halt_reason": null,
  "pipeline": "healthy",
  "motion_dispatched": null,
  "error": null,
  "error_category": null,
  "evidence": {
    "host_agent": {
      "source": "host-agent-summary.json",
      "exit_code": 143,
      "agent_wall_s": 1801.0
    },
    "agent_timeout_s": 1800
  }
}
```

The state is `agent_error` with no error text, as that runner recorded it; the evidence shows the agent was stopped (exit code 143, SIGTERM) after 1801 s of a 1800 s budget.

### `report`: list many trials

```
python -m benchflow.robotics report physical-trials
python -m benchflow.robotics report physical-trials --csv
```

Prints one row per trial directory directly under the given folder (each `*/manifest.json`): the main manifest fields, the contents of `metrics.json`, `host-derived-metrics.json` and `assessment.json` when present, and `execution_status`, `assessment_status` and `reward`. `--csv` prints the same rows as CSV. It reads only. `execution_status` is the state `index` gives for the trial, including `no_motion`: `report` reads the command log's counts for it, but does not index cameras or recordings, so use `index` for streams and evidence.

### `score`: record an assessment

`score` writes files, so try it on a copy of the trial first.

```
python -m benchflow.robotics score physical-trials/20260101T000000Z-0123456789 \
  --placement tape=untied \
  --reviewer alice \
  --interventions 0 \
  --evidence "recordings/trial.rrd, whole run"
```

- `--placement BLOCK=LOCATION`: the observed outcome for each key of the task's `expected` block, one flag per key, exactly those keys (for example `tape=untied`, or `blue=left` for a sort task).
- `--reviewer`, `--evidence` (which files and time ranges you reviewed) and `--interventions` (how many times a person intervened, 0 or more) are required.
- `--cups-upright yes|no`: required for every task in the cup scene (`sort-blue-left`, `sort-green-left`, `pick-yellow`) and for any task whose `expected` block puts an object in a cup (`left` or `right`, the two red cups). A task outside the cup scene, such as `untie-knot`, may leave it out; an explicit `no` still fails any task.
- `--external-interruption TEXT`: an external event you observed. It makes the evidence inadmissible.

`score` writes `assessment.json`, writes `reward.txt` when the evidence is admissible, sets the manifest's `assessment` field, and rewrites `trial-record.json` and `result.json`. It refuses to replace an existing assessment.

In the assessment, `object_accuracy` is the share of placements that match `expected`; `task_success` needs all of them to match and the cups not reported fallen; `autonomous_success` also needs zero interventions. `benchmark_valid` is false when the evidence is not admissible, and `invalid_reasons` then lists every cause:

- camera footage is not complete
- external interruption: the text given with `--external-interruption`
- the final observation failed
- the trial ended in an infrastructure error
- bridge halted: `recording_lost` or `uncertain_outcome`
- model usage was not reported by the provider or agent, including a trial with no `metrics.json`
- model API error (`api_error` or `suspected_api_error`)

The example trial has no `metrics.json`, so its assessment is `unassessable` for the usage reason even when the reviewer records a success.

Operator mistakes (a wrong path, a trial that is already scored, a trial that is not a physical trial, a missing `--cups-upright` for a cup-scene task) print one line, `python -m benchflow.robotics: error: …`, and exit 1.

## Execution states

`outcome.execution.status` says how the run ended. It is set from the first matching row, top to bottom:

| State | When | `error` in `result.json` |
|-------|------|--------------------------|
| `unfinalized` | The runner never recorded a terminal state: no status, `preparing` or `running`, or no finish time. | set |
| `cancelled` | Interrupted by the operator or the process (status `interrupted`, or the operator stopped a host trial). | set |
| `infrastructure_error` | The harness around the agent failed (status `infrastructure_error` or `smoke_failed`). | set |
| `halted` | The bridge stopped accepting commands; `halt_reason` says why: `budget_exhausted`, `recording_lost`, `hardware_abort` or `uncertain_outcome`. | not set |
| `timed_out` | The agent's time budget ran out (error category `timeout` or `idle_timeout`, or "time budget exhausted"). | set |
| `agent_error` | The agent or its provider failed. Older host trials stopped at their deadline also land here; see `evidence`. | set |
| `no_motion` | Ended on its own with a `healthy` pipeline, at least one successful observation and no motion command dispatched: a refusal or a decision not to act. Only the transcript tells which. | not set |
| `completed` | The agent session ended on its own and nothing halted it. | not set |

The execution state never sets a reward; only the assessment does. When `error` is set, it is the recorded error text, or `physical trial <state>` when there is none.

`outcome.execution.pipeline` separates a stall from a broken rig:

- `capture_failure`: the bridge halted with `recording_lost`, or the footage is not complete.
- `controller_failure`: the bridge halted with `uncertain_outcome` or `hardware_abort`, the final observation failed, or the command log shows an uncertain outcome or a transport failure.
- `healthy`: the footage is complete and none of the above happened.
- `unknown`: footage completeness was never recorded, for example in an unfinalized trial.

## Assessment states

`outcome.assessment.status` is the reviewer's verdict:

| State | Meaning | Reward |
|-------|---------|--------|
| `pending` | Nobody has assessed the evidence yet (`reason: awaiting_reviewer`). | none |
| `verified` | Assessed with admissible evidence; the task succeeded autonomously. | 1.0 |
| `failed` | Assessed with admissible evidence; it did not (including a success that needed an intervention). | 0.0 |
| `unassessable` | The evidence cannot support a verdict (`evidence_not_admissible`, with `invalid_reasons`), the execution never finalized (`execution_not_finalized`), or the trial is not scored, such as a probe or smoke (`not_a_scored_trial`). | none |

Only `verified` and `failed` carry a reward. BenchFlow's shared scoring withholds the score of every other state, even if a stale `reward.txt` is present.

## Viewing trials

`bench eval view` does not render physical trials yet. Pointed at a trial directory or its parent folder, it prints the `index` and `report` commands for them and exits 1. Rendering trial records in the viewer is not built yet.
