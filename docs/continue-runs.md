# Continuing timed-out runs (`bench eval continue`)

`bench eval continue` resumes a previous, **unfinished** (timed-out) agent run to
completion. It is a standalone tool — it does **not** modify benchflow's normal
`eval`/run path — and currently targets the **`openhands`** agent.

The goal is a *transparent* resume: the continued run behaves as if the original
timeout had simply been larger. The agent keeps its exact context and
environment and continues its own loop with **no injected prompt**.

> The command lives under the `eval` group (`bench eval continue`). The original
> top-level `bench continue` still works as a hidden, deprecated alias.

## The problem it solves

A finished run keeps nothing of the container unless it opted into snapshot
retention (`bench eval run --keep-snapshots`, rollout-branching RFC §3.6) —
cleanup tears the sandbox down and marks any recorded stage refs `ephemeral`
in `stage_snapshots.json`. What survives on disk otherwise is the run folder:
`config.json`, `result.json`, `prompts.json`, and
`trajectory/llm_trajectory.jsonl`. So a historical timeout has only its
*trajectory* + the *task*; there is no saved container to restore.

`bench eval continue` reconstructs the missing state from the trajectory.

## How it works — record-replay

The recorded `llm_trajectory.jsonl` is the exact sequence of LLM
request/response pairs from the original run. `bench eval continue`:

1. **Loads** the original run folder and the recorded exchanges.
2. **Boots a fresh, pristine sandbox** from the same base image.
3. Stands up a **replay proxy** that OpenHands talks to via `LLM_BASE_URL`. For
   the first *N* requests it returns the recorded responses **in order**, so the
   agent re-executes its own past decisions *for real* — rebuilding the
   byte-exact workspace and its exact internal conversation/event state.
4. When the recorded responses run out (the timeout cut-point), the proxy flips
   to the **live model** and the agent continues — no new prompt.
5. **Re-verifies** with the task verifier and writes a new HF-compatible folder,
   with a stitched `llm_trajectory.jsonl` (recorded prefix + live suffix) and
   `continued_from` provenance — a drop-in replacement for the timed-out entry.

Because the agent rebuilds its own state by re-doing its own steps, no
reverse-engineering of OpenHands internals is needed, and the result is a single
continuous run rather than a fresh agent on a warm filesystem.

## Usage

```bash
bench eval continue path/to/original/run-folder \
  --tasks-dir path/to/tasks          # where the task source (verifier) lives
```

The uploaded run folder does **not** ship the task's verifier, so point
`--tasks-dir` at the directory containing the task (matched by name). If the
`task_path` recorded in `config.json` still exists on disk, `--tasks-dir` is
optional.

### Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--tasks-dir DIR` | recorded `task_path` | Task source (instruction + verifier). |
| `--model MODEL` | original run's model | Override the **live-continuation** model. |
| `--timeout SEC` | original run's timeout | Wall-clock budget for the continuation. |
| `--output DIR` | `<orig-parent>/continued` | Output jobs dir for the new run. |
| `--require-timeout` | off | Refuse runs whose recorded status isn't a timeout. |
| `--strict-divergence` | off | Abort if replay leaves the original rails. |
| `--replay-only` | off | Rebuild via replay and stop at the cut-point (no live model needed). |
| `--max-exchanges K` | all recorded | Replay only the first K recorded exchanges, then go live ([Cut-points](#cut-points)). |
| `--cut-stage STAGE` | — | Cut at a recorded stage boundary by name, resolving K from the run's `stage_snapshots.json` ([Cut-points](#cut-points)). |

### Models and credentials

- The **live-continuation model** defaults to the original run's model so the
  continuation is a faithful continuation of the same brain. Tests use
  `--model gemini-3.1-flash-lite-preview` for a cheap path.
- The **replay phase needs no API key** — responses are served from the
  recording. Only the **live continuation** calls the real provider, so the
  host needs that provider's credentials (e.g. `GEMINI_API_KEY`) in its
  environment. `--replay-only` skips the live leg entirely.

## Cut-points

By default the proxy replays the **entire** recorded prefix before going live.
`--max-exchanges K` cuts the replay short: the first K recorded exchanges are
replayed, then the proxy switches to the live model exactly as if the recording
had ended there. This is the replay cut-point API from the
[rollout-branching RFC §3.5](./rollout-branching-rfc.md) — replay a trajectory
verbatim up to a stage boundary, then go live, to localize which stage a run
went wrong in.

- `K` must satisfy `1 <= K <= n_recorded`; anything else fails closed before a
  sandbox boots.
- The continued run's `source_provenance` gains a `cut_point` block:
  `n_replayed_exchanges` plus two request digests named by what they hash —
  `served_request_digest` (the request the agent *actually* sent at the cut)
  and `recorded_request_digest` (the recorded request it answered for), both
  sha256 over the canonical JSON of the comparable projection
  `{messages, tools}` (`request_digest_basis` states this in the artifact).
  Divergence is checked per replayed exchange on the same basis — a
  same-message-count prompt/content/tool change is detected, not only a count
  mismatch — and every event (exchange index + both digests) is recorded in
  the block's `divergences` list. A divergence annotates rather than aborts
  (fidelity caveats are recorded, not hidden — RFC §3.5); `--strict-divergence`
  remains the opt-in abort. The block also carries `workspace_digest`: a
  deterministic digest of the continuation workspace (`/app` — file contents,
  tree and modes) taken as the run crosses the cut into the live leg; when no
  live sandbox is reachable at that moment (sandbox proxy mode, a run that
  never crossed the cut, a digest failure) the field is `null` and
  `workspace_digest_reason` says why — it is never fabricated. The block's
  `accounting` field names its basis: in **host** proxy mode the orchestrator
  reconciles the block after the run with what the live replay proxy
  *actually served* (`accounting: "served"`, plus `configured_max_exchanges`
  when a cut was requested — so a run that went live before reaching the
  requested cut is visible in artifacts); in **sandbox** proxy mode the
  uploaded recording is truncated to the configured prefix and the block
  records that basis (`accounting: "configured"`, with the live-only fields
  null). A natural-end continuation records the same block, documenting the
  end of the recording.
- The stitched `llm_trajectory.jsonl` contains only the replayed prefix (the
  first K *parsed* recorded exchanges — a malformed recorded line is never
  replayed and never stitched) plus the live suffix.
- Cut-points can be named by **stage** instead of by number:
  `--cut-stage <stage>` (e.g. `--cut-stage post-research`) resolves the
  exchange index the original run recorded when that stage boundary closed.
  A run that captures stage boundaries (`RolloutConfig.snapshot_stages`, or
  `Rollout.mark_stage()` for `post-research`) records
  `exchanges_completed` per stage in its `stage_snapshots.json`; a cut at
  that stage replays exactly that many exchanges. The resolved stage is
  recorded as `branch_stage` in the `cut_point` block. Every miss fails
  closed with a typed `ReplayCutPointError`: a run with no recorded stages,
  an unrecorded stage (the error lists the stages the run *did* record), a
  stage recorded without an index (`exchanges_completed: null` — the usage
  gateway could not count at capture time), or a stage that closed before
  the first exchange. `--cut-stage` and `--max-exchanges` are mutually
  exclusive. Through the Python API
  (`benchflow.continue_run.orchestrator.continue_run`), an explicit
  `stage_tags` mapping (`stage -> 1-based completed-exchange count`)
  overrides the recorded registry.

## Limitations and caveats

- **`openhands` only** for now (the proxy seam relies on `LLM_BASE_URL`).
- **Replay fidelity is best-effort.** Replay re-runs the original shell
  commands for real; if a command's output diverges from the original
  (network, timestamps, nondeterminism), the agent may see a different
  observation than recorded. Divergence warns rather than aborts
  (`--strict-divergence` aborts instead): through the host replay proxy the
  per-exchange check compares content digests of the comparable
  `{messages, tools}` projection ([Cut-points](#cut-points)); the in-sandbox
  proxy checks message counts only.
- **"Identical output" means a faithful continuation**, not a bit-identical
  result — the model samples, and no "original full run" exists past the
  timeout. The bar is: the stitched trajectory reads as one continuous run, as
  if the timeout had been larger.
- Re-running the episode's commands costs wall-clock time (model latency is
  skipped, since recorded responses are served instantly).
