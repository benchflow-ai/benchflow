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

A finished run keeps nothing of the container — cleanup tears the sandbox down.
What survives on disk is the run folder: `config.json`, `result.json`,
`prompts.json`, and `trajectory/llm_trajectory.jsonl`. So a historical timeout
has only its *trajectory* + the *task*; there is no saved container to restore.

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

The stitched manifest preserves both capture boundary and custody. The live
suffix is observed at replay-proxy ingress, before the forwarder and LiteLLM can
replace, filter, or transform the provider request. It is therefore always
labeled `replay_proxy` / `agent_session` with an incomplete provider request,
even when the proxy is host-owned. If replay and the untrusted agent also share
root custody in the sandbox, the manifest records that additional limitation.
The suffix remains useful for audit and continuity, but any continued run with
a live suffix is excluded from training-ready provider evidence.

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

### Models and credentials

- The **live-continuation model** defaults to the original run's model so the
  continuation is a faithful continuation of the same brain. Tests use
  `--model gemini-3.1-flash-lite-preview` for a cheap path.
- The **replay phase needs no API key** — responses are served from the
  recording. Only the **live continuation** calls the real provider, so the
  host needs that provider's credentials (e.g. `GEMINI_API_KEY`) in its
  environment. `--replay-only` skips the live leg entirely.

## Limitations and caveats

- **`openhands` only** for now (the proxy seam relies on `LLM_BASE_URL`).
- **Replay fidelity is best-effort.** Replay re-runs the original shell
  commands for real; if a command's output diverges from the original
  (network, timestamps, nondeterminism), the agent may see a different
  observation than recorded. A message-count check warns on divergence
  (`--strict-divergence` aborts instead).
- **"Identical output" means a faithful continuation**, not a bit-identical
  result — the model samples, and no "original full run" exists past the
  timeout. The bar is: the stitched trajectory reads as one continuous run, as
  if the timeout had been larger.
- Re-running the episode's commands costs wall-clock time (model latency is
  skipped, since recorded responses are served instantly).
