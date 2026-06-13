# Browser Use smoke benchmark

Tiny Browser Use-shaped fixture for BenchFlow 0.7 adapter dogfood.

Run the parity check from the repo root:

```bash
uv run python benchmarks/browser-use-smoke/parity_test.py
```

To emit verifier evidence and close the adoption loop:

```bash
mkdir -p /tmp/benchflow-adapter-parity/browser-use-smoke
uv run python benchmarks/browser-use-smoke/parity_test.py \
  --parity-out /tmp/benchflow-adapter-parity/browser-use-smoke/parity_experiment.json
uv run bench agent verify browser-use-smoke \
  --benchmarks-dir /tmp/benchflow-adapter-parity
```

The parity run also writes
`/tmp/benchflow-adapter-parity/browser-use-smoke/adoption_report.json`, a
scrubbed review manifest with the sandbox, environment adapter, agent adapter,
benchmark adapter, parity counts, artifact counts, timing coverage, and cleanup
summary. It also writes `loop_state.json`, a resumable adapter-adoption flight
recorder with source, commands, artifacts, role status, cleanup, and next queue
items. Use `parity_experiment.json` for the gate, `adoption_report.json` for
human/CI inspection, and `loop_state.json` when another agent or later session
needs to resume the loop.

The script runs the fixture's original runner, materializes the same task
through the BenchFlow Browser Use inbound adapter inside `bench eval create`,
runs BenchFlow with `--agent browser-use-smoke --sandbox docker`, and compares
reward plus trace/artifact shape. The browser environment adapter also records
a scrubbed readiness snapshot for the served local fixture: status, URL, HTTP
status, byte count, content hash, and timing, without storing raw HTML. The
browser runtime session writes the shared trace artifact schema
`benchflow.browser-runtime-trace.v1`, so Browser Use and Stagehand shims share
the same environment/readiness/artifact envelope. The agent is a fixture ACP
shim for dogfooding the agent-adapter seam; it is not the full Browser Use
framework integration.

To exercise the real Browser Use CLI browser harness instead of the fixture
shim:

```bash
uv run python benchmarks/browser-use-smoke/parity_test.py --agent browser-use-cli
```

To exercise the real LLM-driven Browser Use Agent loop with Gemini:

```bash
source /Users/lixiangyi/context/benchflow-0.7/keys.env
uv run python benchmarks/browser-use-smoke/parity_test.py \
  --agent browser-use-agent \
  --model gemini-3.5-flash
```

To exercise the real Stagehand DOM Agent loop with Gemini:

```bash
source /Users/lixiangyi/context/benchflow-0.7/keys.env
uv run python benchmarks/browser-use-smoke/parity_test.py \
  --agent stagehand-agent \
  --model google/gemini-3.5-flash
```

To import a tiny official BU Bench slice without committing plaintext tasks:

```bash
git clone --depth 1 https://github.com/browser-use/benchmark.git /tmp/browser-use-benchmark
uv run python benchmarks/browser-use-smoke/import_upstream.py \
  --upstream-repo /tmp/browser-use-benchmark \
  --out-dir /tmp/benchflow-bu-bench/tasks \
  --task-indices 0 \
  --overwrite
uv run --extra judge bench eval create \
  --tasks-dir /tmp/benchflow-bu-bench/tasks \
  --agent browser-use-agent \
  --model gemini-2.5-flash \
  --sandbox docker \
  --jobs-dir /tmp/benchflow-bu-bench/jobs \
  --concurrency 1
```

`import_upstream.py` decrypts Browser Use's `.enc` suite in memory and writes
selected task dirs only to the requested output directory.

To probe the official runner on the same selected encrypted task without
committing raw traces:

```bash
GOOGLE_API_KEY=<redacted> \
  uv run python benchmarks/browser-use-smoke/original_runner_probe.py \
    --upstream-repo /tmp/browser-use-benchmark \
    --task-indices 0 \
    --browser local_headless \
    --model gemini-2.5-flash \
    --report-out /tmp/benchflow-bu-bench/original_runner_probe.json
```

The probe shells through Browser Use's `run_framework_eval.py`, preserves its
task index, framework, browser, model, timeout, and concurrency knobs, and
writes a scrubbed report. It records summary/task-result paths and raw trace
counts only; it does not read or copy raw `run_data` traces because those may
contain decrypted task text, ground truth, model output, and screenshots. A
blocked original runner is valid evidence only when it is paired with the
BenchFlow run on the same selected task and does not claim parity.

To run the official import, BenchFlow eval, original-runner probe, adoption
report, and loop-state emission in one command:

```bash
source /Users/lixiangyi/context/benchflow-0.7/keys.env
uv run python benchmarks/browser-use-smoke/official_adoption_driver.py \
  --upstream-repo /tmp/browser-use-benchmark \
  --work-dir /tmp/benchflow-bu-bench/work \
  --task-indices 0 \
  --agent browser-use-agent \
  --model gemini-2.5-flash \
  --sandbox docker \
  --parity-out /tmp/benchflow-bu-bench/browser-use-official/parity_experiment.json \
  --overwrite
```

The driver writes `parity_experiment.json`, `adoption_report.json`,
`loop_state.json`, and `original_runner_probe.json`. If the original runner is
blocked, the loop state stays `not-ready` with `original-runner=blocked`;
BenchFlow completion, reward, traces, screenshots, timing, and cleanup still
remain auditable.

For official Stagehand eval task import dogfood, use
`benchmarks/stagehand-smoke/import_upstream.py`. It writes temporary
`stagehand-task.json` task dirs that the normal `bench eval create
--tasks-dir ... --agent stagehand-agent --sandbox docker` path can consume.
