# Computer Use smoke benchmark

Tiny computer-use-shaped fixture for BenchFlow 0.7 desktop/Cua adapter dogfood.

Run the parity check from the repo root:

```bash
BENCHFLOW_CUA_LOCAL=1 BENCHFLOW_CUA_LINUX_KIND=container \
  uv run python benchmarks/computer-use-smoke/parity_test.py
```

To emit verifier evidence and close the adoption loop:

```bash
mkdir -p /tmp/benchflow-adapter-parity/computer-use-smoke
BENCHFLOW_CUA_LOCAL=1 BENCHFLOW_CUA_LINUX_KIND=container \
  uv run python benchmarks/computer-use-smoke/parity_test.py \
  --parity-out /tmp/benchflow-adapter-parity/computer-use-smoke/parity_experiment.json
uv run bench agent verify computer-use-smoke \
  --benchmarks-dir /tmp/benchflow-adapter-parity
```

The parity run also writes
`/tmp/benchflow-adapter-parity/computer-use-smoke/adoption_report.json`, a
scrubbed review manifest with the sandbox, environment adapter, agent adapter,
benchmark adapter, parity counts, artifact counts, timing coverage, and cleanup
summary. It also writes `loop_state.json`, a resumable adapter-adoption flight
recorder with source, commands, artifacts, role status, cleanup, and next queue
items. Use `parity_experiment.json` for the gate, `adoption_report.json` for
human/CI inspection, and `loop_state.json` when another agent or later session
needs to resume the loop.

The script runs the fixture's original Cua SDK runner, materializes the same
task through the BenchFlow computer-use inbound adapter, runs BenchFlow with
`--agent computer-use-smoke --sandbox cua`, and compares reward plus
trace/artifact shape. The agent is a fixture ACP shim for dogfooding the
desktop adapter seam; it is not a full CUA model loop.
