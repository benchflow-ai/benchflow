# Onramp

Onramp converts external benchmarks into [BenchFlow tasks](../docs/task-authoring.md), so you can run them with `bench run`, score them with the standard verifier contract, and slot them into Scene-based multi-turn lifecycles.

Each subdirectory is one adapter: it knows how to read the upstream benchmark's data, then emits a directory of task folders — `task.toml` + `instruction.md` + `environment/Dockerfile` + `tests/test.sh` — that BenchFlow runs unchanged.

## Available adapters

| Adapter | Upstream | Tasks | Notes |
|---|---|---|---|
| [`programbench`](./programbench/) | [ProgramBench](https://programbench.com/) — rebuild a program from binary + docs | 200 | One image per task (`programbench/<id>:task_cleanroom`); per-branch tests pulled from HuggingFace |

## Layout

```
onramp/<adapter>/
├── adapter.py             # parse upstream → emit BenchFlow task dirs
├── main.py                # CLI: `python -m onramp.<adapter>.main --output-dir <path>`
├── parity.py              # run parity check vs. upstream eval
├── README.md
├── run_<adapter>.yaml     # `benchflow.job.Job` config for the full converted dataset
└── templates/             # static template fragments used by adapter.py
```

## Adding a new adapter

1. Read the upstream benchmark's task schema and runner.
2. Mirror the layout above under `onramp/<your-adapter>/`.
3. In `adapter.py`, emit one directory per upstream task. Every `task.toml` must have a stable, lowercase, hyphenated `[task].name` so it survives re-runs and registry lookups.
4. Run `bench tasks check <output>/` on the generated set — every task must pass.
5. Add a `parity.py` that re-runs the same submission through both pipelines and reports score deltas.

See [`programbench/`](./programbench/) for a worked example.
