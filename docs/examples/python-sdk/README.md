# Python SDK examples

Runnable scripts for the Python API. Run one from the repository root:

```bash
uv run python docs/examples/python-sdk/run-oracle.py
uv run python docs/examples/python-sdk/run-oracle.py --help    # its options
```

`--sandbox docker`, the default, needs Docker running. `--sandbox daytona` needs `DAYTONA_API_KEY` and the `sandbox-daytona` extra.

Agent runs need that agent's credential; `bench doctor` shows which one it would use. For Claude, that is `CLAUDE_CODE_OAUTH_TOKEN` or `ANTHROPIC_API_KEY` ([Auth](../../agents/auth.md)).

| Example | Shows | Needs |
|---|---|---|
| `quickstart.py` | A tour in `# %%` cells: one oracle trial on a tiny task on a prebuilt image (about 20 seconds on Docker), a Claude trial and its trajectory, a small batch, then reading the jobs back and comparing them. Open it in VS Code, PyCharm or Jupytext, or run it top to bottom. `BF_QUICKSTART_OFFLINE=1` runs only the reading and comparing cells, on stand-in data. | A sandbox; Claude for the agent cell |
| `run-oracle.py` | One trial with the oracle and the typed result: `reward`, `passed`, `score_outcome`, `rollout_dir`. | A sandbox |
| `run-agent.py` | One trial with a real agent. Walks the trajectory, prints token usage, and reads the trial back with `RolloutResult.load`. | A sandbox, agent credentials |
| `run-many.py` | Several `AGENT[:MODEL]` combinations on one task with `bf.run_batch` (no asyncio). Prints each result as it finishes and exports CSV and JSONL. | A sandbox; credentials for real agents |
| `run-batch.py` | Every task under `--tasks-dir` as one `Evaluation`: `stream()` as tasks finish, CSV export, and `--resume <job_dir>` to finish an interrupted job. | A sandbox |
| `run-with-manifest.py` | A task inside an environment manifest whose HTTP service BenchFlow starts. Writes the task to a temporary folder. | A sandbox |
| `run-branch.py` | `bf.branch`: a Claude run forked after its first prompt into two verifier-scored children, then the parent finished. `--concurrency 2` gives each child its own sandbox. | A sandbox, Claude |
| `compare-jobs.py` | `bf.load_job` and `bf.compare` over two finished jobs, or two globs of folders: paired rewards per task, attempted, scored and errored counts, controls left out. | Nothing: it reads files |

`docs/examples/` also has `branch-agent-run.py` (the `Rollout.branch()` steps behind `bf.branch`), `coder-reviewer-demo.py`, `scene-patterns.py` and `user_dogfood.py`.

## Next steps

- The API: [Python API reference](../../reference/python-api.md).
- Read jobs back: [Analyse runs](../../results/analysing-runs.md).
