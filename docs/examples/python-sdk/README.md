# Python SDK examples

Runnable scripts for the Python API ([reference](../../reference/python-api.md)). Each script runs on Docker or Daytona. Run any of them from the repository root with `uv run python docs/examples/python-sdk/<file>`; `--help` lists the options. `--sandbox docker` (the default) needs Docker running; `--sandbox daytona` needs `DAYTONA_API_KEY` and the `sandbox-daytona` extra. Agent runs need that agent's usual credential (`bench doctor` shows which one it would use): for Claude, `CLAUDE_CODE_OAUTH_TOKEN` or `ANTHROPIC_API_KEY`.

| Example | Shows | Needs |
|---|---|---|
| `quickstart.py` | A tour in `# %%` cells (open it in VS Code, PyCharm or Jupytext, or run it top to bottom): one oracle rollout, a Claude rollout and its trajectory, a small batch, then reading the jobs back and comparing them. `BF_QUICKSTART_OFFLINE=1` runs only the reading and comparing cells, on stand-in data. | A sandbox; Claude for the agent cell |
| `run-oracle.py` | One rollout with the oracle agent and the typed result (`reward`, `passed`, `score_outcome`, `rollout_dir`). | A sandbox |
| `run-agent.py` | One rollout with a real agent; walks the trajectory, prints token usage, reads the rollout back with `RolloutResult.load`. | A sandbox, agent credentials |
| `run-many.py` | Several `AGENT[:MODEL]` combinations on one task with `bf.run_batch` (no asyncio), progress as each finishes, CSV and JSONL export. | A sandbox; credentials for real agents |
| `run-batch.py` | Every task under `--tasks-dir` as one `Evaluation`: `stream()` as tasks finish, CSV export, `--resume <job_dir>` to finish an interrupted job. | A sandbox |
| `run-with-manifest.py` | A task inside an Environment-plane manifest whose HTTP service BenchFlow starts; writes the task to a temporary folder. | A sandbox |
| `run-branch.py` | `bf.branch`: a Claude run forked after its first prompt into two verifier-scored children, then the parent finished (`--concurrency 2` gives each child its own sandbox). | A sandbox, Claude |
| `compare-jobs.py` | `bf.load_job` + `bf.compare` over two finished jobs (or two globs of folders): paired per-task rewards, attempted/scored/errored counts, control runs left out. | Nothing (reads files) |

Also in `docs/examples/`: `branch-agent-run.py` (the manual `Rollout.branch()` lifecycle behind `bf.branch`), `coder-reviewer-demo.py`, `scene-patterns.py`, and the progressive-disclosure `user_dogfood.py`.
