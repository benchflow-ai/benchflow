# Trajectory/result audit rubric

You are auditing a BenchFlow E2E run output bundle. Treat deterministic JSON
reports as the source of truth and use trajectory/log snippets only to explain
root causes.

Inputs you may receive:

- `matrix_config.json` — requested agents, tasks, model, backend, and run paths.
- `matrix_summary.json` — one row per `(agent, task)` entry.
- `artifact_audit.json` — required-file and schema checks for each trial.
- `parity_report.json` — normalized BenchFlow-vs-baseline comparisons.
- sampled `result.json`, `timing.json`, `agent/*.txt`, verifier logs, and
  `trajectory/acp_trajectory.jsonl` excerpts.

Audit goals:

1. Identify framework/runtime regressions separately from model/task failures.
2. Record agents that fail specifically because `gemini-3.1-flash-lite-preview`
   is unsupported or misconfigured for that agent.
3. Check core artifact parity:
   - `result.json`
   - `timing.json`
   - `prompts.json`
   - `rewards.jsonl`
   - `agent/install-stdout.txt`
   - `agent/<agent>.txt`
   - `trajectory/acp_trajectory.jsonl`
   - verifier logs/artifacts
4. Check schema parity for result fields, rewards, errors, trajectory source,
   token/cost fields, and phase timings.
5. Compare reward and error distributions with historical baseline artifacts
   when baselines exist.
6. Call out missing baselines explicitly without treating them as failures.
7. Produce concise findings with evidence paths.

Output:

- A short executive summary.
- A table of agent-level health.
- A table of task-level verifier/task issues.
- A list of actionable BenchFlow bugs.
- A list of known/capability-gap findings.
- A final pass/fail recommendation for whether the E2E run is trustworthy.
