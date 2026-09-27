# End-to-end tier

Real `bench` CLI and public Python API calls against real Daytona sandboxes, with synthetic tasks the harness writes itself. Each scenario checks the real outputs: files in the job folder, JSON documents against the committed schemas in `docs/reference/schemas/`, exit codes, and the trajectory viewer's HTTP answers.

The tier is excluded from the default suite (marker `e2e`, like `live` and `integration`) and never picks a backend on its own, because it spends sandbox time:

```bash
export DAYTONA_API_KEY=...                 # your Daytona key
export BENCHFLOW_DAYTONA_OWNER=e2e-$USER   # every sandbox and snapshot carries this owner
export BENCHFLOW_E2E_SANDBOX=daytona
uv run pytest -m e2e tests/e2e
```

Agents are `oracle` and `nop`. Scenarios that need a real ACP session (branching an agent run, USD budgets, the network allowlist, repeated trials for pass@k) run `claude-agent-acp` against the scripted fake model provider of the deterministic tier (`tests/integration/deterministic/task/environment/fake_llm`), which runs inside the sandbox; no model credentials are read and no provider is billed.

Every run passes `--max-sandbox-seconds` (`BENCHFLOW_E2E_MAX_SANDBOX_SECONDS`, default 900 per job). `BENCHFLOW_E2E_OUT` keeps the jobs, CLI logs and `ledger.jsonl` (one row per scenario: surface, wall time, sandbox-seconds, USD). At the end of the session the tier runs `bench sandbox cleanup --all --max-age 0` for its owner and fails if any sandbox or branch snapshot of that owner is left.

| File | Features |
|---|---|
| `test_eval_run.py` | `bench eval run` single and batch, `nop` control, `--fail-on`/`--fail-under`/`--summary-out`, pre-run checks (CLI and Python), `bf.run_sync`, `bf.run_batch`, `Evaluation.run_sync`, `Evaluation.resume`, `bench eval resume` |
| `test_results.py` | `--matrix --trials`, `bench eval metrics` pass@k/pass^k/`--solve-threshold`, `Job.solve_rates`, `bench eval inspect --json`, `bf.load_job`/`load_trial`, `bench eval compare`/`bf.compare` with `--on-mismatch`/`--vary` |
| `test_budget.py` | `--max-sandbox-seconds`, `--max-cost-usd`, `bf.Budget`, resume after a budget stop |
| `test_branching.py` | `bench eval branch` (oracle and agent), `--retain-snapshots`/`--from-checkpoint`, `--checkpoints` snapshot reuse, `bf.branch`, `bench eval branches`/`branch_views()`, `bench train convert --format branch-tree` and `validate`, `bench eval view` HTTP |
| `test_verifier.py` | `bench eval regrade`/`bf.regrade`, separate verifier sandboxes, verifier recovery after a lost sandbox |
| `test_tasks.py` | `bench tasks init` canaries, `bench tasks check` warnings and `--level equivalence`, `check_equivalence`, lenient legacy `task.toml`, a batch with a symlinked task |
| `test_network.py` | `network_mode: allowlist` on Daytona, block log |
| `test_remote_docker.py` | `--sandbox remote-docker` on a Docker host reached over SSH (a Daytona `docker:dind` sandbox the scenario creates, or `BENCHFLOW_E2E_REMOTE_DOCKER_HOST`): oracle batch with `no-network` and a separate verifier, `nop` control, a task larger than the host, an unreachable host refused before a job, `network_mode: allowlist` armed through the provider API, nothing left on the host, the SSH user absent from every output |
| `test_restore_and_listing.py` | Frozen-workspace file modes restored by `bench eval regrade`, a changed verdict on an unchanged task, regrade of a batch job without `--tasks-dir`, `bench eval list` and `bench eval metrics` over a shared jobs folder, the reviewer-model refusal |
| `test_rl_self_hosted.py` | Self-hosted policy routes (`vllm/`, `sglang/`) with gateway token capture, `bench train token-coverage`, `bench train stream` and `bf.stream_rollouts` while a job runs; selects its sandbox with `BENCHFLOW_DETERMINISTIC_SANDBOX` (Docker or Daytona) |
| `test_training.py` | `bench train convert --reward-vector --group-advantage`, `bench train validate`, `bench train token-coverage` |
| `test_models.py` | Opt-in (`BENCHFLOW_E2E_MODELS=1`, subscription logins only): `bench eval smoke`, Codex Apps policy receipt, native Claude OAuth on a no-network task, rubric review verdicts and reward arithmetic, trials of the bundled public example tasks |
