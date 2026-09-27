# Integration tests

This folder holds two kinds of end-to-end tests.

- **Deterministic tier** (`deterministic/`, run by `tests/test_deterministic_integration.py`): the real `bench` CLI, a real sandbox and a real agent (`claude-agent-acp`), with the model replaced by a scripted fake provider. No model keys, no model cost, same result every run. It runs in the default suite whenever a sandbox is available and on every PR in CI.
- **Live lanes** (`run.sh`, `run_suite.py`, `scenarios.py`, `suites/`, `configs/`): real models on Daytona, opt-in with `-m integration` / `-m live` and provider keys. See [`docs/integration-tests.md`](../../docs/integration-tests.md) and [`docs/integration-tiers.md`](../../docs/integration-tiers.md).

The rest of this page is about the deterministic tier.

## How it works

The fixture task is `deterministic/task/`: a hello-world task whose image also carries the fake provider (`environment/fake_llm/fake_llm.py`, standard library only) and its scripts (`environment/fake_llm/scripts.json`). The task's `environment.toml` declares the fake as an environment-plane service on `127.0.0.1:8911`, so BenchFlow starts it after the sandbox starts and restarts it after every branch or checkpoint restore (running processes are not part of a snapshot).

The fake speaks the Anthropic Messages API (`POST /v1/messages`, streaming SSE and plain JSON, `/v1/messages/count_tokens`, `GET /health`). It is stateless: the reply is a function of the request only.

- The script is chosen by a `[[fake-llm:NAME]]` marker in the most recent user message that has one: the task instruction, a `--prompt`, a branch child's prompt or a `--retry-prompt`.
- The step is the number of assistant messages after that marker message. A restarted server, a branch child or a retried request therefore gets the same reply.
- Tool calls use the tool name the agent offers (`Bash`, or `mcp__acp__Bash`), and their ids are `toolu_fake_<script>_<step>`, so trajectories are comparable across runs.
- Every agent-loop reply reports 1,000 input and 50 output tokens. Requests without tools (Claude Code's session-title call) get `ok` with 0 input and 1 output token. With LiteLLM's `claude-haiku-4-5` prices one agent-loop call costs $0.00125.

Two routes reach the fake; the harness (`deterministic/harness.py`) sets them up.

| Route | Agent env | Path | Used by |
|---|---|---|---|
| `proxy` (BenchFlow's API-key route) | `ANTHROPIC_API_KEY=<dummy>`, `BENCHFLOW_PROVIDER_BASE_URL=<fake>` | agent → BenchFlow's LiteLLM proxy → fake. Records provider usage, cost and `trajectory/llm_trajectory.jsonl`. | `bench eval run` scenarios, checkpoint retry |
| `native` (subscription route) | `ANTHROPIC_AUTH_TOKEN=<dummy>`, `ANTHROPIC_BASE_URL=http://127.0.0.1:8911` | agent → fake, no proxy. Usage comes from ACP; no price. | branch scenarios |

The LiteLLM proxy runs inside the sandbox on Daytona and on the host on Docker. The fake must run where the proxy runs, so on Docker the harness also starts the same fake on the host for the `proxy` route. Branching refuses a rollout that has a provider runtime ("Branching an active provider runtime needs a runtime fork contract"), which is why the branch scenarios use the `native` route. Before each run the harness resolves the agent env and refuses a route that would upload host credentials (`~/.claude/.credentials.json`) or miss the fake.

## Scenarios

| Test | CLI | Asserts |
|---|---|---|
| `test_hello_world_passes` | `bench eval run` | reward 1.0; completed · scored; verifier ran; all timing keys |
| `test_wrong_answer_scores_zero` | `bench eval run` | reward 0.0; completed · scored |
| `test_agent_timeout_with_passing_file_is_timed_out_and_passed` | `bench eval run`, agent timeout 45 s | timed out · scored 1.0; `agent_timeout_info` names the pending tool call; verifier ran |
| `test_agent_crash_is_unscored_and_not_verified` | `bench eval run` | the scripted tool call kills the Claude process: errored (`acp_error`) · unscored; no reward; verifier did not run |
| `test_verifier_error_is_an_assessment_error_never_zero` | `bench eval run`, broken `test.sh` | completed · assessment error; `rewards` null, never 0; `verifier_failure` |
| `test_eval_job_summary` | the same job | `summary.json` counts and categories, token totals |
| `test_branch_two_children_in_place` | `bench eval branch --checkpoint-after-prompt 1 --child … --child …` | V = 0.5, `value_stderr`, children 1.0 / 0.0 from the verifier, `children_mode`, child folders, parent restored and scored 1.0 |
| `test_branch_two_children_parallel` | the same with `--concurrency 2` | as above with isolated children, each child's own trial folder checked like a trial |
| `test_retry_from_checkpoint_keeps_both_scores` | `bench eval run --checkpoints prompt:1 --retry-from-checkpoint on-failure --retry-prompt …` | the trial keeps 0.0, the retry child scores 1.0, both recorded; kept snapshots deleted afterwards |

Every scenario trial is also checked for: token usage and cost exactly as the fake's fixed usage implies (per captured call on the `proxy` route; per agent message on the `native` route), `usage_source`, `usage_tracking` endpoint, non-negative timing, the ATIF-v1.7 structure rules that `export_atif` promises, the public `benchflow.trial` document against `docs/reference/schemas/benchflow-trial.v1.schema.json`, and tool call ids from the fake. Then the run-independent facts are compared with the golden file in `deterministic/golden/<scenario>.json`: outcome labels, a `result.json` subset (rewards, counts, error categories, usage, final metrics, trajectory summary, timing keys), the captured provider calls, the ACP trajectory and the ATIF export, with timestamps, receipts and session ids stripped. Claude Code's session-title call is left out of the golden file, and usage there is net of it: whether that call reaches the proxy before BenchFlow stops the agent is a race (seen on both backends). It is still checked against the recorded totals. Golden files are backend-neutral: the same set passed on Daytona and on Docker.

## Running

```bash
# Docker (default when a daemon answers `docker info`)
uv run python -m pytest tests/test_deterministic_integration.py -v -rs

# Daytona (never chosen implicitly: it spends sandbox time)
BENCHFLOW_DETERMINISTIC_SANDBOX=daytona DAYTONA_API_KEY=... \
  uv run python -m pytest tests/test_deterministic_integration.py -v -rs

# Keep the job folders for inspection
BENCHFLOW_DETERMINISTIC_JOBS_DIR=/tmp/det-jobs uv run python -m pytest tests/test_deterministic_integration.py

# Leave the tier out of a local run
uv run python -m pytest tests/ -m "not deterministic"
```

| Variable | Meaning |
|---|---|
| `BENCHFLOW_DETERMINISTIC_SANDBOX` | `docker`, `daytona` or `off`. Unset: Docker if available, otherwise every scenario skips with the reason. Set to a backend that is unavailable: the scenarios fail instead of skipping (CI sets `docker`). |
| `BENCHFLOW_DETERMINISTIC_JOBS_DIR` | Write job folders here instead of pytest's temporary directory. |
| `BENCHFLOW_DETERMINISTIC_CONCURRENCY` | `--concurrency` of the five-task `bench eval run` job (default 4). |
| `BENCHFLOW_UPDATE_GOLDEN` | `1` rewrites the golden files instead of comparing. |

The scenarios share four CLI jobs (one five-task `bench eval run`, two `bench eval branch`, one checkpoint-retry `bench eval run`), each started once per session on first use, so `-k branch` runs only the branch jobs.

CI: the `deterministic-integration` job in `.github/workflows/test.yml` runs the tier on the runner's Docker on every PR and push to `main` with `BENCHFLOW_DETERMINISTIC_SANDBOX=docker`, and uploads the job folders when it fails. The unit job sets `BENCHFLOW_DETERMINISTIC_SANDBOX=off` so the tier does not run twice.

## Adding a scenario

1. Add a script to `deterministic/task/environment/fake_llm/scripts.json`: a list of steps, each with `text` and optionally `tool` (a tool name the agent offers, matched exactly or as a `__<name>` suffix) and `input` (the tool arguments). Step *n* answers the *n*-th model call after the marked prompt; past the end the fake answers `Done.`. Every step needs `text`, because the checks count one agent message per model call.
2. If the task itself must differ, add a `TaskVariant` (marker, agent timeout, broken verifier) or extend `materialize_task`. Keep `environment/` identical across variants so every scenario shares one image.
3. Add the variant to `EVAL_VARIANTS` (it joins the shared `bench eval run` job) or add a session fixture with its own CLI call, and write the test: call `_check_trial(trial_dir, "<golden name>", route=...)` and assert what the scenario is about.
4. Generate its golden file on purpose (below), read it, and commit it with the test.

`tests/test_fake_llm_provider.py` pins the fake's own contract without a sandbox; extend it when you change the fake.

## Regenerating golden files on purpose

A golden mismatch means agent-visible behaviour changed: the trajectory, the ATIF export, usage, outcome labels or the `result.json` fields listed above. If the change is intended:

```bash
BENCHFLOW_UPDATE_GOLDEN=1 uv run python -m pytest tests/test_deterministic_integration.py -k <scenario>
git diff tests/integration/deterministic/golden/
uv run python -m pytest tests/test_deterministic_integration.py -k <scenario>   # must pass unchanged
```

Review the diff line by line before committing it with the code change that caused it; a regenerated golden file is a claim that the new behaviour is correct.

## Not covered

- `codex-acp` and the other agents: the fake speaks only the Anthropic Messages API. Any LiteLLM-routed agent can be pointed at a fake the same way (`BENCHFLOW_PROVIDER_BASE_URL` sets the proxy's upstream); covering one needs the upstream protocol it uses (OpenAI Responses or Chat Completions) added to the fake.
- The verifier's pytest plugin guard refusals: they need a pytest-based verifier and a plugin location the agent can write, which this task does not have.
- The same scenarios with `--sandbox modal` or other providers.
