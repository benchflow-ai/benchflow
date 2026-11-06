# Integration tests

This folder holds BenchFlow's end-to-end tests. It is for maintainers.

| Part | What it runs | Keys | When |
|---|---|---|---|
| [Release suite](#release-suite) (`run.sh`, `run_suite.py`, `suites/`, `configs/`) | real agents and models on real benchmarks, on Daytona | model providers and Daytona | by hand, before a release or a large runtime change |
| [Judge and integrity tests](#judge-and-integrity-tests) (`scenarios.py`, `agent_judge.py`, `deepagents_harness.py`) | the realness gate and the LLM judge, on fake and real rollouts | none for the deterministic half | the deterministic half on every PR; the live half by hand |
| [Keyless tests](#keyless-tests) (`deterministic/`) | the real CLI, a real sandbox and a real agent, with a scripted fake model | none | on every PR |

CI runs a smaller set on every PR; see [Integration CI](../../.github/integration/README.md) and the [review rubric](../../.github/integration/REVIEW_RUBRIC.md).

## Release suite

```bash
# All 9 agents, 9 tasks each, on Daytona
export DAYTONA_API_KEY=... GEMINI_API_KEY=... CLAUDE_CODE_OAUTH_TOKEN=... \
       OPENAI_API_KEY=... XIAOMI_API_KEY=... XIAOMI_BASE_URL=...
tests/integration/run.sh

# Some agents only
tests/integration/run.sh gemini pi-acp claude-agent-acp

# Run whichever agents have keys; skip the rest instead of failing
BENCHFLOW_INTEGRATION_ALLOW_SKIPS=true tests/integration/run.sh

# Check an earlier run again (no model calls)
BENCHFLOW_INTEGRATION_JOBS_ROOT=jobs/integration-<run-id> tests/integration/run.sh --check-only

# 100 trials at once instead of 64
BENCHFLOW_INTEGRATION_CONCURRENCY=100 tests/integration/run.sh gemini
```

Install Daytona support first:

```bash
uv sync --extra dev --extra sandbox-daytona --locked
```

`run.sh` then does three things:
1. Starts one run per agent, in parallel, from `configs/<agent>.yaml`. Each config takes the 9 tasks from `benchflow-ai/skillsbench`, 64 trials at a time.
2. Prints each agent's score as it finishes.
3. Runs `check_results.py` to check the output files and print a table.

### Keys

| Variable | Needed by |
|---|---|
| `DAYTONA_API_KEY` | every run |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `gemini`, `pi-acp`, `openclaw`, `opencode`, `harvey-lab-harness`, `openhands` |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or `CLAUDE_CODE_OAUTH_TOKEN` | `claude-agent-acp` |
| `OPENAI_API_KEY` | `codex-acp` |
| `XIAOMI_API_KEY` and `XIAOMI_BASE_URL` | `mimo` |

If an agent has no key, `run.sh` prints `SKIP` and exits 1. Set `BENCHFLOW_INTEGRATION_ALLOW_SKIPS=true` to run the rest. Azure Foundry models need `AZURE_API_KEY` and `AZURE_API_ENDPOINT` instead.

`--check-only` needs `BENCHFLOW_INTEGRATION_JOBS_ROOT` or `BENCHFLOW_INTEGRATION_RUN_ID`. A run with neither writes to `jobs/integration-<date>-<time>/`.

### Agents

| Agent | Model |
|---|---|
| `claude-agent-acp` | `claude-haiku-4-5-20251001` |
| `codex-acp` | `gpt-5.4-nano` |
| `mimo` | `xiaomi/mimo-v2.5` |
| `gemini`, `pi-acp`, `openclaw`, `opencode`, `harvey-lab-harness`, `openhands` | `gemini-3.1-flash-lite-preview` |

The 9 tasks are the `nine` set: three low (`jax-computing-basics`, `python-scala-translation`, `jpg-ocr-stat`), three medium (`grid-dispatch-operator`, `threejs-to-obj`, `data-to-d3`) and three high (`lake-warming-attribution`, `weighted-gdp-calc`, `shock-analysis-supply`).

To run one agent without `run.sh`:

```bash
uv run bench run --config tests/integration/configs/gemini.yaml
```

### Output

```
jobs/integration-<run-id>/
├── gemini/
│   ├── <job-name>/
│   │   ├── jax-computing-basics__abc123/
│   │   │   ├── result.json
│   │   │   ├── trajectory/acp_trajectory.jsonl
│   │   │   └── ...
│   │   └── ...
│   └── summary.json
├── claude-agent-acp/
│   └── ...
└── .logs/            # each agent's stdout and stderr
    ├── gemini.log
    └── ...
```

`check_results.py` checks that:
- every `result.json` has `task_name`, `agent`, `rewards`, `error` and `verifier_error`;
- no trial failed for an infrastructure reason;
- `summary.json` has `total`, `passed`, `failed`, `errored`, `verifier_errored` and `score`.

```bash
uv run python tests/integration/check_results.py jobs/integration-<run-id> gemini pi-acp
```

### What a release must pass

Do not publish a release if any check below fails.

| Check | Passes when |
|---|---|
| SkillsBench agent matrix | the 9 tasks with every agent that has a key write a valid `result.json`, trajectory and summary |
| Adapter release set | each benchmark on [Running adapted benchmarks](../../docs/datasets/benchmarks.md#available-benchmarks), and each adapter targeted at the release, has one smoke or parity run with its verifier and reward schema checked |
| Trace-to-task | `bench tasks generate` turns one local or JSONL trace and one Hugging Face trace into tasks, and each task runs and is graded |
| Agent decoupling | core import and task checks work without optional agent packages; two non-oracle agents run the same task |
| Sandbox decoupling | `import benchflow` works without sandbox extras; Docker and Daytona run the same task; a missing extra prints install steps |
| Release sandboxes | `bench run --sandbox docker` and `--sandbox daytona` each run a task with only that backend's normal login. Modal is not required. |

Each adapter is named by its source, written `{repo}:{path}@{ref}`, never by its display name. The current list is in `suites/release.yaml`.

Environments from hosted hubs (OpenReward, benchmark registry, PrimeIntellect) are tracked by `env_uid` and `hub_url`, not by a task-source ID. They get a `benchflow_uid` only once converted to a BenchFlow task source. OpenReward has no environment versions, so the board records its `updated_at` time next to the `env_uid`. Running an OpenReward session may need account credits.

### Lanes and profiles

The suite is a set of named lanes, not every agent × model × sandbox × task combination. Each lane names the risk it covers. Lanes and profiles are defined in `suites/release.yaml`.

Plan before running anything:

```bash
uv run python tests/integration/run_suite.py --list-lanes
uv run python tests/integration/run_suite.py --list-profiles
uv run python tests/integration/run_suite.py --profile near-term --dry-run
uv run python tests/integration/run_suite.py --profile release-gated-cli --dry-run --fail-on-todo
uv run python tests/integration/run_suite.py --profile hosted-envs --dry-run
uv run python tests/integration/run_suite.py --profile full-release --dry-run --fail-on-todo
uv run python tests/integration/run_suite.py --profile backlog --dry-run --fail-on-todo
uv run python tests/integration/run_suite.py --lane shared-sandbox-smoke --dry-run
```

| Profile | Use it for |
|---|---|
| `near-term` | a quick pass: SkillsBench, Daytona, one task per adapter |
| `release-gated-cli` | the CLI release gate: shared sandbox smoke, the SkillsBench agent matrix, the adapter release set and the decoupling checks, on Docker and Daytona |
| `hosted-envs` | the hosted environment board |
| `full-release` | every release check |
| `backlog` | planned lanes (Firecracker, Kubernetes); expected to fail |

Pass `--fail-on-todo` when a plan is release evidence. It fails on any TODO or `blocked_by` entry. `near-term`, `release-gated-cli`, `hosted-envs` and `full-release` pass it today; `backlog` fails until the Firecracker and Kubernetes lane exists.

| Required lane | Covers |
|---|---|
| Shared sandbox smoke | one simple task with the oracle or one cheap agent, on Docker and Daytona |
| SkillsBench agent matrix | the 9 tasks with every agent that has a key, on Daytona |
| Adapter release set | one smoke or parity run per adapter in the release |
| Hosted env compatibility board | the selected OpenReward, benchmark registry and PrimeIntellect environments |
| Trace-to-task e2e | tasks generated from a local or JSONL trace and a Hugging Face trace, then run |
| Decoupling checks | core import without optional packages; one task on two agents and two sandboxes |

| Backlog lane | Becomes required when |
|---|---|
| Security Docker-in-Docker smoke | `bench run --sandbox firecracker` and `--sandbox k8s` run the task, Docker-in-Docker works inside both, and the verifier runs after hardening |

Release checks must pass. Backlog lanes name a real task and when they become required. Nightly runs may add more combinations, but a release decision rests on the named lanes.

`run_suite.py` plans by default. Three lanes can also run:

```bash
# Trace-to-task: generate tasks from traces and run each with the oracle on Docker
uv run python tests/integration/run_suite.py --lane trace-to-task-e2e --execute-trace-evidence --run-trace-eval

# Adapters: check parity artifacts, a SkillsBench result and one worktree per unmerged adapter
uv run python tests/integration/run_suite.py --lane adapter-release-set --execute-adapter-evidence \
  --skillsbench-result <path>/result.json \
  --open-pr-root <AdapterName>=/path/to/adapter-worktree

# Hosted environments: check hub metadata and rebuild the benchmark inventory
uv run python tests/integration/run_suite.py --lane hosted-env-compatibility-board --execute-hosted-env-evidence
```

Each lane writes its evidence under `dogfood/`; `release.yaml` sets the folders, and `--trace-evidence-dir` overrides the trace one. The OpenReward and PrimeIntellect lanes only check hub metadata; the registry lane also lists its registry.

### External comparison evidence

Comparison procedures and baseline outputs are maintained outside this repository.

### Files

```
tests/integration/
├── run.sh                 # launches the agents in parallel and waits
├── run_suite.py           # plans the release suite and runs evidence lanes
├── check_results.py       # checks run outputs and prints the score table
├── check_adapter_evidence.py
├── check_hosted_env_evidence.py
├── check_trace_to_task_evidence.py
├── agent_judge.py         # realness gate + LLM judge
├── rubric_checks.py       # the review rubric's gates
├── scenarios.py           # shared helpers for the judge and integrity tests
├── deepagents_harness.py  # a steerable agent for judge tests
├── deterministic/         # the keyless tier
├── suites/release.yaml    # lanes and profiles
└── configs/               # one run config per agent
```

## Judge and integrity tests

`agent_judge.py` asks an LLM whether a finished rollout is a real attempt: the agent tried the task, the trajectory makes sense, and it did not hack the reward. It uses BenchFlow's `call_judge` with `openai/deepseek-v4-flash` by default (`--model` to change it). It reads `result.json` and the trajectory, and treats the trajectory as data, never as instructions.

```bash
# Judge a rollout folder, or the latest rollout under a jobs folder
uv run python tests/integration/agent_judge.py jobs/integration-eval --json
```

The gate has two parts:
1. **Realness.** A run is real only when `n_tool_calls > 0`, token usage `> 0` and the reward is set. A judge pass cannot rescue a run that isn't real.
2. **The judge.** It must say pass. A missing SDK, an API error, or an answer it cannot parse counts as fail.

In CI, `integration-light.yml` runs one small task with `bench run --sandbox docker`, then judges it. `select_integration_provider.py` picks the first provider that answers a live probe: DeepSeek, GLM, Qwen, a LiteLLM proxy, OpenAI, then GitHub Models. The rollout and the judge use that provider. Install the judge SDKs with `uv sync --extra judge`.

`tests/test_integration_suite.py` tests the gate itself. Each test comes from a bug found while using BenchFlow. Shared helpers (`run_eval`, `reward_of`, `synth_rollout`, the ATIF, ADP and secret-leak checks, `reaper_dryrun_issues`) are in `scenarios.py`. There are two tiers.

### Deterministic (normal CI, no keys)

These run on fake rollouts:

| Test | What it checks |
|---|---|
| Realness gate | rejects a run that was scored but did no work (`n_tool_calls=0`), had no telemetry, was never scored, or errored |
| Judge fails closed | a missing SDK, API error or unreadable verdict reads as fail, never a silent pass |
| Reward hacking caught | a rollout that looks real (tool calls, tokens, reward 1.0) still fails when the judge flags verifier tampering |
| Example `judge.py` fails closed | the shipped generated-skill-eval judge exits non-zero and writes no reward when no LLM judge can run |
| Artifact integrity | ATIF is `ATIF-v1.x` with known step sources; ADP lines parse; no provider key leaks into any file |

```bash
uv run pytest tests/test_integration_suite.py -m "not integration"
```

### Live (`@pytest.mark.integration`, by hand or nightly)

Each scenario skips when its Docker daemon, Daytona key or model keys are missing.

| Scenario | What it checks |
|---|---|
| `oracle_determinism_docker` | every oracle `solve.sh` scores 1.0 |
| `sandbox_parity_docker_daytona` | the same oracle task scores the same on Docker and Daytona |
| `agent_rollout_is_real_and_judged` | a real `openhands` + `deepseek-v4-flash` run is real, and the judge runs on it; the verdict is recorded, not gated |
| `reaper_dryrun_is_safe` | `bench sandbox cleanup --dry-run` deletes nothing and never touches other users' sandboxes |

```bash
export DEEPSEEK_API_KEY=... DEEPSEEK_BASE_URL=https://api.deepseek.com \
       GEMINI_API_KEY=... DAYTONA_API_KEY=...
uv run pytest tests/test_integration_suite.py -m integration
```

`scenarios.py` still runs `bench environment cleanup --dry-run`, the hidden older spelling; it should call `bench sandbox cleanup --dry-run`.

### The deepagents harness

To test the judge against an agent outside BenchFlow, `deepagents_harness.py` runs `deepseek-v4-flash` in LangChain's [deepagents](https://github.com/langchain-ai/deepagents). The agent loop runs on the host. Its shell and file tools run in a Docker container with no network. The run is saved as a BenchFlow rollout and graded like any other. Install it with `uv sync --extra deepagents`.

Pass `extra_system=` to `run_deepagent` to make it cheat. The same harness then produces both honest runs and reward hacks.

The judge ignores instructions inside files the agent reads, and it scans the whole trajectory for edits to test or reward files, not only the excerpt it reads. `tests/test_judge_robustness.py` tests both fixes: scanner tests in normal CI, and a live set of honest runs and hacks the judge must classify. The end-to-end deepagents scenario is `test_integration_suite.py::test_deepagents_deepseek_rollout_is_real_and_judged`.

## Keyless tests

The keyless tier (`deterministic/`) runs the real `bench` CLI, a real sandbox and a real agent (`claude-agent-acp`), with the model replaced by a scripted fake provider. It needs no model keys, costs nothing, and gives the same result every run. `tests/test_deterministic_integration.py`, `tests/test_docker_parallel_isolation.py` and `tests/test_integrity_deterministic.py` run it. It runs in the default suite whenever a sandbox is available, and on every PR in CI.

The scenarios call the older spellings `bench eval run` and `bench eval branch`, which take the same flags as `bench run` and `bench branch`.

### How it works

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

### Scenarios

| Test | CLI | Asserts |
|---|---|---|
| `test_hello_world_passes` | `bench eval run` | reward 1.0; completed · scored; verifier ran; all timing keys |
| `test_wrong_answer_scores_zero` | `bench eval run` | reward 0.0; completed · scored |
| `test_agent_timeout_with_passing_file_is_timed_out_and_passed` | `bench eval run`, agent timeout 45 s | timed out · scored 1.0; `agent_timeout_info` names the pending tool call; verifier ran |
| `test_agent_crash_is_unscored_and_not_verified` | `bench eval run` | the scripted tool call kills the Claude Code CLI (`CLAUDE_CODE_EXECUTABLE`), and the adapter reports it: errored (`acp_error`) · unscored; no reward; verifier did not run |
| `test_verifier_error_is_an_assessment_error_never_zero` | `bench eval run`, broken `test.sh` | completed · assessment error; `rewards` null, never 0; `verifier_failure` |
| `test_eval_job_summary` | the same job | `summary.json` counts and categories, token totals |
| `test_branch_two_children_in_place` | `bench eval branch --checkpoint-after-prompt 1 --child … --child …` | V = 0.5, `value_stderr`, children 1.0 / 0.0 from the verifier, `children_mode`, child folders, parent restored and scored 1.0 |
| `test_branch_two_children_parallel` | the same with `--concurrency 2` | as above with isolated children, each child's own trial folder checked like a trial |
| `test_retry_from_checkpoint_keeps_both_scores` | `bench eval run --checkpoints prompt:1 --retry-from-checkpoint on-failure --retry-prompt …` | the trial keeps 0.0, the retry child scores 1.0, both recorded; kept snapshots deleted afterwards |

`tests/test_docker_parallel_isolation.py` adds two Docker-only scenarios without the fake: the `oracle` agent in in-process `Evaluation`s, with the interleaving that used to lose containers forced by hooks. `test_two_evaluations_at_once_keep_their_containers` holds job A between `compose create` and `compose start` while job B runs from start to end, then checks that A starts the same container on the same network. `test_a_branch_restore_survives_a_sibling_jobs_sweep` runs a sibling job's leftover sweep inside a branch restore, after the old `main` container is removed and before its replacement joins the network. Both fail on the old daemon-wide `docker container prune` / `docker network prune` (`sandbox/_docker_sweep.py` says why).

Every scenario trial is also checked for: token usage and cost exactly as the fake's fixed usage implies (per captured call on the `proxy` route; per agent message on the `native` route), `usage_source`, `usage_tracking` endpoint, non-negative timing, the ATIF-v1.7 structure rules that `export_atif` promises, the public `benchflow.trial` document against `docs/reference/schemas/benchflow-trial.v1.schema.json`, and tool call ids from the fake. Then the run-independent facts are compared with the golden file in `deterministic/golden/<scenario>.json`: outcome labels, a `result.json` subset (rewards, counts, error categories, usage, final metrics, trajectory summary, timing keys), the captured provider calls, the ACP trajectory and the ATIF export, with timestamps, receipts and session ids stripped. Claude Code's session-title call is left out of the golden file, and usage there is net of it: whether that call reaches the proxy before BenchFlow stops the agent is a race (seen on both backends). It is still checked against the recorded totals. Golden files are backend-neutral: the same set passed on Daytona and on Docker.

### Running

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

### Adding a scenario

1. Add a script to `deterministic/task/environment/fake_llm/scripts.json`: a list of steps, each with `text` and optionally `tool` (a tool name the agent offers, matched exactly or as a `__<name>` suffix) and `input` (the tool arguments). Step *n* answers the *n*-th model call after the marked prompt; past the end the fake answers `Done.`. Every step needs `text`, because the checks count one agent message per model call.
2. If the task itself must differ, add a `TaskVariant` (marker, agent timeout, broken verifier) or extend `materialize_task`. Keep `environment/` identical across variants so every scenario shares one image.
3. Add the variant to `EVAL_VARIANTS` (it joins the shared `bench eval run` job) or add a session fixture with its own CLI call, and write the test: call `_check_trial(trial_dir, "<golden name>", route=...)` and assert what the scenario is about.
4. Generate its golden file on purpose (below), read it, and commit it with the test.

`tests/test_fake_llm_provider.py` pins the fake's own contract without a sandbox; extend it when you change the fake.

### Regenerating golden files on purpose

A golden mismatch means agent-visible behaviour changed: the trajectory, the ATIF export, usage, outcome labels or the `result.json` fields listed above. If the change is intended:

```bash
BENCHFLOW_UPDATE_GOLDEN=1 uv run python -m pytest tests/test_deterministic_integration.py -k <scenario>
git diff tests/integration/deterministic/golden/
uv run python -m pytest tests/test_deterministic_integration.py -k <scenario>   # must pass unchanged
```

Review the diff line by line before committing it with the code change that caused it; a regenerated golden file is a claim that the new behaviour is correct.

### Not covered

- `codex-acp` and the other agents: the fake speaks only the Anthropic Messages API. Any LiteLLM-routed agent can be pointed at a fake the same way (`BENCHFLOW_PROVIDER_BASE_URL` sets the proxy's upstream); covering one needs the upstream protocol it uses (OpenAI Responses or Chat Completions) added to the fake.
- The verifier's pytest plugin guard refusals: they need a pytest-based verifier and a plugin location the agent can write, which this task does not have.
- The same scenarios with `--sandbox modal` or other providers.
