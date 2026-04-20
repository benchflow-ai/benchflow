# BenchFlow Architecture

## System Overview

BenchFlow runs AI coding agents inside isolated sandboxes, evaluates their output with a task-specific verifier, and returns structured results. The SDK wraps Harbor environments (Docker or Daytona) and communicates with agents via ACP (Agent Client Protocol) — JSON-RPC 2.0 over stdio. Each trial is one task run by one agent; a `Job` fans out many trials concurrently and aggregates results.

```
  Task (Harbor format) + TrialConfig (YAML or Python)
       |
       v
  bf.run(config) → Trial.create(config) → trial.run()
       |
       |  SETUP (host)
       |    resolve config, create env object, write config
       |
       |  START (sandbox)
       |    spin up sandbox ──> Daytona / Docker container
       |    upload task files, start services
       |
       |  SCENES (one or more)
       |    for each Scene:
       |      install_agent, write credentials, setup sandbox user
       |      deploy_skills, lockdown_paths
       |      for each Turn in scene:
       |        connect_as(role) ── ACP (JSON-RPC/stdio) ──> Agent Process
       |        execute(prompt) <── session/update notifications ─── |
       |        disconnect()
       |
       |  VERIFY
       |    harden_before_verify
       |    _verify ──> Harbor Verifier
       |
       v
  RuntimeResult (rewards, trajectory, error, timing)
```

---

## Trial Run Phases

`Trial.run()` is a strict-order orchestrator. `bf.run()` creates a Trial internally and calls `trial.run()`.

### Phase 1: SETUP (host)

- **`setup()`** — creates `jobs/{job_name}/{trial_name}/` with subdirs `agent/`, `verifier/`, `artifacts/`, `trajectory/`. Pre-creates dirs in Python so Docker doesn't own them as root. Resolves agent env vars, reads `instruction.md`, stages Dockerfile deps, creates the environment object, writes `config.json`.

### Phase 2: START (sandbox)

- **`_start_env_and_upload`** — calls `env.start()`, uploads `instruction.md` and `solution/`.
- **`pre_agent_hooks`** — user-supplied async `(env) -> None` callables; run after container is up, before agent starts. Canonical use: starting background services.

### Phase 3: AGENT

Replaced by **`_run_oracle`** when `agent="oracle"` — runs `solution/solve.sh` directly, no ACP. Oracle still calls `setup_sandbox_user` and `lockdown_paths`. For all real agents:

- **`install_agent`** — looks up the install command string from `AGENT_INSTALLERS[agent]` and runs it as root; streams output to `agent/install-stdout.txt`. Non-zero exit raises `AgentInstallError`.
- **`write_credential_files`** — writes `AgentConfig.credential_files` into the container.
- **`upload_subscription_auth`** — copies host CLI login files when `_BENCHFLOW_SUBSCRIPTION_AUTH` is set (API keys take precedence).
- **`setup_sandbox_user`** — creates sandbox user, copies agent home dirs, sets ownership.
- **`deploy_skills`** — copies skill files into `AgentConfig.skill_paths` (expanding `$HOME` and `$WORKSPACE`).
- **`lockdown_paths`** — `chown root:root` + `chmod 700` on `["/solution", "/tests"]` (and any `sandbox_locked_paths`) so the agent can't read reference answers.
- **`connect_acp`** — starts the agent subprocess, runs ACP handshake: `initialize` → `session_new` → optionally `set_model`.
- **`execute_prompts`** — sends each resolved prompt via `ACPClient.prompt()`; blocks until `end_turn` or `max_tokens`. Streaming `session/update` notifications accumulate in `ACPSession`.

### Phase 4: VERIFY

- **Trajectory fallback** — if ACP trajectory is empty, `_scrape_agent_trajectory` reads agent-native log files from the container (currently only Gemini CLI). Source becomes `"scraped"`. `n_tool_calls` is never overwritten by scraped data.
- **Partial ACP** — if an error interrupted a live session, `finally` calls `_capture_session_trajectory`; source becomes `"partial_acp"`.
- **`harden_before_verify`** — resets permissions for the verifier (undoes lockdown).
- **`_verify`** — calls `verifier.verify()` with timeout from `task.toml`. Returns `rewards` dict or sets `verifier_error`.
- **`_build_result`** — assembles `RunResult`, writes `result.json`, `timing.json`, `prompts.json`, `trajectory/acp_trajectory.jsonl`.

---

## ACP Protocol

BenchFlow is the ACP **client**; the agent is the ACP **server**. The agent process stays alive across all prompts, preserving full conversation context between turns.

### Transport

`StdioTransport`, `DockerProcess`, and `DaytonaProcess` all use line-delimited JSON on stdio. `DockerProcess` and `DaytonaProcess` use a 10 MB readline buffer (`_BUFFER_LIMIT`); `StdioTransport` uses 1 MB. Env vars are sourced from a temp file inside the container, not CLI args, to keep secrets off `ps aux`.

### Session Lifecycle

```
  connect()       start transport
  initialize()    → agent name, version, capabilities
  session_new()   → sessionId
  [set_model()]   → optional model override
  loop:
    prompt()      → session/update notifications accumulate
                  → session/request_permission (auto-approved)
                  ← stopReason: "end_turn" | "max_tokens"
  close()
```

**`session/update`** notification types: `tool_call`, `tool_call_update`, `agent_message_chunk`, `agent_thought_chunk`. Accumulated by `ACPSession.handle_update()` into `ToolCallRecord` objects.

**`session/request_permission`** — auto-approved in benchmark mode (`bypassPermissions` > `allow_always` > `allow_once`).

**`ACPError`** — raised on JSON-RPC error responses; stored in `result.error`.

---

## Registry Pattern

Adding an agent or provider is a **dict entry only** — no changes to `sdk.py`, `job.py`, or other orchestration code. `tests/test_registry_invariants.py` enforces the contract.

### AgentConfig (`src/benchflow/agents/registry.py`)

| Field | Type | Purpose |
|-------|------|---------|
| `name` | `str` | Must equal the dict key |
| `install_cmd` | `str` | Bash run as root to install the agent. Must be idempotent. |
| `launch_cmd` | `str` | Starts the agent process (ACP over stdin/stdout). |
| `protocol` | `str` | `"acp"` (default) or `"cli"`. |
| `requires_env` | `list[str]` | Env vars that must be present; validated before container start. |
| `api_protocol` | `str` | `"anthropic-messages"` or `"openai-completions"`. |
| `env_mapping` | `dict[str, str]` | Maps `BENCHFLOW_PROVIDER_*` keys to agent-native names. |
| `skill_paths` | `list[str]` | Sandbox paths for skill content. Use `$HOME/` or `$WORKSPACE/`. |
| `credential_files` | `list[CredentialFile]` | Files written into the container before launch. |
| `home_dirs` | `list[str]` | Extra dot-dirs under `$HOME` to copy to the sandbox user. |
| `subscription_auth` | `SubscriptionAuth \| None` | Host CLI login files that substitute for an API key. |
| `install_timeout` | `int` | Default: 900 seconds. |
| `default_model` | `str` | Default model ID when `--model` is omitted. |

**Adding an agent** — append to `AGENTS` in `registry.py`, or call `register_agent()` before `bf.run()`:

```python
from benchflow import register_agent

register_agent(
    name="my-agent",
    install_cmd="npm install -g my-agent",
    launch_cmd="my-agent --acp",
    requires_env=["MY_API_KEY"],
    skill_paths=["$HOME/.my-agent/skills"],
)
```

A new code path in `sdk.py` is justified only if the agent fundamentally bypasses ACP (like `oracle`). Don't add more oracle-style exceptions without equivalent justification.

### ProviderConfig (`src/benchflow/agents/providers.py`)

Provider names are model-ID prefixes: `"zai/glm-5"` matches the `"zai"` provider.

| Field | Type | Purpose |
|-------|------|---------|
| `name` | `str` | Must equal the dict key / model-ID prefix. |
| `base_url` | `str` | Primary endpoint. May contain `{placeholder}` tokens. |
| `api_protocol` | `str` | `"anthropic-messages"` or `"openai-completions"`. |
| `auth_type` | `str` | `"api_key"`, `"adc"` (GCP ADC), or `"none"`. |
| `auth_env` | `str \| None` | Env var holding the API key (when `auth_type == "api_key"`). |
| `url_params` | `dict[str, str]` | `{placeholder: ENV_VAR}` for `base_url` expansion. |
| `endpoints` | `dict[str, str]` | `{api_protocol: url}` for multi-protocol providers. |
| `models` | `list[dict]` | Model metadata for agent shims. Each entry requires `id`. |
| `credential_files` | `list[dict]` | Files written for ADC providers. |

**Adding a provider** — append to `PROVIDERS` in `providers.py`. No other changes needed. `find_provider()` matches longest prefix first.

### Tested Agent × Model × Provider Matrix

| Agent | Model | Provider | Auth |
|-------|-------|----------|------|
| `claude-agent-acp` | claude-sonnet-4-6 | anthropic | subscription or API key |
| `claude-agent-acp` | claude-sonnet-4-6 | anthropic-vertex | GCP ADC |
| `claude-agent-acp` | glm-5 | zai | `ZAI_API_KEY` |
| `codex-acp` | gpt-5.4 | openai | subscription or API key |
| `gemini` | gemini-3-flash-preview | google | subscription or `GEMINI_API_KEY` |
| `gemini` | gemini-2.5-flash | google-vertex | GCP ADC |
| `openclaw` | gemini-3-flash-preview | google-vertex | GCP ADC |
| `openclaw` | claude-sonnet-4-6 | anthropic-vertex | GCP ADC |
| `openclaw` | glm-5 | zai | `ZAI_API_KEY` |
| `openclaw` | gpt-5.4 | openai | `OPENAI_API_KEY` |

Auth precedence: Vertex ADC > explicit API key > host subscription auth.

---

## Trajectory Capture

Source is stored in `RunResult.trajectory_source` and written to `result.json`.

| Source | When | Trust |
|--------|------|-------|
| `"acp"` | Normal run — agent emits `session/update` notifications | Trusted. `n_tool_calls` is set from this source only and never overwritten. |
| `"scraped"` | ACP trajectory empty (agent crashed before sending updates). Reads agent-native log files from container. | **Untrusted** — files live in agent-writable dirs. SDK logs a warning. |
| `"partial_acp"` | Error interrupted a live session; `finally` block captures whatever `ACPSession` accumulated. Sets `partial_trajectory = True`. | Partial / best-effort. |

### Trajectory event format

```python
{"type": "tool_call",     "tool_call_id": "...", "kind": "bash", "title": "...", "status": "completed", "content": [...]}
{"type": "agent_message", "text": "..."}
{"type": "agent_thought", "text": "..."}
{"type": "oracle",        "command": "solution/solve.sh", "return_code": 0, "stdout": "..."}  # oracle mode only
```

Written to `trial_dir/trajectory/acp_trajectory.jsonl` (one JSON object per line).

---

## Error Taxonomy

Errors are classified by `classify_error()` in `src/benchflow/_scoring.py`. `Job` uses these categories for retry decisions via `RetryConfig.should_retry()`.

### Agent errors (`RunResult.error`)

| Category | Match string | Retry by default | Meaning |
|----------|-------------|-----------------|---------|
| `INSTALL_FAILED` | `"install failed"` | yes | `install_cmd` exited non-zero. |
| `PIPE_CLOSED` | `"closed stdout"` | yes | Agent process closed stdout unexpectedly (crash/OOM). |
| `ACP_ERROR` | `"ACP error"` | yes | JSON-RPC error from agent (bad key, rate limit, protocol failure). |
| `TIMED_OUT` | `"timed out"` | no | Agent exceeded `task.config.agent.timeout_sec`. |
| (other) | any other string | no | Unexpected exceptions. |

### Verifier errors (`RunResult.verifier_error`)

Verifier errors are always **terminal** — `Job` never retries them. `Job` warns above 20% verifier error rate.

| Category | Match | Meaning |
|----------|-------|---------|
| `VERIFIER_FAILED` | `"verifier crashed"` | Unhandled exception in verifier script. |
| `VERIFIER_TIMEOUT` | `"verifier timed out"` | Verifier exceeded `task.config.verifier.timeout_sec`. |

### Retry logic

```python
RetryConfig(
    max_retries=2,         # Extra attempts beyond the first.
    retry_on_install=True,
    retry_on_pipe=True,
    retry_on_acp=True,
)
```

`_run_task()` calls `Trial.run()` up to `max_retries + 1` times, stopping early on success, `verifier_error`, or when `should_retry()` returns false. Docker resources are pruned between retries via `_prune_docker()`.

---

## Output Directory Layout

```
jobs/{job_name}/{trial_name}/
├── config.json              # Trial parameters (secrets filtered)
├── result.json              # rewards, n_tool_calls, timing, error
├── timing.json              # {environment_setup, agent_setup, agent_execution, verifier, total}
├── prompts.json             # Resolved prompt list
├── agent/
│   ├── install-stdout.txt   # Agent install output
│   └── {agent_name}.txt     # Agent stderr / non-JSON lines
├── trajectory/
│   └── acp_trajectory.jsonl # One JSON event per line
└── verifier/                # Written by Harbor verifier
    ├── reward.txt
    ├── test-stdout.txt
    └── ctrf.json            # pytest results (if verifier uses pytest)
```

`Job.run()` additionally writes `jobs/{job_name}/summary.json` with aggregate counts (passed, failed, errored, verifier_errored, score, score_excl_errors).
