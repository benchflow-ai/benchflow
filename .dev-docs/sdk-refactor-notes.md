# SDK Refactor Notes (April 2026)

Notes from the `refactor` branch: 31 commits, +3740/-1159 lines across 49 files.
Preserves design decisions, rationale, and non-obvious details. For progress
tracking and commit-by-commit history, see `git log main..refactor`.

---

## Contents

1. [Problem Statement](#1-problem-statement)
2. [Agent x Model x Provider Analysis](#2-agent-x-model-x-provider-analysis)
3. [Design Decisions](#3-design-decisions)
4. [Key Metrics](#4-key-metrics)
5. [Public API Contract](#5-public-api-contract)
6. [Risk Post-Mortem](#6-risk-post-mortem)
7. [TDD Refactoring](#7-tdd-refactoring)

---

## 1. Problem Statement

`sdk.py` was a 989-line file whose `run()` method handled 10+ responsibilities
in a single try/finally block. Every new agent or provider required editing this
file, often by adding another hardcoded if-block.

### What was wrong

**A. God method.** `run()` mixed env resolution, Dockerfile preprocessing,
container lifecycle, agent installation, credential injection, ACP communication,
trajectory capture, verification, and result serialization in one 576-line method.

**B. Agent-specific hacks inline in the orchestrator.**

| Hack | Problem |
|------|---------|
| `if "codex" in agent` — write `/root/.codex/auth.json` | New agents needing config files = new if-blocks |
| GCP ADC file write to container | Provider concern baked into orchestrator |
| `for d in .claude .gemini .openclaw .pi .agents .codex` | Adding an agent means editing a shell string |
| `ln -sf /skills /root/.claude/skills && ln -sf /skills /root/.gemini/skills` | Hardcoded skill symlink targets |

**C. `BENCHFLOW_PROVIDER_*` env vars only consumed by openclaw.** For other
agents, custom provider support required manual `--ae` overrides — fragile,
undiscoverable, error-prone.

**D. Zero test coverage for sdk.py internals.** Env resolution (100+ lines of
dict manipulation), credential injection, sandbox user setup, and result building
were all untested because they were buried inside `run()`.

### What was good (kept as-is)

- **`providers.py`** — data-driven provider registry. "Add one dict entry" works.
- **`registry.py`** — agent registry with `AgentConfig`. Clean separation.
- **`openclaw_acp_shim.py`** — reference implementation of consuming `BENCHFLOW_PROVIDER_*`.
- **The try/finally cleanup structure in `run()`** — essential for resource management.

---

## 2. Agent x Model x Provider Analysis

### Agents

| Agent | Native Protocol | Native Auth |
|-------|----------------|-------------|
| claude-agent-acp | anthropic-messages | ANTHROPIC_API_KEY |
| pi-acp | anthropic-messages | ANTHROPIC_API_KEY |
| openclaw | openai-completions | (from model, via shim) |
| codex-acp | openai-completions | OPENAI_API_KEY |
| gemini | google/openai | GOOGLE_API_KEY |

### Providers

| Prefix | Auth | Protocol(s) | Key Env Var |
|--------|------|-------------|-------------|
| *(none — direct Anthropic)* | API key | anthropic-messages | ANTHROPIC_API_KEY |
| *(none — direct OpenAI)* | API key | openai-completions | OPENAI_API_KEY |
| *(none — direct Google)* | API key | openai-completions | GOOGLE_API_KEY |
| google-vertex/ | ADC | openai-completions | GOOGLE_CLOUD_PROJECT + ADC |
| anthropic-vertex/ | ADC | anthropic-messages | GOOGLE_CLOUD_PROJECT + ADC |
| zai/ | API key | openai-completions, anthropic-messages | ZAI_API_KEY |

### Three coupling patterns (before refactor)

**Pattern 1: Native** — agent speaks provider's protocol directly.
`claude-agent-acp + claude-sonnet-4-6 → ANTHROPIC_API_KEY → done`

**Pattern 2: BENCHFLOW_PROVIDER_*** — SDK resolves endpoint, shim configures agent.
Worked only for openclaw.
`openclaw + zai/glm-5 → find_provider() → BENCHFLOW_PROVIDER_* → shim → openclaw.json`

**Pattern 3: Manual --ae override** — user manually maps provider config to
agent-native env vars. This is what the refactor eliminated.
`claude-agent-acp + zai/glm-5 → --ae ANTHROPIC_BASE_URL=... --ae ANTHROPIC_AUTH_TOKEN=...`

---

## 3. Design Decisions

### 3.1 env_mapping on AgentConfig

Translates generic `BENCHFLOW_PROVIDER_*` env vars into agent-native env vars.
Applied by the SDK after provider resolution, before launching the agent.

```python
# Per-agent mapping (registry.py)
"claude-agent-acp": AgentConfig(
    env_mapping={
        "BENCHFLOW_PROVIDER_BASE_URL": "ANTHROPIC_BASE_URL",
        "BENCHFLOW_PROVIDER_API_KEY": "ANTHROPIC_AUTH_TOKEN",
    },
)

# SDK applies via setdefault (preserves explicit --ae overrides)
for src, dst in agent_cfg.env_mapping.items():
    if src in agent_env:
        agent_env.setdefault(dst, agent_env[src])
```

Result: `benchflow run -a claude-agent-acp -m zai/glm-5` just works. No `--ae`.

### 3.2 credential_files on AgentConfig + ProviderConfig

Two kinds of credential files, now declarative in the registry:

- **Agent-level** (e.g. codex `auth.json`): `AgentConfig.credential_files`
- **Provider-level** (e.g. Vertex ADC): `ProviderConfig.credential_files`

```python
@dataclass
class CredentialFile:
    path: str            # Target path in container. {home} → /home/{user} or /root
    env_source: str      # Env var to read value from
    template: str = ""   # Template with {value} placeholder. Empty = raw value.
    mkdir: bool = True
```

**Non-obvious:** The `{home}` placeholder resolves to `/home/{sandbox_user}` when
a sandbox user is set, or `/root` otherwise. This matters because credential files
must be readable by the user running the agent, not just root.

The SDK replaces the two hardcoded if-blocks with one generic loop over
`credential_files` from both agent and provider configs.

**Forward-looking:** Subscription/OAuth auth becomes a registry-only change:
```python
CredentialFile(
    path="{home}/.claude/credentials.json",
    env_source="CLAUDE_SESSION_TOKEN",
)
```
The user exports the token; the existing generic loop writes it. The hard part
is UX/docs, not architecture.

### 3.3 Sandbox dirs derived from registry

Previously a hardcoded shell string. Now derived at runtime from all registered
agents' `skill_paths`, `credential_files`, and `home_dirs` fields. The set always
includes `.local`. Adding an agent to the registry automatically updates the
sandbox dir list.

### 3.4 Module extraction

| Module | Lines | Extracted from sdk.py |
|--------|-------|-----------------------|
| `_models.py` | 67 | `RunResult`, `AgentInstallError`, `AgentTimeoutError` |
| `_trajectory.py` | 82 | ACP native, agent-scraped, Gemini trajectory capture |
| `_env_setup.py` | 242 | Dep staging, skills injection, DinD detection/patching |
| `_scoring.py` | 39 | `extract_reward`, `classify_error`, `pass_rate` (see §7) |

All have zero SDK class dependencies. Re-exported from `sdk.py` for backwards compat.

### 3.5 run() decomposition

Decomposed into 14 private methods (each 10-80 lines). The `run()` body is now
~95 lines orchestrating the call sequence. Methods are either static (testable
without mocks) or async (structural, need a mock Harbor env for testing).

### 3.6 DockerProcess env injection (security fix)

**Problem:** `docker compose exec -e K=V` exposes API keys in `ps aux` on the host.

**Approach tried and failed:** `--env-file` flag — not supported in Compose v5.1.1.

**Final fix:** Write env vars to a file *inside* the container via a separate
`exec + stdin` call, then `source` and `rm` in the main command. Secrets never
appear in `ps aux` on the host. Works with all Compose versions.

---

## 4. Key Metrics

| Metric | Before | After |
|--------|--------|-------|
| `sdk.py` | 1015 lines, 1 god method | 711 lines, 16 methods (none >80 lines) |
| `run()` body | ~560 lines | ~95 lines |
| Extracted modules | 0 | 4 (`_models`, `_trajectory`, `_env_setup`, `_scoring`) |
| Tests | 170 | 232 (+62 new across 8 test files) |
| Adding a new agent | Edit sdk.py (if-blocks, hardcoded lists) | Edit registry.py (one dict entry) |
| Adding a new provider | Manual `--ae` overrides | Edit providers.py (one dict entry) |

sdk.py is 711 (not the target 600) because the 14 extracted private methods are
SDK class methods. The god-method problem is solved — no single method >80 lines.

---

## 5. Public API Contract

The downstream repo (`smolclaws/packages/clawbench`) uses:

```python
from benchflow import SDK, RunResult

result = await SDK().run(
    task_path, agent, prompts, model, agent_env,
    job_name, trial_name, jobs_dir, environment,
    skills_dir, sandbox_user, pre_agent_hooks, context_root,
)
result.trajectory   # list[dict]
result.rewards      # dict | None
result.error        # str | None
```

Also exported: `AgentInstallError`, `AgentTimeoutError`, `register_agent()`,
`stage_dockerfile_deps()`.

### Backwards compatibility guarantees

- `from benchflow.sdk import RunResult` works (re-export from `_models.py`)
- `from benchflow.sdk import _capture_session_trajectory` works (re-export from `_trajectory.py`)
- `from benchflow.sdk import stage_dockerfile_deps` works (re-export from `_env_setup.py`)
- `register_agent()` new kwargs (`env_mapping`, `credential_files`, `home_dirs`) all optional with defaults

### Impact per step

| Step | Breaks downstream? |
|------|--------------------|
| env_mapping on AgentConfig | No — internal to registry |
| credential_files | No — replaces hardcoded if-blocks, same behavior |
| Sandbox dirs from registry | No — internal shell command change |
| Module extraction | No — re-exports preserve both import paths |
| run() decomposition | No — all new methods are `_private` |

---

## 6. Risk Post-Mortem

| Change | Planned Risk | Actual Outcome |
|--------|-------------|----------------|
| env_mapping | Low | No issues. setdefault preserved --ae overrides. |
| credential_files | Medium | Clean. {home} placeholder handled root vs sandbox. |
| Sandbox dir derivation | Low | No issues. |
| Module extraction | Low | Re-exports + import updates worked first try. |
| run() decomposition | Medium | Smooth. Static methods tested via TDD. |

### What was NOT changed (by design)

- `providers.py` — already well-structured
- `openclaw_acp_shim.py` — already consumes BENCHFLOW_PROVIDER_* correctly
- The try/finally structure in `run()` — preserved, essential for cleanup
- Oracle mode — extracted to `_run_oracle()` but kept simple

### Known remaining issue

`AGENT_INSTALLERS` / `AGENT_LAUNCH` in sdk.py are derived from `AGENTS` registry.
`registry.py` imports them back from `sdk.py` in `register_agent()`. This circular
import works because registry imports at function call time, not module level —
but it's fragile. Recommended fix: move both dicts to `registry.py`, add re-exports
in `sdk.py`. Low priority — works as-is.

---

## 7. TDD Refactoring

Separate from the SDK decomposition, this addressed DRY violations and untestable
code in `job.py` and `metrics.py`.

### Why

- `extract_reward`: inner function in `job.py` couldn't be imported or tested.
  Test at `test_job.py` re-implemented it instead of calling the real thing.
- `classify_error`: same magic strings in `job.py` and `metrics.py` with a
  divergence — metrics used broad `"install"` match, job used specific
  `"install failed"`. The actual error message is `"Agent X install failed (rc=N)"`,
  so `"install failed"` is correct.
- `pass_rate` / `pass_rate_excl_errors`: identical properties in `JobResult`
  and `BenchmarkMetrics`.

### Result: `_scoring.py`

4 constants + 4 pure functions, zero dependencies:
- `INSTALL_FAILED`, `PIPE_CLOSED`, `ACP_ERROR`, `TIMED_OUT`
- `extract_reward(result) -> float | None`
- `classify_error(error) -> str | None`
- `pass_rate(*, passed, total) -> float`
- `pass_rate_excl_errors(*, passed, failed) -> float`

### AI-Readiness Scorecard

| Dimension | Before | After |
|-----------|--------|-------|
| D1: Rules & Config | 2 | 3 (PreCommit hook, settings.json) |
| D2: File Organization | 2 | 3 (no god method, 14 focused methods) |
| D3: Test & Verification | 1 | 2 (PreCommit hook, test protection rule) |
| **Total** | **5** | **8** |
