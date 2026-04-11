# Changelog

## 0.2.0 — 2026-04-09

**First public release.** A near-complete rearchitecture from the 0.1.x era. API surface has changed — assume breaking changes. Future releases will maintain compatibility within the 0.2.x line.

### Added

- **Multi-agent, multi-provider, multi-auth matrix** — one YAML config, any supported agent × model × provider × auth combination. 12 end-to-end tested combinations documented in `.dev-docs/tested-agents.md`.
- **Subscription auth support** — use `claude login`, `codex --login`, `gemini` OAuth credentials directly. No API keys required for host-based agent workflows.
- **Vertex AI support** — ADC auth for `google-vertex/`, `anthropic-vertex/`, `vertex-zai/` prefixed models.
- **Provider registry** — data-driven custom LLM endpoints via `ProviderConfig` (`src/benchflow/providers.py`). Adding a new provider = registry dict entry, no code changes.
- **Agent registry overhaul** — `AgentConfig` now holds `env_mapping`, `credential_files`, `home_dirs`, `skill_paths`. New agent = registry edit only.
- **`benchmarks/` directory** with reusable YAML configs and runner scripts:
  - `run_skillsbench.py` + `skillsbench-codex-gpt54.yaml`
  - `run_tb2.py` + `tb2_single-codex-gpt54.yaml` + `tb2_multiturn-codex-gpt54.yaml`
- **Auto task download** via `ensure_tasks()` — `terminal-bench-2` and `skillsbench` clone into `.ref/` on first run.
- **`benchflow tasks init`** command for scaffolding new tasks.
- **`benchflow tasks check`** command for task validation.
- **`benchflow cleanup`** command with `--max-age` filtering (default 24h).
- **Oracle agent support** — run `solution/solve.sh` directly for task validation.
- **Hello-world-task example** for sanity-testing the agent pipeline.
- **Agent test scripts** in `tests/examples/` — `test_claude.sh`, `test_codex.sh`, `test_gemini.sh`, `test_openclaw.sh`.
- **Claude Code skill** in `.claude/skills/benchflow/` teaching agents how to use the framework.
- **Model generation params** via env vars (`BENCHFLOW_TEMPERATURE`, `BENCHFLOW_TOP_P`, `BENCHFLOW_MAX_TOKENS`).
- **OpenClaw ACP shim** with trajectory parsing and skills support.
- **ACP trajectory capture** — full multi-turn agent trajectories via ACP protocol.
- **HTTP proxy + OTel trajectory capture** (implemented, not yet wired).
- **ATIF trajectory export model** (implemented, not yet wired).

### Changed

- **`SDK.run()` decomposed** into 14 private methods, each 10–80 lines. Core modules extracted: `_models.py`, `_trajectory.py`, `_env_setup.py`, `_scoring.py`.
- **`_resolve_agent_env` decomposed** into focused helpers: `_auto_inherit_env`, `_inject_vertex_credentials`, `_resolve_provider_env`, `_check_subscription_auth`.
- **Harbor imports** — replaced star import with explicit allowlist.
- **Sandbox home dirs** derived from agent registry instead of hardcoded lists.
- **Credential file resolution** data-driven via `AgentConfig.credential_files` and `ProviderConfig.credential_files`.
- **Skill loading redesigned** — agent-targeted with proper precedence.
- **Skills auto-distributed** from `task.toml` `skills_dir`.
- **`STATUS.md` consolidated** into `CLAUDE.md`.
- **`DEFAULT_AGENT`** (`claude-agent-acp`) and **`DEFAULT_MODEL`** (`claude-haiku-4-5-20251001`) constants extracted into `job.py`.
- **`openclaw-gemini` merged** into `openclaw` with runtime provider mode selection (`BENCHFLOW_PROVIDER_NAME`).

### Fixed

- **API keys leaking in `ps aux`** — Docker exec `-e K=V` no longer visible in process list. Env vars written inside the container instead.
- **Subscription auth skipped without `-m`** — `benchflow run` without `--model` was skipping subscription auth entirely. Now checks correctly.
- **ADC credentials break with `sandbox_user`** (#111) — Google ADC and Codex credentials now written to sandbox user's home instead of `/root/`.
- **Daytona sandboxes not cleaned up** (#102) — auto-delete after max age, `benchflow cleanup` CLI command.
- **`benchflow cleanup` ignoring `--max-age`** — was deleting everything regardless of age. Now filters correctly.
- **readline buffer overflow crashes trial** (#98) — handled gracefully.
- **OpenClaw ACP shim loses tool command text** (#96) — trajectory now captures full commands.
- **OpenClaw ACP shim hardcodes `anthropic/` prefix** (#95) — now routes correctly for Gemini/GLM models.
- **Oracle agent `PermissionError`** writing `agent/oracle.txt` (#91) — writes inside container.
- **Oracle path skips `pre_agent_hooks`** (#92) — services now start before oracle runs.
- **Trial data parity with Harbor** (#90) — richer `result.json`, agent logs, per-phase timing.
- **`SDK.run()` `PermissionError`** — `jobs_dir` subdirectories were created as root (#88).
- **Partial trajectory lost on timeout** — now saved before timeout raises.
- **Redundant `--version` binary check** — was wasting 30s per trial, removed.
- **`ANTHROPIC_API_KEY` written to OpenClaw's native auth store** for direct OpenAI usage.
- **`codex` binary check 10s timeout** — now passes `agent_env` through.
- **Docker Compose env var compatibility** — replaced `--env-file` with inline export prefix.
- **Trajectory fallback** — scrape agent-native trajectory files when ACP `session/update` is empty (#94).
- **`litellm` upgraded to 1.83.0** for CVE-2026-35030.
- **Transitive dependency security alerts** resolved — `aiohttp`, `cryptography`, `Pygments`, `requests` bumped (13 Dependabot alerts closed).
- **Debug logging added** to silent exception handlers across the codebase.

### Benchmark Results

| Benchmark | Agent | Model | Score |
|-----------|-------|-------|-------|
| TB2 single-turn | `codex-acp` | GPT-5.4\* | **69.7%** (62/89) |
| TB2 single-turn | `claude-agent-acp` | Sonnet 4.6 | 58.4% (52/89) |
| TB2 multi-turn | `codex-acp` | GPT-5.4\* | **62.9%** (56/89) |
| TB2 multi-turn | `claude-agent-acp` | Haiku 4.5 | 37.1% (33/89) |
| SkillsBench | `codex-acp` | GPT-5.4\* | **37.2%** (32/86) |

\*GPT-5.4 runs used `OPENAI_REASONING_EFFORT=medium`.

**Notable finding:** Multi-turn self-critique hurts capable models (GPT-5.4 regresses −6.8pp) but helps weaker models (Haiku 4.5 gains +9.6pp).

### Testing

- **232 unit tests** pass (up from 66 in 0.1.x).
- New test files: `test_tasks.py`, `test_skills.py`, `test_env_setup.py`, `test_resolve_env_helpers.py`, `test_credentials.py`, `test_providers.py`, `test_env_mapping.py`, `test_scoring.py`, `test_sandbox.py`, `test_subscription_auth.py`, `test_agent_model_decouple.py`.

### Documentation

- New `.dev-docs/tested-agents.md` — 12 agent × model × provider × auth combinations.
- New `.dev-docs/sdk-reference.md` — SDK API reference.
- New `.dev-docs/sdk-refactor-notes.md` — architecture rationale.
- `README.md` rewritten with Quick Start, SDK examples, CLI reference, Benchmark Results table.
- `CLAUDE.md` documents architecture, test policy, known issues, backlog.

### Known Issues (P1)

- **Harbor private attributes** — `process.py` accesses `env._sandbox`, `env._strategy`, `env._docker_compose_paths`. No public APIs in Harbor. Blocked on upstream. Planned for 0.3.0 via Harbor internalization.

### Contributors

- [@kywch](https://github.com/kywch) (Kyoung Whan Choe) — core refactor, benchmark runs, agent test scripts
- [@xdotli](https://github.com/xdotli) (Xiangyi Li) — SDK, providers, Vertex AI, OpenClaw shim, subscription auth

### Deprecated (from 0.1.x)

- `BaseAgent` re-export — planned removal in 0.3.0
- `Trial` re-export — planned removal in 0.3.0
- Old author metadata (Hongji Xu / `kirk@benchflow.ai`) — updated

### Migration from 0.1.x

0.1.x users should treat this as a fresh install. The SDK API, CLI, registry pattern, and task format have all changed. There is no automatic migration path. See `.dev-docs/sdk-reference.md` for the new SDK, and `tests/examples/` for end-to-end examples.
