# Changelog

## [Unreleased]

### Renamed (aliased; old names removed in 0.7)
- Benchmark adoption is now `bench eval adopt {init,convert,verify}`. It lives
  under `eval` because `eval` is the universal benchmark entry point (`eval
  create` runs a benchmark; `eval adopt` makes a foreign one runnable). Two prior
  spellings remain as hidden deprecated aliases, each printing a one-line stderr
  notice pointing at `bench eval adopt`: the original `bench agent
  create|run|verify`, and the 0.6-dev intermediate top-level `bench adopt`.
  `bench agent` now means agent management only (`list` / `show`).
- The overloaded `bench environment` group was split and is now a hidden
  **deprecated alias group** (removed in 0.7): the local sandbox lifecycle moved
  to `bench sandbox {create,list,cleanup}`, and hosted-provider browsing to
  `bench hub env {list,show,inspect}`. The old `bench environment
  create|list|cleanup|show|inspect` (plus `list --provider`/`--hub`) still work,
  each printing a one-line stderr deprecation notice. The hosted *run* path stays
  on `bench eval create --source-env`.

### Removed
- **Removed the unwired `OTelCollector`** (`benchflow.OTelCollector` /
  `benchflow.trajectories.OTelCollector`) and its `trajectories/otel.py` module.
  It was a designed-but-never-wired OTLP receiver from the v2 rewrite — never
  instantiated, never tested, and not part of any run path (BenchFlow captures
  trajectories via ACP session events and the LiteLLM callback path instead).
  This drops it from the public `__all__`; re-add it (with a test + real wiring)
  if OpenTelemetry-based capture is revived.
- Removed two unimplemented stub methods (`read_file`, `write_file`) from the
  `@runtime_checkable` `Sandbox` Protocol. No backend implemented them (backends
  expose the `upload_file`/`download_file` family) and there were no call sites,
  so they were a latent `isinstance` trap on the contract surface.
- Dead-code purge, round 3 (no public-API impact; each symbol re-verified
  zero-reference with class context): removed `TaskMetrics.audit_outcome`,
  `OTelCollector.endpoint`, `ReplayRouter.cursor`, `RuntimeResult.to_run_result`
  (legacy SDK-compat converter, unused), the never-read dataclass fields
  `ToolCall.output` and `JudgeConfig.{reference, prompt_template}`, the write-only
  `ReplayProxy._host`, the inert `AgentProtocolError.code` annotation, and an
  unused `retry_if_exception_type` import + fallback in `sandbox/daytona.py`.
- Dead-code purge, round 2 (no public-API impact; each symbol adversarially
  verified zero-reference with class context): removed seven unused `*_path`
  `@property`s from `TaskPaths`/`RolloutPaths` (`readme_path`, `gitignore_path`,
  `verifier_document_path`, `artifacts_manifest_path`, `result_path`,
  `exception_message_path`, `log_path`), the vestigial `ModalSandbox.supports_gpus`
  / `can_disable_internet` capability properties (not on the Sandbox Protocol),
  an unused module-level `logger` in `cli/continue_cmd.py`, and the orphaned
  `mcp_service_hooks_from_config` helper.
- Dead-code purge (no public-API impact unless noted): removed the unused
  `job_config_from_yaml` helper, the nominal `TASK_REPOS` back-compat dict
  (use `TASK_ALIASES`), the `_looks_like_verifier_dep_install_error` shim
  (use `contains_verifier_dep_install_marker`), the unused `parse_binary_verdict`
  reward helper (use `parse_verdict`), the dead `SandboxBackend` type alias,
  an unused `StdioTransport._read_buffer` field, and 12 redundant `rollout`
  package re-export aliases (submodule definitions unchanged).
- Removed the deprecated, hidden `benchflow skills install` CLI command. The
  SDK function `benchflow.skills.install_skill` is unchanged.
- Retired the deprecated top-level legacy CLI (`cli/legacy.py`). The dead
  0.3-era `job`/`agents`/`eval` commands are removed; `metrics` and `view` are
  promoted to first-class `bench eval metrics` / `bench eval view`; and the
  redundant `cleanup` command is dropped in favor of the existing
  `bench environment cleanup`.
- Removed the `experiments/` research/dev tooling tree (never shipped in the
  wheel) and its 6 dependent test modules, completing the dev-tree cleanup
  alongside the earlier `dashboard/` removal and `labs/` → `docs/labs`
  migration. Benchmark result files were preserved out-of-tree, not deleted.

### Changed
- `bench metrics` → `bench eval metrics` and `bench view` → `bench eval view`
  (the deprecated hidden top-level forms are gone; use the `eval` subgroup).

### Fixed
- **CLI errors now go to stderr.** `print_error` (the single CLI error sink) wrote
  to stdout, so a `bench … --json | jq` pipeline could get a non-JSON error line on
  the JSON channel. All CLI errors (and the dataset bench-version remediation hint)
  now route to stderr; exit codes are unchanged, so failures stay detectable.
- **`bench hub env list --json` now emits valid JSON at any width.** The raw
  payload was printed through Rich's console, which soft-wrapped long strings and
  injected literal newlines mid-value (unparseable JSON when piped). It is now
  written verbatim.
- **No more raw tracebacks on bad input.** Hardened the unguarded front doors a
  stress sweep surfaced: `eval create --source-repo` clone failures and
  `--tasks-dir <file>`; `eval view` on corrupt/partial trajectory artifacts
  (`prompts.json`, a bad `acp_trajectory.jsonl` line, `result.json`, a null
  `session_id`); `sandbox create` with an unknown `--sandbox` backend or a missing
  optional sandbox dependency; `tasks digest` on an unreadable file (single = clean
  error, batch = warn-and-skip); and `hub check` with a malformed/missing
  `--registry` (now a user-meaningful message, not a raw `JSONDecodeError`/`OSError`).
- **Markup-safe output.** User/author-controlled strings that look like Rich markup
  no longer crash or silently garble output: `eval list` job names, `eval metrics`
  title, `skills list` cells, and `tasks init`'s reported path are now escaped.
- **`skills eval` schema errors** no longer leak pydantic internals (private model
  name, `[type=…]` tags, the pydantic.dev URL) — just the actionable per-field text.
- **`bench environment` deprecation notice** now fires exactly once (one line,
  once per process) instead of doubling up with Typer's generic
  `DeprecationWarning`, and its aliased verbs are hidden from `--help`, matching the
  `agent` / `eval adopt` alias families.
- `benchmarks/CONVERT.md` now references the canonical `bench eval adopt verify`
  (was the deprecated `bench agent verify`) in the conversion prompt.

## 0.6.0 — 2026-06-10

### Added

- **The `task.md` task standard** — a single-file unified task format (parser,
  verifier planes, prompt sidecars, round-trip export with a machine-readable
  loss report) plus the authoring CLI: `bench tasks init / check / migrate /
  export`, with a layered `check --level` ladder up to a leaderboard-grade
  acceptance gate. See [`docs/task-standard.md`](docs/task-standard.md) and the
  [native authoring guide](docs/task-authoring-task-md.md).
- **`bench agent` benchmark-adoption router** — `create` scaffolds a benchmark
  conversion per [`benchmarks/CONVERT.md`](benchmarks/CONVERT.md), `run` drives
  the host `codex` CLI through the conversion workflow, and `verify` runs the
  parity gate (deterministic per-criterion conversion parity plus the
  agent-scale reward-distribution layer) and emits a confidence verdict, with a
  drafted support issue on divergence. `bench agent verify --rerun` independently
  re-executes the benchmark's `parity_test.py` and scores its fresh output
  (instead of trusting the recorded `parity_experiment.json`), failing closed if
  the output is not scoreable; `bench agent run -c key=value` passes codex config
  overrides through to the host codex driver (e.g. to work around `~/.codex`
  drift). `bench tasks digest` recognizes native `task.md` tasks as well as
  legacy `task.toml`.
- **ATIF and ADP trajectory artifacts** — every scored rollout now emits
  `trainer/atif.json` and `trainer/adp.jsonl` (alongside the existing
  `verifiers.jsonl`), with job-level ADP aggregation. One canonical raw
  trajectory, multiple ecosystem formats out of the box.
- **OpenReward (ORS) reward-format interop** — export BenchFlow rewards in the
  Open Reward Standard shape (`benchflow.adapters.ors`) and the `ors-episode`
  verifier strategy is recognized. (The hosted-environment episode runner that
  executes ORS environments end-to-end is in progress, not in this release.)
- **Daytona sandbox auto-reap** — orphaned sandboxes are cleaned at eval start
  (TTL-tiered; failure states reaped sooner; an idle-activity guard protects
  live runs), gated by `BENCHFLOW_DAYTONA_AUTO_REAP` (any of `0`/`false`/`no`/
  `off`, case-insensitive, disables it).
- **Registry-pinned dataset runs** — `bench eval create -d name@version`
  (e.g. `-d skillsbench@1.1`) resolves a dataset from a git-backed
  `registry.json` (see skillsbench `docs/dataset-versioning.md`): tasks are
  cloned at their pinned `git_commit_id` into `.cache/datasets` and every
  task directory is verified against its sha256 content digest before
  anything runs; the entry's `bench_version` range is checked against the
  installed benchflow. `--registry` overrides the default (skillsbench)
  registry. `result.json`/`config.json` are stamped with `dataset_name`,
  `dataset_version`, and a per-task `task_digest` (`summary.json` carries
  the name/version); `--tasks-dir` dev runs carry no dataset fields but
  still stamp a live-computed `task_digest`, so every trajectory stays
  attributable to exact task content. `bench tasks digest <dir>` prints
  the digest for task authoring, and `check_results.py` audits the stamps.
  See [`docs/running-benchmarks.md`](docs/running-benchmarks.md). (#689,
  #690, #691; `packaging` promoted to a core dependency for the
  `bench_version` check.)
- **`benchflow continue <run-folder>`** — resume a previous, unfinished
  (timed-out) `openhands` run to completion. A standalone tool (it does not
  touch the normal run path) that reconstructs the run's exact workspace and
  agent memory from the recorded `llm_trajectory.jsonl` via record-replay,
  then continues with the live model — no injected prompt — and writes a new
  HF-compatible folder with `continued_from` provenance. See
  [`docs/continue-runs.md`](docs/continue-runs.md).

### Fixed

- `bench tasks migrate` emits minimal, canonical (`schema_version`) front
  matter instead of a full defaults dump.
- Verifier `timeout_sec` is validated as a positive, finite budget
  (fail-closed at parse time; omission inherits the documented default).
- Docker `compose up` retries on the daemon network create/attach race.
- Console error messages truncate at word boundaries instead of mid-token.
- Recorded sandbox-setup timeouts and trajectory artifacts are consistent
  across the Docker and Daytona backends.
- The `task.md` init scaffold is agent-neutral, so `--agent oracle` works on a
  freshly scaffolded task.
- `gemini/`-prefixed judge/simulated-user models now resolve to the Google
  backend instead of passing the slashed name through and 404-ing.
- Model-backed judges raise a clear error naming the provider and pointing at
  `pip install benchflow[judge]` when the judge SDK is missing, instead of the
  misleading "Missing OPENAI_API_KEY" fall-through.
- `bench tasks check` recognizes a rubric-backed `llm-judge` verifier as a valid
  entrypoint and no longer demands a `test.sh`.
- Pre-verifier disk reclaim is workspace-aware and symlink-safe: it rejects
  symlinked cache candidates and realpath-guards every deletion against the
  workspace and `/logs`, so an agent-planted `~/.cache` symlink cannot steer the
  reclaim into workspace or output state (#601).
- Bedrock Claude 4.8+ routes fail closed when LiteLLM's adaptive-thinking patch
  is inactive, instead of silently sending a request the proxy cannot satisfy
  (#602).

### Changed

- Quickstart and CLI reference now match observed run behavior — the real jobs
  directory layout and artifact map, the `<PROVIDER>_API_KEY` /
  `<PROVIDER>_BASE_URL` convention, and exit-code semantics.
- Document the public vs internal preview install/upgrade command matrix,
  including `uv tool` exact pins, internal preview upgrades, and the
  `--force` path for replacing stale entrypoint scripts.

## 0.5.2 — 2026-06-05

### Changed

- **PyPI project README badge** — replace the dynamic PyPI version badge with
  a stable package badge so the rendered project description cannot show a
  stale external version image after a public release.
- **Release documentation refresh** — update public install snippets,
  release-channel docs, examples, and citation metadata to `0.5.2`.

## 0.5.1 — 2026-06-05

### Added

- **Daytona usage telemetry by default** — Daytona runs now start a sandbox-local provider usage proxy so token/cost telemetry works without an external tunnel; use `--usage-tracking off` to bypass proxying when needed.
- **Azure AI Foundry providers** — new `azure-foundry-openai/` and `azure-foundry-anthropic/` prefixes routing through Foundry's unified resource. Export `AZURE_API_KEY` plus `AZURE_API_ENDPOINT` (e.g. `https://<resource>.openai.azure.com/`); benchflow derives the resource name from the endpoint host, builds the per-surface base URL, and maps the key onto the agent-native auth env automatically. Missing/unrecognized endpoints and unsupported agent/provider protocol pairings fail fast with clear errors instead of falling through to the wrong endpoint.
- **Azure Foundry auth guidance** — agent discovery output and docs now call out that provider-prefixed models can use provider-specific credentials instead of the agent's native/default API key.

### Changed

- **PyPI project documentation refresh** — the public package README, install snippets, release-channel docs, examples, and citation metadata now point at `0.5.1`.

### Fixed

- Inherit `BENCHFLOW_PROVIDER_BASE_URL` / `BENCHFLOW_PROVIDER_API_KEY` from the host environment so self-hosted / OpenAI-compatible endpoints route correctly instead of falling back to `api.openai.com`; empty or whitespace-only host values are skipped so they cannot shadow the resolved provider URL (benchflow-ai/skillsbench#817).

## 0.5.0 — 2026-06-04

### Added

- **Public/internal preview release channels** — tag-driven public releases publish stable PyPI packages and GitHub Releases; merges to `main` publish internal preview `.devN` packages after CI passes.
- **v0.5 integration evidence** — release validation docs now cover urgent blocker closure, SkillsBench infra-fix validation, adapter evidence, trace-to-task evidence, hosted env compatibility, and diagnostic fields.
- **Release automation guardrails** — public release tags must point at commits contained in `main`, version tags must match `pyproject.toml`, and PyPI publishing uses Trusted Publishing/OIDC instead of stored tokens.

### Changed

- `main` now tracks the next public version as `0.5.1.dev0`; the published public SDK is `0.5.0`, and internal previews are emitted as `0.5.1.dev<N>`.
- Documentation now directs downstream users to depend on public PyPI releases by default and use prerelease-enabled internal previews only for validation before the next public cut.

### Fixed

- Closed the v0.5 release blocker set covering structured sandbox/verifier diagnostics, Daytona startup/export retries, verifier dependency classification, CTRF path consistency, and SkillsBench task compatibility evidence.

## 0.3.3 — 2026-05-15

### Added

- **Harvey LAB benchmark** — converter, agent shim, and parity validation for 1,251 legal AI tasks (#239).
- **Harvey LAB Claude Sonnet judge** — switched verifier from Gemini to `claude-sonnet-4-6`, matching the original benchmark default (#264).
- **ProgramBench integration** — new benchmark adapter; TB2 removed; `.ref/` migrated to `benchmarks/` (#237).
- **CLI progress output** — `bench eval create` / `bench run` now show progress messages by default (#264).
- **Skill nudge** — optional prompt injection for skill-enhanced agent runs (#207).
- **Self-generated skill mode** for Codex agent (#233).
- **Integration test suite** for ENG-6 + `OPENAI_BASE_URL` inheritance fix (#255).
- **Modal backend support** — Dockerfile compatibility for Modal environments.
- **CITATION.cff** (#246).
- **`AGENTS.md`** — canonical contributor guide; `CLAUDE.md` deprecated (#258).

### Changed

- **Two-field source pattern** for dataset sourcing (#252).
- **Docs overhaul** — synced from www.benchflow.ai; Mintlify config added then orphaned config removed (#259, #257, #226).
- **`uv sync`** for package management (#232).

### Fixed

- Prevent `TypeError` in `metrics.collect_metrics` when reward is `None` (#243).
- Copy eval `requirements.txt` into Docker build context (#245).
- Resolve agent aliases in `bench agent show` and display aliases in `bench agent list` (#251).
- Guard ACP transports against JSON scalar logs (#236).
- Agent timeout reward fallback for Codex (#234).
- Isolate JS agent runtime installs (#231).
- Route Codex ACP through responses API (#224).
- Deploy skills and forward `solution.env` for oracle runs (#223).
- Honor no-internet tasks for agent runs; disable web tools without prompt mutation (#215).
- Propagate `OPENAI_API_KEY` for vllm provider (#3).
- Preserve arrival order of thought/message within flush windows (#214).
- Record user messages and per-turn agent text in ACP trajectory (#745).
- Chown skill-link parent dirs so sandbox user can write into them.
- Dynamic `--rootdir` in `PYTEST_ADDOPTS` based on task workspace.
- Unique env-file path in `DaytonaPtyProcess` to avoid race conditions (#200).

## 0.2.3 — 2026-04-15

### Added

- `benchmarks/tb2_multiturn-claude-haiku45.yaml` — shipped config for the README's TB2 multi-turn Claude result.
- Daytona resource clamping via `BENCHFLOW_DAYTONA_MAX_CPUS` / `MAX_MEMORY_MB`.

### Changed

- Renamed `skillsbench-claude-glm5.yaml` → `skillsbench-claude-glm51.yaml` to match the model ID.
- `codex --login` correction in `docs/getting-started.md`.
- Restricted sdist build to `src/`, `tests/`, and metadata.

### Fixed

- Verifier sandbox hardening follow-ups across several base-image and tooling edge cases.
- Preserve trusted verifier path entries and workspace answer files.
- Redirect oracle output to container log.
- Align YAML path resolution to config file location.

## 0.2.2 — 2026-04-13

### Added

- **Sandbox hardening tiers 1–3** — layered defense (env scrubbing, path lockdown, workspace
  freeze, wider snapshot, oracle privilege drop) blocking F1–F6 red-team findings.
- **`labs/reward-hack-matrix`** — per-trial timeout support and 0.2.2 sweep handoff scripts.

### Fixed

- Multiple sandbox bypass vectors identified in red-team testing.

## 0.2.1 — 2026-04-12

### Added

- **Sandbox hardening on by default** — `sandbox_user` now defaults to `"agent"` (was `None`/root). Blocks conftest-hook and answer-lookup exploit patterns.
- **Path lockdown** — new `sandbox_locked_paths` parameter makes `/solution` and `/tests` read-only before the verifier runs, blocking `.pth`-injection and similar pre-verify tampering.
- **Verifier failure isolation** — agent errors and verifier errors are now stored separately; a crashing verifier no longer masks the agent result.
- **`labs/benchjack-sandbox-hardening`** — cookbook demonstrating three exploit patterns (P1 conftest-hook, P2 answer-lookup, P7 `.pth`-injection) and their defenses.

### Fixed

- **Oracle runs as `sandbox_user`** — oracle agent now respects path lockdown instead of running as root and bypassing it.
- **Multi-endpoint provider routing** — providers with multiple endpoints now route by the agent's native API protocol.
- **Stale API key shadowing subscription auth** — emits a warning when `ANTHROPIC_API_KEY` env var is present alongside `claude login` credentials.
- **pytest `ini`-injection bypass** — closed a verifier hardening edge case.

### Changed

- Version is now single-sourced via `importlib.metadata`; no more duplicate version string in `__init__.py`.
- **User-facing docs** — new `docs/` directory with getting-started guide, CLI reference, architecture overview, task-authoring guide, and labs index. README trimmed; detailed content moved to `docs/`.

## 0.2.0 — 2026-04-09

**First public release.** A near-complete rearchitecture from the 0.1.x era. API surface has changed — assume breaking changes. Future releases will maintain compatibility within the 0.2.x line. 0.1.x users should treat this as a fresh install; see `.dev-docs/sdk-reference.md` for the new SDK.

### Added

- **Multi-agent, multi-provider, multi-auth matrix** — one YAML config, any supported agent × model × provider × auth combination.
- **Subscription auth support** — use `claude login`, `codex --login`, `gemini` OAuth credentials directly. No API keys required for host-based agent workflows.
- **Vertex AI support** — ADC auth for `google-vertex/`, `anthropic-vertex/`, `vertex-zai/` prefixed models.
- **Provider registry** — add a new LLM endpoint via a dict entry in `providers.py`, no code changes.
- **`benchmarks/` directory** with reusable YAML configs and runner scripts for TB2 and SkillsBench.
- **Auto task download** — YAML configs reference datasets as `org/repo/path` (e.g. `harbor-framework/terminal-bench-2`). Repos are cloned on first use and cached under `.cache/datasets/`.
- **`benchflow tasks init`** — scaffold new tasks.
- **`benchflow tasks check`** — validate task structure.
- **`benchflow cleanup`** — delete old sandboxes with `--max-age` filtering (default 24h).
- **Oracle agent support** — run `solution/solve.sh` directly for task validation.
- **Hello-world-task example** for sanity-testing the agent pipeline.
- **Model generation params** via env vars (`BENCHFLOW_TEMPERATURE`, `BENCHFLOW_TOP_P`, `BENCHFLOW_MAX_TOKENS`).
- **OpenClaw ACP shim** with trajectory parsing and skills support.
- **ACP trajectory capture** — full multi-turn agent trajectories via ACP protocol.

### Changed

- **Skill loading** — agent-targeted with proper precedence; auto-distributed from `task.toml` `skills_dir`.
- **`openclaw-gemini` merged** into `openclaw` — provider mode selected at runtime via `BENCHFLOW_PROVIDER_NAME`.

### Fixed

- **API keys leaking in `ps aux`** — env vars now written inside the container instead of passed via Docker exec `-e`.
- **Subscription auth skipped without `-m`** — `benchflow run` without `--model` now checks correctly.
- **ADC credentials break with `sandbox_user`** (#111) — credentials written to sandbox user's home instead of `/root/`.
- **Daytona sandboxes not cleaned up** (#102) — auto-delete after max age.
- **`benchflow cleanup` ignoring `--max-age`** — was deleting everything regardless of age.
- **readline buffer overflow crashes trial** (#98).
- **OpenClaw ACP shim loses tool command text** (#96).
- **OpenClaw ACP shim hardcodes `anthropic/` prefix** (#95) — now routes correctly for Gemini/GLM models.
- **Oracle agent `PermissionError`** writing `agent/oracle.txt` (#91).
- **Oracle path skips `pre_agent_hooks`** (#92) — services now start before oracle runs.
- **Trial data parity with Harbor** (#90) — richer `result.json`, agent logs, per-phase timing.
- **`SDK.run()` `PermissionError`** — `jobs_dir` subdirectories created as root (#88).
- **Partial trajectory lost on timeout** — saved before timeout raises.
- **Redundant `--version` binary check** removed — was wasting 30s per trial.
- **Trajectory fallback** — scrapes agent-native files when ACP `session/update` is empty (#94).
- **`litellm` upgraded to 1.83.0** for CVE-2026-35030; transitive dep security alerts resolved (13 Dependabot alerts closed).

### Deprecated

- `BaseAgent` re-export — planned removal in 0.3.0
- `Trial` re-export — planned removal in 0.3.0
