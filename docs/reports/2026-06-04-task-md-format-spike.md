# task.md Format Spike

Date: 2026-06-04

## Summary

This spike makes `task.md` a first-class BenchFlow task entrypoint for simple
prompt tasks while preserving the legacy `task.toml` + `instruction.md` format.
It also parses the document-level shape needed for agent teams, multi-scene
tasks, and NudgeBench-style simulated users.

Follow-up parity work now treats the YAML frontmatter as the full
Harbor-compatible task config surface, not a reduced subset. Current Harbor
schema fields such as `network_mode`, `allowed_hosts`, `os`, `tpu`,
`healthcheck`, separate verifier environments, `artifacts`, `steps`, and
`multi_step_reward_strategy` parse through both `task.toml` and `task.md`.
Unknown config keys are rejected instead of silently dropped.

The format thesis held up: a task reads better as one executable document than
as one config file plus one prompt file. The migration debt is also real:
several subsystems still reach directly for `task.toml` or `instruction.md`
instead of going through a shared task-format boundary.

## Proposed Standard v0.2

`task.md` starts with YAML frontmatter. Existing and Harbor-compatible
`TaskConfig` keys stay intact: `schema_version`/`version`, `task`, `metadata`,
`agent`, `verifier`, `environment`, `oracle` (legacy alias: `solution`),
`source`, `artifacts`, `steps`, and `multi_step_reward_strategy`.

The config parser rejects unknown fields. Freeform task metadata still belongs
under `metadata`.

Document-only frontmatter keys:

- `agents.roles.<name>` declares role participants for agent teams.
- `scenes[]` declares named scene turns and role assignment.
- `user` declares simulated-user metadata for NudgeBench-style tasks.

Markdown body sections:

- `## prompt` is the canonical agent-facing task prompt.
- `## role:<name>` is role-specific prompt material.
- `## scene:<name>` is scene-specific prompt material.
- `## user-persona` is the simulated user's natural-language persona.

Prompt precedence for parsed scene turns is:

1. Inline turn prompt in frontmatter.
2. `## scene:<name>`.
3. `## role:<name>`.
4. `## prompt`.

## Experiments

Experiment 1: parser and scaffold

- Added `benchflow.task.document.TaskDocument`.
- Added `render_task_md_from_legacy()` for conversion spikes.
- Added `bench tasks init` support for `task.md` scaffolds.
- `Task` now loads `task.md` without requiring `task.toml` or `instruction.md`.
- `Evaluation._get_task_dirs()` discovers `task.md` task directories.
- Dogfood run: a temporary `task.md` package passed
  `bench eval create --agent oracle --sandbox docker` with reward 1.0.

Experiment 3: multi-scene encoding

- `task.md` parses `agents.roles` and `scenes`.
- Parsed scenes compile through existing `compile_scenes_to_steps()`.
- Example: `docs/examples/task-md/multi-scene/task.md`.
- `RolloutConfig.from_legacy()` and direct `RolloutConfig(task_path=...)` now
  adopt document-declared scenes when the caller has not supplied explicit
  prompt overrides and is not using self-gen.
- Focused tests prove document scenes become executable rollout scenes, manual
  prompts still override the document, and self-gen keeps its runtime
  orchestrator contract.
- Scene execution now reconnects at scene boundaries even when consecutive
  scenes use the same role, matching the documented "fresh agent session per
  scene" contract while still reusing a session for same-role turns inside one
  scene.
- Current limit: explicit-agent helper APIs such as
  `bf.run("agent", task_path=...)` and `Runtime(Agent, Environment)` still build
  their own one-agent scene. That is a real interface decision to settle before
  declaring `task.md` the only canonical entrypoint across all public surfaces.

Experiment 5: bake-off

- Legacy demo task: 24 lines across `task.toml` and `instruction.md`.
- Rendered `task.md` demo: 27 lines in one file.
- For tiny tasks, `task.md` is not shorter. Its advantage is locality: prompt,
  config, role semantics, and simulated-user context live in one artifact.
- Harbor's split format remains best as an inbound compatibility target.
- Tau-bench-like simulated-user specs map naturally to `user` frontmatter plus
  `## user-persona`, without absorbing tau-bench runtime semantics.

Experiment 6: Harbor task.toml parity audit

- Checked current Harbor task docs/source and found BenchFlow's model had fallen
  behind the newer `schema_version = "1.3"` shape.
- Added schema support for network policy, task OS, TPU specs, healthchecks,
  workdir, separate verifier environments, root and step artifacts, steps, and
  multi-step reward strategy.
- Added `docs/examples/task-md/harbor-parity/task.md` as a schema-only
  workspace fixture.
- Added tests that parse the same current Harbor-shaped surface through
  `task.toml` and through `task.md` frontmatter.
- Unknown config fields now fail validation instead of disappearing under
  Pydantic's default extra-field behavior.

Experiment 7: migration ergonomics

- Made `bench tasks init` default to `task.md`; legacy split scaffolding remains
  available with `--format legacy`.
- Made native `task.md` scaffolds use `verifier/` and `oracle/` directories;
  legacy `tests/` and `solution/` remain supported aliases.
- Added `bench tasks migrate <task-dir>` to render a legacy
  `task.toml + instruction.md` pair into `task.md`.
- Migration verifies config and prompt round-trip before writing the document.
- Migration is non-destructive by default; `--remove-legacy` deletes the split
  pair only after the generated `task.md` is verified.
- Added utility and CLI tests so authors have a checked adoption path.

## Tech Debt Surfaced

- `src/benchflow/rollout.py` now has a bridge that materializes `task.md` into
  `/instruction.md`. This is intentionally transitional; a task materializer
  should own all runtime views of a task package.
- `src/benchflow/rollout.py` also owns document-scene adoption today. This
  gives the spike an executable path, but the deeper module should be
  `TaskPackage`/`TaskFormat`: one place that answers "what task entrypoint is
  this?", "what prompt should the sandbox see?", and "what scenes should the
  rollout execute?"
- `src/benchflow/runtime.py` has explicit-agent helpers that construct
  `Scene.single(...)` before task-format discovery can happen. Decide whether
  those helpers are agent overrides or convenience defaults; right now they are
  a separate public seam from evaluation.
- `src/benchflow/evaluation.py` has a local `_is_task_dir()` helper. This should
  become a shared task-format detector or registry rather than another filename
  check.
- `src/benchflow/traces/task_gen.py` and `src/benchflow/skill_eval/_core.py`
  still generate only legacy split tasks.
- Some legacy examples and generated trace tasks still use `tests/` and
  `solution/`; native task authoring now prefers `verifier/` and `oracle/`.
- `src/benchflow/adapters/inbound.py` still treats `task.toml` as the native
  format discriminator for inbound adapters.
- Several use-case docs still teach `task.toml` + `instruction.md` as the
  default layout.
- Runtime parity still lags schema parity for newer Harbor features:
  `network_mode = "allowlist"`, Windows `environment.os`, TPU scheduling,
  environment `healthcheck`, `workdir`, separate verifier environments,
  artifact collection declarations, and `steps` are parsed but not yet fully
  executed through BenchFlow's rollout runtime.

## Workflow Pressure Test

Two read-only sidecar agents audited the spike from different angles.

Findings fixed in this pass:

- Mixed-format directories now consistently prefer `task.md` for verifier
  hardening, matching `Task` loading precedence.
- Same-role scene boundaries now force a reconnect, preventing ACP context
  bleed across `scene` sections.
- Document-declared scenes now reach `RolloutConfig` for evaluation/SDK-style
  runs instead of staying parser-only.
- `task.md` frontmatter now covers the current Harbor config schema surface
  instead of only BenchFlow's older subset.
- Unknown task config keys now fail validation in both `task.toml` and
  `task.md`, making schema drift visible during task checks/tests.
- Legacy tasks now have a first-class `bench tasks migrate` path into the
  `task.md` standard.

Findings intentionally left as tech debt:

- Add a `TaskRuntimeView`/`TaskPackage` boundary so rollout, hardening,
  provenance, source discovery, self-gen, and user loops stop re-checking
  filenames independently.
- Decide role/team prompt composition. Today scene prompts shadow role prompts;
  this is simple, but it can drop role guardrails such as "review only".
- Bind simulated-user metadata to a concrete runtime plan. `user.model`,
  `stop_rule`, tools, and `## user-persona` parse today, but they do not yet
  compile into a `BaseUser` loop or scene participant.
- Split native task-format detection from foreign adapter detection. Harbor and
  Terminal-Bench compatibility should stay adapter-owned; `task.md` should be
  native.
- Continue retiring legacy generation at adapter boundaries; trace and
  skill-eval generators now default to `task.md`, but some downstream docs and
  compatibility tests still mention legacy output.

## Real Task Dogfood

The initial spike used tiny fixtures. A follow-up pass converted real
SkillsBench tasks into `task.md`-only packages under
`docs/examples/task-md/real-skillsbench/` by rendering
`task.toml + instruction.md` into frontmatter plus `## prompt`, then deleting
the legacy pair. The real task assets (`environment/`, `tests/`, `solution/`,
bundled skills) were unchanged.

Task checks:

- `citation-check` — `bench tasks check` passed.
- `weighted-gdp-calc` — `bench tasks check` passed.
- `3d-scan-calc` — `bench tasks check` passed.

Oracle / Docker:

- `weighted-gdp-calc`, no-skill: passed with reward 1.0 in
  `jobs/real-taskmd/oracle-docker/.../weighted-gdp-calc__a4dd4f18`.
- `3d-scan-calc`, no-skill: failed because the oracle imports the bundled
  `mesh-analysis` skill and no-skill intentionally withholds it.
- `3d-scan-calc`, with-skill: passed with reward 1.0 in
  `jobs/real-taskmd/oracle-docker-3d-with-skill/.../3d-scan-calc__*`.
- `citation-check`, no-skill: verifier ran and failed with reward 0.0 because
  oracle identified four fake citations while the verifier expects exactly
  three. This is task/oracle behavior, not a task-format failure; the runtime
  reached solution and verifier cleanly.

Gemini / Daytona:

- Verified the exact Gemini key with a raw provider `generateContent` call
  against `gemini-2.5-flash` before running.
- Verified Daytona auth with the SDK list path before creating a sandbox.
- Installed the locked `sandbox-daytona` extra into the `uv` environment; the
  first run failed before sandbox creation because the local `uv` env lacked
  the optional Daytona SDK even though a global Python had it.
- `weighted-gdp-calc`, `task.md`-only, Gemini `gemini-2.5-flash`, Daytona,
  no-skill: completed the cloud rollout with no infra error, 12 ACP tool calls,
  `trajectory_source = "acp"`, and verifier reward 0.0 in
  `jobs/real-taskmd/gemini-daytona-weighted/.../weighted-gdp-calc__fb4d278d`.
  The failure was agent/task quality: the verifier saw many expected spreadsheet
  values left as zero or missing.

Real-task conclusion: the `task.md` path now survives real BenchFlow task
checking, Docker oracle runs, bundled-skill oracle runs, and a live Daytona ACP
agent run. The remaining failures surfaced task/oracle/model issues, not
format-loader or prompt-materialization crashes.

Directory vocabulary update: new native task packages should use `verifier/`
for reward code and `oracle/` for reference solutions. The real SkillsBench
examples above intentionally keep legacy `tests/` and `solution/` directories
to prove compatibility with existing task assets.

Implementation follow-up: BenchFlow now treats `verifier/` and `oracle/` as the
native on-disk package names. Existing `tests/` and `solution/` directories are
still supported as compatibility aliases:

- `TaskPaths` resolves native directories first and falls back to legacy names.
- Native `task.md` scaffolds create `verifier/test.sh`,
  `verifier/test_outputs.py`, and `oracle/solve.sh`.
- `bench tasks init --no-oracle` is the primary skip flag; `--no-solution`
  remains a CLI alias.
- `TaskConfig` accepts either `oracle` or legacy `solution` config blocks and
  serializes `task.md`/TOML output as `oracle`.
- Runtime uploads native oracle code to `/oracle` and native verifier code to
  `/verifier`; legacy tasks continue using `/solution` and `/tests`.
- Sandbox lockdown protects both native and legacy private task directories.
- `bench tasks migrate` rewrites legacy `[solution]` config into native
  `[oracle]` frontmatter.

## Dogfood Follow-Up Fixes

- Demoted misleading hardening warnings from minimal images without `python3`.
- Made verifier hardening opt-outs read from `task.md` frontmatter.
- Made `task.md` win verifier hardening in mixed directories that still carry a
  stale legacy `task.toml`.
- Made Memory-space `expected_skills` fixtures read through `Task`, so `task.md`
  and legacy tasks share the same config path.
- Made source provenance include file hashes when the fetched source path is a
  single `task.md` task.
- Removed the bogus `Unknown agent 'oracle'` warning for oracle evals.
- Added a Harbor parity example under
  `docs/examples/task-md/harbor-parity/task.md`.
- Added `bench tasks migrate` and updated task authoring / CLI docs so
  migration is a documented standard workflow.
- Added regression coverage for native-plus-legacy oracle/verifier directory
  resolution, migration output, runtime oracle mounts, verifier mounts, and
  sandbox lockdown.
- Made trace-generated and skill-eval-generated tasks default to native
  `task.md` packages with `verifier/` and `oracle/`, while preserving explicit
  `output_format="legacy"` / `--task-format legacy` compatibility.

Validation after the vocabulary follow-up:

- `uv run python -m pytest tests/ -q` — 2690 passed, 12 skipped, 1 deselected.
- `uv run ruff check .` — passed.
- `uv run ty check src/` — passed.
- `git diff --check` — passed.
- Real task checks passed for `weighted-gdp-calc`, `3d-scan-calc`, and
  `citation-check` under `docs/examples/task-md/real-skillsbench/`.

Generator parity follow-up:

- Trace generation now defaults to native `task.md` packages with `verifier/`
  and `oracle/`; `output_format="legacy"` and `--task-format legacy` preserve
  the old split layout.
- Skill-eval task generation now defaults to native `task.md` packages with
  `verifier/`; `output_format="legacy"` preserves the old split layout.
- Skill-eval verifier templates detect `/verifier` first and fall back to
  `/tests`, so the same generated verifier works in native and legacy packages.
- Added real generated examples from `benchmarks/models-as-skills` under
  `docs/examples/task-md/generated-skill-eval/models-as-skills/`.
- `bench tasks check` passed for the generated `topo-sort-with-cycle-detection`,
  `regex-email-parser`, and `optimize-quadratic-to-nlogn` tasks.
- Focused generator validation: `uv run python -m pytest
  tests/test_traces_task_gen.py tests/test_trace_task_gen_traversal.py
  tests/integration/check_trace_to_task_evidence.py tests/test_skill_eval.py
  tests/test_skill_eval_dryrun.py tests/test_skill_eval_integration.py
  tests/test_skill_eval_traversal.py tests/test_skill_eval_sweep.py -q` —
  148 passed.

## Recommendation

Keep `task.md` behind explicit authoring and parser support for now:

1. Use it for new BenchFlow-native benchmark design work and generated tasks.
2. Keep legacy split tasks running unchanged.
3. Add a `TaskPackage` or `TaskFormat` boundary before broadening document-scene
   runtime semantics, so direct `task.toml` reads disappear instead of
   spreading.
4. Treat NudgeBench runtime support as separate from the file format. The file
   can declare the simulated user now; the user loop and stop-rule runtime need
   their own design.
