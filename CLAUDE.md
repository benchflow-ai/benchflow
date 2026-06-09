# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Conventions, setup, release mechanics, and batch-experiment guidance live in
[`AGENTS.md`](./AGENTS.md) — read it. This file does not repeat it; it focuses on
the thing this fork exists to build: the **`task.md` task package standard**.

## What this repo is

This is `benchflow-ai/benchflow-task-standard-private`, a full fork of BenchFlow
whose purpose is to design and land the **`task.md` standard** — a single native
authoring format meant to be a *superset* that absorbs and runs other eval
ecosystems (Harbor + its ~1.1k forks, AgentBeats/AAA, ORS, Verifiers, Kaggle,
SWE-bench, Terminal-Bench, METR, Inspect). It is **not** `benchflow-ai/benchflow`
(public main has no `task.md` parser; the whole `TaskDocument`/normalize/standard
stack lives only here). Default branch is `codex/lightweight-task-authoring`; the
standard folds into `docs/task-standard.md` rather than competing proposal files.

The standard's north star, decisions log, milestones, and "how to resume without
re-discovering everything" runbook are in
[`docs/reports/2026-06-08-task-standard-handoff-runbook.md`](./docs/reports/2026-06-08-task-standard-handoff-runbook.md).
**Read that report and `docs/task-standard.md` before doing standard work.**

## Commands

Setup/test/lint are in AGENTS.md. Beyond those, the standard-specific surface is
the `bench tasks` CLI (defined in `src/benchflow/cli/main.py`, ~line 613):

```bash
uv run bench tasks init <name> --format task-md      # scaffold native task.md package
uv run bench tasks migrate <dir> --remove-legacy     # legacy split (task.toml+instruction.md+solution/+tests/) -> task.md + oracle/ + verifier/
uv run bench tasks normalize <dir> --write           # expand minimal preset authoring into canonical task.md
uv run bench tasks export <dir> <out> --target harbor # export native -> Harbor/Pier split layout + compatibility/export-report.json (loss report)
uv run bench tasks check <dir> --level <level>       # validate (see levels below)
```

`bench tasks check` validation levels form a ladder of strictness:
`schema` (authoring/prompt parse only — for schema-only fixtures) →
`structural` (default) → `runtime-capability --sandbox <docker|daytona|modal>`
(fail-closed gate for parsed-but-unrunnable features) → `publication-grade`
(native `task.md`/`oracle/`/`verifier/verifier.md`/rubrics/reward contract) →
`acceptance` (static evidence gate) → `acceptance-live --sandbox <backend>`
(executes oracle/verifier reruns through a real sandbox).

Run a single test (pytest config lives in `pyproject.toml`; `live`/`integration`
markers are excluded by default):

```bash
uv run python -m pytest tests/test_task_document.py::test_task_document_minimal_profile_authoring_parses
uv run python -m pytest tests/test_task_document.py tests/test_task_package.py tests/test_verifier_document.py tests/test_task_export.py tests/test_runtime_capabilities.py
```

The standard's own tests are `tests/test_task_*.py`, `test_verifier_*.py`,
`test_runtime_capabilities.py`, `test_scene*.py`, `test_user.py`, `test_oracle*.py`,
and `test_skillsbench_harbor_parity.py`.

## Architecture of the standard

**Three views — never collapse them** (`docs/task-standard.md`):

| View | What it is | Owner code |
|---|---|---|
| Authoring document | what humans/generators write: one `task.md` + sidecar dirs | `TaskDocument` (`task/document.py`) |
| Runtime task view | what rollout/verifier/hardening/provenance/trajectories consume | `TaskRuntimeView`/`TaskPackage` (`task/runtime_view.py`, `task/package.py`) |
| Foreign adapter view | what Harbor/Pier/TB/SWE/Inspect import/export | adapters + `task/export.py`, `task/imports.py` |

Authoring may carry more than a foreign format can export. **Direction is
asymmetric: absorb maximally, export honestly** (export emits an explicit loss
report; we do not chase bidirectional-lossless parity with any one source).

**Native layout** (`task.md` + `environment/` + `verifier/` + `oracle/` +
`evidence/`). `oracle/` and `verifier/` are the native names; `solution/` and
`tests/` are compatibility aliases. Selection is **fail-closed**: if `task.md`/
`verifier/`/`oracle/` exist they are authoritative and do not fall back to the
alias; duplicate alias trees must be byte-identical after normalized traversal.

**Six planes, three of them authoring axes.** A task is one mode-selection per
axis: `environment × interaction-mode × verifier-strategy`. Supporting planes are
Package, Oracle, Evidence. New normative fields must add/compose a mode on an axis
or cite a fork user story (F1–F8) — otherwise they belong in `metadata` or an
adapter, not the standard.

**Key source files** (all under `src/benchflow/task/`):
- `document.py` — `TaskDocument`: parses frontmatter (Harbor `TaskConfig` keys +
  `agents`/`scenes`/`user` orchestration + reserved `benchflow:` namespace) and
  markdown prompt sections (`## prompt`, `## role:<n>`, `## scene:<n>`,
  `## user-persona`); `normalize_task_document_frontmatter`, `render_task_md*`.
- `runtime_view.py` / `package.py` — the runtime-facing boundary: which entrypoint
  is authoritative, native vs legacy verifier/oracle dirs, source hashes, compat
  metadata, compiled prompt plan, blocking runtime issues.
- `runtime_capabilities.py` — `validate_task_runtime_support(task, *, sandbox,
  task_dir)`: pure validator that reports stable config paths + reasons for
  parsed-but-unrunnable features (`steps`, artifacts, allowlists, separate
  verifier envs, Windows/TPU, healthchecks, unsafe workdirs, doc-only
  `user`/`benchflow` semantics). Wired into `bench tasks check --sandbox` and the
  shared sandbox factory; raises `UnsupportedTaskFeatureError` **before** Docker/
  Daytona/Modal construction. **A parsed field the runtime can't honor is worse
  than a parse error** — fail closed.
- `verifier_document.py` — parses `verifier/verifier.md` (the `task.md` of the
  evaluation side): strategies (`script`, `llm-judge`, `reward-kit`,
  `agent-judge`, `ors-episode`), rubric metadata, output contracts,
  verifier-scoped role prompts. Verifier is an isolated peer package: judge
  models/credentials/prompts are verifier-scoped, never agent-visible.
- `verifier.py` — `Verifier.verify()` selects/executes the strategy and resolves
  reward precedence (`reward.json` preferred over `reward.txt`; both-present
  scalar disagreement fails closed; preserves structured reward fields and
  `reward-details.json`).
- `export.py` — `export_task_to_split_layout()`,
  `build_harbor_roundtrip_conformance_report()` (the deterministic parity proof:
  canonical `TaskConfig` + normalized prompt + env/solution/tests file-map hashes).
- `acceptance_live.py` — the executable evidence gate.
- Authoring helpers (`init/migrate/normalize/check`) live in
  `src/benchflow/_utils/task_authoring.py`.

## Current status (M0, mostly landed) and how to continue

Milestones (`docs/task-standard.md` → "Absorption Frontier"):
- **M0 (today):** native authoring, `script`+`llm-judge` verifier, per-phase
  policy, compat import/export, sequential shared-workspace team handoff. The
  Runtime Capability Matrix in `task-standard.md` is the live parse-vs-runtime
  ledger — update it when a feature moves from `partial` to `yes`.
- **M1:** trajectory/episode reward (G1, G6), declarative simulated-user (G3),
  GAIN aggregation (G2), leaderboard-submission (G5).
- **M2 (heaviest):** `arena-concurrent` runtime (G4) — A2A bridge + concurrent
  assessor. The schema already represents it, so M2 is additive, not a rewrite.

**Validation state (honest):**
- *Done (deterministic, no model):* SkillsBench conversion parity 88/88 via
  `build_harbor_roundtrip_conformance_report`; oracle E2E parity sample
  (`--agent oracle --sandbox docker`) native reward == legacy reward. Evidence in
  `docs/reports/2026-06-08-skillsbench-taskmd-parity.md` and
  `experiments/skillsbench-taskmd-parity/`.
- *In progress:* agent-run parity — all 88 × {legacy, task.md} with a real agent
  on Daytona, comparing per-task mean reward. Use `openhands` (clean litellm
  wiring), not `opencode`. See the runbook §4–5 and §8 for the exact next steps.

**Landmines that have cost real hours** (from the runbook §5):
- `bench eval create` **resumes** an existing job dir when task names match and
  silently reuses stale results — always pass a fresh `--jobs-dir` per batch.
- A run that "passed" may be resumed *oracle* results. Trust a real rollout only
  when `n_tool_calls > 0`, `total_tokens > 0`, and reward is non-None.
- A revoked/wrong model key surfaces as an opaque agent error, not an auth error
  (see AGENTS.md). `agent=oracle` runs `oracle/solve.sh` with no model — best for
  deterministic parity.

When extending the standard: prefer putting draft/BenchFlow-specific ideas under
`benchflow:` until they have a stable typed interface, gate anything the runtime
can't execute as fail-closed in `runtime_capabilities.py`, add a regression test
that names the PR/commit it guards (AGENTS.md), and record decisions in the
handoff runbook's decisions log.
