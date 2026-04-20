---
name: docs-review
description: Review benchflow documentation for drift, staleness, duplication, and alignment
user-invocable: true
---

# Docs Review Skill

Review the repo's documentation against the current codebase and surface
a punch list. **Do not auto-fix.** Report findings; let the user approve
edits.

## Commands

The user may say `/docs-review` with an optional argument:

- `/docs-review` — full review across all in-scope docs
- `/docs-review <path>` — single doc (e.g. `/docs-review README.md`)
- `/docs-review --drift` — fast subset: drift-vs-code + stale refs only

## In-scope docs

### Full review (all seven checks)

- `docs/architecture.md`
- `docs/cli-reference.md`
- `docs/task-authoring.md`
- `docs/getting-started.md`
- `docs/labs.md`
- `README.md`
- `CLAUDE.md`

### Light-touch (checks 1, 2, 6 only — drift, stale refs, link integrity)

- `.dev-docs/sdk-reference.md` — internal SDK surface; verify class/function
  names + signatures still resolve in `src/benchflow/`.
- `.dev-docs/harden-sandbox.md` — sandbox hardening notes; verify referenced
  files / knobs / env vars still exist.
- `.dev-docs/tested-agents.md` — matrix of agent × model × provider; verify
  names still appear in `agents/registry.py` and `agents/providers.py`.

### Skipped entirely

- `.dev-docs/sdk-refactor-notes.md` — dated refactor record (April 2026);
  historical, status language is expected. Do not flag or edit.
- Anything matching `*-notes.md`, `*-archive.md`.
- `.smoke-jobs/`, `trajectories/`, `examples/`, `fixtures/` — generated or
  sample output, not documentation.

## Checks

### 1. Drift vs. code

Project-structure trees, module one-liners, env var names, registry
entries. Cross-check:

- `ls src/benchflow/`, `ls src/benchflow/agents/`, `ls src/benchflow/acp/`,
  `ls src/benchflow/cli/` against trees in `architecture.md` and `Key
  modules` blocks in README/CLAUDE.
- Module descriptions — does `sdk.py` still own what the doc claims?
  Does `job.py` still drive the run loop? Spot-check first ~40 lines of
  each named module.
- **Registry drift** (high-churn surface). For each agent/provider name
  mentioned in docs, grep `src/benchflow/agents/registry.py` and
  `src/benchflow/agents/providers.py`. A name in docs but not in the
  registry dict → stale; a name in the registry but not documented where
  expected (`docs/architecture.md` matrix, `.dev-docs/tested-agents.md`)
  → gap.
- Env vars mentioned in docs (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `GROQ_API_KEY`, `BENCHFLOW_*`, etc.) — still referenced in
  `src/benchflow/`?
- `pyproject.toml` — python version pin, dep names, extras. Verify
  `Setup` / `Install` blocks in README and `docs/getting-started.md`.

### 2. Stale references

Grep each doc for file paths, function names, class names, CLI commands,
task/agent IDs. For each hit, verify it resolves in the current tree:

- File paths → `ls` (watch for renames — e.g. a file split into a
  package, a private module prefix added like `_acp_run.py`).
- Function / class / decorator names (`register_agent`, `SDK`,
  `RunResult`, `detect_services_from_dockerfile`, …) → `Grep` in
  `src/benchflow/` and the `__init__.py` re-exports.
- CLI commands (`benchflow run`, `benchflow ls`, `benchflow view`, …) →
  check the Typer app in `src/benchflow/cli/`.
- Task IDs referenced in examples → check `examples/` and
  `fixtures/`.

### 3. Status-language rot

Grep for implementation-tracking words:
`CURRENT`, `NEXT`, `shipped`, `Phase \d`, `proposed`, `planned`, `not started`, `TODO`, `FIXME`, `WIP`.

For each hit, ask: is this describing the *design* (stays true) or
*in-flight work* (rots)? In-flight language belongs in commit messages,
PR descriptions, or `.dev-docs/*-notes.md`, not user-facing reference
docs.

**Suppress for `.dev-docs/*-notes.md`** — dated refactor notes legitimately
carry status language.

### 4. Duplication

Any fact stated in ≥2 docs that could be a link instead? Big offenders
for benchflow:

- **Project-structure trees** (belongs only in `architecture.md`)
- **SDK Run Phases** (SETUP → START → AGENT → VERIFY) — should live in
  `architecture.md`; others should link.
- **Registry examples** — one copy in `architecture.md` + one in
  `task-authoring.md` or `.dev-docs/sdk-reference.md` is OK if they
  illustrate distinct use cases; two near-identical `register_agent(...)`
  blocks is not.
- **Agent × Model × Provider matrix** — live in
  `.dev-docs/tested-agents.md`; `architecture.md` should link, not
  duplicate.
- **Env var reference** — should live in `docs/getting-started.md` or
  `docs/cli-reference.md`; not re-listed in README.

Target state: `architecture.md` is the sole deep reference for internals,
`cli-reference.md` for commands, `task-authoring.md` for task YAML /
verifier shape. README and CLAUDE.md link to them instead of duplicating.

### 5. Cross-doc alignment

- `docs/cli-reference.md` flag list ↔ actual Typer definitions in
  `src/benchflow/cli/`. Every documented flag resolves; every command in
  the CLI has a documented entry (or an intentional hide).
- `docs/task-authoring.md` YAML schema ↔ `TaskConfig` / loader in
  `src/benchflow/tasks.py`. Every field has a loader path.
- `docs/architecture.md` "Error Taxonomy" / "Trajectory event format"
  sections ↔ the actual dataclass fields in `src/benchflow/models.py`
  and emit sites in `job.py` / `_trajectory.py`.
- `docs/architecture.md` ACP Protocol section ↔ `src/benchflow/acp/` and
  `_acp_run.py`.
- Cross-references between docs — does each `[text](other-doc.md)` link
  still point at a section that exists?

### 6. Link integrity

All markdown links resolve:

- `[text](path)` → file exists
- `[text](path#Lnum)` → line exists (file has ≥ N lines)
- `[text](#heading)` → heading exists in the same doc
- `[text](../foo.md)` → relative path resolves
- Inline backticked paths (`` `src/benchflow/sdk.py` ``) still exist —
  not strictly links, but the same drift vector.

### 7. Doc-role violations

- **README.md**: outward-facing only. Install / quickstart / one-screen
  architecture pointer. No full internals tree (use `Key modules` + link
  to `docs/architecture.md`). No deep rationale.
- **CLAUDE.md**: AI entry-point. Stays compact (~50 lines max — it's
  always loaded into Claude's context). Links to design docs rather than
  inlining them. Encodes conventions, not reference material.
- **docs/architecture.md**: sole deep reference for internals. All module
  descriptions, full project tree, SDK phases, registry pattern, error
  taxonomy live here.
- **docs/cli-reference.md**: flag-level reference. Not narrative.
- **docs/task-authoring.md**: task YAML + verifier contract. Not a
  "how benchflow works" overview — link to architecture for that.
- **docs/getting-started.md / docs/labs.md**: tutorial tone; design
  rationale belongs elsewhere.
- **.dev-docs/**: internal — can carry status language, refactor
  histories, signature tables.

## Execution

For a full review:

1. **Dispatch in parallel.** Spawn one `Explore` agent per full-review
   doc. Each agent gets: the doc path, the seven checks, "report ≤ 250
   words, concrete file:line references only, no prose rewrites."
   Light-touch docs get a trimmed prompt (checks 1, 2, 6 only). Skip
   entirely the docs under "Skipped entirely."

2. **Synthesize.** Merge agent findings into a single punch list. Group
   by severity:

   - **Blocker** — broken link, flat-out wrong fact, dead file reference,
     documented CLI flag that doesn't exist, agent/provider name that
     doesn't resolve in the registry. Reader will be misled.
   - **Stale** — outdated but currently harmless (old phase names,
     settled open questions, retired status labels, superseded rationale
     that's still correct-shaped).
   - **Polish** — duplication, doc-role creep, language smells.

   Each item: `<severity> · <doc>:<line> — <one-line description>`.

3. **Ask for approval.** Present the punch list. Do NOT start editing.
   Wait for "fix 1-4", "ignore 5, it's intentional", "all", or similar.

4. **Apply fixes.** Edit only what was approved. After edits, re-verify
   the specific items you touched (don't re-run the full review).

## Anti-patterns

- **Don't auto-fix.** Surface findings; let the user decide.
- **Don't false-positive on quoted history.** A doc can mention a retired
  module inside a "previously this was structured as …" sentence without
  being stale. Verify each hit reads as a *current* reference, not a
  historical one, before flagging.
- **Don't rewrite for style.** Scope is factual drift and structure, not
  prose quality or tone.
- **Don't grow scope.** If a check isn't in the seven above, don't add it
  mid-review. File a suggestion in the punch list instead.
- **Don't touch archives or refactor notes.** `.dev-docs/*-notes.md`
  legitimately carry status language and reflect state at the time they
  were written; don't normalize them.
- **Don't flag registry drift without reading the registry dict.**
  `registry.py` and `providers.py` are the source of truth — a name
  missing from docs is a doc bug; a name missing from the registry is a
  code bug (surface separately, don't silently "fix" docs).

## Example output

```
Docs review punch list (3 blocker, 4 stale, 2 polish)

Blockers:
- docs/architecture.md:114 — references AgentConfig at src/benchflow/agents/registry.py, but file defines AgentSpec (renamed)
- docs/cli-reference.md:142 — documents `benchflow verify --strict` flag; flag doesn't exist in cli/verify.py
- README.md:62 — example uses register_agent(..., model=...) kwarg; signature takes models=[...] (list)

Stale:
- docs/architecture.md:39 — "Phase 1: SETUP (host)" numbering implies sequential work-in-progress; phases are always-on
- docs/task-authoring.md:88 — "TODO: document verifier timeout knob"
- docs/getting-started.md:121 — example task ID "demo-fizzbuzz" renamed to "examples-fizzbuzz"
- .dev-docs/tested-agents.md:14 — lists claude-code-acp; registry only has claude-agent-acp

Polish:
- README.md:98-134 — full src/ tree duplicates docs/architecture.md:12-38
- CLAUDE.md:14-22 — Setup block duplicates docs/getting-started.md; consider linking

Reply with which to fix, or "all".
```
