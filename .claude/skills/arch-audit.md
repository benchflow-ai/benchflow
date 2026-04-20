---
name: arch-audit
description: Periodic audit of src/benchflow/ shape (names, boundaries, file sizes, stability mix) — propose structural reorg opportunities, don't execute
user_invocable: true
---

# Architecture Audit Skill

Find structural reorg opportunities: vague filenames, orphan tiny files, stability-mixed merges, oversize files. One layer above `/code-cleanup` — that skill edits inside files; this one reshapes *which files exist*.

**Do not auto-apply.** Report ranked candidates; the user approves. Reorgs have import-graph consequences that need human judgment.

## Commands

- `/arch-audit` — full `src/benchflow/` sweep
- `/arch-audit <subtree>` — scoped (e.g. `src/benchflow/agents/`, `src/benchflow/cli/`)

## Scope

In: file-level structure — names, boundaries, merges, splits, package grouping, `__init__.py` re-exports, import depth.
Out: line-level edits (`/code-cleanup`), public API design, dependency upgrades, test strategy, registry-entry additions (those are one-line contributions, not structural).

## Helpers

Reusable scripts in [.claude/skills/arch-audit/](.claude/skills/arch-audit/). They encode edge cases (private-underscore convention, `__init__.py` barrels, test exclusion, `__all__`, re-exports, string-typed imports). Python 3.12+, stdlib `ast` only.

**Primary:**
- `manifest.py [--no-cache]` — one `ast` + `git log` pass over non-test `src/benchflow/**/*.py`. Emits JSON with `{file, loc, commits_6mo, churn_ratio, exports[], importers[], barrel_importers[], importer_selectors{}}`. Cached under `.cache/<sha>[-dirty-<hash>].json`. **Source for Rules 1, 2, 5** — do not per-file-loop when manifest has the answer.

**Per-file (for what manifest doesn't cover):**
- `dominant_symbol.py <file>` — Rule 1 ≥70% exemption. Emits VERDICT: `dominant=X (N%)` / `NONE` / `INSUFFICIENT_IMPORTERS` (<3) / `NO_IMPORTERS`.
- `churn_ratio.sh <file>...` — TSV `ratio<tab>commits<tab>loc<tab>path`. Uses `--follow`; manifest's `churn_ratio` doesn't. **Required for Rule 3.**
- `co_edit_matrix.py <file>` — Rule 4 symbol co-edit clustering via `git log --numstat` + AST line-range mapping. Emits TSV pairs ≥20% or `# status:` (`file_renamed` / `insufficient_commits` / `too_many_symbols` / `no_clusters`).
- `split_costs.py <file> --half-a=names --half-b=names` — emits `{shared_types, shared_helpers, importer_payoff, importers}` for a proposed 2-way split. **Required for every split finding.**

**Spot-check:** `tiny_files.sh`, `importers.py`, `exports.py` — redundant when manifest is fresh; use for ad-hoc or post-rename sanity.

Parallelize per-file helpers with `xargs -P 8` (read-only, safe).

## The five rules

Each has command + threshold + decision. Thresholds are heuristics — note them in findings.

Scan non-test sources: `SRC=$(find src/benchflow -name '*.py' -not -name 'conftest.py' -not -path '*/tests/*')`.

### Rule 1 — One job per module

Source: manifest `exports[]`. A module is vague if **both**:
1. ≥3 top-level value exports (`FunctionDef`/`AsyncFunctionDef`/`ClassDef`/module-level assignments whose names don't start with `_` and aren't in a `__all__` carve-out; drop `TypeAlias`/`Protocol`/`TypedDict`/`NamedTuple` unless they have runtime behavior), **and**
2. exports cluster into ≥2 disjoint name-stem groups (no shared prefix/suffix ≥4 chars after snake_case split).

Before flagging, run `dominant_symbol.py`. Only `dominant=X (≥70%)` exempts. Every other VERDICT (`NONE`, `INSUFFICIENT_IMPORTERS`, `NO_IMPORTERS`) means **no exemption applies — emit the finding**. `INSUFFICIENT_IMPORTERS` is not a "skip this file" signal; it means the exemption check is unreliable, so fall back to the mechanical step-1+2 trip and emit.

Rule 1 is **independent of Rule 4** — a Rule 4 skip doesn't exempt Rule 1.

No "cohesive around X" prose overrides — mechanical check decides.

**Private modules (leading underscore like `_sandbox.py`, `_env_setup.py`) are NOT exempt.** Underscore means "package-private API surface"; it does not mean "cohesive by convention." Rule 1 still fires.

### Rule 2 — Tiny modules need a job

Source: manifest `loc < 100`. Must satisfy one:
- (a) `importers.length >= 2`, OR
- (b) exports are only type aliases / Protocols / TypedDicts / Enums / literal-RHS consts (no runtime functions/classes), OR
- (c) isolates an external dep for mocking (rare — e.g. a thin wrapper around `subprocess.run` specifically so tests can `patch("benchflow.X.run")`; inspect source), OR
- (d) `barrel_importers.length > 0` OR file itself is an `__init__.py` barrel (only re-exports).

Otherwise inline at sole caller. **1 importer does NOT satisfy (a)** — that's the fold signal.

**Skip `__init__.py` with `__all__` and/or re-exports only** — those are barrels, exempt by (d).

### Rule 3 — Don't merge across stability levels

Before any merge, `churn_ratio.sh <a> <b>`. If ratio >**3×** → veto (mixed stability = blame noise, review friction). Label sides "hot" / "cold" in the finding.

Caveat: `--follow` tracks single-rename chains only; squash-merge repos (benchflow uses squash-merge on PRs to `main`) under-count. Signal, not proof — a ratio of 2.5× in a squash-merge repo should be treated as closer to 4×.

### Rule 4 — Big modules split by edit-locality

Applies only to modules >500 LOC. Verify with `wc -l` first. Current candidates in benchflow: `_sandbox.py` (~757), `sdk.py` (~706), `job.py` (~582). Probably `process.py` (~379), `viewer.py` (~378) are under threshold but borderline — skip unless they cross 500.

1. `co_edit_matrix.py <file>`.
2. Pairs ≥20% → candidate clusters; split at min-cut where cross-cluster <20%.
3. Status: `no_clusters` → leave the file. `file_renamed` → extend with `git log --follow` or skip. `insufficient_commits` / `too_many_symbols` → skip. `boundary_drift_hunks: N` → reduce confidence.

Don't downgrade to "manual inspection required" — silence beats a weak claim.

**`no_clusters` fallback**: if file >500 LOC with ≥3 importers and Rule 4 is silent, check `importer_selectors`. Disjoint selector sets = importer-driven cleave (e.g., `sdk.py` imported as `SDK` by one set of callers and as `SetupPhase`/`VerifyPhase` helpers by another). Surface as `needs_signoff`; don't mechanize (facades / `__init__.py` re-exports look disjoint by design).

### Rule 5 — Reorg sequencing (leaves before hubs)

Source: manifest `importers.length` + `barrel_importers.length`.

- 0–1 importers: reorg first.
- >5 importers: codemod + `ty check src/` + `.venv/bin/python -m pytest tests/` in same PR.
- `__init__.py` barrels exempt from Rule 2, count toward Rule 5 — rename behind a barrel still needs codemod if the symbol is imported directly (`from benchflow.sdk import SDK` bypasses the `benchflow/__init__.py` barrel for tree-shaking agents).

Extra hub: `src/benchflow/__init__.py` is benchflow's public API surface. Any symbol re-exported there (check `__all__` or direct `from .X import Y`) counts as >5 importers by default — external consumers aren't visible to the codemod.

## Execution

### Short-circuit

If `$SRC` has <20 files or <3000 LOC, one `Explore` agent runs all rules end-to-end. Benchflow (~39 files / ~9k LOC) is **above** this threshold — default to multi-agent split by subtree.

### Pass 1 — discovery

**One agent per subtree, all five rules.** Splitting Rules 1+2 from Rule 3 means wasted merges Rule 3 vetoes. Suggested split for a whole-repo `/arch-audit`:

- `src/benchflow/` top-level (all `*.py` not in subpackages)
- `src/benchflow/agents/`
- `src/benchflow/acp/` + `src/benchflow/cli/`

**Run `manifest.py` once first** — data source for Rules 1/2/5. Only per-file-loop for what manifest lacks: `dominant_symbol.py`, `co_edit_matrix.py`, `churn_ratio.sh`. Parallelize with `xargs -P 8`.

**Primary output: JSON, one object per line.** If a rule can't produce a finding (e.g. Rule 4 `file_renamed`), emit no row. A Rule 4 skip doesn't suppress other rules.

**Do not pre-filter findings you think Pass 2 will kill.** Pass 1 emits when the mechanical check fires; Pass 2 attacks on `rationale_weak` / `importer_payoff` / traps. Self-suppressing with prose like "single-importer facade pattern" or "semantically coherent around SDK phases" is the exact failure Pass 2 exists to catch — let it do its job. The one legitimate suppression is an explicit `≥70%` dominant-symbol exemption. Everything else emits.

Schema:

```
{
  "file": "src/benchflow/foo.py",
  "rule_id": 2,
  "evidence_cmd": ".claude/skills/arch-audit/importers.py src/benchflow/foo.py",
  "evidence_output": "src/benchflow/bar.py",
  "proposed_change": "fold into src/benchflow/bar.py",
  "loc": 57,
  "importers": ["src/benchflow/bar.py"],
  "churn_6mo": 3,
  "co_edit_cluster": null,
  "split_costs": {
    "shared_types": [],
    "shared_helpers": [],
    "importer_payoff": "high | medium | none",
    "rationale": "one sentence: what concrete PR becomes easier?",
    "rationale_weak": false
  }
}
```

Populating `split_costs`:
- `shared_types`/`shared_helpers`: from `split_costs.py`, else manual AST check.
- `importer_payoff`: 1 importer → `none` + `rationale_weak: true` unless rationale names a workflow win. 2–3 → `medium`. 4+ with disjoint selectors → `high`.
- `rationale`: must name a concrete future PR or bug-fix pattern ("splitting `_sandbox.py` lets the path-lockdown tests stop importing user-setup fixtures and cuts the `tests/test_sandbox_hardening.py` mock graph in half"). "Narrative clarity," "isolated tests," "cognitive load" → `rationale_weak: true`. No non-aesthetic rationale → don't emit.

### Pass 2 — adversarial critique

**Do not skip.** Pass 2's value is attacking weak proposals, not confirming strong ones. Spawn one `Explore` agent with this prompt verbatim:

> You are a skeptical reviewer of arch-audit findings. Default REJECT unless proposals survive concrete mechanical attacks. Each finding ends SHIP / SKIP / NEEDS-REWORK / NEEDS-SIGNOFF.
>
> Run attacks in order, stop at first kill:
>
> 1. **Rule 3 veto (merges)**: recompute `churn_ratio.sh`. Threshold is **3×** — if you typed 5× or 8×, re-read the skill.
> 2. **Aesthetic rationale**: if `rationale_weak: true` OR `importer_payoff: none` → SKIP. Demand a concrete PR pattern.
> 3. **Shared-types trap (splits)**: `shared_types` non-empty → 2-way split forces 3rd types module. NEEDS-REWORK.
> 4. **Shared-helpers trap (splits)**: `shared_helpers` non-empty → duplicated helpers or cross-imports. NEEDS-REWORK unless helpers move cleanly.
> 5. **Semantic-marker exemption (Rule 2 folds)**: if top 15 lines have a module docstring or block comment explaining the boundary (keywords: `boundary`, `variant`, `unguarded`, `scoped`, `isolate`, `bridge`, `shim`) → SKIP.
> 6. **Entry-point trap (renames)**: check `pyproject.toml` `[project.scripts]`, `[project.entry-points]`, and direct imports by `src/benchflow/cli/`, `src/benchflow/__init__.py`, and `src/benchflow/sdk.py`. If hit → NEEDS-SIGNOFF.
> 7. **Registry-surface trap**: check if the file contains or is imported by `agents/registry.py` / `agents/providers.py`. These are the documented extension points — any rename invalidates external registration code and needs user-docs update. NEEDS-SIGNOFF.
> 8. **Cycle/barrel trap**: `importers.py` output showing `# barrel:` lines via `__init__.py`; verify proposed splits don't import each other through a barrel.
> 9. **Test colocation**: `tests/test_<name>.py` must move with source without fixture or `conftest.py` restructuring. If the file has a paired test file, name it and check for `conftest.py` dependencies.
>
> All attacks fail → SHIP with counterfactual rationale verbatim. No hedging.

Expect 40–60% kill rate. <20% → re-run with stronger framing. >80% → Pass 1's `rationale_weak` filter is too loose.

### Synthesize

Three buckets:
- **Leaf-first wins** (0–1 importers) — safe now.
- **Hub moves** (>5 importers or entry-point) — codemod, separate PR.
- **Rejected** — brief list with vetoing rule.

Include PR sequence respecting Rule 5.

### Approval

Do NOT edit. On approval:
- **Rule 2 folds** → `/code-cleanup`.
- **Renames/splits** → manual edit + codemod (`grep -rl 'from benchflow.X' | xargs sed -i ...`) + `ty check src/` + `.venv/bin/python -m pytest tests/` in one PR. "Draft PR 1" means diff plan, not commits.

## Cadence

Monthly or per-major-feature. Signal comes from accumulated drift, not single-commit noise. Benchflow has three files already past the Rule 4 threshold (`_sandbox.py`, `sdk.py`, `job.py`) — those are the current top candidates.

## Anti-patterns

- **Don't skip Pass 2** — it kills aesthetic splits and 1-importer theater Pass 1 rubber-stamps.
- **Don't pre-filter in Pass 1 with "facade" / "coherent" / "cohesive" prose.** If the mechanical check fires and dominant-symbol < 70%, emit. Pass 2 decides.
- **Don't split Pass 1 by rule** — split by subtree or not at all. Rule 3 must see Rules 1/2 proposals.
- **Don't propose renames without checking entry points, `__init__.py` re-exports, and registry surfaces.**
- **Don't batch leaf + hub reorgs in one PR** — blast radius must match review posture.
- **Don't grow the rule list.** Five is enough. New signals → `split_costs`, not new rules.
- **Don't accept aesthetic rationales.** Narrative clarity / isolated tests / cognitive load → `rationale_weak: true`.
- **Don't propose a 2-way split without `split_costs.py`** — unchecked shared types/helpers become 3-way splits or cross-import tangles.
- **Don't touch private-underscore modules without reading them.** `_sandbox.py` / `_env_setup.py` look mergeable by name but carry the repo's most security-sensitive code; the underscore prefix is about API surface, not cohesion.
