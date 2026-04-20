---
name: branch-review
description: Pre-push branch reviewer — runs lint+typecheck+tests, then fans /code-cleanup, /test-review, /docs-review at the branch diff, merges findings by file
user-invocable: true
---

# Branch Review Skill

Pre-push sanity check. Runs correctness gates first (Pass 0), then invokes the three review skills against the branch diff (`origin/main...HEAD`), collects their native outputs verbatim, appends a merge-by-file index. Use before `git push` / PR.

**Do not auto-fix.** Each sibling refuses to auto-edit; this skill preserves that.

Sibling to `/launch-prep` — that skill runs the full release pipeline (CHANGELOG, version bump, PR to main); this one is the lightweight pre-push counterpart.

## Commands

- `/branch-review` — default: Pass 0 + `/code-cleanup` + `/test-review` (docs skipped)
- `/branch-review --full` — includes `/docs-review`
- `/branch-review --fast` — skips Pass 0's test run; uses `--recent` on code-cleanup where useful
- `/branch-review --code-only` / `--tests-only` / `--docs-only` — single-skill shortcut
- `/branch-review --base=<ref>` — non-default base (default: `origin/main`)
- `/branch-review --include-dirty` — add staged+unstaged to the diff scope

## Preflight

1. **Refresh base.** `git fetch origin main --quiet` (or `<base>`). Stale local `main` re-surfaces upstream-merged commits.
2. **Git state check.** Refuse on detached HEAD or active rebase/merge (`git status --porcelain=2 --branch` → look for `in progress`). Warn if working tree is dirty; offer `--include-dirty` to add staged+unstaged to scope, else diff is branch-only.
3. **Compute diff.** `git diff --name-only origin/main...HEAD` (+ `git diff --name-only HEAD` if `--include-dirty`).
4. **Scale gate.**
   - Empty diff → exit with "no changes vs base".
   - ≤30 files → proceed as normal.
   - 31–60 files → proceed but force `--fast` and print a "large branch" warning.
   - \>60 files → refuse. Print the partition counts and suggest: narrow base (`--base=<recent-commit>`), scope to subtree (`--paths=src/benchflow/agents`), or invoke the siblings directly.

## Partition

Route changed files by path:

- **tests** → `/test-review`: `tests/**/test_*.py`, `tests/**/*_test.py`
- **src** → `/code-cleanup`: `src/benchflow/**/*.py`
- **docs** → `/docs-review`: `*.md`, `docs/**/*`, `.dev-docs/**/*`, `src/benchflow/**/*.md`

**Skip silently** (no routing, no findings, no warning): `uv.lock`, `.venv/**`, `__pycache__/**`, `*.egg-info/**`, `dist/**`, `build/**`, `.pytest_cache/**`, generated files.

**Flag in rollup but do not review** — config changes the author should eyeball themselves: `pyproject.toml`, `.pre-commit-config.yaml`, `.github/**`, `.claude/**`, `ruff.toml`, `pytest.ini`.

**Renames:** if `git diff -M --name-status` shows `R100` (pure rename, no content delta) → surface as rename, route nothing. `R<100` → route the new path normally.

**Test-orphan detection:** if `src/benchflow/X.py` changed but `tests/test_X.py` (or `tests/**/test_X.py`) did not, include the matching test file in the test partition. Benchflow's test layout is flat under `tests/` with some subdirs — glob for `test_<basename>.py` rather than assuming co-location. Do not extrapolate beyond direct name match.

## Pass 0 — correctness gates (unless `--fast`)

Run in parallel Bash calls. CI gates all four per `CLAUDE.md`, so this skill mirrors CI:

- `.venv/bin/ruff format --check src/ tests/` — formatting.
- `.venv/bin/ruff check src/ tests/` — lint.
- `.venv/bin/ty check src/` — typecheck (whole-src, cheap).
- `.venv/bin/python -m pytest <changed test files + orphan-detected siblings>` — scoped to the test partition, not full suite. Unit only; **do not pass `-m live`** (requires Docker + API key, not appropriate for a pre-push gate).

With `--fast`: skip the pytest run only. Keep ruff + ty — they're fast enough to always run.

If any gate fails, surface at the top of the report as **BLOCKERS** and short-circuit the review: there's no point polishing a branch that doesn't lint, compile, or pass tests. Author fixes those, then re-runs.

## Pass 1 — review skills

Invoke the enabled siblings via the **Skill tool in the main thread**, one after another. Pass each the space-separated partition as the argument. Each sibling already runs its internal subagent fan-out in parallel, so within-skill concurrency is preserved; the orchestrator just waits on each skill's native report.

Example invocations:
- `Skill("code-cleanup", "<src paths>")`
- `Skill("test-review", "<test paths>")`
- `Skill("docs-review", "<doc paths>")`  (only if `--full`)

With `--fast`, prefer `Skill("code-cleanup", "--recent")` when the branch touches >10 src files — it caps subagent work. `test-review` and `docs-review` take explicit path lists; pass the partition directly.

Collect the three native outputs **verbatim**. Do not summarize, rewrite, or translate their vocabularies — each skill's bucketed structure (cleanup's 7 categories, test-review's delete/collapse/loosen/add, docs-review's drift/stale/duplicate) encodes its calibration. Preserving it lets the author re-use their mental model from direct invocations.

## Output

```
[If BLOCKERS from Pass 0]
## Blockers
- ruff format: <files needing reformat>
- ruff check: <rule violations, file:line>
- ty: <type errors, file:line>
- pytest: <failing test names>

[Else continue to reviews]

<verbatim code-cleanup report>

<verbatim test-review report>

<verbatim docs-review report, if --full>

## Merge-by-file index
Author-oriented cross-reference. Does NOT re-rank; lists which skills flagged which files so co-located fixes surface together.

### src/benchflow/agents/registry.py
- [cleanup] 2 findings (1 verified, 1 needs-judgment)
- [test] tests/test_registry.py: 1 delete, 1 loosen

### docs/architecture.md
- [docs] 1 blocker, 2 stale

### Flagged config (not reviewed)
- pyproject.toml, .github/workflows/ci.yml

...
```

End with a one-line rollup: `N files reviewed · <cleanup-count> cleanup · <test-count> test · <docs-count> docs · <blocker-count> blockers`.

## Anti-patterns

- **Don't re-verify findings.** Each sibling runs its own verification pass. Re-auditing here is duplication.
- **Don't re-rank or translate vocabularies.** The skills calibrate on their native buckets; flattening to a common severity destroys that signal.
- **Don't expand scope beyond test-orphan detection.** If the author wanted arch-level review, they'd invoke `/arch-audit` directly.
- **Don't run at repo scope.** That's the scale gate's job — enforce it, don't silently accept 76-file branches.
- **Don't skip Pass 0 outside `--fast`.** A clean review of uncompileable code is worse than useless.
- **Don't run `pytest -m live`.** Live tests need Docker + an API key; pre-push gates should be hermetic and fast.
- **Don't re-run immediately after fixes.** Use the sibling skills' scoped modes (`<path>` argument, `--recent`) to re-verify just the touched files.
- **Don't duplicate `/launch-prep`.** If the user is cutting a release (version bump, CHANGELOG, PR to main), route them there instead.
