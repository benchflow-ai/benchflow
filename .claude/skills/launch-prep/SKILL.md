---
name: launch-prep
description: Prepare a benchflow release for launch. Use when the user says "prepare for launch", "release prep", "cut a release", "bump version", or "launch checklist". Runs docs/labs/code review, CI gate, e2e smoke test, CHANGELOG update, version bump, and opens a PR to main.
user-invocable: true
---

# BenchFlow Launch Prep

Prepares a release: docs/labs/code review → CI gate → e2e smoke → CHANGELOG → version bump → PR.

Arguments passed: `$ARGUMENTS`

---

## Dispatch

**No args** — show release state: current version (`grep '^version' pyproject.toml`), `[Unreleased]` section of CHANGELOG.md, branch and dirty files (`git status --short`). Recommend next step.

**`check`** — run CI gate, report pass/fail, no edits:
```bash
.venv/bin/ruff format --check src tests && .venv/bin/ruff check src tests
.venv/bin/ty check
.venv/bin/python -m pytest tests/ -q
uv lock --check
```

**`patch` / `minor` / `major`** — follow Steps 0–6 below.

---

## Step 0 — Pre-flight

1. **Version:** `grep '^version' pyproject.toml`, compute new version, ask user to confirm before proceeding. Use their value if they override.
2. **Branch:** `git branch --show-current` — if `main`, stop: *"Switch to a feature branch or: `git checkout -b release/v<NEW_VERSION>`"*. Any non-main branch is fine.

---

## Step 1 — Docs, labs, and code review (3 subagents in parallel)

**A — user-facing docs:** Read `README.md` and all `docs/*.md`. Evaluate as a first-time external user: complete install instructions, working examples, accurate CLI flags and API signatures, no broken relative links. Flag anything confusing or stale.

**B — labs/:** For each experiment: README matches code, no broken links, no stale `src/benchflow` API calls, no notebook tracebacks. Grep for `TODO`, `FIXME`.

**C — src/ code smell:** Grep `src/` and `tests/` for `TODO`, `FIXME`, `HACK`, `XXX` (blocker vs. debt). Grep for hardcoded secrets (`api_key\s*=\s*["']`, etc.). Check `__init__.py` for debug symbols.

Synthesize: **Blockers / Polish / Clean**. Stop on blockers.

---

## Step 2 — CI gate

Mirrors `.github/workflows/test.yml` exactly. Stop on any failure.

```bash
.venv/bin/ruff format src tests       # note: ty check takes no path arg
.venv/bin/ruff check src tests
.venv/bin/ty check
.venv/bin/python -m pytest tests/ -q
uv lock --check
```

If `ruff format` changed files: `git diff --name-only`, then `git add <those files only>` — not `git add .`.

---

## Step 3 — e2e smoke test

```bash
source .env 2>/dev/null || true
.venv/bin/python -m pytest -m live tests/test_smoke.py -v
```

If Docker is unavailable, warn and ask to skip or abort — do not skip silently. Expected: `test_hello_world_smoke` passes with reward > 0 and non-empty trajectory.

---

## Step 4 — CHANGELOG draft

If `[Unreleased]` is non-empty: show it, ask to confirm.

If empty: generate from git log —
```bash
git log $(git describe --tags --abbrev=0)..HEAD --oneline --no-merges
```
Map prefixes: `feat:` → Added, `fix:` → Fixed, `chore:`/`refactor:`/`build:` → Changed. Omit `docs:` and `test:` unless notable. Show draft and wait for user approval before writing anything.

---

## Step 5 — Write CHANGELOG, pyproject.toml, and uv.lock

**CHANGELOG:** Insert approved entries as `## <NEW_VERSION> — $(date +%Y-%m-%d)` immediately after `## [Unreleased]`, leaving `[Unreleased]` empty. Use Edit, not Write.

**pyproject.toml + uv.lock:** Edit only `version = "..."` under `[project]` (line 3). Do not touch `target-version` (ruff) or `python-version` (ty) — those are Python version pins. Then run `uv lock` so the editable `benchflow` package entry in `uv.lock` matches the new version. Do not edit `__init__.py`; it reads version from `importlib.metadata` automatically.

---

## Step 6 — Commit and PR

```bash
git add CHANGELOG.md pyproject.toml uv.lock   # plus any ruff-formatted files from Step 2
git commit -m "chore: release v<NEW_VERSION>"
git push -u origin HEAD
gh pr create --title "chore: release v<NEW_VERSION>" --body "$(cat <<'EOF'
## Release v<NEW_VERSION>

See CHANGELOG.md for details.

## Checklist
- [ ] CI gate passes
- [ ] e2e smoke test passes
- [ ] CHANGELOG updated
- [ ] Version bumped in pyproject.toml
- [ ] uv.lock refreshed
- [ ] Merge and tag after review

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Return the PR URL.

---

## Step 7 — Post-merge (remind user, do not run)

```bash
git checkout main && git pull
git tag v<NEW_VERSION> && git push origin v<NEW_VERSION>

# GitHub release — substitute the actual version for <NEW_VERSION> in the awk pattern
awk '/^## <NEW_VERSION> /{flag=1; next} /^## [0-9]/{flag=0} flag' CHANGELOG.md > /tmp/release_notes.md
gh release create v<NEW_VERSION> --title "v<NEW_VERSION>" --notes-file /tmp/release_notes.md

# PyPI publish — separate step (requires UV_PUBLISH_TOKEN or ~/.pypirc)
uv build && uv publish
```
