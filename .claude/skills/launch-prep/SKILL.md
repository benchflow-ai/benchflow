---
name: launch-prep
description: Prepare a benchflow release for launch. Use when the user says "prepare for launch", "release prep", "cut a release", "bump version", or "launch checklist". Runs docs/labs/code review, CI gate, e2e smoke test, CHANGELOG update, version bump, and opens a PR to main.
user-invocable: true
---

# BenchFlow Launch Prep

Prepares a release: docs/labs/code review → CI gate → e2e smoke → CHANGELOG → version bump → commit → PR.

Arguments passed: `$ARGUMENTS`

---

## Dispatch on arguments

### No args — show current release state

1. Read current version: `grep '^version' pyproject.toml`
2. Show `[Unreleased]` section of CHANGELOG.md (if empty, warn)
3. Show current branch and any uncommitted changes: `git status --short`
4. Recommend next step (e.g. `/launch-prep patch`)

### `check` — preflight only, no changes

Run the CI gate and report pass/fail. Make no edits.

```bash
.venv/bin/ruff format --check src/ tests/
.venv/bin/ruff check src/ tests/
.venv/bin/ty check src/
.venv/bin/python -m pytest tests/ -q
```

Stop and report any failures. All four must pass before proceeding.

### `patch` / `minor` / `major` — full release prep

Follow steps 1–7 in order. Stop and ask the user if anything is unclear or if there are blockers.

---

## Step 1 — Docs, labs, and code review

Run three subagents in parallel before touching anything else.

**Subagent A — docs review:**
- Read `README.md` and all `.dev-docs/*.md`. Check that code examples, CLI flags, file paths, and API signatures match the current `src/benchflow` implementation. Flag any that are stale.
- Check all relative markdown links (`[text](path)`) in README and .dev-docs — do the target files/dirs actually exist?
- Confirm `CHANGELOG.md` has a non-empty `[Unreleased]` section.

**Subagent B — labs/ review:**
- For each experiment under `labs/`, verify: README accurately describes the code, no broken relative links, no stale `src/benchflow` API calls (e.g. wrong SDK method names, wrong result field names).
- Check notebook output cells for tracebacks or errors.
- Grep `labs/` for `TODO`, `FIXME`.

**Subagent C — src/ code smell:**
- Grep `src/` and `tests/` for `TODO`, `FIXME`, `HACK`, `XXX`. Classify each as blocker vs. acceptable debt.
- Grep for hardcoded secret patterns: `api_key\s*=\s*["']`, `token\s*=\s*["']`, `password\s*=\s*["']`.
- Check `__init__.py` exports for obviously unfinished or debug symbols.

Synthesize into a punch list: **Blockers** / **Polish** / **Clean**. Stop if there are blockers and wait for the user to resolve them.

---

## Step 2 — CI gate

Run all four checks. If any fail, report the failures and stop. Do not proceed until the gate is green.

```bash
.venv/bin/ruff format src/ tests/
.venv/bin/ruff check src/ tests/
.venv/bin/ty check src/
.venv/bin/python -m pytest tests/ -q
```

If `ruff format` made changes, stage them and include in the version-bump commit (step 6).

---

## Step 3 — e2e smoke test

Run the live smoke test against a real Docker daemon and API key.

```bash
source .env
.venv/bin/python -m pytest -m live tests/test_smoke.py -v
```

- If Docker is unavailable, warn the user and ask whether to skip or abort. Do not skip silently.
- If the smoke test fails, stop and report. Do not proceed to version bump until e2e passes.
- Expected: `test_hello_world_smoke` passes with reward > 0 and non-empty trajectory.

---

## Step 4 — Draft CHANGELOG entries

Read `CHANGELOG.md`. Check if `[Unreleased]` already has content.

**If `[Unreleased]` is non-empty:** show the entries to the user and ask for confirmation before using them as-is.

**If `[Unreleased]` is empty:** generate entries from git log since the last release tag.

```bash
# Find the last release tag
git describe --tags --abbrev=0

# List commits since that tag
git log <last-tag>..HEAD --oneline
```

Group the commits into Added / Changed / Fixed / Deprecated sections using conventional commit prefixes (`feat:` → Added, `fix:` → Fixed, `chore:`/`refactor:` → Changed). Show the draft to the user and ask them to confirm or edit before proceeding. Do not write to CHANGELOG.md until the user approves the draft.

---

## Step 5 — Compute new version

Read current version from `pyproject.toml` (`version = "X.Y.Z"`).

Apply the requested bump:
- `patch` → X.Y.(Z+1)
- `minor` → X.(Y+1).0
- `major` → (X+1).0.0

Show the user: "Bumping X.Y.Z → X.Y.Z+1. Proceed?"

---

## Step 6 — Update CHANGELOG.md and pyproject.toml

**CHANGELOG:** Insert the approved entries (from Step 4) as a new versioned section immediately after the `## [Unreleased]` heading, leaving `[Unreleased]` empty. Get today's date via `date +%Y-%m-%d`.

Result:
```markdown
## [Unreleased]

## 0.2.2 — 2026-04-12

### Added
- ...

## 0.2.1 — ...
```

Use Edit (not Write) — preserve all existing content below.

**pyproject.toml:** Edit the single `version = "..."` line. No other changes.

---

## Step 7 — Commit and open PR

Stage only the modified files:

```bash
git add CHANGELOG.md pyproject.toml
```

Commit:
```bash
git commit -m "chore: release v<NEW_VERSION>"
```

Push and open a PR to main:
```bash
git push -u origin HEAD
gh pr create --title "chore: release v<NEW_VERSION>" --body "$(cat <<'EOF'
## Release v<NEW_VERSION>

See CHANGELOG.md for details.

## Checklist
- [ ] CI gate passes
- [ ] e2e smoke test passes
- [ ] CHANGELOG updated
- [ ] Version bumped in pyproject.toml
- [ ] Merge and tag after review

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Return the PR URL to the user.

---

## Step 8 — Post-merge (manual, on main)

After the PR is merged, remind the user to run these — do NOT run automatically:

```bash
git checkout main && git pull
git tag v<NEW_VERSION>
git push origin v<NEW_VERSION>
uv build
uv publish
```

`uv publish` requires `UV_PUBLISH_TOKEN` or `~/.pypirc` configured.

---

## Notes

- CI gate mirrors exactly what `.github/workflows/test.yml` enforces.
- `ty check src/` baseline is zero errors — any new error is a blocker.
- Never push directly to main. Always go through a PR.
