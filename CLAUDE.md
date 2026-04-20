# benchflow

Multi-turn agent benchmarking with ACP.

Architecture, CLI, task format: see `docs/architecture.md`, `docs/cli-reference.md`, `docs/task-authoring.md`. Internal refactor notes and SDK reference: `.dev-docs/`.

## Setup

Requires Python 3.12+. Uses `uv`.

```bash
uv venv -p 3.12 .venv && uv pip install -e ".[dev]"
```

## Test

```bash
.venv/bin/python -m pytest tests/          # unit (fast, no Docker)
.venv/bin/python -m pytest -m live tests/  # e2e (Docker + API key)
.venv/bin/ty check src/                    # type check — also the fastest "find references" after any signature change
```

CI gates `ruff format`, `ruff check`, `pytest`, and `ty check src/`. Run all four before pushing. Live tests use Haiku 4.5 (`claude-haiku-4-5-20251001`).

## Conventions

- **Minimal fix.** Do only what was asked; "leave as is" is a valid outcome; generalize on the third repetition. "Narrow" is about *how much* code changes, not *where* — the fix still goes at the root cause, even in a different file.
- **Cite `file:line` before fixing.** If you can't, the bug may not exist.
- **Root cause, sweep callers.** Two PRs patching two files for one bug means the fix is upstream. Grep every caller before patching one — a flag at the call site is a smell the function itself is wrong. After fixing a shared helper, audit its other callers in the same PR; each either hits the same bug (same fix) or demonstrably doesn't (note in PR).
- **Don't change a happy path to fix an edge case.** If a line looks wrong but serves a legitimate flow, narrow the fix. `git log -p` it before flipping — two-week-old "intent" is not intent.
- **Registry over hardcode.** Adding an agent or provider is a dict entry in `agents/registry.py` or `providers.py` — not a new code path. If the registry declares `contextWindow`, `baseURL`, or a model ID, callers read it from there; hardcoded copies drift. The `oracle` special case in `sdk.py` exists because it bypasses the agent loop; don't add more without the same justification.
- **Read siblings before adding one.** New launcher/provider/shim? Open every file in the directory and reconcile defaults before writing.
- **Don't rewrite passing tests.** Updating a test because the code it covers changed shape is fine; rewriting one to match new behavior without understanding why it was written is not. No tautological tests (dataclass reads, stdlib behavior, "does it construct").
- **Test new regressions against `main` first.** A test that passes on buggy `main` pins the bug instead of preventing it. Name the commit/PR it guards.
- **Human review before main.** Commit freely on a feature branch, open a PR. Never push to `main` directly, never force-push it. Self-approval doesn't count — request an independent reviewer.
