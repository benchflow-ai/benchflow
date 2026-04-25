# benchflow

Multi-turn agent benchmarking with ACP.

Docs: `docs/quickstart.md`, `docs/cli-reference.md`, `docs/api-reference.md`, `docs/task-authoring.md`, `docs/use-cases.md`, `docs/progressive-disclosure.md`.

## Setup

Requires Python 3.12+. Uses `uv`.

```bash
uv venv -p 3.12 .venv && uv pip install -e ".[dev]"
```

## Test

```bash
.venv/bin/python -m pytest tests/
.venv/bin/ty check src/
```

## Conventions

- **Don't rewrite passing tests.** Updating a test because the code it covers changed shape is fine; rewriting one to match new behavior without understanding why it was written is not. No tautological tests (dataclass reads, stdlib behavior, "does it construct").
- **Test new regressions against `main` first.** A test that passes on buggy `main` pins the bug instead of preventing it. Name the commit/PR it guards.
- **Human review before main.** Commit freely on a feature branch, open a PR. Never push to `main` directly, never force-push it. Self-approval doesn't count — request an independent reviewer.
