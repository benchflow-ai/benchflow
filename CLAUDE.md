# benchflow

Multi-turn agent benchmarking with ACP.

Architecture, CLI, task format: see `docs/architecture.md`, `docs/cli-reference.md`, `docs/task-authoring.md`. Internal refactor notes and SDK reference: `.dev-docs/`.

## Setup

Requires Python 3.12+. Uses `uv`.

```bash
uv venv -p 3.12 .venv && uv pip install -e ".[dev]"
.venv/bin/pre-commit install
```

## Test

```bash
.venv/bin/python -m pytest tests/          # unit (fast, no Docker)
.venv/bin/python -m pytest -m live tests/  # e2e (Docker + API key)
.venv/bin/ty check src/                    # type check — also the fastest "find references" after any signature change
```

CI gates `ruff format`, `ruff check`, `pytest`, and `ty check src/`. Run all four before pushing. Live tests use Haiku 4.5 (`claude-haiku-4-5-20251001`).

## Conventions

- **Minimal fix.** Do only what was asked. "Leave as is" is a valid outcome. Generalize on the third repetition, not the first.
- **Registry over hardcode.** Adding an agent or provider is a dict entry in `agents/registry.py` or `providers.py` — not a new code path. The `oracle` special case in `sdk.py` exists because it bypasses the agent loop; don't add more without the same justification.
- **Don't rewrite passing tests.** Updating a test because the code it covers changed shape is fine. Rewriting one to match new behavior without understanding why it was written is not. No tautological tests (dataclass reads, stdlib behavior, "does it construct").
- **Human review before main.** Commit freely on a feature branch, open a PR. Never push to `main` directly, never force-push it.
