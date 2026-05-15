# benchflow

Multi-turn agent benchmarking with ACP. Docs live in [`docs/`](./docs/).

`AGENTS.md` is the single source of truth for agent-facing repository
instructions. `CLAUDE.md` is deprecated and kept only as a compatibility
pointer for tools that still auto-load it.

## Setup + test

```bash
uv sync --extra dev --locked
uv run python -m pytest tests/
uv run ty check src/
uv run ruff check .
```

## Conventions

- **Don't rewrite passing tests** to match new behavior. Update for shape changes, not for semantic changes you don't understand. No tautological tests.
- **Regression tests must name the PR/commit they guard** in the docstring (e.g. `Guards the fix from PR #198 against the regression introduced by PR #193`).
- **Human review before `main`.** PRs only. No force-pushes to `main`. Self-approval doesn't count.
- **Trunk-based:** branch off `main`, PR back to `main`. No long-lived release branches.
- **Releases:** bump `pyproject.toml` to the stable version, tag `v<version>` on main, push tag (CI publishes to PyPI), then bump main to the next `.dev0`.
- **Comments:** prefer self-explanatory code; add comments only for non-obvious rationale, invariants, workarounds, or module orientation.
- **Validation:** don't add error handling, fallbacks, or validation for scenarios that can't happen.

## Cursor Cloud specific instructions

- **Python 3.12+ required.** The VM ships with 3.12; do not downgrade.
- **`uv` must be on PATH.** The update script installs it to `~/.local/bin`. If a shell session doesn't find `uv`, run `export PATH="$HOME/.local/bin:$PATH"`.
- **Tests skip live tests by default.** `pyproject.toml` sets `addopts = "-m 'not live'"`. Tests marked `@pytest.mark.live` require Docker and a real API key (e.g. `ANTHROPIC_API_KEY`). The default `pytest` run uses mocks.
- **Running actual benchmarks** (`bench run`, `bench eval create`) requires Docker and at least one LLM API key. These are not needed for lint/test/type-check workflows.
- **CLI entry points:** both `bench` and `benchflow` are registered as console scripts. Use `uv run bench <subcommand>`.
- **Task validation:** `bench tasks check <dir>` validates task structure (needs `task.toml`). `bench tasks init <name>` scaffolds a new task.

## Backlog and known issues

- **ATIF export:** `src/benchflow/trajectories/atif.py` and `src/benchflow/trajectories/claude_code.py` are backlog modules, not wired into the default SDK path yet.
- **Job resume config scoping:** `Job` resume currently keys on completed result files, not the full job config. Changing agent/config values in the same `jobs_dir` can therefore reuse prior task results.
