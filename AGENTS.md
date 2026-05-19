# benchflow

Multi-turn agent benchmarking with ACP. Docs live in [`docs/`](./docs/).

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
- **Compatibility examples must be real benchmark tasks.** For third-party framework integration, docs, demos, and smoke evidence must use real benchmark entries, not `hello-world`, `toy`, or other fake showcase tasks. Synthetic fixtures are acceptable only inside unit tests and should use real benchmark-style names.
