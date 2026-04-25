# benchflow

Multi-turn agent benchmarking with ACP. Docs in [`docs/`](./docs/).

## Setup + test

```bash
uv venv -p 3.12 .venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest tests/
.venv/bin/ty check src/
ruff check .
```

## Conventions

- **Don't rewrite passing tests** to match new behavior. Update for shape changes, not for semantic changes you don't understand. No tautological tests.
- **Regression tests must name the PR/commit they guard** in the docstring (e.g. `Guards the fix from PR #198 against the regression introduced by PR #193`).
- **Human review before `main`.** PRs only. No force-pushes to `main`. Self-approval doesn't count.
- **Trunk-based:** branch off `main`, PR back to `main`. No long-lived release branches.
- **Releases:** bump `pyproject.toml` to the stable version, tag `v<version>` on main, push tag (CI publishes to PyPI), then bump main to the next `.dev0`.
