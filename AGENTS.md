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

## Agent skills

### Issue tracker

GitHub Issues at `benchflow-ai/benchflow`, driven via the `gh` CLI. See [`docs/agents/issue-tracker.md`](./docs/agents/issue-tracker.md).

### Triage labels

Canonical vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). `wontfix` already exists in the repo; the other four are not yet created — run `gh label create` if/when you start using them. See [`docs/agents/triage-labels.md`](./docs/agents/triage-labels.md).

### Domain docs

Single-context layout — `CONTEXT.md` and `docs/adr/` live at the repo root (neither file exists yet; producer skills create them lazily). See [`docs/agents/domain.md`](./docs/agents/domain.md).
