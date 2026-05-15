# benchflow

Multi-turn agent benchmarking with ACP. See `CLAUDE.md` for conventions and `README.md` for full documentation.

## Cursor Cloud specific instructions

### Quick reference

All standard commands are in `CLAUDE.md`:
```bash
uv sync --extra dev --locked
uv run python -m pytest tests/
uv run ty check src/
uv run ruff check .
```

### Notes for cloud agents

- **Python 3.12+ required.** The VM ships with 3.12; do not downgrade.
- **`uv` must be on PATH.** The update script installs it to `~/.local/bin`. If a shell session doesn't find `uv`, run `export PATH="$HOME/.local/bin:$PATH"`.
- **Tests skip live tests by default.** `pyproject.toml` sets `addopts = "-m 'not live'"`. Tests marked `@pytest.mark.live` require Docker and a real API key (e.g. `ANTHROPIC_API_KEY`). The default `pytest` run (811+ tests) uses only mocks.
- **Running actual benchmarks** (`bench run`, `bench eval create`) requires Docker and at least one LLM API key. These are not needed for lint/test/type-check workflows.
- **CLI entry points:** both `bench` and `benchflow` are registered as console scripts. Use `uv run bench <subcommand>`.
- **Task validation:** `bench tasks check <dir>` validates task structure (needs `task.toml`). `bench tasks init <name>` scaffolds a new task.
