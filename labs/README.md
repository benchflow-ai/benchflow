# benchflow labs

Runnable, Docker-heavy scenarios that exercise the full benchflow SDK end-to-end.
Each entry is self-contained with its own README, orchestrator script, and prerequisites.

For an indexed overview of all labs with run commands and key takeaways, see [docs/labs.md](../docs/labs.md).

## Contract

Every `labs/<entry>/` must:

1. Have a `README.md` with a **One-command repro** block showing the exact command and expected stdout.
2. Ship a top-level orchestrator (`run_*.py` or similar) that handles venv setup, image builds, `SDK().run()` calls, and result aggregation.
3. List prerequisites explicitly: Docker, Python version, optional `uv`, network, disk.
4. Be self-contained — no shared state, no cross-entry imports, no assumed ordering.
5. Document what it proves and what it does not.

## Entries

| entry | what it shows |
|---|---|
| [`benchjack-sandbox-hardening/`](./benchjack-sandbox-hardening/) | BenchJack `conftest.py` exploit succeeds against `benchflow==0.2.0` and is blocked under HEAD by the 0.2.1 sandbox hardening. |
