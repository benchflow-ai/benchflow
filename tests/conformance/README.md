# ACP conformance tasks

Small, deterministic tasks that check an agent registered in `src/benchflow/agents/registry.py` speaks ACP end to end.

## Tasks

| Task | Covers |
|---|---|
| `acp_smoke/` | ACP handshake · single tool call · file write · terminal reward |

`acp_smoke/` carries a native `task.md` entrypoint while retaining the legacy
split files used by older conformance docs and tests.

## Run

```bash
bench eval run --tasks-dir tests/conformance/acp_smoke --agent <agent-name> --model claude-haiku-4-5-20251001
```

A working agent returns `reward=1`. `run_conformance.py` runs the smoke task for every registered agent whose credentials are set and writes `conformance-results.json` next to itself. No release step requires a run.
