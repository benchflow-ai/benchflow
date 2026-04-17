# ACP conformance tasks

Small, deterministic tasks that every registered agent in `src/benchflow/agents/registry.py` must pass before a release tag. See `docs/0.3-plan.md` §A6.

## Tasks

| Task | Covers |
|---|---|
| `acp_smoke/` | ACP handshake · single tool call · file write · terminal reward |

## Run

```bash
benchflow run -t tests/conformance/acp_smoke -a <agent-name> -m claude-haiku-4-5-20251001
```

Every agent in the registry must return `reward=1` on the smoke task against the oracle solution, and must complete cleanly for Haiku 4.5 as the driving model.

## Gate for 0.3

Per `docs/0.3-plan.md` §A6, release blocked until every registered agent has at least one green conformance run on record. Agents that can't pass the smoke are either fixed or removed from the registry.
