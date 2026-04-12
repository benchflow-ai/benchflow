# benchflow labs

Research-forward, Docker-heavy scenarios that exercise the full benchflow
SDK. Each entry reproduces a specific situation end-to-end — an attack
against a task, a diagnostic against a trajectory, a before/after sweep
across benchflow versions — and each is self-contained with its own README,
runner, and prerequisites.

## labs vs tutorials

`tutorials/` and `labs/` answer different questions:

| | `tutorials/` | `labs/` |
|---|---|---|
| **Purpose** | teach one concept | reproduce one situation |
| **Setup** | no Docker, no API keys | Docker required, may pin multiple benchflow versions |
| **Runtime** | under 10 seconds | minutes |
| **Scope** | one idea per entry | one end-to-end scenario per entry |
| **Audience** | first-time reader | readers who want to run real SDK against real tasks |

A reader who wants to understand a concept opens `tutorials/`. A reader who
wants to see the concept at work against the real SDK opens `labs/`.

## Contract

Every `labs/<entry>/` directory must:

1. Have its own `README.md` with a **One-command repro** block showing the
   exact command and the expected stdout.
2. Ship a top-level orchestrator script (`run_*.py` or similar) that does
   everything — venv setup, image builds, `SDK().run()` calls, result
   aggregation — so the reader never has to stitch steps together.
3. List its prerequisites explicitly: Docker, Python version, optional `uv`,
   network access, disk space.
4. Be self-contained: no shared state, no cross-entry imports, no assumed
   ordering. Delete one entry and the rest still work.
5. Document what it proves and what it does NOT prove. Labs entries are
   illustrative, not exhaustive; each should point at the broader
   documentation for the claim it's demonstrating.

## Current entries

| entry | what it shows | status |
|---|---|---|
| [`benchjack-sandbox-hardening/`](./benchjack-sandbox-hardening/) | BenchJack `conftest.py` exploit succeeds against `benchflow==0.2.0` and is blocked under HEAD by the 0.2.1 sandbox hardening. | shipped |
| `benchjack-scan/` | CLI that audits existing benchflow tasks for the seven BenchJack attack patterns. | planned |
| `reward-hack-detector/` | Trajectory pattern-matcher that flags reward-hacking attempts post hoc. | planned |

## Non-goals

- **Not tutorials.** Labs assume the reader already understands benchflow's
  basic usage.
- **Not unit tests.** Labs do not gate CI. They run on demand, take minutes,
  and require Docker.
- **Not benchmark task packs.** Full task suites live under
  `benchmarks/` — labs entries use one or a handful of tasks to make a
  specific point.
- **Not documentation.** Design notes and architecture decisions live under
  `docs/` — labs are runnable artifacts.
- **Not promises.** An entry landing in `labs/` does not mean the approach
  it demonstrates will ever become a supported API. Some entries stabilize
  over time into production features; others stay as illustrative
  scenarios. Check each entry's README for its specific stability posture.
