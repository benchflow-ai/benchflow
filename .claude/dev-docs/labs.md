# Labs

Runnable, Docker-heavy experiments that exercise the full benchflow SDK end-to-end. Labs are distinct from unit tests (real Docker, no mocking) and from docs (executable, with expected output). Each lab is self-contained with its own README and orchestrator script.

> **Historical (0.2.x-era).** These labs are archived under [`docs/labs/`](../../docs/labs/). They compare benchflow 0.2.0 against 0.2.1/0.2.2 and are kept as cited security evidence; the hardening they validate still ships. The public write-up is [`docs/sandbox-hardening.md`](../../docs/sandbox-hardening.md).

| Lab | Question summary | Benchflow versions | API key needed |
| --- | --- | --- | --- |
| [`benchjack-sandbox-hardening`](../../docs/labs/benchjack-sandbox-hardening/) | Does 0.2.1 block BenchJack exploits that succeed under 0.2.0? | 0.2.0 vs 0.2.1 | No |
| [`reward-hack-matrix`](../../docs/labs/reward-hack-matrix/) | Do the same exploits succeed on real benchmark tasks, and does 0.2.2 block them? | 0.2.0 vs 0.2.2 | Optional (`DAYTONA_API_KEY`) |

Each lab's README documents its prerequisites, the one-command repro, and key takeaways. See [`docs/sandbox-hardening.md`](../../docs/sandbox-hardening.md) for the narrative and results tables.

## See also

- [`harden-sandbox.md`](./harden-sandbox.md) — full seven-pattern BenchJack threat model and hardening audit
