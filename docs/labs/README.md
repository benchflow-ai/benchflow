# Archived labs (historical, 0.2.x-era)

These are runnable research artifacts from the benchflow 0.2.x cycle. They are
kept as cited security evidence for the sandbox-hardening story
(see [`docs/sandbox-hardening.md`](../../docs/sandbox-hardening.md)); the
defenses they validate still ship. They are archived because they compare
benchflow 0.2.0 against 0.2.1/0.2.2 and are not part of the current release flow.

| Lab | Question | Benchflow versions |
| --- | --- | --- |
| [`benchjack-sandbox-hardening/`](./benchjack-sandbox-hardening/) | Does 0.2.1 block BenchJack exploits that succeed under 0.2.0? | 0.2.0 vs 0.2.1 |
| [`reward-hack-matrix/`](./reward-hack-matrix/) | Do the same exploits succeed on real benchmark tasks, and does 0.2.2 block them? | 0.2.0 vs 0.2.2 |

Each lab is self-contained with its own README and orchestrator script.
