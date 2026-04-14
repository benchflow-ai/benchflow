# Reward-hack matrix — real benchmark tasks, 0.2.0 vs 0.2.2

## Background

[BenchJack](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/) (Berkeley RDI, April 2026)
showed that ~100% of tasks across eight major AI agent benchmarks can be solved by reward-hacking
shortcuts that exploit the *verifier's runtime*, not the agent's intelligence: pytest conftest
hooks, planted PATH binaries, leaked answer keys, and other one-line tricks. [Meerkat](https://debugml.github.io/cheating-agents/)
(Penn, same week) confirmed it from the trace side — frontier models discover and use these
shortcuts in the wild.

Both papers describe the threat. Neither ships a runtime defense.

## What this lab does

Sweeps **666 real tasks** across three production benchmarks (skillsbench, swebench-verified,
terminal-bench-2) with one BenchJack-style exploit per benchmark, run end-to-end through
`SDK().run()` on Daytona under both **benchflow 0.2.0** (no hardening) and **benchflow 0.2.2** (new
sandbox hardening). 1332 trials total. The exploits are direct copies of the BenchJack blog's own
shapes — pytest conftest hook for swebench, PATH-binary trojan for terminal-bench, `.pth` injection
for skillsbench.

## Results

| benchmark | tasks | 0.2.0 EXPLT | 0.2.2 EXPLT | Δ |
|---|---:|---:|---:|---:|
| `skillsbench` | 77 | 16 (20.8%) | 0 (0%) | −20.8 pp |
| `swebench-verified` | 500 | 119 (23.8%) | 1 (0.2%)¹ | −23.6 pp |
| `terminal-bench-2` | 89 | 82 (92.1%) | 0 (0%) | −92.1 pp |
| **total** | **666** | **217 (32.6%)** | **1 (0.15%)** | **−32.4 pp** |

¹ The single 0.2.2 exploit is `swebench-verified/django__django-7530`, which scores reward = 1.0 on
**both** versions because its `FAIL_TO_PASS` test passes at baseline without any patch — a
SWE-bench task-definition quirk, not a 0.2.2 bypass. **True bypass count (cells where 0.2.0 = BLOCK
and 0.2.2 = EXPLT): 0.**

The 0.2.0 numbers are below benchjack's reported ~100% because our exploit is a strict subset of
theirs (single-framework hook per benchmark, not multi-framework). For the hardening claim that
doesn't matter — whether the attacker is naive or sophisticated, 0.2.2 reduces both to zero via
the same defenses. Reproducing benchjack's full multi-framework exploit shape per benchmark is
tracked as the `benchjack-scan` follow-up lab.

Full per-cell results: [`sweep_0.2.0_vs_0.2.2.json`](sweep_0.2.0_vs_0.2.2.json).

## What's new in benchflow 0.2.2

| defense | mechanism | covers |
|---|---|---|
| `sandbox_user="agent"` default | oracle/agent runs as non-root | `/usr/local/bin` and `/usr/local/lib/python3.X/dist-packages` writes blocked at VFS layer (P7) |
| path lockdown | `chown root + chmod 700` on `/tests` and `/solution` before verify | answer-key reads (P2), conftest drops in trusted dirs (P1) |
| build-config snapshot/restore | `setup.py`, `pyproject.toml`, etc. snapshotted pre-agent and restored before verify | install-time RCE via agent-modified setup.py |
| full workspace restore | `/testbed_verify` is the canonical pre-agent copy; `dirs_exist_ok` merge before verify | source-file modification of any project under test |
| pre-verify scrub (`CLEANUP_CMD`) | `find / -name conftest.py -not -path '/tests/*' -delete`; `.pth` / `sitecustomize.py` walk on `sys.path`; `*.py` purge in `/tmp`, `/var/tmp` | agent-planted hooks at any depth (P1, P7) |
| `VERIFIER_ENV` re-pin | hardened `PYTEST_ADDOPTS`, `PYTHONPATH=""`, `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, etc. | env-var injection, plugin entry-point loading |

See [`.dev-docs/harden-sandbox.md`](../../.dev-docs/harden-sandbox.md) for the full hardening audit.

## Reproducing

```sh
./fetch_corpora.sh                         # one-time, ~400 MB
python run_matrix.py --sweep --concurrency 64 \
    --summary-path .jobs/matrix_sweep.json
```

Runs in ~20 minutes on Daytona at concurrency 64. The long-lived worker pool (`_worker.py`) keeps
local RAM at ~1 GB regardless of trial count and wraps every `sdk.run()` in a 900 s
`asyncio.wait_for` so a hung sandbox cannot starve the semaphore. Pass `--limit 1` for a 6-trial
smoke run in ~90 s.

The lab uses benchflow's `agent="oracle"` mode, which runs `solution/solve.sh` directly. We
overwrite `solve.sh` with the BenchJack exploit payload, so the demo is deterministic and free of
LLM API calls. Swap `agent="oracle"` for `agent="claude-agent-acp"` in `_runner.py` to test with
a real frontier model.

## Out of scope

This lab tests two of BenchJack's seven patterns end-to-end on real tasks: **P1 (conftest-hook)**
and **P7 (path-trojan / pth-injection)**. The others:

- **P2 (answer-lookup)** — addressed architecturally by `chmod 700 /solution`. Demonstrated
  synthetically in [`labs/benchjack-sandbox-hardening/pattern2_answer_lookup`](../benchjack-sandbox-hardening/pattern2_answer_lookup/);
  not yet swept against real benchmarks (per-benchmark answer-file shape varies).
- **P3 / P4 / P5** — verifier-code-quality issues, not runtime-hardening issues. benchflow doesn't
  `eval()` agent input, doesn't ship an in-tree LLM judge, and uses reward-based scoring rather
  than substring matching. Out of scope per the threat model in
  [`harden-sandbox.md`](../../.dev-docs/harden-sandbox.md).
- **P6 (trivial verifier)** — addressed architecturally via the `verifier_error` field, not as a
  pattern blocked at runtime.

## Future work

1. **`benchjack-scan` lab** — multi-framework exploit per benchmark (pytest hook + unittest
   monkey-patch + project-specific runners) so the 0.2.0 baseline mirrors benchjack's ~100%
   directly. Proves the 0.2.2 defenses hold against the strongest single-pattern attacker, not
   just the naive one.
2. **Real-LLM attacker mode** — swap `agent="oracle"` for `agent="claude-agent-acp"` and measure
   whether a capable frontier model discovers the exploits without being told to cheat. Reproduces
   the Meerkat trace-detection findings against benchflow specifically.
3. **Memory / disk task-toml clamps** in `_prepare_cell` to mirror the existing CPU clamp. The
   current sweep loses ~8 skillsbench tasks to Daytona resource caps; the clamp would shave them
   off the rollup.

## Independent concurrent validation

Two research groups published complementary work within 24 hours of each other in early April 2026:

- **[Hao Wang et al. (Berkeley RDI) — "How We Broke Top AI Agent Benchmarks"](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/)** — released
  [`moogician/trustworthy-env`](https://github.com/moogician/trustworthy-env), a static + LLM + Z3
  auditor for benchmark source code. They report "Exploit Score 100%" on SWE-bench Verified via a
  `conftest.py` hook injection that matches the P1 cell above. Their release does not include a
  benchmark runner; this lab covers the runtime evidence.
- **[BrachioLab (Penn) — "Finding Widespread Cheating on Popular Agent Benchmarks"](https://debugml.github.io/cheating-agents/)** — released
  [`BrachioLab/Meerkat`](https://github.com/BrachioLab/Meerkat), a post-hoc trace auditor. Their
  headline finding for Terminal-Bench 2 Pilot: agents read answer keys from `/tests` in 415/429
  traces — the exact behavior benchflow 0.2.2's `chmod 700 /tests` lockdown blocks at the VFS layer.

Three complementary angles: Berkeley = source-level audit, Penn = trace-level detection, benchflow
0.2.2 = runtime prevention.
