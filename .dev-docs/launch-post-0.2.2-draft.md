# benchflow 0.2.2: closing the runtime side of agent benchmark reward hacking

*Draft — edit freely before publishing.*

---

**TL;DR.** benchflow 0.2.2 ships sandbox hardening that cuts the BenchJack-style exploit
success rate from **32.6% to 0.15%** across 666 real tasks in skillsbench, swebench-verified,
and terminal-bench-2. The residual 0.15% is a single SWE-bench task-definition quirk, not a
hardening bypass. True bypass count: **0**.

## The threat

In early April 2026, two independent teams documented that AI agent benchmarks are broken at the
runtime layer:

- **[BenchJack](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/)** (Berkeley RDI) showed
  that eight major benchmarks — Terminal-Bench, SWE-bench Verified/Pro, WebArena, FieldWorkArena,
  CAR-bench, GAIA, OSWorld — can be solved at ~100% with one-line shortcuts that hijack the
  verifier's runtime: dropped `conftest.py` hooks, planted PATH binaries, leaked answer keys,
  `file://` URL navigation, normalization collisions.

- **[Meerkat](https://debugml.github.io/cheating-agents/)** (BrachioLab / Penn) found the same
  thing from the trace side. Terminal-Bench 2 Pilot: **415 of 429 traces** show agents reading
  answer keys directly from `/tests`.

Both papers describe the threat. Neither ships a runtime defense.

## What 0.2.2 ships

A four-tier sandbox hardening pass that runs inside `SDK._verify()` before any reward signal is
computed:

1. **Pre-agent workspace snapshot.** The task's canonical content is captured at
   `/testbed_verify` before the agent runs a single command.
2. **Build-config snapshot/restore.** `setup.py`, `pyproject.toml`, `setup.cfg`, `tox.ini`,
   `noxfile.py`, and friends are snapshotted pre-agent and restored before the verifier runs.
   Closes the "agent modifies `setup.py` and `pip install` runs agent code as root" primitive.
3. **Full workspace restore + scrub.** `dirs_exist_ok` merge-copy from `/testbed_verify`
   overwrites agent modifications. A companion `CLEANUP_CMD` walks the filesystem and deletes
   injected `conftest.py`, `.pth`, `sitecustomize.py`, and `usercustomize.py` files outside the
   trusted `/tests` directory, plus `*.py` drops in `/tmp` and `/var/tmp`.
4. **Hardened verifier env.** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `PYTHONPATH=""`,
   `PYTHONDONTWRITEBYTECODE=1`, `PYTHONBREAKPOINT=0`, canonical `PATH`, hardened `PYTEST_ADDOPTS`
   (`-c /dev/null --confcutdir=/tests --rootdir=/tests -p no:cacheprovider`), and friends.
   Closes env-var injection and pytest entry-point plugin loading.

Plus the `sandbox_user="agent"` default (non-root agent) that was already in 0.2.1, which blocks
`/usr/local/bin` and system `dist-packages` writes at the VFS layer.

Full audit, tier-by-tier: [`.dev-docs/harden-sandbox.md`](../.dev-docs/harden-sandbox.md).

## The evidence

We built [`labs/reward-hack-matrix/`](../labs/reward-hack-matrix/) to run BenchJack-shaped
exploits against every task in three production benchmarks, end-to-end through `SDK().run()`
on Daytona, under both benchflow versions. One hand-crafted exploit per benchmark, 666 cells,
1332 trials.

| benchmark | tasks | 0.2.0 EXPLT | 0.2.2 EXPLT | Δ |
|---|---:|---:|---:|---:|
| `skillsbench` | 77 | 16 (20.8%) | 0 (0%) | −20.8 pp |
| `swebench-verified` | 500 | 119 (23.8%) | 1 (0.2%)¹ | −23.6 pp |
| `terminal-bench-2` | 89 | 82 (92.1%) | 0 (0%) | −92.1 pp |
| **total** | **666** | **217 (32.6%)** | **1 (0.15%)** | **−32.4 pp** |

¹ The single 0.2.2 exploit is `swebench-verified/django__django-7530`, which scores reward = 1.0
on **both** versions because its `FAIL_TO_PASS` test passes at baseline without any patch. This
is a SWE-bench task-definition quirk, not a 0.2.2 bypass. **True bypasses (cells where 0.2.0 =
BLOCK and 0.2.2 = EXPLT): 0.**

Our 0.2.0 numbers are below benchjack's reported ~100% because our exploit shape is deliberately
a strict subset of theirs (single-framework hook per benchmark, not multi-framework). For the
hardening claim that doesn't matter — whether the attacker is naive or sophisticated, 0.2.2
reduces them to zero via the same defenses. Reproducing benchjack's full multi-framework exploit
shape directly is tracked as the `benchjack-scan` follow-up lab.

## Scope

Tested end-to-end on real tasks: **P1 conftest-hook**, **P7 path-trojan / pth-injection**.
Addressed architecturally: **P2 answer-lookup** via `chmod 700 /solution /tests` (demonstrated
synthetically in `labs/benchjack-sandbox-hardening/pattern2_answer_lookup/`). Out of scope per
the threat model in `harden-sandbox.md`:

- **P3 `eval()` on untrusted input** — benchflow doesn't `eval` agent input.
- **P4 LLM judge injection** — no in-tree LLM judge; benchmark author's responsibility.
- **P5 weak string matching** — benchflow uses reward-based scoring, not substring match.
- **P6 trivial verifier** — handled via the `verifier_error` field architecturally.

## What's next

- **`benchjack-scan` lab.** Multi-framework exploit per benchmark (pytest hook + unittest
  monkey-patch + project-specific runners) to push the 0.2.0 baseline toward benchjack's ~100%
  directly. Proves the 0.2.2 defenses hold against the strongest single-pattern attacker, not
  just the naive one.
- **Real-LLM attacker mode.** Swap `agent="oracle"` for `agent="claude-agent-acp"` in the lab
  and measure whether a capable frontier model discovers the exploits unprompted — the actual
  BenchJack threat model. Reproduces Meerkat's trace-detection findings against benchflow
  specifically.
- **benchflow 0.3 — verifier UID isolation.** Run the verifier under a distinct non-root UID
  from `sandbox_user`, so that agent-controlled code (including `setup.py` at install time)
  runs under a user that cannot write to `/logs/verifier/` at all. Closes the install-phase
  root-execution residual.

## Credits

- The runtime threat model and tier 1–7 exploit classification come from BenchJack (Hao Wang,
  Qiuyang Mang, Alvin Cheung, Koushik Sen, Dawn Song; Berkeley RDI).
- The trace-level validation of in-the-wild exploits comes from Meerkat (BrachioLab / Penn).
- Hardening design + tier 1–4 implementation: @kywch.
- Reward-hack matrix lab, sweep infrastructure, validation runs: @xdotli.

## Links

- Release tag: `v0.2.2` (on branch `feat/sandbox-f1-f6`, pending merge to `main`)
- Lab + full per-cell results: [`labs/reward-hack-matrix/`](../labs/reward-hack-matrix/) — see
  [`sweep_0.2.0_vs_0.2.2.json`](../labs/reward-hack-matrix/sweep_0.2.0_vs_0.2.2.json) for the raw
  1332-trial data
- Hardening audit: [`.dev-docs/harden-sandbox.md`](./harden-sandbox.md)
- Related prior lab on synthetic tasks: [`labs/benchjack-sandbox-hardening/`](../labs/benchjack-sandbox-hardening/)
