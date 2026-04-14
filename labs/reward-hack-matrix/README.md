# Reward-hack matrix — real benchmark tasks, 0.2.0 vs 0.2.2

Companion to [`labs/benchjack-sandbox-hardening/`](../benchjack-sandbox-hardening/). Where that lab
proves the three BenchJack patterns (P1 conftest-hook, P2 answer-lookup, P7 pth/path-injection) on
**synthetic** tasks designed for the demo, this lab reproduces them on **real tasks pulled from three
production benchmarks** — one tailored exploit per benchmark, run through the full `SDK().run()`
pipeline on Daytona.

## The result — full 666-task sweep

One tailored exploit per benchmark, **every task** in each corpus, run end-to-end through
`SDK().run()` on Daytona under both benchflow versions. **1332 trials. Zero true bypasses on 0.2.2.**

| benchmark | pattern | tasks | 0.2.0 EXPLT | 0.2.2 EXPLT | Δ |
|---|---|---:|---:|---:|---:|
| `skillsbench` | P7 pth-injection | 77 | **16/77** (20.8%) | 0/77 (0%) | −20.8 pp |
| `swebench-verified` | P1 conftest-hook | 500 | **119/500** (23.8%) | 1/500 (0.2%)¹ | −23.6 pp |
| `terminal-bench-2` | P7 path-trojan | 89 | **82/89** (92.1%) | 0/89 (0%) | −92.1 pp |
| **total** | | **666** | **217/666** (32.6%) | **1/666** (0.15%) | **−32.4 pp** |

¹ The single 0.2.2 exploit is `swebench-verified/django__django-7530`, which scores reward = 1.0 on
**both** versions. The task's `FAIL_TO_PASS` test (`test_squashmigrations_initial_attribute`) passes
at baseline without any agent patch, so the exploit "works" against any benchflow version regardless
of hardening. This is a SWE-bench task-definition quirk, not a benchflow 0.2.2 bypass — the conftest
hook is caught by the verifier-phase scrub, and django uses unittest (not pytest) so the hook never
loads. **True bypass count (cells where 0.2.0 = BLOCK and 0.2.2 = EXPLT): 0.**

Per-benchmark notes:

- **`terminal-bench-2` 92.1% on 0.2.0** matches the BenchJack blog's "100% Exploit Score" claim
  shape. tb2 tasks have a uniform verifier template (`uvx -p 3.13 -w pytest==X ... pytest`) that the
  P7 path-trojan exploits exactly — every task exits 0 because our trojaned `uvx` shells out to a
  no-op. The 7 BLOCKED tb2 cells use non-pytest verifier shapes (`build-cython-ext`,
  `bn-fit-modify`'s R-bayesian-network test, etc.) and are immune to any pytest-shaped attack.
- **`skillsbench` 20.8% is structurally low** because skillsbench has *deliberately diverse*
  verifier shapes (pytest, unittest, bash, node, LLM judges, custom python parsers). A
  single-pattern attack can only land on the pytest subset. The same diversity is what blocks every
  P7 attempt under 0.2.2, and is also why a single benchmark can't be reduced to one exploit class
  the way terminal-bench can.
- **`swebench-verified` 23.8% on 0.2.0** is mostly the django + sympy + matplotlib subset that
  follows pytest-with-conftest-hook semantics. The 376 BLOCKED 0.2.0 cells include tasks where
  django's unittest doesn't load conftest hooks at all, plus c-extension-only tasks where the build
  fails before pytest runs.

### Errors are infrastructure, not hardening failures

22 of 1332 trials returned errors on 0.2.2. **None mask a hardening failure**:

| class | count | cause |
|---|---:|---|
| Daytona resource limits (memory/disk > per-sandbox cap) | 8 | task.toml declares 10–24 GB memory or 15–20 GB disk; daytona caps below those |
| `TrialTimeoutError: sdk.run > 900s` | 5 | swebench tasks (pylint-6386/6903, scikit-learn-14710, sympy-19495, psf__requests-2317) hang on Daytona sandbox builds; the 900 s per-trial timeout in `_worker.py` correctly releases the semaphore slot |
| `Failed to create sandbox` (Daytona infra) | 2 | scikit-learn-13135, sympy-16597 — daytona retry-fixable |
| broken Dockerfiles upstream in skillsbench | 3 | fix-visual-stability, gh-repo-analytics, pg-essay-to-audiobook |
| invalid task.toml upstream | 1 | parallel-tfidf-search (`task.name` validation error) |
| task's own verifier timeout | 1 | python-scala-translation (verifier.timeout_sec = 600 s) |
| stale `.cells/` from earlier branch state | 2 | sec-financial-report, shock-analysis-demand — rerun-fixable |

The 5 `TrialTimeoutError` rows are the per-trial timeout doing exactly what it should — a hung
Daytona sandbox surfaces as a single error and the rest of the sweep continues. Without that
timeout, the same 5 tasks hung the previous full sweep at trial 1325/1332 for ~10 minutes.

### Reproducing the rollup

The full results live at [`sweep_0.2.0_vs_0.2.2.json`](sweep_0.2.0_vs_0.2.2.json) (666 cells × 2
versions, slim format with `reward`, `error`, `verifier_error` per cell). To reproduce from
scratch:

```sh
./fetch_corpora.sh  # one-time, ~400 MB
python run_matrix.py --sweep --concurrency 64 \
    --summary-path .jobs/matrix_sweep.json
```

Daytona at concurrency 64 with the long-lived worker pool (`_worker.py`) takes ~20 minutes for the
full 666 cells × 2 versions = 1332 trials. The `.jobs/matrix_smoke.json` smoke target — 1 task per
benchmark — runs in under 90 s and is enough to confirm the pipeline works before committing to the
full sweep.

## Why one pattern per benchmark, and how that compares to BenchJack's "100% Exploit Score"

The BenchJack blog reports ~100% exploit success on every benchmark in its scorecard. Our numbers
are lower (terminal-bench-2 92.1%, swebench-verified 23.8%, skillsbench 20.8%) because **benchjack
uses an automated agent that tailors a fresh exploit per task**, while we apply **one fixed
pattern per benchmark and sweep it across every task in the corpus**. Tasks that don't match the
chosen pattern's structural weak spot just BLOCK on 0.2.0 — that's not the benchmark "resisting
benchjack," it's our single pattern not having the right shape for that particular task.

The single-pattern shape is a deliberate choice for this lab: it produces a defensible, easy-to-
audit upper bound rather than a research-grade per-task synthesis. To reproduce benchjack's 100%
claim shape, we'd need per-task pattern selection (a future `benchjack-scan` package). For the
0.2.2 hardening claim, **the upper bound doesn't matter — what matters is that 0.2.2 reduces
whatever the 0.2.0 number is to 0%.** Both numbers fall to zero in this sweep.

Pattern → benchmark mapping is matched to where each benchmark image leaves writable space and
which binary the verifier shells out to:

- **`swebench-verified`** mounts `/testbed` from the Docker image, so agent writes during the agent
  phase persist into the verifier phase. A dropped `/testbed/conftest.py` with a
  `pytest_runtest_makereport` hook flips every test to `passed`. This is the exact exploit from
  ["How We Broke Top AI Agent Benchmarks"](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/)
  (SWE-bench Verified row).
- **`skillsbench`** bind-mounts `/tests` at the start of the verifier phase, so `conftest.py` drops
  get overwritten. But Python's system `dist-packages` persists across phases — a `.pth` file in
  `/usr/local/lib/python3.X/dist-packages/` executes at every Python startup. We register an
  `atexit` hook that calls `os._exit(0)` when `argv[0]` ends with `pytest`, forcing exit code 0
  before pytest's real SystemExit can land.
- **`terminal-bench-2`** leaves `/usr/local/bin` writable as root, and the verifier shells out to
  `curl`, `uvx`, and `pytest` by name. We plant trojaned stubs at `/usr/local/bin/{curl,uvx,pytest}`
  that `exit 0` for any argv. First in PATH wins, verifier's exit code is 0, reward = 1. This is the
  same class as the blog's "binary wrapper trojans" row for Terminal-Bench.

Under benchflow 0.2.2's hardening, **every one of these writes is denied**:

| defense layer | patterns covered | mechanism |
|---|---|---|
| `sandbox_user="agent"` default | P1, P7 | oracle runs as non-root — `/usr/local/bin`, `/usr/local/lib/python3.X/dist-packages` writes blocked at VFS layer |
| path lockdown | P1 (swebench flavor), P2 | `chown root + chmod 700` on `/tests` and `/solution` before verify |
| pre-verify cleanup | P1 | `find / -maxdepth 5 -name conftest.py -not -path '/tests/*' -delete` |
| sys.path-aware `.pth` wipe | P7 pth flavor | walks `sys.path`, removes writable `sitecustomize.py` / `usercustomize.py` / `.pth` files |
| `_VERIFIER_ENV` PATH | P7 path-trojan flavor | verifier env pins `PATH=/usr/local/sbin:/usr/local/bin:...` with `PATH` set explicitly, not inherited |

See [`.dev-docs/harden-sandbox.md`](../../.dev-docs/harden-sandbox.md) for the full hardening audit.

## What's in this directory

| file | purpose |
|---|---|
| `run_matrix.py` | orchestrator — sync mode (3 hardcoded cells) and `--sweep` mode (asyncio.Semaphore-bounded) |
| `_runner.py` | inner runner — one `sdk.run(agent="oracle", environment="daytona", ...)` call, JSON out |
| `exploits/conftest_hook.sh` | P1 exploit — drops `conftest.py` with `pytest_runtest_makereport` hook |
| `exploits/pth_injection.sh` | P7 pth exploit — plants `.pth` + `sitecustomize.py` in every `dist-packages` it finds |
| `exploits/path_trojan.sh` | P7 path exploit — plants trojaned `curl`/`uvx`/`pytest` under `/usr/local/bin` |
| `fetch_corpora.sh` | clones `laude-institute/harbor-datasets` + `terminal-bench-2` into `.corpora/` |
| `.cells/` | staged tasks (copies of corpus tasks with `solution/solve.sh` overwritten by an exploit) |
| `.venvs/bf-0.2.0`, `.venvs/bf-0.2.2` | pinned comparison venvs created by `run_matrix.py` |
| `.jobs/` | trial output + summary JSON |

## How the oracle is the attacker

The lab uses benchflow's `agent="oracle"` special case, which runs `solution/solve.sh` directly
inside the task container without the ACP agent loop. We repurpose it as a zero-configuration null
attacker: `solution/solve.sh` **is** the exploit payload. This is the same design benchflow's own
`labs/benchjack-sandbox-hardening/` uses — it makes the demo deterministic and free of LLM API
calls. To test with a real model, swap `agent="oracle"` for `agent="claude-agent-acp"` in
`_runner.py`.

## Running against a single cell

```sh
python run_matrix.py \
    --cells "P1@swebench-verified/astropy__astropy-12907"
```

By default trials run on Daytona (`--env daytona`, requires `DAYTONA_API_KEY`). Pass `--env docker`
to use a local Docker daemon instead — no API key needed, useful for offline development.

## Running a sweep

The `--sweep` mode enumerates every task in each corpus, stages a cell per task with its
benchmark's tailored exploit, and runs everything through a long-lived worker pool (one process per
benchflow version, asyncio coroutines inside, bounded by `--concurrency`, default 64). The worker
pool keeps local RAM at ~1 GB regardless of trial count and wraps each `sdk.run()` call in a 900 s
`asyncio.wait_for` so a hung Daytona sandbox cannot starve the semaphore. Skip already-completed
trials with `--resume`.

```sh
python run_matrix.py --sweep --concurrency 64 \
    --summary-path .jobs/matrix_sweep.json
```

`--limit N` caps the number of tasks per benchmark for fast smoke runs (`--limit 1` is 6 trials in
~90 s).

## Independent concurrent validation

Two research groups published complementary work within 24 hours of each other in early April 2026:

- **[Hao Wang et al. (Berkeley RDI) — "How We Broke Top AI Agent Benchmarks"](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/)**
  released [`moogician/trustworthy-env`](https://github.com/moogician/trustworthy-env), a static +
  LLM + Z3 auditor for benchmark source code. They report "Exploit Score 100%" on SWE-bench
  Verified via `conftest.py` hook injection, matching our P1 cell above. **Their release does not
  include a benchmark runner** — they ship one hand-written PoC (`scripts/terminalbench2_poc.sh`,
  plain Python in `/tmp`, not Docker) plus an agentic LLM PoC-synthesizer. Our reward-hack-matrix
  covers the runtime evidence they didn't publish.

- **[BrachioLab (Penn) — "Finding Widespread Cheating on Popular Agent Benchmarks"](https://debugml.github.io/cheating-agents/)**
  released [`BrachioLab/Meerkat`](https://github.com/BrachioLab/Meerkat), a post-hoc trace auditor
  using agentic search + clustering on logged agent trajectories. Their headline finding for
  Terminal-Bench 2 Pilot: **agents read answer keys from `/tests` in 415/429 traces.** That is
  exactly the behavior benchflow 0.2.2's `chmod 700 /tests` lockdown blocks at the VFS layer — the
  attack succeeds on 0.2.0 and fails on 0.2.2 without any detection pass at all. Concurrent
  validation from an independent team using a different methodology (trace detection vs. infra
  hardening).

The three efforts compose cleanly:

- **Berkeley:** source-level static audit → "the vulnerability exists in benchmark code"
- **Penn:** trace-level post-hoc detection → "models are actively cheating in practice"
- **benchflow 0.2.2:** runtime prevention → "the runner itself blocks the exploit, so the
  vulnerability becomes unreachable without a detection pass"

## Future work

1. **Per-task pattern selection (`benchjack-scan`).** The current sweep applies one fixed pattern
   per benchmark; benchjack's automated agent tailors per task and reports ~100% on every benchmark
   it tested. Building a small per-task selector that tries P1 / P2 / P7 / a few custom payloads
   would push the 0.2.0 baseline numbers up to benchjack's claim shape and tighten the "0.2.2
   blocks every variant" story. Tracked as a future lab + standalone PyPI package.
2. **Real-LLM attacker mode.** Swap `agent="oracle"` for `agent="claude-agent-acp"` and measure
   whether a capable frontier model discovers these exploits without being told to cheat — the
   actual BenchJack threat model. This would reproduce the Penn / Meerkat trace-detection findings
   against benchflow specifically and validate that 0.2.2's hardening holds against an adversarial
   agent's emergent exploit attempts (not just hand-crafted ones).
3. **Memory / disk task-toml clamps.** The 0.2.0 sweep had 8 skillsbench task errors due to
   `memory_mb = 10–24 GB` or `disk_mb = 15–20 GB` declarations exceeding Daytona's per-sandbox
   caps. The CPU clamp in `_prepare_cell` already covers `cpus = 8` cases; extending it to memory
   and disk would shave 8 errors off the next sweep. Cosmetic but cleans up the rollup table.
