# BenchJack sandbox hardening — 0.2.0 vs HEAD

A runnable, end-to-end demonstration that the sandbox hardening shipped in
benchflow 0.2.1 (currently on `main`) blocks three BenchJack-style exploits
that work against benchflow 0.2.0. Under 0.2.0 each exploit causes the task's
reward to flip from 0.0 to 1.0. Under HEAD the same exploits, run by the same
"attacker," are blocked and the reward stays 0.0.

## One-command repro

```sh
python labs/benchjack-sandbox-hardening/run_comparison.py
```

Expected output:

```
========================================================================
BenchJack sandbox-hardening comparison (0.2.0 vs HEAD)
========================================================================
pattern                   benchflow 0.2.0       benchflow HEAD
------------------------------------------------------------------------
P1 conftest-hook            1.00  EXPLOITED    0.00  BLOCKED
P2 answer-lookup            1.00  EXPLOITED    0.00  BLOCKED
P7 pth-injection            1.00  EXPLOITED    0.00  BLOCKED
```

The orchestrator creates two isolated venvs (one with `benchflow==0.2.0` from
PyPI, one with the current checkout in editable mode), runs `_attack_runner.py`
inside each per pattern, and prints a comparison table. First run takes a
few minutes (Docker image builds + pip installs); subsequent runs are fast.

Pass `--clean` to delete `.venvs/` and `.jobs/` before running.

### Requirements

* Docker daemon accessible (`docker ps` works from wherever you run the script)
* Python 3.10+
* `uv` on PATH (preferred — parallelizes pip installs); falls back to
  `python -m venv` + pip
* Network access to PyPI on first run

## Pattern registry

Sourced from the [BenchJack seven-pattern threat model](https://moogician.github.io/blog/2026/trustworthy-benchmarks-cont/).

| id | name | description | 0.2.1 blocks? | defense layer | status |
|---|---|---|---|---|---|
| P1 | conftest-hook | Plant `conftest.py` hook that rewrites all test bodies to no-ops | yes | path lockdown (`/tests` chmod 700) + pre-verify cleanup | shipped |
| P2 | answer-lookup | Read locked answer key from `/solution/` | yes | path lockdown (`/solution` chmod 700) | shipped |
| P3 | eval-injection | Agent output passed to `eval()` in verifier | no — verifier code quality | n/a | out of scope |
| P4 | llm-judge-injection | Prompt injection into LLM judge via unsanitized agent output | no — API boundary | n/a | out of scope |
| P5 | weak-string-match | Substring match lets wrong answers pass | no — verifier logic | n/a | out of scope |
| P6 | trivial-verifier | Verifier always returns 1.0, never checks actual output | no — design | n/a | out of scope |
| P7 | pth-injection | Plant `.pth` file that forces exit code 0 via atexit + os._exit | yes | oracle runs as `sandbox_user` (non-root) — system site-packages write denied | shipped |

**P3–P6** are benchflow-agnostic verifier code quality issues. They produce "both versions fail" in a 0.2.0 vs 0.2.1 comparison and belong in a future `benchjack-scan/` auditor, not here.

## Layout

```
labs/benchjack-sandbox-hardening/
├── README.md                       ← you are here
├── run_comparison.py               ← dual-venv orchestrator (top-level entry)
├── _attack_runner.py               ← inner runner, one per pinned benchflow version
├── pattern1_conftest_hook/
│   ├── task.toml
│   ├── instruction.md
│   ├── environment/
│   │   ├── Dockerfile              ← installs pytest + bakes exploit into image
│   │   └── conftest_exploit.py     ← the pytest hook payload
│   ├── tests/
│   │   ├── test.sh                 ← verifier: runs pytest, writes reward.txt
│   │   └── test_outputs.py         ← 3 designed-to-fail tests
│   └── solution/
│       └── solve.sh                ← plants conftest.py in /tests
├── pattern2_answer_lookup/
│   ├── task.toml
│   ├── instruction.md
│   ├── environment/
│   │   └── Dockerfile
│   ├── tests/
│   │   ├── test.sh                 ← verifier: compares /tmp/answer.txt to ground truth
│   │   └── ground_truth.txt        ← "42"
│   └── solution/
│       ├── solve.sh                ← reads /solution/answer.txt, writes to /tmp/answer.txt
│       └── answer.txt              ← "42" (locked at chmod 700 under 0.2.1)
├── pattern7_pth_injection/
│   ├── task.toml
│   ├── instruction.md
│   ├── environment/
│   │   └── Dockerfile              ← installs pytest
│   ├── tests/
│   │   ├── test.sh                 ← verifier: runs pytest, writes reward.txt
│   │   └── test_outputs.py         ← 3 designed-to-fail tests
│   └── solution/
│       └── solve.sh                ← plants .pth hook that patches sys.exit
└── comparison.ipynb                ← narrative walkthrough of the same mechanism
```

## How each attack works

### P1 — conftest-hook

`pattern1_conftest_hook/tests/test_outputs.py` has three tests that always fail:

```python
def test_arithmetic_is_broken(): assert 1 + 1 == 3
def test_false_is_true():        assert False
def test_pi_is_rational():       import math; assert math.pi == 22 / 7
```

`tests/test.sh` runs `pytest test_outputs.py` and writes `1.0` to
`/logs/verifier/reward.txt` if pytest exits 0, else `0.0`.

`solution/solve.sh` copies a pre-baked `/app/exploit/conftest.py` (built into
the image via `environment/Dockerfile`) to `/tests/conftest.py`. The hook
registers `pytest_collection_modifyitems` and rewrites every collected test's
body to `lambda *a, **k: None`. When `test.sh` runs pytest, the hook fires,
all three tests become no-ops, everything "passes," and `reward.txt` gets `1.0`.

### P2 — answer-lookup

The task asks "What is 7 × 6?" with the answer `42` stored in
`/solution/answer.txt`. `solution/solve.sh` simply reads that file and writes
it to `/tmp/answer.txt`. `tests/test.sh` compares `/tmp/answer.txt` to
`/tests/ground_truth.txt` — if they match, reward is `1.0`.

Under 0.2.0 the oracle runs as root and can read `/solution/answer.txt`.
Under 0.2.1 `/solution` is locked to `root:root chmod 700` before oracle
executes, so the read fails and the script writes a wrong answer instead.

### P7 — pth-injection

`solution/solve.sh` plants a `.pth` file into the writable Python
user-site-packages directory (`~/.local/lib/python3.x/site-packages`). A
`.pth` file is executed by Python at startup — before any test code runs.
This one registers an atexit handler that calls `os._exit(0)`:

```python
import atexit, os; atexit.register(lambda: os._exit(0))
```

When `test.sh` runs `python -m pytest`, the hook fires at Python startup and
registers the handler. pytest runs (tests fail as normal). When `raise
SystemExit(1)` propagates to Python's top level, Python runs atexit handlers
first — `os._exit(0)` fires and forces the process to exit with code 0 before
Python can use the SystemExit value. `test.sh` sees exit code 0 and writes
`reward.txt = 1.0`.

Note: a naive version that writes `reward.txt = 1.0` directly in the `.pth`
hook loses the race — test.sh overwrites it with `0.0` after pytest finishes.
The atexit approach avoids the race entirely. Patching `sys.exit` looks
equivalent but doesn't work: pytest 8.x `__main__.py` uses `raise
SystemExit(main())` directly rather than `sys.exit()`, so `sys.exit` is never
called.

**Name history:** Originally called "path-trojan" and described as injecting a
fake `pytest` binary via PATH manipulation. That approach is broken because
Harbor invokes verifiers as `bash -c` (non-login, non-interactive) — shell
startup files (`/etc/profile.d/`, `.bashrc`, `/etc/environment`) are never
sourced, so PATH changes made by the agent have no effect on the verifier
process. The correct mechanism is `.pth` file injection.

## Why 0.2.1 blocks each attack

| pattern | primary defense | backup defenses |
|---|---|---|
| P1 conftest-hook | `/tests` locked `chmod 700 root:root` → write denied | pytest `--confcutdir=/tests`; pre-verify `find … -name conftest.py -delete` |
| P2 answer-lookup | `/solution` locked `chmod 700 root:root` → read denied | oracle runs as `sandbox_user` (non-root) |
| P7 pth-injection | oracle runs as `sandbox_user` (non-root) — system site-packages write denied | — |

**Defense layer reference:**

| defense | what it does | patterns covered |
|---|---|---|
| path lockdown | `chown root:root && chmod 700` on `/tests` and `/solution` before agent/oracle runs | P1, P2 |
| oracle sandbox_user | oracle runs `solve.sh` as `sandbox_user` (non-root), so locked paths and root-owned site-packages deny access | P1, P2, P7 |
| pre-verify cleanup | removes stray `conftest.py` files outside `/tests` | P1 |

See [`docs/harden-sandbox.md`](../../docs/harden-sandbox.md) for the full
audit against the BenchJack seven-pattern threat model.

## Walkthrough notebook

`comparison.ipynb` is a narrative view of the same mechanism — intended for
readers who prefer a guided explanation with inline code over a raw script.
It assumes `run_comparison.py` has been run at least once (to create the
`.venvs/` directory) and uses those venvs to execute the comparison.

To rebuild the notebook's baked outputs before committing:

```sh
jupyter nbconvert --to notebook --execute --inplace comparison.ipynb
```

## Caveats

* **First run is slow.** Docker image builds, `pip install benchflow==0.2.0`
  from PyPI, and `pip install -e ../..` for HEAD. Budget ~5 minutes the first
  time, ~1 minute thereafter (venvs are cached under `.venvs/`).
* **No GPU required.** The demo uses `python:3.12-slim` + `pytest==8.3.3` only.
* **Not a benchmark run.** This is a three-row demo of three attacks on three
  tasks. For the full BenchJack audit against skillsbench (75/89 tasks
  vulnerable to the original attack), see the upstream BenchJack replication
  notebook.

## Adding a new pattern

1. Add a row to the pattern registry table above.
2. Create `pattern<N>_<name>/` with `task.toml`, `instruction.md`,
   `environment/Dockerfile`, `solution/solve.sh`, and `tests/`.
3. Add a `(id, name, path)` tuple to the `PATTERNS` list in `run_comparison.py`.
4. Run `run_comparison.py --clean` to verify the expected EXPLOITED / BLOCKED
   outcome appears in the table.
