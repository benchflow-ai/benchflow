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

## Layout

```
labs/benchjack-sandbox-hardening/
├── README.md                       ← you are here
├── PATTERNS.md                     ← canonical pattern registry (id, defense, status)
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
This one patches `sys.exit` so pytest always exits `0`:

```python
import sys; _real_exit = sys.exit; sys.exit = lambda c=0: _real_exit(0)
```

When `test.sh` runs `python -m pytest`, the hook fires at Python startup,
patches `sys.exit`, then pytest runs (tests fail as normal), but the patched
`sys.exit(1)` becomes `sys.exit(0)`. `test.sh` sees exit code 0 and writes
`reward.txt = 1.0`.

Note: a naive version that writes `reward.txt = 1.0` directly in the `.pth`
hook loses the race — test.sh overwrites it with `0.0` after pytest finishes.
Patching `sys.exit` avoids the race entirely.

## Why 0.2.1 blocks each attack

| pattern | primary defense | backup defenses |
|---|---|---|
| P1 conftest-hook | `/tests` locked `chmod 700 root:root` → write denied | pytest `--confcutdir=/tests`; pre-verify `find … -name conftest.py -delete` |
| P2 answer-lookup | `/solution` locked `chmod 700 root:root` → read denied | oracle runs as `sandbox_user` (non-root) |
| P7 pth-injection | `CLEANUP_CMD` removes writable `.pth` files before verifier | verifier env reset (`PYTHONSAFEPATH`, `PATH`) |

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

1. Add an entry to `PATTERNS.md`.
2. Create `pattern<N>_<name>/` with `task.toml`, `instruction.md`,
   `environment/Dockerfile`, `solution/solve.sh`, and `tests/`.
3. Add a `(id, name, path)` tuple to the `PATTERNS` list in `run_comparison.py`.
4. Run `run_comparison.py --clean` to verify the expected EXPLOITED / BLOCKED
   outcome appears in the table.
