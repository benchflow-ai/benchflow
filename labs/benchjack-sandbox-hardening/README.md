# BenchJack sandbox hardening — 0.2.0 vs HEAD

A runnable, end-to-end demonstration that the sandbox hardening shipped in
benchflow 0.2.1 (currently on `main`) blocks the BenchJack-style `conftest.py`
exploit that works against benchflow 0.2.0. Under 0.2.0 the exploit rewrites
three failing tests to no-ops via a planted pytest hook and the task's reward
flips from 0.0 to 1.0. Under HEAD the same exploit, run by the same
"attacker," is blocked at multiple layers and the reward stays 0.0.

## One-command repro

```sh
python labs/benchjack-sandbox-hardening/run_comparison.py
```

Expected output:

```
benchflow 0.2.0       reward=  1.00  EXPLOITED — conftest.py hook fired
benchflow <HEAD>      reward=  0.00  BLOCKED — hardening layer fired
```

The orchestrator creates two isolated venvs (one with `benchflow==0.2.0` from
PyPI, one with the current checkout in editable mode), runs `_attack_runner.py`
inside each, and prints a comparison table. First run takes a couple of
minutes (Docker image build + pip install); subsequent runs are fast.

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
├── README.md               ← you are here
├── run_comparison.py       ← dual-venv orchestrator (top-level entry)
├── _attack_runner.py       ← inner runner, one per pinned benchflow version
├── attack_task/
│   ├── task.toml                   ← benchflow task metadata
│   ├── instruction.md              ← designed-to-fail task description
│   ├── environment/
│   │   ├── Dockerfile              ← installs pytest + bakes exploit into image
│   │   └── conftest_exploit.py     ← the pytest hook payload
│   ├── tests/
│   │   ├── test.sh                 ← verifier: runs pytest, writes reward.txt
│   │   └── test_outputs.py         ← 3 designed-to-fail tests
│   └── solution/
│       └── solve.sh                ← the "attacker": plants conftest.py
└── comparison.ipynb        ← narrative walkthrough of the same mechanism
```

## How the attack works

`attack_task/tests/test_outputs.py` has three tests that always fail:

```python
def test_arithmetic_is_broken(): assert 1 + 1 == 3
def test_false_is_true():        assert False
def test_pi_is_rational():       import math; assert math.pi == 22 / 7
```

`tests/test.sh` is the verifier — it runs `pytest test_outputs.py` and writes
`1.0` to `/logs/verifier/reward.txt` if pytest exits 0, else `0.0`.

The "attacker" is `attack_task/solution/solve.sh`. It runs via benchflow's
built-in `oracle` agent, which executes `solution/solve.sh` inside the task
container instead of going through an ACP agent loop. For this demo we're
intentionally repurposing the oracle runner as a zero-configuration null
attacker: the "solution" it runs IS the exploit.

`solve.sh` copies the pre-baked `/app/exploit/conftest.py` (built into the
image via `environment/Dockerfile`) to `/tests/conftest.py`. The `conftest.py`
registers a `pytest_collection_modifyitems` hook that rewrites every collected
test's function body to `lambda *a, **k: None`. When `test.sh` runs pytest,
the hook fires, all three test bodies become no-ops, everything "passes," and
`reward.txt` gets `1.0`.

## Why 0.2.1 blocks it

Four independent defense layers, any one of which is sufficient:

1. **`/tests` is locked.** Sandbox setup runs `chown root:root /tests && chmod
   700 /tests`, so the agent user `solve.sh` runs as cannot write
   `conftest.py` there.
2. **pytest `--rootdir=/tests --confcutdir=/tests`.** Even if a `conftest.py`
   ends up at `/app/conftest.py` (the `solve.sh` fallback), pytest's
   collection walk is anchored at `/tests` and never visits `/app`.
3. **Pre-verify cleanup.** `find / -maxdepth 5 -name conftest.py -not -path
   '/tests/*' -delete` runs before the verifier, deleting any stray
   `conftest.py` the agent managed to write.
4. **Canonical verifier env.** `PATH`, `PYTHONSAFEPATH`, and related env vars
   are reset so the agent can't shim `pytest` via PATH trojans or Python
   sitecustomize hooks.

See [`docs/harden-sandbox.md`](../../docs/harden-sandbox.md) (added in commit
`19eeefe`) for the full audit against the BenchJack seven-pattern threat
model.

## Walkthrough notebook

`comparison.ipynb` is a narrative view of the same mechanism — eight cells,
intended for readers who prefer a guided explanation with inline code over a
raw script. It assumes `run_comparison.py` has been run at least once (to
create the `.venvs/` directory) and uses those venvs to execute the
comparison.

To rebuild the notebook's baked outputs before committing:

```sh
jupyter nbconvert --to notebook --execute --inplace comparison.ipynb
```

This requires a Jupyter install and the same Docker prerequisites as
`run_comparison.py`.

## Caveats

* **First run is slow.** Docker image build, `pip install benchflow==0.2.0`
  from PyPI, and `pip install -e ../..` for HEAD. Budget ~3 minutes the first
  time, ~30 seconds thereafter (venvs are cached under `.venvs/`).
* **No GPU required.** The demo uses `python:3.12-slim` + `pytest==8.3.3`
  only.
* **Not a benchmark run.** This is a two-row demo of one attack on one task.
  For the full BenchJack audit against skillsbench (75/89 tasks vulnerable to
  the original attack), see the upstream BenchJack replication notebook.

## Extending

To add a second attack pattern (e.g. BenchJack's PATH-trojan P7), create a
sibling directory `attack_task_path_trojan/` with its own `solve.sh` that
plants a fake `pytest` binary on `$PATH`, and add a second row to
`run_comparison.py`. Each attack should live in its own `task.toml`-rooted
directory so the comparison orchestrator can loop over them.
