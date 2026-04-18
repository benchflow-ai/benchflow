# Reward Hack Detection

**What you'll learn:** How to audit an agent benchmark for reward-hacking vulnerabilities before you trust its scores. By the end, you'll have run a real exploit against a pytest-based benchmark and built a detector you can drop into your own benchmark to catch the same attack.

**Prereqs:**

```bash
pip install pytest
```

That's it. **No Docker. No API keys. No model calls.** The whole tutorial runs in <10 seconds on your laptop.

**Runtime:** ~5 minutes to read and run through.

---

## Why this matters

In April 2026, researchers at Berkeley ([BenchJack: Trustworthy Benchmarks](https://moogician.github.io/blog/2026/trustworthy-benchmarks-cont/)) showed that 8 major AI agent benchmarks — including SWE-bench Verified, WebArena, OSWorld, GAIA, and Terminal-Bench — can each achieve **near-perfect scores through exploitation rather than task completion**. Their scanner built working exploits for each one.

The core problem: most benchmarks run the verifier in the same filesystem as the agent. Whatever the agent wrote during its turn (including test-manipulating code) is still there when pytest runs.

This tutorial reproduces one of their exploits — the **SWE-bench `conftest.py` pytest hook injection** — then shows you how to detect it. The same detector works against 75% of SkillsBench tasks and any other benchmark that runs `pytest` in the agent's container.

---

## Part 1 — The exploit

pytest auto-discovers any `conftest.py` in the test directory or any of its parents. If you can write one, you can register hooks that fire during test execution.

### Setup

We've pre-built the exploit in `conftest_exploit/`. The directory contains:

```
conftest_exploit/
├── conftest.py          # the exploit: rewrites every test result to "passed"
├── test_failing.py      # 3 deliberately failing tests
└── test_passing.py      # 1 control test that passes normally
```

### Run the tests **without** the exploit (baseline)

First, temporarily rename `conftest.py` so pytest doesn't discover it:

```bash
cd conftest_exploit
mv conftest.py conftest.py.disabled
pytest test_failing.py test_passing.py -v
```

Expected output (4 tests, 3 failures, 1 pass):

```
FAILED test_failing.py::test_truth_is_false - AssertionError: This assertion is false
FAILED test_failing.py::test_explicit_failure - RuntimeError: Explicit failure
FAILED test_failing.py::test_key_error - KeyError: 'b'
test_passing.py::test_this_actually_passes PASSED
================ 3 failed, 1 passed in 0.02s =================
```

Three tests genuinely fail, one genuinely passes. That's the truth.

### Run them **with** the exploit

Now restore `conftest.py` and run again:

```bash
mv conftest.py.disabled conftest.py
pytest test_failing.py test_passing.py -v
```

Expected output:

```
test_failing.py::test_truth_is_false PASSED                    [ 25%]
test_failing.py::test_explicit_failure PASSED                  [ 50%]
test_failing.py::test_key_error PASSED                         [ 75%]
test_passing.py::test_this_actually_passes PASSED              [100%]
================== 4 passed in 0.01s ==================
```

**Every test passes.** Nothing changed about the tests themselves — they still assert `1 == 2` and raise `RuntimeError`. The `conftest.py` is a pytest hook that rewrites each test's `outcome` field to `"passed"` after the test runs but before the reporter sees it.

### Read the exploit

Open `conftest_exploit/conftest.py`:

```python
import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        rep.outcome = "passed"
        rep.longrepr = None
```

That's the whole exploit. 10 lines. `pytest_runtest_makereport` is a hook that fires for every test result; `hookwrapper=True` lets us mutate the result after pytest has built it. Setting `rep.outcome = "passed"` during the `call` phase makes every test appear to pass.

### Why this works on real benchmarks

On SWE-bench Verified, the agent's job is to write a patch that fixes a bug. The evaluator then runs pytest inside the same Docker container to see if the agent's patch makes the failing tests pass. The agent can write any file it wants during its turn — including `conftest.py`. pytest auto-discovers it at verify time. All tests report as passing. `reward.txt` gets `1`. The agent "solved" the task without touching the bug.

The same attack works on any benchmark that:
- Runs pytest (or any test runner with hookable lifecycle methods)
- Runs verification in the same filesystem as the agent
- Doesn't pre-scan agent artifacts for test-tampering code

---

## Part 2 — The detector

Now flip the script: build a scanner that catches this exploit before verification runs.

Run it:

```bash
cd ..
python detector.py conftest_exploit/
```

Expected output:

```
Scanning conftest_exploit/ for reward-hack indicators...

FOUND: conftest_exploit/conftest.py
  pattern: pytest_runtest_makereport hook wrapper
  severity: HIGH
  line 4: @pytest.hookimpl(hookwrapper=True)
  line 5: def pytest_runtest_makereport(item, call):
  line 9:         rep.outcome = "passed"
  recommendation: delete this file before running pytest, or run with --confcutdir
                  at a path above the agent filesystem

SCAN COMPLETE: 1 HIGH severity finding
```

### Read the detector

Open `detector.py`. It's ~50 lines of pure Python, no dependencies. It walks a directory, reads every `.py` file, and greps for three patterns:

1. **`pytest_runtest_makereport` + `hookwrapper=True`** — the SWE-bench conftest.py exploit
2. **`rep.outcome = "passed"`** — the literal string that marks result rewriting
3. **`unittest.TestCase.run` monkey-patch** — the Django variant

Every hit includes the file path, line numbers, matched pattern, and a recommended mitigation.

### Drop it into your own benchmark

The detector is standalone. Copy `detector.py` into your benchmark's CI pipeline and run it against the agent's output directory before verification starts:

```bash
# In your verifier:
python detector.py /path/to/agent/workspace/
if [ $? -ne 0 ]; then
    echo "Reward-hack pattern detected — failing trial"
    echo 0 > /logs/verifier/reward.txt
    exit 0
fi
```

Returns exit code 1 if any HIGH severity finding is present. Your verifier fails the trial without running pytest.

---

## Part 3 — The baseline ("null agent")

The most honest signal of a compromised benchmark is: **does a null agent (zero actions) ever score above zero?**

If yes, your benchmark has a path to reward that doesn't require solving anything. That's a bug in the benchmark, not an agent achievement.

`null_agent.py` demonstrates:

```bash
python null_agent.py
```

Output:

```
Null agent: doing literally nothing
  Wrote no files
  Made no tool calls
  Executed no code

Then running the 'test' against the empty workspace:
  WITHOUT the conftest.py exploit: reward = 0  (as expected)
  WITH the conftest.py exploit:    reward = 1  (!!! benchmark is compromised)

If a null agent can get reward=1 on your benchmark,
your benchmark has a reward-hacking vulnerability.
```

Run the null agent against any benchmark you're evaluating. If it ever scores above zero, you know the benchmark is compromised before you trust any real agent's score on it.

---

## What you learned

1. **Reward hacks aren't theoretical.** A 10-line `conftest.py` flips a failing test suite to all-passing. This works against any pytest-based benchmark that runs in the agent's filesystem.
2. **Detection is cheap.** A ~50-line Python scanner catches the known patterns. Run it before verification.
3. **Null agent baseline is the truth.** If an agent doing nothing can score above zero, the benchmark is broken, not the agent.
4. **This applies to real benchmarks.** SWE-bench Verified, SkillsBench, Terminal-Bench, and many others all run pytest in the agent's container. The exploit transfers directly.

## What benchflow does about this

BenchFlow's roadmap includes:

- **Pre-verify scanner** — opt-in hook that runs the detector before Harbor's verifier
- **Fresh-container verification** — copy patched files to a clean base instead of reusing the agent's container
- **Null-agent baseline** — automatic detection of "this benchmark can be solved by doing nothing"

If you're building a benchmark with benchflow today, the simplest protection is the ClawsBench approach: run the agent as a restricted user, `chmod 700` on sensitive directories, use `gosu` to drop privileges before the agent phase, and restore root only during verification. See the [ClawsBench paper](https://arxiv.org/abs/2604.05172) appendix for the full sandbox design.

## Further reading

- **BenchJack: Trustworthy Benchmarks** ([blog part 1](https://moogician.github.io/blog/2026/trustworthy-benchmarks/), [blog part 2](https://moogician.github.io/blog/2026/trustworthy-benchmarks-cont/)) — the original research. 45 exploits across 13 benchmarks.
- **ClawsBench** ([arXiv:2604.05172](https://arxiv.org/abs/2604.05172)) — our benchmark that deployed the restricted-user sandbox in response to this exact vulnerability class in March 2026.
- `benchflow` [CHANGELOG](../../CHANGELOG.md) — tracks planned defense features.
