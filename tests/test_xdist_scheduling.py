"""`pytest -n auto` runs each deterministic job once and starts long tests first.

A session fixture runs once per xdist worker. The deterministic tier's jobs
are session fixtures that several tests read, so under xdist's default
scheduling one job ran on several workers (minutes each, all on the same
Docker), and the tier's tests, contiguous in collection order, queued on one
or two workers. tests/conftest.py groups tests by the job they share and
moves deterministic tests to the front; pyproject's addopts set
``--dist loadgroup`` so the groups are kept. This runs a made-up module of
that shape under two workers with the repository's conftest and pytest
settings, and counts how often each job ran.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

ROOT = Path(__file__).resolve().parents[1]

MODULE = """
import os
from pathlib import Path

import pytest

LOG = Path(os.environ["SCHED_LOG"])


def _log(line):
    with LOG.open("a") as fh:
        fh.write(f"{os.environ.get('PYTEST_XDIST_WORKER', 'main')} {line}\\n")


@pytest.fixture(scope="session")
def shared_root(tmp_path_factory):
    return tmp_path_factory.mktemp("shared")


@pytest.fixture(scope="session")
def job_a(shared_root):
    _log("job job_a")
    return "a"


@pytest.fixture(scope="session")
def job_b(shared_root):
    _log("job job_b")
    return "b"


@pytest.fixture(scope="session")
def job_c(shared_root):
    _log("job job_c")
    return "c"


def _ran(name):
    _log(f"test {name}")

"""

# Jobs a and b are read together by one test, so they must share a worker;
# job c is on its own. The deterministic test is last in the file.
TESTS = [
    ("test_a1", "job_a"),
    ("test_b1", "job_b"),
    ("test_a2", "job_a"),
    ("test_b2", "job_b"),
    ("test_a3", "job_a"),
    ("test_ab", "job_a, job_b"),
    ("test_c1", "job_c"),
    ("test_c2", "job_c"),
    ("test_c3", "job_c"),
]


def _module_text() -> str:
    body = [MODULE]
    for name, fixtures in TESTS:
        body.append(f"def {name}({fixtures}):\n    _ran({name!r})\n\n")
    for i in range(12):
        body.append(f"def test_plain_{i:02d}():\n    _ran('test_plain_{i:02d}')\n\n")
    body.append(
        "@pytest.mark.deterministic\n"
        "def test_long_last(shared_root):\n    _ran('test_long_last')\n"
    )
    return "".join(body)


def _run(pytester: pytest.Pytester, *args: str) -> list[tuple[str, str, str]]:
    log = pytester.path / "sched.log"
    pytester.makepyfile(test_sched=_module_text())
    # The repository's conftest, as a plugin, and its pytest settings
    # (addopts carry --dist loadgroup and the marker set).
    pytester.makeini(
        "[pytest]\n"
        "addopts = -p xdist.plugin --dist loadgroup -p tests.conftest -p no:cacheprovider\n"
        "markers =\n    deterministic: test\n"
    )
    env = {
        # Guards 9b0320f: pytester's isolated subprocess must load the plugin
        # whose scheduler it tests, independently of entry-point autoload.
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "SCHED_LOG": str(log),
        "PYTHONPATH": os.pathsep.join(
            p for p in (str(ROOT), os.environ.get("PYTHONPATH", "")) if p
        ),
    }
    with pytest.MonkeyPatch.context() as mp:
        for key, value in env.items():
            mp.setenv(key, value)
        result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(passed=len(TESTS) + 13)
    events = []
    for line in log.read_text().splitlines():
        worker, kind, name = line.split(" ", 2)
        events.append((worker, kind, name))
    return events


def test_each_session_job_runs_once_under_two_workers(
    pytester: pytest.Pytester,
) -> None:
    events = _run(pytester, "-n", "2")
    jobs = [(worker, name) for worker, kind, name in events if kind == "job"]
    assert sorted(name for _, name in jobs) == ["job_a", "job_b", "job_c"], jobs
    workers = {name: worker for worker, name in jobs}
    assert workers["job_a"] == workers["job_b"], jobs
    tests_on = {name: w for w, kind, name in events if kind == "test"}
    for name, fixtures in TESTS:
        for fixture in fixtures.split(", "):
            assert tests_on[name] == workers[fixture], (name, events)


def test_deterministic_tests_start_first_under_xdist(
    pytester: pytest.Pytester,
) -> None:
    events = _run(pytester, "-n", "2")
    # xdist hands out the groups first (largest first), then the other tests
    # one at a time, and a worker runs what it is given in order. So the
    # deterministic test, last in the file, was handed out before every quick
    # test: none runs before it on its worker.
    (worker,) = {w for w, kind, n in events if kind == "test" and n == "test_long_last"}
    on_worker = [n for w, kind, n in events if kind == "test" and w == worker]
    plain = [n for n in on_worker if n.startswith("test_plain_")]
    assert plain, events
    assert on_worker.index("test_long_last") < on_worker.index(plain[0]), events


def test_a_serial_run_keeps_the_file_order(pytester: pytest.Pytester) -> None:
    events = _run(pytester)
    order = [name for _, kind, name in events if kind == "test"]
    expected = [name for name, _ in TESTS]
    expected += [f"test_plain_{i:02d}" for i in range(12)]
    assert order == [*expected, "test_long_last"]
