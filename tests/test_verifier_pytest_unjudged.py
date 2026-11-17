"""A pytest that judged nothing ends its trial errored, never a scored 0 (EN-647).

programbench@1.0 scored a silent 0.0 on every trial: its run.sh passes
``--timeout=30``, pytest-timeout is installed but not loaded (BenchFlow runs
test.sh with plugin autoload off and loads only plugins the image has or the
task pins), so pytest stopped with ``unrecognized arguments`` (exit 4) and
the task's own verify.py wrote 0/0 = 0.0. The plugin guard now leaves a
``starting`` marker that only a pytest which read its whole command line
removes, and an ``exited`` marker for a run that ended with exit 3, 4 or 5;
the verifier then ends the trial with no usable reward. Where the solution
may have brought pytest to that status (a conftest that fails to import, an
ini file or conftest the agent left, a test path in the agent's files, tests
built from its code, a test module that skips itself without it), the guard
leaves no marker and the run stays the plain fail test.sh scores (F2b,
2026-11-15).

The task's tests live in their own folder, as /tests does in a sandbox; the
agent's workspace is the guarded tree and pytest's working directory.

The verifier runs real pytest subprocesses with the environment it passes to
test.sh; only the container transport is stubbed, as in
``test_verifier_guard_load_failure.py``.
"""

import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow._utils.scoring import (
    NO_USABLE_REWARD,
    VERIFIER_FAILED,
    classify_score_outcome,
    classify_verifier_error,
    is_no_usable_reward,
    mean_reward_every_trial,
    mean_scored_reward,
)
from benchflow.rollout._setup import _verify_rollout
from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.lockdown import (
    _pytest_plugin_guard_source,
    parse_pytest_plugin_guard_markers,
    pytest_plugin_guard_key,
    pytest_plugin_guard_key_file,
    pytest_plugin_guard_markers_cmd,
    pytest_plugin_guard_markers_dir,
)
from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig
from benchflow.task.verifier_core import _unjudged_pytest
from tests.test_verifier_runtime_r5 import Tree

PASSING = "def test_solution():\n    assert True\n"
FAILING = "def test_solution():\n    assert False\n"
NO_TESTS = "def helper():\n    return 1\n"
MISSING_SOLUTION = (
    "import solution_the_agent_never_wrote\n\n"
    "def test_solution():\n    assert solution_the_agent_never_wrote.ok\n"
)
CRASHING_CONFTEST = (
    "def pytest_sessionstart(session):\n    raise RuntimeError('plugin bug')\n"
)
ADDOPTION_CRASH = (
    "def pytest_addoption(parser):\n    raise RuntimeError('addoption bug')\n"
)
# A broken report hook, patched in by the task's test module: an internal
# error raised outside every conftest and the agent's files.
PYTEST_BUG = (
    "import _pytest.reports\n\n"
    "def _broken(cls, item, call):\n"
    "    raise RuntimeError('report hook bug')\n\n"
    "_pytest.reports.TestReport.from_item_and_call = classmethod(_broken)\n\n"
    "def test_solution():\n    assert True\n"
)
SESSIONFINISH_CRASH = (
    "def pytest_sessionfinish(session):\n    raise RuntimeError('report writer bug')\n"
)


def imports_at_load(workspace):
    """A task conftest that imports the solution when it loads."""
    return f"import sys\nsys.path.insert(0, {str(workspace)!r})\nimport solution\n"


def builds_at_sessionstart(workspace):
    """A task conftest that builds the solution at session start."""
    return (
        "import subprocess\n\n"
        "def pytest_sessionstart(session):\n"
        f"    subprocess.run(['sh', {str(workspace / 'compile.sh')!r}], check=True)\n"
    )


def imports_at_configure(workspace):
    """A task conftest that imports the solution in pytest_configure."""
    return (
        "import sys\n\n"
        "def pytest_configure(config):\n"
        f"    sys.path.insert(0, {str(workspace)!r})\n"
        "    import solution  # noqa: F401\n"
    )


# A test module that skips itself when the solution is missing.
IMPORTORSKIP = (
    "import pytest\n\n"
    "solution = pytest.importorskip('solution_the_agent_never_wrote')\n\n"
    "def test_solution():\n    assert solution.ok\n"
)
CONFTEST_IMPORTS_SOLUTION = "import solution_the_agent_never_wrote\n"
UNLOADED_OPTION_INI = "[pytest]\naddopts = --timeout=30\n"


def built_from(workspace):
    """A test module whose tests come from the solution's own list."""
    return (
        "import sys\n"
        f"sys.path.insert(0, {str(workspace)!r})\n"
        "import solution\n\n"
        "for _i, _case in enumerate(solution.CASES):\n"
        "    globals()[f'test_{_i}'] = lambda case=_case: None\n"
    )


# A passing test whose subject is pytest itself: the pytests it starts end
# with a usage error and with no tests, as the test expects, and so does the
# one it runs in its own process.
SUBJECT = """
import subprocess, sys

def test_pytest_reports_its_own_errors(tmp_path):
    (tmp_path / "test_empty.py").write_text("X = 1\\n")
    flag = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--timeout=30", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert flag.returncode == 4, flag.stdout + flag.stderr
    assert "unrecognized arguments" in flag.stderr
    empty = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert empty.returncode == 5, empty.stdout + empty.stderr
    import pytest

    assert pytest.main(["-q", "-p", "no:cacheprovider", str(tmp_path)]) == 5
"""
# A failing solution that, run by the tests, writes the two new markers
# itself, unsigned, where the guard and the verifier keep them.
FORGING_FAILURE = """
import os, re
from pathlib import Path

GUARD = re.search(r"-p (_[a-z]+_guard_[0-9a-f]+)", os.environ["PYTEST_ADDOPTS"])[1]
VERIFIER = Path(os.environ["FORGE_VERIFIER_DIR"])
MARKERS = Path(os.environ["FORGE_GUARD_PARENT"]) / GUARD / ".markers"


def test_solution():
    for kind, body in (("starting", '""'), ("exited", '"4"')):
        (VERIFIER / f"{GUARD}.1-deadbeef.{kind}").write_text(body)
        (MARKERS / f"{GUARD}.2-deadbeef.{kind}").write_text("0" * 64 + "\\t" + body)
    assert False
"""


@pytest.fixture(autouse=True)
def _guard_parent(tmp_path, monkeypatch):
    """Guards live in the test's tmp_path, as they live at / in the sandbox."""
    monkeypatch.setattr(lockdown, "_PYTEST_PLUGIN_GUARD_PARENT", str(tmp_path))


class PytestSandbox:
    """test.sh as a list of pytest runs; the reward is 1 when the last passes."""

    is_mounted = True

    def __init__(self, paths, workspace, runs, log):
        self._paths = paths
        self._workspace = workspace
        self._runs = runs
        self._log = log

    async def upload_dir(self, *args, **kwargs):
        pass

    async def exec(self, command, env=None, **kwargs):
        if "test-stdout.txt" not in command:
            if "/.markers" in command:
                # The verifier reading the guard's markers back.
                result = subprocess.run(
                    ["/bin/sh", "-c", command], capture_output=True, text=True
                )
                return ExecResult(
                    stdout=result.stdout,
                    stderr=result.stderr,
                    return_code=result.returncode,
                )
            return ExecResult(stdout="", stderr="", return_code=0)
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("PYTEST_", "PYTHON"))
        }
        environment.update(env or {}, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        output = ""
        code = 0
        for args in self._runs:
            if args[:1] == ["--script"]:
                command = [sys.executable, *args[1:]]
            elif args[:1] == ["--timeout"]:
                # test.sh's `timeout N pytest ...`
                command = ["timeout", args[1], sys.executable, "-m", "pytest"]
                command += ["-q", *args[2:]]
            else:
                command = [sys.executable, "-m", "pytest", "-q", *args]
            result = subprocess.run(
                command,
                cwd=self._workspace,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120,
            )
            output += result.stdout
            code = result.returncode
        if self._log is not None:
            (self._paths.verifier_dir / self._log).write_text(output)
        # What programbench's verify.py and a plain `pytest && echo 1` both
        # do with a pytest that never ran a test: write 0.
        self._paths.reward_text_path.write_text("1" if code == 0 else "0")
        return ExecResult(stdout="", stderr="", return_code=0)


def _task(tmp_path: Path, env: dict[str, str]) -> MagicMock:
    tests_dir = tmp_path / "task" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text("#!/bin/sh\n")
    task = MagicMock()
    task.name = "task"
    task.task_dir = tests_dir.parent
    task.paths.task_dir = tests_dir.parent
    task.paths.tests_dir = tests_dir
    task.paths.test_path = tests_dir / "test.sh"
    task.paths.uses_native_verifier_dir = False
    task.config = TaskConfig.model_validate_toml('version = "1.0"\n[verifier]\n')
    task.config.verifier.env = env
    return task


async def _run(
    tmp_path,
    test_source,
    *,
    runs=None,
    conftest=None,
    agent_files=None,
    log="test-stdout.txt",
):
    """Verify one rollout whose test.sh runs *runs* (pytest arguments).

    *test_source* and *conftest* are the task's, in its tests folder;
    *agent_files* (name: source) are what the agent left in its workspace.
    """
    guard = lockdown.PYTEST_PLUGIN_GUARD_PREFIX + uuid.uuid4().hex
    guard_dir = tmp_path / guard
    guard_dir.mkdir()
    Path(pytest_plugin_guard_markers_dir(guard)).mkdir(mode=0o700)
    Path(pytest_plugin_guard_key_file(guard)).write_text(pytest_plugin_guard_key(guard))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name, source in (agent_files or {}).items():
        (workspace / name).write_text(source)
    tests_dir = tmp_path / "verifier_tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_outputs.py"
    test_file.write_text(test_source)
    if conftest is not None:
        (tests_dir / "conftest.py").write_text(conftest)
    (guard_dir / f"{guard}.py").write_text(
        _pytest_plugin_guard_source(guard, (str(workspace),), [])
    )
    task = _task(
        tmp_path,
        {
            "PYTEST_ADDOPTS": f"-p no:cacheprovider -p {guard}",
            "PYTHONPATH": str(guard_dir),
        },
    )
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    runs = [[str(test_file)]] if runs is None else runs(test_file)
    sandbox = PytestSandbox(paths, workspace, runs, log)
    planes = SimpleNamespace(harden_before_verify=AsyncMock(), verifier=Verifier)
    rewards, error, _ = await _verify_rollout(sandbox, task, paths, {}, planes)
    return guard, paths, rewards, error


def _kinds(paths, guard):
    """The new markers the verifier copied into the trial's verifier/ folder."""
    return sorted(
        p.name.rsplit(".", 1)[1]
        for p in paths.verifier_dir.glob(guard + ".*")
        if p.name.rsplit(".", 1)[1] in ("starting", "exited")
    )


def _assert_errored(rewards, error, opening):
    assert rewards is None
    assert error is not None
    assert error.startswith(NO_USABLE_REWARD + opening), error
    # The ending every unusable reward has: unscored, never infrastructure,
    # 0 in the every-trial mean and a failure in the scored-only mean.
    row = {"rewards": None, "verifier_error": error}
    passed = {"rewards": {"reward": 1.0}}
    assert is_no_usable_reward(error)
    assert classify_verifier_error(error) == VERIFIER_FAILED
    assert classify_score_outcome(row) == "verifier_errored"
    assert mean_reward_every_trial([row, passed]) == 0.5
    assert mean_scored_reward([row, passed]) == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize("log", ["test-stdout.txt", None], ids=["kept", "discarded"])
async def test_an_option_of_an_unloaded_plugin_ends_errored(tmp_path, log):
    """programbench@1.0's case: ``--timeout=30`` with pytest-timeout not loaded.

    Before EN-647 the trial was scored 0.0. pytest stops while parsing its
    command line, before any hook but registration has run, so only the
    ``starting`` marker records it; it does so even when test.sh discards
    pytest's output.
    """
    guard, paths, rewards, error = await _run(
        tmp_path, PASSING, runs=lambda test: [["--timeout=30", str(test)]], log=log
    )

    assert paths.reward_text_path.read_text() == "0"
    if log is not None:
        said = (paths.verifier_dir / log).read_text()
        assert "unrecognized arguments: --timeout=30" in said
    _assert_errored(
        rewards, error, "pytest stopped before it had read its whole command line"
    )
    assert _kinds(paths, guard) == ["starting"]


@pytest.mark.asyncio
async def test_a_run_that_collects_no_tests_ends_errored(tmp_path):
    guard, paths, rewards, error = await _run(tmp_path, NO_TESTS)

    assert paths.reward_text_path.read_text() == "0"
    _assert_errored(rewards, error, "pytest collected no tests (exit 5)")
    assert _kinds(paths, guard) == ["exited"]


@pytest.mark.asyncio
async def test_an_internal_error_outside_conftests_and_the_agents_files_ends_errored(
    tmp_path,
):
    """Exit 3 raised by pytest's or a plugin's own code (here a broken report
    hook, patched in by the task's test module): no conftest and no file the
    agent left in the traceback, so the verifier is broken. pytest calls no
    pytest_sessionfinish for it; the cmdline_main hook wrapper sees it."""
    guard, paths, rewards, error = await _run(tmp_path, PYTEST_BUG)

    assert "INTERNALERROR" in paths.test_stdout_path.read_text()
    assert paths.reward_text_path.read_text() == "0"
    _assert_errored(rewards, error, "pytest ended with an internal error (exit 3)")
    assert _kinds(paths, guard) == ["exited"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["task-conftest-bug", "build-at-sessionstart", "configure-imports"]
)
async def test_an_internal_error_through_a_conftest_is_a_plain_fail(tmp_path, case):
    """Exit 3 whose traceback passes through a conftest: its hooks may build,
    import or run the solution (R647 F2), so the run stays the plain fail
    test.sh scores, whatever the cause."""
    workspace = tmp_path / "workspace"
    conftest, agent_files = {
        "task-conftest-bug": (CRASHING_CONFTEST, {}),
        "build-at-sessionstart": (
            builds_at_sessionstart(workspace),
            {"compile.sh": "echo 'error: build failed' >&2\nexit 1\n"},
        ),
        "configure-imports": (
            imports_at_configure(workspace),
            {"solution.py": "raise RuntimeError('broken solution')\n"},
        ),
    }[case]
    guard, paths, rewards, error = await _run(
        tmp_path, PASSING, conftest=conftest, agent_files=agent_files
    )

    assert "INTERNALERROR" in paths.test_stdout_path.read_text()
    assert error is None, error
    assert rewards == {"reward": 0.0}
    assert _kinds(paths, guard) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("solution", ["hang", "segfault"])
async def test_a_pytest_killed_while_a_conftest_imports_the_solution_is_a_plain_fail(
    tmp_path, solution
):
    """R647 F1: the task's conftest imports a solution that hangs (test.sh's
    `timeout 5` kills pytest) or segfaults. ``starting`` is held back while
    the first conftests load, so the run stays the solution's 0."""
    workspace = tmp_path / "workspace"
    source = {
        "hang": "while True:\n    pass\n",
        "segfault": "import ctypes\nctypes.string_at(0)\n",
    }[solution]
    guard, paths, rewards, error = await _run(
        tmp_path,
        PASSING,
        conftest=imports_at_load(workspace),
        agent_files={"solution.py": source},
        runs=lambda test: [["--timeout", "5", str(test)]],
    )

    assert error is None, error
    assert rewards == {"reward": 0.0}
    assert _kinds(paths, guard) == []


@pytest.mark.asyncio
async def test_an_unloaded_option_after_the_conftests_load_still_ends_errored(
    tmp_path,
):
    """``starting`` stands again once the first conftests have loaded, so an
    unknown option (checked after them) still ends errored when the task's
    conftest imports a working solution."""
    workspace = tmp_path / "workspace"
    guard, paths, rewards, error = await _run(
        tmp_path,
        PASSING,
        conftest=imports_at_load(workspace),
        agent_files={"solution.py": "ok = True\n"},
        runs=lambda test: [["--timeout=30", str(test)]],
    )

    assert "unrecognized arguments: --timeout=30" in paths.test_stdout_path.read_text()
    _assert_errored(
        rewards, error, "pytest stopped before it had read its whole command line"
    )
    assert _kinds(paths, guard) == ["starting"]


@pytest.mark.asyncio
async def test_a_crash_after_the_tests_ran_is_scored_as_before(tmp_path):
    """R647 F5: a pytest_sessionfinish hook that raises after the tests ran
    escapes pytest, and Python exits 1: no marker, test.sh's 0 stands."""
    guard, paths, rewards, error = await _run(
        tmp_path, FAILING, conftest=SESSIONFINISH_CRASH
    )

    assert "report writer bug" in paths.test_stdout_path.read_text()
    assert error is None, error
    assert rewards == {"reward": 0.0}
    assert _kinds(paths, guard) == []


def test_a_test_folder_the_agent_emptied_is_the_agents(tmp_path, monkeypatch):
    """R647 F4: a workspace folder whose pre-agent copy is a folder counted
    as the image's whatever the agent removed from it, so `pytest
    <workspace>/tests` after the agent deleted the tests (exit 5) ended
    errored. The folder is the image's only while every file of its copy is
    there with the image's bytes. (The copy must be root's alone; the test
    stands in for that check.)"""
    from benchflow.sandbox import _pytest_plugin_guard as guard

    workspace, snapshot = tmp_path / "ws", tmp_path / "snap"
    for root in (workspace, snapshot):
        (root / "tests").mkdir(parents=True)
        (root / "tests" / "test_a.py").write_text(PASSING)
    monkeypatch.setattr(guard, "_BENCHFLOW_WORKSPACE", str(workspace))
    monkeypatch.setattr(guard, "_BENCHFLOW_SNAPSHOT", str(snapshot))
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", (str(workspace),))
    monkeypatch.setattr(guard, "_AGENT_TREES", [])
    monkeypatch.setattr(guard, "_BENCHFLOW_RECORD", "")
    monkeypatch.setattr(guard, "_RECORD", None)
    monkeypatch.setattr(guard, "_private_to_root", lambda directory: True)
    folder = str(workspace / "tests")

    assert not guard._the_agents(folder)
    (workspace / "tests" / "test_a.py").write_text(FAILING)
    assert guard._the_agents(folder)
    (workspace / "tests" / "test_a.py").unlink()
    assert guard._the_agents(folder)


@pytest.mark.asyncio
async def test_a_usage_error_inside_the_session_ends_errored(tmp_path):
    """A path that does not exist is a usage error pytest raises after it read
    its command line; the hook wrapper sees the UsageError itself."""
    guard, paths, rewards, error = await _run(
        tmp_path, PASSING, runs=lambda test: [[str(test.with_name("test_gone.py"))]]
    )

    assert paths.reward_text_path.read_text() == "0"
    assert error is not None and "usage error" in error, error
    _assert_errored(rewards, error, "pytest ")
    assert _kinds(paths, guard) in (["exited"], ["starting"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "reward"),
    [(FAILING, 0.0), (MISSING_SOLUTION, 0.0), (PASSING, 1.0)],
    ids=["failing-test", "collection-error", "passing"],
)
async def test_runs_that_judged_the_solution_are_scored_as_before(
    tmp_path, source, reward
):
    """A failing test, and tests that cannot import the solution (exit 2, a
    collection error: the agent's failure), still score 0 as a real fail;
    a passing run still scores 1. Neither leaves a marker."""
    guard, paths, rewards, error = await _run(tmp_path, source)

    assert error is None, error
    assert rewards == {"reward": reward}
    assert _kinds(paths, guard) == []


@pytest.mark.asyncio
async def test_a_crash_before_the_command_line_is_read_is_scored_as_before(tmp_path):
    """A conftest hook that raises while pytest reads its options kills pytest
    with a traceback (exit 1, not a usage error): the guard registered and
    left ``starting``, then cleared it at exit, so test.sh's 0 stands."""
    guard, paths, rewards, error = await _run(
        tmp_path, PASSING, conftest=ADDOPTION_CRASH
    )

    said = paths.test_stdout_path.read_text()
    assert "RuntimeError: addoption bug" in said, said
    assert error is None, error
    assert rewards == {"reward": 0.0}
    assert _kinds(paths, guard) == []
    registered = Path(pytest_plugin_guard_markers_dir(guard)).glob("*.registered")
    assert list(registered)


@pytest.mark.asyncio
async def test_an_in_process_usage_error_counts_after_a_later_pass(tmp_path):
    """A verifier script calling pytest.main twice in one process: the first
    stops with a usage error, the second passes and the script writes 1.
    The guard is imported once, so the second registration finds the first
    run's ``starting`` and keeps it as ``exited`` 4."""
    script = tmp_path / "verify.py"

    def runs(test):
        script.write_text(
            "import sys, pytest\n"
            f"pytest.main(['-q', '--timeout=30', {str(test)!r}])\n"
            f"sys.exit(pytest.main(['-q', {str(test)!r}]))\n"
        )
        return [["--script", str(script)]]

    guard, paths, rewards, error = await _run(tmp_path, PASSING, runs=runs)

    assert paths.reward_text_path.read_text() == "1"
    _assert_errored(rewards, error, "pytest ended with a usage error (exit 4)")
    assert _kinds(paths, guard) == ["exited"]


@pytest.mark.asyncio
async def test_version_and_help_runs_judge_nothing_and_stop_nothing(tmp_path):
    """``pytest --version`` and ``--help`` return before any session; they
    read their whole command line, so they leave no marker."""
    guard, paths, rewards, error = await _run(
        tmp_path, PASSING, runs=lambda test: [["--version"], ["--help"], [str(test)]]
    )

    assert error is None, error
    assert rewards == {"reward": 1.0}
    assert _kinds(paths, guard) == []


@pytest.mark.asyncio
async def test_a_pytest_a_test_runs_as_its_subject_does_not_count(tmp_path):
    """A task whose tests run pytest and expect exit 4 and exit 5 still passes."""
    guard, paths, rewards, error = await _run(tmp_path, SUBJECT)

    assert error is None, error
    assert rewards == {"reward": 1.0}
    assert _kinds(paths, guard) == []


@pytest.mark.asyncio
async def test_a_failing_solution_cannot_forge_the_new_markers(tmp_path, monkeypatch):
    """Markers live in the guard's own folder and carry an HMAC; unsigned
    ones, there or in the trial's verifier/ folder, are dropped, so the
    failing solution below still scores 0."""
    monkeypatch.setenv("FORGE_VERIFIER_DIR", str(tmp_path / "rollout" / "verifier"))
    monkeypatch.setenv("FORGE_GUARD_PARENT", str(tmp_path))

    guard, paths, rewards, error = await _run(tmp_path, FORGING_FAILURE)

    assert error is None, error
    assert rewards == {"reward": 0.0}
    assert (paths.verifier_dir / f"{guard}.1-deadbeef.exited").exists()
    forged = Path(pytest_plugin_guard_markers_dir(guard)) / f"{guard}.2-deadbeef.exited"
    assert forged.exists()


# Paths to exit 3, 4 or 5 the solution can reach stay plain scored fails
# (F2b 2026-11-15); each row names the path.


@pytest.mark.asyncio
async def test_a_conftest_that_cannot_import_the_solution_is_a_plain_fail(tmp_path):
    """Startup exit 4: the task's conftest imports a solution that is not
    there. pytest calls it a usage error, but a conftest may import the
    solution, so any conftest that fails to import stays the solution's 0."""
    guard, paths, rewards, error = await _run(
        tmp_path, PASSING, conftest=CONFTEST_IMPORTS_SOLUTION
    )

    said = paths.test_stdout_path.read_text()
    assert "solution_the_agent_never_wrote" in said, said
    assert error is None, error
    assert rewards == {"reward": 0.0}
    assert _kinds(paths, guard) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ini_owner", "errored"), [("task", True), ("agent", False)], ids=["task", "agent"]
)
async def test_an_unloaded_option_from_an_ini_file_depends_on_whose_file(
    tmp_path, ini_owner, errored
):
    """Startup exit 4 from ini ``addopts``: the task's ini is a broken
    verifier (errored); an ini the agent left in its workspace makes it the
    solution's (a plain fail)."""
    folder = tmp_path / ("verifier_tests" if ini_owner == "task" else "workspace")

    def runs(test):
        ini = folder / "pytest.ini"
        ini.write_text(UNLOADED_OPTION_INI)
        return [["-c", str(ini), str(test)]]

    guard, paths, rewards, error = await _run(tmp_path, PASSING, runs=runs)

    assert "unrecognized arguments: --timeout=30" in paths.test_stdout_path.read_text()
    if errored:
        _assert_errored(
            rewards, error, "pytest stopped before it had read its whole command line"
        )
        assert _kinds(paths, guard) == ["starting"]
    else:
        assert error is None, error
        assert rewards == {"reward": 0.0}
        assert _kinds(paths, guard) == []


@pytest.mark.asyncio
async def test_a_test_module_that_skips_without_the_solution_is_a_plain_fail(
    tmp_path,
):
    """Exit 5 by importorskip: the only test module skips itself because
    the solution is missing, so nothing was collected; a skipped collector
    keeps it the solution's 0."""
    guard, paths, rewards, error = await _run(tmp_path, IMPORTORSKIP)

    assert error is None, error
    assert rewards == {"reward": 0.0}
    assert _kinds(paths, guard) == []


@pytest.mark.asyncio
async def test_tests_built_from_the_solution_that_yield_none_are_a_plain_fail(
    tmp_path,
):
    """Exit 5 from the solution's code: the task's tests are made from the
    solution's list, which is empty. The tests imported a file the agent
    left, so the run stays the solution's 0."""
    workspace = tmp_path / "workspace"
    guard, paths, rewards, error = await _run(
        tmp_path, built_from(workspace), agent_files={"solution.py": "CASES = []\n"}
    )

    assert error is None, error
    assert rewards == {"reward": 0.0}
    assert _kinds(paths, guard) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_files", "status"),
    [({"test_mine.py": "X = 1\n"}, "no tests"), ({}, "missing")],
    ids=["emptied-exit-5", "gone-exit-4"],
)
async def test_a_test_path_in_the_agents_files_is_a_plain_fail(
    tmp_path, agent_files, status
):
    """Exit 5 or an in-session usage error (4) on a test path in the agent's
    workspace (tests it emptied or removed): the solution's 0. The same
    statuses on the task's own paths end errored (rows above)."""
    workspace = tmp_path / "workspace"
    guard, paths, rewards, error = await _run(
        tmp_path,
        PASSING,
        agent_files=agent_files,
        runs=lambda test: [[str(workspace / "test_mine.py")]],
    )

    assert error is None, error
    assert rewards == {"reward": 0.0}
    assert _kinds(paths, guard) == []


@pytest.mark.asyncio
async def test_an_internal_error_from_a_conftest_the_agent_left_is_a_plain_fail(
    tmp_path,
):
    """Exit 3 from a conftest in the agent's workspace: the solution's 0.
    The same crash in the task's conftest ends errored (row above)."""
    workspace = tmp_path / "workspace"
    guard, paths, rewards, error = await _run(
        tmp_path,
        PASSING,
        agent_files={"conftest.py": CRASHING_CONFTEST, "test_mine.py": PASSING},
        runs=lambda test: [[str(workspace / "test_mine.py")]],
    )

    assert "INTERNALERROR" in paths.test_stdout_path.read_text()
    assert error is None, error
    assert rewards == {"reward": 0.0}
    assert _kinds(paths, guard) == []


def _runner_markers(
    tmp_path,
    tests_source,
    *args,
    conftest=None,
    agent_ini=None,
    target=None,
    agent_ini_symlink=False,
    tests_files=None,
):
    """Run BenchFlow's protected runner and read back its signed markers.

    *agent_ini* is a pytest.ini the agent left in the runner's workspace, and
    the runner is then pointed at a test file there.
    """
    guard = lockdown.PYTEST_PLUGIN_GUARD_PREFIX + uuid.uuid4().hex
    tree = Tree(tmp_path / "tree")
    markers = Path(pytest_plugin_guard_markers_dir(guard))
    markers.mkdir(parents=True, mode=0o700)
    key = Path(pytest_plugin_guard_key_file(guard))
    key.write_text(pytest_plugin_guard_key(guard))
    # The runner writes markers only under /<guard prefix>*; here the guard
    # folder is in tmp_path.
    source = tree.run.read_text()
    check = f'_markers.startswith("/{lockdown.PYTEST_PLUGIN_GUARD_PREFIX}")'
    assert source.count(check) == 1
    source = source.replace(check, f"_markers.startswith({str(tmp_path)!r})")
    if agent_ini is not None:
        assert source.count("WORKSPACE = ''") == 1
        source = source.replace("WORKSPACE = ''", f"WORKSPACE = {str(tree.app)!r}")
        ini = tree.app / "pytest.ini"
        if agent_ini_symlink:
            # The agent's ini, linked from a file it wrote in another tree.
            elsewhere = tmp_path / "agent_tmp"
            elsewhere.mkdir()
            (elsewhere / "evil.ini").write_text(agent_ini)
            ini.symlink_to(elsewhere / "evil.ini")
        else:
            ini.write_text(agent_ini)
        (tree.app / "test_image.py").write_text(tests_source)
        target = str(tree.app / "test_image.py")
    tree.run.write_text(source)
    (tree.tests / "test_outputs.py").write_text(tests_source)
    if conftest is not None:
        (tree.tests / "conftest.py").write_text(conftest)
    for name, source in (tests_files or {}).items():
        (tree.tests / name).write_text(source)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(tree.run),
            *args,
            target or "test_outputs.py",
        ],
        cwd=tree.tests,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "BENCHFLOW_PYTEST_MARKERS": str(markers / guard),
            "BENCHFLOW_PYTEST_KEY": str(key),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    read = subprocess.run(
        ["/bin/sh", "-c", pytest_plugin_guard_markers_cmd(guard)],
        capture_output=True,
        text=True,
    )
    found = parse_pytest_plugin_guard_markers(read.stdout, guard)
    by_kind: dict = {}
    for marker in found:
        by_kind.setdefault(marker.kind, []).append(marker)
    return result, by_kind


@pytest.mark.parametrize(
    ("source", "args", "code", "opening"),
    [
        (PASSING, ("--timeout=30",), 4, "pytest ended with a usage error (exit 4)"),
        (NO_TESTS, (), 5, "pytest collected no tests (exit 5)"),
        (FAILING, (), 1, None),
        (PASSING, (), 0, None),
    ],
    ids=["unloaded-plugin-option", "no-tests", "failing", "passing"],
)
def test_the_protected_runner_records_a_run_that_judged_nothing(
    tmp_path, source, args, code, opening
):
    """``$BENCHFLOW_PYTEST`` calls pytest.main itself, so it has the status."""
    result, markers = _runner_markers(tmp_path, source, *args)

    assert result.returncode == code, result.stdout + result.stderr
    exited = [m.detail.partition("\n")[0] for m in markers.get("exited", [])]
    reason = _unjudged_pytest(markers)
    if opening is None:
        assert exited == []
        assert reason is None
    else:
        assert exited == [str(code)]
        assert reason is not None and reason.startswith(opening), reason


@pytest.mark.parametrize(
    ("source", "conftest", "code"),
    [
        (IMPORTORSKIP, None, 5),
        (PASSING, CONFTEST_IMPORTS_SOLUTION, 4),
        (PASSING, CRASHING_CONFTEST, 3),
    ],
    ids=["importorskip-exit-5", "conftest-import-exit-4", "conftest-crash-exit-3"],
)
def test_the_protected_runner_leaves_the_solutions_failures_scored(
    tmp_path, source, conftest, code
):
    """The runner judges the same paths: a test module skipped without the
    solution and a conftest that cannot import it leave no marker."""
    result, markers = _runner_markers(tmp_path, source, conftest=conftest)

    assert result.returncode == code, result.stdout + result.stderr
    assert "exited" not in markers
    assert _unjudged_pytest(markers) is None


@pytest.mark.parametrize(
    "ini",
    ["[pytest]\naddopts = --timeout=30\n", "[pytest]\nminversion = 99\n"],
    ids=["unloaded-option", "minversion"],
)
def test_the_protected_runner_leaves_an_agents_ini_scored(tmp_path, ini):
    """R647 F3: an ini file the agent left in the runner's workspace (a bogus
    ``addopts``, ``minversion = 99``) gives exit 4 before any test; the runner
    checks the ini as the guard does and leaves no marker."""
    result, markers = _runner_markers(tmp_path, PASSING, agent_ini=ini)

    assert result.returncode == 4, result.stdout + result.stderr
    assert "exited" not in markers
    assert _unjudged_pytest(markers) is None


def _guard_with_trees(monkeypatch, tmp_path):
    """The guard module with a tests folder, a workspace and its root-only
    pre-agent copy (the root-only check stood in), and no record."""
    from benchflow.sandbox import _pytest_plugin_guard as guard

    tests, workspace, snapshot = tmp_path / "tests", tmp_path / "ws", tmp_path / "snap"
    for folder in (tests, workspace, snapshot):
        folder.mkdir()
    monkeypatch.setattr(guard, "_BENCHFLOW_TESTS_DIR", str(tests))
    monkeypatch.setattr(guard, "_BENCHFLOW_WORKSPACE", str(workspace))
    monkeypatch.setattr(guard, "_BENCHFLOW_SNAPSHOT", str(snapshot))
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", (str(workspace),))
    monkeypatch.setattr(guard, "_AGENT_TREES", [])
    monkeypatch.setattr(guard, "_SEEN_CONFTESTS", [])
    monkeypatch.setattr(guard, "_BENCHFLOW_RECORD", "")
    monkeypatch.setattr(guard, "_RECORD", None)
    monkeypatch.setattr(guard, "_private_to_root", lambda directory: True)
    return guard, tests, workspace, snapshot


@pytest.mark.parametrize(
    "where", ["tests-helper", "image-plugin", "import-error", "elsewhere"]
)
def test_an_internal_error_is_judged_by_where_its_frames_run(
    tmp_path, monkeypatch, where
):
    """R647 R3-A: an exit 3 raised from a helper module in the tests folder
    (listed in a conftest's ``pytest_plugins``), from the image's own
    unchanged plugin in the workspace, or as an ImportError, may be the
    solution's, whatever the file is called or holds. A frame anywhere else
    (an installed plugin) is not."""
    guard, tests, workspace, snapshot = _guard_with_trees(monkeypatch, tmp_path)
    hook = "def hook():\n    raise RuntimeError('build failed')\n"
    (workspace / "imgplug.py").write_text(hook)
    (snapshot / "imgplug.py").write_text(hook)
    source, path = {
        "tests-helper": (hook, tests / "build_helper.py"),
        "image-plugin": (hook, workspace / "imgplug.py"),
        "import-error": (
            "def hook():\n    import mypkg_helpers_the_agent_deleted  # noqa: F401\n",
            tmp_path / "site" / "plugin.py",
        ),
        "elsewhere": (hook, tmp_path / "site" / "plugin.py"),
    }[where]
    namespace = {}
    exec(compile(source, str(path), "exec"), namespace)
    try:
        namespace["hook"]()
    except Exception as error:
        raised = error
    assert guard._raised_where_the_solution_runs(raised) is (where != "elsewhere")


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any folder")
def test_an_image_folder_whose_copy_cannot_be_read_is_the_agents(tmp_path, monkeypatch):
    """R647 N3-3: a folder of the pre-agent copy that cannot be read proves
    nothing, so the live folder is not the image's."""
    guard, _, workspace, snapshot = _guard_with_trees(monkeypatch, tmp_path)
    for root in (workspace, snapshot):
        (root / "tests" / "sub").mkdir(parents=True)
        (root / "tests" / "test_a.py").write_text(PASSING)
        (root / "tests" / "sub" / "test_b.py").write_text(PASSING)
    assert not guard._the_agents(str(workspace / "tests"))
    (snapshot / "tests" / "sub").chmod(0)
    try:
        assert guard._the_agents(str(workspace / "tests"))
    finally:
        (snapshot / "tests" / "sub").chmod(0o755)


def test_the_protected_runner_leaves_an_agents_symlinked_ini_scored(tmp_path):
    """R647 R3-B: the agent's ini as a workspace link to a file it wrote in
    another tree resolves outside the workspace; the link counts as the
    agent's."""
    result, markers = _runner_markers(
        tmp_path,
        PASSING,
        agent_ini="[pytest]\naddopts = --timeout=30\n",
        agent_ini_symlink=True,
    )

    assert result.returncode == 4, result.stdout + result.stderr
    assert "exited" not in markers
    assert _unjudged_pytest(markers) is None


def test_the_protected_runner_judges_an_exit_3_by_where_it_runs(tmp_path):
    """R647 R3-A on the runner: a tests-folder helper module the conftest
    lists in ``pytest_plugins`` crashes at session start: no marker."""
    result, markers = _runner_markers(
        tmp_path,
        PASSING,
        conftest="pytest_plugins = ['build_helper']\n",
        tests_files={"build_helper.py": CRASHING_CONFTEST},
    )

    assert result.returncode == 3, result.stdout + result.stderr
    assert "exited" not in markers
    assert _unjudged_pytest(markers) is None


@pytest.mark.parametrize(
    ("where", "solutions"),
    [
        ("agent-module", True),
        ("testsh-venv", False),
        ("tests-folder", True),
        ("testsh-venv-no-record", True),
    ],
)
def test_code_test_sh_installed_in_the_agents_trees_is_the_verifiers(
    tmp_path, monkeypatch, where, solutions
):
    """R647 R4-1: pytest in a venv test.sh made in the workdir after the agent
    stopped is the verifier's code, so its own crash ends errored; a module
    the agent left there, and code in the tests folder, are the solution's.
    The record of the agent's trees (inodes, taken when the agent stopped)
    tells them apart; without one, everything there is the solution's."""
    guard, tests, workspace, _ = _guard_with_trees(monkeypatch, tmp_path)
    (workspace / "agent_mod.py").write_text("X = 1\n")
    record = {
        guard._identity(os.lstat(path)): {None}
        for path in (workspace, *workspace.rglob("*"))
    }
    venv = workspace / ".tbench-testing" / "lib" / "_pytest"
    venv.mkdir(parents=True)
    (venv / "main.py").write_text("X = 1\n")
    monkeypatch.setattr(
        guard, "_record", lambda: None if where.endswith("no-record") else record
    )
    path = {
        "agent-module": workspace / "agent_mod.py",
        "testsh-venv": venv / "main.py",
        "tests-folder": tests / "build_helper.py",
        "testsh-venv-no-record": venv / "main.py",
    }[where]
    namespace = {}
    exec(
        compile("def hook():\n    raise RuntimeError('bug')\n", str(path), "exec"),
        namespace,
    )
    try:
        namespace["hook"]()
    except Exception as error:
        raised = error
    assert guard._raised_where_the_solution_runs(raised) is solutions


@pytest.mark.asyncio
async def test_a_startup_usage_error_after_a_conftest_with_cmdline_preparse_is_scored(
    tmp_path,
):
    """R647 N3-1: on pytest <= 7 a conftest's pytest_cmdline_preparse runs
    after the first conftests load, where solution code could hang, so
    ``starting`` is not written again when a registered conftest defines it
    and the run scores as before (the guard does not implement that
    deprecated hook)."""
    guard, paths, rewards, error = await _run(
        tmp_path,
        PASSING,
        conftest="def pytest_cmdline_preparse(config, args):\n    pass\n",
        runs=lambda test: [["--timeout=30", str(test)]],
    )

    assert "unrecognized arguments: --timeout=30" in paths.test_stdout_path.read_text()
    assert error is None, error
    assert rewards == {"reward": 0.0}
    assert _kinds(paths, guard) == []
