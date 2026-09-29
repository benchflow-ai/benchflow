"""A pytest that cannot load the plugin guard is a verifier error, never a 0.

Guards the fix for the silent-zero regression introduced by the pytest plugin guard,
which passes ``-p _benchflow_guard_<hex>`` to every verifier pytest. The
per-interpreter guard install puts the guard into every interpreter that exists at hardening
time, but an interpreter test.sh creates itself (uvx, a fresh venv, an
apk-installed Python) can still run pytest under ``-I`` or with a replaced
PYTHONPATH. pytest then aborts on the missing module, test.sh writes reward 0,
and a correct solution was scored as a failure.

The verifier runs a real pytest subprocess with the environment the verifier
passes to test.sh; only the container transport is stubbed. ``log=None`` models
a test.sh that sends pytest's output to /dev/null.
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
    VERIFIER_FAILED,
    classify_result,
    classify_verifier_error,
)
from benchflow.rollout._setup import _verify_rollout
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.lockdown import (
    _DISCOVER_PYTEST_PLUGINS_SCRIPT,
    _pytest_plugin_guard_source,
)
from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig

PASSING = "def test_solution():\n    assert True\n"
FAILING = "def test_solution():\n    assert False\n"
HOSTILE_PLUGIN = (
    "def pytest_collection_modifyitems(items):\n"
    "    for item in items:\n"
    "        item.obj = lambda: None\n"
)


def armed(extra=""):
    """The protected guard as hardening writes it, followed by *extra* source."""

    def source(guard, verifier_dir, workspace):
        return (
            _pytest_plugin_guard_source(guard, (str(workspace),), [], str(verifier_dir))
            + extra
        )

    return source


class PytestSandbox:
    """Run test.sh as ``pytest; reward = 1 if rc == 0 else 0`` on this machine."""

    is_mounted = True

    def __init__(self, paths, tests, *, isolated, pythonpath, banner, log, plugins):
        self._paths = paths
        self._tests = tests
        self._isolated = isolated
        self._pythonpath = pythonpath
        self._banner = banner
        self._log = log
        self._plugins = plugins

    async def upload_dir(self, *args, **kwargs):
        pass

    async def exec(self, command, env=None, **kwargs):
        if "test-stdout.txt" not in command:
            return ExecResult(stdout="", stderr="", return_code=0)
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("PYTEST_", "PYTHON"))
        }
        environment.update(env or {}, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        if self._pythonpath is not None:
            # test.sh replaced PYTHONPATH, e.g. ``PYTHONPATH=/app pytest``.
            environment["PYTHONPATH"] = self._pythonpath
        flags = ["-I"] if self._isolated else []
        plugins = [arg for name in self._plugins for arg in ("-p", name)]
        result = subprocess.run(
            [sys.executable, *flags, "-m", "pytest", "-q", *plugins, str(self._tests)],
            cwd=self._tests.parent,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        if self._log is not None:
            log = self._paths.verifier_dir / self._log
            log.write_text(self._banner + result.stdout)
        self._paths.reward_text_path.write_text("1" if result.returncode == 0 else "0")
        return ExecResult(stdout="", stderr="", return_code=result.returncode)


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
    isolated=False,
    pythonpath=None,
    guard_source=None,
    banner="",
    log="test-stdout.txt",
    plugins=(),
):
    guard = "_benchflow_guard_" + uuid.uuid4().hex
    guard_dir = tmp_path / guard
    guard_dir.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test_outputs.py").write_text(test_source)
    for name in plugins:
        (workspace / f"{name}.py").write_text(HOSTILE_PLUGIN)
    task = _task(
        tmp_path,
        {
            "PYTEST_ADDOPTS": f"-p no:cacheprovider -p {guard}",
            "PYTHONPATH": str(guard_dir),
        },
    )
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    if callable(guard_source):
        guard_source = guard_source(guard, paths.verifier_dir, workspace)
    if guard_source is not None:
        (guard_dir / f"{guard}.py").write_text(guard_source)
    sandbox = PytestSandbox(
        paths,
        workspace / "test_outputs.py",
        isolated=isolated,
        pythonpath=pythonpath,
        banner=banner,
        log=log,
        plugins=plugins,
    )
    planes = SimpleNamespace(harden_before_verify=AsyncMock(), verifier=Verifier)
    rewards, error, _ = await _verify_rollout(sandbox, task, paths, {}, planes)
    return guard, paths, rewards, error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("isolated", "pythonpath", "log", "signature"),
    [
        (True, None, "test-stdout.txt", 'Error importing plugin "{}"'),
        (False, "/nonexistent-app", "test-stdout.txt", "No module named '{}'"),
        (True, None, "pytest.log", 'Error importing plugin "{}"'),
    ],
    ids=["python-I", "replaced-PYTHONPATH", "own-log-file"],
)
async def test_unloadable_guard_is_a_verifier_error_not_a_zero(
    tmp_path, isolated, pythonpath, log, signature
):
    """Guards the fix for the silent zero introduced by the pytest plugin guard.

    The solution is correct and test.sh wrote reward 0 only because pytest
    could not import ``-p _benchflow_guard_<hex>``.
    """
    guard, paths, rewards, error = await _run(
        tmp_path, PASSING, isolated=isolated, pythonpath=pythonpath, log=log
    )

    assert signature.format(guard) in (paths.verifier_dir / log).read_text()
    assert paths.reward_text_path.read_text() == "0"
    assert rewards is None
    assert error is not None and guard in error
    assert classify_verifier_error(error) == VERIFIER_FAILED
    assert (
        classify_result(reward=None, error=None, verifier_error=error)
        == "verifier_errored"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "reward"), [(PASSING, 1.0), (FAILING, 0.0)], ids=["pass", "fail"]
)
async def test_loaded_guard_keeps_genuine_results_scored(tmp_path, source, reward):
    """Guards the fix for the pytest plugin guard: only a guard load failure is diverted.

    A genuine test failure, even one whose output names a guessed guard
    module, is still scored normally.
    """
    guessed = "_benchflow_guard_" + "0" * 32
    banner = (
        f'ImportError: Error importing plugin "{guessed}": x\n'
        f"ModuleNotFoundError: No module named '{guessed}'\n"
    )
    _, _, rewards, error = await _run(
        tmp_path, source, guard_source=_DISCOVER_PYTEST_PLUGINS_SCRIPT, banner=banner
    )

    assert error is None
    assert rewards == {"reward": reward}


@pytest.mark.asyncio
async def test_guard_refused_by_pluggy_is_a_verifier_error(tmp_path):
    """Guards the fix for the pytest plugin guard: a guard pytest cannot register.

    an earlier guard fix removed a hook argument (``plugin_name``) pytest 7 does
    not pass; pluggy refuses such a plugin as a whole and pytest aborts the
    same way. An argument no pytest passes reproduces that on any version.
    """
    source = "def pytest_plugin_registered(plugin, manager, unknown):\n    pass\n"
    guard, paths, rewards, error = await _run(tmp_path, PASSING, guard_source=source)

    assert f"Plugin '{guard}' for hook" in paths.test_stdout_path.read_text()
    assert rewards is None
    assert error is not None and guard in error


def _guard_markers(paths, guard):
    return sorted(p.name for p in paths.verifier_dir.glob(guard + ".*"))


@pytest.mark.asyncio
async def test_guard_pytest_did_not_register_is_a_verifier_error_without_logs(
    tmp_path,
):
    """Guards the guard import-failure detection against a test.sh that discards output.

    The guard import-failure detection recognises pluggy refusing the guard (the pytest 7 ``plugin_name``
    failure fixed earlier) only by its message in the verifier logs. A
    test.sh that sends pytest to /dev/null left no message, so the correct
    solution below was scored 0. The guard now leaves a marker when it is
    imported and removes it once pytest has registered it.
    """
    refused = armed(
        "\ndef pytest_plugin_registered(plugin, manager, unknown):\n    pass\n"
    )
    guard, paths, rewards, error = await _run(
        tmp_path, PASSING, guard_source=refused, log=None
    )

    assert paths.reward_text_path.read_text() == "0"
    assert not paths.test_stdout_path.read_text()
    assert rewards is None
    assert error is not None and guard in error
    assert classify_verifier_error(error) == VERIFIER_FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "reward"), [(PASSING, 1.0), (FAILING, 0.0)], ids=["pass", "fail"]
)
async def test_registered_guard_leaves_quiet_verifiers_scored(tmp_path, source, reward):
    """Guards the marker fix for the guard import-failure detection's log-only detection.

    A guard pytest registered removes its marker, so a quiet verifier's
    genuine pass or failure is still scored and nothing is left behind.
    """
    guard, paths, rewards, error = await _run(
        tmp_path, source, guard_source=armed(), log=None
    )

    assert error is None
    assert rewards == {"reward": reward}
    assert _guard_markers(paths, guard) == []


@pytest.mark.asyncio
async def test_guard_rejection_in_a_quiet_verifier_stays_scored(tmp_path):
    """Guards the marker fix for the guard import-failure detection against excusing tampering.

    The guard refusing an agent plugin stops pytest before registration
    completes; that is the intended 0, not a guard that failed to load.
    """
    guard, paths, rewards, error = await _run(
        tmp_path, PASSING, guard_source=armed(), log=None, plugins=("agent_plugin",)
    )

    assert error is None
    assert rewards == {"reward": 0.0}
    assert _guard_markers(paths, guard) == []


@pytest.mark.asyncio
async def test_guard_crash_is_a_verifier_error_without_logs(tmp_path):
    """Guards the fix for guard crashes being scored since the pytest plugin guard.

    A guard hook that fails for its own reasons (here pytest no longer
    exporting a name the guard imports, like the missing invocation_params
    fixed by an earlier guard fix) aborts pytest; test.sh wrote 0 for a correct solution
    and the trial was scored. The guard now records the traceback in a
    ``crashed`` marker and the verifier reports an error, with or without logs.
    """
    crashing = armed(
        "\ndef _validate(names, entry_points=True):\n"
        "    from _pytest.config import name_a_later_pytest_dropped\n"
    )
    guard, paths, rewards, error = await _run(
        tmp_path, PASSING, guard_source=crashing, log=None
    )

    assert paths.reward_text_path.read_text() == "0"
    assert rewards is None
    assert error is not None and guard in error and "crashed" in error
    assert classify_verifier_error(error) == VERIFIER_FAILED
    (marker,) = paths.verifier_dir.glob(guard + ".*.crashed")
    assert "name_a_later_pytest_dropped" in marker.read_text()
