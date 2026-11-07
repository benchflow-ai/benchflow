"""The pytest plugin guard's markers, read back after a verifier that ran no
pytest, or ran it through a library, and a read that failed.

Terminal-Bench 4.1's music-harmony checks its answer with a Python checker
that never imports pytest (its verifier image has none); a sweep scored its
correct oracle 0 with "pytest plugin guard not confirmed" and no guard marker
copied out. No pytest, so no marker existed: what failed was BenchFlow's read
of the guard's markers after test.sh, on a host building several images at
once. One failed exec decided the trial, and the warning listed guard failures
that never happened (load failure, stop, refusal).

The verifier runs real pytest (or none) with the armed guard on its path;
only the container transport is stubbed (``test_verifier_guard_load_failure``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import ClassVar

import pytest

from benchflow._utils.scoring import VERIFIER_INFRA, classify_verifier_error
from benchflow._utils.trial_warnings import recording_trial_warnings
from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from benchflow.task import verifier_core
from tests import test_verifier_guard_load_failure as load_failure

# A checker that is not pytest (music-harmony's test.sh runs one), and one
# that runs pytest in its own process through pytest.main, *n* times.
NO_PYTEST = "import sys\nsys.exit(0 if 'pytest' not in sys.modules else 3)\n"
LIBRARY_PYTEST = (
    "import sys, pytest\n"
    "codes = [pytest.main(['-q', sys.argv[1]]) for _ in range(int(sys.argv[2]))]\n"
    "sys.exit(max(codes))\n"
)
TIMED_OUT = RuntimeError("Command timed out after 30 seconds")


@pytest.fixture(autouse=True)
def guard_parent(tmp_path, monkeypatch):
    """Guards and their markers live under this test's folder, as at / in the sandbox."""
    monkeypatch.setattr(lockdown, "_PYTEST_PLUGIN_GUARD_PARENT", str(tmp_path))
    monkeypatch.setattr(
        verifier_core, "_GUARD_MARKER_RETRY_DELAYS_SEC", (0.0, 0.0), raising=False
    )


class _Checker(load_failure.PytestSandbox):
    """test.sh runs a checker in Python (*checker*), not ``python -m pytest``;
    reward 1 when it exits 0. Each read of the guard's markers fails with
    the next of *read_failures* (an exception, or an exit status) until
    they run out."""

    checker = NO_PYTEST
    runs = 1
    read_failures: ClassVar[list] = []
    reads = 0

    async def exec(self, command, env=None, **kwargs):
        if "/.markers" in command and "test-stdout.txt" not in command:
            type(self).reads += 1
            if self.read_failures:
                failure = self.read_failures.pop(0)
                if isinstance(failure, BaseException):
                    raise failure
                return ExecResult(stdout="", stderr="", return_code=failure)
            return await super().exec(command, env=env, **kwargs)
        if "test-stdout.txt" not in command:
            return await super().exec(command, env=env, **kwargs)
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("PYTEST_", "PYTHON"))
        }
        environment.update(env or {}, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        result = subprocess.run(
            [sys.executable, "-c", self.checker, str(self._tests), str(self.runs)],
            cwd=self._tests.parent,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        (self._paths.verifier_dir / "test-stdout.txt").write_text(result.stdout)
        self._paths.reward_text_path.write_text("1" if result.returncode == 0 else "0")
        return ExecResult(stdout="", stderr="", return_code=result.returncode)


async def _verify(tmp_path, monkeypatch, *, checker=NO_PYTEST, runs=1, failures=()):
    sandbox = type(
        "Sandbox",
        (_Checker,),
        {"checker": checker, "runs": runs, "read_failures": list(failures), "reads": 0},
    )
    monkeypatch.setattr(load_failure, "PytestSandbox", sandbox)
    with recording_trial_warnings() as warnings:
        guard, paths, rewards, error = await load_failure._run(
            tmp_path, load_failure.PASSING, guard_source=load_failure.armed()
        )
    return guard, paths, rewards, error, warnings, sandbox.reads


@pytest.mark.parametrize("runs", [1, 2])
async def test_a_checker_that_runs_pytest_through_a_library_registers_the_guard(
    tmp_path, monkeypatch, runs
):
    """pytest.main in the checker's own process (once, or twice) reads
    PYTEST_ADDOPTS as a pytest command does: the guard loads, registers and
    leaves its ``registered`` marker, and the reward is scored as written.
    Pins what the sweep's diagnosis of music-harmony assumed was broken."""
    guard, paths, rewards, error, warnings, _ = await _verify(
        tmp_path, monkeypatch, checker=LIBRARY_PYTEST, runs=runs
    )
    assert "passed" in (paths.verifier_dir / "test-stdout.txt").read_text()
    left = [path.name for path in (tmp_path / guard / ".markers").iterdir()]
    assert [name.rsplit(".", 1)[1] for name in left] == ["registered"]
    assert (error, rewards, warnings) == (None, {"reward": 1.0}, [])


async def test_a_verifier_without_pytest_is_scored_as_written(tmp_path, monkeypatch):
    """music-harmony's shape: no pytest runs, no marker is left, the read
    finds none and test.sh's reward stands."""
    guard, _, rewards, error, warnings, reads = await _verify(tmp_path, monkeypatch)
    assert list((tmp_path / guard / ".markers").iterdir()) == []
    assert (error, rewards, warnings, reads) == (None, {"reward": 1.0}, [], 1)


@pytest.mark.parametrize("failure", [TIMED_OUT, 1], ids=["timed-out", "exit-1"])
async def test_a_marker_read_that_failed_is_tried_again(tmp_path, monkeypatch, failure):
    """Guards the retried read against the regression of 96155b555 (an
    unreadable read scores 0) and 4f5ba44ad (one read decides): one exec of
    the read that timed out or failed on a loaded host scored music-harmony's
    correct oracle 0. The read only reads, so it is tried again."""
    _, _, rewards, error, warnings, reads = await _verify(
        tmp_path, monkeypatch, failures=[failure, failure]
    )
    assert (error, rewards, reads) == (None, {"reward": 1.0}, 3)
    assert warnings == []


async def test_a_read_that_keeps_failing_scores_0_naming_the_failed_read(
    tmp_path, monkeypatch
):
    """Three failed tries leave no marker read, so a refusal may be among
    them and test.sh's reward is not used (score 0, as before); the warning
    says it was BenchFlow's read that failed and how, not that the guard
    failed to load, stopped or refused a plugin."""
    _, _, rewards, error, warnings, reads = await _verify(
        tmp_path, monkeypatch, failures=[TIMED_OUT] * 3
    )
    assert (error, rewards, reads) == (None, {"reward": 0.0}, 3)
    (warning,) = warnings
    assert warning.startswith("Verifier scored 0: pytest plugin guard not confirmed: ")
    assert "could not read back the markers" in warning
    assert "3 tries, the last: RuntimeError: Command timed out after 30" in warning
    assert "failed to load" not in warning and "refused a plugin" not in warning


async def test_a_transport_failure_of_the_read_is_a_named_verifier_error(
    tmp_path, monkeypatch
):
    """A transport failure on every try is the controller's, which verifier
    code cannot bring about: a named verifier error that is not scored (the
    docs' rule for every verifier step), classed as infrastructure."""
    lost = ConnectionError("connection reset by the daemon")
    guard, _, rewards, error, warnings, reads = await _verify(
        tmp_path, monkeypatch, failures=[lost] * 3
    )
    assert (rewards, warnings, reads) == (None, [], 3)
    assert error is not None
    assert error.startswith(
        "verifier crashed: VerifierInfrastructureError: the pytest plugin guard "
        f"{guard}'s markers could not be read in 3 tries"
    )
    assert "connection reset by the daemon" in error
    assert classify_verifier_error(error) == VERIFIER_INFRA


async def test_a_listing_test_sh_can_shape_is_not_read_again(tmp_path, monkeypatch):
    """Output that came back whole is decided as it is: a marker folder that
    is gone (test.sh's user owns it) scores 0 after one read, naming it."""

    class Gone(_Checker):
        async def exec(self, command, env=None, **kwargs):
            if "/.markers" in command and "test-stdout.txt" not in command:
                guards = f"{lockdown.PYTEST_PLUGIN_GUARD_PREFIX}*/.markers"
                for folder in tmp_path.glob(guards):
                    folder.rmdir()
            return await super().exec(command, env=env, **kwargs)

    monkeypatch.setattr(load_failure, "PytestSandbox", Gone)
    Gone.reads = 0
    with recording_trial_warnings() as warnings:
        _, _, rewards, error = await load_failure._run(
            tmp_path, load_failure.PASSING, guard_source=load_failure.armed()
        )
    assert (error, rewards, Gone.reads) == (None, {"reward": 0.0}, 1)
    (warning,) = warnings
    assert "(the marker directory is gone)" in warning
