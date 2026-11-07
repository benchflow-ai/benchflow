"""coverage.py's start-up hook stays off in the verifier.

``coverage.process_startup()`` (called by a test package, a ``sitecustomize``
or a ``.pth`` file) starts measuring whenever ``COVERAGE_PROCESS_START`` or
``COVERAGE_PROCESS_CONFIG`` is defined, even empty. Hardening named the start
variable empty, so a test suite whose package calls the hook ran under the
tracer, about twice as slow (TB2 fix-code-vulnerability: bottle's 367 tests,
0.66 s -> 1.36 s), and wrote a ``.coverage`` file into its working folder.
The verifier's command now unsets both, so the hook returns before it reads
or writes anything.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchflow._verifier_env import VERIFIER_UNSET_ENV, pinned_shell_prefix
from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig
from tests.test_verifier_env_entry_points import LocalSandbox

# What the exec can carry: hardening's empty start variable, and the image's
# own configuration variable (coverage >= 7.10 reads it first).
STARTUP_ENV = {"COVERAGE_PROCESS_START": "", "COVERAGE_PROCESS_CONFIG": "{}"}


def test_the_prefix_unsets_both_startup_variables():
    assert VERIFIER_UNSET_ENV == ("COVERAGE_PROCESS_START", "COVERAGE_PROCESS_CONFIG")
    prefix = pinned_shell_prefix(None)
    assert prefix.endswith("unset COVERAGE_PROCESS_START COVERAGE_PROCESS_CONFIG; ")
    out = subprocess.run(
        ["/bin/sh", "-c", prefix + "env"],
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **STARTUP_ENV},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    names = {line.split("=", 1)[0] for line in out.splitlines()}
    assert not names & set(VERIFIER_UNSET_ENV)
    # The pins are still exported.
    assert "PIP_CONFIG_FILE=/dev/null" in out.splitlines()


def test_hardening_still_names_the_start_variable_empty():
    """Commands without the prefix keep the old value, never the image's."""
    from benchflow.sandbox.lockdown import VERIFIER_ENV

    assert VERIFIER_ENV["COVERAGE_PROCESS_START"] == ""


def _task(tmp_path: Path, script: str, env: dict[str, str]) -> MagicMock:
    tests_dir = tmp_path / "task" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text(script)
    task = MagicMock()
    task.name = "coverage-startup"
    task.task_dir = tests_dir.parent
    task.paths.task_dir = tests_dir.parent
    task.paths.tests_dir = tests_dir
    task.paths.test_path = tests_dir / "test.sh"
    task.paths.uses_native_verifier_dir = False
    task.config = TaskConfig.model_validate_toml('version = "1.0"\n[verifier]\n')
    task.config.verifier.env = env
    return task


async def _verify(tmp_path: Path, script: str) -> tuple[dict, LocalSandbox]:
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    sandbox = LocalSandbox(tmp_path, paths, tmp_path)
    task = _task(tmp_path, script, dict(STARTUP_ENV))
    result = await Verifier(task=task, rollout_paths=paths, sandbox=sandbox).verify()
    return result.rewards, sandbox


async def test_test_sh_runs_without_the_startup_variables(tmp_path):
    """Failing first on e84d0cc8e: both reached test.sh."""
    script = (
        "#!/bin/sh\n"
        'env > "$OUT/env.txt"\n'
        'if env | grep -q "^COVERAGE_PROCESS_"; then echo 0; else echo 1; fi'
        ' > "$VERIFIER_DIR/reward.txt"\n'
    )
    rewards, sandbox = await _verify(tmp_path, script)
    assert rewards == {"reward": 1.0}, (sandbox.out / "env.txt").read_text()


async def test_the_coverage_hook_does_not_start_in_test_sh(tmp_path):
    """The real hook: it measured, and wrote .coverage, on e84d0cc8e."""
    pytest.importorskip("coverage")
    work = tmp_path / "work"
    work.mkdir()
    probe = (
        "import coverage, sys; "
        "started = coverage.process_startup() is not None; "
        "sys.exit(1 if started else 0)"
    )
    script = (
        "#!/bin/sh\n"
        f"cd {shlex.quote(str(work))} || exit 1\n"
        f"{shlex.quote(sys.executable)} -c {shlex.quote(probe)} "
        '> "$OUT/probe.txt" 2>&1\n'
        'if [ $? -eq 0 ]; then echo 1; else echo 0; fi > "$VERIFIER_DIR/reward.txt"\n'
    )
    rewards, sandbox = await _verify(tmp_path, script)
    assert rewards == {"reward": 1.0}, (sandbox.out / "probe.txt").read_text()
    assert not list(work.glob(".coverage*"))
