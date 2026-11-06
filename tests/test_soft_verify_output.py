"""What soft_verify() hands a scripted user as ``RoundResult.verifier_output``.

Round 3's fix ("soft verify hands a scripted user the tests'
output") against the bug it fixes:
soft_verify read only ``/logs/verifier/*.log`` and ``output.txt``, but the
verifier saves test.sh's stdout and stderr as ``test-stdout.txt``. A task that
writes no log of its own gave every scripted user ``verifier_output=None``, and
every ``--loop-strategy verify-retry`` retry an empty feedback section (found
running ``docs/tasks/scripted-users.md`` on citation-check: round 1's prompt
read "The tests failed:\\nNone"). The fix reads the verifier's copy in the
trial folder, so soft verify still runs exactly one sandbox command that
writes test-stdout.txt (the transport tests count on that).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchflow import rollout
from benchflow.loop_strategies import FeedbackLevel, filter_verifier_feedback
from benchflow.task import RolloutPaths
from benchflow.task.config import TaskConfig

PYTEST_TAIL = (
    "tests/test_outputs.py .F..\n"
    "FAILED tests/test_outputs.py::test_citations - AssertionError: expected 3, got 2\n"
    "========================= 1 failed, 3 passed in 0.52s =========================\n"
)


def _paths(tmp_path: Path) -> RolloutPaths:
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    return paths


def test_reads_the_verifiers_copy_of_the_tests_output(tmp_path):
    paths = _paths(tmp_path)
    (tmp_path / "rollout" / "verifier" / "test-stdout.txt").write_text(PYTEST_TAIL)

    output = rollout._host_test_stdout(paths.test_stdout_path)

    assert output == PYTEST_TAIL.strip()
    assert (
        filter_verifier_feedback(output, FeedbackLevel.NAMES)
        == "tests/test_outputs.py::test_citations"
    )


def test_no_output_file_reads_as_none(tmp_path):
    assert rollout._host_test_stdout(_paths(tmp_path).test_stdout_path) is None


def test_a_huge_output_is_capped(tmp_path):
    paths = _paths(tmp_path)
    (tmp_path / "rollout" / "verifier" / "test-stdout.txt").write_bytes(
        b"x" * (rollout._SOFT_VERIFY_OUTPUT_MAX_BYTES + 10)
    )

    output = rollout._host_test_stdout(paths.test_stdout_path)

    assert output is not None and len(output) == rollout._SOFT_VERIFY_OUTPUT_MAX_BYTES


TEST_SH = """#!/bin/bash
echo "FAILED tests/test_outputs.py::test_citations - AssertionError: expected 3, got 2"
echo 0 > "$VERIFIER_DIR/reward.txt"
"""


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
async def test_soft_verify_hands_the_user_what_test_sh_printed(tmp_path):
    """Through the real Docker exec wrapper, as tests/test_verifier_soft_transport.py runs it."""
    from tests.test_verifier_soft_transport import _docker, _soft

    tests_dir = tmp_path / "task" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text(TEST_SH)
    task = MagicMock()
    task.name = "soft-output"
    task.task_dir = tests_dir.parent
    task.paths.task_dir = tests_dir.parent
    task.paths.tests_dir = tests_dir
    task.paths.test_path = tests_dir / "test.sh"
    task.paths.uses_native_verifier_dir = False
    task.config = TaskConfig.model_validate_toml('version = "1.0"\n[verifier]\n')
    paths = _paths(tmp_path)
    image_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path)}
    sandbox = _docker(tmp_path, paths, image_env)

    rewards, output, error = await _soft(tmp_path, sandbox, task, paths)

    assert error is None, error
    assert rewards == {"reward": 0.0}
    assert (
        output is not None and "FAILED tests/test_outputs.py::test_citations" in output
    )
    ran = getattr(sandbox, "ran", None)  # the transport fixture records commands
    assert ran is not None and len(ran) == 1
