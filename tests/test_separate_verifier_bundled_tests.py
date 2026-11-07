"""A task.toml task's separate verifier keeps the tests its image bundles.

The task.toml format builds a separate verifier's image from ``tests/`` and
bundles the tests in it: every Terminal-Bench 4.1 verifier Dockerfile copies
them to ``/tests``, and some then change them. risk-scorer-replay compiles a
private scorer from ``tests/legacy_score_probe.c``, deletes the source, and
checks that nobody can read one. BenchFlow uploaded ``tests/`` over the
image's ``/tests``, which brought the source back readable, and the reference
solution scored 0 where the format's other runner scores 1.0. The verifier
now uses ``/tests`` as the image built it when its test script is there.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.rollout._separate_verifier import _verifier_task
from benchflow.task import Task
from benchflow.task.verifier_core import Verifier
from benchflow.task.verifier_sandbox import plan_verifier_image


def _task(root: Path, *, tests_dockerfile: bool) -> Task:
    (root / "environment").mkdir(parents=True)
    (root / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (root / "tests").mkdir()
    (root / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    if tests_dockerfile:
        (root / "tests" / "Dockerfile").write_text(
            "FROM ubuntu:24.04\nCOPY . /tests/\n"
        )
    (root / "instruction.md").write_text("Do it.\n")
    (root / "task.toml").write_text(
        'schema_version = "1.0"\n\n[verifier]\nenvironment_mode = "separate"\n'
    )
    return Task(root)


def _view(task: Task):
    return _verifier_task(
        task, plan_verifier_image(task.config, Path(task.paths.task_dir))
    )


def test_a_verifier_built_from_tests_bundles_them(tmp_path):
    assert (
        _view(_task(tmp_path, tests_dockerfile=True))._benchflow_bundled_tests is True
    )


def test_a_verifier_from_the_task_image_does_not(tmp_path):
    view = _view(_task(tmp_path, tests_dockerfile=False))
    assert view._benchflow_bundled_tests is False


def _verifier(task, found: int) -> tuple[Verifier, MagicMock]:
    sandbox = MagicMock()
    sandbox.exec = AsyncMock(return_value=SimpleNamespace(return_code=found, stdout=""))
    return Verifier(task, MagicMock(), sandbox), sandbox


@pytest.mark.asyncio
async def test_an_image_with_its_test_script_is_used_as_built(tmp_path):
    verifier, sandbox = _verifier(_view(_task(tmp_path, tests_dockerfile=True)), 0)
    assert await verifier._bundled_tests(PurePosixPath("/tests"), "main") is True
    command = sandbox.exec.call_args.args[0]
    assert command == "test -f /tests/test.sh"
    assert sandbox.exec.call_args.kwargs["user"] == "root"


@pytest.mark.asyncio
async def test_an_image_without_its_test_script_gets_the_upload(tmp_path):
    verifier, _ = _verifier(_view(_task(tmp_path, tests_dockerfile=True)), 1)
    assert await verifier._bundled_tests(PurePosixPath("/tests"), "main") is False


@pytest.mark.asyncio
async def test_a_shared_verifier_never_asks(tmp_path):
    task = _task(tmp_path, tests_dockerfile=True)
    verifier, sandbox = _verifier(task, 0)
    assert await verifier._bundled_tests(PurePosixPath("/tests"), "main") is False
    sandbox.exec.assert_not_called()
