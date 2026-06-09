"""Mount-routing characterization tests for the verifier (HIGH #10).

The verifier chooses *which* source directory to upload and *which* container
mount point it lands on based on whether the task ships a native ``verifier/``
package or a legacy ``tests/`` package. ``Verifier._verify_test_script``
encodes that choice as::

    uses_native_verifier_dir = task.paths.uses_native_verifier_dir
    verifier_code_dir = (
        SandboxPaths().verifier_code_dir   # /verifier  (native)
        if uses_native_verifier_dir
        else SandboxPaths().tests_dir       # /tests     (legacy)
    )
    upload_dir(source_dir=task.paths.tests_dir, target_dir=str(verifier_code_dir))

These tests assert that selection end to end through the *public*
``TaskPaths`` / ``SandboxPaths`` interface — building real task fixtures on
disk and checking which source dir resolves and which mount target it maps to.
They touch no source and characterize existing behavior, so they pass as-is.
"""

from __future__ import annotations

from pathlib import Path

from benchflow.task.paths import SandboxPaths, TaskPaths


def _resolve_mount(task_dir: Path) -> tuple[Path, str]:
    """Mirror the verifiers mount selection through the public interface.

    Returns ``(source_dir, mount_target)`` — exactly the ``source_dir`` and
    ``target_dir`` the verifier passes to ``sandbox.upload_dir``.
    """
    paths = TaskPaths(task_dir)
    sandbox_paths = SandboxPaths()
    source_dir = paths.tests_dir
    target = (
        sandbox_paths.verifier_code_dir
        if paths.uses_native_verifier_dir
        else sandbox_paths.tests_dir
    )
    return source_dir, str(target)


def _make_task(tmp_path: Path, *, verifier_subdir: str) -> Path:
    """Create a minimal task fixture whose verifier package lives in
    ``verifier_subdir`` ("verifier" for native, "tests" for legacy)."""
    task_dir = tmp_path / "task"
    vdir = task_dir / verifier_subdir
    vdir.mkdir(parents=True)
    (vdir / "test.sh").write_text("#!/bin/sh\necho 0 > reward.txt\n")
    return task_dir


# verifier source dir -> mount target


def test_native_verifier_dir_mounts_to_slash_verifier(tmp_path):
    """A task with a native ``verifier/`` dir uploads it to ``/verifier``."""
    task_dir = _make_task(tmp_path, verifier_subdir="verifier")
    paths = TaskPaths(task_dir)

    assert paths.uses_native_verifier_dir is True
    assert paths.tests_dir == task_dir / "verifier"

    source_dir, target = _resolve_mount(task_dir)
    assert source_dir == task_dir / "verifier"
    assert target == "/verifier"
    assert str(SandboxPaths().verifier_code_dir) == "/verifier"


def test_legacy_tests_dir_mounts_to_slash_tests(tmp_path):
    """A legacy task with only a ``tests/`` dir uploads it to ``/tests``."""
    task_dir = _make_task(tmp_path, verifier_subdir="tests")
    paths = TaskPaths(task_dir)

    assert paths.uses_native_verifier_dir is False
    assert paths.tests_dir == task_dir / "tests"

    source_dir, target = _resolve_mount(task_dir)
    assert source_dir == task_dir / "tests"
    assert target == "/tests"
    assert str(SandboxPaths().tests_dir) == "/tests"


def test_native_verifier_dir_takes_precedence_over_legacy_tests(tmp_path):
    """When BOTH ``verifier/`` and ``tests/`` exist, the native package wins —
    ``tests_dir`` resolves to ``verifier/`` and the mount is ``/verifier``."""
    task_dir = _make_task(tmp_path, verifier_subdir="verifier")
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\n")

    paths = TaskPaths(task_dir)
    assert paths.uses_native_verifier_dir is True
    assert paths.tests_dir == task_dir / "verifier"

    source_dir, target = _resolve_mount(task_dir)
    assert source_dir == task_dir / "verifier"
    assert target == "/verifier"


# oracle / solution dir -> mount target (reachable through the same interface)


def test_native_oracle_dir_mounts_to_slash_oracle(tmp_path):
    """A native ``oracle/`` dir resolves ``solution_dir`` to it and maps to the
    ``/oracle`` mount point."""
    task_dir = tmp_path / "task"
    (task_dir / "oracle").mkdir(parents=True)
    (task_dir / "oracle" / "solve.sh").write_text("#!/bin/sh\n")

    paths = TaskPaths(task_dir)
    assert paths.uses_native_oracle_dir is True
    assert paths.solution_dir == task_dir / "oracle"
    assert str(SandboxPaths().oracle_dir) == "/oracle"


def test_legacy_solution_dir_mounts_to_slash_solution(tmp_path):
    """A legacy ``solution/`` dir (no native ``oracle/``) resolves
    ``solution_dir`` to it and maps to the ``/solution`` mount point."""
    task_dir = tmp_path / "task"
    (task_dir / "solution").mkdir(parents=True)
    (task_dir / "solution" / "solve.sh").write_text("#!/bin/sh\n")

    paths = TaskPaths(task_dir)
    assert paths.uses_native_oracle_dir is False
    assert paths.solution_dir == task_dir / "solution"
    assert str(SandboxPaths().solution_dir) == "/solution"
