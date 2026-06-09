"""Regression tests for migrate_task_to_task_md mount-prefix rewrite (#651).

When ``remove_legacy=True`` promotes ``tests/ -> verifier/`` and
``solution/ -> oracle/``, scripts that hardcode the old absolute mount
prefix (e.g. ``/tests/test_outputs.py``) must be rewritten to the native
prefix or they break at the new mount point.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from benchflow._utils.task_authoring import migrate_task_to_task_md

_LEGACY_FIXTURE = Path(__file__).parent / "examples" / "hello-world-task"


def _make_legacy_task(tmp_path: Path) -> Path:
    """Copy the canonical legacy hello-world task into tmp_path.

    The fixture ships task.toml + instruction.md + tests/ + solution/,
    which is exactly the split-format layout migrate promotes.
    """
    task_dir = tmp_path / "task"
    shutil.copytree(_LEGACY_FIXTURE, task_dir)
    return task_dir


def test_tests_to_verifier_rewrites_mount_prefix(tmp_path: Path) -> None:
    task_dir = _make_legacy_task(tmp_path)
    (task_dir / "tests" / "test.sh").write_text(
        "#!/bin/bash\npytest /tests/test_outputs.py\n"
    )

    migrate_task_to_task_md(task_dir, remove_legacy=True)

    promoted = (task_dir / "verifier" / "test.sh").read_text()
    assert "/verifier/test_outputs.py" in promoted
    assert "/tests/" not in promoted


def test_solution_to_oracle_rewrites_mount_prefix(tmp_path: Path) -> None:
    task_dir = _make_legacy_task(tmp_path)
    (task_dir / "solution" / "solve.sh").write_text(
        "#!/bin/bash\npython /solution/ref.py\n"
    )

    migrate_task_to_task_md(task_dir, remove_legacy=True)

    promoted = (task_dir / "oracle" / "solve.sh").read_text()
    assert "/oracle/ref.py" in promoted
    assert "/solution/" not in promoted


def test_rewrite_is_prefix_scoped(tmp_path: Path) -> None:
    """Only the promoted leading mount segment is rewritten.

    A ``/logs/verifier/ctrf.json`` path (where ``verifier`` is not the
    leading segment) and unrelated prose must survive untouched, while the
    leading ``/tests/`` segment in a tests->verifier promotion maps to
    ``/verifier/`` and ``/solution/`` in solution->oracle maps to ``/oracle/``.
    """
    task_dir = _make_legacy_task(tmp_path)
    (task_dir / "tests" / "test.sh").write_text(
        "#!/bin/bash\n"
        "pytest /tests/test_outputs.py\n"
        "cat /logs/verifier/ctrf.json\n"
        "# run the tests in /tests/ now\n"
    )
    (task_dir / "solution" / "solve.sh").write_text(
        "#!/bin/bash\npython /solution/ref.py  # the solution lives here\n"
    )

    migrate_task_to_task_md(task_dir, remove_legacy=True)

    verifier = (task_dir / "verifier" / "test.sh").read_text()
    assert "pytest /verifier/test_outputs.py" in verifier
    assert "run the tests in /verifier/ now" in verifier
    # unrelated nested path is untouched
    assert "/logs/verifier/ctrf.json" in verifier
    assert "/tests/" not in verifier

    oracle = (task_dir / "oracle" / "solve.sh").read_text()
    assert "python /oracle/ref.py" in oracle
    # the bare word "solution" in prose is not a mount prefix; it stays
    assert "the solution lives here" in oracle
    assert "/solution/" not in oracle


def test_normal_task_migration_round_trips(tmp_path: Path) -> None:
    """A normal legacy task still migrates and produces task.md."""
    task_dir = _make_legacy_task(tmp_path)

    result = migrate_task_to_task_md(task_dir, remove_legacy=True)

    assert result.removed_legacy is True
    assert (task_dir / "task.md").is_file()
    assert not (task_dir / "task.toml").exists()
    assert not (task_dir / "instruction.md").exists()
    assert result.migrated_legacy_dirs == (
        "tests/ -> verifier/",
        "solution/ -> oracle/",
    )


def test_overwrite_false_on_existing_task_md_raises(tmp_path: Path) -> None:
    task_dir = _make_legacy_task(tmp_path)
    (task_dir / "task.md").write_text("---\n---\nexisting\n")

    with pytest.raises(FileExistsError):
        migrate_task_to_task_md(task_dir, overwrite=False)
