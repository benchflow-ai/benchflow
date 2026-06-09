"""Discovery must key off a CHEAP marker, not full validation.

Review MEDIUM (#651): ``_is_task_dir`` ran a full ``check_task()`` on any
directory containing ``task.md`` and only returned ``True`` when there were
*zero* issues. The effect was an asymmetry:

* A misauthored ``task.toml`` dir is discovered (cheap ``.exists()`` check)
  and fails *loudly* at run time.
* A misauthored ``task.md`` dir — one that fails STRUCTURAL validation
  (e.g. missing ``environment/Dockerfile``) — was SILENTLY EXCLUDED from
  the run, vanishing from the summary instead of surfacing an error.

Discovery should answer only "is this plausibly a task directory?" using a
cheap marker (``task.md`` OR ``task.toml`` exists). Validation belongs to
``check_task`` / run time, where failures are loud. These tests pin
``_is_task_dir`` (the unit) to that contract.
"""

from __future__ import annotations

from pathlib import Path

from benchflow._utils.task_authoring import check_task
from benchflow.evaluation import _is_task_dir


def _make_broken_task_md_dir(parent: Path, name: str = "broken-taskmd") -> Path:
    """A dir with a task.md marker that FAILS structural check_task.

    Missing ``environment/Dockerfile`` (and the required dirs) guarantees
    ``check_task`` returns a non-empty issue list.
    """
    task = parent / name
    task.mkdir()
    (task / "task.md").write_text("# Broken task\n\nNo frontmatter, no env.\n")
    return task


def _make_broken_task_toml_dir(parent: Path, name: str = "broken-tasktoml") -> Path:
    """The task.toml analogue of the broken dir above."""
    task = parent / name
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    return task


def test_broken_task_md_dir_fails_check_task(tmp_path):
    """Precondition: the fixture really is structurally invalid."""
    task = _make_broken_task_md_dir(tmp_path)
    issues = check_task(task)
    assert issues, "fixture should fail check_task; otherwise the test is vacuous"


def test_is_task_dir_discovers_broken_task_md(tmp_path):
    """A task.md dir that fails check_task must still be DISCOVERED.

    This is the core regression: discovery keys off the cheap marker, so a
    misauthored task.md dir is surfaced (and fails loudly at run time)
    rather than silently dropped.
    """
    task = _make_broken_task_md_dir(tmp_path)
    assert _is_task_dir(task) is True, (
        "discovery must treat a task.md dir as a task even when check_task "
        "reports issues — otherwise misauthored tasks vanish silently"
    )


def test_is_task_dir_symmetric_for_broken_md_and_toml(tmp_path):
    """task.md and task.toml dirs must be discovered symmetrically.

    Both are equally broken; discovery must not privilege one marker over
    the other.
    """
    md = _make_broken_task_md_dir(tmp_path, "md")
    toml = _make_broken_task_toml_dir(tmp_path, "toml")
    assert _is_task_dir(md) is True
    assert _is_task_dir(toml) is True
    assert _is_task_dir(md) == _is_task_dir(toml)


def test_is_task_dir_rejects_non_task_dir(tmp_path):
    """A dir with neither marker is not a task dir."""
    plain = tmp_path / "not-a-task"
    plain.mkdir()
    (plain / "README.md").write_text("# just docs\n")
    assert _is_task_dir(plain) is False
