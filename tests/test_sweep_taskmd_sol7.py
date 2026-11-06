"""Guards the PY8 fixes against the task.md cache at 363d815b."""

import shutil

import pytest

from benchflow.task.formats import materialize_task_dir
from benchflow.taskmd.materialize import package_tree_hash
from tests._taskmd_helpers import EXAMPLES


@pytest.fixture
def package(tmp_path, monkeypatch):
    source = tmp_path / "task"
    shutil.copytree(EXAMPLES / "hello-world", source)
    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "cache"))
    return source


@pytest.mark.parametrize(
    "relative,mode",
    [("sandbox/helper", 0o755), ("sandbox/helper", 0o600), ("sandbox/data", 0o700)],
)
def test_mode_changes_invalidate_materialization_without_changing_source_tree(
    package, relative, mode
):
    """Permission changes affect runtime caching, not canonical provenance."""
    path = package / relative
    if relative.endswith("data"):
        path.mkdir()
        path.chmod(0o755)
    else:
        path.write_text("#!/bin/sh\necho correct\n")
        path.chmod(0o644)
    before_hash = package_tree_hash(package)
    before = materialize_task_dir(package)
    path.chmod(mode)
    after = materialize_task_dir(package)
    assert package_tree_hash(package) == before_hash
    assert after != before
    copied = after / "environment" / path.name
    assert copied.stat().st_mode & 0o777 == mode


@pytest.mark.parametrize("directory", [False, True])
def test_followed_source_links_are_part_of_the_runtime_cache(
    package, tmp_path, directory
):
    """A sibling case: copytree follows source links, so their target modes count."""
    target = tmp_path / "linked-source"
    if directory:
        target.mkdir()
        helper = target / "helper"
    else:
        helper = target
    helper.write_text("#!/bin/sh\nexit 0\n")
    helper.chmod(0o644)
    (package / "sandbox" / "linked").symlink_to(target, target_is_directory=directory)
    before_hash = package_tree_hash(package)
    before = materialize_task_dir(package)
    helper.chmod(0o755)
    after = materialize_task_dir(package)
    assert package_tree_hash(package) == before_hash
    assert before != after
    copied = after / "environment" / "linked"
    if directory:
        copied /= "helper"
    assert copied.stat().st_mode & 0o777 == 0o755
