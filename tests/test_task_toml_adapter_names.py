"""The task.toml inbound adapter names its format and nothing else.

Uses only ``detect_adapter`` and ``materialize_inbound_task_md``, so the same
checks run against any version of the adapter module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.adapters.inbound import detect_adapter, materialize_inbound_task_md
from benchflow.task import TaskDocument
from benchflow.task.config import TaskConfig

_TASK_TOML = """\
schema_version = "1.1"
vendor_ext = "kept"

[task]
name = "org/sample"

[environment]
cpus = 1
"""


def _task_dir(root: Path) -> Path:
    task_dir = root / "sample"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "task.toml").write_text(_TASK_TOML)
    (task_dir / "instruction.md").write_text("Do the thing.\n")
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\necho 1\n")
    return task_dir


def test_adapter_module_and_class_name_the_format(tmp_path: Path) -> None:
    adapter = detect_adapter(_task_dir(tmp_path))
    assert adapter.__module__ == "benchflow.adapters.task_toml"
    assert adapter.__name__ == "TaskTomlAdapter"


def test_inbound_source_is_task_toml(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path)
    inbound = detect_adapter(task_dir).from_task_dir(task_dir)
    assert inbound.source == "task-toml"
    assert inbound.compatibility is not None
    assert inbound.compatibility.source == "task-toml"


def test_materialized_task_md_records_task_toml_source(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path)
    inbound = detect_adapter(task_dir).from_task_dir(task_dir)
    native = materialize_inbound_task_md(inbound, tmp_path / "native")
    document = TaskDocument.from_path(native / "task.md")
    assert document.benchflow["compat"]["source"] == "task-toml"


def test_missing_instruction_error_names_only_the_file(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path)
    (task_dir / "instruction.md").unlink()
    with pytest.raises(FileNotFoundError) as excinfo:
        detect_adapter(task_dir).from_task_dir(task_dir)
    assert str(excinfo.value).startswith("Task is missing instruction.md: ")


def test_unknown_format_error_names_only_task_toml(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as excinfo:
        detect_adapter(tmp_path)
    assert str(excinfo.value) == (
        f"Unrecognized task format in {tmp_path}: expected a 'task.toml'."
    )


def test_both_spellings_error_calls_environment_the_task_toml_spelling() -> None:
    with pytest.raises(ValueError) as excinfo:
        TaskConfig.model_validate_toml(
            'schema_version = "1.1"\n[environment]\ncpus = 1\n[sandbox]\ncpus = 1\n'
        )
    assert "'environment' is the legacy task.toml spelling" in str(excinfo.value)
