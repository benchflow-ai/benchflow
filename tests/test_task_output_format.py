from pathlib import Path

import pytest

from benchflow.task import (
    TASK_OUTPUT_FORMATS,
    ensure_existing_task_output_format,
    oracle_dir_name,
    task_entrypoint_name,
    validate_task_output_format,
    verifier_dir_name,
)


def test_validate_task_output_format_accepts_supported_values() -> None:
    """Guards PR #1's shared benchmark converter format validation."""
    assert TASK_OUTPUT_FORMATS == ("legacy", "task-md")
    assert validate_task_output_format("legacy") == "legacy"
    assert validate_task_output_format("task-md") == "task-md"


def test_validate_task_output_format_rejects_unknown_value() -> None:
    """Guards PR #1 against adapters accepting silent format drift."""
    with pytest.raises(ValueError, match="task_format must be one of"):
        validate_task_output_format("native")


@pytest.mark.parametrize(
    ("task_format", "entrypoint", "verifier_dir", "oracle_dir"),
    [
        ("legacy", "task.toml", "tests", "solution"),
        ("task-md", "task.md", "verifier", "oracle"),
    ],
)
def test_task_output_format_layout_names(
    task_format,
    entrypoint: str,
    verifier_dir: str,
    oracle_dir: str,
) -> None:
    """Guards adapter layout names from diverging by benchmark."""
    assert task_entrypoint_name(task_format) == entrypoint
    assert verifier_dir_name(task_format) == verifier_dir
    assert oracle_dir_name(task_format) == oracle_dir


def test_ensure_existing_task_output_format_rejects_mixed_entrypoints(
    tmp_path: Path,
) -> None:
    """Guards adapted converters from resuming into the wrong task layout."""
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text("[task]\n")

    with pytest.raises(ValueError, match=r"already contains task\.toml"):
        ensure_existing_task_output_format(task_dir, "task-md")


def test_ensure_existing_task_output_format_rejects_unknown_existing_dir(
    tmp_path: Path,
) -> None:
    """Existing output dirs must already contain the selected entrypoint."""
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    with pytest.raises(ValueError, match=r"missing task\.md"):
        ensure_existing_task_output_format(task_dir, "task-md")
