"""Tests for ``benchflow.cli.eval`` — the future-facing eval CLI module.

NOTE: ``cli/eval.py`` is **not wired into the live CLI** (the active
``bench eval create`` lives in ``cli/main.py``).  These tests cover the
task-reference resolver (``_resolve_task_ref``) which will be used once
the module is promoted to the live entry point.
"""

from __future__ import annotations

from pathlib import Path

import click.exceptions
import pytest

from benchflow.cli.eval import _resolve_task_ref


def _make_task_dir(parent: Path, name: str) -> Path:
    d = parent / name
    d.mkdir()
    (d / "task.toml").write_text('schema_version = "1.1"\n')
    (d / "instruction.md").write_text("instruction\n")
    (d / "environment").mkdir()
    (d / "tests").mkdir()
    (d / "tests" / "test.sh").write_text("#!/bin/bash\necho 1\n")
    return d


def test_resolve_single_task_dir(tmp_path: Path) -> None:
    task = _make_task_dir(tmp_path, "my-task")
    resolved, is_batch = _resolve_task_ref(str(task))
    assert resolved == task.resolve()
    assert is_batch is False


def test_resolve_directory_of_tasks_is_batch(tmp_path: Path) -> None:
    _make_task_dir(tmp_path, "task-a")
    _make_task_dir(tmp_path, "task-b")
    resolved, is_batch = _resolve_task_ref(str(tmp_path))
    assert resolved == tmp_path.resolve()
    assert is_batch is True


def test_resolve_missing_path_exits(tmp_path: Path) -> None:
    with pytest.raises(click.exceptions.Exit):
        _resolve_task_ref(str(tmp_path / "does-not-exist"))


def test_resolve_harbor_ref_routes_through_ensure_tasks(
    tmp_path: Path, monkeypatch
) -> None:
    """`harbor://<name>` resolves to the cached dataset directory (batch)."""

    fake_dataset = tmp_path / "cached-aime"
    fake_dataset.mkdir()
    _make_task_dir(fake_dataset, "aime_60")
    _make_task_dir(fake_dataset, "aime_61")

    captured: dict[str, str] = {}

    def fake_ensure_tasks(ref: str) -> Path:
        captured["ref"] = ref
        return fake_dataset

    from benchflow import task_download

    monkeypatch.setattr(task_download, "ensure_tasks", fake_ensure_tasks)
    resolved, is_batch = _resolve_task_ref("harbor://aime@1.0")
    assert captured["ref"] == "harbor://aime@1.0"
    assert resolved == fake_dataset
    assert is_batch is True


def test_resolve_harbor_ref_with_subtask(tmp_path: Path, monkeypatch) -> None:
    """`harbor://<name>/<task>` resolves to a single task from the dataset."""

    fake_dataset = tmp_path / "cached-aime"
    fake_dataset.mkdir()
    _make_task_dir(fake_dataset, "aime_60")
    _make_task_dir(fake_dataset, "aime_61")

    def fake_ensure_tasks(ref: str) -> Path:
        return fake_dataset

    from benchflow import task_download

    monkeypatch.setattr(task_download, "ensure_tasks", fake_ensure_tasks)
    resolved, is_batch = _resolve_task_ref("harbor://aime/aime_60")
    assert resolved == fake_dataset / "aime_60"
    assert is_batch is False


def test_resolve_benchflow_ref(tmp_path: Path, monkeypatch) -> None:
    """`benchflow://<name>` works identically to `harbor://`."""

    fake_dataset = tmp_path / "cached-benchjack"
    fake_dataset.mkdir()
    _make_task_dir(fake_dataset, "exploit-1")

    def fake_ensure_tasks(ref: str) -> Path:
        assert ref == "benchflow://benchjack"
        return fake_dataset

    from benchflow import task_download

    monkeypatch.setattr(task_download, "ensure_tasks", fake_ensure_tasks)
    resolved, is_batch = _resolve_task_ref("benchflow://benchjack")
    assert resolved == fake_dataset
    assert is_batch is True


def test_resolve_non_task_dir_exits(tmp_path: Path) -> None:
    """A directory with no task.toml children errors cleanly."""

    (tmp_path / "not-a-task").mkdir()
    with pytest.raises(click.exceptions.Exit):
        _resolve_task_ref(str(tmp_path))
