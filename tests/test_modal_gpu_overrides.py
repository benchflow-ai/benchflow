"""Guard effective GPU admission after task.md overlays at 460d52bc3."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from benchflow._utils.config_override import apply_config_override
from benchflow.sandbox.setup import (
    _create_environment,
    _validate_task_runtime_for_launch,
)
from benchflow.task import Task, UnsupportedTaskFeatureError


def _task(tmp_path, kind, initial_gpu):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    environment = task_dir / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n")
    verifier = task_dir / "tests"
    verifier.mkdir()
    (verifier / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    if kind == "markdown":
        resource = "  gpus: 1\n  gpu_types: [T4]\n" if initial_gpu else "  gpus: 0\n"
        (task_dir / "task.md").write_text(
            '---\nschema_version: "1.3"\nsandbox:\n'
            + resource
            + "---\n\n## prompt\n\nSolve the task.\n"
        )
    else:
        resource = 'gpus = 1\ngpu_types = ["T4"]\n' if initial_gpu else "gpus = 0\n"
        (task_dir / "task.toml").write_text(
            'schema_version = "1.3"\n[environment]\n' + resource
        )
        (task_dir / "instruction.md").write_text("Solve the task.\n")
    return Task(task_dir), task_dir


@pytest.mark.parametrize("kind", ["markdown", "legacy"])
@pytest.mark.parametrize(
    ("initial_gpu", "backend", "overlay", "refused_path"),
    [
        (False, "modal", {"gpus": 1, "gpu_types": ["T4"]}, None),
        (True, "docker", {"gpus": 0, "gpu_types": None}, None),
        (False, "docker", {"gpus": 1, "gpu_types": ["T4"]}, "sandbox.gpus"),
        (False, "daytona", {"gpus": 1, "gpu_types": ["T4"]}, "sandbox.gpus"),
        (False, "modal", {"gpus": 1, "gpu_types": ["T4", "H100"]}, "sandbox.gpu_types"),
        (True, "modal", {"gpus": 0}, "sandbox.gpu_types"),
    ],
)
def test_launch_gate_uses_effective_resource_overlay(
    tmp_path, kind, initial_gpu, backend, overlay, refused_path
):
    """Guards stale document.config admission in 460d52bc3 and 67753c4ac."""
    task, task_dir = _task(tmp_path, kind, initial_gpu)
    original = task.config.model_dump()
    document = task.document
    task.config = apply_config_override(task.config, {"sandbox": overlay})
    if refused_path is None:
        _validate_task_runtime_for_launch(
            task, sandbox_type=backend, task_path=task_dir
        )
    else:
        with pytest.raises(UnsupportedTaskFeatureError) as error:
            _validate_task_runtime_for_launch(
                task, sandbox_type=backend, task_path=task_dir
            )
        assert {feature.path for feature in error.value.features} == {refused_path}
    if document is not None:
        assert task.document is document
        assert document.config.model_dump() == original
    assert task.config.sandbox.gpus == overlay["gpus"]


def test_overlay_retains_document_only_refusals(tmp_path):
    """Guards dropping document policy checks while repairing 460d52bc3."""
    task, task_dir = _task(tmp_path, "markdown", False)
    task.document = replace(task.document, benchflow={"runtime_policy": {}})
    task.config = apply_config_override(
        task.config, {"sandbox": {"gpus": 1, "gpu_types": ["T4"]}}
    )
    with pytest.raises(UnsupportedTaskFeatureError) as error:
        _validate_task_runtime_for_launch(
            task, sandbox_type="modal", task_path=task_dir
        )
    assert {feature.path for feature in error.value.features} == {
        "benchflow.runtime_policy"
    }


@pytest.mark.parametrize("kind", ["markdown", "legacy"])
@pytest.mark.parametrize("backend", ["docker", "modal"])
def test_invalid_overlay_refuses_before_staging_or_provider_setup(
    tmp_path, monkeypatch, kind, backend
):
    """Guards post-preflight invalid overlay rejection in 460d52bc3."""
    task, task_dir = _task(tmp_path, kind, False)
    task.config = apply_config_override(
        task.config, {"sandbox": {"gpus": 1, "gpu_types": ["T4", "H100"]}}
    )
    stage = MagicMock()
    provider = MagicMock()
    monkeypatch.setattr("benchflow.sandbox.setup._stage_environment", stage)
    monkeypatch.setattr("benchflow.sandbox.setup._make_sandbox", provider)
    with pytest.raises(UnsupportedTaskFeatureError):
        _create_environment(backend, task, task_dir, "gpu-task", MagicMock())
    stage.assert_not_called()
    provider.assert_not_called()
