"""Guard effective package GPU diagnostics after PR #7 at 41e76366e."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from benchflow._utils.config_override import apply_config_override
from benchflow.task import Task, TaskPackage


def _task(tmp_path: Path, kind: str, initial_gpu: bool) -> Task:
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
    return Task(task_dir)


@pytest.mark.parametrize("kind", ["markdown", "legacy"])
@pytest.mark.parametrize(
    ("initial_gpu", "backend", "supported"),
    [
        (False, "docker", False),
        (True, "docker", True),
        (False, "modal", True),
        (True, "modal", True),
    ],
)
def test_task_package_reports_effective_gpu_resources(
    tmp_path: Path, kind: str, initial_gpu: bool, backend: str, supported: bool
) -> None:
    """Guards stale package diagnostics retained in PR #7 at 41e76366e."""
    task = _task(tmp_path, kind, initial_gpu)
    document = task.document
    original = task.config.model_dump()
    authored = TaskPackage.from_task_dir(task.task_dir, sandbox=backend)
    task.config = apply_config_override(
        task.config,
        {
            "sandbox": {
                "gpus": 0 if initial_gpu else 1,
                "gpu_types": None if initial_gpu else ["T4"],
            }
        },
    )

    package = TaskPackage.from_task(task, sandbox=backend)

    expected_paths = set() if supported else {"sandbox.gpus"}
    assert package.view.config is task.config
    assert package.runtime_supported is supported
    assert {issue.path for issue in package.runtime_issues} == expected_paths
    payload = package.to_dict()
    assert payload["runtime_supported"] is supported
    assert {issue["path"] for issue in payload["runtime_issues"]} == expected_paths
    assert package.document is document
    if document is not None:
        assert document.config.model_dump() == original
    assert authored.view.config.model_dump() == original
    assert package.view.source_hashes == authored.view.source_hashes


def test_task_package_overlay_keeps_document_policy_diagnostics(
    tmp_path: Path,
) -> None:
    """Guards document policy preservation while repairing PR #7's facade."""
    task = _task(tmp_path, "markdown", False)
    assert task.document is not None
    task.document = replace(task.document, benchflow={"runtime_policy": {}})
    document = task.document
    task.config = apply_config_override(
        task.config, {"sandbox": {"gpus": 1, "gpu_types": ["T4"]}}
    )

    package = TaskPackage.from_task(task, sandbox="modal")

    assert package.document is document
    assert document.config.sandbox.gpus == 0
    assert package.view.config.sandbox.gpus == 1
    assert package.runtime_supported is False
    assert {issue.path for issue in package.runtime_issues} == {
        "benchflow.runtime_policy"
    }
    payload = package.to_dict()
    assert payload["runtime_supported"] is False
    assert {issue["path"] for issue in payload["runtime_issues"]} == {
        "benchflow.runtime_policy"
    }
