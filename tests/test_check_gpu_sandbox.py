"""`bench check` without --sandbox checks a GPU task against a sandbox that runs one.

Guards the fix of the regression 46d7305b5 introduced for GPU tasks: the
check validated every task's plan with ``sandbox or "docker"``, and docker
refuses ``sandbox.gpus`` (runtime_capabilities), so `bench tasks check <task>`
with no --sandbox, as a benchmark's CI runs it, refused every GPU task that
BenchFlow 0.7.11's check passed. Without --sandbox a GPU task is now checked
against modal, the one sandbox that runs it, and the check says so; an explicit
``--sandbox docker`` still refuses it. A task without GPUs is checked against
docker, as before, with no extra line.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_ENV", raising=False)
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)


def _task(root: Path, environment: str = "") -> Path:
    root.mkdir(parents=True)
    for folder in ("environment", "verifier", "oracle"):
        (root / folder).mkdir()
    (root / "task.toml").write_text('version = "1.0"\n[environment]\n' + environment)
    (root / "instruction.md").write_text("Write the answer.\n")
    (root / "environment" / "Dockerfile").write_text("FROM alpine:3.20\n")
    (root / "verifier" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (root / "oracle" / "solve.sh").write_text("#!/bin/sh\nexit 0\n")
    return root


ASSUMED = "no --sandbox given: checking against modal"
REFUSED = "GPU task environments require the Modal sandbox backend"


@pytest.mark.parametrize("command", [["tasks", "check"], ["check"]])
@pytest.mark.parametrize("resources", ["gpus = 1\n", 'gpus = 1\ngpu_types = ["T4"]\n'])
def test_a_gpu_task_is_checked_against_modal_without_a_backend_flag(
    tmp_path, command, resources
):
    task = _task(tmp_path / "gpu", resources)
    result = CliRunner().invoke(app, [*command, str(task), "--agent", "oracle"])
    assert result.exit_code == 0, result.output
    assert ASSUMED in " ".join(result.stdout.split())
    assert REFUSED not in result.output


@pytest.mark.parametrize("command", [["tasks", "check"], ["check"]])
def test_an_explicit_docker_sandbox_still_refuses_a_gpu_task(tmp_path, command):
    task = _task(tmp_path / "gpu", "gpus = 1\n")
    result = CliRunner().invoke(
        app, [*command, str(task), "--agent", "oracle", "--sandbox", "docker"]
    )
    assert result.exit_code == 1, result.output
    assert REFUSED in " ".join(result.stderr.split())
    assert ASSUMED not in " ".join(result.stdout.split())


def test_the_runtime_capability_level_checks_a_gpu_task_against_modal(tmp_path):
    task = _task(tmp_path / "gpu", "gpus = 1\n")
    result = CliRunner().invoke(
        app,
        ["check", str(task), "--agent", "oracle", "--level", "runtime-capability"],
    )
    assert result.exit_code == 0, result.output
    assert ASSUMED in " ".join(result.stdout.split())


def test_a_task_without_gpus_is_still_checked_against_docker(tmp_path):
    task = _task(tmp_path / "cpu")
    result = CliRunner().invoke(app, ["check", str(task), "--agent", "oracle"])
    assert result.exit_code == 0, result.output
    assert "checking against" not in result.output


def test_the_assumed_sandbox_follows_the_tasks_gpu_request():
    from types import SimpleNamespace

    from benchflow.task.runtime_capabilities import GPU_SANDBOX, sandbox_for_task

    def config(**sandbox):
        return SimpleNamespace(sandbox=SimpleNamespace(**sandbox))

    assert GPU_SANDBOX == "modal"
    assert sandbox_for_task(config(gpus=0, gpu_types=None)) == "docker"
    assert sandbox_for_task(config(gpus=2, gpu_types=None)) == "modal"
    # A type without a count is still a GPU request, refused on docker.
    assert sandbox_for_task(config(gpus=0, gpu_types=["T4"])) == "modal"
