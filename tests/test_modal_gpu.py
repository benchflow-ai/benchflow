"""Guard Modal GPU admission and allocation against release baseline 67753c4ac.

All provider operations are fakes. These tests establish request propagation,
not GPU availability or successful execution on Modal hardware.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.modal_impl import ModalSandbox
from benchflow.sandbox.setup import (
    _create_benchflow_modal_environment_class,
    _create_environment,
)
from benchflow.task import Task, TaskConfig, UnsupportedTaskFeatureError
from benchflow.task.config import SandboxConfig
from benchflow.task.runtime_capabilities import validate_task_runtime_support


@pytest.fixture
def provider(monkeypatch):
    """Replace the optional SDK, including its final allocation boundary."""
    module = ModuleType("modal")
    module.App = SimpleNamespace(lookup=SimpleNamespace(aio=AsyncMock()))
    module.Image = MagicMock()
    module.Secret = MagicMock()
    module.Volume = MagicMock()
    # The current Sandbox.filesystem API only: Modal refuses the
    # legacy Sandbox.mkdir the launchers used.
    remote = SimpleNamespace(
        filesystem=SimpleNamespace(make_directory=SimpleNamespace(aio=AsyncMock()))
    )
    create = AsyncMock(return_value=remote)
    module.Sandbox = SimpleNamespace(create=SimpleNamespace(aio=create))
    monkeypatch.setitem(sys.modules, "modal", module)
    monkeypatch.setattr(
        "benchflow.sandbox.modal_impl.bfx_delivery.provider_enabled",
        lambda name: False,
    )
    return module


def _sandbox(tmp_path, *, factory, config, override_gpus=None):
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    cls = _create_benchflow_modal_environment_class() if factory else ModalSandbox
    sandbox = cls(
        environment_dir=tmp_path,
        environment_name="gpu-task",
        session_id="gpu-trial",
        rollout_paths=None,
        task_env_config=config,
        override_gpus=override_gpus,
    )
    sandbox._refuse_startup_names = AsyncMock()
    sandbox.exec = AsyncMock(
        return_value=ExecResult(stdout="", stderr="", return_code=0)
    )
    return sandbox


@pytest.mark.parametrize("factory", [False, True], ids=["direct", "factory"])
@pytest.mark.parametrize(
    ("gpus", "gpu_types", "override", "expected"),
    [
        (0, None, None, None),
        (1, None, None, "any:1"),
        (1, [], None, "any:1"),
        (1, ["T4"], None, "T4:1"),
        (2, ["A100-80GB"], None, "A100-80GB:2"),
        (1, ["H100"], 4, "H100:4"),
    ],
)
@pytest.mark.asyncio
async def test_both_launchers_deliver_effective_gpu_request(
    tmp_path, provider, factory, gpus, gpu_types, override, expected
):
    """Guards the two independent GPU launch paths present at 67753c4ac."""
    sandbox = _sandbox(
        tmp_path,
        factory=factory,
        config=SandboxConfig(gpus=gpus, gpu_types=gpu_types, cpus=2, memory_mb=4096),
        override_gpus=override,
    )
    await sandbox.start(force_build=False)

    create = provider.Sandbox.create.aio
    create.assert_awaited_once()
    assert create.await_args.kwargs["gpu"] == expected
    assert create.await_args.kwargs["cpu"] == 2
    assert create.await_args.kwargs["memory"] == 4096
    assert create.await_args.kwargs["name"] == "gpu-trial"
    assert sandbox._sandbox is create.return_value
    assert sandbox._sandbox.filesystem.make_directory.aio.await_count == 2


@pytest.mark.parametrize("factory", [False, True], ids=["direct", "factory"])
@pytest.mark.parametrize(
    ("gpus", "gpu_types", "override"),
    [
        (0, ["T4"], None),
        (1, ["T4", "A100"], None),
        (1, [""], None),
        (1, [" T4"], None),
        (1, ["H100:8"], None),
        (1, ["T4\n"], None),
        (1, ["T4"], 0),
        (1, ["T4"], -1),
        (1, ["T4"], True),
    ],
)
@pytest.mark.asyncio
async def test_bad_gpu_request_has_no_provider_effects(
    tmp_path, provider, factory, gpus, gpu_types, override
):
    """Guards silent truncation and ignored requests in 67753c4ac."""
    sandbox = _sandbox(
        tmp_path,
        factory=factory,
        config=SandboxConfig(gpus=gpus, gpu_types=gpu_types),
        override_gpus=override,
    )
    with pytest.raises(ValueError, match=r"GPU|gpu_types"):
        await sandbox.start(force_build=False)

    provider.App.lookup.aio.assert_not_awaited()
    provider.Image.from_registry.assert_not_called()
    provider.Image.from_aws_ecr.assert_not_called()
    provider.Image.from_dockerfile.assert_not_called()
    provider.Secret.from_name.assert_not_called()
    provider.Volume.from_name.assert_not_called()
    provider.Sandbox.create.aio.assert_not_awaited()


@pytest.mark.parametrize("count", [-1, True, False, 1.0, 1.5, "1"])
def test_gpu_count_is_an_explicit_nonnegative_integer(count):
    """Guards accidental GPU reservations from coercion at 67753c4ac."""
    with pytest.raises(ValidationError):
        SandboxConfig(gpus=count)


@pytest.mark.parametrize("types", [None, ["T4"], ["A100-80GB"]])
def test_modal_gpu_request_passes_capability_gate(types):
    """Guards the blanket GPU refusal in 67753c4ac."""
    config = TaskConfig(sandbox=SandboxConfig(gpus=1, gpu_types=types))
    assert validate_task_runtime_support(config, sandbox="modal") == []


@pytest.mark.parametrize(
    "gpus,types", [(0, ["T4"]), (1, ["T4", "A100"]), (1, ["T4:2"])]
)
def test_invalid_modal_gpu_request_is_reported_at_check(gpus, types):
    """Guards check/start disagreement in 67753c4ac."""
    config = TaskConfig(sandbox=SandboxConfig(gpus=gpus, gpu_types=types))
    issues = validate_task_runtime_support(config, sandbox="modal")
    assert len(issues) == 1
    assert issues[0].path == "sandbox.gpu_types"
    assert issues[0].sandbox == "modal"


@pytest.mark.parametrize(
    "backend", ["docker", "remote-docker", "daytona", "agentcore", "apple-container"]
)
def test_other_backends_keep_gpu_refusal(backend):
    """Guards unsupported backend admission while repairing 67753c4ac."""
    config = TaskConfig(sandbox=SandboxConfig(gpus=1, gpu_types=["T4"]))
    issues = validate_task_runtime_support(config, sandbox=backend)
    assert len(issues) == 1
    assert issues[0].path == "sandbox.gpus"
    assert "Modal" in issues[0].reason


def _task(tmp_path, kind):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    environment = task_dir / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n")
    tests = task_dir / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    if kind == "markdown":
        (task_dir / "task.md").write_text(
            '---\nschema_version: "1.3"\nsandbox:\n'
            "  gpus: 2\n  gpu_types: [A100-80GB]\n---\n\n"
            "## prompt\n\nUse the GPUs to solve the task.\n"
        )
    else:
        (task_dir / "task.toml").write_text(
            'schema_version = "1.3"\n[environment]\n'
            'gpus = 2\ngpu_types = ["A100-80GB"]\n'
        )
        (task_dir / "instruction.md").write_text("Use the GPUs to solve the task.\n")
    return Task(task_dir), task_dir


@pytest.mark.parametrize("kind", ["markdown", "legacy"])
@pytest.mark.asyncio
async def test_parsed_task_reaches_modal_allocation(
    tmp_path, provider, monkeypatch, kind
):
    """Guards task.md/task.toml launch refusal at 67753c4ac."""
    task, task_dir = _task(tmp_path, kind)
    monkeypatch.setattr(ModalSandbox, "preflight", lambda: None)
    sandbox = _create_environment("modal", task, task_dir, "gpu-task", MagicMock())
    sandbox._refuse_startup_names = AsyncMock()
    sandbox.exec = AsyncMock(
        return_value=ExecResult(stdout="", stderr="", return_code=0)
    )
    await sandbox.start(force_build=False)
    provider.Sandbox.create.aio.assert_awaited_once()
    assert provider.Sandbox.create.aio.await_args.kwargs["gpu"] == "A100-80GB:2"


@pytest.mark.parametrize("kind", ["markdown", "legacy"])
def test_gpu_task_refuses_before_loading_other_backend(tmp_path, monkeypatch, kind):
    """Preserves the no-provider-effects gate from 67753c4ac."""
    task, task_dir = _task(tmp_path, kind)
    backend = MagicMock()
    monkeypatch.setattr("benchflow.sandbox.docker.DockerSandbox", backend)
    with pytest.raises(UnsupportedTaskFeatureError, match="Modal"):
        _create_environment("docker", task, task_dir, "gpu-task", MagicMock())
    backend.assert_not_called()
