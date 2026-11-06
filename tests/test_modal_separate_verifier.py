"""Separate verifier sandboxes on Modal.

Guards the port of cb59bd70f's gate ("Modal now runs
separate verifiers") onto 0.9's separate-verifier design: before it, every
task with ``verifier.sandbox_mode: separate`` (such as one
whose verifier has its own L4) was refused on Modal before the run. On
Modal the verifier's sandbox starts only once the agent's has stopped and
Modal says it ended, as on Daytona.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("tenacity")

from benchflow.task import Task
from benchflow.task.verifier_sandbox import (
    SEPARATE_VERIFIER_SANDBOXES,
    separate_verifier_issues,
)

EN67_SHAPE = """---
schema_version: '1.3'
verifier:
  type: test-script
  sandbox_mode: separate
  timeout_sec: 3600
  sandbox:
    network_mode: no-network
    cpus: 8
    memory_mb: 32768
    gpus: 1
    gpu_types:
      - L4
agent:
  timeout_sec: 14400
sandbox:
  network_mode: public
  cpus: 8
  memory_mb: 32768
  gpus: 1
  gpu_types:
    - L4
---

## prompt

Do the work.
"""


def _task(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "task.md").write_text(EN67_SHAPE)
    for folder in ("environment", "verifier", "oracle"):
        (root / folder).mkdir()
    (root / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (root / "verifier" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (root / "verifier" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (root / "oracle" / "solve.sh").write_text("#!/bin/sh\nexit 0\n")
    return root


def test_modal_runs_separate_verifiers():
    assert "modal" in SEPARATE_VERIFIER_SANDBOXES


def test_a_separate_verifier_with_its_own_gpu_passes_the_modal_gate(tmp_path):
    """A declared no-network verifier sandbox with an L4."""
    config = Task(_task(tmp_path / "task")).config
    assert separate_verifier_issues(config, sandbox="modal") == []
    # The verifier sandbox's own GPU request is what Modal is asked for.
    from benchflow._utils.modal_gpu import modal_gpu_request
    from benchflow.task.verifier_sandbox import _verifier_sandbox_config

    verifier = _verifier_sandbox_config(config)
    assert modal_gpu_request(verifier.gpus, verifier.gpu_types) == "L4:1"


def test_apple_and_agentcore_still_refuse_separate_verifiers(tmp_path):
    config = Task(_task(tmp_path / "task")).config
    for sandbox in ("apple-container", "agentcore"):
        issues = separate_verifier_issues(config, sandbox=sandbox)
        assert any("implemented for daytona, modal and docker" in i for i in issues)


def test_modal_releases_the_agent_sandbox_before_the_verifier_starts(tmp_path):
    from benchflow.rollout import Rollout, RolloutConfig

    for environment, order in (
        ("modal", "after_stop"),
        ("daytona", "after_stop"),
        ("docker", "prestart"),
    ):
        rollout = Rollout(
            RolloutConfig(
                task_path=tmp_path / "task",
                agent="claude-agent-acp",
                environment=environment,
            )
        )
        rollout._release_agent_before_verify = True
        rollout._env = object()
        assert rollout._separate_verifier_order() == order, environment


class _Handle:
    object_id = "sb-agent"

    def __init__(self, code):
        self.poll = SimpleNamespace(aio=AsyncMock(return_value=code))


@pytest.mark.asyncio
async def test_the_stop_check_asks_modal_whether_the_agent_sandbox_ended():
    from benchflow.rollout._separate_verifier import agent_sandbox_left_running
    from benchflow.sandbox.modal_impl import ModalSandbox

    env = object.__new__(ModalSandbox)
    # Terminated: Modal reports an exit code (137 after a terminate).
    assert await agent_sandbox_left_running(env, _Handle(137)) is None
    running = await agent_sandbox_left_running(env, _Handle(None))
    assert running is not None and "sb-agent is still running" in running
    unknown = await agent_sandbox_left_running(env, None)
    assert unknown is not None and "could not confirm" in unknown


def test_the_registry_holds_which_backends_run_separate_verifiers():
    """The gate and the start order come from the provider registry (one
    place for these facts, tests/test_sandbox_provider_registry_drift.py)."""
    from benchflow.sandbox.providers import (
        PROVIDERS_BY_NAME,
        SEPARATE_VERIFIER_PROVIDERS,
    )

    assert SEPARATE_VERIFIER_SANDBOXES == SEPARATE_VERIFIER_PROVIDERS
    assert {"docker", "remote-docker", "daytona", "modal"} == set(
        SEPARATE_VERIFIER_PROVIDERS
    )
    starts = {p.name: p.separate_verifier_start for p in PROVIDERS_BY_NAME.values()}
    assert starts == {
        "docker": "prestart",
        "remote-docker": "prestart",
        "daytona": "after_stop",
        "modal": "after_stop",
        "apple-container": None,
        "agentcore": None,
    }
