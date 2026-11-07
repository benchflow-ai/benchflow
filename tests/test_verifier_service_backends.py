"""A verifier in a compose service other than main runs wherever the backend can exec in that service.

Through 0.10.0rc1 the plan gate allowed ``verifier.service`` only on
``docker`` and ``remote-docker``, by name, so every embodied task (its verifier runs in the trusted ``simulator``
service) was refused on Daytona before a sandbox started, although Daytona's Docker-in-Docker strategy execs,
uploads and downloads in any compose service. The gate now reads ``runs_verifier_in_service`` off the provider
registry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.sandbox.providers import (
    PROVIDERS_BY_NAME,
    SANDBOX_PROVIDERS,
    VERIFIER_SERVICE_PROVIDERS,
)
from benchflow.task import Task
from benchflow.task.runtime_capabilities import validate_task_runtime_support

TASK_TOML = """\
version = "1.0"

[metadata]
author_name = "test"

[verifier]
service = "{service}"
timeout_sec = 60

[agent]
timeout_sec = 60
"""


def _task(root: Path, service: str) -> Path:
    root.mkdir(parents=True)
    (root / "task.toml").write_text(TASK_TOML.format(service=service))
    (root / "instruction.md").write_text("Do the task.\n")
    env = root / "environment"
    env.mkdir()
    (env / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (env / "docker-compose.yaml").write_text(
        "services:\n  main: {}\n  simulator:\n    image: python:3.12-slim\n"
    )
    (root / "tests").mkdir()
    (root / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    return root


def _service_issues(task_dir: Path, sandbox: str) -> list[str]:
    config = Task(task_dir).config
    return [
        i.reason
        for i in validate_task_runtime_support(config, sandbox=sandbox)
        if i.path == "verifier.service"
    ]


def test_the_backends_that_run_a_verifier_in_a_service() -> None:
    assert {"docker", "remote-docker", "daytona"} == VERIFIER_SERVICE_PROVIDERS
    for name in VERIFIER_SERVICE_PROVIDERS:
        assert PROVIDERS_BY_NAME[name].supports_compose, name


@pytest.mark.parametrize("sandbox", SANDBOX_PROVIDERS)
def test_a_simulator_verifier_is_refused_only_where_no_service_can_run_it(
    tmp_path, sandbox
) -> None:
    issues = _service_issues(_task(tmp_path / "t", "simulator"), sandbox)
    if sandbox in VERIFIER_SERVICE_PROVIDERS:
        assert issues == []
    else:
        assert len(issues) == 1
        assert "docker, remote-docker, daytona" in issues[0]


@pytest.mark.parametrize("sandbox", SANDBOX_PROVIDERS)
def test_a_main_verifier_is_never_refused_for_its_service(tmp_path, sandbox) -> None:
    assert _service_issues(_task(tmp_path / "t", "main"), sandbox) == []
