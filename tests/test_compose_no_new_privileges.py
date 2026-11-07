"""no-new-privileges once on every service, whatever the task's own Compose file sets.

Guards the fix of the regression a0ac129e5 introduced: BenchFlow's base
Compose files (``docker-compose-base.yaml``, ``docker-compose-remote-base.yaml``)
added ``security_opt: [no-new-privileges:true]`` to ``main`` before the task's
own file, and Compose merges the two lists. A task that sets the option too,
as one benchmark's task does, made a list Compose
2.40 refuses ("services.main.security_opt items at 0 and 1 are equal"), so
`bench check` and every launch refused a task BenchFlow 0.7.11 ran. The option
is now set on the project Compose resolved, by identity: in the frozen launch
(Docker and remote Docker, every start) and the build overlay (Docker-in-Docker
as well); a task with no Compose file gets it from its own overlay. The
kept-service freezer's writable-cgroups is stacked before the build overlay,
whose list is the last word, so that option is not appended twice either.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest
import yaml

from benchflow.sandbox._compose import (
    COMPOSE_BASE_PATH,
    COMPOSE_KEPT_FREEZER_PATH,
    COMPOSE_NO_NEW_PRIVILEGES_PATH,
    COMPOSE_REMOTE_BASE_PATH,
    compose_file_paths,
)
from benchflow.sandbox._compose_runtime import NO_NEW_PRIVILEGES, with_no_new_privileges
from tests.test_compose_launch import _from_task
from tests.test_compose_oracle import _compose_available, _compose_task, _gate

SPELLINGS = [
    "no-new-privileges",
    "no-new-privileges:true",
    "no-new-privileges=true",
    "no-new-privileges:false",
    "no-new-privileges=false",
]

needs_compose = pytest.mark.skipif(
    not _compose_available(),
    reason="the gate runs `docker compose config`, which needs docker compose",
)


@pytest.fixture(autouse=True)
def _operator(monkeypatch, tmp_path):
    """The operator names nothing, has no .env, has not allowed host access,
    and runs from a scratch folder (Environment.from_task writes jobs/)."""
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "")
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)
    monkeypatch.delenv("DOCKER_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)


def _options(service: dict) -> list[str]:
    return [
        option
        for option in service.get("security_opt", [])
        if option.split("=", 1)[0].split(":", 1)[0] == "no-new-privileges"
    ]


@pytest.mark.parametrize("path", [COMPOSE_BASE_PATH, COMPOSE_REMOTE_BASE_PATH])
def test_the_base_files_set_no_security_option_before_the_tasks(path):
    main = yaml.safe_load(path.read_text())["services"]["main"]
    assert "security_opt" not in main


def test_a_task_without_compose_gets_the_option_from_its_own_overlay(tmp_path):
    from benchflow.task.config import SandboxConfig

    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM alpine:3.20\n")
    paths = compose_file_paths(environment, SandboxConfig())
    assert COMPOSE_NO_NEW_PRIVILEGES_PATH in paths
    main = yaml.safe_load(COMPOSE_NO_NEW_PRIVILEGES_PATH.read_text())["services"]
    assert main == {"main": {"security_opt": [NO_NEW_PRIVILEGES]}}
    (environment / "docker-compose.yaml").write_text("services:\n  main: {}\n")
    paths = compose_file_paths(environment, SandboxConfig())
    assert COMPOSE_NO_NEW_PRIVILEGES_PATH not in paths


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_every_spelling_is_replaced_by_one(spelling):
    assert with_no_new_privileges(["apparmor=docker-default", spelling]) == [
        "apparmor=docker-default",
        NO_NEW_PRIVILEGES,
    ]
    assert with_no_new_privileges(None) == [NO_NEW_PRIVILEGES]


@needs_compose
@pytest.mark.parametrize("spelling", SPELLINGS)
def test_a_task_that_sets_the_option_passes_the_gate_and_launches_with_it_once(
    tmp_path, spelling
):
    """twisted-slinky-wave-coupling's shape, in each spelling, beside a
    sidecar that sets nothing: the gate (`bench check`, the launch gate)
    resolves it, and the frozen launch gives each service the option once."""
    task = _compose_task(
        tmp_path / "task",
        "services:\n"
        "  main:\n"
        "    image: alpine:3.20\n"
        f'    security_opt: ["{spelling}", "apparmor=docker-default"]\n'
        "  side:\n"
        "    image: alpine:3.20\n",
    )
    _gate(task)
    sandbox = _from_task(task).inner
    frozen = json.loads(sandbox._launch.path.read_text())
    assert set(frozen["services"]) == {"main", "side"}
    for service in frozen["services"].values():
        assert _options(service) == [NO_NEW_PRIVILEGES]
    assert frozen["services"]["main"]["security_opt"] == [
        "apparmor=docker-default",
        NO_NEW_PRIVILEGES,
    ]


@needs_compose
def test_the_kept_service_freezer_and_the_build_overlay_add_nothing_twice(tmp_path):
    """A keeping start's second freeze stacks the build overlay, whose
    security_opt is read from a project that holds the freezer's
    writable-cgroups already: stacked after the freezer, nothing is appended
    to it again, and Compose 2.40 resolves the project."""
    task = _compose_task(
        tmp_path / "task",
        "services:\n"
        "  main:\n"
        '    security_opt: ["no-new-privileges:true"]\n'
        "  side:\n"
        "    image: alpine:3.20\n",
    )
    sandbox = _from_task(task).inner
    sandbox.kept_service_freezer = True  # what a keeping start decides
    sandbox.freeze_launch()
    asyncio.run(sandbox._prepare_compose_builds())
    try:
        inputs = sandbox._runtime_compose_inputs()
        assert inputs.index(COMPOSE_KEPT_FREEZER_PATH) < inputs.index(
            sandbox._build_compose_path
        )
        sandbox.freeze_launch()
        frozen = json.loads(sandbox._launch.path.read_text())
    finally:
        sandbox._build_compose_path.unlink(missing_ok=True)
    main = frozen["services"]["main"]["security_opt"]
    assert sorted(main) == sorted(["writable-cgroups=true", NO_NEW_PRIVILEGES])
    assert _options(frozen["services"]["side"]) == [NO_NEW_PRIVILEGES]


def test_dind_stacks_the_freezer_before_the_build_overlay(tmp_path):
    from benchflow.sandbox.daytona import _DaytonaDinD
    from benchflow.sandbox.setup import _create_sandbox_environment
    from benchflow.task import RolloutPaths, Task
    from tests.test_kept_service_freezer import _docker_task

    task_dir = _docker_task(tmp_path, True, compose=True)
    with (
        patch("benchflow.sandbox.daytona._load_daytona_sdk"),
        patch("benchflow.sandbox._sdk_ops.apply"),
    ):
        sandbox = _create_sandbox_environment(
            "daytona",
            Task(task_dir),
            task_dir,
            "rollout",
            RolloutPaths(rollout_dir=tmp_path / "rollout"),
        )
    strategy = sandbox._strategy
    assert isinstance(strategy, _DaytonaDinD)
    sandbox.kept_service_freezer = True
    strategy._build_compose_path = f"{_DaytonaDinD._COMPOSE_DIR}/builds.yaml"
    flags = strategy._compose_file_flags()[1::2]
    freezer = f"{_DaytonaDinD._COMPOSE_DIR}/{COMPOSE_KEPT_FREEZER_PATH.name}"
    assert flags.index(freezer) < flags.index(strategy._build_compose_path)
