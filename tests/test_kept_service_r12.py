"""K1's round-12 review: Compose file names and runner-aware freezer setup.

K1-R12-1: provider admission recognised only ``environment/docker-compose.yaml``
as a Daytona Compose task, while Daytona selects DinD (which gives the freezer)
for any of Compose's file names; the other three were refused.

K1-R12-2: the rollout names what runs agent code
(an oracle or an agent); nop names nothing and never asks for the freezer.
(The start's decision is tests/test_kept_service_freezer_request.py.)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchflow.task.plan_validation import validate_task_plan

COMPOSE_NAMES = (
    "docker-compose.yaml",
    "docker-compose.yml",
    "compose.yaml",
    "compose.yml",
)


def _task(root: Path, compose: str | None = None) -> Path:
    task_dir = root / "keeps"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "solution").mkdir()
    (task_dir / "instruction.md").write_text("Do it.\n")
    (task_dir / "solution" / "solve.sh").write_text("#!/bin/bash\ntrue\n")
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n'
        '[metadata]\n"x-keep-services" = [{ port = 5000, name = "api" }]\n'
        "[agent]\ntimeout_sec = 1\n"
        "[verifier]\ntimeout_sec = 1\n"
    )
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    if compose:
        (task_dir / "environment" / compose).write_text("services:\n  main: {}\n")
    return task_dir


@pytest.mark.parametrize("compose", COMPOSE_NAMES)
def test_every_compose_file_name_admits_a_daytona_agent_trial(tmp_path, compose):
    """K1-R12-1: no keep-services refusal for a Daytona Compose task, by any
    of Compose's file names (another check may still stop the plan)."""
    try:
        validate_task_plan(
            _task(tmp_path, compose),
            sandbox="daytona",
            agent="claude-agent-acp",
            sandbox_user="agent",
        )
    except ValueError as exc:
        assert "keeps services" not in str(exc)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent, runner", [("nop", None), ("oracle", "oracle")])
async def test_the_rollout_names_the_runner(tmp_path, agent, runner):
    """nop asks for no freezer; an oracle's sandbox start decides one."""
    from benchflow.rollout import Rollout, RolloutConfig

    env = MagicMock()
    env.kept_service_runner = None
    planes = MagicMock()
    planes.resolve_locked_paths.return_value = []
    planes.resolve_agent_env.return_value = {}
    planes.agent_launch.return_value = agent
    planes.create_environment.return_value = env
    rollout = Rollout(
        RolloutConfig.from_legacy(
            task_path=_task(tmp_path),
            agent=agent,
            jobs_dir=tmp_path / "jobs",
            planes=planes,
        )
    )
    await rollout.setup()
    assert env.kept_service_runner == runner
