"""A compose build that cannot reach the Docker daemon is a sandbox startup failure.

With the daemon down, the rollout logged a traceback that embedded the whole
`docker compose` command and recorded `error_category: other`. Programmatic
callers (and a daemon that stops mid-job) skip the CLI preflight, so the
rollout itself must name the cause.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow._utils.scoring import SANDBOX_SETUP, classify_error
from benchflow.evaluation import RetryConfig
from benchflow.rollout._setup import _start_env_and_upload
from benchflow.sandbox.protocol import SandboxStartupError

# Shape of a compose error, shortened.
_COMPOSE_DAEMON_DOWN = (
    "Docker compose command failed for environment good. Command: docker compose "
    "--project-name good__00000001 --project-directory /tmp/tasks/good/environment "
    "-f /x/docker-compose-base.yaml -f /x/docker-compose-build.yaml build. "
    'Return code: 1. Stdout: time="2026-01-01T00:00:00Z" level=warning '
    'msg="Docker Compose requires buildx plugin to be installed"\n'
    " Image bf__good Building \n"
    "failed to connect to the docker API at unix:///tmp/none.sock; check if the "
    "path is correct and if the daemon is running: dial unix /tmp/none.sock: "
    "connect: no such file or directory\n. Stderr: None. "
)
_COMPOSE_COPY_MISSING = (
    "Docker compose command failed for environment missing-ctx. Command: docker "
    "compose build. Return code: 1. Stdout: COPY failed: file not found in build "
    "context or excluded by .dockerignore: stat data/: file does not exist\n"
)


class _FailingStartEnv:
    def __init__(self, message: str) -> None:
        self.message = message

    async def start(self, force_build: bool) -> None:
        raise RuntimeError(self.message)


def _task(tmp_path: Path) -> Path:
    task = tmp_path / "good"
    task.mkdir()
    (task / "instruction.md").write_text("hi\n")
    return task


@pytest.mark.asyncio
async def test_unreachable_daemon_becomes_a_short_sandbox_startup_error(tmp_path):
    with pytest.raises(SandboxStartupError) as info:
        await _start_env_and_upload(
            _FailingStartEnv(_COMPOSE_DAEMON_DOWN), _task(tmp_path), {}
        )

    message = str(info.value)
    assert message.startswith("Docker daemon unreachable: failed to connect")
    assert "unix:///tmp/none.sock" in message
    assert "bench doctor" in message
    assert "--project-name" not in message
    assert isinstance(info.value.__cause__, RuntimeError)
    # The rollout records it as "Sandbox startup failed: <message>".
    recorded = f"Sandbox startup failed: {message}"
    assert classify_error(recorded) == SANDBOX_SETUP
    # Retrying seconds later meets the same stopped daemon (and each attempt
    # logs a failed `compose down`), so it is not retried.
    assert RetryConfig().should_retry(recorded) is False


@pytest.mark.asyncio
async def test_other_compose_failures_are_left_alone(tmp_path):
    with pytest.raises(RuntimeError) as info:
        await _start_env_and_upload(
            _FailingStartEnv(_COMPOSE_COPY_MISSING), _task(tmp_path), {}
        )

    assert not isinstance(info.value, SandboxStartupError)
    assert str(info.value) == _COMPOSE_COPY_MISSING


# Teardown after such a failed start. Regression test (the failed
# `compose down` line): with the daemon down, stop()
# logged "Docker compose down hung/failed (<the whole compose command>);
# force-killing project" although nothing had been started.

_DAEMON_LINE = (
    "failed to connect to the docker API at unix:///tmp/none.sock; check if the "
    "path is correct and if the daemon is running: dial unix /tmp/none.sock: "
    "connect: no such file or directory"
)


def _docker_env(tmp_path: Path):
    from unittest.mock import AsyncMock

    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task.config import SandboxConfig

    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM alpine:3.20\n")
    env = DockerSandbox(environment, "good", "good__00000001", None, SandboxConfig())
    env._force_kill_project = AsyncMock()
    env._probe_verifier_log_mount = AsyncMock()
    return env


def _compose(fail: dict[str, BaseException]):
    """A compose runner whose listed subcommands raise, the rest succeed."""
    from benchflow.sandbox._base import ExecResult

    calls: list[str] = []

    async def run(command, check=True, timeout_sec=None):
        calls.append(command[0])
        error = fail.get(command[0])
        if error is not None:
            raise error
        return ExecResult(stdout="", stderr="", return_code=0)

    return run, calls


def _daemon_down() -> RuntimeError:
    return RuntimeError(
        "Docker compose command failed for environment good. Command: docker "
        f"compose --project-name good__00000001 down. Return code: 1. Stdout: "
        f"{_DAEMON_LINE}\n. Stderr: None. "
    )


def _no_docker_cli() -> FileNotFoundError:
    return FileNotFoundError(2, "No such file or directory", "docker")


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", [_daemon_down, _no_docker_cli])
async def test_teardown_after_docker_was_unavailable_at_start_is_quiet(
    tmp_path, caplog, unavailable
):
    import logging

    env = _docker_env(tmp_path)
    run, calls = _compose(
        {"build": unavailable(), "down": unavailable(), "exec": unavailable()}
    )
    env._run_docker_compose_command = run

    with pytest.raises((RuntimeError, FileNotFoundError)):
        await env.start(force_build=False)
    with caplog.at_level(logging.DEBUG, logger="benchflow"):
        await env.stop(delete=True)

    assert [r.message for r in caplog.records if r.levelno >= logging.WARNING] == []
    teardown = [r for r in caplog.records if "teardown" in r.message.lower()]
    assert [r.levelno for r in teardown] == [logging.DEBUG]
    assert "--project-name" not in teardown[0].message
    # No container was started, so there is nothing to chown or force-kill.
    assert "exec" not in calls
    env._force_kill_project.assert_not_awaited()


@pytest.mark.asyncio
async def test_teardown_failure_after_compose_up_stays_loud(tmp_path, caplog):
    """Containers may exist once `compose up` ran: a daemon that went away
    mid-run leaves them behind, so the failure stays a warning and the
    project is still force-killed."""
    import logging

    env = _docker_env(tmp_path)
    run, _ = _compose({"down": _daemon_down()})
    env._run_docker_compose_command = run

    await env.start(force_build=False)
    with caplog.at_level(logging.WARNING, logger="benchflow"):
        await env.stop(delete=True)

    warnings = [r.message for r in caplog.records if "compose down" in r.message]
    assert len(warnings) == 1 and "force-killing" in warnings[0]
    env._force_kill_project.assert_awaited_once()


@pytest.mark.asyncio
async def test_other_teardown_failures_before_compose_up_stay_loud(tmp_path, caplog):
    """A start that failed for another reason, then a teardown that hangs,
    points at a wedged daemon rather than an absent one: still a warning."""
    import logging

    env = _docker_env(tmp_path)
    run, _ = _compose(
        {
            "build": RuntimeError("Docker compose command failed: bad Dockerfile"),
            "down": RuntimeError("Command timed out after 120 seconds"),
        }
    )
    env._run_docker_compose_command = run

    with pytest.raises(RuntimeError):
        await env.start(force_build=False)
    with caplog.at_level(logging.WARNING, logger="benchflow"):
        await env.stop(delete=True)

    warnings = [r.message for r in caplog.records if "compose down" in r.message]
    assert len(warnings) == 1 and "timed out" in warnings[0]
    env._force_kill_project.assert_awaited_once()
