"""Sandbox user setup on images without useradd (Alpine / busybox).

Harbor's ``hello-alpine`` example (``FROM alpine`` plus bash) has busybox
``adduser`` but no ``useradd``. setup_sandbox_user chained ``useradd`` in front
of every other step and ignored the exit code, so on Alpine no user was
created, the log still said the user was ready, and the failure surfaced only
later as "Verifier hardening failed: sandbox-user process identity check"
(``id -u agent`` exiting 2), which points at the verifier, not at user setup.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox.lockdown import setup_sandbox_user


def _result(rc: int, stderr: str = "") -> MagicMock:
    return MagicMock(return_code=rc, stdout="", stderr=stderr)


@pytest.mark.asyncio
async def test_setup_falls_back_to_busybox_adduser() -> None:
    env = MagicMock()
    env.exec = AsyncMock(return_value=_result(0))

    await setup_sandbox_user(env, "agent", "/app")

    cmd = env.exec.call_args_list[0].args[0]
    assert "useradd -m -s /bin/bash agent || adduser -D -s /bin/bash agent" in cmd


@pytest.mark.asyncio
async def test_setup_raises_when_the_user_could_not_be_created() -> None:
    env = MagicMock()
    env.exec = AsyncMock(
        side_effect=[
            _result(127, "sh: useradd: not found\nsh: adduser: not found"),
            _result(1),  # id -u agent: no such user
        ]
    )

    with pytest.raises(RuntimeError, match="could not create sandbox user 'agent'"):
        await setup_sandbox_user(env, "agent", "/app")


@pytest.mark.asyncio
async def test_setup_tolerates_a_later_step_failing_once_the_user_exists() -> None:
    env = MagicMock()
    env.exec = AsyncMock(side_effect=[_result(1, "chown: read-only"), _result(0)])

    assert await setup_sandbox_user(env, "agent", "/app") == "/app"
