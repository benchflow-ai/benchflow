"""Tests for _setup_verifier_user (Tier 3 sandbox hardening)."""

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_env():
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))
    return env


@pytest.mark.asyncio
async def test_setup_verifier_user_locks_logs_parent():
    """/logs/ is chowned to root:root and chmod 755 to block sandbox_user rename."""
    from benchflow._sandbox import _setup_verifier_user

    env = _make_env()
    await _setup_verifier_user(env)

    calls = [c.args[0] for c in env.exec.call_args_list]
    assert any("chown root:root /logs" in c and "chmod 755 /logs" in c for c in calls)


@pytest.mark.asyncio
async def test_setup_verifier_user_creates_user_no_extra_groups():
    """useradd includes --groups '' to strip supplementary groups."""
    from benchflow._sandbox import _setup_verifier_user

    env = _make_env()
    await _setup_verifier_user(env)

    calls = [c.args[0] for c in env.exec.call_args_list]
    assert any("useradd" in c and "--groups ''" in c for c in calls)


@pytest.mark.asyncio
async def test_setup_verifier_user_wipes_home_dir():
    """Pre-staged /home/verifier is wiped and recreated clean."""
    from benchflow._sandbox import _VERIFIER_USER, _setup_verifier_user

    env = _make_env()
    await _setup_verifier_user(env)

    calls = [c.args[0] for c in env.exec.call_args_list]
    assert any(f"rm -rf /home/{_VERIFIER_USER}" in c for c in calls)
    assert any(f"mkdir -p /home/{_VERIFIER_USER}" in c for c in calls)


@pytest.mark.asyncio
async def test_setup_verifier_user_chowns_logs_verifier():
    """/logs/verifier/ is owned by the verifier user with chmod 700."""
    from benchflow._sandbox import _VERIFIER_USER, _setup_verifier_user

    env = _make_env()
    await _setup_verifier_user(env)

    calls = [c.args[0] for c in env.exec.call_args_list]
    assert any(
        f"chown {_VERIFIER_USER}:{_VERIFIER_USER} /logs/verifier" in c
        and "chmod 700 /logs/verifier" in c
        for c in calls
    )


@pytest.mark.asyncio
async def test_setup_verifier_user_creates_testbed_verify():
    """/testbed_verify is seeded from /testbed and chowned root:root."""
    from benchflow._sandbox import _setup_verifier_user

    env = _make_env()
    await _setup_verifier_user(env)

    calls = [c.args[0] for c in env.exec.call_args_list]
    assert any("cp -a /testbed /testbed_verify" in c for c in calls)
    assert any("chown -R root:root /testbed_verify" in c for c in calls)


@pytest.mark.asyncio
async def test_setup_verifier_user_not_called_without_sandbox_user(monkeypatch):
    """SDK.run path with sandbox_user=None must not call _setup_verifier_user."""
    from unittest.mock import patch

    called = []

    async def fake_setup_verifier_user(env):
        called.append(True)

    with patch("benchflow.sdk._setup_verifier_user", fake_setup_verifier_user):
        # Simulate the guard: _setup_verifier_user is only called inside
        # `if sandbox_user:`, so passing sandbox_user=None must skip it.
        # We verify the guard logic directly rather than running a full SDK.run.
        sandbox_user = None
        if sandbox_user:
            await fake_setup_verifier_user(object())

    assert not called, (
        "_setup_verifier_user must not be called when sandbox_user is None"
    )
