"""The native floor seeds the service IN-SANDBOX (env-0 contract), never as a
host subprocess — and the orchestrator reaches it on localhost."""

from __future__ import annotations

import subprocess
import textwrap

import pytest

from benchflow.arena import bootstrap as bs


class _FakeEnv:
    """Stands in for ManifestEnvironment — records the in-sandbox lifecycle."""

    def __init__(self):
        self.calls = []

    async def provision(self, ctx):
        self.calls.append(("provision", ctx))

    async def readiness(self):
        self.calls.append(("readiness", None))

    async def teardown(self):
        self.calls.append(("teardown", None))


class _FakeSandbox:
    def __init__(self):
        self.execs = []

    async def exec(self, cmd, *, user="root", timeout_sec=30):
        self.execs.append(cmd)

    async def stop(self, delete=False):
        self.execs.append(("stop", delete))


def _write_manifest(tmp_path):
    p = tmp_path / "environment.toml"
    p.write_text(textwrap.dedent("""
        [environment]
        name = "casinobench"
        image = "casinobench-base:latest"
        owns_lifecycle = false
        [[environment.services]]
        name = "casino"
        command = "casino-service"
        port = 9001
        health_path = "/health"
    """))
    return p


@pytest.mark.asyncio
async def test_bootstrap_starts_service_in_sandbox_on_localhost(tmp_path, monkeypatch):
    # No subprocess.Popen may be spawned — the service is in-sandbox.
    monkeypatch.setattr(subprocess, "Popen", _no_popen)
    env = _FakeEnv()
    sandbox = _FakeSandbox()

    out_sandbox, service_url, teardown = await bs.bootstrap_shared_env(
        _write_manifest(tmp_path), game="six-deck-blackjack-s17",
        _sandbox=sandbox, _env=env,
    )

    assert service_url == "http://localhost:9001"        # in-sandbox localhost, no bridge
    assert out_sandbox is sandbox
    assert ("provision", {"task_id": "six-deck-blackjack-s17"}) in env.calls
    assert ("readiness", None) in env.calls
    await teardown()
    assert ("teardown", None) in env.calls


@pytest.mark.asyncio
async def test_service_env_injected_before_provision(tmp_path):
    env, sandbox = _FakeEnv(), _FakeSandbox()
    await bs.bootstrap_shared_env(
        _write_manifest(tmp_path), service_env={"CASINO_MULTIPLAYER": "1"},
        _sandbox=sandbox, _env=env,
    )
    assert any("CASINO_MULTIPLAYER=1" in c for c in sandbox.execs if isinstance(c, str))


def _no_popen(*a, **k):
    raise AssertionError("bootstrap must NOT spawn a host subprocess — service is in-sandbox")
