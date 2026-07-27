"""The native floor seeds the service IN-SANDBOX (env-0 contract), never as a
host subprocess — and the orchestrator reaches it on localhost."""

from __future__ import annotations

import json
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
    p.write_text(
        textwrap.dedent("""
        [environment]
        name = "casinobench"
        image = "casinobench-base:latest"
        owns_lifecycle = false
        [[environment.services]]
        name = "casino"
        command = "casino-service"
        port = 9001
        health_path = "/health"
    """)
    )
    return p


@pytest.mark.asyncio
async def test_bootstrap_starts_service_in_sandbox_on_localhost(tmp_path, monkeypatch):
    # No subprocess.Popen may be spawned — the service is in-sandbox.
    monkeypatch.setattr(subprocess, "Popen", _no_popen)
    env = _FakeEnv()
    sandbox = _FakeSandbox()

    out_sandbox, service_url, teardown = await bs.bootstrap_shared_env(
        _write_manifest(tmp_path),
        game="six-deck-blackjack-s17",
        _sandbox=sandbox,
        _env=env,
    )

    assert service_url == "http://localhost:9001"  # in-sandbox localhost, no bridge
    assert out_sandbox is sandbox
    assert ("provision", {"task_id": "six-deck-blackjack-s17"}) in env.calls
    assert ("readiness", None) in env.calls
    await teardown()
    assert ("teardown", None) in env.calls


def test_service_env_goes_into_sandbox_persistent_env(tmp_path, monkeypatch):
    # CASINO_MULTIPLAYER (floor mode) must land in the sandbox's persistent env so
    # ManifestEnvironment's own `nohup casino-service` exec inherits it — not a
    # throwaway `export` exec.
    captured = {}

    class _FakeDocker:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr("benchflow.sandbox.docker.DockerSandbox", _FakeDocker)
    bs._make_sandbox("img:latest", tmp_path, "docker", {"CASINO_MULTIPLAYER": "1"})
    cfg = captured["task_env_config"]
    assert cfg.env["CASINO_MULTIPLAYER"] == "1"
    assert cfg.docker_image == "img:latest"


def _no_popen(*a, **k):
    raise AssertionError(
        "bootstrap must NOT spawn a host subprocess — service is in-sandbox"
    )


def test_count_actions_reads_real_casino_actions_per_seat():
    # floor.json "moves" counted ACP tool calls; the REAL activity metric is
    # action_applied events per actor (opencode showed 77 calls but 1066 acts).
    rows = [
        {"type": "action_applied", "actor": "a", "seq": 1},
        {"type": "action_applied", "actor": "a", "seq": 2},
        {"type": "action_applied", "actor": "b", "seq": 3},
        {"type": "action_timeout", "actor": "a", "seq": 4},
        {"type": "settlement", "actor": "", "seq": 5},
    ]
    jsonl = "\n".join(json.dumps(r) for r in rows)
    assert bs._count_actions(jsonl) == {
        "a": {"actions": 2, "timeouts": 1},
        "b": {"actions": 1, "timeouts": 0},
    }
