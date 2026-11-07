"""A slow first image build shows progress (DF-p1-first-run-6).

The first run of a task builds its image (45 to 48 s for SkillsBench
citation-check in the dogfood, "a couple of minutes" by the docs). A TTY showed
only "creating sandbox…" and a log nothing after "Starting environment". A
build still running after a few seconds is now announced in the log, with a
line every 30 s and one when it ends, and the live table shows
"building image…" then "starting sandbox…". A cached rebuild stays quiet.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from benchflow._utils.live_activity import ActivitySnapshot
from benchflow.cli._live_progress import _activity_cell
from benchflow.rollout import Rollout
from benchflow.sandbox import docker as d
from tests.test_docker_build_identity import sandbox


def _messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_a_slow_build_is_announced_with_a_heartbeat_and_its_time(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(d, "_BUILD_ANNOUNCE_AFTER_SEC", 0.05, raising=False)
    monkeypatch.setattr(d, "_BUILD_HEARTBEAT_SEC", 0.05, raising=False)
    env = sandbox(tmp_path, "docker")
    states: list[str] = []

    async def slow_build(args, check=True, timeout_sec=None):
        for _ in range(4):
            await asyncio.sleep(0.05)
            states.append(getattr(env, "image_build", ""))

    env._run_docker_compose_command = slow_build
    with caplog.at_level(logging.INFO):
        await env._run_docker_compose_build()

    messages = _messages(caplog)
    name = env.environment_name
    assert any(
        m.startswith(f"Building the task image for {name}: the first run of a task")
        and "later runs reuse it" in m
        for m in messages
    ), messages
    assert any(
        m.startswith(f"Still building the task image for {name}") for m in messages
    )
    assert any(m.startswith(f"Built the task image for {name} in ") for m in messages)
    assert "building" in states
    assert getattr(env, "image_build", "") == "built"


@pytest.mark.asyncio
async def test_a_cached_rebuild_stays_quiet(tmp_path, caplog):
    env = sandbox(tmp_path, "docker")

    async def quick_build(args, check=True, timeout_sec=None):
        return None

    env._run_docker_compose_command = quick_build
    with caplog.at_level(logging.INFO):
        await env._run_docker_compose_build()

    assert not [m for m in _messages(caplog) if "task image" in m]
    assert getattr(env, "image_build", "") == ""


def test_the_live_table_shows_the_build_then_the_start():
    def cell(build: str) -> str:
        rollout = SimpleNamespace(
            _acp_client=None, _phase="setup", _env=SimpleNamespace(image_build=build)
        )
        return _activity_cell(Rollout.activity_snapshot(rollout))

    assert cell("") == "creating sandbox…"
    assert cell("building") == "building image…"
    assert cell("built") == "starting sandbox…"
    # Once the agent stage starts, the rollout phase wins again.
    started = SimpleNamespace(
        _acp_client=None, _phase="started", _env=SimpleNamespace(image_build="built")
    )
    assert Rollout.activity_snapshot(started) == ActivitySnapshot("started", None)
