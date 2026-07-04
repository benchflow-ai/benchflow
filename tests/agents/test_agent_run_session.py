"""Behavior tests for `bench agent run` sessions (headless run + resume)."""

import sys
from pathlib import Path

from benchflow.agents.session_store import SessionStore

FAKE_AGENT = Path(__file__).parent / "fake_acp_agent.py"


def fake_launch_cmd() -> str:
    return f"{sys.executable} {FAKE_AGENT}"


def test_created_session_can_be_reloaded_with_stored_state(tmp_path):
    store = SessionStore(root=tmp_path)

    rec = store.create(agent="opencode", model="deepseek/deepseek-v4-flash", cwd="/w")
    assert rec.session_id

    store.update(
        rec.session_id,
        acp_session_id="acp-abc",
        capabilities={"loadSession": True},
    )

    loaded = SessionStore(root=tmp_path).load(rec.session_id)
    assert loaded.agent == "opencode"
    assert loaded.model == "deepseek/deepseek-v4-flash"
    assert loaded.cwd == "/w"
    assert loaded.acp_session_id == "acp-abc"
    assert loaded.capabilities == {"loadSession": True}


def test_continue_finds_most_recent_session_for_cwd_only(tmp_path):
    store = SessionStore(root=tmp_path)
    old = store.create(agent="opencode", model="m", cwd="/proj")
    other = store.create(agent="opencode", model="m", cwd="/elsewhere")
    newer = store.create(agent="opencode", model="m", cwd="/proj")
    # touching a session makes it the most recent one
    store.update(newer.session_id, acp_session_id="acp-2")

    latest = store.latest_for_cwd("/proj")
    assert latest is not None and latest.session_id == newer.session_id

    assert store.latest_for_cwd("/nowhere") is None
    assert store.latest_for_cwd("/elsewhere").session_id == other.session_id
    assert old.session_id != newer.session_id


async def test_first_run_returns_reply_and_persists_resumable_session(tmp_path):
    from benchflow.agents.standalone import run_turn

    store = SessionStore(root=tmp_path / "store")
    work = tmp_path / "w"
    work.mkdir()

    result = await run_turn(
        agent="fake-agent",
        prompt="say hi",
        cwd=str(work),
        store=store,
        launch_cmd=fake_launch_cmd(),
        agent_env={"FAKE_ACP_LOG": str(tmp_path / "log.jsonl")},
    )

    assert "hello from fake" in result.text
    assert result.stop_reason == "end_turn"

    rec = store.load(result.session_id)
    assert rec.acp_session_id == "fake-sess-1"
    assert rec.capabilities.get("loadSession") is True


async def test_resume_loads_persisted_session_instead_of_creating(tmp_path):
    import json

    from benchflow.agents.standalone import run_turn

    store = SessionStore(root=tmp_path / "store")
    work = tmp_path / "w"
    work.mkdir()
    log = tmp_path / "log.jsonl"
    env = {"FAKE_ACP_LOG": str(log)}

    first = await run_turn(
        agent="fake-agent",
        prompt="hi",
        cwd=str(work),
        store=store,
        launch_cmd=fake_launch_cmd(),
        agent_env=env,
    )
    second = await run_turn(
        agent="fake-agent",
        prompt="again",
        cwd=str(work),
        store=store,
        resume=first.session_id,
        launch_cmd=fake_launch_cmd(),
        agent_env=env,
    )

    assert second.session_id == first.session_id
    assert "hello from fake" in second.text

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    methods = [c["method"] for c in calls]
    resumed_run = methods[methods.index("session/prompt") + 1 :]
    assert "session/load" in resumed_run
    assert "session/new" not in resumed_run

    load = next(c for c in calls if c["method"] == "session/load")
    assert load["params"]["sessionId"] == "fake-sess-1"
    assert load["params"]["cwd"] == str(work)


async def test_resume_fails_actionably_when_agent_cannot_load_sessions(tmp_path):
    import pytest

    from benchflow.agents.standalone import ResumeUnsupportedError, run_turn

    store = SessionStore(root=tmp_path / "store")
    work = tmp_path / "w"
    work.mkdir()
    env = {"FAKE_LOADSESSION": "0"}

    first = await run_turn(
        agent="fake-agent",
        prompt="hi",
        cwd=str(work),
        store=store,
        launch_cmd=fake_launch_cmd(),
        agent_env=env,
    )

    with pytest.raises(ResumeUnsupportedError) as exc:
        await run_turn(
            agent="fake-agent",
            prompt="again",
            cwd=str(work),
            store=store,
            resume=first.session_id,
            launch_cmd=fake_launch_cmd(),
            agent_env=env,
        )
    assert "fake-agent" in str(exc.value)
    assert "loadSession" in str(exc.value)
