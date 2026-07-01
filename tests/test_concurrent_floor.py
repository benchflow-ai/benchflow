"""Shared-sandbox concurrent floor: fan-out, separate trajectories, drive modes.

The floor consumes a decoupled ``Roster`` (agents only) + a ``FloorConfig`` (the
run-level params that come from CLI flags) — not a task-coupled manifest."""

from __future__ import annotations

import json
import textwrap

import pytest

from benchflow.arena import concurrent_floor as cf
from benchflow.arena.agent_driver import SeatConn
from benchflow.arena.concurrent_floor import FloorConfig
from benchflow.arena.roster import Roster


class _FakeSandbox:
    def __init__(self):
        self.execs, self.uploads = [], []

    async def exec(self, cmd, *, user="root", timeout_sec=30):
        self.execs.append(cmd)

    async def upload_file(self, src, dst):
        self.uploads.append((str(src), dst))


class _FakeSession:
    on_change = None


def _patch_driver(monkeypatch, traj_by_agent):
    """connect_seat → a fake conn; prompt_seat → that agent's scripted trajectory."""
    async def fake_connect_seat(cfg, **kw):
        return SeatConn(protocol=cfg.protocol, session=_FakeSession(), client=None,
                        name=cfg.name)

    async def fake_prompt_seat(conn, prompt, *, timeout, idle_timeout=None):
        traj = traj_by_agent.get(conn.name, [{"type": "tool_call"}])
        return traj, sum(1 for s in traj if s.get("type") == "tool_call")

    async def fake_close_seat(conn):
        pass

    monkeypatch.setattr(cf, "connect_seat", fake_connect_seat)
    monkeypatch.setattr(cf, "prompt_seat", fake_prompt_seat)
    monkeypatch.setattr(cf, "close_seat", fake_close_seat)


def _patch_subscription(monkeypatch):
    """Force the subscription (oauth, acp-only) path deterministically, regardless
    of host creds / API keys in the test environment."""
    async def fake_upload(sandbox, agent, home):
        pass
    monkeypatch.setattr(cf, "upload_subscription_auth", fake_upload)
    monkeypatch.setattr(cf, "uses_native_subscription_auth", lambda *a, **k: True)


@pytest.fixture
def make_roster(tmp_path):
    def _make(agents_body):
        (tmp_path / "claude.md").write_text("be bold")
        p = tmp_path / "roster.yaml"
        p.write_text(textwrap.dedent(agents_body))
        return Roster.from_yaml(p)
    return _make


@pytest.mark.asyncio
async def test_auto_loop_separate_acp_files_and_artifacts(make_roster, tmp_path, monkeypatch):
    roster = make_roster("""
        agents:
          - { name: claude, agent: claude-agent-acp, model: claude-sonnet-4-6, count: 2, instructions: claude.md }
          - { name: codex, agent: codex-acp, model: gpt-5.5 }
    """)
    cfg = FloorConfig(out=str(tmp_path / "out"), drive="auto-loop", prompt="play the game")
    _patch_driver(monkeypatch, {
        "claude-agent-acp": [{"type": "agent_message", "text": "hi"}, {"type": "tool_call"}],
        "codex-acp": [{"type": "tool_call"}],
    })
    _patch_subscription(monkeypatch)
    sb = _FakeSandbox()

    summary = await cf.run_concurrent_floor(roster, sandbox=sb, service_url="http://svc", config=cfg)

    out = tmp_path / "out"
    # fan-out: claude-0, claude-1, codex → 3 seats, 3 separate acp files
    seats = {r["seat"] for r in summary["results"]}
    assert seats == {"claude-0", "claude-1", "codex"}
    for seat in seats:
        traj = (out / seat / "trajectory" / "acp_trajectory.jsonl")
        assert traj.exists() and traj.read_text().strip(), seat
        # subscription seats → NO raw llm trajectory
        assert not (out / seat / "trajectory" / "llm_trajectory.jsonl").exists()
    # roster + floor written
    roster_json = json.loads((out / "roster.json").read_text())
    assert {r["seat"] for r in roster_json} == seats
    assert json.loads((out / "floor.json").read_text())["drive"] == "auto-loop"
    # instructions (CLAUDE.md) uploaded for the two claude seats only
    claude_uploads = [d for _, d in sb.uploads if d.endswith("CLAUDE.md")]
    assert sorted(claude_uploads) == ["/work/claude-0/CLAUDE.md", "/work/claude-1/CLAUDE.md"]


@pytest.mark.asyncio
async def test_proxy_seat_gets_separate_raw_llm_file(make_roster, tmp_path, monkeypatch):
    roster = make_roster("""
        agents:
          - { name: ds, agent: deepagents, model: deepseek-v4-pro }
    """)
    cfg = FloorConfig(out=str(tmp_path / "out"), prompt="play")
    _patch_driver(monkeypatch, {})

    class _RtTraj:
        def __init__(self):
            self.exchanges = [{"a": 1}, {"b": 2}]

        def to_jsonl(self, redact_keys=False):
            return '{"raw": 1}\n{"raw": 2}\n'

    class _Rt:
        server = type("S", (), {"trajectory": _RtTraj()})()

    async def fake_runtime(**kw):
        return {"BENCHFLOW_PROVIDER_BASE_URL": "http://proxy"}, _Rt()

    async def fake_stop(runtime):
        pass

    monkeypatch.setattr(cf, "ensure_litellm_runtime", fake_runtime)
    monkeypatch.setattr(cf, "stop_provider_runtime", fake_stop)
    sb = _FakeSandbox()

    summary = await cf.run_concurrent_floor(roster, sandbox=sb, service_url="http://svc", config=cfg)

    seat_dir = tmp_path / "out" / "ds" / "trajectory"
    assert (seat_dir / "acp_trajectory.jsonl").exists()
    llm = seat_dir / "llm_trajectory.jsonl"
    assert llm.exists() and llm.read_text() == '{"raw": 1}\n{"raw": 2}\n'
    assert summary["results"][0]["llm_calls"] == 2
    assert summary["results"][0]["raw"] is True


@pytest.mark.asyncio
async def test_subscription_capable_agent_with_api_key_uses_proxy(
    make_roster, tmp_path, monkeypatch
):
    # claude-agent-acp WITH an API key (uses_native_subscription_auth → False) must
    # take the proxy path (its own per-seat proxy, raw=True) — NOT be forced
    # subscription-only just because it *can* do oauth.
    roster = make_roster("""
        agents:
          - { name: claude, agent: claude-agent-acp, model: claude-sonnet-4-6 }
    """)
    cfg = FloorConfig(out=str(tmp_path / "out"), prompt="play")
    _patch_driver(monkeypatch, {})
    monkeypatch.setattr(cf, "uses_native_subscription_auth", lambda *a, **k: False)
    started = {}

    async def fake_runtime(**kw):
        started["session_id"] = kw.get("session_id")
        return {"BENCHFLOW_PROVIDER_BASE_URL": "http://proxy"}, object()  # truthy runtime

    async def fake_stop(rt):
        pass

    monkeypatch.setattr(cf, "ensure_litellm_runtime", fake_runtime)
    monkeypatch.setattr(cf, "stop_provider_runtime", fake_stop)

    summary = await cf.run_concurrent_floor(
        roster, sandbox=_FakeSandbox(), service_url="http://s", config=cfg)
    assert started["session_id"] == "floor-claude"  # its OWN per-seat proxy
    assert summary["results"][0]["raw"] is True


@pytest.mark.asyncio
async def test_one_bad_seat_does_not_kill_floor(make_roster, tmp_path, monkeypatch):
    roster = make_roster("""
        agents:
          - { name: good, agent: codex-acp }
          - { name: bad, agent: claude-agent-acp }
    """)
    cfg = FloorConfig(out=str(tmp_path / "out"), prompt="play")
    _patch_subscription(monkeypatch)

    async def fake_connect_seat(cfg_, **kw):
        if cfg_.name == "claude-agent-acp":
            raise RuntimeError("boom")
        return SeatConn(protocol="acp", session=_FakeSession(), client=None, name=cfg_.name)

    async def fake_prompt_seat(conn, prompt, *, timeout, idle_timeout=None):
        return [{"type": "tool_call"}], 1

    monkeypatch.setattr(cf, "connect_seat", fake_connect_seat)
    monkeypatch.setattr(cf, "prompt_seat", fake_prompt_seat)
    monkeypatch.setattr(cf, "close_seat", lambda conn: _noop())

    summary = await cf.run_concurrent_floor(
        roster, sandbox=_FakeSandbox(), service_url="http://s", config=cfg)
    by_seat = {r["seat"]: r for r in summary["results"]}
    assert by_seat["good"]["status"] == "ok"
    assert by_seat["bad"]["status"].startswith("error:")


async def _noop():
    pass


@pytest.mark.asyncio
async def test_service_rounds_nudges_only_on_your_turn(make_roster, tmp_path, monkeypatch):
    roster = make_roster("""
        agents:
          - { name: p, agent: codex-acp }
    """)
    cfg = FloorConfig(out=str(tmp_path / "out"), drive="service-rounds", prompt="take your turn")
    _patch_driver(monkeypatch, {"codex-acp": [{"type": "tool_call"}]})
    _patch_subscription(monkeypatch)  # codex seat → no real proxy in the test

    # service script: wait, your_turn, not_your_turn, your_turn, done
    script = iter([
        {"status": "waiting"},
        {"status": "your_turn", "request_id": "r1",
         "observation": {"public": {"chips": 50}}, "legal_actions": [{"a": "bet"}]},
        {"status": "not_your_turn"},
        {"status": "your_turn", "request_id": "r2",
         "observation": {"public": {"chips": 60}}, "legal_actions": [{"a": "fold"}]},
        {"status": "done"},
    ])
    observed = []

    class _FakeClient:
        def __init__(self, seat_id):
            self.seat = seat_id
        async def observe(self, seat_id):
            p = next(script)
            observed.append(p["status"])
            return p
        async def act(self, *a, **k):
            return {}

    prompts_sent = []

    async def fake_prompt_seat(conn, prompt, *, timeout, idle_timeout=None):
        prompts_sent.append(prompt)
        return [{"type": "tool_call"}], 1

    monkeypatch.setattr(cf, "prompt_seat", fake_prompt_seat)

    summary = await cf.run_concurrent_floor(
        roster, sandbox=_FakeSandbox(), service_url="http://svc", config=cfg,
        seat_client_factory=lambda seat_id: _FakeClient(seat_id),
    )
    r = summary["results"][0]
    assert r["status"].startswith("done")
    assert r["acp_tool_calls"] == 2  # tool calls summed over the two your_turn rounds
    assert len(prompts_sent) == 2
    assert "[Round 1]" in prompts_sent[0] and "[Round 2]" in prompts_sent[1]
    assert "request_id=r1" in prompts_sent[0]  # request_id threaded into the nudge
    assert observed[-1] == "done"
