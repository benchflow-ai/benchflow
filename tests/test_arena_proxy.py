"""Proxy routing + trajectory capture: each seat's raw LLM goes through the
BenchFlow provider proxy (``BENCHFLOW_PROVIDER_*``), and each turn is recorded to
a per-seat trajectory. Uses ``httpx.MockTransport`` — no real proxy or LLM.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from benchflow.arena import (
    Observation,
    ProxyChatPolicy,
    SeatStatus,
    SeatTrajectory,
)


def test_seat_trajectory_writes_per_seat_jsonl(tmp_path) -> None:
    tr = SeatTrajectory(tmp_path)
    tr.record(
        "a",
        status=SeatStatus.YOUR_TURN,
        observation={"pot": 150},
        action={"verb": "pick", "args": {"n": 7}},
    )
    tr.record(
        "a", status=SeatStatus.YOUR_TURN, action={"verb": "pick", "args": {"n": 3}}
    )
    tr.record(
        "b", status=SeatStatus.YOUR_TURN, action={"verb": "pick", "args": {"n": 5}}
    )
    lines = tr.path("a").read_text().strip().splitlines()
    assert len(lines) == 2
    r0 = json.loads(lines[0])
    assert r0["seat"] == "a" and r0["turn"] == 1 and r0["status"] == "your_turn"
    assert r0["action"]["args"]["n"] == 7
    assert json.loads(lines[1])["turn"] == 2
    assert tr.path("b").exists()


def test_proxy_chat_policy_routes_through_provider_and_records(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", "http://proxy.local/v1")
    monkeypatch.setenv("BENCHFLOW_PROVIDER_API_KEY", "bf-key")
    monkeypatch.setenv("BENCHFLOW_PROVIDER_MODEL", "proxy-model")
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers.get("authorization")
        seen["seat"] = req.headers.get("x-bf-seat")
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "I choose rock."}}],
                "usage": {"total_tokens": 11},
            },
        )

    tr = SeatTrajectory(tmp_path)

    async def scenario() -> dict:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            legal = [
                {"verb": "throw", "args": {"hand": h}}
                for h in ("rock", "paper", "scissors")
            ]
            pol = ProxyChatPolicy(
                "seat-0",
                http,
                render=lambda o: "pick one",
                pick=lambda text, lg: next(
                    (a for a in lg if a["args"]["hand"] in text.lower()), lg[0]
                ),
                recorder=tr,
            )
            obs = Observation(
                status=SeatStatus.YOUR_TURN,
                request_id="r1",
                public={"pot": 100},
                legal_actions=legal,
            )
            return await pol.act(obs)

    action = asyncio.run(scenario())
    assert action == {"verb": "throw", "args": {"hand": "rock"}}
    # routed through the PROXY base, with the proxy key/model + a per-seat tag
    assert seen["url"] == "http://proxy.local/v1/chat/completions"
    assert seen["auth"] == "Bearer bf-key"
    assert seen["seat"] == "seat-0"
    assert seen["body"]["model"] == "proxy-model"
    assert seen["body"]["metadata"] == {"seat": "seat-0"}
    # the per-seat trajectory captured the decision + the raw llm call
    rec = json.loads(tr.path("seat-0").read_text().strip())
    assert rec["action"]["args"]["hand"] == "rock"
    assert rec["llm"]["model"] == "proxy-model"
    assert rec["llm"]["usage"]["total_tokens"] == 11
    assert rec["llm"]["response"] == "I choose rock."
