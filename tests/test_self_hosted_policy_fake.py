"""The deterministic fake provider as a self-hosted policy server.

``tests/integration/deterministic/task/environment/fake_llm`` also serves
``/v1/chat/completions`` shaped like vLLM (``/v1``) or SGLang
(``/sglang/v1``), with token ids and logprobs from a deterministic chat
template. The end-to-end scenario in ``tests/e2e`` runs ``bench eval run``
against it in a sandbox; these tests pin its behaviour on the host, behind
BenchFlow's real LiteLLM proxy, with a scripted two-turn tool loop in the
shape an agent sends: the captured calls must be complete and each prompt
must extend the previous prompt and sampled tokens (token-in/token-out).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

import benchflow.providers.litellm_runtime as runtime_mod
from benchflow.providers.litellm_config import resolve_litellm_route
from benchflow.trajectories.token_capture import summarize_token_capture
from tests.integration.deterministic import harness as h

TOOLS_ANTHROPIC = [
    {
        "name": "Bash",
        "description": "Run a command",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
        },
    }
]
TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        },
    }
]
PROMPT = f"Create hello.txt. {h.marker('hello-pass')}"


def _fake() -> Any:
    return h._load_fake_llm_module()


def test_chat_template_makes_the_next_prompt_extend_prompt_and_sample():
    fake = _fake()
    scripts = json.loads((h.FAKE_LLM_DIR / "scripts.json").read_text())
    body = {
        "model": "policy",
        "tools": TOOLS_OPENAI,
        "messages": [{"role": "user", "content": PROMPT}],
        "logprobs": True,
        "return_token_ids": True,
    }
    reply = fake.plan_chat_reply(body, scripts)
    first = fake.chat_response(body, reply, "vllm")
    assert reply["finish_reason"] == "tool_calls"
    message = first["choices"][0]["message"]
    # The agent sends the tool call back with differently formatted JSON.
    call = dict(message["tool_calls"][0])
    args = json.loads(call["function"]["arguments"])
    call["function"] = {"name": "Bash", "arguments": json.dumps(args, indent=1)}
    body2 = dict(body)
    body2["messages"] = [
        *body["messages"],
        {"role": "assistant", "content": message["content"], "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": "done"},
    ]
    second = fake.chat_response(body2, fake.plan_chat_reply(body2, scripts), "vllm")

    expected = first["prompt_token_ids"] + first["choices"][0]["token_ids"]
    assert second["prompt_token_ids"][: len(expected)] == expected
    assert second["choices"][0]["finish_reason"] == "stop"
    assert len(first["choices"][0]["logprobs"]["content"]) == len(
        first["choices"][0]["token_ids"]
    )


def test_streamed_chunks_reassemble_to_the_plain_response():
    fake = _fake()
    scripts = json.loads((h.FAKE_LLM_DIR / "scripts.json").read_text())
    for flavor, flags in (
        ("vllm", {"return_token_ids": True}),
        (
            "sglang",
            {"return_input_ids_in_sglext": True, "return_output_ids_in_sglext": True},
        ),
    ):
        body = {
            "tools": TOOLS_OPENAI,
            "messages": [{"role": "user", "content": PROMPT}],
            "logprobs": True,
            **flags,
        }
        reply = fake.plan_chat_reply(body, scripts)
        plain = fake.chat_response(body, reply, flavor)
        chunks = fake.chat_chunks(body, reply, flavor)
        ids = [
            t for c in chunks for ch in c["choices"] for t in ch.get("token_ids", [])
        ]
        logprobs = [
            e["logprob"]
            for c in chunks
            for ch in c["choices"]
            for e in (ch.get("logprobs") or {}).get("content", [])
        ]
        plain_logprobs = [
            e["logprob"] for e in plain["choices"][0]["logprobs"]["content"]
        ]
        assert logprobs == plain_logprobs
        if flavor == "vllm":
            assert chunks[0]["prompt_token_ids"] == plain["prompt_token_ids"]
            assert ids == plain["choices"][0]["token_ids"]
            assert "sglext" not in chunks[-1]
        else:
            assert ids == []
            assert chunks[-1] == {
                **chunks[-1],
                "choices": [],
                "sglext": plain["sglext"],
            }


async def _loop(route_model: str, base_path: str, tmp_path: Path, kind: str):
    """Two scripted turns through the proxy; returns the captured exchanges."""
    with h.host_fake_llm(tmp_path / "fake.jsonl") as url:
        env = {
            "BENCHFLOW_PROVIDER_BASE_URL": url + base_path,
            "BENCHFLOW_PROVIDER_API_KEY": "fake-key",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "BENCHFLOW_CAPTURE_TOKEN_LOGPROBS": "1",
        }
        route = resolve_litellm_route(route_model, env)
        proc = await runtime_mod._start_host_litellm(
            route=route,
            master_key="sk-master",
            agent_env=env,
            environment="local",
            session_id="s",
            agent_name="fake-policy-test",
        )
        artifact = tmp_path / "trajectory" / "llm_trajectory.jsonl"
        try:
            proc.start_live_capture(artifact)
            async with httpx.AsyncClient(timeout=60) as client:
                if kind == "messages":
                    await _messages_loop(client, proc.base_url, route.model_alias)
                else:
                    await _chat_stream_loop(client, proc.base_url, route.model_alias)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if (
                    proc.log_path.exists()
                    and len(proc.log_path.read_text().splitlines()) >= 2
                ):
                    break
                await asyncio.sleep(0.2)
        finally:
            await proc.stop()
        fake_log = h.read_jsonl(tmp_path / "fake.jsonl")
    return h.read_jsonl(artifact), fake_log


async def _messages_loop(client: httpx.AsyncClient, base: str, alias: str) -> None:
    headers = {"Authorization": "Bearer sk-master", "anthropic-version": "2023-06-01"}
    messages: list[dict[str, Any]] = [{"role": "user", "content": PROMPT}]
    for _ in range(2):
        response = await client.post(
            base + "/v1/messages",
            headers=headers,
            json={
                "model": alias,
                "max_tokens": 256,
                "tools": TOOLS_ANTHROPIC,
                "messages": messages,
            },
        )
        assert response.status_code == 200, response.text
        content = response.json()["content"]
        messages.append({"role": "assistant", "content": content})
        uses = [b for b in content if b["type"] == "tool_use"]
        if not uses:
            return
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": u["id"], "content": "ok"}
                    for u in uses
                ],
            }
        )


async def _chat_stream_loop(client: httpx.AsyncClient, base: str, alias: str) -> None:
    headers = {"Authorization": "Bearer sk-master"}
    messages: list[dict[str, Any]] = [{"role": "user", "content": PROMPT}]
    for _ in range(2):
        text, calls = "", {}
        async with client.stream(
            "POST",
            base + "/v1/chat/completions",
            headers=headers,
            json={
                "model": alias,
                "tools": TOOLS_OPENAI,
                "messages": messages,
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                for choice in json.loads(line[6:]).get("choices") or []:
                    delta = choice.get("delta") or {}
                    text += delta.get("content") or ""
                    for tc in delta.get("tool_calls") or []:
                        slot = calls.setdefault(
                            tc.get("index", 0),
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["function"]["name"] += fn.get("name") or ""
                        slot["function"]["arguments"] += fn.get("arguments") or ""
        assistant: dict[str, Any] = {"role": "assistant", "content": text or None}
        if calls:
            assistant["tool_calls"] = list(calls.values())
        messages.append(assistant)
        if not calls:
            return
        for call in calls.values():
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": "ok"}
            )


@pytest.mark.parametrize(
    ("route_model", "base_path", "kind"),
    [
        ("vllm/policy", "/v1", "messages"),
        ("vllm/policy", "/v1", "chat_stream"),
        ("sglang/policy", "/sglang/v1", "messages"),
        ("sglang/policy", "/sglang/v1", "chat_stream"),
    ],
)
def test_agent_loop_through_the_gateway_is_training_grade(
    tmp_path, route_model, base_path, kind
):
    exchanges, fake_log = asyncio.run(_loop(route_model, base_path, tmp_path, kind))

    summary = summarize_token_capture(exchanges)
    assert summary["calls"] == 2, fake_log
    assert summary["complete_calls"] == 2, summary
    assert summary["prefix"] == {"pairs": 1, "extends_previous_call": 1, "breaks": []}
    assert summary["training_grade"] is True
    flavor = "sglang" if base_path.startswith("/sglang") else "vllm"
    assert [e["flavor"] for e in fake_log] == [flavor, flavor]
    assert all(e["logprobs"] for e in fake_log)
