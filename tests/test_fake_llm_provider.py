"""The deterministic tier's fake model provider, over real HTTP on the host.

The provider (``tests/integration/deterministic/task/environment/fake_llm``)
is what makes the deterministic integration tier reproducible; these checks
pin its contract without a sandbox: script selection by marker, step from the
conversation alone, tool-name resolution, fixed usage, and the Anthropic
streaming event sequence the LiteLLM proxy parses.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from tests.integration.deterministic import harness as h

fake = h._load_fake_llm_module()
SCRIPTS = json.loads((h.FAKE_LLM_DIR / "scripts.json").read_text())
BASH = {"name": "Bash", "input_schema": {"type": "object"}}


def _user(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _assistant() -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": "..."}]}


def _tool_result() -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}],
    }


def test_step_is_the_number_of_assistant_turns_after_the_marker():
    messages = [_user("hi [[fake-llm:hello-pass]]"), _assistant(), _tool_result()]
    assert fake.locate(messages) == ("hello-pass", 1)
    assert fake.locate(messages[:1]) == ("hello-pass", 0)


def test_the_latest_marked_prompt_wins_so_branch_children_follow_their_own_script():
    messages = [
        _user("draft [[fake-llm:draft]]"),
        _assistant(),
        _tool_result(),
        _assistant(),
        _user("child [[fake-llm:hello-wrong]]"),
    ]
    assert fake.locate(messages) == ("hello-wrong", 0)


def test_tool_call_uses_the_offered_name_and_fixed_usage():
    body = {
        "model": "claude-haiku-4-5",
        "tools": [{"name": "mcp__acp__Bash"}],
        "messages": [_user("[[fake-llm:hello-pass]]")],
    }
    reply = fake.plan_reply(body, SCRIPTS)
    tool_use = [b for b in reply["content"] if b["type"] == "tool_use"]
    assert tool_use == [
        {
            "type": "tool_use",
            "id": "toolu_fake_hello-pass_0",
            "name": "mcp__acp__Bash",
            "input": SCRIPTS["hello-pass"][0]["input"],
        }
    ]
    assert reply["stop_reason"] == "tool_use"
    assert reply["usage"]["input_tokens"] == h.MAIN_CALL_USAGE["input"]
    assert reply["usage"]["output_tokens"] == h.MAIN_CALL_USAGE["output"]


def test_calls_without_tools_are_side_calls_with_side_usage():
    reply = fake.plan_reply({"messages": [_user("[[fake-llm:hello-pass]]")]}, SCRIPTS)
    assert reply["content"] == [{"type": "text", "text": "ok"}]
    assert reply["usage"]["input_tokens"] == h.SIDE_CALL_USAGE["input"]
    assert reply["usage"]["output_tokens"] == h.SIDE_CALL_USAGE["output"]


def test_past_the_end_of_a_script_the_agent_is_told_done():
    messages = [
        _user("[[fake-llm:hello-pass]]"),
        _assistant(),
        _tool_result(),
        _assistant(),
        _user("more"),
    ]
    reply = fake.plan_reply({"tools": [BASH], "messages": messages}, SCRIPTS)
    assert reply["content"] == [{"type": "text", "text": "Done."}]
    assert reply["stop_reason"] == "end_turn"


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("no marker here", "no [[fake-llm:NAME]] marker"),
        ("[[fake-llm:nope]]", "unknown script 'nope'"),
    ],
)
def test_unscripted_prompts_get_a_visible_explanation(prompt, expected):
    reply = fake.plan_reply({"tools": [BASH], "messages": [_user(prompt)]}, SCRIPTS)
    assert expected in reply["content"][0]["text"]


def test_every_script_step_is_well_formed():
    for name, steps in SCRIPTS.items():
        assert steps, name
        for step in steps:
            assert set(step) <= {"text", "tool", "input", "delay_sec"}, (name, step)
            # Text on every step: the scenario checks count one agent
            # message per model call.
            assert step.get("text"), (name, step)
            assert ("input" in step) == ("tool" in step), (name, step)
            # A slow model's pause (fake_llm.script_delay)
            delay = step.get("delay_sec", 0)
            assert isinstance(delay, int | float) and delay >= 0, (name, step)


def _post(url: str, body: dict) -> tuple[int, str, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return (
            response.status,
            response.headers.get("Content-Type"),
            response.read().decode(),
        )


def test_streaming_reply_is_the_anthropic_event_sequence(tmp_path):
    log = tmp_path / "requests.jsonl"
    with h.host_fake_llm(log) as url:
        body = {
            "stream": True,
            "tools": [BASH],
            "messages": [_user("[[fake-llm:hello-pass]]")],
        }
        status, content_type, text = _post(f"{url}/v1/messages?beta=true", body)
        with urllib.request.urlopen(f"{url}/health", timeout=10) as response:
            health = json.loads(response.read())
        _, _, count = _post(f"{url}/v1/messages/count_tokens", {"messages": []})
    assert status == 200 and content_type == "text/event-stream"
    events = [
        line.split(": ", 1)[1]
        for line in text.splitlines()
        if line.startswith("event: ")
    ]
    assert events == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    data = [
        json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")
    ]
    assert (
        json.loads(data[5]["delta"]["partial_json"])
        == SCRIPTS["hello-pass"][0]["input"]
    )
    assert data[7]["delta"]["stop_reason"] == "tool_use"
    assert data[7]["usage"]["output_tokens"] == h.MAIN_CALL_USAGE["output"]
    assert health["ok"] is True and "hello-pass" in health["scripts"]
    assert json.loads(count) == {"input_tokens": h.MAIN_CALL_USAGE["input"]}
    logged = [json.loads(line) for line in log.read_text().splitlines()]
    assert [e["kind"] for e in logged] == ["messages", "count_tokens"]
    assert logged[0]["reply_id"] == "msg_fake_hello-pass_0"


def test_plain_json_reply_when_not_streaming():
    with h.host_fake_llm() as url:
        status, content_type, text = _post(
            f"{url}/v1/messages",
            {"tools": [BASH], "messages": [_user("[[fake-llm:draft]]")]},
        )
    assert status == 200 and content_type == "application/json"
    assert json.loads(text)["id"] == "msg_fake_draft_0"


def test_materialized_variants_share_one_environment(tmp_path):
    a = h.materialize_task(
        h.TaskVariant("a", "hello-wrong", agent_timeout_sec=45.0), tmp_path
    )
    b = h.materialize_task(h.TaskVariant("b", "crash", broken_verifier=True), tmp_path)
    assert "[[fake-llm:hello-wrong]]" in (a / "task.md").read_text()
    assert "timeout_sec: 45.0" in (a / "task.md").read_text()
    assert "exit 3" in (b / "tests" / "test.sh").read_text()
    for rel in (
        "environment/Dockerfile",
        "environment/fake_llm/fake_llm.py",
        "environment/fake_llm/scripts.json",
        "environment.toml",
    ):
        assert (
            (a / rel).read_bytes()
            == (b / rel).read_bytes()
            == (h.TEMPLATE_TASK / rel).read_bytes()
        )


@pytest.mark.parametrize("route", ["proxy", "native"])
@pytest.mark.parametrize("sandbox", ["docker", "daytona"])
def test_both_routes_reach_the_fake_without_host_credentials(
    route, sandbox, monkeypatch
):
    """A developer's provider env and host Claude login must not leak into a
    scripted run: the dummy token keeps BenchFlow off the host-credentials
    fallback, and the resolved base URL is the fake."""
    for key in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)
    agent_env = h.route_env(route, sandbox, "http://127.0.0.1:1")
    h.check_route_is_hermetic(route, agent_env)
    if route == "native" or h.proxy_runs_in_sandbox(sandbox):
        assert h.IN_SANDBOX_FAKE_URL in agent_env.values()
