"""The effort a call carried is the one the body LiteLLM sent upstream
carried (lead ruling, R093 P1).

On the gateway's DeepSeek route Claude Code's ``output_config.effort: max``
reached DeepSeek's Anthropic endpoint as ``thinking.budget_tokens: 16384`` and
no effort, and result.json said ``sent {max: 1}, matches: true``: the record
read the request the gateway took in. The gateway's log now keeps the effort
fields of the body sent upstream, and the record reads them, a budget as
itself. Each gateway case runs BenchFlow's route, the proxy's callback module
and LiteLLM's Router against a local stub that records the body it receives.
No model is called.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

import benchflow.providers.litellm_bedrock_patch  # noqa: F401  (the proxy loads it)
from benchflow.effort_record import _body_effort, effort_outcome, effort_warning
from benchflow.providers.litellm_logging import callback_module_source

_ANTHROPIC = {
    "id": "m",
    "type": "message",
    "role": "assistant",
    "model": "m",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
_CHAT = {
    "id": "c",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
_CONVERSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    "metrics": {"latencyMs": 1},
}


class _Upstream(BaseHTTPRequestHandler):
    seen: ClassVar[list[dict[str, Any]]] = []

    def log_message(self, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append({"path": self.path, "body": body})
        if self.path.endswith("/converse"):
            reply = _CONVERSE
        elif self.path.endswith("/chat/completions"):
            reply = _CHAT
        else:
            reply = _ANTHROPIC
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _logged(
    tmp_path: Path, monkeypatch, upstream: str, model: str, env: dict, kind: str
):
    """One call through LiteLLM's Router with the proxy's callback module
    registered; the row it logged, and the body the stub received."""
    import litellm

    from benchflow.providers.litellm_config import resolve_litellm_route

    log = tmp_path / "llm_trajectory.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    namespace: dict[str, Any] = {"__name__": "benchflow_callback_under_test"}
    exec(callback_module_source(), namespace)
    monkeypatch.setattr(litellm, "callbacks", [namespace["proxy_handler_instance"]])
    monkeypatch.setattr(
        litellm, "use_chat_completions_url_for_anthropic_messages", True
    )
    monkeypatch.setattr(litellm, "drop_params", True)
    params = dict(resolve_litellm_route(model, env).litellm_params)
    if model.startswith("aws-bedrock/"):
        params["aws_bedrock_runtime_endpoint"] = upstream
    else:
        params["api_base"] = env["BENCHFLOW_PROVIDER_BASE_URL"]
        params["api_key"] = "fake-not-a-key"
    router = litellm.Router(model_list=[{"model_name": "m", "litellm_params": params}])
    _Upstream.seen.clear()

    async def call() -> None:
        if kind == "messages":
            await router.aanthropic_messages(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=32000,
                thinking={"type": "adaptive"},
                output_config={"effort": "max"},
                tools=[
                    {
                        "name": "Bash",
                        "description": "x",
                        "input_schema": {"type": "object"},
                    }
                ],
            )
        else:
            await router.acompletion(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=32000,
                reasoning_effort="max",
            )
        for _ in range(50):
            if log.exists() and log.read_text().strip():
                return
            await asyncio.sleep(0.1)

    asyncio.run(call())
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    return rows[-1]["request"]["body"], _Upstream.seen[-1]


_AWS = {
    "AWS_BEARER_TOKEN_BEDROCK": "fake-not-a-key",
    "AWS_REGION": "us-east-1",
    "AWS_REGION_NAME": "us-east-1",
}


def test_deepseeks_budget_is_what_the_record_reads(tmp_path, monkeypatch, upstream):
    body, sent = _logged(
        tmp_path,
        monkeypatch,
        upstream,
        "deepseek/deepseek-v4-pro",
        {"DEEPSEEK_API_KEY": "fake-not-a-key", "BENCHFLOW_PROVIDER_BASE_URL": upstream},
        "messages",
    )
    assert sent["body"]["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert "output_config" not in sent["body"]
    assert body["upstream"]["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert "output_config" not in body["upstream"]
    # The request the gateway took in still says max; the record does not.
    assert body["output_config"] == {"effort": "max"}
    assert _body_effort(body) == "budget_tokens=16384"


@pytest.mark.parametrize(
    ("model", "kind", "wire"),
    [
        ("aws-bedrock/us.anthropic.claude-opus-4-8", "messages", "max"),
        ("aws-bedrock/us.anthropic.claude-opus-4-8", "completion", "max"),
    ],
)
def test_a_bedrock_call_records_the_effort_its_body_carried(
    tmp_path, monkeypatch, upstream, model, kind, wire
):
    body, sent = _logged(tmp_path, monkeypatch, upstream, model, dict(_AWS), kind)
    fields = sent["body"].get("additionalModelRequestFields") or sent["body"]
    assert fields["output_config"]["effort"] == wire
    assert body["upstream"]["output_config"]["effort"] == wire
    assert _body_effort(body) == wire


def test_an_openai_compatible_call_records_the_effort_its_body_carried(
    tmp_path, monkeypatch, upstream
):
    body, sent = _logged(
        tmp_path,
        monkeypatch,
        upstream,
        "vllm/anthropic/claude-opus-5.5",
        {
            "OPENAI_API_KEY": "fake-not-a-key",
            "BENCHFLOW_PROVIDER_BASE_URL": upstream + "/v1",
        },
        "completion",
    )
    assert body["upstream"]["reasoning_effort"] == sent["body"]["reasoning_effort"]
    assert _body_effort(body) == sent["body"]["reasoning_effort"]


def test_with_no_effort_asked_a_budget_is_recorded_as_itself_and_matches_nothing():
    """Lead ruling: on a route that maps effort to a budget, a run with no
    -e states the budget as sent, and never a false match."""
    effort = effort_outcome(
        requested=None,
        requested_from=None,
        applied="default",
        sent={"budget_tokens=16384": 2},
        sent_source="proxy",
        calls=2,
    )
    assert effort["sent"] == {"budget_tokens=16384": 2}
    assert (effort["matches"], effort["reason"]) == (None, "nothing_to_compare")
    assert effort_warning(effort) is None


def test_a_log_written_before_the_upstream_body_was_kept_reads_as_before():
    assert _body_effort({"output_config": {"effort": "high"}}) == "high"
    assert _body_effort({"reasoning_effort": "low"}) == "low"
    assert _body_effort({"upstream": {}, "output_config": {"effort": "high"}}) is None
