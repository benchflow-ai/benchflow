"""Gateway must forward agent-sent reasoning params on completions routes.

Live capture (2026-08-08, proxy between the gateway and api.deepseek.com):
prime-agent sent ``thinking: {"type": "enabled"}`` + ``reasoning_effort: high``
and the upstream received neither — LiteLLM's global ``drop_params: True``
strips non-vanilla-OpenAI fields on ``openai/`` passthrough routes, so every
gateway-routed run silently used the provider's default thinking config while
a native run used the agent's. The route now allowlists the reasoning params;
an upstream that does not support them rejects them exactly as it would for a
native client, which is the parity-correct behavior.
"""

from __future__ import annotations

import pytest

from benchflow.providers.litellm_config import resolve_litellm_route


def test_deepseek_routes_via_native_litellm_provider():
    """deepseek/<model>, not openai/<model>: LiteLLM's deepseek integration
    declares thinking + reasoning_effort supported, so drop_params keeps them;
    the generic OpenAI route does not declare DeepSeek thinking support."""
    env = {
        "DEEPSEEK_API_KEY": "sk-test",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
    }
    route = resolve_litellm_route("deepseek/deepseek-v4-flash", env)
    assert route.upstream_model == "deepseek/deepseek-v4-flash"
    assert route.litellm_params["model"] == "deepseek/deepseek-v4-flash"


def test_deepseek_native_route_honors_base_url_override():
    """A custom DEEPSEEK_BASE_URL (mock/capture endpoints in parity runs) must
    survive the native-provider routing."""
    env = {
        "DEEPSEEK_API_KEY": "sk-test",
        "DEEPSEEK_BASE_URL": "http://127.0.0.1:11500/v1",
    }
    route = resolve_litellm_route("deepseek/deepseek-v4-flash", env)
    assert route.upstream_model == "deepseek/deepseek-v4-flash"
    assert route.litellm_params.get("api_base") == "http://127.0.0.1:11500/v1"


def test_pinned_litellm_deepseek_provider_supports_reasoning_params():
    """Guards the assumption the native-provider routing rests on: if a LiteLLM
    upgrade stops declaring these params supported for deepseek, this fails
    before a silent re-drop ships."""
    pytest.importorskip("litellm")
    from litellm.utils import get_supported_openai_params

    supported = (
        get_supported_openai_params(
            model="deepseek-chat", custom_llm_provider="deepseek"
        )
        or []
    )
    assert "thinking" in supported
    assert "reasoning_effort" in supported


@pytest.mark.parametrize(
    "model,env,provider",
    [
        ("azure-foundry-openai/grok-4.7", {"AZURE_RESOURCE": "unused"}, "azure"),
        ("openai/gpt-6", {}, "openai"),
        (
            "xiaomi/mimo-v2-pro",
            {"XIAOMI_BASE_URL": "https://api.xiaomimimo.com/v1"},
            "openai",
        ),
        ("deepseek/deepseek-v4-flash", {}, "deepseek"),
    ],
)
def test_requested_effort_survives_parameter_filter(model, env, provider):
    """Guards issue #1152: inspect the wire params without any API call."""
    from litellm.utils import get_optional_params

    route = resolve_litellm_route(model, env)
    allowed = route.litellm_params.get("allowed_openai_params", [])
    assert isinstance(allowed, list)
    params = get_optional_params(
        model=route.upstream_model.split("/", 1)[1],
        custom_llm_provider=provider,
        reasoning_effort="high",
        allowed_openai_params=allowed,
        drop_params=True,
    )
    # DeepSeek maps effort to its native thinking field.
    if provider == "deepseek":
        assert params.get("thinking") == {"type": "enabled"}
    else:
        assert params.get("reasoning_effort") == "high"


def test_azure_effort_reaches_local_capture_server():
    """Guards #1152 on the real adapter, with no Azure traffic."""
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    import litellm

    captured = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            body = json.dumps(
                {
                    "id": "test",
                    "object": "chat.completion",
                    "model": "grok-4.7",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        route = resolve_litellm_route(
            "azure-foundry-openai/grok-4.7", {"AZURE_RESOURCE": "unused"}
        )
        params = dict(route.litellm_params)
        params.update(
            api_base=f"http://127.0.0.1:{server.server_port}", api_key="local-test"
        )
        litellm.completion(
            **params,
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="xhigh",
            drop_params=True,
        )
        assert captured[0]["reasoning_effort"] == "xhigh"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
