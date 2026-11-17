"""OpenRouter's bill is read on the route 0.9.x builds (EN-374).

0.9.x routes an OpenRouter model as ``openai/<vendor>/<model>`` with
OpenRouter's ``api_base``, so LiteLLM's provider for the call is ``openai``,
not ``openrouter``. The bill check keyed on the provider name never read
OpenRouter's ``usage.cost`` there. Here LiteLLM's Router runs the real route
entry, OpenRouter's address included, and answers through an HTTP transport
that bills each answer as OpenRouter does; the logging callback's bill check
reads what LiteLLM hands it. A route to any other host is never billed, even
when its answer carries a ``usage.cost``.

Each call starts from empty LiteLLM callback lists: LiteLLM adds no second
logger of a class it already holds, so in one process a later test's
capture was dropped beside an earlier one's and saw no callback.

Each call's prompt carries a marker of its own, and a test reads only the
records that carry it. LiteLLM runs a call's success callbacks after the call
returns (on its logging worker, which hands pending tasks to the next event
loop, or on a thread), and BenchFlow's logger reads
``BENCHFLOW_LITELLM_LOG_PATH`` when it writes, not when it was made: a
callback of another test's call that runs during this one writes its record
into this test's log (R093 L5: 1,050 input tokens counted for 1,000).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

# The gateway proxy's sitecustomize imports BenchFlow's stream patch, which
# notes the usage.cost OpenRouter's own stream chunks carry (EN-544); these
# calls run with it, as the gateway's do.
import benchflow.providers.litellm_token_capture_patch  # noqa: F401
from benchflow.providers.litellm_config import (
    litellm_proxy_config,
    resolve_litellm_route,
)

BILL = 0.00731
# The model OpenRouter names in its answer; LiteLLM prices a call by it.
ANSWER_MODEL = "moonshotai/kimi-k3"
USAGE = {
    "prompt_tokens": 400,
    "completion_tokens": 20,
    "total_tokens": 420,
    "cost": BILL,
}


class _BillingUpstream(BaseHTTPRequestHandler):
    """An OpenAI-compatible upstream that bills every answer, as OpenRouter
    does (``usage.cost``), streamed or not."""

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        base = {"id": "gen-1", "created": 1, "model": "moonshotai/kimi-k3"}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            first = {
                **base,
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": "ok"}}
                ],
            }
            last = {
                **base,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": USAGE,
            }
            for chunk in (first, last):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        data = json.dumps(
            {
                **base,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": USAGE,
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # quiet
        pass


@pytest.fixture
def upstream():
    server = HTTPServer(("127.0.0.1", 0), _BillingUpstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/api/v1"
    server.shutdown()


def _callback():
    from benchflow.providers.litellm_logging import callback_module_source

    namespace: dict = {}
    exec(callback_module_source(), namespace)
    return namespace


def _answer(request: httpx.Request) -> httpx.Response:
    """OpenRouter's answer to one chat call, with ``USAGE`` (billed: its
    ``usage.cost``)."""
    body = json.loads(request.content or b"{}")
    base = {"id": "gen-1", "created": 1, "model": ANSWER_MODEL}
    if body.get("stream"):
        first = {
            **base,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}}],
        }
        last = {
            **base,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": USAGE,
        }
        sse = "".join(f"data: {json.dumps(c)}\n\n" for c in (first, last))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(sse + "data: [DONE]\n\n").encode(),
        )
    return httpx.Response(
        200,
        json={
            **base,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": USAGE,
        },
    )


_CALLBACK_LISTS = (
    "callbacks",
    "input_callback",
    "success_callback",
    "failure_callback",
    "_async_success_callback",
    "_async_failure_callback",
)


def _routed_call(model: str, env: dict, *, stream: bool, api_base: str | None = None):
    """One synchronous call through LiteLLM's Router on BenchFlow's own route
    entry for *model*; the (entry, kwargs, response) LiteLLM gives its
    synchronous success hook. Without *api_base* the call goes to the
    route's own address, answered by :func:`_answer` (no network)."""
    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    route = resolve_litellm_route(model, env)
    entry = litellm_proxy_config(route, master_key="local-test")["model_list"][0]
    if api_base is not None:
        entry["litellm_params"]["api_base"] = api_base
    # A key of its own: LiteLLM caches its HTTP clients by key and address.
    entry["litellm_params"]["api_key"] = f"local-test-{uuid.uuid4().hex}"
    marker = f"call-{uuid.uuid4().hex}"
    seen: list = []

    class Capture(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            # This call's event only, not another call's late callback.
            if marker in json.dumps(kwargs.get("messages"), default=str):
                seen.append((kwargs, response_obj))

    # LiteLLM keeps the callbacks it was given in its own lists, and adds no
    # second logger of a class it already holds: a later test's Capture would
    # be dropped beside an earlier one's. Each call starts from empty lists,
    # and the previous ones come back after.
    saved = {name: list(getattr(litellm, name)) for name in _CALLBACK_LISTS}
    previous_session = litellm.client_session
    for name in _CALLBACK_LISTS:
        setattr(litellm, name, [])
    litellm.callbacks = [Capture()]
    litellm.client_session = httpx.Client(transport=httpx.MockTransport(_answer))
    try:
        router = litellm.Router(model_list=[entry])
        response = router.completion(
            model=entry["model_name"],
            messages=[{"role": "user", "content": f"hi {marker}"}],
            stream=stream,
            **({"stream_options": {"include_usage": True}} if stream else {}),
        )
        if stream:
            for _ in response:
                pass
        for _ in range(200):
            if seen:
                break
            time.sleep(0.05)
    finally:
        for name, value in saved.items():
            setattr(litellm, name, value)
        litellm.client_session = previous_session
    assert seen, "LiteLLM called no success callback"
    return entry, seen[-1]


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "streamed"])
def test_the_openrouter_route_s_bill_is_read(stream):
    entry, (kwargs, response) = _routed_call(
        "openrouter/moonshotai/kimi-k3",
        {"OPENROUTER_API_KEY": "local-test"},
        stream=stream,
    )
    # The route 0.9.x builds: LiteLLM's openai provider on OpenRouter's host.
    assert entry["litellm_params"]["model"].startswith("openai/")
    assert "openrouter.ai" in entry["litellm_params"]["api_base"]
    billed = _callback()["_provider_billed_cost"](kwargs, response)
    assert billed == pytest.approx(BILL)


def test_another_route_is_never_billed(upstream):
    """A deepseek route whose answer carries a usage.cost: not billed."""
    _, (kwargs, response) = _routed_call(
        "deepseek/deepseek-v4-pro",
        {"DEEPSEEK_API_KEY": "local-test"},
        stream=False,
        api_base=upstream,
    )
    assert _callback()["_provider_billed_cost"](kwargs, response) is None


def test_the_route_decides_not_the_provider_name():
    is_openrouter = _callback()["_openrouter_call"]
    assert is_openrouter(
        {"litellm_params": {"api_base": "https://openrouter.ai/api/v1"}}
    )
    assert is_openrouter(
        {"litellm_params": {"metadata": {"model_group": "benchflow-openrouter-x-y"}}}
    )
    assert not is_openrouter(
        {"litellm_params": {"api_base": "https://api.deepseek.com/v1"}}
    )
    assert not is_openrouter(
        {"litellm_params": {"metadata": {"model_group": "benchflow-deepseek-v4"}}}
    )
    assert not is_openrouter(
        {"litellm_params": {"api_base": "https://openrouter.ai.example.com/v1"}}
    )
    # Where the call went decides: an openrouter/ model sent elsewhere is not
    # OpenRouter's to bill.
    assert not is_openrouter(
        {
            "model": "openrouter/x/y",
            "litellm_params": {
                "api_base": "https://gateway.example/v1",
                "metadata": {"model_group": "benchflow-openrouter-x-y"},
            },
        }
    )


# The record another call's callback writes, late, into whatever log the
# process names then: an answer of 50 input tokens and 1 output token.
FOREIGN_KWARGS = {
    "model": "benchflow-openai-gpt-4.1-mini",
    "messages": [{"role": "user", "content": "another test's call"}],
    "litellm_params": {"model": "openai/gpt-4.1-mini"},
    "optional_params": {},
    "call_type": "acompletion",
}
FOREIGN_RESPONSE = {
    "model": "gpt-4.1-mini",
    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    "usage": {"prompt_tokens": 50, "completion_tokens": 1, "total_tokens": 51},
}

# OpenRouter's answer with cache reads and writes, both inside prompt_tokens
# (OpenRouter's documented usage object; see test_en374_token_split).
CACHED_USAGE = {
    "prompt_tokens": 10000,
    "completion_tokens": 100,
    "total_tokens": 10100,
    "prompt_tokens_details": {"cached_tokens": 6000, "cache_write_tokens": 3000},
    "cost": BILL,
}


def _gateway_usage(
    tmp_path, monkeypatch, model: str, *, api: str, stream: bool
) -> dict:
    """BenchFlow's own gateway logger on the route 0.9.x builds for *model*:
    one call, answered by :func:`_answer` with ``USAGE``, its log read back as
    the trial's usage. A record another call writes into the log meanwhile
    is not counted."""
    from datetime import datetime

    import litellm

    from benchflow.providers.litellm_logging import (
        extract_usage_from_trajectory,
        trajectory_from_litellm_callback_log,
    )

    tmp_path.mkdir(parents=True, exist_ok=True)
    log = tmp_path / "litellm.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    route = resolve_litellm_route(model, {"OPENROUTER_API_KEY": "local-test"})
    entries = litellm_proxy_config(route, master_key="local-test")["model_list"]
    key = f"local-test-{uuid.uuid4().hex}"
    for entry in entries:
        entry["litellm_params"]["api_key"] = key
    marker = f"call-{uuid.uuid4().hex}"
    name = entries[0]["model_name"]
    if api == "responses-bridge":
        name = next(
            e["model_name"]
            for e in entries
            if e["model_name"].endswith("-responses-bridge")
        )
    saved = {
        list_name: list(getattr(litellm, list_name)) for list_name in _CALLBACK_LISTS
    }
    previous_session = litellm.aclient_session
    for list_name in _CALLBACK_LISTS:
        setattr(litellm, list_name, [])
    litellm.callbacks = [_callback()["BenchFlowLiteLLMLogger"]()]
    litellm.aclient_session = httpx.AsyncClient(transport=httpx.MockTransport(_answer))

    def mine() -> str:
        """The log's records of this call."""
        if not log.exists():
            return ""
        return "".join(
            line for line in log.read_text().splitlines(keepends=True) if marker in line
        )

    async def call():
        router = litellm.Router(model_list=entries)
        # Another call's late callback writes its record here meanwhile.
        moment = datetime(2026, 11, 14, 3, 0, 0)
        await _callback()["BenchFlowLiteLLMLogger"]().async_log_success_event(
            FOREIGN_KWARGS, FOREIGN_RESPONSE, moment, moment
        )
        if api == "responses-bridge":
            response = await router.aresponses(
                model=name, input=f"hi {marker}", stream=stream
            )
        else:
            response = await router.acompletion(
                model=name,
                messages=[{"role": "user", "content": f"hi {marker}"}],
                stream=stream,
                **({"stream_options": {"include_usage": True}} if stream else {}),
            )
        if stream:
            async for _ in response:
                pass
        for _ in range(200):
            if '"event":"success"' in mine():
                break
            await asyncio.sleep(0.05)

    try:
        asyncio.run(call())
    finally:
        for list_name, value in saved.items():
            setattr(litellm, list_name, value)
        litellm.aclient_session = previous_session
    assert mine(), "the gateway's logger wrote no record of this call"
    assert "another test's call" in log.read_text()
    trajectory = trajectory_from_litellm_callback_log(
        mine(), session_id="s", agent_name="a"
    )
    return extract_usage_from_trajectory(trajectory)


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("api", ["chat", "responses-bridge"])
def test_openrouter_s_cache_writes_reach_the_trial_s_split(
    tmp_path, monkeypatch, api, stream
):
    """BenchFlow's own gateway logger on the route 0.9.x builds, its log read
    back as the trial's usage: OpenRouter's cache writes leave the uncached
    input, on the chat route and on the Responses-API bridge a Codex run
    takes, streamed or not, and the bill stays the cost. A record another
    call writes into the log meanwhile is not counted."""
    monkeypatch.setitem(globals(), "USAGE", CACHED_USAGE)
    usage = _gateway_usage(
        tmp_path,
        monkeypatch,
        "openrouter/anthropic/claude-sonnet-5-5",
        api=api,
        stream=stream,
    )
    assert usage["n_input_tokens"] == 1000
    assert usage["n_cache_read_tokens"] == 6000
    assert usage["n_cache_creation_tokens"] == 3000
    assert usage["n_cache_write_5m_tokens"] is None
    assert usage["total_tokens"] == 10100
    assert usage["cost_usd"] == pytest.approx(BILL)
    assert usage["cost_source"] == "provider"


# OpenRouter's answer without its bill (no usage.cost), for a model LiteLLM
# prices.
UNBILLED_USAGE = {"prompt_tokens": 400, "completion_tokens": 20, "total_tokens": 420}


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "streamed"])
def test_an_unbilled_answer_is_no_bill(monkeypatch, stream):
    """LiteLLM writes its own estimate into a stream's usage.cost when
    OpenRouter sent none; that is not OpenRouter's bill (R093 L1, EN-544)."""
    monkeypatch.setitem(globals(), "USAGE", UNBILLED_USAGE)
    monkeypatch.setitem(globals(), "ANSWER_MODEL", "openai/gpt-4o-mini")
    _, (kwargs, response) = _routed_call(
        "openrouter/openai/gpt-4o-mini",
        {"OPENROUTER_API_KEY": "local-test"},
        stream=stream,
    )
    assert _callback()["_provider_billed_cost"](kwargs, response) is None


@pytest.mark.parametrize("api", ["chat", "responses-bridge"])
def test_an_unbilled_stream_is_priced_as_the_whole_answer_is(
    tmp_path, monkeypatch, api
):
    """Through the gateway's logger, an unbilled streamed OpenRouter call is a
    list-price estimate, as the same call whole is: it was recorded as
    OpenRouter's bill (``cost_source`` provider, ``price_source`` openrouter)
    at LiteLLM's own estimate (R093 L1)."""
    monkeypatch.setitem(globals(), "USAGE", UNBILLED_USAGE)
    monkeypatch.setitem(globals(), "ANSWER_MODEL", "openai/gpt-4o-mini")
    model = "openrouter/openai/gpt-4o-mini"
    whole = _gateway_usage(
        tmp_path / "whole", monkeypatch, model, api=api, stream=False
    )
    streamed = _gateway_usage(
        tmp_path / "streamed", monkeypatch, model, api=api, stream=True
    )
    keys = ("cost_usd", "cost_source", "price_source")
    assert {k: streamed.get(k) for k in keys} == {k: whole.get(k) for k in keys}
    assert streamed["cost_usd"] is not None
    assert streamed.get("cost_source") != "provider"
    assert streamed.get("price_source") != "openrouter"
