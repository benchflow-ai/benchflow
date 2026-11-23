"""The model gateway's refusals are counted in model-calls.json (benchflow#1246).

A sub-agent naming another model (Claude Code's Agent tool with ``model:
haiku``, Codex's ``spawn_agent`` with another ``model``) is refused by the
callback's model gate with HTTP 400 ``model_not_served`` before any provider
sees it. On 0.10.0 the refusal was only a warning in the run log:
``model-calls.json`` said ``refused_calls: 0``. The callback now tells the
request guard, whose ``request_end`` record names the rule and the model; the
host copies that record into the trial's ``trajectory/provider_requests.jsonl``,
and ``model-calls.json`` counts it: ``refused_calls``, ``refusals_by_reason``
and ``models_refused``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.providers import litellm_request_guard as guard
from benchflow.providers.litellm_logging import callback_module_source
from benchflow.providers.litellm_runtime import LiteLLMProcess
from benchflow.trajectories.model_calls import write_model_calls

RUN_MODEL = "deepseek-flash"
FOREIGN = ("claude-haiku-5-5", "claude-opus-5-5")


def _body(model: str) -> bytes:
    return json.dumps(
        {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }
    ).encode()


def _scope() -> dict:
    return {
        "type": "http",
        "path": "/v1/messages",
        "method": "POST",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }


def _receive(body: bytes):
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()

    return receive


@pytest.fixture
def proxy(tmp_path, monkeypatch):
    """The callback as the proxy loads it, beside the request guard, with
    the run's model served and the trial folder's request record."""
    log = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    monkeypatch.setenv("BENCHFLOW_LITELLM_SERVED_MODELS", json.dumps([RUN_MODEL]))
    monkeypatch.delenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", raising=False)
    monkeypatch.delenv("BENCHFLOW_EFFORT_PIN", raising=False)
    monkeypatch.setattr(guard, "_GATE", guard._Gate())
    # The proxy's sitecustomize loads the guard under this name; the
    # callback finds it there.
    monkeypatch.setitem(sys.modules, "benchflow_litellm_request_guard", guard)
    namespace: dict = {}
    exec(callback_module_source(), namespace)
    callback = namespace["proxy_handler_instance"]

    async def app(scope, receive, send):
        # LiteLLM's part, as far as this needs it: the pre-call hook, then
        # the exception's status (400) as the answer.
        first = await receive()
        data = json.loads(first["body"])
        try:
            await callback.async_pre_call_hook(None, None, data, "anthropic_messages")
        except Exception as exc:
            status = getattr(exc, "status_code", 500)
            await send({"type": "http.response.start", "status": status})
            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps({"error": {"message": str(exc)}}).encode(),
                }
            )
            return
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"{}"})

    trial = tmp_path / "trial"
    (trial / "trajectory").mkdir(parents=True)
    process = LiteLLMProcess.__new__(LiteLLMProcess)
    process._live_trajectory_path = trial / "trajectory" / "llm_trajectory.jsonl"
    process._request_events = {}
    return SimpleNamespace(
        log=log, route=guard.SingleRouteGuard(app), trial=trial, process=process
    )


async def _post(proxy, model: str) -> int:
    sent: list = []

    async def send(message):
        sent.append(message)

    await proxy.route(_scope(), _receive(_body(model)), send)
    return sent[0]["status"]


def _records(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def _answered_call(trial: Path) -> None:
    """One call the proxy forwarded, as llm_trajectory.jsonl holds it."""
    row = {
        "request": {
            "timestamp": "2026-11-20T00:00:00Z",
            "method": "POST",
            "path": "/v1/messages",
            "body": {"model": RUN_MODEL, "max_tokens": 64},
        },
        "response": {
            "timestamp": "2026-11-20T00:00:01Z",
            "status_code": 200,
            "body": {
                "model": RUN_MODEL,
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        },
        "duration_ms": 1000.0,
        "metadata": {"request_model": RUN_MODEL, "call_purpose": "agent"},
    }
    (trial / "trajectory" / "llm_trajectory.jsonl").write_text(json.dumps(row) + "\n")


# --- J5 ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_refusal_is_named_on_the_guard_s_request_end(proxy):
    assert await _post(proxy, FOREIGN[0]) == 400
    ends = [r for r in _records(proxy.log) if r["event"] == "request_end"]
    assert len(ends) == 1, _records(proxy.log)
    assert ends[0]["status"] == 400
    assert ends[0]["refused"] == {"rule": "model-not-served", "model": FOREIGN[0]}
    # The callback's own record of the refusal is unchanged.
    refused = [r for r in _records(proxy.log) if r["event"] == "refused"]
    assert [(r["rule"], r["request_model"]) for r in refused] == [
        ("model-not-served", FOREIGN[0])
    ]


@pytest.mark.asyncio
async def test_a_refused_model_name_is_kept_printable(proxy):
    await _post(proxy, "claude-\u001b]0;x\u0007haiku")
    (end,) = [r for r in _records(proxy.log) if r["event"] == "request_end"]
    assert end["refused"] == {"rule": "model-not-served", "model": "(unprintable)"}


@pytest.mark.asyncio
async def test_a_served_request_names_no_refusal(proxy):
    assert await _post(proxy, RUN_MODEL) == 200
    (end,) = [r for r in _records(proxy.log) if r["event"] == "request_end"]
    assert "refused" not in end


@pytest.mark.asyncio
async def test_model_calls_json_counts_the_refusals_with_the_models(proxy):
    for model in (*FOREIGN, FOREIGN[0], RUN_MODEL):
        await _post(proxy, model)
    # The host's live capture copies the request records into the trial.
    proxy.process._capture_request_events(proxy.log.read_text())
    _answered_call(proxy.trial)
    summary = write_model_calls(proxy.trial)
    assert summary is not None
    assert summary["refused_calls"] == 3
    assert summary["refusals_by_reason"] == {"model_not_served": 3}
    assert summary["models_refused"] == {FOREIGN[0]: 2, FOREIGN[1]: 1}
    written = json.loads((proxy.trial / "model-calls.json").read_text())
    assert written["refused_calls"] == 3


@pytest.mark.asyncio
async def test_a_trial_whose_every_request_was_refused_still_writes_the_count(proxy):
    await _post(proxy, FOREIGN[1])
    proxy.process._capture_request_events(proxy.log.read_text())
    summary = write_model_calls(proxy.trial)
    assert summary is not None
    assert (summary["calls"], summary["refused_calls"]) == (0, 1)
    assert summary["models_refused"] == {FOREIGN[1]: 1}


def test_a_model_s_own_refusal_and_the_gateway_s_are_told_apart(tmp_path):
    trial = tmp_path / "trial"
    (trial / "trajectory").mkdir(parents=True)
    _answered_call(trial)
    row = json.loads((trial / "trajectory" / "llm_trajectory.jsonl").read_text())
    row["response"]["body"]["stop_reason"] = "refusal"
    (trial / "trajectory" / "llm_trajectory.jsonl").write_text(json.dumps(row) + "\n")
    (trial / "trajectory" / "provider_requests.jsonl").write_text(
        json.dumps(
            {
                "event": "request_end",
                "request_id": "a",
                "status": 400,
                "refused": {"rule": "model-not-served", "model": FOREIGN[0]},
            }
        )
        + "\n"
    )
    summary = write_model_calls(trial)
    assert summary is not None
    assert summary["refused_calls"] == 2
    assert summary["refusals_by_reason"] == {"refusal": 1, "model_not_served": 1}
    # result.json's agent_outcome keeps counting the model's own refusals.
    from benchflow.trajectories.model_calls import agent_outcome

    assert agent_outcome(summary)["refused_calls"] == 1


def test_the_cli_summary_names_the_gateway_s_refusals(tmp_path):
    from benchflow.cli.main import _model_calls_row

    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "model-calls.json").write_text(
        json.dumps(
            {
                "calls": 12,
                "source": "proxy",
                "refused_calls": 8,
                "model_refused_calls": 0,
                "refusals_by_reason": {"model_not_served": 8},
                "models_refused": {FOREIGN[0]: 4, FOREIGN[1]: 4},
            }
        )
    )
    row = _model_calls_row(SimpleNamespace(trials=[SimpleNamespace(path=trial)]))
    assert row is not None
    assert "8 refused by the model gateway (model not served:" in row, row
    assert f"{FOREIGN[0]} x4" in row and f"{FOREIGN[1]} x4" in row, row
    # The model refused none of them.
    assert "; 8 refused;" not in row, row


def test_the_run_card_after_the_run_names_the_gateway_s_refusals(tmp_path):
    from benchflow.run_checks import RunChecks, after_run_lines

    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "result.json").write_text("{}")
    (trial / "model-calls.json").write_text(
        json.dumps(
            {
                "refusals_by_reason": {"model_not_served": 6, "refusal": 1},
                "models_refused": {FOREIGN[0]: 6},
                "efforts_used": {"max": 3},
            }
        )
    )
    checks = RunChecks({}, 1)
    checks.record("trial", trial)
    lines = after_run_lines(checks.summary())
    refused = [line for line in lines if line.startswith("  Refused:")]
    assert refused == [
        "  Refused:   6 requests by the model gateway, before any provider "
        f"(model not served: {FOREIGN[0]} (6)); each trial's model-calls.json"
    ], lines
    # One level: no Efforts line.
    assert not [line for line in lines if line.startswith("  Efforts:")]
