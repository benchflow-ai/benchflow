"""A model call whose stream was cut makes the trial's count partial (EN-646).

X4 (2026-11-15, rel 53cad31d9, a real pi-acp run on MiMo through BenchFlow's
proxy), cell ``rel-pi-acp-acp-mimo-mixed-default-run``: trial
``b-interrupted__2e9bd848`` timed out while its one model call streamed.
LiteLLM logged that call as a success with its own count of the prompt (583
input, 0 output, finish_reason "stop") and the trial stored those figures,
priced at $0.00008162, with no marker; summary.json had no counter for
partial counts. The provider's answer to that call, seen by X4's recorder,
was 1,336 input (1,280 of them cached) and 49 output. Cell
``rel-mimo-acp-mimo-cancel-default-run`` (Ctrl-C, then resume): the stopped
attempt's last request was cut before its first chunk came back, a 499 with
no usage record at all, and the attempt's 24,302 tokens read as whole.

A call cut before the provider reported its usage now marks the trial
``usage_incomplete`` with the reason ``stream interrupted before final
usage``, ``cost_incomplete`` beside its cost, and its tokens as LiteLLM's
estimate under ``usage_details.interrupted_streams``; the cancellation
receipt keeps the marker and summary.json counts such trials
(``trials_usage_incomplete``, ``trials_cost_incomplete``). A stream that
ended is unchanged. Keys and model answers here are made up; counts are X4's.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from benchflow.providers.litellm_logging import (
    callback_module_source,
    extract_usage_from_trajectory,
    trajectory_from_litellm_callback_log,
)

MODEL = "mimo-v2.6-flash"
STREAM_INTERRUPTED = "stream interrupted before final usage"
CUT_REASON = (
    f"{STREAM_INTERRUPTED}: 1 call(s) through the model proxy ended before the "
    "provider reported their usage; the tokens of 1 are LiteLLM's local "
    "estimate, not the provider's count"
)


def _call(
    prompt: int,
    output: int,
    cached: int,
    cost: float,
    finish: str,
    *,
    interrupted: bool = False,
) -> dict:
    """One success line of the proxy's callback log, as the callback writes it."""
    record = {
        "event": "success",
        "request_model": MODEL,
        "provider_model": MODEL,
        "call_type": "acompletion",
        "request": {
            "method": "POST",
            "path": "/v1/chat/completions",
            "body": {"model": MODEL, "messages": [], "stream": True},
        },
        "response": {
            "model": "runtime-xiaomi-mimo-v2.6-flash",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish,
                    "message": {"role": "assistant", "content": "made-up answer"},
                }
            ],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": output,
                "total_tokens": prompt + output,
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        },
        "response_cost": cost,
        "start_time": "2026-11-14T23:26:11.297851",
        "end_time": "2026-11-14T23:26:13.687870",
        "duration_ms": 2390.0,
    }
    if interrupted:
        record["stream_interrupted"] = "client_disconnected"
    return record


def _request(
    request_id: str,
    *,
    upstream: bool = True,
    status: int | None = 200,
    complete: bool | None = True,
    ended: bool = True,
) -> list[dict]:
    """The request guard's lines for one request (provider_requests.jsonl)."""
    rows: list[dict] = [{"event": "request_start", "request_id": request_id}]
    if upstream:
        rows.append(
            {"event": "upstream_start", "request_id": request_id, "attempt_id": "a"}
        )
        rows.append(
            {
                "event": "upstream_end",
                "request_id": request_id,
                "attempt_id": "a",
                "status": 200,
            }
        )
    if ended:
        end = {"event": "request_end", "request_id": request_id, "status": status}
        if complete is not None:
            end["response_complete"] = complete
        rows.append(end)
    return rows


def _log(*rows: dict | list[dict]) -> str:
    lines = []
    for row in rows:
        for record in row if isinstance(row, list) else [row]:
            lines.append(json.dumps(record))
    return "\n".join(lines) + "\n"


def _usage(text: str) -> dict:
    return extract_usage_from_trajectory(
        trajectory_from_litellm_callback_log(text, session_id="s", agent_name="pi-acp")
    )


# X4 rel-pi-acp-acp-mimo-mixed-default-run, b-interrupted__2e9bd848: the
# call the timeout cut, and its request (LiteLLM's 499: the client left).
X4_CUT = _call(583, 0, 0, 8.162e-05, "stop", interrupted=True)
X4_CUT_REQUEST = _request("2e9bd848", status=499, complete=True)
# ... and a-success__222623df: two calls that ended (MiMo's list prices:
# $0.14/M uncached input, $0.0028/M cache reads, $0.28/M output).
X4_DONE = [
    _call(1336, 55, 1280, 2.6824e-05, "tool_calls"),
    _call(1401, 33, 1280, 2.9764e-05, "stop"),
]


def test_x4_cut_call_marks_the_count_and_labels_its_estimate():
    usage = _usage(_log(X4_CUT, X4_CUT_REQUEST))

    # The figures are what was counted, LiteLLM's estimate among them ...
    assert usage["n_input_tokens"] == 583
    assert usage["n_output_tokens"] == 0
    assert usage["cost_usd"] == pytest.approx(8.162e-05)
    # ... never presented as the provider's whole count.
    assert usage["usage_incomplete"] is True
    assert usage["usage_incomplete_reason"] == CUT_REASON
    assert usage["usage_details"]["interrupted_streams"] == {
        "calls_cut": 1,
        "calls_estimated": 1,
        "estimated": {
            "source": "litellm_local_count",
            "n_input_tokens": 583,
            "n_output_tokens": 0,
            "n_cache_read_tokens": 0,
            "n_cache_creation_tokens": 0,
            "cost_usd": pytest.approx(8.162e-05),
        },
    }


def test_x4_calls_that_ended_stay_complete():
    usage = _usage(
        _log(
            X4_DONE[0],
            _request("first"),
            X4_DONE[1],
            _request("second"),
        )
    )

    assert usage["total_tokens"] == 2825
    assert usage["n_input_tokens"] == 177
    assert usage["n_cache_read_tokens"] == 2560
    assert usage["cost_usd"] == pytest.approx(5.6588e-05)
    assert "usage_incomplete" not in usage
    assert "interrupted_streams" not in (usage.get("usage_details") or {})


def test_a_cut_request_with_no_usage_record_still_marks_the_count():
    """LiteLLM logs nothing for a stream its client left before the first
    chunk came back; the guard's 499 after an upstream call says it was cut.
    A 499 before any upstream call cost nothing and is not counted; a
    request the stopped proxy never ended is cut."""
    usage = _usage(
        _log(
            X4_DONE[0],
            _request("done"),
            _request("left-at-first-chunk", status=499, complete=True),
            _request("left-before-upstream", upstream=False, status=499),
        )
    )
    assert usage["usage_incomplete"] is True
    assert usage["usage_incomplete_reason"] == (
        f"{STREAM_INTERRUPTED}: 1 call(s) through the model proxy ended before "
        "the provider reported their usage"
    )
    assert usage["usage_details"]["interrupted_streams"] == {
        "calls_cut": 1,
        "calls_estimated": 0,
        "estimated": None,
    }

    stopped_under_it = _usage(
        _log(X4_DONE[0], _request("done"), _request("open", ended=False))
    )
    assert stopped_under_it["usage_details"]["interrupted_streams"]["calls_cut"] == 1

    mid_stream = _usage(
        _log(X4_DONE[0], _request("done"), _request("cut", status=200, complete=False))
    )
    assert mid_stream["usage_incomplete"] is True
    # A log written before the guard said so (no response_complete) reads
    # as it did.
    older = _usage(_log(X4_DONE[0], _request("done", complete=None)))
    assert "usage_incomplete" not in older


async def _guard_run(app) -> list[dict]:
    from benchflow.providers.litellm_request_guard import SingleRouteGuard

    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }
    ).encode()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    await SingleRouteGuard(app)(scope, receive, send)
    return sent


@pytest.mark.asyncio
async def test_the_guard_says_whether_the_whole_response_went_out(
    tmp_path, monkeypatch
):
    log = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))

    async def cut(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send(
            {"type": "http.response.body", "body": b"data: {}\n\n", "more_body": True}
        )
        # Starlette ends a stream whose client left without its last part.

    async def whole(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send(
            {"type": "http.response.body", "body": b"data: {}\n\n", "more_body": True}
        )
        await send({"type": "http.response.body", "body": b"data: [DONE]\n\n"})

    await _guard_run(cut)
    await _guard_run(whole)
    ends = [
        json.loads(line)
        for line in log.read_text().splitlines()
        if json.loads(line)["event"] == "request_end"
    ]
    assert [end["response_complete"] for end in ends] == [False, True]
    assert [end["status"] for end in ends] == [200, 200]


SSE_CUT = [
    {"role": "assistant", "content": ""},
    {"content": "Hello"},
]


def _sse(deltas: list[dict], *, finish: str | None, usage: dict | None) -> bytes:
    chunks = [
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        for delta in deltas
    ]
    if finish is not None:
        chunks[-1]["choices"][0]["finish_reason"] = finish
    if usage is not None:
        chunks.append(
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": MODEL,
                "choices": [],
                "usage": usage,
            }
        )
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + ("data: [DONE]\n\n" if finish is not None else "")).encode()


async def _litellm_stream(
    log: Path, body: bytes | httpx.AsyncByteStream, *, leave_after: int | None
):
    """LiteLLM's own proxy stream generator (1.103.3, the proxy's pin) on a
    made-up upstream, with BenchFlow's callback; *leave_after* chunks, then
    the client leaves (Starlette closes the generator), or it reads all."""
    import litellm
    import openai
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
    from litellm.proxy.utils import ProxyLogging

    namespace: dict = {}
    exec(callback_module_source(), namespace)
    callbacks = litellm.callbacks
    litellm.callbacks = [namespace["proxy_handler_instance"]]
    try:
        upstream = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    **(
                        {"content": body}
                        if isinstance(body, bytes)
                        else {"stream": body}
                    ),
                )
            )
        )
        client = openai.AsyncOpenAI(
            api_key="sk-made-up",
            base_url="http://upstream.test/v1",
            http_client=upstream,
        )
        response = await litellm.acompletion(
            model=f"openai/{MODEL}",
            messages=[{"role": "user", "content": "Say hello. " * 40}],
            stream=True,
            stream_options={"include_usage": True},
            client=client,
        )
        request_data = {
            "litellm_logging_obj": response.logging_obj,
            "model": MODEL,
            "metadata": {},
        }
        stream = ProxyBaseLLMRequestProcessing.async_streaming_data_generator(
            response=response,
            user_api_key_dict=UserAPIKeyAuth(),
            request_data=request_data,
            proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
            serialize_chunk=lambda chunk: f"data: {chunk}\n\n",
            serialize_error=lambda error: f"data: {error}\n\n",
            request=None,
        )
        if leave_after is None:
            async for _ in stream:
                pass
        else:
            for _ in range(leave_after):
                await stream.__anext__()
            await stream.aclose()
        for _ in range(100):
            text = log.read_text() if log.exists() else ""
            if '"event":"success"' in text or '"event":"failure"' in text:
                break
            await asyncio.sleep(0.05)
    finally:
        litellm.callbacks = callbacks
    return [json.loads(line) for line in log.read_text().splitlines()]


@pytest.mark.asyncio
async def test_litellm_logs_a_stream_its_client_left_as_interrupted(
    tmp_path, monkeypatch
):
    log = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    rows = await _litellm_stream(
        log, _sse(SSE_CUT, finish=None, usage=None), leave_after=2
    )

    (record,) = [row for row in rows if row.get("event") == "success"]
    # LiteLLM's 1.103.3 logs the cut stream as a success it finished
    # itself: "stop", and its own count, as X4's record showed ...
    assert record["response"]["choices"][0]["finish_reason"] == "stop"
    assert record["usage"]["completion_tokens"] >= 1
    # ... and the callback now says the stream was cut.
    assert record["stream_interrupted"] == "client_disconnected"
    usage = _usage(log.read_text())
    assert usage["usage_incomplete"] is True
    assert usage["usage_incomplete_reason"] == CUT_REASON
    estimate = usage["usage_details"]["interrupted_streams"]["estimated"]
    assert estimate["n_input_tokens"] == usage["n_input_tokens"]


@pytest.mark.asyncio
async def test_a_stream_that_reached_its_usage_report_stays_complete(
    tmp_path, monkeypatch
):
    log = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    # X4's provider counts for the cut call, had its stream ended.
    provider = {
        "prompt_tokens": 1336,
        "completion_tokens": 49,
        "total_tokens": 1385,
        "prompt_tokens_details": {"cached_tokens": 1280},
    }
    rows = await _litellm_stream(
        log,
        _sse([*SSE_CUT, {"content": " world"}], finish="stop", usage=provider),
        leave_after=None,
    )

    (record,) = [row for row in rows if row.get("event") == "success"]
    assert "stream_interrupted" not in record
    usage = _usage(log.read_text())
    assert (usage["n_input_tokens"], usage["n_cache_read_tokens"]) == (56, 1280)
    assert usage["n_output_tokens"] == 49
    assert "usage_incomplete" not in usage


class _StoppedProxy:
    """A stopped proxy whose callback log was read (LiteLLMProcess.stop)."""

    def __init__(self, text: str) -> None:
        self.trajectory = trajectory_from_litellm_callback_log(
            text, session_id="s", agent_name="pi-acp"
        )

    async def stop(self) -> None:
        pass


@pytest.mark.asyncio
async def test_the_x4_cut_trial_is_incomplete_in_its_records(tmp_path, monkeypatch):
    """result.json, ATIF's final metrics and summary.json's counts carry the
    marker for X4's cut trial; before, each showed its tokens as whole."""
    from benchflow._utils.evaluation_results import usage_summary
    from tests.test_en374_salvage import _rollout

    home = tmp_path / "sandbox-home"
    home.mkdir()
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="pi-acp", harness="acp", home=home
    )
    rollout._usage_runtime = SimpleNamespace(
        kind="litellm", server=_StoppedProxy(_log(X4_CUT, X4_CUT_REQUEST))
    )
    # The timeout cut the turn before its report; the proxy saw every call.
    rollout._usage_report_missing = True
    rollout._env = None
    await rollout._cleanup_before_teardown()
    rollout._build_result()

    data = json.loads((trial / "result.json").read_text())
    agent = data["agent_result"]
    assert agent["total_tokens"] == 583
    assert agent["usage_incomplete"] is True
    assert agent["usage_incomplete_reason"] == CUT_REASON
    assert agent["cost_incomplete"] is True
    estimate = agent["usage_details"]["interrupted_streams"]["estimated"]
    assert (estimate["n_input_tokens"], estimate["n_output_tokens"]) == (583, 0)

    summary = usage_summary({trial.name: data})
    assert summary["trials_usage_incomplete"] == 1
    assert summary["trials_cost_incomplete"] == 1


# X4 rel-mimo-acp-mimo-cancel-default-run: the stopped attempt
# (cancel__85e27abd), whose last request the Ctrl-C cut before its first chunk
# (a 499, no usage record), and the resumed one (cancel__c563a3f9).
X4_STOPPED_LOG = _log(
    _call(24178, 124, 24064, 0.0001180592, "tool_calls"),
    _request("first"),
    _request("cut-before-first-chunk", status=499, complete=True),
)
X4_STOPPED_TOKENS = 24302
X4_STOPPED_COST = 0.0001180592
X4_RESUMED_TOKENS = 73454
X4_RESUMED_COST = 0.0003868144


def _agent_result(tokens: int, cost: float) -> dict:
    """agent_result as rollout/_results.py writes it for a proxied trial."""
    return {
        "n_input_tokens": tokens // 2,
        "n_output_tokens": tokens // 4,
        "n_cache_read_tokens": tokens - tokens // 2 - tokens // 4,
        "n_cache_creation_tokens": 0,
        "total_tokens": tokens,
        "cost_usd": cost,
        "usage_source": "proxy",
        "cost_source": "list_price_estimate",
        "price_table_date": None,
        "price_source": "litellm",
        "usage_source_legacy": "provider_response",
    }


@pytest.mark.asyncio
async def test_a_resumed_jobs_every_attempt_scope_counts_the_cut_attempt(tmp_path):
    """The stopped attempt's receipt is written by the cancel path from the
    proxy's count. summary.json counts every attempt on this line, the
    stopped one included: its cut tail makes it an incomplete trial there,
    while the resumed trial, whose calls ended, stays complete."""
    from benchflow._utils.result_metadata import incomplete_usage_fields
    from benchflow.budget import cancelled_attempt_receipt
    from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
    from benchflow.models import RolloutResult

    tasks_dir = tmp_path / "tasks"
    (tasks_dir / "cancel").mkdir(parents=True)
    (tasks_dir / "cancel" / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )
    job = Evaluation(
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            agent="pi-acp",
            model=MODEL,
            concurrency=1,
            retry=RetryConfig(max_retries=0, min_wait_sec=0.0),
        ),
        preflight=False,
    )
    job_dir = job._jobs_dir / job._job_name
    stopped_usage = _usage(X4_STOPPED_LOG)
    assert stopped_usage["total_tokens"] == X4_STOPPED_TOKENS
    assert stopped_usage["usage_incomplete"] is True
    assert stopped_usage["usage_details"]["interrupted_streams"]["estimated"] is None

    async def first_run(task_path, _cfg):
        folder = job_dir / "cancel__85e27abd"
        folder.mkdir(parents=True)
        agent_result = {
            **{k: v for k, v in stopped_usage.items() if k != "usage_details"},
            "usage_source": "proxy",
            **incomplete_usage_fields(stopped_usage),
        }
        (folder / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "cancel",
                    "rollout_name": folder.name,
                    "rewards": None,
                    "agent": "pi-acp",
                    "agent_result": agent_result,
                    "error": "Interrupted: the run was stopped",
                    "verifier_error": None,
                    "partial_trajectory": True,
                    "finished_at": "2026-11-14T23:25:05+00:00",
                    "interrupted": True,
                }
            )
        )
        with cancelled_attempt_receipt(lambda: folder, lambda: stopped_usage):
            raise asyncio.CancelledError  # Ctrl-C while the call streamed

    job._run_single_task = AsyncMock(side_effect=first_run)
    with pytest.raises(asyncio.CancelledError):
        await job.run()
    receipt = json.loads(
        (job_dir / "cancel__85e27abd" / "budget-cancelled.json").read_text()
    )
    assert receipt["agent_result"]["usage_incomplete"] is True

    resumed = Evaluation.resume(job_dir)
    resumed._preflight = False

    async def second_run(task_path, _cfg):
        folder = job_dir / "cancel__c563a3f9"
        folder.mkdir(parents=True)
        usage = _agent_result(X4_RESUMED_TOKENS, X4_RESUMED_COST)
        (folder / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "cancel",
                    "rollout_name": folder.name,
                    "rewards": {"reward": 1.0},
                    "agent": "pi-acp",
                    "agent_result": usage,
                    "error": None,
                    "verifier_error": None,
                }
            )
        )
        return RolloutResult(
            task_name="cancel",
            rollout_name=folder.name,
            rewards={"reward": 1.0},
            agent="pi-acp",
            n_input_tokens=usage["n_input_tokens"],
            n_output_tokens=usage["n_output_tokens"],
            n_cache_read_tokens=usage["n_cache_read_tokens"],
            total_tokens=usage["total_tokens"],
            cost_usd=usage["cost_usd"],
            usage_source="proxy",
            cost_source="list_price_estimate",
            rollout_dir=folder,
        )

    resumed._run_single_task = AsyncMock(side_effect=second_run)
    await resumed.run()
    summary = json.loads((job_dir / "summary.json").read_text())

    # Every attempt: both, as X4's rel summary counted them ...
    assert summary["total_tokens"] == X4_STOPPED_TOKENS + X4_RESUMED_TOKENS == 97756
    assert summary["total_cost_usd"] == pytest.approx(X4_STOPPED_COST + X4_RESUMED_COST)
    # ... and the stopped one's cut tail is said, the resumed trial's is not.
    assert summary["trials_usage_incomplete"] == 1
    assert summary["trials_cost_incomplete"] == 1


# ---------------------------------------------------------------------------
# A stream the provider broke mid-way (F2b, 2026-11-15): the same rule, a call
# that never reached its final usage event.


class _BreaksAfterTwoChunks(httpx.AsyncByteStream):
    """A provider whose connection drops after two chunks of its answer."""

    async def __aiter__(self):
        yield _sse(SSE_CUT, finish=None, usage=None)
        raise httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body"
        )


@pytest.mark.asyncio
async def test_litellm_logs_a_stream_the_provider_broke_as_interrupted(
    tmp_path, monkeypatch
):
    log = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    rows = await _litellm_stream(log, _BreaksAfterTwoChunks(), leave_after=None)

    (record,) = [row for row in rows if row.get("event") in {"success", "failure"}]
    # LiteLLM 1.103.3 logs a failure, after counting the chunks itself ...
    assert record["event"] == "failure"
    assert record["partial_usage"]["completion_tokens"] >= 1
    # ... and the callback now says the provider's stream broke.
    assert record["stream_interrupted"] == "provider_stream_error"
    usage = _usage(_log(X4_DONE[0], _request("done")) + log.read_text())
    assert usage["usage_incomplete"] is True
    assert usage["usage_incomplete_reason"] == (
        f"{STREAM_INTERRUPTED}: 1 call(s) through the model proxy ended before "
        "the provider reported their usage"
    )
    assert usage["usage_details"]["interrupted_streams"]["calls_estimated"] == 0


@pytest.mark.asyncio
async def test_the_guard_marks_an_upstream_stream_that_broke(tmp_path, monkeypatch):
    from benchflow.providers import litellm_request_guard as guard

    log = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))

    async def read(stream: httpx.AsyncByteStream, request_id: str) -> None:
        token = guard._REQUEST.set(
            {
                "id": request_id,
                "upstream_seconds": 0.0,
                "retry_wait_seconds": 0.0,
                "failed_at": None,
            }
        )
        try:
            attempt = guard._UpstreamAttempt(
                httpx.Request("POST", "http://upstream.test/v1/chat/completions")
            )
            response = guard._track_response(
                httpx, httpx.Response(200, stream=stream), attempt, asynchronous=True
            )
            async for _ in response.stream:
                pass
        finally:
            guard._REQUEST.reset(token)

    with pytest.raises(httpx.RemoteProtocolError):
        await read(_BreaksAfterTwoChunks(), "broke")
    await read(httpx.ByteStream(_sse(SSE_CUT, finish="stop", usage=None)), "whole")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    ends = {row["request_id"]: row for row in rows if row["event"] == "upstream_end"}
    assert ends["broke"]["stream_error"] is True
    assert ends["whole"]["stream_error"] is False

    # The agent's response ended whole (LiteLLM sends an error frame), yet
    # the call was cut: the import counts it.
    requests = [
        [
            {"event": "request_start", "request_id": name},
            {
                "event": "request_end",
                "request_id": name,
                "status": 200,
                "response_complete": True,
            },
        ]
        for name in ("broke", "whole")
    ]
    usage = _usage(_log(X4_DONE[0], *requests, rows))
    assert usage["usage_incomplete"] is True
    assert usage["usage_details"]["interrupted_streams"]["calls_cut"] == 1


# ---------------------------------------------------------------------------
# A turn cut on a subscription route (lead ruling, 2026-11-15 00:5xZ): an ACP
# agent on a login, or a native CLI, counts its calls from its own report and
# logs, which keep only the responses that ended. A turn ended by a cut (a
# timeout, a cancel, SIGINT, a budget stop, a lost sandbox) marks the trial; a
# normal end, and an agent's own error, do not.

TURN_CUT_REASON = "turn cut on a subscription route; a call in flight may be missing"
OPUS = "claude-opus-5-5"
LOST = "container is not running"


def _login_rollout(tmp_path, monkeypatch, *, harness: str):
    """claude-agent-acp on a Claude login, its session log in the sandbox."""
    from benchflow.acp.session import ACPSession
    from tests.test_en374_salvage import _rollout

    home = tmp_path / "sandbox-home"
    home.mkdir()
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness=harness, home=home
    )
    object.__setattr__(rollout._config, "model", OPUS)
    rollout._timeout = 60
    rollout._resolved_prompts = ["Build the site."]
    rollout._acp_client = object()
    rollout._session = ACPSession("s")
    return rollout, trial, home


def _a_response_that_ended(home: Path) -> None:
    """One response in Claude Code's session log: 1,500 input, 3,000 cache
    writes, 120 output (a line is written once a response has ended)."""
    log = home / ".claude" / "projects" / "-app" / "session-1.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps(
            {
                "type": "assistant",
                "requestId": "req_1",
                "message": {
                    "id": "msg_1",
                    "model": OPUS,
                    "stop_reason": "tool_use",
                    "usage": {
                        "input_tokens": 1500,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 3000,
                        "output_tokens": 120,
                    },
                },
            }
        )
        + "\n"
    )


ENDINGS = (
    "timeout",
    "cancel",
    "sigint",
    "budget_stop",
    "sandbox_loss",
    "normal_end",
    "agent_error",
)


async def _turn(rollout, monkeypatch, home: Path, ending: str, *, native: bool):
    """Run one turn that logs a response that ended, then ends by *ending*."""
    import os
    import signal

    from benchflow import budget
    from benchflow.acp.client import ACPError
    from benchflow.acp.runtime import _agent_prompt_timeout_error

    waiting = asyncio.Event()

    async def turn(*args, **_kwargs):
        session = rollout._session
        session.record_user_prompt("Build the site.")
        _a_response_that_ended(home)
        if ending == "timeout":
            raise _agent_prompt_timeout_error(session, 60)
        if ending == "sandbox_loss":
            raise RuntimeError(LOST)
        if ending == "agent_error":
            raise ACPError(-32603, "Internal error")  # the agent's own error
        if ending == "normal_end":
            session.record_prompt_usage(
                {
                    "inputTokens": 1500,
                    "outputTokens": 120,
                    "cachedReadTokens": 0,
                    "cachedWriteTokens": 3000,
                    "totalTokens": 4620,
                }
            )
            return [], 0
        waiting.set()
        await asyncio.Event().wait()  # a call in flight, until the cut
        raise AssertionError("not cut")

    if native:
        rollout._is_session_factory = True
        monkeypatch.setattr(rollout._planes, "execute_prompts_session_factory", turn)
    else:
        monkeypatch.setattr(rollout._planes, "execute_prompts", turn)

    guard = budget.BudgetGuard(budget=budget.Budget(max_cost_usd=0.01))
    token = budget._trial_budget.set((guard, "trial"))
    try:
        task = asyncio.ensure_future(rollout.execute())
    finally:
        budget._trial_budget.reset(token)
    if ending in {"cancel", "sigint", "budget_stop"}:
        await waiting.wait()
        if ending == "cancel":
            task.cancel()
        elif ending == "sigint":
            # As asyncio.run does on Ctrl-C: SIGINT cancels the running task.
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGINT, task.cancel)
            try:
                os.kill(os.getpid(), signal.SIGINT)
                for _ in range(100):
                    if task.done():
                        break
                    await asyncio.sleep(0.01)
            finally:
                loop.remove_signal_handler(signal.SIGINT)
        else:
            # The job's spend passed its cap: the guard cancels its trials.
            guard._tasks["trial"] = task
            guard.cost_usd = 1.0
            guard.check()
    if ending == "normal_end":
        await task
    else:
        with pytest.raises(BaseException):  # noqa: B017 - each cut its own type
            await task
    await rollout._cleanup_before_teardown()
    rollout._build_result()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["acp-login", "native"])
@pytest.mark.parametrize("ending", ENDINGS)
async def test_a_turn_cut_on_a_subscription_route_is_a_floor(
    tmp_path, monkeypatch, ending, native
):
    from benchflow._utils.evaluation_results import usage_summary

    rollout, trial, home = _login_rollout(
        tmp_path, monkeypatch, harness="native" if native else "acp"
    )
    await _turn(rollout, monkeypatch, home, ending, native=native)
    data = json.loads((trial / "result.json").read_text())
    agent = data["agent_result"]
    summary = usage_summary({trial.name: data})

    # The response that ended is counted every way the turn ends.
    assert agent["n_output_tokens"] == 120
    assert agent["cost_usd"] is not None
    if ending in {"normal_end", "agent_error"}:
        for key in ("usage_incomplete", "usage_incomplete_reason", "cost_incomplete"):
            assert key not in agent
        assert summary["trials_usage_incomplete"] == 0
        return
    assert agent["usage_incomplete"] is True
    assert agent["usage_incomplete_reason"] == TURN_CUT_REASON
    assert agent["cost_incomplete"] is True
    assert data["final_metrics"]["extra"]["usage_incomplete"] is True
    assert summary["trials_usage_incomplete"] == 1
    assert summary["trials_cost_incomplete"] == 1
    # The kind of cut, as execute() recorded it.
    assert (
        rollout._turn_cut
        == {
            "timeout": "timeout",
            "cancel": "cancel",
            "sigint": "cancel",
            "budget_stop": "budget_stop",
            "sandbox_loss": "lost",
        }[ending]
    )


def test_a_proxied_trial_with_a_cut_turn_is_left_to_the_proxy():
    """A count the model proxy made is marked by the stream it saw cut
    (above), never by the turn rule: a call through it is in its log."""
    from benchflow.rollout._cost import mark_incomplete_usage

    for source, details, marked in (
        ("provider_response", {}, False),
        ("provider_response", {"agent_native_acp": {"total_tokens": 5}}, True),
        ("agent_native_acp", {}, True),
        ("session_log", {}, True),
        ("stream_log", {}, True),
        ("agent-reported", {}, False),
    ):
        rollout = SimpleNamespace(
            _usage_metrics={
                "usage_source": source,
                "total_tokens": 10,
                "usage_details": details,
            },
            _usage_report_missing=True,
            _usage_salvaged=True,
            _usage_incomplete_reason=None,
            _turn_cut="timeout",
        )
        mark_incomplete_usage(rollout)
        assert (
            rollout._usage_metrics.get("usage_incomplete_reason") == TURN_CUT_REASON
        ) is marked, source
