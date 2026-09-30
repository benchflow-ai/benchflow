"""Exact token segments of a rollout (``benchflow.trajectories.token_segments``).

PostTrain Arena's GRPO collector had to rebuild all of this on its side: keep
only the agent's own calls, skip failed provider attempts that the agent then
retried, start a new causal segment when the agent compacted its history,
mask tool output, and refuse a call whose ids and logprobs disagree. These
tests pin each rule on synthetic ``llm_trajectory.jsonl`` rows.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchflow.trajectories.token_capture import (
    TOKEN_CAPTURE_SCHEMA_VERSION,
    build_token_capture,
    strip_captured_token_ids,
    token_digest,
)
from benchflow.trajectories.token_segments import (
    segment_digest,
    segment_rollout,
)


def lp(ids: list[int]) -> list[float]:
    return [-0.5 - 0.01 * i for i in range(len(ids))]


def call(
    prompt: list[int] | None,
    sampled: list[int] | None,
    logprobs: list[float] | None = None,
    *,
    tools: tuple[str, ...] = ("Bash",),
    messages: list[dict[str, Any]] | None = None,
    status: int = 200,
    purpose: str = "agent",
    unavailable: dict[str, Any] | None = None,
    routing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if logprobs is None and sampled is not None:
        logprobs = lp(sampled)
    completion: dict[str, Any] = {
        "index": 0,
        "token_ids": sampled,
        "tokens": None,
        "logprobs": logprobs,
        "top_logprobs": None,
    }
    if routing is not None:
        completion["routing"] = routing
    capture: dict[str, Any] = {
        "schema_version": TOKEN_CAPTURE_SCHEMA_VERSION,
        "wire": "openai-chat",
        "provider": "vllm",
        "requested": {"logprobs": True, "top_logprobs": None, "token_ids": True},
        "prompt_token_ids": prompt,
        "completions": [completion] if status < 400 else [],
        "unavailable": unavailable or {},
    }
    if status >= 400:
        capture["unavailable"] = {
            f: {"reason": "request_failed", "detail": "the call failed"}
            for f in ("prompt_token_ids", "completion_token_ids", "logprobs")
        }
    digest = token_digest(prompt, capture["completions"])
    if digest is not None:
        capture["digest"] = digest
    return {
        "request": {
            "body": {
                "messages": messages
                if messages is not None
                else [{"role": "user", "content": f"p{prompt}"}],
                "tools": [{"type": "function", "function": {"name": t}} for t in tools],
            }
        },
        "response": {"status_code": status, "body": {}},
        "metadata": {"call_purpose": purpose, "token_capture": capture},
    }


def test_one_conversation_is_one_segment_with_masks_and_spans():
    first = call([1, 2, 3], [10, 11])
    # The next prompt re-sends history (1,2,3,10,11) plus a tool result (50,51).
    second = call([1, 2, 3, 10, 11, 50, 51], [12])
    third = call([1, 2, 3, 10, 11, 50, 51, 12, 52], [13, 14])
    report = segment_rollout([first, second, third])

    assert report["status"] == "exact"
    assert report["dropped"] == []
    [segment] = report["segments"]
    assert segment["kind"] == "agent"
    assert segment["trainable"] is True
    assert segment["start"] is None
    assert segment["prompt_ids"] == [1, 2, 3]
    assert segment["completion_ids"] == [10, 11, 50, 51, 12, 52, 13, 14]
    assert segment["action_mask"] == [1, 1, 0, 0, 1, 0, 1, 1]
    assert segment["logprobs"] == [
        *lp([10, 11]),
        0.0,
        0.0,
        *lp([12]),
        0.0,
        *lp([13, 14]),
    ]
    assert segment["call_spans"] == [
        {"call": 0, "offset": 0, "length": 2},
        {"call": 1, "offset": 4, "length": 1},
        {"call": 2, "offset": 6, "length": 2},
    ]
    assert segment["digest"] == segment_digest(segment)
    assert [c["segment"] for c in report["calls"]] == [0, 0, 0]


def test_helper_calls_are_their_own_untrained_segments():
    agent_1 = call([1, 2], [10])
    helper = call([7, 7, 7], [70], tools=())  # tool-less side call
    agent_2 = call([1, 2, 10, 50], [11])
    report = segment_rollout([agent_1, helper, agent_2])

    kinds = [(s["kind"], s["trainable"], s["calls"]) for s in report["segments"]]
    assert kinds == [("agent", True, [0, 2]), ("helper", False, [1])]
    assert report["segments"][1]["excluded"] == "kind:helper"
    assert report["status"] == "exact"


def test_labelled_helper_and_compaction_calls_are_excluded_even_with_tools():
    agent_1 = call([1, 2], [10])
    title = call([9, 9], [90], purpose="title")
    compaction = call([8, 8], [80], purpose="compaction")
    report = segment_rollout([agent_1, title, compaction])
    kinds = {s["calls"][0]: (s["kind"], s["trainable"]) for s in report["segments"]}
    assert kinds == {
        0: ("agent", True),
        1: ("helper", False),
        2: ("compaction", False),
    }
    included = segment_rollout(
        [agent_1, title, compaction],
        trainable_kinds=("agent", "subagent", "chat", "compaction"),
    )
    assert [s["trainable"] for s in included["segments"]] == [True, False, True]


def test_failed_attempt_retried_is_left_out_and_linked():
    messages = [{"role": "user", "content": "same request"}]
    first = call([1, 2], [10])
    failed = call([1, 2, 10, 50], None, status=429, messages=messages)
    retry = call([1, 2, 10, 50], [11], messages=messages)
    report = segment_rollout([first, failed, retry])

    assert report["failed_attempts"] == {"retried": 1, "unretried": 0}
    calls = report["calls"]
    assert calls[1]["status"] == "failed"
    assert calls[1]["retried_by"] == 2
    assert calls[2]["retry_of"] == 1
    assert calls[1]["segment"] is None
    [segment] = report["segments"]
    assert segment["calls"] == [0, 2]
    assert report["status"] == "exact"
    assert report["dropped"] == []


def test_failure_without_retry_is_counted_unretried():
    report = segment_rollout(
        [call([1, 2], [10]), call([1, 2, 10, 5], None, status=503)]
    )
    assert report["failed_attempts"] == {"retried": 0, "unretried": 1}
    assert report["calls"][1]["retried_by"] is None
    assert report["status"] == "exact"


def test_compaction_starts_a_new_segment():
    first = call([1, 2, 3, 4], [10])
    second = call([1, 2, 3, 4, 10, 50], [11])
    # The agent summarised its history: the prompt departs before the last
    # sampled turn (inside the previous prompt).
    compacted = call([1, 99, 98], [12])
    after = call([1, 99, 98, 12, 51], [13])
    report = segment_rollout([first, second, compacted, after])

    assert [s["calls"] for s in report["segments"]] == [[0, 1], [2, 3]]
    assert report["segments"][1]["start"] == {"reason": "compaction", "call": 1}
    assert report["segments"][1]["prompt_ids"] == [1, 99, 98]
    assert report["segments"][1]["completion_ids"] == [12, 51, 13]
    assert report["status"] == "exact"
    assert all(s["trainable"] for s in report["segments"])


def test_rerendered_turn_starts_a_new_segment_named_rerender():
    first = call([1, 2], [10, 11])
    # The template kept the whole prompt but rewrote the sampled turn.
    second = call([1, 2, 10, 77, 50], [12])
    report = segment_rollout([first, second])
    assert [s["calls"] for s in report["segments"]] == [[0], [1]]
    assert report["segments"][1]["start"] == {"reason": "rerender", "call": 0}


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        (call([1, 2, 10, 5], [11, 12], [-0.1]), "logprobs:length_mismatch"),
        (
            call(
                [1, 2, 10, 5],
                None,
                None,
                unavailable={
                    "completion_token_ids": {"reason": "not_returned", "detail": "x"}
                },
            ),
            "completion_token_ids:not_returned",
        ),
    ],
)
def test_unusable_call_is_dropped_with_its_reason_and_splits_the_segment(bad, reason):
    report = segment_rollout(
        [call([1, 2], [10]), bad, call([1, 2, 10, 5, 11, 12, 6], [13])]
    )
    assert report["dropped"] == [{"call": 1, "reason": reason}]
    assert [s["calls"] for s in report["segments"]] == [[0], [2]]
    assert report["segments"][1]["start"] == {
        "reason": "after_dropped_call",
        "call": 1,
    }
    assert report["status"] == "partial"


def test_every_call_unusable_leaves_no_trainable_segment():
    report = segment_rollout([call([1], [2], [-0.1, -0.2])])
    assert report["status"] == "none"
    assert report["segments"] == []
    assert report["dropped"] == [{"call": 0, "reason": "logprobs:length_mismatch"}]


def test_calls_without_capture_are_dropped_not_ignored():
    plain = {
        "request": {"body": {"messages": [], "tools": [{"name": "Bash"}]}},
        "response": {"status_code": 200, "body": {}},
        "metadata": {"call_purpose": "agent"},
    }
    report = segment_rollout([plain])
    assert report["dropped"] == [{"call": 0, "reason": "no_token_capture"}]
    assert report["status"] == "none"


def test_subagent_and_chat_kinds():
    main = call([1, 2], [10], tools=("Bash", "Task"))
    sub = call([5, 6], [60], tools=("Read",))
    main_2 = call([1, 2, 10, 50], [11], tools=("Bash", "Task"))
    report = segment_rollout([main, sub, main_2])
    assert [(s["kind"], s["calls"]) for s in report["segments"]] == [
        ("agent", [0, 2]),
        ("subagent", [1]),
    ]
    chat = segment_rollout([call([1], [2], tools=()), call([1, 2, 3], [4], tools=())])
    assert [(s["kind"], s["calls"]) for s in chat["segments"]] == [("chat", [0, 1])]


def test_relay_records_attest_calls_and_tag_policy_versions():
    first = call([1, 2], [10])
    second = call([1, 2, 10, 50], [11])
    relay = [
        {"status": "ok", "digest": first["metadata"]["token_capture"]["digest"], "version": 3},
        {"status": "ok", "digest": second["metadata"]["token_capture"]["digest"], "version": 4},
    ]
    report = segment_rollout([first, second], relay_calls=relay)
    assert report["attestation"]["status"] == "attested"
    assert report["attestation"]["matched"] == 2
    assert [c["policy_version"] for c in report["calls"]] == [3, 4]
    assert report["segments"][0]["policy_versions"] == [3, 4]
    assert report["segments"][0]["trainable"] is True


def test_a_call_the_relay_never_served_is_a_mismatch_and_not_trained():
    first = call([1, 2], [10])
    second = call([1, 2, 10, 50], [11])
    # The server really sampled 12 with another logprob; the store says 11.
    served = token_digest([1, 2, 10, 50], [{"index": 0, "token_ids": [12], "logprobs": [-0.3]}])
    relay = [
        {"status": "ok", "digest": first["metadata"]["token_capture"]["digest"], "version": 1},
        {"status": "ok", "digest": served, "version": 1},
    ]
    report = segment_rollout([first, second], relay_calls=relay)
    assert report["attestation"]["status"] == "mismatch"
    assert report["attestation"]["mismatched_calls"] == [1]
    [segment] = report["segments"]
    assert segment["trainable"] is False
    assert segment["excluded"] == "attestation_mismatch"
    assert report["status"] == "none"


def test_relay_calls_the_store_missed_make_attestation_partial():
    first = call([1, 2], [10])
    extra = token_digest([9], [{"index": 0, "token_ids": [9], "logprobs": [-1.0]}])
    relay = [
        {"status": "ok", "digest": first["metadata"]["token_capture"]["digest"], "version": 1},
        {"status": "ok", "digest": extra, "version": 1},
        {"status": "error", "digest": None, "version": 1},
    ]
    report = segment_rollout([first], relay_calls=relay)
    assert report["attestation"]["status"] == "partial"
    assert report["attestation"]["relay_only"] == 1
    assert report["segments"][0]["trainable"] is True


def test_no_relay_means_attestation_unavailable():
    report = segment_rollout([call([1], [2])])
    assert report["attestation"]["status"] == "unavailable"
    assert report["calls"][0]["attested"] is None


def test_routing_passes_through_to_the_segment():
    routing = {"source": "sglang", "encoding": "int32-base64", "data": "AAAA", "start": 0}
    first = call([1, 2], [10], routing=routing)
    second = call([1, 2, 10, 5], [11], routing=dict(routing, data="BBBB"))
    report = segment_rollout([first, second])
    [segment] = report["segments"]
    assert segment["routing"] == [
        {"call": 0, "sequence_length": 3, **routing},
        {"call": 1, "sequence_length": 5, **dict(routing, data="BBBB")},
    ]


def test_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="unknown segment kinds"):
        segment_rollout([], trainable_kinds=("agent", "tools"))


# --- token_capture: digests, id fallback, routing --------------------------------


def _record(response: dict[str, Any], *, body=None, stream=None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "event": "success",
        "token_capture": {
            "enabled": True,
            "wire": "openai-chat",
            "provider": "vllm",
            "request": "chat",
            "logprobs": True,
            "token_ids": True,
        },
        "request": {"body": body or {"logprobs": True}},
        "response": response,
    }
    if stream is not None:
        record["stream_tokens"] = stream
    return record


def test_capture_takes_ids_from_logprob_entries_and_digests_the_call():
    response = {
        "prompt_token_ids": [1, 2],
        "choices": [
            {
                "index": 0,
                "logprobs": {
                    "content": [
                        {"token": "token_id:10", "logprob": -0.5},
                        {"token": "x", "token_id": 11, "logprob": -0.25},
                    ]
                },
            }
        ],
    }
    capture = build_token_capture(_record(response))
    assert capture is not None
    assert capture["completions"][0]["token_ids"] == [10, 11]
    assert capture["unavailable"] == {}
    assert capture["digest"] == token_digest(
        [1, 2], [{"index": 0, "token_ids": [10, 11], "logprobs": [-0.5, -0.25]}]
    )


def test_partial_logprob_entry_ids_are_not_a_capture():
    response = {
        "prompt_token_ids": [1],
        "choices": [
            {
                "index": 0,
                "logprobs": {
                    "content": [
                        {"token": "token_id:10", "logprob": -0.5},
                        {"token": "plain", "logprob": -0.25},
                    ]
                },
            }
        ],
    }
    capture = build_token_capture(_record(response))
    assert capture is not None
    assert capture["completions"][0]["token_ids"] is None
    assert "completion_token_ids" in capture["unavailable"]
    assert "digest" not in capture


def test_vllm_routed_experts_pass_through_and_leave_the_raw_body():
    response = {
        "prompt_token_ids": [1, 2],
        "choices": [
            {
                "index": 0,
                "token_ids": [10],
                "routed_experts": "npy-b64",
                "logprobs": {"content": [{"token": "a", "logprob": -0.1}]},
            }
        ],
    }
    capture = build_token_capture(
        _record(response, body={"logprobs": True, "routed_experts_prompt_start": 2})
    )
    assert capture is not None
    assert capture["completions"][0]["routing"] == {
        "source": "vllm",
        "encoding": "npy-base64",
        "data": "npy-b64",
        "start": 2,
    }
    stripped = strip_captured_token_ids(response)
    assert "routed_experts" not in stripped["choices"][0]
    assert "token_ids" not in stripped["choices"][0]


def test_sglang_routed_experts_from_a_stream():
    from benchflow.providers.litellm_token_capture_patch import (
        STREAM_TOKENS_KEY,
        record_stream_chunk,
    )

    details: dict[str, Any] = {}
    record_stream_chunk(
        details,
        {
            "choices": [
                {"index": 0, "delta": {"content": "a"}, "logprobs": {"content": [{"token": "a", "logprob": -0.1}]}}
            ]
        },
    )
    record_stream_chunk(
        details,
        {
            "choices": [],
            "sglext": {"input_ids": [1, 2], "output_ids": [[10]], "routed_experts": "i32-b64"},
        },
    )
    capture = build_token_capture(
        _record(
            {"choices": [{"index": 0}]},
            body={"logprobs": True, "extra_body": {"routed_experts_start_len": 1}},
            stream=details[STREAM_TOKENS_KEY],
        )
    )
    assert capture is not None
    assert capture["prompt_token_ids"] == [1, 2]
    assert capture["completions"][0]["token_ids"] == [10]
    assert capture["completions"][0]["routing"] == {
        "source": "sglang",
        "encoding": "int32-base64",
        "data": "i32-b64",
        "start": 1,
    }
    assert capture["digest"].startswith("sha256:")
