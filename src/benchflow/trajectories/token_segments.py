"""Exact token segments of one rollout, ready for a trainer.

:func:`segment_rollout` turns a rollout's captured model calls
(``trajectory/llm_trajectory.jsonl`` rows, see
:mod:`benchflow.trajectories.token_capture`) into token segments. A segment
is one span of a conversation in which every prompt extends the previous
prompt and sampled tokens exactly (token-in/token-out)::

    prompt_ids       the first call's prompt, as the server tokenized it
    completion_ids   everything after it: each call's sampled tokens, and the
                     tokens the next prompt adds after them (tool results,
                     the next turn's template)
    action_mask      1 for a token the policy sampled, 0 for one the
                     environment added
    logprobs         the server's logprob for each sampled token, 0.0 where
                     the mask is 0

What makes a rollout's calls more than one segment, and what is left out, is
reported, never dropped silently:

- **Conversations.** Calls that offer tools are grouped by tool set: the
  first tool set is the agent loop (``kind: "agent"``), another is a
  subagent (``"subagent"``). A call without tools in a rollout that uses
  tools is a helper (``"helper"``: Claude Code's short side prompts, title
  generation); calls the gateway labelled ``title``, ``summary`` or
  ``helper`` are helpers too, and ``compaction`` calls (the agent
  summarising its own history) are their own kind. A rollout without any
  tool-offering call is one ``"chat"`` conversation. By default helpers and
  compaction calls are not trainable.
- **Breaks.** Within a conversation a new segment starts when a prompt does
  not extend the previous prompt and sampled tokens: ``rerender`` when it
  keeps the previous prompt but rewrote the sampled turn (a chat template
  that canonicalises tool calls or drops earlier reasoning), ``compaction``
  when it rewrote the history before that (the agent compacted or refreshed
  its context), and ``after_dropped_call`` after a call whose tokens could
  not be used.
- **Failed attempts.** A call the server or gateway failed has no tokens and
  is never part of a segment. When a later call sends the same request
  (messages and tools), it is that call's retry: the failure is recorded as
  ``retried_by`` and the retry as ``retry_of``. A failure without a retry is
  ``unretried`` (the agent gave up, or the rollout ended).
- **Dropped calls.** A successful call without prompt ids, sampled ids or
  logprobs for its first choice, or with a different number of ids and
  logprobs, is listed in ``dropped`` with its reason.

With ``relay_calls`` (the calls BenchFlow's policy relay forwarded for this
rollout, :mod:`benchflow.training.relay`) each captured call is matched to
the relay's record of the server's raw answer by token digest: that
attests that the server, the gateway's store and the segments hold the same
tokens, and gives each call the policy version that served it. A captured
call whose digest the relay never saw is a mismatch, and its segment is not
trainable.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from benchflow.trajectories.token_capture import (
    TOKEN_CAPTURE_METADATA_KEY,
    TOKEN_CAPTURE_SCHEMA_VERSION,
    token_digest,
)

TOKEN_SEGMENTS_SCHEMA_VERSION = "benchflow.token-segments.v1"
DEFAULT_TRAINABLE_KINDS: tuple[str, ...] = ("agent", "subagent", "chat")
KINDS: tuple[str, ...] = ("agent", "subagent", "chat", "helper", "compaction")
HELPER_PURPOSES = frozenset({"title", "summary", "helper"})
COMPACTION_PURPOSES = frozenset({"compaction"})
BREAK_REASONS: tuple[str, ...] = ("rerender", "compaction", "after_dropped_call")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    ids = [v for v in value if isinstance(v, int) and not isinstance(v, bool)]
    return ids if len(ids) == len(value) else None


def _capture(exchange: Mapping[str, Any]) -> dict[str, Any] | None:
    capture = _dict(_dict(exchange.get("metadata")).get(TOKEN_CAPTURE_METADATA_KEY))
    if capture.get("schema_version") != TOKEN_CAPTURE_SCHEMA_VERSION:
        return None
    return capture


def _request_body(exchange: Mapping[str, Any]) -> dict[str, Any]:
    return _dict(_dict(exchange.get("request")).get("body"))


def _tool_names(exchange: Mapping[str, Any]) -> tuple[str, ...]:
    names = []
    for tool in _request_body(exchange).get("tools") or []:
        tool = _dict(tool)
        name = _dict(tool.get("function")).get("name") or tool.get("name")
        if name:
            names.append(str(name))
    return tuple(sorted(names))


def request_signature(exchange: Mapping[str, Any]) -> str:
    """What a retry of this call sends again: its messages (or input) and tools."""
    body = _request_body(exchange)
    payload = {
        "messages": body.get("messages"),
        "input": body.get("input"),
        "tools": body.get("tools"),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def call_failed(exchange: Mapping[str, Any]) -> bool:
    """True when the call has no usable response (the server or gateway failed it)."""
    status = _dict(exchange.get("response")).get("status_code")
    if isinstance(status, int) and not isinstance(status, bool) and status >= 400:
        return True
    capture = _capture(exchange)
    if capture is not None:
        for why in _dict(capture.get("unavailable")).values():
            if _dict(why).get("reason") == "request_failed":
                return True
    return False


def _first_choice(capture: Mapping[str, Any]) -> dict[str, Any]:
    completions = capture.get("completions")
    if isinstance(completions, list) and completions:
        return _dict(completions[0])
    return {}


def _unusable_reason(capture: Mapping[str, Any] | None) -> str | None:
    """Why a successful call's tokens cannot enter a segment; None when they can."""
    if capture is None:
        return "no_token_capture"
    choice = _first_choice(capture)
    if not choice:
        return "no_choices"
    unavailable = _dict(capture.get("unavailable"))
    for field in ("prompt_token_ids", "completion_token_ids", "logprobs"):
        why = _dict(unavailable.get(field))
        if why:
            return f"{field}:{why.get('reason', 'unknown')}"
    prompt = _int_list(capture.get("prompt_token_ids"))
    ids = _int_list(choice.get("token_ids"))
    logprobs = choice.get("logprobs")
    if prompt is None:
        return "prompt_token_ids:missing"
    if ids is None:
        return "completion_token_ids:missing"
    if not isinstance(logprobs, list):
        return "logprobs:missing"
    if len(ids) != len(logprobs):
        return "logprobs:length_mismatch"
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in logprobs):
        return "logprobs:not_numeric"
    return None


def _segment_digest(segment: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {
            "prompt_ids": segment["prompt_ids"],
            "completion_ids": segment["completion_ids"],
            "action_mask": segment["action_mask"],
            "logprobs": segment["logprobs"],
        },
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def segment_digest(segment: Mapping[str, Any]) -> str:
    """The digest :func:`segment_rollout` records for a segment.

    A trainer can recompute it over the lists it received to check they are
    the ones BenchFlow built (``TokenSegment.verify``).
    """
    return _segment_digest(segment)


class _Builder:
    """One open segment of a conversation."""

    def __init__(
        self,
        *,
        thread: int,
        kind: str,
        start: dict[str, Any] | None,
        index: int,
        prompt: list[int],
        sampled: list[int],
        logprobs: list[float],
        routing: dict[str, Any] | None,
    ) -> None:
        self.thread = thread
        self.kind = kind
        self.start = start
        self.prompt_ids = list(prompt)
        self.completion_ids: list[int] = []
        self.action_mask: list[int] = []
        self.logprobs: list[float] = []
        self.calls: list[int] = []
        self.spans: list[dict[str, int]] = []
        self.routing: list[dict[str, Any]] = []
        self.last_prompt = list(prompt)
        self._append(index, sampled, logprobs, routing, full_prompt=prompt)

    @property
    def expected_prefix(self) -> list[int]:
        return self.prompt_ids + self.completion_ids

    def _append(
        self,
        index: int,
        sampled: list[int],
        logprobs: list[float],
        routing: dict[str, Any] | None,
        *,
        full_prompt: list[int],
    ) -> None:
        offset = len(self.completion_ids)
        self.completion_ids += sampled
        self.action_mask += [1] * len(sampled)
        self.logprobs += [float(v) for v in logprobs]
        self.calls.append(index)
        self.spans.append({"call": index, "offset": offset, "length": len(sampled)})
        if routing is not None:
            # Positions in the segment's full sequence (prompt + completion):
            # the call's own sequence is this call's prompt plus its sampled
            # tokens, which is exactly the segment up to here.
            self.routing.append(
                {
                    "call": index,
                    "sequence_length": len(full_prompt) + len(sampled),
                    **routing,
                }
            )
        self.last_prompt = list(full_prompt)

    def extends(self, prompt: list[int]) -> bool:
        expected = self.expected_prefix
        return prompt[: len(expected)] == expected

    def break_reason(self, prompt: list[int]) -> str:
        """Where ``prompt`` departs from this segment: the sampled turn or before."""
        last = self.last_prompt
        return "rerender" if prompt[: len(last)] == last else "compaction"

    def extend(
        self,
        index: int,
        prompt: list[int],
        sampled: list[int],
        logprobs: list[float],
        routing: dict[str, Any] | None,
    ) -> None:
        added = prompt[len(self.expected_prefix) :]
        self.completion_ids += added
        self.action_mask += [0] * len(added)
        self.logprobs += [0.0] * len(added)
        self._append(index, sampled, logprobs, routing, full_prompt=prompt)

    def close(self, number: int) -> dict[str, Any]:
        segment: dict[str, Any] = {
            "segment": number,
            "thread": self.thread,
            "kind": self.kind,
            "calls": list(self.calls),
            "start": self.start,
            "prompt_ids": self.prompt_ids,
            "completion_ids": self.completion_ids,
            "action_mask": self.action_mask,
            "logprobs": self.logprobs,
            "call_spans": self.spans,
            "routing": self.routing or None,
        }
        segment["digest"] = _segment_digest(segment)
        return segment


def _purpose(exchange: Mapping[str, Any]) -> str:
    purpose = _dict(exchange.get("metadata")).get("call_purpose")
    return purpose if isinstance(purpose, str) and purpose else "agent"


def _match_relay(
    calls: list[dict[str, Any]],
    relay_calls: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Match captured calls to the relay's records by digest; the attestation."""
    if relay_calls is None:
        for call in calls:
            call["attested"] = None
            call["policy_version"] = None
        return {
            "status": "unavailable",
            "detail": "no policy relay record for this rollout",
        }
    pool: dict[str, deque[Mapping[str, Any]]] = defaultdict(deque)
    relay_ok = 0
    relay_without_digest = 0
    for record in relay_calls:
        if record.get("status") != "ok":
            continue
        relay_ok += 1
        digest = record.get("digest")
        if isinstance(digest, str) and digest:
            pool[digest].append(record)
        else:
            relay_without_digest += 1
    matched = 0
    mismatched: list[int] = []
    for call in calls:
        call["attested"] = None
        call["policy_version"] = None
        if call["status"] != "ok" or call["digest"] is None:
            continue
        queue = pool.get(call["digest"])
        if queue:
            record = queue.popleft()
            call["attested"] = True
            call["policy_version"] = record.get("version")
            if record.get("version_end") not in (None, record.get("version")):
                call["policy_version_end"] = record.get("version_end")
            matched += 1
        else:
            call["attested"] = False
            mismatched.append(call["index"])
    relay_only = sum(len(queue) for queue in pool.values())
    if mismatched:
        status = "mismatch"
        detail = (
            f"{len(mismatched)} captured call(s) hold tokens the relay never "
            "received from the server"
        )
    elif relay_only or relay_without_digest:
        status = "partial"
        detail = (
            f"{relay_only} relay call(s) never reached the gateway's store"
            if relay_only
            else f"{relay_without_digest} relay call(s) came back without "
            "complete token data"
        )
    else:
        status = "attested"
        detail = "every captured call matches the server's answer at the relay"
    return {
        "status": status,
        "detail": detail,
        "relay_calls": relay_ok,
        "matched": matched,
        "mismatched_calls": mismatched,
        "relay_only": relay_only,
        "relay_without_digest": relay_without_digest,
    }


def segment_rollout(
    exchanges: Sequence[Mapping[str, Any]],
    *,
    relay_calls: Iterable[Mapping[str, Any]] | None = None,
    trainable_kinds: Iterable[str] = DEFAULT_TRAINABLE_KINDS,
) -> dict[str, Any]:
    """Exact token segments of one rollout, and an account of every call.

    ``exchanges`` are the rows of ``trajectory/llm_trajectory.jsonl`` in
    order. Returns a ``benchflow.token-segments.v1`` mapping::

        {"schema_version", "status": "exact" | "partial" | "none",
         "calls": [{"index", "status", "purpose", "kind", "thread", "digest",
                    "unusable", "retry_of", "retried_by", "segment",
                    "attested", "policy_version"}, ...],
         "segments": [{"segment", "thread", "kind", "trainable",
                       "excluded", "calls", "start", "prompt_ids",
                       "completion_ids", "action_mask", "logprobs",
                       "call_spans", "routing", "policy_versions",
                       "digest"}, ...],
         "dropped": [{"call", "reason"}, ...],
         "failed_attempts": {"retried": n, "unretried": n},
         "attestation": {...}}

    ``status`` is ``exact`` when every successful call of a trainable kind
    is inside a segment, ``partial`` when some were dropped but a trainable
    segment remains, and ``none`` when no trainable segment exists.
    """
    kinds = frozenset(trainable_kinds)
    unknown = kinds - set(KINDS)
    if unknown:
        raise ValueError(f"unknown segment kinds {sorted(unknown)}; use {KINDS}")
    calls: list[dict[str, Any]] = []
    for index, exchange in enumerate(exchanges):
        capture = _capture(exchange)
        failed = call_failed(exchange)
        calls.append(
            {
                "index": index,
                "status": "failed" if failed else "ok",
                "purpose": _purpose(exchange),
                "kind": None,
                "thread": None,
                "digest": (capture or {}).get("digest")
                if capture is not None and not failed
                else None,
                "unusable": None if failed else _unusable_reason(capture),
                "retry_of": None,
                "retried_by": None,
                "segment": None,
            }
        )
        if calls[-1]["digest"] is None and capture is not None and not failed:
            # Captures written before digests existed: compute it the same way.
            calls[-1]["digest"] = token_digest(
                capture.get("prompt_token_ids"), capture.get("completions")
            )

    # Failed attempts and the retries that replaced them.
    pending: dict[str, list[int]] = defaultdict(list)
    for index, exchange in enumerate(exchanges):
        signature = request_signature(exchange)
        if calls[index]["status"] == "failed":
            pending[signature].append(index)
            continue
        earlier = pending.pop(signature, [])
        if earlier:
            calls[index]["retry_of"] = earlier[-1]
            for failed_index in earlier:
                calls[failed_index]["retried_by"] = index
    retried = sum(1 for c in calls if c["status"] == "failed" and c["retried_by"] is not None)
    unretried = sum(1 for c in calls if c["status"] == "failed" and c["retried_by"] is None)

    # Conversations over the successful calls.
    ok = [c for c in calls if c["status"] == "ok"]
    tools_of = {c["index"]: _tool_names(exchanges[c["index"]]) for c in ok}
    agentic = any(
        tools_of[c["index"]]
        for c in ok
        if c["purpose"] not in HELPER_PURPOSES | COMPACTION_PURPOSES
    )
    main_tools: tuple[str, ...] | None = None
    thread_of_tools: dict[tuple[str, ...], int] = {}
    threads: list[dict[str, Any]] = []

    def _new_thread(kind: str) -> int:
        threads.append({"thread": len(threads), "kind": kind, "calls": []})
        return len(threads) - 1

    for call in ok:
        tools = tools_of[call["index"]]
        if call["purpose"] in COMPACTION_PURPOSES:
            number = _new_thread("compaction")
        elif call["purpose"] in HELPER_PURPOSES or (agentic and not tools):
            number = _new_thread("helper")
        elif not agentic:
            if () not in thread_of_tools:
                thread_of_tools[()] = _new_thread("chat")
            number = thread_of_tools[()]
        else:
            if main_tools is None:
                main_tools = tools
            if tools not in thread_of_tools:
                thread_of_tools[tools] = _new_thread(
                    "agent" if tools == main_tools else "subagent"
                )
            number = thread_of_tools[tools]
        threads[number]["calls"].append(call["index"])
        call["thread"] = number
        call["kind"] = threads[number]["kind"]

    attestation = _match_relay(calls, relay_calls)

    # Segments within each conversation.
    segments: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    by_index = {c["index"]: c for c in calls}
    for thread in threads:
        builder: _Builder | None = None
        pending_start: dict[str, Any] | None = None
        for index in thread["calls"]:
            call = by_index[index]
            if call["unusable"] is not None:
                dropped.append({"call": index, "reason": call["unusable"]})
                if builder is not None:
                    segments.append(builder.close(len(segments)))
                    builder = None
                pending_start = {"reason": "after_dropped_call", "call": index}
                continue
            capture = _capture(exchanges[index]) or {}
            choice = _first_choice(capture)
            prompt = _int_list(capture.get("prompt_token_ids")) or []
            sampled = _int_list(choice.get("token_ids")) or []
            logprobs = [float(v) for v in choice.get("logprobs") or []]
            routing = _dict(choice.get("routing")) or None
            if builder is not None and builder.extends(prompt):
                builder.extend(index, prompt, sampled, logprobs, routing)
                continue
            if builder is not None:
                reason = builder.break_reason(prompt)
                previous = builder.calls[-1]
                segments.append(builder.close(len(segments)))
                pending_start = {"reason": reason, "call": previous}
            builder = _Builder(
                thread=thread["thread"],
                kind=thread["kind"],
                start=pending_start,
                index=index,
                prompt=prompt,
                sampled=sampled,
                logprobs=logprobs,
                routing=routing,
            )
            pending_start = None
        if builder is not None:
            segments.append(builder.close(len(segments)))
    segments.sort(key=lambda s: (s["calls"][0], s["segment"]))
    for number, segment in enumerate(segments):
        segment["segment"] = number
        versions = []
        excluded = None
        for index in segment["calls"]:
            by_index[index]["segment"] = number
            versions.append(by_index[index].get("policy_version"))
            if by_index[index].get("attested") is False:
                excluded = "attestation_mismatch"
        if segment["kind"] not in kinds:
            excluded = excluded or f"kind:{segment['kind']}"
        segment["policy_versions"] = versions
        segment["trainable"] = excluded is None
        segment["excluded"] = excluded
    dropped.sort(key=lambda d: d["call"])

    trainable_dropped = [
        d for d in dropped if by_index[d["call"]]["kind"] in kinds
    ]
    trainable_segments = [s for s in segments if s["trainable"]]
    if not trainable_segments:
        status = "none"
    elif trainable_dropped or any(
        not s["trainable"] and s["kind"] in kinds for s in segments
    ):
        status = "partial"
    else:
        status = "exact"
    return {
        "schema_version": TOKEN_SEGMENTS_SCHEMA_VERSION,
        "status": status,
        "calls": calls,
        "threads": threads,
        "segments": segments,
        "dropped": dropped,
        "failed_attempts": {"retried": retried, "unretried": unretried},
        "attestation": attestation,
    }


def read_llm_trajectory(rollout_dir: Any) -> list[dict[str, Any]] | None:
    """The rows of ``trajectory/llm_trajectory.jsonl``; None without the file."""
    from pathlib import Path

    path = Path(rollout_dir) / "trajectory" / "llm_trajectory.jsonl"
    if not path.is_file():
        return None
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


__all__ = [
    "BREAK_REASONS",
    "DEFAULT_TRAINABLE_KINDS",
    "KINDS",
    "TOKEN_SEGMENTS_SCHEMA_VERSION",
    "call_failed",
    "read_llm_trajectory",
    "request_signature",
    "segment_digest",
    "segment_rollout",
]
