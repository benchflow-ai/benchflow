"""Per-call token ids and logprobs captured at BenchFlow's LiteLLM gateway.

With ``BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1`` the gateway asks the model server
for sampled-token logprobs (and, on routes that support it, token ids) and
records what came back. :func:`build_token_capture` turns one gateway callback
record into the ``token_capture`` block stored in that exchange's ``metadata``
in ``trajectory/llm_trajectory.jsonl``. Schema ``benchflow.token_capture.v1``::

    {
      "schema_version": "benchflow.token_capture.v1",
      "wire": "openai-chat" | "openai-responses" | "anthropic-messages" | "gemini" | "other",
      "provider": "<BenchFlow route provider, e.g. vllm>",
      "requested": {"logprobs": bool, "top_logprobs": int | null, "token_ids": bool},
      "prompt_token_ids": [int, ...] | null,
      "completions": [
        {
          "index": 0,
          "token_ids": [int, ...] | null,
          "tokens": [str, ...] | null,
          "logprobs": [float, ...] | null,
          "top_logprobs": [[{"token": str, "logprob": float}, ...], ...] | null
        }
      ],
      "unavailable": {
        "<prompt_token_ids | completion_token_ids | logprobs>":
            {"reason": "<code>", "detail": "<text>"}
      }
    }

Every field in ``unavailable``'s key set is either captured for every choice or
listed there with a reason, never silently missing. Reason codes:

- ``provider_api_unsupported``: the upstream API has no such output (Anthropic
  Messages and Bedrock expose no logprobs or token ids; the OpenAI Responses
  and Gemini APIs have logprobs but no token ids).
- ``not_requested``: capture is on, but the gateway did not ask this route for
  the field (token ids on a route not known to accept ``return_token_ids``;
  logprobs on native Gemini ``generateContent`` pass-through calls).
- ``not_returned``: requested, but the response carried none (the server
  ignored the parameter or the model does not support it).
- ``request_failed``: the call failed, so there is no response.

``tokens``/``logprobs``/``top_logprobs`` are parallel per sampled token; token
ids come from the server (``return_token_ids`` on vLLM; SGLang's ``sglext``
extension, ``return_input_ids_in_sglext`` / ``return_output_ids_in_sglext``)
and are never re-derived by tokenizing text. ``prompt_token_ids`` is the prompt exactly as the server
tokenized it, including the chat template. Streamed calls are assembled from
the chunks (see ``providers/litellm_token_capture_patch.py``).
"""

from __future__ import annotations

from typing import Any

TOKEN_CAPTURE_SCHEMA_VERSION = "benchflow.token_capture.v1"
TOKEN_CAPTURE_METADATA_KEY = "token_capture"
_FIELDS = ("prompt_token_ids", "completion_token_ids", "logprobs")
_RESPONSES_LOGPROBS_INCLUDE = "message.output_text.logprobs"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    ids = [v for v in value if isinstance(v, int) and not isinstance(v, bool)]
    return ids if len(ids) == len(value) else None


def _first(*values: list[int] | None) -> list[int] | None:
    # An empty list is a real (empty) capture, so only None falls through.
    return next((v for v in values if v is not None), None)


def _split_logprobs(
    content: Any,
) -> tuple[list[str], list[float], list[list[dict[str, Any]]] | None] | None:
    """Split OpenAI-style per-token entries into parallel lists."""
    if not isinstance(content, list):
        return None
    tokens: list[str] = []
    logprobs: list[float] = []
    tops: list[list[dict[str, Any]]] = []
    for item in content:
        entry = _dict(item)
        value = entry.get("logprob")
        if not isinstance(value, int | float):
            return None
        tokens.append(str(entry.get("token", "")))
        logprobs.append(float(value))
        tops.append(
            [
                {"token": str(top.get("token", "")), "logprob": top.get("logprob")}
                for top in (entry.get("top_logprobs") or [])
                if isinstance(top, dict)
            ]
        )
    return tokens, logprobs, (tops if any(tops) else None)


def _choice_field(choice: dict[str, Any], *names: str) -> list[int] | None:
    """First int list among ``names`` on a choice or its provider extras.

    LiteLLM moves non-OpenAI choice fields into ``provider_specific_fields``.
    """
    extra = _dict(choice.get("provider_specific_fields"))
    for name in names:
        for source in (choice, extra):
            value = _int_list(source.get(name))
            if value is not None:
                return value
    return None


def _sglext(response: dict[str, Any]) -> dict[str, Any]:
    """SGLang's response-level extension (non-streamed body)."""
    return _dict(response.get("sglext"))


def _sglext_output_ids(response: dict[str, Any], index: Any) -> list[int] | None:
    output_ids = _sglext(response).get("output_ids")
    if not isinstance(output_ids, list) or not isinstance(index, int):
        return None
    if 0 <= index < len(output_ids):
        return _int_list(output_ids[index])
    return None


def _prompt_token_ids(
    response: dict[str, Any], stream: dict[str, Any]
) -> list[int] | None:
    # vLLM: top level (first chunk when streaming); SGLang: ``sglext.input_ids``
    # (final chunk when streaming) or, with ``return_token_ids``, on each choice.
    return _first(
        _int_list(response.get("prompt_token_ids")),
        _int_list(_sglext(response).get("input_ids")),
        _int_list(stream.get("prompt_token_ids")),
        *(
            _choice_field(_dict(choice), "prompt_token_ids")
            for choice in response.get("choices") or []
        ),
    )


def _chat_completions(
    response: dict[str, Any], stream: dict[str, Any]
) -> list[dict[str, Any]]:
    streamed = _dict(stream.get("choices"))
    completions = []
    for position, raw in enumerate(response.get("choices") or []):
        choice = _dict(raw)
        index = choice.get("index", position)
        chunk_data = _dict(streamed.get(str(index)))
        # vLLM ``token_ids``; SGLang ``response_token_ids``.
        token_ids = _first(
            _choice_field(choice, "token_ids", "response_token_ids"),
            _sglext_output_ids(response, index),
            _int_list(chunk_data.get("token_ids")),
        )
        content = _dict(choice.get("logprobs")).get("content")
        if content is None:
            content = chunk_data.get("logprobs")
        completions.append(_completion(index, token_ids, _split_logprobs(content)))
    return completions


def _responses_completions(response: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[Any] | None = None
    for item in response.get("output") or []:
        for part in _dict(item).get("content") or []:
            part = _dict(part)
            if part.get("type") == "output_text" and isinstance(
                part.get("logprobs"), list
            ):
                content = [*(content or []), *part["logprobs"]]
    return [_completion(0, None, _split_logprobs(content))]


def _completion(
    index: Any,
    token_ids: list[int] | None,
    split: tuple[list[str], list[float], list[list[dict[str, Any]]] | None] | None,
) -> dict[str, Any]:
    tokens, logprobs, tops = split if split else (None, None, None)
    return {
        "index": index if isinstance(index, int) else 0,
        "token_ids": token_ids,
        "tokens": tokens,
        "logprobs": logprobs,
        "top_logprobs": tops,
    }


def _missing_reason(
    field: str, plan: dict[str, Any], requested: bool
) -> dict[str, str]:
    wire = plan.get("wire")
    provider = plan.get("provider") or "this route"
    if wire == "anthropic-messages":
        return {
            "reason": "provider_api_unsupported",
            "detail": "the Anthropic Messages API (direct or Bedrock) returns "
            "no token ids or logprobs",
        }
    if wire in {"openai-responses", "gemini"} and field != "logprobs":
        api = "OpenAI Responses" if wire == "openai-responses" else "Gemini"
        return {
            "reason": "provider_api_unsupported",
            "detail": f"the {api} API returns logprobs but no token ids",
        }
    if not requested:
        if field == "logprobs":
            detail = f"the gateway does not request logprobs on {wire} routes"
        else:
            detail = (
                f"route provider {provider!r} is not a vllm or sglang route; "
                "set BENCHFLOW_CAPTURE_TOKEN_IDS=1 to request vLLM's "
                "return_token_ids"
            )
        return {"reason": "not_requested", "detail": detail}
    return {
        "reason": "not_returned",
        "detail": "requested, but the provider response did not include it",
    }


def _requested(plan: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    """What the upstream request actually asked for (the recorded body wins)."""
    include = body.get("include")
    logprobs = body.get("logprobs") is True or (
        isinstance(include, list) and _RESPONSES_LOGPROBS_INCLUDE in include
    )
    extra = _dict(body.get("extra_body"))
    token_ids = any(
        extra.get(flag) is True or body.get(flag) is True
        for flag in ("return_token_ids", "return_output_ids_in_sglext")
    )
    top = body.get("top_logprobs")
    return {
        "logprobs": logprobs or bool(plan.get("logprobs")),
        "top_logprobs": top if isinstance(top, int) else plan.get("top_logprobs"),
        "token_ids": token_ids or bool(plan.get("token_ids")),
    }


def build_token_capture(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return the ``token_capture`` block for one gateway callback record.

    Returns None when the gateway did not run with token capture enabled.
    """
    plan = record.get("token_capture")
    if not isinstance(plan, dict) or not plan.get("enabled"):
        return None
    plan = dict(plan)
    requested = _requested(plan, _dict(_dict(record.get("request")).get("body")))
    response = _dict(record.get("response"))
    if plan.get("wire") == "openai-responses" and isinstance(
        response.get("choices"), list
    ):
        # A ``-responses-bridge`` alias: LiteLLM sent the call to a chat backend.
        plan["wire"] = "openai-chat"
    capture: dict[str, Any] = {
        "schema_version": TOKEN_CAPTURE_SCHEMA_VERSION,
        "wire": plan.get("wire") or "other",
        "provider": plan.get("provider"),
        "requested": requested,
        "prompt_token_ids": None,
        "completions": [],
        "unavailable": {},
    }
    if record.get("event") != "success":
        capture["unavailable"] = {
            field: {"reason": "request_failed", "detail": "the call failed"}
            for field in _FIELDS
        }
        return capture

    stream = _dict(record.get("stream_tokens"))
    if isinstance(response.get("output"), list):
        capture["completions"] = _responses_completions(response)
    elif isinstance(response.get("choices"), list):
        capture["completions"] = _chat_completions(response, stream)
        capture["prompt_token_ids"] = _prompt_token_ids(response, stream)

    completions = capture["completions"]
    captured = {
        "prompt_token_ids": capture["prompt_token_ids"] is not None,
        "completion_token_ids": bool(completions)
        and all(c["token_ids"] is not None for c in completions),
        "logprobs": bool(completions)
        and all(c["logprobs"] is not None for c in completions),
    }
    for field in _FIELDS:
        if not captured[field]:
            was_requested = requested[
                "logprobs" if field == "logprobs" else "token_ids"
            ]
            capture["unavailable"][field] = _missing_reason(field, plan, was_requested)
    return capture


_TOKEN_ID_FIELDS = ("prompt_token_ids", "token_ids", "response_token_ids")


def _without_token_ids(value: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in value.items() if k not in _TOKEN_ID_FIELDS}


def strip_captured_token_ids(body: dict[str, Any]) -> dict[str, Any]:
    """Drop token ids already moved into ``token_capture`` from a response body.

    Prompt token ids are the bulk of a captured call; keeping them once, in the
    ``token_capture`` block, avoids doubling ``llm_trajectory.jsonl``. Provider
    logprobs stay in the raw body as before.
    """
    cleaned = _without_token_ids(body)
    if isinstance(body.get("sglext"), dict):
        rest = {
            k: v
            for k, v in body["sglext"].items()
            if k not in {"input_ids", "output_ids"}
        }
        if rest:
            cleaned["sglext"] = rest
        else:
            cleaned.pop("sglext", None)
    if isinstance(body.get("choices"), list):
        choices = []
        for raw in body["choices"]:
            choice = _without_token_ids(_dict(raw))
            extra = choice.get("provider_specific_fields")
            if isinstance(extra, dict):
                choice["provider_specific_fields"] = _without_token_ids(extra)
            choices.append(choice)
        cleaned["choices"] = choices
    return cleaned


# --- Coverage: is a capture training-grade? -------------------------------


def _complete(capture: dict[str, Any]) -> bool:
    """Prompt ids, and for every choice sampled ids with one logprob each."""
    completions = capture.get("completions")
    return (
        _int_list(capture.get("prompt_token_ids")) is not None
        and isinstance(completions, list)
        and bool(completions)
        and all(
            _int_list(_dict(c).get("token_ids")) is not None
            and isinstance(_dict(c).get("logprobs"), list)
            for c in completions
        )
        and not _length_mismatch(capture)
    )


def _length_mismatch(capture: dict[str, Any]) -> bool:
    """Whether a choice has sampled token ids and logprobs of different lengths.

    A trainer pairs the i-th sampled id with the i-th logprob, so such a call
    cannot be trained on; it is counted as ``logprobs:length_mismatch``.
    """
    completions = capture.get("completions")
    for choice in completions if isinstance(completions, list) else []:
        ids, logprobs = _dict(choice).get("token_ids"), _dict(choice).get("logprobs")
        if (
            isinstance(ids, list)
            and isinstance(logprobs, list)
            and len(ids) != len(logprobs)
        ):
            return True
    return False


def _tool_names(exchange: dict[str, Any]) -> tuple[str, ...]:
    body = _dict(_dict(exchange.get("request")).get("body"))
    names = []
    for tool in body.get("tools") or []:
        tool = _dict(tool)
        name = _dict(tool.get("function")).get("name") or tool.get("name")
        if name:
            names.append(str(name))
    return tuple(sorted(names))


def conversation_threads(exchanges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split a rollout's captured calls into the conversations they belong to.

    One agent run can hold several conversations: the agent loop, helper
    calls (Claude Code's short tool-less prompts, e.g. a session title) and
    subagents with their own tool set. Each is a separate token stream. When
    any captured call offers tools, calls are grouped by their tool set
    (``kind: "agent"``) and every tool-less call is its own one-call
    conversation (``kind: "helper"``). A rollout without any tool-offering
    call is one conversation (``kind: "chat"``). Returns ``[{"thread",
    "kind", "calls": [exchange index, ...]}]`` in order of first call.
    """
    captured = [
        (index, _tool_names(exchange))
        for index, exchange in enumerate(exchanges)
        if _dict(_dict(exchange.get("metadata")).get(TOKEN_CAPTURE_METADATA_KEY)).get(
            "schema_version"
        )
        == TOKEN_CAPTURE_SCHEMA_VERSION
    ]
    agentic = any(tools for _, tools in captured)
    threads: list[dict[str, Any]] = []
    members: dict[tuple[str, ...], list[int]] = {}
    for index, tools in captured:
        if not agentic or tools:
            calls = members.get(tools)
            if calls is None:
                calls = members[tools] = []
                threads.append(
                    {
                        "thread": len(threads),
                        "kind": "agent" if agentic else "chat",
                        "calls": calls,
                    }
                )
            calls.append(index)
        else:
            threads.append({"thread": len(threads), "kind": "helper", "calls": [index]})
    return threads


def summarize_token_capture(exchanges: list[dict[str, Any]]) -> dict[str, Any]:
    """Coverage of the ``token_capture`` blocks in one ``llm_trajectory.jsonl``.

    ``complete_calls`` have prompt token ids and, for every choice, sampled
    token ids and one logprob per sampled id (a choice whose counts differ is
    counted in ``unavailable`` as ``logprobs:length_mismatch``). ``prefix`` checks the token-in/token-out property
    an RL trainer relies on, within each conversation (``threads``, see
    :func:`conversation_threads`): each complete call's prompt should start
    with the previous complete call's prompt of the same conversation
    followed by its sampled tokens. A break means the client re-rendered the
    history (a chat template, a compaction), so the conversation cannot be
    trained on as one token stream without re-alignment. ``training_grade``
    is true only when every call is complete and no pair breaks. ``path``
    names the route provider(s) of the captured calls (``vllm``, ``sglang``,
    …), or None.
    """
    unavailable: dict[str, int] = {}
    providers: set[str] = set()
    captured = complete = 0
    pairs = extends = 0
    breaks: list[int] = []
    threads = conversation_threads(exchanges)
    thread_of = {i: t["thread"] for t in threads for i in t["calls"]}
    previous: dict[int, tuple[list[int], list[int]]] = {}
    for index, exchange in enumerate(exchanges):
        capture = _dict(_dict(exchange.get("metadata")).get(TOKEN_CAPTURE_METADATA_KEY))
        if capture.get("schema_version") != TOKEN_CAPTURE_SCHEMA_VERSION:
            continue
        captured += 1
        if capture.get("provider"):
            providers.add(str(capture["provider"]))
        for field, why in _dict(capture.get("unavailable")).items():
            key = f"{field}:{_dict(why).get('reason', 'unknown')}"
            unavailable[key] = unavailable.get(key, 0) + 1
        if _length_mismatch(capture):
            key = "logprobs:length_mismatch"
            unavailable[key] = unavailable.get(key, 0) + 1
        if not _complete(capture):
            continue
        complete += 1
        prompt = _int_list(capture["prompt_token_ids"]) or []
        sampled = _int_list(_dict(capture["completions"][0]).get("token_ids")) or []
        thread = thread_of[index]
        if thread in previous:
            pairs += 1
            before = previous[thread]
            expected = before[0] + before[1]
            if prompt[: len(expected)] == expected:
                extends += 1
            else:
                breaks.append(index)
        previous[thread] = (prompt, sampled)
    calls = len(exchanges)
    return {
        "calls": calls,
        "captured_calls": captured,
        "complete_calls": complete,
        "unavailable": dict(sorted(unavailable.items())),
        "prefix": {"pairs": pairs, "extends_previous_call": extends, "breaks": breaks},
        "threads": threads,
        "training_grade": calls > 0 and complete == calls and not breaks,
        "path": ",".join(sorted(providers)) or None,
    }


def summarize_rollout_token_capture(rollout_dir: Any) -> dict[str, Any]:
    """:func:`summarize_token_capture` for a rollout, saying why when there is none."""
    import json
    from pathlib import Path

    root = Path(rollout_dir)
    path = root / "trajectory" / "llm_trajectory.jsonl"
    summary: dict[str, Any] = {"rollout": root.name}
    if not path.is_file():
        try:
            result = json.loads((root / "result.json").read_text())
        except (OSError, ValueError):
            result = {}
        usage = _dict(_dict(result).get("usage_tracking"))
        kind = usage.get("endpoint_kind")
        summary.update(
            status="no_gateway_capture",
            path=None,
            reason=(
                f"no llm_trajectory.jsonl: the agent's model calls did not go "
                f"through BenchFlow's gateway (usage_tracking.endpoint_kind="
                f"{kind!r})"
                if kind
                else "no llm_trajectory.jsonl in this rollout"
            ),
            training_grade=False,
        )
        return summary
    exchanges: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            exchanges.append(record)
    summary.update(summarize_token_capture(exchanges))
    summary["status"] = "captured" if summary["captured_calls"] else "capture_off"
    if not summary["captured_calls"]:
        summary["reason"] = (
            "calls went through the gateway without token capture; rerun with "
            "--agent-env BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1"
        )
    return summary
