"""One record per model call, from the source that saw it, naming that source.

BenchFlow's model proxy records each request it forwards
(``trajectory/llm_trajectory.jsonl``): the settings sent, the model that
answered, why the answer ended, tokens, cost and time. A run that goes around
the proxy (a Claude login, a direct provider) leaves no such record. An agent
that exports OpenTelemetry to BenchFlow's receiver (``sandbox/agent_otel.py``)
leaves ``trajectory/agent-otel.jsonl``: Claude Code logs one
``claude_code.api_request`` event per call, with tokens, its own cost estimate
and time. An agent may export none: Claude Code doesn't unless the run turns
it on. The agent's own session log still holds each response, and the
rollout copies it when the proxy saw nothing (``rollout/_session_log.py``):
Claude Code's ``~/.claude/projects`` into ``agent/claude-sessions/`` and
Codex's ``~/.codex/sessions`` into ``agent/codex-sessions/``.

:func:`model_calls` reads the first of these a trial has, in that order. Each
record carries ``source`` and leaves ``None`` what its source cannot see
(a session log holds no request's sampling). :func:`summarize` totals them,
and its ``sources`` says where each figure came from. Both shapes are
versioned (:data:`SCHEMA`, :data:`SUMMARY_SCHEMA`) and documented in
``docs/reference/model-calls.md``, with JSON Schemas in
``docs/reference/schemas/`` (``python -m benchflow.trajectories.model_calls
docs/reference/schemas`` writes them; a test keeps them current).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = "benchflow.model-call.v2"
# Records written before ``tools`` (0.10.0rc1): the same fields without it.
SCHEMA_V1 = "benchflow.model-call.v1"
SUMMARY_SCHEMA = "benchflow.model-calls.v1"
CALLS_FILE = "model-calls.jsonl"  # under the trial's trajectory/
SUMMARY_FILE = "model-calls.json"  # in the trial folder

PROXY = "proxy"
CLAUDE_CODE_OTEL = "claude-code-otel"
CLAUDE_CODE = "claude-code-session-log"
CODEX = "codex-session-log"

PROXY_LOG = Path("trajectory") / "llm_trajectory.jsonl"
# The proxy's own record of each request's attempts upstream (its request
# guard's lifecycle events: request_start, upstream_start, upstream_end, ...).
PROXY_ATTEMPTS = Path("trajectory") / "provider_requests.jsonl"
AGENT_OTEL = Path("trajectory") / "agent-otel.jsonl"
CLAUDE_SESSIONS = Path("agent") / "claude-sessions"
CODEX_SESSIONS = Path("agent") / "codex-sessions"

#: Why an answer ended, as each wire spells a cut at the output cap.
_TRUNCATED = frozenset({"length", "max_tokens", "max_output_tokens"})
#: ... and a refusal or a provider's content filter.
_REFUSED = frozenset({"refusal", "content_filter"})
#: ... and an answer, not a request for a tool (Chat Completions and Anthropic
#: name that apart). A Responses call ends ``completed`` either way.
_ANSWERED = frozenset({"stop", "end_turn", "stop_sequence"})

FIELDS = (
    "schema",
    "source",
    "index",
    "started_at",
    "duration_ms",
    "http_status",
    "purpose",
    "model_requested",
    "model_served",
    "route",
    "effort",
    "thinking",
    "temperature",
    "top_p",
    "max_output_tokens",
    "tools",
    "seed",
    "finish_reason",
    "truncated",
    "refused",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "cost_usd",
    "subagent",
)


def _record(source: str, **values: Any) -> dict[str, Any]:
    record = dict.fromkeys(FIELDS)
    record.update(values, schema=SCHEMA, source=source)
    reason = record.get("finish_reason")
    if isinstance(reason, str):
        record["truncated"] = reason in _TRUNCATED
        record["refused"] = reason in _REFUSED or values.get("refused") is True
    return record


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    # Raw requests can be tens of MB; keep only one while extracting a call.
    try:
        with path.open(errors="replace") as lines:
            for line in lines:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    yield row
    except OSError:
        return


def _dict(value: Any) -> dict[str, Any]:
    """*value* when it is a JSON object, else an empty one."""
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _prompt_tokens(usage: dict[str, Any]) -> int | None:
    """Every prompt token of an Anthropic-shaped usage: its ``input_tokens``
    leave out the cache reads and writes it lists apart. Every source's
    ``input_tokens`` here count cached tokens too (as OpenAI's do)."""
    parts = [
        _int(usage.get(key))
        for key in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    ]
    if parts[0] is None:
        return None
    return sum(part or 0 for part in parts)


def _generation(body: dict[str, Any], key: str) -> Any:
    """``generationConfig[key]`` of a Gemini request (the proxy keeps that
    much of a pass-through request's body)."""
    return _dict(body.get("generationConfig")).get(key)


def _effort(body: dict[str, Any]) -> str | None:
    effort = body.get("reasoning_effort")
    if isinstance(effort, str):
        return effort
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and isinstance(reasoning.get("effort"), str):
        return reasoning["effort"]
    config = body.get("output_config")
    if isinstance(config, dict) and isinstance(config.get("effort"), str):
        return config["effort"]
    # Gemini's thinking level (HIGH, LOW, ...); a token budget is no level.
    level = _dict(_generation(body, "thinkingConfig")).get("thinkingLevel")
    if isinstance(level, str) and level and level != "THINKING_LEVEL_UNSPECIFIED":
        return level.lower()
    return None


def _thinking(body: dict[str, Any]) -> bool | None:
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and isinstance(thinking.get("type"), str):
        return thinking["type"] != "disabled"
    kwargs = body.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and isinstance(kwargs.get("enable_thinking"), bool):
        return kwargs["enable_thinking"]
    gemini = _dict(_generation(body, "thinkingConfig"))
    budget = _int(gemini.get("thinkingBudget"))
    if budget is not None:
        return budget != 0
    if isinstance(gemini.get("thinkingLevel"), str):
        return True
    return None


def _finish(response: Any) -> str | None:
    """Why the answer ended, from an OpenAI, Anthropic or Responses body."""
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        if isinstance(reason, str):
            return reason
    for key in ("stop_reason", "finish_reason"):
        if isinstance(response.get(key), str):
            return response[key]
    details = response.get("incomplete_details")
    if isinstance(details, dict) and isinstance(details.get("reason"), str):
        return details["reason"]
    if response.get("status") == "completed":
        return "completed"
    return None


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _refusal(response: Any) -> bool | None:
    """True when the answer itself is a refusal, which can end as ``stop``
    or ``completed``: a Chat Completions message's ``refusal``, or a
    Responses ``refusal`` output part. None leaves it to the finish reason
    (Anthropic's ``stop_reason`` is ``refusal``)."""
    if not isinstance(response, dict):
        return None
    for choice in _list(response.get("choices")):
        message = choice.get("message") if isinstance(choice, dict) else None
        refusal = message.get("refusal") if isinstance(message, dict) else None
        if isinstance(refusal, str) and refusal:
            return True
    for item in _list(response.get("output")):
        if not isinstance(item, dict):
            continue
        parts = [item, *_list(item.get("content"))]
        if any(
            isinstance(part, dict) and part.get("type") == "refusal" for part in parts
        ):
            return True
    return None


def _detail(usage: dict[str, Any], key: str, *details: str) -> int | None:
    """``usage[detail][key]`` from the first of *details* that has it: the
    Chat Completions names, then the Responses ones."""
    for name in details:
        found = usage.get(name)
        if isinstance(found, dict) and found.get(key) is not None:
            return _int(found.get(key))
    return None


def _tool_names(tools: Any, prefix: str = "") -> list[str]:
    """The names of a request's tools, in order: a function's or custom
    tool's name (a Chat Completions function's under ``function``), a hosted
    tool's type (``web_search``), and a Responses namespace's tools as
    ``namespace.name``."""
    names: list[str] = []
    for tool in _list(tools):
        if not isinstance(tool, dict):
            continue
        name = tool.get("name") or _dict(tool.get("function")).get("name")
        if tool.get("type") == "namespace":
            names += _tool_names(tool.get("tools"), f"{prefix}{name}.")
        elif isinstance(name or tool.get("type"), str):
            names.append(f"{prefix}{name or tool.get('type')}")
    return names


def request_tools(body: dict[str, Any]) -> list[str]:
    """The tools a request offered the model: its ``tools``, and the tools a
    Responses request sends as ``additional_tools`` input items (Codex's code
    mode offers its tools there: ``functions.exec`` and the namespaces of
    tools a script calls)."""
    names = _tool_names(body.get("tools"))
    for item in _list(body.get("input")):
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            names += _tool_names(item.get("tools"))
    return names


def tool_surface(names: list[str]) -> dict[str, Any]:
    """``count``, ``names`` and ``sha256`` (of the names, one per line, in
    order) of a request's tools: two trials with one hash offered the model
    the same tools."""
    return {
        "count": len(names),
        "names": names,
        "sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
    }


def _is_agent_call(purpose: Any) -> bool:
    return purpose in (None, "agent")


def offers_no_tools(
    call: Mapping[str, Any], calls: Sequence[Mapping[str, Any]]
) -> bool:
    """Whether ``call`` is one of the agent's side requests to its own model,
    a session title or a summary: it offered no tools while another of the
    agent's calls did. An agent that names its tools in text offers none in
    any call, so none of its calls is set apart; a source that cannot see
    requests records no ``tools``."""
    if call.get("tools") != 0:
        return False
    return any(
        (other.get("tools") or 0) > 0
        for other in calls
        if _is_agent_call(other.get("purpose"))
    )


# Reading a file, as agents name the tool, and the argument holding its path.
_READ_TOOLS = frozenset({"read", "read_file", "readfile", "view_file"})
_WRITE_TOOLS = frozenset(
    {"write", "edit", "multiedit", "notebookedit", "write_file", "edit_file"}
)
_PATH_KEYS = ("file_path", "filePath", "path", "absolute_path", "notebook_path")
# A file the model can only take as a document or an image block.
_BINARY_SUFFIXES = (
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".docx",
    ".pptx",
    ".xlsx",
)
# Claude Code's line for a read whose content goes in a block of its own:
# "PDF file read: /root/proposal.pdf (47KB)".
_READ_PLACEHOLDER = re.compile(r"(?:[A-Za-z]+ )?file read: \S+(?: \([^)]*\))?", re.I)
# A text this short for a binary file cannot hold its content.
STUB_TEXT_CHARS = 200
# What Claude Code appends to a tool result (the environment update, the
# tokens left): the harness's words, not the file's. One the recorded head
# cuts off runs to the end, where Claude Code puts them.
_REMINDER = re.compile(r"<system-reminder>.*?(?:</system-reminder>|\Z)", re.S)


def _without_reminders(text: str, chars: int) -> tuple[str, int, bool]:
    """``(visible, visible_chars, whole)`` of a tool result's text without
    its system reminders. ``text`` may be the head of a result of ``chars``
    characters; ``whole`` says the visible text is all of it."""
    visible = _REMINDER.sub("", text).strip()
    cut_inside = text.rfind("<system-reminder>") > text.rfind("</system-reminder>")
    rest = 0 if chars <= len(text) or cut_inside else chars - len(text)
    return visible, len(visible) + rest, rest == 0


# BenchFlow's bridge patch's note for a PDF it could not read any text from.
_OMITTED_NOTE = "[PDF omitted:"
# The start of a bridged tool result the proxy records (litellm_logging's
# _bridge_delivery keeps as much).
_BRIDGE_HEAD_CHARS = 240
# A later result this long, from a call that names the file (pdftotext, a
# script), may have carried the content the stub did not.
_RECOVERY_TEXT_CHARS = 1000
_INPUT_FILES_LISTED = 50


def _result_content(content: Any) -> tuple[str, str]:
    """``(kind, text)`` of a tool result: ``document`` when it carries a
    document or file block, ``image`` an image, else ``text``; the text is its
    text parts, joined."""
    if isinstance(content, str):
        return "text", content
    texts: list[str] = []
    kinds: set[str] = set()
    for part in _list(content):
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in ("text", "input_text", "output_text"):
            texts.append(str(part.get("text") or ""))
        elif kind in ("document", "file", "input_file"):
            kinds.add("document")
        elif kind in ("image", "image_url", "input_image"):
            kinds.add("image")
    return (
        "document" if "document" in kinds else "image" if "image" in kinds else "text",
        "\n".join(texts),
    )


def _new_turn(messages: list[Any]) -> tuple[list[dict], list[tuple[str, Any, bool]]]:
    """The tool calls of a request's last assistant message, and the results
    after it: ``(id, name, input)`` and ``(id, content, is_error)``, from
    Anthropic Messages or Chat Completions requests. Each assistant message
    is the last one in exactly the request that answers it, so reading only
    this turn sees every call once, cheaply."""
    start = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if isinstance(messages[i], dict) and messages[i].get("role") == "assistant"
        ),
        None,
    )
    if start is None:
        return [], []
    uses: list[dict] = []
    assistant = messages[start]
    for block in _list(assistant.get("content")):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            uses.append(
                {
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": block.get("input"),
                }
            )
    for call in _list(assistant.get("tool_calls")):
        function = _dict(_dict(call).get("function"))
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (TypeError, ValueError):
            arguments = {}
        uses.append(
            {
                "id": _dict(call).get("id"),
                "name": function.get("name"),
                "input": arguments,
            }
        )
    results: list[tuple[str, Any, bool]] = []
    for message in messages[start + 1 :]:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool":
            results.append(
                (str(message.get("tool_call_id")), message.get("content"), False)
            )
        for block in _list(message.get("content")):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results.append(
                    (
                        str(block.get("tool_use_id")),
                        block.get("content"),
                        block.get("is_error") is True,
                    )
                )
    return uses, results


class _InputScan:
    """Whether each file the agent read reached the model, from the requests.

    A read of a file the agent had not written (a task input) is recorded by
    what its result in the next request carried: a document or image block,
    or text, with the text's length and hash. A stub is a result that cannot
    hold the file: Claude Code's ``PDF file read: <path> (47KB)`` line alone,
    or a few characters of text for a PDF or an image, which is what a route
    that drops the document leaves the model. ``recovered`` is set when a
    later call that names the file brought back a long text (the agent
    extracted it itself).
    """

    def __init__(self) -> None:
        self._pending: dict[str, tuple[str, str]] = {}  # call id -> (kind, path)
        # A retried request sends the same turn again: each call is read once.
        self._seen: set[str] = set()
        self._written: set[str] = set()
        self.files: dict[str, dict[str, Any]] = {}
        self.reads = 0

    def feed(
        self, body: dict[str, Any], index: int, route: Any, bridged: Any = None
    ) -> None:
        """Read one request; ``bridged`` is what the proxy's bridge to chat
        completions made of its tool results that held a document or an
        image (``metadata.bridge_tool_results``): the request recorded is the
        agent's, from before that translation."""
        messages = body.get("messages")
        if not isinstance(messages, list):
            return
        bridged = bridged if isinstance(bridged, dict) else {}
        uses, results = _new_turn(messages)
        for use in uses:
            if str(use.get("id")) in self._seen:
                continue
            self._seen.add(str(use.get("id")))
            name = str(use.get("name") or "").lower()
            arguments = _dict(use.get("input"))
            path = next(
                (arguments[k] for k in _PATH_KEYS if isinstance(arguments.get(k), str)),
                None,
            )
            if name in _WRITE_TOOLS and path:
                self._written.add(path)
            elif name in _READ_TOOLS and path and path not in self._written:
                self._pending[str(use.get("id"))] = ("read", path)
            else:
                named = json.dumps(arguments)
                for stubbed in self.files.values():
                    if stubbed["stub"] and stubbed["path"] in named:
                        self._pending[str(use.get("id"))] = ("other", stubbed["path"])
        for call_id, content, is_error in results:
            if call_id not in self._pending:
                continue
            kind_of_call, path = self._pending.pop(call_id)
            kind, text = _result_content(content)
            if is_error or "<tool_use_error>" in text:
                continue
            chars, digest, head, full = (
                len(text),
                hashlib.sha256(text.encode()).hexdigest(),
                text[:_BRIDGE_HEAD_CHARS],
                text,
            )
            delivered = bridged.get(call_id)
            if kind != "text" and isinstance(delivered, dict):
                # The bridge's word for what the provider got: an image part,
                # or the text it made of the block (BenchFlow's bridge patch
                # turns a PDF into its text), or nothing. An image part that
                # holds a PDF (image_url:application/pdf, LiteLLM's own bridge)
                # is not an image: the text beside it is what the model reads.
                parts = _list(delivered.get("parts"))
                kind = "image" if "image_url" in parts else "text"
                chars = _int(delivered.get("chars")) or 0
                digest = str(delivered.get("sha256") or "")
                head = full = str(delivered.get("head") or "")
            if kind_of_call == "other":
                if chars > _RECOVERY_TEXT_CHARS:
                    self.files[path]["recovered"] = True
                continue
            self.reads += 1
            visible, visible_chars, whole = _without_reminders(full, chars)
            stub = kind == "text" and (
                (whole and _READ_PLACEHOLDER.fullmatch(visible) is not None)
                or _OMITTED_NOTE in head
                or (
                    path.lower().endswith(_BINARY_SUFFIXES)
                    and visible_chars < STUB_TEXT_CHARS
                )
            )
            entry: dict[str, Any] | None = self.files.get(path)
            if entry is None:
                if len(self.files) >= _INPUT_FILES_LISTED:
                    continue
                entry = self.files[path] = {
                    "path": path,
                    "reads": 0,
                    "kind": kind,
                    "text_chars": chars,
                    "text_sha256": digest,
                    "stub": False,
                    "recovered": False,
                    "route": route if isinstance(route, str) else None,
                    "call": index,
                }
            entry["reads"] += 1
            if stub and not entry["stub"]:
                entry.update(
                    stub=True,
                    kind=kind,
                    text_chars=chars,
                    text_sha256=digest,
                    call=index,
                )

    def summary(self) -> dict[str, Any] | None:
        if not self.reads:
            return None
        files = list(self.files.values())
        return {
            "reads": self.reads,
            "stubs": sum(1 for entry in files if entry["stub"]),
            "files": files,
        }


def _proxy_scan(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The proxy's records, and what the totals read from the requests
    themselves: ``first_agent_tools``, the tools the agent's first request
    that offered any had (:func:`tool_surface`), or None; ``input_files``,
    whether each file the agent read reached the model (:class:`_InputScan`),
    or None when it read none."""
    calls: list[dict[str, Any]] = []
    first_tools: dict[str, Any] | None = None
    inputs = _InputScan()
    for row in _jsonl(path):
        request = row.get("request")
        if not isinstance(request, dict) or row.get("event"):
            continue
        meta = _dict(row.get("metadata"))
        body = _dict(request.get("body"))
        if _is_agent_call(meta.get("call_purpose")):
            if first_tools is None and (names := request_tools(body)):
                first_tools = tool_surface(names)
            inputs.feed(
                body,
                len(calls),
                meta.get("provider_model"),
                meta.get("bridge_tool_results"),
            )
        response = _dict(row.get("response"))
        answer = _dict(response.get("body"))
        usage = _dict(answer.get("usage"))
        cached = _detail(
            usage, "cached_tokens", "prompt_tokens_details", "input_tokens_details"
        )
        served = answer.get("model")
        calls.append(
            _record(
                PROXY,
                index=len(calls),
                started_at=request.get("timestamp"),
                duration_ms=_number(row.get("duration_ms")),
                http_status=_int(response.get("status_code")),
                purpose=meta.get("call_purpose"),
                model_requested=meta.get("request_model") or body.get("model"),
                model_served=served if isinstance(served, str) else None,
                route=meta.get("provider_model"),
                effort=_effort(body),
                thinking=_thinking(body),
                temperature=_number(body.get("temperature"))
                if body.get("temperature") is not None
                else _number(_generation(body, "temperature")),
                top_p=_number(body.get("top_p"))
                if body.get("top_p") is not None
                else _number(_generation(body, "topP")),
                max_output_tokens=_int(body.get("max_tokens"))
                or _int(body.get("max_completion_tokens"))
                or _int(body.get("max_output_tokens"))
                or _int(_generation(body, "maxOutputTokens")),
                tools=len(request_tools(body)),
                seed=_int(body.get("seed")),
                finish_reason=_finish(answer),
                refused=_refusal(answer),
                input_tokens=_int(usage.get("prompt_tokens"))
                if usage.get("prompt_tokens") is not None
                else _prompt_tokens(usage),
                output_tokens=_int(usage.get("completion_tokens"))
                if usage.get("completion_tokens") is not None
                else _int(usage.get("output_tokens")),
                cache_read_tokens=cached
                if cached is not None
                else _int(usage.get("cache_read_input_tokens")),
                cache_write_tokens=_int(usage.get("cache_creation_input_tokens")),
                reasoning_tokens=_detail(
                    usage,
                    "reasoning_tokens",
                    "completion_tokens_details",
                    "output_tokens_details",
                ),
                cost_usd=_number(meta.get("response_cost")),
            )
        )
    return calls, {"first_agent_tools": first_tools, "input_files": inputs.summary()}


def _proxy_calls(path: Path) -> list[dict[str, Any]]:
    """One record per request the proxy forwarded (``llm_trajectory.jsonl``).

    Each line holds ``metadata`` (the model asked for, the route's model,
    ``call_purpose``: the agent's, a reviewer's or a judge's), ``request``
    (``body`` as sent, ``timestamp``), ``response`` (``status_code``,
    ``body``: the provider's answer, ``timestamp``) and ``duration_ms``.
    """
    return _proxy_scan(path)[0]


def _logs(folder: Path) -> list[Path]:
    """The session logs below *folder*, never through a link: ``agent/`` is
    a folder the sandbox shares."""
    found: list[Path] = []
    for root, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not os.path.islink(os.path.join(root, d)))
        for name in sorted(files):
            path = Path(root) / name
            if name.endswith(".jsonl") and not path.is_symlink() and path.is_file():
                found.append(path)
    return found


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _claude_code_calls(folder: Path) -> list[dict[str, Any]]:
    """One record per API response in Claude Code's session logs.

    A response with several content blocks is logged once per block under
    one ``message.id``; the last line has the response's final usage and
    ``stop_reason``. ``duration_ms`` is the time from the line before the
    response's first line (the request it answers) to its last line.
    """
    responses: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for log in _logs(folder):
        previous: datetime | None = None
        for row in _jsonl(log):
            when = _timestamp(row.get("timestamp"))
            message = row.get("message")
            if row.get("type") != "assistant" or not isinstance(message, dict):
                previous = when or previous
                continue
            key = str(message.get("id") or row.get("requestId") or len(order))
            if key not in responses:
                order.append(key)
                responses[key] = {"start": previous, "row": row, "thinking": False}
            entry = responses[key]
            entry["row"], entry["end"] = row, when
            content = message.get("content")
            if isinstance(content, list) and any(
                isinstance(block, dict)
                and block.get("type") in ("thinking", "redacted_thinking")
                for block in content
            ):
                entry["thinking"] = True
            previous = when or previous
    calls: list[dict[str, Any]] = []
    for key in order:
        entry = responses[key]
        row = entry["row"]
        message = row["message"]
        usage = _dict(message.get("usage"))
        start, end = entry.get("start"), entry.get("end")
        model = message.get("model")
        calls.append(
            _record(
                CLAUDE_CODE,
                index=len(calls),
                started_at=start.isoformat() if start else row.get("timestamp"),
                duration_ms=(end - start).total_seconds() * 1000
                if start and end and end >= start
                else None,
                model_served=model if isinstance(model, str) else None,
                thinking=entry["thinking"] or None,
                finish_reason=message.get("stop_reason")
                if isinstance(message.get("stop_reason"), str)
                else None,
                input_tokens=_prompt_tokens(usage),
                output_tokens=_int(usage.get("output_tokens")),
                cache_read_tokens=_int(usage.get("cache_read_input_tokens")),
                cache_write_tokens=_int(usage.get("cache_creation_input_tokens")),
                cost_usd=_number(row.get("costUSD")),
                purpose="agent",
                subagent=bool(row.get("isSidechain")) or None,
            )
        )
    return calls


def _otlp_value(value: Any) -> Any:
    """An OTLP/JSON ``AnyValue``'s plain value (int64s arrive as strings)."""
    if not isinstance(value, dict):
        return None
    for key in ("stringValue", "boolValue", "doubleValue"):
        if key in value:
            return value[key]
    if "intValue" in value:
        try:
            return int(value["intValue"])
        except (TypeError, ValueError):
            return None
    return None


def _otel_number(value: Any) -> float | None:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    return _number(value)


def _otel_int(value: Any) -> int | None:
    number = _otel_number(value)
    return int(number) if number is not None and number == int(number) else None


def _claude_code_otel_calls(path: Path) -> list[dict[str, Any]]:
    """One record per ``claude_code.api_request`` event Claude Code exported.

    Its ``input_tokens`` excludes cached tokens (the Anthropic usage), so the
    record's adds both cache counts; ``cost_usd`` is Claude Code's own
    estimate; ``event.timestamp`` is when the response finished, so the call
    started ``duration_ms`` before it.
    """
    events: list[dict[str, Any]] = []
    for line in _jsonl(path):
        body = line.get("body")
        for resource in (
            body.get("resourceLogs") or [] if isinstance(body, dict) else []
        ):
            for scope in resource.get("scopeLogs") or []:
                for log in scope.get("logRecords") or []:
                    attrs = {
                        a.get("key"): _otlp_value(a.get("value"))
                        for a in log.get("attributes") or []
                        if isinstance(a, dict)
                    }
                    name = _otlp_value(log.get("body")) or attrs.get("event.name")
                    if name in ("claude_code.api_request", "api_request"):
                        events.append(attrs)
    events.sort(
        key=lambda a: (
            _otel_int(a.get("event.sequence")) or 0,
            str(a.get("event.timestamp")),
        )
    )
    calls: list[dict[str, Any]] = []
    for attrs in events:
        end = _timestamp(attrs.get("event.timestamp"))
        duration = _otel_number(attrs.get("duration_ms"))
        start = (
            datetime.fromtimestamp(end.timestamp() - duration / 1000, tz=end.tzinfo)
            if end is not None and duration is not None
            else None
        )
        fresh = _otel_int(attrs.get("input_tokens"))
        read = _otel_int(attrs.get("cache_read_tokens"))
        write = _otel_int(attrs.get("cache_creation_tokens"))
        model = attrs.get("model")
        effort = attrs.get("effort")
        calls.append(
            _record(
                CLAUDE_CODE_OTEL,
                index=len(calls),
                started_at=start.isoformat() if start else attrs.get("event.timestamp"),
                duration_ms=duration,
                model_served=model if isinstance(model, str) else None,
                effort=effort if isinstance(effort, str) and effort else None,
                input_tokens=None
                if fresh is None
                else fresh + (read or 0) + (write or 0),
                output_tokens=_otel_int(attrs.get("output_tokens")),
                cache_read_tokens=read,
                cache_write_tokens=write,
                cost_usd=_otel_number(attrs.get("cost_usd")),
                purpose="agent",
            )
        )
    return calls


def _codex_calls(folder: Path) -> list[dict[str, Any]]:
    """One record per model response in Codex's session rollouts.

    Newer Codex writes a ``token_usage_record`` per response (its
    ``response_id`` and ``usage``); older Codex (0.159) only a ``token_count``
    event after each response (``info.last_token_usage``). A log with records
    is read from them alone, so a response is never counted twice. The model
    and effort are those of the ``turn_context`` before it. Codex logs no
    finish reason or request sampling, so those stay ``None``. Codex's
    ``input_tokens`` already count cached ones.

    A sub-agent's thread (a review, a compaction, an agent the model spawned)
    is a rollout of its own whose ``session_meta`` names its source
    ``{"subagent": ...}``, and it may run on another model or effort (Codex's
    ``default_subagent_model``): its calls are ``subagent``. The title thread
    codex-acp starts is ephemeral, and Codex writes no rollout for it.
    """
    calls: list[dict[str, Any]] = []
    for log in _logs(folder):
        rows = list(_jsonl(log))
        records = any(row.get("type") == "token_usage_record" for row in rows)
        subagent = any(
            row.get("type") == "session_meta"
            and "subagent" in _dict(_dict(row.get("payload")).get("source"))
            for row in rows
        )
        model = effort = None
        previous: datetime | None = None
        for row in rows:
            payload = _dict(row.get("payload"))
            when = _timestamp(row.get("timestamp"))
            if row.get("type") == "turn_context":
                if isinstance(payload.get("model"), str):
                    model = payload["model"]
                if isinstance(payload.get("effort"), str):
                    effort = payload["effort"]
            usage = None
            if records and row.get("type") == "token_usage_record":
                usage = payload.get("usage")
            elif (
                not records
                and row.get("type") == "event_msg"
                and payload.get("type") == "token_count"
                and isinstance(payload.get("info"), dict)
            ):
                usage = payload["info"].get("last_token_usage")
            if isinstance(usage, dict):
                calls.append(
                    _record(
                        CODEX,
                        index=len(calls),
                        started_at=previous.isoformat()
                        if previous
                        else row.get("timestamp"),
                        duration_ms=(when - previous).total_seconds() * 1000
                        if when and previous and when >= previous
                        else None,
                        purpose="agent",
                        model_served=model,
                        effort=effort,
                        input_tokens=_int(usage.get("input_tokens")),
                        output_tokens=_int(usage.get("output_tokens")),
                        cache_read_tokens=_int(usage.get("cached_input_tokens")),
                        cache_write_tokens=_int(usage.get("cache_write_input_tokens")),
                        reasoning_tokens=_int(usage.get("reasoning_output_tokens")),
                        subagent=subagent or None,
                    )
                )
            previous = when or previous
    return calls


def model_calls(trial: Path) -> list[dict[str, Any]]:
    """The trial's model calls from the first source it has (see the module)."""
    proxy = _proxy_calls(trial / PROXY_LOG)
    if proxy:
        return proxy
    otel = _claude_code_otel_calls(trial / AGENT_OTEL)
    if otel:
        return otel
    for folder, reader in (
        (CLAUDE_SESSIONS, _claude_code_calls),
        (CODEX_SESSIONS, _codex_calls),
    ):
        if (trial / folder).is_dir():
            calls = reader(trial / folder)
            if calls:
                return calls
    return []


_UPSTREAM_FIGURES = ("upstream_attempts", "upstream_retries", "upstream_errors")


def upstream_attempts(path: Path) -> dict[str, int] | None:
    """What the proxy's request records say of its attempts upstream
    (``provider_requests.jsonl``): every attempt it sent the provider
    (``upstream_attempts``), the attempts of a request that came after one of
    its attempts failed (``upstream_retries``; a request can make more than
    one call upstream on its own, such as a lookup before the model call),
    and the attempts that failed or got no status (``upstream_errors``). None
    without that file."""
    if not path.is_file():
        return None
    attempts = retries = errors = 0
    failed: set[str] = set()  # requests whose last attempt failed
    for row in _jsonl(path):
        request = str(row.get("request_id"))
        if row.get("event") == "upstream_start":
            attempts += 1
            if request in failed:
                retries += 1
                failed.discard(request)
        elif row.get("event") == "upstream_end":
            status = _int(row.get("status"))
            if status is None or status >= 400:
                errors += 1
                failed.add(request)
            else:
                failed.discard(request)
    return {
        "upstream_attempts": attempts,
        "upstream_retries": retries,
        "upstream_errors": errors,
    }


def summarize(
    calls: list[dict[str, Any]],
    *,
    first_agent_tools: dict[str, Any] | None = None,
    input_files: dict[str, Any] | None = None,
    upstream: dict[str, int] | None = None,
) -> dict[str, Any]:
    """The trial's totals, with the source of each (``sources``).

    ``first_agent_tools`` is the proxy's view of the tools the agent's first
    request offered (:func:`tool_surface`); None for a source that cannot
    see requests. ``input_files`` is the proxy's view of the files the agent
    read (:class:`_InputScan`). ``upstream`` is :func:`upstream_attempts`.
    """
    sources = sorted({c["source"] for c in calls})
    known = lambda key: [c[key] for c in calls if c.get(key) is not None]  # noqa: E731
    seen = {key: bool(known(key)) for key in FIELDS}
    truncated = known("truncated")
    refused = known("refused")
    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "calls": len(calls),
        "source": sources[0] if len(sources) == 1 else (sources or None),
        "models_served": dict(Counter(known("model_served"))),
        "models_requested": dict(Counter(known("model_requested"))),
        "effort": dict(Counter(known("effort"))),
        "thinking": dict(Counter(str(v).lower() for v in known("thinking"))),
        "max_output_tokens": dict(Counter(str(v) for v in known("max_output_tokens"))),
        "finish_reasons": dict(Counter(known("finish_reason"))),
        "truncated_calls": sum(truncated) if truncated else None,
        "truncation_rate": sum(truncated) / len(truncated) if truncated else None,
        "refused_calls": sum(refused) if refused else None,
        "http_errors": sum(1 for status in known("http_status") if status >= 400)
        if seen["http_status"]
        else None,
        "http_status_counts": dict(Counter(str(v) for v in known("http_status"))),
        "purposes": dict(Counter(known("purpose"))),
        "tokens": {
            key: sum(known(key)) if seen[key] else None
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "reasoning_tokens",
            )
        },
        "cost_usd": sum(known("cost_usd")) if seen["cost_usd"] else None,
        "model_time_ms": sum(known("duration_ms")) if seen["duration_ms"] else None,
        "first_call_at": min(known("started_at"), default=None),
    }
    agent_calls = [
        call
        for call in calls
        if _is_agent_call(call.get("purpose")) and call.get("subagent") is not True
    ]
    # The agent's turns: a session title or a summary it asked its own model
    # for carries none of the run's settings and is no ending of its own.
    turns = [call for call in agent_calls if not offers_no_tools(call, calls)]
    summary["first_agent_tools"] = first_agent_tools
    summary["input_files"] = input_files
    # A failed call of the agent's followed by another: the agent tried again
    # (Claude Code and Codex retry a 429 or a 5xx themselves, each try a call).
    statuses = [call.get("http_status") for call in agent_calls]
    summary["agent_retries"] = (
        sum(1 for status in statuses[:-1] if isinstance(status, int) and status >= 400)
        if seen["http_status"]
        else None
    )
    summary.update(upstream or dict.fromkeys(_UPSTREAM_FIGURES))
    # The effort each of the agent's own calls carried: side calls (a session
    # title, a judge) and a session log's sub-agents are not the run's effort.
    summary["agent_effort"] = dict(
        Counter(c["effort"] for c in turns if c.get("effort") is not None)
    )
    last = turns[-1] if turns else None
    summary["last_agent_call"] = (
        {
            key: last.get(key)
            for key in ("index", "http_status", "refused", "finish_reason")
        }
        if last is not None
        else None
    )
    source = summary["source"] if isinstance(summary["source"], str) else "mixed"
    # The proxy sees every request whole, so a setting it lacks was not sent;
    # an agent's session log holds no request settings at all.
    absent_setting = "none sent" if source == PROXY else "not recorded by " + source
    summary["sources"] = (
        {
            figure: source if available else absent
            for figure, available, absent in (
                ("models_served", seen["model_served"], "not recorded by " + source),
                ("effort", seen["effort"], absent_setting),
                ("sampling", seen["temperature"] or seen["top_p"], absent_setting),
                ("max_output_tokens", seen["max_output_tokens"], absent_setting),
                ("finish_reasons", seen["finish_reason"], "not recorded by " + source),
                (
                    "tokens",
                    seen["input_tokens"] or seen["output_tokens"],
                    "not recorded by " + source,
                ),
                ("cost_usd", seen["cost_usd"], "not priced by " + source),
                ("model_time_ms", seen["duration_ms"], "not recorded by " + source),
            )
        }
        if calls
        else {}
    )
    return summary


def agent_outcome(
    summary: dict[str, Any] | None,
    *,
    stop_reason: str | None = None,
    agent_error: bool = False,
) -> dict[str, Any]:
    """How the agent's model ended, beside the verifier's own score.

    The counts are every call's (*summary*, ``model-calls.json``). The status
    comes from how the agent's last prompt ended (*stop_reason*, its ACP stop
    reason; None when it did not end) and from its last call
    (``last_agent_call``), the first that applies:

    - ``refusal``: the prompt ended ``refusal``;
    - ``provider_error``: the last call failed with HTTP 429 or 5xx;
    - ``refusal``: the last call was a refusal;
    - ``completed``: the prompt ended ``end_turn``; or, with no stop reason
      and no agent error or timeout (*agent_error*), the last call answered
      200 with a finish reason that ends an answer, not a tool request;
    - ``unknown``: anything else, a timeout among them.
    """
    summary = summary or {}
    last = summary.get("last_agent_call") or {}
    status = last.get("http_status")
    terminal_error = isinstance(status, int) and (status == 429 or 500 <= status < 600)
    if stop_reason == "refusal":
        outcome = "refusal"
    elif terminal_error:
        outcome = "provider_error"
    elif last.get("refused") is True:
        outcome = "refusal"
    elif stop_reason == "end_turn" or (
        stop_reason is None
        and not agent_error
        and status == 200
        and last.get("finish_reason") in _ANSWERED
    ):
        outcome = "completed"
    else:
        outcome = "unknown"
    return {
        "status": outcome,
        "stop_reason": stop_reason,
        "http_errors": summary.get("http_errors"),
        "http_status_counts": summary.get("http_status_counts"),
        "agent_retries": summary.get("agent_retries"),
        "upstream_retries": summary.get("upstream_retries"),
        "refused_calls": summary.get("refused_calls"),
        "terminal_provider_error": terminal_error,
        "last_http_status": status,
    }


def refusal_ending(
    outcome: Mapping[str, Any],
    summary: Mapping[str, Any] | None,
    calls: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """result.json's ``model_refusal``: the evidence for a trial that ended
    on a model refusal or a content filter (``agent_outcome`` status
    ``refusal``), None for any other ending.

    ``by`` says what showed it, as ``agent_outcome`` reads it: ``stop_reason``
    (the agent's last prompt ended ``refusal``) or ``last_call`` (the agent's
    own last call was refused). ``call`` is the refused call
    (``trajectory/model-calls.jsonl``): the agent's last call when it was
    refused, else the last refused call of the trial, else None (only the
    agent's word says so). ``refused_calls`` counts every refused call,
    subagents' among them.
    """
    if outcome.get("status") != "refusal":
        return None
    last = (summary or {}).get("last_agent_call") or {}
    index = last.get("index") if last.get("refused") is True else None
    refused = [c for c in calls if c.get("refused") is True]
    call = next((c for c in calls if c.get("index") == index), None)
    if call is None and refused:
        call = refused[-1]
    return {
        "by": "stop_reason" if outcome.get("stop_reason") == "refusal" else "last_call",
        "stop_reason": outcome.get("stop_reason"),
        "call": {
            key: call.get(key)
            for key in (
                "index",
                "source",
                "finish_reason",
                "http_status",
                "model_served",
                "started_at",
                "subagent",
            )
        }
        if call is not None
        else None,
        "refused_calls": len(refused) if calls else None,
    }


def subagent_usage(trial: Path) -> dict[str, Any] | None:
    """What the agent's subagents used, from its own session log copied
    into the trial (Claude Code marks a subagent's responses ``isSidechain``,
    Codex a sub-agent thread's ``session_meta``): their calls (``turns``) and
    tokens (input, cached included, plus output), and the log's whole calls
    and tokens beside them (``of_turns``, ``of_tokens``). None without such a
    log: the proxy sees every call, but not whose it is."""
    for folder, reader, source in (
        (CLAUDE_SESSIONS, _claude_code_calls, CLAUDE_CODE),
        (CODEX_SESSIONS, _codex_calls, CODEX),
    ):
        if not (trial / folder).is_dir():
            continue
        calls = reader(trial / folder)
        if not calls:
            continue

        def tokens(rows: list[dict[str, Any]]) -> int:
            return sum(
                (row.get("input_tokens") or 0) + (row.get("output_tokens") or 0)
                for row in rows
            )

        theirs = [call for call in calls if call.get("subagent") is True]
        return {
            "turns": len(theirs),
            "tokens": tokens(theirs),
            "of_turns": len(calls),
            "of_tokens": tokens(calls),
            "source": source,
        }
    return None


def effort_outcome(
    summary: dict[str, Any] | None,
    *,
    requested: str | None,
    applied: str | None,
    route: str | None = None,
) -> dict[str, Any]:
    """The reasoning effort the run asked for, beside the effort that ran.

    ``applied`` is the agent session's word (the ``model[effort]`` id
    codex-acp took, the value an adapter answered an effort option with, the
    effort a native CLI was given); ``route`` an effort the model proxy's
    route sets on each request itself (Bedrock's ceiling for some Claude
    models). ``sent`` counts the agent's model calls by the effort each
    carried (*summary*, ``model-calls.json``): the proxy sees every request,
    so ``{}`` there means none carried one; an agent's own log can only say
    what it recorded, so None when it recorded no effort. ``matches`` is
    False when any of them differs from ``requested``, None when there is
    nothing to compare (no effort asked for, nothing recorded).
    """
    summary = summary or {}
    source = summary.get("source") if summary.get("calls") else None
    efforts = summary.get("agent_effort")
    efforts = dict(efforts) if isinstance(efforts, dict) else {}
    sent: dict[str, int] | None = efforts if source == PROXY or efforts else None
    checks: list[bool] = []
    if requested:
        checks += [value == requested for value in (applied, route) if value]
        if sent is not None:
            checks.append(set(sent) == {requested})
    return {
        "requested": requested,
        "applied": applied,
        "route": route,
        "sent": sent,
        "sent_source": source if sent is not None else None,
        "matches": all(checks) if checks else None,
    }


def effort_warning(effort: dict[str, Any]) -> str | None:
    """The trial warning for an effort that is not the one asked for."""
    if effort.get("matches") is not False:
        return None
    requested = effort.get("requested")
    found = [
        f"{who} {effort[key]}"
        for key, who in (
            ("applied", "the agent applied"),
            ("route", "the model proxy's route sets"),
        )
        if effort.get(key) and effort[key] != requested
    ]
    sent = effort.get("sent")
    if sent == {}:
        found.append("none of the agent's model calls carried an effort")
    elif sent and set(sent) != {requested}:
        found.append(
            "the agent's model calls carried "
            + ", ".join(f"{value} ({n} calls)" for value, n in sorted(sent.items()))
        )
    return f"reasoning effort {requested} was asked for, but " + "; ".join(found)


def summarize_proxy_log(path: Path) -> dict[str, Any] | None:
    """The totals of the calls a proxy log holds so far (the live
    ``llm_trajectory.jsonl`` while the trial runs), as ``model-calls.json``
    would give them; None when it holds none."""
    calls, seen = _proxy_scan(path)
    return summarize(calls, **seen) if calls else None


def write_model_calls(trial: Path) -> dict[str, Any] | None:
    """Write ``trajectory/model-calls.jsonl`` and ``model-calls.json``; None
    when no source recorded a call."""
    from benchflow._utils.disk_space import write_text_atomic

    calls, seen = _proxy_scan(trial / PROXY_LOG)
    if not calls:
        calls, seen = model_calls(trial), {}
    if not calls:
        return None
    # Each file is written beside its place and renamed over it, so a link
    # left at either name is replaced, never followed.
    write_text_atomic(
        trial / "trajectory" / CALLS_FILE,
        "".join(json.dumps(call, separators=(",", ":")) + "\n" for call in calls),
    )
    summary = summarize(
        calls, **seen, upstream=upstream_attempts(trial / PROXY_ATTEMPTS)
    )
    write_text_atomic(trial / SUMMARY_FILE, json.dumps(summary, indent=2) + "\n")
    return summary


_STR = {"type": ["string", "null"]}
_INT = {"type": ["integer", "null"]}
_NUM = {"type": ["number", "null"]}
_BOOL = {"type": ["boolean", "null"]}
_FIELD_TYPES: dict[str, dict[str, Any]] = {
    "schema": {"const": SCHEMA},
    "source": {"enum": [PROXY, CLAUDE_CODE_OTEL, CLAUDE_CODE, CODEX]},
    "index": {"type": "integer", "minimum": 0},
    "started_at": _STR,
    "duration_ms": _NUM,
    "http_status": _INT,
    "purpose": _STR,
    "model_requested": _STR,
    "model_served": _STR,
    "route": _STR,
    "effort": _STR,
    "thinking": _BOOL,
    "temperature": _NUM,
    "top_p": _NUM,
    "max_output_tokens": _INT,
    "tools": _INT,
    "seed": _INT,
    "finish_reason": _STR,
    "truncated": _BOOL,
    "refused": _BOOL,
    "input_tokens": _INT,
    "output_tokens": _INT,
    "cache_read_tokens": _INT,
    "cache_write_tokens": _INT,
    "reasoning_tokens": _INT,
    "cost_usd": _NUM,
    "subagent": _BOOL,
}
_COUNTS = {"type": "object", "additionalProperties": {"type": "integer"}}

# 0.10.0rc1 wrote v1 records: neither tools nor seed.
_V1_FIELDS = tuple(name for name in FIELDS if name not in ("tools", "seed"))
SCHEMAS: dict[str, dict[str, Any]] = {
    "benchflow-model-call.v1.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://benchflow.ai/schemas/benchflow-model-call.v1.schema.json",
        "title": "One model call (trajectory/model-calls.jsonl)",
        "type": "object",
        "required": list(_V1_FIELDS),
        "additionalProperties": False,
        "properties": {
            **{name: _FIELD_TYPES[name] for name in _V1_FIELDS},
            "schema": {"const": SCHEMA_V1},
        },
    },
    "benchflow-model-call.v2.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://benchflow.ai/schemas/benchflow-model-call.v2.schema.json",
        "title": "One model call (trajectory/model-calls.jsonl)",
        "type": "object",
        "required": list(FIELDS),
        "additionalProperties": False,
        "properties": {name: _FIELD_TYPES[name] for name in FIELDS},
    },
    "benchflow-model-calls.v1.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://benchflow.ai/schemas/benchflow-model-calls.v1.schema.json",
        "title": "A trial's model calls, totalled (model-calls.json)",
        "type": "object",
        "required": ["schema", "calls", "source", "sources", "tokens"],
        "properties": {
            "schema": {"const": SUMMARY_SCHEMA},
            "calls": {"type": "integer", "minimum": 0},
            "source": {"type": ["string", "array", "null"]},
            "models_served": _COUNTS,
            "models_requested": _COUNTS,
            "effort": _COUNTS,
            "thinking": _COUNTS,
            "max_output_tokens": _COUNTS,
            "finish_reasons": _COUNTS,
            "truncated_calls": _INT,
            "truncation_rate": _NUM,
            "refused_calls": _INT,
            "http_errors": _INT,
            "http_status_counts": _COUNTS,
            "agent_retries": _INT,
            "upstream_attempts": _INT,
            "upstream_retries": _INT,
            "upstream_errors": _INT,
            "purposes": _COUNTS,
            "agent_effort": _COUNTS,
            "input_files": {
                "type": ["object", "null"],
                "required": ["reads", "stubs", "files"],
                "properties": {
                    "reads": {"type": "integer", "minimum": 0},
                    "stubs": {"type": "integer", "minimum": 0},
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "path",
                                "reads",
                                "kind",
                                "text_chars",
                                "text_sha256",
                                "stub",
                                "recovered",
                            ],
                            "properties": {
                                "path": {"type": "string"},
                                "reads": {"type": "integer", "minimum": 0},
                                "kind": {"enum": ["text", "document", "image"]},
                                "text_chars": {"type": "integer", "minimum": 0},
                                "text_sha256": {"type": "string"},
                                "stub": {"type": "boolean"},
                                "recovered": {"type": "boolean"},
                                "route": _STR,
                                "call": {"type": "integer", "minimum": 0},
                            },
                        },
                    },
                },
            },
            "first_agent_tools": {
                "type": ["object", "null"],
                "required": ["count", "names", "sha256"],
                "additionalProperties": False,
                "properties": {
                    "count": {"type": "integer", "minimum": 0},
                    "names": {"type": "array", "items": {"type": "string"}},
                    "sha256": {"type": "string"},
                },
            },
            "tokens": {"type": "object", "additionalProperties": _INT},
            "cost_usd": _NUM,
            "model_time_ms": _NUM,
            "first_call_at": _STR,
            "last_agent_call": {
                "type": ["object", "null"],
                "required": ["http_status", "refused", "finish_reason"],
                "additionalProperties": False,
                "properties": {
                    # The call's index in model-calls.jsonl (0.10.0rc2 on).
                    "index": _INT,
                    "http_status": _INT,
                    "refused": _BOOL,
                    "finish_reason": _STR,
                },
            },
            "sources": {"type": "object", "additionalProperties": {"type": "string"}},
        },
    },
}


def write_schemas(directory: str | Path) -> list[Path]:
    """Write the record and totals schemas into *directory*."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, schema in SCHEMAS.items():
        path = directory / name
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        paths.append(path)
    return paths


if __name__ == "__main__":
    import sys

    for written in write_schemas(sys.argv[1] if len(sys.argv) > 1 else "."):
        print(written)
