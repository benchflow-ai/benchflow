"""Tokens and a cost for every trial, whatever the auth (0.9.3).

A run around BenchFlow's model proxy (a Claude or ChatGPT login, the native
harness) learns its tokens from the agent's end-of-turn report: ACP's
``PromptResponse.usage``, Claude Code's final ``result`` event, Codex's
``turn.completed``. A turn that never ends -- a timeout, a cancel, a lost
sandbox -- sends none, and the trial recorded no tokens and no cost. Here,
at teardown:

- :func:`salvage_from_stream_log` counts a native Claude Code trial from its
  own output stream, which BenchFlow keeps on the host
  (``agent/claude-code.jsonl``), so a lost sandbox loses nothing: a turn
  that reported counts what it reported, a turn that did not counts its
  messages (input and cache from ``message_start``, output from the last
  ``message_delta``, once per message id).
- :func:`salvage_from_session_log` counts a Claude Code trial over ACP from
  Claude Code's session log, copied from the sandbox into
  ``agent/claude-sessions/`` (the copy needs the sandbox).
- Codex is counted from its session log, copied into
  ``agent/codex-sessions/``, by
  :func:`benchflow.rollout._session_log.count_codex_tokens`.
- :func:`apply_list_price` prices every trial that has tokens and no cost,
  at list prices (:mod:`benchflow._utils.pricing`), and labels every cost:
  ``cost_source`` is ``provider`` for a provider's billed figure, else
  ``list_price_estimate``, with the price table's date.

Each step never fails the rollout.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from benchflow._utils.pricing import (
    SNAPSHOT_TABLE,
    TokenSlice,
    anthropic_slice,
    list_rates,
    price_slices,
)
from benchflow.usage_tracking import (
    COST_SOURCE_LIST_PRICE,
    COST_SOURCE_PROVIDER,
    USAGE_END_OF_TURN,
    USAGE_SESSION_LOG,
    USAGE_SOURCE_AGENT_NATIVE_ACP,
    USAGE_SOURCE_PROVIDER_RESPONSE,
    USAGE_STREAM_LOG,
    is_token_usage_available,
)

logger = logging.getLogger(__name__)

#: ``price_source`` of a cost this module set (tokens at list prices).
LIST_PRICE_SOURCE = "list_price"
#: ``price_source`` of the provider's own billed figure (an OpenRouter
#: response's ``usage.cost``, summed by the gateway).
PROVIDER_PRICE_SOURCE = "provider"
STREAM_LOG = "claude-code.jsonl"  # under the trial's agent/ folder
STREAM_TOKENS_SOURCE = "claude-code-stream"
SESSION_TOKENS_SOURCE = "claude-code-session-log"
CLAUDE_SESSIONS = "agent/claude-sessions"
CODEX_SESSIONS = Path("agent") / "codex-sessions"
TOKEN_FIELDS = (
    "n_input_tokens",
    "n_output_tokens",
    "n_cache_read_tokens",
    "n_cache_creation_tokens",
    "n_cache_write_5m_tokens",
    "n_cache_write_1h_tokens",
    "total_tokens",
)
# The kinds a session log can be short of, and their names in a reason.
_KINDS = {
    "n_input_tokens": "input",
    "n_output_tokens": "output",
    "n_cache_read_tokens": "cache-read",
    "n_cache_creation_tokens": "cache-write",
}
_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _merge_usage(held: dict[str, Any], usage: Any) -> None:
    """Keep the largest of each count a message's events report.

    ``message_start`` carries the prompt and an output of 1, each
    ``message_delta`` the output so far, and an ``assistant`` line the usage
    as of that content block: every one is a running count of the same
    response, so the largest is the latest.
    """
    if not isinstance(usage, dict):
        return
    for key in _USAGE_KEYS:
        if isinstance(usage.get(key), int):
            held[key] = max(_int(held.get(key)), _int(usage.get(key)))
    written = usage.get("cache_creation")
    if isinstance(written, dict):
        held_written = held.setdefault("cache_creation", {})
        for key in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"):
            held_written[key] = max(_int(held_written.get(key)), _int(written.get(key)))
    server = usage.get("server_tool_use")
    if isinstance(server, dict):
        held_server = held.setdefault("server_tool_use", {})
        held_server["web_search_requests"] = max(
            _int(held_server.get("web_search_requests")),
            _int(server.get("web_search_requests")),
        )


def _totals(usages: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """result.json's token fields for Anthropic-shaped usages: input leaves
    the cache out, so the total is the sum of the four. The cache writes'
    5-minute and 1-hour shares are summed when every response that wrote to
    the cache reported them (``cache_creation``), else None."""
    fresh = output = read = write = five = hour = 0
    known, seen = True, False
    for usage in usages:
        fresh += _int(usage.get("input_tokens"))
        output += _int(usage.get("output_tokens"))
        read += _int(usage.get("cache_read_input_tokens"))
        written = _int(usage.get("cache_creation_input_tokens"))
        write += written
        lifetimes = usage.get("cache_creation")
        if isinstance(lifetimes, dict) and (
            "ephemeral_5m_input_tokens" in lifetimes
            or "ephemeral_1h_input_tokens" in lifetimes
        ):
            seen = True
            five += _int(lifetimes.get("ephemeral_5m_input_tokens"))
            hour += _int(lifetimes.get("ephemeral_1h_input_tokens"))
        elif written:
            known = False
    split = known and seen
    return {
        "n_input_tokens": fresh,
        "n_output_tokens": output,
        "n_cache_read_tokens": read,
        "n_cache_creation_tokens": write,
        "n_cache_write_5m_tokens": five if split else None,
        "n_cache_write_1h_tokens": hour if split else None,
        "total_tokens": fresh + output + read + write,
    }


def claude_stream_usage(path: Path) -> dict[str, Any] | None:
    """The tokens of the Claude Code turns in a native trial's stream log.

    The log holds each turn's events after a ``benchflow_native_turn``
    marker. A turn with a ``result`` event counts that event's ``usage``, as
    the turn's own report did; a turn without one (killed, cancelled, cut
    off) counts its messages: per message id, the largest of each count its
    ``message_start``, ``message_delta`` and ``assistant`` events carry
    (:func:`_merge_usage`). Returns the token fields, ``turns``,
    ``turns_without_report``, ``messages`` and ``slices`` (what to price);
    None when the log is missing or counts no token.
    """
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    turns: list[dict[str, Any]] = []
    seen: set[str] = set()
    current: dict[Any, str] = {}

    def turn() -> dict[str, Any]:
        if not turns:
            turns.append({"result": None, "messages": {}, "models": []})
        return turns[-1]

    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if "benchflow_native_turn" in event:
            turns.append({"result": None, "messages": {}, "models": []})
            current.clear()
            continue
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            model = event.get("model")
            if isinstance(model, str) and model:
                turn()["models"].append(model)
            continue
        if kind == "result":
            turn()["result"] = event
            continue
        parent = event.get("parent_tool_use_id")
        message: dict[str, Any] | None = None
        usage: Any = None
        if kind == "stream_event":
            raw = event.get("event")
            raw = raw if isinstance(raw, dict) else {}
            if raw.get("type") == "message_start":
                message = (
                    raw.get("message") if isinstance(raw.get("message"), dict) else None
                )
                if message and isinstance(message.get("id"), str):
                    current[parent] = message["id"]
                usage = (message or {}).get("usage")
            elif raw.get("type") == "message_delta" and parent in current:
                message = {"id": current[parent]}
                usage = raw.get("usage")
        elif kind == "assistant" and isinstance(event.get("message"), dict):
            message = event["message"]
            usage = message.get("usage")
        if not message or not isinstance(message.get("id"), str):
            continue
        model = message.get("model")
        if model == "<synthetic>":
            continue
        message_id = message["id"]
        if message_id in seen and message_id not in turn()["messages"]:
            continue  # counted in an earlier turn
        seen.add(message_id)
        held = turn()["messages"].setdefault(message_id, {"model": None, "usage": {}})
        if isinstance(model, str) and model:
            held["model"] = model
        _merge_usage(held["usage"], usage)

    counted: list[dict[str, Any]] = []
    slices: list[TokenSlice] = []
    without = 0
    for item in turns:
        result = item["result"]
        models = [m["model"] for m in item["messages"].values() if m["model"]]
        model = (models or item["models"] or [""])[0]
        usage = result.get("usage") if isinstance(result, dict) else None
        if isinstance(usage, dict):
            counted.append(usage)
            piece = anthropic_slice(model, usage)
            piece.prompt = 0  # a turn's total, not one request
            slices.append(piece)
            continue
        if item["messages"]:
            without += 1
        for held in item["messages"].values():
            counted.append(held["usage"])
            slices.append(anthropic_slice(held["model"] or model, held["usage"]))
    totals = _totals(counted)
    if not totals["total_tokens"]:
        return None
    return {
        **totals,
        "turns": len(turns),
        "turns_without_report": without,
        "messages": len(seen),
        "slices": slices,
    }


#: ``usage_details.token_count.counted_from``: Claude Code's own per-model
#: totals (its ``cost-state`` lines), or the logged responses.
COUNTED_FROM_TOTALS = "claude-code-totals"
COUNTED_FROM_RESPONSES = "responses"
#: What a Claude Code session-log count covers (benchflow-ai/benchflow#1241):
#: the end-of-turn report counts the main session only.
SCOPE_ALL = "main session, subagents and Claude Code's side calls"
SCOPE_LOGGED = "main session and subagents"
SCOPE_MAIN = "main session only"


def _scope(usages: list[dict[str, Any]]) -> dict[str, Any]:
    totals = _totals(usages)
    return {
        "responses": len(usages),
        **{field: totals[field] for field in TOKEN_FIELDS},
    }


def _covered_slices(
    usages: list[tuple[str, dict[str, Any]]], counts: dict[str, dict[str, int]]
) -> list[TokenSlice]:
    """What to price for a count taken from Claude Code's totals: each
    logged response as it was logged (its prompt picks a long-prompt tier,
    its 1-hour cache writes their rate), and per model what the totals count
    beyond its logged responses, at the base tier."""
    slices = [anthropic_slice(model, usage) for model, usage in usages]
    for model, c in counts.items():
        mine = [piece for piece in slices if piece.model == model]
        rest = TokenSlice(
            model=model,
            input=max(c["input"] - sum(p.input for p in mine), 0),
            output=max(c["output"] - sum(p.output for p in mine), 0),
            cache_read=max(c["cache_read"] - sum(p.cache_read for p in mine), 0),
            cache_write=max(c["cache_write"] - sum(p.cache_write for p in mine), 0),
            web_searches=max(c["web_searches"] - sum(p.web_searches for p in mine), 0),
        )
        if (
            rest.input
            or rest.output
            or rest.cache_read
            or rest.cache_write
            or rest.web_searches
        ):
            slices.append(rest)
    return slices


def claude_session_usage(folder: Path) -> dict[str, Any] | None:
    """The tokens of the Claude Code session logs under ``folder`` (the
    trial's ``agent/claude-sessions/``), subagents' included; None when they
    hold no response.

    When each session's last ``cost-state`` counts at least every logged
    response of each kind, Claude Code's own totals are the count: they add
    the calls Claude Code makes for itself (a title, a compaction summary, a
    web fetch's digest), which no response line logs, and the output a
    response's lines logged before it ended (on MiMo a line can log a third
    of the response's output). Otherwise the logged responses, each once.
    ``main`` and ``subagents`` split the logged responses; ``unlogged`` is
    what Claude Code's totals count beyond them.
    """
    from benchflow._utils.reachable_write import ReachablePath
    from benchflow._utils.session_cost import covers, read_scoped, stated_counts

    logs: list[Path | ReachablePath] = (
        list(sorted(folder.rglob("*.jsonl"))) if folder.is_dir() else []
    )
    states, responses, subagent_usages = read_scoped(logs)
    usages = [(model, usage) for rs in responses.values() for model, usage in rs]
    if not usages:
        return None
    logged = _totals(usage for _model, usage in usages)
    if not logged["total_tokens"]:
        return None
    in_subagent = {id(usage) for usage in subagent_usages}
    split = {
        "main": _scope([u for _m, u in usages if id(u) not in in_subagent]),
        "subagents": _scope([u for _m, u in usages if id(u) in in_subagent]),
    }
    counts = stated_counts(states)
    if covers(states, responses) and {m for m, _u in usages} <= set(counts):
        fresh = sum(c["input"] for c in counts.values())
        output = sum(c["output"] for c in counts.values())
        read = sum(c["cache_read"] for c in counts.values())
        write = sum(c["cache_write"] for c in counts.values())
        # The 5-minute/1-hour split is the responses' when the totals add
        # no cache write to theirs; otherwise unknown.
        exact = write == logged["n_cache_creation_tokens"]
        totals = {
            "n_input_tokens": fresh,
            "n_output_tokens": output,
            "n_cache_read_tokens": read,
            "n_cache_creation_tokens": write,
            "n_cache_write_5m_tokens": logged["n_cache_write_5m_tokens"]
            if exact
            else None,
            "n_cache_write_1h_tokens": logged["n_cache_write_1h_tokens"]
            if exact
            else None,
            "total_tokens": fresh + output + read + write,
        }
        unlogged = {
            field: totals[field] - logged[field]
            for field in (
                "n_input_tokens",
                "n_output_tokens",
                "n_cache_read_tokens",
                "n_cache_creation_tokens",
                "total_tokens",
            )
        }
        return {
            **totals,
            "responses": len(usages),
            "counted_from": COUNTED_FROM_TOTALS,
            "scope": SCOPE_ALL,
            **split,
            "unlogged": unlogged,
            "slices": _covered_slices(usages, counts),
        }
    return {
        **logged,
        "responses": len(usages),
        "counted_from": COUNTED_FROM_RESPONSES,
        "scope": SCOPE_LOGGED,
        **split,
        "slices": [anthropic_slice(model, usage) for model, usage in usages],
    }


def _session_logs(folder: Path) -> list[Path]:
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


def _rows(path: Path) -> Iterator[dict[str, Any]]:
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


def codex_session_calls(trial: Path) -> list[dict[str, Any]]:
    """One record per model response in the trial's copy of Codex's session
    rollouts (``agent/codex-sessions/``): its model and usage.

    Newer Codex writes a ``token_usage_record`` per response; Codex 0.159 only
    a ``token_count`` event after each one (``info.last_token_usage``). A log
    with records is read from them alone, so a response is never counted
    twice. The model is the ``turn_context``'s before it. A sub-agent's
    thread is a rollout of its own and is counted too.
    """
    calls: list[dict[str, Any]] = []
    for log in _session_logs(trial / CODEX_SESSIONS):
        rows = list(_rows(log))
        records = any(row.get("type") == "token_usage_record" for row in rows)
        model = None
        for row in rows:
            raw = row.get("payload")
            payload: dict[str, Any] = raw if isinstance(raw, dict) else {}
            if row.get("type") == "turn_context" and isinstance(
                payload.get("model"), str
            ):
                model = payload["model"]
            usage = None
            if records and row.get("type") == "token_usage_record":
                usage = payload.get("usage")
            elif (
                not records
                and row.get("type") == "event_msg"
                and payload.get("type") == "token_count"
                and isinstance(info := payload.get("info"), dict)
            ):
                usage = info.get("last_token_usage")
            if isinstance(usage, dict):
                calls.append(
                    {
                        "model": model,
                        "input_tokens": _int(usage.get("input_tokens")),
                        "output_tokens": _int(usage.get("output_tokens")),
                        "cache_read_tokens": _int(usage.get("cached_input_tokens")),
                        "cache_write_tokens": _int(
                            usage.get("cache_write_input_tokens")
                        ),
                        "cache_write_reported": isinstance(
                            usage.get("cache_write_input_tokens"), int
                        ),
                        "reasoning_tokens": _int(usage.get("reasoning_output_tokens")),
                    }
                )
    return calls


def codex_session_tokens(trial: Path) -> dict[str, Any] | None:
    """Every model call's tokens in the trial's copy of Codex's session log,
    summed in ``result.json``'s terms (BenchFlow's split); None when
    it records no call. Codex's ``input_tokens`` include the cached ones;
    ``n_input_tokens`` leaves them out, so ``total_tokens`` is Codex's input
    plus output. Codex logs ``cache_write_input_tokens`` (0 on every call:
    OpenAI bills no cache write), a reported count, so the cache writes are
    known; None only when some call leaves the field out. ``thought_tokens``
    is the reasoning inside the output."""
    calls = codex_session_calls(trial)
    if not calls:
        return None
    fresh = read = write = output = reasoning = 0
    for call in calls:
        cached = call["cache_read_tokens"]
        written = call["cache_write_tokens"]
        fresh += max(call["input_tokens"] - cached - written, 0)
        read += cached
        write += written
        output += call["output_tokens"]
        reasoning += call["reasoning_tokens"]
    return {
        "n_input_tokens": fresh,
        "n_output_tokens": output,
        "n_cache_read_tokens": read,
        "n_cache_creation_tokens": (
            write if all(call["cache_write_reported"] for call in calls) else None
        ),
        "n_cache_write_5m_tokens": None,
        "n_cache_write_1h_tokens": None,
        "total_tokens": fresh + read + write + output,
        "thought_tokens": reasoning,
        "calls": len(calls),
    }


def codex_session_slices(trial: Path) -> list[TokenSlice]:
    """Each response in the trial's copy of Codex's session log as a slice
    to price. Codex's input counts the cached and written tokens; a slice's
    input leaves them out."""
    slices = []
    for call in codex_session_calls(trial):
        cached = call["cache_read_tokens"]
        written = call["cache_write_tokens"]
        prompt = call["input_tokens"]
        slices.append(
            TokenSlice(
                model=call["model"] or "",
                input=max(prompt - cached - written, 0),
                output=call["output_tokens"],
                cache_read=cached,
                cache_write=written,
                prompt=prompt,
            )
        )
    return slices


def _shares_another_rollouts_turns(rollout: Any) -> bool:
    """The sandbox (and its logs) hold turns another rollout is charged for:
    a fork's in-place child, an isolated child or a continued checkpoint
    (see ``_session_log.price_from_session_log``)."""
    return bool(
        getattr(rollout, "_branch_child_active", False)
        or getattr(rollout, "_branch_forks", None)
        or getattr(rollout, "_from_branch_snapshot", False)
    )


def _needs_salvage(rollout: Any, metrics: dict[str, Any]) -> bool:
    """Whether the trial's count is short: a turn ended without its report,
    or nothing counted its tokens at all."""
    if metrics.get("usage_source") == USAGE_SOURCE_PROVIDER_RESPONSE:
        return False  # the proxy counted every call it forwarded
    if getattr(rollout, "_usage_salvaged", False):
        return False
    return bool(getattr(rollout, "_usage_report_missing", False)) or not (
        is_token_usage_available(metrics)
    )


def _claude_only(rollout: Any) -> bool:
    from benchflow.rollout._session_log import _contributors, _runs_claude_code

    agents = _contributors(rollout)
    return bool(agents) and all(_runs_claude_code(agent) for agent in agents)


def _replace_usage(
    rollout: Any,
    counted: dict[str, Any],
    *,
    source: str,
    detail: dict[str, Any],
) -> None:
    """Make ``counted`` the trial's tokens, from ``source``; keep what the
    end-of-turn reports said in ``usage_details.token_count``."""
    metrics = dict(getattr(rollout, "_usage_metrics", None) or {})
    details = dict(metrics.get("usage_details") or {})
    reported = (
        {field: metrics.get(field) for field in TOKEN_FIELDS}
        if is_token_usage_available(metrics)
        else None
    )
    details["token_count"] = {**detail, "end_of_turn_reported": reported}
    salvaged = {
        **metrics,
        **{field: counted[field] for field in TOKEN_FIELDS},
        "usage_source": source,
        # The count is the agent's own, as the end-of-turn report was.
        "usage_source_legacy": USAGE_SOURCE_AGENT_NATIVE_ACP,
        "usage_details": details,
    }
    if metrics.get("price_source") == LIST_PRICE_SOURCE:
        # A price of the old count: price the new one.
        salvaged.update(cost_usd=None, price_source=None, cost_source=None)
    rollout._usage_metrics = salvaged
    # A later re-finalization (cleanup after a separate verifier) starts
    # from the native metrics: keep the salvage there too.
    rollout._native_usage_metrics = dict(salvaged)
    rollout._usage_slices = list(counted.get("slices") or [])
    rollout._usage_salvaged = True


def salvage_from_stream_log(rollout: Any) -> None:
    """Count a native Claude Code trial from its host-side stream log when
    a turn ended without its report, or nothing was counted."""
    from benchflow.native_harness.spec import HARNESS_NATIVE

    try:
        metrics = getattr(rollout, "_usage_metrics", None) or {}
        paths = getattr(rollout, "_rollout_paths", None)
        if (
            paths is None
            or getattr(rollout._config, "harness", None) != HARNESS_NATIVE
            or not _claude_only(rollout)
            or _shares_another_rollouts_turns(rollout)
            or not _needs_salvage(rollout, metrics)
        ):
            return
        path = Path(paths.rollout_dir) / "agent" / STREAM_LOG
        counted = claude_stream_usage(path)
        if counted is None:
            return
        _replace_usage(
            rollout,
            counted,
            source=USAGE_STREAM_LOG,
            detail={
                "source": STREAM_TOKENS_SOURCE,
                "path": f"agent/{STREAM_LOG}",
                "turns": counted["turns"],
                "turns_without_report": counted["turns_without_report"],
                "messages": counted["messages"],
            },
        )
        logger.info(
            "Tokens counted from Claude Code's stream log: %d tokens over %d "
            "turn(s), %d without a final report",
            counted["total_tokens"],
            counted["turns"],
            counted["turns_without_report"],
        )
    except Exception as exc:  # never fail the rollout over a count
        logger.info("Could not count Claude Code's stream log: %s", exc)


def salvage_from_session_log(rollout: Any) -> None:
    """Count a Claude Code trial over ACP from the session log teardown
    copied into ``agent/claude-sessions/``.

    The end-of-turn report (ACP's ``PromptResponse.usage``) counts the main
    session only: a subagent's calls, and the calls Claude Code makes for
    itself, are left out (benchflow-ai/benchflow#1241). Whenever the log
    was copied, its count (:func:`claude_session_usage`) is the trial's.

    A log can count fewer tokens of a kind than the reports did (its lines
    can be written before a response ends; MiMo's log a fraction of the
    output), so it is short of that kind (``short_of``). Then the reports
    stay, the log's count beside them in ``usage_details.token_count``; or,
    when a turn ended without its report, each kind is the larger of the
    two counts, as each counts some of the calls only. Either way the count
    is marked incomplete (``INCOMPLETE_SHORT_LOG``, through
    :func:`mark_incomplete_usage`), and a cost priced from the log reads
    ``cost_incomplete``. ``usage_details.token_count.scope`` names what the
    count covers.
    """
    try:
        metrics = getattr(rollout, "_usage_metrics", None) or {}
        paths = getattr(rollout, "_rollout_paths", None)
        if (
            paths is None
            or not _claude_only(rollout)
            or _shares_another_rollouts_turns(rollout)
            or metrics.get("usage_source") == USAGE_SOURCE_PROVIDER_RESPONSE
            or getattr(rollout, "_usage_salvaged", False)
        ):
            return
        counted = claude_session_usage(Path(paths.rollout_dir) / CLAUDE_SESSIONS)
        if counted is None:
            _label_scope(rollout)
            return
        detail = {
            "source": SESSION_TOKENS_SOURCE,
            "path": CLAUDE_SESSIONS,
            "responses": counted["responses"],
            "counted_from": counted["counted_from"],
            "scope": counted["scope"],
            "main": counted["main"],
            "subagents": counted["subagents"],
            **({"unlogged": counted["unlogged"]} if "unlogged" in counted else {}),
        }
        # Of each kind, a log that counts less than the reports did (its
        # lines can be written before a response ends, and log less output
        # than it had) is short of that kind.
        short = (
            [
                field
                for field in _KINDS
                if _int(counted.get(field)) < _int(metrics.get(field))
            ]
            if is_token_usage_available(metrics)
            else []
        )
        if short and not _needs_salvage(rollout, metrics):
            # Every turn reported: the reports stay, the log's count beside
            # them, and the cost priced from the log is short of them.
            details = dict(metrics.get("usage_details") or {})
            details["token_count"] = {
                "source": USAGE_END_OF_TURN,
                "scope": SCOPE_MAIN,
                "session_log": {
                    **detail,
                    **{field: counted.get(field) for field in TOKEN_FIELDS},
                    "short_of": short,
                },
            }
            rollout._usage_metrics = {**metrics, "usage_details": details}
            _short_log_incomplete(rollout, short)
            logger.info(
                "Claude Code's session log counts fewer %s than its end-of-turn "
                "reports: the reports stay",
                ", ".join(short),
            )
            return
        if short:
            # A turn ended without its report: the turns that reported and
            # the log each count some of the calls. Each kind is the larger.
            counted = _at_least_reported(counted, metrics, short)
            detail["short_of"] = short
        _replace_usage(rollout, counted, source=USAGE_SESSION_LOG, detail=detail)
        if short:
            _short_log_incomplete(rollout, short)
        logger.info(
            "Tokens counted from Claude Code's session log: %d tokens in %d "
            "responses (%d in subagents), from %s",
            counted["total_tokens"],
            counted["responses"],
            counted["subagents"]["responses"],
            counted["counted_from"],
        )
    except Exception as exc:  # never fail the rollout over a count
        logger.info("Could not count Claude Code's session log: %s", exc)


def _at_least_reported(
    counted: dict[str, Any], metrics: dict[str, Any], short: list[str]
) -> dict[str, Any]:
    """The log's count with each kind in *short* raised to the reports'.
    A cache write raised from the reports has no 5-minute/1-hour split."""
    raised = {**counted, **{field: _int(metrics.get(field)) for field in short}}
    if "n_cache_creation_tokens" in short:
        raised["n_cache_write_5m_tokens"] = None
        raised["n_cache_write_1h_tokens"] = None
    raised["total_tokens"] = sum(_int(raised.get(field)) for field in _KINDS)
    return raised


def _short_log_incomplete(rollout: Any, short: list[str]) -> None:
    """Have :func:`mark_incomplete_usage` mark the count: the log is short
    of the reported kinds *short*. A reason already given is kept, this one
    after it."""
    names = [_KINDS[field] for field in short]
    kinds = " and ".join([", ".join(names[:-1]), names[-1]] if names[:-1] else names)
    reason = INCOMPLETE_SHORT_LOG.format(kinds=kinds)
    held = getattr(rollout, "_usage_incomplete_reason", None)
    rollout._usage_incomplete_reason = f"{held}; {reason}" if held else reason


def _label_scope(rollout: Any) -> None:
    """With no session log to count, name what an end-of-turn count covers:
    Claude Code's main session only (benchflow-ai/benchflow#1241)."""
    metrics = getattr(rollout, "_usage_metrics", None) or {}
    if not is_token_usage_available(metrics) or metrics.get("usage_source") not in (
        USAGE_SOURCE_AGENT_NATIVE_ACP,
        USAGE_END_OF_TURN,
    ):
        return
    details = dict(metrics.get("usage_details") or {})
    count = dict(details.get("token_count") or {})
    count.setdefault("source", USAGE_END_OF_TURN)
    count.setdefault("scope", SCOPE_MAIN)
    details["token_count"] = count
    rollout._usage_metrics = {**metrics, "usage_details": details}


INCOMPLETE_REASON = (
    "a turn ended without its usage report and no log of the agent's could count it"
)
#: An in-sandbox model proxy's log that may miss calls: the sandbox was lost
#: before the log could be read at the end, and only what the live capture
#: read while the turn ran is counted.
INCOMPLETE_SANDBOX_PROXY = (
    "the in-sandbox model proxy's usage log may miss calls: {detail}"
)
INCOMPLETE_SHORT_LOG = (
    "Claude Code's session log counts fewer {kinds} tokens than the turns "
    "reported (its lines can be written before a response ends), so each "
    "count of those kinds is the reports', and the cost priced from the log "
    "is short of them"
)


INCOMPLETE_TURN_CUT = (
    "turn cut on a subscription route; a call in flight may be missing"
)
#: Counts of the calls an agent made itself, on a login or a native CLI:
#: its end-of-turn report, or its own logs.
_SUBSCRIPTION_SOURCES = frozenset(
    {
        USAGE_SOURCE_AGENT_NATIVE_ACP,
        USAGE_STREAM_LOG,
        USAGE_SESSION_LOG,
        USAGE_END_OF_TURN,
    }
)


def mark_incomplete_usage(rollout: Any) -> None:
    """Mark a count that misses calls: ``usage_incomplete`` with
    ``usage_incomplete_reason``, so a partial count (and its cost,
    ``cost_incomplete``) is not read as whole.

    A count step that knows its count is short says why
    (``rollout._usage_incomplete_reason``: a session log short of a
    reported kind, :func:`salvage_from_session_log`). Otherwise a turn that
    ended without its usage report (a timeout, a cancel, a lost sandbox)
    and that no log of the agent's counted leaves the trial with the tokens
    of the turns that reported. A turn cut that way on a login or a native
    CLI leaves a count a floor even when a log counted it: a call in flight
    is in no log (``INCOMPLETE_TURN_CUT``; an agent error is no cut). A
    proxy-only trial is complete: the proxy
    counted every call it forwarded, unless it saw a stream cut before the
    provider's usage report, which the proxy's import marks
    (``litellm_logging._mark_interrupted_streams``). Never fails
    the rollout."""
    try:
        metrics = getattr(rollout, "_usage_metrics", None) or {}
        proxy = getattr(rollout, "_proxy_usage_incomplete_reason", None)
        if isinstance(proxy, str):
            # Whatever counted the tokens, and when nothing did: the proxy's
            # log was the trial's record of its calls. A reason another step
            # gave (already marked, or a count step's) is kept before it.
            held = (
                metrics.get("usage_incomplete_reason")
                if metrics.get("usage_incomplete")
                else None
            ) or getattr(rollout, "_usage_incomplete_reason", None)
            if isinstance(held, str) and held:
                reason = held if proxy in held else f"{held}; {proxy}"
            else:
                reason = proxy
            if (
                metrics.get("usage_incomplete") is not True
                or metrics.get("usage_incomplete_reason") != reason
            ):
                rollout._usage_metrics = {
                    **metrics,
                    "usage_incomplete": True,
                    "usage_incomplete_reason": reason,
                }
                logger.warning("The trial's token count is incomplete: %s", reason)
            return
        if not is_token_usage_available(metrics) or metrics.get("usage_incomplete"):
            return
        reason = getattr(rollout, "_usage_incomplete_reason", None)
        if reason is None:
            details = metrics.get("usage_details") or {}
            proxied_only = (
                metrics.get("usage_source") == USAGE_SOURCE_PROVIDER_RESPONSE
                and USAGE_SOURCE_AGENT_NATIVE_ACP not in details
            )
            if (
                getattr(rollout, "_usage_report_missing", False)
                and not getattr(rollout, "_usage_salvaged", False)
                and not proxied_only  # the proxy counted every call it forwarded
            ):
                reason = INCOMPLETE_REASON
            elif getattr(rollout, "_turn_cut", None) and (
                metrics.get("usage_source") in _SUBSCRIPTION_SOURCES
                or (
                    metrics.get("usage_source") == USAGE_SOURCE_PROVIDER_RESPONSE
                    and USAGE_SOURCE_AGENT_NATIVE_ACP in details
                )
            ):
                # A turn cut (a timeout, a cancel, a budget stop, a lost
                # sandbox) on a login or a native CLI: the agent's log keeps
                # only the responses that ended, so a call in flight is in
                # neither it nor the count.
                reason = INCOMPLETE_TURN_CUT
        if reason is None:
            return
        rollout._usage_metrics = {
            **metrics,
            "usage_incomplete": True,
            "usage_incomplete_reason": reason,
        }
    except Exception as exc:  # never fail the rollout over a marker
        logger.info("Could not mark the trial's count incomplete: %s", exc)


def _fallback_model(rollout: Any) -> str:
    """The model to price an end-of-turn count with when its role named
    none: the one model the agent's own log says answered, else the run's."""
    reported = list(
        dict.fromkeys(getattr(rollout, "_session_reported_models", None) or [])
    )
    if len(reported) == 1:
        return reported[0]
    return getattr(rollout._config, "model", None) or ""


def _end_of_turn_slices(rollout: Any, *, unproxied_only: bool) -> list[TokenSlice]:
    """The trial's end-of-turn counts, one slice per model (BenchFlow's
    split: uncached input beside the cache)."""
    slices = []
    for (model, unproxied), counts in (
        getattr(rollout, "_native_usage_slices", None) or {}
    ).items():
        if unproxied_only and not unproxied:
            continue
        read = counts.get("cached_read_tokens", 0)
        write = counts.get("cached_write_tokens", 0)
        fresh = counts.get("input_tokens", 0)
        slices.append(
            TokenSlice(
                model=model or _fallback_model(rollout),
                input=fresh,
                output=counts.get("output_tokens", 0),
                cache_read=read,
                cache_write=write,
            )
        )
    return slices


def _snapshot_date_for(models: Iterable[str]) -> str | None:
    """The snapshot's date when every model is priced from it, else None."""
    from benchflow.providers.pricing_snapshot import SNAPSHOT_DATE

    names = [m for m in models if m]
    if not names:
        return None
    tables = []
    for name in names:
        rates = list_rates(name)
        tables.append(rates.table if rates else None)
    return (
        SNAPSHOT_DATE if tables and all(t == SNAPSHOT_TABLE for t in tables) else None
    )


def _label_cost(rollout: Any, metrics: dict[str, Any]) -> dict[str, Any]:
    """``cost_source`` and ``price_table_date`` for a cost someone else set."""
    if metrics.get("cost_source") is not None:
        return metrics
    source = metrics.get("price_source")
    labelled = dict(metrics)
    if source == PROVIDER_PRICE_SOURCE:
        labelled.update(cost_source=COST_SOURCE_PROVIDER, price_table_date=None)
        return labelled
    labelled["cost_source"] = COST_SOURCE_LIST_PRICE
    if labelled.get("price_table_date") is None:
        estimate = (metrics.get("usage_details") or {}).get("cost_estimate") or {}
        if source == "litellm":
            routed = getattr(rollout._config, "model", None) or ""
            models = list(getattr(rollout, "_proxy_reported_models", None) or []) or [
                routed
            ]
            # The gateway gives snapshot rates to a vendor's own host only: an
            # OpenRouter route was priced from LiteLLM's table, whatever model
            # it serves, so it has no snapshot date.
            labelled["price_table_date"] = (
                None if routed.startswith("openrouter/") else _snapshot_date_for(models)
            )
        elif estimate.get("method") == "usage-at-list-price" or (
            # Claude Code's own counts, every model priced by BenchFlow (a
            # Claude model's part, if any, is Claude Code's figure: undated).
            estimate.get("method") == "claude-code-tokens-at-list-price"
            and set(estimate.get("models") or {})
            == set(estimate.get("repriced_models") or [])
        ):
            labelled["price_table_date"] = _snapshot_date_for(
                (estimate.get("models") or {}).keys()
            )
    return labelled


def apply_list_price(rollout: Any) -> None:
    """Price a trial that has tokens and no cost; label every cost.

    A cost already set (the gateway's, Claude Code's session-log estimate)
    keeps its figure and gets ``cost_source`` and ``price_table_date``.
    Otherwise the trial's tokens -- salvaged from a log, or the end-of-turn
    reports, per model -- are priced at list prices. A trial whose proxied
    calls the gateway priced and whose login calls it could not see (a
    mixed trial) costs the sum of the two. A trial with no tokens, or with a
    model no table prices, keeps a null cost.
    """
    try:
        metrics = dict(getattr(rollout, "_usage_metrics", None) or {})
        if metrics.get("cost_usd") is not None:
            rollout._usage_metrics = _label_cost(rollout, metrics)
            return
        details = dict(metrics.get("usage_details") or {})
        mixed = isinstance(details.get(USAGE_SOURCE_PROVIDER_RESPONSE), dict)
        provider_usd: float | None = None
        salvaged = getattr(rollout, "_usage_slices", None)
        unreported = getattr(rollout, "_native_usage_unreported", None) or set()
        if not is_token_usage_available(metrics):
            return
        elif unreported and not salvaged:
            # The agent's end-of-turn report left a cache count out: those
            # tokens are null, and a price of the rest would pass for the
            # whole.
            logger.info(
                "The agent's usage report left out %s: the cost stays unknown",
                ", ".join(sorted(unreported)),
            )
            return
        elif (
            metrics.get("usage_source") == USAGE_SOURCE_PROVIDER_RESPONSE and not mixed
        ):
            # The gateway priced what it could; a model no table it reads
            # prices is no model a bare-name guess should price (a private
            # deployment of an open model is not its vendor's API).
            return
        elif mixed:
            provider_usd = details[USAGE_SOURCE_PROVIDER_RESPONSE].get("cost_usd")
            if provider_usd is None:
                return  # the proxied part has no price: the sum would be short
            slices = _end_of_turn_slices(rollout, unproxied_only=True)
        else:
            slices = getattr(rollout, "_usage_slices", None) or _end_of_turn_slices(
                rollout, unproxied_only=False
            )
        priced = price_slices(list(slices))
        if priced is None:
            logger.info(
                "No list price for the trial's model(s) %s: its cost stays unknown",
                sorted({piece.model for piece in slices}),
            )
            return
        usd = priced.usd + (provider_usd or 0.0)
        details["cost_estimate"] = {
            "source": "benchflow-list-price",
            "method": "usage-at-list-price",
            "models": priced.models,
            "tables": priced.tables,
            **({"provider_response_usd": provider_usd} if mixed else {}),
        }
        table_date = priced.table_date
        proxied_models = getattr(rollout, "_proxy_reported_models", None) or []
        if mixed and table_date != _snapshot_date_for(proxied_models):
            table_date = None  # the two parts were priced from different tables
        rollout._usage_metrics = {
            **metrics,
            "cost_usd": round(usd, 10),
            "price_source": LIST_PRICE_SOURCE,
            "cost_source": COST_SOURCE_LIST_PRICE,
            "price_table_date": table_date,
            "usage_details": details,
        }
        logger.info(
            "Cost at list prices: $%.4f (%s)",
            usd,
            ", ".join(f"{m}: {t}" for m, t in sorted(priced.tables.items())),
        )
    except Exception as exc:  # never fail the rollout over a price
        logger.info("Could not price the trial's tokens: %s", exc)


__all__ = [
    "LIST_PRICE_SOURCE",
    "PROVIDER_PRICE_SOURCE",
    "apply_list_price",
    "claude_session_usage",
    "mark_incomplete_usage",
    "claude_stream_usage",
    "codex_session_calls",
    "codex_session_slices",
    "codex_session_tokens",
    "salvage_from_session_log",
    "salvage_from_stream_log",
]
