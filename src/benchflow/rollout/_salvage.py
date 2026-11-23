"""Count a turn that ended without its usage report.

A run around BenchFlow's model proxy (a Claude or ChatGPT login, the native
harness) learns its tokens from the agent's end-of-turn report: ACP's
``PromptResponse.usage``, Claude Code's final ``result`` event, Codex's
``turn.completed``. A turn that never ends -- a timeout, a cancel, a lost
sandbox -- sends none, and the trial recorded no tokens. At teardown:

- :func:`salvage_from_stream_log` counts a native Claude Code trial from its
  own output stream, which BenchFlow keeps on the host
  (``agent/claude-code.jsonl``), so a lost sandbox loses nothing: a turn
  that reported counts what it reported, a turn that did not counts its
  messages (input and cache from ``message_start``, output from the last
  ``message_delta``, once per message id).
- :func:`salvage_from_session_log` counts a Claude Code trial over ACP from
  Claude Code's session log, copied from the sandbox into
  ``agent/claude-sessions/``; with no log (the sandbox is gone) it keeps the
  running cost Claude Code last sent over ACP, if any.
- Codex is counted from its session log by
  :func:`benchflow.rollout._session_log.count_codex_tokens`, with
  :func:`codex_session_slices` to price.

The counts carry what to price (``rollout._usage_slices``) for
:func:`benchflow.rollout._cost.apply_list_price`. Each step never fails the
rollout.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchflow._utils.pricing import TokenSlice, anthropic_slice
from benchflow.rollout._cost import (
    INCOMPLETE_MIRROR,
    INCOMPLETE_SHORT_LOG,
    LIST_PRICE_SOURCE,
)
from benchflow.usage_tracking import (
    COST_SOURCE_LIST_PRICE,
    USAGE_SESSION_LOG,
    USAGE_SOURCE_AGENT_NATIVE_ACP,
    USAGE_SOURCE_PROVIDER_RESPONSE,
    USAGE_STREAM_LOG,
    is_token_usage_available,
)

logger = logging.getLogger(__name__)

STREAM_LOG = "claude-code.jsonl"  # under the trial's agent/ folder
STREAM_TOKENS_SOURCE = "claude-code-stream"
SESSION_TOKENS_SOURCE = "claude-code-session-log"
CLAUDE_SESSIONS = "agent/claude-sessions"
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
            if isinstance(written.get(key), int):
                held_written[key] = max(
                    _int(held_written.get(key)), _int(written.get(key))
                )
    server = usage.get("server_tool_use")
    if isinstance(server, dict):
        held_server = held.setdefault("server_tool_use", {})
        held_server["web_search_requests"] = max(
            _int(held_server.get("web_search_requests")),
            _int(server.get("web_search_requests")),
        )


def _totals(usages: Iterable[dict[str, Any]]) -> dict[str, int | None]:
    """result.json's token fields for Anthropic-shaped usages (split:
    input leaves the cache out; cache writes with their 5-minute and 1-hour
    parts where the usages split them)."""
    fresh = output = read = write = 0
    five: int | None = None
    hour: int | None = None
    for usage in usages:
        fresh += _int(usage.get("input_tokens"))
        output += _int(usage.get("output_tokens"))
        read += _int(usage.get("cache_read_input_tokens"))
        write += _int(usage.get("cache_creation_input_tokens"))
        written = usage.get("cache_creation")
        if isinstance(written, dict):
            if isinstance(written.get("ephemeral_5m_input_tokens"), int):
                five = (five or 0) + _int(written.get("ephemeral_5m_input_tokens"))
            if isinstance(written.get("ephemeral_1h_input_tokens"), int):
                hour = (hour or 0) + _int(written.get("ephemeral_1h_input_tokens"))
    return {
        "n_input_tokens": fresh,
        "n_output_tokens": output,
        "n_cache_read_tokens": read,
        "n_cache_creation_tokens": write,
        "n_cache_write_5m_tokens": five,
        "n_cache_write_1h_tokens": hour,
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
    from benchflow._utils.pricing import TokenSlice

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
            field: _int(totals[field]) - _int(logged[field])
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


def codex_session_slices(trial: Path) -> list[TokenSlice]:
    """Each response in the trial's copy of Codex's session log
    (``agent/codex-sessions/``) as a slice to price. Codex's input counts
    the cached and written tokens; a slice's input leaves them out."""
    from benchflow.trajectories.model_calls import CODEX_SESSIONS, _codex_calls

    slices = []
    for call in _codex_calls(trial / CODEX_SESSIONS):
        cached = call["cache_read_tokens"] or 0
        written = call["cache_write_tokens"] or 0
        prompt = call["input_tokens"] or 0
        slices.append(
            TokenSlice(
                model=call["model_served"] or "",
                input=max(prompt - cached - written, 0),
                output=call["output_tokens"] or 0,
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
        **{field: counted.get(field) for field in TOKEN_FIELDS},
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
        snapshot = getattr(rollout, "_session_log_snapshot", None)
        if snapshot is not None:
            # agent/ is shared with the sandbox, where test.sh runs as root:
            # the copy made at the end of the agent's turn.
            if snapshot.stream is None:
                return
            path = snapshot.stream
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
    :func:`~benchflow.rollout._cost.mark_incomplete_usage`), and a cost
    priced from the log reads ``cost_incomplete``. Without a log, a turn that
    ends before its report keeps the running cost Claude Code last sent over
    ACP, if any. ``usage_details.token_count.scope`` names what the count
    covers.
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
        from benchflow.rollout._session_log import session_log_root

        counted = claude_session_usage(session_log_root(rollout) / CLAUDE_SESSIONS)
        if counted is None:
            if _needs_salvage(rollout, metrics):
                _cost_from_usage_update(rollout)
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
                "source": "end_of_turn",
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
        placed = getattr(rollout, "_mirror_placed", None) or {}
        if placed.get(Path(CLAUDE_SESSIONS).name) is False:
            # The copy made while the turns ran had not caught up.
            rollout._usage_incomplete_reason = INCOMPLETE_MIRROR
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
    """Have :func:`~benchflow.rollout._cost.mark_incomplete_usage` mark the
    count: the log is short of the reported kinds *short*. A reason already
    given (a copy that had not caught up) is kept, this one after it."""
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
        "end_of_turn",
    ):
        return
    details = dict(metrics.get("usage_details") or {})
    count = dict(details.get("token_count") or {})
    count.setdefault("source", "end_of_turn")
    count.setdefault("scope", SCOPE_MAIN)
    details["token_count"] = count
    rollout._usage_metrics = {**metrics, "usage_details": details}


def _cost_from_usage_update(rollout: Any) -> None:
    """With no session log to count (the sandbox is gone), keep the running
    cost Claude Code last sent over ACP (a ``usage_update``'s ``cost``), if
    any: its own list-price estimate. Its tokens stay unknown; ACP's
    ``usage_update`` carries how full the context is, not a count."""
    from benchflow._utils.session_cost import claude_code_prices
    from benchflow.rollout._cost import _fallback_model

    metrics = getattr(rollout, "_usage_metrics", None) or {}
    if metrics.get("cost_usd") is not None or is_token_usage_available(metrics):
        return
    env = getattr(rollout, "_agent_env", None) or {}
    model = _fallback_model(rollout) or str(env.get("ANTHROPIC_MODEL") or "")
    if model and not claude_code_prices(model):
        # Claude Code's running cost of another provider's model is at its
        # default rate, not the model's (benchflow-ai/benchflow#1240).
        return
    update = getattr(rollout, "_last_usage_update", None) or {}
    cost = update.get("cost") if isinstance(update, dict) else None
    amount = cost.get("amount") if isinstance(cost, dict) else None
    currency = str((cost or {}).get("currency") or "USD").upper()
    if (
        isinstance(amount, bool)
        or not isinstance(amount, int | float)
        or currency != "USD"
    ):
        return
    details = dict(metrics.get("usage_details") or {})
    details["cost_estimate"] = {"source": "acp-usage-update", "method": "agent-total"}
    rollout._usage_metrics = {
        **metrics,
        "cost_usd": round(float(amount), 10),
        "price_source": "agent_usage_update",
        "cost_source": COST_SOURCE_LIST_PRICE,
        "price_table_date": None,
        "usage_details": details,
    }
