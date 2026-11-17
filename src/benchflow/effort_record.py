"""result.json's ``effort``: the effort asked for beside the one that ran.

0.9.2 wrote no applied effort at all, so a run asked for ``ultra`` that ran at
``medium`` looked like any other. Each trial now records:

- ``requested``: what the run asked for, however it asked
  (:mod:`benchflow.effort_request`), and ``requested_from``, which ways;
- ``applied``: the agent session's own word (the ``model[effort]`` id
  codex-acp took, the value an effort option reports), never the request
  echoed back: None when the session says nothing;
- ``sent``: the agent's model calls counted by the effort each carried,
  from the model proxy's log (``trajectory/llm_trajectory.jsonl``), else
  from Codex's own session log (``agent/codex-sessions``), with
  ``sent_source``; None when neither says;
- ``matches``: False when ``applied`` or any call differs from
  ``requested``; True only when the calls carried the requested effort;
  None otherwise (unknown is not a match).

A mismatch adds a trial warning, counts in ``summary.json``
(``effort_mismatches``), and fails ``bench run --fail-on effort-mismatch``.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

PROXY = "proxy"
CODEX_SESSION_LOG = "codex-session-log"
PROXY_LOG = Path("trajectory") / "llm_trajectory.jsonl"
CODEX_SESSIONS = Path("agent") / "codex-sessions"
_SIDE_CALLS = frozenset({"title", "summary", "compaction", "helper", "judge"})


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            yield row


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _body_effort(body: Any) -> str | None:
    """The effort one request carried upstream, read from the body LiteLLM
    sent (``upstream`` in the proxy's log), or, in a log
    written before it was kept, from the request the gateway took in:
    Anthropic's ``output_config.effort``, the Responses API's
    ``reasoning.effort``, OpenAI's ``reasoning_effort``, in that order. A call
    that carried a thinking budget and no effort level reads as the budget
    (``budget_tokens=16384``), never as the level asked for."""
    if not isinstance(body, dict):
        return None
    upstream = body.get("upstream")
    fields = upstream if isinstance(upstream, dict) else body
    for key in ("output_config", "reasoning"):
        part = fields.get(key)
        if isinstance(part, dict) and isinstance(part.get("effort"), str):
            return part["effort"]
    if isinstance(fields.get("reasoning_effort"), str):
        return fields["reasoning_effort"]
    thinking = fields.get("thinking")
    if (
        isinstance(thinking, dict)
        and thinking.get("type") == "enabled"
        and isinstance(thinking.get("budget_tokens"), int)
    ):
        return f"budget_tokens={thinking['budget_tokens']}"
    config = fields.get("thinkingConfig")
    if isinstance(config, dict) and isinstance(config.get("thinkingBudget"), int):
        return f"thinking_budget={config['thinkingBudget']}"
    return None


def _effort_level(value: Any) -> str | None:
    """*value* when it is an effort level, else None (``default``, ``auto``:
    nothing applied)."""
    from benchflow._utils.config import EFFORT_LEVELS

    return value if value in EFFORT_LEVELS else None


def _proxy_efforts(trial: Path) -> tuple[dict[str, int], int] | None:
    """The agent's calls through the proxy by effort, and how many calls
    could carry one (``{}`` and the count when none did); None when it saw
    no call. A call offering no tools beside calls that do (a session
    title) is a side call."""
    rows = []
    for row in _rows(trial / PROXY_LOG):
        if _dict(row.get("metadata")).get("call_purpose") in _SIDE_CALLS:
            continue
        rows.append(_dict(_dict(row.get("request")).get("body")))
    if not rows:
        return None
    if any(body.get("tools") for body in rows):
        rows = [body for body in rows if body.get("tools")]
    counts: Counter[str] = Counter()
    for body in rows:
        effort = _body_effort(body)
        if effort:
            counts[effort.lower()] += 1
    return dict(counts), len(rows)


def _codex_efforts(trial: Path) -> tuple[dict[str, int], int] | None:
    """Codex's responses by the effort of the ``turn_context`` before each
    (a ``token_count`` event per response), the main thread's only, and how
    many responses there were; None when its log recorded no effort."""
    folder = trial / CODEX_SESSIONS
    if not folder.is_dir():
        return None
    counts: Counter[str] = Counter()
    responses = 0
    for root, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not os.path.islink(os.path.join(root, d)))
        for name in sorted(files):
            path = Path(root) / name
            if not name.endswith(".jsonl") or path.is_symlink():
                continue
            rows = list(_rows(path))
            if any(
                row.get("type") == "session_meta"
                and isinstance((row.get("payload") or {}).get("source"), dict)
                and "subagent" in row["payload"]["source"]
                for row in rows
            ):
                continue
            effort = None
            for row in rows:
                payload = _dict(row.get("payload"))
                if row.get("type") == "turn_context" and isinstance(
                    payload.get("effort"), str
                ):
                    effort = payload["effort"].lower()
                elif (
                    row.get("type") == "event_msg"
                    and payload.get("type") == "token_count"
                    and isinstance(payload.get("info"), dict)
                ):
                    responses += 1
                    if effort:
                        counts[effort] += 1
    return (dict(counts), responses) if counts else None


def sent_efforts(
    trial: Path,
) -> tuple[dict[str, int] | None, str | None, int | None]:
    """``(sent, sent_source, calls)`` for result.json's effort: *calls* is
    how many of the agent's calls could carry an effort."""
    proxy = _proxy_efforts(trial)
    if proxy is not None:
        return proxy[0], PROXY, proxy[1]
    codex = _codex_efforts(trial)
    if codex is not None:
        return codex[0], CODEX_SESSION_LOG, codex[1]
    return None, None, None


def _is_budget(value: str) -> bool:
    """A thinking budget recorded as itself (``budget_tokens=16384``,
    Gemini's ``thinking_budget=N``): no effort level."""
    return value.startswith(("budget_tokens=", "thinking_budget="))


def effort_outcome(
    *,
    requested: str | None,
    requested_from: str | None,
    applied: str | None,
    sent: dict[str, int] | None,
    sent_source: str | None,
    calls: int | None = None,
) -> dict[str, Any]:
    """The block result.json keeps under ``effort``.

    ``matches``: every agent call that could carry an effort (*calls*) must
    carry the expected one, the request, or with none asked for the
    session's ``applied`` (PR #17: no ``-e``, applied medium, calls at
    xhigh). False on any call that differs, an ``applied`` other than the
    request, or, when an effort was asked for, a proxy that saw calls and
    none carried one; True only when every call carried it; None when some
    carried none (``partly unmeasured``, with ``calls_with_effort`` and
    ``calls_without_effort``; ``calls_with_budget`` counts the calls that
    carried only a thinking budget, apart from both), when no call was seen
    carrying one
    (``unmeasured``), or when nothing was asked or applied
    (``nothing_to_compare``). A call that carried no effort is no
    agreement. A level a provider runs as another is measured here, a
    ``mismatch``: one BenchFlow knows of is refused before launch
    (:func:`benchflow.effort_request.refuse_known_downgrade`).
    """
    carried = {value: n for value, n in (sent or {}).items() if n}
    with_budget = sum(n for value, n in carried.items() if _is_budget(value))
    with_effort = sum(carried.values()) - with_budget
    without = (
        max(0, calls - with_effort - with_budget)
        if isinstance(calls, int) and sent is not None
        else None
    )
    # The session's word counts only as a level (claude-agent-acp answers
    # ``default`` when nothing was asked); the record keeps it as said.
    level = _effort_level(applied)
    expected = requested or level
    matches: bool | None
    reason: str | None
    if requested and level and level != requested:
        matches, reason = False, "mismatch"
    elif expected is None:
        matches, reason = None, "nothing_to_compare"
    elif carried and set(carried) != {expected}:
        matches, reason = False, "mismatch"
    elif carried and without:
        # Calls that carried no effort agree with nothing.
        matches, reason = None, "partly unmeasured"
    elif carried:
        matches, reason = True, None
    elif sent == {} and sent_source == PROXY:
        # The proxy saw the calls and none carried an effort.
        matches, reason = None, "unmeasured: no call carried an effort"
    else:
        matches, reason = None, "unmeasured"
    return {
        "requested": requested,
        "requested_from": requested_from if requested else None,
        "applied": applied,
        "sent": sent,
        "sent_source": sent_source if sent is not None else None,
        "calls_with_effort": with_effort if sent is not None else None,
        "calls_with_budget": with_budget if sent is not None else None,
        "calls_without_effort": without,
        "matches": matches,
        "reason": reason,
    }


def effort_block(
    trial: Path,
    *,
    agent: str | None,
    model: str | None,
    reasoning_effort: str | None,
    agent_env: Mapping[str, str] | None,
    applied: str | None,
    requested_from: str | None = None,
) -> dict[str, Any]:
    """result.json's ``effort`` for the trial in *trial*."""
    from benchflow.effort_request import (
        CLAUDE_EFFORT_ENV,
        effort_request,
        effort_source,
    )

    asked = effort_request(agent, model, reasoning_effort, agent_env)
    if asked.source == CLAUDE_EFFORT_ENV:
        # Claude Code applies its own variable over the session's effort: the
        # session's word is not what runs, and the variable is the request,
        # not evidence; the calls say what ran.
        applied = None
    sent, source, calls = sent_efforts(trial)
    return effort_outcome(
        requested=asked.effort,
        requested_from=effort_source(requested_from, asked.source),
        applied=applied,
        sent=sent,
        sent_source=source,
        calls=calls,
    )


def effort_warning(effort: Mapping[str, Any] | None) -> str | None:
    """The trial warning for an effort that is not the one asked for, or,
    with none asked for, calls at another effort than the session applied;
    and for calls of which only some said their effort."""
    if not effort:
        return None
    requested = effort.get("requested")
    applied = _effort_level(effort.get("applied"))
    expected = requested or applied
    if effort.get("reason") == "unmeasured: no call carried an effort":
        return (
            f"reasoning effort {expected} was "
            + ("asked for" if requested else "applied")
            + ", but none of the agent's model calls carried an effort, so the "
            "effort they ran at is unmeasured"
        )
    if effort.get("reason") == "partly unmeasured":
        with_effort = int(effort.get("calls_with_effort") or 0)
        without = int(effort.get("calls_without_effort") or 0)
        return (
            f"reasoning effort {expected} was "
            + ("asked for" if requested else "applied")
            + f": {with_effort} of the agent's {with_effort + without} model calls "
            f"carried it and {without} carried no effort, so the effort of those "
            "is unmeasured"
        )
    if effort.get("matches") is not False:
        return None
    found = []
    if requested and applied and applied != requested:
        found.append(f"the agent applied {applied}")
    sent = effort.get("sent")
    carried = {value: n for value, n in (sent or {}).items() if n}
    if carried and set(carried) != {expected}:
        found.append(
            "the agent's model calls carried "
            + ", ".join(f"{value} ({n} calls)" for value, n in sorted(carried.items()))
        )
    if requested:
        return f"reasoning effort {requested} was asked for, but " + "; ".join(found)
    return (
        f"no reasoning effort was asked for and the agent applied {expected}, but "
        + "; ".join(found)
    )


def effort_mismatches(results: Mapping[str, Mapping[str, Any]]) -> int:
    """The trials of a job whose effort is not the one asked for."""
    return sum(
        isinstance(result.get("effort"), dict)
        and result["effort"].get("matches") is False
        for result in results.values()
    )


def _unmeasured(reason: object) -> bool:
    """A reason the record gives when nothing measured the effort: ``unmeasured``,
    ``unmeasured: no call carried an effort`` or ``partly unmeasured``."""
    return isinstance(reason, str) and (
        reason.startswith("unmeasured") or reason == "partly unmeasured"
    )


def effort_unmeasured(results: Mapping[str, Mapping[str, Any]]) -> int:
    """The trials that asked for an effort nothing measured."""
    return sum(
        isinstance(result.get("effort"), dict)
        and bool(result["effort"].get("requested"))
        and _unmeasured(result["effort"].get("reason"))
        for result in results.values()
    )
