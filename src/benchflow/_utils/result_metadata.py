"""Shared result metadata derived from usage telemetry and ACP trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: The token split and its provenance, as result.json's ``agent_result``
#: names them; ATIF's ``final_metrics.extra`` carries the same keys.
TOKEN_SPLIT_FIELDS = (
    "n_input_tokens",
    "n_cache_read_tokens",
    "n_cache_write_tokens",
    "n_cache_write_5m_tokens",
    "n_cache_write_1h_tokens",
    "n_output_tokens",
)
PROVENANCE_FIELDS = ("usage_source", "cost_source", "price_table_date")


#: What a count from Codex's session log covers. Codex's WebSocket
#: transport can first send a warm-up (``response.create`` with
#: ``generate: false``) that the provider answers with a usage of its own; the
#: session log records generating responses only, so the warm-up is never in
#: the count. result.json's ``usage_details.token_count.scope``.
CODEX_GENERATING_SCOPE = (
    "generating calls only; Codex's generate=false warm-up is recorded "
    "separately in usage_details.warmup"
)
#: summary.json's ``usage_scope.scope``, beside the trials it covers.
SUMMARY_GENERATING_SCOPE = (
    "token totals count generating calls only; Codex's generate=false warm-up "
    "is recorded per trial in usage_details.warmup (tokens not observed, cost "
    "unknown)"
)
#: ``cost_incomplete_reason`` of a trial whose warm-up is unpriced.
COST_INCOMPLETE_WARMUP = (
    "Codex's generate=false warm-up is unpriced: its tokens are not observed "
    "and its billing is not established"
)


def codex_warmup() -> dict[str, Any]:
    """``usage_details.warmup`` of a trial counted from Codex's session log: what the warm-up is, and that its calls, tokens and cost are
    unknown (None, never 0), with why. Whether the provider bills a
    ``generate: false`` request is not established: OpenAI's WebSocket-mode
    guide does not say."""
    return {
        "request": "response.create generate=false",
        "calls": None,
        "tokens": None,
        "cost_usd": None,
        "reason": (
            "Codex's session log records generating responses only; warm-up "
            "usage is not observed"
        ),
        "billing": (
            "not established: OpenAI's WebSocket-mode guide does not say "
            "whether generate=false requests are billed"
        ),
    }


def unpriced_warmup(usage_details: Any) -> bool:
    """Whether ``usage_details`` records a warm-up whose cost is unknown."""
    warmup = usage_details.get("warmup") if isinstance(usage_details, Mapping) else None
    return isinstance(warmup, Mapping) and warmup.get("cost_usd") is None


#: ``cost_incomplete_reason`` of a cost that leaves out cache writes no answer
#: reported, on a route whose list price has a cache-write rate.
COST_INCOMPLETE_CACHE_WRITES = (
    "the provider's answers named no cache-write count, and the route's list "
    "price has a cache-write rate, so the cost leaves the writes' extra out"
)


def cache_writes_floor(usage_details: Any) -> bool:
    """Whether ``usage_details`` records cache writes that make the cost a
    floor (``cache_writes.cost_floor``)."""
    writes = (
        usage_details.get("cache_writes")
        if isinstance(usage_details, Mapping)
        else None
    )
    return isinstance(writes, Mapping) and writes.get("cost_floor") is True


def incomplete_usage_fields(source: Any) -> dict[str, Any]:
    """``usage_incomplete: true`` and ``usage_incomplete_reason`` when a
    trial's tokens are those of some of its calls only (a turn cut off
    before its report that no log counted, a session log short of a
    reported kind), and ``cost_incomplete: true`` when a cost was priced
    from them: the figures are what was counted, never the whole; and
    ``cost_incomplete`` with ``cost_incomplete_reason`` when the cost leaves
    out a warm-up whose tokens and billing are unknown, or cache
    writes no answer reported on a route that prices them. Present only
    then. *source* is an agent_result (a mapping) or a result object."""

    def get(key: str) -> Any:
        if isinstance(source, Mapping):
            return source.get(key)
        return getattr(source, key, None)

    fields: dict[str, Any] = {}
    if get("usage_incomplete") is True:
        fields = {
            "usage_incomplete": True,
            "usage_incomplete_reason": get("usage_incomplete_reason"),
        }
        if get("cost_usd") is not None:
            fields["cost_incomplete"] = True
    if get("cost_usd") is not None and unpriced_warmup(get("usage_details")):
        # Its tokens count the generating calls, all of them, so they are
        # whole; its cost leaves out a warm-up that may be billed.
        fields["cost_incomplete"] = True
        fields["cost_incomplete_reason"] = COST_INCOMPLETE_WARMUP
    if get("cost_usd") is not None and cache_writes_floor(get("usage_details")):
        # Its tokens are whole but for the writes' count; its cost leaves out
        # what the route bills a write above an input token.
        fields["cost_incomplete"] = True
        fields["cost_incomplete_reason"] = "; ".join(
            reason
            for reason in (
                fields.get("cost_incomplete_reason"),
                COST_INCOMPLETE_CACHE_WRITES,
            )
            if reason
        )
    return fields


def floor_note(partial: int, costed: int) -> str:
    """What is a floor for *partial* trials whose count missed calls,
    *costed* of them with a cost: their tokens, and their cost where one was
    priced (an unknown cost is unknown, not a floor)."""
    if costed >= partial:
        return "their tokens and cost are a floor"
    if not costed:
        return "their tokens are a floor (no cost is known)"
    return f"their tokens are a floor, and so is the cost of {costed} of them"


def incomplete_counts(rows: Any) -> dict[str, int]:
    """``trials_usage_incomplete`` and ``trials_cost_incomplete`` over
    result.json payloads, by summary.json's rule: a trial that
    called no model is in neither; one whose count missed calls
    (``usage_incomplete``) is in the first, and in the second when it has a
    cost, which is then a floor."""
    usage = cost = 0
    for row in rows:
        agent = row.get("agent_result") if isinstance(row, Mapping) else None
        agent = agent if isinstance(agent, Mapping) else {}
        if agent.get("no_usage_reason") == "no_model_call":
            continue
        if agent.get("usage_incomplete") is True:
            usage += 1
            cost += agent.get("cost_usd") is not None
    return {"trials_usage_incomplete": usage, "trials_cost_incomplete": cost}


def _usd_text(value: float) -> str:
    return f"${value:.6f}" if 0 < value < 0.01 else f"${value:.4f}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def spend_note(spend: Any, counted: int) -> str | None:
    """Every attempt's spend (summary.json's ``spend_all_attempts``), for a
    reader whose own figure counts *counted* attempts: None when
    the job ran no other attempt (none stopped, none retried away), so the
    reader's figure already is it, or when the spend is unknown."""
    if not isinstance(spend, Mapping):
        return None
    attempts = spend.get("attempts") or 0
    stopped = spend.get("stopped_attempts") or 0
    if attempts + stopped <= counted:
        return None
    over = [_plural(attempts, "attempt")] if attempts or not stopped else []
    if stopped:
        over.append(_plural(stopped, "stopped attempt"))
    cost, tokens = spend.get("cost_usd"), spend.get("tokens")
    figures = ["USD unknown" if cost is None else _usd_text(cost)]
    if tokens is not None:
        figures.append(f"{tokens:,} tokens")
    note = (
        f"every attempt (spend_all_attempts): {', '.join(figures)} over "
        + " and ".join(over)
    )
    unknown = sum(
        spend.get(key) or 0
        for key in (
            "attempts_without_cost",
            "stopped_attempts_without_cost",
            "stopped_attempts_without_usage",
        )
    )
    floors = []
    partial = spend.get("attempts_cost_incomplete") or 0
    if partial:
        floors.append(f"{_plural(partial, 'attempt')} counted only some of their calls")
    if unknown and cost is not None:
        floors.append(f"{unknown} with no cost")
    if floors:
        note += "; " + " and ".join(floors) + ", so it is a floor"
    return note


def floor_causes(source: Any) -> list[str]:
    """Why a trial's known cost is a floor, one word per cause: ``warmup``
    (an unpriced Codex warm-up), ``cache_writes`` (writes no answer
    reported on a route that prices them) and ``calls`` (a count that missed
    calls). Empty when the cost is whole or unknown."""
    if not cost_is_floor(source):
        return []

    def get(key: str) -> Any:
        if isinstance(source, Mapping):
            return source.get(key)
        return getattr(source, key, None)

    details = get("usage_details")
    causes = []
    if unpriced_warmup(details):
        causes.append("warmup")
    if cache_writes_floor(details):
        causes.append("cache_writes")
    if get("usage_incomplete") is True or not causes:
        causes.append("calls")
    return causes


def cost_is_floor(source: Any) -> bool:
    """Whether a trial's known cost is a floor: priced from some of its calls
    only (``usage_incomplete``), or short of an unpriced warm-up.
    *source* is an agent_result (a mapping) or a result object."""
    return incomplete_usage_fields(source).get("cost_incomplete") is True


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def atif_prompt_tokens(agent_result: Mapping[str, Any]) -> int | None:
    """ATIF's ``total_prompt_tokens``: every prompt token, uncached input plus
    cache reads and writes; None when any of the three is unknown (a sum
    with an unknown part is unknown, not a smaller number).

    An unknown write count from answers that named none still leaves the
    prompt known when each of those answers held its cache in its input
    count (OpenAI's shape, ``usage_details.cache_writes.inside_input``):
    their writes are inside ``n_input_tokens`` already, so the prompt is
    the input, the reads and the writes the other answers named."""
    fresh = _count(agent_result.get("n_input_tokens"))
    read = _count(agent_result.get("n_cache_read_tokens"))
    write = _count(agent_result.get("n_cache_write_tokens"))
    if write is None:
        write = _count(agent_result.get("n_cache_creation_tokens"))
    if write is None:
        details = agent_result.get("usage_details")
        writes = details.get("cache_writes") if isinstance(details, Mapping) else None
        if isinstance(writes, Mapping) and writes.get("inside_input") is True:
            write = _count(writes.get("named_tokens"))
    if fresh is None or read is None or write is None:
        return None
    return fresh + read + write


def atif_metrics_extra(agent_result: Mapping[str, Any]) -> dict[str, Any]:
    """ATIF ``final_metrics.extra``: BenchFlow's token split (uncached input,
    cache reads, cache writes and their 5-minute and 1-hour shares, output)
    and where the counts and cost came from. None for an unknown."""
    from benchflow.usage_tracking import result_usage_source

    extra: dict[str, Any] = {
        field: agent_result.get(field) for field in TOKEN_SPLIT_FIELDS
    }
    if extra["n_cache_write_tokens"] is None:
        extra["n_cache_write_tokens"] = agent_result.get("n_cache_creation_tokens")
    extra["usage_source"] = result_usage_source(agent_result.get("usage_source"))
    extra["cost_source"] = agent_result.get("cost_source")
    extra["price_table_date"] = agent_result.get("price_table_date")
    # A count that missed calls says so here too, as in agent_result.
    extra.update(incomplete_usage_fields(agent_result))
    return extra


def final_metrics_from_agent_result(
    agent_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Return final token/cost metrics in the ATIF ``final_metrics`` shape.

    ATIF's own fields keep ATIF's meaning: ``total_prompt_tokens`` is every
    prompt token (uncached input plus cache reads and writes) and
    ``total_cached_tokens`` the cache reads. BenchFlow's split and the counts'
    provenance go in ``extra`` (:func:`atif_metrics_extra`). Unknown counts
    are None, never 0.
    """
    return {
        "total_prompt_tokens": atif_prompt_tokens(agent_result),
        "total_completion_tokens": agent_result.get("n_output_tokens"),
        "total_cached_tokens": agent_result.get("n_cache_read_tokens"),
        "total_cost_usd": agent_result.get("cost_usd"),
        "extra": atif_metrics_extra(agent_result),
    }


def final_metrics_from_rollout(result: Any) -> dict[str, Any]:
    """Build ATIF-shaped final metrics from a RolloutResult-like object."""
    return final_metrics_from_agent_result(
        {
            "n_input_tokens": getattr(result, "n_input_tokens", None),
            "n_output_tokens": getattr(result, "n_output_tokens", None),
            "n_cache_read_tokens": getattr(result, "n_cache_read_tokens", None),
            "n_cache_creation_tokens": getattr(result, "n_cache_creation_tokens", None),
            "n_cache_write_5m_tokens": getattr(result, "n_cache_write_5m_tokens", None),
            "n_cache_write_1h_tokens": getattr(result, "n_cache_write_1h_tokens", None),
            "cost_usd": getattr(result, "cost_usd", None),
            "usage_source": getattr(result, "usage_source", None),
            "cost_source": getattr(result, "cost_source", None),
            "price_table_date": getattr(result, "price_table_date", None),
        }
    )


def trajectory_summary_from_events(
    trajectory: Sequence[Mapping[str, Any]] | None,
    *,
    partial_trajectory: bool,
    trajectory_source: str | None,
) -> dict[str, Any]:
    """Summarize ACP trajectory steps as event, tool-call and status counts."""
    event_type_counts: dict[str, int] = {}
    tool_call_status_counts: dict[str, int] = {}

    for event in trajectory or ():
        event_type = str(event.get("type") or "unknown")
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        if event_type == "tool_call":
            status = str(event.get("status") or "unknown")
            tool_call_status_counts[status] = tool_call_status_counts.get(status, 0) + 1

    steps = sum(event_type_counts.values())
    tool_call_steps = event_type_counts.get("tool_call", 0)
    known_non_tool_steps = sum(
        event_type_counts.get(event_type, 0)
        for event_type in ("user_message", "agent_message", "agent_thought")
    )

    return {
        "steps": steps,
        "tool_call_steps": tool_call_steps,
        "user_message_steps": event_type_counts.get("user_message", 0),
        "agent_message_steps": event_type_counts.get("agent_message", 0),
        "agent_thought_steps": event_type_counts.get("agent_thought", 0),
        "other_steps": steps - tool_call_steps - known_non_tool_steps,
        "event_type_counts": event_type_counts,
        "tool_call_status_counts": tool_call_status_counts,
        "partial_trajectory": partial_trajectory,
        "trajectory_source": trajectory_source,
    }
