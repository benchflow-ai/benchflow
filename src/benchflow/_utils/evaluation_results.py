"""Helpers for serializing evaluation rollout results."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchflow._utils.benchmark_repos import task_source_provenance
from benchflow._utils.result_metadata import (
    final_metrics_from_rollout,
    incomplete_usage_fields,
    trajectory_summary_from_events,
)
from benchflow._utils.reward_events import (
    memory_score_from_events,
    reward_event_to_dict,
)
from benchflow._utils.source_provenance import artifact_source_provenance
from benchflow.models import RolloutResult
from benchflow.trajectories.metrics import (
    count_skill_invocations,
    result_skill_invocations,
)
from benchflow.trajectories.types import redact_trajectory_text
from benchflow.usage_tracking import (
    NO_MODEL_CALL,
    is_trusted_usage_source,
    result_usage_source,
)

# Phase keys produced by Rollout (see rollout.py — environment_setup,
# agent_setup, agent_execution, verifier, total). Kept here so summary
# aggregation stays in lockstep with the rollout writer.
_TIMING_PHASES: tuple[str, ...] = (
    "environment_setup",
    "agent_setup",
    "agent_execution",
    "verifier",
    "total",
)


def agent_result_from_rollout(result: RolloutResult) -> dict[str, Any]:
    """Return the serialized agent_result block for an in-memory rollout result."""
    n_skill_invocations = result.n_skill_invocations or count_skill_invocations(
        result.trajectory
    )
    agent_result = {
        "n_tool_calls": result.n_tool_calls,
        "n_skill_invocations": n_skill_invocations,
        "n_prompts": result.n_prompts,
        "n_input_tokens": result.n_input_tokens,
        "n_output_tokens": result.n_output_tokens,
        "n_cache_read_tokens": result.n_cache_read_tokens,
        "n_cache_write_tokens": getattr(result, "n_cache_write_tokens", None),
        "n_cache_write_5m_tokens": getattr(result, "n_cache_write_5m_tokens", None),
        "n_cache_write_1h_tokens": getattr(result, "n_cache_write_1h_tokens", None),
        "n_cache_creation_tokens": result.n_cache_creation_tokens,
        "total_tokens": result.total_tokens,
        "cost_usd": result.cost_usd,
        "usage_source": result.usage_source,
        "cost_source": getattr(result, "cost_source", None),
        "price_table_date": getattr(result, "price_table_date", None),
        "price_source": result.price_source,
        "usage_source_legacy": getattr(result, "usage_source_legacy", None),
        "no_usage_reason": getattr(result, "no_usage_reason", None),
        "cost_usd_provider": getattr(result, "cost_usd_provider", None),
        "n_calls_cost_estimated": getattr(result, "n_calls_cost_estimated", None),
        **incomplete_usage_fields(result),
    }
    if getattr(result, "usage_details", None) is not None:
        agent_result["usage_details"] = result.usage_details
    return agent_result


def _masked(text: str | None) -> str | None:
    """Error text as result.json keeps it: secret shapes masked."""
    return redact_trajectory_text(text) if isinstance(text, str) else text


def rollout_result_payload(
    result: RolloutResult,
    *,
    source_provenance: dict[str, Any] | None,
    tasks_dir: Path,
    task_name: str,
) -> dict[str, Any]:
    """Normalize an in-memory rollout result to the persisted result shape."""
    reward_events = result.reward_events or []
    memory_score = memory_score_from_events(reward_events)
    task_source = result.source_provenance or task_source_provenance(
        source_provenance, tasks_dir / task_name
    )
    n_skill_invocations = result.n_skill_invocations or count_skill_invocations(
        result.trajectory
    )
    return {
        "task_name": result.task_name,
        "rollout_name": result.rollout_name,
        "rewards": result.rewards,
        **({"scoring": result.scoring.to_dict()} if result.scoring is not None else {}),
        **(
            {"purpose": result.purpose, "parent_rollout": result.parent_rollout}
            if result.purpose != "task"
            else {}
        ),
        # Masked as the trial's own result.json masks them: this payload is
        # the result.json of a trial that left no folder, and summary rows.
        "error": _masked(result.error),
        "error_category": result.error_category,
        "verifier_error": _masked(result.verifier_error),
        "verifier_error_category": result.verifier_error_category,
        "export_error": _masked(result.export_error),
        # How the trial ended when its model refused or its budget stopped it
        # (benchflow.scoring counts both beside the score).
        "model_refusal": result.model_refusal,
        "trial_budget": result.trial_budget,
        "n_tool_calls": result.n_tool_calls,
        "n_skill_invocations": n_skill_invocations,
        "agent_result": agent_result_from_rollout(result),
        "judge_usage": getattr(result, "judge_usage", None),
        "final_metrics": final_metrics_from_rollout(result),
        "trajectory_summary": trajectory_summary_from_events(
            result.trajectory,
            partial_trajectory=result.partial_trajectory,
            trajectory_source=result.trajectory_source,
        ),
        **(
            {"reward_events": [reward_event_to_dict(event) for event in reward_events]}
            if reward_events
            else {}
        ),
        **({"memory_score": memory_score} if memory_score is not None else {}),
        **({"source": artifact_source_provenance(task_source)} if task_source else {}),
    }


def usage_summary(results: dict[str, dict]) -> dict[str, Any]:
    """Aggregate provider telemetry fields for summary.json.

    Totals cover every trial whose tokens a trusted source counted, errored
    ones included: a trial that timed out or lost its sandbox still
    spent what it spent. A trial nothing counted adds nothing and is not a
    0: with no counted trial, a total is None. An agent that routes its own
    model calls reports usage nobody metered (``usage_source_legacy``
    ``agent-reported``); it is counted in ``usage_sources`` but left out of
    the totals, as before. ``usage_sources`` and ``cost_sources`` count the
    trials by ``usage_source`` and ``cost_source``;
    ``trials_without_usage`` and ``trials_without_cost`` count the rest,
    and ``trials_without_model_call`` the trials whose agent made no model
    call (an oracle, a nop: ``no_usage_reason`` ``no_model_call``), which
    are in no other count. ``trials_usage_incomplete`` counts the trials
    whose tokens are those of some of their calls only (``usage_incomplete``),
    and ``trials_cost_incomplete`` those whose cost was priced from such a
    count: the totals include what they counted and are short by the rest.
    """

    def agent(r: dict) -> dict:
        return r.get("agent_result") or {}

    # A trial whose agent made no model call (an oracle, a nop) has nothing
    # to count: counted apart, never as a trial with 0 tokens.
    no_call = [
        r for r in results.values() if agent(r).get("no_usage_reason") == NO_MODEL_CALL
    ]
    trials = [
        r for r in results.values() if agent(r).get("no_usage_reason") != NO_MODEL_CALL
    ]

    covered = [
        r
        for r in trials
        if is_trusted_usage_source(
            agent(r).get("usage_source"), agent(r).get("usage_source_legacy")
        )
    ]

    def total(field: str) -> int | None:
        values = [
            value
            for r in covered
            if isinstance(value := agent(r).get(field), int)
            and not isinstance(value, bool)
        ]
        return sum(values) if values else None

    # A missing cost is unknown, not zero.
    costs = [
        cost
        for r in covered
        if isinstance(cost := agent(r).get("cost_usd"), int | float)
        and not isinstance(cost, bool)
    ]
    total_cost = round(sum(costs), 10) if costs else None
    billed = [
        value
        for r in covered
        if isinstance(value := agent(r).get("cost_usd_provider"), int | float)
        and not isinstance(value, bool)
    ]
    estimated_calls = [
        value
        for r in covered
        if isinstance(value := agent(r).get("n_calls_cost_estimated"), int)
        and not isinstance(value, bool)
    ]
    usage_sources: dict[str, int] = {}
    cost_sources: dict[str, int] = {}
    for r in trials:
        source = result_usage_source(agent(r).get("usage_source"))
        if source is not None:
            usage_sources[source] = usage_sources.get(source, 0) + 1
        cost_source = agent(r).get("cost_source")
        if agent(r).get("cost_usd") is not None and isinstance(cost_source, str):
            cost_sources[cost_source] = cost_sources.get(cost_source, 0) + 1
    return {
        "total_input_tokens": total("n_input_tokens"),
        "total_output_tokens": total("n_output_tokens"),
        "total_cache_read_tokens": total("n_cache_read_tokens"),
        "total_cache_write_tokens": total("n_cache_write_tokens")
        if any("n_cache_write_tokens" in agent(r) for r in covered)
        else total("n_cache_creation_tokens"),
        "total_cache_write_5m_tokens": total("n_cache_write_5m_tokens"),
        "total_cache_write_1h_tokens": total("n_cache_write_1h_tokens"),
        # Alias of total_cache_write_tokens for one release.
        "total_cache_creation_tokens": total("n_cache_creation_tokens"),
        "total_tokens": total("total_tokens"),
        "total_cost_usd": total_cost,
        # The provider-billed part of total_cost_usd, and the calls priced at
        # list price for want of a bill, over the trials that say.
        "total_cost_usd_provider": round(sum(billed), 10) if billed else None,
        "n_calls_cost_estimated": sum(estimated_calls) if estimated_calls else None,
        "avg_cost_per_trial_usd": (
            round(total_cost / len(costs), 10) if total_cost is not None else None
        ),
        "telemetry_coverage": (len(covered) / len(trials) if trials else 0.0),
        "usage_sources": dict(sorted(usage_sources.items())),
        "cost_sources": dict(sorted(cost_sources.items())),
        "trials_without_usage": sum(
            1
            for r in trials
            if result_usage_source(agent(r).get("usage_source")) is None
        ),
        "trials_without_cost": sum(
            1 for r in trials if agent(r).get("cost_usd") is None
        ),
        "trials_without_model_call": len(no_call),
        "trials_usage_incomplete": sum(
            1 for r in trials if agent(r).get("usage_incomplete") is True
        ),
        "trials_cost_incomplete": sum(
            1
            for r in trials
            if agent(r).get("cost_incomplete") is True
            and agent(r).get("cost_usd") is not None
        ),
    }


#: The token totals of summary.json's ``judge_totals``, each from the
#: field of a trial's ``judge_usage`` (BenchFlow's split).
_JUDGE_TOKEN_TOTALS = (
    ("total_input_tokens", "n_input_tokens"),
    ("total_output_tokens", "n_output_tokens"),
    ("total_cache_read_tokens", "n_cache_read_tokens"),
    ("total_cache_write_tokens", "n_cache_write_tokens"),
    ("total_cache_write_5m_tokens", "n_cache_write_5m_tokens"),
    ("total_cache_write_1h_tokens", "n_cache_write_1h_tokens"),
    ("total_tokens", "total_tokens"),
)


def _known_sum(values: Iterable[Any]) -> int | float | None:
    """The sum of the known numbers; None when none is known (never 0)."""
    known = [
        v for v in values if isinstance(v, int | float) and not isinstance(v, bool)
    ]
    return sum(known) if known else None


def _round_usd(value: int | float | None) -> float | None:
    return round(value, 10) if value is not None else None


def judge_usage_summary(results: dict[str, dict]) -> dict[str, Any]:
    """summary.json's ``judge_totals``: what the verifiers' LLM judges spent
    over the trials :func:`usage_summary` counts (each task's final
    attempt), from each trial's ``judge_usage``, apart from the agents'.

    ``trials`` counts the trials whose verifier called a judge;
    ``trials_without_usage`` and ``trials_without_cost`` those whose
    judge's tokens or cost are unknown (a total leaves them out, and is
    None when no judged trial knows it).
    """
    blocks = [
        block
        for r in results.values()
        if isinstance(block := r.get("judge_usage"), dict)
    ]
    usage_sources: dict[str, int] = {}
    cost_sources: dict[str, int] = {}
    for block in blocks:
        for key, counts in (
            ("usage_source", usage_sources),
            ("cost_source", cost_sources),
        ):
            if isinstance(source := block.get(key), str):
                counts[source] = counts.get(source, 0) + 1
    return {
        "judge_totals": {
            "trials": len(blocks),
            "calls": sum(c for b in blocks if isinstance(c := b.get("calls"), int)),
            **{
                total: _known_sum(block.get(field) for block in blocks)
                for total, field in _JUDGE_TOKEN_TOTALS
            },
            "total_cost_usd": _round_usd(_known_sum(b.get("cost_usd") for b in blocks)),
            "usage_sources": dict(sorted(usage_sources.items())),
            "cost_sources": dict(sorted(cost_sources.items())),
            "trials_without_usage": sum(
                1 for block in blocks if block.get("total_tokens") is None
            ),
            "trials_without_cost": sum(
                1 for block in blocks if block.get("cost_usd") is None
            ),
        }
    }


def kept_usage_rows(
    kept: dict[str, dict[str, Any]], stopped: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """The trials summary.json's ``total_*`` and ``judge_totals`` count: each task's kept attempt (``kept``) and, for a task with none,
    the latest record of an attempt stopped before it finished (``stopped``,
    from :func:`benchflow._utils.result_paths.interrupted_trial_results`),
    which still spent. A stopped attempt of a task kept since counts in
    :func:`attempt_spend_summary` only. ``bench eval metrics`` takes its
    Cost row by the same rule, so the two agree.
    """
    tasks = {row.get("task_name") for row in kept.values()}
    latest: dict[Any, tuple[float, str]] = {}
    for key, row in stopped.items():
        task = row.get("task_name")
        if task in tasks:
            continue
        rank = (_instant(row.get("finished_at")), key)
        if task not in latest or rank > latest[task]:
            latest[task] = rank
    return {**kept, **{key: stopped[key] for _, key in latest.values()}}


def _instant(value: Any) -> float:
    """A result's time as seconds since the epoch: result.json writes local
    time with its UTC offset, so two offsets compare by the instant, never
    as text. A naive time is local; an unreadable one sorts first."""
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(value)).astimezone().timestamp()
    except (TypeError, ValueError):
        return float("-inf")


def attempt_spend_summary(
    attempts: Iterable[dict], *, unrecorded: int = 0
) -> dict[str, Any]:
    """summary.json's ``spend_all_attempts``: what the job spent over every
    attempt it recorded, retries and resumed runs included, its agents' and
    judges' together.

    The ``total_*`` keys of :func:`usage_summary` count each task's final
    attempt (the trial the score reads); this counts every attempt in the
    job folder (``attempts`` from ``evaluation._recorded_attempts``) by the
    rule the budget's ``spent`` uses (:class:`benchflow.budget.AttemptSpend`),
    so the two agree. ``cost_usd`` and ``tokens`` are None when no attempt's
    is known; ``judge_cost_usd`` and ``judge_tokens`` are the judges' part
    of them. ``attempts_without_cost`` and ``attempts_without_tokens`` count
    the attempts whose figure is unknown, and ``attempts_without_model_call``
    those that called no model (an oracle, a nop, a script), which are in no
    other count. ``stopped_attempts`` were stopped before they finished
    (Ctrl-C, SIGTERM, a budget): what their receipt recorded counts,
    ``stopped_attempts_without_usage`` recorded nothing and
    ``stopped_attempts_without_cost`` recorded tokens with no cost.
    ``attempts_cost_incomplete`` counts the attempts whose cost was priced
    from some of their calls only (``cost_incomplete``), so ``cost_usd`` is a
    floor when it is not 0.

    An attempt that ended with no result and no receipt (its run stopped
    mid-turn: a host reset or a killed process) is a stopped attempt too: counted from the logs it left
    (``attempts_usage_salvaged``, a floor, so also in
    ``attempts_cost_incomplete``), or, with none, one of ``unrecorded``
    (``evaluation._recorded_attempts``' count), in ``stopped_attempts`` and
    ``stopped_attempts_without_usage`` and never left out.
    """
    from benchflow._utils.attempt_salvage import SALVAGED_KEY
    from benchflow._utils.result_paths import is_interrupted_result
    from benchflow.budget import AttemptSpend

    attempts = list(attempts)
    unrecorded = max(0, unrecorded)

    def stopped_record(row: dict) -> bool:
        # A cancellation receipt, or the stopped trial's own result.json
        # (a trial folder read alone).
        return bool(row.get("_budget_cancelled")) or is_interrupted_result(row)

    finished = [AttemptSpend.of(a) for a in attempts if not stopped_record(a)]
    stopped = [AttemptSpend.of(a) for a in attempts if stopped_record(a)]
    spends = finished + stopped
    no_call = [s for s in finished if s.called_no_model]
    called = [s for s in finished if not s.called_no_model]
    return {
        "spend_all_attempts": {
            "attempts": len(finished),
            "stopped_attempts": len(stopped) + unrecorded,
            "cost_usd": _round_usd(_known_sum(s.total_cost_usd for s in spends)),
            "tokens": _known_sum(s.total_tokens for s in spends),
            "judge_cost_usd": _round_usd(_known_sum(s.judge_cost_usd for s in spends)),
            "judge_tokens": _known_sum(s.judge_tokens for s in spends),
            "attempts_without_cost": sum(1 for s in called if s.usd_unknown),
            "attempts_without_tokens": sum(1 for s in called if s.tokens is None),
            "attempts_without_model_call": len(no_call),
            "stopped_attempts_without_usage": unrecorded
            + sum(1 for s in stopped if s.tokens is None and s.cost_usd is None),
            "attempts_usage_salvaged": sum(
                1 for a in attempts if stopped_record(a) and a.get(SALVAGED_KEY)
            ),
            # Its tokens counted, its cost not (a model no price table has).
            "stopped_attempts_without_cost": sum(
                1
                for s in stopped
                if s.cost_usd is None and s.tokens is not None and not s.called_no_model
            ),
            "attempts_cost_incomplete": sum(1 for s in spends if s.cost_incomplete),
        }
    }


#: summary.json's usage keys: :func:`usage_summary`'s, then
#: :func:`judge_usage_summary`'s and :func:`attempt_spend_summary`'s. A
#: sharded run's summary adds each up over the shards
#: (:func:`merge_usage_summaries`).
AGENT_USAGE_KEYS = (
    "total_input_tokens",
    "total_output_tokens",
    "total_cache_read_tokens",
    "total_cache_write_tokens",
    "total_cache_write_5m_tokens",
    "total_cache_write_1h_tokens",
    "total_cache_creation_tokens",
    "total_tokens",
    "total_cost_usd",
    "total_cost_usd_provider",
    "n_calls_cost_estimated",
    "avg_cost_per_trial_usd",
    "telemetry_coverage",
    "usage_sources",
    "cost_sources",
    "trials_without_usage",
    "trials_without_cost",
    "trials_without_model_call",
    "trials_usage_incomplete",
    "trials_cost_incomplete",
)
USAGE_SUMMARY_KEYS = (*AGENT_USAGE_KEYS, "judge_totals", "spend_all_attempts")
#: Keys :func:`merge_usage_summaries` derives instead of adding up.
_DERIVED = ("avg_cost_per_trial_usd", "telemetry_coverage")
_COUNTS = (
    "trials_without_usage",
    "trials_without_cost",
    "trials_without_model_call",
    "trials_usage_incomplete",
    "trials_cost_incomplete",
)


def _sum_counts(values: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for counts in values:
        if isinstance(counts, dict):
            for key, n in counts.items():
                if isinstance(n, int) and not isinstance(n, bool):
                    out[key] = out.get(key, 0) + n
    return dict(sorted(out.items()))


def _merge_block(
    blocks: Iterable[Any], keys: Iterable[str] | None = None
) -> dict[str, Any]:
    """Merge usage blocks: counts by name are added, numbers summed over
    the blocks that know them (None when none does), USD rounded."""
    blocks = [b for b in blocks if isinstance(b, dict)]
    names = keys if keys is not None else dict.fromkeys(k for b in blocks for k in b)
    merged: dict[str, Any] = {}
    for key in names:
        values = [b.get(key) for b in blocks]
        if any(isinstance(v, dict) for v in values):
            merged[key] = _sum_counts(values)
        else:
            total = _known_sum(values)
            merged[key] = _round_usd(total) if isinstance(total, float) else total
    return merged


def merge_usage_summaries(shards: list[dict[str, Any]]) -> dict[str, Any]:
    """The usage keys of a sharded job's summary.json (``--worker-concurrency``),
    from each shard's own (:data:`USAGE_SUMMARY_KEYS`).

    A total adds the shards that know it and stays None when none does (a
    shard's None is unknown, never 0); counts by source add up. A shard's
    trials are its ``usage_sources`` and ``trials_without_usage``; its
    priced trials are its total cost over its mean; so the coverage and the
    mean cost per trial are the unsharded job's.
    """
    # A shard whose summary could not be read passed no usage: every total
    # leaves it out, and says so (shards_without_usage), as a spend no shard
    # knows is null and one some shard lacks says parts_unknown.
    unread = sum(1 for usage in shards if not usage)
    merged = _merge_block(shards, [k for k in AGENT_USAGE_KEYS if k not in _DERIVED])
    trials = covered = priced = 0.0
    for usage in shards:
        n = sum((usage.get("usage_sources") or {}).values()) + (
            usage.get("trials_without_usage") or 0
        )
        trials += n
        coverage = usage.get("telemetry_coverage")
        covered += n * coverage if isinstance(coverage, int | float) else 0
        cost, mean = usage.get("total_cost_usd"), usage.get("avg_cost_per_trial_usd")
        if isinstance(cost, int | float) and isinstance(mean, int | float) and mean:
            priced += round(cost / mean)
        elif isinstance(cost, int | float):
            # Every priced trial cost 0: count them as the trials with a cost.
            priced += n - (usage.get("trials_without_cost") or 0)
    cost = merged["total_cost_usd"]
    for key in _COUNTS:
        merged[key] = merged[key] or 0
    merged["usage_sources"] = merged["usage_sources"] or {}
    merged["cost_sources"] = merged["cost_sources"] or {}
    merged["avg_cost_per_trial_usd"] = (
        round(cost / priced, 10) if cost is not None and priced else None
    )
    merged["telemetry_coverage"] = covered / trials if trials else 0.0
    return {
        **{key: merged[key] for key in AGENT_USAGE_KEYS},
        **({"shards_without_usage": unread} if unread else {}),
        "judge_totals": _merge_block(u.get("judge_totals") for u in shards),
        "spend_all_attempts": merge_spends(u.get("spend_all_attempts") for u in shards),
    }


def merge_spends(blocks: Iterable[Any]) -> dict[str, Any] | None:
    """``spend_all_attempts`` blocks of several jobs or shards as one: None
    when none is known, and ``parts_unknown`` counts the parts whose spend
    is unknown (a job folder whose records could not be read, a shard whose
    summary has none), so a sum with an unknown part says so."""
    blocks = list(blocks)
    known = [block for block in blocks if isinstance(block, dict)]
    if not known:
        return None
    merged = _merge_block(known)
    if len(known) < len(blocks):
        merged["parts_unknown"] = len(blocks) - len(known)
    return merged


def _unmeasured(reason: object) -> bool:
    """A reason the record gives when nothing measured the effort: ``unmeasured``,
    ``unmeasured: no call carried an effort`` or ``partly unmeasured``."""
    return isinstance(reason, str) and (
        reason.startswith("unmeasured") or reason == "partly unmeasured"
    )


def model_call_summary(results: dict[str, dict]) -> dict[str, Any]:
    """The ``model_calls`` block of summary.json, from each trial's own record
    of its model calls (its result.json, ``model-calls.json`` behind it).

    ``tool_surfaces`` lists each set of tools the agents' first requests
    offered, by hash, with how many trials had it (a job whose trials ran on
    two surfaces shows two) and ``fallback`` from the trials'
    ``tool_surface``; ``fallback_tool_trials`` counts the trials on an agent's
    fallback surface; ``effort_mismatches`` the trials whose recorded effort
    differs from the one asked for (``effort.matches`` false).

    From each trial's ``agent_outcome``: ``http_errors`` and
    ``http_status_counts`` total the calls by status over the trials that
    recorded one (``trials_with_status``), the failures the agents survived
    among them; ``trials_with_http_errors`` counts the trials with any;
    ``agent_retries`` the failed calls an agent followed with another,
    ``upstream_retries`` the proxy's own extra attempts upstream, and
    ``terminal_provider_errors`` the trials whose agent's last call failed
    with 429 or a 5xx. From ``input_delivery``: ``input_reads`` the agents'
    reads of files they had not written, ``input_stubs`` the files whose read
    reached the model as a stub and ``input_stub_trials`` the trials with
    one. Empty when no trial recorded any of it.
    """
    surfaces: dict[str, dict[str, Any]] = {}
    fallback = mismatches = unmeasured = 0
    statuses: dict[str, int] = {}
    http = {
        "trials_with_status": 0,
        "http_errors": 0,
        "trials_with_http_errors": 0,
        "agent_retries": 0,
        "upstream_retries": 0,
        "terminal_provider_errors": 0,
    }
    inputs = {"input_reads": 0, "input_stubs": 0, "input_stub_trials": 0}
    for result in results.values():
        delivery = result.get("input_delivery")
        if isinstance(delivery, dict) and isinstance(delivery.get("reads"), int):
            inputs["input_reads"] += delivery["reads"]
            stubs = (
                delivery.get("stubs") if isinstance(delivery.get("stubs"), int) else 0
            )
            inputs["input_stubs"] += stubs
            inputs["input_stub_trials"] += stubs > 0
        outcome = result.get("agent_outcome")
        if isinstance(outcome, dict) and isinstance(outcome.get("http_errors"), int):
            http["trials_with_status"] += 1
            http["http_errors"] += outcome["http_errors"]
            http["trials_with_http_errors"] += outcome["http_errors"] > 0
            for key in ("agent_retries", "upstream_retries"):
                if isinstance(outcome.get(key), int):
                    http[key] += outcome[key]
            http["terminal_provider_errors"] += (
                outcome.get("terminal_provider_error") is True
            )
            counts = outcome.get("http_status_counts")
            for status, count in counts.items() if isinstance(counts, dict) else ():
                if isinstance(count, int):
                    statuses[str(status)] = statuses.get(str(status), 0) + count
        surface = result.get("tool_surface")
        if isinstance(surface, dict) and isinstance(surface.get("sha256"), str):
            entry = surfaces.setdefault(
                surface["sha256"],
                {
                    "sha256": surface["sha256"],
                    "count": surface.get("count"),
                    "names": surface.get("names"),
                    "fallback": surface.get("fallback"),
                    "trials": 0,
                },
            )
            entry["trials"] += 1
            fallback += surface.get("fallback") is True
        effort = result.get("effort")
        mismatches += isinstance(effort, dict) and effort.get("matches") is False
        # Beside them, the trials that asked for an effort nothing measured
        # .
        unmeasured += (
            isinstance(effort, dict)
            and bool(effort.get("requested"))
            and _unmeasured(effort.get("reason"))
        )
    if (
        not surfaces
        and not mismatches
        and not unmeasured
        and not http["trials_with_status"]
        and not inputs["input_reads"]
    ):
        return {}
    return {
        "model_calls": {
            **http,
            **inputs,
            "http_status_counts": dict(sorted(statuses.items())),
            "tool_surfaces": sorted(
                surfaces.values(), key=lambda entry: -int(entry["trials"])
            ),
            "fallback_tool_trials": fallback,
            "effort_mismatches": mismatches,
            "effort_unmeasured": unmeasured,
        }
    }


def solve_rate_summary(results: dict[str, dict]) -> dict[str, Any]:
    """The ``solve_rates`` block of summary.json (``benchflow.pass_at_k``).

    One Evaluation holds one result per task, so ``n = 1`` and only pass@1
    is defined here; pool repeated trials with ``bench eval metrics`` over
    the trial folders, ``bf.load_job(...).solve_rates()`` or the matrix
    summary. Every result is a trial: an errored / verifier-errored one is
    unscored and counts as not solved, and the ``*_scored_only`` figures
    leave it out of ``n``; the pass verdict is the shared score classifier's.
    """
    from benchflow._utils.scoring import classify_score_outcome, extract_reward
    from benchflow.pass_at_k import Sample, solve_rates

    samples = []
    for name, r in results.items():
        outcome = classify_score_outcome(r)
        scored = outcome in ("passed", "failed")
        reward = extract_reward(r) if scored else None
        if scored and reward is None:  # a gate verdict without a reward value
            reward = 1.0 if outcome == "passed" else 0.0
        samples.append(
            Sample(
                str(r.get("task_name") or name),
                reward,
                outcome == "passed" if scored else None,
            )
        )
    return {"solve_rates": solve_rates(samples).to_dict()}


def loop_summary(results: dict[str, dict]) -> dict[str, Any]:
    """Aggregate per-rollout ``loop`` blocks into a job-level convergence report.

    The headline loopbench artifact: the **pass@iteration curve** (cumulative
    fraction of tasks passed by each loop iteration) plus convergence rates and
    iteration economics. Returns ``{}`` when no result carries a real
    (non-single-shot) loop strategy, so ordinary jobs never gain empty keys.
    """
    looped = [
        loop
        for r in results.values()
        if isinstance((loop := r.get("loop")), dict)
        and loop.get("strategy") not in (None, "single-shot")
    ]
    if not looped:
        return {}

    n = len(looped)
    first_pass = [
        loop["first_pass_iteration"]
        for loop in looped
        if loop.get("first_pass_iteration") is not None
    ]
    iters_run = [loop.get("iterations_run") or 0 for loop in looped]
    max_iter = max(
        (len(loop.get("reward_trajectory") or []) for loop in looped), default=0
    )
    # pass@iteration: cumulative fraction of tasks passed BY iteration i.
    pass_at_iteration = [
        sum(
            1
            for loop in looped
            if loop.get("first_pass_iteration") is not None
            and loop["first_pass_iteration"] <= i
        )
        / n
        for i in range(max_iter)
    ]
    stop_reasons: dict[str, int] = {}
    for loop in looped:
        sr = loop.get("stop_reason") or "unknown"
        stop_reasons[sr] = stop_reasons.get(sr, 0) + 1
    # Cost-to-converge economics (the cost-curve money axis), over the converged
    # tasks that actually captured token usage. None when nothing converged with
    # usage data (e.g. a LiteLLM path that doesn't surface native tokens).
    tokens_to_pass = [
        loop["tokens_to_pass"]
        for loop in looped
        if loop.get("tokens_to_pass") is not None
    ]
    mean_tokens_to_converge = (
        round(sum(tokens_to_pass) / len(tokens_to_pass), 1) if tokens_to_pass else None
    )

    return {
        "loop_summary": {
            "strategy": looped[0].get("strategy"),
            "n_tasks": n,
            "fraction_converged": len(first_pass) / n,
            "mean_iterations_to_converge": (
                round(sum(first_pass) / len(first_pass), 4) if first_pass else None
            ),
            "mean_iterations_run": round(sum(iters_run) / n, 4),
            "mean_tokens_to_converge": mean_tokens_to_converge,
            "pass_at_iteration": [round(p, 4) for p in pass_at_iteration],
            "stop_reasons": stop_reasons,
        }
    }


def trajectory_step_summary(results: dict[str, dict]) -> dict[str, Any]:
    """Aggregate trajectory step counts for summary.json."""
    summaries: list[dict[str, Any]] = []
    for result in results.values():
        summary = result.get("trajectory_summary")
        if isinstance(summary, dict):
            summaries.append(summary)
    step_counts = [int(s.get("steps") or 0) for s in summaries]
    tool_step_counts = [int(s.get("tool_call_steps") or 0) for s in summaries]
    total_steps = sum(step_counts)
    total_tool_steps = sum(tool_step_counts)

    return {
        "total_trajectory_steps": total_steps,
        "avg_trajectory_steps_per_task": (
            total_steps / len(step_counts) if step_counts else 0.0
        ),
        "max_trajectory_steps_per_task": max(step_counts) if step_counts else 0,
        "total_trajectory_tool_call_steps": total_tool_steps,
        "avg_trajectory_tool_call_steps_per_task": (
            total_tool_steps / len(tool_step_counts) if tool_step_counts else 0.0
        ),
        "max_trajectory_tool_call_steps_per_task": (
            max(tool_step_counts) if tool_step_counts else 0
        ),
        "trajectory_summary_coverage": (
            len(summaries) / len(results) if results else 0.0
        ),
    }


def skill_invocation_summary(results: dict[str, dict]) -> dict[str, Any]:
    """Aggregate structured skill invocation counts for summary.json."""
    total = sum(result_skill_invocations(result) for result in results.values())
    return {
        "total_skill_invocations": total,
        "avg_skill_invocations": (round(total / len(results), 1) if results else 0.0),
    }


def _round_secs(value: float) -> float:
    """Round a duration in seconds to the same precision rollout.py uses."""
    return round(float(value), 1)


def tool_call_summary(results: dict[str, dict]) -> dict[str, Any]:
    """Aggregate per-rollout ``n_tool_calls`` across every result in the job.

    Unlike ``usage_summary``, this counts EVERY rollout — including errored
    and verifier-errored ones — because tool-call cost is paid regardless of
    whether verification succeeded. Reviewers asking "how many tool calls did
    this job consume?" want the literal sum, not a success-filtered one.
    """
    counts = [int(r.get("n_tool_calls") or 0) for r in results.values()]
    total = sum(counts)
    return {
        "total_tool_calls": total,
        "avg_tool_calls_per_task": (total / len(counts)) if counts else 0.0,
        "max_tool_calls_per_task": max(counts) if counts else 0,
    }


def phase_timing_summary(results: dict[str, dict]) -> dict[str, Any]:
    """Aggregate per-phase wall-clock timing across every rollout.

    Sums and averages cover any rollout that recorded a ``timing`` block, so
    reviewers can answer "how much time went to agent vs verifier?" without
    inspecting each ``result.json``. Phase keys follow ``rollout.py``:
    ``environment_setup``, ``agent_setup``, ``agent_execution``, ``verifier``,
    ``total``. The ``timing_coverage`` ratio surfaces when phase data is
    incomplete (e.g. mocked test runs that don't persist ``timing``).
    """
    timings: list[dict[str, float]] = []
    for r in results.values():
        t = r.get("timing")
        if isinstance(t, dict) and t:
            timings.append(t)

    out: dict[str, Any] = {
        "timing_coverage": (len(timings) / len(results)) if results else 0.0,
    }
    for phase in _TIMING_PHASES:
        values = [
            float(t[phase]) for t in timings if isinstance(t.get(phase), (int, float))
        ]
        # Phases with no data get a 0.0 sum + null avg/max so downstream
        # readers can distinguish "ran but cost nothing" from "no data".
        out[f"{phase}_time_sec"] = _round_secs(sum(values)) if values else 0.0
        out[f"avg_{phase}_time_sec"] = (
            _round_secs(sum(values) / len(values)) if values else None
        )
        out[f"max_{phase}_time_sec"] = _round_secs(max(values)) if values else None
    return out
