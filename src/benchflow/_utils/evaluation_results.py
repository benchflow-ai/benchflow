"""Helpers for serializing evaluation rollout results."""

from pathlib import Path
from typing import Any

from benchflow._utils.benchmark_repos import task_source_provenance
from benchflow._utils.result_metadata import (
    SUMMARY_GENERATING_SCOPE,
    cost_is_floor,
    final_metrics_from_rollout,
    incomplete_usage_fields,
    trajectory_summary_from_events,
    unpriced_warmup,
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
from benchflow.usage_tracking import is_trusted_usage_source, result_usage_source

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


def _no_usage_reason(result: RolloutResult) -> str | None:
    """``"no_model_call"`` for a control or an operator script."""
    from benchflow.agents.registry import is_scripted_agent

    return (
        "no_model_call" if is_scripted_agent(getattr(result, "agent", None)) else None
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
        "n_cache_creation_tokens": result.n_cache_creation_tokens,
        "n_cache_write_tokens": getattr(result, "n_cache_write_tokens", None),
        "n_cache_write_5m_tokens": getattr(result, "n_cache_write_5m_tokens", None),
        "n_cache_write_1h_tokens": getattr(result, "n_cache_write_1h_tokens", None),
        "total_tokens": result.total_tokens,
        "cost_usd": result.cost_usd,
        "usage_source": result.usage_source,
        "cost_source": getattr(result, "cost_source", None),
        "price_table_date": getattr(result, "price_table_date", None),
        "price_source": result.price_source,
        "usage_source_legacy": getattr(result, "usage_source_legacy", None),
        "no_usage_reason": _no_usage_reason(result),
        # A count that missed calls says so as its result.json does, so
        # summary.json counts the trials this run finished as it counts the
        # ones it reads back.
        **incomplete_usage_fields(result),
    }
    if getattr(result, "usage_details", None) is not None:
        agent_result["usage_details"] = result.usage_details
    return agent_result


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
        "error": result.error,
        "error_category": result.error_category,
        "verifier_error": result.verifier_error,
        "verifier_error_category": result.verifier_error_category,
        "export_error": result.export_error,
        "n_tool_calls": result.n_tool_calls,
        "n_skill_invocations": n_skill_invocations,
        "agent_result": agent_result_from_rollout(result),
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


def _counted_tokens(agent: dict[str, Any]) -> bool:
    """Whether a trial's total tokens are known: a named usage source with
    null counters counted nothing a total can add."""
    total_tokens = agent.get("total_tokens")
    return isinstance(total_tokens, int) and not isinstance(total_tokens, bool)


def _source_counts(results: dict[str, dict]) -> dict[str, Any]:
    usage_sources: dict[str, int] = {}
    cost_sources: dict[str, int] = {}
    without_usage = without_cost = without_model_call = 0
    usage_incomplete = cost_incomplete = 0
    generating_only = 0
    for r in results.values():
        agent = r.get("agent_result") or {}
        if agent.get("no_usage_reason") == "no_model_call":
            without_model_call += 1
            continue
        # A count that missed calls (usage_incomplete: a turn cut before
        # its report, a stream cut before the provider's usage): what was
        # counted, never the whole. Its priced cost is a floor, and
        # cost_is_floor below counts it (merge: counted once).
        if agent.get("usage_incomplete") is True:
            usage_incomplete += 1
        source = result_usage_source(agent.get("usage_source"))
        if source is None or not _counted_tokens(agent):
            without_usage += 1
        else:
            usage_sources[source] = usage_sources.get(source, 0) + 1
        # A known cost that is a floor: some calls uncounted, or an unpriced
        # warm-up left out.
        cost_incomplete += cost_is_floor(agent)
        generating_only += unpriced_warmup(agent.get("usage_details"))
        cost_source = agent.get("cost_source")
        if agent.get("cost_usd") is None:
            without_cost += 1
        elif isinstance(cost_source, str):
            cost_sources[cost_source] = cost_sources.get(cost_source, 0) + 1
    return {
        "usage_sources": dict(sorted(usage_sources.items())),
        "cost_sources": dict(sorted(cost_sources.items())),
        "trials_without_usage": without_usage,
        "trials_without_cost": without_cost,
        "trials_without_model_call": without_model_call,
        "trials_usage_incomplete": usage_incomplete,
        # The trials whose known cost is a floor: some calls uncounted, or
        # Codex's unpriced generate=false warm-up left out.
        "trials_cost_incomplete": cost_incomplete,
        # What the token totals cover when a count leaves out Codex's
        # generate=false warm-up by design; None for no such trial.
        "usage_scope": (
            {
                "generating_calls_only": generating_only,
                "scope": SUMMARY_GENERATING_SCOPE,
            }
            if generating_only
            else None
        ),
    }


def usage_summary(results: dict[str, dict]) -> dict[str, Any]:
    """Aggregate provider telemetry fields for summary.json.

    Totals cover every trial whose tokens a trusted source counted, errored,
    unscored and stopped ones included: a trial that timed out, lost
    its sandbox or was interrupted still spent what it spent. Tokens are
    BenchFlow's split: ``total_input_tokens`` is uncached input, beside the
    cache reads and writes. ``telemetry_coverage`` is over every trial, and
    like ``usage_sources`` counts a trial only when its total tokens are
    known; ``trials_without_usage`` counts the rest.
    """
    completed = list(results.values())
    covered = [
        r
        for r in completed
        if is_trusted_usage_source((r.get("agent_result") or {}).get("usage_source"))
    ]

    def total(field: str, alias: str | None = None) -> int | None:
        # A trial that did not count a field adds nothing to it, and a field
        # no trial counted is unknown, not 0.
        values = []
        for r in covered:
            agent = r.get("agent_result") or {}
            value = agent.get(field)
            if value is None and alias is not None:
                value = agent.get(alias)
            if isinstance(value, int) and not isinstance(value, bool):
                values.append(value)
        return sum(values) if values else None

    def unknown(field: str) -> int:
        # The trials total() sums over whose count of a field is not known:
        # the total leaves them out.
        return sum(
            1
            for r in covered
            if not isinstance(value := (r.get("agent_result") or {}).get(field), int)
            or isinstance(value, bool)
        )

    # A missing cost is unknown (subscription logins report none), not zero.
    costs = [
        cost
        for r in covered
        if (cost := (r.get("agent_result") or {}).get("cost_usd")) is not None
    ]
    total_cost = round(sum(costs), 10) if costs else None
    return {
        "total_input_tokens": total("n_input_tokens"),
        "total_output_tokens": total("n_output_tokens"),
        "total_cache_read_tokens": total("n_cache_read_tokens"),
        "total_cache_creation_tokens": total("n_cache_creation_tokens"),
        # The trials whose cache-write count no answer reported: the
        # totals of writes are the others'.
        "trials_without_cache_write_count": unknown("n_cache_creation_tokens"),
        # its split: total_input_tokens is uncached input; the cache
        # writes (total_cache_creation_tokens, kept as an alias) with their
        # 5-minute and 1-hour shares where reported.
        "total_cache_write_tokens": total(
            "n_cache_write_tokens", "n_cache_creation_tokens"
        ),
        "total_cache_write_5m_tokens": total("n_cache_write_5m_tokens"),
        "total_cache_write_1h_tokens": total("n_cache_write_1h_tokens"),
        "total_tokens": total("total_tokens"),
        "total_cost_usd": total_cost,
        "avg_cost_per_trial_usd": (
            round(total_cost / len(costs), 10) if total_cost is not None else None
        ),
        "telemetry_coverage": (
            sum(1 for r in covered if _counted_tokens(r.get("agent_result") or {}))
            / len(completed)
            if completed
            else 0.0
        ),
        # the trials by where their tokens came from and who set their
        # cost; those nothing counted; and those that called no model (a
        # control or an operator script), which are neither zeros nor
        # missing usage.
        **_source_counts(results),
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


def _known_sum(values: Any) -> int | float | None:
    """The sum of the known numbers; None when none is known (never 0)."""
    known = [
        v for v in values if isinstance(v, int | float) and not isinstance(v, bool)
    ]
    return sum(known) if known else None


def _round_usd(value: int | float | None) -> float | None:
    return round(value, 10) if value is not None else None


def attempt_spend_summary(attempts: Any) -> dict[str, Any]:
    """summary.json's ``spend_all_attempts``: what the job spent over every
    attempt it recorded, retries, stopped attempts and resumed runs included
    (the 0.10 line's block, which also counts judges).

    The ``total_*`` keys of :func:`usage_summary` count each task's kept
    attempt and the attempts stopped before they finished; this counts every
    attempt in the job folder (``attempts`` from
    ``evaluation._recorded_attempts``: result.json dicts, cancellation
    receipts, or results) by the rule the budget's ``spent`` uses
    (:class:`benchflow.budget.AttemptSpend`). ``cost_usd`` and ``tokens``
    are None when no attempt's is known. ``attempts_without_cost`` and
    ``attempts_without_tokens`` count the finished attempts whose figure is
    unknown, and ``attempts_without_model_call`` those that called no model
    (an oracle, a nop, a script), which are in no other count.
    ``stopped_attempts`` were stopped before they finished (Ctrl-C, SIGTERM,
    a budget): what their record holds counts, ``stopped_attempts_without_usage``
    recorded nothing and ``stopped_attempts_without_cost`` recorded tokens
    with no cost. ``attempts_cost_incomplete`` counts the attempts whose cost
    was priced from some of their calls only (``usage_incomplete``), so
    ``cost_usd`` is a floor when it is not 0.
    """
    from benchflow._utils.result_paths import is_interrupted_result
    from benchflow.budget import AttemptSpend

    attempts = list(attempts)

    def stopped_record(row: Any) -> bool:
        # A cancellation receipt, or the stopped attempt's own result.json.
        return isinstance(row, dict) and (
            bool(row.get("_budget_cancelled")) or is_interrupted_result(row)
        )

    finished = [AttemptSpend.of(a) for a in attempts if not stopped_record(a)]
    stopped = [AttemptSpend.of(a) for a in attempts if stopped_record(a)]
    spends = finished + stopped
    no_call = [s for s in finished if s.called_no_model]
    called = [s for s in finished if not s.called_no_model]
    return {
        "spend_all_attempts": {
            "attempts": len(finished),
            "stopped_attempts": len(stopped),
            "cost_usd": _round_usd(_known_sum(s.cost_usd for s in spends)),
            "tokens": _known_sum(s.tokens for s in spends),
            "attempts_without_cost": sum(1 for s in called if s.usd_unknown),
            "attempts_without_tokens": sum(1 for s in called if s.tokens is None),
            "attempts_without_model_call": len(no_call),
            "stopped_attempts_without_usage": sum(
                1 for s in stopped if s.tokens is None and s.cost_usd is None
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


def merge_spends(blocks: Any) -> dict[str, Any] | None:
    """``spend_all_attempts`` blocks of several job folders as one: None when
    none is known, and ``parts_unknown`` counts the parts whose spend is
    unknown (a job folder whose records could not be read), so a sum with an
    unknown part says so."""
    blocks = list(blocks)
    known = [block for block in blocks if isinstance(block, dict)]
    if not known:
        return None
    merged: dict[str, Any] = {}
    for key in dict.fromkeys(k for block in known for k in block):
        total = _known_sum(block.get(key) for block in known)
        merged[key] = _round_usd(total) if isinstance(total, float) else total
    if len(known) < len(blocks):
        merged["parts_unknown"] = len(blocks) - len(known)
    return merged
