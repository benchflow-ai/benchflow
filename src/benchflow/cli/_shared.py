"""Shared console + display helpers for the benchflow CLI command modules.

These are the cross-cutting, side-effect-free helpers that several CLI command
groups (``cli/main.py`` and the ``cli/<group>.py`` modules) need in common: the
shared Rich :data:`console`, the evaluation-result summary/exit helpers, and the
agent ``Requires`` rendering used by ``agents``/``agent`` listings. The one
file-reading step of the eval report (mining verifier artifacts for failure
evidence) is delegated to :mod:`._failure_evidence` to keep that claim true.

Keeping them here lets each command group import one stable surface instead of
re-deriving the formatting, and lets ``cli/main.py`` stay a thin app + eval
wiring module while preserving identical output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.markup import escape

from benchflow._utils.text import truncate_end
from benchflow.cli._failure_evidence import (
    FailureLine,
    artifact_failure_evidence,
    metric_breakdown,
    verifier_dir_for,
)

if TYPE_CHECKING:
    from pathlib import Path

    from benchflow.evaluation import EvaluationResult, TaskFailure

console = Console()

# stderr console for out-of-band notices (deprecations) so they never corrupt
# stdout consumers like `--json` (e.g. `environment list --json`).
err_console = Console(stderr=True)


def stopped_run_result(exc: BaseException) -> Any:
    """The ``EvaluationResult`` a job that stopped on a usage limit carries.

    ``Evaluation.run`` raises ``UsageLimitError`` once its running trials
    finish; the report then shows what finished. With no result attached
    (the job never started), print the error and exit 1.
    """
    from benchflow.errors import user_message

    result = getattr(exc, "result", None)
    if result is None:
        print_error(user_message(exc))
        raise typer.Exit(1)
    return result


def print_error(message: str) -> None:
    """Print a red error line to **stderr**, escaping Rich markup in ``message``.

    The single safe sink for CLI error messages. Two jobs:

    1. *Escape* — error text routinely interpolates user-supplied values (a task
       path, an agent name, a config error echoing a field) that can contain
       ``[`` / ``[/x]`` tokens. An unescaped ``console.print(f"[red]{value}[/red]")``
       then makes Rich itself raise ``MarkupError`` — turning a clean error into a
       raw traceback. (Messages with NO user input escape to a no-op, so it is
       always safe.)
    2. *Stream* — write to ``err_console`` (stderr), not stdout. Errors on stdout
       corrupt ``--json`` consumers (a ``bench … --json | jq`` pipeline gets a
       non-JSON line on the JSON channel); the same stderr rule the deprecation
       notices follow. Exit codes are unchanged, so failures stay detectable.
    """
    # emoji=False: interpolated user input often contains ``:token:`` patterns
    # (e.g. a hosted-env ref ``primeintellect:a:b``). With Rich's default
    # emoji=True, err_console would substitute ``:a:`` with an emoji, corrupting
    # the echoed-back value. escape() neutralizes [..] markup but not shortcodes.
    # soft_wrap: never hard-wrap an error at the console width; a path split
    # across lines cannot be copied or grepped from a CI log.
    err_console.print(f"[red]{escape(str(message))}[/red]", emoji=False, soft_wrap=True)


_DEPRECATION_WARNED: set[str] = set()


def warn_deprecated(old: str, new: str, *, removal: str = "0.7") -> None:
    """Emit a one-line deprecation notice to stderr, once per ``old`` per process.

    ``old``/``new`` are the user-facing invocations, e.g.
    ``warn_deprecated("bench agent create", "bench eval adopt <name> --scaffold-only")``.
    Printed before the command does its real work so exit codes + stdout stay
    unchanged.
    """
    if old in _DEPRECATION_WARNED:
        return
    _DEPRECATION_WARNED.add(old)
    # Plain "deprecation:" label — NOT "[deprecated]", which Rich would parse as
    # a markup tag and silently swallow.
    err_console.print(
        f"[yellow]deprecation:[/yellow] {old!r} is now {new!r} and will be removed "
        f"in {removal}. Update your scripts."
    )


_PROVIDER_AUTH_MESSAGE = (
    "Provider-prefixed models may use different credentials; Azure Foundry "
    "models use AZURE_API_KEY + AZURE_API_ENDPOINT."
)
_REQUIRES_AUTH_NOTE = (
    "Requires shows native/default agent auth. " + _PROVIDER_AUTH_MESSAGE
)


def _format_requires(agent) -> str:
    sub_env = agent.subscription_auth.replaces_env if agent.subscription_auth else None
    requires = [
        f"{env_var} (or login)" if env_var == sub_env else env_var
        for env_var in agent.requires_env
    ]
    return ", ".join(requires)


def _exit_if_evaluation_had_errors(result: object) -> None:
    errored = int(getattr(result, "errored", 0) or 0)
    verifier_errored = int(getattr(result, "verifier_errored", 0) or 0)
    if errored or verifier_errored:
        raise typer.Exit(1)


# Final-block failure lines: keep the block skimmable on big jobs and each
# line inside a typical terminal width — except lines carrying a multi-failure
# count suffix, which deliberately run ~40 chars past the budget.
_MAX_FAILURE_LINES = 5
_FAILURE_LINE_LIMIT = 100


def _failure_reason(failure: TaskFailure, job_dir: Path | None = None) -> FailureLine:
    """One cheap line explaining why a FAILED (scored, reward != 1) task failed.

    Priority: the verifier's own error if set; else the reward plus a compact
    breakdown of the named metrics in the reward dict — flat, or flattened one
    level from an env0-style ``metrics``/``details`` sub-dict, lowest-signal
    metrics first (see :func:`metric_breakdown`); else the reward plus a
    one-liner mined from the rollout's verifier artifacts (bounded CLI-side
    reads resolved via the ``rollout_name`` the engine records — the engine
    stays file-free); else just the reward. The reason is a ``FailureLine`` so
    the artifact tier's multi-failure count suffix stays separable from the
    truncatable body (only the artifact tier ever sets it). The block's
    ``(details: …)`` pointer is NOT decided here — the caller keys it off
    on-disk artifacts alone, independent of which tier supplied the reason.
    """
    if failure.verifier_error:
        # Collapse whitespace: verifier errors are routinely multi-line.
        return FailureLine(" ".join(failure.verifier_error.split()))
    rewards = failure.rewards or {}
    reward = rewards.get("reward")
    shown = metric_breakdown(rewards)
    if shown is not None:
        return FailureLine(f"reward {reward} — {shown}")
    # No rollout_name (pre-#957-follow-up result.json, sharded aggregation):
    # nothing to resolve — the bare reward is as honest as it gets.
    if job_dir is not None and failure.rollout_name:
        detail = artifact_failure_evidence(job_dir, failure.rollout_name)
        if detail is not None:
            return FailureLine(f"reward {reward} — {detail.body}", detail.suffix)
    return FailureLine(f"reward {reward}")


def _report_checkpoint_retries(result: object) -> None:
    """One line for ``--retry-from-checkpoint`` retries, next to the score
    they never change."""
    from benchflow.checkpoint_retry import retry_summary

    results = getattr(result, "results", None)
    if not isinstance(results, dict):
        return
    counts = retry_summary(list(results.values()))
    if not counts:
        return
    parts = [f"{counts['passed']}/{counts['attempted']} passed"]
    if counts.get("no_work"):
        parts.append(f"{counts['no_work']} made no tool calls")
    if counts.get("failed_to_run"):
        parts.append(f"{counts['failed_to_run']} failed to run")
    if counts.get("no_checkpoint"):
        parts.append(f"{counts['no_checkpoint']} had no checkpoint")
    if counts.get("refused"):
        parts.append(f"{counts['refused']} refused by the restore boundary")
    console.print(
        "Retries from checkpoints: "
        + ", ".join(parts)
        + " (the score above is the trials' own; see each result.json retry block)",
        highlight=False,
        soft_wrap=True,
    )


def _report_eval_result(result: EvaluationResult, job_dir: Path | None = None) -> None:
    """Print the Score/errors summary line, colored by outcome, plus artifacts.

    A clean pass and a total failure used to look identical (both bold white);
    now the line is green only on a full clean pass, red on a shutout, amber
    otherwise, and ``errors=N`` is red when non-zero. Each FAILED task gets one
    dim ``✗ task: reason`` line (capped at ``_MAX_FAILURE_LINES``) so the "why"
    doesn't require opening summary.json; when ``job_dir`` is given, a reason
    that would otherwise be a bare ``reward X`` is upgraded from the rollout's
    verifier artifacts (bounded reads, displayed failures only), and every
    failure block with artifacts on disk gets one ``(details: …/verifier)``
    pointer line — evidence mined or not. When ``job_dir`` is
    given, the result/summary paths are printed so testers know where to look
    (the guide repeatedly says "read summary.json" but the CLI never said
    where).
    """
    reused = int(getattr(result, "reused", 0) or 0)
    ran = int(getattr(result, "ran", 0) or 0)
    if reused:
        where = f" {job_dir}" if job_dir is not None else ""
        if ran:
            note = f"{reused} finished task(s) reused, {ran} ran now"
        else:
            note = (
                f"all {reused} task(s) were already finished, so nothing ran and "
                "these are the earlier results"
            )
        err_console.print(
            f"[yellow]Resumed job{escape(where)}:[/yellow] {note}. Pass --fresh "
            "(or a new --job-name) for a new run.",
            highlight=False,
            soft_wrap=True,
        )
    errors = int(getattr(result, "errored", 0) or 0)
    verifier_errors = int(getattr(result, "verifier_errored", 0) or 0)
    total_errors = errors + verifier_errors
    if result.total and result.passed == result.total and total_errors == 0:
        style, mark = "bold green", "✓"
    elif result.passed > 0:
        style, mark = "bold yellow", "•"
    else:
        style, mark = "bold red", "✗"
    # The displayed count must agree with the colour decision (which uses
    # total_errors): a verifier-error-only run is NOT "errors=0". Break out the
    # verifier bucket when present so the two error kinds stay legible.
    if total_errors:
        detail = f"errors={errors}"
        if verifier_errors:
            detail += f" verifier-errors={verifier_errors}"
        err_part = f", [red]{detail}[/red]"
    else:
        err_part = ", errors=0"
    # Mean reward alongside the binarized counts: pass/fail thresholds at
    # reward==1, so "0/1" alone can't distinguish 0.3 partial credit from a
    # flat 0. getattr(): sharded aggregation and older callers don't carry it.
    mean_reward = getattr(result, "mean_reward", None)
    mean_part = f", mean reward {mean_reward:.2f}" if mean_reward is not None else ""
    console.print(
        f"\n[{style}]{mark} Score: {result.passed}/{result.total} "
        f"({result.score:.1%})[/{style}]{mean_part}{err_part}"
    )
    _report_checkpoint_retries(result)
    budget = getattr(result, "budget", None)
    if budget and budget.get("stopped"):
        console.print(
            f"[yellow]Budget:[/yellow] {escape(str(budget.get('reason')))}; "
            f"{len(budget.get('cancelled') or [])} running trial(s) cancelled, "
            f"{len(budget.get('not_started') or [])} not started (not counted as "
            "failures; resume the job to run them)"
        )
    # One dim reason line per FAILED task, so "0/1" doesn't force a dig into
    # summary.json to learn why. getattr(): sharded aggregation and older
    # SimpleNamespace-style callers don't carry task_failures.
    failures = getattr(result, "task_failures", None) or []
    artifact_pointer: Path | None = None
    for failure in failures[:_MAX_FAILURE_LINES]:
        reason = _failure_reason(failure, job_dir)
        # Pointer rule: every failure block with artifacts on disk gets one
        # (details:) pointer — the first displayed failure whose verifier dir
        # exists, independent of which tier supplied its reason line (byte-
        # identical reasons must not differ on provenance the console can't
        # show, and the pointer matters most when every probe missed).
        if artifact_pointer is None and job_dir is not None and failure.rollout_name:
            artifact_pointer = verifier_dir_for(job_dir, failure.rollout_name)
        # The char budget governs only the free-text body; the multi-failure
        # count suffix is appended AFTER truncation so a long assertion can
        # never swallow the "more than this is broken" signal. Worst case the
        # line runs ~40 chars past the budget — completeness beats strict
        # width for this one signal.
        body = truncate_end(
            f"  ✗ {failure.task_name}: {reason.body}",
            _FAILURE_LINE_LIMIT,
        )
        console.print(f"[dim]{escape(body + reason.suffix)}[/dim]")
    extra = len(failures) - _MAX_FAILURE_LINES
    if extra > 0:
        console.print(f"[dim]  … and {extra} more[/dim]")
    # One pointer per block (not per line): the first displayed failure whose
    # verifier dir exists — artifact-backed or not — so "where do I look next"
    # needs no summary.json dig even when evidence mining came up empty.
    if artifact_pointer is not None:
        # Paths are never wrapped, so they can be copied and grepped from a
        # pipe or CI log.
        console.print(
            f"[dim]  (details: {escape(str(artifact_pointer))})[/dim]", soft_wrap=True
        )
    _report_outcomes(result, job_dir)
    if job_dir is not None:
        console.print(f"[dim]Artifacts:[/dim] {escape(str(job_dir))}", soft_wrap=True)
        console.print(
            f"[dim]Summary:  [/dim] {escape(str(job_dir))}/summary.json",
            soft_wrap=True,
        )
        console.print(
            f"[dim]View:     [/dim] bench eval view {escape(str(job_dir))}",
            soft_wrap=True,
        )


# Task names shown per cause before "and N more".
_MAX_CAUSE_TASKS = 5


def _duration(seconds: float) -> str:
    whole = max(0, round(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def _report_outcomes(result: object, job_dir: Path | None) -> None:
    """Scored, unscored and errored trials, by cause, with whose fault and what next.

    Then the run's cost and time. ``result.results`` (one RolloutResult per
    task) feeds it; a result without them (sharded runs, older callers)
    prints nothing here.
    """
    import json
    from collections.abc import Mapping
    from typing import cast

    from benchflow._utils.scoring import classify_score_outcome
    from benchflow.failures import cause_of, task_dir_finder

    results = getattr(result, "results", None)
    if not isinstance(results, dict) or not results:
        return
    task_dir = task_dir_finder(job_dir)
    groups: dict[tuple[str, str, str], list[tuple[str, str]]] = {}
    counts = {"passed": 0, "failed": 0, "unscored": 0, "errored": 0}
    for key, trial in sorted(results.items()):
        name = str(key)
        # The Score line's buckets: its errors are "errored" here and its
        # verifier errors "unscored". Only those trials' files are read.
        outcome = (
            classify_score_outcome(cast("Mapping[str, Any]", trial))
            if isinstance(trial, Mapping)
            else getattr(trial, "score_outcome", None)
        )
        if outcome in ("passed", "failed"):
            counts[outcome] += 1
            continue
        if outcome is None:
            outcome = "errored" if getattr(trial, "error", None) else "unscored"
        bucket = "errored" if outcome == "errored" else "unscored"
        # The bucket follows the Score line even when no cause is known (a
        # trial whose result.json cannot be read): the two lines always
        # agree, and only the explanation below is missing.
        counts[bucket] += 1
        cause = cause_of(trial, job_dir=job_dir, task_dir=task_dir(name))
        if cause is not None:
            groups.setdefault((bucket, cause.headline(), cause.key), []).append(
                (name, cause.next_step)
            )
    scored = counts["passed"] + counts["failed"]
    console.print(
        f"Outcomes: {scored} scored ({counts['passed']} passed, {counts['failed']} "
        f"failed), {counts['unscored']} unscored, {counts['errored']} errored",
        highlight=False,
        soft_wrap=True,
    )
    for bucket in ("unscored", "errored"):
        for (group_bucket, headline, _key), members in groups.items():
            if group_bucket != bucket:
                continue
            names = [name for name, _ in members]
            shown = ", ".join(names[:_MAX_CAUSE_TASKS])
            if len(names) > _MAX_CAUSE_TASKS:
                shown += f" and {len(names) - _MAX_CAUSE_TASKS} more"
            steps = list(dict.fromkeys(step for _, step in members))
            step = (
                steps[0] if len(steps) == 1 else steps[0] + " (and likewise for each)"
            )
            style = "yellow" if bucket == "unscored" else "red"
            console.print(
                f"  [{style}]{len(names)} {bucket}[/{style}]: {escape(headline)}: "
                f"{escape(shown)}",
                highlight=False,
                soft_wrap=True,
            )
            console.print(
                f"      [cyan]next:[/cyan] {escape(step)}",
                highlight=False,
                soft_wrap=True,
            )
    summary: dict = {}
    if job_dir is not None:
        try:
            loaded = json.loads((job_dir / "summary.json").read_text())
            summary = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            summary = {}
    stop = summary.get("usage_limit")
    if isinstance(stop, dict) and stop.get("not_started"):
        where = str(job_dir) if job_dir is not None else "<job dir>"
        console.print(
            f"  {len(stop['not_started'])} not started: the login's usage limit "
            f"stopped the job; `bench eval resume {escape(where)}` runs them on "
            "another login or after the reset",
            highlight=False,
            soft_wrap=True,
        )
    costs = [
        c
        for trial in results.values()
        if isinstance(c := getattr(trial, "cost_usd", None), int | float)
        and not isinstance(c, bool)
    ]
    tokens = [
        t
        for trial in results.values()
        if isinstance(t := getattr(trial, "total_tokens", None), int)
        and not isinstance(t, bool)
    ]
    token_text = f", {sum(tokens):,} tokens" if tokens else ""
    trials = len(results)
    no_usage = trials - len(tokens)
    usage_note = (
        f" ({no_usage} of {trials} trials reported no usage)" if no_usage else ""
    )
    if costs:
        missing = trials - len(costs)
        note = f" ({missing} of {trials} trials reported no cost)" if missing else ""
        cost_text = f"${sum(costs):.2f}{token_text}{note}"
    elif tokens and not sum(tokens) and not no_usage:
        cost_text = "none: no model tokens were used"
    elif tokens and sum(tokens):
        cost_text = (
            "not reported in USD (subscription logins and unpriced models "
            f"report tokens only){token_text}{usage_note}"
        )
    elif tokens:
        cost_text = f"not reported{usage_note}; the others used no model tokens"
    else:
        cost_text = "not reported"
    console.print(f"[dim]Cost:     [/dim] {cost_text}", soft_wrap=True)
    elapsed = getattr(result, "elapsed_sec", None)
    if isinstance(elapsed, int | float) and not isinstance(elapsed, bool):
        reused = int(getattr(result, "reused", 0) or 0)
        tail = f" (this run; {reused} task(s) reused from before)" if reused else ""
        console.print(
            f"[dim]Time:     [/dim] {_duration(elapsed)}{tail}", soft_wrap=True
        )


def _parse_agent_env(entries: list[str] | None) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` CLI options into a dict."""
    parsed: dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            print_error(f"Invalid env var: {entry}")
            raise typer.Exit(1)
        key, value = entry.split("=", 1)
        parsed[key] = value
    return parsed


def _apply_dotenv_to_process_env() -> None:
    """Expose local .env credentials to provider SDKs without overriding env."""
    import os

    from benchflow._dotenv import load_dotenv_env

    for key, value in load_dotenv_env().items():
        os.environ.setdefault(key, value)
