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
from rich.text import Text

from benchflow._diagnostics import render_text
from benchflow._utils.command_hints import command_arg as command_arg
from benchflow._utils.command_hints import local_path_arg as local_path_arg
from benchflow.cli._failure_evidence import (
    FailureLine,
    _captured_text,
    artifact_failure_evidence,
    metric_breakdown,
    verifier_dir_for,
)

if TYPE_CHECKING:
    from pathlib import Path

    from benchflow._utils.reachable_write import ReachablePath
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
    from benchflow.cli._errors import expected_error_parts

    result = getattr(exc, "result", None)
    if result is None:
        message, hint = expected_error_parts(exc)
        print_error(message, next_step=hint or "`bench explain R213` says more")
        raise typer.Exit(1)
    return result


# The next step of a run that stopped before it had results, for an error
# the run raised that names no step of its own.
RUN_FAILED_NEXT = (
    "`bench check <task folder>` validates the tasks; `bench doctor` checks "
    "this machine"
)
# The next step of a command that reads job folders and was given a path that
# is not one.
JOBS_FOLDER_NEXT = (
    "pass a job folder or a folder of jobs (`bench run` writes them under "
    "./jobs, or --jobs-dir)"
)


def quote_next_lines(message: str) -> str:
    """``message`` with each line that reads as a ``next:`` line marked ``| next:``.

    An error's next step is only ever the ``next_step`` its caller passes to
    :func:`print_error`, which prints it last. A ``next:`` line inside the
    message came from text the message quotes (a YAML snippet or a role key of
    a downloaded task.md, a tool's stderr), so it is marked as quoted and can
    pass as the step neither on the terminal nor in the ``--json`` error.
    Line breaks of every kind ``str.splitlines`` knows (``\\r``, ``\\x0b``,
    ``\\u2028`` ...) become plain newlines first, so none hides a line.
    """
    lines = []
    for line in message.splitlines():
        body = line.lstrip()
        if body.lower().startswith("next:"):
            line = f"{line[: len(line) - len(body)]}| {body}"
        lines.append(line)
    return "\n".join(lines)


def one_line(step: str | None) -> str | None:
    """``step`` on one line: a next step is one ``next:`` line, never more."""
    if not step:
        return None
    return " ".join(part.strip() for part in step.splitlines() if part.strip()) or None


def print_hint(message: str, *, style: str = "", stderr: bool = False) -> None:
    """Render a hint as literal text: no markup, emoji expansion or wrapping.

    A Rich Text object cannot reinterpret a quoted path as formatting. Keep
    the style separate from the message, just as command_arg keeps paths
    separate from shell syntax (D2b-CONF7-1/2).
    """
    from benchflow.cli._errors import _redact, _secrets

    output = err_console if stderr else console
    output.print(
        Text(_redact(message, _secrets()), style=style), highlight=False, soft_wrap=True
    )


def with_next_step(message: str, next_step: str | None) -> str:
    """``message``, its ``next:`` lines quoted, then ``next_step`` as its last line.

    The line is the form the terminal shows and ``--json`` reports as
    ``next``; the step is always the caller's, never read out of ``message``
    (:func:`quote_next_lines`).
    """
    message = quote_next_lines(str(message))
    step = one_line(next_step)
    if step is None:
        return message
    # An empty message (``str()`` of a bare exception) would leave the next:
    # line first, where --json reads the error's message.
    return f"{message.rstrip() or 'error'}\n  next: {step}"


def exit_if_missing(path: Path | None, *, next_step: str) -> None:
    """A path argument that is not there is a usage mistake: its error and
    next step, then exit 2, as every command does (cli.md, Exit codes)."""
    import os

    if path is None or os.path.exists(path):
        return
    from benchflow._utils.command_hints import looked_in

    print_error(f"No such file or directory: {looked_in(path)}", next_step=next_step)
    raise typer.Exit(2)


def print_error(message: str, *, next_step: str | None = None) -> None:
    """Print a red error line to **stderr**, escaping Rich markup in ``message``.

    ``next_step`` is what to do about it, printed last as a ``next:`` line
    (:func:`with_next_step`): a command to run or the fix, in BenchFlow's
    words. For an expected error that is its ``hint`` or the caller's own
    (:func:`benchflow.errors.error_parts`). Every error the CLI prints ends
    with one, with a ``bench explain`` id, or with nothing when the message is
    the whole story; tests/test_cli_error_next_steps.py sweeps every call.

    The single safe sink for CLI error messages. Three jobs:

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
    3. *Redact* — mask credentials before Rich can format their lines, and
       before the message and step are recorded for the JSON error.
    """
    # emoji=False: interpolated user input often contains ``:token:`` patterns
    # (e.g. a hosted-env ref ``primeintellect:a:b``). With Rich's default
    # emoji=True, err_console would substitute ``:a:`` with an emoji, corrupting
    # the echoed-back value. escape() neutralizes [..] markup but not shortcodes.
    # soft_wrap: never hard-wrap an error at the console width; a path split
    # across lines cannot be copied or grepped from a CI log.
    from benchflow._diagnostics import render_diagnostic
    from benchflow.cli._errors import _secrets, record_error

    secrets = _secrets()
    printed = render_diagnostic(
        (str(message), next_step or ""),
        secrets,
        lambda fields: with_next_step(fields[0], fields[1] or None),
    )
    err_console.print(f"[red]{escape(printed)}[/red]", emoji=False, soft_wrap=True)
    # Under --json the message is also the command's JSON error, if it stops:
    # its text and its step kept apart, so the step is never parsed back out.
    record_error(str(message), next_step)


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
        return FailureLine(" ".join(_captured_text(failure.verifier_error).split()))
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


def report_stopped(
    stop: Any,
    job_dir: Path | None,
    *,
    since: float | None = None,
    resume: str | None = None,
) -> None:
    """The line a run stopped by Ctrl-C or SIGTERM ends with (DF-p5-cli-craft-10).

    How many trials this run stopped (attempts it started, ``since`` its
    start, with no result.json: an earlier interrupted run's are not counted
    again), that their sandboxes are gone, and how to finish the job:
    ``resume`` when the job resumes another way (a sharded run), else
    ``bench eval resume <job>``.
    """
    import json
    from pathlib import Path as _Path

    from benchflow.jobs import _interrupted

    known = job_dir is not None and _Path(job_dir).is_dir()
    folders = _interrupted(_Path(job_dir)) if known and job_dir is not None else []
    if since is not None:
        folders = [
            f
            for f in folders
            if (f / "config.json").is_file()
            and (f / "config.json").stat().st_mtime >= since - 1
        ]
    stopped = len(folders)
    what = (
        f"stopped {stopped} running trial{'s' if stopped != 1 else ''} and "
        f"removed {'their sandboxes' if stopped != 1 else 'its sandbox'}"
        if stopped
        else "no trial had started yet"
    )
    print_hint(f"{stop.how}: {what}.", style="yellow", stderr=True)
    summary = _Path(job_dir) / "summary.json" if known and job_dir is not None else None
    if summary is not None and summary.is_file():
        try:
            partial = json.loads(summary.read_text()).get("stopped") is True
        except (OSError, ValueError, AttributeError):
            partial = False
        if partial:
            # Written before the stop went on (DFR-p3-11).
            print_hint(
                f"Partial summary (what finished): {local_path_arg(summary, '<file>')}",
                stderr=True,
            )
    if resume is not None:
        print_hint(f"Resume: {resume}", stderr=True)
    elif known and job_dir is not None:
        print_hint(
            f"Resume with: bench eval resume {local_path_arg(job_dir, '<job dir>')}",
            stderr=True,
        )


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
    # The empty control (nop) is expected to pass nothing: its 0 says the
    # task cannot be passed by doing nothing, which is good news, not a
    # failure (DF-p1-first-run-13).
    agent = str(getattr(getattr(result, "config", None), "agent", "") or "")
    empty_control = agent.lower() in _EMPTY_CONTROL_AGENTS
    # Planned tasks a budget or usage limit kept from running: the score is
    # over what ran, so the line says it and never shows the all-passed tick
    # (DF-p3-eval-runner-5).
    not_run = int(getattr(result, "not_run", 0) or 0)
    if empty_control and result.passed == 0 and total_errors == 0:
        style, mark = "bold", "•"
    elif (
        result.total
        and result.passed == result.total
        and total_errors == 0
        and not not_run
    ):
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
    # The score and mean reward count every trial (an unscored one as a
    # fail); with unscored trials the line says so, and the figures over the
    # scored ones follow, with their own n, so either can be quoted. The
    # scored-only mean is what BenchFlow 0.7 printed as its mean_reward, so
    # the line names it: a 0.7 job resumed here otherwise reads as a drop
    # (DF-p4-upgrader-1).
    scored_part = ""
    if total_errors:
        if mean_reward is not None:
            mean_part += f" over all {result.total} trials (unscored as 0)"
        # The score buckets are disjoint: scored = passed + failed.
        scored = result.total - total_errors
        mean_scored = getattr(result, "mean_reward_scored_only", None)
        scored_part = (
            f"; scored only: {result.passed}/{scored} ({result.passed / scored:.1%})"
            + (
                f", mean reward {mean_scored:.2f} (0.7's mean_reward)"
                if mean_scored is not None
                else ""
            )
            if scored
            else "; scored only: none scored"
        )
    plan_part = ""
    if not_run:
        planned = result.total + not_run
        budget_block = getattr(result, "budget", None) or {}
        why = "budget" if budget_block.get("stopped") else "usage limit"
        plan_part = (
            f"; [yellow]{not_run} of {planned} planned tasks not run ({why})[/yellow]"
            f", so {result.passed}/{planned} ({result.passed / planned:.1%}) of the plan"
        )
    console.print(
        f"\n[{style}]{mark} Score: {result.passed}/{result.total}"
        f"{' run' if not_run else ''} "
        f"({result.score:.1%})[/{style}]{mean_part}{err_part}{scored_part}{plan_part}"
    )
    if empty_control and total_errors == 0:
        if result.passed == 0:
            print_hint(
                f"{agent} control: {result.passed}/{result.total}, as expected: "
                "doing nothing passes none of the tasks"
            )
        else:
            print_hint(
                f"{agent} control: {result.passed}/{result.total} passed with "
                "nothing done: those tasks can be passed without solving them; "
                "check their verifiers",
                style="yellow",
            )
    caveat = getattr(result, "mean_reward_caveat", None)
    if caveat:
        console.print(f"[dim]{escape(caveat)}[/dim]")
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
    if empty_control:
        # Each control task failing is the expected result said above.
        failures = []
    artifact_pointer: ReachablePath | None = None
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
        from benchflow._diagnostics import render_diagnostic
        from benchflow.cli._errors import _secrets

        body = render_diagnostic(
            (failure.task_name, reason.body, reason.suffix),
            _secrets(),
            lambda fields: (
                render_text(f"  ✗ {fields[0]}: {fields[1]}", _FAILURE_LINE_LIMIT)
                + fields[2]
            ),
        )
        console.print(f"[dim]{escape(body)}[/dim]")
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
            f"[dim]  {escape(artifact_pointer.details_message())}[/dim]", soft_wrap=True
        )
    _report_outcomes(result, job_dir)
    if job_dir is not None:
        console.print(f"[dim]Artifacts:[/dim] {escape(str(job_dir))}", soft_wrap=True)
        console.print(
            f"[dim]Summary:  [/dim] {escape(str(job_dir))}/summary.json",
            soft_wrap=True,
        )
        print_hint(f"View:      bench view {local_path_arg(job_dir, '<job dir>')}")


# The empty controls (benchflow.jobs' empty agents): an expected 0.
_EMPTY_CONTROL_AGENTS = frozenset({"nop", "noop", "empty"})

# Task names shown per cause before "and N more".
_MAX_CAUSE_TASKS = 5


def _duration(seconds: float) -> str:
    whole = max(0, round(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def explain_pointer(codes: list[str]) -> str | None:
    """The line a summary that named causes ends with: which ``bench explain`` to run.

    ``bench explain R212 says more (also verifier_failure)``, as Rust's
    ``bench`` words it; None when it named none.
    """
    seen = list(dict.fromkeys(codes))
    if not seen:
        return None
    first, rest = seen[0], seen[1:]
    if not rest:
        return f"`bench explain {first}` says more"
    if len(rest) > 2:
        return f"`bench explain {first}` says more (also {rest[0]}, {rest[1]}, ...)"
    return f"`bench explain {first}` says more (also {', '.join(rest)})"


def _print_verifier_tail(job_dir: Path, rollouts: list[str]) -> None:
    """test.sh's last output lines for the first crashed verifier that has them.

    The crash's own words (a traceback, a NameError) are in its
    verifier/test-stdout.txt; the summary quotes them so the next step is
    not a file hunt (DF-p2-task-author-11).
    """
    from benchflow.cli._failure_evidence import verifier_output_tail

    for rollout in rollouts:
        found = verifier_output_tail(job_dir, rollout)
        if found is None:
            continue
        _, lines = found
        shown = job_dir / rollout / "verifier" / "test-stdout.txt"
        print_hint(f"      test.sh's last lines ({shown}):")
        for line in lines:
            print_hint(f"        | {line}")
        return


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
    from benchflow.failures import RUST_CODES, cause_of, task_dir_finder

    results = getattr(result, "results", None)
    if not isinstance(results, dict) or not results:
        return
    task_dir = task_dir_finder(job_dir)
    groups: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
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
            rollout = (
                cast("Mapping[str, Any]", trial).get("rollout_name")
                if isinstance(trial, Mapping)
                else getattr(trial, "rollout_name", None)
            )
            groups.setdefault((bucket, cause.headline(), cause.key), []).append(
                (name, cause.next_step, str(rollout or ""))
            )
    scored = counts["passed"] + counts["failed"]
    console.print(
        f"Outcomes: {scored} scored ({counts['passed']} passed, {counts['failed']} "
        f"failed), {counts['unscored']} unscored, {counts['errored']} errored",
        highlight=False,
        soft_wrap=True,
    )
    shown_codes: list[str] = []
    for bucket in ("unscored", "errored"):
        for (group_bucket, headline, key), members in groups.items():
            if group_bucket != bucket:
                continue
            shown_codes.append(RUST_CODES.get(key, key))
            names = [name for name, _, _ in members]
            shown = ", ".join(names[:_MAX_CAUSE_TASKS])
            if len(names) > _MAX_CAUSE_TASKS:
                shown += f" and {len(names) - _MAX_CAUSE_TASKS} more"
            steps = list(dict.fromkeys(step for _, step, _ in members))
            step = (
                steps[0] if len(steps) == 1 else steps[0] + " (and likewise for each)"
            )
            style = "yellow" if bucket == "unscored" else "red"
            console.print(
                f"  [{style}]{len(names)} {bucket}[/{style}]: {escape(_captured_text(headline))}: "
                f"{escape(_captured_text(shown))}",
                highlight=False,
                soft_wrap=True,
            )
            if key == "verifier_failure" and job_dir is not None:
                _print_verifier_tail(job_dir, [r for _, _, r in members if r])
            print_hint(f"      next: {step}")
    pointer = explain_pointer(shown_codes)
    if pointer is not None:
        console.print(f"  {escape(pointer)}", highlight=False, soft_wrap=True)
    summary: dict = {}
    if job_dir is not None:
        try:
            loaded = json.loads((job_dir / "summary.json").read_text())
            summary = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            summary = {}
    stop = summary.get("usage_limit")
    if isinstance(stop, dict) and stop.get("not_started"):
        where = command_arg(job_dir, "<job dir>")
        print_hint(
            f"  {len(stop['not_started'])} not started: the login's usage limit "
            f"stopped the job; `bench eval resume {where}` runs them on "
            "another login or after the reset",
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
            print_error(f"Invalid env var: {entry}", next_step="pass it as KEY=VALUE")
            raise typer.Exit(2)
        key, value = entry.split("=", 1)
        parsed[key] = value
    return parsed


def _apply_dotenv_to_process_env() -> None:
    """Expose local .env credentials to provider SDKs without overriding env.

    The opt-ins that let tasks read the operator's variables or reach this
    machine are not taken from it: a ``.env`` in the folder ``bench`` runs
    from may belong to the benchmark being run.
    """
    import os

    from benchflow._dotenv import load_dotenv_env
    from benchflow.task.compose_env import ALLOW_HOST_ACCESS_VAR
    from benchflow.task.env import ALLOW_TASK_ENV_VAR

    for key, value in load_dotenv_env().items():
        if key in (ALLOW_TASK_ENV_VAR, ALLOW_HOST_ACCESS_VAR):
            continue
        os.environ.setdefault(key, value)


def _apply_allow_task_env(names: list[str] | None) -> None:
    """Add ``--allow-task-env`` names to ``BENCHFLOW_ALLOW_TASK_ENV``.

    The process variable carries them to every place a task env table is
    filled, worker processes included.
    """
    import os

    from benchflow.task.env import (
        ALLOW_TASK_ENV_VAR,
        invalid_task_env_names,
        parse_allowed_task_env,
    )

    if not names:
        return
    bad = invalid_task_env_names(names)
    if bad:
        # The entry is never echoed: it may be a value (NAME=VALUE, a key).
        print_error(
            "--allow-task-env takes one variable name (letters, digits and _), "
            "not NAME=VALUE or a list; the value stays in your environment",
            next_step="Pass one variable name after --allow-task-env; repeat "
            "the option for additional names.",
        )
        raise typer.Exit(2)
    allowed = parse_allowed_task_env(os.environ.get(ALLOW_TASK_ENV_VAR)) | set(names)
    os.environ[ALLOW_TASK_ENV_VAR] = ",".join(sorted(allowed))
