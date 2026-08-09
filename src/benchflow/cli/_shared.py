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

from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.markup import escape

from benchflow._utils.text import truncate_end
from benchflow.cli._failure_evidence import FailureLine, artifact_failure_evidence

if TYPE_CHECKING:
    from pathlib import Path

    from benchflow.evaluation import EvaluationResult, TaskFailure

console = Console()

# stderr console for out-of-band notices (deprecations) so they never corrupt
# stdout consumers like `--json` (e.g. `environment list --json`).
err_console = Console(stderr=True)


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
    err_console.print(f"[red]{escape(str(message))}[/red]", emoji=False)


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
_FAILURE_REASON_METRICS = 3


def _failure_reason(
    failure: TaskFailure, job_dir: Path | None = None
) -> tuple[FailureLine, Path | None]:
    """One cheap line explaining why a FAILED (scored, reward != 1) task
    failed, plus the verifier dir when artifacts supplied the evidence.

    Priority: the verifier's own error if set; else the reward plus a compact
    breakdown of the named metrics in the reward dict (zero/failed metrics
    first — they explain the miss); else the reward plus a one-liner mined
    from the rollout's verifier artifacts (bounded CLI-side reads resolved via
    the ``rollout_name`` the engine records — the engine stays file-free);
    else just the reward. The reason is a ``FailureLine`` so the artifact
    tier's multi-failure count suffix stays separable from the truncatable
    body (only the artifact tier ever sets it). The returned path is
    non-``None`` only for the artifact tier, so the caller can print one
    ``(details: …)`` pointer per report block.
    """
    if failure.verifier_error:
        # Collapse whitespace: verifier errors are routinely multi-line.
        return FailureLine(" ".join(failure.verifier_error.split())), None
    rewards = failure.rewards or {}
    reward = rewards.get("reward")
    metrics = [
        (name, value)
        for name, value in rewards.items()
        if name != "reward" and isinstance(value, (bool, int, float))
    ]
    if metrics:
        # Zero/failed metrics first (stable within each group), capped so one
        # metric-happy verifier can't flood the line.
        metrics.sort(key=lambda kv: kv[1] != 0)
        shown = ", ".join(
            f"{name} {value}" for name, value in metrics[:_FAILURE_REASON_METRICS]
        )
        return FailureLine(f"reward {reward} — {shown}"), None
    # No rollout_name (pre-#957-follow-up result.json, sharded aggregation):
    # nothing to resolve — the bare reward is as honest as it gets.
    if job_dir is not None and failure.rollout_name:
        evidence = artifact_failure_evidence(job_dir, failure.rollout_name)
        if evidence is not None:
            detail, verifier_dir = evidence
            return (
                FailureLine(f"reward {reward} — {detail.body}", detail.suffix),
                verifier_dir,
            )
    return FailureLine(f"reward {reward}"), None


def _report_eval_result(result: EvaluationResult, job_dir: Path | None = None) -> None:
    """Print the Score/errors summary line, colored by outcome, plus artifacts.

    A clean pass and a total failure used to look identical (both bold white);
    now the line is green only on a full clean pass, red on a shutout, amber
    otherwise, and ``errors=N`` is red when non-zero. Each FAILED task gets one
    dim ``✗ task: reason`` line (capped at ``_MAX_FAILURE_LINES``) so the "why"
    doesn't require opening summary.json; when ``job_dir`` is given, a reason
    that would otherwise be a bare ``reward X`` is upgraded from the rollout's
    verifier artifacts (bounded reads, displayed failures only), with a single
    ``(details: …/verifier)`` pointer line for the block. When ``job_dir`` is
    given, the result/summary paths are printed so testers know where to look
    (the guide repeatedly says "read summary.json" but the CLI never said
    where).
    """
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
    console.print(
        f"\n[{style}]{mark} Score: {result.passed}/{result.total} "
        f"({result.score:.1%})[/{style}]{err_part}"
    )
    # One dim reason line per FAILED task, so "0/1" doesn't force a dig into
    # summary.json to learn why. getattr(): sharded aggregation and older
    # SimpleNamespace-style callers don't carry task_failures.
    failures = getattr(result, "task_failures", None) or []
    artifact_pointer: Path | None = None
    for failure in failures[:_MAX_FAILURE_LINES]:
        reason, verifier_dir = _failure_reason(failure, job_dir)
        if artifact_pointer is None and verifier_dir is not None:
            artifact_pointer = verifier_dir
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
    # One pointer per block (not per line): the first artifact-backed reason's
    # verifier dir, so "where do I look next" needs no summary.json dig either.
    if artifact_pointer is not None:
        console.print(f"[dim]  (details: {escape(str(artifact_pointer))})[/dim]")
    if job_dir is not None:
        console.print(f"[dim]Artifacts:[/dim] {escape(str(job_dir))}")
        console.print(f"[dim]Summary:  [/dim] {escape(str(job_dir))}/summary.json")


def _parse_agent_env(entries: list[str] | None) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` CLI options into a dict."""
    import typer

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
