"""Shared console + display helpers for the benchflow CLI command modules.

These are the cross-cutting, side-effect-free helpers that several CLI command
groups (``cli/main.py`` and the ``cli/<group>.py`` modules) need in common: the
shared Rich :data:`console`, the evaluation-result summary/exit helpers, and the
agent ``Requires`` rendering used by ``agents``/``agent`` listings.

Keeping them here lets each command group import one stable surface instead of
re-deriving the formatting, and lets ``cli/main.py`` stay a thin app + eval
wiring module while preserving identical output.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.markup import escape

from benchflow._utils.text import truncate_end

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
# line inside a typical terminal width.
_MAX_FAILURE_LINES = 5
_FAILURE_LINE_LIMIT = 100
_FAILURE_REASON_METRICS = 3
# Artifact tier: never read more than this many bytes per failed task (one
# bounded read each, and only for the <= _MAX_FAILURE_LINES displayed tasks).
_ARTIFACT_READ_BYTES = 64 * 1024
# Pytest final summary, decoration stripped: "1 failed, 2 passed in 40.86s".
_PYTEST_SUMMARY_RE = re.compile(r"\d+ failed\b")


def _ctrf_failure_line(ctrf_path: Path) -> str | None:
    """``<test> failed[: <assertion>]`` from the first failed CTRF test.

    The verifier's pytest run writes a CTRF report (``pytest --ctrf``, see the
    standard path in ``task_authoring/structural_checks.py``). Per test the
    useful failure evidence is the ``E …`` assertion line inside ``trace``;
    ``message`` is only a generic phase description, so it is the fallback.
    """
    data = json.loads(ctrf_path.read_text(encoding="utf-8", errors="replace"))
    tests = (data.get("results") or {}).get("tests") or []
    for test in tests:
        if not isinstance(test, dict) or test.get("status") != "failed":
            continue
        # Full pytest node ids ("tests/test_x.py::test_build") waste the
        # 100-char line budget; keep the test function name.
        name = str(test.get("name") or "test").rsplit("::", 1)[-1]
        trace = test.get("trace")
        if isinstance(trace, str):
            for line in trace.splitlines():
                stripped = line.strip()
                if stripped.startswith("E "):
                    return f"{name} failed: {stripped[2:].strip()}"
        message = test.get("message")
        if isinstance(message, str) and message.strip():
            return f"{name} failed: {' '.join(message.split())}"
        return f"{name} failed"
    return None


def _stdout_tail_failure_line(stdout_path: Path) -> str | None:
    """Last pytest summary ("N failed, M passed …") — else last ``FAILED …``
    line — from a bounded tail of the verifier's test-stdout capture."""
    with stdout_path.open("rb") as fh:
        fh.seek(0, 2)  # SEEK_END
        fh.seek(max(0, fh.tell() - _ARTIFACT_READ_BYTES))
        tail = fh.read(_ARTIFACT_READ_BYTES).decode("utf-8", errors="replace")
    last_failed: str | None = None
    for line in reversed(tail.splitlines()):
        bare = line.strip().strip("=").strip()
        if _PYTEST_SUMMARY_RE.match(bare):
            return bare
        if last_failed is None and line.strip().startswith("FAILED "):
            last_failed = line.strip()
    return last_failed


def _artifact_failure_evidence(
    job_dir: Path, task_name: str
) -> tuple[str, Path] | None:
    """One-liner mined from the task's verifier artifacts, or ``None``.

    Best-effort display tier: reads AT MOST ONE small file (the CTRF report if
    present and small, else a bounded tail of test-stdout.txt) from the task's
    newest rollout dir (``<task>__<uuid8>``, newest mtime wins — same rule as
    resume in ``Evaluation._get_completed_tasks``). Returns the one-liner plus
    the verifier dir it came from (for the pointer line). Never raises.
    """
    try:
        candidates = [
            d for d in job_dir.glob(f"{task_name}__*") if (d / "verifier").is_dir()
        ]
        exact = job_dir / task_name
        if (exact / "verifier").is_dir():
            candidates.append(exact)
        if not candidates:
            return None
        verifier_dir = max(candidates, key=lambda d: d.stat().st_mtime) / "verifier"
        ctrf_path = verifier_dir / "ctrf.json"
        # A CTRF report only parses whole, so an oversized one is skipped
        # (not truncated) and the stdout tail is consulted instead — either
        # way exactly one file is read.
        if ctrf_path.is_file() and ctrf_path.stat().st_size <= _ARTIFACT_READ_BYTES:
            detail = _ctrf_failure_line(ctrf_path)
        else:
            stdout_path = verifier_dir / "test-stdout.txt"
            if not stdout_path.is_file():
                return None
            detail = _stdout_tail_failure_line(stdout_path)
        if detail is None:
            return None
        return detail, verifier_dir
    except Exception:
        # Any surprise (unreadable file, bad JSON, permissions) degrades to
        # the bare `reward X` reason — never to a crashed report.
        return None


def _failure_reason(
    failure: TaskFailure, job_dir: Path | None = None
) -> tuple[str, Path | None]:
    """One cheap line explaining why a FAILED (scored, reward != 1) task
    failed, plus the verifier dir when artifacts supplied the evidence.

    Priority: the verifier's own error if set; else the reward plus a compact
    breakdown of the named metrics in the reward dict (zero/failed metrics
    first — they explain the miss); else the reward plus a one-liner mined
    from the task's verifier artifacts (one bounded file read, CLI-side only —
    the engine stays file-free); else just the reward. The returned path is
    non-``None`` only for the artifact tier, so the caller can print one
    ``(details: …)`` pointer per report block.
    """
    if failure.verifier_error:
        # Collapse whitespace: verifier errors are routinely multi-line.
        return " ".join(failure.verifier_error.split()), None
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
        return f"reward {reward} — {shown}", None
    if job_dir is not None:
        evidence = _artifact_failure_evidence(job_dir, failure.task_name)
        if evidence is not None:
            detail, verifier_dir = evidence
            return f"reward {reward} — {detail}", verifier_dir
    return f"reward {reward}", None


def _report_eval_result(result: EvaluationResult, job_dir: Path | None = None) -> None:
    """Print the Score/errors summary line, colored by outcome, plus artifacts.

    A clean pass and a total failure used to look identical (both bold white);
    now the line is green only on a full clean pass, red on a shutout, amber
    otherwise, and ``errors=N`` is red when non-zero. Each FAILED task gets one
    dim ``✗ task: reason`` line (capped at ``_MAX_FAILURE_LINES``) so the "why"
    doesn't require opening summary.json; when ``job_dir`` is given, a reason
    that would otherwise be a bare ``reward X`` is upgraded from the task's
    verifier artifacts (one bounded read per displayed failure), with a single
    ``(details: …/verifier/)`` pointer line for the block. When ``job_dir`` is
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
        line = truncate_end(
            f"  ✗ {failure.task_name}: {reason}",
            _FAILURE_LINE_LIMIT,
        )
        console.print(f"[dim]{escape(line)}[/dim]")
    extra = len(failures) - _MAX_FAILURE_LINES
    if extra > 0:
        console.print(f"[dim]  … and {extra} more[/dim]")
    # One pointer per block (not per line): the first artifact-backed reason's
    # verifier dir, so "where do I look next" needs no summary.json dig either.
    if artifact_pointer is not None:
        console.print(f"[dim]  (details: {escape(str(artifact_pointer))}/)[/dim]")
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
