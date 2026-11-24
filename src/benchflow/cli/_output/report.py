"""Printing a run's end: the default screen, ``--verbose`` and ``--json``.

``bench run --json`` writes exactly one JSON document to stdout when the run
ends (or a controlled failure stops it); every human line, the live display
included, goes to stderr. ``bench run --verbose`` (and ``bench -v run``)
adds the long-form footer after the default screen.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, TextIO

from benchflow.cli._output.run_summary import SCHEMA, summary_document, summary_lines

_JSON_ENV = "BENCHFLOW_RUN_JSON"
_state: dict[str, Any] = {"json": False, "stdout": None, "written": False}


def configure(*, verbose: bool = False, json_out: bool = False) -> None:
    """Set the run's output mode before it starts."""
    from benchflow.cli import _shared

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.NOTSET)
    if json_out:
        _state.update(json=True, stdout=sys.stdout, written=False)
        os.environ[_JSON_ENV] = "1"
        # Every human line, the live display's included, goes to stderr.
        _shared.console.file = sys.stderr


def json_mode() -> bool:
    return bool(_state["json"])


def verbose_mode() -> bool:
    return logging.getLogger().getEffectiveLevel() <= logging.DEBUG


def _write(document: dict[str, Any], out: TextIO | None = None) -> None:
    stream = out or _state["stdout"] or sys.stdout
    stream.write(json.dumps(document, indent=2, allow_nan=False) + "\n")
    stream.flush()
    _state["written"] = True


def write_failure(exit_code: int, message: str | None = None) -> None:
    """The one document of a run that stopped before its result (a refusal,
    a setup error): nothing when the result's document was written. With no
    ``message``, the error is the first one the run printed (a refused
    resume says why on stderr; the document says it too) with its next step,
    redacted as the line was."""
    if not json_mode() or _state["written"]:
        return
    from benchflow.cli import _errors

    next_step = None
    run = _errors._JSON_RUN
    if message is None and run is not None and run.errors:
        message, next_step = run.errors[0]
    if message is not None:
        from benchflow._diagnostics import redact

        secrets = _errors._secrets()
        message = redact(message, secrets)
        next_step = redact(next_step, secrets) if next_step else None
    _write(
        {
            "schema": SCHEMA,
            "status": "error",
            "exit_code": exit_code,
            "error": message,
            "next_step": next_step,
        }
    )


def print_run_result(result: Any, job_dir: Path | None) -> None:
    """The end of a run: its document in ``--json``, else the default screen
    (and with ``--verbose`` the long-form footer after it)."""
    from rich.markup import escape

    from benchflow.cli._shared import (
        _report_eval_result,
        _report_outcomes,
        console,
        print_hint,
    )

    if json_mode():
        document = summary_document(result, job_dir)
        document["status"] = "completed"
        _write(document)
        return
    from benchflow._diagnostics import redact
    from benchflow.cli._errors import _secrets

    _print_notes(result, job_dir)
    console.print()
    secrets = _secrets()
    for line in summary_lines(result, job_dir, width=console.width):
        console.print(escape(redact(line, secrets)), soft_wrap=True, highlight=False)
    # Each unscored trial's cause and next step, kept sandboxes and a usage
    # limit's not-started trials: what the table can't say.
    _report_outcomes(result, job_dir, compact=True)
    if int(getattr(result, "reused", 0) or 0):
        print_hint("A new run of the same job: pass --fresh (or another --job-name)")
    if verbose_mode():
        console.print()
        _report_eval_result(result, job_dir)


def _print_notes(result: Any, job_dir: Path | None) -> None:
    """The lines a user must act on, kept from the long-form footer: the
    controls' verdicts, a budget or run-check stop, and one reason line per
    failed trial (at most five)."""
    from rich.markup import escape

    from benchflow._diagnostics import redact
    from benchflow.cli._errors import _secrets
    from benchflow.cli._shared import (
        _failure_reason,
        console,
        control_verdicts,
        print_hint,
    )

    agent = str(getattr(getattr(result, "config", None), "agent", "") or "").lower()
    passed, total = int(result.passed), int(result.total)
    from benchflow.agents.remote_manifests import core_override

    if (override := core_override(agent)) is not None:
        print_hint(override.text(), style="yellow")
    for text, style in control_verdicts(result):
        print_hint(text, style=style)
    budget = getattr(result, "budget", None) or {}
    checks = getattr(result, "run_checks", None) or {}
    not_run = int(getattr(result, "not_run", 0) or 0)
    if not_run:
        planned = total + not_run
        why = (
            "budget"
            if budget.get("stopped")
            else "a run check"
            if checks.get("stopped")
            else "usage limit"
        )
        console.print(
            f"[yellow]{not_run} of {planned} planned tasks not run ({why})[/yellow], "
            f"so {passed}/{planned} ({passed / planned:.1%}) of the plan",
            soft_wrap=True,
        )
    if budget.get("stopped"):
        console.print(
            f"[yellow]Budget:[/yellow] {escape(str(budget.get('reason')))}; "
            f"{len(budget.get('cancelled') or [])} running trial(s) cancelled, "
            f"{len(budget.get('not_started') or [])} not started",
            soft_wrap=True,
        )
    if checks.get("stopped"):
        console.print(
            f"[red]Run check:[/red] {escape(str(checks.get('reason')))}; "
            f"{len(checks.get('not_started') or [])} trial(s) not started",
            soft_wrap=True,
        )
    failures = [] if agent == "nop" else getattr(result, "task_failures", None) or []
    secrets = _secrets()
    for failure in failures[:5]:
        reason = _failure_reason(failure, job_dir)
        line = redact(f"  ✗ {failure.task_name}: {reason.body}", secrets)
        console.print(f"[dim]{escape(line)}[/dim]", soft_wrap=True)
    if len(failures) > 5:
        console.print(f"[dim]  … and {len(failures) - 5} more failed[/dim]")
