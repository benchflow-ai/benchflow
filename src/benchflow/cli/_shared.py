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

from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.markup import escape

if TYPE_CHECKING:
    from pathlib import Path

    from benchflow.evaluation import EvaluationResult

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


def _report_eval_result(result: EvaluationResult, job_dir: Path | None = None) -> None:
    """Print the Score/errors summary line, colored by outcome, plus artifacts.

    A clean pass and a total failure used to look identical (both bold white);
    now the line is green only on a full clean pass, red on a shutout, amber
    otherwise, and ``errors=N`` is red when non-zero. When ``job_dir`` is given,
    the result/summary paths are printed so testers know where to look (the
    guide repeatedly says "read summary.json" but the CLI never said where).
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
    if job_dir is not None:
        console.print(f"[dim]Artifacts:[/dim] {escape(str(job_dir))}")
        console.print(f"[dim]Summary:  [/dim] {escape(str(job_dir))}/summary.json")
