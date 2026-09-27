"""``bench eval regrade`` — re-score stored trials with a changed verifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from benchflow.cli._shared import _apply_dotenv_to_process_env, console


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:g}"


def eval_regrade(
    path: Annotated[
        Path, typer.Argument(help="A job folder, or one trial folder inside it")
    ],
    tasks_dir: Annotated[
        Path | None,
        typer.Option(
            "--tasks-dir",
            help=(
                "Folder of task folders (or one task folder) holding the changed "
                "verifiers. Default: the task path each trial recorded."
            ),
        ),
    ] = None,
    sandbox: Annotated[
        str | None,
        typer.Option(
            "--sandbox", help="Backend for the fresh sandboxes (default: as run)"
        ),
    ] = None,
    concurrency: Annotated[
        int, typer.Option("--concurrency", min=1, help="Trials regraded at once")
    ] = 4,
    reason: Annotated[
        str | None,
        typer.Option("--reason", help="Why the verifier changed; kept in each block"),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the summary as JSON")
    ] = False,
) -> None:
    """Re-run the task's current verifier on each trial's frozen workspace.

    The new score is written beside the original (regrade.json in each trial,
    regrade-summary.json in PATH); result.json is never changed. Trials whose
    workspace was not frozen (--freeze-workspace) are listed as not regradable.
    """
    from benchflow.eval_regrade import regrade

    _apply_dotenv_to_process_env()
    try:
        summary = regrade(
            path,
            tasks_dir=tasks_dir,
            sandbox=sandbox,
            concurrency=concurrency,
            reason=reason,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Regrade failed: {escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    if as_json:
        console.print_json(json.dumps(summary.to_dict()))
    else:
        table = Table(title=f"Regrade {summary.regrade_id}")
        for column in ("Trial", "Original", "New", "Verdict"):
            table.add_column(column)
        for row in summary.trials:
            if row.status == "regraded":
                verdict = row.change or "same"
            elif row.status == "failed":
                verdict = f"failed: {row.reason}"
            else:
                verdict = f"not regradable: {row.reason}"
            table.add_row(
                escape(row.trial),
                _fmt(row.original_reward),
                _fmt(row.new_reward),
                escape(verdict),
            )
        console.print(table)
        counts = summary.counts()
        console.print(
            f"{counts['regraded']}/{counts['trials']} regraded, "
            f"{counts['changed']} changed ({counts['fail_to_pass']} fail->pass, "
            f"{counts['pass_to_fail']} pass->fail), {counts['failed']} failed, "
            f"{counts['not_regradable']} not regradable. "
            f"Summary: {escape(str(Path(summary.path) / 'regrade-summary.json'))}"
        )
    if summary.failed:
        raise typer.Exit(1)


def register_eval_regrade(eval_app: typer.Typer) -> None:
    eval_app.command("regrade")(eval_regrade)
