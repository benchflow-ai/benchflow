"""``bench monitor`` — score a rollout in production (#386).

API-surface scaffold; the runtime is not yet implemented. Every subcommand
fails closed with the canonical not-implemented status and a distinct exit code.

Registered onto the top-level app by :func:`register_monitor`; ``cli/main.py``
only wires the call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from benchflow.cli._options import MonitorJobsDirOption
from benchflow.cli._shared import console


def _monitor_not_implemented() -> None:
    """Emit the canonical not-implemented message and exit non-zero.

    Centralised so every ``bench monitor`` subcommand fails closed with the
    same wording and exit code, matching the issue's "fail closed with a
    clear not-implemented status" requirement.
    """
    from benchflow.monitor import not_implemented_message

    console.print("[yellow]Monitor mode not implemented yet.[/yellow]")
    console.print(not_implemented_message())
    # Exit code 2 (not 1) — distinguishes "feature absent" from "feature
    # ran and failed", so CI dashboards do not conflate the two.
    raise typer.Exit(2)


def register_monitor(app: typer.Typer) -> None:
    """Attach the ``monitor`` command group to the top-level benchflow app."""
    monitor_app = typer.Typer(
        help=(
            "Monitor mode — score a rollout in production (#386). "
            "API surface scaffold; runtime not yet implemented."
        ),
    )
    app.add_typer(monitor_app, name="monitor")

    @monitor_app.command("run")
    def monitor_run(
        source: Annotated[
            str,
            typer.Argument(
                help="Source trajectory (persisted rollout dir, file, or URI)."
            ),
        ],
        rubric: Annotated[
            Path | None,
            typer.Option(
                "--rubric", help="Rubric/verifier definition to score against."
            ),
        ] = None,
        jobs_dir: MonitorJobsDirOption = "jobs/monitor",
        run_name: Annotated[
            str | None,
            typer.Option("--run-name", help="Human-readable id for this monitor run."),
        ] = None,
    ) -> None:
        """Score one trajectory under monitor semantics. **Not yet implemented (#386).**"""
        del source, rubric, jobs_dir, run_name  # accepted for API stability
        _monitor_not_implemented()

    @monitor_app.command("replay")
    def monitor_replay(
        trajectory_path: Annotated[
            Path,
            typer.Argument(help="Path to a persisted rollout/trajectory to re-score."),
        ],
        jobs_dir: MonitorJobsDirOption = "jobs/monitor",
    ) -> None:
        """Re-score a persisted rollout under monitor semantics. **Not yet implemented (#386).**"""
        del trajectory_path, jobs_dir  # accepted for API stability
        _monitor_not_implemented()

    @monitor_app.command("watch")
    def monitor_watch(
        source: Annotated[
            str,
            typer.Argument(
                help="Live event source (webhook, polling endpoint, queue)."
            ),
        ],
        jobs_dir: MonitorJobsDirOption = "jobs/monitor",
    ) -> None:
        """Stream-score live production events. **Not yet implemented (#386).**"""
        del source, jobs_dir  # accepted for API stability
        _monitor_not_implemented()
