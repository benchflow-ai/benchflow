"""CLI registration for paired eval lift reports."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from benchflow.cli._shared import console, print_error
from benchflow.eval_lift import write_lift_report


def register_eval_lift(eval_app: typer.Typer) -> None:
    @eval_app.command("compare-lift")
    def eval_compare_lift(
        baseline: Annotated[
            Path,
            typer.Option(
                "--baseline",
                exists=True,
                file_okay=False,
                dir_okay=True,
                readable=True,
                help="Baseline BenchFlow job directory.",
            ),
        ],
        trained: Annotated[
            Path,
            typer.Option(
                "--trained",
                exists=True,
                file_okay=False,
                dir_okay=True,
                readable=True,
                help="Trained BenchFlow job directory.",
            ),
        ],
        out: Annotated[
            Path,
            typer.Option("--out", help="Markdown report output path."),
        ],
        json_out: Annotated[
            Path,
            typer.Option("--json-out", help="JSON report output path."),
        ],
        bootstrap_samples: Annotated[
            int,
            typer.Option(
                "--bootstrap-samples",
                min=0,
                help="Bootstrap samples for paired delta confidence intervals.",
            ),
        ] = 1000,
        bootstrap_seed: Annotated[
            int,
            typer.Option(
                "--bootstrap-seed",
                help="Deterministic seed for bootstrap confidence intervals.",
            ),
        ] = 0,
        allow_duplicate_first_by_path: Annotated[
            bool,
            typer.Option(
                "--allow-duplicate-first-by-path",
                help=(
                    "Allow duplicate healthy task rollouts and keep the first by "
                    "deterministic path order."
                ),
            ),
        ] = False,
    ) -> None:
        """Compare paired baseline vs trained eval job directories."""

        try:
            report = write_lift_report(
                baseline_jobs_dir=baseline,
                trained_jobs_dir=trained,
                markdown_path=out,
                json_path=json_out,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                allow_duplicate_first_by_path=allow_duplicate_first_by_path,
            )
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(1) from None
        paired = report["pairing"]["paired_tasks_count"]
        console.print(f"[green]Paired healthy tasks:[/green] {paired}")
        console.print(
            f"[green]Lift report:[/green] {escape(str(out))} "
            f"({paired} paired healthy task(s))"
        )
        console.print(f"[green]Lift JSON:[/green] {escape(str(json_out))}")
