"""``bench compat`` — third-party framework compatibility checks.

Registered onto the top-level app by :func:`register_compat`; ``cli/main.py``
only wires the call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from benchflow.cli._shared import console


def register_compat(app: typer.Typer) -> None:
    """Attach the ``compat`` command group to the top-level benchflow app."""
    compat_app = typer.Typer(help="Third-party framework compatibility checks.")
    app.add_typer(compat_app, name="compat", rich_help_panel="Environments")

    @compat_app.command("harbor-registry")
    def compat_harbor_registry(
        registry: Annotated[
            str,
            typer.Option(
                "--registry",
                help="Harbor registry JSON URL or local file.",
            ),
        ] = "https://raw.githubusercontent.com/harbor-framework/harbor/main/registry.json",
        tasks_per_dataset: Annotated[
            int,
            typer.Option(
                "--tasks-per-dataset",
                help="Number of representative tasks to select per registry dataset.",
                min=1,
            ),
        ] = 2,
        level: Annotated[
            str,
            typer.Option(
                "--level",
                help="Compatibility level to run: inventory or check.",
            ),
        ] = "inventory",
        out: Annotated[
            Path | None,
            typer.Option("--out", help="Optional JSONL output path."),
        ] = None,
        cache_dir: Annotated[
            Path,
            typer.Option("--cache-dir", help="Cache directory for sparse clones."),
        ] = Path(".cache/compat/harbor"),
        limit: Annotated[
            int | None,
            typer.Option("--limit", help="Optional cap on selected task refs."),
        ] = None,
    ) -> None:
        """Inventory or structurally check representative Harbor registry tasks."""
        from benchflow.compat.harbor_registry import (
            check_harbor_registry,
            records_summary,
        )

        try:
            records = check_harbor_registry(
                registry,
                tasks_per_dataset=tasks_per_dataset,
                level=level,
                out=out,
                cache_dir=cache_dir,
                limit=limit,
            )
        except Exception as exc:
            console.print(f"[red]Harbor compatibility check failed:[/red] {exc}")
            raise typer.Exit(1) from exc

        summary = records_summary(records)
        console.print(
            "[bold]Harbor compatibility:[/bold] "
            f"{summary['total']} task refs, "
            f"{summary['pass']} pass, {summary['fail']} fail, "
            f"{summary['blocked']} blocked"
        )
        if out is not None:
            console.print(f"[green]Wrote JSONL report:[/green] {out}")
