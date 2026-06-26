"""``bench train`` — training data conversion commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.markup import escape

from benchflow.cli._shared import console, print_error


def _ensure_prime_sft(format_name: str) -> None:
    if format_name != "prime-sft":
        print_error("--format currently supports only 'prime-sft'")
        raise typer.Exit(1)


def register_train(app: typer.Typer) -> None:
    """Attach the ``train`` command group to the top-level benchflow app."""
    train_app = typer.Typer(help="Training data commands.")
    app.add_typer(train_app, name="train", rich_help_panel="Core")

    @train_app.command("convert")
    def train_convert(
        jobs_dir: Annotated[
            Path,
            typer.Argument(help="BenchFlow rollout or jobs directory"),
        ],
        output: Annotated[
            Path,
            typer.Option("--out", "-o", help="Output JSONL path"),
        ],
        format_name: Annotated[
            str,
            typer.Option("--format", help="Trainer format"),
        ] = "prime-sft",
        min_reward: Annotated[
            float | None,
            typer.Option("--min-reward", help="Only include rows with reward >= value"),
        ] = None,
        row_mode: Annotated[
            Literal["rollout", "exchange"],
            typer.Option(
                "--row-mode",
                help="rollout writes one row per rollout; exchange writes one row per LLM exchange",
            ),
        ] = "rollout",
        manifest: Annotated[
            Path | None,
            typer.Option("--manifest", help="Optional conversion stats JSON path"),
        ] = None,
        expected_rows: Annotated[
            int | None,
            typer.Option(
                "--expected-rows",
                help=(
                    "Fail (before writing the output file) unless exactly this "
                    "many rows would be exported"
                ),
            ),
        ] = None,
    ) -> None:
        """Convert BenchFlow rollout artifacts into trainer-ready data."""
        _ensure_prime_sft(format_name)
        from benchflow.trajectories.export_prime_sft import export_prime_sft_jsonl

        try:
            stats = export_prime_sft_jsonl(
                jobs_dir,
                output,
                min_reward=min_reward,
                row_mode=row_mode,
                expected_rows=expected_rows,
                manifest=manifest,
            )
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(1) from None

        console.print(
            f"[green]Converted {stats.rows_written} row(s)[/green] "
            f"from {stats.rollouts_seen} rollout(s) -> {escape(str(output))}"
        )
        if manifest is not None:
            console.print(f"Stats: {escape(str(manifest))}")

    @train_app.command("validate")
    def train_validate(
        jsonl: Annotated[
            Path,
            typer.Argument(help="Trainer JSONL path to validate"),
        ],
        format_name: Annotated[
            str,
            typer.Option("--format", help="Trainer format"),
        ] = "prime-sft",
        expected_rows: Annotated[
            int | None,
            typer.Option(
                "--expected-rows", help="Fail unless this many rows are present"
            ),
        ] = None,
    ) -> None:
        """Validate trainer-ready data."""
        _ensure_prime_sft(format_name)
        from benchflow.trajectories.export_prime_sft import validate_prime_sft_jsonl

        try:
            result = validate_prime_sft_jsonl(jsonl, expected_rows=expected_rows)
        except (OSError, ValueError) as exc:
            print_error(str(exc))
            raise typer.Exit(1) from None
        console.print(json.dumps(result, sort_keys=True))
