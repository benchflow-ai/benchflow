"""``bench traj upload`` — contribute validated trajectory captures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from benchflow.cli._shared import console, print_error

# The environment variable remains an override for development and disaster
# recovery.
DEFAULT_TRAJ_BROKER_URL: str | None = (
    "https://tasksminer-traj-broker.nicewave-c3abaecf.westus2.azurecontainerapps.io"
)


def register_traj(app: typer.Typer) -> None:
    """Attach the trajectory contribution group to the top-level app."""
    traj_app = typer.Typer(help="Trajectory commands.")
    app.add_typer(traj_app, name="traj", rich_help_panel="Core")

    @traj_app.command("upload")
    def upload(
        path: Annotated[
            Path,
            typer.Argument(help="Trajectory JSONL file, directory, or trial directory"),
        ],
        source_id: Annotated[
            str | None,
            typer.Option("--source-id", help="Stable contributor source identifier"),
        ] = None,
        direct: Annotated[
            bool,
            typer.Option("--direct", help="Upload with local Azure credentials"),
        ] = False,
        container_url: Annotated[
            str | None,
            typer.Option("--container-url", help="Azure container URL for --direct"),
        ] = None,
        dry_run: Annotated[
            bool,
            typer.Option("--dry-run", help="Validate and stage without uploading"),
        ] = False,
    ) -> None:
        """Validate, redact, and upload trajectory JSONL."""
        from benchflow.publish.azure_blob import upload_capture_direct
        from benchflow.publish.broker import upload_capture_via_broker
        from benchflow.publish.traj_capture import (
            default_source_id,
            stage_trajectory_capture,
        )

        try:
            selected_source_id = source_id or default_source_id(path)
            if direct:
                destination = container_url or os.environ.get(
                    "BENCHFLOW_AZURE_CONTAINER_URL"
                )
                if not destination:
                    raise ValueError(
                        "--direct requires --container-url or "
                        "BENCHFLOW_AZURE_CONTAINER_URL"
                    )
            else:
                if container_url:
                    raise ValueError("--container-url is only valid with --direct")
                destination = (
                    os.environ.get("BENCHFLOW_TRAJ_BROKER_URL")
                    or DEFAULT_TRAJ_BROKER_URL
                )
                if not destination:
                    raise ValueError(
                        "no trajectory broker is configured; set "
                        "BENCHFLOW_TRAJ_BROKER_URL, or use --direct with "
                        "--container-url/BENCHFLOW_AZURE_CONTAINER_URL if you have "
                        "Azure credentials"
                    )

            with stage_trajectory_capture(
                path,
                source_id=selected_source_id,
                uploaded_by=os.environ.get("BENCHFLOW_TRAJ_UPLOADED_BY"),
            ) as staged:
                if dry_run:
                    _print_dry_run(staged, destination=destination, direct=direct)
                    return
                if direct:
                    result = upload_capture_direct(
                        staged,
                        container_url=destination,
                    )
                else:
                    result = upload_capture_via_broker(
                        staged,
                        broker_url=destination,
                    )
                size = sum(item.size_bytes for item in staged.files)
                if result.uploaded:
                    console.print(
                        "[green]Uploaded trajectory:[/green] "
                        f"{escape(result.url)} "
                        f"({len(result.uploaded)} uploaded, {len(result.skipped)} skipped, "
                        f"{_format_bytes(size)}, "
                        f"{staged.redaction_replacements} redactions)"
                    )
                else:
                    console.print(
                        f"[green]Already uploaded:[/green] {escape(result.url)} (no-op)"
                    )
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(1) from None


def _print_dry_run(staged, *, destination: str, direct: bool) -> None:
    mode = "Azure container" if direct else "broker"
    console.print(f"[bold]Dry run[/bold] — {mode}: {escape(destination)}")
    console.print(f"Digest: sha256:{staged.traj_digest}")
    for staged_file in staged.files:
        console.print(
            f"  {escape(staged_file.relname)} ({_format_bytes(staged_file.size_bytes)})"
        )
    if staged.ignored:
        console.print(f"Ignored: {escape(', '.join(staged.ignored))}")
    console.print(f"Redactions: {staged.redaction_replacements}")


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")
