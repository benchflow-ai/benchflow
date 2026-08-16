"""``bench traj upload`` — contribute validated trajectory captures."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol

import typer
from rich.markup import escape

from benchflow.cli._shared import console, print_error
from benchflow.cli._traj_upload_ui import (
    format_bytes,
    render_trajectory_report,
    upload_progress,
)
from benchflow.publish.traj_capture import StagedCapture, StagedFile
from benchflow.publish.traj_report import DEFAULT_PREVIEW_STEPS, MAX_PREVIEW_STEPS

# The environment variable remains an override for development and disaster
# recovery.
DEFAULT_TRAJ_BROKER_URL: str | None = (
    "https://tasksminer-traj-broker.nicewave-c3abaecf.westus2.azurecontainerapps.io"
)


@dataclass(frozen=True)
class _UploadOptions:
    path: Path | None
    github_id: str | None
    email: str | None
    source_id: str | None
    direct: bool
    container_url: str | None
    dry_run: bool
    preview_steps: int


@dataclass(frozen=True)
class _UploadDestination:
    url: str
    direct: bool


class _PublishResult(Protocol):
    @property
    def url(self) -> str: ...

    uploaded: tuple[str, ...]
    skipped: tuple[str, ...]


def register_traj(app: typer.Typer) -> None:
    """Attach the trajectory contribution group to the top-level app."""
    traj_app = typer.Typer(help="Trajectory commands.")
    app.add_typer(traj_app, name="traj", rich_help_panel="Core")

    @traj_app.command("upload")
    def upload(
        path: Annotated[
            Path | None,
            typer.Argument(help="Trajectory JSONL file, directory, or trial directory"),
        ] = None,
        github_id: Annotated[
            str | None,
            typer.Option("--github-id", help="Contributor GitHub username"),
        ] = None,
        email: Annotated[
            str | None,
            typer.Option("--email", help="Contributor email stored in the manifest"),
        ] = None,
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
        preview_steps: Annotated[
            int,
            typer.Option(
                "--preview-steps",
                min=0,
                max=MAX_PREVIEW_STEPS,
                help="Number of redacted trajectory steps to preview",
            ),
        ] = DEFAULT_PREVIEW_STEPS,
    ) -> None:
        """Inspect, redact, confirm, and upload trajectory JSONL."""
        try:
            _run_upload(
                _UploadOptions(
                    path=path,
                    github_id=github_id,
                    email=email,
                    source_id=source_id,
                    direct=direct,
                    container_url=container_url,
                    dry_run=dry_run,
                    preview_steps=preview_steps,
                )
            )
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(1) from None


def _run_upload(options: _UploadOptions) -> None:
    from benchflow.publish.traj_capture import (
        default_source_id,
        finalize_trajectory_capture,
        stage_trajectory_artifacts,
    )
    from benchflow.publish.traj_report import build_trajectory_report

    interactive = any(
        value is None for value in (options.path, options.github_id, options.email)
    )
    path = options.path or Path(
        typer.prompt("Trajectory JSONL file or trial directory")
    )
    source_id = options.source_id or default_source_id(path)
    destination = _resolve_destination(options)

    with (
        console.status(
            "[bold cyan]Inspecting trajectory and masking key values…"
        ) as status,
        stage_trajectory_artifacts(path, source_id=source_id) as artifacts,
    ):
        report = build_trajectory_report(
            artifacts.files,
            masked_values=artifacts.redaction_replacements,
            preview_steps=options.preview_steps,
        )
        status.stop()
        render_trajectory_report(report, console=console)

        github_id, email = _resolve_contributor(options.github_id, options.email)
        staged = finalize_trajectory_capture(
            artifacts,
            uploaded_by=os.environ.get("BENCHFLOW_TRAJ_UPLOADED_BY"),
            github_id=github_id,
            email=email,
        )
        if options.dry_run:
            _print_dry_run(staged)
            return
        if interactive and not typer.confirm(
            "Upload this trajectory?",
            default=False,
        ):
            console.print("[yellow]Upload cancelled.[/yellow]")
            return
        with upload_progress(staged.files, console=console) as on_file_complete:
            result = _publish(
                staged,
                destination=destination,
                on_file_complete=on_file_complete,
            )
        _print_upload_result(staged, result)


def _resolve_contributor(
    github_id: str | None,
    email: str | None,
) -> tuple[str, str]:
    return (
        github_id or typer.prompt("GitHub ID"),
        email or typer.prompt("Email"),
    )


def _resolve_destination(options: _UploadOptions) -> _UploadDestination:
    if options.direct:
        destination = options.container_url or os.environ.get(
            "BENCHFLOW_AZURE_CONTAINER_URL"
        )
        if not destination:
            raise ValueError(
                "--direct requires --container-url or BENCHFLOW_AZURE_CONTAINER_URL"
            )
        return _UploadDestination(url=destination, direct=True)
    if options.container_url:
        raise ValueError("--container-url is only valid with --direct")
    destination = os.environ.get("BENCHFLOW_TRAJ_BROKER_URL") or DEFAULT_TRAJ_BROKER_URL
    if not destination:
        raise ValueError(
            "no trajectory broker is configured; set BENCHFLOW_TRAJ_BROKER_URL, "
            "or use --direct with --container-url/BENCHFLOW_AZURE_CONTAINER_URL "
            "if you have Azure credentials"
        )
    return _UploadDestination(url=destination, direct=False)


def _publish(
    staged: StagedCapture,
    *,
    destination: _UploadDestination,
    on_file_complete: Callable[[StagedFile], None],
) -> _PublishResult:
    if destination.direct:
        from benchflow.publish.azure_blob import upload_capture_direct

        return upload_capture_direct(
            staged,
            container_url=destination.url,
            on_file_complete=on_file_complete,
        )
    from benchflow.publish.broker import upload_capture_via_broker

    return upload_capture_via_broker(
        staged,
        broker_url=destination.url,
        on_file_complete=on_file_complete,
    )


def _print_upload_result(staged: StagedCapture, result: _PublishResult) -> None:
    if not result.uploaded:
        console.print(f"[green]Already uploaded:[/green] {escape(result.url)} (no-op)")
        return
    size = sum(item.size_bytes for item in staged.files)
    console.print(
        "[green]Uploaded trajectory:[/green] "
        f"{escape(result.url)} "
        f"({len(result.uploaded)} uploaded, {len(result.skipped)} skipped, "
        f"{format_bytes(size)}, {staged.redaction_replacements} redactions)"
    )


def _print_dry_run(staged: StagedCapture) -> None:
    console.print("[bold]Dry run[/bold] — no files uploaded")
    console.print(f"Digest: sha256:{staged.traj_digest}")
    for staged_file in staged.files:
        console.print(
            f"  {escape(staged_file.relname)} ({format_bytes(staged_file.size_bytes)})"
        )
    if staged.ignored:
        console.print(f"Ignored: {escape(', '.join(staged.ignored))}")
    console.print(f"Redactions: {staged.redaction_replacements}")
