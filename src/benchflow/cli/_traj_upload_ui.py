"""Rich terminal rendering for trajectory inspection and upload progress."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from benchflow.publish.redact import REDACTED
from benchflow.publish.traj_capture import StagedFile
from benchflow.publish.traj_report import PREVIEW_WORD_LIMIT, TrajectoryReport


def render_trajectory_report(report: TrajectoryReport, *, console: Console) -> None:
    """Render upload facts and a bounded, already-redacted trajectory preview."""
    facts = Table.grid(padding=(0, 2))
    facts.add_column(style="bold cyan", no_wrap=True)
    facts.add_column()
    facts.add_row("Primary trajectory", Text(report.primary_file))
    facts.add_row("Format", report.format.value)
    facts.add_row("Created", _format_created_at(report))
    facts.add_row("JSONL files", str(report.file_count))
    facts.add_row("Trajectory size", format_bytes(report.size_bytes))
    facts.add_row("Total steps", str(report.total_steps))
    facts.add_row("Thinking steps", str(report.thinking_steps))
    facts.add_row("Tool-call steps", str(report.tool_call_steps))
    facts.add_row("Human steps", str(report.human_steps))
    facts.add_row(
        "API keys / secrets masked",
        Text(str(report.masked_values), style="bold green"),
    )
    facts.add_row("Safe replacement", Text(REDACTED))
    console.print(Panel(facts, title="Trajectory report", border_style="cyan"))

    if not report.preview:
        return
    preview = Table(
        title=(
            f"First {len(report.preview)} trajectory steps "
            f"(up to {PREVIEW_WORD_LIMIT} words each)"
        ),
        show_lines=True,
        header_style="bold cyan",
    )
    preview.add_column("#", justify="right", style="dim", no_wrap=True)
    preview.add_column("Kind", no_wrap=True)
    preview.add_column("Preview", overflow="fold")
    for step in report.preview:
        preview.add_row(str(step.number), step.kind, Text(step.summary))
    console.print(preview)
    if len(report.preview) < report.total_steps:
        console.print(
            f"[dim]Showing {len(report.preview)} of {report.total_steps} steps. "
            "Use --preview-steps to change the preview.[/dim]"
        )


@dataclass(frozen=True)
class UploadProgressHooks:
    """Transport callbacks that feed one shared progress bar."""

    on_bytes: Callable[[int], None]
    on_file_complete: Callable[[StagedFile], None]


@contextmanager
def upload_progress(
    files: tuple[StagedFile, ...],
    *,
    console: Console,
) -> Iterator[UploadProgressHooks]:
    """Display streamed byte progress and yield the transport callbacks."""
    total_bytes = sum(item.size_bytes for item in files)
    streamed_bytes = 0
    boundary_bytes = 0
    progress = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(binary_units=True),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task_id = progress.add_task("Uploading trajectory", total=total_bytes)

        def on_bytes(count: int) -> None:
            nonlocal streamed_bytes
            streamed_bytes = min(streamed_bytes + count, total_bytes)
            progress.update(task_id, completed=streamed_bytes)

        def on_file_complete(staged_file: StagedFile) -> None:
            # Resync at each file boundary: a transport retry may re-read
            # bytes, so the boundary is authoritative for completed work.
            nonlocal boundary_bytes, streamed_bytes
            boundary_bytes = min(boundary_bytes + staged_file.size_bytes, total_bytes)
            streamed_bytes = boundary_bytes
            progress.update(
                task_id,
                completed=boundary_bytes,
                description=f"Uploaded {staged_file.relname}",
            )

        yield UploadProgressHooks(on_bytes=on_bytes, on_file_complete=on_file_complete)
        progress.update(
            task_id,
            completed=total_bytes,
            description="Upload complete",
        )


def _format_created_at(report: TrajectoryReport) -> str:
    local = report.created_at.astimezone()
    return f"{local:%Y-%m-%d %H:%M:%S %Z} ({report.created_at_source})"


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")
