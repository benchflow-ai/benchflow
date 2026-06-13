"""Shared CLI reporting for inbound adapter diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import typer
from rich.markup import escape

from benchflow.adapters.inbound import UnsupportedInboundTaskError
from benchflow.cli._shared import console


def unsupported_adapter_task_payload(
    task_dir: Path,
    error: UnsupportedInboundTaskError,
) -> dict[str, object]:
    """Machine-readable unsupported-task payload for adapter adoption loops."""
    report = error.report
    return {
        "status": "unsupported-adapter-task",
        "task": str(task_dir),
        "task_name": task_dir.name,
        "adapter": report.source,
        "task_id": report.task_id,
        "dataset": report.dataset,
        "reason": report.reason or "unsupported task shape",
        "details": report.details,
    }


def unsupported_adapter_task_or_exit(
    task_dir: Path,
    error: UnsupportedInboundTaskError,
    *,
    output_json: bool = False,
) -> NoReturn:
    """Print a structured unsupported-task report and exit non-zero."""
    if output_json:
        typer.echo(json.dumps(unsupported_adapter_task_payload(task_dir, error)))
        raise typer.Exit(1)

    report = error.report
    console.print(f"[red]✗[/red] {task_dir.name} — unsupported adapter task")
    console.print(f"  [yellow]→[/yellow] Adapter: {escape(report.source)}")
    if report.task_id:
        console.print(f"  [yellow]→[/yellow] Task: {escape(report.task_id)}")
    if report.dataset:
        console.print(f"  [yellow]→[/yellow] Dataset: {escape(report.dataset)}")
    reason = report.reason or "unsupported task shape"
    console.print(f"  [yellow]→[/yellow] Reason: {escape(reason)}")
    tags = report.details.get("tags")
    if isinstance(tags, list) and tags:
        tag_list = ", ".join(str(tag) for tag in tags)
        console.print(f"  [yellow]→[/yellow] Tags: {escape(tag_list)}")
    for key, value in report.details.items():
        if key == "tags" or value in (None, "", [], {}):
            continue
        label = key.replace("_", " ").title()
        console.print(
            f"  [yellow]→[/yellow] {escape(label)}: {escape(_format_detail(value))}"
        )
    raise typer.Exit(1)


def _format_detail(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}={val}" for key, val in sorted(value.items()))
    return str(value)
