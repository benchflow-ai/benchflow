"""``bench environment {create,list}`` — environment management.

Extracted from cli/main.py per PLAN_V2_impl §13.4 / §13.6 commit 10b.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

env_app = typer.Typer(help="Environment management commands.")


@env_app.command("create")
def environment_create(
    task_dir: Annotated[
        Path,
        typer.Argument(help="Task directory with task.toml + environment/Dockerfile"),
    ],
    backend: Annotated[
        str,
        typer.Option("--backend", "-b", help="Backend: docker or daytona"),
    ] = "daytona",
) -> None:
    """Create an environment from a task directory (does not start it)."""
    from benchflow.api import Environment

    env = Environment.from_task(task_dir, backend=backend)
    console.print(f"[green]Environment created:[/green] {env}")
    console.print(f"  Task:    {env.task_path}")
    console.print(f"  Backend: {env.backend}")
    console.print(
        "  Use [cyan]bench environment start[/cyan] to launch, or pass to [cyan]bf.run()[/cyan]"
    )


@env_app.command("list")
def environment_list() -> None:
    """List active Daytona sandboxes."""
    try:
        from daytona import Daytona
    except ImportError:
        console.print("[red]daytona SDK not installed[/red]")
        raise typer.Exit(1) from None

    d = Daytona()
    table = Table(title="Active Sandboxes")
    table.add_column("ID", style="cyan")
    table.add_column("State", style="green")
    table.add_column("Age")
    table.add_column("Target")

    page = 1
    now = datetime.now(UTC)
    total = 0
    while True:
        result = d.list(page=page, limit=50)
        if not result.items:
            break
        for sb in result.items:
            total += 1
            age = ""
            if sb.created_at:
                created = datetime.fromisoformat(sb.created_at.replace("Z", "+00:00"))
                mins = (now - created).total_seconds() / 60
                age = f"{mins:.0f}m"
            target = getattr(sb, "target", "") or ""
            table.add_row(sb.id[:12] + "…", str(sb.state), age, str(target)[:40])
        if len(result.items) < 50:
            break
        page += 1

    console.print(table)
    console.print(f"\n[bold]{total} sandbox(es)[/bold]")
