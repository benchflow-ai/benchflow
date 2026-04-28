"""``bench tasks {init,check}`` — task authoring helpers.

Extracted from cli/main.py per PLAN_V2_impl §13.4 / §13.6 commit 8b.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()

tasks_app = typer.Typer(help="Task authoring commands")


@tasks_app.command("init")
def tasks_init(
    name: Annotated[str, typer.Argument(help="Task name")],
    parent_dir: Annotated[
        Path,
        typer.Option("--dir", "-p", help="Parent directory (default: tasks/)"),
    ] = Path("tasks"),
    no_pytest: Annotated[
        bool, typer.Option("--no-pytest", help="Skip pytest template")
    ] = False,
    no_solution: Annotated[
        bool, typer.Option("--no-solution", help="Skip solution template")
    ] = False,
) -> None:
    """Scaffold a new benchmark task."""
    from benchflow._utils.task_authoring import init_task

    try:
        task_dir = init_task(
            name, parent_dir=parent_dir, no_pytest=no_pytest, no_solution=no_solution
        )
        console.print(f"[green]Created:[/green] {task_dir}/")
        console.print(
            "  task.toml, instruction.md, environment/Dockerfile, tests/test.sh"
        )
        if not no_solution:
            console.print("  solution/solve.sh")
    except FileExistsError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None


@tasks_app.command("check")
def tasks_check(
    task_dir: Annotated[Path, typer.Argument(help="Path to task directory")],
) -> None:
    """Validate a task directory structure."""
    from benchflow._utils.task_authoring import check_task

    issues = check_task(task_dir)
    if not issues:
        console.print(f"[green]✓[/green] {task_dir.name} — valid")
    else:
        console.print(f"[red]✗[/red] {task_dir.name} — {len(issues)} issue(s):")
        for issue in issues:
            console.print(f"  [yellow]→[/yellow] {issue}")
        raise typer.Exit(1)
