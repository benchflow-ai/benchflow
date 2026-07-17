"""``bench sandbox`` — local sandbox lifecycle (create / list / cleanup).

This is the local execution side of the framework: provision a task as a
runnable environment on a docker/daytona/modal **sandbox** backend, list active
sandboxes, and reap stale ones. It was previously ``bench environment``; that
name now reads as a misnomer (hosted-environment browsing moved to
``bench hub``), so the group is renamed to ``sandbox`` — ``bench
environment`` stays as a hidden deprecated alias group through 0.6.

The command bodies live here as plain functions so the deprecated
``bench environment`` aliases (``cli/environment.py``) can delegate to the same
logic without a fork. The Daytona client + reaper deliberately resolve through
``benchflow.cli.main`` so tests that monkeypatch those names keep working.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from benchflow.cli._options import SandboxOption
from benchflow.cli._shared import console, print_error


def sandbox_create(task_dir: Path, sandbox: str) -> None:
    """Create an environment object from a task directory (does not start it)."""
    from benchflow.runtime import Environment

    if not task_dir.is_dir():
        print_error(f"Not a directory: {task_dir}")
        raise typer.Exit(1)
    try:
        env = Environment.from_task(task_dir, sandbox=sandbox)
    except (OSError, ValueError) as e:
        # An existing dir with no task document — or one where task.md/task.toml
        # is itself a directory — reaches Task's unguarded read_text(), raising
        # FileNotFoundError / IsADirectoryError (both OSError). Surface a clean
        # error instead of a raw traceback.
        print_error(f"Not a valid task directory {task_dir}: {e}")
        raise typer.Exit(1) from None
    except RuntimeError as e:
        # An unknown --sandbox backend (UnsupportedTaskFeatureError, a RuntimeError
        # subclass) and a missing optional sandbox dependency both raise a
        # RuntimeError carrying a clean, user-facing message. Surface it without a
        # traceback, matching how `sandbox list`/`cleanup` handle the same cases.
        print_error(str(e))
        raise typer.Exit(1) from None
    console.print(f"[green]Environment created:[/green] {escape(str(env))}")
    console.print(f"  Task:    {env.task_path}")
    console.print(f"  Sandbox: {env.sandbox}")
    console.print(
        "  Use [cyan]bench eval run[/cyan] for CLI runs, or pass to [cyan]bf.run()[/cyan]"
    )


def _daytona_sdk_available() -> bool:
    """True if the optional Daytona SDK can be imported.

    A plain import rather than ``importlib.util.find_spec``: the test suite
    injects a fake ``daytona`` module into ``sys.modules`` that has no
    ``__spec__``, which makes ``find_spec`` raise/return None. An import sees the
    fake (and a real install) alike.

    The anyio compat shim is applied first — the Daytona sync client imports
    ``anyio.AsyncContextManagerMixin`` at import time, which the pinned anyio may
    not expose. Without it a *real* install would look absent here (and so would
    ``build_sync_client`` further down). Mirrors that bootstrap.
    """
    try:
        from benchflow.sandbox.daytona import _ensure_daytona_anyio_compat

        _ensure_daytona_anyio_compat()
        import daytona  # noqa: F401

        return True
    except ImportError:
        return False


def sandbox_list_local() -> None:
    """List active off-box sandboxes (Daytona).

    Daytona is the only backend with persistent, listable sandboxes; Docker
    sandboxes are ephemeral (built and torn down per run). When the optional
    Daytona SDK is not installed there is nothing to list — an empty result, not
    an error (mirroring how ``sandbox create`` degrades on a missing extra).
    """
    if not _daytona_sdk_available():
        console.print(
            "No active sandboxes. Daytona is the only backend with persistent, "
            "listable sandboxes, and its SDK is not installed "
            "([cyan]uv sync --extra sandbox-daytona[/cyan]). Docker sandboxes are "
            "ephemeral and created per run."
        )
        return
    from benchflow.cli import main as cli_main

    d = cli_main._daytona_client_or_exit()
    table = Table(title="Active Sandboxes")
    table.add_column("ID", style="cyan")
    table.add_column("State", style="green")
    table.add_column("Age")
    table.add_column("Target")

    now = datetime.now(UTC)
    total = 0
    # daytona SDK >=0.18: ``list()`` yields an auto-paginating Iterator[Sandbox].
    for sb in d.list():
        total += 1
        age = ""
        if sb.created_at:
            created = datetime.fromisoformat(sb.created_at.replace("Z", "+00:00"))
            mins = (now - created).total_seconds() / 60
            age = f"{mins:.0f}m"
        target = getattr(sb, "target", "") or ""
        table.add_row(sb.id[:12] + "…", str(sb.state), age, str(target)[:40])

    console.print(table)
    console.print(f"\n[bold]{total} sandbox(es)[/bold]")


def sandbox_cleanup(*, dry_run: bool, max_age_minutes: int) -> None:
    """Clean up orphaned Daytona sandboxes.

    Like ``sandbox list``, this is a no-op (not an error) when the optional
    Daytona SDK is absent: only Daytona has persistent sandboxes to reap.
    """
    if not _daytona_sdk_available():
        console.print(
            "Nothing to clean up. The Daytona SDK is not installed "
            "([cyan]uv sync --extra sandbox-daytona[/cyan]); only Daytona has "
            "persistent sandboxes to reap. Docker sandboxes are torn down per run."
        )
        return
    from benchflow.cli import main as cli_main

    cli_main._cleanup_daytona_sandboxes(
        dry_run=dry_run, max_age_minutes=max_age_minutes
    )
    _cleanup_daytona_snapshots(dry_run=dry_run)


def _cleanup_daytona_snapshots(dry_run: bool) -> None:
    """Reap leaked ``bf-snap-*`` Daytona snapshots (display wrapper).

    The sandbox reaper never touches snapshots, so a snapshot whose owning
    sandbox's ``stop()`` never ran leaks against the account. Scoped by the
    ``bf-snap-`` name prefix benchflow stamps — Daytona snapshots have no labels.
    """
    from benchflow.cli import main as cli_main
    from benchflow.sandbox.daytona import reap_leaked_snapshots

    d = cli_main._daytona_client_or_exit()

    def _show(snap, will_delete):
        verdict = "[red](delete)[/red]" if will_delete else "[green](skip)[/green]"
        if dry_run or not will_delete:
            console.print(
                f"  [dim]{getattr(snap, 'name', '?')}[/dim] "
                f"state={getattr(snap, 'state', '?')} {verdict}"
            )

    counts = reap_leaked_snapshots(d, dry_run=dry_run, on_decision=_show)
    if dry_run:
        console.print(
            f"\n[bold]{counts['found']} snapshots found, "
            f"{counts['deleted']} benchflow-owned[/bold] "
            "(use without --dry-run to delete)"
        )
    else:
        console.print(
            f"\n[bold green]{counts['deleted']} snapshots deleted[/bold green] "
            f"({counts['skipped']} skipped, not benchflow-owned)"
        )


def register_sandbox(app: typer.Typer) -> None:
    """Attach the ``sandbox`` command group to the top-level benchflow app."""
    sandbox_app = typer.Typer(help="Local sandbox lifecycle (create / list / cleanup).")
    app.add_typer(sandbox_app, name="sandbox", rich_help_panel="Environments")

    @sandbox_app.command("create")
    def sandbox_create_cmd(
        task_dir: Annotated[
            Path,
            typer.Argument(
                help="Task directory with task.md or task.toml + Dockerfile"
            ),
        ],
        sandbox: SandboxOption = "daytona",
    ) -> None:
        """Create an environment from a task directory (does not start it)."""
        sandbox_create(task_dir, sandbox)

    @sandbox_app.command("list")
    def sandbox_list_cmd() -> None:
        """List active sandboxes (Daytona; Docker sandboxes are ephemeral)."""
        sandbox_list_local()

    @sandbox_app.command("cleanup")
    def sandbox_cleanup_cmd(
        dry_run: Annotated[
            bool, typer.Option("--dry-run", help="List sandboxes without deleting")
        ] = False,
        max_age_minutes: Annotated[
            int, typer.Option("--max-age", help="Delete sandboxes older than N minutes")
        ] = 1440,
    ) -> None:
        """Clean up orphaned Daytona sandboxes."""
        sandbox_cleanup(dry_run=dry_run, max_age_minutes=max_age_minutes)
