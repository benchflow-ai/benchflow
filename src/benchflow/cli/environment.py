"""``bench environment`` — environment management commands.

Covers local Daytona sandbox lifecycle (create / list / cleanup) and the
read-only hosted-environment hub views (list / show / inspect).

Registered onto the top-level app by :func:`register_environment`;
``cli/main.py`` only wires the call. The Daytona client + cleanup helpers
deliberately live in ``cli/main.py`` (``_daytona_client_or_exit`` /
``_cleanup_daytona_sandboxes``); these commands resolve them through the
``benchflow.cli.main`` module so tests that monkeypatch those names on
``cli.main`` keep working.
"""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from benchflow.cli._options import SandboxOption
from benchflow.cli._shared import console, warn_deprecated


def register_environment(app: typer.Typer) -> None:
    """Attach the ``environment`` command group to the top-level benchflow app."""
    env_app = typer.Typer(help="Environment management commands.")
    app.add_typer(env_app, name="environment", rich_help_panel="Environments")

    @env_app.command("create")
    def environment_create(
        task_dir: Annotated[
            Path,
            typer.Argument(
                help="Task directory with task.md or task.toml + Dockerfile"
            ),
        ],
        sandbox: SandboxOption = "daytona",
    ) -> None:
        """Create an environment from a task directory (does not start it)."""
        from benchflow.runtime import Environment

        if not task_dir.is_dir():
            console.print(f"[red]Not a directory: {escape(str(task_dir))}[/red]")
            raise typer.Exit(1)
        try:
            env = Environment.from_task(task_dir, sandbox=sandbox)
        except (FileNotFoundError, NotADirectoryError, ValueError) as e:
            # An existing dir with no task document reaches Task.__init__'s
            # unguarded read_text() — surface a clean error instead of a raw
            # FileNotFoundError traceback ending at instruction.md/task.md.
            console.print(
                f"[red]Not a valid task directory {escape(str(task_dir))}:[/red] "
                f"{escape(str(e))}"
            )
            raise typer.Exit(1) from None
        console.print(f"[green]Environment created:[/green] {env}")
        console.print(f"  Task:    {env.task_path}")
        console.print(f"  Sandbox: {env.sandbox}")
        console.print(
            "  Use [cyan]bench eval create[/cyan] for CLI runs, or pass to [cyan]bf.run()[/cyan]"
        )

    @env_app.command("list")
    def environment_list(
        provider: Annotated[
            str | None,
            typer.Option(
                "--provider",
                help="Hosted environment provider to list (e.g. primeintellect)",
            ),
        ] = None,
        hub: Annotated[
            str | None,
            typer.Option("--hub", hidden=True, help="[deprecated] use --provider"),
        ] = None,
        owner: Annotated[
            str | None,
            typer.Option("--owner", help="Hosted provider owner/namespace filter"),
        ] = None,
        search: Annotated[
            str | None,
            typer.Option("--search", help="Hosted provider search query"),
        ] = None,
        limit: Annotated[
            int | None,
            typer.Option("--limit", help="Maximum hosted provider results"),
        ] = None,
        output_json: Annotated[
            bool,
            typer.Option("--json", help="Emit raw JSON for hosted provider results"),
        ] = False,
    ) -> None:
        """List active Daytona sandboxes or hosted provider environments."""
        from datetime import datetime

        from benchflow.cli import main as cli_main

        # --hub is the deprecated spelling of --provider (kept through 0.6).
        if hub is not None:
            warn_deprecated(
                "bench environment list --hub", "bench environment list --provider"
            )
            if provider is None:
                provider = hub

        if provider:
            if provider != "primeintellect":
                console.print(
                    "[red]Only --provider primeintellect is supported today[/red]"
                )
                raise typer.Exit(1)
            from benchflow.hosted_env import HostedEnvError, prime_env_list

            try:
                raw = prime_env_list(owner=owner, search=search, limit=limit)
            except HostedEnvError as e:
                console.print(f"[red]{escape(str(e))}[/red]")
                raise typer.Exit(1) from None
            if output_json:
                console.print(raw)
                return
            data = json.loads(raw)
            rows = (
                data
                if isinstance(data, list)
                else data.get("environments", data.get("items", []))
            )
            table = Table(title="PrimeIntellect Environments")
            table.add_column("Environment", style="cyan")
            table.add_column("Version", style="green")
            table.add_column("Visibility")
            table.add_column("Updated", style="dim")
            for item in rows:
                name = (
                    item.get("environment")
                    or item.get("fullName")
                    or item.get("name")
                    or item.get("id")
                    or ""
                )
                version = str(item.get("version") or item.get("latestVersion") or "")
                visibility = str(item.get("visibility") or item.get("private") or "")
                updated = str(item.get("updated_at") or item.get("updatedAt") or "")
                table.add_row(name, version, visibility, updated)
            console.print(table)
            return

        d = cli_main._daytona_client_or_exit()
        table = Table(title="Active Sandboxes")
        table.add_column("ID", style="cyan")
        table.add_column("State", style="green")
        table.add_column("Age")
        table.add_column("Target")

        now = datetime.now(UTC)
        total = 0
        # daytona SDK >=0.18: ``list()`` yields an auto-paginating
        # Iterator[Sandbox] (was a paged ``list(page=, limit=)`` -> page object
        # with ``.items``).
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

    @env_app.command("show")
    def environment_show(
        source_env: Annotated[
            str,
            typer.Argument(
                help="Hosted environment (e.g. primeintellect/general-agent)"
            ),
        ],
        version: Annotated[
            str | None,
            typer.Option("--version", help="Hosted environment version"),
        ] = None,
    ) -> None:
        """Show hosted environment metadata."""
        from benchflow.hosted_env import HostedEnvError, HostedEnvRef, prime_env_info

        try:
            ref = HostedEnvRef.parse(source_env, version=version)
            console.print(prime_env_info(ref))
        except HostedEnvError as e:
            console.print(f"[red]{escape(str(e))}[/red]")
            raise typer.Exit(1) from None

    @env_app.command("inspect")
    def environment_inspect(
        source_env: Annotated[
            str,
            typer.Argument(
                help="Hosted environment (e.g. primeintellect/general-agent)"
            ),
        ],
        version: Annotated[
            str | None,
            typer.Option("--version", help="Hosted environment version"),
        ] = None,
        path: Annotated[
            str,
            typer.Option("--path", help="File inside the hosted environment package"),
        ] = "README.md",
    ) -> None:
        """Inspect a file from a hosted environment package."""
        from benchflow.hosted_env import (
            HostedEnvError,
            HostedEnvRef,
            prime_env_inspect,
        )

        try:
            ref = HostedEnvRef.parse(source_env, version=version)
            console.print(prime_env_inspect(ref, path=path))
        except HostedEnvError as e:
            console.print(f"[red]{escape(str(e))}[/red]")
            raise typer.Exit(1) from None

    @env_app.command("cleanup")
    def environment_cleanup(
        dry_run: Annotated[
            bool,
            typer.Option("--dry-run", help="List sandboxes without deleting"),
        ] = False,
        max_age_minutes: Annotated[
            int,
            typer.Option("--max-age", help="Delete sandboxes older than N minutes"),
        ] = 1440,
    ) -> None:
        """Clean up orphaned Daytona sandboxes."""
        from benchflow.cli import main as cli_main

        cli_main._cleanup_daytona_sandboxes(
            dry_run=dry_run, max_age_minutes=max_age_minutes
        )
