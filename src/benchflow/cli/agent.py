"""``bench agent`` — agent management commands (list / show) plus the adoption
router subcommands wired from :mod:`benchflow.agent_router`.

Registered onto the top-level app by :func:`register_agent`; ``cli/main.py``
only wires the call.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.table import Table

from benchflow.agent_router import register_agent_router
from benchflow.cli._shared import (
    _PROVIDER_AUTH_MESSAGE,
    _REQUIRES_AUTH_NOTE,
    _format_requires,
    console,
)


def register_agent(app: typer.Typer) -> None:
    """Attach the ``agent`` command group to the top-level benchflow app."""
    agent_app = typer.Typer(help="Agent management commands.")
    app.add_typer(agent_app, name="agent", rich_help_panel="Core")
    register_agent_router(agent_app)

    @agent_app.command("list")
    def agent_list() -> None:
        """List all registered agents."""
        from benchflow.agents.registry import AGENT_ALIASES, list_agents

        # Build reverse map: canonical name -> list of aliases
        reverse_aliases: dict[str, list[str]] = {}
        for alias, canonical in AGENT_ALIASES.items():
            if alias != canonical:
                reverse_aliases.setdefault(canonical, []).append(alias)

        table = Table(title="Registered Agents")
        table.add_column("Name", style="cyan")
        table.add_column("Aliases", style="dim")
        table.add_column("Description")
        table.add_column("Protocol", style="green")
        table.add_column("Requires", style="yellow")

        for a in list_agents():
            aliases = ", ".join(sorted(reverse_aliases.get(a.name, [])))
            table.add_row(
                a.name, aliases, a.description, a.protocol, _format_requires(a)
            )

        console.print(table)
        console.print(f"[dim]{_REQUIRES_AUTH_NOTE}[/dim]")

    @agent_app.command("show")
    def agent_show(
        name: Annotated[str, typer.Argument(help="Agent name")],
    ) -> None:
        """Show details for a registered agent."""
        from benchflow.agents.registry import AGENT_ALIASES, AGENTS

        resolved = AGENT_ALIASES.get(name, name)
        cfg = AGENTS.get(resolved)
        if not cfg:
            console.print(f"[red]Unknown agent: {name}[/red]")
            raise typer.Exit(1)

        # Collect aliases that point to this agent
        aliases = sorted(
            a for a, c in AGENT_ALIASES.items() if c == cfg.name and a != cfg.name
        )

        console.print(f"[bold]{cfg.name}[/bold]")
        if aliases:
            console.print(f"  Aliases:     {', '.join(aliases)}")
        console.print(f"  Description: {cfg.description}")
        console.print(f"  Protocol:    {cfg.protocol}")
        console.print(f"  Launch:      {cfg.launch_cmd}")
        console.print(f"  Requires:    {_format_requires(cfg) or '(none)'}")
        console.print(f"  Provider auth: {_PROVIDER_AUTH_MESSAGE}")
        if cfg.subscription_auth:
            console.print(
                f"  Auth:        subscription via {cfg.subscription_auth.detect_file}"
            )
