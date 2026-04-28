"""``bench agent {list,show}`` — agent registry inspection.

Extracted from cli/main.py per PLAN_V2_impl §13.4 / §13.6 commit 8c.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

agent_app = typer.Typer(help="Agent management commands.")


@agent_app.command("list")
def agent_list() -> None:
    """List all registered agents."""
    from benchflow.agents.registry import list_agents

    table = Table(title="Registered Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Protocol", style="green")
    table.add_column("Requires", style="yellow")

    for a in list_agents():
        sub_env = a.subscription_auth.replaces_env if a.subscription_auth else None
        requires = [f"{e} (or login)" if e == sub_env else e for e in a.requires_env]
        table.add_row(a.name, a.description, a.protocol, ", ".join(requires))

    console.print(table)


@agent_app.command("show")
def agent_show(
    name: Annotated[str, typer.Argument(help="Agent name")],
) -> None:
    """Show details for a registered agent."""
    from benchflow.agents.registry import AGENTS

    cfg = AGENTS.get(name)
    if not cfg:
        console.print(f"[red]Unknown agent: {name}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]{cfg.name}[/bold]")
    console.print(f"  Description: {cfg.description}")
    console.print(f"  Protocol:    {cfg.protocol}")
    console.print(f"  Launch:      {cfg.launch_cmd}")
    console.print(f"  Requires:    {', '.join(cfg.requires_env) or '(none)'}")
    if cfg.subscription_auth:
        console.print(
            f"  Auth:        subscription via {cfg.subscription_auth.detect_file}"
        )


@agent_app.command("test")
def agent_test(
    name: Annotated[str, typer.Argument(help="Agent name")],
    level: Annotated[int, typer.Option("--level", "-l", help="0=version, 1=ping")] = 0,
    provider: Annotated[str, typer.Option("--provider")] = "",
    model: Annotated[str, typer.Option("--model")] = "",
) -> None:
    """Run BYOA smoke test (PLAN_V2_byoa.md §6).

    L0 — install + version_cmd from [smoke_test], <2s, $0.
    L1 — L0 + ping_cmd against the configured provider, <30s, ~$0.000004.
    """
    from benchflow.agents.tester import run_l0, run_l1

    if level == 0:
        result = run_l0(name)
    elif level == 1:
        result = run_l1(name, provider=provider, model=model)
    else:
        console.print(f"[red]L{level} is not implemented; only L0 and L1 today.[/red]")
        raise typer.Exit(2)

    color = {"pass": "green", "fail": "red", "skipped": "yellow", "error": "red"}[
        result.outcome
    ]
    console.print(
        f"[{color}]{result.outcome.upper()}[/{color}] "
        f"L{result.level} {result.name} "
        f"({result.latency_ms} ms, fidelity={result.fidelity})"
    )
    if result.version_detected:
        console.print(f"  version: {result.version_detected}")
    if result.detail:
        console.print(f"  detail:  {result.detail}")
    if result.stderr_tail:
        console.print(f"  stderr tail:\n{result.stderr_tail}")
    if result.outcome == "fail":
        raise typer.Exit(1)
