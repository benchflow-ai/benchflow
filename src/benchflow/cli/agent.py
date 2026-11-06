"""``bench agent`` — agent management commands (list / show).

The benchmark-adoption commands now live under ``bench eval adopt`` (see
:mod:`benchflow.cli.adopt`); the legacy ``bench agent create|run|verify`` are
registered here as hidden deprecated aliases (one-release window) so existing
scripts keep working, each emitting a deprecation notice.

Registered onto the top-level app by :func:`register_agent`; ``cli/main.py``
only wires the call.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.table import Table

from benchflow.agent_router import AGENT_ALIAS_VERBS, register_agent_router
from benchflow.cli._shared import (
    _PROVIDER_AUTH_MESSAGE,
    _REQUIRES_AUTH_NOTE,
    _format_requires,
    console,
    print_error,
)


def _first_sentence(text: str | None) -> str:
    """A description's first sentence, on one line."""
    flat = " ".join((text or "").split())
    for end in (". ", "; "):
        cut = flat.find(end)
        if cut != -1:
            return flat[: cut + 1].rstrip(";")
    return flat


# The two controls --agent takes besides the registered agents.
_CONTROLS = {
    "oracle": "The task's reference solution (no model, no key)",
    "nop": "Does nothing: the empty run, a control",
}


def register_agent(app: typer.Typer) -> None:
    """Attach the ``agent`` command group to the top-level benchflow app."""
    agent_app = typer.Typer(help="Agent management commands.")
    app.add_typer(agent_app, name="agent", rich_help_panel="Core")
    # Legacy adoption verbs (create/run/verify) — hidden + deprecated; canonical
    # home is the single `bench eval adopt` command.
    register_agent_router(agent_app, verbs=AGENT_ALIAS_VERBS, deprecated_as="agent")

    @agent_app.command("list")
    def agent_list() -> None:
        """List all registered agents."""
        from benchflow.agents.registry import AGENT_ALIASES, list_agents
        from benchflow.agents.remote_manifests import (
            manifest_catalog_for_listing,
        )

        catalog = manifest_catalog_for_listing()

        # Build reverse map: canonical name -> list of aliases
        reverse_aliases: dict[str, set[str]] = {}
        for alias, canonical in AGENT_ALIASES.items():
            if alias != canonical:
                reverse_aliases.setdefault(canonical, set()).add(alias)
        if not catalog.applied:
            for manifest in catalog.manifests:
                for alias in manifest.aliases:
                    if alias != manifest.config.name:
                        reverse_aliases.setdefault(manifest.config.name, set()).add(
                            alias
                        )

        # One line per agent, so the list fits on a screen (dogfood p5-11): the
        # name in full (it is what -a takes), its aliases, and the first
        # sentence of its description cut to the width; `bench agents <name>`
        # shows the rest.
        agents = list_agents()
        if not catalog.applied:
            agents.extend(manifest.config for manifest in catalog.manifests)
        rows = sorted({a.name: a for a in agents}.values(), key=lambda a: a.name)
        # Columns as the terminal allows: the name in full (aliases beside
        # it, cut before the name is) and what it needs always, the protocol
        # on wide screens; the description takes what is left, on one line.
        widest = console.width >= 120
        alias_text = {
            a.name: ", ".join(sorted(reverse_aliases.get(a.name, []))) for a in rows
        }
        name_width = max(len(a.name) for a in rows) + 12
        needs = {a.name: _format_requires(a) for a in rows}
        needs_width = min(28, max(12, *(len(t) for t in needs.values())))
        widths = [name_width, needs_width, *([9] if widest else [])]
        text_width = max(20, console.width - sum(widths) - 2 * len(widths))
        table = Table(
            title="Registered Agents",
            title_justify="left",
            box=None,
            pad_edge=False,
            show_edge=False,
            header_style="bold",
        )
        table.add_column("Name", style="cyan", no_wrap=True, width=name_width)
        if widest:
            table.add_column("Protocol", style="green", no_wrap=True, width=9)
        table.add_column(
            "Requires",
            style="yellow",
            no_wrap=True,
            overflow="ellipsis",
            width=needs_width,
        )
        table.add_column(
            "Description", no_wrap=True, overflow="ellipsis", max_width=text_width
        )

        def named(name: str, aliases: str) -> str:
            if not aliases:
                return name
            cell = f"{name} ({aliases})"
            if len(cell) <= name_width:
                return cell
            return cell[: max(len(name), name_width - 1)] + "…"

        def row(name: str, aliases: str, protocol: str, need: str, text: str) -> None:
            cells = [named(name, aliases), *([protocol] if widest else [])]
            table.add_row(*cells, need, text)

        for control, text in _CONTROLS.items():
            row(control, "", "-", "", text)
        for a in rows:
            row(
                a.name,
                alias_text[a.name],
                a.protocol,
                needs[a.name],
                _first_sentence(a.description),
            )

        console.print(table)
        console.print(
            f"[dim]`bench agents <name>` shows one in full. {_REQUIRES_AUTH_NOTE}[/dim]"
        )
        if catalog.warnings:
            print_error(
                f"Agent catalog incomplete ({catalog.source}):\n  "
                + "\n  ".join(catalog.warnings),
                next_step="the agents listed above work; BENCHFLOW_AGENTS_SOURCE "
                "names the catalog (off skips it)",
            )

    @agent_app.command("show")
    def agent_show(
        name: Annotated[str, typer.Argument(help="Agent name")],
    ) -> None:
        """Show details for a registered agent."""
        if name in _CONTROLS:
            # A control is no registered agent: it runs no model and needs
            # no key, so it has no launch command or requirements to show.
            console.print(f"[bold]{name}[/bold]")
            console.print(f"  Description: {_CONTROLS[name]}")
            console.print("  Protocol:    - (a control, built into BenchFlow)")
            console.print("  Requires:    (none)")
            return
        from benchflow.agents.registry import (
            AGENT_ALIASES,
            resolve_agent,
        )

        try:
            cfg = resolve_agent(name)
        except KeyError as exc:
            # A KeyError's str() is its message in quotes; print the message.
            print_error(
                str(exc.args[0]) if exc.args else str(exc),
                next_step="`bench agents` lists the agents",
            )
            raise typer.Exit(2) from None

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
