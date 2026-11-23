"""``bench agent`` — agent management commands (list / show).

The benchmark-adoption commands now live under ``bench eval adopt`` (see
:mod:`benchflow.cli.adopt`); the legacy ``bench agent create|run|verify`` are
registered here as hidden deprecated aliases (one-release window) so existing
scripts keep working, each emitting a deprecation notice.

Registered onto the top-level app by :func:`register_agent`; ``cli/main.py``
only wires the call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from benchflow.agent_router import AGENT_ALIAS_VERBS, register_agent_router
from benchflow.cli._shared import (
    _PROVIDER_AUTH_MESSAGE,
    _REQUIRES_AUTH_NOTE,
    _format_requires,
    console,
    err_console,
    print_error,
)


def _apply_dotenv_agent_source() -> None:
    """The ``.env`` keys that decide agent discovery (``BENCHFLOW_AGENTS_*``),
    with ``bench run``'s precedence: an exported value wins (``.env``
    named the agents folder for ``bench run`` but not for ``bench agents``).
    Only these keys, so listing agents imports nothing a run needs."""
    import os

    from benchflow._dotenv import load_dotenv_env

    for key, value in load_dotenv_env().items():
        if key.startswith("BENCHFLOW_AGENTS_"):
            os.environ.setdefault(key, value)


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

    # Reached as `bench agents validate`; hidden here, so the 0.8 group keeps
    # its own commands.
    @agent_app.command("validate", hidden=True)
    def agent_validate(
        path: Annotated[
            Path,
            typer.Argument(help="A manifest.toml, or the agent folder that holds it"),
        ],
    ) -> None:
        """Check an agent manifest against the contract, without Docker: its
        contract_version, every key and its type, with the line of each problem."""
        from benchflow.agents.manifest_check import check_manifest, manifest_identity

        _apply_dotenv_agent_source()
        manifest = path / "manifest.toml" if path.is_dir() else path
        if not manifest.is_file():
            print_error(
                f"no manifest at {manifest.as_posix()}",
                next_step="pass the agent's manifest.toml (agents/<name>/manifest.toml) "
                "or its folder",
            )
            raise typer.Exit(2)
        problems = check_manifest(manifest)
        if problems:
            for problem in problems:
                err_console.print(
                    escape(problem.text()), highlight=False, soft_wrap=True
                )
            print_error(
                f"{manifest.as_posix()} is not a valid agent manifest "
                f"({len(problems)} problem{'s' if len(problems) != 1 else ''})",
                next_step="fix each line above, then run bench agents validate again",
            )
            raise typer.Exit(1)
        identity = manifest_identity(manifest, manifest.parent.parent)
        console.print(
            f"[green]✓[/green] {escape(manifest.as_posix())}: valid; agent "
            f"{escape(identity.text())}",
            highlight=False,
            soft_wrap=True,
        )

    @agent_app.command("list")
    def agent_list() -> None:
        """List all registered agents."""
        from benchflow.agents.registry import AGENT_ALIASES, list_agents
        from benchflow.agents.remote_manifests import (
            manifest_catalog_for_listing,
        )

        # The same .env a run reads names the agents folder.
        _apply_dotenv_agent_source()
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
        # An invalid local manifest is one line of the catalog's warning:
        # the agent, its first problem and its line.
        warnings = catalog.warnings
        console.print(
            f"[dim]`bench agents <name>` shows one in full. {_REQUIRES_AUTH_NOTE}[/dim]"
        )
        if warnings:
            print_error(
                f"Agent catalog incomplete ({catalog.source}):\n  "
                + "\n  ".join(warnings),
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
            MODEL_ROUTE_AGENT,
            resolve_agent,
        )

        _apply_dotenv_agent_source()
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
        from benchflow.agents.remote_manifests import core_override

        if (override := core_override(cfg.name)) is not None:
            console.print(
                f"  Source:      {override.manifest} ({override.sha256[:23]}), "
                f"replacing the core agent ({override.differs_in[0]} and "
                f"{len(override.differs_in) - 1} more differ)"
                if len(override.differs_in) > 1
                else f"  Source:      {override.manifest} ({override.sha256[:23]}), "
                "replacing the core agent",
                style="yellow",
                soft_wrap=True,
            )
        if aliases:
            console.print(f"  Aliases:     {', '.join(aliases)}")
        console.print(f"  Description: {cfg.description}")
        console.print(f"  Protocol:    {cfg.protocol}")
        console.print(f"  Launch:      {cfg.launch_cmd}")
        console.print(f"  Requires:    {_format_requires(cfg) or '(none)'}")
        if getattr(cfg, "model_route", None) == MODEL_ROUTE_AGENT:
            # No provider route to explain: the agent brings its own.
            console.print(
                "  Model route: agent (calls its own endpoint; BenchFlow starts "
                "no model proxy and meters no usage)"
            )
        else:
            console.print(f"  Provider auth: {_PROVIDER_AUTH_MESSAGE}")
        if cfg.subscription_auth:
            console.print(
                f"  Auth:        subscription via {cfg.subscription_auth.detect_file}"
            )
