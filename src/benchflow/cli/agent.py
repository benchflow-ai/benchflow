"""``bench agent`` — agent management commands (list / show / run).

``bench agent run`` is the headless agent runner: one prompt per invocation,
resumable across invocations via ACP ``session/load`` (claude -p parity). It
supersedes the retired benchmark-adoption alias that briefly occupied the name
(canonical adoption spelling: ``bench eval adopt``; the remaining legacy
aliases ``bench agent create|verify`` stay hidden + deprecated).

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
            print_error(f"Unknown agent: {name}")
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

    @agent_app.command("run")
    def agent_run(
        agent: Annotated[
            str, typer.Argument(help="Registered agent name (`bench agent list`)")
        ],
        prompt_arg: Annotated[
            str | None,
            typer.Argument(metavar="[PROMPT]", help="The prompt (or use -p)"),
        ] = None,
        prompt_opt: Annotated[
            str | None, typer.Option("--prompt", "-p", help="The prompt")
        ] = None,
        model: Annotated[
            str, typer.Option("--model", help="Model id for the turn")
        ] = "",
        resume: Annotated[
            str | None,
            typer.Option(
                "--resume", "-r", help="Resume a session by id (see json output)"
            ),
        ] = None,
        continue_: Annotated[
            bool,
            typer.Option(
                "--continue",
                "-c",
                help="Resume the most recent session in this directory",
            ),
        ] = False,
        launch_cmd: Annotated[
            str | None,
            typer.Option(
                "--launch-cmd", help="Override the agent launch command (host binary)"
            ),
        ] = None,
        output_format: Annotated[
            str, typer.Option("--output-format", help="text | json")
        ] = "text",
        timeout: Annotated[
            float, typer.Option("--timeout", help="Turn budget in seconds")
        ] = 600.0,
    ) -> None:
        """Run one headless prompt against an agent; resume it later (claude -p parity).

        First run prints a session id; ``--resume <id>`` / ``-c`` continues that
        conversation in the same working directory via ACP ``session/load``
        (agents must advertise the ``loadSession`` capability). No task, no
        sandbox, no verifier — evaluations stay on ``bench eval run``.
        """
        import asyncio
        import json as _json
        import os

        from benchflow.agents.session_store import SessionStore
        from benchflow.agents.standalone import ResumeUnsupportedError, run_turn

        prompt = prompt_opt or prompt_arg
        if not prompt:
            print_error("a prompt is required (positional or -p/--prompt)")
            raise typer.Exit(1)

        store = SessionStore(
            root=os.environ.get("BENCHFLOW_AGENT_SESSIONS_DIR") or None
        )
        cwd = os.getcwd()

        if continue_ and not resume:
            latest = store.latest_for_cwd(cwd)
            if latest is None:
                print_error(f"no agent session recorded for {cwd}")
                raise typer.Exit(1)
            resume = latest.session_id

        agent_env: dict[str, str] | None = None
        if not launch_cmd:
            from benchflow.agents.env import resolve_agent_env
            from benchflow.agents.registry import AGENT_ALIASES, AGENTS

            canonical = AGENT_ALIASES.get(agent, agent)
            cfg = AGENTS.get(canonical)
            if cfg is None:
                print_error(f"Unknown agent: {agent}")
                raise typer.Exit(1)
            agent = canonical
            launch_cmd = cfg.launch_cmd
            try:
                agent_env = resolve_agent_env(
                    agent, model or cfg.default_model or None, None
                )
            except ValueError as exc:
                print_error(str(exc))
                raise typer.Exit(1) from exc

        try:
            result = asyncio.run(
                asyncio.wait_for(
                    run_turn(
                        agent=agent,
                        prompt=prompt,
                        cwd=cwd,
                        store=store,
                        model=model,
                        resume=resume,
                        launch_cmd=launch_cmd,
                        agent_env=agent_env,
                    ),
                    timeout=timeout,
                )
            )
        except TimeoutError as exc:
            print_error(f"agent turn timed out after {timeout:g}s")
            raise typer.Exit(1) from exc
        except ResumeUnsupportedError as exc:
            print_error(str(exc))
            raise typer.Exit(1) from exc

        if output_format == "json":
            typer.echo(
                _json.dumps(
                    {
                        "result": result.text,
                        "session_id": result.session_id,
                        "stop_reason": result.stop_reason,
                        "agent": agent,
                        "model": model,
                    }
                )
            )
        else:
            typer.echo(result.text)
            typer.echo(
                f"session: {result.session_id}  (resume: bench agent run {agent} -r {result.session_id} -p …)",
                err=True,
            )
