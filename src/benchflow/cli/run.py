"""``bench run`` — execute a single task with an ACP agent.

Extracted from cli/main.py per PLAN_V2_impl §13.4 / §13.6 commit 7.
The ``register(app)`` function attaches the ``run`` command to the
root Typer app — main.py owns the app, this module owns the command.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console

from benchflow.job import DEFAULT_AGENT

console = Console()


def register(app: typer.Typer) -> None:
    @app.command()
    def run(
        task_dir: Annotated[
            Path,
            typer.Argument(help="Task directory (must contain task.toml)"),
        ],
        agent: Annotated[
            str,
            typer.Option("--agent", "-a", help="Agent name from registry"),
        ] = DEFAULT_AGENT,
        model: Annotated[
            str | None,
            typer.Option("--model", "-m", help="Model to use"),
        ] = None,
        environment: Annotated[
            str,
            typer.Option("--backend", "-b", help="Backend: docker or daytona"),
        ] = "docker",
        prompt: Annotated[
            list[str] | None,
            typer.Option(
                "--prompt", "-p", help="Prompt(s) to send (default: instruction.md)"
            ),
        ] = None,
        jobs_dir: Annotated[
            str,
            typer.Option("--jobs-dir", "-o", help="Output directory for results"),
        ] = "jobs",
        agent_env: Annotated[
            list[str] | None,
            typer.Option("--agent-env", "--ae", help="Agent env var (KEY=VALUE)"),
        ] = None,
        skills_dir: Annotated[
            Path | None,
            typer.Option(
                "--skills-dir", "-s", help="Skills directory to deploy into sandbox"
            ),
        ] = None,
        sandbox_user: Annotated[
            str | None,
            typer.Option(
                "--sandbox-user",
                help="Run agent as non-root user (default: 'agent'). Pass 'none' for root.",
            ),
        ] = "agent",
    ) -> None:
        """Run a single task with an ACP agent.

        Examples:
            bench run tasks/regex-log --agent gemini --model gemini-3.1-flash-lite-preview
            bench run tasks/X --agent openhands --backend daytona
        """
        from benchflow.trial import Trial, TrialConfig

        parsed_env: dict[str, str] = {}
        for entry in agent_env or []:
            if "=" not in entry:
                console.print(f"[red]Invalid env var: {entry}[/red]")
                raise typer.Exit(1)
            key, value = entry.split("=", 1)
            parsed_env[key] = value

        # CLI only ever passes plain strings; cast to widen for Trial's
        # `list[str | None] | None` API (None entries mean "use default").
        async def _run():
            config = TrialConfig(
                task_path=Path(task_dir),
                agent=agent,
                prompts=cast("list[str | None] | None", prompt),
                model=model,
                agent_env=parsed_env,
                jobs_dir=jobs_dir,
                environment=environment,
                skills_dir=str(skills_dir) if skills_dir else None,
                sandbox_user=sandbox_user,
            )
            trial = await Trial.create(config)
            return await trial.run()

        result = asyncio.run(_run())

        if result.error:
            console.print(f"[red]Error: {result.error}[/red]")
            raise typer.Exit(1)

        console.print(f"[green]Task:[/green] {result.task_name}")
        console.print(f"[green]Agent:[/green] {result.agent_name}")
        console.print(f"[green]Rewards:[/green] {result.rewards}")
        console.print(f"[green]Tool calls:[/green] {result.n_tool_calls}")
