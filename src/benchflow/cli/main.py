"""benchflow CLI — Harbor + ACP unified."""

import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()

app = typer.Typer(
    name="benchflow",
    help="Agent benchmarking framework. ACP + Harbor.",
    no_args_is_help=True,
)


@app.command()
def run(
    task_dir: Annotated[
        Path,
        typer.Option("--task-dir", "-t", help="Task directory"),
    ],
    agent: Annotated[
        str,
        typer.Option("--agent", "-a", help="ACP agent command"),
    ] = "claude-agent-acp",
    prompt: Annotated[
        list[str] | None,
        typer.Option(
            "--prompt", "-p", help="Prompt(s) to send (default: instruction.md)"
        ),
    ] = None,
    agent_env: Annotated[
        list[str] | None,
        typer.Option("--ae", help="Agent env var (KEY=VALUE)"),
    ] = None,
) -> None:
    """Run a task with an ACP agent in a Docker container."""
    from benchflow.sdk import SDK

    parsed_env: dict[str, str] = {}
    for entry in agent_env or []:
        if "=" not in entry:
            console.print(f"[red]Invalid env var: {entry}[/red]")
            raise typer.Exit(1)
        key, value = entry.split("=", 1)
        parsed_env[key] = value

    # Load API key from environment if not provided
    if "ANTHROPIC_API_KEY" not in parsed_env:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            parsed_env["ANTHROPIC_API_KEY"] = api_key

    sdk = SDK()
    result = asyncio.run(
        sdk.run(
            task_path=task_dir,
            agent=agent,
            prompts=prompt,
            agent_env=parsed_env,
        )
    )

    if result.error:
        console.print(f"[red]Error: {result.error}[/red]")
        raise typer.Exit(1)

    console.print(f"Task: {result.task_name}")
    console.print(f"Agent: {result.agent_name}")
    console.print(f"Rewards: {result.rewards}")
    console.print(f"Tool calls: {result.n_tool_calls}")
    console.print(f"Prompts: {result.n_prompts}")
    console.print(f"Trajectory: {len(result.trajectory)} events")


@app.command()
def view(
    trial_dir: Annotated[
        Path,
        typer.Argument(help="Trial directory containing turn*.txt or trajectory.json"),
    ],
    port: Annotated[int, typer.Option(help="Server port")] = 8888,
) -> None:
    """View a trial trajectory in the browser."""
    from benchflow.viewer import serve

    serve(str(trial_dir), port)


if __name__ == "__main__":
    app()
