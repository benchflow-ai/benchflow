"""``bench skills`` — skill discovery and evaluation.

Registered onto the top-level app by :func:`register_skills`; ``cli/main.py``
only wires the call.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from benchflow.cli._options import (
    ConcurrencyOption,
    JobsDirOption,
    SandboxOption,
)
from benchflow.cli._shared import console, print_error
from benchflow.sandbox.providers import is_known_provider, providers_phrase


def register_skills(app: typer.Typer) -> None:
    """Attach the ``skills`` command group to the top-level benchflow app."""
    skills_app = typer.Typer(help="Skill discovery and evaluation.")
    app.add_typer(skills_app, name="skills", rich_help_panel="Core")

    @skills_app.command("list")
    def skills_list(
        directory: Annotated[
            Path | None,
            typer.Option("--dir", help="Skills directory to scan"),
        ] = None,
    ) -> None:
        """List discovered skills."""
        from benchflow.skills import DEFAULT_SKILLS_DIR, discover_skills

        search_dirs = (
            [directory]
            if directory
            else [DEFAULT_SKILLS_DIR, Path(".claude/skills"), Path("skills")]
        )
        found = discover_skills(*search_dirs)
        if not found:
            console.print(
                "No skills found. Add skill directories under .claude/skills/ or skills/."
            )
            return

        table = Table(title="Discovered Skills")
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="green")
        table.add_column("Description")
        table.add_column("Path", style="dim")

        for s in found:
            # escape(): SKILL.md metadata is author/third-party controlled (e.g.
            # a description mentioning "[/INST]") and would otherwise raise a Rich
            # MarkupError that crashes the listing — matching `skills eval`.
            table.add_row(
                escape(s.name),
                escape(s.version or "-"),
                escape(s.description[:60]),
                escape(str(s.path)),
            )

        console.print(table)

    @skills_app.command("eval")
    def skills_eval(
        skill_dir: Annotated[
            Path,
            typer.Argument(help="Path to skill directory containing evals/evals.json"),
        ],
        agent: Annotated[
            list[str] | None,
            typer.Option("--agent", help="Agent(s) to evaluate (repeatable)"),
        ] = None,
        model: Annotated[
            list[str] | None,
            typer.Option("--model", help="Model(s) (matched 1:1 with agents)"),
        ] = None,
        environment: SandboxOption = "docker",
        concurrency: ConcurrencyOption = 1,
        jobs_dir: JobsDirOption = "jobs",
        no_baseline: Annotated[
            bool,
            typer.Option("--no-baseline", help="Skip baseline (without-skill) runs"),
        ] = False,
        export_gepa: Annotated[
            bool,
            typer.Option("--export-gepa", help="Export GEPA-compatible traces"),
        ] = False,
    ) -> None:
        """Evaluate a skill using its evals/evals.json test cases.

        Generates ephemeral tasks from the skill's eval dataset, runs each agent
        with and without the skill installed, and reports the lift.

        Examples:
            benchflow skills eval ./my-skill/ --agent claude-agent-acp
            benchflow skills eval ./my-skill/ --agent claude-agent-acp --agent codex-acp --sandbox daytona --concurrency 4
            benchflow skills eval ./my-skill/ --agent claude-agent-acp --no-baseline --export-gepa
        """
        from benchflow.skill_eval import SkillEvaluator, export_gepa_traces

        if not is_known_provider(environment):
            print_error(
                f"Invalid --sandbox {environment!r}: choose {providers_phrase()}"
            )
            raise typer.Exit(1)
        if agent is None:
            agent = ["claude-agent-acp"]
        if not (skill_dir / "evals" / "evals.json").exists():
            print_error(
                f"No evals/evals.json found in {skill_dir}\n"
                "Create one with test cases. See: benchflow skills eval --help"
            )
            raise typer.Exit(1)

        try:
            evaluator = SkillEvaluator(skill_dir)
        except (
            json.JSONDecodeError,
            ValueError,
            FileNotFoundError,
            NotADirectoryError,
        ) as e:
            print_error(f"{e}")
            raise typer.Exit(1) from None
        console.print(
            f"[bold]Skill eval:[/bold] {escape(str(evaluator.dataset.skill_name))} "
            f"({len(evaluator.dataset.cases)} cases)"
        )
        console.print(f"  Agents: {', '.join(agent)}")
        console.print(f"  Environment: {environment}")
        if no_baseline:
            console.print("  [dim]Baseline skipped (--no-baseline)[/dim]")

        try:
            result = asyncio.run(
                evaluator.run(
                    agents=agent,
                    models=model,
                    environment=environment,
                    jobs_dir=jobs_dir,
                    no_baseline=no_baseline,
                    concurrency=concurrency,
                )
            )
        except (
            json.JSONDecodeError,
            ValueError,
            FileNotFoundError,
            NotADirectoryError,
        ) as e:
            print_error(f"{e}")
            raise typer.Exit(1) from None

        # Display results
        table = Table(title=f"Skill Eval: {result.skill_name}")
        table.add_column("Agent", style="cyan")
        table.add_column("Mode", style="dim")
        table.add_column("Score")
        table.add_column("Avg Reward")

        for row in result.summary_table():
            style = "bold green" if row["mode"] == "LIFT" else None
            table.add_row(
                row["agent"], row["mode"], row["score"], row["avg_reward"], style=style
            )

        console.print(table)

        if export_gepa:
            gepa_dir = export_gepa_traces(
                result,
                evaluator.dataset,
                output_dir=f"{jobs_dir}/skill-eval/{result.skill_name}/gepa",
            )
            console.print(
                f"[green]GEPA traces exported to {escape(str(gepa_dir))}[/green]"
            )
