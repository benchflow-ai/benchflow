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
        if model is not None and len(model) not in {1, len(agent)}:
            print_error(
                "--model may be provided once for all agents or once per --agent; "
                f"got {len(model)} models for {len(agent)} agents"
            )
            raise typer.Exit(1)
        if not (skill_dir / "evals" / "evals.json").is_file():
            # is_file (not exists): a directory named evals.json otherwise passes
            # this guard, then read_text() raises a raw IsADirectoryError.
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

        # Match `eval run`'s _exit_if_evaluation_had_errors: a run with any
        # errored case is a failure for CI/scripting — exit non-zero so a
        # 100%-error run (e.g. missing credentials) is not reported as success.
        errored = sum(1 for c in result.case_results if c.error)
        if errored:
            print_error(
                f"{errored}/{len(result.case_results)} eval case(s) errored — "
                "the eval did not complete cleanly."
            )
            raise typer.Exit(1)

    @skills_app.command("validate")
    def skills_validate(
        paths: Annotated[
            list[Path],
            typer.Argument(
                help=(
                    "Skill directories, or directories searched recursively for "
                    "SKILL.md (e.g. a SkillsBench tasks/ folder)"
                ),
            ),
        ],
        no_security: Annotated[
            bool,
            typer.Option(
                "--no-security", help="Skip the NVIDIA SkillSpector security scan"
            ),
        ] = False,
        fail_on: Annotated[
            str,
            typer.Option(
                "--fail-on",
                help=(
                    "Exit 1 when a schema or security finding is at or above this "
                    "severity: critical, high, medium, low, or never"
                ),
            ),
        ] = "high",
        checks: Annotated[
            str | None,
            typer.Option(
                "--checks",
                help=(
                    "Comma-separated SkillEvaluator Tier 1 checks (default: "
                    "schema,version,pii,license,code-integrity,dependency,unicode,"
                    "quality,lint)"
                ),
            ),
        ] = None,
        min_score: Annotated[
            int | None,
            typer.Option(
                "--min-score",
                min=0,
                max=100,
                help="Quality score SkillEvaluator's quality check needs (default 70)",
            ),
        ] = None,
        output_dir: Annotated[
            Path | None,
            typer.Option(
                "--output-dir",
                "-o",
                help="Keep each skill's raw JSON reports under DIR/<skill>/",
            ),
        ] = None,
        workers: Annotated[
            int,
            typer.Option("--workers", "-j", min=1, help="Skills checked in parallel"),
        ] = 4,
        max_findings: Annotated[
            int,
            typer.Option(
                "--max-findings",
                min=0,
                help="Blocking findings listed per skill (all are in --json)",
            ),
        ] = 5,
        json_out: Annotated[
            bool,
            typer.Option("--json", help="Print machine-readable JSON"),
        ] = False,
    ) -> None:
        """Run static checks on skills: NVIDIA SkillEvaluator Tier 1 and SkillSpector.

        SkillEvaluator Tier 1 checks schema, version, PII, license, code
        integrity, dependencies, Unicode smuggling, quality score and script
        lint. SkillSpector scans for security issues. Both run without LLM
        calls; install them separately (see docs/skill-eval.md). Exit 1 when a
        schema or security finding reaches --fail-on, or a checker fails to run.

        Examples:
            bench skills validate ./my-skill
            bench skills validate tasks/ --json > skill-checks.json
            bench skills validate tasks/citation-check --fail-on critical -o reports
        """
        from concurrent.futures import ThreadPoolExecutor

        from benchflow.cli._shared import err_console
        from benchflow.integrations.skillevaluator import (
            DEFAULT_TIER1_CHECKS,
            FAIL_ON_CHOICES,
            SKILLEVALUATOR,
            SKILLSPECTOR,
            check_skill,
            install_hint,
            iter_skill_dirs,
            locate_tool,
        )

        fail_on = fail_on.lower()
        if fail_on not in FAIL_ON_CHOICES:
            print_error(
                f"Invalid --fail-on {fail_on!r}: choose {', '.join(FAIL_ON_CHOICES)}"
            )
            raise typer.Exit(1)
        skill_dirs = list(iter_skill_dirs(paths))
        if not skill_dirs:
            print_error(f"No SKILL.md found under {', '.join(map(str, paths))}")
            raise typer.Exit(1)

        evaluator = locate_tool(SKILLEVALUATOR)
        spector = None if no_security else locate_tool(SKILLSPECTOR)
        if evaluator is None and spector is None:
            print_error(install_hint(SKILLEVALUATOR))
            if not no_security:
                print_error(install_hint(SKILLSPECTOR))
            raise typer.Exit(1)
        if evaluator is None:
            err_console.print(
                f"[yellow]{escape(install_hint(SKILLEVALUATOR))} "
                "Schema and quality checks skipped.[/yellow]"
            )
        if spector is None and not no_security:
            err_console.print(
                f"[yellow]{escape(install_hint(SKILLSPECTOR))} "
                "Security scan skipped.[/yellow]"
            )
        for tool in (evaluator, spector):
            warning = tool.warning() if tool else None
            if warning:
                err_console.print(f"[yellow]warning: {escape(warning)}[/yellow]")

        check_list = (
            tuple(c.strip() for c in checks.split(",") if c.strip())
            if checks
            else DEFAULT_TIER1_CHECKS
        )
        labels = [d.name for d in skill_dirs]
        if len(set(labels)) < len(labels):
            labels = [str(d) for d in skill_dirs]

        def _check(index: int):
            return check_skill(
                skill_dirs[index],
                skillevaluator=evaluator,
                skillspector=spector,
                checks=check_list,
                min_score=min_score,
                output_dir=(
                    output_dir / labels[index].replace("/", "__")
                    if output_dir
                    else None
                ),
            )

        with (
            err_console.status(f"Checking {len(skill_dirs)} skill(s)..."),
            ThreadPoolExecutor(max_workers=min(workers, len(skill_dirs))) as pool,
        ):
            results = list(pool.map(_check, range(len(skill_dirs))))

        blocked = [r for r in results if r.is_blocking(fail_on)]
        if json_out:
            tools = {
                t.name: {"path": t.path, "version": t.version}
                for t in (evaluator, spector)
                if t is not None
            }
            typer.echo(
                json.dumps(
                    {
                        "tools": tools,
                        "fail_on": fail_on,
                        "skills": [r.to_dict(fail_on) for r in results],
                        "blocking": len(blocked),
                    },
                    indent=2,
                )
            )
        else:
            title = " + ".join(
                f"{'SkillEvaluator' if t.name == SKILLEVALUATOR else 'SkillSpector'} "
                f"{t.version or '?'}"
                for t in (evaluator, spector)
                if t is not None
            )
            table = Table(title=f"Skill checks ({title})")
            table.add_column("Skill", style="cyan")
            if evaluator:
                table.add_column("Tier 1")
                table.add_column("Quality")
            if spector:
                table.add_column("Security")
            table.add_column("Blocking", justify="right")
            for label, r in zip(labels, results, strict=True):
                row = [escape(label)]
                if evaluator:
                    t1 = r.tier1
                    row.append(t1.status if t1 else "[red]error[/red]")
                    row.append(
                        f"{t1.quality_grade or '?'} {t1.quality_score:g}"
                        if t1 and t1.quality_score is not None
                        else "-"
                    )
                if spector:
                    sec = r.security
                    if sec is None:
                        row.append("[red]error[/red]")
                    else:
                        score = "?" if sec.risk_score is None else sec.risk_score
                        cell = f"{sec.risk_severity or '?'} {score} {sec.recommendation or ''}"
                        if sec.analysis_status not in (None, "complete"):
                            cell += f" ({sec.analysis_status})"
                        row.append(escape(cell.strip()))
                n_blocking = len(r.blocking_findings(fail_on))
                if r.errors:
                    row.append("[red]error[/red]")
                elif n_blocking:
                    row.append(f"[red]{n_blocking}[/red]")
                else:
                    row.append("[green]0[/green]")
                table.add_row(*row)
            console.print(table)

            for label, r in zip(labels, results, strict=True):
                shown = r.blocking_findings(fail_on)[:max_findings]
                if not (r.errors or shown):
                    continue
                console.print(f"[bold]{escape(label)}[/bold]")
                for err in r.errors:
                    console.print(f"  [red]error[/red] {escape(err)}")
                for f in shown:
                    where = f" ({f.location})" if f.location else ""
                    console.print(
                        f"  [red]{f.severity.upper():8}[/red] {f.source} "
                        f"{escape(f.category)} {escape(f.check)}: "
                        f"{escape(f.message[:160])}{escape(where)}"
                    )
                hidden = len(r.blocking_findings(fail_on)) - len(shown)
                if hidden > 0:
                    console.print(f"  [dim]... {hidden} more; see --json[/dim]")
            incomplete = sorted(
                {s for r in results if r.tier1 for s in r.tier1.incomplete_scans}
            )
            if incomplete:
                console.print(
                    "[dim]Incomplete SkillEvaluator scans (scanner missing or "
                    f"partial): {escape(', '.join(incomplete))}[/dim]"
                )
            console.print(
                f"{len(results)} skill(s) checked, {len(blocked)} blocking "
                f"(--fail-on {fail_on})"
            )

        if blocked:
            raise typer.Exit(1)
