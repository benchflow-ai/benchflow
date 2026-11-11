"""``bench review`` — rubric review of finished rollouts.

Lives in its own module per the one-file-per-command-group convention;
:func:`register_review` attaches it to the root Typer app.
"""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from benchflow.cli._shared import (
    _apply_dotenv_to_process_env,
    _parse_agent_env,
    console,
    exit_if_missing,
    print_error,
)
from benchflow.sandbox.providers import is_known_provider, providers_phrase

# The model `bench review` gives its default agent, opencode, which has no
# registry default: a gateway model id.
DEFAULT_REVIEW_MODEL = "gemini/gemini-3.5-flash-lite"

_REVIEW_OUTCOME_STYLES = {
    "pass": "green",
    "fail": "red",
    "not_applicable": "grey50",
}

_REVIEW_SCORE_STYLES = {
    0: "red",
    1: "yellow",
    2: "green",
}


def _render_trial_review(trial) -> None:
    table = Table(title=f"Review: {escape(str(trial.trial_name))}", show_lines=True)
    table.add_column("Criterion")
    table.add_column("Result")
    table.add_column("Explanation")
    for name, check in (trial.checks or {}).items():
        if "score" in check:
            raw_score = check.get("score")
            result = str(raw_score)
            style = (
                _REVIEW_SCORE_STYLES.get(raw_score, "white")
                if type(raw_score) is int
                else "white"
            )
        else:
            result = str(check.get("outcome", ""))
            style = _REVIEW_OUTCOME_STYLES.get(result, "white")
        table.add_row(
            escape(str(name).replace("_", " ").title()),
            escape(result),
            escape(str(check.get("explanation", ""))),
            style=style,
        )
    if trial.summary:
        console.print(f"\n[bold]Summary:[/bold] {escape(str(trial.summary))}\n")
    scoring = getattr(trial, "scoring", None)
    if scoring is not None:
        console.print(
            "[bold]Weighted quality:[/bold] "
            f"{scoring.raw_quality:.3f} raw / {scoring.gated_quality:.3f} gated "
            f"— {escape(scoring.decision.value)}\n"
        )
    console.print(table)


def _render_review_overview(trials) -> None:
    table = Table(title="Rubric Reviews", show_lines=True)
    for column in ("Run", "Pass", "Fail", "N/A", "Quality", "Decision", "Valid"):
        table.add_column(column)
    for trial in trials:
        if trial.error and not trial.checks:
            table.add_row(
                escape(str(trial.trial_name)),
                "-",
                "-",
                "-",
                "-",
                "-",
                "no",
                style="red",
            )
            continue
        counts = trial.outcome_counts()
        scoring = getattr(trial, "scoring", None)
        quality = f"{scoring.raw_quality:.3f}" if scoring is not None else "-"
        decision = scoring.decision.value if scoring is not None else "-"
        if (
            not trial.review_valid
            or counts["fail"]
            or (scoring is not None and decision == "not_publishable")
        ):
            style = "red"
        elif scoring is not None and decision == "presentable_with_revisions":
            style = "yellow"
        else:
            style = "white"
        table.add_row(
            escape(str(trial.trial_name)),
            str(counts["pass"]),
            str(counts["fail"]),
            str(counts["not_applicable"]),
            quality,
            escape(decision),
            "yes" if trial.review_valid else "no",
            style=style,
        )
    console.print(table)
    for trial in trials:
        if trial.error:
            console.print(
                f"[red]✗ {escape(str(trial.trial_name))}: "
                f"{escape(trial.error.splitlines()[0])}[/red]"
            )


def _review_command(
    path: Annotated[
        Path,
        typer.Argument(
            help="A finished rollout directory, or a job directory of rollouts"
        ),
    ],
    rubric: Annotated[
        Path | None,
        typer.Option(
            "--rubric",
            "-r",
            help=(
                "Rubric JSON file. Default: an admitted task copy's "
                "verifier/rubric.json (requires --tasks-root and verified "
                "digest), else the built-in default rubric."
            ),
        ),
    ] = None,
    prompt: Annotated[
        Path | None,
        typer.Option(
            "--prompt",
            "-p",
            help="Custom reviewer instruction template (see benchflow.review.prompts)",
        ),
    ] = None,
    agent: Annotated[
        str,
        typer.Option("--agent", "-a", help="Reviewer agent harness"),
    ] = "opencode",
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help=(
                f"Reviewer model (default: {DEFAULT_REVIEW_MODEL} with opencode, "
                "else the agent's registry default)"
            ),
        ),
    ] = None,
    environment: Annotated[
        str,
        typer.Option("--sandbox", help=f"Sandbox: {providers_phrase()}"),
    ] = "docker",
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", "-n", help="Max concurrent reviews"),
    ] = 4,
    passing: Annotated[
        bool,
        typer.Option("--passing", help="Only review passing rollouts (reward 1.0)"),
    ] = False,
    failing: Annotated[
        bool,
        typer.Option("--failing", help="Only review failing rollouts"),
    ] = False,
    timeout_sec: Annotated[
        int,
        typer.Option("--timeout-sec", help="Reviewer agent timeout per rollout"),
    ] = 1800,
    agent_env: Annotated[
        list[str] | None,
        typer.Option("--agent-env", help="KEY=VALUE for the reviewer (repeatable)"),
    ] = None,
    image: Annotated[
        str | None,
        typer.Option("--image", help="Prebuilt sandbox image for reviewer rollouts"),
    ] = None,
    tasks_root: Annotated[
        Path | None,
        typer.Option(
            "--tasks-root",
            help=(
                "Trusted directory holding the reviewed tasks. Required to "
                "include task definitions in evidence: a task path recorded "
                "inside a rollout is untrusted input and is never read "
                "directly. Without this, reviews use run records only."
            ),
        ),
    ] = None,
    allow_open_network: Annotated[
        bool,
        typer.Option(
            "--allow-open-network",
            help=(
                "Run reviewers WITHOUT the no-internet declaration. Required "
                "on backends that cannot enforce network isolation (e.g. "
                "agentcore) — reviews there fail closed by default. The "
                "report records the relaxed posture."
            ),
        ),
    ] = False,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            "-o",
            help="Review output directory (default: jobs/review-<ts>)",
        ),
    ] = None,
    pdf_tools: Annotated[
        bool,
        typer.Option(
            "--pdf-tools",
            help=(
                "Install PyMuPDF beside numpy and pypdf in each reviewer "
                "sandbox, to read and render a submitted PDF"
            ),
        ),
    ] = False,
    submitted_file_rule: Annotated[
        bool,
        typer.Option(
            "--submitted-file-rule",
            help=(
                "Tell the reviewer to judge a written deliverable by the file "
                "the agent submitted: read it first, cite pages, and take a "
                "compiled PDF over its sources"
            ),
        ),
    ] = False,
) -> None:
    """Grade finished rollouts against a rubric with a reviewer agent.

    Reviews run detached from the rollouts they grade: each review is an
    ordinary sandboxed rollout of a throwaway wrapper task, evidence is a
    read-only copy, and results land in ``review_report.json``. The reviewed
    rollouts' rewards and ``result.json`` are never modified.
    """
    from benchflow._utils.config import normalize_agent_name
    from benchflow.review import run_reviews
    from benchflow.review.wrapper import REVIEWER_IMAGE

    if allow_open_network:
        console.print(
            "[yellow]Warning: reviewers will run without network isolation "
            "(--allow-open-network).[/yellow]"
        )
    if passing and failing:
        print_error(
            "--passing and --failing cannot be used together",
            next_step="pass one of them, or neither to review every rollout",
        )
        raise typer.Exit(2)
    filter_passing = True if passing else (False if failing else None)
    if not path.exists():
        print_error(
            f"✗ {path} is not there",
            next_step="pass a finished rollout directory or a job directory",
        )
        raise typer.Exit(2)
    for flag, input_path in (
        ("--rubric", rubric),
        ("--prompt", prompt),
        ("--tasks-root", tasks_root),
    ):
        exit_if_missing(input_path, next_step=f"pass an existing path to {flag}")
    # The local environment can select the agent manifest catalog as well as
    # credentials, so load it before resolving the reviewer's name.
    _apply_dotenv_to_process_env()
    try:
        agent = normalize_agent_name(agent)
    except (KeyError, ValueError) as exc:
        print_error(str(exc), next_step="pass -a with a name `bench agents` lists")
        raise typer.Exit(2) from None
    if not is_known_provider(environment):
        print_error(
            f"Unknown sandbox: {environment!r}",
            next_step=f"pass --sandbox with one of: {providers_phrase()}",
        )
        raise typer.Exit(2)
    if model is None and agent == "opencode":
        # opencode has no registry default model, and the default reviewer
        # must run with no flag (dogfood p5-18).
        model = DEFAULT_REVIEW_MODEL

    parsed_env = _parse_agent_env(agent_env)
    # A missing reviewer key is one error line before any review starts, not
    # a traceback from inside the runner.
    from types import SimpleNamespace

    from benchflow.cli._errors import expected_error_parts
    from benchflow.errors import MissingCredentialError
    from benchflow.runtime import check_credentials

    try:
        check_credentials(
            [SimpleNamespace(agent=agent, model=model, agent_env=parsed_env)]
        )
    except MissingCredentialError as exc:
        text, hint = expected_error_parts(exc)
        print_error(f"✗ {text}", next_step=hint)
        raise typer.Exit(1) from None
    console.print("\n[blue]Reviewing rollout(s)...[/blue]")
    try:
        report, report_path = asyncio.run(
            run_reviews(
                path,
                agent=agent,
                model=model,
                environment=environment,
                rubric_path=rubric,
                prompt_path=prompt,
                agent_env=parsed_env,
                concurrency=concurrency,
                timeout_sec=timeout_sec,
                image=image if image is not None else REVIEWER_IMAGE,
                open_network=allow_open_network,
                tasks_root=tasks_root,
                filter_passing=filter_passing,
                out_dir=out_dir,
                pdf_tools=pdf_tools,
                submitted_file_rule=submitted_file_rule,
            )
        )
    except (ValueError, FileNotFoundError) as exc:
        print_error(
            f"✗ {exc}",
            next_step="`bench review --help` describes the rollout path and --rubric",
        )
        raise typer.Exit(1) from None

    if len(report.trials) == 1:
        _render_trial_review(report.trials[0])
        if report.trials[0].error:
            console.print(
                f"[red]✗ {escape(report.trials[0].error.splitlines()[0])}[/red]"
            )
    else:
        _render_review_overview(report.trials)
        if report.job_summary:
            console.print(f"\n[bold]Job summary:[/bold] {escape(report.job_summary)}")

    console.print(f"\n[bold]Report:[/bold] {escape(str(report_path))}")
    if any(trial.error for trial in report.trials):
        raise typer.Exit(1)


def register_review(app: typer.Typer) -> None:
    app.command("review", rich_help_panel="Core")(_review_command)
