"""``bench eval ablate`` — stage-level ablation over branch children.

Lives in its own module per the one-file-per-command-group convention;
:func:`register_eval_ablate` attaches it to the ``eval`` group. The command is
a thin caller: parsing arm specs, running the ablation, and turning the report
into a table or JSON all belong to :mod:`benchflow.ablate`, imported lazily
inside the command so ``bench --help`` never pays for the branch engine.

Exit codes mirror ``bench review``: 0 when every arm produced a reward, 1 when
any arm errored or was skipped, and 1 for a request that cannot run — always
as a one-line error on stderr, never a traceback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from benchflow.branch_stage import BRANCH_STAGES, STAGE_ENV_READY
from benchflow.cli._options import AgentOption, ModelOption, SandboxOption
from benchflow.cli._shared import _apply_dotenv_to_process_env, console, print_error

_STATUS_STYLES = {
    "pass": "green",
    "fail": "red",
    "error": "red",
    "skipped": "grey50",
}

DEFAULT_ARMS = "with-skill,no-skill"


def _render_ablation(report) -> None:
    """Print the parent's own reward, then one row per arm."""
    parent = "n/a" if report.parent_reward is None else f"{report.parent_reward:.2f}"
    console.print(
        f"\n[bold]Task:[/bold] {escape(report.task_id)}  "
        f"[bold]stage:[/bold] {escape(report.stage)}  "
        f"[bold]parent reward:[/bold] {parent}"
    )
    if report.parent_error:
        console.print(
            f"[yellow]Parent run error:[/yellow] "
            f"{escape(report.parent_error.splitlines()[0])}"
        )
    table = Table(
        title=f"Ablation at {escape(report.stage)}: {escape(report.task_id)}",
        show_lines=True,
    )
    for column in ("Arm", "Reward", "Result", "Wall clock", "Attribution"):
        table.add_column(column)
    for arm in report.arms:
        table.add_row(
            escape(arm.name),
            "-" if arm.reward is None else f"{arm.reward:.2f}",
            arm.status,
            "-" if arm.wall_clock_sec is None else f"{arm.wall_clock_sec:.0f}s",
            escape(arm.verdict),
            style=_STATUS_STYLES.get(arm.status, "white"),
        )
    console.print(table)
    if report.value is not None:
        console.print(f"[bold]V(parent) over the arms:[/bold] {report.value:.2f}")


def _ablate_command(
    tasks_dir: Annotated[
        Path,
        typer.Option(
            "--tasks-dir",
            help=(
                "Task directory to ablate (or a collection holding exactly one "
                "task) — the arms are the axis, the task is fixed"
            ),
        ),
    ],
    agent: AgentOption = "claude-agent-acp",
    model: ModelOption = None,
    sandbox: SandboxOption = "docker",
    at_stage: Annotated[
        str,
        typer.Option(
            "--at-stage",
            help=(
                f"Stage boundary to fork: {', '.join(BRANCH_STAGES)}. "
                "'post-research' is a mid-execute() cut point only "
                "Rollout.mark_stage() can record, so this command cannot "
                "capture it."
            ),
        ),
    ] = STAGE_ENV_READY,
    arms: Annotated[
        str,
        typer.Option(
            "--arms",
            help=(
                "Comma-separated arms, one branch child each: 'with-skill', "
                "'no-skill', 'inject:<path-to-file>'"
            ),
        ),
    ] = DEFAULT_ARMS,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            "-o",
            help="Ablation output directory (default: jobs/ablate-<ts>)",
        ),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the ablation report as JSON on stdout"),
    ] = False,
) -> None:
    """Run a task once, branch a stage boundary, and compare the arms.

    The counterfactual entry point of the rollout-branching RFC: the task runs
    once, the requested stage boundary is snapshotted as it passes, and that
    one world is forked into a child per arm — ``with-skill`` / ``no-skill``
    switch the skill mode, ``inject:<file>`` hands the child that file as its
    continuation prompt. Every arm therefore starts from a byte-identical world
    and differs by exactly one recorded delta. At ``--at-stage env-ready``
    every arm re-runs agent installation as its own rollout (that boundary
    precedes ``install_agent()``); at later boundaries the arms continue in
    place. Results land in a table plus ``ablation.json``.
    """
    import asyncio
    import json
    from datetime import datetime

    from benchflow.ablate import (
        AblationError,
        AblationRequest,
        parse_arms,
        resolve_ablation_task,
        run_ablation,
        validate_arms_for_stage,
        write_ablation_report,
    )

    try:
        parsed_arms = parse_arms(arms)
        stage = validate_arms_for_stage(parsed_arms, at_stage)
        task_path = resolve_ablation_task(tasks_dir)
    except (AblationError, ValueError) as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None

    if out_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        out_dir = Path("jobs") / f"ablate-{stamp}"

    _apply_dotenv_to_process_env()
    if not output_json:
        console.print(
            f"\n[blue]Ablating {escape(task_path.name)} at {escape(stage)}: "
            f"{escape(', '.join(arm.name for arm in parsed_arms))}[/blue]"
        )
    try:
        report = asyncio.run(
            run_ablation(
                AblationRequest(
                    task_path=task_path,
                    arms=parsed_arms,
                    agent=agent,
                    stage=stage,
                    model=model,
                    sandbox=sandbox,
                    out_dir=out_dir,
                )
            )
        )
    except (AblationError, ValueError, FileNotFoundError) as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None

    report_path = write_ablation_report(report, out_dir)
    if output_json:
        payload = report.to_dict()
        payload["report_path"] = str(report_path)
        typer.echo(json.dumps(payload, sort_keys=True, indent=2))
    else:
        _render_ablation(report)
        console.print(f"\n[bold]Report:[/bold] {escape(str(report_path))}")

    if report.error:
        print_error(report.error.splitlines()[0])
    for arm in report.arms:
        if arm.error:
            print_error(f"{arm.name}: {arm.error.splitlines()[0]}")
    if report.has_errors:
        raise typer.Exit(1)


def register_eval_ablate(eval_app: typer.Typer) -> None:
    eval_app.command("ablate")(_ablate_command)
