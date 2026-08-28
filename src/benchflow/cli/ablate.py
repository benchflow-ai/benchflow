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
from benchflow.cli._options import (
    AgentOption,
    EnvironmentManifestOption,
    ModelOption,
    SandboxOption,
)
from benchflow.cli._shared import _apply_dotenv_to_process_env, console, print_error

_STATUS_STYLES = {
    "pass": "green",
    "fail": "red",
    "error": "red",
    "skipped": "grey50",
}

DEFAULT_ARMS = "with-skill,no-skill"


_TEST_STATUS_STYLES = {"passed": "green", "failed": "red", "skipped": "grey50"}


def _render_sub_test_attribution(report) -> None:
    """Print the sub-test section: only the tests whose outcome differs.

    The second half of the attribution, and the half a scalar tie hides. Tying
    tests are counted, never listed — they are noise for attribution — and the
    section always ends on the summary line, so a reader is told *why* there is
    no table (all tests tie / no per-test data was mined) rather than left to
    read the absence.
    """
    from benchflow.ablate import sub_test_attribution

    section = sub_test_attribution(report.arms)
    differing = section["differing_tests"]
    if differing:
        arms = section["arms_with_tests"]
        table = Table(
            title=f"Sub-test outcomes that differ ({len(differing)})",
            show_lines=True,
        )
        table.add_column("Test")
        for arm in arms:
            table.add_column(escape(arm))
        for entry in differing:
            cells = []
            for arm in arms:
                outcome = entry["outcomes"].get(arm)
                style = _TEST_STATUS_STYLES.get(outcome or "", "yellow")
                cells.append(
                    f"[{style}]{escape(outcome) if outcome else 'not reported'}[/{style}]"
                )
            table.add_row(escape(entry["test"]), *cells)
        console.print(table)
        if section["tying_tests"]:
            console.print(
                f"[dim]{len(section['tying_tests'])} test(s) tie across the "
                "arms and are omitted.[/dim]"
            )
    console.print(f"[bold]Sub-test attribution:[/bold] {escape(section['summary'])}")


def _render_ablation(report) -> None:
    """Print the parent's own reward, then one row per arm."""
    parent = "n/a" if report.parent_reward is None else f"{report.parent_reward:.2f}"
    console.print(
        f"\n[bold]Task:[/bold] {escape(report.task_id)}  "
        f"[bold]stage:[/bold] {escape(report.stage)}  "
        f"[bold]parent reward:[/bold] {parent}"
    )
    if report.environment:
        env = report.environment
        world = env.get("env_hash") or env.get("image") or env.get("base_image")
        console.print(
            f"[bold]Environment:[/bold] {escape(str(env.get('name')))} "
            f"({escape(str(env.get('ref')))}, {escape(str(world))})"
        )
    if report.stage_snapshot:
        # The recorded roll-back handles of the branched stage — enough to
        # restore this world and re-branch it by hand later.
        snap = report.stage_snapshot
        console.print(
            f"[bold]Stage snapshot:[/bold] "
            f"sandbox={escape(str(snap.get('sandbox_ref') or '-'))}  "
            f"environment={escape(str(snap.get('environment_ref') or '-'))}"
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
    _render_sub_test_attribution(report)
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
    reasoning_effort: Annotated[
        str | None,
        typer.Option(
            "--reasoning-effort",
            help=(
                "Agent reasoning/thinking effort when the agent exposes one "
                "(e.g. max) — the same control as bench eval run, recorded in "
                "the parent's and every arm's config"
            ),
        ),
    ] = None,
    sandbox: SandboxOption = "docker",
    environment_manifest: EnvironmentManifestOption = None,
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
                "'no-skill', 'inject:<path-to-file>', "
                "'config:<inline-json-or-@file>', 'env:<registry-ref>'"
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
    keep_snapshots: Annotated[
        bool,
        typer.Option(
            "--keep-snapshots",
            help=(
                "Export the branched stage's sandbox snapshot image (docker "
                "save) to <out-dir>/snapshots/<ref>.tar before cleanup "
                "destroys it; ablation.json records the tar path and sha256. "
                "Without it the snapshot dies with the run and its handle is "
                "recorded as ephemeral"
            ),
        ),
    ] = False,
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
    continuation prompt, ``config:<patch>`` runs the child under the parent's
    config with the allowlisted patch deep-merged on top (#790 machinery),
    ``env:<ref>`` provisions a different registry manifest's service set over
    the restored world (same image only — the tool-outage pattern).
    Every arm therefore starts from a byte-identical world
    and differs by exactly one recorded delta. At ``--at-stage env-ready``
    every arm re-runs agent installation as its own rollout (that boundary
    precedes ``install_agent()``); at later boundaries the arms continue in
    place. ``--environment-manifest`` binds the parent's (and therefore every
    arm's) environment explicitly, beating the task-declared manifest — the
    same precedence as ``bench eval run``. Results land in a table plus
    ``ablation.json`` — which stamps the bound environment (name, ref,
    content hash) and the branched stage's snapshot refs — followed by a
    second table naming the sub-tests whose outcome differs across the arms:
    the behavioral difference a tied scalar reward would otherwise hide.
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
                    reasoning_effort=reasoning_effort,
                    sandbox=sandbox,
                    out_dir=out_dir,
                    environment_manifest=environment_manifest,
                    keep_snapshots=keep_snapshots,
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
