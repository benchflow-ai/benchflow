"""``bench eval regrade`` — re-score stored trials with a changed verifier."""

from __future__ import annotations

import json
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
from benchflow.cli.reviewer_options import (
    ReviewerAgentOption,
    ReviewerConcurrencyOption,
    ReviewerEffortOption,
    ReviewerEnvOption,
    ReviewerEvidenceTimeoutOption,
    ReviewerIdleTimeoutOption,
    ReviewerImageOption,
    ReviewerMemoryOption,
    ReviewerModelOption,
    ReviewerNetworkOption,
    ReviewerPdfToolsOption,
    ReviewerSandboxOption,
    ReviewerSetupTimeoutOption,
    ReviewerStorageOption,
    ReviewerSubmittedFileRuleOption,
    ReviewerTimeoutOption,
    reviewer_from_cli,
    reviewer_idle_timeout,
)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:g}"


def eval_regrade(
    path: Annotated[
        Path, typer.Argument(help="A job folder, or one trial folder inside it")
    ],
    tasks_dir: Annotated[
        Path | None,
        typer.Option(
            "--tasks-dir",
            help=(
                "Folder of task folders (or one task folder) holding the changed "
                "verifiers. Default: the task path each trial recorded."
            ),
        ),
    ] = None,
    sandbox: Annotated[
        str | None,
        typer.Option(
            "--sandbox", help="Backend for the fresh sandboxes (default: as run)"
        ),
    ] = None,
    concurrency: Annotated[
        int, typer.Option("--concurrency", min=1, help="Trials regraded at once")
    ] = 4,
    reason: Annotated[
        str | None,
        typer.Option("--reason", help="Why the verifier changed; kept in each block"),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the summary as JSON")
    ] = False,
    review: Annotated[
        bool,
        typer.Option(
            "--review",
            help=(
                "Also rerun each trial's rubric review on its saved outputs, "
                "gated by the new test, and compare the combined score. The "
                "reviewer is the one the trial recorded, changed by --reviewer-*"
            ),
        ),
    ] = False,
    reviewer_agent: ReviewerAgentOption = None,
    reviewer_model: ReviewerModelOption = None,
    reviewer_reasoning_effort: ReviewerEffortOption = None,
    reviewer_sandbox: ReviewerSandboxOption = None,
    reviewer_timeout_sec: ReviewerTimeoutOption = None,
    reviewer_idle_timeout_sec: ReviewerIdleTimeoutOption = None,
    reviewer_storage_mb: ReviewerStorageOption = None,
    reviewer_memory_mb: ReviewerMemoryOption = None,
    reviewer_setup_timeout_sec: ReviewerSetupTimeoutOption = None,
    reviewer_evidence_timeout_sec: ReviewerEvidenceTimeoutOption = None,
    reviewer_concurrency: ReviewerConcurrencyOption = None,
    reviewer_image: ReviewerImageOption = None,
    reviewer_agent_env: ReviewerEnvOption = None,
    reviewer_open_network: ReviewerNetworkOption = None,
    reviewer_pdf_tools: ReviewerPdfToolsOption = None,
    reviewer_submitted_file_rule: ReviewerSubmittedFileRuleOption = None,
) -> None:
    """Re-run the task's current verifier on each trial's frozen workspace.

    The new score is written beside the original (regrade.json in each trial,
    regrade-summary.json in PATH); result.json is never changed. Trials whose
    workspace was not frozen (--freeze-workspace) are listed as not regradable.
    """
    from benchflow.eval_regrade import regrade

    exit_if_missing(
        path, next_step="pass a job or trial folder of a run with --freeze-workspace"
    )
    _apply_dotenv_to_process_env()
    reviewer = reviewer_from_cli(
        agent=reviewer_agent,
        model=reviewer_model,
        reasoning_effort=reviewer_reasoning_effort,
        environment=reviewer_sandbox,
        timeout_sec=reviewer_timeout_sec,
        idle_timeout_sec=reviewer_idle_timeout(reviewer_idle_timeout_sec),
        storage_mb=reviewer_storage_mb,
        memory_mb=reviewer_memory_mb,
        setup_timeout_sec=reviewer_setup_timeout_sec,
        evidence_timeout_sec=reviewer_evidence_timeout_sec,
        concurrency=reviewer_concurrency,
        image=reviewer_image,
        agent_env=_parse_agent_env(reviewer_agent_env) if reviewer_agent_env else None,
        open_network=reviewer_open_network,
        pdf_tools=reviewer_pdf_tools,
        submitted_file_rule=reviewer_submitted_file_rule,
    )
    if reviewer is not None and not review:
        print_error(
            "the reviewer options (--reviewer-model and the rest) apply only "
            "with --review",
            next_step="add --review to rerun each trial's rubric review",
        )
        raise typer.Exit(2)
    try:
        summary = regrade(
            path,
            tasks_dir=tasks_dir,
            sandbox=sandbox,
            concurrency=concurrency,
            reason=reason,
            review=review,
            reviewer=reviewer,
        )
    except (FileNotFoundError, ValueError) as exc:
        # stderr, like every error: under --json it becomes the one JSON object.
        print_error(
            f"Regrade failed: {exc}",
            next_step="pass a job or trial folder of a run with --freeze-workspace",
        )
        raise typer.Exit(1) from exc
    if as_json:
        console.print_json(json.dumps(summary.to_dict()))
    else:
        table = Table(title=f"Regrade {summary.regrade_id}")
        for column in ("Trial", "Original", "New", "Verdict"):
            table.add_column(column)
        for row in summary.trials:
            if row.status == "regraded":
                verdict = row.change or "same"
                if row.changed and row.task_changed is False:
                    verdict += " (task unchanged: state outside the workspace?)"
            elif row.status == "failed":
                verdict = f"failed: {row.reason}"
            else:
                verdict = f"not regradable: {row.reason}"
            table.add_row(
                escape(row.trial),
                _fmt(row.original_reward),
                _fmt(row.new_reward),
                escape(verdict),
            )
        console.print(table)
        counts = summary.counts()
        console.print(
            f"{counts['regraded']}/{counts['trials']} regraded, "
            f"{counts['changed']} changed ({counts['fail_to_pass']} fail->pass, "
            f"{counts['pass_to_fail']} pass->fail), {counts['failed']} failed, "
            f"{counts['not_regradable']} not regradable. "
            f"Summary: {escape(str(Path(summary.path) / 'regrade-summary.json'))}"
        )
        if any(row.compared == "verifier_reward" for row in summary.trials):
            console.print(
                "Trials with a rubric review compare the test's own reward "
                "(verifier_reward): a regrade reruns the test, not the review "
                "(--review reruns it too)."
            )
        if review:
            console.print(
                "Rubric reviews ran again: rewards are the combined score "
                "(tests, blockers and quality)."
            )
        flipped = [row for row in summary.changed if row.task_changed is False]
        if flipped:
            console.print(
                f"[yellow]{len(flipped)} changed verdict(s) on an unchanged "
                "task: the verifier read state a regrade does not restore "
                "(packages installed system-wide, services, files outside the "
                "workspace) or is nondeterministic. See docs/scale/resume.md#regrade-saved-work.[/yellow]"
            )
    if summary.failed:
        raise typer.Exit(1)


def register_eval_regrade(eval_app: typer.Typer) -> None:
    eval_app.command("regrade")(eval_regrade)
