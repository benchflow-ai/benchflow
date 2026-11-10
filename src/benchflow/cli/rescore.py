"""``bench eval score`` — finish rubric scoring without rerunning the solver."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from benchflow._utils.scoring import score_summary_fields
from benchflow.cli._shared import (
    _apply_dotenv_to_process_env,
    _parse_agent_env,
    console,
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
from benchflow.review.outcome import scoring_from_result
from benchflow.review.persistence import write_json_atomic
from benchflow.review.resume import resume_review
from benchflow.trajectories.results import write_job_results_jsonl


def _refresh_job_artifacts(job_dir: Path) -> None:
    """Rebuild derived exports while retaining job configuration and provenance."""
    import json

    from benchflow.jobs import load_job
    from benchflow.trajectories.export import write_job_verifiers_jsonl
    from benchflow.trajectories.export_adp import write_job_adp_jsonl

    write_job_results_jsonl(job_dir)
    write_job_verifiers_jsonl(job_dir)
    write_job_adp_jsonl(job_dir)
    summary_path = job_dir / "summary.json"
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text())
    # Resume's completed-task filter deliberately omits agent errors. Derived
    # metrics instead use the canonical final attempts, including those errors.
    job = load_job(job_dir)
    rows = [
        json.loads((trial.path / "result.json").read_text()) for trial in job.trials
    ]
    summary.update(score_summary_fields(rows))
    from benchflow.setting_groups import setting_rows

    summary["scoring"]["settings"] = setting_rows(
        job.trials, summaries=[summary]
    ).to_dict()
    if "planned" in summary:
        from benchflow._utils.scoring import pass_rate

        # The scoring rule's passes, as the run's own score_of_plan counts
        # them (summary.json's "passed" is the audit count).
        summary["score_of_plan"] = pass_rate(
            passed=summary["scoring"]["passed"], total=summary["planned"]
        )
    previous_rates = summary.get("solve_rates") or {}
    summary["solve_rates"] = job.solve_rates(
        solve_threshold=previous_rates.get("solve_threshold"), include_controls=True
    ).to_dict()
    write_json_atomic(summary_path, summary)
    root_summary_path = job_dir.parent / "summary.json"
    if root_summary_path.is_file():
        root_summary = json.loads(root_summary_path.read_text())
        if root_summary.get("job_name") == summary.get("job_name"):
            write_json_atomic(root_summary_path, summary)


def eval_score(
    path: Annotated[Path, typer.Argument(help="A saved task rollout directory")],
    tasks_root: Annotated[
        Path,
        typer.Option(
            "--tasks-root", help="Trusted original task directory or collection"
        ),
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Create a new revision even if scoring already completed"
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
    """Retry a failed reviewer using the saved workspace and trajectory."""
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
    try:
        result = asyncio.run(
            resume_review(path, tasks_root=tasks_root, reviewer=reviewer, force=force)
        )
        _refresh_job_artifacts(path.resolve().parent)
        scoring = scoring_from_result(result)
    except (ValueError, RuntimeError, OSError) as exc:
        print_error(
            f"Scoring failed: {exc}",
            next_step="pass a saved trial folder and the task it ran with --tasks-root",
        )
        raise typer.Exit(1) from exc
    if (
        scoring is None
        and result.get("rewards") is not None
        and not result.get("verifier_error")
    ):
        console.print(f"Verification complete: rewards={result['rewards']}")
        return
    if scoring is None or scoring.status != "complete":
        reason = scoring.error if scoring is not None else "No scoring verdict"
        print_error(
            f"Scoring incomplete: {reason}",
            next_step="fix that, then score again with `bench eval score --force`",
        )
        raise typer.Exit(1)
    rewards = scoring.numeric_rewards()
    assert rewards is not None
    console.print(
        f"Scoring complete: passed={scoring.passed}, reward={rewards['reward']:.3f}"
    )


def register_eval_score(eval_app: typer.Typer) -> None:
    eval_app.command("score")(eval_score)
