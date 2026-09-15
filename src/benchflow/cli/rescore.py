"""``bench eval score`` — finish rubric scoring without rerunning the solver."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from benchflow._utils.result_paths import load_task_results
from benchflow._utils.scoring import score_summary_fields
from benchflow.cli._shared import (
    _apply_dotenv_to_process_env,
    _parse_agent_env,
    console,
)
from benchflow.cli.reviewer_options import (
    ReviewerAgentOption,
    ReviewerConcurrencyOption,
    ReviewerEffortOption,
    ReviewerEnvOption,
    ReviewerImageOption,
    ReviewerModelOption,
    ReviewerNetworkOption,
    ReviewerSandboxOption,
    ReviewerTimeoutOption,
    reviewer_from_cli,
)
from benchflow.review.outcome import scoring_from_result
from benchflow.review.persistence import write_json_atomic
from benchflow.review.resume import resume_review
from benchflow.trajectories.results import write_job_results_jsonl


def _refresh_job_artifacts(job_dir: Path) -> None:
    """Rebuild derived exports while retaining job configuration and provenance."""
    import json

    from benchflow.trajectories.export import write_job_verifiers_jsonl
    from benchflow.trajectories.export_adp import write_job_adp_jsonl

    write_job_results_jsonl(job_dir)
    write_job_verifiers_jsonl(job_dir)
    write_job_adp_jsonl(job_dir)
    summary_path = job_dir / "summary.json"
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text())
    summary.update(score_summary_fields(load_task_results(job_dir).values()))
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
    reviewer_concurrency: ReviewerConcurrencyOption = None,
    reviewer_image: ReviewerImageOption = None,
    reviewer_agent_env: ReviewerEnvOption = None,
    reviewer_open_network: ReviewerNetworkOption = None,
) -> None:
    """Retry a failed reviewer using the saved workspace and trajectory."""
    _apply_dotenv_to_process_env()
    reviewer = reviewer_from_cli(
        agent=reviewer_agent,
        model=reviewer_model,
        reasoning_effort=reviewer_reasoning_effort,
        environment=reviewer_sandbox,
        timeout_sec=reviewer_timeout_sec,
        concurrency=reviewer_concurrency,
        image=reviewer_image,
        agent_env=_parse_agent_env(reviewer_agent_env) if reviewer_agent_env else None,
        open_network=reviewer_open_network,
    )
    try:
        result = asyncio.run(
            resume_review(path, tasks_root=tasks_root, reviewer=reviewer, force=force)
        )
        _refresh_job_artifacts(path.resolve().parent)
        scoring = scoring_from_result(result)
    except (ValueError, RuntimeError, OSError) as exc:
        console.print(f"[red]Scoring failed: {escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    if scoring is None or scoring.status != "complete":
        reason = scoring.error if scoring is not None else "No scoring verdict"
        console.print(f"[red]Scoring incomplete: {escape(str(reason))}[/red]")
        raise typer.Exit(1)
    rewards = scoring.numeric_rewards()
    assert rewards is not None
    console.print(
        f"Scoring complete: passed={scoring.passed}, reward={rewards['reward']:.3f}"
    )


def register_eval_score(eval_app: typer.Typer) -> None:
    eval_app.command("score")(eval_score)
