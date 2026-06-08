"""``benchflow continue`` — resume a timed-out run to completion.

Standalone command (does not touch the normal eval/run path): reconstruct a
previous unfinished ``openhands`` run's exact env + memory from its recorded
trajectory via record-replay, continue it live, and write a new HF-compatible
folder linked to the parent. See :mod:`benchflow.continue_run`.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer

logger = logging.getLogger(__name__)


def register_continue(app: typer.Typer) -> None:
    """Attach the ``continue`` command to the top-level benchflow app."""

    @app.command("continue")
    def continue_cmd(
        folder: Annotated[
            Path,
            typer.Argument(
                help="Original run output folder (contains config.json + "
                "trajectory/llm_trajectory.jsonl)."
            ),
        ],
        tasks_dir: Annotated[
            Path | None,
            typer.Option(
                "--tasks-dir",
                help="Directory holding the task source (instruction + verifier). "
                "Required unless the recorded task_path still exists on disk.",
            ),
        ] = None,
        model: Annotated[
            str | None,
            typer.Option(
                "--model",
                help="Override the live-continuation model (default: the "
                "original run's model). Tests use gemini-3.1-flash-lite-preview.",
            ),
        ] = None,
        timeout: Annotated[
            int | None,
            typer.Option(
                "--timeout",
                help="Wall-clock budget for the continuation, in seconds "
                "(default: the original run's timeout).",
            ),
        ] = None,
        output: Annotated[
            Path | None,
            typer.Option(
                "--output",
                help="Output jobs dir for the new run (default: "
                "<orig-parent>/continued).",
            ),
        ] = None,
        require_timeout: Annotated[
            bool,
            typer.Option(
                "--require-timeout/--no-require-timeout",
                help="Refuse runs whose recorded status is not a timeout.",
            ),
        ] = False,
        strict_divergence: Annotated[
            bool,
            typer.Option(
                "--strict-divergence/--no-strict-divergence",
                help="Abort if replay leaves the original rails (message-count "
                "mismatch) instead of warning.",
            ),
        ] = False,
        replay_only: Annotated[
            bool,
            typer.Option(
                "--replay-only/--no-replay-only",
                help="Rebuild the env via replay and stop at the cut-point "
                "(no live model needed) — useful for testing.",
            ),
        ] = False,
    ) -> None:
        """Resume a previous unfinished (timed-out) openhands run to completion."""
        from benchflow._dotenv import load_dotenv_env
        from benchflow.continue_run.orchestrator import continue_run
        from benchflow.continue_run.run_folder import RunFolderError

        load_dotenv_env()

        try:
            result = asyncio.run(
                continue_run(
                    folder,
                    tasks_dir=tasks_dir,
                    model=model,
                    timeout=timeout,
                    output_dir=output,
                    require_timeout=require_timeout,
                    strict_divergence=strict_divergence,
                    replay_only=replay_only,
                )
            )
        except RunFolderError as exc:
            typer.secho(f"benchflow continue: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from exc

        typer.secho(
            f"\n✓ continued run written to {result.rollout_dir}", fg=typer.colors.GREEN
        )
        typer.echo(
            f"  replayed {result.n_recorded} recorded turn(s); "
            f"{result.n_live} live turn(s); {result.divergences} divergence(s)"
        )
        if result.rewards is not None:
            typer.echo(f"  rewards: {result.rewards}")
        if result.error:
            typer.secho(f"  agent error: {result.error}", fg=typer.colors.YELLOW)
