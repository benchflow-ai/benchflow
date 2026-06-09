"""``benchflow continue`` — resume a timed-out run to completion.

Standalone command (does not touch the normal eval/run path): reconstruct a
previous unfinished ``openhands`` run's exact env + memory from its recorded
trajectory via record-replay, continue it live, and write a new HF-compatible
folder linked to the parent. See :mod:`benchflow.continue_run`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Annotated

import typer

logger = logging.getLogger(__name__)


def _load_env_defaults() -> None:
    from benchflow._dotenv import load_dotenv_env

    for key, value in load_dotenv_env().items():
        os.environ.setdefault(key, value)


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
        proxy_mode: Annotated[
            str,
            typer.Option(
                "--proxy-mode",
                help=(
                    "Replay proxy placement: auto, host, or sandbox. Auto uses "
                    "sandbox-local replay for Daytona/Modal and host replay for Docker."
                ),
            ),
        ] = "auto",
    ) -> None:
        """Resume a previous unfinished (timed-out) openhands run to completion."""
        from benchflow.continue_run.orchestrator import continue_run
        from benchflow.continue_run.run_folder import RunFolderError

        _load_env_defaults()

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
                    proxy_mode=proxy_mode,
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

    @app.command("continue-batch")
    def continue_batch_cmd(
        root: Annotated[
            Path,
            typer.Argument(
                help=(
                    "Run folder or directory tree containing timeout run folders "
                    "(config.json + trajectory/llm_trajectory.jsonl)."
                )
            ),
        ],
        tasks_dir: Annotated[
            Path | None,
            typer.Option(
                "--tasks-dir",
                help="Directory holding task sources; required unless recorded task_path exists.",
            ),
        ] = None,
        model: Annotated[
            str | None,
            typer.Option("--model", help="Override live-continuation model."),
        ] = None,
        timeout: Annotated[
            int | None,
            typer.Option("--timeout", help="Wall-clock budget per continuation."),
        ] = None,
        output: Annotated[
            Path | None,
            typer.Option("--output", help="Output jobs dir for continued runs."),
        ] = None,
        concurrency: Annotated[
            int,
            typer.Option(
                "--concurrency",
                help="Maximum number of continuation runs in flight.",
            ),
        ] = 100,
        limit: Annotated[
            int | None,
            typer.Option("--limit", help="Limit discovered timeout folders."),
        ] = None,
        strict_divergence: Annotated[
            bool,
            typer.Option(
                "--strict-divergence/--no-strict-divergence",
                help="Abort a run if replay leaves the original rails.",
            ),
        ] = False,
        proxy_mode: Annotated[
            str,
            typer.Option(
                "--proxy-mode",
                help=(
                    "Replay proxy placement: auto, host, or sandbox. For PR5 "
                    "Daytona runs, use the default auto or sandbox."
                ),
            ),
        ] = "auto",
    ) -> None:
        """Continue all timed-out OpenHands runs under a directory tree."""
        import json

        from benchflow.continue_run.batch import (
            continue_batch,
            discover_timeout_run_folders,
            summarize_batch,
        )

        _load_env_defaults()
        folders = discover_timeout_run_folders(root, limit=limit)
        if not folders:
            typer.secho("No timeout run folders found.", fg=typer.colors.YELLOW)
            return

        typer.echo(
            f"Continuing {len(folders)} timeout run(s) with concurrency={concurrency}"
        )
        results = asyncio.run(
            continue_batch(
                folders,
                concurrency=concurrency,
                tasks_dir=tasks_dir,
                model=model,
                timeout=timeout,
                output_dir=output,
                require_timeout=True,
                strict_divergence=strict_divergence,
                proxy_mode=proxy_mode,
            )
        )
        summary = summarize_batch(results)
        typer.echo(json.dumps(summary, indent=2))
        if summary["failed"]:
            raise typer.Exit(1)
