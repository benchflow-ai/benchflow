"""``bench bundle`` — one shareable zip of a run's evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from benchflow.cli._shared import console, exit_if_missing, print_error


def eval_bundle(
    path: Annotated[
        Path, typer.Argument(help="A job folder, or one trial folder inside it")
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out", "-o", help="Where to write the zip (default: <PATH>.bundle.zip)"
        ),
    ] = None,
    workspace: Annotated[
        bool,
        typer.Option(
            "--workspace/--no-workspace",
            help="Include each trial's saved workspace (evidence/) when it has one",
        ),
    ] = True,
) -> None:
    """Write one zip of a run's evidence to share, redacted.

    Per trial the result, trajectory, verifier output, reviews, regrades,
    artifacts and saved workspace; for a job its run card, summary and report.
    Every text file is redacted as a trajectory upload is. bundle.json lists
    each file with its SHA-256 and what was masked, and what was left out.
    """
    from benchflow.evidence_bundle import BundleError, build_bundle

    exit_if_missing(path, next_step="pass a job folder or a trial folder")
    target = out or path.with_name(path.name + ".bundle.zip")
    try:
        result = build_bundle(path, target, workspace=workspace)
    except BundleError as exc:
        print_error(str(exc), next_step="pass a job folder or a trial folder")
        raise typer.Exit(1) from None
    masked = sum(result.masked.values())
    console.print(
        f"Bundle: {escape(str(result.path))} ({len(result.trials)} trial(s), "
        f"{result.files} files, sha256 {result.sha256[:12]}…); "
        f"{masked} value(s) masked"
        + (
            f"; {len(result.left_out)} file(s) left out (bundle.json says why)"
            if result.left_out
            else ""
        ),
        highlight=False,
        soft_wrap=True,
    )


def register_eval_bundle(eval_app: typer.Typer) -> None:
    eval_app.command("bundle")(eval_bundle)
