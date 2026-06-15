"""``bench eval adopt`` — bring an external benchmark into benchflow.

``bench eval adopt`` is a single multi-mode command:

* ``bench eval adopt <source>`` — scaffold (if needed) and drive the codex
  conversion of an upstream benchmark into ``benchmarks/<name>/``;
* ``bench eval adopt <name> --scaffold-only`` — only scaffold the package;
* ``bench eval adopt <name> --verify`` — run the parity gate for the benchmark.

It lives under ``eval`` because ``eval`` is the universal benchmark entry point
(``eval create`` runs a benchmark; ``eval adopt`` is the manual path to make a
foreign benchmark runnable). It previously was a subgroup with ``init`` /
``convert`` / ``verify`` subcommands, and before that ``bench agent
create|run|verify`` (#735); both ``bench adopt init|convert|verify`` and ``bench
agent create|run|verify`` now stay as hidden deprecated aliases through 0.6.

``register_eval_adopt`` registers the single canonical command onto the ``eval``
Typer; ``register_adopt_deprecated`` mounts the hidden top-level alias group.
``cli/main.py`` only wires the calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from benchflow.agent_router import (
    ADOPT_VERBS,
    DEFAULT_REWARD_TOLERANCE,
    InvalidBenchmarkName,
    default_benchmarks_dir,
    derive_name_from_source,
    register_agent_router,
    run_convert_action,
    run_scaffold_action,
    run_verify_action,
)
from benchflow.cli._shared import console


def register_eval_adopt(eval_app: typer.Typer) -> None:
    """Register the single canonical ``bench eval adopt`` command onto ``eval``."""

    @eval_app.command("adopt")
    def adopt(
        target: Annotated[
            str | None,
            typer.Argument(
                help="In convert mode (default) the SOURCE repo/path to adopt; in "
                "--verify / --scaffold-only mode the benchmark SLUG."
            ),
        ] = None,
        name: Annotated[
            str | None,
            typer.Option(
                "--name",
                help="Benchmark slug for convert mode (default: derived from source)",
            ),
        ] = None,
        verify: Annotated[
            bool,
            typer.Option(
                "--verify", help="Run the parity gate for the named benchmark"
            ),
        ] = False,
        scaffold_only: Annotated[
            bool,
            typer.Option(
                "--scaffold-only", help="Only scaffold the package, do not convert"
            ),
        ] = False,
        # convert-mode flags
        model: Annotated[
            str | None, typer.Option("--model", help="Model for the codex driver")
        ] = None,
        dry_run: Annotated[
            bool,
            typer.Option("--dry-run", help="Print the launch command, do not run"),
        ] = False,
        codex_bin: Annotated[
            str, typer.Option("--codex-bin", help="Host codex binary")
        ] = "codex",
        codex_config: Annotated[
            list[str] | None,
            typer.Option(
                "--codex-config",
                "-c",
                help="Codex config override as key=value, passed to codex as "
                "`-c key=value`; repeatable (e.g. -c service_tier=flex to work "
                "around host ~/.codex/config.toml drift)",
            ),
        ] = None,
        # verify-mode flags (--benchmarks-dir is also used by scaffold)
        benchmarks_dir: Annotated[
            Path | None,
            typer.Option("--benchmarks-dir", help="Target benchmarks/ directory"),
        ] = None,
        tolerance: Annotated[
            float,
            typer.Option("--tolerance", help="Max abs reward delta (statistical)"),
        ] = DEFAULT_REWARD_TOLERANCE,
        issue_out: Annotated[
            Path | None,
            typer.Option("--issue-out", help="Write the divergence issue draft here"),
        ] = None,
        roundtrip_task: Annotated[
            Path | None,
            typer.Option(
                "--roundtrip-task",
                help="Also run the structural round-trip check on this task dir",
            ),
        ] = None,
        rerun: Annotated[
            bool,
            typer.Option(
                "--rerun",
                help="Independently re-execute parity_test.py --mode side-by-side "
                "and score its fresh output, instead of the recorded "
                "parity_experiment.json",
            ),
        ] = False,
    ) -> None:
        """Adopt an external benchmark into benchflow.

        Default (convert): `bench eval adopt <source>` scaffolds benchmarks/<name>/
        if missing, then drives the codex conversion. `--scaffold-only` only writes
        the package; `--verify` runs the parity gate for the named benchmark.
        """
        if verify and scaffold_only:
            console.print(
                "[red]--verify and --scaffold-only are mutually exclusive[/red]"
            )
            raise typer.Exit(2)
        if target is None:
            console.print(
                "[red]missing target: pass a SOURCE (convert) or a benchmark "
                "SLUG (--verify / --scaffold-only)[/red]"
            )
            raise typer.Exit(2)

        if verify:
            run_verify_action(
                target,
                benchmarks_dir=benchmarks_dir,
                tolerance=tolerance,
                issue_out=issue_out,
                roundtrip_task=roundtrip_task,
                rerun=rerun,
                console=console,
            )
            return

        if scaffold_only:
            run_scaffold_action(target, benchmarks_dir, console=console)
            return

        # Convert mode: resolve the slug, auto-scaffold if the package is
        # missing (a no-op if it already exists), then drive the conversion.
        # --dry-run only prints the command, so it must not write any files.
        try:
            slug = name or derive_name_from_source(target)
        except InvalidBenchmarkName as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")
            raise typer.Exit(1) from exc
        root = benchmarks_dir or default_benchmarks_dir()
        if not dry_run and not (root / slug).exists():
            run_scaffold_action(slug, benchmarks_dir, console=console)
        run_convert_action(
            target,
            slug,
            model=model,
            dry_run=dry_run,
            codex_bin=codex_bin,
            codex_config=codex_config,
            console=console,
        )


def register_adopt_deprecated(app: typer.Typer) -> None:
    """Attach the hidden deprecated top-level ``bench adopt`` → ``bench eval adopt``."""
    adopt_app = typer.Typer(help="deprecated — use `bench eval adopt`.")
    app.add_typer(adopt_app, name="adopt", hidden=True)
    register_agent_router(adopt_app, verbs=ADOPT_VERBS, deprecated_as="adopt")
