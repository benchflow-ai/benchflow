"""``bench tasks`` — task authoring commands (init / check / migrate /
normalize / export), plus the trace-import generators wired from
:mod:`benchflow.cli.trace_import`.

Registered onto the top-level app by :func:`register_tasks`; ``cli/main.py``
only wires the call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from rich.markup import escape

from benchflow.cli._shared import console
from benchflow.cli.trace_import import register_tasks_generate
from benchflow.sandbox.providers import providers_phrase


def register_tasks(app: typer.Typer) -> None:
    """Attach the ``tasks`` command group to the top-level benchflow app."""
    tasks_app = typer.Typer(help="Task authoring commands")
    app.add_typer(tasks_app, name="tasks", rich_help_panel="Core")

    register_tasks_generate(tasks_app)

    @tasks_app.command("init")
    def tasks_init(
        name: Annotated[str, typer.Argument(help="Task name")],
        parent_dir: Annotated[
            Path,
            typer.Option("--dir", help="Parent directory (default: tasks/)"),
        ] = Path("tasks"),
        no_pytest: Annotated[
            bool, typer.Option("--no-pytest", help="Skip pytest template")
        ] = False,
        no_oracle: Annotated[
            bool,
            typer.Option(
                "--no-oracle",
                "--no-solution",
                help="Skip oracle template",
            ),
        ] = False,
        task_format: Annotated[
            str, typer.Option("--format", help="Task format: legacy or task-md")
        ] = "task-md",
    ) -> None:
        """Scaffold a new benchmark task."""
        from benchflow._utils.task_authoring import scaffold_task

        try:
            result = scaffold_task(
                name,
                parent_dir=parent_dir,
                no_pytest=no_pytest,
                no_oracle=no_oracle,
                task_format=cast(Literal["legacy", "task-md"], task_format),
            )
            console.print(f"[green]Created:[/green] {result.task_dir}/")
            # List every file actually written, derived from the scaffold itself
            # so the summary can never under-report (e.g. omit
            # verifier/test_outputs.py or verifier/rubrics/verifier.toml, both of
            # which `bench tasks check` validates).
            for rel in result.files:
                console.print(f"  {rel}")
        except (OSError, ValueError) as e:
            # OSError covers FileExistsError plus the NotADirectoryError /
            # PermissionError that mkdir() raises for `--dir <file>` or a
            # read-only parent — siblings (migrate/normalize/export) already
            # degrade gracefully; init was the outlier that dumped a traceback.
            # escape(): the OSError message echoes the user-supplied path.
            console.print(f"[red]{escape(str(e))}[/red]")
            raise typer.Exit(1) from None

    @tasks_app.command("check")
    def tasks_check(
        task_dir: Annotated[Path, typer.Argument(help="Path to task directory")],
        validation_level: Annotated[
            Literal[
                "schema",
                "structural",
                "runtime-capability",
                "publication-grade",
                "acceptance",
                "acceptance-live",
            ],
            typer.Option(
                "--level",
                help=(
                    "Validation level: schema, structural, runtime-capability, "
                    "publication-grade, acceptance, or acceptance-live"
                ),
            ),
        ] = "structural",
        sandbox: Annotated[
            str | None,
            typer.Option(
                "--sandbox",
                help=f"Also validate parsed runtime semantics for {providers_phrase()}",
            ),
        ] = None,
        report_output: Annotated[
            Path | None,
            typer.Option(
                "--report-output",
                help=(
                    "Write the acceptance-live report to this host path instead "
                    "of the task-declared report path"
                ),
            ),
        ] = None,
        no_report_write: Annotated[
            bool,
            typer.Option(
                "--no-report-write",
                help=(
                    "Validate acceptance-live without writing the declared report "
                    "or its .sha256 sidecar (report-only dogfood; leaves the task "
                    "package unmodified). Takes precedence over --report-output."
                ),
            ),
        ] = False,
    ) -> None:
        """Validate a task directory structure."""
        from benchflow._utils.task_authoring import check_task

        issues = check_task(
            task_dir,
            sandbox_type=sandbox,
            validation_level=validation_level,
            acceptance_live_report_output=report_output,
            acceptance_live_write_report=not no_report_write,
        )
        if not issues:
            console.print(
                f"[green]✓[/green] {task_dir.name} — valid ({validation_level})"
            )
        else:
            console.print(f"[red]✗[/red] {task_dir.name} — {len(issues)} issue(s):")
            for issue in issues:
                # Escape Rich markup so literal section names like "[agent]"
                # render verbatim instead of being parsed as styling (#379).
                console.print(f"  [yellow]→[/yellow] {escape(issue)}")
            raise typer.Exit(1)

    @tasks_app.command("migrate")
    def tasks_migrate(
        task_dir: Annotated[Path, typer.Argument(help="Legacy task directory")],
        overwrite: Annotated[
            bool,
            typer.Option("--overwrite", help="Replace an existing task.md"),
        ] = False,
        remove_legacy: Annotated[
            bool,
            typer.Option(
                "--remove-legacy",
                help=(
                    "Delete split files and promote tests/solution aliases after "
                    "task.md is verified"
                ),
            ),
        ] = False,
    ) -> None:
        """Convert task.toml + instruction.md into the unified task.md format."""
        from benchflow._utils.task_authoring import migrate_task_to_task_md

        try:
            result = migrate_task_to_task_md(
                task_dir,
                overwrite=overwrite,
                remove_legacy=remove_legacy,
            )
        except (
            FileExistsError,
            FileNotFoundError,
            NotADirectoryError,
            ValueError,
        ) as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None

        console.print(f"[green]Created:[/green] {result.task_md}")
        if result.removed_legacy:
            console.print("  removed task.toml and instruction.md")
            for migrated_dir in result.migrated_legacy_dirs:
                console.print(f"  promoted {migrated_dir}")
        else:
            console.print("  kept task.toml and instruction.md")

    @tasks_app.command("normalize")
    def tasks_normalize(
        task_dir: Annotated[Path, typer.Argument(help="Task directory with task.md")],
        output: Annotated[
            Path | None,
            typer.Option(
                "--output",
                "-o",
                help="Write normalized task.md to this path instead of stdout",
            ),
        ] = None,
        write: Annotated[
            bool,
            typer.Option(
                "--write",
                help="Replace task.md in place with the normalized canonical form",
            ),
        ] = False,
    ) -> None:
        """Expand minimal task.md authoring profiles into canonical task.md."""
        from benchflow._utils.task_authoring import normalize_task_md

        try:
            result = normalize_task_md(task_dir, output_path=output, write=write)
        except (FileNotFoundError, NotADirectoryError, ValueError) as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None

        if result.output_path is None:
            typer.echo(result.normalized_text, nl=False)
        else:
            console.print(f"[green]Normalized:[/green] {result.output_path}")

    @tasks_app.command("export")
    def tasks_export(
        task_dir: Annotated[Path, typer.Argument(help="Task directory to export")],
        output_dir: Annotated[
            Path | None,
            typer.Argument(
                help="Destination split-layout directory (omit with --report-only)",
            ),
        ] = None,
        target: Annotated[
            str,
            typer.Option("--target", help="Compatibility target: harbor or pier"),
        ] = "harbor",
        overwrite: Annotated[
            bool,
            typer.Option("--overwrite", help="Replace an existing export directory"),
        ] = False,
        report_only: Annotated[
            bool,
            typer.Option(
                "--report-only",
                help="Print the compatibility loss report without writing files",
            ),
        ] = False,
    ) -> None:
        """Export a task to a Harbor/Pier split layout with a loss report."""
        from benchflow.task import (
            build_compatibility_export_report,
            export_task_to_split_layout,
        )

        if target not in {"harbor", "pier"}:
            console.print("[red]target must be 'harbor' or 'pier'[/red]")
            raise typer.Exit(1)

        try:
            if report_only:
                report = build_compatibility_export_report(
                    task_dir,
                    target=cast(Literal["harbor", "pier"], target),
                )
                typer.echo(report.to_json(), nl=False)
                return
            if output_dir is None:
                console.print(
                    "[red]Missing output_dir; pass one or use --report-only[/red]"
                )
                raise typer.Exit(1)
            report = export_task_to_split_layout(
                task_dir,
                output_dir,
                target=cast(Literal["harbor", "pier"], target),
                overwrite=overwrite,
            )
        except (
            FileExistsError,
            FileNotFoundError,
            NotADirectoryError,
            ValueError,
        ) as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None

        console.print(f"[green]Exported:[/green] {output_dir}")
        console.print(f"  target: {report.target}")
        console.print(f"  status: {report.status}")
        console.print(f"  losses: {len(report.losses)}")
        console.print("  report: compatibility/export-report.json")

    @tasks_app.command("digest")
    def tasks_digest(
        path: Annotated[
            Path,
            typer.Argument(
                help="Path to a task directory, or a directory of task directories"
            ),
        ],
    ) -> None:
        """Compute the content digest that pins a task's files, independent of git.

        Matches the digests in the skillsbench dataset registry (registry.json).
        Given a single task directory (a legacy ``task.toml`` or a native
        ``task.md`` task), prints the digest; given a directory of tasks, prints
        one "<name> <digest>" line per task.
        """
        from benchflow._utils.task_authoring import task_digest

        # A task directory is either a legacy task.toml task or a native task.md
        # task (the universal-adapter format) — recognize both, not just legacy.
        def _is_task_dir(p: Path) -> bool:
            return (p / "task.toml").is_file() or (p / "task.md").is_file()

        if not path.is_dir():
            console.print(f"[red]Not a directory: {path}[/red]")
            raise typer.Exit(1)

        if _is_task_dir(path):
            # typer.echo, not console.print: Rich wraps lines at terminal width,
            # which would corrupt piped machine-readable output.
            typer.echo(task_digest(path))
            return

        task_dirs = sorted(d for d in path.iterdir() if d.is_dir() and _is_task_dir(d))
        if not task_dirs:
            console.print(
                f"[red]No tasks under {path} — expected task.toml or task.md "
                f"in it or in its immediate subdirectories[/red]"
            )
            raise typer.Exit(1)
        for task_dir in task_dirs:
            typer.echo(f"{task_dir.name} {task_digest(task_dir)}")
