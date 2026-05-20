"""`bench tasks generate` — generate BenchFlow tasks from agent traces.

Supports three trace sources:

1. **Local Claude Code sessions** — ``bench tasks generate --from-local``
2. **JSONL trace files** — ``bench tasks generate --from-file <path>``
3. **HuggingFace datasets** — ``bench tasks generate --from-hf <dataset>``

Also adds ``bench tasks list-sources`` for discovering known HF datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def register_tasks_generate(tasks_app: typer.Typer) -> None:
    """Register ``generate`` and ``list-sources`` on the tasks sub-app."""

    @tasks_app.command("generate")
    def tasks_generate(
        from_local: Annotated[
            bool,
            typer.Option(
                "--from-local",
                help="Generate from local Claude Code sessions (~/.claude/projects/)",
            ),
        ] = False,
        from_file: Annotated[
            Path | None,
            typer.Option(
                "--from-file",
                help="Generate from a JSONL trace file (Claude Code or opentraces)",
            ),
        ] = None,
        from_hf: Annotated[
            str | None,
            typer.Option(
                "--from-hf",
                help=(
                    "Generate from a HuggingFace dataset "
                    "(ID or alias; run `bench tasks list-sources` for aliases)"
                ),
            ),
        ] = None,
        output_dir: Annotated[
            Path,
            typer.Option("--output", help="Output directory for generated tasks"),
        ] = Path("tasks"),
        # ── source-specific options ──────────────────────────────────
        projects_dir: Annotated[
            Path | None,
            typer.Option(
                "--projects-dir",
                help="Claude Code projects directory (default: ~/.claude/projects/)",
            ),
        ] = None,
        project_filter: Annotated[
            str | None,
            typer.Option(
                "--project",
                help="Filter local sessions by project path substring",
            ),
        ] = None,
        format: Annotated[
            str,
            typer.Option(
                "--format",
                help="Trace format override: auto, claude-code, opentraces, claude-messages",
            ),
        ] = "auto",
        split: Annotated[
            str,
            typer.Option("--split", help="HuggingFace dataset split"),
        ] = "train",
        max_rows: Annotated[
            int,
            typer.Option("--max-rows", help="Max rows to download from HuggingFace"),
        ] = 100,
        # ── shared filtering / output options ────────────────────────
        limit: Annotated[
            int,
            typer.Option("--limit", help="Max traces to process"),
        ] = 20,
        min_steps: Annotated[
            int,
            typer.Option("--min-steps", help="Minimum steps per trace"),
        ] = 2,
        outcome: Annotated[
            str | None,
            typer.Option(
                "--outcome", help="Filter by outcome: success, failure, unknown"
            ),
        ] = None,
        author: Annotated[
            str,
            typer.Option("--author", help="Author name for task.toml"),
        ] = "benchflow-traces",
        dry_run: Annotated[
            bool,
            typer.Option("--dry-run", help="Preview traces without generating tasks"),
        ] = False,
    ) -> None:
        """Generate benchmark tasks from agent traces.

        Exactly one source flag (``--from-local``, ``--from-file``, or
        ``--from-hf``) is required.

        Examples::

            bench tasks generate --from-local
            bench tasks generate --from-local --project my-repo --limit 5
            bench tasks generate --from-file session.jsonl --dry-run
            bench tasks generate --from-hf opentraces-test --limit 50 --outcome success
        """
        sources = sum([from_local, from_file is not None, from_hf is not None])
        if sources == 0:
            console.print(
                "[red]Specify a source: --from-local, --from-file, or --from-hf[/red]"
            )
            raise typer.Exit(1)
        if sources > 1:
            console.print("[red]Only one source allowed at a time[/red]")
            raise typer.Exit(1)

        if from_local:
            traces = _load_local(projects_dir, project_filter, limit)
        elif from_file is not None:
            traces = _load_file(from_file, format)
        else:
            assert from_hf is not None
            traces = _load_hf(from_hf, format, split, max_rows)

        # Apply shared --limit across all sources
        if limit and len(traces) > limit:
            traces = traces[:limit]

        if not traces:
            console.print("[yellow]No traces found.[/yellow]")
            raise typer.Exit(1)

        console.print(f"Loaded {len(traces)} trace(s)")

        if dry_run:
            from benchflow.traces.task_gen import filter_traces_for_generation

            eligible_traces, skipped = filter_traces_for_generation(
                traces,
                min_steps=min_steps,
                outcome_filter=outcome,
            )
            if skipped:
                console.print(
                    f"[yellow]Skipped {skipped} trace(s) that would not generate a task[/yellow]"
                )
            if not eligible_traces:
                console.print(
                    "[yellow]No traces match generation filters; generation would create 0 tasks.[/yellow]"
                )
                return
            _print_traces_table(eligible_traces)
            return

        from benchflow.traces.task_gen import generate_tasks_from_traces

        results = generate_tasks_from_traces(
            traces,
            output_dir,
            author=author,
            min_steps=min_steps,
            outcome_filter=outcome,
        )

        console.print(f"\n[green]Generated {len(results)} tasks[/green] → {output_dir}")
        _print_task_summary(results)

    @tasks_app.command("list-sources")
    def tasks_list_sources() -> None:
        """List known HuggingFace trace datasets.

        These aliases can be passed to ``bench tasks generate --from-hf``.
        """
        from benchflow.traces.huggingface import KNOWN_DATASETS

        table = Table(title="Known Trace Datasets")
        table.add_column("Alias", style="cyan")
        table.add_column("HuggingFace Repo", style="green")
        table.add_column("Format", style="yellow")
        table.add_column("Description")

        for alias, info in KNOWN_DATASETS.items():
            table.add_row(alias, info["repo"], info["format"], info["description"])

        console.print(table)


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------


def _load_local(
    projects_dir: Path | None, project_filter: str | None, limit: int
) -> list:
    """Load traces from local Claude Code sessions."""
    from benchflow.traces.local import load_local_sessions

    traces = load_local_sessions(
        projects_dir, project_filter=project_filter, limit=limit
    )
    if not traces:
        console.print(
            "[dim]Tip: make sure Claude Code is installed and you have "
            "sessions in ~/.claude/projects/[/dim]"
        )
    return traces


def _load_file(path: Path, format: str) -> list:
    """Load traces from a single JSONL file.

    For Claude Code format, splits by ``sessionId`` so multi-session
    files produce one trace per session.  Also supports HuggingFace
    ``claude-messages`` rows (with ``messages_json`` field).
    """
    from benchflow.traces.parsers import (
        parse_claude_code_file,
        parse_opentraces_file,
    )

    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    detected_format = format
    if format == "auto":
        detected_format = _detect_format(path)
        console.print(f"[dim]Detected format: {detected_format}[/dim]")

    if detected_format == "claude-code":
        return parse_claude_code_file(path)
    elif detected_format == "opentraces":
        return parse_opentraces_file(path)
    elif detected_format == "claude-messages":
        return _parse_hf_messages_file(path)
    else:
        console.print(f"[red]Unknown format: {detected_format}[/red]")
        raise typer.Exit(1)


def _load_hf(dataset: str, format: str | None, split: str, max_rows: int) -> list:
    """Load traces from a HuggingFace dataset."""
    from benchflow.traces.huggingface import KNOWN_DATASETS, load_hf_dataset

    fmt = None if format == "auto" else format
    # CLI uses "claude-code"; HF loader uses "claude-messages"
    if fmt == "claude-code":
        fmt = "claude-messages"
    if dataset in KNOWN_DATASETS:
        info = KNOWN_DATASETS[dataset]
        console.print(
            f"[dim]Using known dataset: {info['repo']} ({info['description']})[/dim]"
        )

    return load_hf_dataset(dataset, format=fmt, split=split, max_rows=max_rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_hf_messages_file(path: Path) -> list:
    """Parse a JSONL file of HuggingFace ``claude-messages`` rows."""
    import json as _json

    from benchflow.traces.huggingface import _parse_claude_messages_row

    traces = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                parsed = _parse_claude_messages_row(row, idx=i)
                if parsed:
                    traces.append(parsed)
    return traces


def _detect_format(path: Path) -> str:
    """Auto-detect trace file format from the first non-empty JSONL line."""
    first_line = ""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                first_line = line
                break

    if not first_line:
        return "claude-code"

    try:
        data = json.loads(first_line)
    except json.JSONDecodeError:
        return "claude-code"

    if "schema_version" in data or ("agent" in data and "steps" in data):
        return "opentraces"

    if data.get("type") in ("user", "assistant", "system"):
        return "claude-code"

    # HuggingFace merged-messages format
    if "messages_json" in data or ("messages" in data and "model" in data):
        return "claude-messages"

    return "claude-code"


def _print_traces_table(traces: list) -> None:
    """Print a Rich summary table of parsed traces."""
    table = Table(title=f"Parsed Traces ({len(traces)})")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Agent", style="cyan")
    table.add_column("Prompt (first 80 chars)")
    table.add_column("Steps", justify="right")
    table.add_column("Tools", justify="right")
    table.add_column("Outcome", style="yellow")
    table.add_column("Files", justify="right")

    for i, trace in enumerate(traces[:50], 1):
        prompt = trace.first_user_prompt or "(no prompt)"
        prompt = prompt[:80].replace("\n", " ")
        table.add_row(
            str(i),
            trace.agent_name,
            prompt,
            str(len(trace.steps)),
            str(trace.n_tool_calls),
            trace.outcome or "?",
            str(len(trace.files_edited)),
        )

    console.print(table)

    if len(traces) > 50:
        console.print(f"[dim]... and {len(traces) - 50} more traces[/dim]")


def _print_task_summary(task_dirs: list[Path]) -> None:
    """Print a Rich summary table of generated task directories."""
    if not task_dirs:
        return

    table = Table(title="Generated Tasks")
    table.add_column("Task", style="cyan")
    table.add_column("Files")

    for task_dir in task_dirs[:20]:
        files = [f.name for f in task_dir.iterdir() if f.is_file()]
        table.add_row(task_dir.name, ", ".join(sorted(files)))

    console.print(table)

    if len(task_dirs) > 20:
        console.print(f"[dim]... and {len(task_dirs) - 20} more tasks[/dim]")
