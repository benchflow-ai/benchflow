"""`bench import` — import agent traces as BenchFlow tasks.

Supports three sources:

1. **Local Claude Code sessions** — ``bench import local``
2. **opentraces JSONL files** — ``bench import file <path>``
3. **HuggingFace datasets** — ``bench import hf <dataset>``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

app = typer.Typer(
    name="import",
    help="Import agent traces as BenchFlow benchmark tasks.",
    no_args_is_help=True,
)


@app.command("local")
def import_local(
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for generated tasks"),
    ] = Path("tasks"),
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
            "-p",
            help="Filter sessions by project path substring",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Max sessions to import"),
    ] = 20,
    min_steps: Annotated[
        int,
        typer.Option("--min-steps", help="Minimum steps per trace"),
    ] = 2,
    outcome: Annotated[
        str | None,
        typer.Option("--outcome", help="Filter by outcome: success, failure, unknown"),
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
    """Import local Claude Code sessions as BenchFlow tasks.

    Scans ~/.claude/projects/ for JSONL session files and converts them
    into task directories with task.toml + instruction.md.

    Examples:
        bench import local
        bench import local --project my-repo --limit 5
        bench import local --outcome success -o benchmarks/from-traces
    """
    from benchflow.traces.local import load_local_sessions
    from benchflow.traces.task_gen import generate_tasks_from_traces

    traces = load_local_sessions(
        projects_dir, project_filter=project_filter, limit=limit
    )

    if not traces:
        console.print("[yellow]No Claude Code sessions found.[/yellow]")
        console.print(
            "Make sure Claude Code is installed and you have sessions in "
            "~/.claude/projects/"
        )
        raise typer.Exit(1)

    if dry_run:
        _print_traces_table(traces)
        return

    results = generate_tasks_from_traces(
        traces,
        output_dir,
        author=author,
        min_steps=min_steps,
        outcome_filter=outcome,
    )

    console.print(
        f"\n[green]Generated {len(results)} tasks[/green] "
        f"from {len(traces)} sessions → {output_dir}"
    )
    _print_task_summary(results)


@app.command("file")
def import_file(
    path: Annotated[
        Path,
        typer.Argument(help="Path to a JSONL trace file (Claude Code or opentraces)"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for generated tasks"),
    ] = Path("tasks"),
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Trace format: auto, claude-code, opentraces",
        ),
    ] = "auto",
    min_steps: Annotated[
        int,
        typer.Option("--min-steps", help="Minimum steps per trace"),
    ] = 2,
    outcome: Annotated[
        str | None,
        typer.Option("--outcome", help="Filter by outcome: success, failure, unknown"),
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
    """Import traces from a JSONL file.

    Supports Claude Code session files and opentraces format. Use --format
    to override auto-detection.

    Examples:
        bench import file ~/.claude/projects/-my-repo/abc123.jsonl
        bench import file traces.jsonl --format opentraces
        bench import file session.jsonl --dry-run
    """
    from benchflow.traces.parsers import (
        parse_claude_code_session,
        parse_opentraces_file,
    )
    from benchflow.traces.task_gen import generate_tasks_from_traces

    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    detected_format = format
    if format == "auto":
        detected_format = _detect_format(path)
        console.print(f"[dim]Detected format: {detected_format}[/dim]")

    if detected_format == "claude-code":
        traces = [parse_claude_code_session(path)]
    elif detected_format == "opentraces":
        traces = parse_opentraces_file(path)
    else:
        console.print(f"[red]Unknown format: {detected_format}[/red]")
        raise typer.Exit(1)

    if not traces:
        console.print("[yellow]No traces found in file.[/yellow]")
        raise typer.Exit(1)

    console.print(f"Parsed {len(traces)} trace(s) from {path.name}")

    if dry_run:
        _print_traces_table(traces)
        return

    results = generate_tasks_from_traces(
        traces,
        output_dir,
        author=author,
        min_steps=min_steps,
        outcome_filter=outcome,
    )

    console.print(
        f"\n[green]Generated {len(results)} tasks[/green] → {output_dir}"
    )
    _print_task_summary(results)


@app.command("hf")
def import_hf(
    dataset: Annotated[
        str,
        typer.Argument(
            help=(
                "HuggingFace dataset ID (e.g. nlile/misc-merged-claude-code-traces-v1) "
                "or alias (opentraces-test, cc-traces-merged, claudeset-community)"
            )
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for generated tasks"),
    ] = Path("tasks"),
    format: Annotated[
        str | None,
        typer.Option(
            "--format",
            "-f",
            help="Dataset format override: opentraces, claude-messages",
        ),
    ] = None,
    split: Annotated[
        str,
        typer.Option("--split", help="Dataset split to load"),
    ] = "train",
    max_rows: Annotated[
        int,
        typer.Option("--max-rows", "-n", help="Max rows to download"),
    ] = 100,
    min_steps: Annotated[
        int,
        typer.Option("--min-steps", help="Minimum steps per trace"),
    ] = 2,
    outcome: Annotated[
        str | None,
        typer.Option("--outcome", help="Filter by outcome: success, failure, unknown"),
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
    """Import traces from a HuggingFace dataset.

    Downloads the dataset, parses traces, and generates BenchFlow tasks.

    Aliases:
        opentraces-test    → Jayfarei/opentraces-test (58 traces)
        cc-traces-merged   → nlile/misc-merged-claude-code-traces-v1 (32k traces)
        claudeset-community → lelouch0110/claudeset-community
        cc-traces-weka     → semianalysisai/cc-traces-weka-no-subagents-051226

    Examples:
        bench import hf opentraces-test --dry-run
        bench import hf nlile/misc-merged-claude-code-traces-v1 -n 50
        bench import hf cc-traces-merged -n 100 --outcome success
    """
    from benchflow.traces.huggingface import KNOWN_DATASETS, load_hf_dataset

    # Show known dataset info if alias
    if dataset in KNOWN_DATASETS:
        info = KNOWN_DATASETS[dataset]
        console.print(
            f"[dim]Using known dataset: {info['repo']} "
            f"({info['description']})[/dim]"
        )

    traces = load_hf_dataset(
        dataset, format=format, split=split, max_rows=max_rows
    )

    if not traces:
        console.print("[yellow]No traces found in dataset.[/yellow]")
        raise typer.Exit(1)

    console.print(f"Loaded {len(traces)} trace(s) from HuggingFace")

    if dry_run:
        _print_traces_table(traces)
        return

    from benchflow.traces.task_gen import generate_tasks_from_traces

    results = generate_tasks_from_traces(
        traces,
        output_dir,
        author=author,
        min_steps=min_steps,
        outcome_filter=outcome,
    )

    console.print(
        f"\n[green]Generated {len(results)} tasks[/green] → {output_dir}"
    )
    _print_task_summary(results)


@app.command("list-datasets")
def list_datasets() -> None:
    """List known HuggingFace trace datasets."""
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
# Helpers
# ---------------------------------------------------------------------------


def _detect_format(path: Path) -> str:
    """Auto-detect trace file format from first line."""
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

    # opentraces has schema_version and agent fields
    if "schema_version" in data or ("agent" in data and "steps" in data):
        return "opentraces"

    # Claude Code has type field with user/assistant/system
    if data.get("type") in ("user", "assistant", "system"):
        return "claude-code"

    return "claude-code"


def _print_traces_table(traces: list) -> None:
    """Print a summary table of parsed traces."""
    table = Table(title=f"Parsed Traces ({len(traces)})")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Agent", style="cyan")
    table.add_column("Prompt (first 80 chars)")
    table.add_column("Steps", justify="right")
    table.add_column("Tools", justify="right")
    table.add_column("Outcome", style="yellow")
    table.add_column("Files", justify="right")

    for i, trace in enumerate(traces[:50], 1):  # Cap display at 50
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
    """Print summary of generated tasks."""
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
