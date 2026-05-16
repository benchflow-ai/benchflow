"""Generate BenchFlow tasks from parsed agent traces.

Converts a :class:`~benchflow.traces.models.ParsedTrace` into a task
directory containing ``task.toml`` and ``instruction.md``, optionally
with a ``test.sh`` verifier derived from the trace outcome.
"""

from __future__ import annotations

import hashlib
import logging
import re
import textwrap
from pathlib import Path

from benchflow.traces.models import ParsedTrace

logger = logging.getLogger(__name__)

# Default difficulty heuristics based on trace properties
_DIFFICULTY_THRESHOLDS = {
    "easy": (0, 5),       # <=5 tool calls
    "medium": (6, 20),    # 6-20 tool calls
    "hard": (21, 50),     # 21-50 tool calls
    "expert": (51, None), # >50 tool calls
}


def _estimate_difficulty(trace: ParsedTrace) -> str:
    """Estimate task difficulty from trace complexity signals."""
    n = trace.n_tool_calls
    for level, (lo, hi) in _DIFFICULTY_THRESHOLDS.items():
        if hi is None:
            if n >= lo:
                return level
        elif lo <= n <= hi:
            return level
    return "medium"


def _slugify(text: str, max_len: int = 60) -> str:
    """Convert text to a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())
    slug = slug.strip("-")
    return slug[:max_len].rstrip("-") or "trace-task"


def _task_id_from_trace(trace: ParsedTrace) -> str:
    """Generate a deterministic, short task ID from the trace."""
    h = hashlib.sha256(trace.trace_id.encode()).hexdigest()[:8]
    prompt = trace.first_user_prompt or trace.trace_id
    slug = _slugify(prompt, max_len=40)
    return f"{slug}-{h}"


def _build_instruction(trace: ParsedTrace) -> str:
    """Synthesize instruction.md content from the trace.

    Uses the first user prompt as the core instruction and annotates
    with context about what the original agent did (files edited,
    tools used) as implicit acceptance criteria.
    """
    prompt = trace.first_user_prompt
    if not prompt:
        prompt = "(No user prompt found in trace — manual instruction needed)"

    lines = [prompt, ""]

    # Add context about what the agent actually did as implicit requirements
    files = trace.files_edited
    if files:
        lines.append("## Expected Changes")
        lines.append("")
        lines.append("The following files should be created or modified:")
        lines.append("")
        for f in files[:20]:  # cap at 20 files
            lines.append(f"- `{f}`")
        if len(files) > 20:
            lines.append(f"- ... and {len(files) - 20} more files")
        lines.append("")

    return "\n".join(lines)


def _build_task_toml(
    trace: ParsedTrace,
    *,
    author: str = "benchflow-traces",
    timeout_sec: int = 300,
    verifier_timeout_sec: int = 60,
) -> str:
    """Generate task.toml content from a trace."""
    difficulty = _estimate_difficulty(trace)
    tags = list(trace.tags) if trace.tags else []
    tags.append("from-trace")
    if trace.agent_name:
        tags.append(f"agent:{trace.agent_name}")

    # Build tags line
    tags_str = ", ".join(f'"{t}"' for t in tags)

    category = "trace-import"
    if trace.git.repo:
        # Use repo name as category
        repo_name = trace.git.repo.rstrip("/").split("/")[-1]
        if repo_name:
            category = repo_name

    toml_lines = [
        'version = "1.0"',
        "",
        "[metadata]",
        f'author_name = "{author}"',
        f'difficulty = "{difficulty}"',
        f'category = "{category}"',
        f"tags = [{tags_str}]",
        f'source_trace_id = "{trace.trace_id}"',
        f'source_session_id = "{trace.session_id}"',
    ]

    if trace.model:
        toml_lines.append(f'source_model = "{trace.model}"')
    if trace.outcome:
        toml_lines.append(f'source_outcome = "{trace.outcome}"')

    toml_lines.extend(
        [
            "",
            "[agent]",
            f"timeout_sec = {timeout_sec}",
            "",
            "[verifier]",
            f"timeout_sec = {verifier_timeout_sec}",
            "",
            "[environment]",
            "cpus = 1",
            "memory_mb = 2048",
        ]
    )

    return "\n".join(toml_lines) + "\n"


def _build_test_sh(trace: ParsedTrace) -> str | None:
    """Generate a basic test.sh verifier from the trace.

    If the trace has files_edited, generates a verifier that checks
    those files exist. Returns None if no verifier can be generated.
    """
    files = trace.files_edited
    if not files:
        return None

    checks: list[str] = []
    for f in files[:10]:
        checks.append(f'  if [ ! -f "{f}" ]; then')
        checks.append(f'    echo "Missing: {f}"')
        checks.append("    PASS=0")
        checks.append("  fi")

    script = textwrap.dedent("""\
        #!/bin/bash
        # Auto-generated verifier from trace {trace_id}
        # Checks that expected files were created/modified.
        set -euo pipefail

        PASS=1

        {checks}

        if [ "$PASS" = "1" ]; then
            echo "1.0" > reward.txt
        else
            echo "0.0" > reward.txt
        fi
    """).format(
        trace_id=trace.trace_id,
        checks="\n".join(checks),
    )

    return script


def generate_task(
    trace: ParsedTrace,
    output_dir: Path,
    *,
    author: str = "benchflow-traces",
    timeout_sec: int = 300,
    overwrite: bool = False,
) -> Path:
    """Generate a complete BenchFlow task directory from a parsed trace.

    Creates:
        ``<output_dir>/<task-slug>/task.toml``
        ``<output_dir>/<task-slug>/instruction.md``
        ``<output_dir>/<task-slug>/test.sh`` (if verifiable)

    Args:
        trace: Parsed trace to convert.
        output_dir: Parent directory for generated tasks.
        author: Author name for task.toml metadata.
        timeout_sec: Agent timeout in seconds.
        overwrite: If True, overwrite existing task directories.

    Returns:
        Path to the created task directory.
    """
    output_dir = Path(output_dir)
    task_id = _task_id_from_trace(trace)
    task_dir = output_dir / task_id

    if task_dir.exists() and not overwrite:
        logger.info("Task %s already exists, skipping (use overwrite=True)", task_id)
        return task_dir

    task_dir.mkdir(parents=True, exist_ok=True)

    # Write task.toml
    toml_content = _build_task_toml(trace, author=author, timeout_sec=timeout_sec)
    (task_dir / "task.toml").write_text(toml_content)

    # Write instruction.md
    instruction = _build_instruction(trace)
    (task_dir / "instruction.md").write_text(instruction)

    # Write test.sh if we can generate a verifier
    test_sh = _build_test_sh(trace)
    if test_sh:
        test_path = task_dir / "test.sh"
        test_path.write_text(test_sh)
        test_path.chmod(0o755)

    logger.info(
        "Generated task %s (difficulty=%s, outcome=%s, tools=%d)",
        task_id,
        _estimate_difficulty(trace),
        trace.outcome,
        trace.n_tool_calls,
    )

    return task_dir


def generate_tasks_from_traces(
    traces: list[ParsedTrace],
    output_dir: Path,
    *,
    author: str = "benchflow-traces",
    timeout_sec: int = 300,
    overwrite: bool = False,
    min_steps: int = 2,
    outcome_filter: str | None = None,
) -> list[Path]:
    """Batch-generate tasks from multiple traces with filtering.

    Args:
        traces: List of parsed traces.
        output_dir: Parent directory for generated tasks.
        author: Author name for task.toml metadata.
        timeout_sec: Agent timeout in seconds.
        overwrite: If True, overwrite existing task directories.
        min_steps: Minimum number of steps to include a trace.
        outcome_filter: If set, only include traces with this outcome.

    Returns:
        List of paths to created task directories.
    """
    results: list[Path] = []
    skipped = 0

    for trace in traces:
        # Filter by minimum complexity
        if len(trace.steps) < min_steps:
            skipped += 1
            continue

        # Filter by outcome
        if outcome_filter and trace.outcome != outcome_filter:
            skipped += 1
            continue

        # Skip traces with no user prompt
        if not trace.first_user_prompt:
            skipped += 1
            continue

        task_dir = generate_task(
            trace,
            output_dir,
            author=author,
            timeout_sec=timeout_sec,
            overwrite=overwrite,
        )
        results.append(task_dir)

    if skipped:
        logger.info("Skipped %d traces (filtered by steps/outcome/prompt)", skipped)

    return results
