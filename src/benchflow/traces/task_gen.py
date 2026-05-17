"""Generate BenchFlow tasks from parsed agent traces.

Converts a :class:`~benchflow.traces.models.ParsedTrace` into a task
directory containing ``task.toml`` and ``instruction.md``, optionally
with a ``test.sh`` verifier derived from the trace outcome.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shlex
import textwrap
from pathlib import Path

from benchflow.traces.models import ParsedTrace

logger = logging.getLogger(__name__)

# Difficulty heuristics — weighted score from multiple trace signals
_DIFFICULTY_WEIGHTS = {
    "tool_calls": 0.4,
    "files_edited": 0.3,
    "tokens": 0.2,
    "duration": 0.1,
}

_DIFFICULTY_LEVELS = [
    ("easy", 0, 15),
    ("medium", 16, 45),
    ("hard", 46, 75),
    ("expert", 76, None),
]

# Regex matching path segments that contain dates/timestamps and should
# be replaced with globs so verifiers tolerate agent-generated variants.
_TIMESTAMP_SEGMENT_RE = re.compile(
    r"^\d{4}[-_]\d{2}[-_]\d{2}"
    r"(?:[-_T]\d{2,6})?"
)

# Timeout scaling by difficulty (seconds)
_TIMEOUT_BY_DIFFICULTY = {
    "easy": 300,
    "medium": 600,
    "hard": 1200,
    "expert": 1800,
}


def _estimate_difficulty(trace: ParsedTrace) -> str:
    """Estimate task difficulty from multiple trace complexity signals.

    Combines tool call count, files edited, token usage, and duration
    into a weighted score mapped to easy/medium/hard/expert.
    """
    # Normalize each signal to 0-100 scale
    tool_score = min(trace.n_tool_calls * 2, 100)
    file_score = min(len(trace.files_edited) * 10, 100)
    total_tokens = trace.total_input_tokens + trace.total_output_tokens
    token_score = min(total_tokens / 100, 100) if total_tokens else 0
    duration = trace.duration_sec
    duration_score = min(duration / 6, 100) if duration else 0

    score = (
        tool_score * _DIFFICULTY_WEIGHTS["tool_calls"]
        + file_score * _DIFFICULTY_WEIGHTS["files_edited"]
        + token_score * _DIFFICULTY_WEIGHTS["tokens"]
        + duration_score * _DIFFICULTY_WEIGHTS["duration"]
    )

    for level, lo, hi in _DIFFICULTY_LEVELS:
        if hi is None:
            if score >= lo:
                return level
        elif lo <= score <= hi:
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


_SESSION_CONTINUATION_RE = re.compile(
    r"^This session is being continued from a previous conversation.*?"
    r"(?:Please continue|Continue with).*?$",
    re.DOTALL | re.IGNORECASE,
)


def _clean_user_prompt(prompt: str) -> str:
    """Clean session artifacts from a user prompt.

    Strips session-continuation boilerplate that leaks from multi-turn
    trace formats and extracts the actionable instruction.
    """
    # Remove session continuation wrapper — extract the actual task
    m = _SESSION_CONTINUATION_RE.search(prompt)
    if m:
        # Try to find the real task content between summary markers
        # e.g. "## Final Solution" or "### Implementation Required"
        sections = re.split(r"\n##+ ", prompt)
        if len(sections) > 1:
            # Use the section that looks most like an instruction
            for section in sections[1:]:
                if any(
                    kw in section.lower()
                    for kw in ("solution", "implementation", "required", "code")
                ):
                    prompt = section.strip()
                    break

    return prompt.strip()


def _relativize_path(path: str) -> str:
    """Convert an absolute workspace path to a relative project path.

    Traces from real environments contain absolute paths like
    ``/workspace/archit/trace_generation/repos/hyperswitch_pool_3/crates/...``.
    This strips everything up to and including the repo root directory,
    returning just the project-relative portion.
    """
    # Common workspace prefixes from known trace sources
    patterns = [
        r"/workspace/[^/]+/trace_generation/repos/[^/]+/",
        r"/workspace/[^/]+/repos/[^/]+/",
        r"/workspace/[^/]+/[^/]+/",
        r"/home/[^/]+/repos/[^/]+/",
        r"/home/[^/]+/[^/]+/",
        r"/tmp/[^/]+/",
    ]
    for pattern in patterns:
        m = re.match(pattern, path)
        if m:
            return path[m.end():]
    # If it's an absolute path but no pattern matched, strip leading /
    if path.startswith("/"):
        parts = path.split("/")
        # Heuristic: skip to the first directory that looks like project code
        for i, part in enumerate(parts):
            if part in ("src", "crates", "lib", "app", "tests", "migrations", "pkg"):
                return "/".join(parts[i:])
        # Fallback: just use the basename
        return os.path.basename(path)
    return path


def _globify_path(path: str) -> str:
    """Replace timestamp-bearing directory segments with globs.

    For example::

        migrations/2025-11-28-131040_create_invoices/up.sql
        → migrations/*_create_invoices/up.sql

    This lets verifiers match agent-generated files that use different
    timestamps than the original trace.
    """
    parts = path.split("/")
    out: list[str] = []
    for part in parts:
        if _TIMESTAMP_SEGMENT_RE.match(part):
            # Keep the meaningful suffix after the timestamp prefix
            rest = _TIMESTAMP_SEGMENT_RE.sub("", part).lstrip("-_")
            out.append(f"*_{rest}" if rest else "*")
        else:
            out.append(part)
    return "/".join(out)


def _has_dynamic_segments(path: str) -> bool:
    """Return True if the path contains timestamp-like directory segments."""
    return any(_TIMESTAMP_SEGMENT_RE.match(p) for p in path.split("/"))


def _build_instruction(trace: ParsedTrace) -> str:
    """Synthesize instruction.md content from the trace.

    Uses the first user prompt as the core instruction and annotates
    with context about what the original agent did (files edited,
    tools used) as implicit acceptance criteria.
    """
    prompt = trace.first_user_prompt
    if not prompt:
        prompt = "(No user prompt found in trace — manual instruction needed)"
    else:
        prompt = _clean_user_prompt(prompt)

    lines = [prompt, ""]

    # Add context about what the agent actually did as implicit requirements
    files = [_relativize_path(f) for f in trace.files_edited]
    if files:
        lines.append("## Expected Changes")
        lines.append("")
        lines.append("The following files should be created or modified:")
        lines.append("")
        for f in files[:20]:
            display = _globify_path(f) if _has_dynamic_segments(f) else f
            lines.append(f"- `{display}`")
        if len(files) > 20:
            lines.append(f"- ... and {len(files) - 20} more files")
        lines.append("")

    return "\n".join(lines)


def _sanitize_toml_string(value: str) -> str:
    """Escape a string for safe embedding in a TOML quoted value."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _build_task_toml(
    trace: ParsedTrace,
    *,
    task_name: str = "",
    author: str = "benchflow-traces",
    timeout_sec: int | None = None,
    verifier_timeout_sec: int = 60,
) -> str:
    """Generate task.toml content from a trace.

    If *timeout_sec* is ``None``, scales automatically by estimated difficulty.
    """
    difficulty = _estimate_difficulty(trace)
    if timeout_sec is None:
        timeout_sec = _TIMEOUT_BY_DIFFICULTY.get(difficulty, 300)

    tags = list(trace.tags) if trace.tags else []
    tags.append("from-trace")
    if trace.agent_name:
        tags.append(f"agent:{trace.agent_name}")

    tags_str = ", ".join(f'"{ _sanitize_toml_string(t)}"' for t in tags)

    category = "trace-import"
    if trace.git.repo:
        repo_name = trace.git.repo.rstrip("/").split("/")[-1]
        if repo_name:
            category = repo_name

    safe_author = _sanitize_toml_string(author)
    safe_trace_id = _sanitize_toml_string(trace.trace_id)
    safe_session_id = _sanitize_toml_string(trace.session_id)
    safe_category = _sanitize_toml_string(category)
    safe_name = _sanitize_toml_string(task_name)

    toml_lines = [
        'version = "1.0"',
        "",
        "[task]",
        f'name = "{safe_name}"',
        "",
        "[metadata]",
        f'author_name = "{safe_author}"',
        f'difficulty = "{difficulty}"',
        f'category = "{safe_category}"',
        f"tags = [{tags_str}]",
        f'source_trace_id = "{safe_trace_id}"',
        f'source_session_id = "{safe_session_id}"',
    ]

    if trace.model and trace.model != "None":
        toml_lines.append(f'source_model = "{_sanitize_toml_string(trace.model)}"')
    if trace.outcome:
        toml_lines.append(f'source_outcome = "{_sanitize_toml_string(trace.outcome)}"')

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
            "build_timeout_sec = 600",
            "cpus = 1",
            "memory_mb = 2048",
            "storage_mb = 10240",
        ]
    )

    return "\n".join(toml_lines) + "\n"


def _build_test_sh(trace: ParsedTrace) -> str:
    """Generate a test.sh verifier from the trace.

    When git context is available (repo was cloned), checks that files
    were **modified** relative to the base commit — not just that they
    exist.  Otherwise falls back to file-existence checks.
    Writes reward to /logs/verifier/reward.txt per BenchFlow contract.
    """
    files = [_relativize_path(f) for f in trace.files_edited]
    if not files:
        return (
            "#!/bin/bash\n"
            f"# Auto-generated verifier from trace {trace.trace_id}\n"
            "# No file checks available — manual verification needed.\n"
            'echo "1.0" > /logs/verifier/reward.txt\n'
        )

    has_git = bool(trace.git.repo and trace.git.commit_before)

    checks: list[str] = []
    for f in files[:20]:
        if _has_dynamic_segments(f):
            pattern = _globify_path(f)
            quoted_pattern = shlex.quote(pattern)
            checks.append(f'if ! compgen -G {quoted_pattern} > /dev/null 2>&1; then')
            checks.append(f'  echo "Missing (pattern): {quoted_pattern}"')
            checks.append('  PASS=0')
            checks.append('fi')
        elif has_git:
            quoted = shlex.quote(f)
            checks.append(f'if ! git diff --name-only HEAD {quoted} 2>/dev/null | grep -q .; then')
            checks.append(f'  echo "Not modified: {quoted}"')
            checks.append('  PASS=0')
            checks.append('fi')
        else:
            quoted = shlex.quote(f)
            checks.append(f'if [ ! -f {quoted} ]; then')
            checks.append(f'  echo "Missing: {quoted}"')
            checks.append('  PASS=0')
            checks.append('fi')

    checks_block = "\n".join(checks)

    header = (
        "#!/bin/bash\n"
        f"# Auto-generated verifier from trace {trace.trace_id}\n"
    )
    if has_git:
        header += "# Checks that expected files were modified vs base commit.\n"
    else:
        header += "# Checks that expected files were created/modified.\n"

    return (
        f"{header}"
        "set -euo pipefail\n"
        "\n"
        "PASS=1\n"
        "\n"
        f"{checks_block}\n"
        "\n"
        'if [ "$PASS" = "1" ]; then\n'
        '  echo "1.0" > /logs/verifier/reward.txt\n'
        "else\n"
        '  echo "0.0" > /logs/verifier/reward.txt\n'
        "fi\n"
    )


def _build_dockerfile(trace: ParsedTrace | None = None) -> str:
    """Generate a Dockerfile for trace-generated tasks.

    When the trace has git context (repo + commit), clones the repo at
    the specified commit so the agent has the actual codebase to work with.
    """
    base = textwrap.dedent("""\
        FROM ubuntu:24.04

        RUN apt-get update -qq && apt-get install -y -qq curl git python3 && rm -rf /var/lib/apt/lists/*

        RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
    """)

    if trace and trace.git.repo and trace.git.commit_before:
        repo = trace.git.repo
        commit = trace.git.commit_before
        # Use HTTPS clone URL for public repos
        clone_url = f"https://github.com/{repo}.git"
        base += textwrap.dedent(f"""\
            RUN git clone --depth 50 {clone_url} /app && \\
                cd /app && git checkout {commit} || true

            WORKDIR /app
        """)
    else:
        base += "\nWORKDIR /app\n"

    return base


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
        timeout_sec: Agent timeout in seconds (0 = auto-scale by difficulty).
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
    effective_timeout = timeout_sec if timeout_sec > 0 else None
    task_name = f"trace-import/{task_id}"
    toml_content = _build_task_toml(
        trace, task_name=task_name, author=author, timeout_sec=effective_timeout,
    )
    (task_dir / "task.toml").write_text(toml_content)

    # Write instruction.md
    instruction = _build_instruction(trace)
    (task_dir / "instruction.md").write_text(instruction)

    # Write environment/Dockerfile
    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(_build_dockerfile(trace))

    # Write tests/test.sh
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_sh = _build_test_sh(trace)
    test_path = tests_dir / "test.sh"
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
    timeout_sec: int = 0,
    overwrite: bool = False,
    min_steps: int = 2,
    outcome_filter: str | None = None,
) -> list[Path]:
    """Batch-generate tasks from multiple traces with filtering.

    Args:
        traces: List of parsed traces.
        output_dir: Parent directory for generated tasks.
        author: Author name for task.toml metadata.
        timeout_sec: Agent timeout in seconds (0 = auto-scale by difficulty).
        overwrite: If ``True``, overwrite existing task directories.
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

        # Skip traces with no tool calls (e.g. pure explanation sessions)
        if trace.n_tool_calls == 0:
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
