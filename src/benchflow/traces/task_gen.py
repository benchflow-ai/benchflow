"""Generate BenchFlow tasks from parsed agent traces.

Converts a :class:`~benchflow.traces.models.ParsedTrace` into a native
``task.md`` package with ``verifier/`` and ``oracle/`` directories. Pass
``output_format="legacy"`` for split ``task.toml`` / ``instruction.md`` tasks.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import textwrap
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml

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

_WHOLE_FILE_WRITE_TOOLS = {"Write", "write_to_file"}
_AMBIGUOUS_EDIT_TOOLS = {"Edit", "MultiEdit", "edit_file"}
_GITHUB_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
TaskOutputFormat = Literal["task-md", "legacy"]


@dataclass(frozen=True)
class _FileExpectation:
    """A generated verifier/oracle target extracted from a trace."""

    path: str
    expected_content: str | None

    @property
    def has_dynamic_path(self) -> bool:
        return _has_dynamic_segments(self.path)

    @property
    def target(self) -> str:
        return _globify_path(self.path) if self.has_dynamic_path else self.path

    def verifier_payload(self, *, has_git: bool) -> dict[str, str]:
        if self.expected_content is not None:
            return {
                "kind": "content",
                "path": self.path,
                "target": self.target,
                "content_b64": base64.b64encode(self.expected_content.encode()).decode(
                    "ascii"
                ),
            }
        if has_git:
            return {
                "kind": "git_status",
                "path": self.path,
                "target": self.target,
            }
        return {
            "kind": "manual",
            "path": self.path,
            "target": self.target,
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
            return path[m.end() :]
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


def _is_safe_workspace_path(path: str) -> bool:
    """Return True if ``path`` is safe to embed in a generated task artifact.

    Generated ``test.sh`` and ``solve.sh`` ``cd /app`` before operating on
    these paths. A path containing ``..`` segments or starting with ``/``
    escapes the workspace, so the verifier and oracle can read or write
    arbitrary host locations (see GitHub issue #376).

    Empty strings are also unsafe — they'd produce shell commands that
    silently target the current directory.
    """
    if not path:
        return False
    if path.startswith("/"):
        return False
    if "\x00" in path:
        return False
    # Use PurePosixPath-style splitting; trace paths use forward slashes
    # even when the agent ran on Windows-like surfaces.
    return ".." not in Path(path).parts


def _safe_relativize(path: str) -> str | None:
    """Return ``_relativize_path(path)`` if the result is workspace-safe.

    Returns ``None`` (with a warning log) for unsafe inputs — the caller
    should drop the offending entry from generated artifacts so the task
    does not encode writes or verification outside the workspace.
    """
    relative = _relativize_path(path)
    if not _is_safe_workspace_path(relative):
        logger.warning(
            "Dropping trace path %r (relativized to %r) — escapes task workspace",
            path,
            relative,
        )
        return None
    return relative


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

    # Add context about what the agent actually did as implicit requirements.
    # Drop trace paths that would escape the task workspace (issue #376) —
    # ``..`` segments or absolute paths in instruction.md mislead the agent
    # and pair badly with the verifier and oracle that consume the same list.
    files = [
        rel for f in trace.files_edited if (rel := _safe_relativize(f)) is not None
    ]
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

    tags_str = ", ".join(f'"{_sanitize_toml_string(t)}"' for t in tags)

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


def _build_task_md(toml_content: str, instruction: str) -> str:
    raw_config = tomllib.loads(toml_content)
    if "solution" in raw_config and "oracle" not in raw_config:
        raw_config["oracle"] = raw_config.pop("solution")
    frontmatter = yaml.safe_dump(raw_config, sort_keys=False)
    return f"---\n{frontmatter}---\n\n## prompt\n\n{instruction.strip()}\n"


def _validate_output_format(output_format: str) -> TaskOutputFormat:
    if output_format == "task-md":
        return "task-md"
    if output_format == "legacy":
        return "legacy"
    raise ValueError("output_format must be 'task-md' or 'legacy'")


def _build_test_sh(trace: ParsedTrace) -> str:
    """Generate a test.sh verifier from the trace.

    Whole-file writes become exact content checks. Ambiguous edit-only traces
    use git status when a base commit is available; otherwise the verifier
    reports 0.0 and requires manual authoring instead of auto-passing a weak
    existence check.
    Writes reward to /logs/verifier/reward.txt per BenchFlow contract.
    """
    expectations = _file_expectations_from_trace(trace)
    has_git = bool(trace.git.repo and trace.git.commit_before)
    payload = [
        expectation.verifier_payload(has_git=has_git)
        for expectation in expectations[:20]
    ]
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    header = (
        "#!/bin/bash\n"
        f"# Auto-generated verifier from trace {_shell_comment_text(trace.trace_id)}\n"
    )
    if not payload:
        header += "# No file checks available; manual verification required.\n"
    elif has_git:
        header += (
            "# Checks exact content where known, otherwise git working-tree status.\n"
        )
    else:
        header += (
            "# Checks exact content where known; unverifiable edits fail closed.\n"
        )
    for expectation in expectations[:20]:
        header += f"# Expected: {_shell_comment_text(expectation.target)}\n"

    return (
        f"{header}"
        "set -euo pipefail\n"
        "\n"
        "python3 - <<'PY'\n"
        "import base64\n"
        "import glob\n"
        "import json\n"
        "import pathlib\n"
        "import subprocess\n"
        "import sys\n"
        "\n"
        f"CHECKS = json.loads({payload_json!r})\n"
        "REWARD = pathlib.Path('/logs/verifier/reward.txt')\n"
        "REWARD.parent.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "def targets(check):\n"
        "    target = check['target']\n"
        "    if target != check['path']:\n"
        "        return sorted(p for p in glob.glob(target) if pathlib.Path(p).is_file())\n"
        "    return [target] if pathlib.Path(target).is_file() else []\n"
        "\n"
        "def git_changed(path):\n"
        "    try:\n"
        "        result = subprocess.run(\n"
        "            ['git', 'status', '--porcelain', '--', path],\n"
        "            check=False,\n"
        "            text=True,\n"
        "            stdout=subprocess.PIPE,\n"
        "            stderr=subprocess.DEVNULL,\n"
        "        )\n"
        "    except OSError:\n"
        "        return False\n"
        "    return result.returncode == 0 and bool(result.stdout.strip())\n"
        "\n"
        "passed = True\n"
        "if not CHECKS:\n"
        "    print('No file checks available for this trace-generated task.')\n"
        "    passed = False\n"
        "\n"
        "for check in CHECKS:\n"
        "    kind = check['kind']\n"
        "    matched = targets(check)\n"
        "    if kind == 'content':\n"
        "        expected = base64.b64decode(check['content_b64'])\n"
        "        if not matched:\n"
        "            print(f\"Missing: {check['target']}\")\n"
        "            passed = False\n"
        "            continue\n"
        "        if not any(pathlib.Path(path).read_bytes() == expected for path in matched):\n"
        "            print(f\"Content mismatch: {check['target']}\")\n"
        "            passed = False\n"
        "    elif kind == 'git_status':\n"
        "        candidates = matched or [check['path']]\n"
        "        if not any(git_changed(path) for path in candidates):\n"
        "            print(f\"Not modified: {check['target']}\")\n"
        "            passed = False\n"
        "    else:\n"
        "        print(f\"Manual verification required: {check['target']}\")\n"
        "        passed = False\n"
        "\n"
        "REWARD.write_text('1.0\\n' if passed else '0.0\\n')\n"
        "sys.exit(0)\n"
        "PY\n"
    )


def _tool_call_path(tc_input: dict[str, object]) -> str | None:
    path = tc_input.get("file_path") or tc_input.get("path")
    if not isinstance(path, str) or not path:
        return None
    return _safe_relativize(path)


def _whole_file_content(tc_input: dict[str, object]) -> str | None:
    """Return final file content only for whole-file write tool inputs."""
    for key in ("content", "text"):
        value = tc_input.get(key)
        if isinstance(value, str):
            return value
    return None


def _file_expectations_from_trace(trace: ParsedTrace) -> list[_FileExpectation]:
    """Return final per-path expectations without inventing edit content.

    ``Write``/``write_to_file`` carry whole-file content and can be replayed
    exactly. ``Edit``/``MultiEdit`` carry replacement fragments, so they are
    intentionally classified as content-unknown unless a later whole-file
    write supersedes them.
    """
    by_path: dict[str, _FileExpectation] = {}

    for step in trace.steps:
        for tc in step.tool_calls:
            if tc.name in _WHOLE_FILE_WRITE_TOOLS:
                path = _tool_call_path(tc.input)
                if path is None:
                    continue
                by_path[path] = _FileExpectation(
                    path=path,
                    expected_content=_whole_file_content(tc.input),
                )
            elif tc.name in _AMBIGUOUS_EDIT_TOOLS:
                path = _tool_call_path(tc.input)
                if path is None:
                    continue
                by_path[path] = _FileExpectation(path=path, expected_content=None)

    return list(by_path.values())


def _solution_writes_from_trace(trace: ParsedTrace) -> list[tuple[str, str]]:
    """Return deterministic file writes that can replay a trace as an oracle."""
    return [
        (expectation.path, expectation.expected_content)
        for expectation in _file_expectations_from_trace(trace)
        if expectation.expected_content is not None
    ]


def _build_solution_sh(trace: ParsedTrace) -> str:
    """Generate an oracle solution that replays simple file writes from a trace."""
    writes = _solution_writes_from_trace(trace)
    header = (
        "#!/bin/bash\n"
        f"# Auto-generated oracle solution from trace {_shell_comment_text(trace.trace_id)}\n"
        "set -euo pipefail\n"
        "cd /app\n\n"
    )

    if not writes:
        return header + (
            'echo "No replayable file writes were found for this trace." >&2\nexit 1\n'
        )

    payload = [
        {
            "path": path,
            "content_b64": base64.b64encode(content.encode()).decode("ascii"),
        }
        for path, content in writes
    ]
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    return (
        header
        + "python3 - <<'PY'\n"
        + "import base64\n"
        + "import json\n"
        + "import pathlib\n"
        + "\n"
        + f"WRITES = json.loads({payload_json!r})\n"
        + "for write in WRITES:\n"
        + "    path = pathlib.Path(write['path'])\n"
        + "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        + "    path.write_bytes(base64.b64decode(write['content_b64']))\n"
        + "PY\n"
    )


def _build_dockerfile(trace: ParsedTrace | None = None) -> str:
    """Generate a Dockerfile for trace-generated tasks.

    When the trace has git context (repo + commit), clones the repo at
    the specified commit so the agent has the actual codebase to work with.
    The checkout must succeed; building against clone HEAD would make the
    generated benchmark stale and misleading.
    """
    base = textwrap.dedent("""\
        FROM ubuntu:24.04

        RUN apt-get update -qq && apt-get install -y -qq curl git python3 && rm -rf /var/lib/apt/lists/*

        RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
    """)

    if trace and trace.git.repo and trace.git.commit_before:
        clone_url = shlex.quote(_github_clone_url(trace.git.repo))
        commit = _validate_git_commit(trace.git.commit_before)
        quoted_commit = shlex.quote(commit)
        base += textwrap.dedent(f"""\
            RUN git clone --no-checkout {clone_url} /app && \\
                cd /app && \\
                git fetch --depth 1 origin {quoted_commit} && \\
                git checkout --detach {quoted_commit} && \\
                case "$(git rev-parse --verify HEAD)" in {commit}*) ;; *) exit 1 ;; esac

            WORKDIR /app
        """)
    else:
        base += "\nWORKDIR /app\n"

    return base


def _github_clone_url(repo: str) -> str:
    """Return a public HTTPS clone URL for GitHub shorthand or full URLs."""
    normalized = repo.strip()
    if normalized.endswith(".git"):
        normalized = normalized[:-4]

    if normalized.startswith("git@github.com:"):
        return _github_https_url(normalized.removeprefix("git@github.com:"))

    if normalized.startswith("ssh://git@github.com/"):
        return _github_https_url(normalized.removeprefix("ssh://git@github.com/"))

    if normalized.startswith(("https://", "http://")):
        parsed = urlparse(normalized)
        if parsed.netloc.lower() != "github.com":
            raise ValueError(
                "Trace git.repo must be a public GitHub HTTPS URL or owner/repo"
            )
        return _github_https_url(parsed.path.lstrip("/"))

    if normalized.startswith("git@"):
        raise ValueError("Trace git.repo SSH remotes are only supported for GitHub")

    if normalized.startswith("github.com/"):
        return _github_https_url(normalized.removeprefix("github.com/"))

    return _github_https_url(normalized)


def _github_https_url(owner_repo: str) -> str:
    normalized = owner_repo.strip().strip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    if not _GITHUB_REPO_RE.fullmatch(normalized):
        raise ValueError(
            "Trace git.repo must identify a GitHub repository as owner/repo"
        )
    return f"https://github.com/{normalized}.git"


def _validate_git_commit(commit: str) -> str:
    normalized = commit.strip()
    if not _GIT_COMMIT_RE.fullmatch(normalized):
        raise ValueError(
            "Trace git.commit_before must be a 7-40 character hexadecimal commit"
        )
    return normalized.lower()


def _shell_comment_text(value: str) -> str:
    """Keep trace-controlled metadata inside a single shell comment line."""
    clean = re.sub(r"[\r\n\t]+", " ", value)
    return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in clean)


def generate_task(
    trace: ParsedTrace,
    output_dir: Path,
    *,
    author: str = "benchflow-traces",
    timeout_sec: int = 300,
    overwrite: bool = False,
    output_format: TaskOutputFormat = "task-md",
) -> Path:
    """Generate a complete BenchFlow task directory from a parsed trace.

    Creates:
        ``<output_dir>/<task-slug>/task.md``
        ``<output_dir>/<task-slug>/verifier/test.sh``
        ``<output_dir>/<task-slug>/oracle/solve.sh``

        Pass ``output_format="legacy"`` for the split ``task.toml`` +
        ``instruction.md`` + ``tests/`` + ``solution/`` layout.

    Args:
        trace: Parsed trace to convert.
        output_dir: Parent directory for generated tasks.
        author: Author name for task.toml metadata.
        timeout_sec: Agent timeout in seconds (0 = auto-scale by difficulty).
        overwrite: If True, overwrite existing task directories.
        output_format: ``"task-md"`` for native tasks or ``"legacy"`` for
            split-format compatibility.

    Returns:
        Path to the created task directory.
    """
    output_format = _validate_output_format(output_format)
    output_dir = Path(output_dir)
    task_id = _task_id_from_trace(trace)
    task_dir = output_dir / task_id

    if task_dir.exists() and not overwrite:
        logger.info("Task %s already exists, skipping (use overwrite=True)", task_id)
        return task_dir

    # Render every artifact before replacing an existing task directory. If
    # trace metadata is unsafe (for example an unsupported git remote), fail
    # without leaving a half-regenerated benchmark behind.
    effective_timeout = timeout_sec if timeout_sec > 0 else None
    task_name = f"trace-import/{task_id}"
    toml_content = _build_task_toml(
        trace,
        task_name=task_name,
        author=author,
        timeout_sec=effective_timeout,
    )
    instruction = _build_instruction(trace)
    dockerfile = _build_dockerfile(trace)
    test_sh = _build_test_sh(trace)
    solution_sh = _build_solution_sh(trace)

    if task_dir.exists():
        if task_dir.is_symlink() or task_dir.is_file():
            task_dir.unlink()
        else:
            shutil.rmtree(task_dir)

    task_dir.mkdir(parents=True, exist_ok=True)

    if output_format == "task-md":
        (task_dir / "task.md").write_text(_build_task_md(toml_content, instruction))
    else:
        (task_dir / "task.toml").write_text(toml_content)
        (task_dir / "instruction.md").write_text(instruction)

    # Write environment/Dockerfile
    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(dockerfile)

    verifier_dir = task_dir / ("verifier" if output_format == "task-md" else "tests")
    verifier_dir.mkdir(exist_ok=True)
    test_path = verifier_dir / "test.sh"
    test_path.write_text(test_sh)
    test_path.chmod(0o755)

    oracle_dir = task_dir / ("oracle" if output_format == "task-md" else "solution")
    oracle_dir.mkdir(exist_ok=True)
    solution_path = oracle_dir / "solve.sh"
    solution_path.write_text(solution_sh)
    solution_path.chmod(0o755)

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
    output_format: TaskOutputFormat = "task-md",
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
        output_format: ``"task-md"`` for native tasks or ``"legacy"`` for
            split-format compatibility.

    Returns:
        List of paths to created task directories.
    """
    results: list[Path] = []
    output_format = _validate_output_format(output_format)
    eligible_traces, skipped = filter_traces_for_generation(
        traces,
        min_steps=min_steps,
        outcome_filter=outcome_filter,
    )

    for trace in eligible_traces:
        task_dir = generate_task(
            trace,
            output_dir,
            author=author,
            timeout_sec=timeout_sec,
            overwrite=overwrite,
            output_format=output_format,
        )
        results.append(task_dir)

    if skipped:
        logger.info("Skipped %d traces (filtered by steps/outcome/prompt)", skipped)

    return results


def filter_traces_for_generation(
    traces: list[ParsedTrace],
    *,
    min_steps: int = 2,
    outcome_filter: str | None = None,
) -> tuple[list[ParsedTrace], int]:
    """Return traces that would produce objective task directories."""
    eligible: list[ParsedTrace] = []
    skipped = 0

    for trace in traces:
        if len(trace.steps) < min_steps:
            skipped += 1
            continue
        if outcome_filter and trace.outcome != outcome_filter:
            skipped += 1
            continue
        if not trace.first_user_prompt:
            skipped += 1
            continue
        if trace.n_tool_calls == 0:
            skipped += 1
            continue
        if not trace.files_edited:
            skipped += 1
            continue
        eligible.append(trace)

    return eligible, skipped
