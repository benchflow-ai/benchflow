"""Discover and parse local Claude Code sessions.

Scans ``~/.claude/projects/`` for JSONL session files and parses them
into :class:`~benchflow.traces.models.ParsedTrace` objects.
"""

from __future__ import annotations

import logging
from pathlib import Path

from benchflow.traces.models import ParsedTrace
from benchflow.traces.parsers import parse_claude_code_session

logger = logging.getLogger(__name__)

# Default Claude Code session directory
CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"


def _decode_project_path(encoded: str) -> str:
    """Decode an encoded Claude Code project directory name.

    Claude Code encodes the working directory by replacing ``/`` with ``-``.
    For example: ``/Users/dev/my-project`` → ``-Users-dev-my-project``.
    """
    # The leading dash represents the root /
    if encoded.startswith("-"):
        parts = encoded.split("-")
        # Filter empty parts from leading dash
        return "/" + "/".join(p for p in parts if p)
    return encoded


def discover_sessions(
    projects_dir: Path | None = None,
    *,
    project_filter: str | None = None,
    limit: int | None = None,
) -> list[Path]:
    """Find Claude Code JSONL session files on disk.

    Args:
        projects_dir: Override the default ``~/.claude/projects/`` path.
        project_filter: Only include sessions from projects whose decoded
            path contains this substring (case-insensitive).
        limit: Maximum number of session files to return.

    Returns:
        List of paths to ``.jsonl`` session files, newest first.
    """
    base = projects_dir or CLAUDE_PROJECTS_DIR

    if not base.exists():
        logger.info("Claude Code projects directory not found: %s", base)
        return []

    sessions: list[tuple[float, Path]] = []

    for project_dir in base.iterdir():
        if not project_dir.is_dir():
            continue

        # Apply project filter
        if project_filter:
            decoded = _decode_project_path(project_dir.name)
            if project_filter.lower() not in decoded.lower():
                continue

        for jsonl_file in project_dir.glob("*.jsonl"):
            if jsonl_file.stat().st_size > 0:
                sessions.append((jsonl_file.stat().st_mtime, jsonl_file))

    # Sort by modification time, newest first
    sessions.sort(key=lambda x: x[0], reverse=True)

    paths = [p for _, p in sessions]
    if limit:
        paths = paths[:limit]

    logger.info("Found %d Claude Code session files", len(paths))
    return paths


def load_local_sessions(
    projects_dir: Path | None = None,
    *,
    project_filter: str | None = None,
    limit: int | None = None,
) -> list[ParsedTrace]:
    """Discover and parse local Claude Code sessions.

    Args:
        projects_dir: Override the default session directory.
        project_filter: Only include sessions from matching projects.
        limit: Maximum number of sessions to parse.

    Returns:
        List of parsed traces.
    """
    paths = discover_sessions(projects_dir, project_filter=project_filter, limit=limit)

    traces: list[ParsedTrace] = []
    errors = 0
    for path in paths:
        try:
            trace = parse_claude_code_session(path)
            traces.append(trace)
        except Exception as e:
            logger.debug("Failed to parse %s: %s", path, e)
            errors += 1

    if errors:
        logger.info("Failed to parse %d/%d session files", errors, len(paths))

    logger.info("Parsed %d local Claude Code sessions", len(traces))
    return traces
