"""Discover recent Claude Code and Codex session files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_CLAUDE_SKIP_PARTS = frozenset({"subagents", "workflows"})


@dataclass(frozen=True)
class SessionHit:
    path: Path
    source: str
    mtime: float
    snippet: str

    @property
    def when(self) -> str:
        return datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d %H:%M")


def encode_claude_project_dir(absolute_path: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "-", absolute_path)


def list_recent_sessions(
    *, cwd: Path | None = None, home: Path | None = None, limit: int = 8
) -> list[SessionHit]:
    cwd = (cwd or Path.cwd()).resolve()
    home = home or Path.home()
    # Machines accumulate thousands of session files (12 GB of Codex JSONL is
    # normal), so sort candidates by mtime first and read prompt snippets only
    # for the files we will actually show.
    candidates = [
        *_claude_candidates(home, cwd),
        *_codex_candidates(home),
        *_trial_candidates(cwd),
    ]
    candidates.sort(key=lambda item: item[2], reverse=True)
    return [
        SessionHit(
            path=display_path,
            source=source,
            mtime=mtime,
            snippet=_first_prompt_snippet(snippet_path),
        )
        for display_path, source, mtime, snippet_path in candidates[:limit]
    ]


# (display_path, source, mtime, path-to-read-snippet-from)
_Candidate = tuple[Path, str, float, Path]


def _claude_candidates(home: Path, cwd: Path) -> list[_Candidate]:
    # Scan every project, not just the one for cwd: people run their agent
    # work in one directory and ask to submit from another. Global mtime
    # ranking surfaces "my most recent session anywhere", which is what a
    # submit flow means by "best recent session". Stat-ing thousands of files
    # is cheap; only the shown hits get their contents read.
    root = home / ".claude" / "projects"
    if not root.is_dir():
        return []
    candidates: list[_Candidate] = []
    for path in root.rglob("*.jsonl"):
        if any(part in _CLAUDE_SKIP_PARTS for part in path.parts):
            continue
        candidates.append((path, "claude", path.stat().st_mtime, path))
    return candidates


def _codex_candidates(home: Path) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for root in (home / ".codex" / "sessions", home / ".codex" / "archived_sessions"):
        if not root.is_dir():
            continue
        for path in root.rglob("*.jsonl"):
            candidates.append((path, "codex", path.stat().st_mtime, path))
    return candidates


def _trial_candidates(cwd: Path) -> list[_Candidate]:
    jobs = cwd / "jobs"
    if not jobs.is_dir():
        return []
    candidates: list[_Candidate] = []
    for path in jobs.rglob("acp_trajectory.jsonl"):
        candidates.append((path.parent.parent, "trial", path.stat().st_mtime, path))
    return candidates


def _first_prompt_snippet(path: Path, limit: int = 120) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]
    except OSError:
        return ""
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        text = _event_prompt(event)
        if text:
            return re.sub(r"\s+", " ", text).strip()[:limit]
    return ""


def _event_prompt(event: dict) -> str:
    etype = event.get("type")
    if etype == "user":
        content = (event.get("message") or {}).get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                str(block.get("text") or "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "\n".join(part for part in parts if part)
    if etype == "user_message":
        return str(event.get("text") or "")
    raw_payload = event.get("payload")
    payload: dict = raw_payload if isinstance(raw_payload, dict) else {}
    if etype == "event_msg" and payload.get("type") == "user_message":
        return str(payload.get("message") or "")
    return ""
