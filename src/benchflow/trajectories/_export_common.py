"""Shared plumbing for the trajectory export formats.

The format emitters (:mod:`benchflow.trajectories.export` for Verifiers/ORS,
:mod:`benchflow.trajectories.export_atif` for ATIF, and
:mod:`benchflow.trajectories.export_adp` for ADP) all walk the same captured
ACP events and write per-rollout artifacts under ``rollout_dir/trainer/``.
This module owns the format-independent pieces:

- :func:`content_blocks_to_text` — render ACP tool-call content blocks to
  plain text. Sibling of the Verifiers-specific
  :func:`benchflow.trajectories.export._tool_call_to_content`, which renders
  the same blocks into a titled summary string.
- :class:`ThoughtBuffer` — the ``agent_thought`` accumulator shared by the
  ATIF and ADP event walkers.
- :func:`aggregate_rollout_jsonl` — the job-level JSONL aggregator behind
  ``write_job_verifiers_jsonl`` and ``write_job_adp_jsonl``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from benchflow.trajectories.types import redact_trajectory_text

logger = logging.getLogger(__name__)


def content_blocks_to_text(content: Any) -> str:
    """Render ACP tool-call content blocks to plain text.

    Handles both the flat shape (``{"text": ...}`` / ``{"content": "..."}``)
    and the nested ACP shape (``{"type": "content", "content": {"type":
    "text", "text": ...}}``). Non-text blocks are skipped.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        inner = item.get("content")
        if isinstance(inner, dict):
            inner = inner.get("text")
        text = item.get("text") or inner
        if text:
            parts.append(str(text))
    return "\n".join(parts)


class ThoughtBuffer:
    """Accumulate consecutive ``agent_thought`` texts between agent actions.

    The ATIF and ADP walkers both join buffered thoughts with a blank line
    and attach them as ``reasoning_content`` on the next agent action —
    flushing them as a standalone item when a user event or the end of the
    trajectory would otherwise drop them. The buffer owns the join-and-clear
    bookkeeping; what a flushed thought becomes stays format-specific.
    """

    def __init__(self) -> None:
        self._pending: list[str] = []

    def push(self, text: str) -> None:
        self._pending.append(text)

    def take(self) -> str | None:
        """Return the buffered thoughts joined by blank lines, then clear.

        ``None`` when nothing is buffered, so callers attach
        ``reasoning_content`` only when it exists.
        """
        if not self._pending:
            return None
        joined = "\n\n".join(self._pending)
        self._pending.clear()
        return joined


def aggregate_rollout_jsonl(
    job_dir: str | Path,
    *,
    rollout_relpath: str,
    out_filename: str,
) -> Path | None:
    """Concatenate per-rollout JSONL artifacts into one job-level dataset.

    Scans ``job_dir/*/<rollout_relpath>`` and writes their lines to
    ``job_dir/<out_filename>``. Each artifact is normalized to end with a
    newline and re-redacted on the way through, so a legacy or hand-placed
    raw rollout file cannot leak secrets into the job dataset. Unreadable
    artifacts are skipped with a warning rather than failing the whole
    aggregation. Returns the artifact path, or ``None`` when no rollouts
    have emitted records yet.
    """
    job_path = Path(job_dir)
    if not job_path.is_dir():
        return None
    rollout_files = sorted(job_path.glob(f"*/{rollout_relpath}"))
    if not rollout_files:
        return None
    out = job_path / out_filename
    with out.open("w") as fout:
        for src in rollout_files:
            try:
                text = src.read_text()
            except OSError as e:
                logger.warning("Skipping unreadable trainer artifact %s: %s", src, e)
                continue
            if not text.endswith("\n"):
                text = text + "\n"
            fout.write(redact_trajectory_text(text))
    return out
