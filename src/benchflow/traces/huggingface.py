"""Download and parse trace datasets from HuggingFace Hub.

Supports three dataset layouts:

1. **opentraces format** — JSONL with ``TraceRecord`` schema
2. **Claude Code merged traces** — JSONL with ``messages`` or
   ``messages_json`` field (e.g. ``nlile/misc-merged-claude-code-traces-v1``)
3. **Claude Code request traces** — rows with ``requests`` list containing
   API-level request metadata (e.g. ``semianalysisai/cc-traces-weka-…``)

Datasets are cached locally under ``.cache/traces/``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from benchflow.traces.models import GitContext, ParsedTrace, ToolCall, TraceStep
from benchflow.traces.parsers import parse_opentraces_record

logger = logging.getLogger(__name__)

# Well-known HuggingFace trace datasets
KNOWN_DATASETS: dict[str, dict[str, str]] = {
    "opentraces-test": {
        "repo": "Jayfarei/opentraces-test",
        "format": "opentraces",
        "description": "58 traces, 19.6k steps — opentraces schema",
    },
    "cc-traces-merged": {
        "repo": "nlile/misc-merged-claude-code-traces-v1",
        "format": "claude-messages",
        "description": "32,133 deduplicated Claude Code traces",
    },
    "claudeset-community": {
        "repo": "lelouch0110/claudeset-community",
        "format": "claude-messages",
        "description": "Community Claude Code sessions",
    },
    "cc-traces-weka": {
        "repo": "semianalysisai/cc-traces-weka-no-subagents-051226",
        "format": "claude-requests",
        "description": "949 production traces, 136k requests (metadata only)",
    },
}


def _cache_dir() -> Path:
    """Local cache for downloaded datasets."""
    d = Path.cwd()
    root = d
    while d != d.parent:
        if (d / ".git").exists():
            root = d
            break
        d = d.parent
    cache = root / ".cache" / "traces"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _pick_split_file(repo_files: list[str], split: str, suffix: str) -> str | None:
    """Pick the repo file matching *split* and *suffix*, if any.

    Matches either the exact ``{split}{suffix}`` basename (e.g.
    ``data/test.jsonl`` for ``split="test"``) or the HF sharded-parquet
    convention ``{split}-NNNNN-of-NNNNN`` (e.g.
    ``data/test-00000-of-00001.parquet``). The sharded match is anchored on
    ``\\d+-of-\\d+`` so a sibling *subset* such as
    ``test-small-00000-of-00001.parquet`` is not mistaken for ``split="test"``.
    Returns ``None`` when no file matches so the caller can fall back to
    constructed filename candidates.
    """
    sharded_re = re.compile(rf"^{re.escape(split)}-\d+-of-\d+")
    candidates = [
        f
        for f in repo_files
        if f.endswith(suffix)
        and (
            Path(f).name == f"{split}{suffix}"
            or sharded_re.match(Path(f).name) is not None
        )
    ]
    if not candidates:
        return None
    # Prefer files under data/, then the shortest path for determinism.
    candidates.sort(key=lambda f: (not f.startswith("data/"), len(f), f))
    return candidates[0]


def _split_filename_candidates(
    matched: str | None, split: str, suffix: str
) -> list[str]:
    """Build an ordered list of filenames to try for *split* and *suffix*.

    If a file was matched from the repo listing it is tried first; otherwise
    fall back to the conventional ``data/{split}-00000-of-00001`` and
    ``data/{split}`` layouts. All candidates are split-specific so a
    ``split="test"`` request never resolves to ``train`` data.
    """
    candidates: list[str] = []
    if matched:
        candidates.append(matched)
    for guess in (
        f"data/{split}-00000-of-00001{suffix}",
        f"data/{split}{suffix}",
    ):
        if guess not in candidates:
            candidates.append(guess)
    return candidates


def _download_hf_dataset(
    repo_id: str,
    *,
    split: str = "train",
    max_rows: int | None = None,
    cache: Path | None = None,
) -> Path:
    """Download a HuggingFace dataset to a local JSONL file.

    Uses ``huggingface_hub`` if available, falls back to the datasets API.
    """
    cache = cache or _cache_dir()
    safe_name = repo_id.replace("/", "__")
    rows_suffix = f"_n{max_rows}" if max_rows else ""
    out_path = cache / f"{safe_name}__{split}{rows_suffix}.jsonl"

    if out_path.exists():
        logger.info("Using cached dataset: %s", out_path)
        return out_path

    logger.info("Downloading %s (split=%s) from HuggingFace...", repo_id, split)

    # Try huggingface_hub first
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        # List repo files so we can pick a file matching the requested split,
        # rather than hardcoding `train`. Picking the wrong file would silently
        # mislabel data (e.g. caching `train` rows under a `__test` filename).
        repo_files: list[str] = []
        try:
            repo_files = list(list_repo_files(repo_id, repo_type="dataset"))
        except Exception:
            logger.debug("Could not list files for %s", repo_id, exc_info=True)

        parquet_file = _pick_split_file(repo_files, split, ".parquet")
        jsonl_file = _pick_split_file(repo_files, split, ".jsonl")

        # Try to download a parquet data file for the split
        for filename in _split_filename_candidates(parquet_file, split, ".parquet"):
            try:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                )
                # Convert parquet to JSONL. The conversion is kept inside the
                # try/fallback scope: if a parquet file downloads but pyarrow
                # is missing (or decoding fails), the failure must fall through
                # to the JSONL candidates / `_download_via_api` rather than
                # propagating immediately (regression guarded — PR #323).
                _parquet_to_jsonl(Path(downloaded), out_path, max_rows=max_rows)
            except Exception:
                continue
            return out_path

        # Try a JSONL data file for the split
        for filename in _split_filename_candidates(jsonl_file, split, ".jsonl"):
            try:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                )
            except Exception:
                continue
            _copy_jsonl(Path(downloaded), out_path, max_rows=max_rows)
            return out_path

    except ImportError:
        logger.debug("huggingface_hub not installed, using API fallback")

    # Fallback: use the HF datasets API via httpx
    _download_via_api(repo_id, split=split, max_rows=max_rows, out_path=out_path)
    return out_path


def _parquet_to_jsonl(
    parquet_path: Path, out_path: Path, *, max_rows: int | None = None
) -> None:
    """Convert a parquet file to JSONL."""
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        rows = table.to_pylist()
        if max_rows:
            rows = rows[:max_rows]
        with out_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
    except ImportError as e:
        raise ImportError(
            "pyarrow is required for parquet datasets. "
            "Install with: pip install pyarrow"
        ) from e


def _copy_jsonl(src: Path, dst: Path, *, max_rows: int | None = None) -> None:
    """Copy a JSONL file with optional row limit."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open() as fin, dst.open("w") as fout:
        for i, line in enumerate(fin):
            if max_rows and i >= max_rows:
                break
            fout.write(line)


def _download_via_api(
    repo_id: str,
    *,
    split: str = "train",
    max_rows: int | None = None,
    out_path: Path,
) -> None:
    """Download dataset rows via the HuggingFace datasets API."""
    import httpx

    limit = max_rows or 100
    url = f"https://datasets-server.huggingface.co/rows?dataset={repo_id}&config=default&split={split}&offset=0&length={limit}"

    try:
        resp = httpx.get(url, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Failed to download {repo_id}: {e}") from e

    rows = data.get("rows", [])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row_wrapper in rows:
            row = row_wrapper.get("row", row_wrapper)
            f.write(json.dumps(row, default=str) + "\n")

    logger.info("Downloaded %d rows to %s", len(rows), out_path)


# Format-specific parsers for HuggingFace datasets


_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


def _strip_system_reminders(text: str) -> str:
    """Remove ``<system-reminder>…</system-reminder>`` blocks from text."""
    return _SYSTEM_REMINDER_RE.sub("", text).strip()


_GITDIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/", re.MULTILINE)

_TASK_DESCRIPTION_RE = re.compile(
    r"TASK DESCRIPTION:\s*\n(.*)",
    re.DOTALL,
)

_REPO_RE = re.compile(r"Repository:\s*(\S+)")
_COMMIT_RE = re.compile(r"Base commit:\s*(\S+)")


_TRAILING_BOILERPLATE_RE = re.compile(
    r"\n\s*CRITICAL INSTRUCTIONS.*",
    re.DOTALL | re.IGNORECASE,
)


def _extract_task_from_prompt(prompt: str) -> str:
    """Extract the actionable task from a structured user prompt.

    Many HuggingFace traces wrap the real task inside boilerplate
    (repo setup, base commit, task ID, "CRITICAL INSTRUCTIONS").
    This extracts just the ``TASK DESCRIPTION:`` section and strips
    trailing agent instructions.
    """
    m = _TASK_DESCRIPTION_RE.search(prompt)
    text = m.group(1).strip() if m else prompt.strip()
    # Remove trailing boilerplate like "CRITICAL INSTRUCTIONS - WORK FAST..."
    text = _TRAILING_BOILERPLATE_RE.sub("", text).strip()
    return text


def _parse_claude_messages_row(row: dict[str, Any], idx: int = 0) -> ParsedTrace | None:
    """Parse a row from a Claude Code merged-messages dataset.

    Handles two sub-formats found in datasets like cc-traces-merged:

    1. Rows with ``messages`` / ``messages_json`` — a list of
       ``{role, content}`` dicts, sometimes with ``tool_use`` blocks.
    2. Rows with ``user_prompt`` + ``assistant_response`` + ``gitdiff``
       but no messages array. File paths are extracted from the diff.
    """
    import uuid

    messages = row.get("messages") or row.get("messages_json")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            messages = None

    if not isinstance(messages, list) or not messages:
        # Fall back to user_prompt + assistant_response + gitdiff format
        return _parse_prompt_diff_row(row, idx=idx)

    steps: list[TraceStep] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = str(msg.get("role", "unknown"))
        content = msg.get("content", "")

        # Extract tool calls from content blocks
        tool_calls: list[ToolCall] = []
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(str(block.get("text", "")))
                    elif block.get("type") == "tool_use":
                        tool_calls.append(
                            ToolCall(
                                name=str(block.get("name", "unknown")),
                                input=block.get("input", {})
                                if isinstance(block.get("input"), dict)
                                else {},
                            )
                        )
                    elif block.get("type") == "tool_result":
                        result_content = block.get("content", "")
                        if isinstance(result_content, str):
                            text_parts.append(result_content)
                elif isinstance(block, str):
                    text_parts.append(block)
            content_str = "\n".join(text_parts)
        else:
            content_str = str(content) if content else ""

        # Strip system-reminder blocks from user messages
        if role == "user" and content_str:
            content_str = _strip_system_reminders(content_str)

        if content_str.strip() or tool_calls:
            steps.append(
                TraceStep(role=role, content=content_str, tool_calls=tool_calls)
            )

    if not steps:
        return None

    trace_id = str(
        row.get("id", row.get("trace_id", f"hf-{idx}-{uuid.uuid4().hex[:8]}"))
    )
    session_id = str(row.get("session_id", trace_id))
    model = str(row.get("model", "")) or None
    source = str(row.get("source", "")) or None

    tags: list[str] = ["huggingface"]
    if source:
        tags.append(f"source:{source}")

    outcome = _infer_outcome(steps)

    return ParsedTrace(
        trace_id=trace_id,
        session_id=session_id,
        agent_name="claude-code",
        model=model,
        steps=steps,
        outcome=outcome,
        tags=tags,
    )


def _infer_outcome(steps: list[TraceStep]) -> str:
    """Infer task outcome from the last assistant step."""
    for step in reversed(steps):
        if step.role == "assistant":
            lower = step.content.lower()
            if any(
                w in lower
                for w in (
                    "complete",
                    "done",
                    "finished",
                    "success",
                    "fixed",
                    "created",
                    "built",
                    "updated",
                    "refactored",
                    "implemented",
                    "added",
                )
            ):
                return "success"
            if any(w in lower for w in ("error", "failed", "cannot")):
                return "failure"
            break
    return "unknown"


def _parse_prompt_diff_row(row: dict[str, Any], idx: int = 0) -> ParsedTrace | None:
    """Parse a row that uses ``user_prompt`` + ``gitdiff`` instead of messages.

    Many rows in cc-traces-merged store the prompt in ``user_prompt``,
    the response in ``assistant_response``, and the actual code changes
    in ``gitdiff``.  File paths are extracted from the diff headers.
    """
    import uuid

    user_prompt = row.get("user_prompt")
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        return None

    assistant_response = row.get("assistant_response", "") or ""
    gitdiff = row.get("gitdiff", "") or ""

    # Extract the actionable task from boilerplate
    task_text = _extract_task_from_prompt(str(user_prompt))
    if not task_text:
        return None

    steps: list[TraceStep] = [
        TraceStep(role="user", content=task_text),
    ]

    # Build synthetic tool calls from gitdiff file paths
    diff_files = _GITDIFF_FILE_RE.findall(str(gitdiff))
    tool_calls: list[ToolCall] = []
    for fp in diff_files:
        tool_calls.append(ToolCall(name="Edit", input={"file_path": fp}))

    # Strip <think> blocks from assistant response
    clean_response = re.sub(
        r"<think>.*?</think>", "", str(assistant_response), flags=re.DOTALL
    ).strip()
    if clean_response or tool_calls:
        steps.append(
            TraceStep(
                role="assistant",
                content=clean_response[:2000] if clean_response else "",
                tool_calls=tool_calls,
            )
        )

    trace_id = str(row.get("id", f"hf-{idx}-{uuid.uuid4().hex[:8]}"))
    session_id = str(row.get("session_id", trace_id))
    model = str(row.get("model", "")) or None
    source_repo = str(row.get("source_repo", "")) or None

    tags: list[str] = ["huggingface"]
    if source_repo:
        tags.append(f"source:{source_repo}")

    # Extract git context from structured prompt
    raw_prompt = str(user_prompt)
    repo_m = _REPO_RE.search(raw_prompt)
    commit_m = _COMMIT_RE.search(raw_prompt)
    git_ctx = GitContext(
        repo=repo_m.group(1) if repo_m else None,
        commit_before=commit_m.group(1) if commit_m else None,
    )

    outcome = _infer_outcome(steps)
    # If we have a non-empty gitdiff, it's likely a success
    if outcome == "unknown" and gitdiff.strip():
        outcome = "success"

    return ParsedTrace(
        trace_id=trace_id,
        session_id=session_id,
        agent_name="claude-code",
        model=model,
        steps=steps,
        git=git_ctx,
        outcome=outcome,
        tags=tags,
    )


def _parse_claude_requests_row(row: dict[str, Any], idx: int = 0) -> ParsedTrace | None:
    """Parse a row from the cc-traces-weka request-metadata dataset.

    These datasets have a ``requests`` list with API-level metadata
    (token counts, timing, model) but no message content. Useful for
    difficulty estimation but produces minimal task instructions.
    """
    import uuid

    requests = row.get("requests")
    if not isinstance(requests, list) or not requests:
        return None

    trace_id = str(row.get("id", f"hf-req-{idx}-{uuid.uuid4().hex[:8]}"))
    models = row.get("models", [])
    model = str(models[0]) if isinstance(models, list) and models else None

    total_input = 0
    total_output = 0
    steps: list[TraceStep] = []
    for req in requests:
        if not isinstance(req, dict):
            continue
        total_input += int(req.get("in", 0))
        total_output += int(req.get("out", 0))
        req_model = str(req.get("model", ""))
        steps.append(
            TraceStep(
                role="assistant",
                content=f"API request: model={req_model}, "
                f"in={req.get('in', 0)}, out={req.get('out', 0)}",
            )
        )

    tags = ["huggingface", "request-metadata"]
    return ParsedTrace(
        trace_id=trace_id,
        session_id=trace_id,
        agent_name="claude-code",
        model=model,
        steps=steps,
        outcome="unknown",
        tags=tags,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
    )


def load_hf_dataset(
    repo_id: str,
    *,
    format: str | None = None,
    split: str = "train",
    max_rows: int | None = None,
) -> list[ParsedTrace]:
    """Download and parse a HuggingFace trace dataset.

    Args:
        repo_id: HuggingFace dataset ID (e.g. ``nlile/misc-merged-claude-code-traces-v1``),
            or an alias from :data:`KNOWN_DATASETS`.
        format: Dataset format override. Auto-detected from known datasets if None.
            Options: ``"opentraces"``, ``"claude-messages"``.
        split: Dataset split to load.
        max_rows: Maximum number of rows to download.

    Returns:
        List of parsed traces.
    """
    # Resolve aliases
    if repo_id in KNOWN_DATASETS:
        info = KNOWN_DATASETS[repo_id]
        repo_id = info["repo"]
        if format is None:
            format = info["format"]

    if format is None:
        format = "claude-messages"  # default assumption

    jsonl_path = _download_hf_dataset(repo_id, split=split, max_rows=max_rows)

    traces: list[ParsedTrace] = []
    with jsonl_path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(row, dict):
                continue

            if format == "opentraces":
                traces.append(parse_opentraces_record(row))
            elif format == "claude-messages":
                parsed = _parse_claude_messages_row(row, idx=i)
                if parsed:
                    traces.append(parsed)
            elif format == "claude-requests":
                parsed = _parse_claude_requests_row(row, idx=i)
                if parsed:
                    traces.append(parsed)
            else:
                logger.warning("Unknown format %r, skipping row %d", format, i)

    logger.info("Loaded %d traces from %s", len(traces), repo_id)
    return traces
