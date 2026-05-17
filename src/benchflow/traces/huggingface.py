"""Download and parse trace datasets from HuggingFace Hub.

Supports two dataset layouts:

1. **opentraces format** — JSONL with ``TraceRecord`` schema
2. **Claude Code merged traces** — JSONL with ``messages`` field
   (e.g. ``nlile/misc-merged-claude-code-traces-v1``)

Datasets are cached locally under ``.cache/traces/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from benchflow.traces.models import ParsedTrace, ToolCall, TraceStep
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
        "format": "claude-messages",
        "description": "949 production traces, 136k requests",
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
        from huggingface_hub import hf_hub_download

        # Try to download a data file directly
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename="data/train-00000-of-00001.parquet",
                repo_type="dataset",
            )
            # Convert parquet to JSONL
            _parquet_to_jsonl(Path(downloaded), out_path, max_rows=max_rows)
            return out_path
        except Exception:
            pass

        # Try JSONL file
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename="data/train.jsonl",
                repo_type="dataset",
            )
            _copy_jsonl(Path(downloaded), out_path, max_rows=max_rows)
            return out_path
        except Exception:
            pass

    except ImportError:
        logger.debug("huggingface_hub not installed, using API fallback")

    # Fallback: use the HF datasets API via httpx
    _download_via_api(repo_id, split=split, max_rows=max_rows, out_path=out_path)
    return out_path


def _parquet_to_jsonl(parquet_path: Path, out_path: Path, *, max_rows: int | None = None) -> None:
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


# ---------------------------------------------------------------------------
# Format-specific parsers for HuggingFace datasets
# ---------------------------------------------------------------------------


def _parse_claude_messages_row(row: dict[str, object], idx: int = 0) -> ParsedTrace | None:
    """Parse a row from a Claude Code merged-messages dataset.

    These datasets have a ``messages`` field containing a list of
    ``{role, content}`` dicts, sometimes with ``tool_use`` blocks.
    """
    import uuid

    messages = row.get("messages")
    if not isinstance(messages, (list, str)):
        return None

    # Some datasets store messages as a JSON string
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return None

    if not isinstance(messages, list) or not messages:
        return None

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
                elif isinstance(block, str):
                    text_parts.append(block)
            content_str = "\n".join(text_parts)
        else:
            content_str = str(content) if content else ""

        if content_str.strip() or tool_calls:
            steps.append(TraceStep(role=role, content=content_str, tool_calls=tool_calls))

    if not steps:
        return None

    trace_id = str(row.get("id", row.get("trace_id", f"hf-{idx}-{uuid.uuid4().hex[:8]}")))
    session_id = str(row.get("session_id", trace_id))
    model = str(row.get("model", "")) or None
    source = str(row.get("source", "")) or None

    tags: list[str] = ["huggingface"]
    if source:
        tags.append(f"source:{source}")

    return ParsedTrace(
        trace_id=trace_id,
        session_id=session_id,
        agent_name="claude-code",
        model=model,
        steps=steps,
        outcome="unknown",
        tags=tags,
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
            else:
                logger.warning("Unknown format %r, skipping row %d", format, i)

    logger.info("Loaded %d traces from %s", len(traces), repo_id)
    return traces
