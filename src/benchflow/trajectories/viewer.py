"""Trace viewer for one BenchFlow rollout.

Python owns *data*, JavaScript owns *pixels*. This module reads a rollout
directory, normalizes ACP events (or legacy Claude Code ``turn*.txt``
stream-json) into one versioned payload, and embeds that payload in a page
whose renderer is ``viewer_assets/render.js``.

The split is deliberate: the published static site (slice 5) serves the same
``render.js`` against a payload fetched from HuggingFace, so a locally rendered
page and a published one cannot drift apart. ``bench eval view`` inlines the
payload instead of fetching it, which keeps ``trajectory.html`` a single
portable file that works with no network.

The canonical input is ``trajectory/acp_trajectory.jsonl`` plus ``result.json``
/ ``timing.json``. ``turn*.txt`` is a compatibility fallback only, never merged
with ACP.
"""

from __future__ import annotations

import base64
import html
import json
import math
import sys
from dataclasses import dataclass, field
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

from benchflow._utils.scoring import IDLE_TIMEOUT, TIMED_OUT, classify_error
from benchflow.trajectories._export_common import content_blocks_to_text

_ASSET_DIR = Path(__file__).with_name("viewer_assets")

#: Bump when a payload field changes meaning or disappears. ``render.js``
#: refuses a payload it was not written for rather than rendering it wrong,
#: which matters once pages and data ship from different places (slice 5).
PAYLOAD_SCHEMA_VERSION = 1

_TRAJECTORY_RELPATH = "trajectory/acp_trajectory.jsonl"

# Per-observation inlining budget. The page embeds every observation, so one
# runaway log would otherwise decide the file size. Both ends are kept: a
# failing assertion lands at the tail.
_MAX_OUTPUT_CHARS = 100_000
_OUTPUT_HEAD_CHARS = 60_000
_OUTPUT_TAIL_CHARS = 40_000

# Sentinel returned when a directory contains neither a rollout nor rollout
# children. ``serve`` uses the exact value to fail before starting a server.
_NO_TRAJECTORIES_HTML = "<p>No trajectory files found</p>"


@dataclass
class ViewEvent:
    """One normalized event shared by the ACP and stream-json paths."""

    type: str
    text: str = ""
    kind: str = ""
    title: str = ""
    status: str = ""
    blocks: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ViewTurn:
    """A setup group or one user-started turn in the timeline."""

    number: int | None
    events: list[ViewEvent]


@dataclass
class ViewRun:
    """Everything one rollout page is built from."""

    name: str
    source: str
    result: dict[str, Any]
    timing: dict[str, Any]
    turns: list[ViewTurn]
    has_trajectory_artifact: bool = False
    stream_meta: dict[str, Any] = field(default_factory=dict)


# ── artifact loading ──────────────────────────────────────────────────


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_json_list(path: Path) -> list[str] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, list):
        return None
    return [str(item) for item in value if item is not None]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read complete JSON-object lines, skipping malformed trailing data."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


# ── tool observations ─────────────────────────────────────────────────


def _is_binary_text(value: str) -> bool:
    if len(value) <= 20:
        return False
    printable = sum(char.isprintable() or char in "\n\r\t" for char in value)
    return printable / len(value) < 0.7


def _is_binary_block(block: dict[str, Any]) -> bool:
    """Decide per block, so one image cannot suppress its sibling text."""
    block_type = str(block.get("type", "")).lower()
    mime_type = str(block.get("mimeType", block.get("mime_type", ""))).lower()
    if block_type in {"image", "audio", "blob"}:
        return True
    if "blob" in block:
        return True
    if mime_type:
        return not mime_type.startswith("text/")
    nested = block.get("content", block.get("resource"))
    if isinstance(nested, dict):
        return _is_binary_block(nested)
    if isinstance(nested, list):
        return any(isinstance(item, dict) and _is_binary_block(item) for item in nested)
    return "data" in block and block_type != "text"


def _clipped_text_block(text: str, *, kind: str = "text") -> dict[str, Any]:
    """One text-ish payload block, bounded but honest about what it dropped."""
    if len(text) <= _MAX_OUTPUT_CHARS:
        return {"kind": kind, "text": text}
    return {
        "kind": kind,
        "text": f"{text[:_OUTPUT_HEAD_CHARS]}{text[-_OUTPUT_TAIL_CHARS:]}",
        "clip": {
            "dropped": len(text) - _OUTPUT_HEAD_CHARS - _OUTPUT_TAIL_CHARS,
            "at": _OUTPUT_HEAD_CHARS,
            "artifact": _TRAJECTORY_RELPATH,
        },
    }


def _tool_block(block: Any) -> dict[str, Any] | None:
    """Convert one ACP content block. Never drops a block silently.

    ``content_blocks_to_text`` renders only text-bearing blocks, so anything it
    skips — an ACP ``diff`` from an edit tool, an unknown provider block —
    falls back to structure or raw JSON here instead of vanishing. Slice 1's
    contract is that whatever the tool observed reaches the page.
    """
    if isinstance(block, str):
        if not block:
            return None
        if _is_binary_text(block):
            return {"kind": "binary"}
        return _clipped_text_block(block)
    if not isinstance(block, dict):
        return None if block is None else {"kind": "text", "text": str(block)}
    if _is_binary_block(block):
        return {"kind": "binary"}
    if str(block.get("type", "")).lower() == "diff":
        return {
            "kind": "diff",
            "path": str(block.get("path") or ""),
            "old": str(block.get("oldText") or block.get("old_text") or ""),
            "new": str(block.get("newText") or block.get("new_text") or ""),
        }
    text = content_blocks_to_text([block])
    if not text:
        resource = block.get("resource")
        if isinstance(resource, dict) and isinstance(resource.get("text"), str):
            text = resource["text"]
    if not text:
        try:
            return _clipped_text_block(
                json.dumps(block, indent=2, ensure_ascii=False), kind="json"
            )
        except (TypeError, ValueError):
            text = str(block)
    if _is_binary_text(text):
        return {"kind": "binary"}
    return _clipped_text_block(text)


def _tool_blocks(value: Any) -> list[dict[str, Any]]:
    """Normalize ACP ``tool_call.content`` into payload blocks."""
    if value in (None, "", [], {}):
        return []
    items = value if isinstance(value, list) else [value]
    blocks = [_tool_block(item) for item in items]
    return [block for block in blocks if block is not None]


# ── ACP normalization ─────────────────────────────────────────────────


def _normalize_acp_events(events: list[dict[str, Any]]) -> list[ViewEvent]:
    normalized: list[ViewEvent] = []
    for event in events:
        event_type = event.get("type")
        if event_type in {"user_message", "agent_message", "agent_thought"}:
            normalized.append(
                ViewEvent(type=str(event_type), text=str(event.get("text") or ""))
            )
        elif event_type == "tool_call":
            normalized.append(
                ViewEvent(
                    type="tool_call",
                    kind=str(event.get("kind") or "tool"),
                    title=str(event.get("title") or ""),
                    status=str(event.get("status") or ""),
                    blocks=_tool_blocks(event.get("content")),
                    tool_call_id=str(event.get("tool_call_id") or ""),
                )
            )
        elif event_type == "agent_timeout":
            normalized.append(
                ViewEvent(
                    type="agent_timeout",
                    text=str(event.get("reason") or "Agent timed out"),
                    meta={
                        "timeout_sec": _finite(event.get("timeout_sec")),
                        "pending_tool_call_ids": [
                            str(item)
                            for item in (event.get("pending_tool_call_ids") or [])
                        ],
                        "terminal_trajectory_complete": bool(
                            event.get("terminal_trajectory_complete")
                        ),
                    },
                )
            )
        elif event_type == "oracle":
            return_code = event.get("return_code")
            if return_code is None:
                oracle_status = "unknown"
            elif isinstance(return_code, bool):
                oracle_status = "completed" if return_code else "failed"
            else:
                oracle_status = "completed" if return_code == 0 else "failed"
            normalized.append(
                ViewEvent(
                    type="oracle",
                    title=str(event.get("command") or "Oracle solution"),
                    status=oracle_status,
                    blocks=_tool_blocks(event.get("stdout")),
                    meta={"return_code": return_code},
                )
            )
    return normalized


def _group_acp_turns(
    events: list[ViewEvent], prompts: list[str] | None
) -> list[ViewTurn]:
    """Group at user-message boundaries, keeping pre-prompt setup separate."""
    has_user_events = any(event.type == "user_message" for event in events)
    if not has_user_events and prompts:
        setup = [event for event in events if event.type == "oracle"]
        trace = [event for event in events if event.type != "oracle"]
        turns: list[ViewTurn] = []
        if setup:
            turns.append(ViewTurn(number=None, events=setup))
        for index, prompt in enumerate(prompts, start=1):
            turn_events = [ViewEvent(type="user_message", text=prompt)]
            if index == 1:
                turn_events.extend(trace)
            turns.append(ViewTurn(number=index, events=turn_events))
        return turns

    setup_events: list[ViewEvent] = []
    turns = []
    current: ViewTurn | None = None
    for event in events:
        if event.type == "user_message":
            current = ViewTurn(number=len(turns) + 1, events=[event])
            turns.append(current)
        elif current is None:
            setup_events.append(event)
        else:
            current.events.append(event)
    if setup_events:
        turns.insert(0, ViewTurn(number=None, events=setup_events))
    return turns


# ── stream-json fallback ──────────────────────────────────────────────


def _legacy_tool_title(name: str, arguments: Any) -> str:
    if not isinstance(arguments, dict):
        return str(arguments or name)
    if name.lower() in {"bash", "shell", "execute"}:
        return str(arguments.get("command") or arguments.get("cmd") or name)
    path = arguments.get("file_path", arguments.get("path"))
    if path:
        return str(path)
    try:
        rendered = json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        rendered = str(arguments)
    return rendered if rendered != "{}" else name


def _normalize_stream_turn(
    raw_events: list[dict[str, Any]], prompt: str | None
) -> tuple[list[ViewEvent], dict[str, Any]]:
    """Convert one Claude Code stream-json file to canonical view events."""
    normalized: list[ViewEvent] = []
    tools: dict[str, ViewEvent] = {}
    metadata: dict[str, Any] = {}
    if prompt:
        normalized.append(ViewEvent(type="user_message", text=prompt))

    for raw in raw_events:
        event_type = raw.get("type")
        if event_type == "system":
            for key in ("model", "session_id", "claude_code_version"):
                if raw.get(key) is not None:
                    metadata[key] = raw[key]
        elif event_type == "assistant":
            message = raw.get("message")
            blocks = message.get("content", []) if isinstance(message, dict) else []
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "thinking":
                    normalized.append(
                        ViewEvent(
                            type="agent_thought",
                            text=str(block.get("thinking") or ""),
                        )
                    )
                elif block_type == "text":
                    normalized.append(
                        ViewEvent(
                            type="agent_message", text=str(block.get("text") or "")
                        )
                    )
                elif block_type == "tool_use":
                    name = str(block.get("name") or "tool")
                    tool_id = str(block.get("id") or "")
                    view_event = ViewEvent(
                        type="tool_call",
                        kind=name,
                        title=_legacy_tool_title(name, block.get("input")),
                        status="pending",
                        tool_call_id=tool_id,
                    )
                    normalized.append(view_event)
                    if tool_id:
                        tools[tool_id] = view_event
        elif event_type == "user":
            message = raw.get("message")
            blocks = message.get("content", []) if isinstance(message, dict) else []
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_id = str(block.get("tool_use_id") or "")
                target = tools.get(tool_id)
                observed = _tool_blocks(block.get("content"))
                if target is not None:
                    target.blocks = observed
                    target.status = "failed" if block.get("is_error") else "completed"
                else:
                    normalized.append(
                        ViewEvent(
                            type="tool_call",
                            kind="tool output",
                            title=tool_id or "Unmatched tool result",
                            status="failed" if block.get("is_error") else "completed",
                            blocks=observed,
                            tool_call_id=tool_id,
                        )
                    )
        elif event_type == "result":
            if isinstance(raw.get("total_cost_usd"), (int, float)):
                metadata["cost_usd"] = metadata.get("cost_usd", 0.0) + float(
                    raw["total_cost_usd"]
                )
            result_text = str(raw.get("result") or "")
            if result_text:
                metadata["result"] = result_text
                if not any(
                    event.type == "agent_message" and event.text == result_text
                    for event in normalized
                ):
                    normalized.append(ViewEvent(type="agent_message", text=result_text))
    return normalized, metadata


def _stream_turns(
    turn_files: list[Path], prompts: list[str] | None
) -> tuple[list[ViewTurn], dict[str, Any]]:
    turns: list[ViewTurn] = []
    metadata: dict[str, Any] = {}
    for index, path in enumerate(turn_files, start=1):
        prompt = prompts[index - 1] if prompts and index <= len(prompts) else None
        events, turn_meta = _normalize_stream_turn(_read_jsonl(path), prompt)
        turns.append(ViewTurn(number=index, events=events))
        for key, value in turn_meta.items():
            if key == "cost_usd":
                metadata[key] = metadata.get(key, 0.0) + value
            elif value not in (None, ""):
                metadata[key] = value
    return turns, metadata


# ── run assembly ──────────────────────────────────────────────────────


def _load_timing(rollout_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    timing_path = rollout_dir / "timing.json"
    timing = _read_json_object(timing_path) if timing_path.exists() else {}
    if timing:
        return timing
    embedded = result.get("timing")
    return embedded if isinstance(embedded, dict) else {}


def _build_acp_run(
    rollout_dir: Path, acp_path: Path, prompts: list[str] | None
) -> ViewRun:
    result = _read_json_object(rollout_dir / "result.json")
    events = _normalize_acp_events(_read_jsonl(acp_path))
    return ViewRun(
        name=rollout_dir.name,
        source="acp",
        result=result,
        timing=_load_timing(rollout_dir, result),
        turns=_group_acp_turns(events, prompts),
        has_trajectory_artifact=True,
    )


def _build_stream_run(
    rollout_dir: Path, turn_files: list[Path], prompts: list[str] | None
) -> ViewRun:
    result = _read_json_object(rollout_dir / "result.json")
    turns, stream_meta = _stream_turns(turn_files, prompts)
    return ViewRun(
        name=rollout_dir.name,
        source="stream-json",
        result=result,
        timing=_load_timing(rollout_dir, result),
        turns=turns,
        stream_meta=stream_meta,
    )


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _reward(result: dict[str, Any]) -> float | None:
    rewards = result.get("rewards")
    if not isinstance(rewards, dict):
        return None
    return _finite(rewards.get("reward"))


def _status(run: ViewRun) -> tuple[str, str]:
    result = run.result
    if result.get("verifier_error"):
        return "verifier-errored", "Verifier errored"
    has_timeout_event = any(
        event.type == "agent_timeout" for turn in run.turns for event in turn.events
    )
    error = result.get("error")
    # ``error_category`` is canonical, but pre-#503 rollouts predate the field;
    # re-deriving it with the shared classifier keeps one definition of what
    # counts as a timeout instead of a second copy of the substring table.
    error_category = str(
        result.get("error_category")
        or classify_error(str(error) if error else None)
        or ""
    ).lower()
    if has_timeout_event or error_category in {TIMED_OUT, IDLE_TIMEOUT}:
        return "timeout", "Timeout"
    if error:
        return "errored", "Errored"
    reward = _reward(result)
    if reward is None:
        return "not-scored", "Not scored"
    # Project-wide convention (see benchflow.eval_lift.Trial.passed): only a
    # perfect reward is a pass; partial credit is a failure.
    if reward == 1.0:
        return "passed", "Passed"
    return "failed", "Failed"


# ── payload ───────────────────────────────────────────────────────────


def _event_payload(event: ViewEvent) -> dict[str, Any]:
    if event.type in {"user_message", "agent_message", "agent_thought"}:
        return {"type": event.type, "text": event.text}
    if event.type == "tool_call":
        return {
            "type": "tool_call",
            "kind": event.kind,
            "title": event.title,
            "status": event.status,
            "tool_call_id": event.tool_call_id,
            "blocks": event.blocks,
        }
    if event.type == "agent_timeout":
        return {"type": "agent_timeout", "text": event.text, **event.meta}
    return {
        "type": "oracle",
        "title": event.title,
        "status": event.status,
        "return_code": event.meta.get("return_code"),
        "blocks": event.blocks,
    }


def _usage_payload(
    agent_result: dict[str, Any], *, stream_meta: dict[str, Any]
) -> dict[str, Any]:
    def _count(key: str) -> int | None:
        value = agent_result.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    # A stream-json run has no agent_result; its cost is the sum the fallback
    # parser took from the `result` events.
    cost = _finite(agent_result.get("cost_usd"))
    if cost is None:
        cost = _finite(stream_meta.get("cost_usd"))

    # Read ``agent_result``, never ``final_metrics``: the latter has no
    # cache-creation field, so its four numbers do not add up to total_tokens.
    return {
        "input": _count("n_input_tokens"),
        "output": _count("n_output_tokens"),
        "cache_creation": _count("n_cache_creation_tokens"),
        "cache_read": _count("n_cache_read_tokens"),
        "total": _count("total_tokens"),
        "cost_usd": cost,
        "source": agent_result.get("usage_source") or None,
        "price_source": agent_result.get("price_source") or None,
    }


def _notices_payload(result: dict[str, Any]) -> list[dict[str, str]]:
    notices: list[dict[str, str]] = []
    if result.get("error"):
        notices.append(
            {"level": "error", "title": "Agent error", "body": str(result["error"])}
        )
    if result.get("verifier_error"):
        notices.append(
            {
                "level": "error",
                "title": "Verifier error",
                "body": str(result["verifier_error"]),
            }
        )
    if result.get("partial_trajectory"):
        notices.append(
            {
                "level": "warning",
                "title": "Partial trajectory",
                "body": "This trace may end before the agent stopped.",
            }
        )
    return notices


def _run_payload(run: ViewRun) -> dict[str, Any]:
    result = run.result
    agent_result = result.get("agent_result")
    if not isinstance(agent_result, dict):
        agent_result = {}
    status_slug, status_label = _status(run)

    captured_tool_calls = sum(
        event.type == "tool_call" for turn in run.turns for event in turn.events
    )
    tool_calls = result.get("n_tool_calls")
    if tool_calls is None:
        tool_calls = agent_result.get("n_tool_calls", captured_tool_calls)

    timing = {
        key: value
        for key, value in (
            (key, _finite(run.timing.get(key)))
            for key in (
                "environment_setup",
                "agent_setup",
                "agent_execution",
                "verifier",
                "total",
            )
        )
        if value is not None
    }

    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "name": run.name,
        "source": run.source,
        "status": {"slug": status_slug, "label": status_label},
        "reward": _reward(result),
        "meta": {
            "task_name": result.get("task_name") or None,
            "agent_name": result.get("agent_name") or None,
            "agent": result.get("agent") or None,
            "model": result.get("model") or run.stream_meta.get("model") or None,
            "skill_mode": result.get("skill_mode") or None,
            "n_tool_calls": tool_calls,
            "n_skill_invocations": result.get("n_skill_invocations"),
        },
        "usage": _usage_payload(agent_result, stream_meta=run.stream_meta),
        "timing": timing,
        "notices": _notices_payload(result),
        "turns": [
            {
                "number": turn.number,
                "events": [_event_payload(event) for event in turn.events],
            }
            for turn in run.turns
        ],
        "artifacts": {
            "trajectory": _TRAJECTORY_RELPATH if run.has_trajectory_artifact else None
        },
    }


def build_run_payload(
    rollout_dir: Path | str, prompts: list[str] | None = None
) -> dict[str, Any] | None:
    """Normalize one rollout directory into the viewer payload.

    Returns ``None`` when the directory holds no trajectory this viewer can
    read. The payload is the contract between this module and ``render.js``;
    it is also what slice 5 will publish as ``viewer_data/<run_id>.json``.
    """
    rollout_dir = Path(rollout_dir)
    if prompts is None:
        prompts = _read_json_list(rollout_dir / "prompts.json")

    acp_path = rollout_dir / "trajectory" / "acp_trajectory.jsonl"
    if acp_path.is_file():
        return _run_payload(_build_acp_run(rollout_dir, acp_path, prompts))

    turn_files = sorted(rollout_dir.glob("turn*.txt"))
    if turn_files:
        return _run_payload(_build_stream_run(rollout_dir, turn_files, prompts))
    return None


# ── document assembly ─────────────────────────────────────────────────


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


@cache
def _asset_text(name: str) -> str:
    return (_ASSET_DIR / name).read_text()


@lru_cache(maxsize=1)
def _font_data() -> str:
    return base64.b64encode(
        (_ASSET_DIR / "JetBrainsMono-Regular.woff2").read_bytes()
    ).decode("ascii")


def _embed_json(payload: dict[str, Any]) -> str:
    """Serialize for a ``<script type="application/json">`` island.

    Tool output routinely contains ``</script>``. Escaping the three markup
    characters keeps the island from being closed early and stays valid JSON.
    """
    return (
        json.dumps(payload, ensure_ascii=False, default=str)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _document(title: str, body: str) -> str:
    # Order matters: the vendored PostTrainBench sheet is the base and
    # benchflow.css only ever adds to it, so it has to load second.
    css = _asset_text("ptb-styles.css") + _asset_text("benchflow.css").replace(
        "__BF_FONT_DATA__", _font_data()
    )
    template = _asset_text("viewer.html")
    replacements = {
        "__BF_TITLE__": _escape(title),
        "__BF_CSS__": css,
        "__BF_BODY__": body,
        "__BF_JS__": f"{_asset_text('render.js')}\n{_asset_text('boot.js')}",
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def _run_document(title: str, payload: dict[str, Any]) -> str:
    """A run page: mount point, data island, and a no-JS escape hatch."""
    body = (
        '<div id="bf-app"><noscript>This trace renders with JavaScript. The '
        f"canonical events are in <code>{_TRAJECTORY_RELPATH}</code> next to "
        "this file.</noscript></div>"
        '<script id="bf-run-data" type="application/json">'
        f"{_embed_json(payload)}</script>"
    )
    return _document(title, body)


def _render_job_hint(rollout_dir: Path, rollouts: list[str]) -> str:
    """A job directory has no single run to draw — list what it holds.

    Server-rendered in the vendored stylesheet's vocabulary rather than
    handed to ``render.js``: there is no run payload here, and boot.js keys
    off the absent ``#bf-app`` mount to leave this body alone.
    """
    items = "".join(f"<li><code>{_escape(name)}</code></li>" for name in rollouts)
    message = (
        '<header class="topbar"><div class="topbar-inner">'
        '<a href="#" class="logo">Bench<span class="logo-accent">Flow</span></a>'
        '<span class="logo-sub">/ traces</span></div></header>'
        '<div class="layout bf-no-right-rail"><main class="content">'
        '<div class="card"><span class="summary-label">Job directory</span>'
        f"<h2>{_escape(rollout_dir.name)}</h2>"
        f"<p class='muted'>This looks like a job directory with "
        f"{len(rollouts)} rollout(s). The viewer opens one rollout at a "
        "time:</p>"
        f"<ul>{items}</ul>"
        f"<pre>bench eval view {_escape(rollout_dir.name)}/&lt;rollout&gt;</pre>"
        "</div></main></div>"
    )
    return _document(f"benchflow — {rollout_dir.name}", message)


def render_rollout(rollout_dir: Path, prompts: list[str] | None = None) -> str:
    """Render one rollout, preferring canonical ACP over legacy stream-json."""
    rollout_dir = Path(rollout_dir)
    payload = build_run_payload(rollout_dir, prompts)
    if payload is not None:
        return _run_document(f"benchflow — {rollout_dir.name}", payload)

    try:
        rollouts = sorted(
            child.name
            for child in rollout_dir.iterdir()
            if child.is_dir()
            and (
                (child / "trajectory" / "acp_trajectory.jsonl").is_file()
                or any(child.glob("turn*.txt"))
            )
        )
    except OSError:
        rollouts = []
    if rollouts:
        return _render_job_hint(rollout_dir, rollouts)
    return _NO_TRAJECTORIES_HTML


def _render_acp_trajectory(
    rollout_dir: Path, acp_path: Path, prompts: list[str] | None
) -> str:
    """Compatibility entry point used by older viewer tests and callers."""
    return _run_document(
        f"benchflow — {Path(rollout_dir).name}",
        _run_payload(_build_acp_run(Path(rollout_dir), Path(acp_path), prompts)),
    )


def serve(
    rollout_path: str, port: int = 8888, prompts: list[str] | None = None
) -> None:
    """Write and serve a self-contained trajectory page for one rollout."""
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    path = Path(rollout_path)
    if not path.is_dir():
        print(f"Not a directory: {path}")
        sys.exit(1)

    html_content = render_rollout(path, prompts)
    if html_content == _NO_TRAJECTORIES_HTML:
        print(f"No trajectories found in {path}")
        sys.exit(1)
    (path / "trajectory.html").write_text(html_content)

    print(f"Trajectory viewer: http://localhost:{port}")
    print(f"Trial: {path}")
    print("Press Ctrl+C to stop\n")

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_content.encode())

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = HTTPServer(("localhost", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m benchflow.trajectories.viewer <rollout_dir> [port]")
        sys.exit(1)
    selected_port = int(sys.argv[2]) if len(sys.argv) > 2 else 8888
    serve(sys.argv[1], selected_port)
