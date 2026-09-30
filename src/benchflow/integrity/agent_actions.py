"""Agent action records projected from the ACP trajectory the host recorded.

Ported from BenchGuard (arXiv 2609.11028;
``src/benchflow/benchguard/agent_actions.py``). An ACP agent acts inside the
sandbox: its shell commands and file edits never pass through BenchFlow's
sandbox object, so the trajectory is where they exist. This module projects
that stream into action records with ``actor_class="Agent"``, which the trace
checker's agent-violation rules key on.

Changes from BenchGuard: tool arguments are read from 0.8's ``raw_input``
field as well as ``arguments``; ``TaskRuntime.bash`` calls (``tool_name:
"bash"`` with a top-level ``command``) are commands; previews go through
BenchFlow's own trajectory redaction (``redact_trajectory_text``).

The input is always the host's copy of the stream (the rollout's in-memory
trajectory, or ``trajectory/acp_trajectory.jsonl`` that the host wrote), never
a file read back from the sandbox. An agent can still hide what a command did
(an interpreter script is opaque); it cannot make a recorded command vanish.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from benchflow.integrity.classify import (
    classify_resource_class,
    normalize_manifest_resources,
)
from benchflow.integrity.constants import ACTION_RECORD_SCHEMA_VERSION
from benchflow.integrity.exec_decompose import _resolve_path, decompose_exec_command

# Tool kinds that carry a shell command, mapped to the argument holding it.
_COMMAND_KINDS: dict[str, str] = {
    "Bash": "command",
    "bash": "command",
    "bash_command": "keystrokes",
    "exec_command": "cmd",
    "execute": "command",
    "exec": "command",
    "shell": "command",
    "run_command": "command",
    "search": "command",
    "read": "command",
    "write_stdin": "chars",
}
# Tool kinds that write a file, mapped to the argument holding the path.
_WRITE_KINDS: dict[str, str] = {
    "Write": "file_path",
    "Edit": "file_path",
    "MultiEdit": "file_path",
    "NotebookEdit": "notebook_path",
    "create_file": "path",
    "str_replace_editor": "path",
    "edit": "file_path",
    "write_file": "file_path",
}
# Tool kinds that read a file, mapped to the argument holding the path.
_READ_KINDS: dict[str, str] = {
    "Read": "file_path",
    "view_image": "path",
    "open_file": "path",
    "read_file": "file_path",
}
_NETWORK_KINDS: frozenset[str] = frozenset(
    {"WebFetch", "WebSearch", "web_search_call", "fetch", "web_search", "navigate"}
)
# Harness bookkeeping: no sandbox side effect to witness.
_IGNORED_KINDS: frozenset[str] = frozenset(
    {
        "TaskCreate",
        "TaskUpdate",
        "TodoWrite",
        "ToolSearch",
        "update_plan",
        "mark_task_complete",
        "harbor_observation",
        "Agent",
        "Skill",
        "skill",
        "ListAgents",
        "SendMessage",
        "AskUserQuestion",
        "EnterPlanMode",
        "ExitPlanMode",
        "switch_mode",
        "create_goal",
        "update_goal",
        "get_goal",
        "think",
    }
)
# codex-acp sends `read`/`search` as shell commands; claude-agent-acp sends
# them as a Read carrying `file_path` and a Grep/Glob carrying the directory.
# The arguments decide which one a call is (see _ambiguous_read_path).
_AMBIGUOUS_READ_KINDS: frozenset[str] = frozenset({"read", "search"})

_FALLBACK_COMMAND_KEYS = ("command", "cmd", "keystrokes", "script")
_FALLBACK_PATH_KEYS = ("file_path", "path", "filename")
_PREVIEW_LIMIT = 200


@dataclass
class TrajectorySynthesis:
    """Records projected from a trajectory plus a coverage summary."""

    records: list[dict[str, Any]]
    summary: dict[str, Any] = field(default_factory=dict)


def action_records_from_trajectory(
    trajectory: list[dict[str, Any]] | None,
    *,
    agent_cwd: str | None = None,
    manifest_resources: Any = None,
    network_class: str | None = None,
) -> TrajectorySynthesis:
    """Project trajectory tool calls into Agent-attributed action records.

    ``manifest_resources`` takes ``(path, resource_class)`` pairs (the
    contract's roots); ``network_class`` is the task's authorized egress
    class, stamped onto network records so the forbidden-network rule has a
    policy to compare against.
    """

    resources = normalize_manifest_resources(manifest_resources)
    records: list[dict[str, Any]] = []
    kind_counts: dict[str, int] = {}
    unmapped_kinds: dict[str, int] = {}
    # A mapped kind whose command or path could not be recovered is a
    # coverage gap, counted so the summary shows how much went unwitnessed.
    dropped_missing_input: dict[str, int] = {}
    tool_calls = _dedupe_tool_calls(trajectory or [])

    for index, event in enumerate(tool_calls, start=1):
        kind = _tool_kind(event)
        arguments = _arguments_from_event(event)
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        cwd = (
            arguments.get("workdir")
            if isinstance(arguments.get("workdir"), str)
            else agent_cwd
        )

        base = _base_record(event, sequence=index)
        ambiguous_path = _ambiguous_read_path(kind, arguments)
        if ambiguous_path is not None:
            records.append(
                _file_record(
                    base, path=ambiguous_path, write=False, resources=resources, cwd=cwd
                )
            )
        elif kind in _COMMAND_KINDS or _fallback_command_key(kind, arguments):
            key = _COMMAND_KINDS.get(kind) or _fallback_command_key(kind, arguments)
            command = arguments.get(key) if key else None
            unvouched_title = False
            if not isinstance(command, str) or not command.strip():
                command = _title_command_fallback(kind, event)
                unvouched_title = isinstance(
                    command, str
                ) and _title_command_is_verbatim(event, command)
            if not isinstance(command, str) or not command.strip():
                dropped_missing_input[kind] = dropped_missing_input.get(kind, 0) + 1
                continue
            records.append(
                _command_record(
                    base,
                    command=command,
                    cwd=cwd,
                    resources=resources,
                    via_stdin=kind == "write_stdin",
                    command_from_title=unvouched_title,
                    network_class=network_class,
                )
            )
        elif kind in _WRITE_KINDS or kind in _READ_KINDS:
            # A spec-defined ``diff`` content block names the file the agent
            # changed; prefer it, one record per block.
            diff_paths = _diff_content_paths(event)
            if diff_paths:
                for path in diff_paths:
                    records.append(
                        _file_record(
                            dict(base),
                            path=path,
                            write=True,
                            resources=resources,
                            cwd=cwd,
                        )
                    )
                continue
            key = _WRITE_KINDS.get(kind) or _READ_KINDS[kind]
            path = (
                arguments.get(key)
                or _fallback_path(arguments)
                or _title_path_fallback(event)
            )
            if not isinstance(path, str) or not path:
                dropped_missing_input[kind] = dropped_missing_input.get(kind, 0) + 1
                continue
            records.append(
                _file_record(
                    base,
                    path=path,
                    write=kind in _WRITE_KINDS,
                    resources=resources,
                    cwd=cwd,
                )
            )
        elif kind == "apply_patch":
            patch_text = arguments.get("input")
            targets = _patch_targets(patch_text if isinstance(patch_text, str) else "")
            if not targets:
                dropped_missing_input[kind] = dropped_missing_input.get(kind, 0) + 1
            for path in targets:
                records.append(
                    _file_record(
                        dict(base), path=path, write=True, resources=resources, cwd=cwd
                    )
                )
        elif kind in _NETWORK_KINDS:
            records.append(
                _network_record(base, arguments=arguments, network_class=network_class)
            )
        elif kind in _IGNORED_KINDS:
            continue
        else:
            path = _fallback_path(arguments)
            if isinstance(path, str) and path:
                # An unknown tool touching a concrete path: witness it as an
                # observation rather than dropping it.
                records.append(
                    _file_record(
                        base, path=path, write=False, resources=resources, cwd=cwd
                    )
                )
            else:
                unmapped_kinds[kind] = unmapped_kinds.get(kind, 0) + 1

    summary = {
        "tool_call_count": len(tool_calls),
        "record_count": len(records),
        "kind_counts": kind_counts,
        "unmapped_kinds": unmapped_kinds,
        "dropped_missing_input": dropped_missing_input,
    }
    return TrajectorySynthesis(records=records, summary=summary)


def redacted_preview(value: str | None, *, limit: int = _PREVIEW_LIMIT) -> str:
    """BenchFlow's trajectory redaction, newlines flattened, truncated."""

    from benchflow.trajectories.types import redact_trajectory_text

    return redact_trajectory_text(value or "").replace("\n", "\\n")[:limit]


def _tool_kind(event: dict[str, Any]) -> str:
    kind = event.get("kind")
    if isinstance(kind, str) and kind and kind != "other":
        return kind
    # TaskRuntime.bash and other externally recorded calls carry only a name.
    name = event.get("tool_name")
    if isinstance(name, str) and name:
        return name
    return str(kind or "")


def _dedupe_tool_calls(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per tool_call_id, preferring the completed sighting."""

    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    anonymous: list[dict[str, Any]] = []
    for event in trajectory:
        if not isinstance(event, dict) or event.get("type") != "tool_call":
            continue
        call_id = event.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            anonymous.append(event)
            continue
        if call_id not in by_id:
            by_id[call_id] = event
            order.append(call_id)
        elif by_id[call_id].get("status") != "completed":
            by_id[call_id] = event
    return [by_id[call_id] for call_id in order] + anonymous


def _base_record(event: dict[str, Any], *, sequence: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": ACTION_RECORD_SCHEMA_VERSION,
        "sequence": sequence,
        "operation": "agent.tool_call",
        "phase": "Agent",
        "actor": "actor.agent",
        "actor_class": "Agent",
        "trust_domain": "Untrusted",
        "attribution_source": "trajectory",
        "status": _record_status(event),
        "tool_kind": _tool_kind(event),
        "tool_call_id": event.get("tool_call_id"),
    }
    status = event.get("status")
    if status != "completed":
        record["tool_status"] = status
    timestamp_ms = _timestamp_ms(event.get("timestamp"))
    if timestamp_ms is not None:
        record["timestamp_unix_ms"] = timestamp_ms
    return record


def _record_status(event: dict[str, Any]) -> str:
    status = event.get("status")
    if status in {"failed", "error", "cancelled"}:
        return "error"
    # ``pending`` means the call was issued and its completion never landed:
    # the action happened, it was not denied.
    return "ok"


def _command_record(
    base: dict[str, Any],
    *,
    command: str,
    cwd: str | None,
    resources: Any,
    via_stdin: bool,
    command_from_title: bool = False,
    network_class: str | None = None,
) -> dict[str, Any]:
    decomposition = decompose_exec_command(command, cwd=cwd)
    derived_targets = [
        {
            "path": target.path,
            "mode": target.mode,
            "mechanism": target.mechanism,
            "resource_class": classify_resource_class(target.path, resources),
        }
        for target in decomposition.targets
    ]
    record = {
        **base,
        "event_type": "AgentExec",
        "action_class": "Execute",
        "cwd": cwd,
        "command_sha256": _sha256_text(command),
        "command_preview": redacted_preview(command),
    }
    if via_stdin:
        record["via_stdin"] = True
    if command_from_title:
        record["command_from_title"] = True
    if derived_targets:
        record["derived_targets"] = derived_targets
    if decomposition.opaque:
        record["exec_opaque"] = True
    if decomposition.truncated:
        record["exec_targets_truncated"] = True
    if decomposition.referenced_paths:
        record["referenced_paths"] = list(decomposition.referenced_paths)
    if decomposition.network_targets:
        record["derived_network"] = [
            {"url": redacted_preview(url, limit=240)}
            for url in decomposition.network_targets
        ]
        if network_class:
            record["network_class"] = network_class
    if decomposition.loopback_targets:
        # Not egress, but a declared protected service route lives behind a
        # loopback URL, so the contract gets to classify it.
        record["derived_loopback"] = [
            {"url": url} for url in decomposition.loopback_targets
        ]
    return {key: value for key, value in record.items() if value is not None}


def _file_record(
    base: dict[str, Any],
    *,
    path: str,
    write: bool,
    resources: Any,
    cwd: str | None = None,
) -> dict[str, Any]:
    """One Read/Write record, the path resolved against the agent's cwd.

    A file tool names its target as the agent typed it, usually relative to
    the workspace; ``tests/x`` under ``/app`` is ``/app/tests/x``, not the
    verifier's ``/tests/x``.
    """

    resolved = _resolve_path(path, cwd) if cwd else None
    witnessed = resolved or path
    record = {
        **base,
        "event_type": "AgentWrite" if write else "AgentRead",
        "action_class": "Write" if write else "Read",
        "target_path" if write else "source_path": witnessed,
        "resource_class": classify_resource_class(witnessed, resources),
    }
    if cwd:
        record["cwd"] = cwd
    return {key: value for key, value in record.items() if value is not None}


def _network_record(
    base: dict[str, Any],
    *,
    arguments: dict[str, Any],
    network_class: str | None,
) -> dict[str, Any]:
    url = arguments.get("url")
    query = arguments.get("query") or arguments.get("q")
    record = {
        **base,
        "event_type": "AgentNetworkRequest",
        "action_class": "NetworkRequest",
        "resource": "network:web",
        "url": redacted_preview(url, limit=240) if isinstance(url, str) else None,
        "query": redacted_preview(query, limit=240) if isinstance(query, str) else None,
        "network_class": network_class,
    }
    return {key: value for key, value in record.items() if value is not None}


def _patch_targets(patch_text: str) -> list[str]:
    targets: list[str] = []
    for line in patch_text.splitlines():
        stripped = line.strip()
        path: str | None = None
        for prefix in ("*** Update File: ", "*** Add File: ", "*** Delete File: "):
            if stripped.startswith(prefix):
                path = stripped.removeprefix(prefix).strip()
                break
        if path is None and stripped.startswith("+++ "):
            candidate = stripped.removeprefix("+++ ").strip()
            if candidate.startswith("b/"):
                candidate = candidate[2:]
            if candidate not in {"/dev/null", ""}:
                path = candidate
        if path and path not in targets:
            targets.append(path)
    return targets


def _title_command_fallback(kind: str, event: dict[str, Any]) -> str | None:
    """The title as the command, for shims that put the command there.

    codex-acp and gemini-cli title an ``execute`` call with the raw command;
    openhands prefixes it with prose ("...: $ ls -la"). A "<kind> {json}"
    title is handled by the JSON fallback, and a bare kind label is not a
    command.
    """

    title = event.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    stripped = title.strip()
    if (
        stripped == kind
        or stripped.startswith(f"{kind} {{")
        or stripped.startswith("{")
    ):
        return None
    _, sep, command = stripped.partition(": $ ")
    if sep and command.strip():
        return command.strip()
    if stripped.startswith("$ ") and stripped[2:].strip():
        return stripped[2:].strip()
    return stripped


def _title_command_is_verbatim(event: dict[str, Any], command: str) -> bool:
    """Whether ``command`` is the whole title, with no ``$`` marker to vouch for it.

    A plan-step title is prose, and prose that reaches the decomposer makes an
    opaque exec whose scraped paths could fail a run closed for an agent that
    only mentioned a path. The record is still emitted; the marker lets the
    fail-closed opaque-reference rule abstain on input it cannot trust.
    """

    title = event.get("title")
    return isinstance(title, str) and title.strip() == command


_TITLE_PATH_RE = re.compile(
    r"(?:Editing|Reading|Creating|Writing|Viewing|Opening)\s+(/[^\s:]+?)\.?\s*$"
)


def _diff_content_paths(event: dict[str, Any]) -> list[str]:
    """Target paths from ACP ``diff`` content blocks (the file actually changed)."""

    paths: list[str] = []
    for block in event.get("content") or []:
        if not isinstance(block, dict) or block.get("type") != "diff":
            continue
        path = block.get("path")
        if isinstance(path, str) and path.strip() and path not in paths:
            paths.append(path.strip())
    return paths


def _title_path_fallback(event: dict[str, Any]) -> str | None:
    title = event.get("title")
    if not isinstance(title, str):
        return None
    match = _TITLE_PATH_RE.search(title.strip())
    return match.group(1) if match else None


def _arguments_from_event(event: dict[str, Any]) -> dict[str, Any]:
    """The tool call's arguments: ``arguments``, 0.8's ``raw_input``, then fallbacks.

    ``TaskRuntime.bash`` records its command at the top level of the event.
    Some shims send no arguments at all but embed them in the title as
    ``"<kind> {json}"``; a truncated or non-JSON title parses to nothing
    rather than to wrong arguments.
    """

    for key in ("arguments", "raw_input", "rawInput"):
        value = event.get(key)
        if isinstance(value, dict):
            return value
    top_level = {
        key: event[key]
        for key in (*_FALLBACK_COMMAND_KEYS, *_FALLBACK_PATH_KEYS, "url", "query")
        if isinstance(event.get(key), str)
    }
    if top_level:
        return top_level
    title = event.get("title")
    if isinstance(title, str):
        brace = title.find("{")
        if brace != -1:
            try:
                parsed, _ = json.JSONDecoder().raw_decode(title[brace:])
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return parsed
    return {}


def _fallback_command_key(kind: str, arguments: dict[str, Any]) -> str | None:
    if kind in _COMMAND_KINDS or kind in _WRITE_KINDS or kind in _READ_KINDS:
        return None
    if kind in _NETWORK_KINDS or kind in _IGNORED_KINDS or kind == "apply_patch":
        return None
    for key in _FALLBACK_COMMAND_KEYS:
        if isinstance(arguments.get(key), str):
            return key
    return None


def _ambiguous_read_path(kind: str, arguments: dict[str, Any]) -> str | None:
    """The path a ``read``/``search`` call read, or None to keep it a command.

    Only the path is witnessed, never a search pattern: synthesizing an
    operand from a pattern is how ``grep -r 'a/b' /app`` would invent
    ``/app/a/b``.
    """

    if kind not in _AMBIGUOUS_READ_KINDS:
        return None
    command_key = _COMMAND_KINDS.get(kind)
    if command_key:
        command = arguments.get(command_key)
        if isinstance(command, str) and command.strip():
            return None
    return _fallback_path(arguments)


def _fallback_path(arguments: dict[str, Any]) -> str | None:
    for key in _FALLBACK_PATH_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _timestamp_ms(value: Any) -> int | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return int(datetime.fromisoformat(text).timestamp() * 1000)
    except ValueError:
        return None


def _sha256_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"


__all__ = [
    "TrajectorySynthesis",
    "action_records_from_trajectory",
    "redacted_preview",
]
