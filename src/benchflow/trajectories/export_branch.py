"""Export branch trees as training rows (``bench train convert --format branch-tree``).

Each fork in a trial's ``tree.json`` is a sibling group: its children started
from the same checkpoint, so they share a *prefix* (the conversation up to
the fork) and differ in their *continuation*. One row per child
(``kind: branch_child``, ``schema_version`` 2) carries both as chat messages
ready for ``tokenizer.apply_chat_template(row["messages"], tools=row["tools"])``:
one ``assistant`` message per reply (``content``, ``reasoning_content``,
OpenAI-style ``tool_calls`` with the agent's tool name and JSON-string
arguments), then one ``tool`` message per call. It also carries the child's
reward, the fork's value V (mean reward of its children) and the advantage
``reward - V``. ``tools`` lists the tools the row calls, inferred from the
calls (the agents' real tool schemas and system prompts are not recorded).

With ``pairs_out``, one ``branch_pair`` row per sibling pair whose rewards
differ: ``prompt`` (the shared prefix plus the shared user turn), ``chosen``
and ``rejected`` (assistant and tool messages), rewards and margin. By
default only siblings that were asked the same thing are paired (their first
user message is identical); ``any_request_pairs=True`` keeps the others, with
the differing user turns inside ``chosen``/``rejected``.

Sources are the ACP events BenchFlow records for every agent (runs that can
branch have no LLM-proxy trajectory): the parent's
``trajectory/acp_trajectory.jsonl`` and each child's ``observation.json``.
A tree node's ``step_id`` is ``step-<i>-<type>``, the index into the
trajectory of the rollout that produced it, so the prefix of a fork is that
rollout's events up to the fork node; a nested fork adds the enclosing
child's events; a trial started from a kept checkpoint (``--from-checkpoint``)
starts with the source trial's events up to the checkpoint
(``checkpoint_source.json``; ``prefix_complete: false`` when they cannot be
found). ``session`` is the fork's ``agent_session``: ``resumed`` children
continued the parent's conversation (the prefix is context they remember),
``fresh`` children only see its effects in their files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from benchflow.trajectories.types import redact_trajectory_obj

_STEP_RE = re.compile(r"^step-(\d+)-")


@dataclass
class BranchExportStats:
    trials: int = 0
    forks: int = 0
    child_rows: int = 0
    pair_rows: int = 0
    unscored_children: int = 0
    missing_observations: int = 0
    skipped_oracle: int = 0
    below_min_reward: int = 0
    pairs_skipped_different_request: int = 0


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def events_to_messages(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """ACP events as chat messages (user, assistant, reasoning, tool calls)."""
    messages: list[dict[str, Any]] = []
    for event in events:
        kind = event.get("type")
        if kind == "user_message":
            messages.append({"role": "user", "content": event.get("text") or ""})
        elif kind == "agent_message":
            messages.append({"role": "assistant", "content": event.get("text") or ""})
        elif kind == "agent_thought":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning": event.get("text") or "",
                }
            )
        elif kind == "tool_call":
            call_id = event.get("tool_call_id")
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "name": event.get("title") or event.get("kind") or "tool",
                            "kind": event.get("kind"),
                            "arguments": event.get("raw_input"),
                        }
                    ],
                }
            )
            output = event.get("raw_output")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _text(output) if output is not None else "",
                }
            )
        elif kind == "oracle":
            messages.append(
                {
                    "role": "assistant",
                    "content": f"[oracle] {event.get('command')} exited "
                    f"{event.get('return_code')}",
                }
            )
    return messages


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _step_index(step_id: Any) -> int | None:
    match = _STEP_RE.match(step_id) if isinstance(step_id, str) else None
    return int(match.group(1)) if match else None


def _observation_events(trial: Path, child: dict[str, Any]) -> list[dict] | None:
    path = (child.get("artifacts") or {}).get("path")
    if not isinstance(path, str):
        return None
    target = (trial / path).resolve()
    if not target.is_relative_to(trial.resolve()):
        return None
    try:
        observation = json.loads((target / "observation.json").read_text())
    except (OSError, ValueError):
        return None
    events = observation.get("trajectory") if isinstance(observation, dict) else None
    return (
        [e for e in events if isinstance(e, dict)] if isinstance(events, list) else None
    )


_TOOL_KIND_NAMES = {"read": "Read", "fetch": "WebFetch", "think": "TodoWrite"}


def _infer_tool_name(event: dict[str, Any]) -> str:
    """Claude Code's tool name for a call recorded without one."""
    kind = str(event.get("kind") or "")
    raw = event.get("raw_input")
    args: dict[str, Any] = raw if isinstance(raw, dict) else {}
    if kind == "edit" or (
        "file_path" in args and ("content" in args or "old_string" in args)
    ):
        if "edits" in args:
            return "MultiEdit"
        return "Edit" if "old_string" in args else "Write"
    if kind == "execute" or "command" in args:
        return "Bash"
    if kind == "search" or "pattern" in args:
        return (
            "Grep"
            if any(k in args for k in ("output_mode", "glob", "type"))
            or not any(ch in str(args.get("pattern", "")) for ch in "*?")
            else "Glob"
        )
    if "url" in args:
        return "WebFetch"
    if "query" in args:
        return "WebSearch"
    return _TOOL_KIND_NAMES.get(kind) or kind or "tool"


def events_to_chat(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """ACP events as chat-template messages, the tools they call, and where
    the tool names came from (``agent``, ``inferred``, ``mixed`` or None)."""
    messages: list[dict[str, Any]] = []
    tools: dict[str, set[str]] = {}
    sources: set[str] = set()
    turn: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal turn, results
        if turn is not None:
            messages.append(turn)
            messages.extend(results)
        turn, results = None, []

    def assistant() -> dict[str, Any]:
        nonlocal turn
        if turn is None:
            turn = {"role": "assistant", "content": ""}
        return turn

    for event in events:
        kind = event.get("type")
        if kind == "user_message":
            flush()
            messages.append({"role": "user", "content": event.get("text") or ""})
        elif kind == "agent_thought":
            if turn is not None and (turn.get("tool_calls") or turn["content"]):
                flush()
            current = assistant()
            current["reasoning_content"] = current.get("reasoning_content", "") + (
                event.get("text") or ""
            )
        elif kind == "agent_message":
            if turn is not None and turn.get("tool_calls"):
                flush()
            current = assistant()
            current["content"] += event.get("text") or ""
        elif kind == "tool_call":
            name = event.get("tool_name")
            sources.add("agent" if name else "inferred")
            name = name or _infer_tool_name(event)
            args = event.get("raw_input")
            tools.setdefault(name, set()).update(args if isinstance(args, dict) else {})
            call_id = event.get("tool_call_id")
            assistant().setdefault("tool_calls", []).append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(
                            args if args is not None else {}, ensure_ascii=False
                        ),
                    },
                }
            )
            output = event.get("raw_output")
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _text(output) if output is not None else "",
                }
            )
        elif kind == "oracle":
            flush()
            messages.append(
                {
                    "role": "assistant",
                    "content": f"[oracle] {event.get('command')} exited "
                    f"{event.get('return_code')}",
                }
            )
    flush()
    tool_list = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "Inferred from the recorded calls; the agent's own "
                "tool schema is not recorded.",
                "parameters": {
                    "type": "object",
                    "properties": {key: {} for key in keys},
                },
            },
        }
        for name, keys in tools.items()
    ]
    source = (
        None if not sources else next(iter(sources)) if len(sources) == 1 else "mixed"
    )
    return messages, tool_list, source


def _merge_tools(*lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for tool_list in lists:
        for tool in tool_list:
            name = tool["function"]["name"]
            if name in merged:
                merged[name]["function"]["parameters"]["properties"].update(
                    tool["function"]["parameters"]["properties"]
                )
            else:
                merged[name] = json.loads(json.dumps(tool))
    return list(merged.values())


def _source_prefix(trial: Path) -> tuple[list[dict], dict[str, Any] | None, bool]:
    """Events of the source trial up to a kept checkpoint this trial started
    from: (events, prefix_source, complete)."""
    document = _read_json(trial / "checkpoint_source.json")
    if not isinstance(document, dict):
        return [], None, True
    events_n = document.get("prefix_events")
    source_path = document.get("trial_path")
    info = {
        "trial": document.get("trial"),
        "checkpoint": document.get("fork_id"),
        "events": events_n,
    }
    if not isinstance(events_n, int) or not isinstance(source_path, str):
        return [], info, False
    events = _read_jsonl(Path(source_path) / "trajectory" / "acp_trajectory.jsonl")
    if len(events) < events_n:
        return [], info, False
    return events[:events_n], info, True


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _first_user(messages: list[dict[str, Any]]) -> tuple[int, str | None]:
    for i, message in enumerate(messages):
        if message.get("role") == "user":
            return i, message.get("content")
    return -1, None


def _trial_rows(
    trial: Path, stats: BranchExportStats, *, include_oracle: bool, any_request: bool
) -> tuple[list[dict], list[dict]]:
    tree = json.loads((trial / "tree.json").read_text())
    if tree.get("kind") != "benchflow-branch-tree" or tree.get("schema_version") != 1:
        return [], []
    nodes = {n["id"]: n for n in tree.get("nodes", []) if isinstance(n, dict)}
    parent_events = _read_jsonl(trial / "trajectory" / "acp_trajectory.jsonl")
    result = _read_json(trial / "result.json") or {}
    task = result.get("task_name")
    agent = result.get("agent")
    model = result.get("model")
    oracle = agent == "oracle"
    source_events, prefix_source, prefix_complete = _source_prefix(trial)
    # node id of a child -> (its prefix events, its own events, depth)
    child_context: dict[str, tuple[list[dict], list[dict], int]] = {}
    child_rows: list[dict] = []
    pair_rows: list[dict] = []
    for fork in tree.get("forks", []):
        if not isinstance(fork, dict):
            continue
        stats.forks += 1
        owner = fork.get("rollout")
        parent_node = fork.get("parent_node")
        index = _step_index((nodes.get(parent_node) or {}).get("step_id"))
        cut = 0 if index is None else index + 1
        if owner in child_context:
            base, own, depth = child_context[owner]
            prefix_events = base + own[:cut]
            depth += 1
            parent_child = owner
        else:
            prefix_events = source_events + parent_events[:cut]
            depth = 1
            parent_child = None
        prefix, prefix_tools, prefix_names = events_to_chat(prefix_events)
        value = fork.get("value")
        session = (fork.get("snapshot") or {}).get("agent_session") or "fresh"
        members = []
        for child in fork.get("children", []):
            if not isinstance(child, dict):
                continue
            events = _observation_events(trial, child)
            if events is None:
                stats.missing_observations += 1
                events = []
            node_id = child.get("node_id")
            if isinstance(node_id, str):
                child_context[node_id] = (prefix_events, events, depth)
            if oracle and not include_oracle:
                stats.skipped_oracle += 1
                continue
            reward = child.get("reward")
            if reward is None:
                stats.unscored_children += 1
            intervention = child.get("intervention") or {}
            continuation, tools, names = events_to_chat(events)
            name_sources = {n for n in (prefix_names, names) if n}
            row = {
                "kind": "branch_child",
                "schema_version": 2,
                "task": task,
                "trial": trial.name,
                "agent": agent,
                "model": model,
                "fork_id": fork.get("id"),
                "fork_kind": fork.get("kind") or "fork",
                "parent_node": parent_node,
                "parent_child": parent_child,
                "depth": depth,
                "fork_status": fork.get("status"),
                "value": value,
                "value_stderr": fork.get("value_stderr"),
                "siblings": len(fork.get("children", [])),
                "child": {
                    "index": child.get("index"),
                    "node_id": node_id,
                    "label": intervention.get("label"),
                    "requested": intervention.get("requested"),
                    "status": child.get("status"),
                    "reward": reward,
                    "reward_source": child.get("reward_source"),
                    "path": (child.get("artifacts") or {}).get("path"),
                },
                "advantage": round(reward - value, 6)
                if isinstance(reward, int | float) and isinstance(value, int | float)
                else None,
                "session": session,
                "prefix_source": prefix_source,
                "prefix_complete": prefix_complete,
                "prefix": prefix,
                "continuation": continuation,
                "messages": prefix + continuation,
                "tools": _merge_tools(prefix_tools, tools),
                "tool_names": None
                if not name_sources
                else next(iter(name_sources))
                if len(name_sources) == 1
                else "mixed",
            }
            child_rows.append(row)
            members.append(row)
        scored = [m for m in members if isinstance(m["child"]["reward"], int | float)]
        for i, first in enumerate(scored):
            for second in scored[i + 1 :]:
                a, b = first["child"], second["child"]
                if a["reward"] == b["reward"]:
                    continue
                chosen, rejected = (
                    (first, second) if a["reward"] > b["reward"] else (second, first)
                )
                ci, c_user = _first_user(chosen["continuation"])
                ri, r_user = _first_user(rejected["continuation"])
                same = c_user is not None and c_user == r_user and ci == 0 and ri == 0
                if not same and not any_request:
                    stats.pairs_skipped_different_request += 1
                    continue
                prompt = prefix + (chosen["continuation"][:1] if same else [])
                pair_rows.append(
                    {
                        "kind": "branch_pair",
                        "schema_version": 2,
                        "task": task,
                        "trial": trial.name,
                        "agent": agent,
                        "model": model,
                        "fork_id": fork.get("id"),
                        "parent_node": parent_node,
                        "depth": depth,
                        "session": session,
                        "prefix_complete": prefix_complete,
                        "same_request": same,
                        "prompt": prompt,
                        "chosen": chosen["continuation"][1:]
                        if same
                        else chosen["continuation"],
                        "rejected": rejected["continuation"][1:]
                        if same
                        else rejected["continuation"],
                        "chosen_reward": a["reward"]
                        if a is chosen["child"]
                        else b["reward"],
                        "rejected_reward": b["reward"]
                        if a is chosen["child"]
                        else a["reward"],
                        "chosen_label": chosen["child"]["label"],
                        "rejected_label": rejected["child"]["label"],
                        "margin": round(
                            chosen["child"]["reward"] - rejected["child"]["reward"], 6
                        ),
                        "tools": _merge_tools(chosen["tools"], rejected["tools"]),
                    }
                )
    return child_rows, pair_rows


def export_branch_jsonl(
    jobs_dir: Path,
    out: Path,
    *,
    pairs_out: Path | None = None,
    redact: bool = True,
    min_reward: float | None = None,
    expected_rows: int | None = None,
    manifest: Path | None = None,
    include_oracle: bool = False,
    any_request_pairs: bool = False,
) -> BranchExportStats:
    """Write child rows (and optionally pair rows) for every tree under jobs_dir.

    ``min_reward`` keeps child rows whose reward is at least that (unscored
    rows are dropped by it); ``expected_rows`` fails, before writing, unless
    exactly that many child rows would be written; ``manifest`` writes the
    stats as JSON.
    """
    stats = BranchExportStats()
    trees = sorted(Path(jobs_dir).rglob("tree.json"))
    if not trees:
        raise ValueError(f"no tree.json under {jobs_dir}: nothing was branched")
    children: list[dict] = []
    pairs: list[dict] = []
    for tree in trees:
        try:
            rows, pair_rows = _trial_rows(
                tree.parent,
                stats,
                include_oracle=include_oracle,
                any_request=any_request_pairs,
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read {tree}: {exc}") from None
        stats.trials += 1
        children.extend(rows)
        pairs.extend(pair_rows)
    if min_reward is not None:
        kept = [
            row
            for row in children
            if isinstance(row["child"]["reward"], int | float)
            and row["child"]["reward"] >= min_reward
        ]
        stats.below_min_reward = len(children) - len(kept)
        children = kept
    if expected_rows is not None and len(children) != expected_rows:
        raise ValueError(
            f"branch-tree export would write {len(children)} child row(s), "
            f"expected {expected_rows}"
        )
    for path, rows in ((out, children), (pairs_out, pairs)):
        if path is None:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                if redact:
                    row = redact_trajectory_obj(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats.child_rows = len(children)
    stats.pair_rows = len(pairs) if pairs_out is not None else 0
    if manifest is not None:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(stats.__dict__, indent=2) + "\n")
    return stats


_ROLES = {"system", "user", "assistant", "tool"}


def _check_messages(where: str, messages: Any, tool_names: set[str] | None) -> int:
    """Check one message list; return how many tool calls it holds."""
    if not isinstance(messages, list):
        raise ValueError(f"{where}: must be a message list")
    calls: set[str] = set()
    n_calls = 0
    for i, item in enumerate(messages):
        message: dict[str, Any] = cast("dict[str, Any]", item)
        if not isinstance(item, dict) or message.get("role") not in _ROLES:
            raise ValueError(f"{where}[{i}]: role must be one of {sorted(_ROLES)}")
        for raw_call in message.get("tool_calls") or []:
            call: dict[str, Any] = raw_call if isinstance(raw_call, dict) else {}
            function = call.get("function")
            if (
                not call
                or call.get("type") != "function"
                or not isinstance(call.get("id"), str)
                or not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not isinstance(function.get("arguments"), str)
            ):
                raise ValueError(
                    f"{where}[{i}]: tool calls must be "
                    '{"id", "type": "function", "function": {"name", "arguments"}}'
                )
            try:
                json.loads(function["arguments"])
            except ValueError:
                raise ValueError(
                    f"{where}[{i}]: tool call arguments are not a JSON string"
                ) from None
            if tool_names is not None and function["name"] not in tool_names:
                raise ValueError(
                    f"{where}[{i}]: tool {function['name']!r} is not in tools"
                )
            calls.add(call["id"])
            n_calls += 1
        if message["role"] == "tool" and message.get("tool_call_id") not in calls:
            raise ValueError(
                f"{where}[{i}]: tool_call_id {message.get('tool_call_id')!r} "
                "answers no earlier tool call"
            )
    return n_calls


def _tool_names(where: str, tools: Any) -> set[str]:
    if not isinstance(tools, list):
        raise ValueError(f"{where}: tools must be a list")
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str):
            names.add(name)
    return names


def validate_branch_jsonl(
    path: Path, *, expected_rows: int | None = None
) -> dict[str, Any]:
    """Check branch-tree rows (``branch_child`` or ``branch_pair``, schema 2).

    Raises ``ValueError`` naming the first bad row; returns counts.
    """
    kinds: dict[str, int] = {}
    prefixes: dict[tuple, Any] = {}
    rows = with_calls = incomplete = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            where = f"row {line_no}"
            try:
                row = json.loads(line)
            except ValueError:
                raise ValueError(f"{where}: not JSON") from None
            kind = row.get("kind") if isinstance(row, dict) else None
            if kind not in {"branch_child", "branch_pair"}:
                raise ValueError(
                    f"{where}: kind must be branch_child or branch_pair, got {kind!r}"
                )
            if row.get("schema_version") != 2:
                raise ValueError(
                    f"{where}: schema_version must be 2 (re-export with this benchflow)"
                )
            names = _tool_names(where, row.get("tools"))
            if kind == "branch_child":
                prefix, continuation = row.get("prefix"), row.get("continuation")
                n = _check_messages(f"{where} prefix", prefix, names)
                n += _check_messages(f"{where} continuation", continuation, names)
                if row.get("messages") != prefix + continuation:
                    raise ValueError(f"{where}: messages must be prefix + continuation")
                _check_messages(f"{where} messages", row["messages"], names)
                key = (row.get("trial"), row.get("fork_id"))
                if prefixes.setdefault(key, prefix) != prefix:
                    raise ValueError(
                        f"{where}: prefix differs from its siblings' in fork "
                        f"{row.get('fork_id')}"
                    )
                reward = (row.get("child") or {}).get("reward")
                value, advantage = row.get("value"), row.get("advantage")
                expected = (
                    reward - value
                    if isinstance(reward, int | float)
                    and isinstance(value, int | float)
                    else None
                )
                if expected is not None and (
                    not isinstance(advantage, int | float)
                    or abs(advantage - expected) > 1e-6
                ):
                    raise ValueError(
                        f"{where}: advantage {advantage!r} != reward {reward} "
                        f"- value {value}"
                    )
                if row.get("prefix_complete") is False:
                    incomplete += 1
            else:
                prompt = row.get("prompt")
                if not isinstance(prompt, list) or not prompt:
                    raise ValueError(f"{where}: prompt must be a non-empty list")
                n = 0
                for side in ("chosen", "rejected"):
                    n += _check_messages(
                        f"{where} prompt+{side}", prompt + row.get(side, []), names
                    )
                if row.get("same_request") and prompt[-1].get("role") != "user":
                    raise ValueError(
                        f"{where}: a same-request pair's prompt must end with the "
                        "user turn"
                    )
                good, bad = row.get("chosen_reward"), row.get("rejected_reward")
                if not (
                    isinstance(good, int | float)
                    and isinstance(bad, int | float)
                    and good > bad
                ):
                    raise ValueError(
                        f"{where}: chosen_reward {good!r} must beat "
                        f"rejected_reward {bad!r}"
                    )
                if row.get("prefix_complete") is False:
                    incomplete += 1
                key = (row.get("trial"), row.get("fork_id"))
                prefixes.setdefault(key, None)
            rows += 1
            with_calls += 1 if n else 0
            kinds[kind] = kinds.get(kind, 0) + 1
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"{path}: {rows} row(s), expected {expected_rows}")
    return {
        "path": str(path),
        "rows": rows,
        "kinds": kinds,
        "forks": len(prefixes),
        "rows_with_tool_calls": with_calls,
        "prefix_incomplete": incomplete,
    }
