"""Attribute captured LLM exchanges to the parent agent or to its subagents.

Claude Code runs each subagent (the ``Agent`` tool, ``Task`` in older
releases) as a separate conversation that shares the rollout's model gateway,
so the subagent's model calls land in ``trajectory/llm_trajectory.jsonl`` next
to the parent's with no marker. The request metadata does not separate them
either: the gateway capture records no client headers, and Claude Code sends
the same ``metadata.user_id`` session id for the parent and every subagent.
SFT conversion therefore has to recover the owner of each exchange from the
conversations themselves before it builds parent rows.

Evidence, strongest first:

1. **ACP linkage.** ``claude-agent-acp`` tags every child ACP event with
   ``_meta.claudeCode.parentToolUseId``, which BenchFlow records as
   ``parent_tool_call_id`` in ``acp_trajectory.jsonl``. ACP tool-call ids are
   the provider ``tool_use`` ids, so an exchange that emits or replays a tool
   call whose ACP event has a parent belongs to that subagent. Only positive
   linkage counts: captures made before the linkage existed record child tool
   calls with no parent at all.
2. **Spawn prompt.** A subagent conversation starts with a user message whose
   text, apart from ``<system-reminder>`` blocks, is exactly the ``prompt``
   argument of the parent's spawn tool call. A conversation never counts as the
   child of a spawn call it issued itself.
3. **Spawn tool declaration.** Claude Code offers the spawn tool only to the
   main agent (subagents cannot start subagents), so a conversation that
   declares it and is not a spawned prompt is the parent.
4. **No tools.** A conversation that declares no tools and matches none of the
   above is a nested helper call, such as the model call Claude Code makes
   inside its ``WebSearch`` tool. It is not part of any agent's conversation.

A tool-using conversation that none of these attribute, or whose evidence
conflicts, raises :class:`SubagentAttributionError` instead of guessing.
Rollouts with no spawn call and no ACP linkage are not attributed at all, so
other agents' conversion is unchanged.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from benchflow.trajectories.types import redact_trajectory_text

SPAWN_TOOL_NAMES = frozenset({"Agent", "Task"})

AgentRole = Literal["parent", "subagent", "helper"]

_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_PREVIEW_CHARS = 80


class SubagentAttributionError(ValueError):
    """Raised when captured exchanges cannot be attributed without mixing."""


@dataclass(frozen=True)
class SubagentSpawn:
    """One spawn tool call issued by an agent."""

    tool_call_id: str
    prompt: str
    subagent_type: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class ExchangeAttribution:
    role: AgentRole
    spawn: SubagentSpawn | None = None
    # Spawn ids a subagent exchange could belong to when identical prompts were
    # spawned more than once and no ACP linkage tells them apart.
    ambiguous_spawn_ids: tuple[str, ...] = ()

    @property
    def parent_tool_call_id(self) -> str | None:
        return self.spawn.tool_call_id if self.spawn is not None else None


@dataclass
class RolloutAttribution:
    """Owner of every successful exchange in one rollout."""

    spawns: dict[str, SubagentSpawn]
    by_exchange: dict[int, ExchangeAttribution]

    def indices(self, role: AgentRole) -> list[int]:
        return sorted(i for i, a in self.by_exchange.items() if a.role == role)

    def subagent_groups(self, *, source: str) -> list[tuple[SubagentSpawn, list[int]]]:
        """Subagent exchange indices grouped by spawn, in capture order.

        Raises when a group cannot be linked to a single spawn call, because
        emitting such rows would attach them to the wrong parent call.
        """
        groups: dict[str, list[int]] = {}
        spawns: dict[str, SubagentSpawn] = {}
        for index in self.indices("subagent"):
            attribution = self.by_exchange[index]
            if attribution.spawn is None:
                raise SubagentAttributionError(
                    f"{source}: exchange {index} matches several spawn calls "
                    f"with the same prompt ({', '.join(attribution.ambiguous_spawn_ids)}) "
                    "and no ACP linkage tells them apart; cannot link subagent rows"
                )
            spawn_id = attribution.spawn.tool_call_id
            groups.setdefault(spawn_id, []).append(index)
            spawns[spawn_id] = attribution.spawn
        return sorted(
            ((spawns[key], indices) for key, indices in groups.items()),
            key=lambda item: item[1][0],
        )


@dataclass
class SubagentConversionCounts:
    """Subagent bookkeeping shared by the SFT conversion reports."""

    rollouts_with_subagents: int = 0
    subagent_calls_seen: int = 0
    subagent_exchanges_seen: int = 0
    subagent_exchanges_excluded: int = 0
    subagent_rows_written: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "rollouts_with_subagents": self.rollouts_with_subagents,
            "subagent_calls_seen": self.subagent_calls_seen,
            "subagent_exchanges_seen": self.subagent_exchanges_seen,
            "subagent_exchanges_excluded": self.subagent_exchanges_excluded,
            "subagent_rows_written": self.subagent_rows_written,
        }


@dataclass
class _Thread:
    key: tuple[str, frozenset[str]]
    texts: frozenset[str]
    tool_names: frozenset[str]
    indices: list[int] = field(default_factory=list)
    tool_call_ids: set[str] = field(default_factory=set)


def subagent_row_tags(spawn: SubagentSpawn) -> dict[str, Any]:
    tags: dict[str, Any] = {
        "agent_role": "subagent",
        "parent_tool_call_id": spawn.tool_call_id,
        "subagent_type": spawn.subagent_type,
        "subagent_description": spawn.description,
    }
    return {key: value for key, value in tags.items() if value is not None}


def load_acp_parent_links(rollout_dir: Path) -> dict[str, str]:
    """Map ACP tool-call ids to the spawn call that owns them, if recorded."""
    for relpath in ("trajectory/acp_trajectory.jsonl", "agent/acp_trajectory.jsonl"):
        path = rollout_dir / relpath
        if not path.is_file():
            continue
        links: dict[str, str] = {}
        try:
            lines = path.read_text().splitlines()
        except OSError:
            return {}
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            call_id = event.get("tool_call_id")
            parent = event.get("parent_tool_call_id")
            if isinstance(call_id, str) and isinstance(parent, str) and parent:
                links[call_id] = parent
        return links
    return {}


def attribute_rollout_exchanges(
    exchanges: list[dict[str, Any]],
    successful: list[int],
    *,
    acp_parent_links: dict[str, str] | None = None,
    source: str = "",
) -> RolloutAttribution | None:
    """Attribute each successful exchange, or return ``None`` without subagents."""
    links = acp_parent_links or {}
    calls = [list(_exchange_tool_calls(exchange)) for exchange in exchanges]
    spawns: dict[str, SubagentSpawn] = {}
    for exchange_calls in calls:
        for call_id, name, arguments in exchange_calls:
            if name in SPAWN_TOOL_NAMES and call_id not in spawns:
                spawn = _spawn_from_arguments(call_id, arguments)
                if spawn is not None:
                    spawns[call_id] = spawn
    if not spawns and not links:
        return None

    threads: dict[tuple[str, frozenset[str]], _Thread] = {}
    exchange_ids: dict[int, set[str]] = {}
    for index in successful:
        body = _request_body(exchanges[index])
        joined, texts = _first_user_texts(body)
        tool_names = frozenset(_declared_tool_names(body))
        key = (joined, tool_names)
        thread = threads.setdefault(
            key, _Thread(key=key, texts=frozenset(texts), tool_names=tool_names)
        )
        thread.indices.append(index)
        ids = {call_id for call_id, _, _ in calls[index]}
        exchange_ids[index] = ids
        thread.tool_call_ids |= ids

    by_exchange: dict[int, ExchangeAttribution] = {}
    unattributed: list[_Thread] = []
    for thread in threads.values():
        matches = [
            spawn
            for spawn in spawns.values()
            if spawn.prompt in thread.texts
            and spawn.tool_call_id not in thread.tool_call_ids
        ]
        declares_spawn_tool = bool(thread.tool_names & SPAWN_TOOL_NAMES)
        fallback: ExchangeAttribution | None = None
        if matches and declares_spawn_tool:
            raise SubagentAttributionError(
                f"{source}: exchanges {thread.indices} declare the spawn tool, "
                "which only the parent agent has, but start with the prompt of "
                f"spawn call {matches[0].tool_call_id}; cannot attribute them"
            )
        if len(matches) == 1:
            fallback = ExchangeAttribution("subagent", spawn=matches[0])
        elif matches:
            fallback = ExchangeAttribution(
                "subagent",
                ambiguous_spawn_ids=tuple(s.tool_call_id for s in matches),
            )
        elif declares_spawn_tool:
            fallback = ExchangeAttribution("parent")
        elif not thread.tool_names:
            fallback = ExchangeAttribution("helper")

        for index in thread.indices:
            owners = {links[i] for i in exchange_ids[index] if i in links}
            if len(owners) > 1:
                raise SubagentAttributionError(
                    f"{source}: exchange {index} replays tool calls that ACP "
                    f"attributes to different subagents ({', '.join(sorted(owners))})"
                )
            if not owners:
                if fallback is not None:
                    by_exchange[index] = fallback
                continue
            owner = owners.pop()
            if declares_spawn_tool or (
                matches and owner not in {s.tool_call_id for s in matches}
            ):
                expected = (
                    "the parent agent"
                    if declares_spawn_tool
                    else "spawn call "
                    + ", ".join(sorted(s.tool_call_id for s in matches))
                )
                raise SubagentAttributionError(
                    f"{source}: ACP attributes exchange {index} to subagent "
                    f"{owner}, but its conversation belongs to {expected}"
                )
            by_exchange[index] = ExchangeAttribution(
                "subagent",
                spawn=spawns.get(owner, SubagentSpawn(tool_call_id=owner, prompt="")),
            )
        if any(index not in by_exchange for index in thread.indices):
            unattributed.append(thread)

    if unattributed:
        details = "; ".join(
            f"exchanges {[i for i in t.indices if i not in by_exchange]} "
            f"(first user message {_preview(t.key[0])!r}, "
            f"tools {sorted(t.tool_names)[:6]})"
            for t in unattributed
        )
        raise SubagentAttributionError(
            f"{source}: cannot attribute LLM exchanges to the parent agent or to "
            f"a subagent spawn: {details}. Refusing to mix them into parent rows."
        )
    if not any(a.role == "parent" for a in by_exchange.values()):
        raise SubagentAttributionError(
            f"{source}: no successful LLM exchange belongs to the parent agent"
        )
    return RolloutAttribution(spawns=spawns, by_exchange=by_exchange)


def _preview(text: str) -> str:
    text = " ".join(text.split())
    if len(text) > _PREVIEW_CHARS:
        text = text[:_PREVIEW_CHARS] + "..."
    return redact_trajectory_text(text)


def _as_dict(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _request_body(exchange: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(_as_dict(exchange.get("request")).get("body"))


def _response_body(exchange: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(_as_dict(exchange.get("response")).get("body"))


def _conversation(body: dict[str, Any]) -> list[Any]:
    messages = body.get("messages")
    if isinstance(messages, list):
        return messages
    raw_input = body.get("input")
    if isinstance(raw_input, list):
        return raw_input
    if isinstance(raw_input, str):
        return [{"role": "user", "content": raw_input}]
    return []


def _block_texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    texts: list[str] = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            texts.append(block["text"])
    return texts


def _first_user_texts(body: dict[str, Any]) -> tuple[str, set[str]]:
    """First user message text without system reminders, joined and per block."""
    for message in _conversation(body):
        if isinstance(message, dict) and message.get("role") == "user":
            blocks = [
                _REMINDER.sub("", text).strip()
                for text in _block_texts(message.get("content"))
            ]
            blocks = [text for text in blocks if text]
            joined = "\n".join(blocks)
            return joined, {*blocks, joined} - {""}
    return "", set()


def _declared_tool_names(body: dict[str, Any]) -> Iterator[str]:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else tool.get("name")
        if isinstance(name, str) and name:
            yield name


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    return cast(dict[str, Any], arguments) if isinstance(arguments, dict) else {}


def _spawn_from_arguments(
    call_id: str, arguments: dict[str, Any]
) -> SubagentSpawn | None:
    prompt = arguments.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    subagent_type = arguments.get("subagent_type")
    description = arguments.get("description")
    return SubagentSpawn(
        tool_call_id=call_id,
        prompt=prompt.strip(),
        subagent_type=subagent_type if isinstance(subagent_type, str) else None,
        description=description if isinstance(description, str) else None,
    )


def _openai_tool_calls(
    message: dict[str, Any],
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = _as_dict(call.get("function"))
        call_id = call.get("id") or call.get("tool_call_id")
        name = function.get("name") or call.get("name")
        if isinstance(call_id, str) and call_id and isinstance(name, str):
            arguments = function.get("arguments", call.get("arguments"))
            yield call_id, name, _parse_arguments(arguments)


def _message_tool_calls(message: Any) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Tool calls carried by one message in Anthropic, OpenAI or Responses shape."""
    if not isinstance(message, dict):
        return
    if message.get("type") in {"function_call", "tool_call"}:
        call_id = message.get("call_id") or message.get("id")
        name = message.get("name")
        if isinstance(call_id, str) and call_id and isinstance(name, str):
            yield call_id, name, _parse_arguments(message.get("arguments"))
        return
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and isinstance(block.get("id"), str)
                and isinstance(block.get("name"), str)
            ):
                yield block["id"], block["name"], _parse_arguments(block.get("input"))
    yield from _openai_tool_calls(message)


def _exchange_tool_calls(
    exchange: dict[str, Any],
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Tool calls replayed in the request history or emitted by the response."""
    for message in _conversation(_request_body(exchange)):
        if isinstance(message, dict) and message.get("role") not in {None, "assistant"}:
            continue
        yield from _message_tool_calls(message)
    response = _response_body(exchange)
    for choice in response.get("choices") or []:
        if isinstance(choice, dict):
            yield from _openai_tool_calls(_as_dict(choice.get("message")))
    yield from _message_tool_calls(response)
    for item in response.get("output") or []:
        yield from _message_tool_calls(item)
    if isinstance(response.get("message"), dict):
        yield from _openai_tool_calls(response["message"])


@dataclass
class RolloutExchangeSplit:
    """A rollout's successful exchanges, split by owner for row building."""

    parent: list[tuple[int, dict[str, Any]]]
    subagents: list[tuple[SubagentSpawn, list[tuple[int, dict[str, Any]]]]]
    helper_calls: int = 0
    subagent_exchanges: int = 0


def split_rollout_exchanges(
    rollout_dir: Path,
    exchanges: list[dict[str, Any]],
    successful: list[tuple[int, dict[str, Any]]],
    *,
    counts: SubagentConversionCounts,
    subagent_rows: bool,
) -> RolloutExchangeSplit:
    """Attribute *successful* exchanges and record the subagent counts.

    Without subagent activity every successful exchange stays a parent
    candidate, exactly as before attribution existed. Subagent groups are
    returned only when *subagent_rows* asks for them.
    """
    if not successful:
        return RolloutExchangeSplit(parent=[], subagents=[])
    source = str(rollout_dir / "trajectory" / "llm_trajectory.jsonl")
    attribution = attribute_rollout_exchanges(
        exchanges,
        [index for index, _ in successful],
        acp_parent_links=load_acp_parent_links(rollout_dir),
        source=source,
    )
    if attribution is None:
        return RolloutExchangeSplit(parent=list(successful), subagents=[])
    by_index = dict(successful)
    subagent_indices = attribution.indices("subagent")
    counts.rollouts_with_subagents += 1
    counts.subagent_calls_seen += len(attribution.spawns)
    counts.subagent_exchanges_seen += len(subagent_indices)
    groups = attribution.subagent_groups(source=source) if subagent_rows else []
    return RolloutExchangeSplit(
        parent=[(index, by_index[index]) for index in attribution.indices("parent")],
        subagents=[
            (spawn, [(index, by_index[index]) for index in indices])
            for spawn, indices in groups
        ],
        helper_calls=len(attribution.indices("helper")),
        subagent_exchanges=len(subagent_indices),
    )


# Roles whose steps may stay in a parent row; only the parent may end it.
_PARENT_ROLES = frozenset({"parent", "helper"})


def reject_results_row_subagent_steps(steps: Any, *, where: str) -> None:
    """Refuse a ``results.jsonl`` row whose trajectory mixes in subagent calls.

    The row's steps (and its top-level prompt/completion, taken from the final
    step) come from every captured exchange, subagents' included, and the row
    keeps neither the tools each call declared nor the ACP linkage, so the
    remaining steps cannot be proven to be the parent's. Converting the
    rollout or jobs directory attributes every call from llm_trajectory.jsonl.
    """
    if not isinstance(steps, list) or not steps:
        return
    roles = [
        step.get("extras", {}).get("agent_role")
        if isinstance(step, dict) and isinstance(step.get("extras"), dict)
        else None
        for step in steps
    ]
    if all(isinstance(role, str) for role in roles):
        # The writer attributed every call while it had the whole capture.
        foreign = [i for i, role in enumerate(roles) if role not in _PARENT_ROLES]
        if foreign or roles[-1] != "parent":
            shown = foreign[:10] or [len(roles) - 1]
            raise SubagentAttributionError(
                f"{where}: trajectory steps {shown} are not the parent agent's "
                f"calls (the writer attributed them to "
                f"{', '.join(sorted({str(roles[i]) for i in shown}))}), so the "
                "row's prompt and completion could be a subagent's. Convert the "
                "rollout or jobs directory instead."
            )
        return
    subagent_steps = results_row_subagent_steps(steps)
    if subagent_steps:
        raise SubagentAttributionError(
            f"{where}: {len(subagent_steps)} trajectory step(s) are Claude "
            f"subagent calls (steps {subagent_steps[:10]}); results.jsonl does "
            "not record which agent made each call, so they cannot be separated "
            "from the parent agent's calls. Convert the rollout or jobs "
            "directory instead."
        )


def results_row_subagent_steps(steps: list[Any]) -> list[int]:
    """Indices of ``results.jsonl`` trajectory steps that are subagent calls.

    ``results.jsonl`` steps keep each call's messages but not the tools it
    declared, so only the spawn-prompt evidence is available: a step whose
    first user message is the prompt of a spawn call issued by another step.
    This is positive evidence only. The results writer drops a spawning call
    that no later request consumed, so a capture that ends inside a subagent
    can leave that subagent's steps undetectable here.
    """
    typed = [
        cast(dict[str, Any], step) if isinstance(step, dict) else {} for step in steps
    ]
    step_calls: list[set[str]] = []
    spawns: dict[str, SubagentSpawn] = {}
    for step in typed:
        ids: set[str] = set()
        for field_name in ("prompt", "completion"):
            messages = step.get(field_name)
            for message in messages if isinstance(messages, list) else []:
                for call_id, name, arguments in _message_tool_calls(message):
                    ids.add(call_id)
                    if name in SPAWN_TOOL_NAMES and call_id not in spawns:
                        spawn = _spawn_from_arguments(call_id, arguments)
                        if spawn is not None:
                            spawns[call_id] = spawn
        step_calls.append(ids)
    if not spawns:
        return []
    subagent_steps: list[int] = []
    for index, step in enumerate(typed):
        prompt = step.get("prompt")
        _, texts = _first_user_texts({"messages": prompt})
        if any(
            spawn.prompt in texts and spawn.tool_call_id not in step_calls[index]
            for spawn in spawns.values()
        ):
            subagent_steps.append(index)
    return subagent_steps
