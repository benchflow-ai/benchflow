"""Read ATIF trajectories (ATIF-v1.0 to ATIF-v1.8) that another runner or a dataset wrote.

An ATIF (Agent Trajectory Interchange Format) trajectory is a list of steps
(``system``, ``user``, ``agent``); an agent step carries the model's message,
its tool calls, the observation that answered them and the step's metrics.
ATIF-v1.6 added image content parts, v1.7 embedded subagent trajectories and
per-step call counts, v1.8 audio parts. BenchFlow writes ATIF-v1.7
(:mod:`benchflow.trajectories.export_atif`); this module reads every 1.x
version up to 1.8 and turns a trajectory into what BenchFlow's own tools read:

- :func:`atif_conversation`: the conversation as chat messages, the agent's
  tool definitions, and what was left out (media parts, subagents, copied
  context);
- :func:`atif_exchanges`: one record per model call in the shape of
  BenchFlow's proxy log (``trajectory/llm_trajectory.jsonl``), so
  ``bench train convert`` reads a trajectory as it reads a proxied run;
- :func:`atif_totals`: steps, tool calls and the trajectory's own metrics.

Tool calls stay structured (OpenAI ``tool_calls``, answered by ``tool``
messages through ``source_call_id``) when the trajectory declares the agent's
``tool_definitions``: the model called functions. Without them the agent's
calls were parsed from its text (a terminal agent's keystrokes), so the
assistant turn is rendered as text, the format's own way: the message, then
one ``<tool_call>`` block per call. An observation result naming no call is
the environment's reply, one ``user`` message. Reasoning is left out, as
BenchFlow leaves it out of every training row. Steps copied from an earlier
trajectory (``is_copied_context``) and agent steps that made no model call
(``llm_call_count`` 0) stay in later prompts but are never a call of their
own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

SUPPORTED_VERSIONS = tuple(f"ATIF-v1.{minor}" for minor in range(9))
SOURCE = "atif"
#: Where a task.toml-package runner's trial keeps its agent's ATIF trajectory.
TRIAL_ATIF_RELPATH = "agent/trajectory.json"


class AtifError(ValueError):
    """Not an ATIF trajectory BenchFlow can read."""


@dataclass
class AtifConversation:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    structured: bool
    #: Indexes into ``messages`` of the assistant turns that were model calls.
    calls: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    dropped_media: int = 0
    subagents: int = 0
    copied_steps: int = 0


def load_atif(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """The trajectory at *source* (a path or a parsed document), checked."""
    if isinstance(source, dict):
        data = source
    else:
        try:
            data = json.loads(Path(source).read_text())
        except (OSError, ValueError) as exc:
            raise AtifError(f"{source}: not readable JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AtifError("an ATIF trajectory is a JSON object")
    version = data.get("schema_version")
    if version not in SUPPORTED_VERSIONS:
        raise AtifError(
            f"ATIF version {version!r} is not one BenchFlow reads "
            f"({SUPPORTED_VERSIONS[0]} to {SUPPORTED_VERSIONS[-1]})"
        )
    agent = data.get("agent")
    if not isinstance(agent, dict) or not isinstance(agent.get("name"), str):
        raise AtifError("agent.name is required")
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        raise AtifError("steps must hold at least one step")
    # Typing only: each step is still checked to be an object below.
    for number, step in enumerate(cast("list[dict[str, Any]]", steps), start=1):
        if not isinstance(step, dict) or step.get("step_id") != number:
            raise AtifError(f"step {number}: step_id must be {number}")
        source_ = step.get("source")
        if source_ not in ("system", "user", "agent"):
            raise AtifError(
                f"step {number}: source {source_!r} is not system, user or agent"
            )
        calls = step.get("tool_calls") or []
        if calls and source_ != "agent":
            raise AtifError(f"step {number}: only an agent step makes tool calls")
        ids = {call.get("tool_call_id") for call in calls if isinstance(call, dict)}
        for call in calls:
            if not (
                isinstance(call, dict)
                and isinstance(call.get("tool_call_id"), str)
                and isinstance(call.get("function_name"), str)
                and isinstance(call.get("arguments", {}), dict)
            ):
                raise AtifError(
                    f"step {number}: a tool call needs tool_call_id, function_name "
                    "and object arguments"
                )
        for result in (step.get("observation") or {}).get("results") or []:
            ref = result.get("source_call_id") if isinstance(result, dict) else None
            if ref is not None and ref not in ids:
                raise AtifError(
                    f"step {number}: observation answers {ref!r}, no tool call of this step"
                )
    return data


def _text(content: Any) -> tuple[str, int]:
    """A message's text and how many media parts it held (left out)."""
    if content is None:
        return "", 0
    if isinstance(content, str):
        return content, 0
    texts, media = [], 0
    for part in content if isinstance(content, list) else [content]:
        if isinstance(part, dict) and part.get("type") == "text":
            texts.append(str(part.get("text") or ""))
        else:
            media += 1
    return "\n".join(texts), media


def atif_conversation(trajectory: dict[str, Any]) -> AtifConversation:
    """The trajectory's steps as chat messages (see the module docstring)."""
    agent = trajectory["agent"]
    tools = [t for t in agent.get("tool_definitions") or [] if isinstance(t, dict)]
    out = AtifConversation(messages=[], tools=tools, structured=bool(tools))
    out.subagents = len(trajectory.get("subagent_trajectories") or [])
    default_model = agent.get("model_name")
    for step in trajectory["steps"]:
        text, media = _text(step.get("message"))
        out.dropped_media += media
        source = step["source"]
        if source in ("system", "user"):
            out.messages.append({"role": source, "content": text})
            continue
        calls = step.get("tool_calls") or []
        if out.structured:
            message: dict[str, Any] = {"role": "assistant", "content": text}
            if calls:
                message["tool_calls"] = [
                    {
                        "id": call["tool_call_id"],
                        "type": "function",
                        "function": {
                            "name": call["function_name"],
                            "arguments": json.dumps(
                                call.get("arguments") or {}, ensure_ascii=False
                            ),
                        },
                    }
                    for call in calls
                ]
        else:
            parts = [text] if text else []
            parts += [
                "<tool_call>\n"
                + json.dumps(
                    {
                        "name": call["function_name"],
                        "arguments": call.get("arguments") or {},
                    },
                    ensure_ascii=False,
                )
                + "\n</tool_call>"
                for call in calls
            ]
            message = {"role": "assistant", "content": "\n".join(parts)}
        if step.get("is_copied_context"):
            out.copied_steps += 1
        elif step.get("llm_call_count") != 0:
            out.calls.append(
                (
                    len(out.messages),
                    {
                        "model": step.get("model_name") or default_model,
                        "metrics": step.get("metrics") or {},
                        "step_id": step["step_id"],
                        "aggregated": (step.get("llm_call_count") or 1) > 1,
                    },
                )
            )
        out.messages.append(message)
        unnamed = []
        for result in (step.get("observation") or {}).get("results") or []:
            content, media = _text(result.get("content"))
            out.dropped_media += media
            if out.structured and result.get("source_call_id"):
                out.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result["source_call_id"],
                        "content": content,
                    }
                )
            else:
                unnamed.append(content)
        if unnamed:
            out.messages.append({"role": "user", "content": "\n".join(unnamed)})
    return out


def atif_exchanges(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """One proxy-log record per model call: the messages before the call as
    the request, the call's assistant turn as the response, its metrics as
    the usage."""
    conversation = atif_conversation(trajectory)
    exchanges = []
    for index, call in conversation.calls:
        reply = conversation.messages[index]
        metrics = call["metrics"]
        usage: dict[str, Any] = {}
        if isinstance(metrics.get("prompt_tokens"), int):
            usage["prompt_tokens"] = metrics["prompt_tokens"]
        if isinstance(metrics.get("completion_tokens"), int):
            usage["completion_tokens"] = metrics["completion_tokens"]
        if isinstance(metrics.get("cached_tokens"), int):
            usage["prompt_tokens_details"] = {"cached_tokens": metrics["cached_tokens"]}
        request: dict[str, Any] = {
            "model": call["model"],
            "messages": conversation.messages[:index],
        }
        if conversation.tools:
            request["tools"] = conversation.tools
        exchanges.append(
            {
                "request": {"body": request},
                "response": {
                    "status_code": 200,
                    "body": {
                        "model": call["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": reply,
                                "finish_reason": (
                                    "tool_calls" if reply.get("tool_calls") else "stop"
                                ),
                            }
                        ],
                        "usage": usage,
                    },
                },
                "metadata": {
                    "source": SOURCE,
                    "schema_version": trajectory["schema_version"],
                    "step_id": call["step_id"],
                    "call_purpose": "agent",
                    "aggregated_calls": call["aggregated"],
                    "cost_usd": metrics.get("cost_usd"),
                },
            }
        )
    return exchanges


def atif_totals(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Steps, agent steps, tool calls and the trajectory's final metrics."""
    steps = trajectory["steps"]
    final = trajectory.get("final_metrics") or {}
    return {
        "schema_version": trajectory["schema_version"],
        "steps": len(steps),
        "agent_steps": sum(1 for s in steps if s["source"] == "agent"),
        "tool_calls": sum(len(s.get("tool_calls") or []) for s in steps),
        "prompt_tokens": final.get("total_prompt_tokens"),
        "completion_tokens": final.get("total_completion_tokens"),
        "cached_tokens": final.get("total_cached_tokens"),
        "cost_usd": final.get("total_cost_usd"),
    }


def trial_atif(rollout_dir: str | Path) -> dict[str, Any] | None:
    """The ATIF trajectory a task.toml-package runner's trial kept, if any.

    BenchFlow's own ``trainer/atif.json`` is not read here: it is projected
    from ACP events and carries no tool arguments, so the trial's proxy log
    is the record to train from.
    """
    path = Path(rollout_dir) / TRIAL_ATIF_RELPATH
    if not path.is_file():
        return None
    try:
        return load_atif(path)
    except AtifError:
        return None
