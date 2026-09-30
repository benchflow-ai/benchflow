"""Codex in exec mode: its command builder and ``--json`` event parser.

The harness runs the pinned ``codex`` CLI as::

    codex exec [resume <thread-id>] --json --ignore-user-config
               --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox
               -c <key>=<value> ... -

with the prompt on stdin. Every setting comes from the run's CODEX_CONFIG as
``-c`` overrides (model provider, model, effort, web-search policy), and
``--ignore-user-config`` keeps a config.toml already in the sandbox out of the
run: the model provider is BenchFlow's gateway or nothing. Codex's approval
prompts and its own sandbox are off, as ``codex-acp`` runs with
``INITIAL_AGENT_MODE=agent-full-access``: BenchFlow's non-root sandbox user
and uid firewall are the isolation.

``codex exec --json`` (``codex-rs/exec/src/exec_events.rs``) reports a thread,
then per turn ``item.*`` events for messages, reasoning, commands, file
changes, MCP calls and web searches, and ``turn.completed`` (with usage) or
``turn.failed``. Items arrive whole: there are no text deltas, so a long
reasoning step shows no activity until it ends (see the harness docs).
Item ids restart at ``item_0`` in every process, so each turn's ids are
prefixed with the turn number.
"""

from __future__ import annotations

import json
from typing import Any

from benchflow.acp.types import McpServerSpec, StopReason
from benchflow.agents.codex_config import codex_config_overrides
from benchflow.native_harness.spec import (
    NativeLaunch,
    NativeTurn,
    NativeTurnOutcome,
)

CLI = "codex"
# model_reasoning_effort values Codex accepts.
CODEX_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})

_FLAGS = (
    "--json",
    "--ignore-user-config",
    "--skip-git-repo-check",
    "--dangerously-bypass-approvals-and-sandbox",
)
# Settings that keep Codex off the network apart from its model provider.
# Measured on 0.156.1 through a logging HTTPS proxy: at startup Codex opens
# github.com, api.github.com and chatgpt.com (twice) for its plugin
# marketplace (gone with features.plugins=false) and ab.chatgpt.com for
# analytics (gone with analytics.enabled=false). Neither changes what the model
# is offered. Feedback and the update check are off for the same reason.
_OFFLINE_OVERRIDES = (
    "features.plugins=false",
    "analytics.enabled=false",
    "feedback.enabled=false",
    "check_for_update_on_startup=false",
)


def codex_mcp_overrides(servers: tuple[McpServerSpec, ...]) -> dict[str, Any]:
    """Task MCP servers as Codex's ``mcp_servers`` config table."""
    table: dict[str, Any] = {}
    for spec in servers:
        entry: dict[str, Any]
        if spec.type == "stdio":
            entry = {"command": spec.command, "args": list(spec.args)}
            if spec.env:
                entry["env"] = dict(spec.env)
            if spec.cwd:
                entry["cwd"] = spec.cwd
        else:
            entry = {"url": spec.url}
            if spec.headers:
                entry["http_headers"] = dict(spec.headers)
        if spec.tools is not None:
            entry["enabled_tools"] = list(spec.tools)
        table[spec.name] = entry
    return {"mcp_servers": table} if table else {}


def codex_launch(turn: NativeTurn, codex_config: dict[str, Any]) -> NativeLaunch:
    """Codex's arguments for one turn (the prompt goes on stdin as ``-``)."""
    config = dict(codex_config)
    if turn.model:
        config["model"] = turn.model
    if turn.reasoning_effort:
        config["model_reasoning_effort"] = turn.reasoning_effort
    config.update(codex_mcp_overrides(turn.mcp_servers))
    provider = _provider(config)
    if provider is not None and isinstance(provider.get("base_url"), str):
        # A thread that names the built-in provider reaches the gateway too
        # (codex_home_config does the same for codex-acp).
        config.setdefault("openai_base_url", provider["base_url"])
    argv: list[str] = ["exec"]
    if turn.resume_id:
        argv += ["resume", turn.resume_id]
    argv += list(_FLAGS)
    for override in (*_OFFLINE_OVERRIDES, *codex_config_overrides(config)):
        argv += ["-c", override]
    argv.append("-")
    return NativeLaunch(tuple(argv))


def _provider(config: dict[str, Any]) -> dict[str, Any] | None:
    providers = config.get("model_providers")
    provider_id = config.get("model_provider")
    if isinstance(providers, dict) and isinstance(provider_id, str):
        provider = providers.get(provider_id)
        return provider if isinstance(provider, dict) else None
    return None


def codex_usage(usage: Any) -> dict[str, int] | None:
    """A ``turn.completed`` usage block in ACP ``PromptResponse.usage`` fields.

    OpenAI's ``input_tokens`` includes the cached ones, as the proxy's
    provider usage counts them.
    """
    if not isinstance(usage, dict):
        return None

    def count(key: str) -> int:
        value = usage.get(key)
        return value if isinstance(value, int) and value >= 0 else 0

    snapshot = {
        "input_tokens": count("input_tokens"),
        "output_tokens": count("output_tokens"),
        "cached_read_tokens": count("cached_input_tokens"),
        "cached_write_tokens": count("cache_write_input_tokens"),
        "thought_tokens": count("reasoning_output_tokens"),
    }
    snapshot["total_tokens"] = snapshot["input_tokens"] + snapshot["output_tokens"]
    return snapshot


def _console(output: str) -> list[dict[str, Any]]:
    if not output.strip():
        return []
    text = f"```console\n{output.rstrip()}\n```"
    return [{"type": "content", "content": {"type": "text", "text": text}}]


_TERMINAL = {"completed": "completed", "failed": "failed", "declined": "failed"}


class CodexExecParser:
    """One ``codex exec --json`` turn, as ACP session updates."""

    def __init__(self, cwd: str | None = None, *, turn: int = 1) -> None:
        self._cwd = cwd
        self._prefix = f"turn{turn}-"
        self._session_id: str | None = None
        self._started: set[str] = set()
        self._completed = False
        self._usage: dict[str, int] | None = None
        self._failure: str | None = None
        self._errors: list[str] = []
        self.warnings: list[str] = []

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def feed(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        kind = event.get("type")
        if kind == "thread.started":
            thread = event.get("thread_id")
            if isinstance(thread, str) and thread:
                self._session_id = thread
            return []
        if kind == "turn.completed":
            self._completed = True
            # The thread's running total (resumed turns included), not the
            # turn's own usage: exec reports its last token-usage update.
            self._usage = codex_usage(event.get("usage"))
            return []
        if kind == "turn.failed":
            error = event.get("error")
            message = error.get("message") if isinstance(error, dict) else None
            self._failure = str(message or "turn failed")
            return []
        if kind == "error":
            self._errors.append(str(event.get("message") or "error"))
            return []
        if kind in ("item.started", "item.updated", "item.completed"):
            item = event.get("item")
            if isinstance(item, dict):
                return self._item(item, done=kind == "item.completed")
        return []

    def _item(self, item: dict[str, Any], *, done: bool) -> list[dict[str, Any]]:
        kind = item.get("type")
        raw_id = item.get("id")
        if not isinstance(raw_id, str):
            return []
        call_id = self._prefix + raw_id
        if kind == "agent_message":
            text = item.get("text")
            return (
                [
                    {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": text},
                    }
                ]
                if done and isinstance(text, str) and text
                else []
            )
        if kind == "reasoning":
            text = item.get("text")
            return (
                [
                    {
                        "sessionUpdate": "agent_thought_chunk",
                        "content": {"type": "text", "text": text},
                    }
                ]
                if done and isinstance(text, str) and text
                else []
            )
        if kind == "error":
            if done and isinstance(item.get("message"), str):
                self.warnings.append(item["message"])
            return []
        if kind == "todo_list":
            # codex-acp sends the to-do list as an ACP plan, which BenchFlow's
            # ACP session does not record.
            return []
        call = self._tool_call(kind, item)
        if call is None:
            return []
        title, tool_kind, raw_input, raw_output, content, status = call
        if not done:
            if call_id in self._started:
                return []
            self._started.add(call_id)
            return [
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": call_id,
                    "title": title,
                    "kind": tool_kind,
                    "status": "in_progress",
                    "rawInput": raw_input,
                }
            ]
        update: dict[str, Any] = {
            "toolCallId": call_id,
            "status": status,
            "rawOutput": raw_output,
            "content": content,
        }
        if call_id in self._started:
            return [{"sessionUpdate": "tool_call_update", **update}]
        # Codex reports file changes and web searches only when they are done.
        self._started.add(call_id)
        return [
            {
                "sessionUpdate": "tool_call",
                "title": title,
                "kind": tool_kind,
                "rawInput": raw_input,
                **update,
            }
        ]

    @staticmethod
    def _tool_call(
        kind: Any, item: dict[str, Any]
    ) -> tuple[str, str, Any, Any, list[dict[str, Any]], str] | None:
        """(title, kind, raw input, raw output, content, terminal status)."""
        if kind == "command_execution":
            command = str(item.get("command") or "")
            output = str(item.get("aggregated_output") or "")
            return (
                command or "Terminal",
                "execute",
                {"command": command},
                {"output": output, "exit_code": item.get("exit_code")},
                _console(output),
                _TERMINAL.get(str(item.get("status")), "completed"),
            )
        if kind == "file_change":
            changes = [c for c in item.get("changes") or [] if isinstance(c, dict)]
            paths = [str(c.get("path")) for c in changes]
            return (
                f"Edit {', '.join(paths)}" if paths else "Edit",
                "edit",
                {"changes": changes},
                None,
                [
                    {
                        "type": "content",
                        "content": {
                            "type": "text",
                            "text": f"{c.get('kind')} {c.get('path')}",
                        },
                    }
                    for c in changes
                ],
                _TERMINAL.get(str(item.get("status")), "completed"),
            )
        if kind == "mcp_tool_call":
            result = item.get("result")
            error = item.get("error")
            blocks = result.get("content") if isinstance(result, dict) else None
            content = [
                {"type": "content", "content": block}
                for block in blocks or []
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if isinstance(error, dict) and error.get("message"):
                content.append(
                    {
                        "type": "content",
                        "content": {"type": "text", "text": str(error["message"])},
                    }
                )
            return (
                f"{item.get('server')}: {item.get('tool')}",
                "other",
                item.get("arguments"),
                result if result is not None else error,
                content,
                _TERMINAL.get(str(item.get("status")), "completed"),
            )
        if kind == "web_search":
            query = str(item.get("query") or "")
            return (
                f'Search "{query}"' if query else "Web search",
                "search",
                {"query": query, "action": item.get("action")},
                item.get("results"),
                [],
                "completed",
            )
        if kind == "collab_tool_call":
            return (
                str(item.get("tool") or "collab"),
                "think",
                {
                    "prompt": item.get("prompt"),
                    "receiver_thread_ids": item.get("receiver_thread_ids"),
                },
                item.get("agents_states"),
                [],
                _TERMINAL.get(str(item.get("status")), "completed"),
            )
        return None

    def outcome(self) -> NativeTurnOutcome:
        outcome = NativeTurnOutcome(
            session_id=self._session_id, usage_total=self._usage
        )
        if self._failure is not None:
            outcome.error = self._failure
            outcome.completed = True
        elif self._completed:
            outcome.stop_reason = StopReason.END_TURN
            outcome.completed = True
        elif self._errors:
            outcome.error = self._errors[-1]
        return outcome


def codex_config_from_env(agent_env: dict[str, str]) -> dict[str, Any]:
    """The run's CODEX_CONFIG object (empty when unset)."""
    raw = agent_env.get("CODEX_CONFIG")
    if not raw:
        return {}
    config = json.loads(raw)
    if not isinstance(config, dict):
        raise ValueError("CODEX_CONFIG must decode to a JSON object")
    return config
