"""OpenRouter Ori built-in harness adapter.

Ori does not expose ACP, but ``ori code --output jsonl`` has the same useful
surface: a headless turn runner, normalized runtime events, stable session ids
for follow-up turns, and terminal token usage.  This module adapts that CLI to
BenchFlow's protocol-agnostic ``Agent`` / ``Session`` contracts.
"""

from __future__ import annotations

import asyncio
import base64
import json
import shlex
import tempfile
import uuid
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from benchflow.acp.types import StopReason
from benchflow.agents.protocol import AgentCapabilities, AskUserHandler
from benchflow.usage_tracking import USAGE_SOURCE_AGENT_NATIVE

ORI_BINARY = "/opt/benchflow/bin/ori"
ORI_VERSION = "0.12.0+68f9a36"

_ORI_GLOBAL_ORI_MD = f"""---
model: openrouter/auto
version: {ORI_VERSION}
---
"""
_ORI_GLOBAL_PACKAGE_JSON = """{
  "name": "benchflow-ori-runtime",
  "private": true,
  "type": "module"
}
"""
_ORI_TERMINAL_EVENTS = frozenset(
    {"turn.succeeded", "turn.failed", "session.succeeded", "session.failed"}
)
_ORI_TEXT_EVENTS = frozenset({"assistant.text.delta", "content.delta"})
_ORI_REASONING_EVENTS = frozenset({"reasoning.delta"})
_ORI_EFFORTS = frozenset(
    {"max", "xhigh", "high", "medium", "low", "minimal", "none"}
)


def _b64(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


def _result_detail(result: Any) -> str:
    stderr = str(getattr(result, "stderr", "") or "").strip()
    stdout = str(getattr(result, "stdout", "") or "").strip()
    return (stderr or stdout or "no diagnostics")[-2000:]


def _nonnegative_int(value: object) -> int:
    try:
        return max(int(str(value)), 0)
    except (TypeError, ValueError):
        return 0


def _json_text(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _tool_kind(name: str) -> str:
    lowered = name.lower()
    if lowered in {"bash", "shell", "terminal"}:
        return "bash"
    if lowered in {"read", "read_file"}:
        return "read"
    if lowered in {"write", "write_file", "edit", "apply_patch"}:
        return "write"
    if lowered in {"glob", "grep", "search"}:
        return "search"
    if lowered in {"browser", "web", "web_search", "web_fetch"}:
        return "browser"
    if "skill" in lowered:
        return "skill"
    return "other"


def _tool_title(name: str, tool_input: object) -> str:
    if not isinstance(tool_input, dict):
        return name
    values: dict[str, Any] = {str(key): value for key, value in tool_input.items()}
    for key in (
        "command",
        "path",
        "file_path",
        "pattern",
        "query",
        "prompt",
        "name",
    ):
        value = values.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:500]
    return name


def _content_blocks(value: object) -> list[dict[str, object]]:
    return [
        {
            "type": "content",
            "content": {"type": "text", "text": _json_text(value)},
        }
    ]


def _decode_jsonl(raw: str) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Ori emitted invalid JSONL on line {line_number}: {line[:300]}"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError(
                f"Ori emitted a non-object JSONL value on line {line_number}"
            )
        documents.append(value)
    return documents


class OriSession:
    """A persistent, multi-turn Ori coding session inside one task sandbox."""

    usage_source = USAGE_SOURCE_AGENT_NATIVE

    def __init__(
        self,
        sandbox: Any,
        *,
        agent_env: dict[str, str],
        cwd: str,
        exec_user: str | None,
        reasoning_effort: str | None,
        command_timeout: float,
        runtime_dir: str,
    ) -> None:
        self._sandbox = sandbox
        self._agent_env = dict(agent_env)
        self._cwd = cwd
        self._exec_user = exec_user
        self._reasoning_effort = self._normalize_effort(reasoning_effort)
        self._command_timeout = max(int(command_timeout), 1)
        self._runtime_dir = runtime_dir
        self._session_id: str | None = None
        self._steps: list[dict[str, Any]] = []
        self._tool_records: dict[str, dict[str, Any]] = {}
        self._tool_call_count = 0
        self._usage_totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_read_tokens": 0,
            "cached_write_tokens": 0,
            "thought_tokens": 0,
            "total_tokens": 0,
        }
        self._has_usage = False
        self._ask_user_handler: AskUserHandler | None = None
        self._current_exec: asyncio.Task[Any] | None = None
        self.on_change: Callable[[Any], None] | None = None

    @staticmethod
    def _normalize_effort(value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        normalized = str(value).strip().lower()
        if normalized not in _ORI_EFFORTS:
            accepted = ", ".join(sorted(_ORI_EFFORTS))
            raise ValueError(
                f"Ori reasoning effort {value!r} is unsupported; choose: {accepted}"
            )
        return normalized

    @property
    def steps(self) -> list[dict[str, Any]]:
        return self._steps

    @property
    def tool_call_count(self) -> int:
        """Cumulative tool calls, consumed by the session-factory drive loop."""
        return self._tool_call_count

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def latest_usage_totals(self) -> dict[str, int] | None:
        """Return cumulative trusted usage from Ori terminal events."""
        return dict(self._usage_totals) if self._has_usage else None

    def on_ask_user(self, handler: AskUserHandler) -> None:
        # Ori's headless JSONL surface does not currently expose a response
        # channel for elicitation events. Keep the handler so the Session
        # contract is honored and a future Ori responder can bind without an
        # API change; capabilities() intentionally advertises ask_user=False.
        self._ask_user_handler = handler

    async def cancel(self) -> None:
        task = self._current_exec
        if task is None or task.done():
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    def _notify_change(self) -> None:
        if self.on_change is not None:
            self.on_change(self)

    def _command(self, prompt_path: str, result_path: str) -> str:
        model = (
            self._agent_env.get("ORI_MODEL")
            or self._agent_env.get("BENCHFLOW_PROVIDER_MODEL")
            or "openrouter/auto"
        )
        args = [
            ORI_BINARY,
            "code",
            "--harness",
            "ori",
            "--model",
            model,
            "--approvals",
            "self-drive",
            "--output",
            "jsonl",
        ]
        if self._reasoning_effort is not None:
            args.extend(["--reasoning-effort", self._reasoning_effort])
        if self._session_id is not None:
            args.extend(["--session", self._session_id])
        args.extend(["--prompt-file", prompt_path])
        command = " ".join(shlex.quote(part) for part in args)
        return f"{command} > {shlex.quote(result_path)}"

    def _exec_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "cwd": self._cwd,
            "env": {
                **self._agent_env,
                "CI": "true",
                "ORI_TELEMETRY": "0",
            },
            "timeout_sec": self._command_timeout,
        }
        if self._exec_user is not None:
            kwargs["user"] = self._exec_user
        return kwargs

    async def prompt(self, text: str) -> StopReason:
        self._steps.append({"type": "user_message", "text": text})
        self._notify_change()

        turn_id = uuid.uuid4().hex
        prompt_path = f"{self._runtime_dir}/prompt-{turn_id}.txt"
        result_path = f"{self._runtime_dir}/result-{turn_id}.jsonl"
        with tempfile.TemporaryDirectory(prefix="benchflow-ori-") as host_tmp:
            host_tmp_path = Path(host_tmp)
            local_prompt = host_tmp_path / "prompt.txt"
            local_result = host_tmp_path / "result.jsonl"
            local_prompt.write_text(text, encoding="utf-8")
            await self._sandbox.upload_file(local_prompt, prompt_path)
            await self._make_prompt_private(prompt_path)

            self._current_exec = asyncio.create_task(
                self._sandbox.exec(self._command(prompt_path, result_path), **self._exec_kwargs())
            )
            try:
                result = await self._current_exec
            finally:
                self._current_exec = None

            raw = ""
            try:
                await self._sandbox.download_file(result_path, local_result)
                raw = local_result.read_text(encoding="utf-8")
            except Exception as exc:
                if getattr(result, "return_code", 1) == 0:
                    raise RuntimeError(
                        "Ori exited successfully without writing its JSONL result"
                    ) from exc

        documents = _decode_jsonl(raw) if raw else []
        result_document = self._record_documents(documents)
        self._notify_change()

        return_code = int(getattr(result, "return_code", 1))
        if return_code != 0:
            message = self._result_error(result_document) or _result_detail(result)
            raise RuntimeError(f"Ori code failed with exit code {return_code}: {message}")
        if result_document is None:
            raise RuntimeError("Ori JSONL stream ended without a terminal result line")
        if result_document.get("ok") is not True:
            message = self._result_error(result_document) or "unknown Ori failure"
            raise RuntimeError(f"Ori code failed: {message}")
        return StopReason.END_TURN

    async def _make_prompt_private(self, path: str) -> None:
        command = f"chmod 600 {shlex.quote(path)}"
        if self._exec_user is not None:
            owner = shlex.quote(self._exec_user)
            command = f"chown {owner}:{owner} {shlex.quote(path)} && {command}"
        result = await self._sandbox.exec(command, timeout_sec=30, user="root")
        if getattr(result, "return_code", 1) != 0:
            raise RuntimeError(f"Could not secure Ori prompt file: {_result_detail(result)}")

    @staticmethod
    def _result_error(document: dict[str, Any] | None) -> str:
        if not document:
            return ""
        error = document.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message
        if isinstance(error, str):
            return error
        return ""

    def _record_documents(
        self, documents: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        terminal: dict[str, Any] | None = None
        for document in documents:
            if document.get("kind") == "result":
                terminal = document
                session_id = document.get("sessionId")
                if isinstance(session_id, str) and session_id:
                    self._session_id = session_id
                self._steps.append({"type": "ori_result", "result": document})
                continue

            wrapper = document.get("event")
            if not isinstance(wrapper, dict):
                self._steps.append({"type": "ori_output", "output": document})
                continue
            wrapper_type = wrapper.get("type")
            if wrapper_type == "runtime.event" and isinstance(
                wrapper.get("event"), dict
            ):
                self._record_runtime_event(wrapper["event"])
            elif wrapper_type == "audit.event":
                self._steps.append({"type": "ori_audit", "event": wrapper})
            else:
                self._steps.append({"type": "ori_event", "event": wrapper})
        return terminal

    def _record_runtime_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        payload = event.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        session_id = event.get("sessionId") or payload.get("sessionId")
        if isinstance(session_id, str) and session_id:
            self._session_id = session_id

        if event_type in _ORI_TEXT_EVENTS:
            self._record_text_delta("agent_message", payload.get("delta"))
            return
        if event_type in _ORI_REASONING_EVENTS:
            self._record_text_delta("agent_thought", payload.get("delta"))
            return
        if event_type == "tool.started":
            self._start_tool(event, payload)
            return
        if event_type in {"tool.progress", "tool.succeeded", "tool.failed"}:
            self._update_tool(event_type, event, payload)
            return
        if event_type in _ORI_TERMINAL_EVENTS:
            self._record_usage(payload.get("usage"))
        self._steps.append({"type": "ori_event", "event": event})

    def _record_text_delta(self, step_type: str, delta: object) -> None:
        if not isinstance(delta, str) or not delta:
            return
        if self._steps and self._steps[-1].get("type") == step_type:
            self._steps[-1]["text"] = str(self._steps[-1].get("text", "")) + delta
        else:
            self._steps.append({"type": step_type, "text": delta})

    def _start_tool(self, event: dict[str, Any], payload: dict[str, Any]) -> None:
        name = str(payload.get("name") or "tool")
        tool_input = payload.get("input")
        tool_call_id = str(
            payload.get("toolCallId")
            or f"ori-tool-{self._tool_call_count + 1}"
        )
        record = {
            "type": "tool_call",
            "tool_call_id": tool_call_id,
            "kind": _tool_kind(name),
            "title": _tool_title(name, tool_input),
            "status": "in_progress",
            "content": _content_blocks(tool_input) if tool_input is not None else [],
            "ori_events": [event],
        }
        self._tool_records[tool_call_id] = record
        self._tool_call_count += 1
        self._steps.append(record)

    def _update_tool(
        self,
        event_type: str,
        event: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        tool_call_id = str(
            payload.get("toolCallId")
            or f"ori-tool-{self._tool_call_count + 1}"
        )
        record = self._tool_records.get(tool_call_id)
        if record is None:
            self._start_tool(event, {**payload, "toolCallId": tool_call_id})
            record = self._tool_records[tool_call_id]
        else:
            record["ori_events"].append(event)

        if event_type == "tool.succeeded":
            record["status"] = "completed"
        elif event_type == "tool.failed":
            record["status"] = "failed"
        else:
            record["status"] = "in_progress"
        output = payload.get("result")
        if output is None:
            output = payload.get("partialResult")
        if output is not None:
            record["content"] = _content_blocks(output)

    def _record_usage(self, usage: object) -> None:
        if not isinstance(usage, dict):
            return
        values: dict[str, Any] = {str(key): value for key, value in usage.items()}
        input_tokens = _nonnegative_int(values.get("inputTokens"))
        output_tokens = _nonnegative_int(values.get("outputTokens"))
        cached_read = _nonnegative_int(values.get("cacheReadTokens"))
        cached_write = _nonnegative_int(values.get("cacheCreationTokens"))
        self._usage_totals["input_tokens"] += input_tokens
        self._usage_totals["output_tokens"] += output_tokens
        self._usage_totals["cached_read_tokens"] += cached_read
        self._usage_totals["cached_write_tokens"] += cached_write
        # Ori's contextTokens is the final request's context size, whereas
        # input/output are turn totals across every tool-loop model call.
        self._usage_totals["total_tokens"] += input_tokens + output_tokens
        self._has_usage = True


class OriAgent:
    """Factory for the native Ori JSONL session adapter."""

    def __init__(self, *, exec_user: str | None = None) -> None:
        self._exec_user = exec_user

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            protocol="ori-jsonl",
            nudges=True,
            ask_user=False,
            token_logprobs=False,
        )

    async def connect(self, sandbox: Any, role: str) -> OriSession:
        del role
        cwd = sandbox.agent_cwd or sandbox.agent_env.get("BENCHFLOW_AGENT_CWD")
        if not cwd:
            raise ValueError("Ori requires the resolved BenchFlow agent workspace")
        await self._ensure_global_workspace(sandbox)

        runtime_dir = f"/tmp/benchflow-ori-{uuid.uuid4().hex}"
        kwargs: dict[str, object] = {"timeout_sec": 30}
        if self._exec_user is not None:
            kwargs["user"] = self._exec_user
        result = await sandbox.exec(
            f"mkdir -p {shlex.quote(runtime_dir)} && chmod 700 {shlex.quote(runtime_dir)}",
            **kwargs,
        )
        if getattr(result, "return_code", 1) != 0:
            raise RuntimeError(
                f"Could not prepare Ori runtime directory: {_result_detail(result)}"
            )

        return OriSession(
            sandbox,
            agent_env=sandbox.agent_env,
            cwd=cwd,
            exec_user=self._exec_user,
            reasoning_effort=getattr(sandbox, "reasoning_effort", None),
            command_timeout=getattr(sandbox, "prompt_timeout", 3600),
            runtime_dir=runtime_dir,
        )

    async def _ensure_global_workspace(self, sandbox: Any) -> None:
        home = f"/home/{self._exec_user}" if self._exec_user else "/root"
        global_root = f"{home}/.ori/global"
        ori_md = f"{global_root}/ori.md"
        package_json = f"{global_root}/package.json"
        command = (
            f"if [ ! -f {shlex.quote(ori_md)} ]; then "
            f"mkdir -p {shlex.quote(f'{global_root}/features')} && "
            f"printf '%s' {shlex.quote(_b64(_ORI_GLOBAL_ORI_MD))} | base64 -d "
            f"> {shlex.quote(ori_md)} && "
            f"printf '%s' {shlex.quote(_b64(_ORI_GLOBAL_PACKAGE_JSON))} | base64 -d "
            f"> {shlex.quote(package_json)}; "
            "fi"
        )
        kwargs: dict[str, object] = {"timeout_sec": 30}
        if self._exec_user is not None:
            kwargs["user"] = self._exec_user
        result = await sandbox.exec(command, **kwargs)
        if getattr(result, "return_code", 1) != 0:
            raise RuntimeError(
                f"Could not prepare Ori global workspace: {_result_detail(result)}"
            )


def build_ori_agent(*, exec_user: str | None = None) -> OriAgent:
    """Session-factory entrypoint declared by the built-in agent registry."""
    return OriAgent(exec_user=exec_user)


__all__ = [
    "ORI_BINARY",
    "ORI_VERSION",
    "OriAgent",
    "OriSession",
    "build_ori_agent",
]
