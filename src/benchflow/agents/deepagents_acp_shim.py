#!/usr/bin/env python3
"""ACP shim for deepagents — wraps LangChain's `create_deep_agent` as an ACP server.

deepagents (https://github.com/langchain-ai/deepagents) is a LangGraph-based
harness that runs its OWN agent loop (planning, sub-agents, a filesystem
backend, and shell tools). Because the loop lives inside the library, the agent
is integrated via BenchFlow's ACP-shim path: this shim speaks ACP on stdio and,
on each prompt, builds and invokes an in-process deep agent, streaming its steps
back as ACP session/update notifications.

Architecture:
  benchflow ACP client ←stdio→ this shim ←in-process→ deepagents create_deep_agent
                                          ←LocalShellBackend→ <task cwd> (FS + shell)

The shim:
1. Installs `deepagents` + `langchain-openai` into an isolated venv at install
   time (see registry.py); this module is uploaded and run inside the sandbox.
2. On session/prompt, resolves the deepseek-v4-pro chat model from the
   OpenAI-compatible provider env (BENCHFLOW_PROVIDER_* with DEEPSEEK_* fallback),
   roots a LocalShellBackend at the task workspace, builds the deep agent, and
   invokes it on the instruction.
3. Emits ACP session/update notifications for agent text and tool calls so
   BenchFlow can capture the trajectory, then returns stopReason and waits for
   the next prompt.

Self-contained: like the other shims (openclaw, harvey-lab), this file is read
and uploaded into the sandbox with NO benchflow installed, so it must NOT
`import benchflow`. It depends only on the stdlib plus the deepagents /
langchain packages installed into the shim's venv.
"""

import json
import os
import sys
import time
import uuid
from collections.abc import Mapping

_DIAG_TRUNCATE = 2000  # max chars for diagnostic/text output in ACP updates
_TOOL_RESULT_TRUNCATE = 1000  # max chars for tool result text
_TOOL_INPUT_TRUNCATE = 500  # max chars for tool input echoed in ACP updates
_DEFAULT_MODEL = "deepseek-v4-pro"
# deepseek's public OpenAI-compatible endpoint; used only when neither the SDK
# (BENCHFLOW_PROVIDER_BASE_URL) nor the provider env (DEEPSEEK_BASE_URL) supply
# one. Kept as a last-resort default so a misconfigured run fails loudly at the
# model call rather than silently with no base_url.
_DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
# Recursion budget for the LangGraph loop — deepagents drives many tool turns,
# so the LangChain default (25) is far too small for real tasks.
_RECURSION_LIMIT = 1000


# ── ACP stdio I/O ─────────────────────────────────────────────────────────────


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def recv():
    while True:
        line = sys.stdin.readline()
        if not line:
            raise EOFError("stdin closed")
        line = line.strip()
        if not line:
            continue
        return json.loads(line)


# ── Provider / model resolution ───────────────────────────────────────────────
#
# deepseek-v4-pro is an OpenAI-compatible model. BenchFlow injects provider
# config two ways and this shim accepts either, in priority order:
#   1. BENCHFLOW_PROVIDER_BASE_URL / BENCHFLOW_PROVIDER_API_KEY — set by the SDK
#      when the run uses the `deepseek/` provider prefix (resolve_provider_env).
#   2. DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY — the provider-native vars, copied
#      into the sandbox unconditionally by auto_inherit_env().
# These are pure functions of the environment so they are unit-testable without
# a live model.


def resolve_base_url(env: Mapping[str, str] | None = None) -> str:
    """Resolve the OpenAI-compatible base URL for the deepseek provider."""
    env = os.environ if env is None else env
    for key in ("BENCHFLOW_PROVIDER_BASE_URL", "DEEPSEEK_BASE_URL"):
        val = env.get(key)
        if val and val.strip():
            return val.strip()
    return _DEFAULT_DEEPSEEK_BASE_URL


def resolve_api_key(env: Mapping[str, str] | None = None) -> str:
    """Resolve the API key for the deepseek provider (may be empty)."""
    env = os.environ if env is None else env
    for key in ("BENCHFLOW_PROVIDER_API_KEY", "DEEPSEEK_API_KEY"):
        val = env.get(key)
        if val and val.strip():
            return val.strip()
    return ""


def resolve_model_id(requested: str, env: Mapping[str, str] | None = None) -> str:
    """Resolve the bare deepseek model id to hand to the chat model.

    Priority: the ACP-supplied model (``set_model``) → BENCHFLOW_PROVIDER_MODEL
    (the SDK's prefix-stripped model) → the built-in default. Any leftover
    ``provider/`` prefix is stripped so the OpenAI-compatible endpoint receives
    the bare model name it expects.
    """
    env = os.environ if env is None else env
    model = (requested or "").strip()
    if not model:
        model = (env.get("BENCHFLOW_PROVIDER_MODEL") or "").strip()
    if not model:
        model = _DEFAULT_MODEL
    # Strip a single leading provider segment (e.g. "deepseek/deepseek-v4-pro").
    # Keep nested ids intact otherwise — deepseek model ids carry no inner slash,
    # but this mirrors strip_provider_prefix's "drop only the first segment" rule.
    if "/" in model:
        model = model.split("/", 1)[1]
    return model


def build_chat_model(model_id: str, env: Mapping[str, str] | None = None):
    """Build a LangChain chat model for deepseek-v4-pro via the OpenAI protocol.

    deepseek-v4-pro is OpenAI-compatible, so we use ``langchain_openai.ChatOpenAI``
    pointed at the deepseek base_url with the resolved api key. Optional
    generation params (temperature / top_p / max_tokens) come from the
    BENCHFLOW_MODEL_* env vars BenchFlow sets per run.

    Imported lazily so the module parses (and its pure helpers are testable)
    even when langchain_openai is not installed in the host dev venv — it is
    only present inside the shim's sandbox venv.
    """
    from langchain_openai import ChatOpenAI

    env = os.environ if env is None else env
    base_url = resolve_base_url(env)
    api_key = resolve_api_key(env)

    kwargs = {
        "model": model_id,
        "base_url": base_url,
        # ChatOpenAI requires a non-empty key; pass a placeholder if the run
        # forgot to supply one so the failure surfaces at the API call with a
        # clear auth error rather than a confusing constructor error.
        "api_key": api_key or "EMPTY",
    }
    temperature = _float_env(env, "BENCHFLOW_MODEL_TEMPERATURE")
    if temperature is not None:
        kwargs["temperature"] = temperature
    top_p = _float_env(env, "BENCHFLOW_MODEL_TOP_P")
    if top_p is not None:
        kwargs["top_p"] = top_p
    max_tokens = _int_env(env, "BENCHFLOW_MODEL_MAX_TOKENS")
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    return ChatOpenAI(**kwargs)


def _float_env(env: Mapping[str, str], key: str):
    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _int_env(env: Mapping[str, str], key: str):
    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


# ── deep agent construction + run ─────────────────────────────────────────────


def build_deep_agent(model_id: str, cwd: str, env: Mapping[str, str] | None = None):
    """Construct a deep agent rooted at ``cwd`` with shell + filesystem tools.

    Uses ``LocalShellBackend`` (a ``FilesystemBackend`` that also exposes the
    ``execute`` shell tool) rooted at the task workspace, with ``virtual_mode``
    so the agent's POSIX paths normalize under that root — the sandboxed FS the
    task expects. Imports are lazy so this module parses without deepagents
    installed.
    """
    from deepagents import create_deep_agent
    from deepagents.backends import LocalShellBackend

    chat_model = build_chat_model(model_id, env)
    backend = LocalShellBackend(root_dir=cwd, virtual_mode=True)
    return create_deep_agent(
        model=chat_model,
        backend=backend,
        system_prompt=(
            "You are an autonomous software engineering agent operating inside a "
            "sandboxed workspace. Use your filesystem and shell tools to complete "
            "the user's task end to end. Verify your work before finishing."
        ),
    )


def _message_role(message) -> str:
    """Best-effort role for a LangChain message object or dict."""
    if isinstance(message, dict):
        return message.get("type") or message.get("role") or ""
    return getattr(message, "type", "") or message.__class__.__name__.lower()


def _message_text(message) -> str:
    """Flatten a LangChain message's content to plain text."""
    content = (
        message.get("content")
        if isinstance(message, dict)
        else getattr(message, "content", "")
    )
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                # Content blocks are {"type": "text", "text": "..."} etc. — any
                # block carrying a "text" key contributes its text.
                if block.get("text"):
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content else ""


def _message_tool_calls(message) -> list:
    """Extract tool calls from an assistant message, as a list of dicts."""
    if isinstance(message, dict):
        calls = message.get("tool_calls") or []
    else:
        calls = getattr(message, "tool_calls", None) or []
    normalized = []
    for call in calls:
        if isinstance(call, dict):
            normalized.append(
                {
                    "id": call.get("id") or "",
                    "name": call.get("name") or "",
                    "args": call.get("args") or call.get("arguments") or {},
                }
            )
    return normalized


def run_deep_agent(
    model_id: str,
    instruction: str,
    cwd: str,
    session_id: str,
    env: Mapping[str, str] | None = None,
) -> dict:
    """Build and invoke a deep agent, emitting ACP updates for its trajectory.

    Returns a small metrics dict. Streams the LangGraph trajectory, emitting
    agent text, tool_call, and tool_call_update events keyed by tool call id as
    each step happens so BenchFlow's trajectory reconstructs the step sequence
    and its idle watchdog stays fed during long model turns.
    """
    start = time.time()
    agent = build_deep_agent(model_id, cwd, env)

    # Stream the LangGraph trajectory so ACP updates emit incrementally per step,
    # NOT one blocking invoke() that stays silent until it returns. BenchFlow's
    # idle watchdog counts ACP updates (tool calls + message/thought chunks), so a
    # long-but-productive model turn under a single invoke() emits nothing for
    # >600s and false-fires the idle timeout; streaming also preserves the
    # trajectory if the run is cancelled mid-flight (it was discarded before).
    message_count = 0
    tool_call_count = 0
    text_count = 0
    # Track which tool_call ids we have already announced so a ToolMessage maps
    # back onto its originating tool_call as a tool_call_update (and so a repeated
    # streamed update does not double-count).
    announced: set[str] = set()

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": instruction}]},
        config={"recursion_limit": _RECURSION_LIMIT},
        stream_mode="updates",
    ):
        for node_update in (chunk or {}).values():
            for message in (node_update or {}).get("messages", []):
                message_count += 1
                role = _message_role(message)

                if role in ("ai", "aimessage", "assistant"):
                    text = _message_text(message)
                    if text.strip():
                        _emit_text(session_id, text)
                        text_count += 1
                    for call in _message_tool_calls(message):
                        call_id = call["id"] or f"tc_{tool_call_count + 1}"
                        if call_id in announced:
                            continue
                        tool_call_count += 1
                        announced.add(call_id)
                        _emit_tool_call(session_id, call_id, call["name"], call["args"])

                elif role in ("tool", "toolmessage"):
                    tool_call_id = (
                        message.get("tool_call_id")
                        if isinstance(message, dict)
                        else getattr(message, "tool_call_id", "")
                    ) or ""
                    _emit_tool_result(session_id, tool_call_id, _message_text(message))

    elapsed = time.time() - start
    return {
        "message_count": message_count,
        "tool_call_count": tool_call_count,
        "text_count": text_count,
        "wall_clock_seconds": round(elapsed, 2),
    }


# ── ACP notification helpers ──────────────────────────────────────────────────


def _emit_text(session_id: str, text: str):
    """Emit agent text as an ACP agent_message_chunk."""
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": text[:_DIAG_TRUNCATE]},
                },
            },
        }
    )


def _emit_tool_call(session_id: str, tool_call_id: str, name: str, args):
    """Emit an ACP tool_call notification (status=in_progress)."""
    kind = _tool_kind(name)
    try:
        input_text = json.dumps(args)
    except (TypeError, ValueError):
        input_text = str(args)
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": tool_call_id,
                    "title": name or "tool",
                    "kind": kind,
                    "status": "in_progress",
                    "input": input_text[:_TOOL_INPUT_TRUNCATE],
                },
            },
        }
    )


def _emit_tool_result(session_id: str, tool_call_id: str, result: str):
    """Emit an ACP tool_call_update with completion status."""
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": tool_call_id,
                    "status": "completed",
                    "content": [
                        {
                            "type": "content",
                            "content": {
                                "type": "text",
                                "text": result[:_TOOL_RESULT_TRUNCATE],
                            },
                        }
                    ],
                },
            },
        }
    )


def _tool_kind(name: str) -> str:
    """Map a deepagents tool name onto an ACP tool kind for display."""
    n = (name or "").lower()
    # Order matters: planning tools (write_todos) carry "write" in their name,
    # so the todo/plan check must precede the write/edit check.
    if "todo" in n or "plan" in n:
        return "think"
    if "execute" in n or "shell" in n or "bash" in n:
        return "execute"
    if "write" in n or "edit" in n:
        return "edit"
    if "read" in n or "ls" in n or "glob" in n:
        return "read"
    if "search" in n or "grep" in n:
        return "search"
    return "other"


# ── Main ACP loop ─────────────────────────────────────────────────────────────


def main():
    session_id = "deepagents-shim"
    cwd = "/app"
    model = ""

    while True:
        try:
            msg = recv()
        except EOFError:
            break

        method = msg.get("method", "")
        req_id = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": 1,
                        "agentCapabilities": {
                            "loadSession": False,
                            "promptCapabilities": {"image": False, "audio": False},
                        },
                        "agentInfo": {"name": "deepagents", "version": "1.0"},
                    },
                }
            )

        elif method == "session/new":
            cwd = params.get("cwd", "/app")
            session_id = f"deepagents-{uuid.uuid4().hex[:8]}"
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"sessionId": session_id},
                }
            )

        elif method == "session/set_model":
            model = params.get("modelId", "")
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})

        elif method == "session/prompt":
            prompt_parts = params.get("prompt", [])
            text = ""
            for part in prompt_parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    text += part.get("text", "")

            model_id = resolve_model_id(model)
            stop_reason = "end_turn"
            try:
                metrics = run_deep_agent(
                    model_id=model_id,
                    instruction=text,
                    cwd=cwd,
                    session_id=session_id,
                )
                _emit_text(
                    session_id,
                    f"[deepagents completed: {metrics['message_count']} messages, "
                    f"{metrics['tool_call_count']} tool calls, "
                    f"{metrics['wall_clock_seconds']}s]",
                )
            except Exception as e:  # surface any failure to the client as text
                _emit_text(session_id, f"[deepagents error: {type(e).__name__}: {e}]")
                stop_reason = "end_turn"

            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"stopReason": stop_reason},
                }
            )

        elif method == "session/cancel":
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})

        elif method == "session/request_permission":
            options = params.get("options", [])
            option_id = options[0].get("optionId", "default") if options else "default"
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "outcome": {"outcome": "selected", "optionId": option_id}
                    },
                }
            )

        else:
            # Unknown method — return empty result to keep the client unblocked.
            if req_id is not None:
                send({"jsonrpc": "2.0", "id": req_id, "result": {}})


if __name__ == "__main__":
    main()
