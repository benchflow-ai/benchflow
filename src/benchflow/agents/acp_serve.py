#!/usr/bin/env python3
"""Generic ACP server for *pure* agents.

This is BenchFlow's "the harness handles the shim" — instead of every agent
writing its own ACP shim, a pure agent exposes a single callable and this server
wraps it on stdio, re-emitting each step as ACP ``session/update`` so BenchFlow
captures the trajectory.

Usage (the launch command deploys this file and runs):

    python acp_serve.py mini_computer_agent.core:run

The pure-agent contract — a single callable::

    run(task: str, model: str, *, on_step=None, max_steps=15, model_kwargs=None) -> str

``on_step(step, reply_text, action)`` is called once per turn for trajectory
capture (``action`` may be None). The agent imports no benchflow code.

Standalone (stdlib only): deployed into the sandbox next to the agent venv; it
imports the agent callable dynamically. Model routing is reconstructed from
``BENCHFLOW_PROVIDER_*`` exactly like the per-agent shims.
"""

import importlib
import json
import os
import sys

_OUT = sys.stdout


def _isolate_stdout() -> None:
    """Reserve fd 1 for the JSON-RPC channel; divert stray prints to stderr."""
    global _OUT
    real = os.dup(1)
    os.dup2(2, 1)
    _OUT = os.fdopen(real, "w", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr


def send(msg: dict) -> None:
    _OUT.write(json.dumps(msg) + "\n")
    _OUT.flush()


def recv() -> dict:
    while True:
        line = sys.stdin.readline()
        if not line:
            raise EOFError("stdin closed")
        line = line.strip()
        if line:
            return json.loads(line)


# ── provider routing (reconstruct the litellm model from BenchFlow env) ────────
def _is_anthropic_model(model: str) -> bool:
    m = model.lower()
    return "claude" in m or "anthropic" in m


def _litellm_prefix(protocol: str, model: str) -> str:
    if protocol == "anthropic-messages":
        return "anthropic"
    if protocol == "openai-completions":
        return "openai"
    if protocol == "openai-responses":
        return "anthropic" if _is_anthropic_model(model) else "openai"
    return ""


def resolve_model(override: str) -> tuple[str, dict]:
    bare = override or os.environ.get("BENCHFLOW_PROVIDER_MODEL") or ""
    model = bare
    if bare and "/" not in bare:
        prefix = _litellm_prefix(
            os.environ.get("BENCHFLOW_PROVIDER_PROTOCOL", ""), bare
        )
        if prefix:
            model = f"{prefix}/{bare}"
        elif "gemini" in bare.lower():
            # litellm needs a provider for gemini; protocol routing doesn't cover it.
            model = f"gemini/{bare}"
    kwargs: dict = {}
    if os.environ.get("BENCHFLOW_PROVIDER_BASE_URL"):
        kwargs["api_base"] = os.environ["BENCHFLOW_PROVIDER_BASE_URL"]
    if os.environ.get("BENCHFLOW_PROVIDER_API_KEY"):
        kwargs["api_key"] = os.environ["BENCHFLOW_PROVIDER_API_KEY"]
    return model, kwargs


# ── ACP session/update emission ───────────────────────────────────────────────
def _emit(session_id: str, update: dict) -> None:
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": session_id, "update": update},
        }
    )


def _emit_text(session_id: str, text: str) -> None:
    _emit(
        session_id,
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": text[:2000]},
        },
    )


def _step_emitter(session_id: str):
    def on_step(step: int, reply: str, action) -> None:
        _emit(
            session_id,
            {
                "sessionUpdate": "tool_call",
                "toolCallId": f"step-{step}",
                "title": (
                    json.dumps(action, sort_keys=True) if action else "(no action)"
                )[:2000],
                "kind": "other",
                "status": "completed",
                "content": [
                    {
                        "type": "content",
                        "content": {"type": "text", "text": (reply or "")[:2000]},
                    }
                ],
            },
        )

    return on_step


def _load_run(spec: str):
    """Import a 'module:callable' agent entry point."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"agent spec must be 'module:callable', got {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: acp_serve.py <module:callable>", file=sys.stderr)
        return 2
    spec = argv[1]
    os.environ.setdefault("LITELLM_LOG", "ERROR")
    _isolate_stdout()
    run = _load_run(spec)  # the pure agent's run callable

    agent_name = os.environ.get("BENCHFLOW_AGENT_NAME", spec.split(":")[0])
    max_steps = int(os.environ.get("BENCHFLOW_AGENT_MAX_STEPS", "15"))
    session_id = agent_name
    model_override = ""

    while True:
        try:
            msg = recv()
        except EOFError:
            return 0
        method = msg.get("method", "")
        rid = msg.get("id")
        params = msg.get("params") or {}

        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {
                        "protocolVersion": 1,
                        "agentCapabilities": {
                            "loadSession": False,
                            "promptCapabilities": {"image": False, "audio": False},
                        },
                        # Protocol-server version; intentionally decoupled from
                        # the wrapped agent's version (the pure-agent contract is
                        # just ``run`` — the shim can't read the agent's version).
                        "agentInfo": {"name": agent_name, "version": "0.1.0"},
                    },
                }
            )
        elif method == "session/new":
            session_id = agent_name
            send({"jsonrpc": "2.0", "id": rid, "result": {"sessionId": session_id}})
        elif method == "session/set_model":
            model_override = params.get("modelId", "") or model_override
            send({"jsonrpc": "2.0", "id": rid, "result": {}})
        elif method == "session/prompt":
            task = "".join(
                p.get("text", "")
                for p in params.get("prompt", [])
                if isinstance(p, dict) and p.get("type") == "text"
            )
            model, model_kwargs = resolve_model(model_override)
            try:
                result = run(
                    task,
                    model,
                    on_step=_step_emitter(session_id),
                    max_steps=max_steps,
                    model_kwargs=model_kwargs,
                )
                _emit_text(session_id, f"[{agent_name} done: {result}]")
                send(
                    {"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}}
                )
            except Exception as exc:  # infra/provider error -> JSON-RPC error
                _emit_text(session_id, f"[{agent_name} error: {exc}]")
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": rid,
                        "error": {"code": -32603, "message": str(exc)},
                    }
                )
        elif method == "session/cancel":
            send({"jsonrpc": "2.0", "id": rid, "result": {}})
        else:
            if rid is not None:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": rid,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}",
                        },
                    }
                )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
