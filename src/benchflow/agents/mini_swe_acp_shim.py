#!/usr/bin/env python3
"""ACP shim for mini-swe-agent — wraps it as an ACP server on stdio.

mini-swe-agent (https://github.com/SWE-agent/mini-swe-agent) is a deliberately
minimal coding harness: a single ``bash`` tool, one shared system prompt, no
vendor editing primitives. That uniformity is the whole point — it lets you
compare models apples-to-apples on the same scaffold.

This shim runs mini-swe's own ``DefaultAgent`` loop in-process against the task
checkout and re-emits each step as ACP ``session/update`` notifications so
BenchFlow captures the trajectory. The agent's bundled ``mini.yaml`` config is
loaded verbatim (minus the interactive ``mode`` key) so the guardrails that the
upstream harness ships with are reproduced faithfully:

  - single ``bash`` tool (tool-calling, ``BASH_TOOL``)
  - shared system / instance templates (find → reproduce → fix → verify → edge → submit)
  - >10k-char command output truncated head/tail (``observation_template``)
  - malformed tool calls caught and retried with guidance (``format_error_template``)

Architecture:
  benchflow ACP client ←stdio→ this shim ←in-process→ minisweagent DefaultAgent
                                          ←subprocess→  bash in the task cwd

stdout discipline: mini-swe (and litellm) print to stdout at import / runtime.
stdout is the ACP JSON-RPC channel, so before importing anything we save the
real stdout fd and redirect fd 1 to stderr — every stray print then lands on
stderr and only ``send()`` writes framed JSON-RPC to the client.
"""

import json
import os
import sys

# ── stdout isolation (must happen before importing minisweagent/litellm) ──────
#
# Duplicate the real stdout (the pipe the ACP client reads), then point fd 1 at
# stderr so anything writing to sys.stdout/print/fd-1 is diverted away from the
# JSON-RPC channel. ``_OUT`` is the only writer to the client.
_real_stdout_fd = os.dup(1)
os.dup2(2, 1)
_OUT = os.fdopen(_real_stdout_fd, "w", buffering=1, encoding="utf-8")
sys.stdout = sys.stderr

# Silence mini-swe's import banner and make litellm cost-tracking failures
# non-fatal (BenchFlow routes through a usage proxy where litellm often can't
# price a model; without this the run dies in _calculate_cost).
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
os.environ.setdefault("LITELLM_LOG", "ERROR")

import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402
from minisweagent import package_dir  # noqa: E402
from minisweagent.agents.default import AgentConfig, DefaultAgent  # noqa: E402
from minisweagent.environments.local import LocalEnvironment  # noqa: E402
from minisweagent.models.litellm_model import LitellmModel  # noqa: E402

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

_DIAG_TRUNCATE = 2000
_TOOL_RESULT_TRUNCATE = 2000
_DEFAULT_CWD = "/app"
# Which mini-swe config to mirror. mini.yaml is the generic (non-SWE-bench)
# tool-calling harness; swebench.yaml hardcodes /testbed + "don't touch tests".
_CONFIG_NAME = os.environ.get("MINI_SWE_CONFIG", "mini.yaml")

# Map BenchFlow's resolved provider protocol → litellm provider prefix. litellm
# infers the provider from the model name, so a bare name (the SDK strips the
# prefix before set_model) needs one prepended when we know the protocol.
_PROTOCOL_PREFIX = {
    "anthropic-messages": "anthropic",
    "openai-completions": "openai",
    "openai-responses": "openai",
}


# ── ACP stdio I/O ─────────────────────────────────────────────────────────────


def send(msg: dict) -> None:
    _OUT.write(json.dumps(msg) + "\n")
    _OUT.flush()


def recv() -> dict:
    while True:
        line = sys.stdin.readline()
        if not line:
            raise EOFError("stdin closed")
        line = line.strip()
        if not line:
            continue
        return json.loads(line)


# ── ACP notification helpers ──────────────────────────────────────────────────


def _emit(session_id: str, update: dict) -> None:
    """Wrap an ACP ``update`` payload in the session/update envelope and send."""
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
            "content": {"type": "text", "text": text[:_DIAG_TRUNCATE]},
        },
    )


def _emit_tool_call(session_id: str, tool_call_id: str, command: str) -> None:
    _emit(
        session_id,
        {
            "sessionUpdate": "tool_call",
            "toolCallId": tool_call_id,
            "title": command[:_DIAG_TRUNCATE],
            "kind": "execute",
            "status": "in_progress",
        },
    )


def _emit_tool_result(session_id: str, tool_call_id: str, output: str) -> None:
    _emit(
        session_id,
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": tool_call_id,
            "status": "completed",
            "content": [
                {
                    "type": "content",
                    "content": {"type": "text", "text": output[:_TOOL_RESULT_TRUNCATE]},
                }
            ],
        },
    )


# ── Trajectory-emitting agent ─────────────────────────────────────────────────


class _ACPAgent(DefaultAgent):
    """DefaultAgent that re-emits each step as ACP session updates.

    ``query`` fires once per model turn (assistant text + the bash command);
    ``execute_actions`` fires after the command runs (its output). The action's
    ``tool_call_id`` (assigned by mini-swe's tool-call parser) is reused as the
    ACP ``toolCallId`` so start/result pair up.
    """

    def __init__(self, *args, session_id: str, **kwargs):
        self._session_id = session_id
        super().__init__(*args, **kwargs)

    def query(self) -> dict:
        message = super().query()
        text = (message.get("content") or "").strip()
        if text:
            _emit_text(self._session_id, text)
        for action in message.get("extra", {}).get("actions", []):
            _emit_tool_call(
                self._session_id,
                action.get("tool_call_id", ""),
                action.get("command", ""),
            )
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        # env.execute may raise Submitted (task complete) mid-list; emit results
        # for whatever ran before re-raising so the trajectory isn't lost.
        observations = super().execute_actions(message)
        for action, obs in zip(actions, observations, strict=False):
            _emit_tool_result(
                self._session_id,
                action.get("tool_call_id", ""),
                obs.get("content", ""),
            )
        return observations


# ── Model / config construction ───────────────────────────────────────────────


def _load_config() -> dict:
    """Load mini-swe's bundled config, split into agent/model/environment dicts."""
    cfg_path = Path(package_dir) / "config" / _CONFIG_NAME
    raw = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    agent_cfg = dict(raw.get("agent", {}))
    # Drop interactive-only / unknown keys (e.g. ``mode: confirm``) — this shim
    # runs headless. Keep only fields DefaultAgent's AgentConfig accepts.
    agent_cfg = {k: v for k, v in agent_cfg.items() if k in AgentConfig.model_fields}
    return {
        "agent": agent_cfg,
        "model": dict(raw.get("model", {})),
        "environment": dict(raw.get("environment", {})),
    }


def _build_model(model_cfg: dict, model_override: str) -> LitellmModel:
    """Build a LitellmModel wired to BenchFlow's resolved provider, if any."""
    bare = (
        model_override
        or os.environ.get("BENCHFLOW_PROVIDER_MODEL")
        or os.environ.get("MSWEA_MODEL_NAME")
        or ""
    )
    protocol = os.environ.get("BENCHFLOW_PROVIDER_PROTOCOL", "")
    base_url = os.environ.get("BENCHFLOW_PROVIDER_BASE_URL")
    api_key = os.environ.get("BENCHFLOW_PROVIDER_API_KEY")

    model_name = bare
    if "/" not in bare and protocol in _PROTOCOL_PREFIX:
        model_name = f"{_PROTOCOL_PREFIX[protocol]}/{bare}"

    cfg = dict(model_cfg)
    model_kwargs = dict(cfg.pop("model_kwargs", {}) or {})
    if base_url:
        model_kwargs["api_base"] = base_url
    if api_key:
        model_kwargs["api_key"] = api_key

    return LitellmModel(model_name=model_name, model_kwargs=model_kwargs, **cfg)


# ── Main ACP loop ──────────────────────────────────────────────────────────────


def main() -> None:
    session_id = "mini-swe-shim"
    cwd = _DEFAULT_CWD
    model_override = ""
    config = _load_config()

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
                        "agentInfo": {"name": "mini-swe-agent", "version": "1.0"},
                    },
                }
            )

        elif method == "session/new":
            cwd = params.get("cwd") or _DEFAULT_CWD
            session_id = "mini-swe-shim"
            send({"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": session_id}})

        elif method == "session/set_model":
            model_override = params.get("modelId", "") or model_override
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})

        elif method == "session/prompt":
            text = "".join(
                part.get("text", "")
                for part in params.get("prompt", [])
                if isinstance(part, dict) and part.get("type") == "text"
            )

            try:
                model = _build_model(config["model"], model_override)
                env = LocalEnvironment(cwd=cwd, **config["environment"])
                agent = _ACPAgent(
                    model,
                    env,
                    session_id=session_id,
                    **config["agent"],
                )
                result = agent.run(text)
                exit_status = result.get("exit_status", "")
                submission = result.get("submission", "")
                _emit_text(
                    session_id,
                    f"[mini-swe-agent finished: exit_status={exit_status}, "
                    f"steps={agent.n_calls}]"
                    + (f"\n{submission}" if submission else ""),
                )
            except Exception as e:
                # Surface the error in the trajectory rather than crashing the
                # shim — the ACP client still needs a prompt response.
                _emit_text(session_id, f"[mini-swe-agent error: {e}]")

            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"stopReason": "end_turn"},
                }
            )

        elif method == "session/cancel":
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})

        else:
            if req_id is not None:
                send({"jsonrpc": "2.0", "id": req_id, "result": {}})


if __name__ == "__main__":
    main()
