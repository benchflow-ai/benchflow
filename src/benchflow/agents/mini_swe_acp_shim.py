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

Import safety: the module top level has NO side effects and does NOT import
``minisweagent`` — stdout redirection and the (banner-printing) minisweagent
import happen inside ``main()``. This keeps the pure routing policy
(``_litellm_prefix``) importable and unit-testable without minisweagent
installed and without clobbering the importer's stdout.
"""

import json
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

_DIAG_TRUNCATE = 2000
_TOOL_RESULT_TRUNCATE = 2000
_DEFAULT_CWD = "/app"
# Which mini-swe config to mirror. mini.yaml is the generic (non-SWE-bench)
# tool-calling harness; swebench.yaml hardcodes /testbed + "don't touch tests".
_CONFIG_NAME = os.environ.get("MINI_SWE_CONFIG", "mini.yaml")

# Set in main() by _isolate_stdout(); the only writer to the ACP client.
_OUT = sys.stdout


# ── Provider routing policy (pure; unit-tested) ────────────────────────────────


def _is_anthropic_model(model: str) -> bool:
    m = model.lower()
    return "claude" in m or "anthropic" in m


def _litellm_prefix(protocol: str, model: str) -> str:
    """Pick the litellm provider prefix for BenchFlow's resolved endpoint.

    The SDK strips the provider prefix before handing the model to the agent, so
    litellm can't infer the dialect from a bare name — we reconstruct it from the
    resolved ``BENCHFLOW_PROVIDER_PROTOCOL``. mini-swe drives ``litellm.completion``,
    which speaks chat-completions and anthropic-messages but NOT the OpenAI
    Responses API; ``openai-responses`` only ever comes from aws-bedrock, whose
    proxy also exposes an anthropic-messages surface, so Anthropic models route
    there. Returns "" for unknown protocols (let litellm infer from the name).
    """
    if protocol == "anthropic-messages":
        return "anthropic"
    if protocol == "openai-completions":
        return "openai"
    if protocol == "openai-responses":
        return "anthropic" if _is_anthropic_model(model) else "openai"
    return ""


# ── ACP stdio I/O ─────────────────────────────────────────────────────────────


def _isolate_stdout() -> None:
    """Reserve fd 1 (the ACP JSON-RPC channel) for framed output only.

    mini-swe and litellm print to stdout at import and at runtime. Duplicate the
    real stdout (the pipe the client reads) into ``_OUT``, then redirect fd 1 to
    stderr so every stray print is diverted away from the JSON-RPC channel.
    """
    global _OUT
    real_stdout_fd = os.dup(1)
    os.dup2(2, 1)
    _OUT = os.fdopen(real_stdout_fd, "w", buffering=1, encoding="utf-8")
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


def _acp_agent_class():
    """Build the trajectory-emitting DefaultAgent subclass.

    Defined behind a factory so the module stays importable without
    ``minisweagent`` (only available inside the sandbox). ``main()`` calls this
    once after the runtime is installed.
    """
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.exceptions import Submitted

    class _ACPAgent(DefaultAgent):
        """DefaultAgent that re-emits each step as ACP session updates.

        ``query`` fires once per model turn (assistant text + the bash command);
        ``execute_actions`` fires after the command runs (its output). The
        action's ``tool_call_id`` (assigned by mini-swe's tool-call parser) is
        reused as the ACP ``toolCallId`` so start/result pair up.
        """

        def __init__(self, *args, session_id: str, **kwargs):
            self._session_id = session_id
            super().__init__(*args, **kwargs)

        def query(self) -> dict:
            message = super().query()
            text = (message.get("content") or "").strip()
            if text:
                _emit_text(self._session_id, text)
            return message

        def execute_actions(self, message: dict) -> list[dict]:
            # Mirrors DefaultAgent.execute_actions, but drives the env loop
            # per-action so the ACP tool-call lifecycle is modeled accurately:
            # each action emits start→result around its own env.execute, and an
            # action that never runs (e.g. anything after a submit) emits nothing
            # rather than being falsely marked completed. mini-swe's instance
            # template asks for submit alone, but a turn may carry several tool
            # calls, so we don't assume one.
            actions = message.get("extra", {}).get("actions", [])
            outputs: list[dict] = []
            tool_call_id = ""
            try:
                for action in actions:
                    tool_call_id = action.get("tool_call_id", "")
                    _emit_tool_call(
                        self._session_id, tool_call_id, action.get("command", "")
                    )
                    output = self.env.execute(action)  # may raise Submitted
                    outputs.append(output)
                    _emit_tool_result(
                        self._session_id, tool_call_id, output.get("output", "")
                    )
            except Submitted as e:
                # The submit command makes env.execute raise before returning, so
                # close out only that action (re-raise; DefaultAgent.run handles
                # Submitted via InterruptAgentFlow).
                submission = e.messages[0].get("content", "") if e.messages else ""
                _emit_tool_result(
                    self._session_id,
                    tool_call_id,
                    submission or "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
                )
                raise
            return self.add_messages(
                *self.model.format_observation_messages(
                    message, outputs, self.get_template_vars()
                )
            )

    return _ACPAgent


# ── Model / config construction ───────────────────────────────────────────────


def _load_config() -> dict:
    """Load mini-swe's bundled config, split into agent/model/environment dicts."""
    from minisweagent import package_dir
    from minisweagent.agents.default import AgentConfig

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


def _build_model(model_cfg: dict, model_override: str):
    """Build a LitellmModel wired to BenchFlow's resolved provider, if any."""
    from minisweagent.models.litellm_model import LitellmModel

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
    if "/" not in bare:
        prefix = _litellm_prefix(protocol, bare)
        if prefix:
            model_name = f"{prefix}/{bare}"

    cfg = dict(model_cfg)
    model_kwargs = dict(cfg.pop("model_kwargs", {}) or {})
    if base_url:
        model_kwargs["api_base"] = base_url
    if api_key:
        model_kwargs["api_key"] = api_key

    return LitellmModel(model_name=model_name, model_kwargs=model_kwargs, **cfg)


# ── Main ACP loop ──────────────────────────────────────────────────────────────


def main() -> None:
    # Silence mini-swe's import banner and make litellm cost-tracking failures
    # non-fatal (BenchFlow routes through a usage proxy where litellm often can't
    # price a model; without this the run dies in _calculate_cost).
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    os.environ.setdefault("LITELLM_LOG", "ERROR")
    _isolate_stdout()

    from minisweagent.environments.local import LocalEnvironment

    config = _load_config()
    acp_agent_class = _acp_agent_class()

    session_id = "mini-swe-shim"
    cwd = _DEFAULT_CWD
    model_override = ""

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
                agent = acp_agent_class(
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
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"stopReason": "end_turn"},
                    }
                )
            except Exception as e:
                # Unexpected failures here are auth/provider/protocol/runtime
                # errors (the agent's own task failures return normally above
                # with an exit_status). Return a JSON-RPC error so BenchFlow
                # classifies it as an agent/infra error instead of masking it as
                # a task failure (matches the openclaw shim).
                _emit_text(session_id, f"[mini-swe-agent error: {e}]")
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32603, "message": str(e)},
                    }
                )

        elif method == "session/cancel":
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})

        else:
            if req_id is not None:
                send({"jsonrpc": "2.0", "id": req_id, "result": {}})


if __name__ == "__main__":
    main()
