#!/usr/bin/env python3
"""ACP-over-stdio shim for OpenRouter Ori's headless coding harness.

Ori exposes a persistent JSONL CLI rather than an ACP server.  This process is
the narrow transport adapter: it speaks ACP on stdin/stdout, invokes
``ori code --output jsonl`` for each prompt, resumes Ori's native session id,
and translates the streamed runtime events into ACP ``session/update``
notifications. The sibling :mod:`ori_jsonl` and :mod:`ori_events` modules own
tolerant decoding, typed token arithmetic, and event translation.

All three files are installed into the sandbox and run without BenchFlow installed,
so this module depends only on the Python standard library.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Installed scripts are sibling top-level modules.
    from ori_events import TurnTranslator
    from ori_jsonl import OriUsage, decode_line, terminal_result_error
except ModuleNotFoundError:  # Package import used by BenchFlow's unit tests.
    from .ori_events import TurnTranslator  # type: ignore[no-redef]
    from .ori_jsonl import OriUsage, decode_line, terminal_result_error

ORI_BINARY = "/opt/benchflow/bin/ori"
ORI_VERSION = "0.12.0+68f9a36"
_DEFAULT_MODEL = "openrouter/auto"
_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def send(message: dict[str, Any]) -> None:
    """Write one JSON-RPC frame without contaminating stdout."""
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def recv() -> dict[str, Any]:
    """Read the next non-empty JSON-RPC frame."""
    while True:
        line = sys.stdin.readline()
        if not line:
            raise EOFError("stdin closed")
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("JSON-RPC frame must be an object")
            return value


def _ensure_global_workspace() -> None:
    """Create only Ori's missing offline persona metadata, never credentials."""
    home = Path(os.environ.get("BENCHFLOW_AGENT_HOME") or Path.home())
    global_root = home / ".ori" / "global"
    (global_root / "features").mkdir(parents=True, exist_ok=True)
    ori_md = global_root / "ori.md"
    package_json = global_root / "package.json"
    if not ori_md.exists():
        ori_md.write_text(
            f"---\nmodel: {_DEFAULT_MODEL}\nversion: {ORI_VERSION}\n---\n",
            encoding="utf-8",
        )
    if not package_json.exists():
        package_json.write_text(
            '{\n  "name": "benchflow-ori-runtime",\n'
            '  "private": true,\n  "type": "module"\n}\n',
            encoding="utf-8",
        )


def build_ori_command(
    *,
    model: str,
    prompt_path: str,
    reasoning_effort: str | None = None,
    native_session_id: str | None = None,
    binary: str | None = None,
) -> list[str]:
    """Build Ori's argv as data; prompts never pass through a shell."""
    binary = binary or ORI_BINARY
    command = [
        binary,
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
    if reasoning_effort:
        command.extend(["--reasoning-effort", reasoning_effort])
    if native_session_id:
        command.extend(["--session", native_session_id])
    command.extend(["--prompt-file", prompt_path])
    return command


@dataclass
class SessionState:
    cwd: str
    model: str
    reasoning_effort: str | None = None
    native_session_id: str | None = None
    usage: OriUsage = field(default_factory=OriUsage)
    has_usage: bool = False


class OriACPServer:
    """Minimal ACP server with one or more independent Ori sessions."""

    def __init__(self) -> None:
        self.sessions: dict[str, SessionState] = {}
        self.current_process: subprocess.Popen[str] | None = None

    def handle(self, message: dict[str, Any]) -> None:
        method = str(message.get("method") or "")
        request_id = message.get("id")
        params = message.get("params")
        params = params if isinstance(params, dict) else {}

        if method == "initialize":
            self._reply(
                request_id,
                {
                    "protocolVersion": 1,
                    "agentCapabilities": {
                        "loadSession": False,
                        "promptCapabilities": {"image": False, "audio": False},
                    },
                    "agentInfo": {"name": "openrouter-ori", "version": ORI_VERSION},
                },
            )
            return
        if method == "session/new":
            self._new_session(request_id, params)
            return
        if method == "session/set_model":
            state = self._state(params)
            model = params.get("modelId")
            if not isinstance(model, str) or not model:
                raise ValueError("modelId must be a non-empty string")
            state.model = model
            self._reply(request_id, {})
            return
        if method == "session/set_config_option":
            self._set_config_option(request_id, params)
            return
        if method == "session/prompt":
            self._prompt(request_id, params)
            return
        if method == "session/cancel":
            if self.current_process is not None and self.current_process.poll() is None:
                self.current_process.terminate()
            return
        if request_id is not None:
            self._error(request_id, -32601, f"Method not found: {method}")

    @staticmethod
    def _reply(request_id: object, result: dict[str, Any]) -> None:
        send({"jsonrpc": "2.0", "id": request_id, "result": result})

    @staticmethod
    def _error(request_id: object, code: int, message: str) -> None:
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": message[-4000:]},
            }
        )

    def _state(self, params: dict[str, Any]) -> SessionState:
        session_id = params.get("sessionId")
        if not isinstance(session_id, str) or session_id not in self.sessions:
            raise ValueError(f"unknown sessionId: {session_id!r}")
        return self.sessions[session_id]

    def _new_session(self, request_id: object, params: dict[str, Any]) -> None:
        _ensure_global_workspace()
        cwd = params.get("cwd")
        if not isinstance(cwd, str) or not cwd:
            cwd = os.getcwd()
        session_id = f"ori-{uuid.uuid4().hex[:12]}"
        model = (
            os.environ.get("ORI_MODEL")
            or os.environ.get("BENCHFLOW_PROVIDER_MODEL")
            or _DEFAULT_MODEL
        )
        self.sessions[session_id] = SessionState(cwd=cwd, model=model)
        # The registry declares the private config id BenchFlow should use for
        # requested effort. Do not advertise a fictitious current value when
        # no effort was requested and Ori owns its default.
        self._reply(request_id, {"sessionId": session_id})

    def _set_config_option(self, request_id: object, params: dict[str, Any]) -> None:
        state = self._state(params)
        if params.get("configId") != "reasoning_effort":
            raise ValueError(f"unknown configId: {params.get('configId')!r}")
        value = params.get("value")
        if value not in _EFFORTS:
            raise ValueError(f"unsupported Ori reasoning effort: {value!r}")
        state.reasoning_effort = str(value)
        self._reply(request_id, {})

    def _prompt(self, request_id: object, params: dict[str, Any]) -> None:
        state = self._state(params)
        prompt = "".join(
            str(part.get("text") or "")
            for part in params.get("prompt", [])
            if isinstance(part, dict) and part.get("type") == "text"
        )
        session_id = str(params["sessionId"])
        translator = TurnTranslator(session_id, send)
        prompt_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="benchflow-ori-prompt-",
                suffix=".txt",
                delete=False,
            ) as prompt_file:
                prompt_file.write(prompt)
                prompt_path = prompt_file.name
            os.chmod(prompt_path, 0o600)
            command = build_ori_command(
                model=state.model,
                prompt_path=prompt_path,
                reasoning_effort=state.reasoning_effort,
                native_session_id=state.native_session_id,
            )
            child_env = {
                **os.environ,
                "CI": "true",
                "ORI_TELEMETRY": "0",
            }
            self.current_process = subprocess.Popen(
                command,
                cwd=state.cwd,
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert self.current_process.stdout is not None
            for line_number, line in enumerate(self.current_process.stdout, start=1):
                decoded = decode_line(line, line_number)
                if decoded is None:
                    continue
                if decoded.document is None:
                    translator.consume_diagnostic(decoded.line_number, decoded.raw)
                else:
                    translator.consume_document(decoded.document)
            return_code = self.current_process.wait()
        finally:
            self.current_process = None
            if prompt_path:
                Path(prompt_path).unlink(missing_ok=True)

        if translator.native_session_id:
            state.native_session_id = translator.native_session_id
        if translator.turn_usage is not None:
            state.usage = state.usage + translator.turn_usage
            state.has_usage = True

        result = translator.result
        if return_code != 0:
            detail = (
                terminal_result_error(result)
                or translator.last_diagnostic
                or f"Ori exited with status {return_code}"
            )
            raise RuntimeError(
                f"Ori code failed with exit code {return_code}: {detail}"
            )
        if result is None:
            raise RuntimeError("Ori JSONL stream ended without a terminal result")
        if result.get("ok") is not True:
            detail = terminal_result_error(result) or "unknown Ori failure"
            raise RuntimeError(f"Ori code failed: {detail}")

        response: dict[str, Any] = {"stopReason": "end_turn"}
        if state.has_usage:
            response["usage"] = state.usage.as_acp()
        self._reply(request_id, response)


def main() -> None:
    server = OriACPServer()
    while True:
        try:
            message = recv()
        except EOFError:
            break
        except Exception as exc:
            print(f"ori-acp-shim input error: {exc}", file=sys.stderr)
            continue
        request_id = message.get("id")
        try:
            server.handle(message)
        except Exception as exc:
            if request_id is not None:
                server._error(request_id, -32603, f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()


__all__ = [
    "ORI_BINARY",
    "ORI_VERSION",
    "OriACPServer",
    "SessionState",
    "TurnTranslator",
    "build_ori_command",
    "main",
]
