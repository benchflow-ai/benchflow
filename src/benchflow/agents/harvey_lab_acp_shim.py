#!/usr/bin/env python3
"""ACP shim for Harvey LAB — wraps the Harvey LAB harness as an ACP server.

Runs the Harvey LAB agent loop (6 tools: bash, read, write, edit, glob, grep)
directly on the filesystem, without Podman. BenchFlow's Docker container
provides equivalent sandboxing.

Architecture:
  benchflow ACP client ←stdio→ this shim ←in-process→ Harvey LAB agent loop
                                          ←filesystem→  /workspace/{documents,output}

The shim:
1. Clones harveyai/harvey-labs into /opt/harvey-labs at install time.
2. On session/prompt, reads the instruction, sets up workspace directories,
   creates a Harvey LAB model adapter, and runs the agent loop in-process.
3. Emits ACP session/update notifications for tool calls and text so
   BenchFlow can capture the trajectory.
"""

import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

_DIAG_TRUNCATE = 2000
_TOOL_RESULT_TRUNCATE = 1000
_HARVEY_LABS_ROOT = Path(os.environ.get("HARVEY_LABS_ROOT", "/opt/harvey-labs"))


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


# ── Lightweight filesystem Sandbox ────────────────────────────────────────────
#
# Implements the subset of sandbox.sandbox.Sandbox that ToolExecutor needs,
# backed by direct filesystem and subprocess calls (no Podman).


class DirectSandbox:
    """Drop-in replacement for Harvey LAB's Sandbox, operating on local dirs.

    Harvey LAB's ToolExecutor calls:
      - sandbox.exec(cmd, timeout=N) -> ExecResult
      - sandbox.read_file(path) -> bytes
      - sandbox.write_file(path, content)
      - sandbox.exists(path) -> bool
      - sandbox.list_files(path) -> list[str]
    Plus class methods:
      - Sandbox.assert_sandbox_path(path)
      - Sandbox.is_writable(path)

    This class maps sandbox paths (/workspace/...) to real host paths.
    """

    WORKSPACE_PATH = "/workspace"
    DOCUMENTS_PATH = "/workspace/documents"
    OUTPUT_PATH = "/workspace/output"

    def __init__(
        self,
        documents_dir: Path,
        output_dir: Path,
        workspace_dir: Path,
        default_timeout: int = 60,
        **kwargs,
    ):
        self.documents_dir = documents_dir
        self.output_dir = output_dir
        self.workspace_dir = workspace_dir
        self.default_timeout = default_timeout
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        """No-op: no container to start."""

    def stop(self):
        """No-op: no container to stop."""

    def _to_host_path(self, sandbox_path: str) -> Path:
        """Map a /workspace/... path to a real host path."""
        if sandbox_path == self.DOCUMENTS_PATH or sandbox_path.startswith(
            self.DOCUMENTS_PATH + "/"
        ):
            rel = sandbox_path[len(self.DOCUMENTS_PATH) :].lstrip("/")
            return self.documents_dir / rel if rel else self.documents_dir
        if sandbox_path == self.OUTPUT_PATH or sandbox_path.startswith(
            self.OUTPUT_PATH + "/"
        ):
            rel = sandbox_path[len(self.OUTPUT_PATH) :].lstrip("/")
            return self.output_dir / rel if rel else self.output_dir
        if sandbox_path == self.WORKSPACE_PATH or sandbox_path.startswith(
            self.WORKSPACE_PATH + "/"
        ):
            rel = sandbox_path[len(self.WORKSPACE_PATH) :].lstrip("/")
            return self.workspace_dir / rel if rel else self.workspace_dir
        raise ValueError(f"Path outside sandbox: {sandbox_path}")

    def exec(self, command: str, timeout: int | None = None) -> "ExecResult":
        """Run a shell command in the workspace directory."""
        timeout = timeout if timeout is not None else self.default_timeout
        env = {
            **os.environ,
            "WORKSPACE_DIR": str(self.workspace_dir),
            "DOCUMENTS_DIR": str(self.documents_dir),
            "OUTPUT_DIR": str(self.output_dir),
        }
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace_dir),
                env=env,
            )
            return ExecResult(
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(stdout="", stderr="", returncode=None, timed_out=True)

    def read_file(self, sandbox_path: str) -> bytes:
        host_path = self._to_host_path(sandbox_path)
        if host_path.is_dir():
            raise IsADirectoryError(sandbox_path)
        return host_path.read_bytes()

    def write_file(self, sandbox_path: str, content: str | bytes) -> None:
        host_path = self._to_host_path(sandbox_path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            host_path.write_text(content)
        else:
            host_path.write_bytes(content)

    def exists(self, sandbox_path: str) -> bool:
        try:
            host_path = self._to_host_path(sandbox_path)
            return host_path.exists()
        except ValueError:
            return False

    def list_files(self, sandbox_path: str) -> list[str]:
        host_path = self._to_host_path(sandbox_path)
        if not host_path.is_dir():
            return []
        return [
            str(p.relative_to(host_path)) for p in host_path.rglob("*") if p.is_file()
        ]

    @staticmethod
    def assert_sandbox_path(path: str) -> None:
        if not (path == "/workspace" or path.startswith("/workspace/")):
            raise ValueError(f"Path outside sandbox: {path}")

    @staticmethod
    def is_writable(path: str) -> bool:
        return (
            path == "/workspace/output"
            or path.startswith("/workspace/output/")
            or (
                (path == "/workspace" or path.startswith("/workspace/"))
                and not (
                    path == "/workspace/documents"
                    or path.startswith("/workspace/documents/")
                )
            )
        )


class ExecResult:
    """Mirrors sandbox.sandbox.ExecResult."""

    def __init__(
        self, stdout: str, stderr: str, returncode: int | None, timed_out: bool = False
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.timed_out = timed_out

    @property
    def ok(self) -> bool:
        return self.returncode == 0


# ── Harvey LAB agent runner ──────────────────────────────────────────────────


def _load_system_prompt() -> str:
    """Load Harvey LAB's system prompt and skill manuals."""
    prompt_path = _HARVEY_LABS_ROOT / "harness" / "system_prompt.md"
    if not prompt_path.exists():
        return "You are an AI agent executing a task provided by the user within a workspace."
    prompt = prompt_path.read_text(encoding="utf-8")

    # Load skill manuals
    skills_dir = _HARVEY_LABS_ROOT / "harness" / "skills"
    if skills_dir.exists():
        for skill_path in sorted(skills_dir.glob("*/SKILL.md")):
            name = skill_path.parent.name
            prompt += f"\n\n## Skill: {name}\n\n{skill_path.read_text()}"

    return prompt


def _patch_sandbox_module() -> None:
    """Inject a mock ``sandbox.sandbox`` module into sys.modules.

    Harvey LAB's ``harness/tools.py`` does ``from sandbox.sandbox import
    Sandbox, ...``.  We intercept that import so it gets our lightweight
    DirectSandbox instead of the real Podman-backed one.
    """
    import types

    inner = types.ModuleType("sandbox.sandbox")
    outer = types.ModuleType("sandbox")
    for name, value in [
        ("Sandbox", DirectSandbox),
        ("ExecResult", ExecResult),
        ("WORKSPACE_PATH", "/workspace"),
        ("DOCUMENTS_PATH", "/workspace/documents"),
        ("OUTPUT_PATH", "/workspace/output"),
        ("DEFAULT_IMAGE", "lab-sandbox:latest"),
    ]:
        setattr(inner, name, value)
    outer.sandbox = inner  # ty: ignore[unresolved-attribute]
    sys.modules["sandbox"] = outer
    sys.modules["sandbox.sandbox"] = inner


def _create_adapter(
    model: str, temperature: float = 0.0, reasoning_effort: str | None = None
):
    """Create a Harvey LAB model adapter for the given model string.

    Imports are dynamic (importlib) because the harness package is only
    available at runtime inside the sandbox, not in BenchFlow's dev venv.
    """
    sys.path.insert(0, str(_HARVEY_LABS_ROOT))

    model_id = model.split("/", 1)[-1] if "/" in model else model
    kwargs = dict(
        model=model_id, temperature=temperature, reasoning_effort=reasoning_effort
    )

    if model_id.startswith("claude"):
        mod = importlib.import_module("harness.adapters.anthropic")
        return mod.AnthropicAdapter(**kwargs)
    elif model_id.startswith(("gpt", "o1", "o3", "o4")):
        mod = importlib.import_module("harness.adapters.openai")
        return mod.OpenAIAdapter(**kwargs)
    elif model_id.startswith("gemini"):
        mod = importlib.import_module("harness.adapters.google")
        return mod.GoogleAdapter(**kwargs)
    else:
        raise ValueError(
            f"Can't determine provider for model: {model}. "
            "Model name should start with claude, gpt, o1/o3/o4, or gemini."
        )


def _run_harvey_lab_agent(
    model: str,
    instruction: str,
    documents_dir: Path,
    output_dir: Path,
    workspace_dir: Path,
    session_id: str,
    max_turns: int = 200,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
) -> dict:
    """Run the Harvey LAB agent loop and emit ACP updates for trajectory.

    Returns the agent loop result dict.
    """
    sys.path.insert(0, str(_HARVEY_LABS_ROOT))

    # Monkey-patch sandbox module so Harvey LAB's tools.py imports our
    # DirectSandbox instead of the real Podman-backed one.
    _patch_sandbox_module()

    tools_mod = importlib.import_module("harness.tools")
    ToolExecutor = tools_mod.ToolExecutor
    get_all_tool_definitions = tools_mod.get_all_tool_definitions

    # Set up workspace layout
    workspace_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Symlink documents into workspace if not already there
    ws_docs = workspace_dir / "documents"
    if not ws_docs.exists():
        ws_docs.symlink_to(documents_dir)
    ws_output = workspace_dir / "output"
    if not ws_output.exists():
        ws_output.symlink_to(output_dir)

    # Copy skill scripts into workspace
    skills_dir = _HARVEY_LABS_ROOT / "harness" / "skills"
    if skills_dir.exists():
        ws_skills = workspace_dir / "skills"
        ws_skills.mkdir(parents=True, exist_ok=True)
        for skill_dir in skills_dir.iterdir():
            scripts_dir = skill_dir / "scripts"
            if scripts_dir.exists():
                dest = ws_skills / skill_dir.name / "scripts"
                if not dest.exists():
                    shutil.copytree(scripts_dir, dest, dirs_exist_ok=True)

    # Create adapter and tool executor
    adapter = _create_adapter(model, temperature, reasoning_effort)

    sandbox = DirectSandbox(
        documents_dir=documents_dir,
        output_dir=output_dir,
        workspace_dir=workspace_dir,
    )

    tool_executor = ToolExecutor(sandbox=sandbox)
    tools = get_all_tool_definitions()
    system_prompt = _load_system_prompt()

    # Run the agent loop with ACP update emission
    messages = [
        adapter.make_system_message(system_prompt),
        adapter.make_user_message(instruction),
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    turn_count = 0
    start_time = time.time()
    tool_call_counter = 0

    for turn in range(max_turns):
        turn_count = turn + 1

        try:
            response = adapter.chat(messages, tools)
        except Exception as e:
            err_msg = str(e)
            if "prompt is too long" in err_msg or "context_length_exceeded" in err_msg:
                _emit_text(session_id, f"[Context overflow on turn {turn_count}]")
                break
            raise

        messages.append(response.message)
        total_input_tokens += response.input_tokens
        total_output_tokens += response.output_tokens

        # Emit text content as agent message
        if response.text:
            _emit_text(session_id, response.text)

        # If no tool calls, agent is done
        if not response.tool_calls:
            break

        # Execute tools and emit ACP updates
        tool_results = []
        for tc in response.tool_calls:
            tool_call_counter += 1
            tool_call_id = tc.id or f"tc_{tool_call_counter}"

            # Emit tool_call start
            _emit_tool_call(session_id, tool_call_id, tc.name, tc.arguments)

            result = tool_executor.execute(tc.name, tc.arguments)

            # Emit tool_call completion
            _emit_tool_result(session_id, tool_call_id, result)

            tool_results.append((tool_call_id, result))

        # Feed results back via adapter
        result_messages = adapter.make_tool_result_messages(tool_results)
        messages.extend(result_messages)

    elapsed = time.time() - start_time

    return {
        "turn_count": turn_count,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "wall_clock_seconds": round(elapsed, 2),
        "tool_metrics": tool_executor.get_metrics(),
        "tool_call_count": tool_call_counter,
    }


# ── ACP notification helpers ──────────────────────────────────────────────────


def _emit_text(session_id: str, text: str):
    """Emit agent text as ACP agent_message_chunk."""
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


def _emit_tool_call(session_id: str, tool_call_id: str, name: str, arguments: str):
    """Emit ACP tool_call notification."""
    kind = {
        "bash": "bash",
        "read": "read",
        "write": "write",
        "edit": "write",
        "glob": "search",
        "grep": "search",
    }.get(name, "other")

    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": tool_call_id,
                    "title": name,
                    "kind": kind,
                    "status": "in_progress",
                    "input": arguments[:_DIAG_TRUNCATE] if arguments else "",
                },
            },
        }
    )


def _emit_tool_result(session_id: str, tool_call_id: str, result: str):
    """Emit ACP tool_call_update with completion status."""
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


# ── Main ACP loop ─────────────────────────────────────────────────────────────


def main():
    session_id = "harvey-lab-shim"
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
                        "agentInfo": {
                            "name": "harvey-lab-harness",
                            "version": "1.0",
                        },
                    },
                }
            )

        elif method == "session/new":
            cwd = params.get("cwd", "/app")
            session_id = f"harvey-lab-{uuid.uuid4().hex[:8]}"
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

            # Determine workspace layout from BenchFlow's task directory.
            # Harvey LAB's Dockerfile copies docs to /app/documents/.
            app_dir = Path(cwd)
            documents_dir = app_dir / "documents"
            if not documents_dir.exists():
                documents_dir = app_dir / "environment" / "documents"
            if not documents_dir.exists():
                documents_dir = app_dir

            output_dir = app_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            workspace_dir = app_dir / "workspace"
            workspace_dir.mkdir(parents=True, exist_ok=True)

            stop_reason = "end_turn"
            try:
                result = _run_harvey_lab_agent(
                    model=model,
                    instruction=text,
                    documents_dir=documents_dir,
                    output_dir=output_dir,
                    workspace_dir=workspace_dir,
                    session_id=session_id,
                )
                _emit_text(
                    session_id,
                    f"[Harvey LAB agent completed: {result['turn_count']} turns, "
                    f"{result['tool_call_count']} tool calls, "
                    f"{result['wall_clock_seconds']}s]",
                )
            except Exception as e:
                _emit_text(session_id, f"[Harvey LAB agent error: {e}]")
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

        else:
            # Unknown method — return empty result
            if req_id is not None:
                send({"jsonrpc": "2.0", "id": req_id, "result": {}})


if __name__ == "__main__":
    main()
