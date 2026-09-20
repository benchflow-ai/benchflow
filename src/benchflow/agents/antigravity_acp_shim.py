#!/usr/bin/env python3
"""ACP shim for the Google Antigravity CLI — wraps ``agy`` as an ACP server.

The Antigravity CLI (``agy``, the successor of the hosted Gemini CLI) ships no
ACP mode of its own; its headless surface is an NDJSON ``stream-json``
protocol over stdin/stdout. This shim speaks ACP (JSON-RPC 2.0 over stdio) to
BenchFlow and drives ``agy`` through that headless protocol:

    benchflow ACP client <-stdio/JSON-RPC-> this shim <-pipes/NDJSON-> agy

Architecture:
  - ``initialize``      validates the pinned ``agy`` binary and prepares
                        ``~/.gemini/antigravity-cli/settings.json`` (Gemini
                        API-key mode, telemetry off, terminal sandbox off).
  - ``session/new``     records the workspace; ``agy`` is spawned lazily on the
                        first prompt with ``--add-dir=<cwd>`` so workspace
                        skills under ``<cwd>/.agents/skills`` are discovered.
  - ``session/set_model`` / ``session/set_config_option`` (``thinking``)
                        select the model and reasoning effort. ``agy`` requires
                        an explicit ``--effort`` for every model in API-key
                        mode; ids such as ``gemini-3.8-flash-high`` carry their
                        effort as a suffix and are split.
  - ``session/prompt``  writes one ``user`` event, streams ``step_update``
                        events back as ACP ``session/update`` notifications
                        (message chunks, tool calls with ``rawInput`` /
                        ``rawOutput``), and returns the turn's token usage.
  - ``session/cancel``  interrupts the running turn (SIGINT to the process
                        group, SIGKILL fallback).

The shim is deployed into the sandbox by ``registry.py`` (``install_cmd``) and
must stay dependency-free: standard library only, Python >= 3.8.
"""

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import uuid
from pathlib import Path

AGY_BIN_ENV = "BENCHFLOW_AGY_BIN"
DEFAULT_AGY_BIN = "/opt/benchflow/antigravity/agy"
MODEL_ENV = "ANTIGRAVITY_MODEL"
EFFORT_ENV = "ANTIGRAVITY_EFFORT"
EXTRA_ARGS_ENV = "ANTIGRAVITY_EXTRA_ARGS"

EFFORT_LEVELS = ("low", "medium", "high")
DEFAULT_EFFORT = "high"
# BenchFlow's normalized reasoning efforts (none/minimal/low/medium/high/
# xhigh/max) collapse onto agy's three levels. Anything above ``high`` maps
# to ``high`` — the strongest level the CLI accepts.
_EFFORT_ALIASES = {
    "none": "low",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}
_MODEL_EFFORT_SUFFIX = re.compile(r"^(?P<model>.+?)-(?P<effort>low|medium|high)$")
_MODELS_DEV_PREFIXES = ("google/", "gemini/")

_TOOL_OUTPUT_LIMIT = 32_000  # chars of tool output forwarded per call
_TOOL_TITLE_LIMIT = 200
_THOUGHT_STEP_TYPES = frozenset({"thinking", "thought", "planning", "reasoning"})
# agy tool name -> ACP ToolKind.
_TOOL_KINDS = {
    "run_command": "execute",
    "send_command_input": "execute",
    "command_status": "execute",
    "read_terminal": "execute",
    "notebook_execution": "execute",
    "view_file": "read",
    "view_code_item": "read",
    "view_content_chunk": "read",
    "list_dir": "read",
    "read_resource": "read",
    "list_resources": "read",
    "grep_search": "search",
    "find_by_name": "search",
    "codebase_search": "search",
    "search_web": "fetch",
    "google_search": "fetch",
    "read_url_content": "fetch",
    "open_browser_url": "fetch",
    "write_to_file": "edit",
    "replace_file_content": "edit",
    "multi_replace_file_content": "edit",
    "sed_file": "edit",
    "notebook_edit": "edit",
}
_TITLE_PARAMS = (
    "CommandLine",
    "AbsolutePath",
    "TargetFile",
    "DirectoryPath",
    "SearchDirectory",
    "SearchPath",
    "Query",
    "query",
    "Pattern",
    "Url",
    "url",
    "Path",
    "File",
)
_PATH_PARAMS = ("AbsolutePath", "TargetFile", "DirectoryPath", "SearchDirectory")

_stdout_lock = threading.Lock()

# ── model / effort helpers (mirrored in benchflow.agents.antigravity_config) ──


def split_model_effort(model):
    """Split ``gemini-3.8-flash-high`` into ``("gemini-3.8-flash", "high")``.

    Bare ids come back with ``None`` effort. models.dev style ``google/`` and
    ``gemini/`` prefixes are dropped — agy takes bare Google model ids.
    """
    bare = (model or "").strip()
    for prefix in _MODELS_DEV_PREFIXES:
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
    match = _MODEL_EFFORT_SUFFIX.match(bare)
    if match:
        return match.group("model"), match.group("effort")
    return bare, None


def normalize_effort(value):
    """Map a BenchFlow/agy effort label onto ``low`` | ``medium`` | ``high``."""
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    if key not in _EFFORT_ALIASES:
        raise ValueError(
            f"Unsupported reasoning effort {value!r} for Antigravity; "
            f"expected one of {', '.join(_EFFORT_ALIASES)}"
        )
    return _EFFORT_ALIASES[key]


# ── stdio plumbing ────────────────────────────────────────────────────────────


def send(msg):
    with _stdout_lock:
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()


def log(msg):
    sys.stderr.write(f"[antigravity-acp-shim] {msg}\n")
    sys.stderr.flush()


def _respond(req_id, result):
    send({"jsonrpc": "2.0", "id": req_id, "result": result})


def _fail(req_id, message, code=-32603):
    send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def _notify(session_id, update):
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": session_id, "update": update},
        }
    )


# ── agy home preparation ──────────────────────────────────────────────────────


def agent_home():
    return os.environ.get("BENCHFLOW_AGENT_HOME") or os.environ.get(
        "HOME", os.path.expanduser("~")
    )


def agy_binary():
    return os.environ.get(AGY_BIN_ENV) or DEFAULT_AGY_BIN


def prepare_settings(home=None):
    """Write ``~/.gemini/antigravity-cli/settings.json`` for headless runs.

    ``modelProvider: "gemini"`` switches agy to Gemini API-key mode (no Google
    sign-in) and is only written when a key is present, so an existing
    sign-in configuration is never clobbered. The remaining keys are
    ``setdefault``-merged: telemetry off, agy's own terminal sandbox off
    (BenchFlow's sandbox owns isolation; nested sandboxes fail inside
    Docker/Daytona), non-workspace paths readable (tasks keep inputs under
    ``/logs``, ``/tests`` and friends), and every review gate set to proceed.
    Returns the settings path, or ``None`` when the file could not be written.
    """
    base = Path(home or agent_home()) / ".gemini" / "antigravity-cli"
    path = base / "settings.json"
    try:
        base.mkdir(parents=True, exist_ok=True)
        data = {}
        if path.exists():
            raw = path.read_text().strip()
            if raw:
                data = json.loads(raw)
                if not isinstance(data, dict):
                    data = {}
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            data["modelProvider"] = "gemini"
        data.setdefault("enableTelemetry", "off")
        data.setdefault("enableTerminalSandbox", "off")
        data.setdefault("allowNonWorkspaceAccess", "on")
        data.setdefault("toolPermission", "always-proceed")
        data.setdefault("artifactReviewPolicy", "always-proceed")
        data.setdefault("agentMode", "accept-edits")
        path.write_text(json.dumps(data, indent=2) + "\n")
        return path
    except Exception as exc:  # pragma: no cover - depends on sandbox FS state
        log(f"could not prepare {path}: {exc!r}")
        return None


# ── event mapping ─────────────────────────────────────────────────────────────


def _skill_name_from_params(name, params):
    """Return the skill name when a tool call is agy loading a skill.

    agy has no dedicated skill tool: skills are progressively disclosed and a
    skill is *used* when the agent reads ``.../skills/<name>/SKILL.md`` with
    ``view_file``. Reporting that read with ACP kind ``skill`` is what lets
    BenchFlow count skill invocations for this harness the way it does for
    harnesses with an explicit skill tool.
    """
    if name != "view_file" or not isinstance(params, dict):
        return None
    path = params.get("AbsolutePath")
    if not isinstance(path, str):
        return None
    parts = path.rstrip("/").split("/")
    if len(parts) >= 3 and parts[-1] == "SKILL.md" and parts[-3] == "skills":
        return parts[-2]
    return None


def _tool_title(name, params):
    if isinstance(params, dict):
        for key in _TITLE_PARAMS:
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                return f"{name}: {value.strip()}"[:_TOOL_TITLE_LIMIT]
        for value in params.values():
            if isinstance(value, str) and value.strip():
                return f"{name}: {value.strip()}"[:_TOOL_TITLE_LIMIT]
    return name


def _tool_locations(params):
    if not isinstance(params, dict):
        return []
    locations = []
    for key in _PATH_PARAMS:
        value = params.get(key)
        if isinstance(value, str) and value.startswith("/"):
            locations.append({"path": value})
    return locations


def _truncate(text, limit=_TOOL_OUTPUT_LIMIT):
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def _text_content(text):
    return [{"type": "content", "content": {"type": "text", "text": text}}]


def usage_to_acp(usage):
    """Project agy's ``result.usage`` onto ACP ``PromptResponse.usage``."""
    if not isinstance(usage, dict):
        return None
    mapping = {
        "inputTokens": "input_tokens",
        "outputTokens": "output_tokens",
        "totalTokens": "total_tokens",
        "cachedReadTokens": "cache_read_tokens",
        "thoughtTokens": "thinking_tokens",
    }
    out = {}
    for acp_key, agy_key in mapping.items():
        value = usage.get(agy_key)
        if isinstance(value, (int, float)):
            out[acp_key] = int(value)
    return out or None


# ── agy session ───────────────────────────────────────────────────────────────


class AgyExited(RuntimeError):
    """agy terminated before the turn produced a ``result`` event."""


class AgySession:
    def __init__(self, session_id, cwd):
        self.session_id = session_id
        self.cwd = cwd
        self.model = None
        self.effort = None
        self.proc = None
        self.conversation_id = None
        self.last_agy_error = ""
        self.cancel_requested = False
        self.prompt_active = False
        self._stderr_thread = None

    # -- configuration --

    def resolved_model(self):
        model = self.model or os.environ.get(MODEL_ENV, "")
        bare, _suffix_effort = split_model_effort(model)
        if not bare:
            raise RuntimeError(
                "No Antigravity model configured: send session/set_model or set "
                f"{MODEL_ENV}"
            )
        return bare

    def resolved_effort(self):
        if self.effort:
            return self.effort
        _bare, suffix_effort = split_model_effort(
            self.model or os.environ.get(MODEL_ENV, "")
        )
        if suffix_effort:
            return suffix_effort
        env_effort = normalize_effort(os.environ.get(EFFORT_ENV))
        return env_effort or DEFAULT_EFFORT

    def set_model(self, model):
        bare, suffix_effort = split_model_effort(model)
        if not bare:
            raise ValueError("modelId must be a non-empty string")
        self.model = bare
        if suffix_effort:
            self.effort = suffix_effort
        self._restart_on_next_prompt()

    def set_effort(self, effort):
        normalized = normalize_effort(effort)
        if normalized is None:
            raise ValueError("thinking effort must be a non-empty string")
        self.effort = normalized
        self._restart_on_next_prompt()

    def config_options(self):
        effort = self.resolved_effort()
        return [
            {
                "id": "thinking",
                "name": "Thinking effort",
                "category": "thought_level",
                "type": "select",
                "currentValue": effort,
                "options": [
                    {"value": level, "name": level.capitalize()}
                    for level in EFFORT_LEVELS
                ],
            }
        ]

    def model_state(self):
        try:
            model = self.resolved_model()
        except RuntimeError:
            return None
        return {
            "availableModels": [{"modelId": model, "name": model}],
            "currentModelId": model,
        }

    # -- process lifecycle --

    def _restart_on_next_prompt(self):
        if self.proc is not None:
            log("model/effort changed; restarting agy for the next prompt")
            self.stop()

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        model = self.resolved_model()
        effort = self.resolved_effort()
        args = [
            agy_binary(),
            "--input-format=stream-json",
            "--output-format=stream-json",
            "--dangerously-skip-permissions",
            "--mode=accept-edits",
            f"--add-dir={self.cwd}",
            f"--model={model}",
            f"--effort={effort}",
        ]
        extra = os.environ.get(EXTRA_ARGS_ENV, "").strip()
        if extra:
            args.extend(extra.split())
        env = dict(os.environ)
        env.setdefault("AGY_CLI_HIDE_LOGO", "1")
        env.setdefault("NO_COLOR", "1")
        log(f"starting agy: model={model} effort={effort} cwd={self.cwd}")
        self.proc = subprocess.Popen(
            args,
            cwd=self.cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.conversation_id = None
        self.cancel_requested = False
        self._stderr_thread = threading.Thread(
            target=self._pump_stderr, args=(self.proc,), daemon=True
        )
        self._stderr_thread.start()

    def _pump_stderr(self, proc):
        try:
            for line in proc.stderr:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith("AGY_ERROR:"):
                    self.last_agy_error = line[len("AGY_ERROR:") :].strip()
                sys.stderr.write(f"[agy] {line}\n")
                sys.stderr.flush()
        except Exception:  # pragma: no cover - pipe torn down on exit
            pass

    def _signal_group(self, sig):
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        except OSError:
            with contextlib.suppress(OSError):
                proc.send_signal(sig)

    def stop(self, timeout=5.0):
        proc = self.proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except OSError:
                pass
            self._signal_group(signal.SIGTERM)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=timeout)
        self.proc = None

    def cancel(self):
        if not self.running():
            return
        self.cancel_requested = True
        log("cancelling the running turn (SIGINT)")
        self._signal_group(signal.SIGINT)

        def _force_kill(proc):
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)

        threading.Thread(target=_force_kill, args=(self.proc,), daemon=True).start()

    # -- prompting --

    def prompt(self, text):
        """Run one turn. Returns agy's ``result`` payload."""
        if not self.running():
            self.start()
        proc = self.proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise AgyExited("agy process is not running")
        stdin, stdout = proc.stdin, proc.stdout
        message = {"event": "user", "message": {"content": text}}
        try:
            stdin.write(json.dumps(message) + "\n")
            stdin.flush()
        except (OSError, ValueError) as exc:
            raise AgyExited(f"could not write the prompt to agy: {exc}") from exc

        open_tools = set()
        for raw in stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                log(f"non-JSON line from agy: {line[:200]}")
                continue
            kind = event.get("event")
            if kind == "init":
                self.conversation_id = event.get("conversation_id") or None
                continue
            if kind == "step_update":
                self._handle_step(event.get("step_update") or {}, open_tools)
                continue
            if kind == "result":
                result = event.get("result") or {}
                if not self.conversation_id:
                    self.conversation_id = result.get("conversation_id") or None
                self._close_open_tools(open_tools, result)
                return result
        # stdout hit EOF: reap the process so the exit code is real, and let
        # the stderr pump drain so a trailing AGY_ERROR line reaches the
        # message instead of racing it.
        try:
            code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            code = proc.poll()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2)
        detail = f" ({self.last_agy_error})" if self.last_agy_error else ""
        raise AgyExited(f"agy exited (rc={code}) before finishing the turn{detail}")

    def _tool_call_id(self, step_index):
        conversation = (self.conversation_id or "agy")[:8]
        return f"{conversation}-{step_index}"

    def _handle_step(self, step, open_tools):
        step_type = step.get("step_type", "")
        state = step.get("state", "")
        text = step.get("text_delta")
        if step_type == "agent_response":
            if text:
                _notify(
                    self.session_id,
                    {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": text},
                    },
                )
            return
        if step_type in _THOUGHT_STEP_TYPES:
            if text:
                _notify(
                    self.session_id,
                    {
                        "sessionUpdate": "agent_thought_chunk",
                        "content": {"type": "text", "text": text},
                    },
                )
            return
        if step_type in ("tool", "tool_call"):
            self._handle_tool_step(step, state, open_tools)
            return
        if step_type == "user_input":
            return
        if text:
            # Unknown textual step: keep it visible rather than dropping it.
            _notify(
                self.session_id,
                {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"type": "text", "text": f"[{step_type}] {text}"},
                },
            )

    def _handle_tool_step(self, step, state, open_tools):
        info = step.get("tool_info") or {}
        name = info.get("name") or step.get("tool_name") or "tool"
        params = info.get("parameters")
        call_id = self._tool_call_id(step.get("step_index", 0))
        acp_kind = _TOOL_KINDS.get(name, "other")
        title = _tool_title(name, params)
        skill_name = _skill_name_from_params(name, params)
        if skill_name:
            acp_kind = "skill"
            title = f"skill: {skill_name}"
        if call_id not in open_tools:
            update = {
                "sessionUpdate": "tool_call",
                "toolCallId": call_id,
                "title": title,
                "kind": acp_kind,
                "status": "in_progress",
            }
            if isinstance(params, dict):
                update["rawInput"] = params
            locations = _tool_locations(params)
            if locations:
                update["locations"] = locations
            _notify(self.session_id, update)
            open_tools.add(call_id)
        if state == "ACTIVE":
            return
        error = info.get("error")
        output = info.get("output")
        if state == "ERROR" or error:
            message = ""
            if isinstance(error, dict):
                message = str(error.get("message") or error.get("type") or "")
            elif error:
                message = str(error)
            if not message:
                message = "tool failed"
            update = {
                "sessionUpdate": "tool_call_update",
                "toolCallId": call_id,
                "status": "failed",
                "content": _text_content(_truncate(message)),
                "rawOutput": {"error": message},
            }
            if output:
                update["rawOutput"]["output"] = _truncate(str(output))
        else:
            text = _truncate(str(output)) if output else ""
            update = {
                "sessionUpdate": "tool_call_update",
                "toolCallId": call_id,
                "status": "completed",
                "rawOutput": text,
            }
            if text:
                update["content"] = _text_content(text)
        _notify(self.session_id, update)
        open_tools.discard(call_id)

    def _close_open_tools(self, open_tools, result):
        status = "cancelled" if result.get("status") == "CANCELED" else "failed"
        for call_id in sorted(open_tools):
            _notify(
                self.session_id,
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": call_id,
                    "status": status,
                },
            )
        open_tools.clear()


# ── ACP server ────────────────────────────────────────────────────────────────


class Server:
    def __init__(self):
        self.sessions = {}
        self._prompt_threads = {}

    def run(self):
        try:
            for raw in sys.stdin:
                line = raw.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    log(f"ignoring non-JSON input: {line[:200]}")
                    continue
                self.dispatch(msg)
        finally:
            for session in list(self.sessions.values()):
                session.stop()

    def dispatch(self, msg):
        method = msg.get("method", "")
        req_id = msg.get("id")
        params = msg.get("params") or {}
        try:
            if method == "initialize":
                self.on_initialize(req_id)
            elif method == "authenticate":
                _respond(req_id, {})
            elif method == "session/new":
                self.on_session_new(req_id, params)
            elif method == "session/set_model":
                self.on_set_model(req_id, params)
            elif method == "session/set_config_option":
                self.on_set_config_option(req_id, params)
            elif method == "session/set_mode":
                _respond(req_id, {})
            elif method == "session/prompt":
                self.on_prompt(req_id, params)
            elif method == "session/cancel":
                self.on_cancel(req_id, params)
            elif method == "session/close":
                self.on_close(req_id, params)
            elif req_id is not None:
                _fail(req_id, f"Method not found: {method}", code=-32601)
        except Exception as exc:
            log(f"{method} failed: {exc!r}")
            if req_id is not None:
                _fail(req_id, f"{method} failed: {exc}")

    def on_initialize(self, req_id):
        binary = agy_binary()
        if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
            _fail(
                req_id,
                f"Antigravity CLI binary not found or not executable: {binary}",
            )
            return
        prepare_settings()
        version = ""
        try:
            probe = subprocess.run(
                [binary, "--version"], capture_output=True, text=True, timeout=30
            )
            version = (
                (probe.stdout or "").strip().splitlines()[0] if probe.stdout else ""
            )
        except Exception as exc:  # pragma: no cover - depends on sandbox
            log(f"agy --version probe failed: {exc!r}")
        _respond(
            req_id,
            {
                "protocolVersion": 1,
                "agentCapabilities": {
                    "loadSession": False,
                    "promptCapabilities": {
                        "image": False,
                        "audio": False,
                        "embeddedContext": True,
                    },
                },
                "agentInfo": {
                    "name": "antigravity",
                    "version": version or "unknown",
                    "title": "Google Antigravity CLI",
                },
                "authMethods": [],
            },
        )

    def _session(self, params):
        session_id = params.get("sessionId", "")
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id!r}")
        return session

    def on_session_new(self, req_id, params):
        cwd = params.get("cwd") or os.getcwd()
        session_id = str(uuid.uuid4())
        session = AgySession(session_id, cwd)
        self.sessions[session_id] = session
        if params.get("mcpServers"):
            # Task MCP servers reach agy through ~/.gemini/config/mcp_config.json
            # (registry task_mcp_transport="native-config"); anything sent over
            # ACP is acknowledged but not wired.
            log("ignoring session/new mcpServers (agy loads mcp_config.json)")
        result = {"sessionId": session_id, "configOptions": session.config_options()}
        model_state = session.model_state()
        if model_state:
            result["models"] = model_state
        _respond(req_id, result)

    def on_set_model(self, req_id, params):
        session = self._session(params)
        if session.prompt_active:
            _fail(req_id, "cannot change the model while a prompt is running")
            return
        session.set_model(params.get("modelId", ""))
        log(
            f"model set to {session.resolved_model()} (effort {session.resolved_effort()})"
        )
        _respond(req_id, {})

    def on_set_config_option(self, req_id, params):
        session = self._session(params)
        config_id = params.get("configId")
        if config_id != "thinking":
            _fail(
                req_id, f"Unsupported session config option: {config_id!r}", code=-32602
            )
            return
        if session.prompt_active:
            _fail(req_id, "cannot change the effort while a prompt is running")
            return
        session.set_effort(params.get("value"))
        log(f"thinking effort set to {session.resolved_effort()}")
        _respond(req_id, {"configOptions": session.config_options()})

    def on_prompt(self, req_id, params):
        session = self._session(params)
        if session.prompt_active:
            _fail(req_id, "a prompt is already in progress on this session")
            return
        text = _prompt_text(params.get("prompt"))
        session.prompt_active = True
        thread = threading.Thread(
            target=self._run_prompt, args=(req_id, session, text), daemon=True
        )
        self._prompt_threads[session.session_id] = thread
        thread.start()

    def _run_prompt(self, req_id, session, text):
        # The session is released *before* the response goes out: the client
        # may send its next request the moment it sees the response, and a
        # still-set prompt_active would reject it as "already in progress".
        try:
            result = session.prompt(text)
        except AgyExited as exc:
            session.stop()
            session.prompt_active = False
            if session.cancel_requested:
                _respond(req_id, {"stopReason": "cancelled"})
            else:
                _fail(req_id, str(exc))
            return
        except Exception as exc:
            session.stop()
            session.prompt_active = False
            _fail(req_id, f"session/prompt failed: {exc}")
            return
        session.prompt_active = False
        status = str(result.get("status") or "").upper()
        usage = usage_to_acp(result.get("usage"))
        if (
            status in ("CANCELED", "CANCELLED", "INTERRUPTED")
            or session.cancel_requested
        ):
            response = {"stopReason": "cancelled"}
            if usage:
                response["usage"] = usage
            _respond(req_id, response)
            return
        if status in ("ERROR", "INVALID"):
            message = str(result.get("error") or session.last_agy_error or status)
            _fail(req_id, f"agy turn ended with {status}: {message}")
            return
        response = {"stopReason": "end_turn"}
        if usage:
            response["usage"] = usage
        _respond(req_id, response)

    def on_cancel(self, req_id, params):
        session = self.sessions.get(params.get("sessionId", ""))
        if session is not None:
            session.cancel()
        if req_id is not None:
            _respond(req_id, {})

    def on_close(self, req_id, params):
        session = self.sessions.pop(params.get("sessionId", ""), None)
        if session is not None:
            session.stop()
        if req_id is not None:
            _respond(req_id, {})


def _prompt_text(prompt):
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list):
        return str(prompt or "")
    parts = []
    for block in prompt:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append(str(block.get("text", "")))
        elif block_type == "resource_link":
            parts.append(f"[{block.get('name') or 'resource'}]({block.get('uri', '')})")
        elif block_type == "resource":
            resource = block.get("resource") or {}
            if isinstance(resource.get("text"), str):
                parts.append(resource["text"])
    return "\n".join(part for part in parts if part)


def main():
    Server().run()


if __name__ == "__main__":
    main()
