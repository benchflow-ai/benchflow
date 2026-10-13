#!/usr/bin/env python3
"""ACP shim for Meta's Muse Code CLI — wraps ``muse exec`` as an ACP server.

Muse Code ships no ACP mode of its own; its headless surface is ``muse exec``,
one prompt per process, with ``--session-id`` continuing an earlier session.
This shim speaks ACP (JSON-RPC 2.0 over stdio) to BenchFlow and runs one
``muse exec`` per ACP prompt:

    benchflow ACP client <-stdio/JSON-RPC-> this shim <-argv, files-> muse exec

Architecture:
  - ``initialize``      checks the pinned ``muse`` binary and the subscription
                        login (``~/.config/muse/auth.json``, written from
                        ``MUSE_AUTH_JSON``). ``META_API_KEY`` is always withheld
                        from Muse, which would otherwise prefer it over the login
                        and bill pay-as-you-go.
  - ``session/new``     records the workspace and picks the Muse session id.
  - ``session/set_model`` / ``session/set_config_option`` (``effort``, and
                        ``model``) select the model and reasoning effort.
                        Contributor model ids (``*-contributor``, whose content
                        Meta may train on) are refused, and every run passes
                        ``--contributor-mode off``.
  - ``session/prompt``  runs ``muse exec --json --session-id <id>`` and tails the
                        session log Muse writes (``$XDG_DATA_HOME/muse/sessions/
                        <date>/<id>/session.jsonl``). The log, not the ``--json``
                        stream, carries the tool calls with their arguments, so
                        its records become ACP ``session/update`` notifications
                        (message and thought chunks, tool calls with ``rawInput``
                        / ``rawOutput``, plans). Token usage is summed over the
                        session's own model calls and those of its sub-agent and
                        reminder sessions, and returned with the turn.
  - Quota               a subscription answers ``429 Subscription quota
                        exhausted. Your usage window resets at <time>``. With
                        ``BENCHFLOW_MUSE_QUOTA_WAIT_SEC`` > 0 and the reset within
                        that bound, the turn waits for the reset (one ``other``
                        tool call that stays in progress, refreshed every two
                        minutes) and then continues the same session with a
                        one-line note; otherwise the prompt fails and names the
                        reset time.
  - ``session/cancel``  interrupts the running turn (SIGINT to the process
                        group, SIGKILL fallback) or a quota wait.

The shim is deployed into the sandbox by ``registry.py`` (``install_cmd``) and
must stay dependency-free: standard library only, Python >= 3.8.
"""

import contextlib
import datetime
import glob
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

MUSE_BIN_ENV = "BENCHFLOW_MUSE_BIN"
DEFAULT_MUSE_BIN = "/opt/benchflow/muse/muse"
MODEL_ENV = "MUSE_MODEL"
EFFORT_ENV = "MUSE_EFFORT"
AUTH_JSON_ENV = "MUSE_AUTH_JSON"
API_KEY_ENV = "META_API_KEY"
QUOTA_WAIT_ENV = "BENCHFLOW_MUSE_QUOTA_WAIT_SEC"
QUOTA_SLACK_ENV = "BENCHFLOW_MUSE_QUOTA_SLACK_SEC"
# Launch flags appended by the registry when a BenchFlow web policy is active.
NO_WEB_FLAG = "--disable-web-tools"
NO_HOSTED_SEARCH_FLAG = "--disable-hosted-search"

# Muse's effort tiers. Meta's API refuses ``none``; ``ultra`` is client-side and
# runs at ``xhigh`` where a provider lacks it. BenchFlow's normalized efforts
# map onto these unchanged except ``none``.
EFFORT_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max", "ultra")
_EFFORT_ALIASES = {"none": "minimal"}
_MODELS_DEV_PREFIXES = ("meta/", "muse/")

_TOOL_OUTPUT_LIMIT = 32_000  # chars of tool output forwarded per call
_TOOL_TITLE_LIMIT = 200
_POLL_SEC = 0.25
_QUOTA_HEARTBEAT_SEC = 120
_QUOTA_RESET_SLACK_SEC = 30
_QUOTA_MAX_WAITS_PER_TURN = 4
_QUOTA_RE = re.compile(
    r"Subscription quota exhausted\.\s*Your usage window resets at\s+"
    r"(?P<at>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z)"
)
CONTINUE_AFTER_QUOTA = (
    "The previous model request stopped on the provider's usage limit, which has "
    "now reset. Continue the task from where you left off."
)

# Muse tool name -> ACP ToolKind.
_TOOL_KINDS = {
    "bash": "execute",
    "bash_input": "execute",
    "code_exec": "execute",
    "code_wait": "execute",
    "read_file": "read",
    "search": "search",
    "tool_search": "search",
    "write_file": "edit",
    "edit_file": "edit",
    "apply_patch": "edit",
    "web_fetch": "fetch",
    "web_search": "fetch",
    "write_todos": "think",
    "update_plan": "think",
}
_TITLE_PARAMS = (
    "command",
    "path",
    "file_path",
    "pattern",
    "query",
    "url",
    "name",
    "description",
    "objective",
    "prompt",
)

_stdout_lock = threading.Lock()


# ── model / effort helpers ────────────────────────────────────────────────────


def normalize_model(model):
    """Strip a models.dev style prefix; refuse contributor ids."""
    bare = (model or "").strip()
    for prefix in _MODELS_DEV_PREFIXES:
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
    if bare.endswith("-contributor"):
        raise ValueError(
            f"Refusing contributor model id {bare!r}: Meta may train on the content "
            "of contributor-mode runs. Use the standard id."
        )
    return bare


def normalize_effort(value):
    """Map a BenchFlow effort label onto one of Muse's tiers."""
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    key = _EFFORT_ALIASES.get(key, key)
    if key not in EFFORT_LEVELS:
        raise ValueError(
            f"Unsupported reasoning effort {value!r} for Muse Code; "
            f"expected one of {', '.join(EFFORT_LEVELS)}"
        )
    return key


def parse_quota_reset(text):
    """Return the reset time of a subscription quota error as epoch seconds, else None."""
    match = _QUOTA_RE.search(text or "")
    if not match:
        return None
    stamp = match.group("at").replace("Z", "+00:00")
    if "." in stamp:
        head, tail = stamp.split(".", 1)
        stamp = head + tail[tail.index("+") :]
    try:
        return datetime.datetime.fromisoformat(stamp).timestamp()
    except ValueError:
        return None


# ── stdio plumbing ────────────────────────────────────────────────────────────


def send(msg):
    with _stdout_lock:
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()


def log(msg):
    sys.stderr.write(f"[muse-acp-shim] {msg}\n")
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


# ── credentials and paths ─────────────────────────────────────────────────────


def agent_home():
    return os.environ.get("BENCHFLOW_AGENT_HOME") or os.environ.get(
        "HOME", os.path.expanduser("~")
    )


def config_home():
    return os.environ.get("XDG_CONFIG_HOME") or os.path.join(agent_home(), ".config")


def data_home():
    return os.environ.get("XDG_DATA_HOME") or os.path.join(
        agent_home(), ".local", "share"
    )


def auth_path():
    return os.path.join(config_home(), "muse", "auth.json")


def muse_binary():
    return os.environ.get(MUSE_BIN_ENV) or DEFAULT_MUSE_BIN


def prepare_credentials():
    """Write MUSE_AUTH_JSON to the login file Muse reads; return the auth mode.

    BenchFlow normally writes the file itself (registry ``credential_files``);
    this covers launches that only carry the variable. The variable is removed
    from the environment either way, so no Muse process or agent shell
    inherits it. Returns ``"subscription"``, or ``None`` without a login.
    """
    raw = os.environ.pop(AUTH_JSON_ENV, "")
    path = auth_path()
    if raw.strip() and not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(raw)
    if os.path.isfile(path):
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)
        return "subscription"
    return None


def muse_env(auth_mode):
    env = dict(os.environ)
    env.pop(AUTH_JSON_ENV, None)
    # Muse prefers META_API_KEY over the login file: a stray key would move the
    # run off the subscription onto pay-as-you-go billing.
    env.pop(API_KEY_ENV, None)
    env["MUSE_NO_AUTO_UPDATE"] = "1"
    env["TBH_CREDENTIAL_BACKEND"] = "file"
    env.setdefault("NO_COLOR", "1")
    return env


# ── session log → ACP ─────────────────────────────────────────────────────────


class LogTail:
    """Incremental reader of one JSONL file that may still be growing."""

    def __init__(self, path):
        self.path = path
        self.offset = 0
        self._rest = b""

    def read(self):
        try:
            with open(self.path, "rb") as handle:
                handle.seek(self.offset)
                chunk = handle.read()
        except OSError:
            return []
        if not chunk:
            return []
        self.offset += len(chunk)
        data = self._rest + chunk
        lines = data.split(b"\n")
        self._rest = lines.pop()
        records = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line.decode("utf-8", "replace")))
            except ValueError:
                continue
        return records


def run_event(record):
    """The ``payload.event`` of a ``runtime.session`` run record, else None."""
    payload = record.get("payload") if isinstance(record, dict) else None
    if not isinstance(payload, dict) or payload.get("kind") != "run":
        return None
    event = payload.get("event")
    return event if isinstance(event, dict) else None


def _truncate(text, limit=_TOOL_OUTPUT_LIMIT):
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def _text_content(text):
    return [{"type": "content", "content": {"type": "text", "text": text}}]


def _parse_args(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            return {"input": raw}
        return value if isinstance(value, dict) else {"input": value}
    return {}


def _skill_name(name, params):
    if name == "read_skill" and isinstance(params, dict):
        for key in ("name", "skill", "id"):
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _tool_title(name, params):
    if isinstance(params, dict):
        for key in _TITLE_PARAMS:
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                return f"{name}: {value.strip()}"[:_TOOL_TITLE_LIMIT]
    return name


def _result_failed(text):
    """Whether a tool result reports the call itself failing (not a nonzero exit)."""
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return False
    if not isinstance(value, dict):
        return False
    if value.get("error"):
        return True
    status = str(value.get("terminal_status") or "").lower()
    return status in ("failed", "error", "rejected", "denied")


class Usage:
    """Token counters summed over model_completed records."""

    FIELDS = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens")

    def __init__(self):
        self.totals = dict.fromkeys(self.FIELDS, 0)
        self.calls = 0

    def add(self, usage):
        if not isinstance(usage, dict):
            return
        self.calls += 1
        cached = usage.get("cached_tokens") or usage.get("cache_read_tokens") or 0
        values = {
            "input_tokens": usage.get("input_tokens") or 0,
            "cached_tokens": cached,
            "output_tokens": usage.get("output_tokens") or 0,
            "reasoning_tokens": usage.get("reasoning_tokens") or 0,
        }
        for key, value in values.items():
            if isinstance(value, (int, float)):
                self.totals[key] += int(value)

    def to_acp(self):
        if not self.calls:
            return None
        t = self.totals
        # Muse reports OpenAI-style counts: input includes the cached prefix and
        # output includes reasoning.
        return {
            "inputTokens": t["input_tokens"],
            "cachedReadTokens": t["cached_tokens"],
            "outputTokens": t["output_tokens"],
            "thoughtTokens": t["reasoning_tokens"],
            "totalTokens": t["input_tokens"] + t["output_tokens"],
        }


# ── muse session ──────────────────────────────────────────────────────────────


class MuseExited(RuntimeError):
    """muse exec ended without a usable terminal state."""


class QuotaExhausted(RuntimeError):
    def __init__(self, message, reset_at):
        super().__init__(message)
        self.reset_at = reset_at


class MuseSession:
    def __init__(self, session_id, cwd, auth_mode, flags):
        self.session_id = session_id
        self.cwd = cwd
        self.auth_mode = auth_mode
        self.flags = flags
        self.muse_session_id = str(uuid.uuid4())
        self.model = None
        self.effort = None
        self.proc = None
        self.turns = 0
        self.cancel_requested = False
        self.prompt_active = False
        self.tails = {}
        self.open_tools = {}
        self.seen_calls = set()
        self._wake = threading.Event()

    # -- configuration --

    def resolved_model(self):
        return normalize_model(self.model or os.environ.get(MODEL_ENV, ""))

    def resolved_effort(self):
        return self.effort or normalize_effort(os.environ.get(EFFORT_ENV))

    def set_model(self, model):
        bare = normalize_model(model)
        if not bare:
            raise ValueError("modelId must be a non-empty string")
        self.model = bare

    def set_effort(self, effort):
        normalized = normalize_effort(effort)
        if normalized is None:
            raise ValueError("effort must be a non-empty string")
        self.effort = normalized

    def config_options(self):
        return [
            {
                "id": "effort",
                "name": "Reasoning effort",
                "category": "thought_level",
                "type": "select",
                "currentValue": self.resolved_effort() or "high",
                "options": [
                    {"value": level, "name": level.capitalize()}
                    for level in EFFORT_LEVELS
                ],
            }
        ]

    # -- argv --

    def argv(self, prompt_file):
        args = [
            muse_binary(),
            "exec",
            "--json",
            "--yolo",
            "--user-input-auto-resolve",
            "--no-foreign-personal-context",
            "--contributor-mode",
            "off",
            "--session-id",
            self.muse_session_id,
            "--prompt-file",
            prompt_file,
            "-c",
            "telemetry.enabled=false",
        ]
        model = self.resolved_model()
        if model:
            args += ["--model", model]
        effort = self.resolved_effort()
        if effort:
            args += ["--reasoning-effort", effort]
        if (
            NO_WEB_FLAG in self.flags
            or os.environ.get("BENCHFLOW_DISALLOW_WEB_TOOLS") == "1"
        ):
            args.append("--disable-web-tools")
        elif (
            NO_HOSTED_SEARCH_FLAG in self.flags
            or os.environ.get("BENCHFLOW_EGRESS_DENYLIST") == "1"
        ):
            # web_search runs on Meta's side, out of the egress proxy's reach;
            # web_fetch fetches from the sandbox through the proxy and stays.
            args += ["-c", 'tools.web_search.mode="off"']
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if proxy:
            args += ["-c", "endpoint_transport.proxy=" + json.dumps(proxy)]
        ca_bundle = os.environ.get("SSL_CERT_FILE")
        if ca_bundle and os.path.isfile(ca_bundle):
            args += ["-c", "endpoint_transport.ca_bundle=" + json.dumps(ca_bundle)]
        return args

    # -- process lifecycle --

    def running(self):
        return self.proc is not None and self.proc.poll() is None

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
            self._signal_group(signal.SIGTERM)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=timeout)
        self.proc = None

    def cancel(self):
        self.cancel_requested = True
        self._wake.set()
        if not self.running():
            return
        log("cancelling the running turn (SIGINT)")
        self._signal_group(signal.SIGINT)

        def _force_kill(proc):
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)

        threading.Thread(target=_force_kill, args=(self.proc,), daemon=True).start()

    # -- session log --

    def _session_dir(self):
        pattern = os.path.join(
            data_home(), "muse", "sessions", "*", "*", "*", self.muse_session_id
        )
        found = glob.glob(pattern)
        return found[0] if found else None

    def _pump_logs(self, usage):
        directory = self._session_dir()
        if directory is None:
            return
        main = os.path.join(directory, "session.jsonl")
        paths = [
            main,
            *sorted(
                glob.glob(os.path.join(directory, "subagent", "*", "session.jsonl"))
            ),
        ]
        for path in paths:
            tail = self.tails.get(path)
            if tail is None:
                tail = self.tails[path] = LogTail(path)
            for record in tail.read():
                event = run_event(record)
                if event is None:
                    continue
                if event.get("kind") == "model_completed":
                    usage.add(event.get("usage"))
                elif path == main:
                    self._handle_event(event)

    def _handle_event(self, event):
        kind = event.get("kind")
        if kind == "assistant_message_committed":
            text = event.get("text")
            if isinstance(text, str) and text:
                _notify(
                    self.session_id,
                    {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": text},
                    },
                )
        elif kind == "reasoning_summary_committed":
            text = event.get("text")
            if isinstance(text, str) and text:
                _notify(
                    self.session_id,
                    {
                        "sessionUpdate": "agent_thought_chunk",
                        "content": {"type": "text", "text": text},
                    },
                )
        elif kind == "assistant_tool_calls_committed":
            for call in event.get("tool_calls") or []:
                self._tool_call(call)
        elif kind == "tool_result_batch_committed":
            for result in event.get("results") or []:
                self._tool_result(result)
        elif kind == "todo_snapshot_updated":
            entries = []
            for item in event.get("items") or []:
                if not isinstance(item, dict):
                    continue
                status = item.get("status")
                if status not in ("pending", "in_progress", "completed"):
                    status = "pending"
                entries.append(
                    {
                        "content": str(item.get("text") or ""),
                        "status": status,
                        "priority": "medium",
                    }
                )
            if entries:
                _notify(self.session_id, {"sessionUpdate": "plan", "entries": entries})

    def _tool_call(self, call):
        if not isinstance(call, dict):
            return
        call_id = call.get("call_id") or call.get("id")
        if not call_id or call_id in self.seen_calls:
            return
        self.seen_calls.add(call_id)
        name = call.get("name") or "tool"
        params = _parse_args(call.get("args"))
        kind = _TOOL_KINDS.get(name, "other")
        title = _tool_title(name, params)
        skill = _skill_name(name, params)
        if skill:
            kind, title = "skill", f"skill: {skill}"
        self.open_tools[call_id] = name
        _notify(
            self.session_id,
            {
                "sessionUpdate": "tool_call",
                "toolCallId": call_id,
                "title": title,
                "kind": kind,
                "status": "in_progress",
                "rawInput": params,
            },
        )

    def _tool_result(self, result):
        if not isinstance(result, dict):
            return
        call_id = result.get("tool_call_id") or result.get("call_id")
        if not call_id:
            return
        self.open_tools.pop(call_id, None)
        text = result.get("text")
        text = text if isinstance(text, str) else json.dumps(text)
        clipped = _truncate(text)
        update = {
            "sessionUpdate": "tool_call_update",
            "toolCallId": call_id,
            "status": "failed" if _result_failed(text) else "completed",
            "rawOutput": clipped,
        }
        if clipped:
            update["content"] = _text_content(clipped)
        _notify(self.session_id, update)

    def _close_open_tools(self, status):
        for call_id in list(self.open_tools):
            _notify(
                self.session_id,
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": call_id,
                    "status": status,
                },
            )
        self.open_tools.clear()

    # -- prompting --

    def run_exec(self, text, usage):
        """Run one ``muse exec``; return its ``run_terminal`` payload."""
        fd, prompt_file = tempfile.mkstemp(prefix="muse-prompt-", suffix=".md")
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        args = self.argv(prompt_file)
        log(
            f"muse exec turn {self.turns + 1}: model={self.resolved_model() or 'default'} "
            f"effort={self.resolved_effort() or 'default'} auth={self.auth_mode} cwd={self.cwd}"
        )
        terminal = {}
        stderr_tail = []

        def pump_stdout(stream):
            for raw in stream:
                line = raw.strip()
                if not line.startswith("{"):
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                payload = record.get("payload")
                if isinstance(payload, dict) and payload.get("kind") == "run_terminal":
                    terminal.clear()
                    terminal.update(payload)

        def pump_stderr(stream):
            for raw in stream:
                line = raw.rstrip("\n")
                if not line:
                    continue
                stderr_tail.append(line)
                del stderr_tail[:-40]
                sys.stderr.write(f"[muse] {line}\n")
                sys.stderr.flush()

        try:
            self.proc = subprocess.Popen(
                args,
                cwd=self.cwd,
                env=muse_env(self.auth_mode),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as exc:
            os.unlink(prompt_file)
            raise MuseExited(f"could not start muse: {exc}") from exc
        threads = [
            threading.Thread(target=pump_stdout, args=(self.proc.stdout,), daemon=True),
            threading.Thread(target=pump_stderr, args=(self.proc.stderr,), daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            while self.proc.poll() is None:
                self._pump_logs(usage)
                time.sleep(_POLL_SEC)
            for thread in threads:
                thread.join(timeout=10)
            # The log may trail the process exit by a few writes.
            for _ in range(3):
                self._pump_logs(usage)
                time.sleep(_POLL_SEC)
        finally:
            code = self.proc.returncode if self.proc else None
            self.proc = None
            with contextlib.suppress(OSError):
                os.unlink(prompt_file)
        self.turns += 1
        if not terminal:
            detail = " | ".join(stderr_tail[-5:])
            raise MuseExited(
                f"muse exited (rc={code}) without a terminal state: {detail}"
            )
        terminal["_stderr"] = "\n".join(stderr_tail)
        return terminal

    def wait_for_quota(self, reset_at):
        """Hold the turn until ``reset_at`` as one in-progress tool call; False if cancelled."""
        call_id = f"muse-quota-wait-{uuid.uuid4().hex[:8]}"
        # datetime.UTC needs Python 3.11; the shim runs on the task image's python3.
        until = datetime.datetime.fromtimestamp(reset_at, datetime.timezone.utc)  # noqa: UP017
        title = f"wait for the Muse usage window to reset at {until:%Y-%m-%dT%H:%M:%SZ}"
        log(title)
        _notify(
            self.session_id,
            {
                "sessionUpdate": "tool_call",
                "toolCallId": call_id,
                "title": title,
                "kind": "other",
                "status": "in_progress",
                "rawInput": {"reset_at": until.strftime("%Y-%m-%dT%H:%M:%SZ")},
            },
        )
        self._wake.clear()
        slack = _QUOTA_RESET_SLACK_SEC
        with contextlib.suppress(ValueError):
            slack = float(os.environ.get(QUOTA_SLACK_ENV) or slack)
        deadline = reset_at + slack
        while time.time() < deadline and not self.cancel_requested:
            self._wake.wait(
                timeout=min(_QUOTA_HEARTBEAT_SEC, max(1.0, deadline - time.time()))
            )
            if self.cancel_requested:
                break
            _notify(
                self.session_id,
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": call_id,
                    "status": "in_progress",
                },
            )
        status = "cancelled" if self.cancel_requested else "completed"
        _notify(
            self.session_id,
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": call_id,
                "status": "failed" if status == "cancelled" else "completed",
                "rawOutput": status,
            },
        )
        return not self.cancel_requested

    def prompt(self, text):
        """Run one ACP turn; return ``(stop_reason, usage)``."""
        self.cancel_requested = False
        usage = Usage()
        wait_cap = 0.0
        with contextlib.suppress(ValueError):
            wait_cap = float(os.environ.get(QUOTA_WAIT_ENV) or 0)
        waits = 0
        while True:
            terminal = self.run_exec(text, usage)
            state = str(terminal.get("terminal") or "").lower()
            if state == "completed":
                return "end_turn", usage
            if (
                state in ("cancelled", "canceled", "interrupted")
                or self.cancel_requested
            ):
                self._close_open_tools("cancelled")
                return "cancelled", usage
            self._close_open_tools("failed")
            message = (
                " ".join(
                    str(part)
                    for part in (terminal.get("reason"), terminal.get("text"))
                    if part
                )
                or terminal.get("_stderr", "")
                or f"muse run ended {state or 'without a state'}"
            )
            reset_at = parse_quota_reset(f"{message}\n{terminal.get('_stderr', '')}")
            if reset_at is None:
                raise MuseExited(f"muse run {state or 'failed'}: {message[:2000]}")
            remaining = reset_at - time.time()
            if waits >= _QUOTA_MAX_WAITS_PER_TURN or remaining > wait_cap:
                stamp = datetime.datetime.fromtimestamp(reset_at, datetime.timezone.utc)  # noqa: UP017
                raise QuotaExhausted(
                    "Muse subscription quota exhausted; the usage window resets at "
                    f"{stamp:%Y-%m-%dT%H:%M:%SZ}",
                    reset_at,
                )
            waits += 1
            if not self.wait_for_quota(reset_at):
                return "cancelled", usage
            text = CONTINUE_AFTER_QUOTA


# ── ACP server ────────────────────────────────────────────────────────────────


class Server:
    def __init__(self, flags=()):
        self.flags = tuple(flags)
        self.sessions = {}
        self.auth_mode = None

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
        binary = muse_binary()
        if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
            _fail(req_id, f"Muse Code binary not found or not executable: {binary}")
            return
        self.auth_mode = prepare_credentials()
        if self.auth_mode is None:
            _fail(
                req_id,
                "No Muse Code login: set MUSE_AUTH_JSON to the contents of the "
                "auth.json that `TBH_CREDENTIAL_BACKEND=file muse login` writes",
            )
            return
        version = ""
        try:
            probe = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                text=True,
                timeout=60,
                env=muse_env(self.auth_mode),
            )
            version = (
                (probe.stdout or "").strip().splitlines()[0] if probe.stdout else ""
            )
        except Exception as exc:  # pragma: no cover - depends on sandbox
            log(f"muse --version probe failed: {exc!r}")
        log(
            f"{version or 'muse'} ready; auth={self.auth_mode}; flags={list(self.flags)}"
        )
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
                    "name": "muse-code",
                    "version": version or "unknown",
                    "title": "Meta Muse Code",
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
        session = MuseSession(session_id, cwd, self.auth_mode, self.flags)
        self.sessions[session_id] = session
        if params.get("mcpServers"):
            log("ignoring session/new mcpServers (not wired for Muse Code)")
        _respond(
            req_id, {"sessionId": session_id, "configOptions": session.config_options()}
        )

    def on_set_model(self, req_id, params):
        session = self._session(params)
        if session.prompt_active:
            _fail(req_id, "cannot change the model while a prompt is running")
            return
        session.set_model(params.get("modelId", ""))
        log(f"model set to {session.resolved_model()}")
        _respond(req_id, {})

    def on_set_config_option(self, req_id, params):
        session = self._session(params)
        config_id = params.get("configId")
        if config_id not in ("effort", "model"):
            _fail(
                req_id, f"Unsupported session config option: {config_id!r}", code=-32602
            )
            return
        if session.prompt_active:
            _fail(req_id, f"cannot change the {config_id} while a prompt is running")
            return
        if config_id == "model":
            session.set_model(params.get("value"))
        else:
            session.set_effort(params.get("value"))
        log(f"{config_id} set to {params.get('value')}")
        _respond(req_id, {"configOptions": session.config_options()})

    def on_prompt(self, req_id, params):
        session = self._session(params)
        if session.prompt_active:
            _fail(req_id, "a prompt is already in progress on this session")
            return
        text = _prompt_text(params.get("prompt"))
        session.prompt_active = True
        threading.Thread(
            target=self._run_prompt, args=(req_id, session, text), daemon=True
        ).start()

    def _run_prompt(self, req_id, session, text):
        # Release the session before responding: the client may send its next
        # request the moment it reads the response.
        try:
            stop_reason, usage = session.prompt(text)
        except QuotaExhausted as exc:
            session.prompt_active = False
            _fail(req_id, str(exc))
            return
        except MuseExited as exc:
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
        response = {"stopReason": stop_reason}
        acp_usage = usage.to_acp()
        if acp_usage:
            response["usage"] = acp_usage
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
            session.cancel()
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


def main(argv=None):
    flags = [
        arg for arg in (sys.argv[1:] if argv is None else argv) if arg.startswith("--")
    ]
    Server(flags).run()


if __name__ == "__main__":
    main()
