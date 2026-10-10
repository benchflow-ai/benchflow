"""The event writer BenchFlow's Python runner scripts share.

Written for BenchFlow. A runner builds an agent from its framework, runs it
once on the instruction, and reports what it does through `Events`: one JSON
object per line on stdout, in the `benchflow-runner` format that BenchFlow's
headless harness parses (crates/benchflow-agent/src/headless/parsers/runner.rs
lists every field).

Frameworks print their own progress to stdout. `Events()` keeps the real
stdout for events and points file descriptor 1 at stderr, so only events
reach the parser and everything else lands in the agent log.

Exit codes: 0 when the agent finished, 1 when it failed, 2 when it could not
start because the user has to give it something (`SetupError`).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

# Tool output longer than this is cut; the trajectory needs the gist, and
# BenchFlow shrinks long lines anyway.
MAX_TEXT = 64 * 1024


class SetupError(Exception):
    """The agent lacks something only the user can give it.

    `message` says what is missing, `fix` what to set or add.
    """

    def __init__(self, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.fix = fix


class Events:
    """Writes `benchflow-runner` events to the original stdout."""

    def __init__(self) -> None:
        sys.stdout.flush()
        events_fd = os.dup(1)
        os.dup2(2, 1)
        self._out = os.fdopen(events_fd, "w", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._usage: dict[str, float] = {}
        self.streamed_message = False

    def emit(self, event_type: str, /, **fields: Any) -> None:
        event = {"type": event_type}
        event.update({k: v for k, v in fields.items() if v is not None})
        line = json.dumps(event, default=jsonable, ensure_ascii=False)
        with self._lock:
            self._out.write(line + "\n")
            self._out.flush()

    def start(self, agent: str, framework: str, version: str | None, model: str | None) -> None:
        self.emit("start", agent=agent, framework=framework, version=version, model=model)

    def message(self, text: str | None) -> None:
        if text:
            self.streamed_message = True
            self.emit("message", text=text)

    def thought(self, text: str | None) -> None:
        if text:
            self.emit("thought", text=text)

    def tool_call(self, call_id: str | None, name: str, arguments: Any = None, title: str | None = None) -> str:
        """Report a tool call starting; returns its id (made up when the framework gave none)."""
        call_id = call_id or f"call_{uuid.uuid4().hex[:12]}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {"input": arguments}
        if arguments is not None and not isinstance(arguments, dict):
            arguments = {"input": arguments}
        self.emit("tool_call", id=call_id, name=name or "tool", input=arguments, title=title)
        return call_id

    def tool_result(self, call_id: str, output: Any, failed: bool = False) -> None:
        self.emit("tool_result", id=call_id, output=text(output), failed=bool(failed))

    def add_usage(self, **counts: float | None) -> None:
        """Add one model call's counts to the run's totals and report the totals."""
        for key, value in counts.items():
            if value:
                self._usage[key] = self._usage.get(key, 0) + value
        self.report_usage()

    def set_usage(self, **counts: float | None) -> None:
        """Replace the totals with counts the framework keeps itself."""
        self._usage = {k: v for k, v in counts.items() if v}
        self.report_usage()

    def report_usage(self) -> None:
        if self._usage:
            fields = {k: (int(v) if k != "cost_usd" else float(v)) for k, v in self._usage.items()}
            self.emit("usage", **fields)

    def error(self, message: str, kind: str = "agent", fix: str | None = None, fatal: bool = True) -> None:
        self.emit("error", message=message, kind=kind, fix=fix, fatal=fatal)

    def done(self, answer: str | None = None) -> None:
        self.emit("done", answer=answer)


def jsonable(value: Any) -> Any:
    """A JSON-friendly stand-in for objects json.dumps cannot write."""
    for attr in ("model_dump", "to_dict", "dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:  # noqa: BLE001 - fall back to repr
                break
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return repr(value)


def text(value: Any, limit: int = MAX_TEXT) -> str:
    """Tool output as text, cut to `limit` characters."""
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, default=jsonable, ensure_ascii=False)
        except (TypeError, ValueError):
            value = str(value)
    if len(value) > limit:
        return value[:limit] + f"\n[BenchFlow cut {len(value) - limit} more characters]"
    return value


def instruction() -> str:
    """The task's instruction: the file named by the runner's first argument
    (the manifests pass "$BENCHFLOW_INSTRUCTION_FILE"), else by that variable."""
    path = (sys.argv[1] if len(sys.argv) > 1 else "") or os.environ.get("BENCHFLOW_INSTRUCTION_FILE", "")
    if not path:
        raise SetupError(
            "BENCHFLOW_INSTRUCTION_FILE is not set; this runner is meant to be started by BenchFlow",
            "run the agent with `bench run`",
        )
    return Path(path).read_text(encoding="utf-8")


def model() -> str | None:
    return os.environ.get("BENCHFLOW_MODEL", "").strip() or None


def effort() -> str | None:
    return os.environ.get("BENCHFLOW_EFFORT", "").strip() or None


def option(name: str, default: str | None = None) -> str | None:
    """An agent option: a variable set with `bench run --env NAME=VALUE`."""
    value = os.environ.get(name, "").strip()
    return value or default


def flag(name: str, default: bool) -> bool:
    value = option(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def json_option(name: str) -> dict[str, Any]:
    raw = option(name)
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise SetupError(f"{name} is not JSON: {exc}", f"pass --env {name}='{{...}}' with a JSON object") from exc
    if not isinstance(value, dict):
        raise SetupError(f"{name} must be a JSON object", f"pass --env {name}='{{...}}' with a JSON object")
    return value


def workdir() -> Path:
    return Path.cwd()


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return None


def _chain(exc: BaseException) -> list[BaseException]:
    """The exception and the ones it wraps (cause, context, `original_exception`)."""
    seen: list[BaseException] = []
    todo: list[BaseException | None] = [exc]
    while todo and len(seen) < 12:
        current = todo.pop(0)
        if current is None or any(current is s for s in seen):
            continue
        seen.append(current)
        todo.extend(
            [
                getattr(current, "original_exception", None),
                current.__cause__,
                current.__context__,
            ]
        )
        for inner in getattr(current, "exceptions", None) or []:
            todo.append(inner)
    return seen


def _status(exc: BaseException) -> int | None:
    for attr in ("status_code", "status", "code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def describe(exc: BaseException) -> str:
    """The most telling exception in the chain, as `Name: message`."""
    chain = _chain(exc)
    chosen = next((e for e in chain if classify_one(e) != "agent"), chain[-1] if chain else exc)
    # One line: BenchFlow reads a message line by line, and a provider's
    # JSON error body spread over lines would lose its telling part.
    message = " ".join(str(chosen).split()) or type(chosen).__name__
    return f"{type(chosen).__name__}: {message}"[:2000]


def classify_one(exc: BaseException) -> str:
    name = type(exc).__name__.lower()
    words = str(exc).lower()
    status = _status(exc)
    if "quota" in words and ("exceeded" in words or "insufficient" in words) or "credit balance" in words:
        return "usage_limit"
    if (
        "authentication" in name
        or "permissiondenied" in name
        or "unauthorized" in name
        or status in (401, 403)
        or "[401]" in words
        or "[403]" in words
        or "authorization failed" in words
        or "api key not valid" in words
        or "invalid api key" in words
        or "incorrect api key" in words
        or "invalid x-api-key" in words
        or "api_key_invalid" in words
    ):
        return "login"
    if "ratelimit" in name or status == 429:
        return "provider"
    if status == 410 or "[410]" in words or "end of life" in words:
        return "model"
    if ("notfound" in name or status == 404) and "model" in words:
        return "model"
    if (
        "connection" in name
        or "timeout" in name
        or "serviceunavailable" in name
        or "internalserver" in name
        or (status is not None and status >= 500)
    ):
        return "provider"
    return "agent"


def classify(exc: BaseException) -> str:
    """What a failure is: login, usage_limit, provider, model, or agent."""
    for current in _chain(exc):
        kind = classify_one(current)
        if kind != "agent":
            return kind
    return "agent"


def load_callable(reference: str, search: list[Path], what: str, fix: str) -> Any:
    """Load `module:attr` or `path/to/file.py:attr` (dotted attrs allowed),
    looking for modules and files in `search`."""
    module_ref, sep, attr = reference.rpartition(":")
    if not sep or not module_ref or not attr:
        raise SetupError(f"{what} `{reference}` is not `module:name` or `file.py:name`", fix)
    for folder in reversed(search):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    if module_ref.endswith(".py") or "/" in module_ref:
        candidates = [Path(module_ref)] if Path(module_ref).is_absolute() else [d / module_ref for d in search]
        path = next((c for c in candidates if c.is_file()), None)
        if path is None:
            looked = ", ".join(str(c) for c in candidates)
            raise SetupError(f"{what} file `{module_ref}` is not in the sandbox (looked at {looked})", fix)
        spec = importlib.util.spec_from_file_location(f"bf_user_{path.stem}_{uuid.uuid4().hex[:8]}", path)
        if spec is None or spec.loader is None:
            raise SetupError(f"cannot import {path}", fix)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(module_ref)
        except ModuleNotFoundError as exc:
            if exc.name and module_ref.split(".")[0] == exc.name.split(".")[0]:
                folders = ", ".join(str(d) for d in search)
                raise SetupError(f"{what} module `{module_ref}` is not in {folders}", fix) from exc
            raise
    value: Any = module
    for part in attr.split("."):
        if not hasattr(value, part):
            raise SetupError(f"{what} `{reference}`: `{module_ref}` has no `{attr}`", fix)
        value = getattr(value, part)
    return value


def run(main: Callable[[Events], str | None]) -> None:
    """Run `main(events)` and report how it ended. `main` returns the final
    answer; an exception becomes an `error` event."""
    events = Events()
    try:
        answer = main(events)
    except SetupError as exc:
        events.error(str(exc), kind="setup", fix=exc.fix)
        sys.exit(2)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001 - every failure is reported
        traceback.print_exc()
        events.error(describe(exc), kind=classify(exc))
        sys.exit(1)
    events.done(answer)
