"""Model judges for task.md draft 2: ``judge-loop@1`` over ``judge-prompt@1``.

A task.md criterion judged by ``llm`` or ``agent`` goes to a model. BenchFlow
runs the spec's own harness, ``judge-loop@1`` (docs/runtime/judging.md, "The
harness"), on the host:

- **The prompt** is ``judge-prompt@1``, compiled by the vendored reference
  compiler (``benchflow.taskmd._vendor.judgeprompt``): the brief,
  ``FIXED_TEXT_1``, the assignment, and for the ``llm`` role the evidence.
  ``tests/test_taskmd_judge_vectors.py`` holds the bytes to the spec's vectors.
- **The tools** are ``judge-tools@1`` in the Anthropic Messages wire form: an
  agent session gets ``read``, ``run``, and ``submit_review``; an ``llm``
  session gets ``submit_review`` only. ``read`` serves the kept copy of the
  solver's saved outputs, the session's ``/judge/`` files, and its views;
  ``run`` executes one command in a fresh copy of the submission's
  environment (``benchflow.taskmd.verify.SandboxRunner``).
- **The loop** keeps judge-loop@1's rules: ``REMINDER_1`` after a response
  without a tool call and the end at the third; ``submit_review`` checked
  against the assignment, with at most three rejections; token, tool-call, and
  time budgets, and the runtime's budget line once 80 percent is spent.
- **The review**: each accepted verdict's citations are checked against what
  the judge was shown, labeled by who produced the bytes, and flagged; the
  samples of each criterion are combined by the role's ``aggregate``.

Where this departs from the spec, the review records it: the provider is
called directly, not through ``proxy@1``; with a Claude Code OAuth token the
system message is Claude Code's identity followed by ``SYSTEM_PROMPT_1``
(OAuth tokens are accepted only so); the Messages API takes no seed, so each
``judge-seed@1`` seed is recorded, not sent; and a criterion left ``error``
leaves the trial unscored rather than being attributed against a reference
run.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

from benchflow.taskmd._vendor import judgeprompt as jp

CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
ANTHROPIC_VERSION = "2023-06-01"
OAUTH_BETA = "oauth-2025-04-20"
DEFAULT_BASE_URL = "https://api.anthropic.com"
MAX_TOKENS = jp.MAX_TOKENS  # judge-loop@1 sends 32,000 (docs/runtime/judging.md)
READ_CAP = 65_536
RUN_TIMEOUT_DEFAULT = 120
RUN_TIMEOUT_MAX = 600
MAX_CITATIONS = 32
MAX_RATIONALE = 4000
MAX_QUOTE = 2000
MODEL_ENV = "BENCHFLOW_TASKMD_JUDGE_MODEL"
SESSION_LIMIT_ENV = "BENCHFLOW_TASKMD_MAX_JUDGE_SESSIONS"
HARNESS = "judge-loop"
HARNESS_VERSION = "1"


class JudgeError(RuntimeError):
    """A judge could not run: no credentials, no model, a provider failure, or a spent session limit.

    Raised as a verifier error, so the trial is unscored rather than scored 0.
    """


# Credentials and the Messages client ----------------------------------------------------------------


@dataclass(frozen=True)
class Credentials:
    """How the judge authenticates to the Anthropic Messages API."""

    kind: str  # "api-key" or "claude-code-oauth"
    token: str = field(repr=False)
    base_url: str = DEFAULT_BASE_URL

    def headers(self) -> dict[str, str]:
        headers = {"anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}
        if self.kind == "api-key":
            headers["x-api-key"] = self.token
        else:
            headers["authorization"] = f"Bearer {self.token}"
            headers["anthropic-beta"] = OAUTH_BETA
        return headers

    def system(self) -> Any:
        """judge-loop@1's system message; a Claude Code OAuth token needs Claude Code's identity first."""
        if self.kind == "api-key":
            return jp.SYSTEM_PROMPT_1
        return [{"type": "text", "text": CLAUDE_CODE_IDENTITY}, {"type": "text", "text": jp.SYSTEM_PROMPT_1}]

    @property
    def system_note(self) -> str | None:
        if self.kind == "api-key":
            return None
        return "claude-code-oauth: the system message is Claude Code's identity, then SYSTEM_PROMPT_1"


def credentials_from_env(env: dict[str, str] | None = None) -> Credentials | None:
    """``ANTHROPIC_API_KEY``, or else a Claude Code OAuth token (``CLAUDE_CODE_OAUTH_TOKEN``)."""
    source = dict(os.environ if env is None else env)
    base = source.get("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL
    if source.get("ANTHROPIC_API_KEY"):
        return Credentials("api-key", source["ANTHROPIC_API_KEY"], base)
    token = source.get("CLAUDE_CODE_OAUTH_TOKEN") or source.get("CLAUDE_OAUTH_TOKEN")
    if token:
        return Credentials("claude-code-oauth", token, base)
    return None


ModelCall = Callable[[dict[str, Any], float], Awaitable[dict[str, Any]]]


class MessagesClient:
    """A minimal Anthropic Messages client (httpx), with retries for rate limits and server errors."""

    def __init__(self, credentials: Credentials, *, retries: int = 4) -> None:
        self.credentials = credentials
        self.retries = retries

    async def __call__(self, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        import httpx

        url = self.credentials.base_url.rstrip("/") + "/v1/messages"
        delay = 2.0
        last: str = ""
        deadline = time.monotonic() + max(timeout, 1.0)
        for attempt in range(self.retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(min(remaining, 600.0), connect=30.0)) as client:
                    response = await client.post(url, headers=self.credentials.headers(), json=body)
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    return response.json()
                last = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code not in (408, 409, 429, 500, 502, 503, 504, 529):
                    raise JudgeError(f"the judge model's provider refused the request: {last}")
            if attempt < self.retries:
                await asyncio.sleep(min(delay, max(deadline - time.monotonic(), 0)))
                delay *= 2
        raise JudgeError(f"the judge model's provider failed after {self.retries + 1} tries: {last}")


class SessionLimit:
    """A process-wide cap on judge sessions (``BENCHFLOW_TASKMD_MAX_JUDGE_SESSIONS``), for budgeted runs."""

    def __init__(self) -> None:
        self.started = 0

    def acquire(self) -> None:
        raw = os.environ.get(SESSION_LIMIT_ENV)
        if raw is None or raw == "":
            self.started += 1
            return
        try:
            cap = int(raw)
        except ValueError as exc:
            raise JudgeError(f"{SESSION_LIMIT_ENV}={raw!r} is not a whole number") from exc
        if self.started >= cap:
            raise JudgeError(f"the judge-session limit {SESSION_LIMIT_ENV}={cap} is spent in this process")
        self.started += 1


SESSIONS = SessionLimit()


def model_for(role: str, settings: dict[str, Any], env: dict[str, str] | None = None) -> str | None:
    """The model a role runs: an override from ``BENCHFLOW_TASKMD_JUDGE_MODEL``, else the task's.

    The override is ``<model>`` for every role, or ``role=model`` pairs joined by commas.
    task.md lets a run substitute a model; every verdict records the model that produced it.
    """
    raw = (os.environ if env is None else env).get(MODEL_ENV, "").strip()
    if raw:
        if "=" not in raw:
            return raw
        for pair in raw.split(","):
            name, _, model = pair.partition("=")
            if name.strip() == role and model.strip():
                return model.strip()
    model = settings.get("model")
    return model if isinstance(model, str) and model else None


# What a session can read ----------------------------------------------------------------------------


@dataclass
class JudgeFiles:
    """The session's file system on the host: the kept copy at its absolute paths under
    ``root``, ``/judge/`` under ``root/judge``, and the views."""

    root: Path
    kept: list[str]  # absolute paths the runtime saved (outputs, or the working folder)
    views: dict[str, bytes] = field(default_factory=dict)
    separate_verifier: bool = True
    scripted_mounts: tuple[str, ...] = ()  # a scripted seat's mount paths, for seat-visible

    def host(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    def in_kept(self, path: str) -> bool:
        return any(path == k or path.startswith(k.rstrip("/") + "/") or k == "/" for k in self.kept)


def _served(files: JudgeFiles, path: str, assignment: dict[str, Any]) -> bool:
    if path in files.views:
        return True
    if path == "/judge" or path.startswith("/judge/"):
        return jp._judge_file_served(path, assignment) and files.host(path).exists()
    return files.in_kept(path) and (files.host(path).exists() or files.host(path).is_symlink())


def _block(fence: str, text: str) -> str:
    body = text if not text or text.endswith("\n") else text + "\n"
    return f"<<<DATA {fence}\n{body}END DATA {fence}>>>\n"


def _lines_of(text: str) -> list[str]:
    rows = text.split("\n")
    if rows and rows[-1] == "":
        rows.pop()
    return rows


def read_result(files: JudgeFiles, assignment: dict[str, Any], step: int, fence: str, args: dict[str, Any]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """The ``read`` tool: (result text, extra content blocks such as an image, a record for citations)."""
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.startswith("/"):
        return f"[judge step {step}] read: the path must be absolute", [], {"tool": "read"}
    try:
        path = jp.normalize_path(raw_path)
    except jp.JudgePromptError:
        return f"[judge step {step}] read: {raw_path} is not served; use run", [], {"tool": "read"}
    record: dict[str, Any] = {"tool": "read", "path": path}
    if not _served(files, path, assignment):
        return f"[judge step {step}] read: {path} is not served; use run", [], record
    if path in files.views:
        data = files.views[path]
    else:
        target = files.host(path)
        if target.is_dir():
            entries = []
            for child in sorted(target.iterdir(), key=lambda p: p.name.encode("utf-8")):
                if child.is_symlink():
                    continue
                if child.is_dir():
                    entries.append(f"dir {child.name}/")
                elif child.is_file():
                    entries.append(f"file {child.name} {child.stat().st_size}")
            text = "\n".join(entries)
            record["text"] = text
            return f"[judge step {step}] read {path}: {len(entries)} entries\n" + _block(fence, text), [], record
        data = target.read_bytes()
    record["bytes"] = data
    suffix = PurePosixPath(path).suffix.lower()
    digest = hashlib.sha256(data).hexdigest()
    if suffix in jp.IMAGE_TYPES:
        media = jp.IMAGE_TYPES[suffix]
        image = {"type": "image", "source": {"type": "base64", "media_type": media, "data": base64.b64encode(data).decode()}}
        return f"[judge step {step}] read {path}: {media}, {len(data)} bytes, sha256:{digest}", [image], record
    if suffix == ".pdf":
        return f"[judge step {step}] read {path}: PDF, {len(data)} bytes, sha256:{digest}; this runtime extracts no PDF text", [], record
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return f"[judge step {step}] read {path}: binary file, {len(data)} bytes, sha256:{digest}", [], record
    text, removed = jp.render_text(data)
    rows = _lines_of(text)
    total = len(rows)
    lines = args.get("lines")
    if isinstance(lines, list) and len(lines) == 2 and all(isinstance(n, int) and not isinstance(n, bool) for n in lines):
        first, last = max(lines[0], 1), min(lines[1], total)
        if total == 0 or first > last:
            shown, span = "", "0-0"
        else:
            shown, span = "\n".join(rows[first - 1 : last]) + "\n", f"{first}-{last}"
    else:
        shown, span = text, f"1-{total}" if total else "0-0"
    shown = jp.cap(shown, READ_CAP)
    record["text"] = shown
    note = f"; {removed} invisible characters removed" if removed else ""
    return f"[judge step {step}] read {path}: lines {span} of {total}{note}\n" + _block(fence, shown), [], record


Runner = Callable[[str, int], Awaitable[tuple[int | None, bool, bytes]]]


async def run_result(runner: Runner | None, step: int, fence: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The ``run`` tool: one command in a fresh copy of the submission's environment."""
    command = args.get("command")
    record: dict[str, Any] = {"tool": "run", "command": command}
    if not isinstance(command, str):
        return f"[judge step {step}] run: the command must be a string", record
    timeout = args.get("timeout", RUN_TIMEOUT_DEFAULT)
    if isinstance(timeout, float) and timeout.is_integer():
        timeout = int(timeout)
    if not isinstance(timeout, int) or isinstance(timeout, bool) or not 1 <= timeout <= RUN_TIMEOUT_MAX:
        return f"[judge step {step}] run: timeout must be a whole number of seconds from 1 to {RUN_TIMEOUT_MAX}", record
    if runner is None:
        return f"[judge step {step}] run: this session has no runner", record
    code, timed_out, output = await runner(command, timeout)
    from benchflow.taskmd.trajectory import shell_output

    kept, _ = shell_output(output)
    text, removed = jp.render_text(kept.encode("utf-8"))
    head = f"run: timed out after {timeout} s" if timed_out else f"run: exit {code}"
    note = f"; {removed} invisible characters removed" if removed else ""
    record["text"] = text
    return f"[judge step {step}] {head}{note}\n" + _block(fence, text), record


# submit_review ---------------------------------------------------------------------------------------

_CITATION_KEYS = {"source", "path", "lines", "page", "cell", "step", "quote"}
_SOURCES = ("file", "trajectory", "judge", "tests", "instruction", "view")


def _int(value: Any, minimum: int = 1) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def validate_submission(args: Any, assignment: dict[str, Any]) -> list[str]:
    """judge-loop@1's checks 1 to 4 on a ``submit_review`` call; an empty list accepts it."""
    problems: list[str] = []
    if not isinstance(args, dict) or set(args) != {"verdicts", "tags"} or not isinstance(args.get("verdicts"), list) or not isinstance(args.get("tags"), list):
        return ["the arguments are an object with exactly verdicts and tags, both lists"]
    criteria = {c["id"]: c for c in assignment.get("criteria", [])}
    behaviors = {b["id"]: b for b in assignment.get("behaviors", [])}
    seen: list[str] = []
    for n, v in enumerate(args["verdicts"]):
        at = f"verdicts[{n}]"
        if not isinstance(v, dict):
            problems.append(f"{at} is an object")
            continue
        extra = set(v) - {"id", "verdict", "level", "value", "citations", "rationale"}
        missing = {"id", "verdict", "citations", "rationale"} - set(v)
        if extra:
            problems.append(f"{at} has keys the schema does not define: {', '.join(sorted(extra))}")
        if missing:
            problems.append(f"{at} needs {', '.join(sorted(missing))}")
            continue
        if not isinstance(v["id"], str) or v["verdict"] not in ("pass", "fail", "level", "value"):
            problems.append(f"{at}: id is a string and verdict is pass, fail, level, or value")
            continue
        if not isinstance(v["rationale"], str) or len(v["rationale"]) > MAX_RATIONALE:
            problems.append(f"{at}: rationale is a string of at most {MAX_RATIONALE} characters")
        cites = v["citations"]
        if not isinstance(cites, list) or len(cites) > MAX_CITATIONS:
            problems.append(f"{at}: citations is a list of at most {MAX_CITATIONS}")
        else:
            for m, c in enumerate(cites):
                where = f"{at}.citations[{m}]"
                if not isinstance(c, dict) or c.get("source") not in _SOURCES or set(c) - _CITATION_KEYS:
                    problems.append(f"{where}: a citation has a source ({', '.join(_SOURCES)}) and only path, lines, page, cell, step, quote")
                    continue
                if "lines" in c and not (isinstance(c["lines"], list) and len(c["lines"]) == 2 and all(_int(x) for x in c["lines"])):
                    problems.append(f"{where}: lines is [first, last], counted from 1")
                for key in ("page", "cell", "step"):
                    if key in c and not _int(c[key]):
                        problems.append(f"{where}: {key} is a whole number, at least 1")
                if "path" in c and not isinstance(c["path"], str):
                    problems.append(f"{where}: path is a string")
                if "quote" in c and (not isinstance(c["quote"], str) or len(c["quote"]) > MAX_QUOTE):
                    problems.append(f"{where}: quote is a string of at most {MAX_QUOTE} characters")
        seen.append(v["id"])
        criterion = criteria.get(v["id"])
        if criterion is None:
            problems.append(f"{at}: {v['id']!r} is not a criterion in the assignment")
            continue
        if "levels" in criterion:
            if v["verdict"] != "level" or str(v.get("level")) not in criterion["levels"]:
                problems.append(f"{at}: {v['id']} takes verdict \"level\" with level one of {', '.join(criterion['levels'])}")
            elif not isinstance(v.get("level"), str):
                problems.append(f"{at}: level is a string")
        elif "score" in criterion:
            lo, hi = criterion["score"]["min"], criterion["score"]["max"]
            value = v.get("value")
            if v["verdict"] != "value" or not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not lo <= value <= hi:
                problems.append(f"{at}: {v['id']} takes verdict \"value\" with a number from {lo} to {hi}")
        elif v["verdict"] not in ("pass", "fail"):
            problems.append(f"{at}: {v['id']} takes pass or fail")
    for ident in sorted(set(criteria) - set(seen)):
        problems.append(f"no verdict for criterion {ident}")
    for ident in sorted({x for x in seen if seen.count(x) > 1}):
        problems.append(f"more than one verdict for criterion {ident}")
    tags_seen = []
    for n, t in enumerate(args["tags"]):
        at = f"tags[{n}]"
        if not isinstance(t, dict) or {"id", "detected", "severity", "spans"} - set(t) or set(t) - {"id", "detected", "severity", "spans", "note"}:
            problems.append(f"{at} has id, detected, severity, spans, and optionally note")
            continue
        tags_seen.append(t["id"])
        behavior = behaviors.get(t["id"])
        if behavior is None:
            problems.append(f"{at}: {t['id']!r} is not a behavior in the assignment")
            continue
        detected, severity = t["detected"], t["severity"]
        binary = behavior.get("scale", "binary") == "binary"
        ok = (
            (detected is None and severity is None)
            or (detected is False and severity == 0)
            or (detected is True and binary and severity == 1)
            or (detected is True and not binary and severity in (1, 2, 3))
        )
        if not ok:
            problems.append(f"{at}: severity is null when detected is null, 0 when false, 1 for a detected binary behavior, and 1 to 3 for a 0-3 behavior")
    for ident in sorted(set(behaviors) - set(tags_seen)):
        problems.append(f"no tag for behavior {ident}")
    return problems


# One session ----------------------------------------------------------------------------------------


@dataclass
class SessionSpec:
    role: str
    unit: str
    model: str
    assignment: dict[str, Any]
    prompt_masked: str  # parts 1-3, {fence} and {budget} as written
    brief: str | None
    budget_text: str
    seconds: int
    tokens: int | None
    tool_calls: int | None
    tools: tuple[str, ...]


@dataclass
class SessionResult:
    accepted: dict[str, Any] | None
    end: str
    fence: str
    usage: dict[str, Any]
    judge_steps: dict[int, dict[str, Any]]
    transcript: list[dict[str, Any]]


def _budget_line(spec: SessionSpec, calls: int, tokens: int, started: float) -> str | None:
    """The runtime's line once 80 percent of any budget is spent."""
    spent = []
    if spec.tool_calls:
        spent.append(calls / spec.tool_calls)
    if spec.tokens:
        spent.append(tokens / spec.tokens)
    elapsed = time.monotonic() - started
    spent.append(elapsed / spec.seconds if spec.seconds else 0)
    if max(spent) < 0.8:
        return None
    parts = []
    if spec.tool_calls:
        parts.append(f"{max(spec.tool_calls - calls, 0)} tool calls")
    if spec.tokens:
        parts.append(f"{max(spec.tokens - tokens, 0)} tokens")
    parts.append(f"{max(int(spec.seconds - elapsed), 0)} seconds")
    joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + (", and " if len(parts) > 2 else " and ") + parts[-1]
    return f"[runtime: {joined} left]"


async def run_session(
    spec: SessionSpec,
    *,
    call: ModelCall,
    credentials: Credentials,
    files: JudgeFiles,
    evidence_text: Callable[[str], str] | None,
    runner: Runner | None,
) -> SessionResult:
    """One judge-loop@1 session: prompt, tools, and at most one accepted review."""
    fence = secrets.token_hex(8)
    prompt = jp.prompt_text(spec.brief, spec.assignment, fence=fence, budget=spec.budget_text)
    if evidence_text is not None:
        prompt += evidence_text(fence)
    tools = jp.render_tools("anthropic-messages", spec.tools)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    usage = {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "tool_calls": 0, "calls": 0}
    judge_steps: dict[int, dict[str, Any]] = {}
    started = time.monotonic()
    step = 0
    quiet = 0
    rejected = 0
    tokens = 0

    def result(end: str, accepted: dict[str, Any] | None = None) -> SessionResult:
        usage["seconds"] = round(time.monotonic() - started, 3)
        return SessionResult(accepted, end, fence, usage, judge_steps, messages)

    while True:
        remaining = spec.seconds - (time.monotonic() - started)
        if remaining <= 0:
            return result("timeout")
        if spec.tokens is not None and tokens >= spec.tokens:
            return result("token-budget")
        body = {
            "model": spec.model,
            "max_tokens": MAX_TOKENS,
            "system": credentials.system(),
            "messages": messages,
            "tools": tools,
            "tool_choice": {"type": "auto"},
        }
        try:
            response = await asyncio.wait_for(call(body, remaining), timeout=remaining + 5)
        except TimeoutError:
            return result("timeout")
        except JudgeError as exc:
            return result(f"provider: {exc}")
        u = response.get("usage") or {}
        cache_read = int(u.get("cache_read_input_tokens") or 0)
        prompt_tokens = int(u.get("input_tokens") or 0) + int(u.get("cache_creation_input_tokens") or 0) + cache_read
        completion = int(u.get("output_tokens") or 0)
        usage["prompt_tokens"] += prompt_tokens
        usage["cached_tokens"] += cache_read
        usage["completion_tokens"] += completion
        usage["calls"] += 1
        tokens += prompt_tokens + completion
        content = response.get("content") or []
        messages.append({"role": "assistant", "content": content})
        uses = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
        if not uses:
            quiet += 1
            if quiet >= 3:
                return result("no-tool-call")
            messages.append({"role": "user", "content": jp.REMINDER_1})
            continue
        results: list[dict[str, Any]] = []
        accepted: dict[str, Any] | None = None
        ended: str | None = None
        for use in uses:
            name, args, use_id = use.get("name"), use.get("input"), use.get("id")
            blocks: list[dict[str, Any]] = []
            if accepted is not None or ended is not None:
                text = "the session has ended"
            elif name == "submit_review" and "submit_review" in spec.tools:
                problems = validate_submission(args, spec.assignment)
                if not problems:
                    accepted = args
                    text = "Review accepted."
                else:
                    rejected += 1
                    text = "Review not accepted:\n" + "".join(f"- {p}\n" for p in problems) + "Fix these and submit again."
                    if rejected >= 3:
                        ended = "rejected"
            elif name in ("read", "run") and name in spec.tools:
                step += 1
                if spec.tool_calls is not None and usage["tool_calls"] >= spec.tool_calls:
                    text = f"[judge step {step}] {name}: the tool-call budget is spent; submit your review"
                    judge_steps[step] = {"tool": name, "text": ""}
                else:
                    usage["tool_calls"] += 1
                    arguments = args if isinstance(args, dict) else {}
                    if name == "read":
                        text, blocks, record = read_result(files, spec.assignment, step, fence, arguments)
                    else:
                        text, record = await run_result(runner, step, fence, arguments)
                    judge_steps[step] = record
                line = _budget_line(spec, usage["tool_calls"], tokens, started)
                if line:
                    text += line + "\n" if text.endswith("\n") else "\n" + line + "\n"
            else:
                text = f"error: unknown tool {name!r}"
            content_out: Any = [{"type": "text", "text": text}, *blocks] if blocks else text
            results.append({"type": "tool_result", "tool_use_id": use_id, "content": content_out})
        messages.append({"role": "user", "content": results})
        if accepted is not None:
            return result("accepted", accepted)
        if ended is not None:
            return result(ended)


# Checking citations ---------------------------------------------------------------------------------


@dataclass
class CitationContext:
    files: JudgeFiles
    record: dict[str, Any]  # the solver's trajectory-1 record
    tests: dict[str, dict[str, Any]]  # /judge/tests.json entries by id
    instruction: bytes
    workdir: str | None
    reasoning_served: bool = False


def _trajectory_fields(step: dict[str, Any], reasoning: bool) -> list[str]:
    fields: list[str] = []
    for key in ("text",) + (("reasoning",) if reasoning else ()):
        if isinstance(step.get(key), str):
            fields.append(step[key])
    for call in step.get("tool_calls") or []:
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            fields.append(arguments)
            try:
                parsed = json.loads(arguments)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                fields.append(jp.jcs(parsed))
                fields += [v for v in parsed.values() if isinstance(v, str)]
        result = call.get("result")
        if isinstance(result, dict) and isinstance(result.get("text"), str):
            fields.append(result["text"])
    return fields


def check_citation(cite: dict[str, Any], ctx: CitationContext, steps: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """A citation as the review records it: verified, label, match, and detail when it did not resolve."""
    out = {k: cite[k] for k in ("source", "path", "lines", "page", "cell", "step", "quote") if k in cite}
    source = cite.get("source")
    quote = cite.get("quote") if isinstance(cite.get("quote"), str) else ""
    spans: list[str] = []
    label = "solver"
    detail = None
    if source == "file":
        path = str(cite.get("path", ""))
        try:
            path = jp.normalize_path(path, ctx.workdir)
        except jp.JudgePromptError as exc:
            detail = str(exc)
        else:
            out["path"] = path
            target = ctx.files.host(path)
            if ctx.files.in_kept(path) and target.is_file():
                data = target.read_bytes()
                if PurePosixPath(path).suffix.lower() in jp.IMAGE_TYPES:
                    out.update(verified=None, label="solver")
                    return out
                if "page" in cite or "cell" in cite:
                    detail = "pages and notebook cells are not resolved by this runtime"
                else:
                    span = jp.span_text(data, cite.get("lines"))
                    if span is None:
                        detail = "not UTF-8 text, or the lines do not exist"
                    else:
                        spans.append(span)
            else:
                detail = "not a saved file (the task's own inputs are not checked by this runtime)"
    elif source == "trajectory":
        label = "solver"
        wanted = cite.get("step")
        step = next((s for s in ctx.record.get("steps") or [] if s.get("id") == wanted), None)
        if step is None:
            detail = f"the trajectory has no step {wanted}"
        else:
            if step.get("source") == "runtime":
                label = "environment"
            for text in _trajectory_fields(step, ctx.reasoning_served):
                spans.append(text)
                spans.append(json.dumps(text)[1:-1])
    elif source == "judge":
        wanted = cite.get("step")
        record = steps.get(wanted) if isinstance(wanted, int) else None
        if record is None:
            detail = f"no judge step {wanted}"
        elif record.get("tool") == "run":
            command = record.get("command") if isinstance(record.get("command"), str) else ""
            text = record.get("text") or ""
            if quote and quote in command:
                label, spans = "judge", [command]
            else:
                # No access watch records whether the command opened the kept copy, so its output is solver-executed.
                label, spans = "solver-executed", [text, command]
        else:
            path = str(record.get("path", ""))
            label = _read_label(path, ctx.files)
            spans.append(record.get("text") or "")
            if isinstance(record.get("bytes"), bytes):
                span = jp.span_text(record["bytes"])
                if span is not None:
                    spans.append(span)
    elif source == "tests":
        label = "environment" if ctx.files.separate_verifier else "solver-executed"
        entry = ctx.tests.get(str(cite.get("path")))
        if entry is None:
            detail = "no test of that id in /judge/tests.json"
        else:
            spans.append(jp.jcs(entry))
    elif source == "instruction":
        label = "environment"
        span = jp.span_text(ctx.instruction, cite.get("lines"))
        if span is None:
            detail = "the lines do not exist"
        else:
            spans.append(span)
    elif source == "view":
        path = str(cite.get("path", ""))
        label = "solver"
        data = ctx.files.views.get(path)
        if data is None and path.startswith("/judge/") and ctx.files.host(path).is_file():
            data = ctx.files.host(path).read_bytes()
            label = _read_label(path, ctx.files)
        if data is None:
            detail = "no such view"
        else:
            span = jp.span_text(data, cite.get("lines"))
            if span is None:
                detail = "not UTF-8 text, or the lines do not exist"
            else:
                spans.append(span)
    out["label"] = label
    if detail is not None:
        out.update(verified=False, detail=detail)
        return out
    if not quote:
        out.update(verified=False, detail="no quote")
        return out
    match = None
    for span in spans:
        match = jp.quote_match(quote, span)
        if match == "exact":
            break
        if match is not None:
            break
    out["verified"] = match is not None
    if match is not None:
        out["match"] = match
    return out


def _read_label(path: str, files: JudgeFiles) -> str:
    if path == "/judge/instruction.md":
        return "environment"
    if path == "/judge/tests.json":
        return "environment" if files.separate_verifier else "solver-executed"
    return "solver"


# Combining samples ----------------------------------------------------------------------------------


def sample_number(criterion: dict[str, Any], verdict: dict[str, Any]) -> Fraction:
    """A valid sample's number: 1 for pass, 0 for fail, the level's key, or the value."""
    if verdict["verdict"] == "level":
        return Fraction(str(verdict["level"]))
    if verdict["verdict"] == "value":
        return Fraction(str(verdict["value"]))
    return Fraction(1 if verdict["verdict"] == "pass" else 0)


def combine(numbers: list[Fraction], aggregate: str) -> Fraction:
    """docs/runtime/judging.md: median is the lower median; majority's tie goes to the lower; mean picks as median."""
    ordered = sorted(numbers)
    if aggregate == "min":
        return ordered[0]
    if aggregate == "majority":
        counts: dict[Fraction, int] = {}
        for n in numbers:
            counts[n] = counts.get(n, 0) + 1
        best = max(counts.values())
        return min(n for n, c in counts.items() if c == best)
    return ordered[(len(ordered) - 1) // 2]  # median, and mean's verdict


def verdict_value(criterion: dict[str, Any], verdict: dict[str, Any]) -> Any:
    """The score() input of a verdict: True or False, a level key, or the value."""
    if verdict["verdict"] == "level":
        return str(verdict["level"])
    if verdict["verdict"] == "value":
        return verdict["value"]
    return verdict["verdict"] == "pass"


def apply_citation_rules(criterion: dict[str, Any], verdict: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Flags, and ``cite = "independent"``: a pass without a verified judge or environment citation is a fail."""
    flags: list[str] = []
    verified = [c for c in verdict.get("citations", []) if c.get("verified") is True]
    if not verified:
        flags.append("no-verified-citation")
    passing = verdict["verdict"] == "pass" or (verdict["verdict"] in ("level", "value") and sample_number(criterion, verdict) > 0)
    independent = [c for c in verified if c.get("label") in ("judge", "environment")]
    if passing and verified and not independent:
        flags.append("self-cited")
    if criterion.get("cite") == "independent" and verdict["verdict"] == "pass" and not independent:
        verdict = {**verdict, "verdict": "fail"}
        if "self-cited" not in flags:
            flags.append("self-cited")
    return verdict, flags
