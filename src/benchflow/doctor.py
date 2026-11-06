"""Host readiness checks behind ``bench doctor``.

One place that answers "can this machine run a BenchFlow eval right now, and if
not, what do I do?" Each check returns a :class:`Check` with a pass / warn /
fail / skip status, a one-line summary and a concrete fix. ``fail`` is reserved
for problems that stop every run on the selected sandbox (no Docker daemon, no
Daytona SDK); everything else is a warning, including a machine with no agent
credential at all (the oracle and nop controls still run).

The Claude subscription check is the one model request doctor makes: one
8-token ``claude-haiku-5-5`` request with the login's OAuth token,
whose ``anthropic-ratelimit-unified-*`` headers give the 5-hour and 7-day
windows' use and resets (``--offline`` skips it).

Credentials are read to decide *whether* they exist and *when* they expire.
Values never leave this module: summaries, fixes and ``details`` carry variable
names, file paths, sources and timestamps only, and any text taken from a
subprocess or SDK error is passed through :func:`redact` with every
credential value known to the environment.

All side effects go through :class:`DoctorProbes`, so tests drive every branch
with fakes and no Docker, network or credential file is touched.
"""

from __future__ import annotations

import base64
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from benchflow._diagnostics import redact, redact_stream, render_diagnostic
from benchflow._utils.command_hints import command_arg

Status = Literal["pass", "warn", "fail", "skip"]

GIB = 1024**3
# Below this the Docker VM runs out of memory while building a task image and
# installing a Node-based agent side by side; 8 GiB leaves room for concurrency.
MIN_DOCKER_MEMORY_BYTES = 4 * GIB
MIN_DAYTONA_SDK = (0, 207, 0)
PROBE_TIMEOUT_SEC = 5.0
# Below this Docker's data root runs out while building task images.
MIN_DOCKER_FREE_BYTES = 10 * GIB
HEADROOM_TIMEOUT_SEC = 20.0
# The current Haiku. Haiku 4.5 is a legacy model (Anthropic's models page,
# read 2026-11-04); a 2026-11-04 request with a subscription login answered 200
# with the unified usage headers.
HEADROOM_MODEL = "claude-haiku-5-5"
ANTHROPIC_API = "https://api.anthropic.com"
# Where a Daytona sandbox installs the LiteLLM model proxy from.
PYPI_ENDPOINT = "https://pypi.org/simple/"

# Endpoints every JS agent install needs from inside the sandbox (Node tarball
# + npm packages). Probed from the host as a proxy for the sandbox's egress.
AGENT_INSTALL_ENDPOINTS = ("https://nodejs.org/", "https://registry.npmjs.org/")
DOCKER_REGISTRY_ENDPOINT = "https://registry-1.docker.io/v2/"
DAYTONA_DEFAULT_API = "https://app.daytona.io/api"
# The one key-check request (the SDK import is not counted), and how often
# ``bench doctor`` asks again when the API does not answer. The run preflight
# asks once: only a rejected key stops a run (DAYTONA-START).
DAYTONA_CHECK_TIMEOUT_SEC = 10.0
DOCTOR_DAYTONA_TRIES = 3


# ── Data model ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Check:
    """One doctor line."""

    id: str
    group: str
    name: str
    status: Status
    summary: str
    fix: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CredentialSource:
    """A credential that exists for an agent. Never holds the value itself."""

    name: str  # env var name or ~/-relative file path
    kind: str  # api-key | auth-token | oauth-token | access-token | login-file | auth-json
    origin: str  # env | .env | file
    expires_at: datetime | None = None
    usable: bool = True
    note: str = ""
    # The agent refreshes this credential itself, so an expired access token
    # is normal and its expiry is not shown as a problem.
    self_refreshing: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["expires_at"] = self.expires_at.isoformat() if self.expires_at else None
        return data


@dataclass(frozen=True)
class AgentAuth:
    """Every credential found for one agent, in the precedence order it uses."""

    agent: str
    label: str
    sources: tuple[CredentialSource, ...]
    effective: CredentialSource | None
    endpoints: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.effective is not None and self.effective.usable


@dataclass
class DoctorReport:
    checks: list[Check]
    sandbox: str
    agents: dict[str, AgentAuth] = field(default_factory=dict)
    unreachable: frozenset[str] = frozenset()

    @property
    def ok(self) -> bool:
        return not any(c.status == "fail" for c in self.checks)

    def counts(self) -> dict[str, int]:
        out = {"pass": 0, "warn": 0, "fail": 0, "skip": 0}
        for c in self.checks:
            out[c.status] += 1
        return out

    def agent_blocked_by_network(self, agent: str) -> list[str]:
        auth = self.agents.get(agent)
        if auth is None:
            return []
        return [url for url in auth.endpoints if url in self.unreachable]

    def to_dict(self) -> dict[str, Any]:
        from benchflow import __version__

        return {
            "ok": self.ok,
            "benchflow_version": __version__,
            "sandbox": self.sandbox,
            "counts": self.counts(),
            "checks": [asdict(c) for c in self.checks],
            "agents": {
                name: {
                    "ready": auth.ready,
                    "effective": auth.effective.to_dict() if auth.effective else None,
                    "sources": [s.to_dict() for s in auth.sources],
                    "endpoints": [_redact_url(url) for url in auth.endpoints],
                    "notes": list(auth.notes),
                }
                for name, auth in self.agents.items()
            },
        }


# ── Side-effect boundary ────────────────────────────────────────────────


@dataclass(frozen=True)
class CommandResult:
    returncode: int | None  # None: not found or timed out
    stdout: str = ""
    stderr: str = ""


def _run_command(argv: Sequence[str], timeout: float) -> CommandResult:
    # Captures must be protected before consumers parse or shorten them. Keep
    # byte/newline distinctions until the same boundary used by smoke logs.
    from benchflow.cli._errors import _secrets

    secrets = _secrets()

    def captured(value: bytes) -> str:
        return "".join(
            redact_stream((value.decode("utf-8", "surrogateescape"),), secrets)
        )

    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CommandResult(None, "", redact(f"{type(exc).__name__}: {exc}", secrets))
    return CommandResult(proc.returncode, captured(proc.stdout), captured(proc.stderr))


def _http_status(url: str, timeout: float) -> int:
    """Return the HTTP status of a GET; raise on connect/TLS/timeout failure.

    Any HTTP answer (401, 404, 403 …) proves egress to the host; only the
    transport outcome matters here.
    """
    import logging

    import httpx

    # The CLI logs at INFO; httpx would print one "HTTP Request" line per probe.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    response = httpx.get(url, timeout=timeout, follow_redirects=False)
    return response.status_code


def _claude_headroom_request(
    token: str, base_url: str, timeout: float
) -> tuple[int | None, dict[str, str], str]:
    """One 8-token Haiku request with a Claude OAuth token: (status, headers, error).

    The headers are returned whatever the status (a spent login answers 429
    with the same ``anthropic-ratelimit-unified-*`` headers); ``error`` is a
    transport failure. The token goes only into the Authorization header.
    """
    import logging

    import httpx

    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        response = httpx.post(
            base_url.rstrip("/") + "/v1/messages",
            headers={
                "Authorization": f"Bearer {token}",
                "anthropic-beta": "oauth-2025-04-20",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": HEADROOM_MODEL,
                "max_tokens": 8,
                "system": "You are Claude Code, Anthropic's official CLI for Claude.",
                "messages": [{"role": "user", "content": "Reply with ready."}],
            },
            timeout=timeout,
        )
    except Exception as exc:
        return None, {}, f"{type(exc).__name__}: {exc}"
    return response.status_code, dict(response.headers), ""


def _disk_free(path: str) -> int | None:
    """Free bytes on the filesystem holding ``path``, or None when it is not here."""
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


def _dist_version(dist: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def _daytona_list_one(environ: Mapping[str, str], timeout: float) -> None:
    """Authenticate against Daytona by listing at most one sandbox.

    *timeout* covers the one request only. The SDK's import and the client
    (no network) come first: counted inside it, a CPU-starved machine (64 or
    more ``bench`` processes starting at once) turned a 0.1 s answer into a
    15 s TimeoutError that refused the run (DAYTONA-START).
    """
    from benchflow.sandbox.daytona import build_sync_client

    client = build_sync_client(environ.get("DAYTONA_API_KEY"))
    try:
        from daytona import ListSandboxesQuery

        query: Any = ListSandboxesQuery(limit=1)
    except ImportError:
        query = None

    def _call() -> None:
        iterator = iter(client.list() if query is None else client.list(query))
        next(iterator, None)

    # No context manager: its exit would wait for a hung SDK call and defeat
    # the timeout. The worker thread is abandoned instead.
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        pool.submit(_call).result(timeout=timeout)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _remote_docker_info(host: Any, environ: Mapping[str, str], timeout: float) -> Any:
    """``docker info`` on a remote host (:func:`probe_remote_docker`): its
    :class:`HostCapacity`, or :class:`SandboxStartupError` naming the host
    without its ssh user."""
    from benchflow.sandbox.remote_docker import probe_remote_docker

    return probe_remote_docker(host, timeout_sec=timeout, environ=environ)


@dataclass
class DoctorProbes:
    """Everything the checks read from the outside world."""

    environ: Mapping[str, str]
    origins: Mapping[str, str]
    home: Path
    which: Callable[[str], str | None] = shutil.which
    run: Callable[[Sequence[str], float], CommandResult] = _run_command
    http_status: Callable[[str, float], int] = _http_status
    dist_version: Callable[[str], str | None] = _dist_version
    daytona_check: Callable[[Mapping[str, str], float], None] = _daytona_list_one
    claude_headroom: Callable[
        [str, str, float], tuple[int | None, dict[str, str], str]
    ] = _claude_headroom_request
    disk_free: Callable[[str], int | None] = _disk_free
    remote_docker_info: Callable[[Any, Mapping[str, str], float], Any] = (
        _remote_docker_info
    )
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    sleep: Callable[[float], None] = time.sleep
    system: str = platform.system()
    python_version: tuple[int, int, int] = (
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
    )
    python_executable: str = sys.executable
    # The path this process was started as (``~/.local/bin/bench``), which
    # tells this install's ``bench`` from another one on PATH.
    entry_point: str = field(default_factory=lambda: sys.argv[0] if sys.argv else "")
    read_bench: Callable[[Path], BenchFile] = field(
        default_factory=lambda: read_bench_file
    )

    @classmethod
    def from_host(cls) -> DoctorProbes:
        """Real probes. ``.env`` in the cwd fills gaps, like ``bench eval run``."""
        from benchflow._dotenv import load_dotenv_env

        merged: dict[str, str] = {}
        origins: dict[str, str] = {}
        for key, value in load_dotenv_env().items():
            if value.strip():
                merged[key] = value
                origins[key] = ".env"
        for key, value in os.environ.items():
            if value.strip():
                merged[key] = value
                origins[key] = "env"
        return cls(environ=merged, origins=origins, home=Path.home())

    def get(self, key: str) -> str | None:
        value = self.environ.get(key)
        return value if value and value.strip() else None

    def origin(self, key: str) -> str:
        return self.origins.get(key, "env")

    def read_json(self, path: Path) -> Any:
        return json.loads(path.read_text())


# ── Helpers ─────────────────────────────────────────────────────────────


_SECRET_NAME_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|AUTH_JSON|CREDENTIAL)")
# BenchFlow's own switches whose names match _SECRET_NAME_RE but hold no
# secret: masking ``BENCHFLOW_KEY_CHECK=0``'s value hid every "0" in an
# error, a hint or a path (job folders are dated).
_NOT_SECRET_NAMES = frozenset({"BENCHFLOW_KEY_CHECK"})


# Fields of the agents' login files that name a mode, a plan or a time, not a
# credential. Masking them hid words everywhere: a Codex login with
# "auth_mode": "apikey" turned doctor's own Google AI Studio link into
# https://aistudio.google.com/***.
_NOT_SECRET_JSON_KEYS = frozenset(
    {"auth_mode", "last_refresh", "scopes", "subscriptionType", "rateLimitTier"}
)


def _json_string_leaves(raw: str) -> list[str]:
    try:
        stack: list[tuple[str | None, Any]] = [(None, json.loads(raw))]
    except json.JSONDecodeError:
        return []
    leaves: list[str] = []
    while stack:
        key, node = stack.pop()
        if key in _NOT_SECRET_JSON_KEYS:
            continue
        if isinstance(node, dict):
            stack.extend((str(k), v) for k, v in node.items())
        elif isinstance(node, list):
            stack.extend((key, item) for item in node)
        elif isinstance(node, str) and node:
            leaves.append(node)
    return leaves


def secret_values(environ: Mapping[str, str]) -> list[str]:
    """Values of credential-looking variables, longest first, for :func:`redact`.

    Inline JSON credentials (``CODEX_AUTH_JSON``) also contribute each nonempty
    string inside them, so a lone token from the file is caught too.
    """
    values: list[str] = []
    for key, value in environ.items():
        if not _SECRET_NAME_RE.search(key) or not value or key in _NOT_SECRET_NAMES:
            continue
        values.append(value)
        if key.endswith("_JSON"):
            values.extend(_json_string_leaves(value))
    return sorted(set(values), key=lambda value: len(value), reverse=True)


def redact_exact(text: str, secrets: Iterable[str]) -> str:
    """Compatibility entry point for the shared diagnostic choke point."""
    return redact(text, secrets)


def redacted_traceback(
    exc: BaseException,
    secrets: Iterable[str],
    *,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Mask raw exception fields before formatting groups; never render notes."""
    root = traceback.TracebackException.from_exception(exc, capture_locals=False)
    pending = [root]
    seen: set[int] = set()
    fields: list[str] = []
    targets: list[tuple[Any, str]] = []
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        # TracebackException snapshots str(exc) in _str; formatting adds the
        # group's line prefixes afterwards. Causes/contexts have snapshots
        # of their own, as do every nested group's children.
        fields.append(item.__dict__["_str"])
        targets.append((item, "_str"))
        for frame in item.stack:
            for name in ("filename", "name", "_line", "_lines", "_lines_dedented"):
                value = getattr(frame, name, None)
                if isinstance(value, str):
                    fields.append(value)
                    targets.append((frame, name))
        notes = getattr(item, "__notes__", None)
        if notes is not None:
            # Notes are arbitrary payloads, including scalars, byte strings
            # and sequences of individual credential characters. Never
            # iterate or stringify them for diagnostics (FF-CLI-C6).
            item.__notes__ = ["*** (exception notes omitted)"]
        # SyntaxError has separately formatted message and source fields.
        for name in ("msg", "filename", "text"):
            value = getattr(item, name, None)
            if isinstance(value, str):
                fields.append(value)
                targets.append((item, name))
        for linked in (item.__cause__, item.__context__):
            if linked is not None:
                pending.append(linked)
        pending.extend(item.exceptions or ())
    fields.extend((prefix, suffix))

    def render(protected: Sequence[str]) -> str:
        for (item, name), value in zip(targets, protected[:-2], strict=True):
            setattr(item, name, value)
        return protected[-2] + "".join(root.format()).rstrip() + protected[-1]

    return render_diagnostic(fields, secrets, render)


def _first_line(text: str, limit: int = 200, *, secrets: Iterable[str] = ()) -> str:
    """First non-blank line, redacted *before* it is shortened to ``limit``.

    Redaction runs on the whole line first: shortening a line that still holds
    a secret can keep a prefix of the secret that :func:`redact` no longer
    matches, leaking part of a key (a reviewer reproduced 14- and 31-character
    leaks). Pass ``secrets`` at every site whose text may carry credentials.
    """

    def render(fields: Sequence[str]) -> str:
        for line in fields[0].splitlines():
            line = line.strip()
            if line:
                return line if len(line) <= limit else line[: limit - 3] + "..."
        return ""

    return render_diagnostic((text,), secrets, render)


def _login_file(probes: DoctorProbes, path: str) -> tuple[Path, str]:
    """A login file's path on the host and how lines name it.

    ``$CLAUDE_CONFIG_DIR`` and ``$CODEX_HOME`` move Claude Code's and Codex's
    logins out of ``~/.claude`` and ``~/.codex``, as the agents read them.
    """
    from benchflow.agents.login_dirs import login_display, login_path

    found = login_path(path, environ=probes.environ, home=probes.home)
    shown = login_display(path, environ=probes.environ)
    return found, (shown if shown != path else _tilde(found, probes.home))


def _tilde(path: Path, home: Path) -> str:
    try:
        return "~/" + str(path.relative_to(home))
    except ValueError:
        return str(path)


def _fmt_bytes(n: int | float) -> str:
    return f"{n / GIB:.1f} GiB"


def _fmt_when(when: datetime, now: datetime) -> str:
    delta = when - now
    stamp = when.strftime("%Y-%m-%d %H:%M UTC")
    seconds = delta.total_seconds()
    if seconds < 0:
        days = int(-seconds // 86400)
        ago = f"{days} days ago" if days >= 1 else f"{int(-seconds // 3600)} h ago"
        return f"expired {stamp} ({ago})"
    days = int(seconds // 86400)
    ahead = f"in {days} days" if days >= 1 else f"in {int(seconds // 3600)} h"
    return f"valid until {stamp} ({ahead})"


def _epoch_ms(value: Any) -> datetime | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return datetime.fromtimestamp(value / 1000, UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _jwt_claims(token: Any) -> dict[str, Any]:
    """Decode a JWT payload without verifying it (expiry display only)."""
    if not isinstance(token, str) or token.count(".") != 2:
        return {}
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, json.JSONDecodeError):
        return {}
    return claims if isinstance(claims, dict) else {}


def _parse_version(text: str) -> tuple[int, ...]:
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not match:
        return ()
    return tuple(int(part) for part in match.groups() if part is not None)


def _host(url: str) -> str:
    """``host[:port]`` of ``url``, without userinfo (which can carry a token)."""
    netloc = urlparse(url).netloc
    return netloc.rsplit("@", 1)[-1] if netloc else url


def _redact_url(url: str) -> str:
    """``url`` with any userinfo (``user:token@``) replaced by ``***``."""
    if url.startswith("//"):
        return redact("https:" + url, ())[len("https:") :]
    return redact(url, ())


# ── Runtime checks ──────────────────────────────────────────────────────


def _checker(group: str, name: str, cid: str | None = None) -> Callable[..., Check]:
    """Build :class:`Check` rows for one named line (its id defaults to its name)."""

    def make(
        status: Status, summary: str, fix: str = "", details: dict | None = None
    ) -> Check:
        return Check(cid or name, group, name, status, summary, fix, details or {})

    return make


def check_python(probes: DoctorProbes) -> Check:
    from benchflow import __version__

    row = _checker("runtime", "python")
    major, minor, micro = probes.python_version
    version = f"{major}.{minor}.{micro}"
    details = {
        "python": version,
        "executable": probes.python_executable,
        "benchflow": __version__,
    }
    if (major, minor) < (3, 12):
        return row(
            "fail",
            f"Python {version} is older than 3.12",
            "uv tool install --python 3.12 --upgrade benchflow",
            details,
        )
    summary = f"Python {version}, benchflow {__version__} ({probes.python_executable})"
    return row("pass", summary, details=details)


def check_uv(probes: DoctorProbes) -> Check:
    row = _checker("runtime", "uv")
    if probes.which("uv") is None:
        return row(
            "warn",
            "uv not found on PATH (needed to install, upgrade or add extras)",
            "Install uv: https://docs.astral.sh/uv/getting-started/installation/",
        )
    result = probes.run(["uv", "--version"], PROBE_TIMEOUT_SEC)
    version = _first_line(result.stdout) or "version unknown"
    return row("pass", version, details={"uv": version})


BFX_MISSING = (
    "bfx is not installed: this BenchFlow was built without its pinned bfx "
    "binaries, so Docker trials run root's steps after the agent without it, "
    "as with BENCHFLOW_BFX_DELIVERY=off (those steps then need a Python in "
    "the image)"
)
BFX_FIX = (
    "install a release wheel, or build bfx with `make bfx` before building "
    "the wheel (docs/getting-started/install.md)"
)


def check_bfx() -> Check:
    """Whether this install carries bfx, root's helper on Docker (REL r4 M7):
    a wheel built from source without ``make bfx`` has none, and runs
    without it."""
    from benchflow._bfx.locate import available

    row = _checker("runtime", "bfx")
    have = [arch for arch in ("x86_64", "aarch64") if available(arch)]
    if have:
        return row("pass", f"bfx {', '.join(have)} (pinned)", details={"bfx": have})
    return row("warn", BFX_MISSING, BFX_FIX, {"bfx": []})


# ── Which bench runs ────────────────────────────────────────────────────

# Both this package and the Rust engine install a program called ``bench``,
# by default into the same folder (~/.local/bin), so which one a shell runs
# depends on install order and PATH. The check names the first
# ``bench`` on PATH and every other one it hides. It tells them apart by
# reading each file, and runs only a BenchFlow one, by its absolute path, for
# its version; relative PATH folders (found from the working folder) are
# skipped.

# Every Rust ``bench`` binary holds this string (its sandbox label); the
# Python entry script and other programs called bench do not. The Rust
# install script (install.sh on bench-rs) tells them apart the same way.
RUST_BENCH_MARKER = b"ai.benchflow.owner"
# A Rust binary is tens of MiB; stop looking for the marker after this much.
_MARKER_SCAN_LIMIT = 256 * 1024**2
# The published install page, not a file path in the repo: the site keeps
# old page paths working with redirects.
INSTALL_GUIDE_URL = "https://www.benchflow.ai/docs/benchflow/getting-started/install"
_ENTRY_NAMES = ("bench", "benchflow")

BenchKind = Literal["python", "rust", "other"]


@dataclass(frozen=True)
class BenchFile:
    """What a ``bench`` file is, read from its bytes without running it.

    ``interpreter`` is a Python entry script's interpreter, from its ``#!``
    line or, for the ``/bin/sh`` form uv and pip write when the path has a
    space, from its ``'''exec'`` line.
    """

    kind: BenchKind
    interpreter: str | None = None


_SH_EXEC_RE = re.compile(rb"^'''exec' '((?:[^']|'\"'\"')+)' ")
# The import line uv and pip write into a ``[project.scripts]`` entry script
# (``from benchflow.cli.main import main``; ``import app`` before 0.8). A
# script that only mentions benchflow is another program and is never run.
_ENTRY_IMPORT_RE = re.compile(rb"^from benchflow(?:\.\w+)* import \w+", re.MULTILINE)


def _script_interpreter(head: bytes) -> str | None:
    lines = head.split(b"\n", 2)
    first = lines[0][2:].strip()
    if first == b"/bin/sh" and len(lines) > 1:
        match = _SH_EXEC_RE.match(lines[1])
        if not match:
            return None
        first = match.group(1).replace(b"'\"'\"'", b"'")
    return first.decode(errors="replace") or None


def read_bench_file(path: Path) -> BenchFile:
    """Tell a Python entry script, a Rust ``bench`` and anything else apart."""
    try:
        with path.open("rb") as fh:
            head = fh.read(4096)
            if head.startswith(b"#!"):
                if _ENTRY_IMPORT_RE.search(head):
                    return BenchFile("python", _script_interpreter(head))
                return BenchFile("other")
            # Chunks overlap by the marker's length, so one cut in two is found.
            keep = len(RUST_BENCH_MARKER) - 1
            chunk, read = head, len(head)
            while chunk:
                if RUST_BENCH_MARKER in chunk:
                    return BenchFile("rust")
                if read >= _MARKER_SCAN_LIMIT:
                    break
                more = fh.read(1024 * 1024)
                read += len(more)
                chunk = chunk[-keep:] + more if more else b""
    except OSError:
        pass
    return BenchFile("other")


def _is_program(path: Path) -> bool:
    # os.path.isfile, unlike Path.is_file on 3.12, returns False for any
    # OSError, so a PATH folder this user cannot search is skipped as a
    # shell skips it.
    return os.path.isfile(path) and os.access(path, os.X_OK)


@dataclass(frozen=True)
class _FoundBench:
    path: Path
    real: str
    file: BenchFile
    this: bool
    version: str | None


def _describe_bench(found: _FoundBench) -> str:
    if found.file.kind == "other":
        return "another program called bench"
    name = "BenchFlow (Python)" if found.file.kind == "python" else "Rust bench"
    return f"{name} {found.version}" if found.version else name


def check_bench_on_path(probes: DoctorProbes) -> Check:
    """Which ``bench`` a shell runs, and which others it hides."""
    from benchflow import __version__

    row = _checker("runtime", "bench on PATH", "bench-path")
    path_var = probes.environ.get("PATH")
    if path_var is None:
        return row("skip", "PATH is not set")
    if probes.system == "Windows":
        # Entry points there are .exe launchers, which this check cannot read.
        return row("skip", "not checked on Windows")
    home = probes.home
    entry = Path(probes.entry_point)
    executable = probes.python_executable
    # This install's own bench: beside its interpreter (a venv, uv's tool
    # environment) or beside the entry point it was started from.
    own = {os.path.realpath(Path(executable).parent / "bench")}
    if entry.name in _ENTRY_NAMES:
        own.add(os.path.realpath(entry.parent / "bench"))
        this_shown = _tilde(entry, home)
    else:
        this_shown = f"{executable} -m benchflow"

    found: list[_FoundBench] = []
    dirs_seen: set[str] = set()
    for folder in path_var.split(os.pathsep):
        # An empty, "." or other relative entry is found from the working
        # folder; skip it rather than read, or run for its version, a bench
        # that depends on where doctor was run (a downloaded task's bin/).
        if not os.path.isabs(folder) or folder in dirs_seen:
            continue
        dirs_seen.add(folder)
        path = Path(folder) / "bench"
        if not _is_program(path):
            continue
        file = probes.read_bench(path)
        real = os.path.realpath(path)
        this = file.kind == "python" and (real in own or file.interpreter == executable)
        # Another bench is never run, not even for its version: what it is
        # comes from its bytes, and bytes prove nothing about what runs (a
        # shell script can carry BenchFlow's import line).
        version = __version__ if this else None
        found.append(_FoundBench(path, real, file, this, version))

    details = {
        "running": this_shown,
        "on_path": [
            {
                "path": str(f.path),
                "resolves_to": f.real,
                "kind": f.file.kind,
                "version": f.version,
                "this_install": f.this,
            }
            for f in found
        ],
    }
    if not found:
        folder = (
            command_arg(entry.parent, "<bin folder>")
            if entry.name in _ENTRY_NAMES
            else None
        )
        return row(
            "warn",
            f"no bench on PATH; this one is {this_shown}",
            f"Add {folder} to PATH (for a uv tool install: uv tool update-shell)"
            if folder
            else "Install the CLI: uv tool install --python 3.12 benchflow",
            details,
        )
    first = found[0]
    hidden: list[_FoundBench] = []
    for f in found[1:]:
        # The same program reached from two folders hides nothing.
        if f.real != first.real and all(f.real != h.real for h in hidden):
            hidden.append(f)
    if not first.this:
        this_dir = next((f.path.parent for f in found if f.this), None)
        if this_dir is None and entry.name in _ENTRY_NAMES:
            this_dir = entry.parent
        if this_dir is None:
            # Started as python -m benchflow, with no entry point of its own
            # on PATH: neither name runs it, and a reinstall would not either.
            fix = (
                "This install has no bench on PATH; run it as "
                + command_arg(executable, "<python executable>")
                + " -m benchflow"
            )
        elif this_dir != first.path.parent:
            fix = (
                "benchflow always runs this install; for bench to run it too, "
                f"put {command_arg(this_dir, '<bin folder>')} before "
                f"{command_arg(first.path.parent, '<bin folder>')} on PATH"
            )
        else:
            # The other bench sits where this install's would: only a
            # reinstall (which replaces it) gives this one the name back.
            # The user's own install command, so its version, extras and
            # constraints stay (a bare one drops them).
            fix = (
                "benchflow always runs this install; for bench to run it too, "
                "run the install command you installed it with again, with "
                f"--force (benchflow=={__version__} and the same extras and "
                "constraints), which replaces "
                f"{command_arg(first.path, '<bench executable>')}"
            )
        return row(
            "warn",
            f"bench runs {_tilde(first.path, home)} ({_describe_bench(first)}), "
            f"not this install ({this_shown})",
            f"{fix}: {INSTALL_GUIDE_URL}#which-bench-runs",
            details,
        )
    summary = f"{_tilde(first.path, home)}, this install"
    if not hidden:
        return row("pass", summary, details=details)
    hides = ", ".join(f"{_tilde(h.path, home)} ({_describe_bench(h)})" for h in hidden)
    return row(
        "warn",
        f"{summary}; it hides {hides}",
        "Run a hidden one by its full path, or put its folder first on PATH: "
        f"{INSTALL_GUIDE_URL}#which-bench-runs",
        details,
    )


# ── Sandbox checks ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class _ColimaState:
    profile: str
    status: str | None = None
    cpus: int | None = None
    memory: int | None = None

    @property
    def spec(self) -> str:
        parts = [f"{self.cpus} CPU" if self.cpus else ""]
        parts.append(_fmt_bytes(self.memory) if self.memory else "")
        return ", ".join(part for part in parts if part)

    @property
    def resize(self) -> str:
        p = self.profile
        return f"colima stop {p} && colima start {p} --memory 8 --cpu 4"


def _docker_context(probes: DoctorProbes) -> str | None:
    host = probes.get("DOCKER_HOST")
    if host:
        # A remote ssh:// DOCKER_HOST carries an ssh user that is often an
        # access token; redact it the way remote_docker does before it reaches
        # a summary line or the details JSON.
        from benchflow.sandbox.remote_docker import _safe as _redact_docker_host

        return f"DOCKER_HOST={_redact_docker_host(host)}"
    result = probes.run(["docker", "context", "show"], PROBE_TIMEOUT_SEC)
    if result.returncode != 0:
        return None
    return _first_line(result.stdout) or None


def _colima_profile(context: str | None) -> str | None:
    if not context:
        return None
    match = re.search(r"\.colima/([^/]+)/docker\.sock", context)
    if match:
        return match.group(1)
    if context == "colima":
        return "default"
    if context.startswith("colima-"):
        return context[len("colima-") :]
    return None


def _colima_state(probes: DoctorProbes, profile: str) -> _ColimaState:
    if probes.which("colima") is None:
        return _ColimaState(profile)
    result = probes.run(["colima", "list", "--json"], PROBE_TIMEOUT_SEC)
    if result.returncode != 0:
        return _ColimaState(profile)
    for line in result.stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("name") == profile:
            cpus = row.get("cpus")
            memory = row.get("memory")
            return _ColimaState(
                profile,
                status=str(row.get("status") or "") or None,
                cpus=cpus if isinstance(cpus, int) else None,
                memory=memory if isinstance(memory, int) else None,
            )
    return _ColimaState(profile)


def check_colima(colima: _ColimaState, *, bad: Status) -> Check:
    row = _checker("sandbox", "colima")
    details = {
        "profile": colima.profile,
        "status": colima.status,
        "cpus": colima.cpus,
        "memory_bytes": colima.memory,
    }
    start = f"colima start {colima.profile}"
    if colima.status is None:
        return row(
            "warn",
            f"docker context points at Colima profile {colima.profile!r}, "
            "but `colima list` does not show it",
            start,
            details,
        )
    if colima.status.lower() != "running":
        spec = f" ({colima.spec})" if colima.spec else ""
        return row(
            bad, f"profile {colima.profile} is {colima.status}{spec}", start, details
        )
    if colima.memory is not None and colima.memory < MIN_DOCKER_MEMORY_BYTES:
        return row(
            "warn",
            f"profile {colima.profile} is running with only {_fmt_bytes(colima.memory)}",
            colima.resize,
            details,
        )
    return row(
        "pass", f"profile {colima.profile} is running ({colima.spec})", details=details
    )


def check_docker(
    probes: DoctorProbes, *, required: bool, buildx: bool = True
) -> list[Check]:
    """Docker CLI, daemon, VM resources, and the Colima profile behind it.

    The Colima line (the likely cause) comes before the Docker line.
    ``buildx=False`` leaves out the buildx plugin check, which can only warn
    and starts two processes (``bench eval run``'s preflight keeps failures).
    """
    row = _checker("sandbox", "docker")
    bad: Status = "fail" if required else "warn"
    linux = probes.system == "Linux"
    if probes.which("docker") is None:
        return [
            row(
                bad,
                "docker CLI not found on PATH",
                "Install Docker Engine: https://docs.docker.com/engine/install/"
                if linux
                else "Install Docker Desktop, OrbStack or Colima (brew install colima docker)",
            )
        ]
    context = _docker_context(probes)
    profile = _colima_profile(context)
    colima = _colima_state(probes, profile) if profile else None
    checks = [check_colima(colima, bad=bad)] if colima else []

    result = probes.run(["docker", "info", "--format", "{{json .}}"], 10.0)
    try:
        parsed = json.loads(result.stdout) if result.stdout.strip() else {}
    except json.JSONDecodeError:
        parsed = {}
    info: dict[str, Any] = parsed if isinstance(parsed, dict) else {}
    via = context or "default"
    if result.returncode != 0 or not info.get("ServerVersion"):
        stderr = result.stderr
        if host := probes.get("DOCKER_HOST"):
            # ssh's own errors name the ssh user ("<user>@host: Permission
            # denied"), which is often a token.
            from benchflow.sandbox.remote_docker import redact_docker_host

            stderr = redact_docker_host(stderr, host)
        reason = (
            _first_line(stderr, secrets=secret_values(probes.environ))
            or "no response from `docker info`"
        )
        if probes.get("DOCKER_HOST"):
            start = (
                f"{context} is set: unset DOCKER_HOST, or point it at a running daemon"
            )
        elif colima:
            start = f"colima start {colima.profile}"
        elif linux:
            start = (
                "sudo systemctl start docker; if the socket's permission is "
                "denied, `sudo usermod -aG docker $USER` and log in again"
            )
        else:
            start = "Start Docker Desktop / OrbStack, or `colima start`"
        checks.append(
            row(
                bad,
                f"daemon unreachable (context {via}): {reason}",
                start,
                {"context": context},
            )
        )
        return checks

    version = str(info.get("ServerVersion"))
    ncpu = info.get("NCPU")
    mem = info.get("MemTotal")
    mem_text = _fmt_bytes(mem) if isinstance(mem, int) else "? GiB"
    root = str(info.get("DockerRootDir") or "")
    # Only a daemon on this machine keeps its data root on this filesystem;
    # Docker Desktop, Colima and a remote DOCKER_HOST keep it in their VM.
    local = context in (None, "default") or (context or "").startswith(
        "DOCKER_HOST=unix://"
    )
    free = probes.disk_free(root) if root and local and not colima else None
    details = {
        "context": context,
        "server_version": version,
        "cpus": ncpu,
        "memory_bytes": mem,
        "os": info.get("OperatingSystem"),
        "root_dir": root or None,
        "free_bytes": free,
    }
    disk_text = f", {_fmt_bytes(free)} free in {root}" if free is not None else ""
    checks.append(
        row(
            "pass",
            f"Docker {version} via {via} ({ncpu} CPU, {mem_text}{disk_text})",
            details=details,
        )
    )
    if free is not None and free < MIN_DOCKER_FREE_BYTES:
        disk_row = _checker("sandbox", "docker disk", "docker-disk")
        checks.append(
            disk_row(
                "warn",
                f"only {_fmt_bytes(free)} free in {root}; task images and agent "
                f"installs need at least {_fmt_bytes(MIN_DOCKER_FREE_BYTES)}",
                "`docker system df` shows what uses it; remove images you no "
                "longer need with `docker image rm`",
                {"root_dir": root, "free_bytes": free},
            )
        )
    buildx_result = (
        probes.run(["docker", "buildx", "version"], PROBE_TIMEOUT_SEC)
        if buildx
        else None
    )
    if buildx_result is not None and buildx_result.returncode != 0:
        buildx_row = _checker("sandbox", "docker buildx", "docker-buildx")
        summary = (
            'buildx plugin not found, so Compose prints "Docker Compose requires '
            'buildx plugin to be installed" and uses the legacy builder, which '
            "cannot build Dockerfiles that use BuildKit features (RUN --mount, "
            "heredocs)"
        )
        fix = (
            "sudo apt-get install docker-buildx-plugin (Debian/Ubuntu; other "
            "distributions: https://docs.docker.com/build/install-buildx/)"
            if linux
            else "Docker Desktop and OrbStack include it; with Colima, run "
            "`brew install docker-buildx` and add "
            '"cliPluginsExtraDirs": ["/opt/homebrew/lib/docker/cli-plugins"] '
            "to ~/.docker/config.json"
        )
        checks.append(buildx_row("warn", summary, fix))
    if isinstance(mem, int) and mem < MIN_DOCKER_MEMORY_BYTES:
        memory_row = _checker("sandbox", "docker memory", "docker-memory")
        fix = (
            colima.resize
            if colima
            else "Give the Docker VM at least 8 GiB (Docker Desktop: Settings > Resources)"
        )
        summary = (
            f"Docker VM has {mem_text}; image builds plus agent installs "
            f"need at least {_fmt_bytes(MIN_DOCKER_MEMORY_BYTES)}"
        )
        checks.append(memory_row("warn", summary, fix, {"memory_bytes": mem}))
    checks.extend(check_docker_leftovers(probes))
    return checks


# The containers BenchFlow's compose files label with the process that made
# them (sandbox/_docker_sweep.py), with that process, project and state.
_LEFTOVER_FORMAT = (
    '{{.ID}}\t{{.Label "benchflow.process"}}\t'
    '{{.Label "com.docker.compose.project"}}\t{{.State}}'
)


def check_docker_leftovers(probes: DoctorProbes) -> list[Check]:
    """Sandboxes that runs which have ended left on the Docker daemon.

    A run stopped hard (killed, the machine asleep, a crash) leaves its
    containers; the next run on this daemon removes only the stopped ones,
    so a container left running keeps its CPU, memory and ports. Counted
    by the ``benchflow.process`` label: a container whose process is gone
    (same host, and another boot, no such pid, or a later process on it).
    Containers of a live run, of another host, or without the label are not
    counted. No row when nothing is left or the listing fails.
    """
    from benchflow.sandbox._docker_sweep import (
        OWNED_LABEL,
        OWNED_VALUE,
        kept_projects,
        owner_state,
    )

    listing = probes.run(
        [
            "docker",
            "ps",
            "-a",
            "--no-trunc",
            "--filter",
            f"label={OWNED_LABEL}={OWNED_VALUE}",
            "--format",
            _LEFTOVER_FORMAT,
        ],
        PROBE_TIMEOUT_SEC,
    )
    if listing.returncode != 0:
        return []
    left: list[tuple[str, str, str]] = []
    # Kept with --keep-sandbox: outlives its run on purpose, and `bench
    # sandbox release` deletes it.
    kept = kept_projects()
    for line in listing.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 4 or not parts[0].strip():
            continue
        cid, token, project, state = (p.strip() for p in parts)
        if project not in kept and owner_state(token) == "gone":
            left.append((cid[:12], project, state))
    if not left:
        return []
    running = sum(1 for _, _, state in left if state in ("running", "restarting"))
    runs = len({project for _, project, _ in left})
    summary = (
        f"{len(left)} container{'s' if len(left) != 1 else ''} left by "
        f"{runs} ended run{'s' if runs != 1 else ''}"
    )
    if running:
        summary += f" ({running} still running)"
    shown = " ".join(cid for cid, _, _ in left[:8])
    more = " ..." if len(left) > 8 else ""
    fix = (
        f"`docker rm -f {shown}{more}` removes them; the next `bench run` "
        "on this daemon removes only the stopped ones"
    )
    row = _checker("sandbox", "docker sandboxes", "docker-leftovers")
    return [
        row(
            "warn",
            summary,
            fix,
            {
                "containers": [cid for cid, _, _ in left],
                "running": running,
                "runs": runs,
            },
        )
    ]


def check_remote_docker(probes: DoctorProbes, *, required: bool) -> Check | None:
    """The Docker host ``--sandbox remote-docker`` runs on: reachable, its
    Docker version, CPUs and memory, named without its ssh user.

    Checked when it is the selected sandbox or ``BENCHFLOW_REMOTE_DOCKER_HOST``
    is set. No free-space figure: ``docker info`` has none, and ``df`` here
    would read this machine's disk.
    """
    from benchflow.sandbox.protocol import SandboxStartupError
    from benchflow.sandbox.remote_docker import (
        REMOTE_DOCKER_HOST_ENV,
        RemoteDockerConfigError,
        resolve_remote_docker_host,
    )

    if not required and probes.get(REMOTE_DOCKER_HOST_ENV) is None:
        return None
    row = _checker("sandbox", "remote-docker")
    bad: Status = "fail" if required else "warn"
    try:
        host = resolve_remote_docker_host(probes.environ, home=probes.home)
    except RemoteDockerConfigError as exc:
        return row(
            bad,
            str(exc),
            f"Set {REMOTE_DOCKER_HOST_ENV}=ssh://user@host[:port], or "
            "tcp://host:2376 with DOCKER_TLS_VERIFY=1 and DOCKER_CERT_PATH",
        )
    details: dict[str, Any] = {"host": host.display}
    if probes.which("docker") is None:
        return row(
            bad,
            "docker CLI not found on PATH; remote-docker runs this machine's "
            "docker CLI against the remote daemon",
            "Install the Docker CLI: https://docs.docker.com/engine/install/",
            details,
        )
    try:
        capacity = probes.remote_docker_info(host, probes.environ, 20.0)
    except SandboxStartupError as exc:
        reason = redact(str(exc), secret_values(probes.environ))
        return row(
            bad,
            reason.removeprefix("Remote Docker host "),
            "Check that this machine's ssh login to the host works and that "
            "the remote user can use Docker (tcp://: the TLS certificates)",
            details,
        )
    memory = capacity.memory_mb * 1024 * 1024
    details |= {
        "server_version": capacity.server_version,
        "cpus": capacity.cpus,
        "memory_bytes": memory,
    }
    return row(
        "pass",
        f"Docker {capacity.server_version} on {host.display} "
        f"({capacity.cpus} CPU, {_fmt_bytes(memory)})",
        details=details,
    )


def check_daytona(
    probes: DoctorProbes, *, required: bool, offline: bool, for_run: bool = False
) -> Check:
    """Doctor's Daytona row.

    *for_run* is the run preflight's form: one try, and an API that does not
    answer (timeout, 429, 5xx, connection error) is a ``warn``, not a
    ``fail``. Sandbox creation retries those within its own budget, so a busy
    API or a busy machine must not refuse the run. A rejected key, or an
    error a retry would only repeat, still fails before any job exists.
    """
    from benchflow.sandbox import daytona_start
    from benchflow.sandbox.providers import extra_install_hint, provider_extra

    row = _checker("sandbox", "daytona")
    key = probes.get("DAYTONA_API_KEY")
    if not required and not key:
        return row(
            "skip", "not configured (set DAYTONA_API_KEY to use --sandbox daytona)"
        )
    bad: Status = "fail" if required else "warn"
    extra = provider_extra("daytona") or "sandbox-daytona"
    install_fix = extra_install_hint(extra)
    sdk = probes.dist_version("daytona")
    details: dict[str, Any] = {"sdk_version": sdk}
    if sdk is None:
        return row(
            bad, "Daytona SDK not installed in this environment", install_fix, details
        )
    if _parse_version(sdk) < MIN_DAYTONA_SDK:
        floor = ".".join(str(p) for p in MIN_DAYTONA_SDK)
        return row(
            bad, f"Daytona SDK {sdk} is older than {floor}", install_fix, details
        )
    if not key:
        return row(
            bad,
            f"Daytona SDK {sdk} installed but DAYTONA_API_KEY is not set",
            "export DAYTONA_API_KEY=... (https://app.daytona.io/dashboard/keys)",
            details,
        )
    origin = probes.origin("DAYTONA_API_KEY")
    api_url = probes.get("DAYTONA_API_URL") or DAYTONA_DEFAULT_API
    details.update(
        {"key": "DAYTONA_API_KEY", "origin": origin, "api_url": _redact_url(api_url)}
    )
    if offline:
        summary = (
            f"SDK {sdk}, DAYTONA_API_KEY set ({origin}); not validated (--offline)"
        )
        return row("pass", summary, details=details)
    tries = 1 if for_run else DOCTOR_DAYTONA_TRIES
    error: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            probes.daytona_check(probes.environ, DAYTONA_CHECK_TIMEOUT_SEC)
        except Exception as exc:
            error = exc
            kind = daytona_start.error_kind(exc)
            wait = daytona_start.wait_sec(exc, attempt)
            if (
                attempt < tries
                and kind in daytona_start.RETRIED_KINDS
                and wait <= daytona_start.READ_MAX_WAIT_SEC
            ):
                probes.sleep(wait)
                continue
        else:
            error = None
        break
    if error is not None:
        reason = _first_line(
            f"{type(error).__name__}: {error}", secrets=secret_values(probes.environ)
        )
        kind = daytona_start.error_kind(error)
        url_set = bool(probes.get("DAYTONA_API_URL"))
        if kind == "config":
            # Only a rejected key refuses (401, or 403 without a quota
            # message): it fails the same way at create.
            fix = "Check the key at https://app.daytona.io/dashboard/keys"
            if url_set:
                fix += " and DAYTONA_API_URL"
            summary = (
                f"SDK {sdk}, but the Daytona API rejected DAYTONA_API_KEY: {reason}"
            )
            return row(bad, summary, fix, details)
        details["unanswered"] = kind
        if for_run:
            # Any other outcome lets the run start: the create reports the
            # real error, with its own retries.
            if kind in daytona_start.RETRIED_KINDS:
                summary = (
                    f"SDK {sdk}, DAYTONA_API_KEY set ({origin}); the key check "
                    f"got no answer ({reason}); sandbox creation retries"
                )
            else:
                summary = (
                    f"SDK {sdk}, DAYTONA_API_KEY set ({origin}); the key check "
                    f"failed without rejecting the key ({reason}); sandbox "
                    "creation reports the error"
                )
            return row("warn", summary, details=details)
        fix = f"Check the network path to {_host(api_url)}"
        if url_set:
            fix += " and DAYTONA_API_URL"
        if kind in daytona_start.RETRIED_KINDS:
            summary = (
                f"SDK {sdk}, but the Daytona API did not answer after {tries} "
                f"tries ({reason}); the key was not checked"
            )
        else:
            summary = (
                f"SDK {sdk}, but the Daytona API answered the key check with an "
                f"error that does not reject the key ({reason})"
            )
        return row(bad, summary, fix + ", then run doctor again", details)
    return row(
        "pass",
        f"SDK {sdk}, DAYTONA_API_KEY accepted by {_host(api_url)}",
        details=details,
    )


# ── Agent credentials ───────────────────────────────────────────────────


def _env_source(probes: DoctorProbes, name: str, kind: str) -> CredentialSource | None:
    if probes.get(name) is None:
        return None
    return CredentialSource(name, kind, probes.origin(name))


def claude_auth(probes: DoctorProbes) -> AgentAuth:
    """Claude Code auth, in Claude's own precedence order.

    ``~/.claude/.credentials.json`` is only used when no env credential is set.
    On macOS the Claude CLI keeps live logins in the Keychain, so this file is
    often a stale copy whose access token expired long ago.
    """
    sources: list[CredentialSource] = []
    for name, kind in (
        ("ANTHROPIC_API_KEY", "api-key"),
        ("ANTHROPIC_AUTH_TOKEN", "auth-token"),
        ("CLAUDE_CODE_OAUTH_TOKEN", "oauth-token"),
        ("CLAUDE_OAUTH_TOKEN", "oauth-token"),
    ):
        src = _env_source(probes, name, kind)
        if src:
            sources.append(src)
    path, label = _login_file(probes, "~/.claude/.credentials.json")
    if path.is_file():
        try:
            data = probes.read_json(path)
            oauth = data.get("claudeAiOauth") if isinstance(data, dict) else None
        except (OSError, ValueError):
            oauth = None
        if not isinstance(oauth, dict) or not oauth.get("accessToken"):
            sources.append(
                CredentialSource(
                    label,
                    "login-file",
                    "file",
                    usable=False,
                    note="no OAuth token in file",
                )
            )
        else:
            expires = _epoch_ms(oauth.get("expiresAt"))
            expired = expires is not None and expires <= probes.now()
            sources.append(
                CredentialSource(
                    label,
                    "login-file",
                    "file",
                    expires_at=expires,
                    usable=not expired,
                    note=(
                        "usable only if its refresh token still works"
                        if expired
                        else ""
                    ),
                )
            )
    notes: list[str] = []
    env_names = [s.name for s in sources if s.origin != "file"]
    if "ANTHROPIC_API_KEY" in env_names and any(
        n in env_names for n in ("CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_OAUTH_TOKEN")
    ):
        notes.append(
            "ANTHROPIC_API_KEY overrides the OAuth token; "
            "unset it to run on the subscription"
        )
    effective = sources[0] if sources else None
    endpoints: tuple[str, ...] = ()
    if effective is not None:
        base = probes.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
        endpoints = (base.rstrip("/") + "/",)
    return AgentAuth(
        "claude-agent-acp", "Claude", tuple(sources), effective, endpoints, tuple(notes)
    )


def _codex_auth_json_source(
    raw: Any, name: str, origin: str, probes: DoctorProbes
) -> CredentialSource:
    if not isinstance(raw, dict):
        return CredentialSource(
            name, "auth-json", origin, usable=False, note="not a JSON object"
        )
    if raw.get("OPENAI_API_KEY"):
        # Codex signs in with this key, not a ChatGPT login: an API key, whose
        # endpoint is the API's.
        return CredentialSource(
            name, "api-key", origin, note="API key stored in file; sign-in not tested"
        )
    tokens = raw.get("tokens")
    if not isinstance(tokens, dict) or not tokens.get("refresh_token"):
        return CredentialSource(
            name, "auth-json", origin, usable=False, note="no ChatGPT tokens in file"
        )
    claims = _jwt_claims(tokens.get("access_token"))
    exp = claims.get("exp")
    expires = (
        datetime.fromtimestamp(exp, UTC)
        if isinstance(exp, int | float) and not isinstance(exp, bool)
        else None
    )
    auth_claims = claims.get("https://api.openai.com/auth")
    plan = (
        auth_claims.get("chatgpt_plan_type") if isinstance(auth_claims, dict) else None
    )
    parts = [f"ChatGPT login (auth_mode={raw.get('auth_mode') or 'chatgpt'})"]
    if isinstance(plan, str) and plan:
        parts.append(f"plan {plan}")
    last = raw.get("last_refresh")
    if isinstance(last, str):
        try:
            refreshed = datetime.fromisoformat(last.replace("Z", "+00:00"))
            if refreshed.tzinfo is None:
                refreshed = refreshed.replace(tzinfo=UTC)
            days = max(0, int((probes.now() - refreshed).total_seconds() // 86400))
            parts.append(f"last refreshed {days} days ago")
        except ValueError:
            pass
    # Codex refreshes an expired access token itself, so expiry is informational.
    return CredentialSource(
        name,
        "auth-json",
        origin,
        expires_at=expires,
        note=", ".join(parts),
        self_refreshing=True,
    )


def codex_auth(probes: DoctorProbes) -> AgentAuth:
    """Codex auth in BenchFlow's precedence order (API key wins over logins)."""
    sources: list[CredentialSource] = []
    for name, kind in (
        ("OPENAI_API_KEY", "api-key"),
        ("CODEX_API_KEY", "api-key"),
        ("CODEX_ACCESS_TOKEN", "access-token"),
    ):
        src = _env_source(probes, name, kind)
        if src:
            sources.append(src)
    inline = probes.get("CODEX_AUTH_JSON")
    if inline is not None:
        try:
            raw: Any = json.loads(inline)
        except json.JSONDecodeError:
            raw = None
        sources.append(
            _codex_auth_json_source(
                raw, "CODEX_AUTH_JSON", probes.origin("CODEX_AUTH_JSON"), probes
            )
        )
    path, label = _login_file(probes, "~/.codex/auth.json")
    if path.is_file():
        try:
            raw = probes.read_json(path)
        except (OSError, ValueError):
            raw = None
        sources.append(_codex_auth_json_source(raw, label, "file", probes))
    notes: list[str] = []
    has_key = any(s.kind == "api-key" for s in sources)
    has_login = any(s.kind in ("auth-json", "access-token") for s in sources)
    if has_key and has_login:
        notes.append(
            "OPENAI_API_KEY/CODEX_API_KEY override the ChatGPT login; "
            "unset them to run on the subscription"
        )
    base = probes.get("OPENAI_BASE_URL")
    if base and base.rstrip("/") != "https://api.openai.com/v1" and has_login:
        notes.append(
            "OPENAI_BASE_URL points at a custom endpoint; "
            "ChatGPT login only works against OpenAI itself"
        )
    effective = sources[0] if sources else None
    endpoints: tuple[str, ...] = ()
    if effective is not None:
        if effective.kind == "api-key":
            endpoints = ((base or "https://api.openai.com/v1").rstrip("/") + "/",)
        else:
            endpoints = ("https://chatgpt.com/", "https://auth.openai.com/")
    return AgentAuth(
        "codex-acp", "Codex", tuple(sources), effective, endpoints, tuple(notes)
    )


def gemini_auth(probes: DoctorProbes) -> AgentAuth:
    sources: list[CredentialSource] = []
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        src = _env_source(probes, name, "api-key")
        if src:
            sources.append(src)
    path = probes.home / ".gemini" / "oauth_creds.json"
    if path.is_file():
        label = _tilde(path, probes.home)
        try:
            data = probes.read_json(path)
        except (OSError, ValueError):
            data = None
        if not isinstance(data, dict) or not data.get("refresh_token"):
            sources.append(
                CredentialSource(
                    label, "login-file", "file", usable=False, note="no refresh token"
                )
            )
        else:
            # The mandatory LiteLLM proxy (#820) routes every Gemini run and
            # needs an API key; it cannot use this login, so it is not a
            # working credential for BenchFlow even though the file is valid.
            sources.append(
                CredentialSource(
                    label,
                    "login-file",
                    "file",
                    usable=False,
                    expires_at=_epoch_ms(data.get("expiry_date")),
                    note=(
                        "Google login, not used: BenchFlow runs Gemini through "
                        "its LiteLLM proxy, which needs GEMINI_API_KEY"
                    ),
                    self_refreshing=True,
                )
            )
    notes: list[str] = []
    adc = probes.get("GOOGLE_APPLICATION_CREDENTIALS") or str(
        probes.home / ".config" / "gcloud" / "application_default_credentials.json"
    )
    if Path(adc).is_file() and probes.get("GOOGLE_CLOUD_PROJECT"):
        notes.append(
            "Vertex ADC + GOOGLE_CLOUD_PROJECT available for google-vertex/ models"
        )
    effective = sources[0] if sources else None
    endpoints: tuple[str, ...] = ()
    if effective is not None and effective.kind == "api-key":
        endpoints = ("https://generativelanguage.googleapis.com/",)
    return AgentAuth(
        "gemini", "Gemini", tuple(sources), effective, endpoints, tuple(notes)
    )


def bedrock_auth(probes: DoctorProbes) -> AgentAuth:
    token = _env_source(probes, "AWS_BEARER_TOKEN_BEDROCK", "api-key")
    region = probes.get("AWS_REGION") or probes.get("AWS_DEFAULT_REGION")
    if token is None:
        return AgentAuth("bedrock", "Bedrock", (), None)
    if region is None:
        token = CredentialSource(
            token.name,
            token.kind,
            token.origin,
            usable=False,
            note="AWS_REGION / AWS_DEFAULT_REGION not set",
        )
    endpoints = (f"https://bedrock-runtime.{region}.amazonaws.com/",) if region else ()
    notes = (f"region {region}; use aws-bedrock/<model> ids",) if region else ()
    return AgentAuth("bedrock", "Bedrock", (token,), token, endpoints, notes)


# Keys owned by the per-agent checks above, and GITHUB_TOKEN, which is usually
# set for git rather than for GitHub Models.
_NOT_PROVIDER_KEYS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GITHUB_TOKEN",
    }
)
# A URL parameter that another variable can stand in for.
_URL_PARAM_ALTERNATIVES = {"AZURE_RESOURCE": "AZURE_API_ENDPOINT"}


def provider_key_sources(probes: DoctorProbes) -> list[CredentialSource]:
    """Provider API keys usable through provider-prefixed ``--model`` ids."""
    from benchflow.agents.providers import PROVIDERS

    by_key: dict[str, list[tuple[str, list[str]]]] = {}
    for provider_name, cfg in PROVIDERS.items():
        key = cfg.auth_env
        if not key or key in _NOT_PROVIDER_KEYS or probes.get(key) is None:
            continue
        # A parameter with a default (DeepSeek's base URL) is never missing:
        # the run uses the default when the variable is unset.
        missing = [
            var
            for param, var in cfg.url_params.items()
            if not cfg.url_param_defaults.get(param)
            and probes.get(var) is None
            and probes.get(_URL_PARAM_ALTERNATIVES.get(var, var)) is None
        ]
        by_key.setdefault(key, []).append((provider_name, missing))
    sources: list[CredentialSource] = []
    for key in sorted(by_key):
        ready = [name for name, missing in by_key[key] if not missing]
        if ready:
            note = "for " + ", ".join(f"{name}/" for name in ready)
            sources.append(
                CredentialSource(key, "api-key", probes.origin(key), note=note)
            )
            continue
        name, missing = by_key[key][0]
        sources.append(
            CredentialSource(
                key,
                "api-key",
                probes.origin(key),
                usable=False,
                note=f"{name}/ also needs {', '.join(missing)}",
            )
        )
    llm = _env_source(probes, "LLM_API_KEY", "api-key")
    if llm:
        sources.append(
            CredentialSource(llm.name, llm.kind, llm.origin, note="for openhands")
        )
    return sources


def _describe_source(src: CredentialSource, now: datetime) -> str:
    text = f"{src.name} ({src.origin})" if src.origin != "file" else src.name
    extras: list[str] = []
    if src.expires_at is not None and not src.self_refreshing:
        extras.append("access token " + _fmt_when(src.expires_at, now))
    if src.note:
        extras.append(src.note)
    return text + (f": {'; '.join(extras)}" if extras else "")


_AGENT_FIXES = {
    "claude-agent-acp": (
        "Run `claude setup-token` and export CLAUDE_CODE_OAUTH_TOKEN=<token> "
        "(or export ANTHROPIC_API_KEY)"
    ),
    "codex-acp": "Run `codex login` (ChatGPT) or export OPENAI_API_KEY",
    "gemini": "export GEMINI_API_KEY=<key> (Google AI Studio: https://aistudio.google.com/apikey)",
    "bedrock": "export AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION=us-west-2",
}


def check_agent_auth(auth: AgentAuth, now: datetime) -> Check:
    details = {
        "effective": auth.effective.to_dict() if auth.effective else None,
        "sources": [s.to_dict() for s in auth.sources],
        "notes": list(auth.notes),
    }
    row = _checker("agents", auth.agent, f"auth.{auth.agent}")
    if auth.effective is None:
        return row(
            "skip", "no credential found", _AGENT_FIXES.get(auth.agent, ""), details
        )
    parts = [_describe_source(auth.effective, now)]
    others = auth.sources[1:]
    for src in others:
        if not src.usable:
            parts.append(f"ignored {_describe_source(src, now)}")
    also = [s.name for s in others if s.usable]
    if also:
        parts.append("also found " + ", ".join(also))
    parts.extend(auth.notes)
    summary = "; ".join(parts)
    if not auth.effective.usable:
        fix = _AGENT_FIXES.get(auth.agent, "")
        if auth.agent == "claude-agent-acp" and auth.effective.kind == "login-file":
            fix += (
                ". On macOS the Claude CLI keeps live logins in the Keychain, "
                "so ~/.claude/.credentials.json is not refreshed"
            )
        return row("warn", summary, fix, details)
    return row("pass", summary, details=details)


def _claude_token(probes: DoctorProbes, src: CredentialSource) -> str | None:
    """The subscription OAuth token behind a Claude credential, or None.

    Only ``CLAUDE_CODE_OAUTH_TOKEN``/``CLAUDE_OAUTH_TOKEN`` and the login file
    hold one; ``ANTHROPIC_AUTH_TOKEN`` is usually a gateway's token, which
    must not be sent to api.anthropic.com.
    """
    if src.kind == "oauth-token":
        return probes.get(src.name)
    if src.kind == "login-file" and src.usable:
        try:
            data = probes.read_json(
                _login_file(probes, "~/.claude/.credentials.json")[0]
            )
        except (OSError, ValueError):
            return None
        oauth = data.get("claudeAiOauth") if isinstance(data, dict) else None
        token = oauth.get("accessToken") if isinstance(oauth, dict) else None
        return token if isinstance(token, str) and token else None
    return None


def _window_text(headroom: Any, window: str, now: datetime) -> str | None:
    from benchflow.agents.usage_limits import format_reset

    used = headroom.used.get(window)
    if used is None:
        return None
    resets = headroom.resets.get(window)
    reset = f", resets {format_reset(resets)}" if resets is not None else ""
    return f"{window} window {used:.0%} used{reset}"


def check_claude_headroom(
    probes: DoctorProbes, auth: AgentAuth, *, offline: bool
) -> Check | None:
    """The Claude subscription login: accepted, and how much of each window is left.

    One 8-token Haiku request with the login's OAuth token (the headers a
    spent login answers with are the same, on HTTP 429). None when Claude's
    effective credential is an API key or absent: there is no subscription.
    """
    src = auth.effective
    if src is None or src.kind == "api-key":
        return None
    row = _checker("agents", "claude usage", "usage.claude-agent-acp")
    who = f"{src.name} ({src.origin})" if src.origin != "file" else src.name
    if offline:
        return row("skip", f"{who}: usage not checked (--offline)")
    if src.kind not in ("oauth-token", "login-file"):
        return row(
            "skip",
            f"{who}: usage not checked (only a subscription's OAuth token is, "
            "and only against api.anthropic.com)",
        )
    custom = probes.get("ANTHROPIC_BASE_URL")
    if custom and _host(custom) != _host(ANTHROPIC_API):
        return row(
            "skip",
            f"{who}: usage not checked: ANTHROPIC_BASE_URL points at "
            f"{_host(custom)}, and the check sends the token only to "
            f"{_host(ANTHROPIC_API)}",
        )
    token = _claude_token(probes, src)
    if token is None:
        return row(
            "skip",
            f"{who}: no usable OAuth token to check (see the claude-agent-acp line)",
        )
    from benchflow.agents.usage_limits import format_reset, parse_unified_headers

    status, headers, error = probes.claude_headroom(
        token, ANTHROPIC_API, HEADROOM_TIMEOUT_SEC
    )
    secrets = [token, *secret_values(probes.environ)]
    request = f"one 8-token {HEADROOM_MODEL} request"
    if status is None:
        return row(
            "warn",
            f"{who}: could not check its usage ({_first_line(error, secrets=secrets)})",
            f"Check that {_host(ANTHROPIC_API)} is reachable; "
            "HTTPS_PROXY is honored if set",
        )
    headroom = parse_unified_headers(headers)
    details: dict[str, Any] = {"http_status": status, "request": request}
    if headroom is not None:
        details.update(
            {
                "status": headroom.status,
                "used": headroom.used,
                "resets": {w: t.isoformat() for w, t in headroom.resets.items()},
                "rejected": list(headroom.rejected),
            }
        )
    if status in (401, 403):
        details["blocks_runs"] = True
        return row(
            "warn",
            f"{who} was refused (HTTP {status}): the login is invalid or expired",
            _AGENT_FIXES["claude-agent-acp"],
            details,
        )
    if headroom is None:
        if status == 200:
            return row(
                "pass",
                f"{who} accepted; the answer carried no usage windows ({request})",
                details=details,
            )
        if status == 429:
            # Refused for want of quota, with no window to name: still a
            # login no run can use, so it counts as unusable like a spent one.
            details["blocks_runs"] = True
        return row(
            "warn",
            f"{who}: HTTP {status} and no usage windows ({request})",
            _AGENT_FIXES["claude-agent-acp"],
            details,
        )
    now = probes.now()
    windows = [
        text
        for window in ("5-hour", "7-day", "7-day Opus", "7-day Sonnet")
        if (text := _window_text(headroom, window, now)) is not None
    ]
    if headroom.limited:
        details["blocks_runs"] = True
        spent = headroom.window or ", ".join(headroom.rejected) or "a"
        when = format_reset(headroom.resets_at)
        return row(
            "warn",
            f"{who} is out of usage: its {spent} window is spent until {when}"
            + (f" ({'; '.join(windows)})" if windows else ""),
            "Claude runs on this login stop at once with a usage-limit error "
            f"until {when}: use another login (CLAUDE_CODE_OAUTH_TOKEN from "
            "`claude setup-token` on another account) or wait",
            details,
        )
    summary = f"{who} accepted: " + ("; ".join(windows) or "no window reported")
    high = [w for w in ("5-hour", "7-day") if (headroom.used.get(w) or 0.0) >= 0.9]
    if high:
        return row(
            "warn",
            summary + f" ({request})",
            f"The {' and '.join(high)} window is nearly spent; a long run may stop "
            "on the usage limit",
            details,
        )
    return row("pass", summary + f" ({request})", details=details)


def check_model_proxy(probes: DoctorProbes, *, sandbox: str) -> Check:
    """The LiteLLM proxy that API-key and provider runs go through.

    On Docker it runs on this machine (the ``litellm`` next to this Python);
    on the providers that run it in the sandbox (Daytona, remote Docker and
    the rest of ``ModelProxyLocation.SANDBOX``) it is installed there from
    PyPI. Subscription runs (a Claude or ChatGPT login) do not use it.
    """
    from benchflow.providers.litellm_runtime import LITELLM_VERSION_SPEC

    row = _checker("proxy", "litellm", "proxy.litellm")
    version = probes.dist_version("litellm")
    sibling = Path(probes.python_executable).with_name("litellm")
    executable = str(sibling) if sibling.exists() else probes.which("litellm")
    details = {
        "version": version,
        "executable": executable,
        "pinned": LITELLM_VERSION_SPEC,
        "sandbox": sandbox,
    }
    from benchflow.sandbox.providers import PROVIDERS_BY_NAME, ModelProxyLocation

    provider = PROVIDERS_BY_NAME.get(sandbox)
    in_sandbox = (
        provider is not None and provider.model_proxy is ModelProxyLocation.SANDBOX
    )
    where = (
        "installed inside each sandbox from PyPI"
        if in_sandbox
        else "runs on this machine"
    )
    custom = [
        f"{name}={_redact_url(value)}"
        for name in ("BENCHFLOW_PROVIDER_BASE_URL", "LLM_BASE_URL")
        if (value := probes.get(name))
    ]
    tail = f"; custom endpoint {', '.join(custom)}" if custom else ""
    if executable is None and not in_sandbox:
        return row(
            "warn",
            f"LiteLLM CLI not found, so API-key and provider-routed runs on "
            f"{sandbox} cannot start their model proxy (subscription logins "
            f"do not need it){tail}",
            "Reinstall BenchFlow so its pinned "
            f"{LITELLM_VERSION_SPEC} is installed next to it "
            "(`uv tool install --reinstall benchflow`)",
            details,
        )
    shown = f"LiteLLM {version}" if version else "LiteLLM"
    return row(
        "pass",
        f"{shown} for API-key and provider runs, {where}{tail}",
        details=details,
    )


def check_provider_keys(sources: list[CredentialSource], now: datetime) -> Check:
    row = _checker("agents", "provider keys", "auth.providers")
    details = {"sources": [s.to_dict() for s in sources]}
    if not sources:
        return row(
            "skip",
            "no other provider API keys (DEEPSEEK_API_KEY, OPENROUTER_API_KEY, ...)",
            details=details,
        )
    status: Status = "pass" if any(s.usable for s in sources) else "warn"
    fix = "" if status == "pass" else "Set the missing base-URL variables listed above"
    summary = (
        "; ".join(_describe_source(s, now) for s in sources)
        + ": set, not tested against the provider (use with openhands, opencode or "
        "pi-acp and a provider-prefixed --model)"
    )
    return row(status, summary, fix, details)


# ── Agent versions ──────────────────────────────────────────────────────


_VERSION_AGENTS = (
    ("claude-agent-acp", "claude"),
    ("codex-acp", "codex"),
    ("gemini", "gemini"),
)
_PIN_RE = re.compile(r"npm install -g --prefix \S+ (\S+@\d[\w.\-]*)")
# Pinned packages an agent's install carries besides its own: claude-agent-acp
# runs the separately pinned Claude Code CLI, which the no-web gate verifies.
_COMPANION_PINS = {"claude-agent-acp": ("claude-code",)}


def _installed_pin(agent: str) -> str | None:
    from benchflow.agents.registry import AGENTS

    cfg = AGENTS.get(agent)
    if cfg is None:
        return None
    return " + ".join(_PIN_RE.findall(cfg.install_cmd)) or None


def _builtin_pin(agent: str) -> str | None:
    from benchflow.agents.registry import pinned_npm_package

    try:
        pins = [
            pinned_npm_package(name)
            for name in (agent, *_COMPANION_PINS.get(agent, ()))
        ]
    except KeyError:
        return None
    return " + ".join(f"{package}@{version}" for package, version in pins)


def check_agent_versions(probes: DoctorProbes) -> list[Check]:
    """Sandbox pins from the registry next to the host CLI versions.

    The sandbox always installs the registry pin; the host CLI is used only to
    log in (``claude setup-token``, ``codex login``), so a newer host CLI is
    informational. A registry entry whose install command no longer carries
    the built-in pin (a manifest override) is flagged: the Codex Apps and
    Claude no-web gates verify the built-in versions.
    """

    def _host_version(binary: str) -> str | None:
        if probes.which(binary) is None:
            return None
        result = probes.run([binary, "--version"], 10.0)
        if result.returncode != 0:
            return None
        match = re.search(r"\d+\.\d+\.\d+[\w.\-+]*", result.stdout)
        return match.group(0) if match else _first_line(result.stdout) or None

    with ThreadPoolExecutor(max_workers=len(_VERSION_AGENTS)) as pool:
        host_versions = list(
            pool.map(lambda pair: _host_version(pair[1]), _VERSION_AGENTS)
        )
    checks: list[Check] = []
    for (agent, binary), host in zip(_VERSION_AGENTS, host_versions, strict=True):
        row = _checker("versions", agent, f"version.{agent}")
        pin = _installed_pin(agent)
        builtin = _builtin_pin(agent)
        details = {"sandbox_pin": pin, "builtin_pin": builtin, "host_cli": host}
        host_text = (
            f"host {binary} {host} (login only)"
            if host
            else f"host {binary} CLI not found (only needed to log in)"
        )
        if builtin and pin != builtin:
            summary = (
                f"registry installs {pin or 'an unpinned build'}, not the built-in "
                f"{builtin}; {host_text}"
            )
            fix = (
                "Remove the agent manifest override, or re-run the policy "
                "conformance fixtures for the new version"
            )
            checks.append(row("warn", summary, fix, details))
            continue
        summary = f"sandbox installs {pin or 'unpinned'}; {host_text}"
        checks.append(row("pass", summary, details=details))
    return checks


# ── Network ─────────────────────────────────────────────────────────────


def check_network(
    probes: DoctorProbes,
    required: Mapping[str, str],
    optional: Mapping[str, str],
) -> tuple[list[Check], frozenset[str]]:
    """Probe each URL once; ``required`` failures are ``fail``.

    ``required``/``optional`` map URL -> why it is probed.
    """
    urls = list(dict.fromkeys([*required, *optional]))
    secrets = secret_values(probes.environ)

    def _probe(url: str) -> tuple[str, int | None, str]:
        try:
            return url, probes.http_status(url, PROBE_TIMEOUT_SEC), ""
        except Exception as exc:
            return (
                url,
                None,
                _first_line(f"{type(exc).__name__}: {exc}", secrets=secrets),
            )

    with ThreadPoolExecutor(max_workers=max(1, min(8, len(urls)))) as pool:
        results = list(pool.map(_probe, urls))
    checks: list[Check] = []
    unreachable: set[str] = set()
    for url, status_code, error in results:
        why = required.get(url) or optional.get(url, "")
        host = _host(url)
        row = _checker("network", host, f"net.{host}")
        if status_code is not None:
            # Any HTTP answer proves egress; a 404 or 401 here is expected, so
            # the status is kept in the details, not shown next to PASS.
            summary = f"reachable — {why}"
            details = {"url": _redact_url(url), "http_status": status_code}
            checks.append(row("pass", summary, details=details))
            continue
        unreachable.add(url)
        checks.append(
            row(
                "fail" if url in required else "warn",
                f"unreachable — {why}: {error}",
                "Check VPN/proxy/firewall; HTTPS_PROXY is honored if set",
                {"url": _redact_url(url), "error": error},
            )
        )
    return checks, frozenset(unreachable)


# ── Orchestration ───────────────────────────────────────────────────────


def run_doctor(
    *,
    sandbox: str = "docker",
    offline: bool = False,
    probes: DoctorProbes | None = None,
) -> DoctorReport:
    probes = probes or DoctorProbes.from_host()
    now = probes.now()
    checks: list[Check] = [
        check_python(probes),
        check_uv(probes),
        check_bench_on_path(probes),
        check_bfx(),
    ]

    checks.extend(check_docker(probes, required=sandbox == "docker"))
    remote = check_remote_docker(probes, required=sandbox == "remote-docker")
    if remote is not None:
        checks.append(remote)
    checks.append(check_daytona(probes, required=sandbox == "daytona", offline=offline))
    if sandbox not in ("docker", "daytona", "remote-docker"):
        row = _checker("sandbox", sandbox, f"sandbox.{sandbox}")
        checks.append(row("skip", f"doctor has no checks for --sandbox {sandbox} yet"))

    auths = {
        auth.agent: auth
        for auth in (
            claude_auth(probes),
            codex_auth(probes),
            gemini_auth(probes),
            bedrock_auth(probes),
        )
    }
    headroom = check_claude_headroom(probes, auths["claude-agent-acp"], offline=offline)
    for auth in auths.values():
        checks.append(check_agent_auth(auth, now))
        if auth.agent == "claude-agent-acp" and headroom is not None:
            checks.append(headroom)
    provider_sources = provider_key_sources(probes)
    checks.append(check_provider_keys(provider_sources, now))
    agents_end = len(checks)
    checks.extend(check_agent_versions(probes))
    checks.append(check_model_proxy(probes, sandbox=sandbox))

    unreachable: frozenset[str] = frozenset()
    if offline:
        checks.append(_checker("network", "network")("skip", "skipped (--offline)"))
    else:
        required: dict[str, str] = {}
        optional: dict[str, str] = {}
        if sandbox == "docker":
            required[DOCKER_REGISTRY_ENDPOINT] = "base image pulls"
        if sandbox == "daytona" or probes.get("DAYTONA_API_KEY"):
            url = (probes.get("DAYTONA_API_URL") or DAYTONA_DEFAULT_API).rstrip("/")
            (required if sandbox == "daytona" else optional)[url + "/"] = "Daytona API"
        for url in AGENT_INSTALL_ENDPOINTS:
            required[url] = "agent install inside the sandbox"
        if sandbox == "daytona":
            optional[PYPI_ENDPOINT] = "model proxy install inside the sandbox"
        for name in ("BENCHFLOW_PROVIDER_BASE_URL", "LLM_BASE_URL"):
            if value := probes.get(name):
                optional.setdefault(value.rstrip("/") + "/", f"model proxy ({name})")
        for auth in auths.values():
            if auth.effective is None:
                continue
            for url in auth.endpoints:
                optional.setdefault(url, f"{auth.label} model API")
        net_checks, unreachable = check_network(probes, required, optional)
        checks.extend(net_checks)

    # Only a spent or refused login stops Claude runs; a nearly spent window,
    # a network blip or a 5xx does not.
    claude_spent = headroom is not None and bool(headroom.details.get("blocks_runs"))
    usable = [
        a
        for a in auths.values()
        if a.ready
        and not any(url in unreachable for url in a.endpoints)
        and not (a.agent == "claude-agent-acp" and claude_spent)
    ]
    if not usable and not any(s.usable for s in provider_sources):
        found_any = bool(provider_sources) or any(
            a.effective is not None for a in auths.values()
        )
        why = (
            "every credential found is expired, spent, incomplete or unreachable"
            if found_any
            else "no credential found for any agent"
        )
        fix = (
            "Log in with `codex login` or `claude setup-token`, or export an API key; "
            "see https://github.com/benchflow-ai/benchflow/blob/main/docs/agents/auth.md"
        )
        row = _checker("agents", "agent credentials", "auth.any")
        # A warning: the oracle and nop controls need no credential. Listed
        # with the other credential lines, not after the network ones.
        checks.insert(
            agents_end,
            row(
                "warn",
                f"no model agent can run: {why} (the oracle and nop controls still can)",
                fix,
            ),
        )
    return DoctorReport(checks, sandbox, auths, unreachable)


def host_summary() -> str:
    return f"{platform.system()} {platform.machine()}"
