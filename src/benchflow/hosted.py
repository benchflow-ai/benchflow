"""Hosted evals: runs that the hosted service executes on its own servers, not this machine.

``bench hosted`` and the functions below talk to the hosted-eval service (its ``/api/v1`` API):

- the address is ``--hub``, else ``BENCHFLOW_HOSTED_URL``; there is no default, and with neither every command
  stops before sending anything. It must be https, except to this machine (localhost, 127.0.0.1);
- the key is sent as ``Authorization: Bearer``. It comes from ``BENCHFLOW_HUB_KEY`` (or ``BENCHFLOW_API_KEY``), else
  from a key file that only its owner can read (``--key-file``, ``BENCHFLOW_HUB_KEY_FILE``, or
  ``~/.config/benchflow/hub-key``); never from a command's arguments. A BenchFlow Hub key (``bfk_`` and 36 base62
  characters) is checked offline before any call; the service's own ``bh_<prefix>_<secret>`` keys are sent as they
  are.
  The key is never printed or logged: messages name only its kind (``bfk_…``, ``bh_…``). The client masks the key,
  its secret parts and anything else shaped like a key in what the service sends back: error messages, run records,
  events, metrics, traces and logs (a trajectories export is written to its file as sent).

The wire format: ``POST /api/v1/runs`` queues a run of ``env_id`` (a hub version id, ``PKG-...``) and answers
``{"run_id", "warnings", "runtime"}``; ``POST /api/v1/runs/estimate`` takes the same body and starts nothing. ``GET
/api/v1/runs/{id}`` is the run record (``status`` queued, starting or running while it goes; ``error``, ``n_traces``,
``n_expected``, ``mean_reward``); ``/events`` its events, ``/log`` its log, ``/metrics`` its reward summary, ``GET
/api/v1/traces?run_id={id}`` its trials, ``/export?format=atif`` (or ``messages``) its trajectories as JSON lines, and
``POST /cancel`` stops it. The run's page is ``{base}/runs/{id}``. A 503 means "try again": a ``bfk_`` key gets one
while the hub's key service is down, which says nothing about the key. Standard library only, like the Hub client
(:mod:`benchflow.hub.client`), whose redirect rule it shares: the key goes only to the origin it was sent to.
"""

from __future__ import annotations

import errno
import json
import os
import re
import stat
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request

from benchflow.errors import UserError
from benchflow.hub.client import _OPENER, _DowngradeRefused

HOSTED_URL_ENV = "BENCHFLOW_HOSTED_URL"
KEY_ENV = "BENCHFLOW_HUB_KEY"
#: Read when ``BENCHFLOW_HUB_KEY`` is not set.
ALT_KEY_ENV = "BENCHFLOW_API_KEY"
KEY_FILE_ENV = "BENCHFLOW_HUB_KEY_FILE"
DEFAULT_KEY_FILE = Path("~/.config/benchflow/hub-key")
#: Skips the offline checksum of a ``bfk_`` key (the key service still checks it).
SKIP_CHECKSUM_ENV = "BENCHFLOW_HUB_KEY_NO_CHECKSUM"
RUNS_PATH = "/api/v1/runs"
TRACES_PATH = "/api/v1/traces"
SECRETS_PATH = "/api/v1/secrets"
#: Providers whose own API key the service's key proxy takes. The key stays on the service; the run's sandbox gets a
#: token for that run only. An account key (``bfk_``) may store only these kinds.
KEY_KINDS = ("anthropic", "deepseek", "kimi", "zai", "mimo")
#: A run in one of these is still going; every other status is final.
ACTIVE_STATUSES = frozenset({"queued", "starting", "running"})
DEFAULT_AGENT = "oracle"
EXPORT_FORMATS = ("atif", "messages")

_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_BFK = re.compile(r"^bfk_([0-9A-Za-z]{30})([0-9A-Za-z]{6})$")
# The hosted service's own keys: bh_, a prefix of letters and digits, _, a URL-safe secret.
_BH = re.compile(r"^bh_[0-9A-Za-z]+_[0-9A-Za-z_-]+$")
_TIMEOUT_S = 60
_CHUNK = 1 << 20
_DETAIL_MAX = 500
_TRACES_PAGE = 200
_KEY_FILE_MAX = 4096


class HostedError(UserError):
    """The hosted service could not be reached, refused a request, or answered something unreadable."""

    exit_code: int = 1

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        hint: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        if hint:
            self.hint = hint
        if exit_code is not None:
            self.exit_code = exit_code


class HostedTimeout(HostedError):
    """``wait`` gave up before the run finished; the run goes on."""


def key_checksum(body: str) -> str:
    """The 6-character checksum a ``bfk_`` key ends with, for its 30-character body.

    Assumed, until the hub team fixes the encoding: zlib's CRC-32 of the body as UTF-8, written in base62
    (``0-9A-Za-z``) and left-padded with ``0`` to 6 characters (62**6 > 2**32, so it always fits). Should the
    service's encoding differ, ``BENCHFLOW_HUB_KEY_NO_CHECKSUM=1`` sends the key unchecked and the service judges it.
    """
    value = zlib.crc32(body.encode("utf-8"))
    digits = ""
    while value:
        value, rem = divmod(value, 62)
        digits = _BASE62[rem] + digits
    return digits.rjust(6, "0")


def key_label(key: str) -> str:
    """How a message names a key: its fixed prefix (``bfk_``, ``bh_``) and no
    character of its secret."""
    kind, sep, _ = key.partition("_")
    return f"{kind}_…" if sep and kind in ("bfk", "bh") else "(hidden)"


def check_api_key(key: str | None) -> str:
    """``key`` stripped, or :class:`HostedError` when it is missing or cannot be a key; no network call."""
    key = (key or "").strip()
    if not key:
        raise HostedError(
            f"no API key: {KEY_ENV} is not set and there is no key file",
            hint=f"set {KEY_ENV} to your BenchFlow Hub key (bfk_...), or put it in ~/.config/benchflow/hub-key (mode 600)",
            exit_code=2,
        )
    match = _BFK.match(key)
    if match:
        skip = os.environ.get(SKIP_CHECKSUM_ENV, "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if not skip and key_checksum(match.group(1)) != match.group(2):
            raise HostedError(
                f"the key {key_label(key)} is not valid: its checksum does not match, so it was mistyped or cut short",
                hint="copy the whole key again",
                exit_code=2,
            )
        return key
    if _BH.match(key):
        return key
    shape = (
        "a bfk_ key of the wrong length"
        if key.startswith("bfk_")
        else "not a BenchFlow Hub or hosted-service key"
    )
    raise HostedError(
        f"the key {key_label(key)} is {shape}",
        hint="use a key that starts bfk_ (36 letters and digits after it) or bh_",
        exit_code=2,
    )


# What opening a link with O_NOFOLLOW raises: ELOOP on Linux and macOS, EMLINK on FreeBSD.
_LINK_ERRNOS = frozenset({errno.ELOOP, errno.EMLINK})


def _read_owner_only(path: Path, limit: int | None = None) -> bytes:
    """The bytes of ``path`` (at most *limit*), a regular file of this user's that only its owner can
    read or write (mode 600 or stricter), or :class:`HostedError` names it, with the fix.

    The file is opened once, without following a link where the platform can refuse one
    (``O_NOFOLLOW``), and without waiting on a FIFO (``O_NONBLOCK``); the check (``fstat``) and the read
    use that descriptor, so a file or link swapped in after the check is never read."""
    flags = os.O_RDONLY
    for name in ("O_NOFOLLOW", "O_NONBLOCK", "O_CLOEXEC"):
        flags |= getattr(os, name, 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno in _LINK_ERRNOS:
            raise HostedError(
                f"the key file {path} is not a regular file",
                hint="point at the file itself",
                exit_code=2,
            ) from None
        raise HostedError(
            f"cannot read the key file {path}: {exc.strerror}",
            hint=f"write the key to that file with mode 600 (for the hub key, or set {KEY_ENV})",
            exit_code=2,
        ) from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise HostedError(
                f"the key file {path} is not a regular file",
                hint="point at the file itself",
                exit_code=2,
            )
        if info.st_mode & 0o077:
            raise HostedError(
                f"the key file {path} can be read by others (mode {stat.S_IMODE(info.st_mode):o})",
                hint=f"chmod 600 {path}",
                exit_code=2,
            )
        getuid = getattr(os, "getuid", None)
        if getuid is not None and info.st_uid != getuid():
            raise HostedError(
                f"the key file {path} belongs to another user",
                hint="use a key file of your own, with mode 600",
                exit_code=2,
            )
        chunks: list[bytes] = []
        size = 0
        while limit is None or size < limit:
            chunk = os.read(fd, 65536 if limit is None else min(65536, limit - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise HostedError(
            f"cannot read the key file {path}: {exc.strerror or exc}",
            hint="write the key alone, as text",
            exit_code=2,
        ) from None
    finally:
        os.close(fd)


def read_key_file(path: Path | str) -> str:
    """The key in ``path``, a regular file only its owner can read or write (mode 600 or stricter)."""
    path = Path(path).expanduser()
    data = _read_owner_only(path, _KEY_FILE_MAX)
    try:
        return data.decode("utf-8").strip()
    except UnicodeDecodeError:
        raise HostedError(
            f"cannot read the key file {path}",
            hint="write the key alone, as text",
            exit_code=2,
        ) from None


def find_key(key_file: Path | str | None = None) -> str | None:
    """The key: from *key_file* when the caller names one (``--key-file``),
    which wins over the environment; else ``BENCHFLOW_HUB_KEY`` /
    ``BENCHFLOW_API_KEY``; else ``BENCHFLOW_HUB_KEY_FILE`` or the default key
    file; None when none has one."""
    if key_file:
        return read_key_file(key_file)
    for name in (KEY_ENV, ALT_KEY_ENV):
        if value := os.environ.get(name, "").strip():
            return value
    if named := os.environ.get(KEY_FILE_ENV):
        return read_key_file(named)
    default = DEFAULT_KEY_FILE.expanduser()
    return read_key_file(default) if default.exists() else None


# Anything shaped like a key, this client's or another's: a bfk_ key (or the
# start of one) and a bh_<prefix>_<secret> key.
_KEY_SHAPED = re.compile(r"\b(?:bfk_[0-9A-Za-z]{4,}|bh_[0-9A-Za-z]+_[0-9A-Za-z_-]{4,})")


def read_secret(source: str) -> str:
    """A model key's value from where its owner keeps it, never from a command's arguments: ``file:PATH``,
    ``env:VAR``, or ``-`` (standard input)."""
    import sys

    source = (source or "").strip()
    if source == "-":
        value = sys.stdin.read()
    elif source.startswith("file:"):
        path = Path(source[5:]).expanduser()
        # A model key is as secret as the hub key: owner-only, as read_key_file asks.
        try:
            value = _read_owner_only(path).decode("utf-8")
        except UnicodeDecodeError:
            raise HostedError(
                f"can't read {path}: it is not UTF-8 text",
                hint="write the key alone, as text",
                exit_code=2,
            ) from None
    elif source.startswith("env:"):
        value = os.environ.get(source[4:], "")
        if not value:
            raise HostedError(f"${source[4:]} is not set", exit_code=2)
    else:
        raise HostedError(
            "a key's source is file:PATH, env:VAR or - (standard input)",
            hint="never the key itself: a key on a command line stays in shell history and the process list",
            exit_code=2,
        )
    value = value.strip()
    if not value:
        raise HostedError(f"{source} holds no key", exit_code=2)
    return value


def _masking(values: list[str], call: Callable[[], Any]) -> Any:
    """``call()``, with ``values`` (keys this request carries) masked out of any error it raises."""
    try:
        return call()
    except HostedError as exc:
        text = str(exc)
        for value in values:
            if value and len(value) >= 4:
                text = text.replace(value, "***")
        raise HostedError(
            text,
            status=exc.status,
            hint=getattr(exc, "hint", None),
            exit_code=exc.exit_code,
        ) from None


def redact(text: str, key: str | None) -> str:
    """``text`` with ``key`` masked, and its secret parts on their own (the
    part after ``bfk_``, and its body without the checksum, which the body
    gives; a ``bh_<prefix>_<secret>`` key's ``<prefix>_<secret>`` and
    ``<secret>``, split as the server splits it, ``split("_", 2)``, since a
    URL-safe secret holds ``_`` itself), and anything else shaped like a key."""
    if key:
        parts = key.split("_", 2)
        secrets = [key, *("_".join(parts[i:]) for i in range(1, len(parts)))]
        if match := _BFK.match(key):
            secrets.append(match.group(1))
        for secret in secrets:
            if len(secret) >= 8:
                text = text.replace(secret, "***")
    return _KEY_SHAPED.sub("***", text)


def normalize_endpoint(url: str) -> str:
    """A model endpoint for a hosted run: https only, reachable from the internet.

    The hosted service's workers call it from the service, so a CLIProxy on the user's own machine (localhost, a
    private or link-local address) can't be reached, and plain http would send its key in the clear. A user name
    or password in the URL would put a key on the command line; the key goes in ``--endpoint-key``.
    """
    import ipaddress

    url = url.strip().rstrip("/")
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    shown = f"{parts.scheme}://{host}" if host else url.split("@")[-1]
    bad = (
        parts.scheme != "https"
        or not host
        or parts.username is not None
        or parts.password is not None
    )
    if not bad:
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            bad = host == "localhost" or host.endswith(
                (".localhost", ".local", ".internal")
            )
        else:
            bad = not ip.is_global
    if bad:
        raise HostedError(
            f"the endpoint must be https only, reachable from the internet: {shown} is not",
            hint="the hosted workers call it from the service, so a CLIProxy on your own machine can't be reached; "
            "give its public https:// address, and its key with --endpoint-key",
            exit_code=2,
        )
    return url


def check_secret_kind(kind: str, key: str | None) -> str:
    """``kind``, or :class:`HostedError` before anything is sent when an account key (``bfk_``) can't store it."""
    kind = kind.strip().lower()
    if (key or "").startswith("bfk_") and kind not in KEY_KINDS:
        raise HostedError(
            f"an account key stores only these kinds: {', '.join(KEY_KINDS)} (not {kind!r})",
            hint="pick one of them with --kind; nothing was sent",
            exit_code=2,
        )
    return kind


def normalize_url(url: str) -> str:
    """``https://host[:port][/prefix]`` without a trailing slash; plain http only to localhost or 127.0.0.1."""
    url = url.strip().rstrip("/")
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise HostedError(
            f"not a hosted service address: {url!r}",
            hint="pass --hub https://host (or set BENCHFLOW_HOSTED_URL)",
            exit_code=2,
        )
    if parts.scheme == "http" and parts.hostname not in {"127.0.0.1", "localhost"}:
        raise HostedError(
            f"refusing plain http to a hosted service that is not on this machine: {url}",
            hint="use the service's https:// address",
            exit_code=2,
        )
    return url


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _detail(raw: bytes, key: str | None) -> str:
    """The service's message, masked (:func:`redact`) and only then cut to
    _DETAIL_MAX characters, so a cut never leaves part of a key behind."""
    try:
        payload = json.loads(raw or b"{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return redact(raw.decode("utf-8", "replace").strip(), key)[:_DETAIL_MAX]
    detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
    text = detail if isinstance(detail, str) else json.dumps(detail)
    return redact(text, key)[:_DETAIL_MAX]


@dataclass(frozen=True)
class HostedClient:
    """One hosted-eval service and the key it is called with."""

    url: str
    key: str = field(repr=False)

    @classmethod
    def from_env(
        cls,
        url: str | None = None,
        key: str | None = None,
        *,
        key_file: Path | str | None = None,
    ) -> HostedClient:
        """A client for ``url`` (else ``BENCHFLOW_HOSTED_URL``; :class:`HostedError` with neither) with
        ``key`` (else :func:`find_key`); both are checked offline, before any request."""
        base = url or os.environ.get(HOSTED_URL_ENV)
        if not base or not base.strip():
            raise HostedError(
                f"no hosted service named: pass --hub or set {HOSTED_URL_ENV}",
                hint=f"pass --hub https://<service> or set {HOSTED_URL_ENV}; "
                "nothing was sent",
                exit_code=2,
            )
        return cls(
            normalize_url(base),
            check_api_key(key if key is not None else find_key(key_file)),
        )

    def run_page(self, run_id: str) -> str:
        """The run's web page (a browser signs in to read it)."""
        return f"{self.url}/runs/{quote(run_id, safe='')}"

    # ---- requests ---------------------------------------------------------------------------------------
    def _open(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        accept: str = "application/json",
    ):
        headers = {
            "Accept": accept,
            "User-Agent": "benchflow-hosted-client",
            "Authorization": f"Bearer {self.key}",
        }
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        request = Request(self.url + path, data=data, method=method, headers=headers)
        what = f"{method} {path}"
        try:
            return _OPENER.open(request, timeout=_TIMEOUT_S)
        except _DowngradeRefused as exc:
            raise HostedError(
                redact(f"{self.url} {exc.msg} for {what}", self.key),
                hint="use the service's https:// address",
            ) from None
        except HTTPError as exc:
            raise self._http_error(exc, what) from None
        except (URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise HostedError(
                redact(
                    f"could not reach the hosted service at {self.url}: {reason}",
                    self.key,
                ),
                hint="check the address (--hub, BENCHFLOW_HOSTED_URL) and this machine's network",
            ) from None

    def _http_error(self, exc: HTTPError, what: str) -> HostedError:
        try:
            raw = exc.read()
        except OSError:
            raw = b""
        detail = _detail(raw, self.key) or redact(str(exc.reason or ""), self.key)
        code = exc.code
        after = exc.headers.get("Retry-After") if exc.headers else None
        later = f"after {after} s" if after and after.isdigit() else "later"
        if code in (401, 403):
            return HostedError(
                f"the hosted service refused the key {key_label(self.key)} ({code}) for {what}: {detail}",
                status=code,
                hint="the key may be revoked or expired, or lack the scope this needs (evals:run to start runs)",
            )
        if code == 429:
            return HostedError(
                f"the hosted service is limiting this key's requests (429): {detail}",
                status=code,
                hint=f"try again {later}; BenchFlow does not retry by itself",
            )
        if code == 503:
            return HostedError(
                f"the hosted service cannot answer right now (503): {detail}",
                status=code,
                hint=f"try again {later}; this says nothing about the key",
            )
        if code == 400:
            return HostedError(
                f"the hosted service refused {what} (400): {detail}",
                status=code,
                exit_code=2,
            )
        if code == 404:
            return HostedError(
                f"the hosted service has nothing at {what} (404): {detail}",
                status=code,
                hint="check the id: a key reads its own workspace's runs and the versions it can run",
            )
        return HostedError(
            f"the hosted service answered {code} for {what}: {detail}", status=code
        )

    def _json(self, method: str, path: str, *, body: Any = None) -> Any:
        with self._open(method, path, body=body) as response:
            raw = response.read()
        try:
            # What the service sends back is masked before anyone sees it.
            return json.loads(redact(raw.decode("utf-8"), self.key))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise HostedError(
                f"{self.url}{path} did not answer JSON",
                hint="check that --hub (or BENCHFLOW_HOSTED_URL) names the hosted service; this answer came from something else",
            ) from None

    def _run_path(self, run_id: str, suffix: str = "") -> str:
        if not run_id or not str(run_id).strip():
            raise HostedError(
                "no run id given",
                hint="pass the run id `bench hosted run` printed",
                exit_code=2,
            )
        return f"{RUNS_PATH}/{quote(str(run_id).strip(), safe='')}{suffix}"

    # ---- the API ----------------------------------------------------------------------------------------
    def run_body(
        self,
        env_id: str,
        *,
        agent: str | None = DEFAULT_AGENT,
        model: str | None = None,
        tasks: list[str] | None = None,
        max_tasks: int | None = None,
        n_trials: int = 1,
        concurrency: int = 1,
        name: str | None = None,
        max_sandbox_seconds: int | None = None,
        secrets: list[str] | None = None,
        max_cost_usd: float | None = None,
        secrets_once: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """The JSON body of a run (and of its estimate). ``secrets_once``: model keys or an endpoint for this run only
        (``{"name", "kind", "value"}``), which the service keeps until the run ends; ``max_cost_usd``: the run's
        budget, held by the service's key proxy."""
        if not env_id or not env_id.strip():
            raise HostedError(
                "no version named",
                hint="pass a hub version id, e.g. PKG-TERMINALBENCH-2-0",
                exit_code=2,
            )
        body: dict[str, Any] = {
            "env_id": env_id.strip(),
            "agent": agent or DEFAULT_AGENT,
            "model": model,
            "max_tasks": max_tasks,
            "tasks": list(tasks or []),
            "n_trials": n_trials,
            "concurrency": concurrency,
        }
        if name:
            body["name"] = name
        extra: dict[str, Any] = {}
        if max_sandbox_seconds is not None:
            extra["max_sandbox_seconds"] = max_sandbox_seconds
        if secrets:
            extra["secrets"] = list(secrets)
        if max_cost_usd is not None:
            extra["max_cost_usd"] = max_cost_usd
        if extra:
            body["extra"] = extra
        if secrets_once:
            body["secrets_once"] = [dict(s) for s in secrets_once]
        return body

    def submit(self, env_id: str, **options: Any) -> dict[str, Any]:
        """Queue a run of ``env_id``, a hub version id (``PKG-...``) or the hosted service's catalog id; the answer is
        ``{"run_id", "warnings", "runtime"}``. ``options`` are :meth:`run_body`'s."""
        once = [s.get("value", "") for s in options.get("secrets_once") or []]
        answer = _masking(
            once,
            lambda: self._json(
                "POST", RUNS_PATH, body=self.run_body(env_id, **options)
            ),
        )
        if not isinstance(answer, dict) or not answer.get("run_id"):
            raise HostedError(
                f"{self.url}{RUNS_PATH} answered without a run_id",
                hint="check that --hub (or BENCHFLOW_HOSTED_URL) names the hosted service; this answer came from something else",
            )
        answer.setdefault("warnings", [])
        answer.setdefault("runtime", {})
        return answer

    def estimate(self, env_id: str, **options: Any) -> dict[str, Any]:
        """What a run would likely do, without starting it (``warnings``, ``budget``, ...)."""
        options = {
            k: v for k, v in options.items() if k != "secrets_once"
        }  # an estimate needs no key
        answer = self._json(
            "POST", f"{RUNS_PATH}/estimate", body=self.run_body(env_id, **options)
        )
        return answer if isinstance(answer, dict) else {"estimate": answer}

    def list_secrets(self) -> list[dict[str, Any]]:
        """The workspace's stored keys: names, kinds and a 4-character hint, never values."""
        answer = self._json("GET", SECRETS_PATH)
        return list(answer.get("items") or []) if isinstance(answer, dict) else []

    def save_secret(self, name: str, kind: str, value: str) -> dict[str, Any]:
        """Store a provider key in the workspace (an account key may store the kinds in :data:`KEY_KINDS`)."""
        body = {"name": name, "kind": kind, "value": value}
        return _masking([value], lambda: self._json("POST", SECRETS_PATH, body=body))

    def save_endpoint(self, name: str, url: str, key: str) -> dict[str, Any]:
        """Store a model endpoint and its key together, as ``<name>_URL`` and ``<name>_KEY``: the service sends a
        stored endpoint only the keys stored with it."""
        body = {"name": name, "url": url, "key": key}
        return _masking(
            [key], lambda: self._json("POST", f"{SECRETS_PATH}/endpoint", body=body)
        )

    def delete_secret(self, name: str) -> bool:
        """Delete a stored key by name; False when there is none."""
        found = [s for s in self.list_secrets() if s.get("name") == name]
        if not found:
            return False
        answer = self._json(
            "DELETE", f"{SECRETS_PATH}/{quote(str(found[0]['id']), safe='')}"
        )
        return bool(isinstance(answer, dict) and answer.get("ok"))

    def status(self, run_id: str) -> dict[str, Any]:
        """The run record: ``status``, ``error``, ``n_traces`` of ``n_expected``, ``mean_reward``, times."""
        record = self._json("GET", self._run_path(run_id))
        if not isinstance(record, dict):
            raise HostedError(
                f"{self.url}{self._run_path(run_id)} did not answer a run record",
                hint="check that --hub (or BENCHFLOW_HOSTED_URL) names the hosted service; this answer came from something else",
            )
        return record

    def wait(
        self,
        run_id: str,
        *,
        timeout: float | None = None,
        poll: float = 2.0,
        max_poll: float = 30.0,
        on_status: Callable[[dict[str, Any]], None] | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> dict[str, Any]:
        """Poll the run until its status is final, waiting ``poll`` seconds and then half again as long each
        time, up to ``max_poll``. ``on_status`` gets each record whose status or trace count changed. Raises
        :class:`HostedTimeout` after ``timeout`` seconds (None: no limit); the run itself goes on. A refusal
        (a 429 or 503 included) stops the wait: it is never retried in a loop."""
        pause = sleep or _sleep
        start = clock()
        delay = max(poll, 0.1)
        last: tuple[str, Any] | None = None
        while True:
            record = self.status(run_id)
            current = str(record.get("status") or "")
            seen = (current, record.get("n_traces"))
            if seen != last and on_status is not None:
                on_status(record)
            last = seen
            if current not in ACTIVE_STATUSES:
                return record
            left = None if timeout is None else timeout - (clock() - start)
            if left is not None and left <= 0:
                raise HostedTimeout(
                    f"run {run_id} is still {current or 'unknown'} after {timeout:g} s",
                    hint=f"keep waiting with `bench hosted wait {run_id}`; the run goes on meanwhile",
                )
            pause(delay if left is None else min(delay, left))
            delay = min(delay * 1.5, max_poll)

    def events(self, run_id: str, *, after: int = 0) -> list[Any]:
        """The run's events (``{kind, message, ...}``) after event id ``after``."""
        suffix = "/events" + (f"?{urlencode({'after': after})}" if after else "")
        answer = self._json("GET", self._run_path(run_id, suffix))
        return answer if isinstance(answer, list) else [answer]

    def logs(self, run_id: str, *, tail: int | None = None) -> str:
        """The run's text log (its last ``tail`` characters, when given)."""
        path = self._run_path(run_id, "/log") + (
            f"?{urlencode({'tail': tail})}" if tail else ""
        )
        with self._open("GET", path, accept="text/plain") as response:
            return redact(response.read().decode("utf-8", "replace"), self.key)

    def metrics(self, run_id: str) -> dict[str, Any]:
        """The run's reward summary: ``n_traces``, each reward key's mean and spread (``keys``), its errors."""
        answer = self._json("GET", self._run_path(run_id, "/metrics"))
        return answer if isinstance(answer, dict) else {}

    def traces(self, run_id: str) -> list[dict[str, Any]]:
        """Every trial of the run (``task_name``, ``reward``, ``outcome``, ...), all pages of them."""
        items: list[dict[str, Any]] = []
        while True:
            query = urlencode(
                {"run_id": run_id, "limit": _TRACES_PAGE, "offset": len(items)}
            )
            page = self._json("GET", f"{TRACES_PATH}?{query}")
            batch = page.get("items") if isinstance(page, dict) else None
            if not isinstance(batch, list):
                raise HostedError(
                    f"{self.url}{TRACES_PATH} did not answer a list of traces",
                    hint="check that --hub (or BENCHFLOW_HOSTED_URL) names the hosted service; this answer came from something else",
                )
            items.extend(x for x in batch if isinstance(x, dict))
            total = page.get("total")
            if not batch or not isinstance(total, int) or len(items) >= total:
                return items

    def results(self, run_id: str) -> dict[str, Any]:
        """The run record, its metrics and its trials; :class:`HostedError` while the run is still going."""
        record = self.status(run_id)
        if str(record.get("status") or "") in ACTIVE_STATUSES:
            raise HostedError(
                f"run {run_id} is still {record.get('status')}",
                hint=f"wait for it with `bench hosted wait {run_id}`",
            )
        return {
            "run": record,
            "metrics": self.metrics(run_id),
            "traces": self.traces(run_id),
        }

    def trajectories(
        self, run_id: str, dest: Path | str, *, format: str = "atif"
    ) -> Path:
        """Save the run's trajectories, one JSON document per line, to ``dest``; returns the path written."""
        if format not in EXPORT_FORMATS:
            raise HostedError(
                f"format {format!r} is not one of {', '.join(EXPORT_FORMATS)}",
                hint="pass --format atif or --format messages",
                exit_code=2,
            )
        dest = Path(dest)
        if dest.is_dir():
            raise HostedError(
                f"{dest} is a folder",
                hint="name the file to write, e.g. -o run.jsonl",
                exit_code=2,
            )
        path = self._run_path(run_id, "/export") + "?" + urlencode({"format": format})
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".part")
        with (
            self._open("GET", path, accept="*/*") as response,
            partial.open("wb") as out,
        ):
            while chunk := response.read(_CHUNK):
                out.write(chunk)
        partial.replace(dest)
        return dest

    def cancel(self, run_id: str) -> dict[str, Any]:
        """Ask the service to stop the run."""
        answer = self._json("POST", self._run_path(run_id, "/cancel"))
        return answer if isinstance(answer, dict) else {"ok": answer}


def summary_line(record: dict[str, Any]) -> str:
    """One line on a run record: its status, trials done of expected, mean reward, and error."""
    parts = [f"status {record.get('status') or 'unknown'}"]
    done, expected = record.get("n_traces"), record.get("n_expected")
    if isinstance(done, int) and not isinstance(done, bool):
        parts.append(
            f"trials {done}/{expected}"
            if isinstance(expected, int)
            else f"trials {done}"
        )
    mean = record.get("mean_reward")
    if isinstance(mean, int | float) and not isinstance(mean, bool):
        parts.append(f"mean reward {mean:.3g}")
    if record.get("error"):
        parts.append(f"error: {record['error']}")
    return ", ".join(parts)


def _client(url: str | None, api_key: str | None) -> HostedClient:
    return HostedClient.from_env(url, api_key)


def submit(
    env_id: str, *, url: str | None = None, api_key: str | None = None, **options: Any
) -> dict[str, Any]:
    """Queue a hosted run (:meth:`HostedClient.submit`) with the environment's address and key."""
    return _client(url, api_key).submit(env_id, **options)


def status(
    run_id: str, *, url: str | None = None, api_key: str | None = None
) -> dict[str, Any]:
    """The run record (:meth:`HostedClient.status`)."""
    return _client(url, api_key).status(run_id)


def wait(
    run_id: str, *, url: str | None = None, api_key: str | None = None, **options: Any
) -> dict[str, Any]:
    """Wait for the run to finish (:meth:`HostedClient.wait`)."""
    return _client(url, api_key).wait(run_id, **options)


def logs(
    run_id: str,
    *,
    url: str | None = None,
    api_key: str | None = None,
    tail: int | None = None,
) -> str:
    """The run's text log (:meth:`HostedClient.logs`)."""
    return _client(url, api_key).logs(run_id, tail=tail)


def results(
    run_id: str, *, url: str | None = None, api_key: str | None = None
) -> dict[str, Any]:
    """The finished run's record, metrics and trials (:meth:`HostedClient.results`)."""
    return _client(url, api_key).results(run_id)


def download(
    run_id: str,
    dest: Path | str,
    *,
    url: str | None = None,
    api_key: str | None = None,
    format: str = "atif",
) -> Path:
    """Save the run's trajectories to ``dest`` (:meth:`HostedClient.trajectories`)."""
    return _client(url, api_key).trajectories(run_id, dest, format=format)


def cancel(
    run_id: str, *, url: str | None = None, api_key: str | None = None
) -> dict[str, Any]:
    """Stop the run (:meth:`HostedClient.cancel`)."""
    return _client(url, api_key).cancel(run_id)
