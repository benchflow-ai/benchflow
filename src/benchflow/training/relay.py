"""The policy relay: how rollouts reach a trainer's inference server.

A trainer serves its policy from its own OpenAI-compatible server (vLLM,
SGLang, a TRL ``vllm-serve``). Rollouts must reach it without exposing it:
the server often has unauthenticated control routes (TRL's weight
synchronisation), its key must not enter a task sandbox, and on providers
where BenchFlow's model gateway runs inside the sandbox (Daytona) a
loopback URL on the trainer's machine is unreachable.

:class:`PolicyRelay` is a small reverse proxy that runs in the trainer's
process, next to the server:

- It accepts only ``POST /v1/chat/completions``, ``POST /v1/completions`` and
  ``GET /v1/models``, each authenticated with a per-rollout *grant*: a bearer
  token that is valid until the rollout ends (:meth:`PolicyRelay.revoke`),
  never the server's own key. The relay adds the server's key itself.
- It listens on loopback, and on a Docker sandbox's route to the host or a
  public address only when asked (:meth:`PolicyRelay.url_for`). Error
  answers never name the upstream server.
- It streams answers through unbuffered and, from the raw bytes the server
  sent, records every call: status, the policy version at the time
  (``version``, e.g. a training step), and the token digest
  (:func:`benchflow.trajectories.token_capture.token_digest`) of the prompt
  ids, sampled ids and logprobs. BenchFlow compares those digests with what
  its gateway stored (:func:`benchflow.trajectories.token_segments.segment_rollout`),
  which attests that the server, the store and the trainer hold identical
  tokens, and tags each call with the version that served it.

``bf.Policy`` owns a relay; most code never touches this module directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

_GRANT_PREFIX = "bfr"
_FORWARDED_POST_PATHS = frozenset({"/v1/chat/completions", "/v1/completions"})
_FORWARDED_GET_PATHS = frozenset({"/v1/models"})
_DEFAULT_MAX_BODY_BYTES = 64 * 1024 * 1024
_DEFAULT_REQUEST_TIMEOUT_SEC = 3600.0
_DEFAULT_IDLE_TIMEOUT_SEC = 900.0


@dataclass
class RelayCall:
    """One request the relay forwarded for a grant."""

    seq: int
    path: str
    stream: bool
    status: str  # "ok" (the server answered 2xx) or "error"
    http_status: int | None
    version: Any
    version_end: Any
    started_at: float
    duration_sec: float
    digest: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RelayGrant:
    """A per-rollout credential for the relay; ``token`` is the secret."""

    id: str
    label: str
    token: str = field(repr=False)
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    active: bool = True
    calls: list[RelayCall] = field(default_factory=list)
    in_flight: int = 0

    def usable(self, now: float) -> bool:
        return self.active and (self.expires_at is None or now < self.expires_at)


class RelayError(RuntimeError):
    """The relay cannot serve a request the way it was asked to."""


def _loopback(host: str | None) -> bool:
    return (host or "").lower() in {"127.0.0.1", "localhost", "::1"}


def _error_body(message: str, kind: str) -> bytes:
    return json.dumps({"error": {"message": message, "type": kind}}).encode()


def _capture_provider(model: str | None) -> str | None:
    head = (model or "").split("/", 1)[0].lower()
    return head or None


class _SSE:
    """Incremental parser for ``data:`` lines of a server-sent event stream."""

    def __init__(self) -> None:
        self._buffer = b""

    def feed(self, data: bytes) -> list[dict[str, Any]]:
        self._buffer += data
        events: list[dict[str, Any]] = []
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:") :].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events


def _capture_record(
    body: dict[str, Any],
    provider: str | None,
    *,
    response: dict[str, Any] | None = None,
    stream_tokens: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A gateway-shaped record, so the relay builds captures the gateway's way."""
    record: dict[str, Any] = {
        "event": "success",
        "token_capture": {
            "enabled": True,
            "wire": "openai-chat",
            "provider": provider,
            "request": "chat",
            "logprobs": True,
            "token_ids": True,
        },
        "request": {"body": body},
        "response": response or {},
    }
    if stream_tokens is not None:
        record["stream_tokens"] = stream_tokens
        choices = stream_tokens.get("choices")
        if not record["response"].get("choices"):
            record["response"] = {
                "choices": [
                    {"index": int(key)}
                    for key in sorted(choices or {}, key=lambda k: int(k))
                    if str(key).isdigit()
                ]
            }
    return record


def _load_stream_tools() -> tuple[str, Callable[[dict[str, Any], Any], None]]:
    """The gateway's stream-chunk recorder (importing it loads LiteLLM: run off-loop)."""
    from benchflow.providers.litellm_token_capture_patch import (
        STREAM_TOKENS_KEY,
        record_stream_chunk,
    )

    return STREAM_TOKENS_KEY, record_stream_chunk


def _docker_address() -> str:
    from benchflow.providers.litellm_runtime import _docker_host_address

    return _docker_host_address()


def _summarise_capture(
    record: dict[str, Any],
) -> tuple[str | None, int | None, int | None]:
    from benchflow.trajectories.token_capture import build_token_capture

    capture = build_token_capture(record) or {}
    prompt = capture.get("prompt_token_ids")
    completions = capture.get("completions") or []
    first = completions[0] if completions and isinstance(completions[0], dict) else {}
    sampled = first.get("token_ids")
    return (
        capture.get("digest"),
        len(prompt) if isinstance(prompt, list) else None,
        len(sampled) if isinstance(sampled, list) else None,
    )


class PolicyRelay:
    """Authenticated reverse proxy in front of a trainer's policy server.

    ``upstream_url`` is the server's ``/v1`` base URL as the trainer's
    machine sees it (``http://127.0.0.1:8000/v1`` is fine: only the relay
    connects to it). ``version`` is called at the start and end of every
    request to tag it with the current policy version. ``public_bind``
    (``"0.0.0.0:8443"``) and ``public_url`` (how remote sandboxes reach that
    listener, e.g. ``https://relay.example.com``) are needed only for
    sandboxes on another machine; put TLS in front of it or pass
    ``ssl_context``.
    """

    def __init__(
        self,
        upstream_url: str,
        *,
        upstream_api_key: str | None = None,
        model: str | None = None,
        version: Callable[[], Any] | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        public_bind: str | None = None,
        public_url: str | None = None,
        ssl_context: Any | None = None,
        max_body_bytes: int = _DEFAULT_MAX_BODY_BYTES,
        request_timeout_sec: float = _DEFAULT_REQUEST_TIMEOUT_SEC,
        idle_timeout_sec: float = _DEFAULT_IDLE_TIMEOUT_SEC,
        max_concurrent_requests: int | None = None,
    ) -> None:
        parts = urlsplit(upstream_url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError(
                f"upstream_url must be an http(s) URL such as "
                f"http://127.0.0.1:8000/v1, got {upstream_url!r}"
            )
        if public_url is not None and public_bind is None:
            raise ValueError("public_url needs public_bind (host:port to listen on)")
        self._upstream = upstream_url.rstrip("/")
        self._upstream_key = upstream_api_key or None
        self._provider = _capture_provider(model)
        self._version = version or (lambda: None)
        self._host = host
        self._port = port
        self._public_bind = public_bind
        self._public_url = public_url.rstrip("/") if public_url else None
        self._ssl = ssl_context
        self._max_body = max_body_bytes
        self._request_timeout = request_timeout_sec
        self._idle_timeout = idle_timeout_sec
        self._limit = (
            asyncio.Semaphore(max_concurrent_requests)
            if max_concurrent_requests
            else None
        )
        self._grants: dict[str, RelayGrant] = {}
        self._seq = 0
        self._runner: Any = None
        self._session: Any = None
        self._urls: dict[str, str] = {}
        self._docker_url: str | None = None
        self._lock = asyncio.Lock()
        self._stream_key = ""
        self._record_chunk: Callable[[dict[str, Any], Any], None] = lambda d, c: None

    # --- lifecycle -----------------------------------------------------------

    @property
    def started(self) -> bool:
        return self._runner is not None

    async def start(self) -> None:
        """Start listening on loopback (and on ``public_bind`` when given)."""
        if self._runner is not None:
            return
        from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

        # The stream recorder lives with the gateway's LiteLLM patch; loading
        # it imports LiteLLM, which takes seconds: never on a request.
        self._stream_key, self._record_chunk = await asyncio.to_thread(
            _load_stream_tools
        )
        app = web.Application(client_max_size=self._max_body)
        app.router.add_route("*", "/{tail:.*}", self._handle)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        try:
            site = web.TCPSite(runner, self._host, self._port)
            await site.start()
            self._urls["host"] = (
                f"http://{self._format_host(self._host)}:{self._bound_port(site)}/v1"
            )
            if self._public_bind is not None:
                host, _, port = self._public_bind.rpartition(":")
                public = web.TCPSite(
                    runner, host or "0.0.0.0", int(port), ssl_context=self._ssl
                )
                await public.start()
                scheme = "https" if self._ssl is not None else "http"
                self._urls["public"] = (
                    f"{self._public_url}/v1"
                    if self._public_url
                    else f"{scheme}://{host}:{self._bound_port(public)}/v1"
                )
        except BaseException:
            await runner.cleanup()
            raise
        self._runner = runner
        self._session = ClientSession(
            timeout=ClientTimeout(
                total=self._request_timeout, sock_read=self._idle_timeout
            ),
            connector=TCPConnector(limit=0),
        )
        logger.info("Policy relay listening on %s", self._urls["host"])

    async def close(self) -> None:
        """Stop listening; revoke every grant."""
        for grant in self._grants.values():
            grant.active = False
        if self._session is not None:
            await self._session.close()
            self._session = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self._urls.clear()
        self._docker_url = None

    async def __aenter__(self) -> PolicyRelay:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @staticmethod
    def _format_host(host: str) -> str:
        return f"[{host}]" if ":" in host and not host.startswith("[") else host

    @staticmethod
    def _bound_port(site: Any) -> int:
        server = getattr(site, "_server", None)
        sockets = getattr(server, "sockets", None) or []
        if sockets:
            return int(sockets[0].getsockname()[1])
        return int(getattr(site, "_port", 0))

    # --- addresses -----------------------------------------------------------

    @property
    def host_url(self) -> str:
        """The relay's ``/v1`` URL on this machine (for a gateway on the host)."""
        if "host" not in self._urls:
            raise RelayError("the policy relay is not started")
        return self._urls["host"]

    async def url_for(self, environment: str) -> str | None:
        """The ``/v1`` URL a gateway *inside* a sandbox of ``environment`` uses.

        Docker: the host as containers see it (the bridge gateway on Linux,
        ``host.docker.internal`` elsewhere), on a listener the relay opens
        there. Other providers: ``public_url``; None when none was given.
        """
        if not self.started:
            raise RelayError("the policy relay is not started")
        if environment == "docker":
            async with self._lock:
                if self._docker_url is None:
                    self._docker_url = await self._open_docker_listener()
            return self._docker_url
        return self._urls.get("public")

    async def _open_docker_listener(self) -> str:
        import socket

        from aiohttp import web

        address = await asyncio.to_thread(_docker_address)
        port = int(urlsplit(self._urls["host"]).port or 0)
        try:
            socket.inet_aton(address)
        except OSError:
            # host.docker.internal (Docker Desktop, OrbStack, Colima) reaches
            # the host's loopback, so the loopback listener already serves it.
            return f"http://{address}:{port}/v1"
        site = web.TCPSite(self._runner, address, 0)
        await site.start()
        return f"http://{address}:{self._bound_port(site)}/v1"

    # --- grants --------------------------------------------------------------

    def grant(self, label: str, *, ttl_sec: float | None = None) -> RelayGrant:
        """A new credential for one rollout; pass ``grant.token`` as its API key."""
        gid = secrets.token_hex(6)
        token = f"{_GRANT_PREFIX}-{gid}-{secrets.token_urlsafe(24)}"
        grant = RelayGrant(
            id=gid,
            label=label,
            token=token,
            expires_at=None if ttl_sec is None else time.time() + ttl_sec,
        )
        self._grants[gid] = grant
        return grant

    async def revoke(
        self, grant: RelayGrant, *, drain_sec: float = 10.0
    ) -> list[RelayCall]:
        """End a grant; returns the calls it made, oldest first.

        New calls are refused at once; calls already in flight get up to
        ``drain_sec`` to finish so they are in the list, not lost.
        """
        grant.active = False
        self._grants.pop(grant.id, None)
        deadline = time.monotonic() + drain_sec
        while grant.in_flight > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        return list(grant.calls)

    def _authorize(self, header: str) -> RelayGrant | None:
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer":
            return None
        parts = token.strip().split("-", 2)
        if len(parts) != 3 or parts[0] != _GRANT_PREFIX:
            return None
        grant = self._grants.get(parts[1])
        if grant is None or not hmac.compare_digest(grant.token, token.strip()):
            return None
        return grant if grant.usable(time.time()) else None

    # --- forwarding ----------------------------------------------------------

    def _upstream_url_for(self, path: str) -> str:
        return self._upstream + path[len("/v1") :]

    def _upstream_headers(self, request: Any) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        accept = request.headers.get("Accept")
        if accept:
            headers["Accept"] = accept
        if self._upstream_key:
            headers["Authorization"] = f"Bearer {self._upstream_key}"
        return headers

    async def _handle(self, request: Any) -> Any:
        from aiohttp import web

        path = request.path
        allowed = (request.method == "POST" and path in _FORWARDED_POST_PATHS) or (
            request.method == "GET" and path in _FORWARDED_GET_PATHS
        )
        if not allowed:
            return web.Response(
                status=404,
                body=_error_body("not served by the policy relay", "not_found"),
                content_type="application/json",
            )
        grant = self._authorize(request.headers.get("Authorization", ""))
        if grant is None:
            return web.Response(
                status=401,
                body=_error_body(
                    "invalid, expired or revoked policy relay credential",
                    "invalid_api_key",
                ),
                content_type="application/json",
            )
        if request.method == "GET":
            return await self._forward_get(request, path)
        try:
            raw = await request.read()
        except web.HTTPRequestEntityTooLarge:
            return web.Response(
                status=413,
                body=_error_body("request body too large", "request_too_large"),
                content_type="application/json",
            )
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return web.Response(
                status=400,
                body=_error_body("the body must be a JSON object", "invalid_request"),
                content_type="application/json",
            )
        grant.in_flight += 1
        try:
            if self._limit is None:
                return await self._forward_post(request, path, raw, body, grant)
            async with self._limit:
                return await self._forward_post(request, path, raw, body, grant)
        finally:
            grant.in_flight -= 1

    async def _forward_get(self, request: Any, path: str) -> Any:
        from aiohttp import ClientError, web

        try:
            async with self._session.get(
                self._upstream_url_for(path), headers=self._upstream_headers(request)
            ) as upstream:
                data = await upstream.read()
                return web.Response(
                    status=upstream.status,
                    body=data,
                    content_type=upstream.content_type or "application/json",
                )
        except (ClientError, TimeoutError) as exc:
            return self._unreachable(exc)

    def _unreachable(self, exc: BaseException) -> Any:
        from aiohttp import web

        # Never name the upstream: a sandbox must not learn where it is.
        logger.warning("Policy relay: upstream request failed: %s", type(exc).__name__)
        return web.Response(
            status=502,
            body=_error_body(
                f"the policy relay could not reach the policy server ({type(exc).__name__})",
                "relay_upstream_unreachable",
            ),
            content_type="application/json",
        )

    def _new_call(self) -> tuple[int, float, Any]:
        self._seq += 1
        return self._seq, time.time(), self._version()

    async def _forward_post(
        self,
        request: Any,
        path: str,
        raw: bytes,
        body: dict[str, Any],
        grant: RelayGrant,
    ) -> Any:
        from aiohttp import ClientError, web

        stream_key, record_stream_chunk = self._stream_key, self._record_chunk
        stream = body.get("stream") is True
        seq, started, version = self._new_call()
        clock = time.monotonic()
        recorded = False

        def finish(
            status: str,
            http_status: int | None,
            *,
            record: dict[str, Any] | None = None,
            error: str | None = None,
        ) -> None:
            nonlocal recorded
            if recorded:  # one record per call, whichever path ends it
                return
            recorded = True
            digest = prompt_tokens = completion_tokens = None
            if record is not None and status == "ok":
                try:
                    digest, prompt_tokens, completion_tokens = _summarise_capture(
                        record
                    )
                except Exception:  # a capture problem must never fail the call
                    logger.debug("Policy relay: capture failed", exc_info=True)
            grant.calls.append(
                RelayCall(
                    seq=seq,
                    path=path,
                    stream=stream,
                    status=status,
                    http_status=http_status,
                    version=version,
                    version_end=self._version(),
                    started_at=started,
                    duration_sec=round(time.monotonic() - clock, 3),
                    digest=digest,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    error=error,
                )
            )

        try:
            upstream_cm = self._session.post(
                self._upstream_url_for(path),
                data=raw,
                headers=self._upstream_headers(request),
            )
            upstream = await upstream_cm.__aenter__()
        except (ClientError, TimeoutError) as exc:
            finish("error", None, error=type(exc).__name__)
            return self._unreachable(exc)
        try:
            is_sse = (upstream.content_type or "").startswith("text/event-stream")
            if not is_sse:
                data = await upstream.read()
                ok = 200 <= upstream.status < 300
                record = None
                if ok:
                    try:
                        parsed = json.loads(data or b"{}")
                    except ValueError:
                        parsed = None
                    if isinstance(parsed, dict):
                        record = _capture_record(body, self._provider, response=parsed)
                finish(
                    "ok" if ok else "error",
                    upstream.status,
                    record=record,
                    error=None if ok else f"HTTP {upstream.status}",
                )
                return web.Response(
                    status=upstream.status,
                    body=data,
                    content_type=upstream.content_type or "application/json",
                )
            out = web.StreamResponse(
                status=upstream.status,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                },
            )
            await out.prepare(request)
            parser = _SSE()
            details: dict[str, Any] = {}
            failed: str | None = None
            try:
                async for chunk in upstream.content.iter_any():
                    await out.write(chunk)
                    for event in parser.feed(chunk):
                        record_stream_chunk(details, event)
            except (ClientError, TimeoutError, ConnectionResetError) as exc:
                failed = type(exc).__name__
            ok = 200 <= upstream.status < 300 and failed is None
            finish(
                "ok" if ok else "error",
                upstream.status,
                record=_capture_record(
                    body,
                    self._provider,
                    stream_tokens=details.get(stream_key) or {},
                )
                if ok
                else None,
                error=failed or (None if ok else f"HTTP {upstream.status}"),
            )
            with contextlib.suppress(ConnectionResetError, RuntimeError):
                await out.write_eof()
            return out
        except (ClientError, TimeoutError) as exc:
            # Reading a non-streamed answer failed: record it, answer 502.
            finish("error", upstream.status, error=type(exc).__name__)
            return self._unreachable(exc)
        except BaseException as exc:
            # Cancelled (the client went away) or a bug: still one record.
            finish("error", upstream.status, error=type(exc).__name__)
            raise
        finally:
            await upstream_cm.__aexit__(None, None, None)

    # --- inspection ----------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Open grants and their call counts (no secrets)."""
        return {
            "grants": [
                {
                    "id": g.id,
                    "label": g.label,
                    "calls": len(g.calls),
                    "in_flight": g.in_flight,
                }
                for g in self._grants.values()
            ],
            "urls": {k: v for k, v in self._urls.items()},
        }


def is_loopback_url(url: str) -> bool:
    """Whether ``url``'s host is this machine's loopback."""
    return _loopback(urlsplit(url).hostname)


def with_host(url: str, host: str) -> str:
    """``url`` with its host replaced (port, path and scheme kept)."""
    parts = urlsplit(url)
    netloc = host if parts.port is None else f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


__all__ = [
    "PolicyRelay",
    "RelayCall",
    "RelayError",
    "RelayGrant",
    "is_loopback_url",
    "with_host",
]
