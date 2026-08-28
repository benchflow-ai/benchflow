"""Record-replay LLM proxy for ``benchflow continue``.

OpenHands talks to this proxy via ``LLM_BASE_URL`` (OpenAI chat-completions
wire protocol). For the first *N* requests it returns the *recorded* responses
from the original run's ``llm_trajectory.jsonl`` **in order**, so the agent
re-executes its own past decisions for real and rebuilds its exact workspace and
internal state. Once the recorded responses are exhausted (the timeout
cut-point), it forwards to the **live** upstream and the agent continues with no
injected prompt.

Two layers, kept separate so the routing logic is testable without sockets:

- :class:`ReplayRouter` — pure: maps the *i*-th incoming request to the *i*-th
  recorded response, then to the live forwarder. Captures the live-leg exchanges
  so the caller can stitch a continuous ``llm_trajectory.jsonl``.
- :class:`ReplayProxy` — a tiny stdlib ``http.server`` wrapper (no extra deps)
  that parses requests, calls the router, and emits the result as JSON or as
  reconstructed SSE depending on the request's ``stream`` flag.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

from benchflow._utils.config_override import overlay_hash
from benchflow.trajectories.types import LLMExchange, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

# A callable that takes the agent's request body and returns a final
# (non-streamed) ChatCompletion dict from the real model. The orchestrator wires
# this to a LiteLLM gateway; ``None`` means "no live leg" (replay-only).
LiveForwarder = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class ReplayResult:
    """Outcome of routing one request: a final ChatCompletion to emit."""

    source: str  # "replay" | "live" | "error"
    status: int
    body: dict[str, Any]


def _n_messages(body: dict[str, Any]) -> int:
    msgs = body.get("messages")
    return len(msgs) if isinstance(msgs, list) else 0


class ReplayCutPointError(ValueError):
    """Raised when a requested replay cut-point lies outside the recording."""


def validate_max_exchanges(max_exchanges: int | None, n_recorded: int) -> int:
    """Resolve the replay prefix length; fail closed on an out-of-range cut.

    ``None`` means the full recorded prefix (today's behavior). An explicit cut
    must satisfy ``0 < K <= n_recorded`` — validated here, at configuration
    time, never mid-flight.
    """
    if max_exchanges is None:
        return n_recorded
    if not 0 < max_exchanges <= n_recorded:
        raise ReplayCutPointError(
            f"max_exchanges must be in 1..{n_recorded} (the recorded exchange "
            f"count), got {max_exchanges}"
        )
    return max_exchanges


def request_body_digest(body: dict[str, Any]) -> str:
    """Content address of a request body — canonical JSON, ``sha256:`` form.

    A thin wrapper over the config-overlay hash
    (:func:`benchflow._utils.config_override.overlay_hash`, #790) so the two
    canonicalizations can never drift: sorted keys, compact separators,
    ``sha256:``-prefixed digest.
    """
    return overlay_hash(body)


#: The basis every request digest in cut-point accounting is computed on —
#: recorded next to the digests so a reader knows what "equal" means.
REQUEST_DIGEST_BASIS = (
    "sha256 over canonical JSON of the comparable projection {messages, tools}"
)


def comparable_request_projection(body: dict[str, Any]) -> dict[str, Any]:
    """The content-bearing fields a recorded and a live request share.

    The recorded request is a *normalized projection* written by the LiteLLM
    logger (model/messages/tools/stream…), not the verbatim HTTP body — and
    the live replay request carries the proxy's placeholder model
    (``openai/replay``) plus transport fields (``stream``, ``temperature``,
    …) the recording may lack. Digesting either side whole would therefore
    flag every exchange as divergent. ``messages`` and ``tools`` are the
    fields that exist on both sides and actually carry the conversation, so
    they are the honest comparison basis (:data:`REQUEST_DIGEST_BASIS`).
    """
    return {
        key: body[key] for key in ("messages", "tools") if body.get(key) is not None
    }


def comparable_request_digest(body: dict[str, Any]) -> str:
    """Digest of :func:`comparable_request_projection` — the comparison unit."""
    return request_body_digest(comparable_request_projection(body))


class ReplayRouter:
    """Serve recorded responses in order, then live — the core replay logic.

    ``max_exchanges`` cuts the replay short (rollout-branching RFC §3.5): at
    most the first K recorded exchanges are served, then the router switches to
    live passthrough exactly as if the recording had ended there. ``None``
    replays the full recorded prefix.

    Thread-safe: a single agent is typically serial, but ``ThreadingHTTPServer``
    may overlap requests, so the cursor is advanced under a lock.
    """

    def __init__(
        self,
        recorded: list[LLMExchange],
        *,
        live_forwarder: LiveForwarder | None = None,
        strict_divergence: bool = False,
        max_exchanges: int | None = None,
        workspace_digest_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._recorded = recorded
        self._n_replay = validate_max_exchanges(max_exchanges, len(recorded))
        self._live_forwarder = live_forwarder
        self._strict = strict_divergence
        self._lock = threading.Lock()
        self._cursor = 0
        self.divergences = 0
        #: One event per detected divergence: the exchange index and both
        #: sides' comparable digests, so the artifact can say *where* the
        #: replay left the rails, not only that it did.
        self.divergence_events: list[dict[str, Any]] = []
        # Comparable digests of the last replayed exchange — the actually
        # served request and the recorded one it answered for.
        self._last_served_digest: str | None = None
        self._last_recorded_digest: str | None = None
        # RFC §3.5 workspace accounting: called exactly once, at the first
        # request past the cut, while the continuation's sandbox still holds
        # the workspace the replay rebuilt. ``None`` = no sandbox reachable
        # in this mode; the provenance then records null with a reason.
        self.workspace_digest_fn = workspace_digest_fn
        self.workspace_digest: dict[str, Any] | None = None
        self._workspace_digest_error: str | None = None
        self._cut_crossed = False
        # Live-leg exchanges, in order, for stitching onto the recorded prefix.
        self.live_exchanges: list[LLMExchange] = []

    @property
    def exhausted(self) -> bool:
        return self._cursor >= self._n_replay

    @property
    def n_replayed_exchanges(self) -> int:
        """Recorded exchanges actually served so far (== the cut once live)."""
        return min(self._cursor, self._n_replay)

    @property
    def served_request_digest(self) -> str | None:
        """Comparable digest of the ACTUAL incoming request last replayed.

        Cut-point accounting (RFC §3.5): this is the request the agent really
        sent at the cut, digested on :data:`REQUEST_DIGEST_BASIS` — compare it
        to :attr:`recorded_request_digest` to see whether the replayed
        conversation was still the recorded one. ``None`` while nothing has
        been replayed.
        """
        return self._last_served_digest

    @property
    def recorded_request_digest(self) -> str | None:
        """Comparable digest of the recorded request at the same position."""
        return self._last_recorded_digest

    @property
    def cut_crossed(self) -> bool:
        """Whether any request arrived past the replay prefix."""
        return self._cut_crossed

    @property
    def workspace_digest_reason(self) -> str | None:
        """Why :attr:`workspace_digest` is ``None`` — never fabricated."""
        if self.workspace_digest is not None:
            return None
        if self._workspace_digest_error is not None:
            return self._workspace_digest_error
        if not self._cut_crossed:
            return (
                "the run never crossed the cut point into the live leg, so "
                "there was no moment to digest the workspace at"
            )
        return (
            "no live sandbox was reachable at the cut point in this mode; "
            "no workspace digest hook was configured"
        )

    def _check_divergence(
        self,
        incoming: dict[str, Any],
        recorded_req: dict[str, Any],
        *,
        index: int,
        served_digest: str,
        recorded_digest: str,
    ) -> None:
        """Validate that replay is still on the original rails — by content.

        Compares the comparable projections' canonical-JSON digests, so a
        same-message-count change to prompt content, tool definitions or tool
        arguments is detected — the case the old message-count heuristic was
        blind to. A recorded request with no ``messages`` (a differently
        shaped recording) offers nothing to compare against and is skipped,
        exactly as the count heuristic skipped it.

        A divergence *annotates* rather than aborts by default: replay
        fidelity is best-effort by design (RFC §3.5 — "recorded, not
        hidden"), and legitimately shifting content (timestamps in prompts,
        nondeterministic tool output) would otherwise kill continuations the
        user asked for. Every event is recorded in
        :attr:`divergence_events`, so the artifact is truthful either way;
        ``strict_divergence`` remains the opt-in abort.
        """
        if not comparable_request_projection(recorded_req).get("messages"):
            return
        if served_digest == recorded_digest:
            return
        event = {
            "exchange_index": index,
            "served_request_digest": served_digest,
            "recorded_request_digest": recorded_digest,
            "n_messages_served": _n_messages(incoming),
            "n_messages_recorded": _n_messages(recorded_req),
        }
        self.divergences += 1
        self.divergence_events.append(event)
        msg = (
            f"replay divergence at exchange {index}: the agent's request "
            f"({event['n_messages_served']} messages, {served_digest}) does "
            f"not match the recorded one ({event['n_messages_recorded']} "
            f"messages, {recorded_digest})"
        )
        if self._strict:
            raise ReplayDivergenceError(msg)
        logger.warning(msg)

    def _capture_workspace_digest(self) -> None:
        """Run the cut-point workspace hook once; record failure, never raise."""
        if self.workspace_digest_fn is None:
            return
        try:
            self.workspace_digest = self.workspace_digest_fn()
        except Exception as exc:
            self._workspace_digest_error = (
                f"workspace digest failed at the cut point: {exc}"
            )
            logger.warning(self._workspace_digest_error, exc_info=True)

    def next_response(self, request_body: dict[str, Any]) -> ReplayResult:
        """Route one request to its recorded response, or to the live model."""
        with self._lock:
            if self._cursor < self._n_replay:
                exchange = self._recorded[self._cursor]
                index = self._cursor
                served_digest = comparable_request_digest(request_body)
                recorded_digest = comparable_request_digest(exchange.request.body)
                self._last_served_digest = served_digest
                self._last_recorded_digest = recorded_digest
                self._check_divergence(
                    request_body,
                    exchange.request.body,
                    index=index,
                    served_digest=served_digest,
                    recorded_digest=recorded_digest,
                )
                self._cursor += 1
                return ReplayResult(
                    source="replay",
                    status=exchange.response.status_code or 200,
                    body=dict(exchange.response.body),
                )
            # Past the cut-point: live continuation.
            self._cursor += 1
            crossing = not self._cut_crossed
            self._cut_crossed = True
            forwarder = self._live_forwarder

        if crossing:
            # Outside the lock: the hook may exec into the sandbox. The
            # workspace it digests is the one the replay just rebuilt —
            # this request is the first the recording no longer covers.
            self._capture_workspace_digest()

        if forwarder is None:
            logger.error(
                "recorded responses exhausted at turn %d and no live forwarder "
                "is configured — returning an error to the agent.",
                self._cursor - 1,
            )
            return ReplayResult(
                source="error",
                status=503,
                body={
                    "error": {
                        "message": (
                            "benchflow continue: recorded trajectory exhausted "
                            "and no live model configured."
                        ),
                        "type": "replay_exhausted",
                    }
                },
            )

        body = forwarder(request_body)
        # Capture the live exchange so the caller can stitch a continuous
        # llm_trajectory.jsonl (recorded prefix + live suffix).
        self.live_exchanges.append(
            LLMExchange(
                request=LLMRequest(body=request_body),
                response=LLMResponse(status_code=200, body=body),
            )
        )
        return ReplayResult(source="live", status=200, body=body)


class ReplayDivergenceError(RuntimeError):
    """Raised (only in ``strict_divergence`` mode) when replay leaves the rails."""


# ── SSE reconstruction ────────────────────────────────────────────────────
#
# The recorded/live body is a *final* ChatCompletion. When the agent asked for
# ``stream: true`` we must re-emit it as ``chat.completion.chunk`` SSE events so
# the agent's LiteLLM client reassembles the identical final message. Emitting
# the whole content / each tool call in a single delta is accepted by
# OpenAI-compatible stream parsers (LiteLLM included).


def completion_to_sse(body: dict[str, Any]) -> list[str]:
    """Render a final ChatCompletion as a list of SSE ``data:`` payload strings.

    The returned strings are the JSON payloads (without the ``data: `` prefix or
    trailing blank line); the HTTP layer frames them. The final ``[DONE]``
    sentinel is appended by the caller.
    """
    base_id = body.get("id") or f"chatcmpl-replay-{int(time.time() * 1000)}"
    created = body.get("created") or int(time.time())
    model = body.get("model") or "replay"
    choices = body.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason") or "stop"

    def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
        return json.dumps(
            {
                "id": base_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )

    payloads: list[str] = [chunk({"role": "assistant"})]

    content = message.get("content")
    if content:
        delta: dict[str, Any] = {"content": content}
        # Preserve reasoning when the provider surfaces it on the message.
        for key in ("reasoning_content", "thinking"):
            if message.get(key):
                delta[key] = message[key]
        payloads.append(chunk(delta))

    tool_calls = message.get("tool_calls") or []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        payloads.append(
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": tc.get("index", i),
                            "id": tc.get("id"),
                            "type": tc.get("type", "function"),
                            "function": {
                                "name": fn.get("name"),
                                "arguments": fn.get("arguments", ""),
                            },
                        }
                    ]
                }
            )
        )

    final = {
        "id": base_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    usage = body.get("usage")
    if isinstance(usage, dict):
        final["usage"] = usage
    payloads.append(json.dumps(final))
    return payloads


# ── HTTP layer ────────────────────────────────────────────────────────────


@dataclass
class _ProxyState:
    router: ReplayRouter
    requests_seen: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def bump(self) -> int:
        with self._lock:
            self.requests_seen += 1
            return self.requests_seen


class _ReplayHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that carries the proxy state for its handlers."""

    def __init__(
        self, addr: tuple[str, int], handler: type, state: _ProxyState
    ) -> None:
        super().__init__(addr, handler)
        self.state = state


class _ReplayHandler(BaseHTTPRequestHandler):
    # Silence the default per-request stderr logging; route through logging.
    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("replay-proxy %s - " + format, self.address_string(), *args)

    @property
    def _state(self) -> _ProxyState:
        return cast("_ReplayHTTPServer", self.server).state

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, payloads: list[str]) -> None:
        # A finite SSE reply: emit all chunks + [DONE], then close the
        # connection so length-agnostic clients see a clean end-of-stream
        # (no Content-Length is sent for event-streams).
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for payload in payloads:
            self.wfile.write(f"data: {payload}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/health/liveliness", "/v1/health"):
            self._send_json(200, {"status": "ok"})
            return
        if path in ("/v1/models", "/models"):
            self._send_json(
                200,
                {"object": "list", "data": [{"id": "replay", "object": "model"}]},
            )
            return
        self._send_json(404, {"error": {"message": f"not found: {path}"}})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, {"error": {"message": f"not found: {path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            request_body = json.loads(raw or b"{}")
            if not isinstance(request_body, dict):
                raise ValueError("request body must be a JSON object")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": {"message": f"bad request: {exc}"}})
            return

        self._state.bump()
        want_stream = bool(request_body.get("stream"))
        try:
            result = self._state.router.next_response(request_body)
        except ReplayDivergenceError as exc:
            self._send_json(409, {"error": {"message": str(exc), "type": "divergence"}})
            return
        except Exception as exc:
            logger.exception("replay router failed")
            self._send_json(500, {"error": {"message": str(exc)}})
            return

        if result.status >= 400 or not result.body.get("choices"):
            # Errors (and recorded failure exchanges) are returned verbatim as
            # JSON regardless of stream, so the agent's client surfaces them.
            self._send_json(result.status, result.body)
            return

        if want_stream:
            self._send_sse(completion_to_sse(result.body))
        else:
            self._send_json(result.status, result.body)


class ReplayProxy:
    """A running record-replay proxy bound to ``host:port``.

    ``base_url`` is the OpenAI-style root (``http://host:port/v1``) to hand the
    agent as ``LLM_BASE_URL``. Use as a context manager or call
    :meth:`start`/:meth:`stop`.
    """

    def __init__(
        self,
        router: ReplayRouter,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        advertise_host: str | None = None,
    ) -> None:
        self._router = router
        self._advertise_host = advertise_host or host
        self._server = _ReplayHTTPServer(
            (host, port), _ReplayHandler, _ProxyState(router=router)
        )
        self._thread: threading.Thread | None = None

    @property
    def router(self) -> ReplayRouter:
        return self._router

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        """The ``/v1`` root the agent should use as ``LLM_BASE_URL``."""
        return f"http://{self._advertise_host}:{self.port}/v1"

    def start(self) -> ReplayProxy:
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="benchflow-replay-proxy",
            daemon=True,
        )
        self._thread.start()
        logger.info("replay proxy listening on %s", self.base_url)
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> ReplayProxy:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
