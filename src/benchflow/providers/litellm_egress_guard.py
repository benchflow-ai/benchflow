"""Model gateway guard for no-web and denylist runs: LiteLLM serves the agent model calls only.

The agent reaches the in-sandbox LiteLLM proxy with the key in its environment,
and LiteLLM serves far more than model calls: provider pass-through routes, file
and batch APIs, WebSocket sessions, and request fields it merges into the
provider call. Any of those can have the provider fetch the web outside every
sandbox rule. ``install()`` wraps LiteLLM's app so that only the model-call
routes the harnesses use are served, every request body passes the egress
policy, and each refusal is logged.

benchflow ships this module and the egress policy modules into the LiteLLM
runtime directory at their package paths, and the runtime's ``sitecustomize`` calls
``install()`` first thing for a guarded gateway.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.abc
import importlib.util
import json
import os
import re
import sys
from collections.abc import Awaitable, Callable, MutableMapping, Sequence
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any

from benchflow.sandbox._egress_policy import Log
from benchflow.sandbox._hosted_fetch import (
    BODY_INSPECT_LIMIT,
    content_codings,
    model_request_rule,
)

_Message = MutableMapping[str, Any]
_Receive = Callable[[], Awaitable[_Message]]
_Send = Callable[[_Message], Awaitable[None]]

_PROXY_SERVER = "litellm.proxy.proxy_server"
# The routes harness traffic uses, by method: chat, completions, Responses,
# Messages and embeddings, Gemini GenerateContent through LiteLLM's
# pass-through, model listings, and the readiness probes. All else is refused.
_MODEL_ROUTES = {
    "GET": re.compile(
        r"/health(?:/[a-z]+)?|/(?:v1/)?models(?:/[^/]+)?"
        r"|/gemini/v1(?:beta)?/models(?:/[^/:]+)?"
    ),
    "POST": re.compile(
        r"/(?:v1/)?(?:chat/completions|completions|embeddings|responses"
        r"|responses/compact|messages|messages/count_tokens)"
        r"|/gemini/v1(?:beta)?/models/[^/:]+:(?:generateContent"
        r"|streamGenerateContent|countTokens|embedContent|batchEmbedContents)"
    ),
}


def install(*, block_log: str | None) -> None:
    """Wrap LiteLLM's proxy app in the guard as soon as its module has been executed.

    Must run before anything imports LiteLLM. It also stops LiteLLM from
    downloading image URLs itself: its own traffic bypasses the egress proxy.
    """
    os.environ["MAX_IMAGE_URL_DOWNLOAD_SIZE_MB"] = "0"
    sys.meta_path.insert(0, _GuardOnImport(block_log))


class _GuardOnImport(importlib.abc.MetaPathFinder):
    """Add the guard middleware to LiteLLM's app before the server starts it."""

    def __init__(self, block_log: str | None):
        self.block_log = block_log

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname != _PROXY_SERVER:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot guard {fullname}: module not found")
        execute = spec.loader.exec_module
        block_log = self.block_log

        def exec_module(module: Any) -> None:
            execute(module)
            module.app.add_middleware(ModelEndpointGuard, block_log=block_log)

        spec.loader.exec_module = exec_module  # ty: ignore[invalid-assignment]
        return spec


class ModelEndpointGuard:
    """ASGI middleware: refuse every non-model route, WebSocket and hosted-fetch request."""

    def __init__(self, app: Callable[..., Awaitable[None]], block_log: str | None):
        self.app = app
        self.log = Log(block_log)  # stderr when there is no block log (no-web runs)

    async def __call__(self, scope: _Message, receive: _Receive, send: _Send) -> None:
        if scope["type"] == "websocket":
            await receive()  # the handshake request
            self._record(scope, "WEBSOCKET", "gateway-websocket")
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method = scope["method"]
        route = _MODEL_ROUTES.get(method)
        if route is None or route.fullmatch(scope["path"]) is None:
            await self._refuse(scope, send, "gateway-route")
            return
        if method == "POST":
            body, receive = await _buffered(receive)
            if body is None:
                rule = "oversized-body"
            elif _is_form(scope):
                # LiteLLM reads a form body field by field, not as the JSON judged here.
                rule = "uninspectable-body"
            else:
                # A large body takes a while to read; keep LiteLLM's loop serving.
                rule = await asyncio.to_thread(
                    model_request_rule, body, _codings(scope)
                )
            if rule:
                await self._refuse(scope, send, rule)
                return
        await self.app(scope, receive, _marked(send))

    async def _refuse(self, scope: _Message, send: _Send, rule: str) -> None:
        self._record(scope, scope["method"], rule)
        body = json.dumps(
            {
                "error": {
                    "message": f"Blocked by the task network policy: {rule}",
                    "type": "permission_error",
                    "code": "403",
                }
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"x-benchflow-blocked", b"1"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    def _record(self, scope: _Message, method: str, rule: str) -> None:
        host, port = scope.get("server") or ("127.0.0.1", 0)
        url = f"http://{host}:{port}{scope['path']}"
        # A refusal stands even when it cannot be logged.
        with contextlib.suppress(OSError):
            self.log.write(action="blocked", method=method, url=url, rule=rule)


def _is_form(scope: _Message) -> bool:
    return any(
        name.lower() == b"content-type"
        and value.split(b";", 1)[0].strip().lower()
        in (b"application/x-www-form-urlencoded", b"multipart/form-data")
        for name, value in scope.get("headers", ())
    )


def _codings(scope: _Message) -> list[str]:
    return content_codings(
        value.decode("latin-1")
        for name, value in scope.get("headers", ())
        if name.lower() == b"content-encoding"
    )


def _marked(send: _Send) -> _Send:
    """``send`` that tags every response it starts, so benchflow can see the guard is on."""

    async def send_marked(message: _Message) -> None:
        if message["type"] == "http.response.start":
            headers = [*message.get("headers", ()), (b"x-benchflow-egress-guard", b"1")]
            message = {**message, "headers": headers}
        await send(message)

    return send_marked


async def _buffered(receive: _Receive) -> tuple[bytes | None, _Receive]:
    """The whole request body (None past the inspection limit) and a receive that replays it."""
    messages: list[_Message] = []
    size = 0
    while True:
        message = await receive()
        messages.append(message)
        if message["type"] != "http.request":
            break
        size += len(message.get("body", b""))
        if size > BODY_INSPECT_LIMIT:
            return None, receive
        if not message.get("more_body", False):
            break
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.request")

    async def replay() -> _Message:
        return messages.pop(0) if messages else await receive()

    return body, replay
