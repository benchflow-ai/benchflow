"""Local Bedrock proxy that exposes Anthropic Messages and OpenAI Responses."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

import httpx

from benchflow.providers.bedrock_runtime import (
    anthropic_request_to_bedrock_converse,
    bedrock_response_to_anthropic,
    bedrock_response_to_openai_response,
    bedrock_stream_event_to_anthropic_sse,
    bedrock_stream_event_to_openai_response_sse,
    build_bedrock_client,
    openai_responses_request_to_bedrock_converse,
)

logger = logging.getLogger(__name__)


_HTTP_REASONS: dict[int, str] = {
    200: "OK",
    404: "Not Found",
    405: "Method Not Allowed",
    500: "Internal Server Error",
}


def _json_http_response(status: int, body: dict[str, Any]) -> bytes:
    payload = json.dumps(body).encode()
    reason = _HTTP_REASONS.get(status, "OK")
    return (
        f"HTTP/1.1 {status} {reason}\r\n"
        "content-type: application/json\r\n"
        f"content-length: {len(payload)}\r\n"
        "\r\n"
    ).encode() + payload


def _chunk_bytes(payload: bytes) -> bytes:
    return f"{len(payload):x}\r\n".encode() + payload + b"\r\n"


def _sse_json_bytes(payload: bytes) -> bytes:
    return f"data: {payload.decode()}\n\n".encode()


_STREAM_END = object()


def _next_stream_event(iterator: Any) -> Any:
    return next(iterator, _STREAM_END)


def _match_bedrock_model_path(path: str, suffix: str) -> str | None:
    prefix = "/model/"
    marker = f"/{suffix}"
    if not path.startswith(prefix) or not path.endswith(marker):
        return None
    return path[len(prefix) : -len(marker)]


def _http_headers_bytes(status_code: int, headers: list[tuple[str, str]]) -> bytes:
    lines = [f"HTTP/1.1 {status_code} OK"]
    lines.extend(f"{key}: {value}" for key, value in headers)
    lines.append("")
    lines.append("")
    return "\r\n".join(lines).encode()


def _copy_upstream_headers(
    response: httpx.Response,
    *,
    include_content_length: bool = False,
    default_content_type: str | None = None,
) -> list[tuple[str, str]]:
    copied: list[tuple[str, str]] = []
    seen: set[str] = set()

    content_type = response.headers.get("content-type", default_content_type)
    if content_type:
        copied.append(("content-type", content_type))
        seen.add("content-type")

    for key, value in response.headers.items():
        key_lower = key.lower()
        if key_lower in {
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailer",
            "transfer-encoding",
            "upgrade",
        }:
            continue
        if key_lower == "content-length" and not include_content_length:
            continue
        if key_lower in seen:
            continue
        if key_lower.startswith("x-amzn-") or key_lower == "content-length":
            copied.append((key_lower, value))
            seen.add(key_lower)

    return copied


class BedrockProxyServer:
    """HTTP server backed by Bedrock Runtime."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8099,
        client: Any | None = None,
        backend_model: str | None = None,
        frontend_model: str | None = None,
        bedrock_http: httpx.AsyncClient | None = None,
        runtime_env: dict[str, str] | None = None,
    ):
        self._host = host
        self._port = port
        self._client = client
        self._backend_model = backend_model
        self._frontend_model = frontend_model or backend_model
        self._server: asyncio.Server | None = None
        self._bedrock_http = bedrock_http
        self._owns_bedrock_http = bedrock_http is None
        self._runtime_env = (
            dict(runtime_env) if runtime_env is not None else dict(os.environ)
        )

    @property
    def port(self) -> int:
        return self._port

    def _require_client(self) -> Any:
        client = self._client
        assert client is not None
        return client

    async def start(self) -> None:
        if self._client is None:
            self._client = build_bedrock_client(dict(self._runtime_env))
        if self._bedrock_http is None:
            self._bedrock_http = httpx.AsyncClient(timeout=None)
        self._server = await asyncio.start_server(
            self._handle_connection,
            self._host,
            self._port,
        )
        sock = self._server.sockets[0]
        self._port = sock.getsockname()[1]
        logger.info("Bedrock proxy listening on http://%s:%s", self._host, self._port)

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        if self._owns_bedrock_http and self._bedrock_http is not None:
            await self._bedrock_http.aclose()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await reader.readline()
            if not request_line:
                return
            method, path, *_ = request_line.decode().strip().split(" ")
            headers: dict[str, str] = {}
            while True:
                header_line = await reader.readline()
                if header_line in (b"\r\n", b"\n", b""):
                    break
                key, _, value = header_line.decode().partition(":")
                headers[key.strip().lower()] = value.strip()

            body_bytes = b""
            content_length = int(headers.get("content-length", "0"))
            if content_length:
                body_bytes = await reader.readexactly(content_length)
            body = json.loads(body_bytes) if body_bytes else {}

            if method == "GET" and path == "/healthz":
                writer.write(_json_http_response(200, {"ok": True}))
                await writer.drain()
                return
            if method == "GET" and path == "/v1/models":
                writer.write(_json_http_response(200, self._models_list_response()))
                await writer.drain()
                return
            if method == "GET" and path.startswith("/v1/models/"):
                model_id = path[len("/v1/models/") :]
                writer.write(_json_http_response(200, self._model_response(model_id)))
                await writer.drain()
                return

            if method != "POST":
                writer.write(_json_http_response(405, {"error": "method_not_allowed"}))
                await writer.drain()
                return

            if path == "/v1/messages":
                await self._handle_messages(body, writer)
                return
            if path == "/v1/responses":
                await self._handle_responses(body, writer)
                return
            model_id = _match_bedrock_model_path(path, "invoke")
            if model_id is not None:
                await self._handle_bedrock_invoke(model_id, body_bytes, headers, writer)
                return
            model_id = _match_bedrock_model_path(path, "invoke-with-response-stream")
            if model_id is not None:
                await self._handle_bedrock_invoke_stream(
                    model_id, body_bytes, headers, writer
                )
                return
            model_id = _match_bedrock_model_path(path, "count-tokens")
            if model_id is not None:
                await self._handle_bedrock_count_tokens(
                    model_id, body_bytes, headers, writer
                )
                return

            writer.write(_json_http_response(404, {"error": "not_found"}))
            await writer.drain()
        except Exception as exc:
            logger.exception("Bedrock proxy request failed")
            writer.write(_json_http_response(500, {"error": str(exc)}))
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                logger.debug("writer close failed")

    async def _handle_messages(
        self,
        body: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        client = self._require_client()
        model = self._backend_model or body["model"]
        payload = anthropic_request_to_bedrock_converse({**body, "model": model})
        if body.get("stream"):
            response = await asyncio.to_thread(client.converse_stream, **payload)
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"content-type: text/event-stream\r\n"
                b"transfer-encoding: chunked\r\n\r\n"
            )
            await writer.drain()
            iterator = iter(response["stream"])
            while True:
                event = await asyncio.to_thread(_next_stream_event, iterator)
                if event is _STREAM_END:
                    break
                for frame in bedrock_stream_event_to_anthropic_sse(
                    event,
                    model=body["model"],
                ):
                    writer.write(_chunk_bytes(frame.encode()))
                    await writer.drain()
            writer.write(b"0\r\n\r\n")
            await writer.drain()
            return

        response = await asyncio.to_thread(client.converse, **payload)
        normalized = bedrock_response_to_anthropic(response, model=body["model"])
        writer.write(_json_http_response(200, normalized))
        await writer.drain()

    async def _handle_responses(
        self,
        body: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        client = self._require_client()
        model = self._backend_model or body["model"]
        payload = openai_responses_request_to_bedrock_converse({**body, "model": model})
        if body.get("stream"):
            response = await asyncio.to_thread(client.converse_stream, **payload)
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"content-type: text/event-stream\r\n"
                b"transfer-encoding: chunked\r\n\r\n"
            )
            await writer.drain()
            iterator = iter(response["stream"])
            block_types: dict[int, str] = {}
            while True:
                event = await asyncio.to_thread(_next_stream_event, iterator)
                if event is _STREAM_END:
                    break
                for frame in bedrock_stream_event_to_openai_response_sse(
                    event,
                    model=body["model"],
                    block_types=block_types,
                ):
                    writer.write(_chunk_bytes(frame.encode()))
                    await writer.drain()
            writer.write(b"0\r\n\r\n")
            await writer.drain()
            return

        response = await asyncio.to_thread(client.converse, **payload)
        normalized = bedrock_response_to_openai_response(
            response,
            model=body["model"],
        )
        writer.write(_json_http_response(200, normalized))
        await writer.drain()

    def _bedrock_runtime_base_url(self) -> str:
        region = self._runtime_env.get("AWS_REGION") or self._runtime_env.get(
            "AWS_DEFAULT_REGION"
        )
        if not region:
            raise RuntimeError(
                "AWS_REGION or AWS_DEFAULT_REGION required for Bedrock runtime proxy."
            )
        return (
            self._runtime_env.get("ANTHROPIC_BEDROCK_BASE_URL")
            or f"https://bedrock-runtime.{region}.amazonaws.com"
        )

    def _bedrock_auth_headers(self, inbound_headers: dict[str, str]) -> dict[str, str]:
        token = self._runtime_env.get("AWS_BEARER_TOKEN_BEDROCK")
        if not token:
            raise RuntimeError(
                "AWS_BEARER_TOKEN_BEDROCK required for Bedrock runtime proxy."
            )
        headers = {
            "authorization": f"Bearer {token}",
            "content-type": inbound_headers.get("content-type", "application/json"),
        }
        if "accept" in inbound_headers:
            headers["accept"] = inbound_headers["accept"]
        return headers

    async def _proxy_bedrock_http_request(
        self,
        model_id: str,
        *,
        suffix: str,
        body_bytes: bytes,
        inbound_headers: dict[str, str],
    ) -> httpx.Response:
        assert self._bedrock_http is not None
        return await self._bedrock_http.post(
            f"{self._bedrock_runtime_base_url()}/model/{model_id}/{suffix}",
            content=body_bytes,
            headers=self._bedrock_auth_headers(inbound_headers),
        )

    async def _stream_bedrock_http_request(
        self,
        model_id: str,
        *,
        suffix: str,
        body_bytes: bytes,
        inbound_headers: dict[str, str],
    ) -> httpx.Response:
        assert self._bedrock_http is not None
        request = self._bedrock_http.build_request(
            "POST",
            f"{self._bedrock_runtime_base_url()}/model/{model_id}/{suffix}",
            content=body_bytes,
            headers=self._bedrock_auth_headers(inbound_headers),
        )
        return await self._bedrock_http.send(request, stream=True)

    async def _handle_bedrock_invoke(
        self,
        model_id: str,
        body_bytes: bytes,
        inbound_headers: dict[str, str],
        writer: asyncio.StreamWriter,
    ) -> None:
        response = await self._proxy_bedrock_http_request(
            model_id,
            suffix="invoke",
            body_bytes=body_bytes,
            inbound_headers=inbound_headers,
        )
        headers = _copy_upstream_headers(
            response,
            include_content_length=False,
            default_content_type="application/json",
        )
        headers.append(("content-length", str(len(response.content))))
        writer.write(_http_headers_bytes(response.status_code, headers))
        writer.write(response.content)
        await writer.drain()

    async def _handle_bedrock_invoke_stream(
        self,
        model_id: str,
        body_bytes: bytes,
        inbound_headers: dict[str, str],
        writer: asyncio.StreamWriter,
    ) -> None:
        response = await self._stream_bedrock_http_request(
            model_id,
            suffix="invoke-with-response-stream",
            body_bytes=body_bytes,
            inbound_headers=inbound_headers,
        )
        try:
            headers = _copy_upstream_headers(
                response,
                include_content_length=False,
                default_content_type="application/vnd.amazon.eventstream",
            )
            headers.append(("transfer-encoding", "chunked"))
            writer.write(_http_headers_bytes(response.status_code, headers))
            await writer.drain()
            async for chunk in response.aiter_bytes():
                if chunk:
                    writer.write(_chunk_bytes(chunk))
                    await writer.drain()
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        finally:
            await response.aclose()

    async def _handle_bedrock_count_tokens(
        self,
        model_id: str,
        body_bytes: bytes,
        inbound_headers: dict[str, str],
        writer: asyncio.StreamWriter,
    ) -> None:
        client = self._require_client()
        body = json.loads(body_bytes) if body_bytes else {}
        try:
            response = await asyncio.to_thread(
                client.count_tokens,
                modelId=model_id,
                input={"invokeModel": {"body": json.dumps(body)}},
            )
            normalized = {"input_tokens": response["inputTokens"]}
        except Exception as exc:
            if "doesn't support counting tokens" not in str(exc):
                raise
            fallback = await asyncio.to_thread(
                client.invoke_model,
                modelId=model_id,
                body=json.dumps(body),
                contentType=inbound_headers.get("content-type", "application/json"),
                accept="application/json",
            )
            payload = json.loads(fallback["body"].read().decode())
            normalized = {
                "input_tokens": payload.get("usage", {}).get("input_tokens", 0)
            }

        payload = json.dumps(normalized).encode()
        writer.write(
            _http_headers_bytes(
                200,
                [
                    ("content-type", "application/json"),
                    ("content-length", str(len(payload))),
                ],
            )
        )
        writer.write(payload)
        await writer.drain()

    def _model_response(self, model_id: str) -> dict[str, Any]:
        created_at = (
            datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )
        display_name = model_id.replace("anthropic.", "").replace(
            "claude-", "Claude ", 1
        )
        return {
            "id": model_id,
            "type": "model",
            "display_name": display_name,
            "created_at": created_at,
        }

    def _models_list_response(self) -> dict[str, Any]:
        assert self._frontend_model is not None
        model = self._model_response(self._frontend_model)
        return {
            "data": [model],
            "first_id": model["id"],
            "has_more": False,
            "last_id": model["id"],
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BenchFlow Bedrock proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    return parser.parse_args()


async def _main_async() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    server = BedrockProxyServer(host=args.host, port=args.port)
    await server.serve_forever()


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
