#!/usr/bin/env python3
"""Sandbox-side research gateway and stdio MCP relay (stdlib only).

This file is copied into task containers.  It deliberately imports no
BenchFlow modules so it works in minimal task images that only provide Python.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import http.client
import ipaddress
import json
import posixpath
import socket
import ssl
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

MAX_FETCH_BYTES = 8 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024
MAX_REQUEST_BYTES = 64 * 1024
MAX_PROVIDER_REQUEST_BYTES = 64 * 1024 * 1024
MAX_REDIRECTS = 5
ANTHROPIC_RELAY_PREFIX = "/provider/anthropic"
ANTHROPIC_RELAY_HOST = "api.anthropic.com"
_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class PolicyBlocked(Exception):
    pass


class GatewayError(Exception):
    pass


class ProviderBlocked(Exception):
    pass


@dataclass(frozen=True)
class Response:
    url: str
    status: int
    content_type: str
    body: bytes


def _normalized_url(url: str) -> tuple[str, int, str]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise GatewayError("only HTTP(S) URLs are supported")
    if parsed.username is not None or parsed.password is not None:
        raise GatewayError("URL userinfo is not supported")
    try:
        host = parsed.hostname.encode("idna").decode("ascii").lower().rstrip(".")
        # Treat default HTTP and HTTPS origins as the same policy resource so
        # a scheme flip or spelling the default port cannot bypass a blocked
        # paper URL. Explicit non-default ports remain distinct.
        explicit_port = parsed.port
        default_port = 443 if parsed.scheme == "https" else 80
        port = 0 if explicit_port in {None, default_port} else explicit_port
    except (UnicodeError, ValueError) as exc:
        raise GatewayError("invalid URL authority") from exc
    decoded_path = parsed.path or "/"
    for _ in range(4):
        expanded = unquote(decoded_path)
        if expanded == decoded_path:
            break
        decoded_path = expanded
    decoded_path = decoded_path.replace("\\", "/")
    path = posixpath.normpath(decoded_path)
    if not path.startswith("/"):
        path = "/" + path
    return host, port, path


class Policy:
    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.urls = {_normalized_url(value) for value in raw["blocked_urls"]}
        self.prefixes = {
            _normalized_url(value) for value in raw["blocked_url_prefixes"]
        }
        self.hosts = {value.lower().rstrip(".") for value in raw["blocked_hosts"]}
        self.terms = tuple(value.casefold() for value in raw["blocked_terms"])
        self.hashes = set(raw["blocked_content_sha256"])
        self.search_endpoint = raw["search_endpoint"]
        self.sha256 = raw["policy_sha256"]

    def check_url(self, url: str) -> None:
        host, port, path = _normalized_url(url)
        if host in self.hosts or any(host.endswith("." + item) for item in self.hosts):
            raise PolicyBlocked
        if (host, port, path) in self.urls:
            raise PolicyBlocked
        for prefix_host, prefix_port, prefix_path in self.prefixes:
            prefix = prefix_path.rstrip("/")
            if (
                host == prefix_host
                and port == prefix_port
                and (path == prefix or path.startswith(prefix + "/"))
            ):
                raise PolicyBlocked

    def check_text(self, value: str) -> None:
        folded = value.casefold()
        if any(term in folded for term in self.terms):
            raise PolicyBlocked

    def check_body(self, body: bytes, content_type: str) -> None:
        if hashlib.sha256(body).hexdigest() in self.hashes:
            raise PolicyBlocked
        if self.terms and (
            content_type.startswith("text/")
            or "json" in content_type
            or "xml" in content_type
        ):
            self.check_text(body.decode("utf-8", errors="replace"))


def _resolve_public_ip(host: str, port: int) -> str:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise GatewayError("DNS resolution failed") from exc
    addresses = []
    for info in infos:
        address = info[4][0]
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise GatewayError("private or non-global destinations are not allowed")
        addresses.append(address)
    if not addresses:
        raise GatewayError("DNS resolution returned no addresses")
    return addresses[0]


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, port: int, ip: str, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout)
        self._ip = ip

    def connect(self) -> None:
        self.sock = socket.create_connection((self._ip, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, port: int, ip: str, timeout: float) -> None:
        context = ssl.create_default_context()
        super().__init__(host, port, timeout=timeout, context=context)
        self._ip = ip
        self._ssl_context = context

    def connect(self) -> None:
        sock = socket.create_connection((self._ip, self.port), self.timeout)
        self.sock = self._ssl_context.wrap_socket(sock, server_hostname=self.host)


def _request(
    policy: Policy,
    url: str,
    *,
    max_bytes: int,
    scan_content: bool = True,
) -> Response:
    current = url
    for redirect_count in range(MAX_REDIRECTS + 1):
        policy.check_url(current)
        parsed = urlsplit(current)
        host, _policy_port, _ = _normalized_url(current)
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError as exc:
            raise GatewayError("invalid URL port") from exc
        ip = _resolve_public_ip(host, port)
        connection_cls = (
            _PinnedHTTPSConnection
            if parsed.scheme == "https"
            else _PinnedHTTPConnection
        )
        connection = connection_cls(host, port, ip, 20.0)
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        default_port = 443 if parsed.scheme == "https" else 80
        header_host = f"[{host}]" if ":" in host else host
        host_header = header_host if port == default_port else f"{header_host}:{port}"
        try:
            connection.request(
                "GET",
                target,
                headers={
                    "Host": host_header,
                    "User-Agent": "BenchFlow-Research-Gateway/1.0",
                    "Accept": "*/*",
                    "Connection": "close",
                },
            )
            upstream = connection.getresponse()
            if upstream.status in {301, 302, 303, 307, 308}:
                location = upstream.getheader("Location")
                if not location:
                    raise GatewayError("redirect missing Location header")
                if redirect_count == MAX_REDIRECTS:
                    raise GatewayError("too many redirects")
                current = urljoin(current, location)
                continue
            body = upstream.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise GatewayError("upstream response exceeds size limit")
            content_type = (upstream.getheader("Content-Type") or "").split(";", 1)[0]
            response = Response(current, upstream.status, content_type.lower(), body)
            if scan_content:
                policy.check_body(body, response.content_type)
            return response
        except (OSError, http.client.HTTPException) as exc:
            raise GatewayError("upstream request failed") from exc
        finally:
            connection.close()
    raise GatewayError("too many redirects")


class _SearchParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._href: str | None = None
        self._title: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = (values.get("class") or "").split()
        if tag == "a" and "result__a" in classes:
            self._href = values.get("href")
            self._title = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._title.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._href is None:
            return
        href = self._href
        if href.startswith("//"):
            href = "https:" + href
        query = parse_qs(urlsplit(href).query)
        if query.get("uddg"):
            href = query["uddg"][0]
        self.results.append(
            {"title": html.unescape("".join(self._title).strip()), "url": href}
        )
        self._href = None
        self._title = []


def _search(policy: Policy, query: str, max_results: int) -> list[dict[str, str]]:
    policy.check_text(query)
    separator = "&" if "?" in policy.search_endpoint else "?"
    response = _request(
        policy,
        f"{policy.search_endpoint}{separator}q={quote_plus(query)}",
        max_bytes=2 * 1024 * 1024,
        scan_content=False,
    )
    if response.status >= 400:
        raise GatewayError(f"search provider returned HTTP {response.status}")
    parser = _SearchParser()
    parser.feed(response.body.decode("utf-8", errors="replace"))
    visible = []
    for result in parser.results:
        try:
            policy.check_url(result["url"])
            policy.check_text(result["title"])
        except (PolicyBlocked, GatewayError):
            continue
        visible.append(result)
        if len(visible) >= max_results:
            break
    return visible


def _result_payload(policy: Policy, url: str, max_bytes: int) -> dict[str, Any]:
    response = _request(policy, url, max_bytes=max_bytes)
    if response.status >= 400:
        raise GatewayError(f"upstream returned HTTP {response.status}")
    textual = response.content_type.startswith("text/") or any(
        kind in response.content_type for kind in ("json", "xml")
    )
    return {
        "url": response.url,
        "content_type": response.content_type,
        "encoding": "utf-8" if textual else "base64",
        "content": (
            response.body.decode("utf-8", errors="replace")
            if textual
            else base64.b64encode(response.body).decode("ascii")
        ),
    }


def _check_anthropic_provider_payload(value: Any) -> None:
    """Reject Anthropic features that could perform server-side research."""

    if isinstance(value, list):
        for item in value:
            _check_anthropic_provider_payload(item)
        return
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        normalized_key = str(key).casefold().replace("-", "_")
        if normalized_key in {"mcp_servers", "web_search_options"}:
            raise ProviderBlocked
        if normalized_key == "type" and isinstance(item, str):
            normalized_type = item.casefold().replace("-", "_")
            if normalized_type in {"web_search", "web_fetch"} or any(
                normalized_type.startswith(prefix)
                for prefix in ("web_search_", "web_fetch_")
            ):
                raise ProviderBlocked
        _check_anthropic_provider_payload(item)


def _provider_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ProviderBlocked
        value[key] = item
    return value


def _validate_anthropic_provider_body(
    body: bytes | None, target: str, headers: Any
) -> None:
    if not body:
        return
    content_encoding = (headers.get("Content-Encoding") or "").casefold()
    if content_encoding not in {"", "identity"}:
        raise ProviderBlocked
    content_type = (headers.get("Content-Type") or "").casefold()
    is_messages_request = urlsplit(target).path.startswith("/v1/messages")
    if "json" not in content_type and not is_messages_request:
        return
    try:
        payload = json.loads(body, object_pairs_hook=_provider_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        if is_messages_request:
            raise ProviderBlocked from exc
        return
    _check_anthropic_provider_payload(payload)


class GatewayHandler(BaseHTTPRequestHandler):
    policy: Policy

    def log_message(self, format: str, *args: Any) -> None:
        del format, args
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _anthropic_target(self) -> str | None:
        if self.path == ANTHROPIC_RELAY_PREFIX:
            return "/"
        if self.path.startswith(ANTHROPIC_RELAY_PREFIX + "/"):
            return self.path[len(ANTHROPIC_RELAY_PREFIX) :]
        if self.path.startswith(ANTHROPIC_RELAY_PREFIX + "?"):
            return "/" + self.path[len(ANTHROPIC_RELAY_PREFIX) :]
        return None

    def _relay_anthropic(self, target: str) -> None:
        """Relay one native Claude API request to a fixed HTTPS origin."""

        raw_length = self.headers.get("Content-Length")
        if self.headers.get("Transfer-Encoding"):
            self._json(400, {"error": "chunked provider requests are not supported"})
            return
        try:
            length = int(raw_length or "0")
        except ValueError:
            self._json(400, {"error": "invalid provider request length"})
            return
        if length < 0 or length > MAX_PROVIDER_REQUEST_BYTES:
            self._json(413, {"error": "provider request exceeds size limit"})
            return

        body = self.rfile.read(length) if length else None
        try:
            _validate_anthropic_provider_body(body, target, self.headers)
        except ProviderBlocked:
            self._json(403, {"error": "provider-side web access is disabled"})
            return
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in _HOP_BY_HOP_HEADERS | {"host", "content-length"}
        }
        headers["Host"] = ANTHROPIC_RELAY_HOST
        if body is not None:
            headers["Content-Length"] = str(len(body))

        connection: _PinnedHTTPSConnection | None = None
        response_started = False
        try:
            ip = _resolve_public_ip(ANTHROPIC_RELAY_HOST, 443)
            connection = _PinnedHTTPSConnection(ANTHROPIC_RELAY_HOST, 443, ip, 300.0)
            connection.request(self.command, target, body=body, headers=headers)
            upstream = connection.getresponse()
            self.send_response(upstream.status, upstream.reason)
            for key, value in upstream.getheaders():
                if key.lower() not in _HOP_BY_HOP_HEADERS | {"content-length"}:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            response_started = True
            while True:
                chunk = upstream.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
            self.close_connection = True
        except (OSError, http.client.HTTPException):
            if response_started:
                self.close_connection = True
            else:
                self._json(502, {"error": "native provider relay failed"})
        finally:
            if connection is not None:
                connection.close()

    def do_GET(self) -> None:
        provider_target = self._anthropic_target()
        if provider_target is not None:
            self._relay_anthropic(provider_target)
            return
        if self.path == "/health":
            self._json(200, {"ok": True, "policy_sha256": self.policy.sha256})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        provider_target = self._anthropic_target()
        if provider_target is not None:
            self._relay_anthropic(provider_target)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > MAX_REQUEST_BYTES:
                raise GatewayError("request exceeds size limit")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise GatewayError("request must be a JSON object")
            if self.path == "/search":
                query = payload.get("query")
                maximum = payload.get("max_results", 10)
                if not isinstance(query, str) or not query.strip():
                    raise GatewayError("query must be a non-empty string")
                if not isinstance(maximum, int) or not 1 <= maximum <= 20:
                    raise GatewayError("max_results must be between 1 and 20")
                self._json(200, {"results": _search(self.policy, query, maximum)})
                return
            if self.path in {"/fetch", "/download"}:
                url = payload.get("url")
                if not isinstance(url, str):
                    raise GatewayError("url must be a string")
                limit = (
                    MAX_DOWNLOAD_BYTES if self.path == "/download" else MAX_FETCH_BYTES
                )
                self._json(200, _result_payload(self.policy, url, limit))
                return
            self._json(404, {"error": "not found"})
        except PolicyBlocked:
            self._json(403, {"error": "blocked by research policy"})
        except (GatewayError, json.JSONDecodeError, ValueError, TypeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception:
            self._json(502, {"error": "research gateway request failed"})

    def do_DELETE(self) -> None:
        provider_target = self._anthropic_target()
        if provider_target is None:
            self._json(404, {"error": "not found"})
            return
        self._relay_anthropic(provider_target)

    do_PATCH = do_DELETE
    do_PUT = do_DELETE


def _call_gateway(endpoint: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except Exception as exc:
        detail = "research gateway rejected the request"
        if isinstance(exc, HTTPError):
            try:
                raw = json.loads(exc.read())
                detail = str(raw.get("error") or detail)
            except Exception:
                pass
        raise GatewayError(detail) from exc


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "web_search",
            "description": "Search the public web through the benchmark research policy.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "web_fetch",
            "description": "Fetch an allowed HTTP(S) resource through the benchmark policy.",
            "inputSchema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "additionalProperties": False,
            },
        },
        {
            "name": "web_download",
            "description": "Download an allowed resource to a file in the task workspace.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "destination": {"type": "string"},
                },
                "required": ["url", "destination"],
                "additionalProperties": False,
            },
        },
    ]


def _mcp_call(endpoint: str, name: str, arguments: dict[str, Any]) -> str:
    if name == "web_search":
        return json.dumps(_call_gateway(endpoint, "/search", arguments), indent=2)
    if name == "web_fetch":
        return json.dumps(_call_gateway(endpoint, "/fetch", arguments), indent=2)
    if name == "web_download":
        destination = arguments.get("destination")
        if not isinstance(destination, str) or not destination.strip():
            raise GatewayError("destination must be a non-empty path")
        payload = _call_gateway(endpoint, "/download", {"url": arguments.get("url")})
        content = payload["content"]
        data = (
            base64.b64decode(content)
            if payload.get("encoding") == "base64"
            else content.encode("utf-8")
        )
        target = Path(destination).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return json.dumps(
            {
                "saved_to": str(target),
                "bytes": len(data),
                "content_type": payload.get("content_type"),
            }
        )
    raise GatewayError("unknown research tool")


def run_mcp(endpoint: str) -> None:
    for line in sys.stdin:
        request: Any = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            method = request.get("method")
            if request_id is None:
                continue
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "benchflow-research", "version": "1.0"},
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": _tools()}
            elif method == "tools/call":
                params = request.get("params") or {}
                output = _mcp_call(
                    endpoint, params.get("name", ""), params.get("arguments") or {}
                )
                result = {
                    "content": [{"type": "text", "text": output}],
                    "isError": False,
                }
            else:
                raise GatewayError("unsupported MCP method")
            response = {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id") if isinstance(request, dict) else None,
                "error": {"code": -32000, "message": str(exc)},
            }
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()


def run_server(policy_path: str, port: int) -> None:
    raw = json.loads(Path(policy_path).read_text())
    policy = Policy(raw)
    GatewayHandler.policy = policy
    server = ThreadingHTTPServer(("127.0.0.1", port), GatewayHandler)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--policy", required=True)
    serve.add_argument("--port", required=True, type=int)
    mcp = subparsers.add_parser("mcp")
    mcp.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    if args.command == "serve":
        run_server(args.policy, args.port)
    else:
        run_mcp(args.endpoint)


if __name__ == "__main__":
    main()
