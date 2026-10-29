"""Egress policy shared by the denylist proxy and the model gateway guard. Stdlib only.

``Policy`` decides which URLs a denylist task keeps out of the agent's reach.
``model_request_rule`` refuses model requests that would have the provider
fetch or search the web for the agent, which no sandbox-side URL rule can see.
``Log`` appends each refusal to the block log the enforcement points share.

The proxy and the gateway guard run inside the sandbox, where benchflow is not
installed, so benchflow ships this file there at its package path
(``benchflow/sandbox/_egress_policy.py``) next to them; both import it exactly
as they do in-process.
"""

from __future__ import annotations

import ipaddress
import json
import re
import stringprep
import sys
import threading
import unicodedata
import urllib.parse
import zlib
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timezone
from typing import Any

# A body that has to be read whole to be checked is refused above this size.
BODY_INSPECT_LIMIT = 64 * 1024 * 1024


def host_key(host: str) -> str:
    """Comparison form of a hostname: lowercase, no trailing dot, no leading www."""
    host = host.strip().rstrip(".").lower()
    return host[4:] if host.startswith("www.") else host


def _unquoted(text: str) -> str:
    """``text`` with percent-encoding decoded, repeatedly (up to three layers)."""
    for _ in range(3):
        decoded = urllib.parse.unquote(text)
        if decoded == text:
            break
        text = decoded
    return text


def _path_key(path: str) -> str:
    """Comparison form of a path: decoded, dot segments resolved, one slash between segments."""
    path = _unquoted(path.split("?", 1)[0].split("#", 1)[0])
    path = path.replace("\\", "/")
    segments: list[str] = []
    for segment in path.split("/"):
        segment = segment.split(";", 1)[0]
        if segment in ("", "."):
            continue
        if segment == "..":
            if segments:
                segments.pop()
            continue
        segments.append(segment)
    key = "/" + "/".join(segments)
    if path.endswith(("/", "/.", "/..")) and key != "/":
        key += "/"
    return key.lower()


_EMBEDDED_IPV4 = re.compile(r"(?:^|[.-])(?:\d{1,3}[.-]){3}\d{1,3}(?:[.-]|$)")
_HEX_IPV4_LABEL = re.compile(r"^[0-9a-f]{8}$")
_WILDCARD_DNS = (
    "nip.io",
    "sslip.io",
    "xip.io",
    "traefik.me",
    "localtest.me",
    "lvh.me",
    "vcap.me",
)
# A hostname (a trailing dot allowed) and the path after it, written inside
# another URL's path or query.
_EMBEDDED_URL = re.compile(
    r"(?<![a-z0-9.-])((?:[a-z0-9-]+\.)+[a-z0-9-]+)\.?(?::[0-9]+)?(/[^?#&\s]*)?"
)
# Dots a URL parser reads as the label separator (UTS #46).
_FULL_STOPS = str.maketrans({"\u3002": ".", "\uff0e": ".", "\uff61": "."})


def looks_like_address(host: str) -> bool:
    """True for anything that names an address rather than a site.

    glibc accepts decimal, octal, hex and short dotted forms (``3232235777``,
    ``0xc0a80101``, ``0300.0250.1.1``, ``127.1``), and wildcard DNS services
    resolve an address embedded in the name (``1-2-3-4.sslip.io``).
    """
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    labels = host.split(".")
    if not any(c.isalpha() for c in host):
        return True
    if labels[-1].isdigit():
        return True
    if any(label.startswith("0x") or _HEX_IPV4_LABEL.match(label) for label in labels):
        return True
    if _EMBEDDED_IPV4.search(host):
        return True
    return any(host == d or host.endswith("." + d) for d in _WILDCARD_DNS)


class Policy:
    """Match hosts and URLs against the denylist; scheme, port and query are ignored."""

    def __init__(
        self,
        blocked_urls: list[str],
        blocked_hosts: list[str],
        model_gateway_port: int | None = None,
    ):
        # Controller-supplied runtime state, never a task-authored allowlist.
        if model_gateway_port is not None and (
            type(model_gateway_port) is not int
            or not 1024 <= model_gateway_port <= 65535
        ):
            raise ValueError("invalid model gateway port")
        self.model_gateway_port = model_gateway_port
        self.prefixes: list[tuple[str, str]] = []
        for raw in blocked_urls:
            url = raw if "://" in raw else "https://" + raw
            parts = urllib.parse.urlsplit(url)
            if not parts.hostname:
                raise ValueError(f"blocked_urls entry has no host: {raw!r}")
            prefix = _path_key(parts.path).rstrip("/") or "/"
            self.prefixes.append((host_key(parts.hostname), prefix))
        self.hosts = {h.strip().rstrip(".").lower() for h in blocked_hosts if h.strip()}
        self.inspect_hosts = {host for host, _ in self.prefixes}

    @classmethod
    def load(cls, path: str) -> Policy:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(
            list(data.get("blocked_urls") or []),
            list(data.get("blocked_hosts") or []),
            data.get("model_gateway_port"),
        )

    def host_rule(self, host: str, port: int = 0) -> str | None:
        if (host, port) == ("127.0.0.1", self.model_gateway_port):
            return None
        name = host.strip().rstrip(".").lower()
        if looks_like_address(name):
            return "ip-literal"
        return self._listed_host(name)

    def url_rule(self, host: str, path: str, port: int = 0) -> str | None:
        """The rule ``host`` + ``path`` breaks, including a listed page named inside the path or query."""
        return (
            self.host_rule(host, port)
            or self._listed_prefix(host, path)
            or self._embedded_rule(path)
        )

    def inspect(self, host: str) -> bool:
        return host_key(host) in self.inspect_hosts

    def _listed_host(self, name: str) -> str | None:
        for blocked in self.hosts:
            if name == blocked or name.endswith("." + blocked):
                return f"host:{blocked}"
        return None

    def _listed_prefix(self, host: str, path: str) -> str | None:
        key, pkey = host_key(host), _path_key(path)
        for bhost, bpath in self.prefixes:
            if key == bhost and pkey.startswith(bpath):
                return f"url:{bhost}{bpath}"
        return None

    def _embedded_rule(self, target: str) -> str | None:
        """Refuse a request that hands a listed page's address to another site.

        Reader services, web archives and redirectors fetch the address they
        are given (``r.jina.ai/https://arxiv.org/abs/2401.12345``), so the page
        would reach the agent from a host the denylist does not name. The
        address is read the way a URL parser would: decoded, without the
        characters URL parsing and IDNA drop (tab and newlines, soft hyphen,
        zero-width joiners), NFKC-folded, with ideographic and fullwidth full
        stops as dots.
        """
        text = "".join(
            c
            for c in _unquoted(target)
            if c not in "\t\n\r" and not stringprep.in_table_b1(c)
        )
        text = unicodedata.normalize("NFKC", text).translate(_FULL_STOPS)
        for match in _EMBEDDED_URL.finditer(text.replace("\\", "/").lower()):
            host, path = match.group(1), match.group(2) or "/"
            rule = self._listed_host(host) or self._listed_prefix(host, path)
            if rule:
                return rule
        return None


class Log:
    """The block log: one JSON object per refused request, from the proxy and the gateway guard."""

    def __init__(self, path: str | None):
        self.path = path
        self._lock = threading.Lock()

    def write(self, **fields: object) -> None:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017
        line = json.dumps({"ts": stamp, **fields}, sort_keys=True)
        with self._lock:
            if self.path:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            else:
                print(line, file=sys.stderr, flush=True)


# Model-request features that make the provider fetch or search the web for
# the caller. The provider's servers reach the page, so the sandbox network
# never sees it. Server tools are matched by type prefix (OpenAI, Anthropic,
# xAI, Mistral, Zhipu, Groq, Moonshot); Gemini names its tools by key instead.
_HOSTED_TOOL_TYPES = (
    "web_search",
    "web_fetch",
    "url_context",
    "google_search",
    "x_search",
    "browser_search",
    "builtin_function",
    "mcp",
)
_HOSTED_TOOL_KEYS = (
    "google_search",
    "googleSearch",
    "google_search_retrieval",
    "googleSearchRetrieval",
    "url_context",
    "urlContext",
    "enterprise_web_search",
    "enterpriseWebSearch",
    "retrieval",
)
# Request fields that turn on provider-side search, fetching or remote servers:
# OpenAI search models, xAI live search, DashScope search, OpenRouter plugins,
# Anthropic's MCP connector, Azure OpenAI "on your data" retrieval, and
# LiteLLM's ``extra_body``, merged into the provider call unchecked (model
# included).
_HOSTED_REQUEST_FIELDS = (
    "web_search_options",
    "search_parameters",
    "enable_search",
    "plugins",
    "mcp_servers",
    "data_sources",
    "extra_body",
)
# Content parts that hand over a file, image, audio or video. Any address
# inside one is fetched by the provider, or by LiteLLM, which downloads some
# itself (``file_id`` and ``file_data`` included).
_ADDRESSED_PART_TYPES = frozenset(
    {
        "image_url",
        "input_image",
        "input_file",
        "file",
        "document_url",
        "video_url",
        "audio_url",
    }
)
# Where a value is never acted on by the provider: tool definitions (classified
# whole), JSON Schema property maps, and the arguments of a call already made.
_SKIPPED_CHILDREN = frozenset({"tools", "properties"})
_CALL_ARGUMENTS = {
    "functionCall": "args",
    "function_call": "args",
    "functionResponse": "response",
    "function_response": "response",
}
_TOOL_CALL_TYPES = ("tool_use", "server_tool_use")
# A scheme and authority: an address the provider would fetch itself. A data:
# URL carries its bytes and has no authority.
_REMOTE_URL = re.compile(r"[\x00-\x20]*[a-z][a-z0-9+.-]*://", re.IGNORECASE)
# Files uploaded to the provider itself are referenced by an https URI too.
_PROVIDER_FILE_STORE = "https://generativelanguage.googleapis.com/"
# Content codings undone before a body is read; 32 + MAX_WBITS takes gzip and zlib.
_INFLATE_CODINGS = ("gzip", "x-gzip", "deflate")
# Clients send one gzip member; many cost a reader time for nothing.
_MAX_COMPRESSED_MEMBERS = 16


def content_codings(header_values: Iterable[str]) -> list[str]:
    """The content codings named by ``Content-Encoding`` values, in order, without identity."""
    return [
        coding
        for value in header_values
        for coding in (part.strip().lower() for part in value.split(","))
        if coding not in ("", "identity")
    ]


def model_request_rule(body: bytes, codings: Sequence[str] = ()) -> str | None:
    """The rule a model API request body breaks: a hosted fetch, or a body that cannot be read.

    ``codings`` are the request's content codings, outermost last.
    """
    if not body:
        return None
    try:
        request = read_model_request(decoded_body(body, codings))
    except (ValueError, RecursionError):
        return "uninspectable-body"
    return hosted_fetch_rule(request)


def decoded_body(body: bytes, codings: Sequence[str]) -> bytes:
    """``body`` with its gzip and deflate codings undone; ValueError for any other or a bad one."""
    for coding in reversed(codings):
        if coding not in _INFLATE_CODINGS:
            raise ValueError(f"unsupported content coding {coding!r}")
        body = _inflated(body)
    return body


def read_model_request(body: bytes) -> Any:
    """Parse a decoded request body the way lenient servers would read it.

    Detects UTF-16 and UTF-32 as ``json`` does, replaces invalid bytes, and
    parses the first JSON value, ignoring anything after it. Raises ValueError
    for a body that holds no JSON value.
    """
    text = body.decode(json.detect_encoding(body), errors="replace")
    value, _end = json.JSONDecoder().raw_decode(text.lstrip("\ufeff \t\r\n"))
    return value


def _inflated(data: bytes) -> bytes:
    out = bytearray()
    for _member in range(_MAX_COMPRESSED_MEMBERS):
        inflater = zlib.decompressobj(32 + zlib.MAX_WBITS)
        try:
            out += inflater.decompress(data, BODY_INSPECT_LIMIT + 1 - len(out))
        except zlib.error as exc:
            raise ValueError("corrupt compressed body") from exc
        if len(out) > BODY_INSPECT_LIMIT or inflater.unconsumed_tail:
            raise ValueError("compressed body too large")
        if not inflater.eof:
            raise ValueError("truncated compressed body")
        data = inflater.unused_data
        if not data:
            return bytes(out)
    raise ValueError("too many compressed members")


def hosted_fetch_rule(request: Any) -> str | None:
    """``hosted-fetch:<feature>`` when a model request asks the provider to fetch or search the web.

    Every object in the request is looked at, whatever envelope wraps it
    (batch requests, ``request`` and ``params`` wrappers, ``extra_body``),
    except where the provider never acts on a value.
    """
    stack: list[tuple[str, Any]] = [("", request)]
    while stack:
        key, node = stack.pop()
        if isinstance(node, list):
            stack.extend((key, item) for item in node)
        elif isinstance(node, dict):
            feature = _hosted_feature(node)
            if feature:
                return f"hosted-fetch:{feature}"
            stack.extend(_acted_on(key, node))
    return None


def _acted_on(key: str, node: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    """The children of ``node``, found under ``key``, that the provider may act on."""
    skipped = set(_SKIPPED_CHILDREN)
    if key in _CALL_ARGUMENTS:
        skipped.add(_CALL_ARGUMENTS[key])
    if key == "content" and node.get("type") in _TOOL_CALL_TYPES:
        skipped.add("input")
    return ((k, v) for k, v in node.items() if k not in skipped)


def _hosted_feature(node: dict[str, Any]) -> str | None:
    for field in _HOSTED_REQUEST_FIELDS:
        # An empty object still switches OpenAI's search on; only null, false
        # and an empty list leave a feature off.
        if node.get(field) not in (None, False, []):
            return field
    tools = node.get("tools")
    # Google's JSON parser takes a single object where a list is expected.
    entries = [tools] if isinstance(tools, dict) else tools
    for tool in entries if isinstance(entries, list) else ():
        name = _hosted_tool(tool)
        if name:
            return f"tool:{name}"
    return _addressed_input(node)


def _hosted_tool(tool: Any) -> str | None:
    if not isinstance(tool, dict):
        return None
    kind = tool.get("type")
    if isinstance(kind, str) and kind.startswith(_HOSTED_TOOL_TYPES):
        return kind
    return next((key for key in _HOSTED_TOOL_KEYS if key in tool), None)


def _addressed_input(part: dict[str, Any]) -> str | None:
    kind = part.get("type")
    if isinstance(kind, str) and kind in _ADDRESSED_PART_TYPES and _has_remote(part):
        return f"input:{kind}"
    source = part.get("source")
    if (
        kind in ("image", "document")
        and isinstance(source, dict)
        and (source.get("type") == "url" or _has_remote(source))
    ):
        return f"input:{kind}"
    if any(_has_remote(part.get(field)) for field in ("file_data", "fileData")):
        return "input:file_data"
    return None


def _has_remote(value: Any) -> bool:
    """Whether any string in ``value`` is an address the provider would fetch."""
    stack = [value]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            if _REMOTE_URL.match(node) and not node.strip().startswith(
                _PROVIDER_FILE_STORE
            ):
                return True
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return False
