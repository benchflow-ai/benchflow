"""A request body as the denylist proxy reads it: framing, streaming, and the checks. Stdlib only.

``RequestBody`` reads one framed request body and checks it before the
request can complete: a body that opens like JSON (or is content-coded) is
read whole and judged against ``benchflow.sandbox._hosted_fetch`` and the
denylist, and any other text body is checked piece by piece as it streams
(``_TextScan``). Pipelined requests and trailers are never forwarded.

benchflow ships this file into the sandbox at its package path
(``benchflow/sandbox/_egress_body.py``) next to the proxy.
"""

from __future__ import annotations

import codecs
import json
import re
import socket
from collections.abc import Iterator

from benchflow.sandbox._egress_policy import Policy
from benchflow.sandbox._hosted_fetch import (
    BODY_INSPECT_LIMIT,
    content_codings,
    decoded_body,
    hosted_fetch_rule,
    named_addresses,
    opens_json,
    read_body,
)

# Longest chunk-size or trailer line read.
_LINE_LIMIT = 64 * 1024


def request_framing(headers: list[tuple[str, str]]) -> tuple[int, bool]:
    """Accept one unambiguous HTTP request body, never a following request."""
    lengths = [v for n, v in headers if n.lower() == "content-length"]
    encodings = [v for n, v in headers if n.lower() == "transfer-encoding"]
    if len(lengths) > 1 or len(encodings) > 1 or (lengths and encodings):
        raise ValueError("ambiguous request framing")
    if encodings:
        if encodings[0].lower() != "chunked":
            raise ValueError("unsupported transfer encoding")
        return 0, True
    if lengths and not re.fullmatch(r"[0-9]+", lengths[0]):
        raise ValueError("invalid content length")
    return int(lengths[0]) if lengths else 0, False


class _BodyReader:
    def __init__(self, sock: socket.socket, initial: bytes):
        self.sock, self.buffer = sock, initial

    def take(self, size: int) -> bytes:
        if not self.buffer:
            self.buffer = self.sock.recv(min(size, 65536))
            if not self.buffer:
                raise ConnectionError("incomplete request body")
        data, self.buffer = self.buffer[:size], self.buffer[size:]
        return data

    def line(self) -> bytes:
        while b"\r\n" not in self.buffer:
            if len(self.buffer) > _LINE_LIMIT:
                raise ConnectionError("body line too large")
            data = self.sock.recv(4096)
            if not data:
                raise ConnectionError("incomplete chunked body")
            self.buffer += data
        line, self.buffer = self.buffer.split(b"\r\n", 1)
        if len(line) > _LINE_LIMIT:
            raise ConnectionError("body line too large")
        return line


def _body_pieces(reader: _BodyReader, length: int, chunked: bool) -> Iterator[bytes]:
    """Yield only this request's body, de-chunked; trailers and pipelined bytes are discarded."""
    if not chunked:
        while length:
            data = reader.take(min(length, 65536))
            length -= len(data)
            yield data
        return
    while True:
        size_text = reader.line().split(b";", 1)[0]
        if not re.fullmatch(rb"[0-9a-fA-F]+", size_text):
            raise ConnectionError("invalid chunk size")
        size = int(size_text, 16)
        if size == 0:
            trailer_size = 0
            while line := reader.line():
                trailer_size += len(line) + 2
                if trailer_size > _LINE_LIMIT:
                    raise ConnectionError("trailers too large")
            return
        while size:
            data = reader.take(min(size, 65536))
            size -= len(data)
            yield data
        if reader.line() != b"":
            raise ConnectionError("invalid chunk terminator")


def body_prefix(headers: list[tuple[str, str]], rest: bytes) -> bytes:
    """Bytes after the head that belong to this request's body; a pipelined request is dropped."""
    names = {n.lower(): v for n, v in headers}
    if "transfer-encoding" in names:
        return rest
    try:
        return rest[: int(names.get("content-length", "0"))]
    except ValueError:
        return b""


class Refused(Exception):
    """A streamed body broke a rule partway; the request is abandoned before it completes."""

    def __init__(self, rule: str):
        super().__init__(rule)
        self.rule = rule


# Uploads whose bytes no service reads as a form, text or JSON Lines.
_BINARY_TYPES = (
    "image/",
    "video/",
    "audio/",
    "font/",
    "application/octet-stream",
    "application/zip",
    "application/gzip",
    "application/x-gzip",
    "application/x-tar",
    "application/x-xz",
    "application/x-bzip2",
    "application/x-7z",
    "application/zstd",
    "application/pdf",
    "application/vnd.",
    "application/x-protobuf",
    "application/grpc",
)


class RequestBody:
    """A request body, checked before the request it belongs to can complete.

    A body that opens like a JSON document (or is content-coded) is read whole
    and judged before anything is forwarded: a model request that asks the
    provider to fetch the web, or a body that names a listed page for a service
    to fetch, never leaves the sandbox. Any other text body streams through,
    each piece checked before it is sent (``_TextScan``). A binary upload
    streams unread.
    """

    def __init__(
        self,
        sock: socket.socket,
        headers: list[tuple[str, str]],
        rest: bytes,
        length: int,
        chunked: bool,
        policy: Policy,
    ):
        self.chunked = chunked
        self.length = length
        self.policy = policy
        self.codings = content_codings(
            v for n, v in headers if n.lower() == "content-encoding"
        )
        types = [
            v.split(";", 1)[0].strip().lower()
            for n, v in headers
            if n.lower() == "content-type"
        ]
        self._text = not types or not types[0].startswith(_BINARY_TYPES)
        reader = _BodyReader(sock, body_prefix(headers, rest))
        self._unread = _body_pieces(reader, length, chunked)
        self._read = b""
        self._scan: _TextScan | None = None

    def check(self) -> str | None:
        """Read as much of the body as the policy needs; the rule it breaks, if any."""
        data = bytearray()
        json_like: bool | None = True if self.codings else None
        scanned = 0  # leading bytes already known to be whitespace
        for piece in self._unread:
            data += piece
            if json_like is None:
                json_like, scanned = opens_json(data, scanned)
                if json_like is False:
                    break
            if len(data) > BODY_INSPECT_LIMIT:
                return "oversized-body"
        self._read = bytes(data)
        if json_like is False:
            if not self._text:
                return None
            # Streamed: what was read is checked now, before the origin is
            # contacted; the rest piece by piece as it is sent.
            self._scan = _TextScan(self.policy)
            rule = self._scan.feed(self._read)
            if not rule and not self.chunked and len(self._read) >= self.length:
                rule = self._scan.finish()
            return rule
        if json_like is None:
            # Whitespace, or a few bytes: no request fits in either.
            return None
        # Read whole; a content-coded body shows whether it is JSON once decoded.
        try:
            decoded = decoded_body(self._read, self.codings)
        except ValueError:
            return "uninspectable-body"
        if not opens_json(decoded, 0)[0]:
            return _TextScan(self.policy).judge(decoded) if self._text else None
        try:
            documents = read_body(decoded)
        except json.JSONDecodeError:
            # No provider takes a top-level array, so an array-opening body that
            # is not JSON (TOML, an INI file, a log) is text, not a request.
            if decoded.lstrip(b" \t\r\n\xef\xbb\xbf")[:1] == b"[" and self._text:
                return _TextScan(self.policy).judge(decoded)
            return "uninspectable-body"
        except ValueError:
            return "uninspectable-body"
        return hosted_fetch_rule(documents) or next(
            filter(None, map(self.policy.named_page_rule, named_addresses(documents))),
            None,
        )

    def send(self, upstream: socket.socket) -> None:
        """Send what was read, then stream the rest, keeping the request's own framing.

        A streamed text body is checked piece by piece before each is sent,
        and whole before the piece or chunk that completes the request, so a
        refused body (``Refused``) never reaches the origin complete.
        """
        self._send_piece(upstream, self._read)
        sent = len(self._read)
        for piece in self._unread:
            sent += len(piece)
            if self._scan is not None:
                completes = not self.chunked and sent >= self.length
                rule = self._scan.feed(piece) or (
                    self._scan.finish() if completes else None
                )
                if rule:
                    raise Refused(rule)
            self._send_piece(upstream, piece)
        if self.chunked:
            rule = self._scan.finish() if self._scan is not None else None
            if rule:
                raise Refused(rule)
            upstream.sendall(b"0\r\n\r\n")

    def _send_piece(self, upstream: socket.socket, piece: bytes) -> None:
        if not self.chunked:
            upstream.sendall(piece)
        elif piece:
            upstream.sendall(f"{len(piece):x}\r\n".encode("ascii") + piece + b"\r\n")


class _TextScan:
    """Checks a text body as it streams: a listed page it names for a service to
    fetch (a form field ``url=...``), and each JSON line that asks a model
    provider to fetch the web (a batch file uploaded as JSON Lines)."""

    # Text kept from the previous piece, for an address split across two.
    OVERLAP = 8192

    def __init__(self, policy: Policy):
        self.policy = policy
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._tail = ""
        self._line = bytearray()
        self._json_line: bool | None = None

    def judge(self, body: bytes) -> str | None:
        """Check a whole body at once."""
        return self.feed(body) or self.finish()

    def feed(self, piece: bytes) -> str | None:
        text = self._tail + self._decoder.decode(piece)
        self._tail = text[-self.OVERLAP :]
        return self.policy.named_page_rule(text) or self._lines(piece)

    def finish(self) -> str | None:
        text = self._decoder.decode(b"", final=True)
        rule = self.policy.named_page_rule(self._tail + text) if text else None
        return rule or self._judge_line()

    def _lines(self, piece: bytes) -> str | None:
        parts = piece.split(b"\n")
        for index, part in enumerate(parts):
            if self._json_line is not False:
                self._line += part
                if self._json_line is None and self._line.lstrip(b" \t\r"):
                    self._json_line = self._line.lstrip(b" \t\r")[:1] in (b"{", b"[")
                if self._json_line is False:
                    self._line.clear()
                elif len(self._line) > BODY_INSPECT_LIMIT:
                    return "oversized-body"
            if index < len(parts) - 1:
                rule = self._judge_line()
                if rule:
                    return rule
        return None

    def _judge_line(self) -> str | None:
        line, json_line = bytes(self._line), self._json_line
        self._line.clear()
        self._json_line = None
        if not json_line:
            return None
        try:
            documents = read_body(line)
        except ValueError:
            # A line that does not parse is skipped, as JSON Lines readers skip it.
            return None
        return hosted_fetch_rule(documents)
