"""The URL Standard's parser for the special schemes, and the one typed form.

Selectors (at contract load) and URLs the agent was observed to fetch (in
the emitter's matcher) go through this one parser into the same
:class:`SpecialURL`, and are compared only through :meth:`SpecialURL.key`,
so the two sides can never disagree on a default port, a host spelling or a
path (R12-1, R12-2). It follows https://url.spec.whatwg.org/ for absolute
URLs with the special schemes http, https, ws, wss and ftp: the C0/space trim
and tab/newline removal, any run of ``/`` and ``\\`` before the authority,
userinfo, host parsing (percent-decoding, forbidden domain code points, the
"ends in a number" IPv4 rule with hex and octal parts, IPv6 with
compression), ports with each scheme's default removed, and the path with
``\\`` as a separator, dot segments and the path percent-encode set. It is
held to WPT's ``urltestdata.json`` (``tests/fixtures/integrity``).

Hosts with non-ASCII code points (or ``xn--`` labels) would need UTS #46
(IDNA) processing, which this module does not implement: they raise
:class:`URLUnsupported`, and callers fail closed on it.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from benchflow.integrity.exec_decompose import is_loopback_host

DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443, "ftp": 21}

_SCHEME_RE = re.compile(r"([A-Za-z][A-Za-z0-9+.\-]*):")
_FORBIDDEN_DOMAIN = set(
    "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f"
    "\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f"
    " #%/:<>?@[\\]^|\x7f"
)
_PATH_ENCODE = set(chr(c) for c in range(0x20)) | set(' "#<>?^`{}')


class URLFailure(ValueError):
    """The URL Standard's parser returns failure for this input."""


class URLUnsupported(ValueError):
    """Valid perhaps, but outside what this parser implements (IDNA hosts)."""


@dataclass(frozen=True)
class SpecialURL:
    scheme: str
    host: str  # serialized: a domain, dotted IPv4, or [IPv6]
    port: int | None  # None for the scheme's default
    path: str  # serialized, starts with "/"
    has_userinfo: bool = False
    has_query: bool = False
    has_fragment: bool = False

    def key(self) -> str:
        """The one serialization used for matching: no userinfo, query or
        fragment, the default port omitted, no trailing ``/``."""
        port = "" if self.port is None else f":{self.port}"
        return f"{self.scheme}://{self.host}{port}{self.path.rstrip('/')}"


def special_scheme(text: str) -> str | None:
    """The lowercase special scheme ``text`` starts with, if any."""
    cleaned = _clean(text)
    match = _SCHEME_RE.match(cleaned)
    if match and match.group(1).lower() in DEFAULT_PORTS:
        return match.group(1).lower()
    return None


def parse_special_url(text: str) -> SpecialURL:
    """``text`` as an absolute special-scheme URL, per the URL Standard."""

    cleaned = _clean(text)
    match = _SCHEME_RE.match(cleaned)
    if not match or match.group(1).lower() not in DEFAULT_PORTS:
        raise URLFailure("not an absolute URL with a special scheme")
    scheme = match.group(1).lower()
    rest = cleaned[match.end() :].lstrip("/\\")
    end = len(rest)
    for index, char in enumerate(rest):
        if char in "/\\?#":
            end = index
            break
    authority, remainder = rest[:end], rest[end:]
    has_userinfo = "@" in authority
    if has_userinfo:
        authority = authority.rsplit("@", 1)[1]
    host_text, port_text = _split_host_port(authority)
    if not host_text:
        raise URLFailure("empty host")
    host = _parse_host(host_text)
    port = _parse_port(port_text, scheme)
    path_text, has_query, has_fragment = remainder, False, False
    for index, char in enumerate(remainder):
        if char in "?#":
            path_text = remainder[:index]
            has_query = char == "?"
            has_fragment = "#" in remainder[index:]
            break
    return SpecialURL(
        scheme=scheme,
        host=host,
        port=port,
        path=_parse_path(path_text),
        has_userinfo=has_userinfo,
        has_query=has_query,
        has_fragment=has_fragment,
    )


def _clean(text: str) -> str:
    # A USVString: lone surrogates are U+FFFD, as the Standard's input is.
    text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    stripped = text.strip("".join(chr(c) for c in range(0x21)))
    return re.sub(r"[\t\n\r]", "", stripped)


def _split_host_port(authority: str) -> tuple[str, str | None]:
    inside = False
    for index, char in enumerate(authority):
        if char == "[":
            inside = True
        elif char == "]":
            inside = False
        elif char == ":" and not inside:
            return authority[:index], authority[index + 1 :]
    return authority, None


def _parse_port(text: str | None, scheme: str) -> int | None:
    if text is None or text == "":
        return None
    if not text.isascii() or not text.isdigit():
        raise URLFailure("bad port")
    port = int(text)
    if port > 65535:
        raise URLFailure("port out of range")
    return None if port == DEFAULT_PORTS[scheme] else port


def _parse_host(text: str) -> str:
    if text.startswith("["):
        if not text.endswith("]"):
            raise URLFailure("unterminated IPv6 host")
        return "[" + _serialize_ipv6(_parse_ipv6(text[1:-1])) + "]"
    raw = _percent_decode(text)
    try:
        domain = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise URLUnsupported("host is not UTF-8") from None
    if not domain.isascii():
        raise URLUnsupported("non-ASCII host needs IDNA processing")
    domain = domain.lower()
    if any(label.startswith("xn--") for label in domain.split(".")):
        raise URLUnsupported("punycode host needs IDNA validation")
    if not domain or any(char in _FORBIDDEN_DOMAIN for char in domain):
        raise URLFailure("forbidden domain code point")
    if _ends_in_number(domain):
        return _parse_ipv4(domain)
    return domain


def _percent_decode(text: str) -> bytes:
    data = text.encode("utf-8")
    out = bytearray()
    index = 0
    while index < len(data):
        byte = data[index]
        if (
            byte == 0x25
            and index + 2 < len(data) + 0
            and re.fullmatch(rb"[0-9A-Fa-f]{2}", data[index + 1 : index + 3])
        ):
            out.append(int(data[index + 1 : index + 3], 16))
            index += 3
            continue
        out.append(byte)
        index += 1
    return bytes(out)


def _ipv4_number(text: str) -> int | None:
    if text == "":
        return None
    radix = 10
    if len(text) >= 2 and text[:2] in ("0x", "0X"):
        text, radix = text[2:], 16
    elif len(text) >= 2 and text[0] == "0":
        text, radix = text[1:], 8
    if text == "":
        return 0
    digits = {10: "0123456789", 16: "0123456789abcdefABCDEF", 8: "01234567"}[radix]
    if not all(char in digits for char in text):
        return None
    return int(text, radix)


def _ends_in_number(domain: str) -> bool:
    parts = domain.split(".")
    if parts[-1] == "":
        if len(parts) == 1:
            return False
        parts = parts[:-1]
    last = parts[-1]
    if last and last.isascii() and last.isdigit():
        return True
    return _ipv4_number(last) is not None


def _parse_ipv4(domain: str) -> str:
    parts = domain.split(".")
    if parts[-1] == "" and len(parts) > 1:
        parts = parts[:-1]
    if len(parts) > 4:
        raise URLFailure("too many IPv4 parts")
    numbers = []
    for part in parts:
        number = _ipv4_number(part)
        if number is None:
            raise URLFailure("bad IPv4 part")
        numbers.append(number)
    if any(number > 255 for number in numbers[:-1]):
        raise URLFailure("IPv4 part out of range")
    if numbers[-1] >= 256 ** (5 - len(numbers)):
        raise URLFailure("IPv4 address out of range")
    address = numbers[-1]
    for index, number in enumerate(numbers[:-1]):
        address += number * 256 ** (3 - index)
    return ".".join(str((address >> shift) & 255) for shift in (24, 16, 8, 0))


def _parse_ipv6(text: str) -> list[int]:
    """The URL Standard's IPv6 parser, step by step."""
    address = [0] * 8
    piece, compress, pointer = 0, None, 0
    chars = [*text, ""]
    if chars[0] == ":":
        if chars[1] != ":":
            raise URLFailure("IPv6 starts with a lone ':'")
        pointer += 2
        piece += 1
        compress = piece
    while chars[pointer] != "":
        if piece == 8:
            raise URLFailure("too many IPv6 pieces")
        if chars[pointer] == ":":
            if compress is not None:
                raise URLFailure("two '::' in IPv6")
            pointer += 1
            piece += 1
            compress = piece
            continue
        value = length = 0
        while (
            length < 4
            and chars[pointer] != ""
            and chars[pointer] in "0123456789abcdefABCDEF"
        ):
            value = value * 16 + int(chars[pointer], 16)
            pointer += 1
            length += 1
        if chars[pointer] == ".":
            if length == 0:
                raise URLFailure("empty IPv4 in IPv6")
            pointer -= length
            if piece > 6:
                raise URLFailure("IPv4 in IPv6 too late")
            seen = 0
            while chars[pointer] != "":
                ipv4 = None
                if seen > 0:
                    if chars[pointer] == "." and seen < 4:
                        pointer += 1
                    else:
                        raise URLFailure("bad IPv4 in IPv6")
                if chars[pointer] == "" or chars[pointer] not in "0123456789":
                    raise URLFailure("bad IPv4 in IPv6")
                while chars[pointer] != "" and chars[pointer] in "0123456789":
                    number = int(chars[pointer])
                    if ipv4 is None:
                        ipv4 = number
                    elif ipv4 == 0:
                        raise URLFailure("leading zero in IPv4 in IPv6")
                    else:
                        ipv4 = ipv4 * 10 + number
                    if ipv4 > 255:
                        raise URLFailure("IPv4 part in IPv6 out of range")
                    pointer += 1
                address[piece] = address[piece] * 0x100 + (ipv4 or 0)
                seen += 1
                if seen in (2, 4):
                    piece += 1
            if seen != 4:
                raise URLFailure("short IPv4 in IPv6")
            break
        if chars[pointer] == ":":
            pointer += 1
            if chars[pointer] == "":
                raise URLFailure("IPv6 ends with ':'")
        elif chars[pointer] != "":
            raise URLFailure("bad IPv6 code point")
        address[piece] = value
        piece += 1
    if compress is not None:
        swaps = piece - compress
        piece = 7
        while piece != 0 and swaps > 0:
            address[piece], address[compress + swaps - 1] = (
                address[compress + swaps - 1],
                address[piece],
            )
            piece -= 1
            swaps -= 1
    elif piece != 8:
        raise URLFailure("too few IPv6 pieces")
    return address


def _serialize_ipv6(address: list[int]) -> str:
    best_start, best_len, start, length = None, 1, None, 0
    for index, value in enumerate([*address, 1]):
        if value == 0 and index < 8:
            if start is None:
                start, length = index, 0
            length += 1
        else:
            if start is not None and length > best_len:
                best_start, best_len = start, length
            start = None
    out, index = "", 0
    while index < 8:
        if index == best_start:
            out += "::" if index == 0 else ":"
            index += best_len
            continue
        out += format(address[index], "x")
        if index != 7:
            out += ":"
        index += 1
    return out


def _parse_path(text: str) -> str:
    segments: list[str] = []
    if text == "":
        return "/"
    parts = re.split(r"[/\\]", text)
    if parts and parts[0] == "":
        parts = parts[1:]
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        lowered = part.lower()
        if lowered in ("..", ".%2e", "%2e.", "%2e%2e"):
            if segments:
                segments.pop()
            if last:
                segments.append("")
        elif lowered in (".", "%2e"):
            if last:
                segments.append("")
        else:
            segments.append(_encode_path(part))
    return "/" + "/".join(segments)


def _encode_path(segment: str) -> str:
    out = []
    for char in segment:
        if char in _PATH_ENCODE or ord(char) > 0x7E:
            out.extend(f"%{byte:02X}" for byte in char.encode("utf-8", "surrogatepass"))
        else:
            out.append(char)
    return "".join(out)


# --- matching an observed request against a route ------------------------------
#
# An observation is a read of a protected route only when it addresses the
# route unambiguously (R13-1..R13-3): the same scheme, the same effective
# port, the same host identity (no https/http or loopback folding; no alias
# equivalence is established anywhere) and a path under the route's. The
# path a request really carries depends on its client: a browser applies the
# URL Standard (dot segments resolved, "\" a separator), Python's urllib and
# http.client send the path as written. Where the client is not known, both
# readings are tried, and if they disagree, or two different loopback names
# are involved, the observation is ambiguous: unknown, never a blamed read.

Target = tuple[str, str, int, str]  # scheme, host identity, effective port, path

# Tool kinds whose client follows the URL Standard (a browser).
WHATWG_CLIENT_KINDS = frozenset({"navigate", "browser_navigate"})


def whatwg_target(text: str) -> Target | None:
    try:
        url = parse_special_url(text)
    except (URLFailure, URLUnsupported):
        return None
    port = DEFAULT_PORTS[url.scheme] if url.port is None else url.port
    return (url.scheme, url.host, port, url.path)


def literal_target(text: str) -> Target | None:
    """The request a client that sends the URL as written makes (Python's
    urllib and http.client): urlsplit's host and port, the raw path."""
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    host = parts.hostname
    if scheme not in DEFAULT_PORTS or not host:
        return None
    try:
        address = ipaddress.ip_address(host)
        host = f"[{address.compressed}]" if address.version == 6 else str(address)
    except ValueError:
        host = host.lower()
    return (
        scheme,
        host,
        DEFAULT_PORTS[scheme] if port is None else port,
        parts.path or "/",
    )


def route_target(selector_key: str) -> Target | None:
    return whatwg_target(selector_key)


def route_relation(observed: list | tuple | None, route: Target) -> str:
    """``yes``, ``no`` or ``ambiguous``: whether one reading of an observed
    request addresses one route (benchflow.integrity.routes resolves them)."""
    if observed is None:
        return "no"  # this client would not have sent the request
    scheme, host, port, path = observed
    if scheme != route[0] or port != route[2]:
        return "no"
    prefix = route[3].rstrip("/")
    under = not prefix or path == prefix or path.startswith(prefix + "/")
    if not under:
        return "no"
    if host == route[1]:
        return "yes"
    if is_loopback_host(host) and is_loopback_host(route[1]):
        return "ambiguous"  # loopback names are not an established alias
    return "no"


__all__ = [
    "DEFAULT_PORTS",
    "WHATWG_CLIENT_KINDS",
    "literal_target",
    "route_relation",
    "route_target",
    "whatwg_target",
    "SpecialURL",
    "URLFailure",
    "URLUnsupported",
    "parse_special_url",
    "special_scheme",
]
