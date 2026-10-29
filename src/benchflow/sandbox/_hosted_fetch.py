"""Model requests that have the provider fetch or search the web for the caller. Stdlib only.

An agent that can reach its model API can ask the provider to fetch a page for
it: with a server tool, a request switch, a browsing model or agent, a hosted
container with network access, or content given by address. The provider's
servers reach the page, so no sandbox-side URL rule sees it (the "universal
offline sandbox escape"). ``hosted_fetch_rule`` refuses such requests for the
denylist proxy and the model gateway guard.

A body is judged on every reading a provider might make of it:

* gzip and deflate codings are undone, UTF-16 and UTF-32 are detected, invalid
  bytes are replaced, and every JSON value in the body is read, line by line
  past a line that does not parse (a JSON Lines batch file).
* Names are compared folded: NFKC, case-folded, letters and digits only. Some
  servers match names without regard to case, and folding also removes what a
  lenient decoder drops (``To\\u0000ols``, ``web-search``, ``WEB_SEARCH``).
* A name given more than once counts with every value: parsers keep the first,
  the last, or (Google) all of them.
* A string is an address when a URL parser would fetch it, including
  ``https:host`` and a tab inside the scheme, which Azure OpenAI fetches.

Request switches and server tools count inside a model request: an object
with a model, an agent or a model input, and everything under it (envelopes
and batches name theirs). Elsewhere a ``plugins`` or ``tools`` field is someone
else's data. Provider content parts given by address count anywhere, since
conversation items can be stored before any request names them.

benchflow ships this file into the sandbox at its package path
(``benchflow/sandbox/_hosted_fetch.py``) next to the proxy and the gateway guard.
"""

from __future__ import annotations

import functools
import json
import re
import unicodedata
import zlib
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from benchflow.sandbox._egress_policy import URL_IGNORED, url_text

# A body that has to be read whole to be checked is refused above this size.
BODY_INSPECT_LIMIT = 64 * 1024 * 1024

# Content codings undone before a body is read; 32 + MAX_WBITS takes gzip and zlib.
_INFLATE_CODINGS = ("gzip", "x-gzip", "deflate")
# Clients send one gzip member; many cost a reader time for nothing.
_MAX_COMPRESSED_MEMBERS = 16
# Model requests hold thousands of objects; a body with more is refused unread,
# which bounds the time a check can take.
_MAX_OBJECTS = 1_000_000
_BLANK = re.compile("[ \t\r\n\ufeff]*")
_SIGNIFICANT = re.compile(rb"[^ \t\r\n]")
# Bytes a JSON document can open with in any encoding ``json`` detects: the
# document itself, whitespace, a NUL of UTF-16 or UTF-32, or a byte order mark.
_JSON_OPENERS = b"{[ \t\r\n\x00\xef\xfe\xff"


def content_codings(header_values: Iterable[str]) -> list[str]:
    """The content codings named by ``Content-Encoding`` values, in order, without identity."""
    return [
        coding
        for value in header_values
        for coding in (part.strip().lower() for part in value.split(","))
        if coding not in ("", "identity")
    ]


def opens_json(data: bytes | bytearray, scanned: int) -> tuple[bool | None, int]:
    """Whether a body opens a JSON document, and how far that was read.

    The first byte usually decides. A body that opens with whitespace, a NUL
    or a byte order mark may be UTF-16 or UTF-32 text, which ``json`` and some
    servers accept: that takes four bytes to tell, then the first significant
    byte decides. Until then the answer is None. Scanning resumes at
    ``scanned``, so a long run of leading whitespace is read once.
    """
    if not data or data[0] not in _JSON_OPENERS:
        return False, len(data)
    if data[0] in b"{[":
        return True, len(data)
    if len(data) < 4:
        return None, 0
    if json.detect_encoding(bytes(data[:4])) not in ("utf-8", "utf-8-sig"):
        return True, len(data)
    start = max(scanned, 3 if data.startswith(b"\xef\xbb\xbf") else 0)
    first = _SIGNIFICANT.search(data, start)
    if first is None:
        return None, len(data)
    return data[first.start()] in b"{[", len(data)


def model_request_rule(body: bytes, codings: Sequence[str] = ()) -> str | None:
    """The rule a request body breaks: a hosted fetch, or a body that cannot be read.

    ``codings`` are the request's content codings, outermost last.
    """
    if not body:
        return None
    try:
        documents = read_body(body, codings)
    except ValueError:
        return "uninspectable-body"
    return hosted_fetch_rule(documents)


def decoded_body(body: bytes, codings: Sequence[str]) -> bytes:
    """``body`` with its gzip and deflate codings undone; ValueError for any other or a bad one."""
    for coding in reversed(codings):
        if coding not in _INFLATE_CODINGS:
            raise ValueError(f"unsupported content coding {coding!r}")
        body = _inflated(body)
    return body


def read_body(body: bytes, codings: Sequence[str] = ()) -> list[Any]:
    """Every JSON value in a body, with objects under folded names.

    Undoes gzip and deflate codings, detects UTF-16 and UTF-32 as ``json``
    does, and replaces invalid bytes. A line that does not parse is skipped,
    as JSON Lines readers skip it; bytes after the last value are ignored.
    Raises ``json.JSONDecodeError`` for a body that does not open with a JSON
    value, and ValueError for one that cannot be decoded or is too deep or
    large to read.
    """
    raw = decoded_body(body, codings)
    text = raw.decode(json.detect_encoding(raw), errors="replace")
    count = 0

    def objects(pairs: list[tuple[str, Any]]) -> _Object:
        nonlocal count
        count += 1
        if count > _MAX_OBJECTS:
            raise _TooManyObjects
        return _Object.of(pairs)

    decoder = json.JSONDecoder(object_pairs_hook=objects)
    try:
        first, index = decoder.raw_decode(text, _skip_blank(text, 0))
        documents = [first]
        while (index := _skip_blank(text, index)) < len(text):
            try:
                value, index = decoder.raw_decode(text, index)
            except ValueError:
                break
            documents.append(value)
        else:
            return documents
        # Past a value that does not parse, each line is read on its own: a
        # JSON Lines reader goes on at the next line, and a value that runs to
        # the end of the body is read once rather than once per line.
        for line in text[index:].splitlines():
            line = line.strip(" \t\ufeff")
            if line[:1] in ("{", "["):
                try:
                    documents.append(decoder.raw_decode(line)[0])
                except ValueError:
                    continue
    except RecursionError as exc:
        raise ValueError("body nests deeper than the parser reads") from exc
    except _TooManyObjects as exc:
        raise ValueError("body holds more objects than a request") from exc
    return documents


class _TooManyObjects(Exception):
    """Raised past ``_MAX_OBJECTS``; not a ValueError, so line skipping cannot swallow it."""


def hosted_fetch_rule(documents: Iterable[Any]) -> str | None:
    """``hosted-fetch:<feature>`` when a request asks the provider to fetch or search the web."""
    for document in documents:
        for key, node, in_request in _nodes(document):
            feature = (in_request and _requested_feature(node)) or _addressed_input(
                key, node, in_request
            )
            if feature:
                return f"hosted-fetch:{feature}"
    return None


def named_addresses(documents: Iterable[Any]) -> Iterator[str]:
    """The strings in a body that a service could take as an address to fetch.

    Inside a model request only fields named for an address count (``url``,
    ``uri``, ``href``, ``link``): the rest is conversation. Elsewhere every
    string counts, as a reader service's ``{"url": ...}`` or a search API's
    ``{"q": ...}`` does.
    """
    for document in documents:
        yield from _strings(document)
        for _key, node, in_request in _nodes(document):
            for name, values in node.items():
                if not in_request or name in _ADDRESS_NAMES:
                    yield from _strings(values)


def _strings(value: Any) -> Iterator[str]:
    """The strings in ``value`` and in lists inside it, not those inside objects."""
    stack = [value]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            yield node
        elif isinstance(node, list) and not isinstance(node, _Object):
            stack.extend(node)


def _skip_blank(text: str, index: int) -> int:
    match = _BLANK.match(text, index)
    return match.end() if match else index


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


def _fold(name: str) -> str:
    """The comparison form of a name: NFKC, case-folded, letters and digits only."""
    return _fold_short(name) if len(name) <= 256 else _folded(name)


def _folded(name: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKC", name).casefold() if c.isalnum()
    )


_fold_short = functools.lru_cache(maxsize=4096)(_folded)


class _Object(dict):
    """A JSON object under folded names, each with every value it was given.

    Parsers disagree about a name given twice (first, last, or for Google all
    values), so every value counts. ``names`` keeps each name as first written.
    """

    __slots__ = ("names",)
    names: dict[str, str]

    @classmethod
    def of(cls, pairs: list[tuple[str, Any]]) -> _Object:
        obj = cls()
        obj.names = {}
        for name, value in pairs:
            key = _fold(name)
            obj.setdefault(key, []).append(value)
            obj.names.setdefault(key, name)
        return obj

    def each(self, name: str) -> list[Any]:
        """Every value given for the folded ``name``."""
        return self.get(name, [])

    def kinds(self) -> list[tuple[str, str]]:
        """Every ``type`` the object declares, as written and folded."""
        return [(v, _fold(v)) for v in self.each("type") if isinstance(v, str)]


# Fields that make an object a model request (folded): the model or its input.
_REQUEST_FIELDS = frozenset(
    {"model", "messages", "input", "contents", "prompt", "systeminstruction"}
)
# Request switches that turn on provider-side search, fetching or remote
# servers (folded name: name reported): OpenAI search models, xAI live search,
# DashScope, Baidu and Tencent search, OpenRouter plugins, Anthropic's MCP
# connector, Azure OpenAI "on your data", and LiteLLM's ``extra_body``, merged
# into the provider call unchecked (model included).
_SWITCHES = {
    "websearchoptions": "web_search_options",
    "searchparameters": "search_parameters",
    "enablesearch": "enable_search",
    "websearch": "web_search",
    "enableenhancement": "enable_enhancement",
    "plugins": "plugins",
    "mcpservers": "mcp_servers",
    "datasources": "data_sources",
    "extrabody": "extra_body",
}
# Server tools, by folded type prefix (OpenAI, Anthropic, xAI, Mistral, Zhipu,
# Groq, Moonshot, DashScope) and by folded key (Gemini, Bedrock system tools).
_HOSTED_TOOL_TYPES = (
    "websearch",
    "webfetch",
    "webextractor",
    "urlcontext",
    "googlesearch",
    "enterprisewebsearch",
    "xsearch",
    "imagesearch",
    "browse",
    "builtinfunction",
    "mcp",
    "retrieval",
)
_HOSTED_TOOL_KEYS = frozenset(
    {
        "googlesearch",
        "googlesearchretrieval",
        "urlcontext",
        "enterprisewebsearch",
        "retrieval",
        "mcpservers",
        "systemtool",
    }
)
# Any other tool type or key that names a search or fetch is treated as one,
# except searches over the caller's own tools and files.
_FETCHING = re.compile(r"search|fetch|brows|crawl|scrap|urlcontext|internet|grounding")
_LOCAL_SEARCH = ("toolsearch", "filesearch")
# Tools the client runs or defines itself: only a built-in name makes one hosted.
_CLIENT_TOOL_TYPES = frozenset({"function", "custom", "namespace"})
# Settings that give a hosted container network access (OpenAI and Azure
# hosted shell); a reference to an existing container has unknown access.
_NETWORK_SETTINGS = frozenset(
    {
        "networkpolicy",
        "alloweddomains",
        "domainsecrets",
        "networkaccess",
        "internetaccess",
    }
)
# Models that browse on every request: OpenAI search and deep research models,
# Perplexity Sonar, Groq Compound, OpenRouter's ":online".
_BROWSING_MODEL = re.compile(
    r":online$|search-preview|search-api|deep-research|(?:^|/)sonar(?:$|-)"
    r"|(?:^|/)compound(?:$|-)"
)
# Content parts that hand over a file, image, audio or video, and the fields
# that may hold its address. The provider fetches an address there, or LiteLLM
# does (``file_id`` and ``file_data`` included). Provider part types count
# anywhere; generic ones, which Notion, Slack and search indexes share, only
# inside a model request.
_PART_ADDRESSES = {
    "imageurl": ("imageurl", "fileid", "url", "uri"),
    "inputimage": ("imageurl", "fileid", "url", "uri"),
    "inputfile": ("fileurl", "fileid", "filedata", "url", "uri"),
    "fileurl": ("fileurl", "url", "uri"),
    "documenturl": ("documenturl", "url", "uri"),
    "videourl": ("videourl", "url", "uri"),
    "audiourl": ("audiourl", "url", "uri"),
    "inputaudio": ("inputaudio", "audiourl", "url", "uri"),
    "inputvideo": ("videourl", "fileid", "url", "uri"),
}
_MEDIA_ADDRESSES = {
    "file": ("file", "fileurl", "fileid", "filedata", "url", "uri"),
    "image": ("image", "url", "uri"),
    "video": ("video", "url", "uri"),
    "audio": ("audio", "url", "uri"),
    "document": ("url", "uri"),
}
# Fields of an address object (``image_url: {"url": ...}``, a Gemini ``fileData``).
_ADDRESS_FIELDS = frozenset(
    {"url", "uri", "fileid", "fileurl", "fileuri", "filedata", "data", "imageurl"}
)
# Fields named for an address, which a service handed the object would fetch.
_ADDRESS_NAMES = frozenset({"url", "uri", "href", "link"})
# DashScope's native API takes untyped parts holding an address.
_UNTYPED_MEDIA = ("image", "video", "audio")
# Where the provider never acts on a value: tool definitions (classified
# whole), JSON Schemas, and the arguments of a call already made.
_UNREAD = frozenset(
    {
        "tools",
        "properties",
        "schema",
        "jsonschema",
        "responseschema",
        "responsejsonschema",
        "inputschema",
        "outputschema",
        "defs",
        "definitions",
    }
)
_CALL_ARGUMENTS = {"functioncall": "args", "functionresponse": "response"}
_TOOL_CALL_TYPES = frozenset({"tooluse", "servertooluse"})


def _nodes(document: Any) -> Iterator[tuple[str, _Object, bool]]:
    """Every object the provider may act on, once, with the key it was found under
    and whether it sits inside a model request."""
    stack: list[tuple[str, Any, bool]] = [("", document, False)]
    while stack:
        key, node, in_request = stack.pop()
        if isinstance(node, list):
            stack.extend((key, item, in_request) for item in node)
        elif isinstance(node, _Object):
            in_request = in_request or not _REQUEST_FIELDS.isdisjoint(node)
            yield key, node, in_request
            unread = _unread(key, node)
            stack.extend(
                (name, value, in_request)
                for name, values in node.items()
                if name not in unread
                for value in values
            )


def _unread(key: str, node: _Object) -> frozenset[str]:
    """The children of ``node``, found under ``key``, that the provider never acts on."""
    if key in _CALL_ARGUMENTS:
        return _UNREAD | {_CALL_ARGUMENTS[key]}
    if key == "content" and "type" in node:
        kinds = {kind for _written, kind in node.kinds()}
        if kinds and kinds <= _TOOL_CALL_TYPES:
            return _UNREAD | {"input"}
    return _UNREAD


def _requested_feature(node: _Object) -> str | None:
    for switch, name in _SWITCHES.items():
        # An empty object still switches OpenAI's search on; only null, false
        # and an empty list leave a feature off.
        if any(value not in (None, False, []) for value in node.each(switch)):
            return name
    for tools in node.each("tools"):
        tool = _hosted_tool_in(tools)
        if tool:
            return f"tool:{tool}"
    if _REQUEST_FIELDS.isdisjoint(node):
        return None
    # A hosted agent given the input (Gemini's Interactions API), or a model
    # that browses on every request.
    if "input" in node:
        for value in node.each("agent"):
            if isinstance(value, str) and value.strip():
                return f"agent:{value}"
    models = [
        value
        for field in ("model", "models")
        for given in node.each(field)
        for value in (given if isinstance(given, list) else [given])
    ]
    for value in models:
        if isinstance(value, str) and _BROWSING_MODEL.search(value.strip().lower()):
            return f"model:{value}"
    return None


def _hosted_tool_in(tools: Any) -> str | None:
    """The first hosted tool in a ``tools`` value, nested tool groups included."""
    pending = [tools]
    while pending:
        tools = pending.pop()
        if isinstance(tools, str) and tools.lstrip()[:1] in ("[", "{"):
            # Tools written as a JSON string, for a server that decodes it.
            try:
                tools = json.loads(tools, object_pairs_hook=_Object.of)
            except (ValueError, RecursionError):
                continue
        # Google's JSON parser takes a single object where a list is expected.
        entries = [tools] if isinstance(tools, dict) else tools
        for tool in entries if isinstance(entries, list) else ():
            if isinstance(tool, _Object):
                name = _hosted_tool(tool)
                if name:
                    return name
                pending.extend(tool.each("tools"))
    return None


def _hosted_tool(tool: _Object) -> str | None:
    kinds = tool.kinds()
    for written, kind in kinds:
        if kind.startswith(_HOSTED_TOOL_TYPES) or (
            _FETCHING.search(kind) and not kind.startswith(_LOCAL_SEARCH)
        ):
            return written
    for key in tool:
        if key in _HOSTED_TOOL_KEYS or (
            _FETCHING.search(key) and not key.startswith(_LOCAL_SEARCH)
        ):
            return tool.names[key]
    if all(kind in _CLIENT_TOOL_TYPES for _written, kind in kinds):
        # Moonshot names its built-in tools with a leading "$" ("$web_search").
        names = [
            name
            for holder in [tool, *tool.each("function")]
            if isinstance(holder, _Object)
            for name in holder.each("name")
        ]
        return next(
            (n for n in names if isinstance(n, str) and n.strip().startswith("$")),
            None,
        )
    if _reaches_network(tool):
        return kinds[0][0]
    return None


def _reaches_network(tool: _Object) -> bool:
    """Whether a provider-run tool's definition gives it network access or an address."""
    stack: list[Any] = [tool]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            if _is_address(node):
                return True
        elif isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, _Object):
            if not _NETWORK_SETTINGS.isdisjoint(node) or any(
                kind == "containerreference" for _written, kind in node.kinds()
            ):
                return True
            stack.extend(
                value
                for name, values in node.items()
                if name not in _UNREAD and name != "parameters"
                for value in values
            )
    return False


def _addressed_input(key: str, part: _Object, in_request: bool) -> str | None:
    for written, kind in part.kinds():
        fields = _PART_ADDRESSES.get(kind) or (
            _MEDIA_ADDRESSES.get(kind, ()) if in_request else ()
        )
        addressed = any(
            _holds_address(value) for field in fields for value in part.each(field)
        ) or (
            in_request
            and kind in ("image", "document")
            and any(_remote_source(source) for source in part.each("source"))
        )
        if addressed:
            return f"input:{written}"
    if any(_holds_address(value) for value in part.each("filedata")):
        return "input:file_data"
    if in_request and key == "content" and "type" not in part:
        for media in _UNTYPED_MEDIA:
            if any(_holds_address(value) for value in part.each(media)):
                return f"input:{media}"
    return None


def _remote_source(source: Any) -> bool:
    """Whether an Anthropic content source is given by address."""
    if not isinstance(source, _Object):
        return False
    kinds = {kind for _written, kind in source.kinds()}
    if "url" in kinds:
        return True
    inline = bool(kinds & {"text", "base64", "content"})
    return any(
        _is_address(value)
        for name, values in source.items()
        if name in _ADDRESS_FIELDS and not (name == "data" and inline)
        for value in values
        if isinstance(value, str)
    )


def _holds_address(value: Any) -> bool:
    """Whether ``value`` is an address, or a list or address object holding one.

    Only the address fields of an object are read (``image_url: {"url": ...}``),
    so a Notion or Slack block that shares a part's type name is not taken for one.
    """
    stack = [value]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            if _is_address(node):
                return True
        elif isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, _Object):
            stack.extend(
                field
                for name, values in node.items()
                if name in _ADDRESS_FIELDS
                for field in values
            )
    return False


_SCHEME = re.compile(r"([a-z][a-z0-9+.-]*):")
# Schemes a WHATWG URL parser reads with or without slashes (``https:host``).
_SPECIAL_SCHEMES = frozenset({"http", "https", "ftp", "ws", "wss", "file"})
_SCHEME_RELATIVE = re.compile(r"[\\/]{2}[^\\/\s]*\.")
# Files uploaded to the provider itself are referenced by an https URI too.
_PROVIDER_FILE_STORE = "https://generativelanguage.googleapis.com/"
# C0 controls and space, which a URL parser strips around an address.
_C0_AND_SPACE = "".join(map(chr, range(33)))
# Enough of what a URL parser keeps of a string to hold any scheme and the file store.
_HEAD = 256


def _is_address(text: str) -> bool:
    """Whether a URL parser would read ``text`` as an address to fetch.

    Only the start decides, so a long data: URL is not read through. A data:
    URL carries its bytes, and a one-letter scheme is a drive letter.
    """
    kept: list[str] = []
    for c in text:
        if c in URL_IGNORED or (not kept and (c <= " " or c.isspace())):
            continue
        kept.append(c)
        if len(kept) == _HEAD:
            break
    head = url_text("".join(kept)).strip().lstrip(_C0_AND_SPACE).lower()
    if head.startswith(_PROVIDER_FILE_STORE):
        return False
    scheme = _SCHEME.match(head)
    if scheme is None:
        return _SCHEME_RELATIVE.match(head) is not None
    name = scheme.group(1)
    if name == "data" or len(name) == 1:
        return False
    return name in _SPECIAL_SCHEMES or head[scheme.end() :].startswith(("/", "\\"))
