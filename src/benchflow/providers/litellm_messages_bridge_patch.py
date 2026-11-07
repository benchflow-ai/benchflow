"""LiteLLM startup patch: Claude Code's requests through the chat-completions bridge.

When an Anthropic Messages client (Claude Code) talks to a chat-completions
model through the proxy (Azure DeepSeek, a vLLM server, ...), LiteLLM 1.103.3
translates the request in
``litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py``.
This module corrects what that translation still gets wrong:

- A PDF never reaches the model as text. The bridge makes each base64
  ``document`` block, in a ``tool_result`` (BerriAI/litellm#38251) or a user
  turn, an ``image_url`` part holding ``data:application/pdf;base64,...``,
  which a chat-completions model does not read as a PDF (Azure OpenAI answers
  400), and it drops a text document and every other kind. Claude Code's Read
  of a PDF is ``[text "PDF file read: ...", document {base64 PDF}]``. Before
  the bridge reads the request, each ``document`` block becomes one ``text``
  block (``_documents_as_text``, copy on write: the request the agent sent
  keeps the document): a PDF's text as pypdf extracts it, page by page, a
  text document's own text, a note for anything else.
- A model that reads text only gets an image part for a tool result that
  holds one image. The bridge gives every model such an image as an
  ``image_url`` part (BerriAI/litellm#34462), and an image part makes a model
  that reads text only refuse that request and every later one (Azure: 400
  "Model only supports text input"). For such a model (``_accepts_images``)
  the tool message holds the image's ``data:image/...;base64,...`` URI as its
  text instead, as LiteLLM 1.91.5's bridge sent it: the model cannot see the
  image either way, and the text costs about one token per 1.44 characters,
  but the request is answered.

LiteLLM 1.103.3 itself fixes the two other faults this module corrected on
1.91.5: a streamed answer opens with its own first block (its thinking or
first tool call) rather than an empty text block, which made Claude Code
count the thinking twice and compact far too early
(BerriAI/litellm#34433), and a model that reads images gets a lone
tool-result image as an image part, not as base64 text
(BerriAI/litellm#34462).

``litellm_logging.callback_module_source`` appends this file to the proxy's
callback module, so every BenchFlow LiteLLM proxy (host and sandbox) installs
it when LiteLLM loads its callback, before the first request. It imports only
the standard library, LiteLLM and pypdf (a dependency of the proxy's install)
and patches only the LiteLLM it was checked with. It has no ``__future__``
import: the callback module's own opens the file, and Python takes one only
there.
"""

import base64
import binascii
import functools
import hashlib
import importlib.metadata
import importlib.resources
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from collections import OrderedDict
from typing import Any

BRIDGE_LITELLM_VERSION = "1.103.3"
# Set on each LiteLLM function this module replaces, so it is replaced once
# however many times the callback module is loaded, and a layer above can see
# the bridge is fixed.
BRIDGE_PATCH_MARK = "__benchflow_messages_bridge_patch__"
# The models BenchFlow's provider registry (agents/providers.py) lists with
# text input only. This file runs without BenchFlow, so they are copied here;
# tests/test_litellm_bridge_tool_images.py keeps the two equal.
REGISTRY_TEXT_ONLY_MODELS = frozenset({"glm-5", "glm-5.1", "glm-5.3", "glm-5.3-flash"})
# The run's route as config.yaml names it (provider/model); the bridge sees
# the model without its provider.
_ROUTE_MODEL_ENV = "BENCHFLOW_LITELLM_UPSTREAM_MODEL"
# LiteLLM's model map as the pinned wheel ships it. litellm.model_cost may be
# a copy fetched at start, so it answers only for the run's own deployment.
_PINNED_MODEL_MAP = "model_prices_and_context_window_backup.json"
# The most characters a PDF's text block holds (about 75k tokens); a longer
# text is cut there with a line that says so, so one long PDF cannot fill the
# model's whole window at once.
PDF_TEXT_CAP = 300_000
# A PDF comes from the agent, so pypdf reads it in a child process: at most
# PDF_MAX_BYTES and PDF_MAX_PAGES, with PDF_READ_SECONDS of CPU and of wall
# time, _PDF_READ_MEMORY of address space and no secret in its environment.
# A PDF built to make pypdf work for hours or fill memory (thousands of pages
# reusing one inflating stream) then costs a note, not the proxy. The bridge
# runs on the proxy's event loop, which waits up to PDF_READ_SECONDS for each
# PDF it has not read before. macOS does not enforce the address-space
# limit, so a host proxy there bounds the child by its time alone.
PDF_MAX_BYTES = 32 * 1024 * 1024
PDF_MAX_PAGES = 2000
PDF_READ_SECONDS = 30
_PDF_READ_MEMORY = 2 * 1024**3
# How much of the child's stderr the proxy reads: the end of a traceback.
_PDF_ERROR_TAIL_BYTES = 4096
# The child: limits first (no core file when the CPU limit stops it), then
# pypdf. It replaces each page's lone UTF-16 surrogates, as _clean_text does,
# and cuts the pages' text so that together it passes the cap by one
# character at most: its answer stays bounded however much text one page
# holds. It answers with one JSON object.
_PDF_READER = r"""
import io, json, logging, resource, sys, warnings
cap, max_pages, seconds, memory, answer = (int(value) for value in sys.argv[1:6])
for limit, value in (
    (resource.RLIMIT_CPU, seconds),
    (resource.RLIMIT_AS, memory),
    (resource.RLIMIT_FSIZE, answer),
    (resource.RLIMIT_CORE, 0),
):
    try:
        resource.setrlimit(limit, (value, value))
    except (ValueError, OSError):
        pass
logging.disable(logging.CRITICAL)
warnings.simplefilter("ignore")
import pypdf
def say(read):
    sys.stdout.buffer.write(json.dumps(read, ensure_ascii=False).encode("utf-8"))
reader = pypdf.PdfReader(io.BytesIO(sys.stdin.buffer.read()), strict=False)
if reader.is_encrypted and not reader.decrypt(""):
    say({"encrypted": True})
    raise SystemExit
count = len(reader.pages)
if count > max_pages:
    say({"pages": count, "too_many": True})
    raise SystemExit
texts, total, cut = [], 0, False
for page in reader.pages:
    if total > cap:
        cut = True
        break
    text = page.extract_text() or ""
    room = cap + 1 - total
    if len(text) > room:
        # Cut before the copies the replacement makes; one more character
        # keeps a surrogate pair at the edge whole.
        text, cut = text[: room + 1], True
    text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")[:room]
    texts.append(text)
    total += len(text)
say({"pages": count, "texts": texts, "cut": cut, "version": pypdf.__version__})
"""
# Converted documents kept for the requests that send them again (a
# conversation sends every earlier turn each time), by total characters.
_DOCUMENT_CACHE_CHARS = 32_000_000

_bridge_logger = logging.getLogger("benchflow.litellm_messages_bridge")


def _marked(function: Any, original: Any) -> Any:
    setattr(function, BRIDGE_PATCH_MARK, True)
    function.__wrapped__ = original
    return function


@functools.cache
def _pinned_vision() -> dict[str, bool]:
    """``supports_vision`` of each model the pinned map states it for, by lower-case key."""
    found: dict[str, bool] = {}
    try:
        text = importlib.resources.files("litellm").joinpath(_PINNED_MODEL_MAP)
        for key, info in json.loads(text.read_text(encoding="utf-8")).items():
            vision = info.get("supports_vision") if isinstance(info, dict) else None
            if isinstance(vision, bool):
                found.setdefault(key.lower(), vision)
    except Exception:
        _bridge_logger.warning("LiteLLM's model map could not be read", exc_info=True)
    return found


def _accepts_images(model: Any) -> bool:
    """Whether the request's model reads image parts.

    The first source that knows decides: the run's deployment as LiteLLM
    registered it from config.yaml (its ``model_info``), BenchFlow's provider
    registry, then LiteLLM's pinned model map, under the route's
    provider/model, the model's own id or its last path segment, ignoring
    case (Azure's ``DeepSeek-V4-Pro`` is the map's text-only
    ``deepseek-v4-pro``). A model none of them knows keeps the bridge's image
    part, as an image beside text does. The bridge's model is None in the
    context-management polyfill's own calls: those are the route's.
    """
    route = os.environ.get(_ROUTE_MODEL_ENV, "")
    names = []
    if isinstance(model, str) and model:
        if route and route.split("/", 1)[-1] == model:
            names.append(route)
        names.append(model)
    elif route:
        names.append(route)
    if names and names[0] == route:
        import litellm

        deployment = litellm.model_cost.get(route)
        vision = (
            deployment.get("supports_vision") if isinstance(deployment, dict) else None
        )
        if isinstance(vision, bool):
            return vision
    names += [name.rsplit("/", 1)[-1] for name in names]
    keys = list(dict.fromkeys(name.lower() for name in names))
    if any(key in REGISTRY_TEXT_ONLY_MODELS for key in keys):
        return False
    pinned = _pinned_vision()
    for key in keys:
        if key in pinned:
            return pinned[key]
    return True


def _lone_image_urls(adapter: Any, messages: Any) -> dict[str, str]:
    """The URL the bridge makes of each tool_result holding one image, by tool_use_id."""
    urls: dict[str, str] = {}
    for message in messages if isinstance(messages, list) else ():
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        for block in content if isinstance(content, list) else ():
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            inner = block.get("content")
            if (
                isinstance(inner, list)
                and len(inner) == 1
                and isinstance(inner[0], dict)
                and inner[0].get("type") == "image"
            ):
                # The bridge's own translation of the image, and its default id.
                url = adapter._translate_anthropic_image_to_openai(
                    inner[0].get("source", {})
                )
                if url:
                    urls[block.get("tool_use_id", "")] = url
    return urls


class _DocumentTexts:
    """Converted documents by payload, least recently used dropped past ``limit`` characters.

    Each payload is converted once, so a PDF reads the same in every later
    request of the conversation and the provider's prompt cache holds.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.chars = 0
        self.texts: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: tuple[str, str, str]) -> str | None:
        with self.lock:
            text = self.texts.get(key)
            if text is not None:
                self.texts.move_to_end(key)
            return text

    def put(self, key: tuple[str, str, str], text: str) -> str:
        with self.lock:
            if key in self.texts:
                return self.texts[key]
            self.texts[key] = text
            self.chars += len(text)
            while self.chars > self.limit and len(self.texts) > 1:
                self.chars -= len(self.texts.popitem(last=False)[1])
            return text


_document_texts = _DocumentTexts(_DOCUMENT_CACHE_CHARS)


def _clean_text(text: str) -> str:
    """``text`` with each lone UTF-16 surrogate as U+FFFD, a surrogate pair as its character.

    pypdf reads a malformed ToUnicode map as lone surrogates (U+D835 without
    its pair); LiteLLM encodes the request as UTF-8, and one such character
    fails that request and every later one ("surrogates not allowed").
    """
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def _pdf_note(name: str, size: int, why: str) -> str:
    named = f" {name}" if name else ""
    return (
        "[PDF omitted: this model cannot read PDF files, and no text could be "
        f"extracted from the PDF{named} ({size} bytes): {why}]"
    )


class _Uncached(str):
    """A note for this request only: a later one may read the PDF (its
    reader could not start, say for want of a temporary file)."""


def _pdf_answer_bytes() -> int:
    """The most bytes of the child's answer: at most PDF_TEXT_CAP + 1
    characters of text, each at most 6 bytes in its JSON (a control
    character's escape), plus the page list."""
    return 6 * (PDF_TEXT_CAP + 1) + 8 * PDF_MAX_PAGES + 4096


def _read_pdf(payload: bytes) -> dict[str, Any] | str:
    """pypdf's reading of *payload* in a child process (:data:`_PDF_READER`),
    or why there is none.

    The child writes to files, of which the proxy reads back at most
    :func:`_pdf_answer_bytes` of the answer (nor can the child write more:
    RLIMIT_FSIZE) and the last :data:`_PDF_ERROR_TAIL_BYTES` of stderr, never
    a pipe it would drain whole.
    """
    bound = _pdf_answer_bytes()
    environment = {
        key: os.environ[key] for key in ("PATH", "LD_LIBRARY_PATH") if key in os.environ
    }
    try:
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            try:
                done = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-c",
                        _PDF_READER,
                        str(PDF_TEXT_CAP),
                        str(PDF_MAX_PAGES),
                        str(PDF_READ_SECONDS),
                        str(_PDF_READ_MEMORY),
                        str(bound),
                    ],
                    input=payload,
                    stdout=out,
                    stderr=err,
                    timeout=PDF_READ_SECONDS,
                    env=environment,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return f"reading it took more than {PDF_READ_SECONDS} s"
            out.seek(0)
            answer = out.read(bound + 1)
            err.seek(max(0, err.seek(0, os.SEEK_END) - _PDF_ERROR_TAIL_BYTES))
            errors = err.read()
    except OSError as exc:
        return _Uncached(f"the PDF reader could not start ({exc})")
    if done.returncode != 0:
        lines = errors.decode("utf-8", "replace").strip().splitlines()
        if any("No module named 'pypdf'" in line for line in lines):
            return "this proxy has no PDF reader (pypdf)"
        last = lines[-1] if lines else f"it stopped with status {done.returncode}"
        return f"pypdf could not read it ({last})"[:300]
    if len(answer) > bound:
        return "pypdf's reading of it is longer than this proxy takes"
    try:
        read = json.loads(answer)
    except ValueError:
        return "pypdf could not read it (its reader gave no answer)"
    return read if isinstance(read, dict) else "pypdf could not read it"


def _pdf_text(data: str, name: str) -> str:
    """The text block that stands for a base64 PDF: a header, its pages, an end line."""
    compact = "".join(data.split())
    try:
        payload = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError):
        return _pdf_note(
            name, len(compact.rstrip("=")) * 3 // 4, "its base64 data is not valid"
        )
    if len(payload) > PDF_MAX_BYTES:
        return _pdf_note(
            name, len(payload), f"it is over the {PDF_MAX_BYTES} bytes this proxy reads"
        )
    try:
        import pypdf  # noqa: F401  (the child reads it; a missing one is said plainly)
    except ImportError:
        return _pdf_note(name, len(payload), "this proxy has no PDF reader (pypdf)")
    read = _read_pdf(payload)
    if isinstance(read, str):
        note = _pdf_note(name, len(payload), read)
        return _Uncached(note) if isinstance(read, _Uncached) else note
    if read.get("encrypted"):
        return _pdf_note(name, len(payload), "it is encrypted with a password")
    count = read.get("pages")
    texts = read.get("texts")
    if read.get("too_many"):
        return _pdf_note(
            name,
            len(payload),
            f"it has {count} pages, over the {PDF_MAX_PAGES} this proxy reads",
        )
    if not isinstance(count, int) or not isinstance(texts, list):
        return _pdf_note(name, len(payload), "pypdf could not read it")
    pages = [_clean_text(str(page)) for page in texts]
    cut = read.get("cut") is True or len(pages) < count
    if len(pages) == count and not any(page.strip() for page in pages):
        return _pdf_note(
            name,
            len(payload),
            f"none of its {count} pages holds text (scanned pages or pictures only)",
        )
    named = f" {name}" if name else ""
    noun = "page" if count == 1 else "pages"
    lines = [
        f"[PDF converted to text: this model cannot read PDF files, so the PDF{named} "
        f"({len(payload)} bytes, {count} {noun}) is given below as the text "
        f"pypdf {read.get('version')} extracted from it. Figures, images and page "
        "layout are not included.]"
    ]
    for number, page in enumerate(pages, 1):
        lines.append(f"=== Page {number} of {count} ===")
        body = page.strip("\n")
        lines.append(body if body.strip() else "(no extractable text on this page)")
    end = f"[End of the text extracted from the PDF{named}]"
    text = "\n".join([*lines, end])
    if len(text) <= PDF_TEXT_CAP and not cut:
        return text
    length = (
        f"was read up to page {len(pages)} of {count}, past {PDF_TEXT_CAP} characters"
        if cut
        else f"is {len(text)} characters"
    )
    return "\n".join(
        (
            text[:PDF_TEXT_CAP].rstrip("\n"),
            f"[truncated: the text extracted from the PDF{named} {length}; only "
            f"the first {PDF_TEXT_CAP} are given here]",
            end,
        )
    )


def _document_text(block: dict[str, Any], name: str) -> str:
    """The text that stands for one ``document`` block (once per payload)."""
    source = block.get("source")
    source = source if isinstance(source, dict) else {}
    kind, data = source.get("type"), source.get("data")
    pdf = kind == "base64" and source.get("media_type") == "application/pdf"
    if not (pdf or kind == "text") or not isinstance(data, str):
        what = " ".join(str(part) for part in (kind, source.get("media_type")) if part)
        named = f"; {name}" if name else ""
        return (
            "[document omitted: this model cannot read documents of this kind "
            f"({what or 'unknown source'}){named}]"
        )
    digest = hashlib.sha256(data.encode("utf-8", "surrogatepass")).hexdigest()
    key = ("pdf" if pdf else "text", digest, name)
    cached = _document_texts.get(key)
    if cached is not None:
        return cached
    if pdf:
        text = _pdf_text(data, name)
        if isinstance(text, _Uncached):
            return str(text)
    else:
        text = f"[Text document {name or '(unnamed)'}]\n{_clean_text(data)}"
    return _document_texts.put(key, text)


def _document_name(block: dict[str, Any], call_input: Any) -> str:
    """The file a document came from: its Read's ``file_path`` (and pages), else its title."""
    args = call_input if isinstance(call_input, dict) else {}
    path = args.get("file_path") or args.get("path")
    if isinstance(path, str) and path:
        pages = args.get("pages")
        return (
            f"{path} (pages {pages})"
            if isinstance(pages, str | int) and pages
            else path
        )
    title = block.get("title")
    return title if isinstance(title, str) else ""


def _blocks_as_text(
    blocks: list[Any], call_input: Any, calls: dict[str, Any]
) -> list[Any]:
    """``blocks`` with each document (also inside a tool_result) as text; the same list when none."""
    out = None
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result" and isinstance(
            block.get("content"), list
        ):
            tool_id = block.get("tool_use_id")
            inner = _blocks_as_text(
                block["content"],
                calls.get(tool_id) if isinstance(tool_id, str) else None,
                calls,
            )
            if inner is block["content"]:
                continue
            new = {**block, "content": inner}
        elif block.get("type") == "document":
            new = {
                "type": "text",
                "text": _document_text(block, _document_name(block, call_input)),
            }
            if "cache_control" in block:
                new["cache_control"] = block["cache_control"]
        else:
            continue
        if out is None:
            out = list(blocks)
        out[index] = new
    return blocks if out is None else out


def _turn_as_text(turn: Any, calls: dict[str, Any]) -> Any:
    """A user turn with its documents as text; the same object when it has none."""
    if not isinstance(turn, dict) or turn.get("role") != "user":
        return turn
    content = turn.get("content")
    if not isinstance(content, list):
        return turn
    new = _blocks_as_text(content, None, calls)
    return turn if new is content else {**turn, "content": new}


def _documents_as_text(messages: Any) -> Any:
    """``messages`` with every document of a user turn as one text block; copy on write."""
    if not isinstance(messages, list):
        return messages
    calls: dict[str, Any] = {}
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        for block in content if isinstance(content, list) else ():
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and isinstance(block.get("id"), str)
            ):
                calls[block["id"]] = block.get("input")
    turns = [_turn_as_text(turn, calls) for turn in messages]
    if all(new is old for new, old in zip(turns, messages, strict=True)):
        return messages
    return turns


def _translated(original: Any) -> Any:
    # The bridge passes custom_llm_provider and preserve_midturn_system by
    # keyword; they go through unchanged.
    def translate_anthropic_messages_to_openai(
        self: Any, messages: Any, model: Any = None, **kwargs: Any
    ) -> Any:
        try:
            messages = _documents_as_text(messages)
        except Exception:
            _bridge_logger.warning("documents left as they came", exc_info=True)
        built = original(self, messages, model, **kwargs)
        try:
            urls = _lone_image_urls(self, messages)
            if urls and not _accepts_images(model):
                for message in built:
                    if (
                        isinstance(message, dict)
                        and message.get("role") == "tool"
                        and message.get("tool_call_id") in urls
                    ):
                        url = urls[message["tool_call_id"]]
                        if message.get("content") == [
                            {"type": "image_url", "image_url": {"url": url}}
                        ]:
                            message["content"] = url
        except Exception:
            _bridge_logger.warning(
                "tool result images left as the bridge made them", exc_info=True
            )
        return built

    return _marked(translate_anthropic_messages_to_openai, original)


def _litellm_checked() -> bool:
    try:
        return importlib.metadata.version("litellm") == BRIDGE_LITELLM_VERSION
    except importlib.metadata.PackageNotFoundError:
        return False


def install_messages_bridge_patch() -> bool:
    """Patch LiteLLM's bridge once; False when this is not the LiteLLM it was checked with."""
    if not _litellm_checked():
        return False
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    name = "translate_anthropic_messages_to_openai"
    original = LiteLLMAnthropicMessagesAdapter.__dict__[name]
    if not getattr(original, BRIDGE_PATCH_MARK, False):
        setattr(LiteLLMAnthropicMessagesAdapter, name, _translated(original))
    return True


try:
    install_messages_bridge_patch()
except Exception:
    # The proxy must start; the request then goes through LiteLLM's own bridge.
    _bridge_logger.warning("LiteLLM messages bridge patch not installed", exc_info=True)
