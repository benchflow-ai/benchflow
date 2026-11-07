"""Guards the report that a PDF Claude Code read never reached a bridged model.

LiteLLM 1.91.5's ``/v1/messages`` bridge kept only the text and image items
of a ``tool_result``: Claude Code's Read of ``/root/proposal.pdf``, the first
step of every task in one benchmark, reached the model as "PDF file read: ..."
and nothing else, a lone document as no tool message, and a document in a
user turn became an ``image_url`` part holding the PDF. LiteLLM 1.103.3 keeps
a ``tool_result``'s base64 document too, but as that same ``image_url`` part
(BerriAI/litellm#38251), which a chat-completions model does not read as a
PDF, and it still drops a text document. Each document now reaches the model
as its text: the PDF's pages as pypdf extracts them, a text document's own
text, or a note that says why there is none.

The callback module is loaded as the proxy loads it, from a runtime folder,
and LiteLLM's own translation is driven with synthetic requests and small
PDFs made here.
"""

from __future__ import annotations

import base64
import copy
import importlib.metadata
import importlib.util
import io
import json
import logging
import subprocess
import sys
import tempfile
from typing import Any

import pypdf
import pytest
from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)

from benchflow.providers.litellm_logging import callback_module_source

MODEL = "DeepSeek-V4.1-Flash"
EXTRACTOR = f"pypdf {pypdf.__version__}"
# A ToUnicode map that, as malformed maps in real PDFs do, sends A and B to
# the two halves of U+1D400 and so gives a lone surrogate for A alone.
SPLIT_SURROGATES = """/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CMapName /Split def
/CMapType 2 def
1 begincodespacerange
<00> <FF>
endcodespacerange
3 beginbfchar
<41> <D835>
<42> <DC00>
<43> <0043>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end"""


@pytest.fixture(autouse=True)
def _proxy_callback(tmp_path, monkeypatch):
    """Import callback_module_source() from a runtime folder, as the proxy does."""
    monkeypatch.setenv("BENCHFLOW_LITELLM_UPSTREAM_MODEL", f"azure/{MODEL}")
    path = tmp_path / "benchflow_litellm_callback.py"
    path.write_text(callback_module_source(), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("benchflow_litellm_callback", path)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def _pdf(pages: list[list[str]], to_unicode: str | None = None) -> bytes:
    """A PDF whose pages show these lines of Helvetica text (a page may have none)."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    def stream(data: bytes) -> bytes:
        return b"<< /Length %d >>\nstream\n%s\nendstream" % (len(data), data)

    unicode_map = ""
    if to_unicode is not None:
        unicode_map = f" /ToUnicode {add(stream(to_unicode.encode('latin-1')))} 0 R"
    font = add(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica"
        b" /Encoding /WinAnsiEncoding%s >>" % unicode_map.encode()
    )
    tree = add(b"")
    kids = []
    for lines in pages:
        ops = ["BT", "/F1 10 Tf", "12 TL", "40 780 Td"]
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({escaped}) Tj T*")
        ops.append("ET")
        content = add(stream("\n".join(ops).encode("latin-1")))
        kids.append(
            add(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] /Contents %d 0 R"
                b" /Resources << /Font << /F1 %d 0 R >> >> >>" % (tree, content, font)
            )
        )
    refs = " ".join(f"{kid} 0 R" for kid in kids)
    objects[tree - 1] = f"<< /Type /Pages /Kids [{refs}] /Count {len(kids)} >>".encode()
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R >>" % tree)
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n%s\nendobj\n" % (number, body))
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(
        b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objects) + 1, catalog, xref)
    )
    return out.getvalue()


def _encrypted(raw: bytes, user_password: str) -> bytes:
    writer = pypdf.PdfWriter(clone_from=pypdf.PdfReader(io.BytesIO(raw)))
    writer.encrypt(
        user_password=user_password, owner_password="owner", algorithm="AES-128"
    )
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _document(raw: bytes, **extra: Any) -> dict[str, Any]:
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(raw).decode(),
        },
        **extra,
    }


def _read(path: str, *items: dict[str, Any], **args: Any) -> list[dict[str, Any]]:
    """Claude Code's turn that reads ``path`` and gets ``items`` back."""
    return [
        {"role": "user", "content": [{"type": "text", "text": "Read the proposal."}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Read it.", "signature": "s"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Read",
                    "input": {"file_path": path, **args},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": list(items),
                }
            ],
        },
    ]


def _file_read(path: str, raw: bytes) -> dict[str, Any]:
    return {"type": "text", "text": f"PDF file read: {path} ({len(raw)} bytes)"}


def _pdf_part(raw: bytes) -> dict[str, Any]:
    """The part LiteLLM 1.103.3's bridge makes of a base64 PDF document."""
    url = f"data:application/pdf;base64,{base64.b64encode(raw).decode()}"
    return {"type": "image_url", "image_url": {"url": url}}


def _translate(messages: list[Any], model: str = MODEL) -> list[Any]:
    return LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
        messages=copy.deepcopy(messages), model=model
    )


def _stock_translate(messages: list[Any]) -> list[Any]:
    """LiteLLM's own translation, whatever wraps it now."""
    function = LiteLLMAnthropicMessagesAdapter.__dict__[
        "translate_anthropic_messages_to_openai"
    ]
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    return function(
        LiteLLMAnthropicMessagesAdapter(), messages=copy.deepcopy(messages), model=MODEL
    )


def _tool_content(built: list[Any]) -> Any:
    tools = [m["content"] for m in built if m.get("role") == "tool"]
    assert len(tools) == 1, f"one tool message expected, the bridge made {len(tools)}"
    return tools[0]


def _pdf_text(name: str, raw: bytes, pages: list[str]) -> str:
    noun = "page" if len(pages) == 1 else "pages"
    lines = [
        f"[PDF converted to text: this model cannot read PDF files, so the PDF {name} "
        f"({len(raw)} bytes, {len(pages)} {noun}) is given below as the text "
        f"{EXTRACTOR} extracted from it. Figures, images and page layout are not "
        "included.]"
    ]
    for number, page in enumerate(pages, 1):
        lines += [f"=== Page {number} of {len(pages)} ===", page]
    return "\n".join([*lines, f"[End of the text extracted from the PDF {name}]"])


def _note(name: str, size: int, why: str) -> str:
    return (
        "[PDF omitted: this model cannot read PDF files, and no text could be "
        f"extracted from the PDF {name} ({size} bytes): {why}]"
    )


def test_a_pdf_read_reaches_the_model_as_its_text():
    raw = _pdf([["Proposal: ground state", "energy (eV) per site"], ["Page two"], []])
    messages = _read(
        "/root/proposal.pdf", _file_read("/root/proposal.pdf", raw), _document(raw)
    )

    assert _tool_content(_stock_translate(messages)) == [
        _file_read("/root/proposal.pdf", raw),
        _pdf_part(raw),
    ]
    assert _tool_content(_translate(messages)) == [
        _file_read("/root/proposal.pdf", raw),
        {
            "type": "text",
            "text": _pdf_text(
                "/root/proposal.pdf",
                raw,
                [
                    "Proposal: ground state\nenergy (eV) per site",
                    "Page two",
                    "(no extractable text on this page)",
                ],
            ),
        },
    ]


def test_a_lone_document_result_becomes_a_tool_message_of_its_text():
    raw = _pdf([["Only the PDF came back."]])
    messages = _read("/root/a.pdf", _document(raw), pages="1-2")

    assert _tool_content(_stock_translate(messages)) == [_pdf_part(raw)]
    assert _tool_content(_translate(messages)) == _pdf_text(
        "/root/a.pdf (pages 1-2)", raw, ["Only the PDF came back."]
    )


def test_a_document_in_a_user_turn_is_text_not_an_image_part():
    raw = _pdf([["Attached by the user."]])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarise this."},
                _document(raw, title="brief.pdf"),
            ],
        }
    ]

    (stock,) = _stock_translate(messages)
    assert stock["content"][1]["type"] == "image_url"
    (built,) = _translate(messages)
    assert built["content"] == [
        {"type": "text", "text": "Summarise this."},
        {
            "type": "text",
            "text": _pdf_text("brief.pdf", raw, ["Attached by the user."]),
        },
    ]


def test_a_text_document_keeps_its_text_and_another_kind_becomes_a_note():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "title": "notes.txt",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": "k = 3",
                    },
                },
                {
                    "type": "document",
                    "source": {
                        "type": "content",
                        "content": [{"type": "text", "text": "a"}],
                    },
                },
            ],
        }
    ]

    (built,) = _translate(messages)

    assert built["content"] == [
        {"type": "text", "text": "[Text document notes.txt]\nk = 3"},
        {
            "type": "text",
            "text": "[document omitted: this model cannot read documents of this kind "
            "(content)]",
        },
    ]


def test_lone_surrogates_become_replacement_characters():
    """A request holding one fails LiteLLM's UTF-8 encoding ("surrogates not
    allowed") and so does every later request of the conversation."""
    raw = _pdf([["ABC AC"]], to_unicode=SPLIT_SURROGATES)
    assert "\ud835C" in pypdf.PdfReader(io.BytesIO(raw)).pages[0].extract_text()
    messages = _read("/root/odd.pdf", _document(raw))
    messages[-1]["content"].append(
        {
            "type": "document",
            "title": "t.txt",
            "source": {"type": "text", "media_type": "text/plain", "data": "x\udc00y"},
        }
    )

    built = _translate(messages)

    assert _tool_content(built) == _pdf_text("/root/odd.pdf", raw, ["\U0001d400C �C"])
    assert built[-1]["content"] == [
        {"type": "text", "text": "[Text document t.txt]\nx�y"}
    ]
    json.dumps(built, ensure_ascii=False).encode("utf-8")


def test_a_text_over_the_cap_is_cut_with_a_line_saying_how_far_it_was_read():
    pages = [[f"page {n} line {i} " + "x" * 60 for i in range(50)] for n in range(80)]
    raw = _pdf(pages)
    texts = [page.extract_text() for page in pypdf.PdfReader(io.BytesIO(raw)).pages]
    full = _pdf_text("/root/long.pdf", raw, [page.strip("\n") for page in texts])
    assert len(full) > 300_000
    # The reader stops once the pages it read hold more than the cap.
    read = total = 0
    while total <= 300_000:
        total += len(texts[read])
        read += 1
    assert read < len(texts)

    text = _tool_content(_translate(_read("/root/long.pdf", _document(raw))))

    assert text == "\n".join(
        [
            full[:300_000].rstrip("\n"),
            "[truncated: the text extracted from the PDF /root/long.pdf was read up "
            f"to page {read} of 80, past 300000 characters; only the first 300000 "
            "are given here]",
            "[End of the text extracted from the PDF /root/long.pdf]",
        ]
    )


def _bridge_globals() -> dict:
    """The bridge patch's module namespace, as the proxy loaded it."""
    return LiteLLMAnthropicMessagesAdapter.translate_anthropic_messages_to_openai.__globals__


@pytest.mark.parametrize(
    ("limit", "value", "why"),
    [
        ("PDF_MAX_BYTES", 100, "it is over the 100 bytes this proxy reads"),
        ("PDF_MAX_PAGES", 2, "it has 3 pages, over the 2 this proxy reads"),
        ("PDF_READ_SECONDS", 0.0001, "reading it took more than 0.0001 s"),
    ],
    ids=["bytes", "pages", "time"],
)
def test_a_pdf_over_a_reading_limit_becomes_a_note(monkeypatch, limit, value, why):
    """Guards the review of the PDF fix: pypdf read an agent-supplied
    PDF on the proxy's event loop with no limit, so a PDF built to make it
    work for hours or fill memory stalled every stream of the run."""
    monkeypatch.setitem(_bridge_globals(), limit, value)
    raw = _pdf([[f"Over the {limit} limit."]] * 3)

    text = _tool_content(_translate(_read(f"/root/{limit}.pdf", _document(raw))))

    assert text == _note(f"/root/{limit}.pdf", len(raw), why)


def test_one_page_with_more_text_than_the_cap_is_cut_by_the_reader(monkeypatch):
    """Guards the review of the reader's limits: the reader checked the cap
    only before each page, so one page's text, however long, went to the
    proxy whole, and the proxy held it several times over."""
    answers = []
    real_run = subprocess.run

    def recording_run(args, **kwargs):
        done = real_run(args, **kwargs)
        out = kwargs.get("stdout")
        if hasattr(out, "seek"):
            out.seek(0)
            answers.append(json.loads(out.read()))
        else:
            answers.append(json.loads(done.stdout))
        return done

    monkeypatch.setattr(subprocess, "run", recording_run)
    monkeypatch.setitem(_bridge_globals(), "PDF_TEXT_CAP", 1000)
    raw = _pdf([[f"line {i} " + "y" * 60 for i in range(60)], ["The second page."]])

    text = _tool_content(_translate(_read("/root/big.pdf", _document(raw))))

    (answer,) = answers
    assert answer.get("cut") is True
    assert len(answer["texts"]) == 1
    assert len(answer["texts"][0]) == 1001
    assert text.endswith(
        "\n[truncated: the text extracted from the PDF /root/big.pdf was read up to "
        "page 1 of 2, past 1000 characters; only the first 1000 are given here]\n"
        "[End of the text extracted from the PDF /root/big.pdf]"
    )


class _CountedFile:
    """A temporary file that counts the bytes read back from it."""

    def __init__(self, real):
        self._real, self.read_bytes = real, 0

    def read(self, *args):
        data = self._real.read(*args)
        self.read_bytes += len(data)
        return data

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)


def _count_temporary_files(monkeypatch) -> list[_CountedFile]:
    files: list[_CountedFile] = []
    real = tempfile.TemporaryFile

    def counted(*args, **kwargs):
        files.append(_CountedFile(real(*args, **kwargs)))
        return files[-1]

    monkeypatch.setattr(tempfile, "TemporaryFile", counted)
    return files


def test_the_proxy_reads_a_bounded_part_of_what_the_reader_writes(monkeypatch):
    """The proxy reads at most the reader's largest answer, and the end of its
    stderr: a reader that writes more costs a note, not the proxy's memory."""
    limits = _bridge_globals()
    bound = 6 * (limits["PDF_TEXT_CAP"] + 1) + 8 * limits["PDF_MAX_PAGES"] + 4096
    files = _count_temporary_files(monkeypatch)
    flood = (
        "import sys\n"
        "sys.stdout.buffer.write(b'{}' + b' ' * 16 * 1024 * 1024)\n"
        "sys.stderr.buffer.write(b'warning\\n' * 2_000_000)\n"
    )
    monkeypatch.setitem(_bridge_globals(), "_PDF_READER", flood)
    raw = _pdf([["Answered at length."]])

    text = _tool_content(_translate(_read("/root/flood.pdf", _document(raw))))

    why = "pypdf's reading of it is longer than this proxy takes"
    assert text == _note("/root/flood.pdf", len(raw), why)
    out, err = files
    assert out.read_bytes == bound + 1
    assert 0 < err.read_bytes <= 4096

    failing = "import sys\nsys.stderr.write('noise\\n' * 2_000_000 + 'ValueError: x')\n"
    monkeypatch.setitem(_bridge_globals(), "_PDF_READER", failing + "sys.exit(1)\n")
    raw = _pdf([["Failed at length."]])

    text = _tool_content(_translate(_read("/root/fails.pdf", _document(raw))))

    why = "pypdf could not read it (ValueError: x)"
    assert text == _note("/root/fails.pdf", len(raw), why)


ASTRAL = """/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CMapName /Astral def
/CMapType 2 def
1 begincodespacerange
<00> <FF>
endcodespacerange
1 beginbfchar
<41> <D835DC00>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end"""


def test_the_answer_bound_follows_the_limits_in_force(monkeypatch):
    """The reader's answer bound is reckoned from the cap and page limit of
    the moment, and holds a page of characters that take 4 bytes each in
    UTF-8 (and 12 as JSON escapes, which would not fit)."""
    runs = []
    real_run = subprocess.run

    def recording_run(args, **kwargs):
        runs.append(args)
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)
    monkeypatch.setitem(_bridge_globals(), "PDF_TEXT_CAP", 1000)
    monkeypatch.setitem(_bridge_globals(), "PDF_MAX_PAGES", 1)
    raw = _pdf([["A" * 60] * 20], to_unicode=ASTRAL)

    text = _tool_content(_translate(_read("/root/astral.pdf", _document(raw))))

    ((*_, bound),) = runs
    assert bound == str(6 * 1001 + 8 + 4096)
    assert text.count("\U0001d400") > 600
    assert "was read up to page 1 of 1, past 1000 characters" in text


def test_a_reader_that_could_not_start_is_tried_again_later(monkeypatch):
    """A note that the reader could not start (no temporary file, say) is not
    kept for the conversation: the next request reads the PDF."""
    real = tempfile.TemporaryFile
    calls = []

    def failing_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError(28, "dummy: no space left")
        return real(*args, **kwargs)

    monkeypatch.setattr(tempfile, "TemporaryFile", failing_once)
    raw = _pdf([["Read once the disk has room."]])
    messages = _read("/root/later.pdf", _document(raw))

    first = _tool_content(_translate(messages))
    later = _tool_content(_translate([*messages, {"role": "user", "content": "And?"}]))

    why = "the PDF reader could not start ([Errno 28] dummy: no space left)"
    assert first == _note("/root/later.pdf", len(raw), why)
    assert later == _pdf_text("/root/later.pdf", raw, ["Read once the disk has room."])


def test_the_pdf_reader_runs_apart_with_no_secret_in_its_environment(monkeypatch):
    runs = []
    real_run = subprocess.run

    def recording_run(args, **kwargs):
        runs.append((args, kwargs))
        return real_run(args, **kwargs)

    monkeypatch.setenv("AZURE_API_KEY", "dummy-secret-not-for-the-reader")
    monkeypatch.setattr(subprocess, "run", recording_run)
    raw = _pdf([["Read in a child process."]])

    text = _tool_content(_translate(_read("/root/apart.pdf", _document(raw))))

    assert "Read in a child process." in text
    ((args, kwargs),) = runs
    assert args[:3] == [sys.executable, "-I", "-c"]
    assert set(kwargs["env"]) <= {"PATH", "LD_LIBRARY_PATH"}
    assert kwargs["timeout"] == 30


@pytest.mark.parametrize(
    ("data", "why"),
    [
        (
            base64.b64encode(_encrypted(_pdf([["Secret."]]), "user")).decode(),
            "it is encrypted with a password",
        ),
        ("QUJD=", "its base64 data is not valid"),
        (base64.b64encode(_pdf([[], []])).decode(), None),
        (base64.b64encode(b"not a pdf at all").decode(), "pypdf could not read it"),
    ],
    ids=["user-password", "invalid-base64", "no-text", "not-a-pdf"],
)
def test_a_pdf_without_readable_text_becomes_a_note_that_says_why(data, why):
    document = {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": data},
    }
    text = _tool_content(_translate(_read("/root/x.pdf", document)))

    size = len(base64.b64decode(data)) if why != "its base64 data is not valid" else 3
    expected = why or "none of its 2 pages holds text (scanned pages or pictures only)"
    if why == "pypdf could not read it":
        assert text.startswith(_note("/root/x.pdf", size, why)[:-1] + " (")
    else:
        assert text == _note("/root/x.pdf", size, expected)


def test_an_owner_password_alone_does_not_hide_the_text():
    raw = _encrypted(_pdf([["Owner-only protection."]]), "")

    text = _tool_content(_translate(_read("/root/o.pdf", _document(raw))))

    assert text == _pdf_text("/root/o.pdf", raw, ["Owner-only protection."])


def test_without_pypdf_a_pdf_becomes_a_note_that_says_so(monkeypatch):
    raw = _pdf([["No reader in this proxy."]])
    monkeypatch.setitem(sys.modules, "pypdf", None)

    text = _tool_content(_translate(_read("/root/n.pdf", _document(raw))))

    assert text == _note(
        "/root/n.pdf", len(raw), "this proxy has no PDF reader (pypdf)"
    )


def test_each_payload_is_converted_once_and_reads_the_same_later(monkeypatch):
    opened = []
    real_run = subprocess.run

    def counting_run(args, **kwargs):
        opened.append(1)
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", counting_run)
    raw = _pdf([["Read again in every later request."]])
    messages = _read("/root/again.pdf", _document(raw))

    first = _translate(messages)
    later = _translate([*messages, {"role": "user", "content": "And now?"}])

    assert len(opened) == 1
    assert _tool_content(first) == _tool_content(later)
    assert "Read again in every later request." in _tool_content(first)


def test_the_request_claude_code_sent_keeps_its_document():
    raw = _pdf([["Kept in the agent's request."]])
    messages = _read("/root/k.pdf", _file_read("/root/k.pdf", raw), _document(raw))
    sent = copy.deepcopy(messages)

    LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
        messages=messages, model=MODEL
    )

    assert messages == sent


def test_a_converted_block_keeps_its_cache_breakpoint():
    raw = _pdf([["Cached."]])
    breakpoint_ = {"type": "ephemeral"}
    messages = [
        {
            "role": "user",
            "content": [_document(raw, title="c.pdf", cache_control=breakpoint_)],
        }
    ]

    # The bridge keeps cache_control for Claude ids only.
    (built,) = _translate(messages, model="claude-fable-5-1")

    assert built["content"] == [
        {
            "type": "text",
            "text": _pdf_text("c.pdf", raw, ["Cached."]),
            "cache_control": breakpoint_,
        }
    ]


def test_the_handler_sends_the_text_upstream():
    """The bridged request as the proxy builds it for litellm.acompletion."""
    raw = _pdf([["Through the handler."]])
    kwargs, _ = (
        LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
            max_tokens=1024,
            messages=_read(
                "/root/h.pdf", _file_read("/root/h.pdf", raw), _document(raw)
            ),
            model=MODEL,
            stream=True,
            extra_kwargs={"custom_llm_provider": "azure"},
        )
    )

    assert _tool_content(kwargs["messages"]) == [
        _file_read("/root/h.pdf", raw),
        {
            "type": "text",
            "text": _pdf_text("/root/h.pdf", raw, ["Through the handler."]),
        },
    ]


def test_a_request_without_documents_is_the_bridges_own():
    messages = _read("/root/t.txt", {"type": "text", "text": "plain file"})
    messages.append({"role": "user", "content": "go on"})

    assert _translate(messages) == _stock_translate(messages)


_TRANSLATE = "translate_anthropic_messages_to_openai"


def _stock_function() -> Any:
    function = LiteLLMAnthropicMessagesAdapter.__dict__[_TRANSLATE]
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    return function


def test_the_patch_passes_the_bridges_keyword_arguments_on():
    """LiteLLM 1.103.3's bridge passes custom_llm_provider and
    preserve_midturn_system by keyword; a wrapper that took 1.91.5's
    arguments only failed every bridged request."""
    calls = []

    def bridge(self, messages, model=None, **kwargs):
        calls.append((messages, model, kwargs))
        return []

    patched = _bridge_globals()["_translated"](bridge)
    patched(
        LiteLLMAnthropicMessagesAdapter(),
        [],
        MODEL,
        custom_llm_provider="azure",
        preserve_midturn_system=True,
    )

    assert calls == [
        ([], MODEL, {"custom_llm_provider": "azure", "preserve_midturn_system": True})
    ]


def test_only_the_checked_litellm_is_patched_and_only_once(monkeypatch):
    install = _bridge_globals()["install_messages_bridge_patch"]
    stock = _stock_function()
    monkeypatch.setattr(LiteLLMAnthropicMessagesAdapter, _TRANSLATE, stock)
    version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "1.91.5" if name == "litellm" else version(name),
    )

    assert install() is False
    assert LiteLLMAnthropicMessagesAdapter.__dict__[_TRANSLATE] is stock

    monkeypatch.setattr(importlib.metadata, "version", version)
    assert install() is True
    patched = LiteLLMAnthropicMessagesAdapter.__dict__[_TRANSLATE]
    assert patched.__wrapped__ is stock
    assert install() is True
    assert LiteLLMAnthropicMessagesAdapter.__dict__[_TRANSLATE] is patched


def test_the_callback_module_loads_when_the_patch_cannot_install(
    tmp_path, monkeypatch, caplog
):
    """The proxy must start: on a bridge whose shape changed, the patch says
    it is not installed and the request goes through LiteLLM's own bridge."""
    monkeypatch.delattr(LiteLLMAnthropicMessagesAdapter, _TRANSLATE)
    path = tmp_path / "changed" / "benchflow_litellm_callback.py"
    path.parent.mkdir()
    path.write_text(callback_module_source(), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("benchflow_litellm_callback", path)
    assert spec is not None and spec.loader is not None
    with caplog.at_level(logging.WARNING, logger="benchflow.litellm_messages_bridge"):
        spec.loader.exec_module(importlib.util.module_from_spec(spec))

    assert "LiteLLM messages bridge patch not installed" in caplog.text
    assert _TRANSLATE not in LiteLLMAnthropicMessagesAdapter.__dict__
