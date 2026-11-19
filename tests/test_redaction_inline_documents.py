"""A service account's JSON pasted under GOOGLE_APPLICATION_CREDENTIALS.

A follow-up to part B (the lead's ruling, 2026-11-14). Every value
here is made up.

In running text the redactor follows the document from the `{` after the
name to the `}` that closes it, across its lines (a pretty-printed
document), reading strings as JSON does: a brace or an escaped quote inside
a string does not count. It masks all of it, up to 64 KB (_JSON_CAP). A
document not closed, nested deeper than three levels (a service account's
is flat) or longer than the cap is masked to its first line's end and then
line by line while each next line keeps a JSON key's shape, after its key,
to that line's end; from the first line that does not, it is text again
(the lead's ruling, 2026-11-14). A line of other text after the document is
kept. Inside a JSON string (a serialized record) the document is read the
same way, as the copy spells it (its lines end at `\\n`), and never past
the string's end.
"""

from __future__ import annotations

import json

import pytest

from benchflow._utils.config_redaction import _should_record_env_entry
from benchflow.publish.redact import redact_value
from benchflow.trajectories import types
from benchflow.trajectories.types import (
    redact_acp_trajectory_jsonl,
    redact_trajectory_obj,
    redact_trajectory_text,
)
from tests._timing import assert_linear, linear_seconds, within_ceiling
from tests.test_redaction_ansi_spellings import _repeat
from tests.test_redaction_escape_boundary import _leaks as _boundary_leaks
from tests.test_redaction_shape_matrix import _leaks as _path_leaks

MASK = "***REDACTED***"
EXPORT_MASK = "<XXX-benchflow-key-values-XXX>"
GAC = "GOOGLE_APPLICATION_CREDENTIALS"
PEM = (
    "--"
    + "---BEGIN"
    + " PRIVATE"
    + " KEY----"
    + "-\nFAKEfakeFAKEfake0123\n-----END PRIVATE KEY-----\n"
)
ACCOUNT = {
    "type": "service_account",
    "project_id": "fake-project",
    "private_key_id": "fakekeyid0123456789",
    "private_key": PEM,
    "client_email": "fake-sa@fake-project.iam.gserviceaccount.com",
    "client_id": "000000000000000000001",
    "token_uri": "https://oauth2.example.com/token",
    "note": 'a } and a { and an escaped "quote" in a string',
}
# The parts of the document that must not survive.
PARTS = [
    "fakekeyid0123456789",
    "FAKEfakeFAKEfake0123",
    "fake-sa@",
    "000000000000000000001",
    "an escaped",
]
PRETTY = json.dumps(ACCOUNT, indent=2)
ONE_LINE = json.dumps(ACCOUNT)
NEXT = "NEXT_SETTING=plain-value"


def _leaked(text: str, secret: str) -> set[str]:
    return {*_boundary_leaks(text, secret), *_path_leaks(text, secret)}


def _text_forms(document: str) -> dict[str, str]:
    return {
        "env": f"{GAC}={document}",
        "export": f"export {GAC}={document}",
        "yaml": f"{GAC}: {document}",
        "after an escaped newline": f"out\\n{GAC}={document}",
        "after a colour code": f"out\x1b[32m{GAC}={document}",
        "after a JSON copy's colour code": f"out\\u001b[32m{GAC}={document}",
    }


@pytest.mark.parametrize("document", [PRETTY, ONE_LINE], ids=["pretty", "one line"])
def test_a_closed_document_is_masked_whole_in_every_layer(document):
    """The lead's ruling: closed, it is masked from its `{` to its `}` in
    text and its escaped and colour forms, in a JSON string, in trajectory
    files, at export and in config.json; a brace or an escaped quote inside
    a string does not end it."""
    for form, text in _text_forms(document).items():
        for part in PARTS:
            assert _leaked(text, part) == set(), (form, part)
        assert redact_trajectory_text(text).endswith(f"={MASK}") or text.startswith(
            f"{GAC}: "
        ), form
    for record in (
        json.dumps({"message": f"{GAC}={document}", "n": 1}),
        json.dumps({"env": {GAC: document}}),
    ):
        out = redact_trajectory_text(record)
        json.loads(out)
        assert not any(part in out for part in PARTS), record
    assert redact_trajectory_obj({GAC: document}) == {GAC: MASK}
    assert redact_trajectory_obj({GAC: ACCOUNT}) == {GAC: MASK}
    assert redact_value({GAC: document})[0] == {GAC: EXPORT_MASK}
    assert not _should_record_env_entry(GAC, document)


@pytest.mark.parametrize("document", [PRETTY, ONE_LINE], ids=["pretty", "one line"])
def test_a_line_of_other_text_after_the_document_is_kept(document):
    """The lead's ruling: the masked document ends at its `}`; the next
    line is read as text of its own."""
    for form, text in _text_forms(document).items():
        out = redact_trajectory_text(f"{text}\n{NEXT}")
        assert out.endswith(f"{MASK}\n{NEXT}"), form
    record = json.dumps({"message": f"{GAC}={document}\n{NEXT}"})
    assert NEXT in json.loads(redact_trajectory_text(record))["message"]


OPEN = PRETTY[: PRETTY.rindex("}")]
DEEP = {
    "type": "service_account",
    "a": {"b": {"c": {"d": {"e": "deepFAKE0123456"}}}},
    "private_key_id": "fakekeyid0123456789",
    "client_email": "fake-sa@fake-project.iam.gserviceaccount.com",
}
DEEP_PARTS = ["deepFAKE0123456", "fakekeyid0123456789", "fake-sa@"]
# A deep document pretty-printed: its keys before the first line of closing
# braces, then that line, then a key after it.
DEEP_PRETTY = (
    "{\n"
    '  "private_key_id": "fakekeyid0123456789",\n'
    '  "client_email": "fake-sa@fake-project.iam.gserviceaccount.com",\n'
    '  "a": {"b": {"c": {"d": {\n'
    '    "e": "deepFAKE0123456"\n'
    "  }}}},\n"
    '  "universe_domain": "googleapis.example.com"\n'
    "}"
)


@pytest.mark.parametrize(
    ("document", "parts"),
    [(OPEN, PARTS), (json.dumps(DEEP), DEEP_PARTS), (DEEP_PRETTY, DEEP_PARTS)],
    ids=["not closed", "deep, one line", "deep, pretty"],
)
def test_a_document_not_closed_or_too_deep_is_masked_line_by_line(document, parts):
    """The lead's ruling: the first line, then each next line that keeps a
    JSON key's shape, after its key, to the line's end; the line of other
    text after it is kept, in text and its escaped and colour forms and in
    a JSON string; in trajectory files, at export and in config.json the
    whole value is masked."""
    for form, text in _text_forms(document).items():
        out = redact_trajectory_text(f"{text}\n{NEXT}")
        assert out.endswith(NEXT), form
        for part in parts:
            assert part not in out, (form, part)
    record = json.dumps({"message": f"{GAC}={document}\n{NEXT}", "n": 1})
    out = redact_trajectory_text(record)
    assert json.loads(out)["message"].endswith(NEXT)
    assert not any(part in out for part in parts)
    assert redact_trajectory_obj({GAC: document}) == {GAC: MASK}
    assert redact_value({GAC: document})[0] == {GAC: EXPORT_MASK}
    assert not _should_record_env_entry(GAC, document)


def test_the_lines_are_masked_after_their_keys_and_stop_at_a_line_of_another_shape():
    """The lead's ruling: a key line keeps its key and its value is masked
    to the line's end; the first line that keeps no key's shape (a line of
    closing braces) ends the masking, and what follows is text again."""
    out = redact_trajectory_text(f"{GAC}={DEEP_PRETTY}")
    assert out.splitlines()[:3] == [
        f"{GAC}={MASK}",
        f'  "private_key_id": {MASK}',
        f'  "client_email": {MASK}',
    ]
    assert "  }}}},\n" in out
    assert '"universe_domain": "googleapis.example.com"' in out
    assert redact_trajectory_text(out) == out


def test_a_document_past_the_cap_counts_as_not_closed():
    """The lead's ruling: past 64 KB a document is masked as one not
    closed: the first line, then its key lines; the rest is text again."""
    filler = "".join(f'  "field_{n}": "plain value {n}",\n' for n in range(3000))
    big = "{\n" + filler + '  "api_key": "FAKEcapkey0123456789"\n}\n' + NEXT
    assert len(big) > types._JSON_CAP
    out = redact_trajectory_text(f"{GAC}={big}")
    assert out.startswith(f"{GAC}={MASK}\n")
    assert f'"field_7": {MASK}' in out
    assert "FAKEcapkey0123456789" not in out and out.endswith(f"}}\n{NEXT}")


def test_the_document_reading_is_the_names_alone():
    """A `{` after any other name is read as before: its value is the run
    the carrier masks, never a document across lines."""
    text = 'KEY_2={\n  "a": "FAKEvalue0123"\n}'
    out = redact_trajectory_text(text)
    assert out.startswith(f"KEY_2={MASK}\n")


@pytest.mark.parametrize(
    "unit",
    [
        f"{GAC}={PRETTY}\n",
        f"{GAC}={PRETTY}\n{NEXT}\n",
        f'{GAC}={{\n  "k": "v",\n',
        f'  "{GAC}": {{\n',
        f"{GAC}={ONE_LINE[:-1]}\n",
        '  "note": "a } brace",\n',
        f"{GAC}={OPEN}\n{NEXT}\n",
        f"{GAC}={DEEP_PRETTY}\n",
        f"{GAC}={json.dumps(DEEP)}\n",
    ],
)
def test_documents_repeated_stay_linear(unit):
    """The lead's ruling: one reading per document, none overlapping. Each
    repeated to 50,000 and 200,000 characters: four times the text takes
    about four times as long."""
    seconds, outs = linear_seconds(redact_trajectory_text, lambda n: _repeat(unit, n))
    assert all("fakekeyid0123456789" not in out for out in outs)
    assert_linear(seconds)


def test_an_unclosed_document_of_the_caps_length_is_read_once():
    """The lead's ruling: an unclosed document of 16 and 64 KB, alone and
    after many closed ones: four times the text takes about four times as
    long."""

    def open_document(lines: int) -> str:
        return f"{GAC}={{\n" + "".join(f'  "f{n}": "v{n}",\n' for n in range(lines))

    for make in (
        open_document,
        lambda n: f"{GAC}={PRETTY}\n" * (n // 20) + open_document(n),
    ):
        seconds, outs = linear_seconds(redact_trajectory_text, make, sizes=(1000, 4000))
        assert all(out.startswith(f"{GAC}={MASK}") for out in outs)
        assert_linear(seconds)


# Review R192, M1: an unclosed document inside a JSON string went back
# through the whole redactor once per further document (RecursionError at
# 160 truncated lines, quadratic before that), and the line walk re-read the
# rest once per line. A truncated env line, as a tool's output cuts one.
TRUNCATED = (
    f'{GAC}={{"type": "service_account", "project_id": "placeholder-project", '
    '"private_key_id": "fakekeyid0123456789"'
)


def _layers(text: str) -> dict[str, str]:
    """The JSON-string layers: the object path, the ACP record and a
    serialized record redacted as text."""
    return {
        "object": json.dumps(redact_trajectory_obj({"output": text})),
        "acp": redact_acp_trajectory_jsonl([{"type": "x", "output": text}]),
        "record text": redact_trajectory_text(json.dumps({"output": text})),
    }


_LAYER = {
    "object": lambda text: json.dumps(redact_trajectory_obj({"output": text})),
    "acp": lambda text: redact_acp_trajectory_jsonl([{"type": "x", "output": text}]),
    "record text": lambda text: redact_trajectory_text(json.dumps({"output": text})),
}
# A timing test times its own layer alone, at 25,000 against 100,000
# characters: a smaller working set, so a loaded machine's shared caches
# slow both sizes alike.
JSON_STRING_SIZES = (25_000, 100_000)


@pytest.mark.parametrize("lines", [160, 1000])
def test_truncated_documents_in_a_json_string_finish_and_mask(lines):
    """R192's case: no exception and nothing of a document left, in the
    object, ACP and record-text layers."""
    for layer, out in _layers("\n".join([TRUNCATED] * lines)).items():
        leaked = "fakekeyid0123456789" in out
        assert not leaked, layer


@pytest.mark.parametrize("layer", ["object", "acp", "record text"])
@pytest.mark.ceiling
def test_ten_thousand_unclosed_documents_in_one_json_string_finish(layer):
    """The lead's guard: 10,000 unclosed documents in one JSON string finish
    with no exception and under the 10 s ceiling, in the object, ACP and
    record-text layers (no ratio here; the next test times them)."""
    out = within_ceiling(_layers, "\n".join([TRUNCATED] * 10_000))[layer]
    assert "fakekeyid0123456789" not in out


@pytest.mark.parametrize("layer", ["object", "acp", "record text"])
def test_unclosed_documents_in_one_json_string_are_linear(layer):
    """The same at 1,000 and 4,000 documents, at the shared check (best of
    seven, the sizes in turn): four times the documents take about four
    times as long."""

    seconds, outs = linear_seconds(
        _LAYER[layer], lambda n: "\n".join([TRUNCATED] * n), sizes=(1000, 4000)
    )
    assert all("fakekeyid0123456789" not in out for out in outs)
    assert_linear(seconds)


KEY_LINE_RUN = "".join(f'  "field_{n}": "plain value {n}",\n' for n in range(20))


@pytest.mark.parametrize(
    "unit",
    [
        TRUNCATED + "\n",
        f"{GAC}={PRETTY}\n",
        f"{GAC}={OPEN}\n{NEXT}\n",
        f"{GAC}={DEEP_PRETTY}\n",
        KEY_LINE_RUN,
    ],
    ids=["truncated", "closed", "not closed", "deep", "key lines"],
)
@pytest.mark.parametrize("layer", ["object", "acp", "record text"])
def test_documents_in_a_json_string_stay_linear(unit, layer):
    """R192, M1: the documents as a JSON string's text, in each layer, at
    50,000 and 200,000 characters. The key-line run follows one unclosed
    document (the line walk read the rest once per line)."""

    def make(size: int) -> str:
        if unit is KEY_LINE_RUN:
            return f"{GAC}={{\n" + _repeat(unit, size)
        return _repeat(unit, size)

    seconds, outs = linear_seconds(_LAYER[layer], make, sizes=JSON_STRING_SIZES)
    leaked = any("fakekeyid0123456789" in out for out in outs)
    assert not leaked
    assert_linear(seconds)


def test_an_open_document_with_a_long_key_line_run_is_read_once():
    """R192, M1's second shape: one unclosed document followed by key lines,
    in running text, at 100,000 and 400,000 characters."""
    seconds, outs = linear_seconds(
        redact_trajectory_text,
        lambda n: f"{GAC}={{\n" + _repeat(KEY_LINE_RUN, n),
        sizes=(100_000, 400_000),
    )
    assert all(out.startswith(f"{GAC}={MASK}") for out in outs)
    assert_linear(seconds)


# Review R192, L2: a list under the name, as a JSON file read as text holds
# one, was left as it was in text. The ruling: a mapping or a list is masked
# whole in every layer. A list is read as a document is, from its `[` to the
# `]` that closes it; one not closed is masked line by line, its lines of
# brackets, braces and commas alone kept and read past.
BRACKETED = {**ACCOUNT, "note": "a ] and a [ and a }, in a string"}
LISTS = {
    "one line": json.dumps([ACCOUNT]),
    "pretty": json.dumps([ACCOUNT], indent=2),
    "two items": json.dumps([ACCOUNT, BRACKETED], indent=2),
    "nested": json.dumps([[BRACKETED]], indent=2),
    "strings": json.dumps(["fakekeyid0123456789", "fake-sa@fake-project.example.com"]),
}
COMPACT_ITEM = '{"private_key_id": "fakekeyid0123456789"}'
COMPACT_ITEMS = "[\n  " + COMPACT_ITEM + ",\n  " + COMPACT_ITEM[:-1]
OPEN_LISTS = {
    "one line": LISTS["one line"][:-2],
    "pretty": LISTS["pretty"][: LISTS["pretty"].rindex("]")],
    "two items": LISTS["two items"][: LISTS["two items"].rindex("]")],
    "two items, CRLF": LISTS["two items"][: LISTS["two items"].rindex("]")].replace(
        "\n", "\r\n"
    ),
    "too deep": json.dumps([[[{"a": {"b": "x"}}]], ACCOUNT], indent=2),
    # Review R192B, L-3: compact items, one a line, are masked after their
    # `{`, CRLF too, after an inline first item and after a close.
    "compact items": COMPACT_ITEMS,
    "compact items, CRLF": COMPACT_ITEMS.replace("\n", "\r\n"),
    "a compact item after an inline one": "["
    + COMPACT_ITEM
    + ",\n  "
    + COMPACT_ITEM[:-1],
    "an item after a close": '[\n  {"type": "service_account"\n  }, '
    + COMPACT_ITEM[:-1],
}


def _parts(document: str) -> list[str]:
    return [part for part in PARTS if part in document]


@pytest.mark.parametrize("document", LISTS.values(), ids=LISTS.keys())
def test_a_list_under_the_name_is_masked_whole_in_every_layer(document):
    """R192, L2: masked from its `[` to its `]` in text and its escaped and
    colour forms, in a JSON string, in a JSON file read as text, in the
    object, ACP and record-text layers, at export and in config.json; the
    next line is kept."""
    parts = _parts(document)
    assert parts
    for form, text in _text_forms(document).items():
        for part in parts:
            assert _leaked(text, part) == set(), (form, part)
        out = redact_trajectory_text(f"{text}\n{NEXT}")
        assert out.endswith(f"{MASK}\n{NEXT}"), form
    record = json.dumps({"message": f"{GAC}={document}\n{NEXT}", "n": 1})
    out = redact_trajectory_text(record)
    assert json.loads(out)["message"].endswith(f"{MASK}\n{NEXT}")
    for record in (
        json.dumps({"env": {GAC: document}}),
        json.dumps({GAC: json.loads(document), "n": 1}),
        json.dumps({GAC: json.loads(document), "n": 1}, indent=2),
    ):
        out = redact_trajectory_text(record)
        assert not any(part in out for part in parts), record
        assert redact_trajectory_text(out) == out
    for layer, out in _layers(f"{GAC}={document}\n{NEXT}").items():
        assert not any(part in out for part in parts), layer
        assert NEXT in out, layer
    for value in (document, json.loads(document)):
        assert redact_trajectory_obj({GAC: value}) == {GAC: MASK}
        assert redact_value({GAC: value})[0] == {GAC: EXPORT_MASK}
    assert not _should_record_env_entry(GAC, document)


@pytest.mark.parametrize("document", OPEN_LISTS.values(), ids=OPEN_LISTS.keys())
def test_a_list_not_closed_or_too_deep_is_masked_line_by_line(document):
    """R192, L2, as the lead's ruling for a document: the first line, then
    each key line after its key; a line of brackets, braces and commas
    alone is kept and read past; the line of other text after it is
    kept."""
    parts = _parts(document)
    assert parts
    for form, text in _text_forms(document).items():
        out = redact_trajectory_text(f"{text}\n{NEXT}")
        assert out.endswith(NEXT), form
        assert not any(part in out for part in parts), form
        assert redact_trajectory_text(out) == out, form
    record = json.dumps({"message": f"{GAC}={document}\n{NEXT}", "n": 1})
    out = redact_trajectory_text(record)
    assert json.loads(out)["message"].endswith(NEXT)
    assert not any(part in out for part in parts)
    for layer, out in _layers(f"{GAC}={document}\n{NEXT}").items():
        assert not any(part in out for part in parts), layer
        assert NEXT in out, layer
    assert redact_trajectory_obj({GAC: document}) == {GAC: MASK}
    assert redact_value({GAC: document})[0] == {GAC: EXPORT_MASK}
    assert not _should_record_env_entry(GAC, document)


def test_a_list_under_any_other_name_is_read_as_before():
    """Only the name that holds a credentials file's content reads a list."""
    text = 'KEY_2=["FAKEvalue0123", "b"]\n  {\n  "a": "kept"'
    assert redact_trajectory_text(text).endswith('\n  {\n  "a": "kept"')


LIST_UNITS = [
    f"{GAC}={LISTS['pretty']}\n",
    f"{GAC}={LISTS['two items']}\n{NEXT}\n",
    f"{GAC}={OPEN_LISTS['two items']}\n{NEXT}\n",
    f"{GAC}={OPEN_LISTS['too deep']}\n",
    f"{GAC}={LISTS['one line'][:-2]}\n",
    f"{GAC}=[[[[{{\n",
    f"{GAC}=[\n  {{\n  }},\n  [\n",
    f'  "{GAC}": [\n',
    f"{GAC}=[\n" + ("  " + COMPACT_ITEM + ",\n") * 20,
]


@pytest.mark.parametrize("unit", LIST_UNITS)
def test_lists_repeated_stay_linear(unit):
    """R192, L2: one reading per list, none overlapping; four times the text
    takes about four times as long."""
    seconds, outs = linear_seconds(redact_trajectory_text, lambda n: _repeat(unit, n))
    assert all("fakekeyid0123456789" not in out for out in outs)
    assert_linear(seconds)


@pytest.mark.parametrize("unit", LIST_UNITS[:5])
@pytest.mark.parametrize("layer", ["object", "acp", "record text"])
def test_lists_in_a_json_string_stay_linear(unit, layer):
    """The same lists inside a JSON string, in the object, ACP and
    record-text layers."""

    seconds, outs = linear_seconds(
        _LAYER[layer], lambda n: _repeat(unit, n), sizes=JSON_STRING_SIZES
    )
    assert all("fakekeyid0123456789" not in out for out in outs)
    assert_linear(seconds)


def test_a_bracket_run_under_the_name_is_read_once():
    """A `[` run of the text's length, under the name, is read once."""
    seconds, outs = linear_seconds(
        redact_trajectory_text,
        lambda n: f"{GAC}=" + "[" * n + '"fakekeyid0123456789"',
        sizes=(100_000, 400_000),
    )
    assert all("fakekeyid0123456789" not in out for out in outs)
    assert_linear(seconds)
