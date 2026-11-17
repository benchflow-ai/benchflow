"""(the lead, 2026-11-15 12:22Z): the decoded views read the escapes
other languages and URLs write, as a named list: a percent escape (`%73`,
nested `%25` wrappers at any depth), `\\x73`, an octal `\\163`, JavaScript's
`\\u{73}`, and mixes of these with the JSON and HTML forms. A
provider key written `%73k-<key>` or `\\x73k-<key>` was shown on every line
(0.9.2, rel, 0.10.0rc3, the rc and). Every value here is made up.
"""

from __future__ import annotations

import codecs
import html
import json
import random
import re
import urllib.parse

import pytest

from benchflow.publish.redact import redact_value
from benchflow.trajectories import types
from benchflow.trajectories.types import redact_trajectory_text
from tests._timing import assert_linear, linear_seconds
from tests.test_redaction_escape_boundary import LAYERS, _leaks, _unchanged

KEY = "FAKEq7Xw2Lr9Tb4Nc6Vd1Hs8Kf3Mj5Pg0Yt2Rb7Qn4Wx9"
AWS_ID = "FAKEQ7XW2LR9TB4N"

FAMILIES = {
    "pct": lambda c: f"%{ord(c):02X}",
    "pct lower": lambda c: f"%{ord(c):02x}",
    "pct25": lambda c: f"%25{ord(c):02x}",
    "pct25 x6": lambda c: "%" + "25" * 6 + f"{ord(c):02X}",
    "x": lambda c: f"\\x{ord(c):02x}",
    "x upper": lambda c: f"\\x{ord(c):02X}",
    "octal": lambda c: f"\\{ord(c):03o}",
    "u brace": lambda c: f"\\u{{{ord(c):x}}}",
    "u brace zeros": lambda c: f"\\u{{{ord(c):06X}}}",
    "x brace": lambda c: f"\\x{{{ord(c):x}}}",
    "U eight": lambda c: f"\\U{ord(c):08x}",
    "entity without semicolon": lambda c: f"&#{ord(c)}",
    "hex entity without semicolon": lambda c: f"&#x{ord(c):x}",
}
# One family writing another's syntax: two or more levels to the character.
CHAINS = {
    "pct of x": lambda c: f"%5Cx{ord(c):02x}",
    "x of pct": lambda c: f"\\x25{ord(c):02x}",
    "json of pct": lambda c: f"\\u0025{ord(c):02x}",
    "octal of pct": lambda c: f"\\045{ord(c):02x}",
    "u brace of pct": lambda c: f"\\u{{25}}{ord(c):02x}",
    "entity of pct": lambda c: f"&#37;{ord(c):02x}",
    "pct of octal": lambda c: f"%5C{ord(c):03o}",
    "pct of u brace": lambda c: f"%5Cu{{{ord(c):x}}}",
    "x of x": lambda c: f"\\x5Cx{ord(c):02x}",
    "pct of entity": lambda c: f"%26%23{ord(c)};",
    "x of json": lambda c: f"\\x5Cu{ord(c):04x}",
    "pct25 of x": lambda c: f"%255Cx{ord(c):02x}",
    "entity of entity": lambda c: f"&amp;#{ord(c)};",
}
VIEWED = {
    "json": lambda c: f"\\u{ord(c):04x}",
    "entity": lambda c: f"&#{ord(c)};",
    "hex entity": lambda c: f"&#x{ord(c):X};",
}

# (the cue as written, a row holding it that the redactor masks as written, the secret it carries)
ROWS = [
    ("sk-", f"log line sk-{KEY} end", KEY),
    ("ghp_", f"log line ghp_{KEY[:36]} end", KEY[:36]),
    ("AKIA", f"log line AKIA{AWS_ID} end", AWS_ID),
    ("gsk_", f"log line gsk_{KEY} end", KEY),
    ("xoxb-", f"log line xoxb-{KEY} end", KEY),
    ("x_token", f"x_token={KEY}", KEY),
    ("api_key", f'{{"api_key": "{KEY}"}}', KEY),
    ("password", f"password: {KEY}", KEY),
    ("Authorization", f"Authorization: Bearer {KEY}", KEY),
]


def _spell(row: str, cue: str, encode, positions) -> str:
    at = row.index(cue)
    chars = list(cue)
    for i in positions:
        chars[i] = encode(chars[i])
    return row[:at] + "".join(chars) + row[at + len(cue) :]


def _forms(encode):
    """Each row in the family, alone, as a JSON copy and inside a JSON string;
    a spelling Python's decoders do not read back as the row is not one
    (`&#x2d` before `FA` reads as U+2DFA)."""
    for cue, row, secret in ROWS:
        for positions in ([0], [len(cue) - 1], range(len(cue))):
            text = _spell(row, cue, encode, positions)
            for wrap in (
                lambda s: s,
                lambda s: json.dumps(s)[1:-1],
                lambda s: json.dumps({"m": s}),
            ):
                if _oracle(wrap(text)) == _oracle(wrap(row)):
                    yield cue, wrap(text), secret


@pytest.mark.parametrize("cue, row, secret", ROWS)
def test_every_row_is_masked_as_written(cue, row, secret):
    assert _leaks(row, secret) == []


@pytest.mark.parametrize("family", FAMILIES)
def test_a_cue_written_in_each_family_is_masked(family):
    """Each row's cue, its first character, its last or all of it written in
    the family, alone, as a JSON copy and inside a JSON string: no layer
    shows the secret."""
    shown = [
        (form, layer)
        for _cue, form, secret in _forms(FAMILIES[family])
        for layer in _leaks(form, secret)
    ]
    assert shown == []


@pytest.mark.parametrize("chain", CHAINS)
def test_a_family_writing_another_reaches_the_character(chain):
    shown = [
        (form, layer)
        for _cue, form, secret in _forms(CHAINS[chain])
        for layer in _leaks(form, secret)
    ]
    assert shown == []


@pytest.mark.parametrize("wrappers", range(13))
def test_nested_percent_is_read_at_any_depth(wrappers):
    text = f"log line %{'25' * wrappers}73k-{KEY} end"
    assert _leaks(text, KEY) == []


def test_the_output_keeps_its_own_escaping():
    """Only the secret's span is masked; the escapes around it stay as
    written."""
    for text in (
        f"a %41 log line %73k-{KEY} end %0A",
        f"a \\x41 then \\163k-{KEY} \\u{{41}}",
    ):
        out = redact_trajectory_text(text)
        assert KEY not in out
        assert out.startswith(text[:8]) and out.endswith(text[-5:])


def test_a_url_scheme_written_with_a_percent_colon_is_read():
    """`%3A` stays for the main pass's encoded-separator rules, except the
    colon of `://`."""
    for text in (
        f"get https%3A//fakeuser:{KEY}@example.invalid/x",
        f"get https%3A%2F%2Ffakeuser:{KEY}@example.invalid/x",
    ):
        assert _leaks(text, KEY) == []


# Text with these escapes and no secret.
KEPT = [
    "GET /search?q=hello%20world&lang=en%2Dus HTTP/1.1",
    "path C:\\x64\\bin\\tool.exe",
    "printf '\\101\\102\\103\\n'",
    "emoji \\u{1F600} and \\u{41}\\u{42}",
    "progress 100% done, 50%25 left",
    "url https%3A%2F%2Fexample.invalid%2Fdocs%3Fpage%3D2",
    "TESTS_PASS%3D12",
]


@pytest.mark.parametrize("text", KEPT)
def test_escaped_text_with_no_secret_is_kept(text):
    assert _unchanged(text) == dict.fromkeys(LAYERS, True)
    assert _unchanged(json.dumps({"m": text})) == dict.fromkeys(LAYERS, True)


# The oracle: Python's own decoders, applied together until nothing changes.
_BACKSLASH = re.compile(
    r"\\(?:x[0-9a-fA-F]{2}|[0-7]{3}|[ux]\{[0-9a-fA-F]+\}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|[\\'\"abfnrtv/])"
)


def _backslash(esc: str) -> str:
    if esc.startswith(("\\u{", "\\x{")):
        return chr(int(esc[3:-1], 16))
    if esc == "\\/":
        return "/"
    return codecs.decode(esc, "unicode_escape")


def _oracle(text: str) -> str:
    for _ in range(12):
        decoded = urllib.parse.unquote(text, encoding="latin-1")
        decoded = _BACKSLASH.sub(lambda m: _backslash(m[0]), html.unescape(decoded))
        if decoded == text:
            return text
        text = decoded
    return text


# The layers that read a text inside one more JSON level (a record's or a
# probe's): a cue there needs one more level.
WRAPPED = ("obj", "acp", "live", "publish")


def test_the_views_agree_with_pythons_decoders():
    """Random mixes: each character of a cue raw or in any family (the JSON
    and HTML forms included), the row then JSON-copied 0 to 2 times. Python's
    decoders read the row back. Where the levels the cue needs (the copies,
    the deepest family's levels, one more in a wrapped layer) are within the
    views' depth, no layer shows the secret, and decoding the text layer's
    output gives none back. Past it is a named residual."""
    rand = random.Random(687)
    families = [
        *((f, 1) for f in FAMILIES.values()),
        *((f, 2) for f in CHAINS.values()),
        *((f, 1) for f in VIEWED.values()),
    ]
    shown = []
    for _ in range(400):
        cue, row, secret = rand.choice(ROWS)
        at = row.index(cue)
        deepest, chars = 0, []
        for c in cue:
            if rand.random() < 0.7:
                encode, levels = rand.choice(families)
                chars.append(encode(c))
                deepest = max(deepest, levels)
            else:
                chars.append(c)
        text, plain = row[:at] + "".join(chars) + row[at + len(cue) :], row
        copies = rand.randrange(3)
        for _ in range(copies):
            text, plain = json.dumps(text)[1:-1], json.dumps(plain)[1:-1]
        if _oracle(text) != _oracle(plain):
            continue
        needed = copies + deepest
        layers = [
            layer
            for layer in _leaks(text, secret)
            if needed + (layer in WRAPPED) <= types._VIEW_DEPTH
        ]
        if needed <= types._VIEW_DEPTH and secret in _oracle(
            redact_trajectory_text(text)
        ):
            layers.append("decoded output")
        if layers:
            shown.append((text, layers))
    assert shown == []


# Rows only publish's own rules mask (VALUE_PATTERNS, the command-line options).
PUBLISH_ROWS = [
    ("bearer", f"bearer {KEY}"),
    ("Bearer", f"note Bearer {KEY} end"),
    ("api-key", f"run --api-key {KEY} now"),
    ("token", f"run --token={KEY} now"),
    ("AQ.", f"log AQ.{KEY} end"),
]


@pytest.mark.parametrize("family", [*FAMILIES, *CHAINS])
def test_publish_reads_the_views_with_its_own_rules(family):
    """The lead, 2026-11-15 13:18Z: publish is the layer that leaves our
    hands, so its own rules read the same views and decoder: each row's cue
    written in the family, alone and as a JSON copy, is masked as its plain
    form is, and a second pass changes nothing."""
    encode = {**FAMILIES, **CHAINS}[family]
    shown = []
    for cue, row in PUBLISH_ROWS:
        assert KEY not in json.dumps(redact_value({"m": row})[0])
        for positions in ([0], [len(cue) - 1], range(len(cue))):
            text = _spell(row, cue, encode, positions)
            for form, plain in (
                (text, row),
                (json.dumps(text)[1:-1], json.dumps(row)[1:-1]),
            ):
                if _oracle(form) != _oracle(plain):
                    continue
                once = redact_value({"m": form})[0]
                if KEY in json.dumps(once) or redact_value(once)[0] != once:
                    shown.append(form)
    assert shown == []


def test_publish_adds_its_masks_after_the_main_pass():
    """Publish's view masks are added after the main pass, by position: a
    word that only looks like an option (`retry--Authorization :`, review
    R192C's ma2 rows) hides no value the main pass masks."""
    for text in (
        f"Authorization: please retry--Authorization : &#34;{KEY}&#34;",
        f'Authorization: please retry--Authorization : \\"{KEY}\\"',
        f'run --api-key : "{KEY}" now',
    ):
        assert KEY not in json.dumps(redact_value({"m": text})[0]), text


@pytest.mark.parametrize("family", [*FAMILIES, *CHAINS])
def test_the_gate_admits_a_record_with_the_family(family):
    """Its per-record gate lets a line through to the views when the
    line's only escapes are this family's."""
    encode = {**FAMILIES, **CHAINS}[family]
    closed = [
        text
        for cue, row, _secret in ROWS
        for text in (
            _spell(row, cue, encode, [0]),
            _spell(row, cue, encode, range(len(cue))),
        )
        if _oracle(text) == _oracle(row) and types._escape_flags(text) is None
    ]
    assert closed == []


def _repeat(unit: str, size: int) -> str:
    return (unit * (size // len(unit) + 1))[:size]


# Long runs of each family's syntax that never complete an escape, or complete
# one at every step. No cue: the gate's finders run to the end (R528's trap: a
# cue opens the gate before them).
RUNS = {
    "percent-25 run": lambda size: "%" + "25" * (size // 2) + "zz",
    "percent starts": lambda size: _repeat("%2", size),
    "backslash run": lambda size: "a " + "\\" * size + " q",
    "octal starts": lambda size: _repeat("\\12", size),
    "octal escapes": lambda size: _repeat("\\101", size),
    "x starts": lambda size: _repeat("\\x7", size),
    "u brace starts": lambda size: _repeat("\\u{", size),
    "u brace zeros": lambda size: "\\u{" + "0" * size,
    "u brace digits": lambda size: "\\u{" + "1" * size + "}",
}


@pytest.mark.parametrize("run", RUNS)
def test_the_decode_set_reads_each_run_once(run):
    """The views' decoder on each run at 50,000 and 200,000 characters."""
    seconds, _ = linear_seconds(types._decode_once, RUNS[run])
    assert_linear(seconds)


@pytest.mark.parametrize("run", RUNS)
def test_the_gate_reads_each_run_once(run):
    seconds, _ = linear_seconds(types._escape_flags, RUNS[run])
    assert_linear(seconds)


@pytest.mark.parametrize("run", RUNS)
def test_redaction_stays_linear_on_each_run_after_a_cue(run):
    """With a cue first, the gate opens and the views read the run."""
    seconds, _ = linear_seconds(
        redact_trajectory_text, lambda size: f"x_token={KEY} " + RUNS[run](size)
    )
    assert_linear(seconds)


@pytest.mark.parametrize("family", ["pct", "x", "octal", "u brace"])
def test_a_key_block_with_an_escaped_head_is_read_whole(family):
    """A private-key block spans lines and its body holds no escape: a head
    written in its families carries its per-record gate through the
    block's reach (its carry, #273), so the body is masked in every
    layer (FX688's rows; a JSON `\\u` head too)."""
    encode = FAMILIES[family]
    body = "\\n".join(KEY[i : i + 16] for i in range(0, len(KEY), 16))
    shown = []
    for heads in ([5], [0], range(10)):
        head = _spell("-----BEGIN", "-----BEGIN", encode, heads)
        for text in (
            f"key {head} PRIVATE KEY-----\\n{body}\\n-----END PRIVATE KEY----- end",
            f"key {_spell('-----BEGIN', '-----BEGIN', VIEWED['json'], heads)} PRIVATE KEY-----\\n{body}\\n-----END PRIVATE KEY-----",
        ):
            shown += [(text, layer) for layer in _leaks(text, KEY[16:32])]
    assert shown == []
