"""A secret value is masked through its backslashes (EN-688).

A carrier's value (``DB_PASSWORD=...``, ``"password": "..."``,
``Authorization: ...``) was masked only up to its first backslash, so
``DB_PASSWORD=\\X<rest>``, ``password: \\x41<rest>`` and ``token=a\\x41<rest>``
showed the rest, in every layer and on every released line. A backslash is
now part of the value wherever it stands, escape or not, and the value still
stops where it did: at its closing quote (by the opening quote's depth),
whitespace, ``,`` ``}`` ``&`` ``*``, and an escaped line break that starts
the next line's own ``name:`` or ``name=``.

A quote inside a value is part of it where the value goes on right after
it: a single quote, or an escaped one. Two quote cases stay as they were
(named on EN-688): a double quote written raw inside a value (R1: it may
close a json.dumps copy's own string), and a value that opens with an
escaped quote and holds it again (R2), which reads as a quoted value
followed by more text.

Every value here is made up.
"""

from __future__ import annotations

import json
import re
import time

import pytest

from benchflow.publish.redact import REDACTED as PUBLISH_MASK
from benchflow.publish.redact import redact_value, redact_value_to_stability
from benchflow.trajectories.export import export_trajectories_to_jsonl
from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    Trajectory,
    redact_acp_trajectory_jsonl,
    redact_trajectory_obj,
    redact_trajectory_text,
)

MASK = "***REDACTED***"
# The made-up value around each backslash: letters and digits, found by any
# eight of them in a row.
HEAD = "Fx688hEaDqZ1vW"
TAIL = "Fx688tAiLmK7pR"
TAIL20 = "Fx688tAiLmK7pRzQ2wE9"

# Every carrier name the redactor reads a value after, in its spellings.
NAMES = (
    "x-api-key",
    "X-Api-Key",
    "x-goog-api-key",
    "api-key",
    "api_key",
    "apiKey",
    "OPENAI_API_KEY",
    "master_key",
    "private-key",
    "PRIVATE_KEY",
    "AWS_BEARER_TOKEN_BEDROCK",
    "GITHUB_TOKEN",
    "token",
    "access_token",
    "SECRET",
    "client_secret",
    "DB_PASSWORD",
    "password",
    "Password",
    "PASSWD",
    "passwd",
    "AWS_SECRET_ACCESS_KEY",
    "AZURE_STORAGE_ACCOUNT_KEY",
    "SECRET_KEY",
    "ENCRYPTION_KEY",
    "CREDENTIALS",
    "credential",
    "Authorization",
    "authorization",
)


def _inner(value: str) -> str:
    return json.dumps(value)[1:-1]


# How a name and its value are written. `json_enc` writes the value as JSON
# does; `copy` is a json.dumps copy of the `json_raw` text.
SHAPES = {
    "env": lambda n, v: f"{n}={v}",
    "export": lambda n, v: f"export {n}={v}",
    "yaml": lambda n, v: f"{n}: {v}",
    "json_raw": lambda n, v: '{"' + n + '": "' + v + '"}',
    "json_enc": lambda n, v: '{"' + n + '": "' + _inner(v) + '"}',
    "repr": lambda n, v: "{'" + n + "': '" + v + "'}",
    "curl": lambda n, v: f"> GET /v1/models HTTP/1.1\n> {n}: {v}\n> Accept: */*",
    "copy": lambda n, v: '{\\"' + n + '\\": \\"' + _inner(v) + '\\"}',
}
QUOTED = {"json_raw", "json_enc", "repr", "copy"}

# The backslash pieces: no escape, each escape family, escaped whitespace and
# delimiters, and runs of backslashes up to past _ESCQ's eight.
PIECES = {
    "not an escape": "\\q",
    "punctuation": "\\!",
    "hex": "\\x41",
    "unicode": "\\u0073",
    "long unicode": "\\U00000041",
    "octal": "\\163",
    "braced code point": "\\u{1F600}",
    "perl braced": "\\x{1b}",
    "slash": "\\/",
    "newline": "\\n",
    "tab": "\\t",
    "carriage return": "\\r",
    "CRLF": "\\r\\n",
    "comma": "\\,",
    "ampersand": "\\&",
    "brace": "\\}",
    "asterisk": "\\*",
    "space": "\\ ",
    "two": "\\\\a",
    "three": "\\\\\\b",
    "eight": "\\" * 8 + "c",
    "nine": "\\" * 9 + "d",
    "sixteen": "\\" * 16 + "e",
}
# Quotes inside a value, where a quote is no close: escaped, or a single
# quote in a double-quoted value.
QUOTE_PIECES = {
    "escaped double quote": '\\"',
    "escaped single quote": "\\'",
    "double quote": '"',
    "single quote": "'",
}


def _values(piece: str) -> dict[str, str]:
    return {
        "first": piece + HEAD + TAIL,
        "middle": HEAD + piece + TAIL,
        "last": HEAD + TAIL + piece.rstrip("abcde"),
        "twice": piece + HEAD + piece + TAIL,
    }


def _layers(text: str) -> dict[str, str]:
    """The five layers the leak was measured in."""
    return {
        "text": redact_trajectory_text(text),
        "record": redact_trajectory_text(json.dumps({"output": text})),
        "object": json.dumps(redact_trajectory_obj({"output": text})),
        "acp": redact_acp_trajectory_jsonl(
            [{"type": "tool_call_update", "content": [{"type": "text", "text": text}]}]
        ),
        "publish": json.dumps(redact_value({"output": text})[0]),
    }


def _shown(out: str, *parts: str) -> bool:
    return any(p[i : i + 8] in out for p in parts for i in range(len(p) - 7))


def _shown_layers(text: str, *parts: str) -> list[str]:
    return [k for k, v in _layers(text).items() if _shown(v, *parts)]


def test_every_name_and_shape_masks_a_plain_value():
    # The sweep's control: with no backslash every row is masked everywhere,
    # so a row shown below is the backslash's.
    bad = [
        (name, shape)
        for name in NAMES
        for shape, write in SHAPES.items()
        if _shown_layers(write(name, HEAD + TAIL), HEAD, TAIL)
    ]
    assert bad == []


@pytest.mark.parametrize("piece", list(PIECES.values()), ids=list(PIECES))
def test_a_value_is_masked_through_its_backslashes(piece):
    bad = [
        (name, shape, position, layers)
        for name in NAMES
        for shape, write in SHAPES.items()
        for position, value in _values(piece).items()
        if (layers := _shown_layers(write(name, value), HEAD, TAIL))
    ]
    assert bad == []


@pytest.mark.parametrize("piece", list(QUOTE_PIECES.values()), ids=list(QUOTE_PIECES))
def test_a_quote_inside_a_value_is_part_of_it(piece):
    rows = []
    for name in NAMES:
        for shape, write in SHAPES.items():
            for position, value in _values(piece).items():
                raw_quote = '"' in piece and "\\" not in piece
                if raw_quote and shape != "json_enc":
                    continue  # R1: a raw quote ends the value, as before
                if piece == "'" and shape not in ("json_raw", "json_enc", "copy"):
                    continue  # R1: a raw single quote outside a double-quoted value
                if shape not in QUOTED and position in ("first", "twice"):
                    continue  # R2: it opens the value as its quote
                rows.append((name, shape, position, write(name, value)))
    bad = [
        (name, shape, position, layers)
        for name, shape, position, text in rows
        if (layers := _shown_layers(text, HEAD, TAIL))
    ]
    assert rows and bad == []


@pytest.mark.parametrize(
    "text",
    [
        "DB_PASSWORD=\\X" + TAIL,
        "DB_PASSWORD=\\x41" + TAIL,
        "password: \\x41" + TAIL,
        '{"password": "\\\\x41' + TAIL + '"}',
        "token=a\\x41" + TAIL,
        "OPENAI_API_KEY=\\x6b-" + TAIL,
        "OPENAI_API_KEY=%6b-" + TAIL,
        'OPENAI_API_KEY="\\x6bey-' + TAIL + '"',
    ],
    ids=[
        "non-escape",
        "hex",
        "yaml",
        "json",
        "after a letter",
        "escaped key",
        "percent key",
        "quoted escaped key",
    ],
)
def test_the_reported_rows_are_masked_in_every_layer(text, tmp_path):
    record = {
        "prompt": [{"role": "user", "content": text}],
        "completion": [{"role": "assistant", "content": text}],
    }
    out = tmp_path / "export.jsonl"
    export_trajectories_to_jsonl([record], out)
    stable, _ = redact_value_to_stability(record)
    proxied = Trajectory(
        session_id="t",
        exchanges=[
            LLMExchange(
                request=LLMRequest(body={"messages": [{"content": text}]}),
                response=LLMResponse(),
            )
        ],
    ).to_jsonl(redact_keys=True)
    layers = {
        **_layers(text),
        "export": out.read_text(),
        "publish stable": json.dumps(stable),
        "proxy exchange": proxied,
    }
    assert [k for k, v in layers.items() if _shown(v, TAIL)] == []


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("token", "password"),
        ("GITHUB_TOKEN", "DB_PASSWORD"),
        ("password", "api_key"),
        ("api_key", "client_secret"),
        ("DB_PASSWORD", "GITHUB_TOKEN"),
    ],
)
@pytest.mark.parametrize("separator", [": ", "=", " = "])
@pytest.mark.parametrize(
    "piece", ["", "\\q", "\\x41", "\\\\a"], ids=["none", "X", "x41", "two"]
)
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["LF", "CRLF"])
def test_the_next_lines_value_stays_masked(first, second, separator, piece, newline):
    # Read through an escaped line break, the first value would take the next
    # line's name and leave its value; it stops there, as at a line break as
    # written (the json.dumps copies write this line break escaped).
    text = f"{first}={HEAD}{piece}{newline}{second}{separator}{TAIL}"
    assert _shown_layers(text, HEAD, TAIL) == []


@pytest.mark.parametrize(
    "piece", ["", "\\q", "\\x41", "\\\\a", "\\n"], ids=["none", "X", "x41", "two", "n"]
)
def test_an_env_dump_keeps_its_other_lines(piece):
    text = f"GITHUB_TOKEN={HEAD}{piece}\nHOME=/root\nPATH=/usr/local/bin:/usr/bin\nLANG=C.UTF-8"
    for layer, out in _layers(text).items():
        assert not _shown(out, HEAD), layer
        for kept in ("HOME=/root", "PATH=/usr/local/bin:/usr/bin", "LANG=C.UTF-8"):
            assert kept in out, (layer, kept)


@pytest.mark.parametrize(
    "value",
    [
        HEAD + "\\" + TAIL,
        HEAD + "\\\\" + TAIL,
        HEAD + '\\"' + TAIL,
        HEAD + '"' + TAIL,
        HEAD + "\\",
        HEAD + "\\\\",
        "\\" * 9 + HEAD,
        HEAD + "\\u{41}" + TAIL,
    ],
)
@pytest.mark.parametrize("depth", [1, 2])
def test_a_redacted_json_copy_still_parses(value, depth):
    # The exporters redact serialized JSON as text: a value read through its
    # backslashes takes every escape whole, so the copy keeps every escape.
    doc = {"password": value, "next": "kept", "note": f"token={value} and more"}
    text = json.dumps(doc)
    for _ in range(depth - 1):
        text = json.dumps({"output": text})
    out = redact_trajectory_text(text)
    for _ in range(depth - 1):
        out = json.loads(out)["output"]
    parsed = json.loads(out)
    assert parsed["next"] == "kept"
    assert not _shown(json.dumps(parsed), HEAD, TAIL)


@pytest.mark.parametrize("piece", ["\\q", "\\x41", '\\"', "\\\\", "\\n", "\\u{41}"])
def test_publish_redaction_is_stable(piece):
    record = {"output": f"password={HEAD}{piece}{TAIL} and DB_PASSWORD: {piece}{TAIL}"}
    once, count = redact_value(record)
    assert count
    assert redact_value(once)[1] == 0
    stable, _ = redact_value_to_stability(record)
    assert stable == once
    assert PUBLISH_MASK in stable["output"]


# Linear time: the value reads each run of backslashes once.
def _cpu_ratio(run, small: str, large: str, rounds: int = 5) -> float:
    """The best round's CPU-time ratio of *run* on *large* over *small*."""
    best = float("inf")
    for _ in range(rounds):
        t0 = time.process_time()
        run(small)
        t1 = time.process_time()
        run(large)
        t2 = time.process_time()
        best = min(best, (t2 - t1) / max(t1 - t0, 1e-4))
    return best


_RUNS = {
    "run at the end": lambda n: "password=" + "\\" * n,
    "run before a quote": lambda n: "password=" + "\\" * n + '"',
    "quoted run": lambda n: 'password: "' + "\\" * n + '"',
    "deep quote run": lambda n: 'password: \\\\\\\\\\\\\\\\"' + "\\" * n + '"',
    "escaped line breaks": lambda n: "password=" + "\\n" * n,
    "escaped line breaks and names": lambda n: "password=a" + "\\nb" * n,
    "unclosed braces": lambda n: "password=" + "\\u{" * n,
    "escaped quotes": lambda n: '\\"password\\": \\"' + '\\\\\\"' * n,
    "many carriers": lambda n: "password=\\X " * (n // 10),
    "runs and letters": lambda n: "token=" + "\\\\\\a" * (n // 4),
}


@pytest.mark.parametrize("make", list(_RUNS.values()), ids=list(_RUNS))
def test_long_backslash_runs_are_read_in_linear_time(make):
    small, large = make(20_000), make(80_000)
    t0 = time.process_time()
    redact_trajectory_text(large)
    assert time.process_time() - t0 < 10
    # 4x the text: about 4x the time; a quadratic read is about 16x.
    assert _cpu_ratio(redact_trajectory_text, small, large) < 8


def test_the_ratio_catches_a_quadratic_read():
    quadratic = re.compile(r"\\+x").search
    assert _cpu_ratio(quadratic, "\\" * 2_000, "\\" * 8_000, rounds=3) > 8


# R5 (named on EN-688): a value whose first word reads as prose, a switch or
# a count, then an escaped line break, tab or carriage return, and more: the
# carriers that judge their value by its words (the rc's Authorization
# names, a Bearer scheme, the lower-case pass names) take the word alone, as
# at a line break. A provider-shaped key after the break is still masked by
# its shape, whatever the first word is.
_KEY_SHAPES = {
    "openai": "sk-" + "Fx688madeUpKeyAbCdEfGhIjKlMnOpQrStUvWxYz012345",
    "github": "ghp_" + "Fx688madeUpGhpAbCdEfGhIjKlMnOpQrStUv01",
    "aws": "AKIA" + "FX688MADEUPKEY01",
    "jwt": "eyJ"
    + "hbGciOiJIUzI1NiJ9"
    + "."
    + "eyJzdWIiOiJmeDY4OCJ9"
    + "."
    + "Fx688madeUpSignatureAbCdEfGhIjKlMn",
}


@pytest.mark.parametrize("key", list(_KEY_SHAPES.values()), ids=list(_KEY_SHAPES))
@pytest.mark.parametrize(
    "first",
    [
        "api_key=yes",
        "Authorization: required",
        "password: none",
        "GITHUB_TOKEN: Bearer word",
        "db_pass: True",
        "token=1",
    ],
)
@pytest.mark.parametrize("brk", ["\\n", "\\r", "\\t", "\n"], ids=["n", "r", "t", "LF"])
def test_a_provider_key_after_a_word_and_a_break_is_masked(key, first, brk):
    text = f"{first}{brk}{key}"
    assert [k for k, v in _layers(text).items() if key[-12:] in v] == []


# A second name after an escape inside the first value: the value stops before
# it, so the second value stays masked (review R632, 2026-11-15: an escaped
# ANSI colour code, an escaped `&` or quote, any escape and more letters).
@pytest.mark.parametrize(
    "junk",
    [
        "\\u001b[32m",
        "\\x1b[0m",
        "\\&#34;",
        ";\\u001b[32m",
        "\\X" + "w1",
        "\\u0026",
        "\\u003c",
        "\\x41",
    ],
    ids=[
        "ansi-u",
        "ansi-x",
        "entity-quote",
        "semicolon-ansi",
        "escape-word",
        "amp",
        "lt",
        "hex",
    ],
)
@pytest.mark.parametrize(
    "second", ["GITHUB_TOKEN", "password", "api_key", "Authorization"]
)
@pytest.mark.parametrize("separator", [": ", "=", " = ", "\\t: "])
def test_a_second_name_after_an_escape_keeps_its_value_masked(junk, second, separator):
    # Where the second entry, after the same junk, is masked standing alone,
    # a first value before it never unmasks it.
    alone = f"{junk}{second}{separator}{TAIL}"
    glued = f"password={HEAD}{junk}{second}{separator}{TAIL}"
    masked_alone = set(_layers(alone)) - set(_shown_layers(alone, TAIL))
    assert masked_alone & set(_shown_layers(glued, TAIL)) == set()


def _leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _leaves(item)
    elif isinstance(value, str):
        yield value


def _parses(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


_NESTED = [
    {"api_key": HEAD + TAIL},
    {"api_key": HEAD + "\\\\x41" + TAIL, "next": "kept"},
    {"password": HEAD + '"' + TAIL, "n": 1},
    {"cmd": f'curl -H "Authorization: Bearer {HEAD}{TAIL}" https://example.test'},
    {
        "headers": {"Authorization": "Bearer " + HEAD + TAIL},
        "token": HEAD + "\\\\" + TAIL,
    },
    {"env": f"GITHUB_TOKEN={HEAD}\\nHOME=/root", "password": HEAD + "\\\\"},
]


@pytest.mark.parametrize("doc", _NESTED)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_json_inside_a_string_still_parses_in_every_layer(doc, depth):
    # Every redacted string leaf that parsed as JSON before still parses
    # after, in every layer (review R632: the pairs before an escaped
    # quote were read, which broke the inner document's own escape).
    leaf = json.dumps(doc)
    for _ in range(depth - 1):
        leaf = json.dumps({"body": leaf})
    record = {"output": leaf}
    outputs = {
        "text": redact_trajectory_text(leaf),
        "record": json.loads(redact_trajectory_text(json.dumps(record)))["output"],
        "object": redact_trajectory_obj(record)["output"],
        "acp": json.loads(
            redact_acp_trajectory_jsonl(
                [
                    {
                        "type": "tool_call_update",
                        "content": [{"type": "text", "text": leaf}],
                    }
                ]
            )
        )["content"][0]["text"],
        "publish": redact_value(record)[0]["output"],
    }
    for layer, out in outputs.items():
        assert _parses(out), layer
        inner = json.loads(out)
        while isinstance(inner, dict) and set(inner) == {"body"}:
            assert _parses(inner["body"]), layer
            inner = json.loads(inner["body"])
        assert [x for x in _leaves(inner) if TAIL in x] == [], layer


@pytest.mark.parametrize(
    "text",
    [
        "password = '''" + HEAD + TAIL + "'''",
        "DB_PASSWORD='''" + HEAD + TAIL + "'''",
    ],
)
def test_a_triple_quoted_value_is_masked(text):
    assert _shown_layers(text, HEAD, TAIL) == []


def test_a_triple_double_quoted_value_is_masked_in_its_copies():
    # As written, a double quote is never read (R1); in a json.dumps copy the
    # quotes are escapes, read as the value's own.
    text = 'GITHUB_TOKEN = """' + HEAD + TAIL + '"""'
    shown = _shown_layers(text, HEAD, TAIL)
    assert set(shown) <= {"text", "publish", "result"}, shown


# The glue guard stops a value only before a credential's word that starts a
# name's segment (the run's start, after anything but a letter, or a
# lower-case letter turning upper-case), or before a name that ends in a
# carrier's own ending (the lead's ruling, 2026-11-15 15:12Z): a word inside
# a word (`monkey=`, `turkey=`) no longer cuts a value short.
@pytest.mark.parametrize(
    "word", ["monkey", "turkey", "Monkey", "TURKEY", "donkey", "bypass"]
)
@pytest.mark.parametrize("piece", ["\\x41", "\\X", "\\u0041"], ids=["hex", "X", "u"])
def test_a_word_inside_a_word_does_not_cut_a_value_short(word, piece):
    text = f"password={HEAD}{piece}{word}=1{TAIL}"
    assert _shown_layers(text, HEAD, TAIL) == []


@pytest.mark.parametrize(
    "name",
    [
        "MY_API_KEY=",
        "x-api-key: ",
        "apiKey: ",
        "ghToken=",
        "\\u001b[32mGITHUB_TOKEN: ",
        "GITHUBTOKEN: ",
        "APIKEY=",
        "dbpassword: ",
    ],
)
@pytest.mark.parametrize("piece", ["\\x41", "\\X", "\\\\"], ids=["hex", "X", "two"])
def test_a_second_carrier_after_an_escape_still_stops_the_value(name, piece):
    # Its value stays masked, as the base masked it once the first value
    # stopped at the backslash.
    text = f"password={HEAD}{piece}{name}{TAIL}"
    assert _shown_layers(text, TAIL) == []


# The shape grammar's glue (FXCANON, 2026-11-15 15:47Z): a second carrier
# whose value is on a later line, a second name written with escapes before
# `Bearer <token>`, a glued key name after a quote. Wherever the second entry
# is masked standing alone, a first value before it never unmasks it. The
# token is 20 characters: publish masks a bearer token from 16, so each row
# is masked standing alone in some layer (review R632, 16:26Z).
@pytest.mark.parametrize(
    ("first", "rest"),
    [
        (f"api_key={HEAD}\\n", f"\\u001b[32mAWS_SECRET_ACCESS_KEY=\n\n{TAIL20}"),
        (f"password={HEAD}|", f"\\u001b[32mdbpassword =\n\n{TAIL20}"),
        (f"password={HEAD}%22", f"\\e[0mgithubtoken =\n{TAIL20}"),
        (f"password={HEAD}", f"\\&#34;'encryption_key' :'{TAIL20}'"),
        (f"token: {HEAD};", f"\\u{{41}}uthorization=Bearer {TAIL20}"),
        (f"password={HEAD}\\n", f"Authorizatio%5Cx6e=Bearer {TAIL20}"),
        (f"password={HEAD}|", f"Authorization\\x3dBearer {TAIL20}"),
    ],
)
def test_a_second_carrier_keeps_its_value_masked_after_a_first_value(first, rest):
    masked_alone = set(_layers(rest)) - set(_shown_layers(rest, TAIL20))
    assert masked_alone  # each row is masked standing alone in some layer
    assert masked_alone & set(_shown_layers(first + rest, TAIL20)) == set()


# EN-697: a credential name written inside two or three quote marks, in any
# spelling a copy or a log writes them.
_QUOTE_SPELLINGS = [
    '"',
    "'",
    '\\"',
    '\\\\\\"',
    "&quot;",
    "&#34;",
    "%22",
    "\\x22",
    "\\u0022",
]


@pytest.mark.parametrize("quote", _QUOTE_SPELLINGS)
@pytest.mark.parametrize("times", [1, 2, 3])
@pytest.mark.parametrize(
    "name", ["password", "api_key", "GITHUB_TOKEN", "client_secret"]
)
@pytest.mark.parametrize("separator", [": ", "="])
def test_a_name_inside_quote_marks_carries_its_value(quote, times, name, separator):
    q = quote * times
    assert _shown_layers(f"{q}{name}{q}{separator}{TAIL}", TAIL) == []


# EN-697: a value followed directly by `;` or `|` and another credential name.
@pytest.mark.parametrize("joiner", [";", "|"])
@pytest.mark.parametrize(
    "second",
    ["GITHUB_TOKEN: ", "password : ", "apiKey: ", "client_secret= ", "\\x1b[1mtoken: "],
)
def test_a_second_entry_after_a_semicolon_keeps_its_value_masked(joiner, second):
    assert _shown_layers(f"password={HEAD}{joiner}{second}{TAIL}", HEAD, TAIL) == []


@pytest.mark.parametrize(
    "text",
    [
        f"password=pa;ss{HEAD}{TAIL}",
        f"password=pa|ss{HEAD}{TAIL}",
        f"password={HEAD};GITHUB_TOKEN={TAIL}",
        f"password={HEAD};%61pi_key={TAIL}",
        f"password={HEAD}|client_secret%3d{TAIL}",
    ],
)
def test_a_semicolon_inside_a_value_keeps_it_whole(text):
    assert _shown_layers(text, HEAD, TAIL) == []


def test_a_semicolon_before_a_plain_name_keeps_the_value_whole():
    # `monkey` holds no credential's word where a name's segment starts, so
    # the value reads on through `;monkey:` as before (and TAIL is no value).
    assert _shown_layers(f"password={HEAD};monkey: {TAIL}", HEAD) == []
    assert "monkey" not in redact_trajectory_text(f"password={HEAD};monkey: {TAIL}")


# EN-697: Authorization right after a terminal colour code.
@pytest.mark.parametrize(
    "code", ["\\033[1m", "\x1b[1m", "\\u001b[32m", "\\e[0m", "\\x1b[0m"]
)
@pytest.mark.parametrize("value", [f"Bearer {TAIL}", TAIL], ids=["scheme", "bare"])
def test_authorization_after_a_colour_code_is_masked(code, value):
    assert _shown_layers(f"{code}Authorization: {value}", TAIL) == []


# The glue guard and the carriers' tables (F2b, 2026-11-15 16:06Z): the
# guard's list of glued name endings is written by hand, so every name
# ending this line's tables hold (the carriers' and the classifier's), glued
# to a prefix, in any case, with or without its `_`, is one the guard stops
# a value before; and every name a carrier masks on its own stays masked
# right after an escape. A table that gains a name cannot reopen the glue.
def _expand(pattern: str) -> list[str]:
    """Every string a finite pattern of literals, classes, groups, `|` and `?` matches."""
    import re._constants as c
    import re._parser as p

    def walk(items):
        outs = [""]
        for op, av in items:
            if op is c.LITERAL:
                alts = [chr(av)]
            elif op is c.IN:
                alts = [chr(v) for o, v in av if o is c.LITERAL]
                alts += [
                    chr(x) for o, v in av if o is c.RANGE for x in range(v[0], v[1] + 1)
                ]
            elif op is c.BRANCH:
                alts = [s for branch in av[1] for s in walk(branch)]
            elif op is c.SUBPATTERN:
                alts = walk(av[3])
            elif op in (c.MAX_REPEAT, c.MIN_REPEAT):
                low, high, sub = av
                assert high <= 1, pattern
                alts = ([""] if low == 0 else []) + walk(sub)
            else:
                raise ValueError((pattern, op))
            outs = [o + a for o in outs for a in alts]
        return outs

    return walk(p.parse(pattern))


def _table_endings() -> list[str]:
    import benchflow.trajectories.types as t

    endings, _words = _line_tables()
    sources = ["TOKEN", *endings]
    for name in (
        "_SECRET_NAME_ENDINGS",
        "_PASSPHRASE_END",
        "_API_KEY_NAME",
        "_KEY_FIELD_NAME",
    ):
        if isinstance(getattr(t, name, None), str):
            sources.append(getattr(t, name))
    for pattern, _replacement, _category in t._REDACTION_PATTERNS:
        text = getattr(pattern, "pattern", "")
        sources += [m.group(1) for m in re.finditer(r"\(\?:TOKEN\|([A-Z_|?]+)\)", text)]
    return sorted({e for s in sources for e in _expand(s)})


def test_every_carrier_name_ending_stops_a_glued_value():
    from benchflow.trajectories.types import _GLUE_AHEAD

    guard = re.compile(_GLUE_AHEAD, re.IGNORECASE)
    endings = _table_endings()
    assert len(endings) >= 10, endings
    bad = []
    for ending in endings:
        for glued in {ending, ending.replace("_", "").replace("-", "")}:
            for name in (
                f"myapp{glued.lower()}",
                f"MYAPP{glued.upper()}",
                f"myApp{glued}",
            ):
                for separator in (": x", "=x", " =\nx", '": "x'):
                    if not guard.match(name + separator):
                        bad.append(name + separator)
    assert bad == []


VALUE32 = "Fx688kEyR4nD0mS7qW2eZ9xC1vB5nM3a"
_CREDENTIAL_WORDS_SEEN = (
    "secret",
    "token",
    "key",
    "password",
    "passwd",
    "pwd",
    "pass",
    "auth",
    "cookie",
    "credential",
    "credentials",
    "bearer",
    "authorization",
    "passphrase",
    "session",
    "sessionid",
    "sid",
    "proxy_authorization",
    "proxy-authorization",
    "primary_key",
    "secondary_key",
    "api_key",
    "apikey",
    "access_key",
    "secret_key",
    "private_key",
)


def _carrier_names() -> list[str]:
    _endings, words = _line_tables()
    names = set()
    for word in {*_CREDENTIAL_WORDS_SEEN, *words}:
        names |= {
            f"MY_{word.upper()}",
            f"my.{word}",
            f"my-{word}",
            f"x_{word}",
            word.upper(),
        }
        names.add("my" + word[:1].upper() + word[1:])
    for ending in _table_endings():
        for glued in {ending, ending.replace("_", "")}:
            names |= {f"myapp{glued.lower()}", f"MYAPP{glued.upper()}", f"myApp{glued}"}
    return sorted(names)


# A name a carrier masks right after an escape (`\x41GITHUB_TOKEN: v`) is
# masked as well after a value that runs into that escape: the value never
# reads the name. (Whether a carrier starts after an escape is its own rule;
# the property is that a value before it changes nothing.)
@pytest.mark.parametrize(
    "escape", ["\\x41", "\\X", "\\u0041", "\\101", "\\\\x41", "\\u0026", "\\x20", "\\t"]
)
def test_a_name_masked_after_an_escape_stays_masked_after_a_value(escape):
    alone = [
        n
        for n in _carrier_names()
        if _shown_layers(f"{escape}{n}: {VALUE32}", VALUE32) == []
    ]
    assert len(alone) >= 40, alone
    bad = [
        (name, layers)
        for name in alone
        if (
            layers := _shown_layers(
                f"password={HEAD}{escape}{name}: {VALUE32}", VALUE32
            )
        )
    ]
    assert bad == []


def _line_tables() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """0.9.x has no classifier: its carriers' own tables are all."""
    return (), ()


_RUNS_697 = {
    "quoted names": lambda n: '""password"": v ' * (n // 16),
    "semicolons": lambda n: "password=v" + ";x" * (n // 2),
    "semicolon entries": lambda n: "password=v;GITHUB_TOKEN: w " * (n // 27),
    "colour codes": lambda n: "\\033[1mAuthorization: Bearer v " * (n // 31),
}


@pytest.mark.parametrize("make", list(_RUNS_697.values()), ids=list(_RUNS_697))
def test_the_en697_shapes_are_read_in_linear_time(make):
    small, large = make(20_000), make(80_000)
    assert _cpu_ratio(redact_trajectory_text, small, large) < 8


# 0.9.x's Authorization carriers start after an escape that ends a word, as
# _WORD_START reads one: a JSON copy of headers (`Host: x\r\nAuthorization:
# Bearer <token>`), an escaped tab, space or `&`. Before (0.9.2, 040c1d0c1)
# the value was shown after every one of them.
@pytest.mark.parametrize(
    "escape", ["\\n", "\\r\\n", "\\t", "\\x20", "\\u0020", "\\u0026", "\\\\n", "\\\\t"]
)
@pytest.mark.parametrize("name", ["Authorization", "AUTHORIZATION"])
@pytest.mark.parametrize(
    "value",
    [f"Bearer {VALUE32}", VALUE32, f"Basic {VALUE32}"],
    ids=["bearer", "bare", "basic"],
)
def test_authorization_after_an_escape_that_ends_a_word_is_masked(escape, name, value):
    assert VALUE32 not in redact_trajectory_text(f"x{escape}{name}: {value}")
    assert _shown_layers(f"x{escape}{name}: {value}", VALUE32) == []


def test_a_json_copy_of_headers_keeps_its_authorization_masked():
    headers = f"Host: example.test\r\nAuthorization: Bearer {VALUE32}\r\nAccept: */*"
    for text in (
        json.dumps({"output": headers}),
        json.dumps({"headers": f"x\t{headers}"}),
    ):
        out = redact_trajectory_text(text)
        assert VALUE32 not in out
        assert json.loads(out)


# An escape that writes a letter or a digit continues the word it is in, so
# `\x41Authorization` is the name `AAuthorization`. It ends in Authorization,
# so it is that header's carrier (F4, any prefix), and its value is masked.
@pytest.mark.parametrize("escape", ["\\x41", "\\X", "\\101"])
def test_an_escaped_letter_before_authorization_joins_a_carrier_name(escape):
    assert _shown_layers(f"x{escape}Authorization: Bearer {VALUE32}", VALUE32) == []


# Review R632 over R528's 21,091 names (2026-11-15 16:26Z): a name that
# starts with a credential's word right after an ANSI code
# (`\u001b[32mpassword2:`), or ends in one and a number (`PGPASSWORD2:`,
# `admintoken_2:`), stops a value before it, as it does standing alone.
_NUMBERED_WORDS = (
    "password",
    "token",
    "secret",
    "passphrase",
    "credential",
    "passvalue",
    "key",
)
_NUMBERED_TAILS = ("", "2", "_2", "2_url", "_v2_name", "s12345_b64")
_NUMBERED_HEADS = ("", "get", "PG", "db_")
_PIECES_BEFORE = ("\\u001b[32m", "\\x1b[1;31m", "\\e[0m", "\x1b[32m", "\\X", "\\u0026")


def _numbered_names() -> list[str]:
    names = set()
    for word in _NUMBERED_WORDS:
        for tail in _NUMBERED_TAILS:
            for head in _NUMBERED_HEADS:
                names |= {(head + word + tail).lower(), (head + word + tail).upper()}
    return sorted(names)


@pytest.mark.parametrize("piece", _PIECES_BEFORE)
def test_a_numbered_or_coloured_name_stays_masked_after_a_value(piece):
    alone = [
        n
        for n in _numbered_names()
        if _shown_layers(f"{piece}{n}: {VALUE32}", VALUE32) == []
    ]
    assert len(alone) >= 20, alone
    bad = [
        (name, layers)
        for name in alone
        if (
            layers := _shown_layers(f"password={HEAD}{piece}{name}: {VALUE32}", VALUE32)
        )
    ]
    assert bad == []


# The property over the shape grammar's carriers (v2, FXCANON; the lead,
# 2026-11-15 16:33Z): every carrier and spelling it writes, in every case
# style and after every colour code, masked standing alone, is masked after a
# first value and each piece between them. (The full cross of quotes,
# separators and first entries runs as a gate on the shared grammar.)
_GRAMMAR_CARRIERS = (
    ("api_key", ""),
    ("apiKey", ""),
    ("x-api-key", ""),
    ("x-goog-api-key", ""),
    ("api-key", ""),
    ("master_key", ""),
    ("private_key", ""),
    ("AWS_BEARER_TOKEN_BEDROCK", ""),
    ("GITHUB_TOKEN", ""),
    ("token", ""),
    ("client_secret", ""),
    ("secret", ""),
    ("DB_PASSWORD", ""),
    ("password", ""),
    ("PASSWD", ""),
    ("AWS_SECRET_ACCESS_KEY", ""),
    ("AZURE_STORAGE_ACCOUNT_KEY", ""),
    ("ENCRYPTION_KEY", ""),
    ("GOOGLE_CREDENTIALS", ""),
    ("credential", ""),
    ("Authorization", "Bearer "),
    ("Authorization", ""),
    *(
        (spelling, scheme)
        for spelling in (
            "Proxy-Authorization",
            "proxy_authorization",
            "proxyAuthorization",
            "PROXY_AUTHORIZATION",
            "proxy-authorization",
        )
        for scheme in ("Bearer ", "Basic ", "")
    ),
    ("primary_key", ""),
    ("primaryKey", ""),
    ("secondaryMasterKey", ""),
)
_GRAMMAR_PIECES = (
    "&",
    "\\n",
    "\n",
    "\\t",
    "\t",
    " ",
    "\\u001b[32m",
    "\x1b[32m",
    "\\&#34;",
    "&#34;",
    "&quot;",
    '\\"',
    "%22",
    ";",
    "|",
    "\\r",
    " please retry&",
    "\\x1b[0m",
    ",",
)
_COLOUR_CODES = (
    "",
    "\\u001b[32m",
    "\\x1b[0m",
    "\x1b[32m",
    "\\033[1m",
    "\\e[0m",
    "^[[0m",
)


def _case_styles(name: str) -> set[str]:
    words = [
        w.lower()
        for w in re.findall(
            r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[A-Z]+|[0-9]+",
            name.replace("-", "_").replace("_", " "),
        )
    ]
    return {
        name,
        name.lower(),
        name.upper(),
        words[0] + "".join(w.title() for w in words[1:]),
        "_".join(words),
        "-".join(words),
        "".join(words),
    }


def _grammar_entries() -> list[str]:
    entries = set()
    for name, scheme in _GRAMMAR_CARRIERS:
        for spelled in _case_styles(name):
            entries.add(f"{spelled}={scheme}{VALUE32}")
        for code in _COLOUR_CODES:
            entries.add(f"{code}{name}: {scheme}{VALUE32}")
    return sorted(entries)


@pytest.mark.parametrize("piece", _GRAMMAR_PIECES)
def test_a_grammar_carrier_masked_alone_stays_masked_after_a_first_value(piece):
    alone = [e for e in _grammar_entries() if _shown_layers(e, VALUE32) == []]
    assert len(alone) >= 150, len(alone)
    bad = [
        (entry, layers)
        for entry in alone
        if (layers := _shown_layers(f"password={HEAD}{piece}{entry}", VALUE32))
    ]
    assert bad == []


# Proxy-Authorization is a carrier, as Authorization (the lead's ruling,
# 2026-11-15 16:24Z), in every spelling, with Bearer, Basic and bare values.
_BASIC = "Zng2ODh1c2VyOkZ4Njg4cEFzU3dPckQ3cVo="


@pytest.mark.parametrize(
    "name",
    [
        "Proxy-Authorization",
        "proxy_authorization",
        "proxyAuthorization",
        "PROXY_AUTHORIZATION",
        "proxy-authorization",
    ],
)
@pytest.mark.parametrize(
    ("value", "secret"),
    [(f"Bearer {VALUE32}", VALUE32), (f"Basic {_BASIC}", _BASIC), (VALUE32, VALUE32)],
    ids=["bearer", "basic", "bare"],
)
@pytest.mark.parametrize(
    "write",
    [
        lambda n, v: f"{n}: {v}",
        lambda n, v: f"GET / HTTP/1.1\r\nHost: example.test\r\n{n}: {v}\r\nAccept: */*",
        lambda n, v: json.dumps({n: v}),
        lambda n, v: f"curl -H '{n}: {v}' https://example.test/",
        lambda n, v: f"{n}={v}",
        lambda n, v: f"x\\n{n}: {v}",
    ],
    ids=["line", "headers", "json", "curl", "env", "escaped-break"],
)
def test_proxy_authorization_is_a_carrier(name, value, secret, write):
    assert _shown_layers(write(name, value), secret) == []


# Azure's and Cosmos's primary and secondary keys are credentials (the
# lead's ruling, 2026-11-15 16:34Z), under a key-shaped value, as the rc
# reads them; a database's primary key is none.
_AZURE_KEY = "Fx688AzUrEkEy0vB5nM3aQ1wE2rT4yU6iO8pAsDfGhJkLzXcVbNm1234567890ab+cd/efGHijklMNopqrST=="
_KEY_SIZES = {"azure": _AZURE_KEY, "key32": VALUE32}
_KEY_FORMS = {
    "env": lambda n, v: f"{n}={v}",
    "json": lambda n, v: json.dumps({n: v}),
    "yaml": lambda n, v: f"{n}: {v}",
}


@pytest.mark.parametrize(
    "name",
    [
        "primary_key",
        "primaryKey",
        "PrimaryKey",
        "PRIMARY_KEY",
        "primary-key",
        "secondary_key",
        "secondaryKey",
        "SecondaryKey",
        "SECONDARY_KEY",
        "secondary-key",
        "primaryMasterKey",
        "secondaryMasterKey",
        "AZURE_STORAGE_PRIMARY_KEY",
    ],
)
@pytest.mark.parametrize("size", list(_KEY_SIZES))
@pytest.mark.parametrize("form", list(_KEY_FORMS))
def test_a_primary_or_secondary_key_is_masked(name, size, form):
    if (form, size) in _KEY_FORMS_KEPT:
        pytest.skip("the rc keeps its own masking here (the lead's ruling, 16:34Z)")
    secret = _KEY_SIZES[size]
    assert _shown_layers(_KEY_FORMS[form](name, secret), secret) == []


@pytest.mark.parametrize(
    "text",
    [
        "models.AutoField(primary_key=True)",
        '{"primaryKey": "id"}',
        "primary_key: user_id",
    ],
)
def test_a_database_primary_key_is_no_credential(text):
    assert redact_trajectory_text(text) == text


# Grammar v2's Proxy-Authorization family (FX688, 17:2xZ): a scheme after a
# blank line in a json.dumps copy, a run of colour codes before the name, a
# URL-encoded backslash before a braced escape in the value.
@pytest.mark.parametrize(
    "name", ["proxy_authorization", "Proxy-Authorization", "Authorization"]
)
@pytest.mark.parametrize("separator", ["=", ":", "%3d"])
@pytest.mark.parametrize("breaks", ["\n", "\n\n", "\r\n\r\n"])
def test_a_scheme_after_blank_lines_keeps_its_token_masked(name, separator, breaks):
    if separator in _SEPARATORS_NOT_READ:
        pytest.skip("this line reads no URL-encoded separator after Authorization")
    text = f"password={HEAD}\t{name}{separator}{breaks}Bearer {VALUE32}"
    out = redact_trajectory_text(json.dumps({"output": text}))
    assert VALUE32 not in out
    assert json.loads(out)


@pytest.mark.parametrize(
    "codes",
    [
        "\x1b[32m\x1b[32m",
        "\\u001b[32m\\u001b[1m",
        "\\e[0m\\e[1m",
        "\x1b[1m\x1b[4m\x1b[32m",
    ],
)
def test_a_first_value_stops_before_a_run_of_colour_codes_and_a_name(codes):
    text = f"password={HEAD}{codes}GITHUB_TOKEN: {VALUE32}"
    assert _shown_layers(text, VALUE32) == []
    assert _shown_layers(text, HEAD) == []


@pytest.mark.parametrize("escape", ["%5Cu{79}", "%255Cx{41}", "%5CU{1F600}", "\\u{79}"])
@pytest.mark.parametrize(
    "name", ["Authorization", "proxy_authorization", "password", "token"]
)
def test_a_braced_escape_is_part_of_the_value(escape, name):
    assert _shown_layers(f"{name}: {escape}{VALUE32}", VALUE32) == []


# FXCANON's cellgate on #274 (2026-11-16 04:51Z): an Authorization-family
# value (Proxy-Authorization's included) never reads the next
# Authorization-family entry where that entry's own carrier can start: the
# second value, masked standing alone, stays masked after the first entry.
_AUTH_FIRSTS = (
    "Proxy-Authorization",
    "proxy_authorization",
    "proxyAuthorization",
    "Authorization",
    "&#34;Authorization&#34;",
    "&quot;Authorization&quot;",
    '"Authorization&quot;',
    "&#x22;Authorization&#x22;",
)
_AUTH_JOINTS = (
    ".",
    ";",
    "/",
    ",",
    "'",
    '"',
    "&quot;",
    "&#34;",
    "none.",
    "none.'",
    "none-",
    "none_",
    "",
)
_AUTH_SECONDS = (
    "Authorization: {v}",
    '"authorization"={v}',
    "'Authorization: {v}",
    "authorization={v}",
    "AUTHORIZATION: {v}",
    "Proxy-Authorization: {v}",
    "Authorization: Bearer {v}",
    'Authorization: "{v}"',
    'authorization"={v}',
)


@pytest.mark.parametrize("first", _AUTH_FIRSTS)
@pytest.mark.parametrize("joint", _AUTH_JOINTS)
def test_an_authorization_value_never_reads_the_next_authorization_entry(first, joint):
    bad = []
    for second in _AUTH_SECONDS:
        alone = joint + second.format(v=VALUE32)
        masked_alone = set(_layers(alone)) - set(_shown_layers(alone, VALUE32))
        if shown := masked_alone & set(_shown_layers(f"{first}: {alone}", VALUE32)):
            bad.append((second, sorted(shown)))
    assert bad == []


def test_an_authorization_value_reads_on_where_no_carrier_can_start():
    # `none-authorization=<v>`: no carrier starts after `-`, so the value
    # reads on and masks it whole, as 040c1d0c1 did.
    for first in ("Authorization", "Proxy-Authorization"):
        text = f"{first}: none-authorization={VALUE32}"
        assert _shown_layers(text, VALUE32) == []


# Review R632 (2026-11-15 22:00Z) over R528's names_all: a name the rc's
# classifier decides by its value, a credential's word anywhere in it and a
# number after (`FAKEsession0123456`, `fakekeyid0123456789`), right after an
# escape that writes a letter; and a quoted entry of any name right after an
# escaped quote and more quotes (`\&#34;''FAKEyaml0123': '<v>'`). Wherever
# the entry is masked standing alone, a first value before it leaves it so.
_DECIDED_NAMES = (
    "FAKEsession0123456",
    "fakekeyid0123456789",
    "db_token_v2",
    "API_SECRET2",
    "session_key_hint",
    "my_credential3",
    "token_2fa_backup",
)
_DECIDED_VALUES = (
    "1dx8PpD5KHXe3PYi",
    "08dWgW5hQxdGlvxz",
    "3SCb8FNVgnX6JVR7siFTrUTL",
    "1NIuk8Io5yaGKbXJD2aSJd7n2Khd9fsuUeRSeZ5J",
    "Fx688kEyR4nD0mS7qW2eZ9xC1vB5nM3a",
)


@pytest.mark.parametrize("name", _DECIDED_NAMES)
@pytest.mark.parametrize(
    "write",
    [
        lambda n, v: (f"\\X{n}: {v}", f"password={HEAD}\\X{n}: {v}"),
        lambda n, v: (f"\\x41{n}: {v}", f"password={HEAD}\\x41{n}: {v}"),
        lambda n, v: (f"\\&#34;''{n}': '{v}'", f"password={HEAD}\\&#34;''{n}': '{v}'"),
        lambda n, v: (f"\\\"'{n}': '{v}'", f"password={HEAD}\\\"'{n}': '{v}'"),
    ],
    ids=["escaped-letter", "hex-letter", "entity-quotes", "escaped-quotes"],
)
def test_a_decided_or_quoted_entry_stays_masked_after_a_first_value(name, write):
    for value in _DECIDED_VALUES:
        alone, glued = write(name, value)
        masked_alone = set(_layers(alone)) - set(_shown_layers(alone, value))
        assert masked_alone & set(_shown_layers(glued, value)) == set(), (name, value)


# F4 (review R632 on #274; the lead via F2b, 2026-11-16 06:45Z): a name that
# ends in Authorization or Proxy-Authorization after any prefix is that
# header's carrier, as the rc reads it.
@pytest.mark.parametrize(
    "name",
    [
        "HTTP_PROXY_AUTHORIZATION",
        "X-Proxy-Authorization",
        "HTTP_AUTHORIZATION",
        "X-Authorization",
        "x_authorization",
        "REDIRECT_HTTP_AUTHORIZATION",
        "basicAuthorization",
    ],
)
@pytest.mark.parametrize(
    "write",
    [
        lambda n, v: f"{n}: {v}",
        lambda n, v: f"{n}={v}",
        lambda n, v: f"{n}: Bearer {v}",
        lambda n, v: json.dumps({n: v}),
        lambda n, v: f"x\\n{n}: {v}",
    ],
    ids=["line", "env", "bearer", "json", "escaped-break"],
)
def test_a_prefixed_authorization_name_is_a_carrier(name, write):
    assert _shown_layers(write(name, VALUE32), VALUE32) == []


def test_a_name_that_goes_on_past_authorization_is_no_carrier():
    text = "X-Authorization-Mode: strict"
    assert redact_trajectory_text(text) == text


# The docstring guard (the lead, 2026-11-16 04:20Z): after a separator, a
# value on a later line does not open with three quotes, so a class and its
# docstring are no entry; a value on its own line, or a later line's value
# that opens otherwise, is masked as before.
@pytest.mark.parametrize("quotes", ['"""', "'''"])
def test_a_docstring_after_a_class_line_is_no_value(quotes):
    text = f"class Token:\n    {quotes}Immutable token representation.{quotes}\n    value: str\n"
    assert redact_trajectory_text(text) == text


@pytest.mark.parametrize(
    "text",
    [
        f"token:\n    {VALUE32}\n",
        f'token:\n    "{VALUE32}"\n',
        f'token: """{VALUE32}"""',
        f"API_TOKEN=\n{VALUE32}",
    ],
)
def test_a_value_after_a_separator_is_still_masked(text):
    assert VALUE32 not in redact_trajectory_text(text)


# The BEARER_TOKEN carrier reads a scheme word before its token, as the rc's
# {_SCHEME} does (F2b, 2026-11-16 06:57Z): the token is masked, the scheme
# kept; a bare scheme word with no token after it is the value, as before.
@pytest.mark.parametrize(
    "text",
    [
        f"BEARER_TOKEN_AUTHORIZATION: Bearer {VALUE32}",
        f"AWS_BEARER_TOKEN_BEDROCK: Bearer {VALUE32}",
        f"MY_BEARER_TOKEN=Token {VALUE32}",
        f"BEARER_TOKEN: {VALUE32}",
        f"BEARER_TOKEN={VALUE32}",
    ],
)
def test_a_bearer_token_name_masks_the_token_after_its_scheme(text):
    assert _shown_layers(text, VALUE32) == []


def test_a_bare_scheme_word_is_the_bearer_token_value():
    assert (
        redact_trajectory_text("BEARER_TOKEN=Bearer") == "BEARER_TOKEN=***REDACTED***"
    )


# Review R632 on R7 (2026-11-16 07:08Z): the rc's decided names and quoted
# entries of any name are no carriers on 0.9.x, so 0.9.x's block keeps no stop
# for them; a value reads on through them and masks what 94e7736d5 masks. A
# real carrier's name in the same place stops the value and keeps its own
# carrier.
@pytest.mark.parametrize(
    "text",
    [
        f"password={HEAD}\\Xmonkey9={TAIL}",
        f"password={HEAD}\\XFAKEsession0123456={TAIL}",
        f"password=ab\\'cd:{TAIL}",
    ],
)
def test_a_value_reads_on_through_a_name_no_carrier_masks(text):
    assert _shown_layers(text, TAIL) == []


@pytest.mark.parametrize(
    "text",
    [
        f"password={HEAD}\\XGITHUB_TOKEN: {TAIL}",
        f"password={HEAD}\\Xapi_key: {TAIL}",
        f"password=ab\\'GITHUB_TOKEN': {TAIL}",
        f"password=ab\\'password: {TAIL}",
    ],
)
def test_a_carrier_name_in_the_same_place_keeps_its_value_masked(text):
    assert _shown_layers(text, TAIL) == []


# The forms whose masking this line keeps as it was (the lead, 16:34Z): the rc
# masks a 32-character key under `primary_key: ` only as an env entry or JSON.
_KEY_FORMS_KEPT = ()
# The separators this line's Authorization carriers do not read at all.
_SEPARATORS_NOT_READ = ("%3d",)


# Review R632 (2026-11-16): a triple-quoted value on the line after a
# credential name's separator is the value, masked as 94e7736d5 masked it;
# only a class's or a def's header makes the next line's three quotes a
# docstring.
@pytest.mark.parametrize("quote", ["'''", '"""'])
@pytest.mark.parametrize(
    "name", ["password", "token", "api_key", "GITHUB_TOKEN", "client_secret"]
)
@pytest.mark.parametrize("separator", [":", "=", ": "])
def test_a_triple_quoted_value_on_the_next_line_is_masked(quote, name, separator):
    text = f"{name}{separator}\n    {quote}{TAIL20}{quote}"
    assert {"text", "publish"} & set(_shown_layers(text, TAIL20)) == set()


@pytest.mark.parametrize("quote", ["'''", '"""'])
@pytest.mark.parametrize(
    "header",
    [
        "class Token:",
        "class AccessToken:",
        "def get(self) -> Token:",
        "    def token(self) -> Secret:",
    ],
)
def test_a_docstring_after_a_class_or_def_header_is_kept(quote, header):
    text = f"{header}\n    {quote}Immutable token representation.{quote}\n"
    assert redact_trajectory_text(text) == text


# R632's N3 (F2b, 2026-11-16 07:53Z): an entry whose separator has a space
# before it or a line break after it (`NAME :v`, `NAME = v`, `NAME =\n\nv`),
# after `;`, `|`, `%22`, a colour code, or glued onto the first value with no
# boundary at all: wherever the entry is masked standing alone, the first
# value before it leaves it so.
_N3_NAMES = (
    "token",
    "GITHUB_TOKEN",
    "client_secret",
    "password",
    "Accesskey",
    "ENCRYPTION_KEY",
    "credentials",
    "Authorization",
    "PROXY-AUTHORIZATION",
)


def _n3_bad(names, joiner, separator):
    bad = []
    for name in names:
        for value in (TAIL20, f"Bearer {TAIL20}"):
            alone = f"{joiner}{name}{separator}{value}"
            masked_alone = set(_layers(alone)) - set(_shown_layers(alone, TAIL20))
            after = set(_shown_layers(f"password={HEAD}{alone}", TAIL20))
            if masked_alone & after:
                bad.append((name, value[:6], sorted(masked_alone & after)))
    return bad


@pytest.mark.parametrize(
    "joiner",
    [";", "|", "%22", "\x1b[32m", "\\x1b[32m", "^[[0m", ""],
    ids=["semi", "pipe", "pct22", "esc", "escaped-esc", "caret", "glued"],
)
@pytest.mark.parametrize(
    "separator",
    [" :", " = ", ": ", "\t: ", " =\n\n", "=\n\n", ":\n"],
    ids=[
        "sp-colon",
        "sp-eq",
        "colon",
        "tab-colon",
        "sp-eq-blank",
        "eq-blank",
        "colon-next",
    ],
)
def test_an_entry_masked_alone_stays_masked_after_a_first_value(joiner, separator):
    assert _n3_bad(_N3_NAMES, joiner, separator) == []


def test_every_carrier_name_ending_glued_onto_a_value_leaves_the_next_masked():
    # Every name ending this line's tables hold, glued onto a first value with
    # no boundary, in lower or upper case, before a separator with a space.
    names = sorted(
        {
            spelling
            for ending in _table_endings()
            for glued in {ending, ending.replace("_", "").replace("-", "")}
            for spelling in (glued.lower(), glued.upper())
        }
    )
    assert len(names) >= 20, names
    bad = [
        cell
        for separator in (" :", " = ", ": ", " =\n\n")
        for cell in _n3_bad(names, "", separator)
    ]
    assert bad == []


def test_the_glued_endings_spell_out_every_ending_the_guard_reads():
    from benchflow.trajectories.types import _GLUE_END, _GLUED_END_WORDS

    words = {w.lower() for w in _GLUED_END_WORDS}
    assert {e.lower() for e in _expand(_GLUE_END)} <= words


@pytest.mark.parametrize(
    "text",
    [
        f"password={HEAD}monkey :{TAIL20}",
        f"password={HEAD}token:\nHOME=/{TAIL20}",
        f'password={HEAD}Token:\n    """{TAIL20}"""',
        f"password={HEAD}token, {TAIL20}",
    ],
    ids=["no-carrier-ending", "next-line-entry", "docstring", "no-separator"],
)
def test_a_glued_value_reads_no_further_than_a_carrier_s_next_value(text):
    out = redact_trajectory_text(text)
    assert HEAD not in out
    assert TAIL20 in out


# The same after an escape inside the first value, the name glued onto the
# rest of its run (`password=<v1>\X<v1>authorization=<v2>`; F2b 09:06Z: an
# escape piece is in scope): the entry standing alone is the measure.
@pytest.mark.parametrize(
    "escape",
    ["\\X", "\\x41", "\\u0041", "\\101", "\\\\X"],
    ids=["letter", "hex", "u4", "octal", "doubled"],
)
@pytest.mark.parametrize(
    "separator",
    [": ", "=", " :", " = ", " =\n\n"],
    ids=["colon", "eq", "sp-colon", "sp-eq", "sp-eq-blank"],
)
def test_an_entry_glued_after_an_escape_in_a_first_value_stays_masked(
    escape, separator
):
    bad = []
    for name in _N3_NAMES:
        for value in (TAIL20, f"Bearer {TAIL20}"):
            alone = f"{name}{separator}{value}"
            masked_alone = set(_layers(alone)) - set(_shown_layers(alone, TAIL20))
            text = f"password={HEAD}{escape}{TAIL}{alone}"
            if masked_alone & set(_shown_layers(text, TAIL20)):
                bad.append((name, value[:6]))
    assert bad == []


def test_a_run_after_an_escape_that_ends_in_no_carrier_name_keeps_its_own():
    # The value stops at the escape (the guard sees `password:` ahead); the
    # run `Xfoo=bar;` ends in no carrier's ending, so the value does not read
    # on, and `password: <v2>` is masked by its own carrier.
    out = redact_trajectory_text(f"password={HEAD}\\Xfoo=bar;password: {TAIL20}")
    assert "\\Xfoo=bar;password: " in out
    assert HEAD not in out
    assert TAIL20 not in out


# A name that holds BEARER_TOKEN in its middle, glued onto a first value
# with no boundary (the zero-length piece): its value stays masked.
@pytest.mark.parametrize("separator", [": ", " :", " = ", " =\n\n", "\t: "])
def test_a_bearer_token_name_glued_onto_a_value_leaves_the_next_masked(separator):
    names = (
        "AWS_BEARER_TOKEN_BEDROCK",
        "aws_bearer_token_bedrock",
        "MY_BEARER_TOKEN_X",
    )
    assert _n3_bad(names, "", separator) == []
    alone = f"aws_bearer_token_bedrock{separator}{TAIL20}"
    for first in ("api_key", "password", "token"):
        text = f"{first}={HEAD}{alone}"
        shown = set(_shown_layers(text, TAIL20)) - set(_shown_layers(alone, TAIL20))
        assert shown == set(), (first, separator)


# A glued Authorization name, its separator and its scheme with no space
# between (`password=<v1>PROXY-AUTHORIZATION=Bearer <v2>`).
@pytest.mark.parametrize("separator", ["=", ":"])
@pytest.mark.parametrize("scheme", ["Bearer", "Basic", "Token"])
def test_a_glued_name_with_its_scheme_after_the_separator_leaves_the_next_masked(
    separator, scheme
):
    names = ("Authorization", "PROXY-AUTHORIZATION", "proxyAuthorization", "token")
    for name in names:
        alone = f"{name}{separator}{scheme} {TAIL20}"
        masked_alone = set(_layers(alone)) - set(_shown_layers(alone, TAIL20))
        text = f"password={HEAD}{alone}"
        assert masked_alone & set(_shown_layers(text, TAIL20)) == set(), name


# Review R307B (2026-11-16): only a real class or def header makes the next
# line's three quotes a docstring; prose with `->` or `.class` before a
# credential name is no header, and its triple-quoted value is masked.
@pytest.mark.parametrize("quote", ["'''", '"""'])
@pytest.mark.parametrize(
    "head",
    ["maps user -> password:", "x.class password:", "see -> Token:", "a->token:"],
)
def test_prose_before_a_credential_name_is_no_class_or_def_header(quote, head):
    text = f"{head}\n    {quote}{TAIL20}{quote}"
    assert {"text", "publish"} & set(_shown_layers(text, TAIL20)) == set()


@pytest.mark.parametrize("quote", ["'''", '"""'])
@pytest.mark.parametrize(
    "header",
    ["def f(x) -> T:", "class C:", "    class Token:", "def get(self) -> Token:"],
)
def test_a_real_class_or_def_header_keeps_its_docstring(quote, header):
    text = f"{header}\n    {quote}Immutable token representation.{quote}\n"
    assert redact_trajectory_text(text) == text
