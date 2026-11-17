"""A key right after a colour code, however the code is spelled.

Review of PR #148, 2026-11-13, L2. Every value here is made up.

A colour or cursor code (CSI: an ESC, a ``[``, parameter characters,
perhaps an intermediate character, a final letter) ends in a letter, so a
key written right after one followed a letter, and the shared left
boundary (``_KEY_START``) read it as starting mid-word and kept it. The
boundary knew six spellings of the opener and up to 48 parameter
characters. The review found five more ways to write a code that hid a key
in all eight layers, before ``sk-``, ``AQ.``, ``sk_live_``, a JWT,
``DB_PASS=``, ``Cookie`` and ``Authorization``: the opener fully
URL-encoded (``%1B%5B32m``), the escape in upper case (``\\x1B[32m``,
``\\u001B[32m``), caret notation (``^[[32m``), an intermediate character
(``ESC[2 q``) and more than 48 parameter characters.

The tables decide every combination of the parts of a code: how the ESC is
written (ESCAPES), how the ``[`` is (BRACKETS), and what follows them
(BODIES); OTHER holds the forms outside them. A key right after a code the
boundary reads is masked in every layer the boundary tests reach and in
what the task judge is shown. A code it does not read says why, as a
strict xfail: a key after it is still kept somewhere, and the day it is
masked everywhere its entry must go.
"""

from __future__ import annotations

import json
import re

import pytest

from benchflow.trajectories import types
from benchflow.trajectories.types import redact_key_families, redact_trajectory_text
from tests._timing import assert_linear, catches_quadratic, linear_seconds
from tests.test_redaction_escape_boundary import (
    AQ_KEY,
    BEARER,
    JWT,
    LAYERS,
    PASSWORD,
    SID,
    SK_KEY,
    SK_LIVE,
    _unchanged,
)
from tests.test_redaction_escape_boundary import _leaks as _boundary_leaks
from tests.test_redaction_shape_matrix import _leaks as _path_leaks

# The key shapes the review named: the text that carries each, and the
# part of it that must not survive.
KEYS = {
    "sk-": (SK_KEY, SK_KEY),
    "AQ.": (AQ_KEY, AQ_KEY),
    "sk_live_": (SK_LIVE, SK_LIVE),
    "JWT": (JWT, JWT.rsplit(".", 1)[1]),
    "DB_PASS=": (f"DB_PASS={PASSWORD}", PASSWORD),
    "Cookie": (f"Cookie: sid={SID}", SID),
    "Authorization": (f"Authorization: Bearer {BEARER}", BEARER),
}
# The key families of these: the judge masks them and keeps a name's value.
JUDGE_KEYS = ("sk-", "AQ.", "sk_live_", "JWT")

# How a writer spells the ESC.
ESCAPES = {
    "ESC": "\x1b",  # the byte, as a terminal log has it
    "\\e": "\\e",  # printf, bash's $'...', a zsh prompt
    "\\E": "\\E",  # terminfo (infocmp)
    "^[": "^[",  # caret notation (cat -v, less, Vim)
    "\\x1b": "\\x1b",  # Python's repr, C, JavaScript
    "\\x1B": "\\x1B",
    "\\033": "\\033",  # shell, C
    "\\33": "\\33",
    "\\u001b": "\\u001b",  # JSON (json.dumps, JavaScript)
    "\\u001B": "\\u001B",  # JSON (Jackson, .NET)
    "\\u{1b}": "\\u{1b}",  # Rust's Debug format, JavaScript
    "\\x{1b}": "\\x{1b}",  # Perl
    "#x1B": "#x1B",  # pytest's JUnit XML report
    "%1B": "%1B",  # URL-encoded
    "%1b": "%1b",
}
# How it spells the `[`.
BRACKETS = {"[": "[", "%5B": "%5B", "%5b": "%5b"}


def _run(length: int) -> str:
    """*length* parameter characters, as a truecolor list has them."""
    run = ("38;2;255;255;255;48;2;0;0;0;" * 8)[:length]
    assert len(run) == length
    return run


# What follows the opener: parameter characters, perhaps an intermediate
# character, and the final letter.
BODIES = {
    "m": "m",  # reset, no parameter
    "32m": "32m",
    "1;32m": "1;32m",
    "?25l": "?25l",
    "?2004h": "?2004h",
    "38:5:208m": "38:5:208m",
    "truecolor fg+bg": "38;2;255;255;255;48;2;0;0;0m",
    "48 parameters": _run(48) + "m",
    "2 q (an intermediate)": "2 q",  # the cursor's shape
    "!p (an intermediate alone)": "!p",  # soft reset
    "?1$p": "?1$p",
    "47 parameters and an intermediate": _run(47) + " q",
    "49 parameters": _run(49) + "m",
    "fg, bg and underline colour": (
        "0;1;3;4;38;2;255;255;255;48;2;255;255;255;58;2;255;255;255m"
    ),
    "200 parameters": _run(200) + "m",
    "ten private modes": "?1049;1000;1002;1003;1004;1005;1006;1015;1016;2004h",
}
# Past 48 characters, a list (digits, `;` and `:`) before `m`, `h` or `l`
# ends a code whatever opened it.
LONG = (
    "49 parameters",
    "fg, bg and underline colour",
    "200 parameters",
    "ten private modes",
)
MIXED = (
    "the ESC and the `[` in two encodings (\\x1b%5B, ESC%5B): no writer "
    "spells a code so; %1B%5B, both URL-encoded, is read"
)
ENCODED_RUN = (
    "the separators URL-encoded too, as Python's quote and "
    "encodeURIComponent write a code of more than one parameter: an encoded "
    "separator is three characters, so the run is no longer one class of "
    "characters and needs a second lookbehind per length; one parameter "
    "(%1B%5B32m) and plain separators (%1B%5B1;32m, as encodeURI writes "
    "them) are read"
)
C1 = (
    "the one-byte CSI (U+009B): terminals take it only in 8-bit mode, and "
    "no UTF-8 log writes it"
)
# Codes outside the table, each decided on its own: None where it is read.
OTHER = {
    "a list of 49 parameters, no opener": (_run(49) + "m", None),
    "ESC[ 48 parameters and an intermediate": (
        "\x1b[" + _run(48) + " q",
        "49 characters, an intermediate last: past 48 only a list before m, "
        "h or l is read, and no command puts an intermediate after so long a "
        "list",
    ),
    "ESC[ 49 digits": (
        "\x1b[" + "1" * 49 + "m",
        "a run of 49 digits or more with no ; or : among the last six is no "
        "list, so that a long number or a hash glued to a word holds no key "
        "start; no code is written so",
    ),
    "ESC[ 49 parameters before H": (
        "\x1b[" + _run(49) + "H",
        "past 48 characters only colour and mode codes (m, h, l) are read: "
        "no other code takes a list that long, and a list of numbers glued to "
        "a word (1;2;...;20;FAQ.) holds no key start",
    ),
    "%1B%5B1%3B32m": ("%1B%5B1%3B32m", ENCODED_RUN),
    "%1B%5B%3F25l": ("%1B%5B%3F25l", ENCODED_RUN),
    "%1B%5B2%20q": ("%1B%5B2%20q", ENCODED_RUN),
    "%251B%255B32m": (
        "%251B%255B32m",
        "URL-encoded twice: every rule reads one %XX escape",
    ),
    "8-bit CSI": ("\x9b32m", C1),
    "\\u009b": ("\\u009b32m", C1),
    "&#27;[": ("&#27;[32m", "an HTML character reference: rare in a log"),
    "\\c[": ("\\c[[32m", "Perl's control-character escape: rare"),
    "two intermediates": (
        "\x1b[1 !q",
        "rare: the commands with an intermediate have one",
    ),
    "ESC(B": (
        "\x1b(B",
        "a character-set selection, no CSI; tput sgr0 writes it before ESC[m",
    ),
}


def _codes():
    for e_name, esc in ESCAPES.items():
        for b_name, bracket in BRACKETS.items():
            mixed = bracket != "[" and not esc.startswith("%")
            for body_name, body in BODIES.items():
                why = MIXED if mixed and body_name not in LONG else None
                yield f"{e_name}{b_name}{body_name}", esc + bracket + body, why
    for name, (code, why) in OTHER.items():
        yield name, code, why


CODES = list(_codes())
READ = [(name, code) for name, code, why in CODES if why is None]


def _kept(code: str) -> list[tuple[str, str]]:
    """The (key, layer) cells where a key right after *code* survives."""
    kept = []
    for name, (carrier, secret) in KEYS.items():
        text = f"note{code}{carrier} done"
        layers = {*_boundary_leaks(text, secret), *_path_leaks(text, secret)}
        if name in JUDGE_KEYS and secret in redact_key_families(text):
            layers.add("judge")
        kept += [(name, layer) for layer in sorted(layers)]
    return kept


def _param(name: str, code: str, why: str | None):
    marks = []
    if why is not None:
        marks = [pytest.mark.xfail(strict=True, raises=AssertionError, reason=why)]
    return pytest.param(code, id=name, marks=marks)


@pytest.mark.parametrize("code", [_param(*row) for row in CODES])
def test_a_key_right_after_a_colour_code_is_masked_in_every_layer(code):
    """Review of PR #148, 2026-11-13, L2: each key shape the review named,
    right after a code, in text, result.json's ``_masked``, a dict value,
    both ACP writers, a serialized line, the export redactor and the
    judge's key families."""
    assert _kept(code) == []


def test_the_tables_hold_every_form_the_review_named():
    read = {code for _, code in READ}
    for form in ("%1B%5B32m", "\\x1B[32m", "\\u001B[32m", "^[[32m", "\x1b[2 q"):
        assert form in read, form
    assert "\x1b[" + _run(49) + "m" in read
    assert len(BODIES["48 parameters"]) == 49
    assert len(BODIES["47 parameters and an intermediate"]) == 49
    assert len(CODES) == len(ESCAPES) * len(BRACKETS) * len(BODIES) + len(OTHER)


# The review's forms and the spellings added with them.
NEW_CODES = {
    "%1B%5B32m": "%1B%5B32m",
    "%1b%5b1;32m": "%1b%5b1;32m",
    "\\x1B[32m": "\\x1B[32m",
    "\\u001B[32m": "\\u001B[32m",
    "^[[32m": "^[[32m",
    "ESC[2 q": "\x1b[2 q",
    "ESC[ fg, bg and underline colour": "\x1b[" + BODIES["fg, bg and underline colour"],
    "\\E[1m": "\\E[1m",
    "\\33[0m": "\\33[0m",
    "#x1B[32m": "#x1B[32m",
    "\\u{1b}[31m": "\\u{1b}[31m",
    "\\x{1b}[1m": "\\x{1b}[1m",
}
AROUND = ("{}", '"{}"', "'{}'", "({})", "[{}]", '\\"{}\\"', "`{}`")
AFTER = ("", ".", ";", ",", ")", "!", '"', "\\n", "%22", "\n", "\r\n")


@pytest.mark.parametrize("code", list(NEW_CODES.values()), ids=list(NEW_CODES))
def test_a_key_after_a_new_spelling_is_masked_quoted_escaped_and_by_punctuation(
    code,
):
    """Review of PR #148, 2026-11-13, L2: the code and its key quoted, in
    brackets, before punctuation, an escaped or encoded line break or
    quote, and the whole line already JSON-escaped (a tool printing a JSON
    file)."""
    kept = []
    for name, (carrier, secret) in KEYS.items():
        texts = [
            "out: " + around.format(code + carrier) + after
            for around in AROUND
            for after in AFTER
        ]
        texts.append(json.dumps(f"note{code}{carrier} done")[1:-1])
        for text in texts:
            layers = {*_boundary_leaks(text, secret), *_path_leaks(text, secret)}
            kept += [(name, text, layer) for layer in sorted(layers)]
    assert kept == []


# What must stay as it is.

KEPT_AFTER_A_CODE = (
    "done",
    "<task-notification>",
    "FAQ.troubleshooting-network-errors",
    ("mask" + "_live_FA" + "KEFAKEFA" + "KEFAKE12" + "34"),
    "TOTAL_PASS=12",
    "cookie_policy: strict",
)
# (the lead, 2026-11-14 19:42Z): a generic sk- key glued to a word and
# a word ending in `sk` before `-` and twelve letters are masked after a code
# too, as in 0.9.2.
MASKED_AFTER_A_CODE = (
    # One prose word after an Authorization name is its value (the lead's ruling of 2026-11-16 08:46Z).
    ("Authorization: required", "required"),
    ("task-instructions", "sk-instructions"),
    ("keysk-" + "0f1e2d3c" * 4, "0f1e2d3c" * 4),
)


@pytest.mark.parametrize("code", [code for _, code in READ], ids=[n for n, _ in READ])
def test_text_with_no_key_after_a_code_is_kept(code):
    """Review of PR #148, 2026-11-13, L2: the code is a boundary and nothing
    more; a word after it, a key glued to a word after it, prose after a
    header name and a test count stay in every layer."""
    changed = [
        (text, layer)
        for text in KEPT_AFTER_A_CODE
        for layer, same in _unchanged(f"note{code}{text}").items()
        if not same
    ]
    assert changed == []
    for text, secret in MASKED_AFTER_A_CODE:
        assert _boundary_leaks(f"note{code}{text}", secret) == [], text


# Text that looks like part of a code and is none.
# `task-implementation` is masked as in 0.9.2
# (test_redaction_escape_boundary.py, SK_WORDS_MASKED_AS_IN_092).
LOOK_ALIKES = [
    "a^[b",
    "a^[bc",
    "x^[1m",
    "100%5Bx",
    "100%5bxyz",
    "q=%5B1;32m",
    "x%1B32m",
    "7" * 64 + " tokens",
    "7" * 64 + "word",
    "1" * 200 + "e10",
    "0123456789" * 10 + "abcdef",
    "=" * 80 + "x",
    ">" * 60 + "q",
    "?" * 60 + "y",
    "iVBORw0KGgoAAAANSUhEUgAA" * 4,
    "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVoxMjM0NTY3ODkw",
    "aGVsbG8td29ybGRfMTIzNDU2Nzg5MA-_",
    "deadbeef" * 8,
    "0123456789abcdef" * 4,
    "2026-11-13T20:15:00.123456+00:00",
    "20261113T201500Z",
    "1763064900123ms",
    "v1.2.3-rc.1+build.5",
    "0.10.0a1",
    "py3.12",
    "a1b",
    "0x3f",
    "0X3F",
    "1e10",
    "1080p",
    "4K60fps",
    "FAQ.troubleshooting-network-errors",
    "at 12 noon",
    "in 2026 the",
    "job done",
    "pub quiz",
    "e.g. foo",
    "2 quick",
    "[link](https://example.test/x)",
    "arr[12]x",
    "C:\\33dir\\x1Bq",
    # Lists of numbers before a word, not a code's final m, h or l.
    ";".join(str(n) for n in range(1, 31)) + ";FAQ.troubleshooting-network-errors",
    "1;" * 30 + "x",
    "12:34:56:" * 8 + "Z",
]
PLAIN_KEY_START = rf"(?<![A-Za-z0-9]{types._NOT_AFTER_ESCAPE})"
PLAIN_IDENT_START = rf"(?<![A-Za-z0-9_-]{types._NOT_AFTER_ESCAPE})"


def _starts(pattern: str, text: str) -> set[int]:
    return {match.start() for match in re.finditer(pattern, text)}


@pytest.mark.parametrize("text", LOOK_ALIKES)
def test_text_that_only_looks_like_a_code_holds_no_key_start(text):
    """Review of PR #148, 2026-11-13, L2: in a look-alike a key starts only
    where it would if no colour code were read at all, for the redactor and
    for the judge, and the text, and a key glued to its last word, are kept
    in every layer. (the lead, 2026-11-14 19:42Z): a generic sk- key
    glued to that word is masked, as in 0.9.2."""
    assert _starts(types._KEY_START, text) == _starts(PLAIN_KEY_START, text)
    assert _starts(types._IDENT_START, text) == _starts(PLAIN_IDENT_START, text)
    assert _unchanged(text) == dict.fromkeys(LAYERS, True)
    if text[-1].isalnum():
        assert _unchanged(text + AQ_KEY) == dict.fromkeys(LAYERS, True)
        assert _boundary_leaks(text + "sk-" + "0f1e2d3c" * 4, "0f1e2d3c" * 4) == []


@pytest.mark.parametrize(
    "text",
    [
        "s/^[[:space:]]*//",
        "grep -E '^[[:alpha:]_]+$'",
        "^[[A-Z]",
        ";".join(str(n) for n in range(1, 31)) + "m",
    ],
)
def test_a_posix_class_after_a_caret_is_kept(text):
    """Review of PR #148, 2026-11-13, L2: ``^[[`` is caret notation's
    opener, so the letter after ``^[[:`` or ``^[[`` reads as a code's end,
    as does an ``m`` after a list of 49 characters or more; what follows is
    no key, and the text is kept."""
    assert _unchanged(text) == dict.fromkeys(LAYERS, True)


# Texts of 50,000 to 200,000 characters that try the boundary at every
# position: each opener repeated, and the characters a code holds after
# one, with no final letter; then letters after `[`, a `B`, a run of
# digits and an intermediate.
def _repeat(unit: str, length: int) -> str:
    return unit * (length // len(unit))


LONG_RUNS = {
    # The three the boundary was timed on when it learnt colour codes.
    "ESC[ then 1; and no final letter": lambda n: "\x1b[" + _repeat("1;", n),
    "%1B[ repeated": lambda n: _repeat("%1B[", n),
    "_ repeated": lambda n: _repeat("_", n),
    "%1B%5B repeated": lambda n: _repeat("%1B%5B", n),
    "\\x1B[ repeated": lambda n: _repeat("\\x1B[", n),
    "\\u001B[ repeated": lambda n: _repeat("\\u001B[", n),
    "^[[ repeated": lambda n: _repeat("^[[", n),
    "\\E[ repeated": lambda n: _repeat("\\E[", n),
    "#x1B[ repeated": lambda n: _repeat("#x1B[", n),
    "\\u{1b}[ repeated": lambda n: _repeat("\\u{1b}[", n),
    "ESC[ then digits": lambda n: "\x1b[" + _repeat("1", n),
    "%1B%5B then 1;": lambda n: "%1B%5B" + _repeat("1;", n),
    "ESC[2 then spaces": lambda n: "\x1b[2" + _repeat(" ", n),
    "ESC[ then !": lambda n: "\x1b[" + _repeat("!", n),
    "[a repeated": lambda n: _repeat("[a", n),
    "%5Ba repeated": lambda n: _repeat("%5Ba", n),
    "1111a repeated": lambda n: _repeat("1111a", n),
    "2 q repeated": lambda n: _repeat("2 q", n),
    "a list of 49 and m, repeated": lambda n: _repeat("1;" * 25 + "m", n),
}


def _long_run_text(name: str, size: int) -> str:
    run = LONG_RUNS[name](size)
    return run + (" " if run[-1].isalnum() else "") + SK_KEY


@pytest.mark.parametrize("name", list(LONG_RUNS))
def test_the_boundary_stays_linear_on_long_runs_of_every_form(name):
    """Review of PR #148, 2026-11-13, L2: four times the text takes about
    four times as long, not sixteen, in the redactor and in the judge's key
    families, and the key after the run is masked. (Quiet, 200,000
    characters take at most 0.05 s to redact and 0.9 s for the judge.)"""
    for redact in (redact_trajectory_text, redact_key_families):
        seconds, outs = linear_seconds(redact, lambda size: _long_run_text(name, size))
        # The judge starts a key only where an identifier starts.
        judge_after_underscore = (
            redact is redact_key_families and LONG_RUNS[name](50_000)[-1] == "_"
        )
        assert all(SK_KEY not in out for out in outs) or judge_after_underscore
        assert_linear(seconds)


# The lead's ruling (2026-11-14): the timing tests above call the
# shared check (tests/_timing.py). The positive control: the check, on one
# of each test's inputs at its sizes, fails once a quadratic term is added.
@pytest.mark.parametrize("name", list(LONG_RUNS)[:1])
def test_the_long_run_timing_check_catches_a_quadratic(name):
    assert catches_quadratic(
        redact_trajectory_text, lambda size: _long_run_text(name, size)
    )
