"""(the lead, 2026-11-14 18:1xZ): one mechanism for escaped text.

A value written with JSON string escapes or HTML entities is the same text
to a reader. The redactor builds a decoded view of escaped text for matching
only, runs every pattern on it, and maps the view's masks back to the text's
own bytes. The forms below are review R192C's (r192c/p_escaped.py), plus
deeper copies and a mix. The value is made up.
"""

from __future__ import annotations

import json

import pytest

from benchflow.trajectories import types
from benchflow.trajectories.types import (
    redact_acp_trajectory_jsonl,
    redact_trajectory_obj,
    redact_trajectory_text,
)
from tests._timing import assert_linear, catches_quadratic, linear_seconds
from tests.test_redaction_escape_boundary import LAYERS, _leaks, _unchanged

KEY = "FAKEq7Xw2Lr9Tb4Nc6Vd1Hs8Kf3Mj5Pg0Yt2Rb7Qn4Wx9"
MASK = "***REDACTED***"

# Text the redactor masks as written, each carrying KEY.
ROWS = [
    f"x_token={KEY}",
    f'api_key: "{KEY}"',
    f'{{"api_key": "{KEY}"}}',
    f"password='{KEY}'",
    f"Authorization: Bearer {KEY}",
    f'has_authorization="{KEY}"',
    f'export GITHUB_TOKEN="{KEY}"',
    f'curl -H "X-Api-Key: {KEY}" https://example.invalid',
]

U00H = {"<": "\\u003c", ">": "\\u003e", "&": "\\u0026"}
ENTITIES = {
    "ent-quot": ("&quot;", "&apos;"),
    "ent-dec": ("&#34;", "&#39;"),
    "ent-hex": ("&#x22;", "&#x27;"),
}


def _jq(text: str) -> str:
    return json.dumps(text)[1:-1]


def _u00a(text: str) -> str:
    return "".join(c if c.isalnum() or c in "_.-" else f"\\u{ord(c):04x}" for c in text)


FORMS = {
    "jq": _jq,
    "jq2": lambda s: _jq(_jq(s)),
    "jq3": lambda s: _jq(_jq(_jq(s))),
    "tab": lambda s: s.replace("\t", "\\t").replace(" ", "\\t"),
    "u00h": lambda s: "".join(U00H.get(c, c) for c in s),
    "u00a": _u00a,
    "u00q": lambda s: s.replace('"', "\\u0022").replace("'", "\\u0027"),
    **{
        name: lambda s, dq=dq, sq=sq: s.replace('"', dq).replace("'", sq)
        for name, (dq, sq) in ENTITIES.items()
    },
    "jq of u00a": lambda s: _jq(_u00a(s)),
    "jq2 of entities": lambda s: _jq(_jq(s.replace('"', "&quot;"))),
    "surrogate pair before": lambda s: "\\ud83d\\ude00 " + _jq(s),
}


@pytest.mark.parametrize("row", ROWS)
def test_every_row_is_masked_as_written(row):
    assert _leaks(row, KEY) == []


@pytest.mark.parametrize("form", FORMS)
def test_an_escaped_value_is_masked_in_every_layer(form):
    """Each row, written in the form, alone and inside a JSON string, shows
    its value in no layer."""
    shown = [
        (row, layer)
        for row in ROWS
        for text in (FORMS[form](row), json.dumps({"m": FORMS[form](row)}))
        for layer in _leaks(text, KEY)
    ]
    assert shown == []


# Review R192C's B6 on PR #192, moved to by the lead's ruling (b), 2026-11-15 01:3xZ:
# an escaped tab in the gap before `=` or `:` (a JSON copy of a tab-separated
# line), raw, as a JSON copy, a copy of a copy and inside a JSON string.
B6_NAMES = [
    "has_authorization",
    "copy_source_authorization",
    "self.basic_authorization",
    "x_token",
    "api_key",
    "password",
    "HTTP_AUTHORIZATION",
]
B6_SEPARATORS = ["\t=\t", "\t= ", " =\t", "\t:\t", "\t=", "\t\t=\t\t"]


def test_an_escaped_tab_before_the_separator_hides_no_value():
    shown = []
    for name in B6_NAMES:
        for separator in B6_SEPARATORS:
            line = f"{name}{separator}{KEY}"
            once = json.dumps(line)[1:-1]
            for text in (line, once, json.dumps(once)[1:-1], json.dumps({"m": once})):
                shown += [(name, separator, text, layer) for layer in _leaks(text, KEY)]
    assert shown == []


# The lead's ruling A (2026-11-15 06:0xZ): a quote written as an escape or an
# entity beside raw ones (review R192C's mixed forms) is read by the quote view,
# a level at a time; a copy of a copy is one JSON string a level up, so a line
# break inside it does not end the value (the inner-escape row). Each value
# shows on the rc and on #246.
MIXED_VALUE = "FAKEma5beyond0123456789xyz"
MIXED = [
    '"Authorization\\u0022: %22proxyAuthorization%22: "' + MIXED_VALUE + "\\u0022",
    'HTTP_AUTHORIZATION=\'proxyAuthorization": \\"' + MIXED_VALUE + '"',
    'Authorization: \\\\\\"proxyAuthorization"x: \\\\\\"' + MIXED_VALUE + '"',
    'HTTP_AUTHORIZATION="proxyAuthorization\': \\\\\\"' + MIXED_VALUE + '"',
]
INNER_VALUE = "FAKEpost0123456789abcdQRS"
INNER = 'has_authorization: \\\\"FAKEpre01234\\n' + INNER_VALUE + '\\\\"'


@pytest.mark.parametrize("text", MIXED)
def test_a_quote_escaped_beside_raw_ones_hides_no_value(text):
    assert _leaks(text, MIXED_VALUE) == []


def test_a_line_break_inside_a_copy_of_a_copy_hides_no_value():
    assert _leaks(INNER, INNER_VALUE) == []


def test_the_output_keeps_its_own_escaping():
    """The masks map back to the text's bytes: the escapes around a value
    stay as written, and the value is masked once."""
    cases = {
        f"x_token\\u003d{KEY}": f"x_token\\u003d{MASK}",
        f"api_key=&quot;{KEY}&quot;": f"api_key=&quot;{MASK}&quot;",
        f"api_key: \\u0022{KEY}\\u0022": f"api_key: \\u0022{MASK}\\u0022",
        f"x_token\\t=\\t{KEY}": f"x_token\\t=\\t{MASK}",
    }
    for text, expected in cases.items():
        assert redact_trajectory_text(text) == expected, text


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_a_json_copy_stays_valid_json(depth):
    """A bare quote, a JSON string's own, is never masked: a copy of a
    record, nested to *depth*, still parses once its value is masked."""
    record = {"message": f'api_key: "{KEY}" and x_token\\u003d{KEY}'}
    text = json.dumps(record)
    for _ in range(depth - 1):
        text = json.dumps({"inner": text})
    out = redact_trajectory_text(text)
    assert KEY not in out
    value = json.loads(out)
    for _ in range(depth - 1):
        value = json.loads(value["inner"])
    assert MASK in value["message"]


# Escaped text with no secret in it, kept in every layer.
KEPT = [
    'note\\nline \\"quoted\\" here',
    "&lt;div class=&quot;x&quot;&gt;fine&lt;/div&gt;",
    'out\\nALL_BLOCKERS_PASS=["tests/a.py::t", "tests/b.py::u[1]"]',
    "note\\nTESTS_PASS: true",
    "\\u003ctask-notification\\u003e done \\u003c/task-notification\\u003e",
    "&lt;task-notification&gt;",
    "path C:\\\\Users\\\\me\\\\file.txt",
    '{\\"status\\": \\"ok\\", \\"count\\": 3}',
    "TESTS_PASS\\u003d12/15",
]


@pytest.mark.parametrize("text", KEPT)
def test_escaped_text_with_no_secret_is_kept(text):
    assert _unchanged(text) == dict.fromkeys(LAYERS, True)
    assert _unchanged(json.dumps({"m": text})) == dict.fromkeys(LAYERS, True)


@pytest.mark.parametrize(
    ("text", "masked"),
    [
        ("note\\nAuthorization: required", f"note\\nAuthorization: {MASK}"),
        ("note\\nHTTP_AUTHORIZATION=none", f"note\\nHTTP_AUTHORIZATION={MASK}"),
    ],
)
def test_a_bare_value_under_an_authorization_name_is_masked(text, masked):
    """These two were kept as prose until that change: a bare value under an
    Authorization name is masked, prose or not (the lead's ruling of
    2026-11-16 08:46Z)."""
    assert redact_trajectory_text(text) == masked


def _repeat(unit: str, size: int) -> str:
    return (unit * (size // len(unit) + 1))[:size]


SHAPES = {
    "JSON lines with escaped strings": (
        '{"message": "note\\nline \\"q\\" x_token\\u003d' + KEY + '", "n": 1}\n'
    ),
    "u00a lines": "has_authorization\\u003a\\u0020\\u0022abc\\u0022\\u000a",
    "entities": f"api_key=&quot;{KEY}&quot; &amp; more ",
    "one line of short literals": '"ab\\"c", ',
    "one line of unpaired quotes": 'a "b\\n c ',
    "a backslash run": "\\",
}


@pytest.mark.parametrize("shape", SHAPES)
def test_the_view_stays_linear(shape):
    """Escaped text at 50,000 and 200,000 characters, at the shared check
    (tests/_timing.py)."""
    seconds, _ = linear_seconds(
        redact_trajectory_text, lambda size: _repeat(SHAPES[shape], size)
    )
    assert_linear(seconds)


# The per-record gate's cue-escape finder (`\\+u00..`, `\\+x..`, `\\+/`) on a long
# backslash run with no cue after it: each run is read from its start and once
# (possessive), so the gate stays linear. Before the fix the finder tried `\\+`
# from every backslash of the run, so each shape here was quadratic (4.7 s on
# 40,000 backslashes; F2b, 2026-11-15 12:04Z): a run alone, and a run before
# `u00` and before `x` with no hex digit after them. `%(?:25)*` starts only at
# a `%`, so it was linear before the fix too.
GATE_SHAPES = {
    "backslash run": lambda size: "a note " + "\\" * size + " q",
    "backslash run before u00": lambda size: "a note " + "\\" * size + "u00 q",
    "backslash run before x": lambda size: "a note " + "\\" * size + "x q",
}


@pytest.mark.parametrize("shape", GATE_SHAPES)
def test_the_gate_reads_a_backslash_run_once(shape):
    seconds, _ = linear_seconds(types._escape_flags, GATE_SHAPES[shape])
    assert_linear(seconds)


# A private-key block spans lines and its body holds no escape: a line that
# opens one carries the views through the block (FX687's and R192C's rows on
# 34958aff0, F2b 2026-11-15 14:42Z). Made-up base64.
PEM_BODY = [
    "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQ" + str(n) * 14 for n in range(6)
]
PEM_BEGIN = "--" + "---BEGIN" + " PRIVATE" + " KEY----" + "-"
PEM_END = "-----END PRIVATE KEY-----"
PEM_SHAPES = {
    "BEGIN with \\u0020": ("--" + "---BEGIN\\u0020PRIVATE KEY-----", PEM_END),
    "BEGIN with &#45;": ("&#45;--" + "--BEGIN PRIVATE KEY-----", PEM_END),
    "BEGIN with \\\\u002d": ("\\\\u002d--" + "--BEGIN PRIVATE KEY-----", PEM_END),
    "END with \\u002d": (PEM_BEGIN, "\\u002d----END PRIVATE KEY-----"),
    "END with &#32;": (PEM_BEGIN, "-----END&#32;PRIVATE KEY-----"),
}


@pytest.mark.parametrize("shape", PEM_SHAPES)
def test_a_key_block_with_an_escaped_marker_is_read_whole(shape):
    head, foot = PEM_SHAPES[shape]
    text = "\n".join(["key file:", head, *PEM_BODY, foot, "done"])
    for line in PEM_BODY:
        assert _leaks(text, line) == []


PEM_LONG = {
    "one block, no END": lambda size: (
        "--" + "---BEGIN\\u0020PRIVATE KEY-----\n" + _repeat(PEM_BODY[0] + "\n", size)
    ),
    "a BEGIN on every line": lambda size: _repeat(
        "--" + "---BEGIN\\u0020PRIVATE KEY-----\n", size
    ),
    "plain BEGIN, escaped body": lambda size: _repeat(
        PEM_BEGIN + "\n" + PEM_BODY[0] + "\\u002b\n", size
    ),
}


@pytest.mark.parametrize("shape", PEM_LONG)
def test_the_key_block_carry_stays_linear(shape):
    seconds, _ = linear_seconds(types._escape_flags, PEM_LONG[shape])
    assert_linear(seconds)


# A cue on one line and an escape on the next line that isn't blank as the
# views read it: both lines are read together, with the blank lines between
# (R192C's families A and B on 8e64f92b1, F2b 2026-11-15 16:06Z). Made-up value.
PAIR_VALUE = "FAKEpairQz8Wm3Rt6Yp1Xk4Vn7Jb2"
PAIR_SHAPES = {
    "a YAML name, an escaped space before the value": "api_key:\n  \\u0020"
    + PAIR_VALUE,
    "a JSON name, the value opened with &#x22;": '{\n  "api_key":\n    &#x22;'
    + PAIR_VALUE
    + '"\n}',
    "a cookie, &#32; before the pair": "Cookie:\n&#32; sid=" + PAIR_VALUE,
    "a name, a blank line, an escaped value": "APIKEY\n\n= "
    + PAIR_VALUE[:8]
    + "\\u0043"
    + PAIR_VALUE[8:],
    "an escaped name, blank lines, a plain value": "AUTHORI\\u005aATION\n\n\n= "
    + PAIR_VALUE,
    "a name, a literal \\t line, a plain value": "SECRET\n\\t\n= " + PAIR_VALUE,
    "a name, an &#32; line, a plain value": "TOKEN\n&#32;\n= " + PAIR_VALUE,
}


@pytest.mark.parametrize("shape", PAIR_SHAPES)
def test_a_value_below_its_name_is_read_with_it(shape):
    assert _leaks(PAIR_SHAPES[shape], PAIR_VALUE[8:20]) == []


PAIR_LONG = {
    "names, blank lines, escaped values": lambda size: _repeat(
        "api_key:\n\n\n  \\u0020v\n", size
    ),
    "a name, then blank lines": lambda size: "token:" + "\n" * size + "\\u0020v",
    "a name, then escaped blank lines": lambda size: (
        "token:\n" + _repeat("\\t\n", size) + "= v"
    ),
}


@pytest.mark.parametrize("shape", PAIR_LONG)
def test_the_pair_rule_stays_linear(shape):
    seconds, _ = linear_seconds(types._escape_flags, PAIR_LONG[shape])
    assert_linear(seconds)


# A scheme word with a token after it, under an Authorization name the rules
# read only as prose (its first letter escaped), is masked with its token,
# quoted plainly or with escaped quotes (F2b 2026-11-16 04:08Z; R192C's
# bearer_probe). Made-up token.
SCHEME_TOKEN = "FAKEschemeL87N4xvwTv6mKJ"
SCHEME_SHAPES = {
    "the first letter as \\x, raw quotes": '"\\x70roxyAuthorization"="Bearer '
    + SCHEME_TOKEN
    + '"',
    "the first letter as \\x, entity quotes": "&quot;\\x70roxyAuthorization&quot;"
    "=&quot;Bearer " + SCHEME_TOKEN + "&quot;",
    "the first letter as &#x70, entity quotes": "&quot;&#x70roxyAuthorization&quot;"
    "=&quot;Bearer " + SCHEME_TOKEN + "&quot;",
    "the first letter as \\u{70}, \\u0022 quotes": "\\u0022\\u{70}roxyAuthorization"
    "\\u0022=\\u0022Bearer " + SCHEME_TOKEN + "\\u0022",
}


@pytest.mark.parametrize("shape", SCHEME_SHAPES)
def test_a_scheme_word_is_masked_with_its_token(shape):
    assert _leaks(SCHEME_SHAPES[shape], SCHEME_TOKEN[4:16]) == []


# A scheme word's token is masked whole: through `&`, `&amp;` and the escapes
# the views decode, up to whitespace, a quote or a delimiter (R192C's
# scheme_edge, F2b 2026-11-16 05:54Z). Made-up token halves.
EDGE_A, EDGE_Z = "FAKEedgeQz8Wm3Rt", "Xk4Vn7Jb2Hd5FAKE"
EDGE_NAME = '"\\x70roxyAuthorization"='
EDGE_SHAPES = {
    "a plain token": EDGE_NAME + '"Bearer ' + EDGE_A + EDGE_Z + '"',
    "a token with &": EDGE_NAME + '"Bearer ' + EDGE_A + "&" + EDGE_Z + '"',
    "a token with &amp;": EDGE_NAME + '"Bearer ' + EDGE_A + "&amp;" + EDGE_Z + '"',
    "a token with \\/": EDGE_NAME + '"Bearer ' + EDGE_A + "\\/" + EDGE_Z + '"',
    "a token with \\u002d": EDGE_NAME + '"Bearer ' + EDGE_A + "\\u002d" + EDGE_Z + '"',
    "entity quotes, a token with &amp;": "&quot;\\x70roxyAuthorization&quot;=&quot;Bearer "
    + EDGE_A
    + "&amp;"
    + EDGE_Z
    + "&quot;",
    "a lowercase scheme": EDGE_NAME + '"bearer ' + EDGE_A + EDGE_Z + '"',
    "Basic with padding": EDGE_NAME + '"Basic ' + EDGE_A + EDGE_Z + '=="',
}


@pytest.mark.parametrize("shape", EDGE_SHAPES)
def test_a_scheme_token_is_masked_whole(shape):
    for part in (EDGE_A, EDGE_Z):
        assert _leaks(EDGE_SHAPES[shape], part) == []


def test_the_view_timing_check_catches_a_quadratic():
    assert catches_quadratic(
        redact_trajectory_text,
        lambda size: _repeat(SHAPES["JSON lines with escaped strings"], size),
    )


# Under the 08:46Z ruling the decoded view masks the first Authorization name's
# bare value and the quoted second name with it; the second name's value on the
# next line is masked too, in the object and ACP layers (#358's next-line rule
# for an entity-quoted name; F2b's ruling (a) on the #289 merge named it a
# residual: #289 c25a207d9 and rel-0.9.x e41cd6195 show it).
SECOND_NAME_AFTER_AN_AUTHORIZATION_VALUE = 'authorization%3A%20&quot;x_y_z_authorization&quot;=\n  \\"FX307Cma5beyond0123456789xyz\\"{tail}'


def _object_layer(text: str) -> str:
    return json.dumps(redact_trajectory_obj({"output": text}))


def _acp_layer(text: str) -> str:
    event = {"type": "tool_call_update", "content": [{"type": "text", "text": text}]}
    return redact_acp_trajectory_jsonl([event])


@pytest.mark.parametrize("layer", [_object_layer, _acp_layer], ids=["obj", "acp"])
@pytest.mark.parametrize("tail", ["", " retry"], ids=["end", "retry"])
def test_a_second_name_after_an_authorization_value_in_a_view_is_masked(tail, layer):
    text = SECOND_NAME_AFTER_AN_AUTHORIZATION_VALUE.format(tail=tail)
    assert "ma5beyond0123456789" not in layer(text)
