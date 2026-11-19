"""Authorization names: identifiers, several words, camel-case own names.

Part A. Every value here is made up.

The lead's rulings (2026-11-14):

1. A prefixed name in lower case that ends in the Authorization header's
   name, joined by ``_`` or ``.`` (has_authorization,
   self.basic_authorization, copy_source_authorization), is an identifier,
   mostly code's in text. Its unquoted value gets the code checks a
   lower-case pass name's and a prefixed cookie name's get: masked in an env
   line and where no variable could be the value, kept where it is code
   (``copy_source_authorization: Optional[str] = None``,
   ``source_authorization = kwargs.pop(...)``: 45 such library lines in the
   corpus were masked). A quoted value is judged whole, as before. A
   hyphenated or capitalised prefix (x-proxy-authorization,
   X-Authorization, note_Authorization) spells no identifier and keeps the
   strict rule: anything but one prose word is a secret's.
2. After the header's own name (Authorization, Proxy-Authorization,
   proxy_authorization) and an upper-case env name that ends in it
   (HTTP_AUTHORIZATION, REDIRECT_HTTP_AUTHORIZATION), a value of two or
   more words on the name's line is masked from the second word on,
   whatever the first word is; a single prose word stays. "A deliberate
   over-mask of prose after an Authorization name; one expected class"
   (test_redaction_gaps_en374.py, EXPECTED_OVER_MASKED). After a
   URL-encoded ``%3A`` or ``%3D`` only ``%20`` or ``+`` joins words, so a
   literal space ends the value (the reviewer's fixture keeps
   ``set the Authorization%3A header to ...``).
3. The header's own name in camel case as a quoted key (``"setCookie"``,
   ``"SetCookie"``, ``"proxyAuthorization"``, ``"ProxyAuthorization"``) is
   read as the header's own name, in text and trajectory files, in every
   layer: cookie pairs under the cookie's, anything but one prose word
   under the authorization's.
"""

from __future__ import annotations

import json

import pytest

from benchflow._utils.credential_names import entry_kind
from benchflow.publish.redact import redact_value
from benchflow.trajectories.types import (
    is_secret_field,
    redact_acp_trajectory_jsonl,
    redact_trajectory_obj,
    redact_trajectory_text,
)
from tests._timing import assert_linear, linear_seconds
from tests.test_redaction_ansi_spellings import _repeat
from tests.test_redaction_escape_boundary import (
    BARE,
    BEARER,
    LAYERS,
    SID,
    _dict_leaks,
    _unchanged,
)
from tests.test_redaction_escape_boundary import _leaks as _boundary_leaks
from tests.test_redaction_shape_matrix import _dict_key_leaks
from tests.test_redaction_shape_matrix import _leaks as _path_leaks

MASK = "***REDACTED***"
EXPORT_MASK = "<XXX-benchflow-key-values-XXX>"
WORDS = "please sign in first"
PAIRS = f"sid={SID}; theme=dark"


def _leaked(text: str, secret: str) -> set[str]:
    """The layers (text, trajectory files, export, publish) that keep it."""
    return {*_boundary_leaks(text, secret), *_path_leaks(text, secret)}


def _kept_everywhere(text: str) -> bool:
    return _unchanged(text) == dict.fromkeys(LAYERS, True)


def _masked_everywhere(text: str) -> bool:
    """Every layer changes *text*: what a code Authorization name holds is
    masked (the lead's ruling (b), 2026-11-15 01:3xZ)."""
    return _unchanged(text) == dict.fromkeys(LAYERS, False)


# --- 1. A prefixed lower-case name joined by `_` or `.` ---------------------

# Code from the corpus (the installed packages), each masked before: an
# annotation, a call, an identifier or a keyword argument after the name.
CODE = [
    "        copy_source_authorization: Optional[str] = None,",
    "        source_authorization: Optional[str] = None,",
    "    source_authorization = kwargs.pop('source_authorization', None)",
    "    return Envelope(identity=opened.identity, "
    "upstream_authorization=SecretStr(upstream_authorization))",
    "has_authorization = check_authorization(request)",
    "has_authorization: bool",
    "requires_authorization = self.flag",
    # The name's own words (the lead's ruling after the delta review of PR
    # #192; the classifier's own-words rule).
    "        'copy_source_authorization': source_authorization,",
    "            copy_source_authorization=copy_source_authorization,",
    "self.basic_authorization = basic_authorization",
    # A short all-letters word is prose: the named residual.
    "copy_source_authorization = flag",
]
# A bare identifier that is neither code, nor the name's own words, nor prose
# (the lead's ruling after the delta review of PR #192): masked.
BARE_IDENTIFIERS = [
    "copy_source_authorization=authorization_source",
    "copy_source_authorization = flag2x9",
    "has_authorization = source_authorization",
]


@pytest.mark.parametrize("text", CODE)
def test_code_after_a_prefixed_lower_case_authorization_name_is_masked(text):
    """Part A.1 kept code as no value. Under the lead's ruling (b), 2026-11-15 01:3xZ every value
    under a code Authorization name is masked, code included, in every
    layer (finer display of code returns in a later release)."""
    assert _masked_everywhere(text), text


@pytest.mark.parametrize("text", BARE_IDENTIFIERS)
def test_a_bare_identifier_after_such_a_name_is_masked(text):
    """The lead's ruling (code, the name's own words or prose): any other bare
    identifier is masked in every layer, the name kept."""
    out = redact_trajectory_text(text)
    assert MASK in out and out.split(MASK)[0] in text
    assert _unchanged(text) == dict.fromkeys(LAYERS, False)


# Values under such a name, and the part that must not survive.
VALUES = [
    (f"has_authorization={BARE}", BARE),
    (f"export copy_source_authorization={BARE}", BARE),
    (f'"has_authorization": "{BARE}"', BARE),
    (f"self.basic_authorization = '{BARE}'", BARE),
    ("has_authorization: FAKEpw-0123!", "FAKEpw-0123!"),
    (f"has_authorization: Bearer {BEARER}", BEARER),
    (f'"x_authorization": "{WORDS}"', WORDS),
]


@pytest.mark.parametrize("text,secret", VALUES)
def test_a_value_under_a_prefixed_lower_case_authorization_name_is_masked(text, secret):
    """Part A.1: an env line, a quoted value (judged whole), a long
    number, a value no variable could be and a scheme's token are masked in
    every layer."""
    assert _leaked(text, secret) == set(), text


STRICT = [
    "x-proxy-authorization",
    "X-Authorization",
]
# A capitalised header word after a prefix of a name in code: a code
# Authorization name to the classifier (_code_name), so every value is
# masked (the lead's ruling (b), 2026-11-15 01:3xZ); one prose word was kept.
STRICT_CODE_NAMES = [
    "note_Authorization",
    "X_Proxy_Authorization",
]


@pytest.mark.parametrize("name", STRICT)
def test_a_hyphenated_or_capitalised_prefix_keeps_the_strict_rule(name):
    """Part A.1: no identifier is spelled so; an identifier after `:`
    is masked as before, and so is one prose word (the lead's ruling of 2026-11-16 08:46Z)."""
    assert _leaked(f"{name}: {BARE}", BARE) == set()
    assert _masked_everywhere(f"{name}: required")


@pytest.mark.parametrize("name", STRICT_CODE_NAMES)
def test_a_capitalised_header_word_after_a_code_prefix_masks_every_value(name):
    """The lead's ruling (b), 2026-11-15 01:3xZ: a code Authorization name's
    value is masked, one prose word included, in every layer."""
    assert _leaked(f"{name}: {BARE}", BARE) == set()
    assert _masked_everywhere(f"{name}: required")


# --- 2. Several words after the header's own name or an upper-case name -----

SPOKEN = [
    "Authorization",
    "authorization",
    "Proxy-Authorization",
    "proxy_authorization",
    "HTTP_AUTHORIZATION",
    "REDIRECT_HTTP_AUTHORIZATION",
    "X_PROXY_AUTHORIZATION",
]


def _forms(name: str, value: str) -> dict[str, str]:
    return {
        "header": f"{name}: {value}",
        "env": f"{name}={value}",
        "export": f"export {name}={value}",
        "after an escaped newline": f"out\\n{name}: {value}",
        "after a colour code": f"out\x1b[32m{name}: {value}",
        "entry": f'"{name}": "{value}"',
        "escaped entry": f'\\"{name}\\": \\"{value}\\"',
        "single-quoted entry": f"'{name}': '{value}'",
        "nested entry": json.dumps({"headers": {name: value}}),
    }


@pytest.mark.parametrize("name", SPOKEN)
def test_several_words_are_masked_from_the_second_word_on(name):
    """In every form and every layer every word is gone, the first prose word
    too (the lead's ruling of 2026-11-16 08:46Z, part A.2 kept it in text)."""
    for form, text in _forms(name, WORDS).items():
        assert _leaked(text, "sign in first") == set(), (form, text)
        assert _leaked(text, "please") == set(), (form, text)
    assert redact_trajectory_text(f"{name}: {WORDS}") == f"{name}: {MASK}"
    assert redact_trajectory_text(f"{name}={WORDS}") == f"{name}={MASK}"


@pytest.mark.parametrize("name", SPOKEN)
def test_a_first_word_that_is_no_prose_is_masked_with_the_rest(name):
    """Part A.2: whatever the first word is, the second is masked; a
    first word that is no prose was masked before and still is."""
    for form, text in _forms(name, f"{BARE} and more").items():
        assert _leaked(text, BARE) == set(), (form, text)
        assert _leaked(text, "and more") == set(), (form, text)


@pytest.mark.parametrize("name", SPOKEN)
def test_one_prose_word_after_the_name_is_masked(name):
    """A single prose word is the value, masked in every form and layer
    (the lead's ruling of 2026-11-16 08:46Z, part A.2 kept it)."""
    for value in ("required", "none.", "Missing", "1", "true"):
        for form, text in _forms(name, value).items():
            assert _masked_everywhere(text), (form, text)


def test_after_an_encoded_separator_only_an_encoded_space_joins_words():
    """Part A.2 (the lead's OK, 2026-11-14): `%20`-joined words are one
    value, masked whole as before. Under (A) (lead 2026-11-15
    13:18Z) with ruling (b), the decoded views read `%3A` as the colon it
    writes: text after an encoded separator is masked as after a literal one
    (`see the Authorization%3A header in the docs` loses `in the docs`, as
    `see the Authorization: header in the docs` does; one prose word stays)."""
    for text, secret in (
        ("Authorization%3A%20please%20sign%20in", "sign%20in"),
        ("HTTP_AUTHORIZATION%3Dplease+sign+in", "sign+in"),
        ("see the Authorization%3A header in the docs", "in the docs"),
    ):
        assert _leaked(text, secret) == set(), text
    # The value's one word is masked, prose or not (the lead's ruling of 2026-11-16 08:46Z), and
    # the views read `%3A` as the colon ((A)): the encoded form masks what the literal form masks.
    encoded = redact_trajectory_text("see the Authorization%3A header in the docs")
    literal = redact_trajectory_text("see the Authorization: header in the docs")
    assert literal == f"see the Authorization: {MASK}"
    assert encoded.startswith(f"see the Authorization%3A {MASK}")
    assert "header" not in encoded and "docs" not in encoded
    assert _masked_everywhere("Authorization%3A%20required")


def test_the_rows_the_lead_added():
    """Part A (the lead, 2026-11-14): HTTP_AUTHORIZATION=Bearer <tok>
    is masked, scheme kept, in every layer. has_authorization=true was kept;
    under the lead's ruling (b), 2026-11-15 01:3xZ it was masked; a flag's
    switch is kept again (the lead, 2026-11-16 10:53Z: every name that ends in `authorization` is a carrier; a has_/requires_ name's switch, number or null is kept)."""
    text = f"HTTP_AUTHORIZATION=Bearer {BEARER}"
    assert _leaked(text, BEARER) == set()
    assert redact_trajectory_text(text) == f"HTTP_AUTHORIZATION=Bearer {MASK}"
    assert _masked_everywhere("has_authorization=true")


def test_the_words_are_the_names_own_line():
    """Part A.2: a value on the next indented line is read by its
    first word, as before; a line of code there is no value's second word.
    That first word is masked, prose or not (the lead's ruling of 2026-11-16 08:46Z; 0.9.3 masks the
    first one too)."""
    for text, masked in (
        (
            "docs say authorization:\n    def run(self, **kwargs):",
            f"docs say authorization:\n    {MASK} run(self, **kwargs):",
        ),
        (
            "Authorization:\n    if self.auto_error:",
            f"Authorization:\n    {MASK} self.auto_error:",
        ),
        (
            "the header is HTTP_AUTHORIZATION=\n  return True",
            f"the header is HTTP_AUTHORIZATION=\n  {MASK} True",
        ),
    ):
        assert redact_trajectory_text(text) == masked, text
    assert _leaked(f"Authorization:\n  {BARE}", BARE) == set()


# --- 3. The header's own name in camel case, quoted --------------------------

CAMEL_COOKIE = ["setCookie", "SetCookie"]
CAMEL_AUTHORIZATION = ["proxyAuthorization", "ProxyAuthorization"]
QUOTED = ("entry", "escaped entry", "single-quoted entry", "nested entry")


@pytest.mark.parametrize("name", CAMEL_COOKIE)
def test_a_camel_case_set_cookie_key_masks_its_pairs(name):
    """Part A.3: cookie pairs under `"setCookie"` are gone in every
    layer, as a quoted entry, a dict value and a headers mapping's key; a
    cookie named in prose is kept, as under Set-Cookie."""
    assert entry_kind(name, PAIRS) == "credential" and is_secret_field(name, PAIRS)
    assert redact_value({name: PAIRS})[0] == {name: EXPORT_MASK}
    for form in QUOTED:
        text = _forms(name, PAIRS)[form]
        assert _leaked(text, SID) == set(), (form, text)
        assert _kept_everywhere(_forms(name, "header too large")[form]), form
    assert _dict_leaks({name: PAIRS}, SID) == []
    assert _dict_key_leaks(name, PAIRS, SID) == []


@pytest.mark.parametrize("name", CAMEL_AUTHORIZATION)
def test_a_camel_case_proxy_authorization_key_is_the_headers(name):
    """Part A.3: under `"proxyAuthorization"` a scheme's token, an
    opaque value and every word are gone in every layer; one prose word is
    masked too (the lead's ruling of 2026-11-16 08:46Z: a bare value under an Authorization name is a
    credential, as 0.9.3 masks it)."""
    for value, secret in (
        (f"Bearer {BEARER}", BEARER),
        (BARE, BARE),
        (WORDS, "sign in first"),
    ):
        assert entry_kind(name, value) == "credential" and is_secret_field(name, value)
        for form in QUOTED:
            text = _forms(name, value)[form]
            assert _leaked(text, secret) == set(), (form, text)
        assert _dict_leaks({name: value}, secret) == []
        assert _dict_key_leaks(name, value, secret) == []
    for form in QUOTED:
        assert _masked_everywhere(_forms(name, "required")[form]), form
    assert redact_trajectory_obj({name: f"Bearer {BEARER}"}) == {name: f"Bearer {MASK}"}


def test_a_one_word_camel_spelling_is_no_header_name():
    """Part A.3: case is kept; `setcookie` is one word, no header."""
    assert _kept_everywhere(f'"setcookie": "{PAIRS}"')


# --- Every rule: masked once, and linear -------------------------------------


def test_part_a_masks_once():
    """Part A: a second pass finds nothing more to mask (the export
    redactor runs to a fixed point)."""
    for text in (
        f"Authorization: {WORDS}",
        f"Authorization: {BARE} and more",
        f"HTTP_AUTHORIZATION={WORDS}",
        f'"Authorization": "{WORDS}"',
        f"has_authorization={BARE}",
        f'"setCookie": "{PAIRS}"',
        f'"proxyAuthorization": "Bearer {BEARER}"',
        f'"ProxyAuthorization": "{WORDS}"',
    ):
        once = redact_trajectory_text(text)
        assert once != text and redact_trajectory_text(once) == once, text


# Review R192, M2: a second Authorization-family name inside A.2's words, its
# value quoted. The words end before it, and its value is masked.
SECOND_NAME_VALUE = "FAKEsecondauth0123"
LONG_PREFIX = "X_" + "A" * 70
SECOND_NAMES = [
    f'Authorization: none, Proxy-Authorization: "{SECOND_NAME_VALUE}"',
    f'Authorization: missing Authorization: "{SECOND_NAME_VALUE}"',
    f'Authorization: none has_authorization="{SECOND_NAME_VALUE}"',
    f'Authorization: please retry; HTTP_AUTHORIZATION="{SECOND_NAME_VALUE}"',
    f'Proxy-Authorization: none, Authorization: "{SECOND_NAME_VALUE}"',
    f'HTTP_AUTHORIZATION=please retry Proxy-Authorization="{SECOND_NAME_VALUE}"',
    f'Authorization: none x-proxy-authorization: "{SECOND_NAME_VALUE}"',
    f'Authorization: Proxy-Authorization: "{SECOND_NAME_VALUE}"',
    # Review R192B, M-A: the second name after a punctuation mark inside the
    # words, and after a prefix longer than 64 characters.
    f'Authorization: please retry;HTTP_AUTHORIZATION="{SECOND_NAME_VALUE}"',
    f'Authorization: please retry (Proxy-Authorization: "{SECOND_NAME_VALUE}")',
    f'Authorization: see [Authorization: "{SECOND_NAME_VALUE}"]',
    f'Authorization: none {{Proxy-Authorization: "{SECOND_NAME_VALUE}"}}',
    f'Authorization: none <Authorization: "{SECOND_NAME_VALUE}">',
    f'Authorization: none /Proxy-Authorization: "{SECOND_NAME_VALUE}"',
    f'Authorization: none |Authorization="{SECOND_NAME_VALUE}"',
    f'Authorization: none #HTTP_AUTHORIZATION="{SECOND_NAME_VALUE}"',
    f'Authorization: none $HTTP_AUTHORIZATION="{SECOND_NAME_VALUE}"',
    f'Authorization: none @Authorization: "{SECOND_NAME_VALUE}"',
    f'Authorization: none :Authorization: "{SECOND_NAME_VALUE}"',
    f'error: Authorization: header rejected (Proxy-Authorization: "{SECOND_NAME_VALUE}")',
    f'HTTP_AUTHORIZATION=please retry;Proxy-Authorization="{SECOND_NAME_VALUE}"',
    f'Authorization: none {LONG_PREFIX}_AUTHORIZATION="{SECOND_NAME_VALUE}"',
]


@pytest.mark.parametrize("text", SECOND_NAMES)
def test_a_second_authorization_name_in_the_words_keeps_its_value_masked(text):
    """R192, M2: in text, after an escaped newline and a colour code, in a
    JSON string, in trajectory files and at export, the second name's value
    is gone; the second name itself is kept."""
    for prefix in ("", "out\\n", "out\x1b[32m"):
        assert _leaked(prefix + text, SECOND_NAME_VALUE) == set(), prefix + text
    out = redact_trajectory_text(text)
    assert MASK in out and out.count("uthorization") == text.count("uthorization")
    assert redact_trajectory_obj({"Authorization": text}) != {"Authorization": text}
    entry = redact_trajectory_obj({"message": text})["message"]
    assert SECOND_NAME_VALUE not in entry


@pytest.mark.parametrize(
    "unit",
    [
        f"Authorization: please {BARE} ",
        "Authorization: please sign in first ",
        "Authorization: a b\n",
        "authorization: ",
        f"HTTP_AUTHORIZATION=please {BARE}\n",
        f'"proxyAuthorization": "please {BARE}", ',
        '"proxyAuthorization": "please ',
        f'"setCookie": "sid={BARE}; a=b", ',
        '"setCookie": "sid=',
        f"has_authorization={BARE}\n",
        "copy_source_authorization: Optional[str] = None,\n",
        "x_authorization = ",
        "Authorization:\n  ",
        "Authorization: none has_authorization=",
        'Authorization: none, Proxy-Authorization: "please ',
        "Authorization: a b c d e f g h ",
        "Authorization: " + "word " * 20 + "\n",
    ],
)
def test_part_a_repeated_stays_linear(unit):
    """Part A: each repeated to 50,000 and 200,000 characters: four
    times the text takes about four times as long, and a masked value is
    gone."""
    seconds, outs = linear_seconds(redact_trajectory_text, lambda n: _repeat(unit, n))
    assert all(BARE not in out and BEARER not in out for out in outs)
    assert_linear(seconds)


# Review R192C, B-1: after a header name, one word glued to the header's word
# with no joiner (`noneAuthorization=<v>`), or the camel `proxyAuthorization`
# unquoted, is no name a carrier reads; the one-word branch stopped its value
# before it and nothing read the value. The rc masks the word whole, and so
# does the branch now, in all six text layers. R192C's sweeps: 21,912 of 63,396
# rows and 46 of 42,845. The second name a carrier does read still ends the
# first value (SECOND_NAMES above).
B1_VALUE = "FAKEb1glued0123456789xyz"
B1_OPENERS = [
    "Authorization: ", "Proxy-Authorization: ", "authorization: ", "proxy-authorization: ",
    "HTTP_AUTHORIZATION=", "REDIRECT_HTTP_AUTHORIZATION=", "x-proxy-authorization: ",
    "Authorization=", "Authorization:", "Authorization:\t", "authorization%3A%20",
    "Authorization%3D", "MY_AUTHORIZATION=", "Authorization : ", '"Authorization": ',
    "set Authorization: ",
]  # fmt: skip
B1_FIRST = ["none", "x", "2", "Bearer", "a.b"]
B1_SECOND = [
    "Authorization", "authorization", "AUTHORIZATION", "proxyAuthorization",
    "ProxyAuthorization", "proxyauthorization", "PROXYAUTHORIZATION",
]  # fmt: skip
B1_SEPARATORS = ["=", "=%22", ":", "%3A", "%3D"]


def _six_layers(text: str) -> dict[str, str]:
    """Text, the rollout's _masked, the object layer, ACP, a record's text and
    export."""
    from benchflow.rollout._results import _masked

    return {
        "text": redact_trajectory_text(text),
        "_masked": _masked(text),
        "obj": json.dumps(redact_trajectory_obj({"output": text})),
        "acp": redact_acp_trajectory_jsonl(
            [{"type": "tool_call_update", "content": [{"type": "text", "text": text}]}]
        ),
        "record text": redact_trajectory_text(json.dumps({"output": text})),
        "publish": json.dumps(redact_value({"output": text})[0]),
    }


def _b1_shown(rows: list[str]) -> list[tuple[str, list[str]]]:
    return [
        (row, [layer for layer, out in _six_layers(row).items() if B1_VALUE in out])
        for row in rows
        if any(B1_VALUE in out for out in _six_layers(row).values())
    ]


def _b1_value(separator: str) -> str:
    return f"{separator}{B1_VALUE}%22" if separator == "=%22" else separator + B1_VALUE


@pytest.mark.parametrize("opener", B1_OPENERS)
def test_one_word_glued_to_an_unread_header_word_is_masked_whole(opener):
    """R192C's focused sweep: every opener, first word, spelling and glued
    separator, with and without a next line, in all six layers."""
    rows = [
        opener + first + second + _b1_value(separator) + tail
        for first in B1_FIRST
        for second in B1_SECOND
        for separator in B1_SEPARATORS
        for tail in ("", "\nnext line")
    ]
    assert _b1_shown(rows) == []


def test_the_wide_sweeps_glued_shapes_are_masked_whole():
    """R192C's wide sweep: `Proxy-Authorization: none<glue><name>=<v>`, glued by
    a letter or a digit, or by joiners before the camel spelling unquoted."""
    rows = [
        f"Proxy-Authorization: none{glue}{name}{_b1_value(separator)}"
        for glue, names in (
            *((glue, ("Authorization", "authorization", "AUTHORIZATION", "proxyAuthorization"))
              for glue in ("X", "x", "2")),
            *((glue, ("proxyAuthorization",)) for glue in ("_", "-", ".", "__", "--", "_-", "._")),
        )
        for name in names
        for separator in ("=", "=%22")
    ]  # fmt: skip
    assert _b1_shown(rows) == []


def test_the_glued_word_before_more_words_is_masked_whole():
    """The word is the value though more words follow: the rest is kept."""
    rows = [
        opener + first + second + _b1_value(separator) + " please retry"
        for opener in (
            "Authorization: ",
            "HTTP_AUTHORIZATION=",
            "Proxy-Authorization: ",
        )
        for first in ("none", "retry", "x", "2")
        for second in ("Authorization", "authorization", "proxyAuthorization")
        for separator in ("=", "=%22")
    ]
    assert _b1_shown(rows) == []
    out = redact_trajectory_text(
        f"Authorization: retryAuthorization={B1_VALUE} please retry"
    )
    assert out == "Authorization: ***REDACTED*** please retry"


# After the one-word branch's single word, a second
# Authorization name that the word itself took: after a punctuation mark, an
# encoded space, `=`, `==`, `:=`, `->` or `=>`, or past the lookahead's
# 64-character prefix. The word was masked and the second name's value, after
# a space or a quote, was read by nothing, on the rc too (1,204 rows of R192C's
# sweep). The branch now fails closed: the value after the taken name is
# masked with the word, in all six text layers.
EN599_VALUE = "FAKEen599took0123456789xyz"
EN599_TAKING_GLUES = ["%20", "+", "->", ":=", ";(", "=", "==", "=>"]
EN599_GLUES = [
    "", "X", "x", "2", "_", "-", ".", "__", "--", "_-", "._", *EN599_TAKING_GLUES,
]  # fmt: skip
EN599_SECONDS = [
    "Authorization", "authorization", "AUTHORIZATION", "Proxy-Authorization",
    "proxy_authorization", "HTTP_AUTHORIZATION", "REDIRECT_HTTP_AUTHORIZATION",
    "has_authorization", "x-proxy-authorization", "X-Authorization", "my-authorization",
    "HTTP_PROXY_AUTHORIZATION",
]  # fmt: skip
EN599_LONG = [
    "X_" + "A" * 70 + "_AUTHORIZATION", "X_" + "A" * 130 + "_AUTHORIZATION",
    "note_" + "b" * 90 + "_authorization", "a" * 200 + "_authorization",
]  # fmt: skip
EN599_SEPARATORS = [
    ': "{v}"',
    '="{v}"',
    ' : "{v}"',
    '%3A%20"{v}"',
    ": '{v}'",
    '= "{v}"',
    ": {v}",
]


def _en599_shown(rows: list[str]) -> list[tuple[str, list[str]]]:
    shown = []
    for row in rows:
        layers = [
            layer for layer, out in _six_layers(row).items() if EN599_VALUE in out
        ]
        if layers:
            shown.append((row, layers))
    return shown


@pytest.mark.parametrize("separator", EN599_SEPARATORS)
def test_a_second_name_the_one_word_took_has_its_value_masked(separator):
    """R192C's L-2 set, whole: `Proxy-Authorization: none<glue><name><sep><v>`
    for the glues that hide the name from the lookahead with every name, and
    every glue with a name past its 64-character prefix."""
    value = separator.format(v=EN599_VALUE)
    rows = [
        f"Proxy-Authorization: none{glue}{name}{value}"
        for glue in EN599_TAKING_GLUES
        for name in EN599_SECONDS
    ] + [
        f"Proxy-Authorization: none{glue}{name}{value}"
        for glue in EN599_GLUES
        for name in EN599_LONG
    ]
    assert _en599_shown(rows) == []


def test_the_fail_closed_value_after_other_header_names():
    """The same after the header's own names, a prefixed name and a JSON copy;
    the rest of the line is kept."""
    rows = [
        f'{opener}x{glue}Authorization: "{EN599_VALUE}" retry'
        for opener in (
            "Authorization: ",
            "HTTP_AUTHORIZATION=",
            "proxy-authorization: ",
        )
        for glue in (";(", "%20", "=>")
    ]
    assert _en599_shown(rows) == []
    out = redact_trajectory_text(
        f'Proxy-Authorization: none;(Authorization: "{EN599_VALUE}" retry'
    )
    assert out == "Proxy-Authorization: ***REDACTED*** retry"


def test_a_word_ending_in_the_header_word_with_no_separator_is_masked_alone():
    """No separator after the taken name: only the word is the value, and the
    next line is kept."""
    out = redact_trajectory_text("Proxy-Authorization: none;(Authorization\nnext: line")
    assert out == "Proxy-Authorization: ***REDACTED***\nnext: line"
    # One prose word is the value too (the lead's ruling of 2026-11-16 08:46Z).
    assert redact_trajectory_text("Authorization: required") == f"Authorization: {MASK}"


@pytest.mark.parametrize(
    "unit",
    [
        f'Proxy-Authorization: none;(Authorization: "{EN599_VALUE}" ',
        "Proxy-Authorization: none%20Authorization%3A%20 ",
        "Authorization: x;Authorization:",
        "Authorization: x" + ";(" * 40 + "Authorization= ",
    ],
)
def test_the_fail_closed_value_stays_linear(unit):
    """one reading per value; four times the text takes about four times
    as long."""
    seconds, outs = linear_seconds(redact_trajectory_text, lambda n: _repeat(unit, n))
    assert all(EN599_VALUE not in out for out in outs)
    assert_linear(seconds)


# Review R192C, B-2 and B-3 (its edge sweep, 12,160 rows): the one-word branch
# stopped its value before a quoted camel name whose quote never closes
# (`"proxyAuthorization=<v>`, which no carrier reads: 168 rows the rc masks),
# and no longer stopped before a name other carriers read after any run that
# ends in a joiner (`_authorization`, `-authorization`, a long run of joiners)
# or a quoted camel name the entry rule reads (`"noneAuthorization": "<v>"`): 310
# rows 8a2bb197a masked. Both are masked now, in all six text layers.
B23_VALUE = "FAKEb23edge0123456789xyz"
B23_OPENERS = [
    "Authorization: ", "Proxy-Authorization: ", "HTTP_AUTHORIZATION=", "authorization%3A%20",
    '"Authorization": ', "Authorization:", "x-proxy-authorization: ",
]  # fmt: skip


def _b23_shown(rows: list[str]) -> list[tuple[str, list[str]]]:
    shown = []
    for row in rows:
        layers = [layer for layer, out in _six_layers(row).items() if B23_VALUE in out]
        if layers:
            shown.append((row, layers))
    return shown


@pytest.mark.parametrize("opener", B23_OPENERS)
def test_a_quoted_camel_name_left_open_is_masked_with_the_word(opener):
    """B-2: `"proxyAuthorization` and its JSON copy, the quote never closed,
    with each glued separator, with and without more words."""
    rows = [
        opener + name + separator.format(v=B23_VALUE) + tail
        for name in ('"proxyAuthorization', '\\"proxyAuthorization')
        for separator in ("={v}", "=%22{v}%22", ":{v}", "%3A{v}", "%3D{v}", "%3A%20{v}")
        for tail in ("", " retry")
    ]
    assert _b23_shown(rows) == []


@pytest.mark.parametrize("opener", B23_OPENERS)
def test_a_name_after_a_joiner_run_keeps_its_value_masked(opener):
    """B-3: a second name after any run that ends in a joiner, read by the
    carrier that reads it, its value masked; and the quoted camel name the
    entry rule reads."""
    rows = [
        opener + name + separator.format(v=B23_VALUE) + tail
        for name in (
            "_authorization", "-authorization", ".authorization", "__authorization",
            "none" + "_" * 9 + "authorization",
        )
        for separator in (': "{v}"', ": {v}", '="{v}"', '=\\"{v}\\"')
        for tail in ("", " retry")
    ]  # fmt: skip
    if opener not in ("HTTP_AUTHORIZATION=", "x-proxy-authorization: "):
        rows += [
            opener + '"noneAuthorization"' + separator.format(v=B23_VALUE) + tail
            for separator in (': "{v}"', '="{v}"', '=\\"{v}\\"')
            for tail in ("", " retry")
        ]
    assert _b23_shown(rows) == []
    assert redact_trajectory_text(f"Authorization: _authorization: {B23_VALUE}") == (
        "Authorization: _authorization: ***REDACTED***"
    )


# Round 2 (the lead's rule, 17:0xZ): four shapes whose value nothing
# read, on the rc too. (1) A.2's words before a name glued to the header's word
# (`Authorization: please retryAuthorization: "<v>"`), and the glued name alone;
# (2) the header's name in a shell or batch reference (`${HTTP_AUTHORIZATION}=<v>`,
# `$(Authorization)`, `%AUTHORIZATION%`, `${HTTP_AUTHORIZATION:-x}=<v>`); (3) a
# name with more after the header's word (`AuthorizationHeader`,
# `authorization_header`, `authorization2`), alone or after a header name; (4) a
# code Authorization name, then a value with a JSON copy's escaped quote
# (`has_authorization: noneAuthorization=\"<v>\"`). Each value is masked now, in
# all six text layers; the classifier still keeps code, the name's own words and
# prose.
R2_VALUE = "FAKEr2class0123456789xyz"
R2_SEPARATORS = [
    ': "{v}"', '="{v}"', ' : "{v}"', '%3A%20"{v}"', '=%22{v}%22', ": '{v}'", '= "{v}"',
    '\t: "{v}"', ": {v}", "={v}",
]  # fmt: skip
R2_GLUED = [
    "Authorization", "authorization", "AUTHORIZATION", "proxyAuthorization",
    "ProxyAuthorization", "proxyauthorization", "PROXYAUTHORIZATION",
]  # fmt: skip
R2_SUFFIXED = [
    "AuthorizationHeader", "authorization_header", "authorization2", "AUTHORIZATION_HEADER",
    "authorizationHeader", "Authorization-Header", "AUTHORIZATION2", "authorization_2",
    "authorizationToken", "authorization_value",
]  # fmt: skip


def _r2_shown(rows: list[str]) -> list[tuple[str, list[str]]]:
    shown = []
    for row in rows:
        layers = [layer for layer, out in _six_layers(row).items() if R2_VALUE in out]
        if layers:
            shown.append((row, layers))
    return shown


@pytest.mark.parametrize(
    "opener",
    [
        "Authorization: please retry", "HTTP_AUTHORIZATION=please retry",
        "x-proxy-authorization: see the docs", "error: Authorization: header rejected",
        "Authorization: a b c d e f g h", "Proxy-Authorization: please sign in",
    ],
)  # fmt: skip
def test_words_before_a_glued_header_word_leave_its_value_masked(opener):
    """Class 1: A.2's words, then a name glued to the header's word by a
    letter or a digit, with each separator."""
    rows = [
        opener + glue + name + separator.format(v=R2_VALUE)
        for glue in ("", "X", "x", "2")
        for name in R2_GLUED
        for separator in R2_SEPARATORS
    ]
    assert _r2_shown(rows) == []


def test_a_glued_header_word_after_a_header_name_on_its_line_is_masked():
    """Class 1 holds after any header name earlier on the line, not only A.2's
    words. Alone, a name glued to the header's word was not read (the rc's
    rule: in the corpus every such name before `:` or `=` is code); it is a
    code Authorization name, so under the lead's ruling (b), 2026-11-15 01:3xZ its value is
    masked."""
    rows = [
        lead + first + name + separator.format(v=R2_VALUE)
        for lead in (
            "Authorization: none; ",
            "HTTP_AUTHORIZATION=x ",
            "Proxy-Authorization: see ",
        )
        for first in ("retry", "none", "x", "2", "abc123", "hasProxy")
        for name in R2_GLUED
        for separator in R2_SEPARATORS
    ]
    assert _r2_shown(rows) == []
    alone = f"httpAuthorization: {R2_VALUE}"
    assert redact_trajectory_text(alone) == "httpAuthorization: ***REDACTED***"


@pytest.mark.parametrize(
    "lead", ["", "Authorization: none ", "export ", "echo ", 'run: "', "x=1 "]
)
def test_a_name_in_a_shell_or_batch_reference_has_its_value_masked(lead):
    """Class 2: `${NAME}`, `$NAME`, `$(NAME)`, `%NAME%` and `${NAME:-x}`, each
    followed by a separator and the value."""
    rows = [
        lead + form.format(n=name) + separator.format(v=R2_VALUE)
        for name in (
            "HTTP_AUTHORIZATION", "Authorization", "AUTHORIZATION", "PROXY_AUTHORIZATION",
            "has_authorization", "Proxy-Authorization",
        )
        for form in ("${{{n}}}", "${n}", "$({n})", "%{n}%", "${{{n}:-x}}")
        for separator in ('="{v}"', "={v}", ': "{v}"', ": {v}", " = {v}")
    ]  # fmt: skip
    assert _r2_shown(rows) == []


@pytest.mark.parametrize(
    "opener",
    [
        "", "Authorization: none", "Proxy-Authorization: none", "HTTP_AUTHORIZATION=none",
        "Authorization: please retry",
    ],
)  # fmt: skip
def test_a_name_with_more_after_the_header_word_has_its_value_masked(opener):
    """Class 3: the name alone, and after a header name glued to the word
    before it, after a space, a letter or a punctuation mark."""
    glues = ("", " ", "X", ";(") if opener else ("",)
    rows = [
        opener + glue + name + separator.format(v=R2_VALUE)
        for glue in glues
        for name in R2_SUFFIXED
        for separator in R2_SEPARATORS
    ]
    assert _r2_shown(rows) == []


@pytest.mark.parametrize(
    "name",
    [
        "has_authorization", "requires_authorization", "copy_source_authorization",
        "http_authorization", "my_authorization",
    ],
)  # fmt: skip
def test_a_code_name_then_an_escaped_quote_has_its_value_masked(name):
    """Class 4: a code Authorization name whose value holds a JSON copy's
    escaped quote, as written in the raw text. A call or a subscript around
    the quoted value is in test_a_key_shaped_literal_in_code_is_masked."""
    rows = [
        name + separator + value.format(v=R2_VALUE)
        for separator in (": ", "=", " = ")
        for value in (
            'noneAuthorization=\\"{v}\\"', 'x=\\"{v}\\"', '\\"{v}\\"', 'b\\"{v}\\"',
            "noneAuthorization=\\'{v}\\'", '{{\\"k\\": \\"{v}\\"}}',
        )
    ]  # fmt: skip
    assert _r2_shown(rows) == []


def test_what_each_class_keeps():
    """The name, the separator and the quotes stay; prose after the header's
    own name is masked (the lead's ruling of 2026-11-16 08:46Z). Code and the name's own words under a code
    Authorization name were kept; under the lead's ruling (b), 2026-11-15 01:3xZ they are
    masked."""
    v = R2_VALUE
    assert redact_trajectory_text(
        f'Authorization: please retryAuthorization: "{v}"'
    ) == ('Authorization: ***REDACTED*** retryAuthorization: "***REDACTED***"')
    assert redact_trajectory_text(f"${{HTTP_AUTHORIZATION}}={v} next") == (
        "${HTTP_AUTHORIZATION}=***REDACTED*** next"
    )
    assert redact_trajectory_text(f'AuthorizationHeader: "{v}" retry') == (
        'AuthorizationHeader: "***REDACTED***" retry'
    )
    assert redact_trajectory_text(
        f'has_authorization: noneAuthorization=\\"{v}\\"'
    ) == ("has_authorization: ***REDACTED***")
    for text, masked in (
        ("retryAuthorization = self.flag", "retryAuthorization = ***REDACTED***"),
        (
            "copySourceAuthorization=copy_source_authorization,",
            "copySourceAuthorization=***REDACTED***,",
        ),
        ("has_authorization: bool", "has_authorization: ***REDACTED***"),
        ("Authorization: required", "Authorization: ***REDACTED***"),
    ):
        assert redact_trajectory_text(text) == masked, text
    for text in (
        "${HTTP_AUTHORIZATION}",
        'headers = {"Modal-Authorization": f"Bearer {session.token}"}',
        '"DescribeEndpointAuthorization": {',
    ):
        assert redact_trajectory_text(text) == text, text


@pytest.mark.parametrize(
    "unit",
    [
        f'Authorization: please retryAuthorization: "{R2_VALUE}" ',
        f"${{HTTP_AUTHORIZATION}}={R2_VALUE} ",
        f"${{HTTP_AUTHORIZATION:-x}}={R2_VALUE} ",
        f'AuthorizationHeader: "{R2_VALUE}" ',
        "Authorization: noneAuthorizationHeader" + "x" * 60 + " ",
        "Authorization: x" + "y" * 60 + " ",
        f'has_authorization: x=\\"{R2_VALUE}\\" ',
        f'Authorization: please retry AuthorizationHeader: "{R2_VALUE}" ',
        "Authorization: please retry authorization_" + "x" * 60 + " ",
    ],
)
def test_round_two_stays_linear(unit):
    """Round 2: one reading per name and value; four times the text
    takes about four times as long."""
    seconds, outs = linear_seconds(redact_trajectory_text, lambda n: _repeat(unit, n))
    assert all(R2_VALUE not in out for out in outs)
    assert_linear(seconds)


# The lead's ruling (17:5xZ) on the code shape: the classifier keeps a call or a
# subscript around a quoted name of at most 32 characters as code; under an
# Authorization name a key-shaped literal there (24 characters or more, or
# opaque: digits inside a word) is masked, its quotes escaped or not. Only a
# short name-like literal stays, the residual (`f("abc123")`, `x["abc"]`), and
# an identifier of letters of any length (the lead's ruling of 2026-11-14).
R2_KEY_LITERALS = [
    "FAKEr2class0123456789xyz", "a1b2c3d4", "sk-FAKE0123abcd", "FAKEabcdefghijklmnopqrst1",
    "FAKEk44abcdefghijklmnopqrstuvwxyz0123456789AB",
]  # fmt: skip


@pytest.mark.parametrize(
    "name",
    ["has_authorization", "copy_source_authorization", "requires_authorization",
     "self.basic_authorization"],
)  # fmt: skip
def test_a_key_shaped_literal_in_code_is_masked(name):
    """In every layer, under each code name."""
    shown = []
    for literal in R2_KEY_LITERALS:
        for form in (
            'f("{v}")', 'x["{v}"]', 'kwargs.pop("{v}", None)', "f('{v}')", 'f(\\"{v}\\")',
            'x[\\"{v}\\"]', 'os.environ.get("{v}")',
        ):  # fmt: skip
            for separator in (": ", "=", " = "):
                text = name + separator + form.format(v=literal)
                layers = [
                    layer for layer, out in _six_layers(text).items() if literal in out
                ]
                if layers:
                    shown.append((text, layers))
    assert shown == []


def test_a_short_name_like_literal_in_code_is_masked():
    """A short name-like literal in code was the residual; under
    the lead's ruling (b), 2026-11-15 01:3xZ every value under a code Authorization name is
    masked, code included."""
    for text in (
        'has_authorization: f("abc123")',
        'has_authorization = x["abc"]',
        'has_authorization=kwargs.pop("user_id2", None)',
        'has_authorization = f("table.id")',
    ):
        head = text.split("f(")[0].split("x[")[0].split("kwargs")[0]
        assert redact_trajectory_text(text) == head + "***REDACTED***", text
    # A quoted string of more than 23 characters inside code is masked, the
    # name's own words too: the code-shape cap's setting 1 wins (the lead,
    # 20:2xZ; #192).
    text = 'copy_source_authorization = kwargs.pop("copy_source_authorization", None)'
    assert redact_trajectory_text(text) == "copy_source_authorization = ***REDACTED***"


# Review R192C, B-4 (its p_ma5 sweep: 60 rows that 8a2bb197a masked, shown by
# the rc too): a joiner inside a quoted camel prefix after a header name
# (`Authorization: "none_proxyAuthorization": "<v>"`, `"x.noneAuthorization"`).
# The one-word branch's word now stops before it, and the entry rule reads it.
@pytest.mark.parametrize(
    "opener",
    ["Authorization: ", "Authorization:", "Proxy-Authorization: ", "authorization%3A%20",
     '"Authorization": '],
)  # fmt: skip
def test_a_joiner_in_a_quoted_camel_prefix_leaves_its_value_masked(opener):
    rows = [
        opener + name + separator.format(v=R2_VALUE) + tail
        for name in ('"none_proxyAuthorization"', '"x.noneAuthorization"')
        for separator in (': "{v}"', '="{v}"', '=\\"{v}\\"')
        for tail in ("", " retry")
    ]
    assert _r2_shown(rows) == []


# Review R192C, its p_ma5 sweep beyond v4 (4,678 rows the rc, 8a2bb197a and
# 1ec2bf789 all show): after a header name, a second name whose quotes are
# doubled, left open, mixed, encoded or followed by more letters, or a quoted
# camel or prefixed name in another case or past 64 letters. The one-word
# branch or the entry rule read the name and left its value. Each now masks the
# value with the name where the name holds the header's word, and a name no
# branch reads is read alone (_AUTHORIZATION_ANY_NAME).
MA5_OPENERS = [
    "Authorization: ", "Proxy-Authorization: ", "HTTP_AUTHORIZATION=", "authorization%3A%20",
    '"Authorization": ', "Authorization:", "x-proxy-authorization: ", "Authorization: none ",
]  # fmt: skip
MA5_NAMES = [
    '""proxyAuthorization""', '"proxyAuthorization""', "\\u0022proxyAuthorization\\u0022",
    '"proxyAuthorization"x', "&quot;proxyAuthorization&quot;", "`proxyAuthorization`",
    '"proxyAuthorization ', "\"proxyAuthorization'", '"none_x_Authorization"', '"_Authorization"',
    '"none_' + "b" * 70 + '_Authorization"', '"noneAUTHORIZATION"', '"noneproxyauthorization"',
    '"' + "a" * 70 + 'Authorization"', '"noneAuthorizationX"', "noneauthorization",
    "NONEAUTHORIZATION",
]  # fmt: skip
MA5_SEPARATORS = ["={v}", "=%22{v}%22", ": {v}", ': "{v}"', "%3A%20{v}", '=\\"{v}\\"']


@pytest.mark.parametrize("opener", MA5_OPENERS)
def test_a_second_name_in_odd_quotes_or_case_leaves_its_value_masked(opener):
    rows = [
        opener + name + separator.format(v=R2_VALUE) + tail
        for name in MA5_NAMES
        for separator in MA5_SEPARATORS
        for tail in ("", " retry")
    ]
    assert _r2_shown(rows) == []


# Review R192C, code-escq and code-unclosed (the lead's hard rule, 18:0xZ):
# after a code Authorization name, a value that opens with a quote and never
# closes it the way it opened: a truncated string (`has_authorization: "<v>`,
# `'<v>`), a JSON copy's escaped quote left open (`has_authorization=\"<v>`,
# `has_authorization: \"<v> retry`), or a raw quote whose only close is an
# escaped one or an entity. The code branch claimed the name, but its run
# cannot start at a quote and the entry rule needs the same quote to close, so
# nothing read the value; the rc masks it. The open string is now the value, a
# quoted string to the classifier, masked in all six layers.
ESCQ_VALUES = [
    "FAKEc227val0123456789xyz", "FAKEk44abcdefghijklmnopqrstuvwxyz0123456789",
    "FAKEk24abcdefghij0123456", "FAKE+opq/0123=abc-def_ghi.jkl9",
]  # fmt: skip
ESCQ_FORMS = [
    '{n}=\\"{v}', '{n}: \\"{v} retry', '{n} = \\"{v}', '{n}=\\\\\\"{v}', '{n}: "{v}', "{n} = '{v}",
    '{n}: "{v} retry', '{n}="{v}\\"', '{n}: "{v}&quot;', '{n}: "{v}&#34;', "{n}='{v}\\'",
]  # fmt: skip
# Review R192C, H-1: a JSON copy's escaped quote at exactly eight backslashes,
# a close one level deeper than the open, and a string past 4,096 characters.
B = "\\"
ESCQ_FORMS += [
    "{n}: " + B * 8 + '"{v}', "{n}: " + B * 8 + '"{v}' + B * 9 + '"', "{n}=" + B * 3 + '"{v}' + B * 4 + '"',
    "{n}: " + B * 7 + '"{v}' + B * 9 + '"', "{n}: " + B * 9 + '"{v}' + B * 9 + '"',
    '{n}: "{v}' + "x" * 5200 + '"', '{n}: "{v}' + "x" * 5200, "{n} = '" + "x" * 5200 + "{v}'",
]  # fmt: skip


@pytest.mark.parametrize(
    "name",
    ["has_authorization", "copy_source_authorization", "requires_authorization",
     "self.basic_authorization"],
)  # fmt: skip
def test_a_quote_left_open_after_a_code_name_is_masked(name):
    shown = []
    for value in ESCQ_VALUES:
        for form in ESCQ_FORMS:
            text = form.format(n=name, v=value)
            layers = [layer for layer, out in _six_layers(text).items() if value in out]
            if layers:
                shown.append((text, layers))
    assert shown == []


def test_code_prose_and_an_empty_or_open_string_are_masked():
    """Under the lead's ruling (b), 2026-11-15 01:3xZ code and prose are masked too (A.1 kept them, as
    review R192C's oracle reads A.1). An empty string, a lone quote and a
    string left open are masked with their opening quote kept (after the
    mask for a lone quote or a string left open, where it may close the JSON
    string around the line)."""
    for text, masked in (
        (
            'copy_source_authorization=kwargs.pop(\\"source_authorization\\", None)',
            "copy_source_authorization=***REDACTED***",
        ),
        ("has_authorization: bool", "has_authorization: ***REDACTED***"),
        ("has_authorization: self.flag", "has_authorization: ***REDACTED***"),
        ('has_authorization: "required"', 'has_authorization: "***REDACTED***"'),
    ):
        assert redact_trajectory_text(text) == masked, text
    for text, masked in (
        ('has_authorization=\\"\\"', 'has_authorization=\\"***REDACTED***\\"'),
        ('has_authorization = "', 'has_authorization = ***REDACTED***"'),
        ('has_authorization: "required', 'has_authorization: ***REDACTED***"'),
        (
            'has_authorization=\\"FAKEescq0123456789xyz',
            'has_authorization=***REDACTED***\\"',
        ),
        (
            'has_authorization: "FAKEescq0123456789xyz',
            'has_authorization: ***REDACTED***"',
        ),
    ):
        assert redact_trajectory_text(text) == masked, text
    # The lead's code-shape ruling (setting 1): a quoted string of more than 23
    # characters inside code is masked, even a name's own words.
    assert redact_trajectory_text(
        'copy_source_authorization=kwargs.pop(\\"copy_source_authorization\\", None)'
    ) == ("copy_source_authorization=***REDACTED***")


@pytest.mark.parametrize(
    "unit",
    [
        'has_authorization=\\"FAKEescq0123456789xyz ',
        'has_authorization: \\"' + "x" * 60 + " ",
        'copy_source_authorization=\\"\\\\x' * 10 + " ",
        'has_authorization: "FAKEescq0123456789xyz ',
        "requires_authorization='" + "x" * 60 + " ",
        "has_authorization: " + "\\" * 8 + '"FAKEescq0123456789xyz ',
    ],
)
def test_the_open_string_value_stays_linear(unit):
    """code-escq, code-unclosed: one reading per value; four times the text
    takes about four times as long."""
    seconds, _ = linear_seconds(redact_trajectory_text, lambda n: _repeat(unit, n))
    assert_linear(seconds)


def test_one_long_string_value_stays_linear():
    """H-1: a single value of 200,000 characters, and of four times that, is
    read whole, masked to its end; its time grows about four times."""
    seconds, outs = linear_seconds(
        redact_trajectory_text,
        lambda n: 'has_authorization: "FAKEescq0123456789xyz' + "x" * n,
        sizes=(200_000, 800_000),
    )
    assert all("FAKEescq" not in out and "xxxx" not in out for out in outs)
    assert_linear(seconds)


# Review R192C, H-2: a value made only of punctuation after a code name
# (`has_authorization: "!@#$%^&*()_+"`, an open `"!!!!...`), which the rc
# masks. The code branch now reads every value to its end and keeps only code,
# the name's own words and prose (the lead's ruling, 19:5xZ); anything else is
# masked, in all six layers. tests/test_redaction_code_name_values.py states
# the rule as a property over arbitrary values.
H2_VALUES = [
    '"!@#$%^&*()_+"', '"!!!!!!!!!!!!!!!!', "'%%%%%%%%%%'", "!@#$%^&*()_+", '\\"#$%&()\\"',
    "abc*def", '"*-*-*-*-*"', "`!!!!`", '"0¡"',
]  # fmt: skip


@pytest.mark.parametrize(
    "name",
    ["has_authorization", "copy_source_authorization", "requires_authorization",
     "self.basic_authorization"],
)  # fmt: skip
def test_a_value_that_is_no_code_words_or_prose_is_masked_after_a_code_name(name):
    shown = []
    for value in H2_VALUES:
        for separator in (": ", "=", " = "):
            text = name + separator + value
            escaped = json.dumps(value)[1:-1]
            for layer, out in _six_layers(text).items():
                if value in out or escaped in out:
                    shown.append((text, layer))
    assert shown == []


@pytest.mark.parametrize("value", ["\u09e6", "\U00011650", "\u212a"])
def test_a_digit_or_letter_of_another_script_is_masked_after_a_glued_name(value):
    # As after a code name (#192): the classifier reads these as ASCII.
    text = "Authorization: please retryAuthorization: " + value
    for layer, out in _six_layers(text).items():
        assert value not in out, (layer, out)
