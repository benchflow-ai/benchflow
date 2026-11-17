"""The A.1 rule as an invariant (review R192C, H-2; the lead's ruling, 19:5xZ).

After a code Authorization name (has_authorization, copy_source_authorization,
requires_authorization, self.basic_authorization) a value stays visible only if
it is code, the name's own words or prose; anything else is masked from the
separator on. Hypothesis generates arbitrary values (a bare token, or a string
opened by a quote at any escape depth, closed or not) under each name and
separator; each line is redacted as written and in its JSON-escaped forms (one
level, two levels, HTML-safe `\\u003c`), in all six text layers.

The oracle is A.1 restated from the lead's rulings (tests/_a1_oracle.py,
review R192C's, byte for byte), independent of the redactor and the
classifier: each clause quotes the ruling it restates. It reads an unmatched
trailing `)`, `]` or `}` as closing the code around the value (the lead,
21:21Z). Nothing else is exempt: a test result (`12/15`, `0.`), a stray
quote (`2"`), a value of quotes alone and a scheme's word in quotes are
masked. A comma at a value's end separates (`x=True,`), as the fuzz's `,`
tail reads it.

The values: a bare token (any characters but spaces, quotes only after its
first character), a string opened by a quote at any escape depth (closed or
not), a call or subscript of up to four arguments joined by commas and
spaces (names, digit-and-dot runs, short strings, key-shaped tokens), closed,
left open or with extra closers, and a callee followed by stray brackets.
A second property puts a key-shaped token after a kept value on its line
(`f(x) or <key>`): it is masked in every form and layer.

The CI budget is BUDGET examples with a fixed seed (SEED); set
BENCHFLOW_PROPERTY_EXAMPLES for a larger run and BENCHFLOW_PROPERTY_SEED for
another stream.

Under the lead's ruling (b), 2026-11-15 01:3xZ, A.1's show-code branch is out of this release:
every value under a code Authorization name is masked, in every path. The
oracle, these properties and review R192C's fuzzers stay for the A.1 work after
the rc. The shown side of the first property now holds with nothing shown;
the masked side and the second property run as before. Tests that asserted a
value kept here now assert it masked, citing the ruling.

tests/_a1_oracle.py is review R192C's oracle, byte for byte (sha256
3dee521d55a4de011748cd26495b382e07c6d3d59ecc9a1ed55345b8368efec5), so it is
excluded from ruff; test_the_oracle_is_the_reviewed_copy pins it.
"""

from __future__ import annotations

import json
import os
import re

import pytest
from hypothesis import HealthCheck, given, seed, settings
from hypothesis import strategies as st

from benchflow._utils.credential_names import flag_authorization_entry
from benchflow.publish.redact import redact_value
from benchflow.trajectories import types
from benchflow.trajectories.types import (
    redact_acp_trajectory_jsonl,
    redact_trajectory_obj,
    redact_trajectory_text,
)
from tests._a1_oracle import allowed, name_like

BUDGET = int(os.environ.get("BENCHFLOW_PROPERTY_EXAMPLES", "150"))
SEED = int(os.environ.get("BENCHFLOW_PROPERTY_SEED", "20261114"))
NAMES = [
    "has_authorization",
    "copy_source_authorization",
    "requires_authorization",
    "self.basic_authorization",
]
SEPARATORS = [": ", "=", " = "]
MASKS = ("***REDACTED***", "<XXX-benchflow-key-values-XXX>")


def _forms(text: str) -> list[str]:
    once = json.dumps(text)[1:-1]
    return [
        text,
        once,
        json.dumps(once)[1:-1],
        text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"),
        # Review R192C's tab form: every space and tab written `\\t` (B6, an
        # escaped tab before `=`; its decoded view, the lead's ruling (b), 2026-11-15 01:3xZ).
        text.replace("\t", "\\t").replace(" ", "\\t"),
    ]


def _six_layers(text: str) -> dict[str, str]:
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


def _allowed(name: str, value: str) -> bool:
    # A flag's switch, number or null under a has_/requires_ Authorization
    # name is kept (the lead, 2026-11-16 10:53Z), a `,` tail too.
    if flag_authorization_entry(name, value.removesuffix(",")):
        return True
    # A comma at a value's end separates, as code and JSON write it (`x=True,`
    # at a line's end) and as review R192C's fuzz reads its `,` tail.
    return allowed(value, name) is not None or (
        value[-1:] == "," and allowed(value[:-1], name) is not None
    )


_SPACE = " \t\r\n\x0b\x0c"
_TOKEN = st.builds(
    lambda first, rest: first + rest,
    st.characters(
        codec="utf-8",
        exclude_characters=_SPACE + "\"'",
        exclude_categories=("Cs", "Cc", "Zs", "Zl", "Zp"),
    ),
    st.text(
        st.characters(
            codec="utf-8",
            exclude_characters=_SPACE,
            exclude_categories=("Cs", "Cc", "Zs", "Zl", "Zp"),
        ),
        max_size=39,
    ),
)
_CONTENT = st.text(
    st.characters(
        codec="utf-8", exclude_characters="\r\n", exclude_categories=("Cs", "Cc")
    ),
    max_size=60,
)
_OPENERS = st.sampled_from(
    ['"', "'", '\\"', "\\'", '\\\\\\"', "\\" * 8 + '"', "\\" * 9 + '"']
)
_QUOTED = st.builds(
    lambda opener, content, close: opener + content + (opener if close else ""),
    _OPENERS,
    _CONTENT,
    st.booleans(),
)
_IDENT = st.from_regex(
    r"[A-Za-z_][A-Za-z0-9_]{0,24}(?:\.[A-Za-z_][A-Za-z0-9_]{0,12}){0,2}", fullmatch=True
)
_DIGITS = st.from_regex(
    r"[-+]?[0-9]{0,14}(?:\.[0-9]{0,12}){0,3}", fullmatch=True
).filter(bool)
_KEY = st.from_regex(r"[A-Za-z0-9]{24,48}", fullmatch=True).filter(
    lambda k: not name_like(k)
)
_ARG = st.one_of(
    _IDENT,
    _DIGITS,
    _KEY,
    st.builds(
        lambda q, c: q + c + q,
        st.sampled_from(['"', "'"]),
        st.text(st.sampled_from("abcXYZ019 ,.()[]_-"), max_size=30),
    ),
)
_CALL = st.builds(
    lambda callee, opener, args, join, close: callee + opener + join.join(args) + close,
    _IDENT,
    st.sampled_from(["(", "["]),
    st.lists(_ARG, max_size=4),
    st.sampled_from([", ", ",", " , ", ",\t"]),
    st.sampled_from([")", "]", "", "))", ")]", "}", ")}"]),
)
_STRAY = st.builds(
    lambda callee, rest: callee + rest,
    _IDENT,
    st.text(st.sampled_from("()[]{}=,._ab01'\""), min_size=1, max_size=12),
)


@seed(SEED)
@settings(
    max_examples=BUDGET,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    name=st.sampled_from(NAMES),
    separator=st.sampled_from(SEPARATORS),
    value=st.one_of(_TOKEN, _QUOTED, _CALL, _STRAY, _DIGITS),
)
def test_a_value_shown_after_a_code_name_is_code_its_own_words_or_prose(
    name, separator, value
):
    if _allowed(name, value):
        return
    # The value is masked from the separator on; an opening quote (escaped as
    # the layer writes it) may stay before the mask.
    masked = re.compile(
        re.escape(name + separator)
        + r"[\\\"']*(?:"
        + "|".join(map(re.escape, MASKS))
        + ")"
    )
    # The tab form writes the separator's spaces as `\\t`, which the mask may
    # take with the value: there no character of the value may stand before it.
    tabbed = re.compile(
        re.escape(name)
        + r"(?:[ \t:=]|\\{1,8}t)*[\\\"']*(?:"
        + "|".join(map(re.escape, MASKS))
        + ")"
    )
    forms = _forms(name + separator + value)
    for text, check in zip(forms, [masked] * (len(forms) - 1) + [tabbed], strict=True):
        for layer, out in _six_layers(text).items():
            # A value of quotes and escaped spaces alone (`"`, `"\\r`) has nothing
            # to show; in the tab form the decoded view reads `\\r` as a line
            # break and never masks a bare quote (a JSON string's own).
            empty = check is tabbed and not re.sub(r"\\+[nrt]|[\"'\\]", "", value)
            assert check.search(out) or empty, (layer, text, out)


def test_the_oracle_is_the_reviewed_copy():
    import hashlib
    from pathlib import Path

    # The reviewed copy with one docstring change (0.10.0rc4): it named Stripe's
    # published example live key verbatim, and now names its shape; the code
    # is the reviewed code.
    oracle = Path(__file__).with_name("_a1_oracle.py").read_bytes()
    assert hashlib.sha256(oracle).hexdigest() == (
        "555c5c29cdc2acbe748f05ff93f940f9416af8078967078e54e941e0b2286ca6"
    )


def test_the_vowel_free_words_are_the_oracle_s():
    from tests import _a1_oracle

    # The same words in the same order (the lead's 19, then this branch's 60).
    assert types._VOWEL_FREE_TEXT.split(", ") == list(_a1_oracle.VOWEL_FREE_WORDS)
    assert types._CAPS_CONSTANT and _a1_oracle.CAPS_CONSTANT


# Made-up tokens of 24+ characters, none a real key.
_KEY44 = "Zq" + "7Xw2Lr9T" + "b4Nc6Vd1" + "Hs8Kf3Mj" + "5Pg0Yt2R" + "b7Qn4Wx9" + "Ce1"
_SHOWN = [
    # The lead's 20:05Z rows.
    "f(scaled_dot_product_attention)",
    "torch.nn.functional.softmax(x)",
    "getUserAccountSettingsV2(x)",
    # 20:34Z A: a constant, every part all caps.
    "f(MAX_SEQUENCE_LENGTH_FOR_MODEL)",
    "x[GOOGLE_API_USE_CLIENT_CERTIFICATE]",
    # 20:34Z list: a listed word passes the vowel test, any case, 2 digits.
    "f(HTTP_server_connection_pool)",
    "f(llms_provider_config_object)",
    "f(pkcs12_certificate_loader_fn)",
    "f(xmlrpc_server_connection_pool)",
    # A listed word of 6 capitals passes the case test under A only.
    "f(XMLRPC_SERVER_CONNECTION_POOL)",
]
_MASKED = [
    # The lead's 20:05Z rows (made-up values of the same shapes).
    (
        ("f(sk" + "_live_FA" + "KE4eC39H" + "qLyjWDar" + "jtT1zd)"),
        ("sk" + "_live_FA" + "KE4eC39H" + "qLyjWDar" + "jtT1zd"),
    ),
    ("f(0123456789abcdef0123456789abcdef)", "0123456789abcdef0123456789abcdef"),
    (
        "f(QWxhZGRpbjpvcGVuIHNlc2FtZQ9xR2tLmN4pQ8sT)",
        "QWxhZGRpbjpvcGVuIHNlc2FtZQ9xR2tLmN4pQ8sT",
    ),
    ("f(AbCdEfGhIjKlMnOpQrStUvWx)", "AbCdEfGhIjKlMnOpQrStUvWx"),
    ("x[123456789012]", "123456789012"),
    (f"abc.{_KEY44[:40]}", _KEY44[:40]),
    (f"f({_KEY44})", _KEY44),
    # 20:2xZ: an acronym joined to a word; a vowel-less part.
    ("f(getHTTPResponseCodeForTheUser)", "getHTTPResponseCodeForTheUser"),
    ("f(load_bcdfg_settings_for_model)", "load_bcdfg_settings_for_model"),
    # 20:34Z A reaches only a token whose every part is all caps.
    ("f(obj.CLIENT_LOGGING_SUPPORTED)", "CLIENT_LOGGING_SUPPORTED"),
    ("f(BCDFGH_SETTINGS_FOR_THE_MODEL)", "BCDFGH_SETTINGS_FOR_THE_MODEL"),
    # 20:34Z list: exact words, case-tested.
    ("f(hTTp_server_connection_pool)", "hTTp_server_connection_pool"),
    ("f(httpss_server_connection_pool)", "httpss_server_connection_pool"),
    ("f(XMLRPC_server_connection_pool)", "XMLRPC_server_connection_pool"),
    # A key in a second or middle argument (shown on rc c03771584).
    (f"f(a, {_KEY44}, b)", _KEY44),
    (f'self.fn("x", {_KEY44})', _KEY44),
]


@pytest.mark.parametrize("separator", [": ", "="])
@pytest.mark.parametrize("value", _SHOWN)
def test_a_name_like_token_in_code_is_masked(value, separator):
    """The cap showed a name-like token in code (the lead, 20:05Z-20:34Z; the
    oracle keeps the rule for the A.1 work after the rc). Under
    the lead's ruling (b), 2026-11-15 01:3xZ the value is masked."""
    text = "has_authorization" + separator + value
    assert redact_trajectory_text(text) == f"has_authorization{separator}{MASKS[0]}"
    for layer, out in _six_layers(text).items():
        assert value not in out.replace('\\"', '"'), (layer, out)


@pytest.mark.parametrize("separator", [": ", "="])
@pytest.mark.parametrize(("value", "token"), _MASKED)
def test_a_key_shaped_token_in_code_is_masked(value, token, separator):
    for text in _forms("has_authorization" + separator + value):
        for layer, out in _six_layers(text).items():
            assert token not in out, (layer, text, out)
            assert "has_authorization" in out, (layer, text, out)


def test_a_string_opened_before_the_name_is_no_entry():
    # Found by the property test at 50,000 examples: in the object layer the
    # line is a JSON string whose opening quote stands just before the name;
    # a quoted name (an entry) is closed by the same quote.
    # A flag's number is kept (the lead, 2026-11-16 10:53Z: every name that ends in `authorization` is a carrier; a has_/requires_ name's switch, number or null is kept); the
    # same shape with a word is masked, its quotes kept.
    assert redact_trajectory_text('has_authorization: "0"') == (
        'has_authorization: "***REDACTED***"'
    )
    text = 'has_authorization: "x"'
    for layer, out in _six_layers(text).items():
        shown = out.replace('\\"', '"')
        masked = any('"' + mask in shown for mask in MASKS)
        assert masked and '"x' not in shown, (layer, out)
    # An entry's prose or number was kept; under the lead's ruling (b), 2026-11-15 01:3xZ
    # it is masked, its quotes kept.
    for entry, masked in (
        ('{"has_authorization": "1"}', '{"has_authorization": "***REDACTED***"}'),
        # `none` is no flag's switch word: masked, as #289, #279 and the twin
        # mask it (F2b, 2026-11-16 21:26Z).
        ("{'has_authorization': 'none'}", "{'has_authorization': '***REDACTED***'}"),
    ):
        assert redact_trajectory_text(entry) == masked


@pytest.mark.parametrize(
    "value", ["\u09e6", "\U00011650", "0\u09e6", "\u212a", "f(x)\u0130"]
)
def test_a_digit_or_letter_of_another_script_is_masked(value):
    # The classifier's \\d takes every script's digits (a test result to it)
    # and IGNORECASE folds the Kelvin sign and the dotted I; A.1 is ASCII.
    text = "has_authorization: " + value
    for layer, out in _six_layers(text).items():
        assert value not in out, (layer, out)


_KEPT = ["f(x)", "self.flag", "True", "1", "required", '"required"', "Optional[str]"]


@seed(SEED)
@settings(
    max_examples=max(BUDGET // 3, 50),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    name=st.sampled_from(NAMES),
    separator=st.sampled_from(SEPARATORS),
    kept=st.sampled_from(_KEPT),
    joiner=st.sampled_from([" or ", " ", " and ", " + ", " if x else ", ", "]),
    key=_KEY,
)
def test_a_key_after_a_kept_value_on_its_line_is_masked(
    name, separator, kept, joiner, key
):
    for text in _forms(name + separator + kept + joiner + key):
        for layer, out in _six_layers(text).items():
            assert key not in out, (layer, text, out)


# Review R192C on dd3abdd8d, under the lead's ruling (b) (2026-11-15 01:3xZ):
# every value under a code Authorization name is masked, what is glued to it
# included. B2 (the rc masks it), B5's backtick and entity tails, B7 (an entry
# whose value opens with closers), closers alone (the rc masks them), and a
# camel-case entry after an upper-case header name, which the carrier read
# before its prose (``xAuthorization``).
_GLUED_ROWS = [
    "has_authorization: x`)" + _KEY44,
    "has_authorization: x`;" + _KEY44,
    "has_authorization: x`'" + _KEY44,
    "has_authorization: x`," + _KEY44,
    'has_authorization: &quot;x"}' + _KEY44,
    '{"has_authorization": ")' + _KEY44 + '"}',
    '{"has_authorization": "}.zUpm,T<i7p!' + _KEY44 + '"}',
    'HTTP_AUTHORIZATION="xAuthorization": "' + _KEY44 + '"',
    'HTTP_AUTHORIZATION=\\"xAuthorization\\": \\"' + _KEY44 + '\\"',
]


@pytest.mark.parametrize("text", _GLUED_ROWS)
def test_a_glued_tail_or_a_closer_led_value_is_masked(text):
    shown = [
        (form, layer)
        for form in (text, json.dumps({"m": text}))
        for layer, out in _six_layers(form).items()
        if _KEY44 in out.replace("\\", "")
    ]
    assert shown == []


def test_closers_alone_are_masked_and_structure_stays():
    assert redact_trajectory_text("cfg.proxy_authorization: ) done") == (
        "cfg.proxy_authorization: ***REDACTED*** done"
    )
    # A code span's closing backtick and a JSON string's closing quote stay.
    assert redact_trajectory_text("`has_authorization: x` done") == (
        "`has_authorization: ***REDACTED***` done"
    )
    # A flag's value as text is masked, a number too (F2b, 2026-11-16 21:51Z).
    assert redact_trajectory_text("`has_authorization: 1` done") == (
        "`has_authorization: ***REDACTED***` done"
    )
    compact = '{"m":"has_authorization: x`","b_authorization":"' + _KEY44 + '"}'
    out = redact_trajectory_text(compact)
    assert (
        out
        == '{"m":"has_authorization: ***REDACTED***","b_authorization":"***REDACTED***"}'
    )
    assert json.loads(out)
