"""A bare value under an Authorization name is a credential.

0.9.3 (rel #288) masks `Authorization: <token>` with no scheme before the
token; the rc kept a value its prose test reads as one word of prose (letters
only, fifteen or fewer: `YmkkyClHmFD`; a count of five digits or fewer; `true`)
and showed it in every layer. The lead ruled on 2026-11-16 at 08:46Z
that the rc masks at least what rel masks here: a bare value under any
Authorization spelling is a carrier, in every layer. A scheme before a token
is read as before, and a scheme word alone stays, as 0.9.3 keeps it.

The prose control, `Authorization: please retry later`:
- rc before (e0e32b6c8): `Authorization: please ***REDACTED***`;
- rel 0.9.3 (#288, 4ce66f4d1): `Authorization: ***REDACTED*** retry later`;
- rc now: `Authorization: ***REDACTED***`.
Every value is made up.
"""

from __future__ import annotations

import json

import pytest

from benchflow.publish.redact import redact_value
from benchflow.trajectories.types import (
    redact_acp_trajectory_jsonl,
    redact_trajectory_obj,
    redact_trajectory_text,
)

MASK = "***REDACTED***"
TOKENS = ["YmkkyClHmFD", "qwErtyZx", "48213", "true", "required"]
NAMES = [
    "Authorization",
    "AUTHORIZATION",
    "authorization",
    "Proxy-Authorization",
    "proxy_authorization",
    "HTTP_AUTHORIZATION",
]
CLASSIFIED_KEYS = {"HTTP_AUTHORIZATION"}


def _text_layers(text: str) -> dict[str, str]:
    """The five layers' output for *text* carried in a record."""
    return {
        "text": redact_trajectory_text(text),
        "rec": redact_trajectory_text(json.dumps(text)),
        "obj": json.dumps(redact_trajectory_obj({"content": text})),
        "acp": redact_acp_trajectory_jsonl(
            [{"type": "tool_call", "raw_input": {"content": text}}]
        ),
        "publish": json.dumps(redact_value({"content": text})[0]),
    }


def _object_layers(value: dict) -> dict[str, str]:
    """The layers' output for an object that holds the name as a key."""
    return {
        "text": redact_trajectory_text(json.dumps(value)),
        "rec": redact_trajectory_text(json.dumps(json.dumps(value))),
        "obj": json.dumps(redact_trajectory_obj(value)),
        "acp": redact_acp_trajectory_jsonl([{"type": "tool_call", "raw_input": value}]),
        "publish": json.dumps(redact_value(value)[0]),
    }


def _shown(layers: dict[str, str], word: str) -> list[str]:
    return [layer for layer, out in layers.items() if word in out]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("token", TOKENS)
def test_the_four_forms_mask_a_bare_value_in_every_layer(name, token):
    forms = {
        "colon": _text_layers(f"{name}: {token}"),
        "equals": _text_layers(f"{name}={token}"),
        "json key": _object_layers({name: token}),
        "nested object": _object_layers({"cfg": {name: token}}),
    }
    shown = {form: _shown(layers, token) for form, layers in forms.items()}
    if name in CLASSIFIED_KEYS:
        # publish decides an object's keys with the classifier (entry_kind),
        # which calls a prose-shaped value under a prefixed name a setting,
        # as 0.9.3 does; every other layer masks it.
        for form in ("json key", "nested object"):
            shown[form] = [layer for layer in shown[form] if layer != "publish"]
    assert shown == {form: [] for form in forms}, shown


@pytest.mark.parametrize("name", NAMES)
def test_a_scheme_before_the_token_is_read_as_before(name):
    token = "eyJhbGciOiJIUzI1NiJ9FAKEtoken"
    assert redact_trajectory_text(f"{name}: Bearer {token}") == f"{name}: Bearer {MASK}"
    assert redact_trajectory_obj({name: f"Bearer {token}"}) == {name: f"Bearer {MASK}"}
    for layers in (
        _text_layers(f"{name}: Basic {token}"),
        _object_layers({name: f"Token {token}"}),
    ):
        assert _shown(layers, token) == []


@pytest.mark.parametrize("name", NAMES)
def test_a_scheme_word_alone_stays(name):
    assert redact_trajectory_text(f"{name}: Bearer") == f"{name}: Bearer"
    assert redact_trajectory_obj({name: "Bearer"}) == {name: "Bearer"}


def test_the_prose_control_is_masked_whole():
    text = "Authorization: please retry later"
    assert redact_trajectory_text(text) == f"Authorization: {MASK}"
    assert redact_trajectory_text(f"Authorization: {MASK}") == f"Authorization: {MASK}"
    for word in ("please", "retry", "later"):
        assert _shown(_text_layers(text), word) == [], word


def test_prose_that_names_the_header_without_a_separator_stays():
    for text in (
        "the Authorization header is missing",
        "set Authorization in the request",
    ):
        assert redact_trajectory_text(text) == text


# R632's N4 (2026-11-16): after a name that ends in a shell, template or batch
# reference's close, a JSON copy's escaped tab after the separator is a space,
# as it is after every other Authorization name. f0c7afb90 masked the c227-tab
# sweep's `Authorization:\tnone\t${HTTP_AUTHORIZATION}:\t<v>` only because
# the first value's run took the second name with it; e80b37ed1, 35663d5ea and
# 307169b69 show <v>.
SHELL_NAMES = [
    "${HTTP_AUTHORIZATION}",
    "${HTTP_AUTHORIZATION:-x}",
    "{{HTTP_AUTHORIZATION}}",
    "$(Authorization)",
    "%AUTHORIZATION%",
]
N4_VALUE = "FAKEn4val0123456789xyz"


@pytest.mark.parametrize("name", SHELL_NAMES)
@pytest.mark.parametrize("lead", ["", "x ", "x\\t", "Authorization:\\tnone\\t"])
@pytest.mark.parametrize("gap", [":\\t", "=\\t", ":\\t\\t", ": \\t"])
def test_an_escaped_tab_after_a_shell_names_separator_is_a_space(name, lead, gap):
    for value in (N4_VALUE, f'"{N4_VALUE}"'):
        layers = _text_layers(f"{lead}{name}{gap}{value}")
        assert _shown(layers, N4_VALUE) == [], layers


@pytest.mark.parametrize("name", SHELL_NAMES)
def test_an_escaped_tab_and_a_space_read_the_same_after_a_shell_name(name):
    spaced = redact_trajectory_text(f"{name}: {N4_VALUE}")
    assert N4_VALUE not in spaced
    for gap in (":\t", ":\\t", ":\\\\t"):
        out = redact_trajectory_text(f"{name}{gap}{N4_VALUE}")
        assert out == spaced.replace(": ", gap), (gap, out)
