"""A carrier's `:` or `=` URL-encoded, and a form-encoded space.

Review of PR #173, 2026-11-13 (RPUB, MEDIUM item 4). Every value here is
made up.

Every carrier read its name's separator as a literal `:` or `=`, so
``Authorization%3A%20Basic%20<b64>`` (a header in a URL or a form body) and
``NAME%3D<value>`` (every carrier name: HF_TOKEN, DB_PASS, password, ...)
kept their secret in every layer. The carriers now share one separator
(``_SEP``: `:` or `=`, or ``%3A`` / ``%3D`` encoded once or twice), and a
space between a separator, a scheme and a token may be ``%20``, ``%2520``
or ``+`` (``_WS``), though not as a next line's indent (``_INDENT``: a
diff's added line).

A secret name's value that opens with an Authorization scheme and a space
(``x-api-key%3DBearer <token>``, ``api_key: Bearer <token>``) keeps the
scheme and masks the token, as the Authorization carrier does (RPUB's
MEDIUM at 860713e39: the encoded form's token reached publish, and the
plain forms leaked in every layer on the rc too).

The reviewer's matrix (234 cells) is committed unchanged as
``fixtures/redaction/rpub_auth_encoded_cells.json`` and driven cell by cell
through the layer it names; the forms beside it are this file's own.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from benchflow._utils.config_redaction import _is_secret_env_value
from benchflow._utils.evaluation_results import _masked
from benchflow.hosted_env import _masked_env_args
from benchflow.publish.redact import redact_value
from benchflow.trajectories.types import (
    redact_acp_trajectory_jsonl,
    redact_key_families,
    redact_trajectory_obj,
    redact_trajectory_text,
)
from tests._timing import assert_linear, catches_quadratic, linear_seconds
from tests.test_redaction_ansi_spellings import _repeat
from tests.test_redaction_escape_boundary import LAYERS, _unchanged
from tests.test_redaction_escape_boundary import _leaks as _boundary_leaks
from tests.test_redaction_shape_matrix import _leaks as _path_leaks

FIXTURE = (
    Path(__file__).parent / "fixtures" / "redaction" / "rpub_auth_encoded_cells.json"
)
FIXTURE_SHA256 = "5f2d806bd71f44fb41d2509ce73a522eff09db48bf3cbdaa34c668639846aa40"
CELLS = json.loads(FIXTURE.read_text())
MASK = "***REDACTED***"

# How each of the matrix's nine layers is driven.
DRIVERS = {
    "text": redact_trajectory_text,
    "export": redact_trajectory_text,  # a serialized line, redacted as text
    "_masked": _masked,  # result.json's error text
    "judge": redact_key_families,  # what the task judge is shown
    "obj": redact_trajectory_obj,
    "acp": redact_acp_trajectory_jsonl,
    "hosted": _masked_env_args,  # a hosted run's recorded env args
    "publish": lambda value: redact_value(value)[0],
}


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def _flat(value) -> str:
    if isinstance(value, dict):
        return " ".join(_flat(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_flat(v) for v in value)
    return str(value)


def _prefix(cell) -> tuple[str, str]:
    """The prefix the cell's `expected` says stays, and its tail."""
    match = re.search(r"the prefix '(.*)' stays", cell["expected"])
    head, _, tail = match.group(1).partition(" ... ") if match else ("", "", "")
    return head, tail


def _outcome(cell) -> str:
    """masked: the made-up value is gone and the stated prefix stays;
    kept: the input comes through unchanged (or, under a by-design judge
    cell, the value does); leak: the value survives a masking layer."""
    layer, value = cell["layer"], cell["input"]
    if layer == "config":
        secret = _is_secret_env_value(value["env_name"], value["env_value"])
        return "masked" if secret else "kept"
    out = DRIVERS[layer](value)
    if cell["expected"] == "kept":
        if layer == "acp":
            return "kept" if out == "\n".join(map(json.dumps, value)) else "changed"
        return "kept" if out == value else "changed"
    head, tail = _prefix(cell)
    if not head:
        return "kept" if _flat(value) in _text(out) else "changed"
    text = _flat(value)
    secret = text[text.index(head) + len(head) :]
    secret = secret[: secret.index(tail)] if tail else secret
    if secret in _text(out):
        return "leak"
    return "masked" if head in _text(out) and tail in _text(out) else "prefix lost"


# The reviewer's control under a code Authorization name, kept in the
# file, changes in every redacting layer: every value there is masked
# (the lead's ruling (b), 2026-11-15 01:3xZ).
RULED_MASKED_CONTROLS = {"has_authorization%3D1"}
# Lead 2026-11-15 13:18Z, (A), with ruling (b): the decoded views
# read `%3A` and `%20` as the colon and space they write, so the doc-text
# control reads as `set the Authorization: header to Bearer <token>`, which
# every redacting layer masks after `header`, as it masks the literal form.
RULED_DECODED_CONTROLS = {"doc text naming the header"}

# The reviewer's prose controls after an Authorization name and an encoded
# separator change too: a bare value under an Authorization name is masked,
# prose or not (the lead's ruling of 2026-11-16 08:46Z). The file stays as the reviewer wrote it.
RULED_MASKED_CONTROLS |= {
    "Authorization%3A%20required (prose)",
    "state=authorization%3Apending in a query",
    "doc text naming the header",
    "authorization%3A%20none",
}


def _wanted(cell) -> str:
    if (
        cell["form"] in RULED_MASKED_CONTROLS | RULED_DECODED_CONTROLS
        and cell["layer"] != "judge"
    ):
        return "masked" if cell["layer"] == "config" else "changed"
    if cell["expected"].startswith(
        ("the made-up value is gone", "the env value is not")
    ):
        return "masked"
    return "kept"  # "kept", and the judge's by-design cells


def test_the_matrix_is_the_reviewers_file():
    """The matrix the reviewer re-runs is the one the suite runs."""
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    assert len(CELLS) == 234
    assert sum(cell["in_the_92"] is True for cell in CELLS) == 92


@pytest.mark.parametrize(
    "cell",
    CELLS,
    ids=[f"{n}-{cell['group']}-{cell['layer']}" for n, cell in enumerate(CELLS)],
)
def test_each_cell_of_the_reviewers_matrix(cell):
    """Review of PR #173, 2026-11-13 (RPUB): the 92 are masked with their
    prefix kept, every control is kept, and the judge's by-design cells keep
    the carrier's value (it masks key families only)."""
    assert _outcome(cell) == _wanted(cell), cell["form"]


# The forms beside the reviewer's: every carrier name with its separator
# URL-encoded, in a query string, an encoded query and a form body.
SECRET = "FAKEsep0aZ9bY7Lm2Qx"
NAME_FORMS = [
    "DB_PASS%3D{v}",
    "SMTP_PASS%3d{v}",
    "ADMIN_PASS%3D{v}",
    "MYSQL_PWD%253D{v}",
    "password%3D{v}",
    "passwd%3D{v}",
    "PASSPHRASE%3D{v}",
    "HF_TOKEN%3D{v}",
    "GITHUB_TOKEN%253D{v}",
    "AWS_BEARER_TOKEN_BEDROCK%3D{v}",
    "api_key%3D{v}",
    "master_key%3D{v}",
    "x-api-key%3A%20{v}",
    "Cookie%3A%20sid%3D{v}",
    "Authorization%3A%20{v}",
    "Authorization%3A+Token+{v}",
]
CONTEXTS = {
    "bare": "{}",
    "query": "https://h.example/cb?state=x&data={}&n=1",
    "encoded query": "https://app.example/login?next=https%3A%2F%2Fapi.example%2Fv1%3F{}%26page%3D2",
    "form body": "grant_type=client_credentials&{}&scope=read",
}


@pytest.mark.parametrize("form", NAME_FORMS)
@pytest.mark.parametrize("context", list(CONTEXTS))
def test_a_carrier_name_with_an_encoded_separator_is_masked(form, context):
    """Review of PR #173, 2026-11-13 (RPUB): `NAME%3D<value>` kept the value
    for every carrier name. In text, a dict value, both ACP writers,
    result.json's ``_masked``, a serialized line, the export redactor and a
    hosted run's env args, and config.json does not record it."""
    raw = CONTEXTS[context].format(form.format(v=SECRET))
    assert {*_boundary_leaks(raw, SECRET), *_path_leaks(raw, SECRET)} == set()
    assert SECRET not in json.dumps(_masked_env_args({"note": raw}))
    assert _is_secret_env_value("NOTE", raw)


KEPT = [
    "q%3Dhello",
    "page%3D2&sort%3Dname",
    "redirect%3Ahome",
    "https://h.example/search?q=100%25%3A&x=a%3Db",
    "/files/a%3Db/c",
    "dGVzdCs/dGVzdA==",
    "aGVsbG8rd29ybGQ+Pz8/",
    "C++ and C#",
    "1+1=2",
    "+1 555 0100",
    "x=1&sort=a+b",
    "key = a+b",
    "-api_token=\n+the next line of a diff",
]


@pytest.mark.parametrize("text", KEPT)
def test_ordinary_encoded_text_and_plus_signs_are_kept(text):
    """Review of PR #173, 2026-11-13 (RPUB): encoded queries without a
    carrier name, percent signs, base64 with `+`, `/` and `=`, sums, phone
    numbers, and a diff's added line after a name with no value (`+` is no
    indent)."""
    assert _unchanged(text) == dict.fromkeys(LAYERS, True)


def test_a_plus_is_a_space_on_the_same_line():
    """Review of PR #173, 2026-11-13 (RPUB): a scheme joined to its token
    with `+` is masked; a `+` before a value is kept as the space it is."""
    assert redact_trajectory_text(f"Authorization: Bearer+{SECRET}") == (
        "Authorization: Bearer+***REDACTED***"
    )
    assert redact_trajectory_text("token=+1") == "token=+***REDACTED***"


# A secret name, a separator (plain or encoded), an Authorization scheme,
# a space and a token: RPUB's four forms (review of PR #173, 2026-11-13,
# MEDIUM), each scheme in at least one, and the other carriers whose value
# is the token (the key field, BEARER_TOKEN, a secret's name, *_PASS, a
# passphrase's name).
SCHEME_TOKEN = "FAKEsc" + "0aZ9bY" * 3
SCHEMES = [
    "Bearer",
    "Token",
    "Basic",
    "ApiKey",
    "Negotiate",
    "NTLM",
    "Bot",
    "Key",
    "SSWS",
]
SCHEME_FORMS = [
    "x-api-key%3D{scheme} {token}",
    "x-api-key: {scheme} {token}",
    "api_key: {scheme} {token}",
    "OPENAI_API_KEY={scheme} {token}",
]
SCHEME_CARRIERS = [
    "x-api-key%253D{scheme}%2520{token}",
    "x-goog-api-key={scheme}+{token}",
    '"x-api-key": "{scheme} {token}"',
    "master_key: {scheme} {token}",
    "AWS_BEARER_TOKEN_BEDROCK={scheme} {token}",
    "GITHUB_TOKEN={scheme} {token}",
    "password: {scheme} {token}",
    "DB_PASS={scheme} {token}",
    "SSH_KEY_PASSPHRASE={scheme} {token}",
]


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("form", SCHEME_FORMS + SCHEME_CARRIERS)
def test_a_scheme_before_a_token_under_a_secret_name_is_masked(form, scheme):
    """Review of PR #173, 2026-11-13 (RPUB, MEDIUM): in every layer the token
    goes and the scheme stays, as written, after an escaped newline and a
    colour code, and as a string value; a second pass changes nothing."""
    text = form.format(scheme=scheme, token=SCHEME_TOKEN)
    assert redact_trajectory_text(text) == text.replace(SCHEME_TOKEN, MASK)
    for raw in (text, f"out\\n{text}", f"out\x1b[32m{text}"):
        assert _boundary_leaks(raw, SCHEME_TOKEN) == [], raw
        assert _path_leaks(raw, SCHEME_TOKEN) == [], raw
        once = redact_trajectory_text(raw)
        assert redact_trajectory_text(once) == once, raw


@pytest.mark.parametrize(
    "text,masked",
    [
        # The scheme word is the whole value: masked as before.
        ("TOKEN=Bearer", f"TOKEN={MASK}"),
        ("api_key: Bearer", f"api_key: {MASK}"),
        ("x-api-key%3DBearer", f"x-api-key%3D{MASK}"),
        ("api_key: Bearer\nnext", f"api_key: {MASK}\nnext"),
        # Prose after the scheme word: its first word masked as before.
        (
            "password: Basic authentication is required",
            f"password: {MASK} authentication is required",
        ),
        ("api_key: Key rotation", f"api_key: {MASK} rotation"),
    ],
)
def test_a_scheme_word_alone_or_before_prose_is_masked_as_before(text, masked):
    """Review of PR #173, 2026-11-13 (RPUB, MEDIUM): with no value after the
    scheme, or prose (a plain word), the scheme is the value."""
    assert redact_trajectory_text(text) == masked
    assert redact_trajectory_text(masked) == masked


@pytest.mark.parametrize(
    "unit", ["x-api-key: Bearer ", "api_key: Bearer Bearer ", "Bearer ", "TOKEN=Basic "]
)
def test_a_scheme_repeated_stays_linear(unit):
    """Review of PR #173, 2026-11-13 (RPUB, MEDIUM): a carrier and its
    scheme repeated, and the scheme word repeated with no value, to 50,000
    and 200,000 characters: four times the text takes about four times as
    long."""
    seconds, _ = linear_seconds(
        redact_trajectory_text, lambda size: _repeat(unit, size) + SCHEME_TOKEN
    )
    assert_linear(seconds)


@pytest.mark.parametrize(
    "unit",
    [
        "%3D",
        "+",
        "%2520",
        "password%3D",
        "DB_PASS%3D",
        "Authorization%3A+",
        "%3A+Basic+",
    ],
)
def test_encoded_separators_repeated_stay_linear(unit):
    """Review of PR #173, 2026-11-13 (RPUB): each repeated to 50,000 and
    200,000 characters, then a value: four times the text takes about four
    times as long."""
    seconds, _ = linear_seconds(
        redact_trajectory_text, lambda size: _repeat(unit, size) + SECRET
    )
    assert_linear(seconds)


# The lead's ruling (2026-11-14): the timing tests above call the
# shared check (tests/_timing.py). The positive control: the check, on one
# of each test's inputs at its sizes, fails once a quadratic term is added.
def test_the_scheme_timing_check_catches_a_quadratic():
    assert catches_quadratic(
        redact_trajectory_text,
        lambda size: _repeat("x-api-key: Bearer ", size) + SCHEME_TOKEN,
    )


def test_the_encoded_separator_timing_check_catches_a_quadratic():
    assert catches_quadratic(
        redact_trajectory_text, lambda size: _repeat("%3D", size) + SECRET
    )
