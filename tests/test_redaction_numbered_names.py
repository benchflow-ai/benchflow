"""A secret's name numbered after its ending.

Review of PR #173, 2026-11-13: DB_PASSWORD_2, API_TOKEN_2, GITHUB_TOKEN_1,
SECRET_2, AWS_SECRET_ACCESS_KEY_2 and DB_PASSPHRASE_2 kept an opaque value
in every layer (the classifier calls each a credential), as they had at
63f22b045: no rule read a number after a secret's name. The lead's ruling:
the secret-name carrier takes `_` and one to four digits after the name's
ending, as the *_PASS carrier does; here also the digits alone (dbPassword2,
as the snake path gives a camel-case name), the api-key and key-field
carriers, and the whole-name rule. Every value here is made up.
"""

from __future__ import annotations

import json

import pytest

from benchflow._utils.config_redaction import _should_record_env_entry
from benchflow._utils.credential_names import entry_kind
from benchflow.hosted_env import _masked_env_args
from benchflow.publish.redact import redact_value
from benchflow.trajectories.types import (
    is_secret_field,
    is_secret_number,
    redact_trajectory_obj,
    redact_trajectory_text,
)
from tests._timing import assert_linear, catches_quadratic, linear_seconds
from tests.test_redaction_ansi_spellings import _repeat
from tests.test_redaction_escape_boundary import LAYERS, _dict_leaks, _unchanged
from tests.test_redaction_escape_boundary import _leaks as _boundary_leaks
from tests.test_redaction_shape_matrix import _leaks as _path_leaks

MASK = "***REDACTED***"
SECRET = "FAKEnum0aZ9bY7Lm2Qx"

# The lead's six, the api-key and key-field carriers' names, and spellings.
NUMBERED = [
    "DB_PASSWORD_2",
    "API_TOKEN_2",
    "GITHUB_TOKEN_1",
    "SECRET_2",
    "AWS_SECRET_ACCESS_KEY_2",
    "DB_PASSPHRASE_2",
    "API_KEY_2",
    "OPENAI_API_KEY_2",
    "PRIVATE_KEY_1",
    "master_key_3",
    "db_password_2",
    "dbPassword2",
    "githubToken1",
    "GITHUB_TOKEN_1234",
    "SECRET2",
]


@pytest.mark.parametrize("name", NUMBERED)
def test_a_numbered_secrets_name_is_masked_in_every_layer(name):
    """Review of PR #173, 2026-11-13: in text (as written, quoted, after an
    escaped newline and a colour code, as a JSON string), as an object key,
    at export, in the hosted env and in config.json, as the classifier
    says."""
    assert entry_kind(name, SECRET) == "credential"
    assert redact_trajectory_text(f"{name}={SECRET}") == f"{name}={MASK}"
    for text in (
        f"{name}={SECRET}",
        f"{name}: {SECRET}",
        f'"{name}": "{SECRET}"',
        f"out\\n{name}={SECRET}",
        f"out\x1b[32m{name}={SECRET}",
    ):
        assert _boundary_leaks(text, SECRET) == [], text
        assert _path_leaks(text, SECRET) == [], text
    assert _dict_leaks({name: SECRET}, SECRET) == []
    assert is_secret_field(name, SECRET)
    assert redact_trajectory_obj({name: SECRET}) == {name: MASK}
    assert redact_value({name: SECRET})[0] == {name: "<XXX-benchflow-key-values-XXX>"}
    assert _masked_env_args({name: SECRET}) == {name: MASK}
    assert not _should_record_env_entry(name, SECRET)
    assert is_secret_number(name, 482139)


# A count, a switch, a hint and a price after a number: settings to the
# classifier, kept. (Eight more rows were here, which the classifier calls
# credentials: they moved to CLASSIFIER_CREDENTIALS below by the lead's
# ruling, part B, 2026-11-14.)
KEPT = [
    "MAX_TOKENS_2=8192",
    "TOKEN_2FA_ENABLED=true",
    "PASSWORD_2_HINT=dog",
    "input_cost_per_token_2=0.1",
]
# The lead's ruling (part B, 2026-11-14): a numbered name is the
# classifier's (R1: a number right after a credential word, perhaps after
# one to three letters), so these, which it calls credentials, are masked in
# every layer: whole name, export, trajectory files and text. They were
# KEPT here; the bare words key, tokens and session still name nothing.
CLASSIFIER_CREDENTIALS = [
    "TOKEN_2FA=required",
    "SECRET_2_OF_3=north",
    "KEY_2=north",
    "TOKENS_2=north",
    "SESSION_2=north",
    "PRIMARY_KEY_2=north",
    "TOKEN_V2=north",
    "TOKEN_12345=north",
]


@pytest.mark.parametrize("text", KEPT)
def test_names_beside_a_numbered_secret_are_kept(text):
    """Review of PR #173, 2026-11-13: in text, after an escaped newline and
    a colour code, as a JSON string, and as an object key at export."""
    name, value = text.split("=", 1)
    for form in (
        text,
        f"out\\n{text}",
        f"out\x1b[32m{text}",
        json.dumps({name: value}),
    ):
        assert _unchanged(form) == dict.fromkeys(LAYERS, True), form
    assert not is_secret_field(name, value)
    assert redact_trajectory_obj({name: value}) == {name: value}
    assert redact_value({name: value}) == ({name: value}, 0)


@pytest.mark.parametrize("text", CLASSIFIER_CREDENTIALS)
def test_names_the_classifier_calls_credentials_are_masked(text):
    """Part B, the lead's ruling: in text, after an escaped newline
    and a colour code, as a JSON string, as an object key at export and in
    trajectory files, and as a whole name."""
    name, value = text.split("=", 1)
    assert entry_kind(name, value) == "credential"
    for form in (
        text,
        f"out\\n{text}",
        f"out\x1b[32m{text}",
        json.dumps({name: value}),
    ):
        leaked = {*_boundary_leaks(form, value), *_path_leaks(form, value)}
        assert leaked == set(), form
    assert is_secret_field(name, value)
    assert redact_trajectory_obj({name: value}) == {name: MASK}
    assert redact_value({name: value})[0] == {name: "<XXX-benchflow-key-values-XXX>"}


def test_a_numbered_passphrase_flag_is_a_bare_passphrase_name():
    """Review of PR #173, 2026-11-13: has_passphrase_2 is no flag to the
    classifier (its last word is the number), so a switch under it is
    masked, as under SSH_KEY_PASSPHRASE; has_passphrase keeps it."""
    assert entry_kind("has_passphrase_2", "False") == "credential"
    assert (
        redact_trajectory_text("has_passphrase_2=False") == f"has_passphrase_2={MASK}"
    )
    assert is_secret_field("has_passphrase_2", "False")
    assert redact_trajectory_text("has_passphrase=False") == "has_passphrase=False"


def test_a_numbered_price_is_a_price():
    """Review of PR #173, 2026-11-13: the price rule holds with a number
    after the name, as the classifier says."""
    assert entry_kind("input_cost_per_token_2", "0.1") == "setting"
    assert redact_trajectory_text("input_cost_per_token_2=0.1") == (
        "input_cost_per_token_2=0.1"
    )
    assert not is_secret_field("input_cost_per_token_2", "0.1")


@pytest.mark.parametrize("unit", ["SECRET_2=", "TOKEN_1234", "TOKEN_12345", "_2"])
def test_numbered_names_repeated_stay_linear(unit):
    """Review of PR #173, 2026-11-13: each repeated to 50,000 and 200,000
    characters, then a value: four times the text takes about four times as
    long."""
    seconds, _ = linear_seconds(
        redact_trajectory_text, lambda size: _repeat(unit, size) + SECRET
    )
    assert_linear(seconds)


# The lead's ruling (2026-11-14): the timing tests above call the
# shared check (tests/_timing.py). The positive control: the check, on one
# of each test's inputs at its sizes, fails once a quadratic term is added.
def test_the_numbered_name_timing_check_catches_a_quadratic():
    assert catches_quadratic(
        redact_trajectory_text, lambda size: _repeat("SECRET_2=", size) + SECRET
    )
