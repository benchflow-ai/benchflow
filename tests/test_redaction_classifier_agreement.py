"""The redactor's whole-name answers are the classifier's.

Review of PR #173, 2026-11-13 (RPUB, MEDIUM, item 2). Every value here is
made up.

Two copies decided each rule, and they disagreed: GATES_PASS=3 and
BLOCKER_PASS=3 were settings to the classifier and masked by the redactor,
cost_per_user_token="0.000003" the same. Here, for every name of the
classifier's price, pass and PER_ rows (its SETTINGS and CREDENTIALS rows
whose words hold `per`, `pass` or a price word, PER_PRICES, PER_CREDENTIALS,
PER_TOKEN_PROBES, PER_AGREEMENT, KEPT_NUMBERS, MASKED_NUMBERS), every name
of the shape matrix, the lead's five rows and the classifier's
COMPARED_NAMES (but SESSION, a record field decided after the rc), in each
spelling (as written, upper, lower and camel case), with a count and an
opaque value:

- ``is_secret_field(name, value)`` is True exactly when
  ``entry_kind(name, value) == "credential"``;
- ``is_secret_number(name, n)`` likewise, but for the number rule's own
  exemptions, stated as rules in ``_number_exempt``.

The lead approved two pending sets, LOWER_CASE_PASS_PENDING and
HEADER_VALUE_RULE_PENDING, and no other exception. emptied both
and removed them (a ``pass`` name in any case is the classifier's, and so
is a name that ends in a Cookie or Authorization header's word): no name
compared here is an exception. part B compares the numbered names
too (NEXT_PART is gone): a numbered name, a hint and a primary or a
secondary key's name are the classifier's, and is_secret_field asks it for
every name it reads (test_is_secret_field_follows_the_classifier).
"""

from __future__ import annotations

import json
import re

import pytest

from benchflow._utils.credential_names import _PASS_AT, _PRICE_WORDS, _words, entry_kind
from benchflow.trajectories.types import (
    _snake_name,
    is_secret_field,
    is_secret_number,
    redact_trajectory_obj,
    redact_trajectory_text,
)
from tests import test_credential_names as classifier_tables
from tests import test_redaction_shape_matrix as shape_matrix

COUNT, OPAQUE = "12", "hunter2FAKE0aZ9"
SMALL, PIN = 1234, 123456
MASK = "***REDACTED***"


def _in_the_rules(name: str) -> bool:
    words = _words(_PASS_AT.sub(r"\1_at_\2", name))
    return "per" in words or "pass" in words or any(w in _PRICE_WORDS for w in words)


FIVE_ROWS = ["TESTS_PASS", "GATES_PASS", "BLOCKER_PASS", "FIRST_PASS", "DB_PASS"]
# Numbered secrets' names (the lead's ruling, review of PR #173, 2026-11-13).
NUMBERED = [
    "DB_PASSWORD_2",
    "API_TOKEN_2",
    "GITHUB_TOKEN_1",
    "SECRET_2",
    "AWS_SECRET_ACCESS_KEY_2",
    "DB_PASSPHRASE_2",
]
# The classifier's COMPARED_NAMES but this one: the bare word session is
# one of the four record fields that share a name with a credential (key,
# session, session_id, tokens), decided after the rc (the lead's ruling);
# no rule of the redactor reads it. No pending set: the numbered
# names, the hints and the primary key's (NEXT_PART until that change part B)
# are compared.
DEFERRED = frozenset({"SESSION"})
# Names the decided branch's pattern reads and its shape check did not
# (review R192B, M-B): the carrier kept their values whole.
PATTERN_ONLY = [
    "KEYV2_TOKEN",
    "TOKENS_2_SECRET",
    "APP_KEYS_2_PASSWORD",
    "mcpcanvasadmintoken1",
    "getpassword2",
]
# The names part B moved in (they were NEXT_PART).
PART_B = frozenset(
    {
        "KEY_2", "TOKEN_12345", "SECRET_2_OF_3", "PRIMARY_KEY_2", "TOKENS_2",
        "TOKEN_2FA", "SESSION_2", "TOKEN_V2",
        "KEY_10", "CIRCLE_OIDC_TOKEN_V2", "TOKEN_2_FILE_PATH",
        "KEY_X509C_FILE", "id_token_hint", "key_hint",
    }
)  # fmt: skip
NAMES = sorted(
    {
        *(name for name in classifier_tables.COMPARED_NAMES if name not in DEFERRED),
        *(
            name
            for name in (*classifier_tables.SETTINGS, *classifier_tables.CREDENTIALS)
            if _in_the_rules(name)
        ),
        *classifier_tables.PER_PRICES,
        *classifier_tables.PER_CREDENTIALS,
        *classifier_tables.PER_TOKEN_PROBES,
        *classifier_tables.PER_AGREEMENT,
        *classifier_tables.KEPT_NUMBERS,
        *classifier_tables.MASKED_NUMBERS,
        *(name for name, _, _ in shape_matrix.CARRIERS.values()),
        *FIVE_ROWS,
        *NUMBERED,
        # A path or the file's content (the lead's ruling, part B).
        "GOOGLE_APPLICATION_CREDENTIALS",
        *PATTERN_ONLY,
    }
)


def _camel(name: str) -> str:
    words = [word for word in re.split(r"[_.\-/\s]+", name) if word]
    if len(words) < 2:
        return name
    return words[0].lower() + "".join(word.capitalize() for word in words[1:])


SPELLINGS = sorted(
    {
        spelled
        for name in NAMES
        for spelled in (name, name.upper(), name.lower(), _camel(name))
    }
)


def _number_exempt(name: str, number: int) -> bool:
    """The number rule's own exemptions: no integer under four digits, and
    none under a name with the word `tokens` (a count of them), one that
    ends in `id` or `ids`, or a schema's `primary_key` or `foreign_key`."""
    snake = _snake_name(name)
    words = snake.split("_")
    return (
        abs(number) < 1000
        or "tokens" in words
        or words[-1] in ("id", "ids")
        or snake in ("primary_key", "foreign_key")
    )


def _disagreements(name: str) -> list[tuple[str, object, bool, bool]]:
    """(kind, value, the redactor's answer, the classifier's) where they differ."""
    rows = []
    for value in (COUNT, OPAQUE):
        mine, theirs = (
            is_secret_field(name, value),
            entry_kind(name, value) == "credential",
        )
        if mine != theirs:
            rows.append(("string", value, mine, theirs))
    for number in (SMALL, PIN):
        mine = is_secret_number(name, number)
        theirs = (
            not _number_exempt(name, number)
            and entry_kind(name, number) == "credential"
        )
        if mine != theirs:
            rows.append(("number", number, mine, theirs))
    return rows


# The header names the two pending sets held, by name, as the comparison
# reaches them.
HEADER_NAMES = {
    "HTTP_COOKIE", "httpCookie", "http_cookie",
    "HTTP_AUTHORIZATION", "httpAuthorization", "http_authorization",
    "REDIRECT_HTTP_AUTHORIZATION", "redirectHttpAuthorization",
    "redirect_http_authorization",
}  # fmt: skip


def test_the_comparison_covers_the_tables():
    """The header names the pending set held, the numbered names part B
    moved in and the classifier's COMPARED_NAMES (but the deferred record
    field) are among the spellings compared, and the comparison is not
    empty."""
    assert len(NAMES) > 120 and len(SPELLINGS) > 300
    assert set(SPELLINGS) >= HEADER_NAMES
    assert set(classifier_tables.COMPARED_NAMES) > DEFERRED
    assert set(NAMES) >= set(classifier_tables.COMPARED_NAMES) - DEFERRED
    assert set(NAMES) >= PART_B and not set(NAMES) & DEFERRED


@pytest.mark.parametrize("name", SPELLINGS)
def test_the_redactor_and_the_classifier_agree(name):
    """Review of PR #173, 2026-11-13 (RPUB, MEDIUM item 2): one answer per
    name, string and number, count and opaque value."""
    assert _disagreements(name) == []


def test_is_secret_field_follows_the_classifier(monkeypatch):
    """Part B, the lead's ruling: is_secret_field holds no copy of
    the classifier's rules. Over every compared spelling, whatever
    entry_kind answers for a name the redactor reads, is_secret_field
    answers the same; a spelling it reads under one answer it reads under
    the other."""
    from benchflow.trajectories import types

    read = {}
    for kind in ("credential", "setting"):
        monkeypatch.setattr(types, "entry_kind", lambda name, value, k=kind: k)
        read[kind] = {name for name in SPELLINGS if is_secret_field(name, OPAQUE)}
    assert read["setting"] == set()
    monkeypatch.undo()
    real = {name for name in SPELLINGS if entry_kind(name, OPAQUE) == "credential"}
    assert real <= read["credential"]
    assert {name for name in SPELLINGS if is_secret_field(name, OPAQUE)} == real


def test_the_whole_name_rule_is_the_classifiers_without_a_value():
    """Part B: with no value (a caller that has none, config.json's
    name test), is_secret_field is entry_kind's answer for null, which is
    name_kind's but for the names whose value decides."""
    from benchflow._utils.credential_names import name_kind

    for name in SPELLINGS:
        secret = is_secret_field(name)
        assert secret is (entry_kind(name, None) == "credential"), name
        if entry_kind(name, None) == entry_kind(name, OPAQUE) == name_kind(name):
            assert secret is (name_kind(name) == "credential"), name


def test_no_compared_name_is_an_exception():
    """Review of PR #173, 2026-11-13, and no pending set is left
    and no name disagrees. (It was test_the_two_pending_sets_are_the_only_
    exceptions; emptied LOWER_CASE_PASS_PENDING and then
    HEADER_VALUE_RULE_PENDING, and removed both.)"""
    disagreeing = {name for name in SPELLINGS if _disagreements(name)}
    assert disagreeing == set()


# A passphrase's name holds a switch as no secret only as a flag's (the
# lead's ruling, review of PR #173, 2026-11-13): the classifier's rows for
# passphrase names (its ENTRY_KINDS), with the lead's four.
PASSPHRASE_ROWS = sorted(
    {
        *(row for row in classifier_tables.ENTRY_KINDS if "phrase" in row[0].lower()),
        ("has_passphrase", "False", "setting"),
        ("requires_passphrase", "true", "setting"),
        ("SSH_KEY_PASSPHRASE", "false", "credential"),
        ("PASSPHRASE", "no", "credential"),
    },
    key=repr,
)


@pytest.mark.parametrize(("name", "value", "kind"), PASSPHRASE_ROWS)
def test_a_passphrase_row_is_the_classifiers_in_the_redactors_rules(name, value, kind):
    """Review of PR #173, 2026-11-13: the whole-name rule, the number rule,
    the text rule (``name=value``, and the row as JSON) and the object path
    give the classifier's answer for the value as it is."""
    assert entry_kind(name, value) == kind
    masked = kind == "credential"
    assert is_secret_field(name, value) is masked
    serialized = json.dumps({name: value})
    assert (redact_trajectory_text(serialized) != serialized) is masked
    if isinstance(value, str):
        text = f"{name}={value}"
        assert redact_trajectory_text(text) == (f"{name}={MASK}" if masked else text)
        assert redact_trajectory_obj({name: value}) == {name: MASK if masked else value}
    else:
        number = isinstance(value, int) and not isinstance(value, bool)
        expected = masked and number and not _number_exempt(name, value)
        assert is_secret_number(name, value) is expected


# The lead's five rows (TESTS_PASS twice), in the redactor's text rule, its
# object path, its whole-name rule and its number rule; the classifier
# agent tests config.json and the override record.
FIVE = [
    ("TESTS_PASS", "hunter2FAKE", True),
    ("TESTS_PASS", "12", False),
    ("GATES_PASS", "3", False),
    ("BLOCKER_PASS", "3", False),
    ("FIRST_PASS", "hunter2FAKE", True),
    ("DB_PASS", "hunter2FAKE", True),
]


@pytest.mark.parametrize("name,value,masked", FIVE)
def test_the_five_rows_in_the_redactors_callers(name, value, masked):
    """Review of PR #173, 2026-11-13: masked where the classifier says
    credential, kept where it says setting."""
    assert entry_kind(name, value) == ("credential" if masked else "setting")
    text = f"{name}={value}"
    assert redact_trajectory_text(text) == (f"{name}={MASK}" if masked else text)
    assert redact_trajectory_obj({name: value}) == {name: MASK if masked else value}
    assert is_secret_field(name, value) is masked
    if masked:
        assert is_secret_number(name, PIN)
    else:
        assert not is_secret_number(name, SMALL)
        assert redact_trajectory_obj({name: SMALL}) == {name: SMALL}


@pytest.mark.parametrize(
    "name", sorted({*PATTERN_ONLY, *(n.upper() for n in PATTERN_ONLY)})
)
def test_the_text_carrier_masks_what_the_classifier_calls_a_credential(name):
    """Review R192B, M-B: one implementation in text too. Where entry_kind
    calls NAME=<value> a credential's, the carrier masks the value in env
    text, JSON text and the object layer."""
    assert entry_kind(name, OPAQUE) == "credential"
    for out in (
        redact_trajectory_text(f"{name}={OPAQUE}"),
        redact_trajectory_text(json.dumps({name: OPAQUE})),
        json.dumps(redact_trajectory_obj({name: OPAQUE})),
    ):
        assert OPAQUE not in out


# The lead's ruling after the delta review of PR #192 (A.1): under a code
# Authorization name (has_authorization, requires_authorization) a value is a
# setting's only as code (the classifier's _CODE_VALUE) or prose, and the
# text carrier reads it so. R192B's differential found 20 cells the rc masked
# and part A kept: an identifier-shaped opaque value and a 44-character key
# after `name:` and in `name=<v> done`, under five names.
A1_NAMES = [
    "has_authorization",
    "http_authorization",
    "my_authorization",
    "redirect_http_authorization",
    "requires_authorization",
]
A1_OPAQUE = "FAKEdiff0123456789abcdefXYZ"
A1_KEY = "FAKEdk44abcdefghijklmnopqrstuvwxyz0123456789"
A1_CODE = [
    ("has_authorization: bool", "bool"),
    ("requires_authorization = self.flag", "self.flag"),
    # The name's own words, and a short all-letters word (prose, the named
    # residual).
    (
        "copy_source_authorization=copy_source_authorization",
        "copy_source_authorization",
    ),
    ("'copy_source_authorization': source_authorization", "source_authorization"),
    ("copy_source_authorization = flag", "flag"),
]
A1_MASKED = [
    ("copy_source_authorization=authorization_source", "authorization_source"),
    ("copy_source_authorization = flag2x9", "flag2x9"),
]


@pytest.mark.parametrize(("text", "value"), A1_CODE)
def test_code_under_a_code_authorization_name_is_masked_as_the_classifier_says(
    text, value
):
    """Code, the name's own words and prose were a setting's to the
    classifier and kept in text. Under the lead's ruling (b), 2026-11-15 01:3xZ they are a
    credential's, and masked in text."""
    name = re.split(r"\s*[:=]\s*", text)[0].strip("'")
    assert entry_kind(name, value) == "credential"
    out = redact_trajectory_text(text)
    assert out != text
    assert out.endswith("***REDACTED***") or "***REDACTED***," in out, out


@pytest.mark.parametrize(("text", "value"), A1_MASKED)
def test_another_bare_identifier_is_masked_as_the_classifier_says(text, value):
    """A reordered name and a word with digits are a credential's: masked."""
    name = re.split(r"\s*[:=]\s*", text)[0]
    assert entry_kind(name, value) == "credential"
    assert value not in redact_trajectory_text(text)


@pytest.mark.parametrize("name", A1_NAMES)
@pytest.mark.parametrize("value", [A1_OPAQUE, A1_KEY])
def test_the_twenty_cells_are_masked_as_the_classifier_says(name, value):
    """R192B's 20 cells: a credential's to the classifier, masked in text."""
    assert entry_kind(name, value) == "credential"
    for text in (f"{name}: {value}", f"{name}={value} done"):
        assert value not in redact_trajectory_text(text), text


def test_a_call_around_a_long_quoted_key_is_masked():
    """The lead's ruling: a call around a 44-character quoted key is no code
    (its quoted name is longer than code's), so it is masked. The call
    around a short quoted name was the accepted residual; under
    the lead's ruling (b), 2026-11-15 01:3xZ it is masked too."""
    text = f'has_authorization = f("{A1_KEY}")'
    assert entry_kind("has_authorization", f'f("{A1_KEY}")') == "credential"
    assert A1_KEY not in redact_trajectory_text(text)
    assert A1_KEY not in json.dumps(redact_trajectory_obj({"message": text}))
    assert redact_trajectory_text('has_authorization = f("abc123")') == (
        "has_authorization = ***REDACTED***"
    )
