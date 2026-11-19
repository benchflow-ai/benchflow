"""The export redactor's placeholder key name is a setting (review of PR #192).

The export redactor (``benchflow.publish.redact.redact_value``) renames a
mapping key that is itself a credential to ``***REDACTED_KEY_<n>***``.
made the redactors follow the classifier for numbered names, and the
classifier read that name's words as a numbered key's (redacted, key, 1): a
credential. So a record that was redacted a second time lost what its
placeholder keys held, and
``tests/test_publish_azure_blob.py::test_contribution_redactor_replaces_credential_mapping_keys_without_collision``
(PR #989) failed. The lead's ruling (2026-11-14): exactly that spelling,
asterisks included, is a setting, through one shape the export redactor also
builds its renamed keys from; ``REDACTED_KEY_1``, ``KEY_1`` and every other
numbered name stay credentials; a key-shaped value under the placeholder is
still masked.

Every value here is made up; the key-shaped ones are built at run time.
"""

from __future__ import annotations

import json

import pytest

from benchflow._utils import credential_names
from benchflow._utils.credential_names import entry_kind, name_kind, placeholder_key
from benchflow.publish.redact import REDACTED, redact_value
from benchflow.trajectories.types import (
    is_secret_field,
    redact_trajectory_obj,
    redact_trajectory_text,
)

TOKENS = [
    "dtn_" + "1234567890abcdefghijklmnop",
    "hf_" + "1234567890abcdefghijklmnop",
    "ghp_" + "FAKE" * 9,
    "sk-ant-api03-" + "FAKE" * 4 + "1234",
]
OPAQUE = "FAKEtok" + "0123456789abcdef"

PLACEHOLDERS = [placeholder_key(index) for index in (0, 1, 2, 9, 10, 123456)]

#: Near the placeholder, not it: each stays what the numbered-name rules say.
NOT_PLACEHOLDERS = [
    "REDACTED_KEY_1",
    "KEY_1",
    "**REDACTED_KEY_1**",
    "***REDACTED_KEY_1**",
    "***REDACTED_KEY_1***_2",
    "x***REDACTED_KEY_1***",
    "***REDACTED_KEY_1*** ",
    "***redacted_key_1***",
    "***REDACTED_KEY_1***\n",
    "***REDACTED_TOKEN_1***",
]

#: The same with no number: credentials to the classifier. The redactor's
#: whole-name rule reads a ``key`` name only with a number or a provider's
#: word, here as on the release line, so these rows ask the classifier only.
NOT_PLACEHOLDERS_UNNUMBERED = ["***REDACTED_KEY_***", "***REDACTED_KEY_one***"]


def test_the_placeholder_is_spelled_in_one_place():
    """The shape the classifier reads is built from the template the export
    redactor's keys are built from."""
    assert placeholder_key(1) == "***REDACTED_KEY_1***"
    assert credential_names._PLACEHOLDER_KEY.format(7) == placeholder_key(7)
    for name in PLACEHOLDERS:
        assert credential_names._PLACEHOLDER_KEY_NAME.fullmatch(name), name


@pytest.mark.parametrize("name", PLACEHOLDERS)
@pytest.mark.parametrize("value", ["reserved", OPAQUE, None, 482139, ["a"], {"a": 1}])
def test_the_exact_placeholder_is_a_setting(name, value):
    """The lead's ruling (2026-11-14), for the classifier and the redactor's
    whole-name rule alike."""
    assert name_kind(name) == "setting"
    assert entry_kind(name, value) == "setting"
    assert is_secret_field(name, value) is False


@pytest.mark.parametrize("name", NOT_PLACEHOLDERS)
def test_any_other_spelling_stays_a_credential(name):
    """Only the exact spelling is the placeholder: a numbered key's name
    with other asterisks, case or text around it is a credential still."""
    assert credential_names._PLACEHOLDER_KEY_NAME.fullmatch(name) is None
    assert name_kind(name) == "credential"
    assert entry_kind(name, OPAQUE) == "credential"
    assert is_secret_field(name, OPAQUE) is True


@pytest.mark.parametrize("name", NOT_PLACEHOLDERS_UNNUMBERED)
def test_an_unnumbered_spelling_is_no_placeholder(name):
    """The placeholder always carries its number."""
    assert credential_names._PLACEHOLDER_KEY_NAME.fullmatch(name) is None
    assert name_kind(name) == "credential"
    assert entry_kind(name, OPAQUE) == "credential"


def test_every_key_the_export_redactor_renames_is_the_placeholder():
    """Each renamed key matches the classifier's shape and is a setting,
    with placeholder names already in the mapping skipped over."""
    value = {token: f"held {index}" for index, token in enumerate(TOKENS)}
    value[placeholder_key(1)] = "reserved"
    value[placeholder_key(3)] = "kept too"
    value["plain"] = "x"

    redacted, replacements = redact_value(value)

    assert replacements == len(TOKENS)
    assert not any(token in str(redacted) for token in TOKENS)
    renamed = [key for key in redacted if key not in value]
    assert len(renamed) == len(TOKENS)
    for key in renamed:
        assert credential_names._PLACEHOLDER_KEY_NAME.fullmatch(key), key
        assert entry_kind(key, redacted[key]) == "setting"
    assert redacted[placeholder_key(1)] == "reserved"
    assert redacted[placeholder_key(3)] == "kept too"
    assert sorted(redacted[key] for key in renamed) == [
        f"held {index}" for index in range(len(TOKENS))
    ]


def test_a_redacted_record_is_unchanged_by_a_second_pass():
    """What the ruling is for: redacting an exported record again replaces
    nothing and keeps what the placeholder keys hold, in the export and the
    trajectory object redactors."""
    value = {
        TOKENS[0]: "first",
        "nested": {TOKENS[1]: {"note": "hello there", "n": 3}},
        "rows": [{TOKENS[2]: "third"}],
    }
    once, replacements = redact_value(value)
    assert replacements == 3

    twice, again = redact_value(once)

    assert (twice, again) == (once, 0)
    assert redact_trajectory_obj(once) == once
    for text in (json.dumps(once), json.dumps(once, indent=1)):
        assert redact_trajectory_text(text) == text


@pytest.mark.parametrize("token", TOKENS)
def test_a_key_shaped_value_under_the_placeholder_is_still_masked(token):
    """The value rules do not depend on the name."""
    redacted, replacements = redact_value({placeholder_key(1): token})

    assert replacements == 1
    assert token not in str(redacted)
    assert redacted == {placeholder_key(1): REDACTED}
    assert token not in str(redact_trajectory_obj({placeholder_key(1): token}))
    assert token not in redact_trajectory_text(json.dumps({placeholder_key(1): token}))
