"""Numbered names, hints and a primary key's value: the classifier decides.

Part B. Every value here is made up.

The lead's rulings (2026-11-14):

- The redactor gives the classifier's answer for a numbered name (R1: a
  number right after a credential word, perhaps after one to three letters:
  KEY_2, KEY_10, TOKEN_12345, SECRET_2_OF_3, PRIMARY_KEY_2, TOKENS_2,
  TOKEN_2FA, SESSION_2, TOKEN_V2, CIRCLE_OIDC_TOKEN_V2, TOKEN_2_FILE_PATH,
  KEY_X509C_FILE) and for a hint (R2: id_token_hint, key_hint), as a whole
  name, at export, in trajectory files and in text. Its settings stay kept
  (MAX_TOKENS_2, TOKEN_2FA_ENABLED, PASSWORD_2_HINT, KEY_2_FILE).
- A usage or count name is a setting whatever number follows: the bare
  words' counts by cache TTL (tokens_1h, tokens_5m: LiteLLM's Bedrock cache
  writes) and every usage and cost field BenchFlow records.
- A primary or a secondary key's value decides (the classifier's _KEY_NAMES
  and _KEY_VALUE): a key of 44 characters is masked; ``id``, an identifier,
  a number and ``["id", "ts"]`` are a schema's and kept.
- is_secret_field asks the classifier for every name it reads; where the
  classifier has no answer the carriers' name rule stands (fail closed).
- GOOGLE_APPLICATION_CREDENTIALS holds a path, kept; an inline JSON
  document there is masked.
- The bare words key, session, tokens and session_id are record fields,
  decided after the rc: unchanged.
"""

from __future__ import annotations

import collections
import itertools
import json
import random

import pytest

from benchflow._utils.config_redaction import _should_record_env_entry
from benchflow._utils.credential_names import entry_kind, name_words
from benchflow.publish.redact import redact_value
from benchflow.trajectories import types
from benchflow.trajectories.types import (
    is_secret_field,
    is_secret_number,
    redact_trajectory_obj,
    redact_trajectory_text,
)
from tests._timing import assert_linear, linear_seconds
from tests.test_redaction_ansi_spellings import _repeat
from tests.test_redaction_escape_boundary import LAYERS, _unchanged
from tests.test_redaction_escape_boundary import _leaks as _boundary_leaks
from tests.test_redaction_shape_matrix import _leaks as _path_leaks

MASK = "***REDACTED***"
EXPORT_MASK = "<XXX-benchflow-key-values-XXX>"
OPAQUE = "FAKEdecided0aZ9bY7"
RUN = "FAKE-decided0aZ9!"
KEY44 = "FAKEaZ09fake/AZ+09fakeAZ09fakeAZ09fakeAZ09="
PEM = (
    "--"
    + "---BEGIN"
    + " PRIVATE"
    + " KEY----"
    + "-\nFAKEfakeFAKEfake0123\n-----END PRIVATE KEY-----\n"
)


def _leaked(text: str, secret: str) -> set[str]:
    return {*_boundary_leaks(text, secret), *_path_leaks(text, secret)}


def _forms(name: str, value: str) -> dict[str, str]:
    return {
        "env": f"{name}={value}",
        "export": f"export {name}={value}",
        "yaml": f"{name}: {value}",
        "after an escaped newline": f"out\\n{name}={value}",
        "after a colour code": f"out\x1b[32m{name}={value}",
        "entry": f'"{name}": "{value}"',
        "escaped entry": f'\\"{name}\\": \\"{value}\\"',
        "nested entry": json.dumps({"env": {name: value}}),
    }


DECIDED = [
    "KEY_2",
    "KEY_10",
    "TOKEN_12345",
    "SECRET_2_OF_3",
    "PRIMARY_KEY_2",
    "TOKENS_2",
    "TOKEN_2FA",
    "SESSION_2",
    "TOKEN_V2",
    "CIRCLE_OIDC_TOKEN_V2",
    "TOKEN_2_FILE_PATH",
    "KEY_X509C_FILE",
    "id_token_hint",
    "key_hint",
    "ACCESS_TOKEN_2",
]


@pytest.mark.parametrize("name", DECIDED)
def test_a_name_the_classifier_calls_a_credential_is_masked_in_every_layer(name):
    """Part B: in every text form and layer, as a key at export and
    in trajectory files, and as a whole name, in each spelling."""
    assert entry_kind(name, OPAQUE) == "credential"
    lower = name != name.upper()
    for form, text in _forms(name, OPAQUE).items():
        if lower and form == "yaml":
            # A lower-case name's unquoted identifier after `:` reads as code
            # (the lower-case pass names' residual): kept; a value no
            # variable could be is masked.
            assert OPAQUE in redact_trajectory_text(text), text
            assert _leaked(f"{name}: {RUN}", RUN) == set()
            continue
        assert _leaked(text, OPAQUE) == set(), (form, text)
    for spelled in {name, name.upper(), name.lower()}:
        assert is_secret_field(spelled, OPAQUE), spelled
        assert redact_value({spelled: OPAQUE})[0] == {spelled: EXPORT_MASK}
        assert redact_trajectory_obj({spelled: OPAQUE}) == {spelled: MASK}


# Settings by the classifier's own rules, and a usage count with a number.
SETTINGS = [
    "MAX_TOKENS_2=8192",
    "TOKEN_2FA_ENABLED=true",
    "PASSWORD_2_HINT=dog",
    "PASSWORD_HINT=dog",
    "KEY_2_FILE=/run/keys/k2",
    "TOKEN_V2_PATH=/run/t",
    "input_tokens_2=1234",
]


@pytest.mark.parametrize("text", SETTINGS)
def test_a_setting_of_the_same_shape_is_kept(text):
    """Part B: the classifier's settings read by the same shapes are
    kept, in every layer, at export and as a whole name."""
    name, value = text.split("=", 1)
    assert entry_kind(name, value) == "setting"
    for form, rendered in _forms(name, value).items():
        assert _unchanged(rendered) == dict.fromkeys(LAYERS, True), (form, rendered)
    assert not is_secret_field(name, value)
    assert redact_value({name: value}) == ({name: value}, 0)


@pytest.mark.parametrize("name", ["key", "session", "tokens", "session_id"])
def test_the_four_record_fields_are_as_they_were(name):
    """the bare record fields are decided after the rc; no rule here
    reads them, in text, in trajectory files or at export."""
    for form, text in _forms(name, OPAQUE).items():
        assert redact_trajectory_text(text) == text, (form, text)
    assert not is_secret_field(name, OPAQUE)
    assert redact_trajectory_obj({name: OPAQUE}) == {name: OPAQUE}
    assert redact_value({name: OPAQUE}) == ({name: OPAQUE}, 0)


# Usage and cost names: BenchFlow's own records (TokenUsage, the provider
# usage keys it reads, result.json and the cost records), LiteLLM's cache
# writes by TTL, and a usage name with a number after it.
USAGE = sorted(
    {
        *(field.name for field in types.TokenUsage.__dataclass_fields__.values()),
        *types._USAGE_KEYS,
        *types._USAGE_DETAIL_KEYS,
        *types._USAGE_METADATA_KEYS,
        "n_input_tokens",
        "n_output_tokens",
        "n_cache_read_tokens",
        "n_cache_creation_tokens",
        "n_cache_tokens",
        "total_input_tokens",
        "total_prompt_tokens",
        "total_cache_read_tokens",
        "total_cached_tokens",
        "total_cache_creation_tokens",
        "cached_read_tokens",
        "cached_write_tokens",
        "cache_write_tokens",
        "cache_write_input_tokens",
        "cost_usd",
        "total_cost_usd",
        "ephemeral_5m_input_tokens",
        "ephemeral_1h_input_tokens",
        "cache_write_5m",
        "cache_write_1h",
        "write_5m",
        "write_1h",
        "tokens_1h",
        "tokens_5m",
        "input_tokens_2",
        "output_tokens_v2",
        "prompt_tokens_12345",
        "completion_tokens_1",
        "total_tokens_2",
        "cache_read_tokens_1h",
        "cache_write_tokens_5m",
        "cache_creation_input_tokens_1h",
    }
)


@pytest.mark.parametrize("name", USAGE)
def test_a_usage_or_cost_name_keeps_a_real_count(name):
    """Part B, the lead's condition: a usage or count name is a
    setting whatever number follows. 1234 and 5, as numbers and as text,
    are kept at export, in trajectory files and in config.json."""
    for value in (1234, 5, "1234", "5"):
        assert redact_value({name: value}) == ({name: value}, 0), value
        assert redact_trajectory_obj({name: value}) == {name: value}, value
        assert _should_record_env_entry(name, str(value)), value
        if isinstance(value, int):
            assert not is_secret_number(name, value)
        else:
            assert not is_secret_field(name, value)
    assert redact_trajectory_text(f"{name}=1234") == f"{name}=1234"


# A primary or a secondary key's name, and its value.
KEY_NAMES = ["primary_key", "primaryKey", "secondary_key", "COSMOS_PRIMARY_KEY"]


@pytest.mark.parametrize("name", KEY_NAMES)
def test_a_primary_keys_value_decides(name):
    """Part B, the classifier's shapes: an access key of 44
    characters is masked in text, trajectory files and at export; a schema's
    key (``id``, an identifier, a number, a list of them) is kept."""
    assert entry_kind(name, KEY44) == "credential"
    for form, text in _forms(name, KEY44).items():
        assert _leaked(text, KEY44) == set(), (form, text)
    assert redact_value({name: KEY44})[0] == {name: EXPORT_MASK}
    assert redact_trajectory_obj({name: KEY44}) == {name: MASK}
    for value in ("id", "external_subscription_id", ["id", "ts"], 123456, True):
        assert entry_kind(name, value) == "setting", value
        assert redact_value({name: value}) == ({name: value}, 0), value
        assert redact_trajectory_obj({name: value}) == {name: value}, value
    for text in (
        f"{name}=id_column",
        f'"{name}": "id"',
        f'"{name}": ["id", "ts"]',
        f"{name}=123456789012345678901234567",
    ):
        assert _unchanged(text) == dict.fromkeys(LAYERS, True), text


SERVICE_ACCOUNT = {
    "type": "service_account",
    "project_id": "fake-project",
    "private_key_id": "fakekeyid0123456789",
    "private_key": PEM,
    "client_email": "fake-sa@fake-project.iam.gserviceaccount.com",
    "client_id": "000000000000000000001",
}
# The parts of the made-up document that must not survive.
ACCOUNT_PARTS = [
    "fakekeyid0123456789",
    "FAKEfakeFAKEfake0123",
    "fake-sa@",
    "000000000000000000001",
]
GAC = "GOOGLE_APPLICATION_CREDENTIALS"
PATH = "/secrets/fake-sa.json"


def _gac_forms(value: str) -> dict[str, str]:
    """The name and a value in every text form: as written, after an
    escape, as a JSON string (the value escaped as JSON escapes it), in a
    serialized record and nested."""
    return {
        "env": f"{GAC}={value}",
        "export": f"export {GAC}={value}",
        "yaml": f"{GAC}: {value}",
        "after an escaped newline": f"out\\n{GAC}={value}",
        "after a colour code": f"out\x1b[32m{GAC}={value}",
        "after a JSON copy's colour code": f"out\\u001b[32m{GAC}={value}",
        "entry": f'"{GAC}": {json.dumps(value)}',
        "nested entry": json.dumps({"env": {GAC: value}}),
        "serialized log line": json.dumps({"message": f"{GAC}={value}", "n": 1}),
    }


def test_google_application_credentials_keeps_a_path_in_every_layer():
    """Part B, the lead's ruling: a path is the classifier's setting,
    kept in text, the escaped and colour forms, a JSON string, trajectory
    files, export and config.json."""
    for name in (GAC, GAC.lower()):
        assert entry_kind(name, PATH) == "setting"
        assert not is_secret_field(name, PATH)
        assert redact_value({name: PATH}) == ({name: PATH}, 0)
        assert redact_trajectory_obj({name: PATH}) == {name: PATH}
        assert _should_record_env_entry(name, PATH)
    for form, text in _gac_forms(PATH).items():
        assert _unchanged(text) == dict.fromkeys(LAYERS, True), (form, text)


@pytest.mark.parametrize(
    "document",
    [
        json.dumps(SERVICE_ACCOUNT),
        json.dumps(SERVICE_ACCOUNT, indent=2).replace("\n", "\\n"),
        json.dumps(SERVICE_ACCOUNT)[:-1],
    ],
    ids=["one line", "multi-line, escaped newlines", "not closed"],
)
def test_google_application_credentials_masks_inline_json_whole(document):
    """Part B, the lead's ruling: inline JSON is masked whole, from
    its `{` to the end of the object or of the line, in text, the escaped
    and colour forms, a JSON string, trajectory files, export and
    config.json; nothing of the document survives."""
    for form, text in _gac_forms(document).items():
        for part in ACCOUNT_PARTS:
            assert _leaked(text, part) == set(), (form, part)
    if document.endswith("}"):
        assert (
            redact_trajectory_text(f"{GAC}={document} NEXT=1") == f"{GAC}={MASK} NEXT=1"
        )
    for record in (
        json.dumps({"message": f"{GAC}={document}", "n": 1}),
        json.dumps({GAC: document}),
    ):
        json.loads(redact_trajectory_text(record))
    for name in (GAC, GAC.lower()):
        assert is_secret_field(name, document)
        assert redact_value({name: document})[0] == {name: EXPORT_MASK}
        assert redact_trajectory_obj({name: document}) == {name: MASK}
        assert not _should_record_env_entry(name, document)


def test_google_application_credentials_masks_a_mapping_or_a_list():
    """Part B: the document as a mapping or a list under the name is
    masked whole in trajectory files and at export."""
    for value in (SERVICE_ACCOUNT, [SERVICE_ACCOUNT]):
        assert redact_trajectory_obj({GAC: value}) == {GAC: MASK}
        assert redact_value({GAC: value})[0] == {GAC: EXPORT_MASK}
        nested = json.dumps(redact_trajectory_obj({"env": {GAC: value}}))
        assert not any(part in nested for part in ACCOUNT_PARTS)


# (a) A kept match beside a masked one on one line, in both orders.
BESIDE = [
    ("KEY_2_FILE=/run/keys/k2", f"KEY_2={OPAQUE}"),
    ("MAX_TOKENS_2=8192", f"TOKEN_V2={OPAQUE}"),
    ("primary_key=id_column", f"id_token_hint={RUN}"),
    (f"{GAC}={PATH}", f"KEY_10={OPAQUE}"),
]


@pytest.mark.parametrize(("kept", "masked"), BESIDE)
def test_a_kept_match_never_hides_the_next_one(kept, masked):
    """Part B, the lead's condition (a): a match the classifier keeps
    ends where its value does, so the next name on the line is read, in
    text, the escaped and colour forms, a JSON string, trajectory files and
    export, in both orders. (A lower-case name's identifier followed by
    more text reads as code, as after a lower-case pass name: its row has a
    value no variable could be.)"""
    name, value = masked.split("=", 1)
    kept_name, kept_value = kept.split("=", 1)
    for line in (f"{kept} {masked}", f"{masked} {kept}", f"{kept}, {masked}"):
        for text in (
            line,
            f"out\\n{line}",
            f"out\x1b[32m{line}",
            json.dumps({"m": line}),
        ):
            assert _leaked(text, value) == set(), text
            assert kept in redact_trajectory_text(text) or kept in json.dumps(
                redact_trajectory_text(text)
            ), text
    pair = {kept_name: kept_value, name: value}
    assert redact_trajectory_obj(pair) == {kept_name: kept_value, name: MASK}
    assert redact_value(pair)[0] == {kept_name: kept_value, name: EXPORT_MASK}


def test_the_classifier_failing_masks_the_value(monkeypatch):
    """Part B, the lead's condition (b): where the classifier raises
    (or has no answer) for a name the carrier asks it about, the value is
    masked, and a second pass counts nothing, so the export redactor's
    run-to-zero loop settles."""
    from benchflow.publish.redact import redact_value_to_stability

    def broken(name, value):
        raise RuntimeError("made-up classifier failure")

    for answer in (broken, lambda name, value: None):
        monkeypatch.setattr(types, "entry_kind", answer)
        for text in ("MAX_TOKENS_2=8192", "KEY_2_FILE=/run/keys/k2", f"KEY_2={OPAQUE}"):
            once, count = types.redact_trajectory_text_with_count(text)
            assert once == text.split("=", 1)[0] + "=" + MASK and count == 1
            assert types.redact_trajectory_text_with_count(once) == (once, 0)
        settled = redact_value_to_stability({"message": "MAX_TOKENS_2=8192"})
        assert "8192" not in json.dumps(settled[0])


def test_is_secret_field_asks_the_classifier(monkeypatch):
    """Part B, the lead's ruling: no third copy decides a name the
    redactor reads. Whatever entry_kind answers, is_secret_field follows;
    where it answers None, only a carriers' name stays a secret's (fail
    closed), and a shape it decides (KEY_2) is none."""
    names = ["GITHUB_TOKEN", "KEY_2", "id_token_hint", "MAX_TOKENS_2", "primary_key"]
    for kind, expected in (("credential", True), ("setting", False)):
        monkeypatch.setattr(types, "entry_kind", lambda name, value, k=kind: k)
        assert [is_secret_field(name, OPAQUE) for name in names] == [expected] * 5
    monkeypatch.setattr(types, "entry_kind", lambda name, value: None)
    assert [is_secret_field(name, OPAQUE) for name in names] == [
        True,
        False,
        False,
        False,
        False,
    ]


# The classifier's gaps: a run-together name it does not split. The
# carriers' name rule stands until it has an answer (after the rc).
GAPS = ["ACCOUNTKEY", "Clientcredentials", "defaultazurecredential", "gitcredentials"]


@pytest.mark.parametrize("name", GAPS)
def test_a_name_the_classifier_has_no_answer_for_stays_masked(name):
    assert entry_kind(name, OPAQUE) is None
    assert is_secret_field(name, OPAQUE)
    assert redact_value({name: OPAQUE})[0] == {name: EXPORT_MASK}


def test_the_numbered_shape_reads_a_function_key_as_the_classifier_does():
    """Part B: KEY_F1 .. KEY_F63 (curses' function keys) are
    numbered names to the classifier, masked as whole names at export and in
    config.json: an over-mask the lead accepted (2026-11-14)."""
    for number in (1, 12, 63):
        name = f"KEY_F{number}"
        assert entry_kind(name, "265") == "credential"
        assert is_secret_field(name, "265")


def test_part_b_masks_once():
    for text in (
        f"KEY_2={OPAQUE}",
        f'"TOKEN_V2": "{OPAQUE}"',
        f"id_token_hint={OPAQUE}",
        f"primary_key={KEY44}",
        f'"primaryKey": "{KEY44}"',
    ):
        once = redact_trajectory_text(text)
        assert once != text and redact_trajectory_text(once) == once, text


def test_a_kept_setting_counts_no_replacement():
    """Part B: the carrier's matches the classifier keeps are no
    replacements, so the export redactor, which runs until a pass counts
    none, stops at once."""
    text = "MAX_TOKENS_2=8192 TOKEN_2FA_ENABLED=true primary_key=id_column"
    assert types.redact_trajectory_text_with_count(text) == (text, 0)
    assert types.redact_trajectory_text_with_count(f"KEY_2={OPAQUE}")[1] == 1


@pytest.mark.parametrize(
    "unit",
    [
        f"KEY_2={OPAQUE} ",
        "KEY_2=",
        "TOKEN_12345",
        "key_hint=",
        f"primary_key={KEY44}\n",
        "primary_key=",
        "MAX_TOKENS_2=8192 ",
        "tokens_1h=1234 ",
        "SESSION_2",
        "a_key_2_b_token_v2_",
        f"{GAC}={{",
        f'{GAC}={{"a": "{OPAQUE}", ',
        f"{GAC}={PATH} ",
    ],
)
def test_part_b_repeated_stays_linear(unit, record_property):
    """Part B: each repeated to 50,000 and 200,000 characters: four
    times the text takes about four times as long, and a masked value is
    gone."""
    seconds, outs = linear_seconds(redact_trajectory_text, lambda n: _repeat(unit, n))
    record_property("seconds", seconds)
    assert all(OPAQUE not in out and KEY44 not in out for out in outs)
    assert_linear(seconds)


def test_the_linear_check_catches_a_quadratic_redactor():
    """The check above fails where matching is quadratic: a stand-in that
    scans the rest of the text from every 32nd character takes about
    sixteen times as long on four times the text."""

    def quadratic(text: str) -> str:
        for start in range(0, len(text), 32):
            text.find("\x00\x01", start)
        return text

    seconds, _ = linear_seconds(quadratic, lambda n: _repeat("a_key_2_b_token_v2_", n))
    with pytest.raises(AssertionError):
        assert_linear(seconds)


def test_the_shapes_are_built_from_the_classifiers_lists():
    """Part B: the numbered words are the carrier's, the hint is the
    classifier's _PASSWORD_QUALIFIERS, the key names its _KEY_NAMES."""
    from benchflow._utils import credential_names as c

    assert set(types._NUMBERED_WORDS) <= set(types._SECRET_NAME_WORDS)
    for qualifier in c._PASSWORD_QUALIFIERS:
        assert types._decided_by_name(["key", qualifier])
    for words in c._KEY_NAMES:
        assert types._decided_by_name(list(words))
        assert "[_-]?".join(words) in types._DECIDED_NAME
    assert not any(
        types._decided_by_name(name_words(n)) for n in ("tokens_1h", "tokens_5m")
    )


# Review R192, L1 (the lead's ruling, fail closed): under
# GOOGLE_APPLICATION_CREDENTIALS only a path is kept; anything else is
# masked in every layer and withheld from config.json. A Windows path is read
# through its backslashes and judged whole.
PATHS = [
    "/secrets/fake-sa.json",
    r"C:\Users\me\sa.json",
    r"creds\sa.json",
    r"%USERPROFILE%\adc.json",
    "~/keys/sa.json",
    "./sa.json",
]
NOT_PATHS = [
    ("FAKEopaque0123456789abcdef", "FAKEopaque0123456789abcdef"),
    ("Bearer FAKEbearer0123456789", "FAKEbearer0123456789"),
    ("sid=FAKEsid0123456; theme=dark", "FAKEsid0123456"),
    (KEY44, KEY44),
    (r"FAKE\opaque\0123secretvalue", "0123secretvalue"),
]


def _value_forms(value: str) -> dict[str, str]:
    return {
        "env": f"{GAC}={value}",
        "quoted entry": f'"{GAC}": "{value}"',
        "JSON string": json.dumps({GAC: value}),
        "record": json.dumps({"message": f"{GAC}={value}"}),
        "after a colour code": f"out\x1b[32m{GAC}={value}",
    }


@pytest.mark.parametrize("value", PATHS)
def test_a_path_under_the_name_is_kept_in_every_layer(value):
    """R192, L1: a path, a Windows one too, is kept in text, a JSON string,
    a record, trajectory files, export and config.json."""
    for name in (GAC, "googleApplicationCredentials"):
        assert entry_kind(name, value) == "setting"
        assert redact_trajectory_obj({name: value}) == {name: value}
        assert redact_value({name: value}) == ({name: value}, 0)
        assert _should_record_env_entry(name, value)
    for form, text in _value_forms(value).items():
        assert redact_trajectory_text(text) == text, form


@pytest.mark.parametrize(("value", "secret"), NOT_PATHS)
def test_anything_but_a_path_under_the_name_is_masked(value, secret):
    """R192, L1 (fail closed): an opaque run, a scheme and token, a cookie
    pair, a key and an opaque value with backslashes are gone in every
    layer and withheld from config.json."""
    for name in (GAC, "googleApplicationCredentials"):
        assert entry_kind(name, value) == "credential"
        assert redact_trajectory_obj({name: value}) == {name: MASK}
        assert redact_value({name: value})[0] == {name: EXPORT_MASK}
        assert not _should_record_env_entry(name, value)
    for form, text in _value_forms(value).items():
        assert _leaked(text, secret) == set(), form


# Review R192, L3: part B ran the whole secret-name carrier on any text that
# held the word session, and asked the classifier once per match. The word
# now counts only as a decided name holds it (_WORD_TAILS), and one run of
# the carrier asks the classifier once per name and value (a run's own
# answers, _classifier_answers).
SESSION_PROSE = "The session was resumed after the user reopened the session window.\n"
SESSION_NAMES = [
    "SESSION_2",
    "SESSIONS_2",
    "SESSION_V2",
    "SESSION-2",
    "session_2",
    "session2",
    "sessionV2",
    "sessionId2",
    "session_hint",
    "SESSION_HINT",
]
DENSE_ROW = (
    "KEY_2=FAKEvalue0123 TOKEN_V2=FAKEvalue0123 MAX_TOKENS_2=8192 "
    "tokens_1h=5 SESSION_2=FAKEvalue0123\n"
)


def _secret_name_carrier():
    (carrier,) = [
        pattern
        for pattern, replacement, _category in types._REDACTION_PATTERNS
        if replacement is types._secret_name_value
    ]
    return carrier


def test_the_word_session_alone_does_not_run_the_carrier():
    """R192, L3: prose with the word, `session_id`, `session-cookie` and
    `sessions` hold no decided name's tail, and no other word of the
    carrier: it is not run."""
    text = SESSION_PROSE * 3 + "session_id session-cookie the sessions\n"
    assert not _secret_name_carrier().may_match(text, types._folded(text))
    assert redact_trajectory_text(text) == text


def _session_forms(name: str) -> list[str]:
    value = "FAKEsession0123456"
    return [
        f"{SESSION_PROSE}{name}={value}\n",
        f"export {name}={value}",
        f'"{name}": "{value}"',
        f"{name}: {value}",
        json.dumps({"message": f"{name}={value}"}),
        f"out\x1b[32m{name}={value}",
    ]


@pytest.mark.parametrize("name", SESSION_NAMES)
def test_a_decided_session_name_reads_as_before(name, monkeypatch):
    """The tail changes no output: each form redacts as it does with the
    plain word, and an upper-case numbered name's value is masked."""
    forms = _session_forms(name)
    with_tails = [redact_trajectory_text(text) for text in forms]
    for text in forms:
        assert _secret_name_carrier().may_match(text, types._folded(text)), text
    monkeypatch.setattr(types, "_WORD_TAILS", {})
    assert [redact_trajectory_text(text) for text in forms] == with_tails
    if name in ("SESSION_2", "SESSIONS_2", "SESSION_V2"):
        assert not any("FAKEsession0123456" in out for out in with_tails), name


def test_the_session_tail_changes_no_output_on_mixed_text(monkeypatch):
    """2,000 texts built from the word, separators, numbers, short words,
    the hint, the carrier's other words and values (a fixed seed): the same
    output with the tail and with the plain word."""
    rng = random.Random(528)
    parts = [
        "session",
        "SESSION",
        "Session",
        "sessions",
        "_",
        "-",
        "2",
        "10",
        "v",
        "Id",
        "hint",
        "=",
        ": ",
        '"',
        '\\"',
        " ",
        "\n",
        "FAKEval0123",
        "token",
        "KEY",
        "id",
        "cookie",
        "{",
        "}",
        "\x1b[32m",
        "\\n",
        "%3D",
    ]
    texts = [
        "".join(rng.choice(parts) for _ in range(rng.randint(3, 24)))
        for _ in range(2000)
    ]
    with_tails = [redact_trajectory_text(text) for text in texts]
    monkeypatch.setattr(types, "_WORD_TAILS", {})
    plain = [redact_trajectory_text(text) for text in texts]
    assert [t for t, a, b in zip(texts, with_tails, plain, strict=True) if a != b] == []


def test_one_run_asks_the_classifier_once_per_name_and_value(monkeypatch):
    """R192, L3: 1,000 dense rows ask the classifier once per decided name
    and value, and redact as each row alone does."""
    expected = redact_trajectory_text(DENSE_ROW) * 1000
    asked: collections.Counter[tuple[str, str]] = collections.Counter()
    real = types.entry_kind

    def counting(name: str, value: str):
        asked[name, value] += 1
        return real(name, value)

    monkeypatch.setattr(types, "entry_kind", counting)
    assert redact_trajectory_text(DENSE_ROW * 1000) == expected
    assert asked and set(asked.values()) == {1}, asked
    assert types._classifier_answers.get() is None


def test_the_answers_are_per_value_within_one_run(monkeypatch):
    """The same name with another value in the same run is asked again: a
    path and an opaque run under GOOGLE_APPLICATION_CREDENTIALS, a schema's
    and an access key under primary_key; a value past _ANSWER_VALUE_MAX is
    asked each time."""
    text = (
        f"{GAC}=/run/secrets/sa.json\n{GAC}=FAKEopaque0123456789\n"
        f"{GAC}=/run/secrets/sa.json\n"
        f"primary_key=external_subscription_id\nprimary_key={KEY44}\n"
    )
    assert redact_trajectory_text(text).splitlines() == [
        f"{GAC}=/run/secrets/sa.json",
        f"{GAC}={MASK}",
        f"{GAC}=/run/secrets/sa.json",
        "primary_key=external_subscription_id",
        f"primary_key={MASK}",
    ]
    asked: collections.Counter[str] = collections.Counter()
    real = types.entry_kind

    def counting(name: str, value: str):
        asked[value] += 1
        return real(name, value)

    monkeypatch.setattr(types, "entry_kind", counting)
    long_value = "FAKE" + "x" * types._ANSWER_VALUE_MAX
    out = redact_trajectory_text(f"KEY_2={long_value}\nKEY_2={long_value}\n")
    assert long_value not in out
    assert asked[long_value] == 2


# Review R192B, M-B: the decided branch kept the whole match, value and all,
# where its pattern read a name the shape check (_decided_by_name) rejected:
# a numbered word glued to letters and a digit (KEYV2), or a plural one
# followed by more words (TOKENS_2_SECRET). The classifier calls each a
# credential's; the value was left in env text, JSON text, the object layer
# and ACP. Now the classifier decides every name the branch reads.
PATTERN_ONLY = [
    "KEYV2_TOKEN",
    "TOKENS_2_SECRET",
    "APP_KEYS_2_PASSWORD",
    "mcpcanvasadmintoken1",
    "getpassword2",
]
GENERATED_WORDS = [
    word + plural
    for word in (
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "PASSWD",
        "KEY",
        "CREDENTIAL",
        "PASSPHRASE",
        "SESSION",
    )
    for plural in ("", "S")
]


def _generated(word: str) -> list[str]:
    """R192B's generated names for one numbered word: a number or a short
    word and a number, a suffix, a prefix, both cases."""
    mids = [
        "2",
        "_2",
        "-2",
        "V2",
        "_V2",
        "-V2",
        "AB12",
        "_AB12",
        "X1",
        "_X1",
        "12345",
        "_12345",
    ]
    suffixes = [
        "", "_TOKEN", "_SECRET", "_KEY", "_PASSWORD", "_API_KEY", "_FILE", "_HINT",
        "_ID", "_URL", "_NAME", "_ENABLED", "_COUNT", "_PATH", "_B64", "_VALUE",
    ]  # fmt: skip
    names = []
    for prefix, mid, suffix in itertools.product(("", "MY_", "APP_"), mids, suffixes):
        name = prefix + word + mid + suffix
        names += [name, name.lower()]
    return names


def _carried(name: str, value: str) -> list[str]:
    """The entry through env text, JSON text, the object layer and ACP."""
    return [
        redact_trajectory_text(f"{name}={value}"),
        redact_trajectory_text(json.dumps({name: value})),
        json.dumps(redact_trajectory_obj({name: value})),
        types.redact_acp_trajectory_jsonl([{"type": "x", "env": {name: value}}]),
    ]


@pytest.mark.parametrize("name", PATTERN_ONLY)
def test_a_name_only_the_decided_pattern_reads_is_masked_in_every_layer(name):
    """R192B's five: the classifier calls each a credential's, and the value
    is gone from text (env, export, JSON, colon), the object layer, ACP, a
    record's text, export and hosted, and withheld from config.json."""
    from benchflow.hosted_env import _masked_env_args

    assert entry_kind(name, OPAQUE) == "credential"
    texts = [
        f"{name}={OPAQUE}",
        f'export {name}="{OPAQUE}"',
        json.dumps({name: OPAQUE}),
        f"{name}: {OPAQUE}",
        json.dumps({"message": f"{name}={OPAQUE}"}),
    ]
    for text in texts:
        assert OPAQUE not in redact_trajectory_text(text), text
    for out in _carried(name, OPAQUE):
        assert OPAQUE not in out
    assert redact_value({name: OPAQUE})[0] == {name: EXPORT_MASK}
    assert OPAQUE not in json.dumps(_masked_env_args({name: OPAQUE}))
    assert not _should_record_env_entry(name, OPAQUE)
    assert is_secret_field(name, OPAQUE)


@pytest.mark.parametrize("word", GENERATED_WORDS)
def test_every_generated_decided_name_the_classifier_calls_a_credential_is_masked(word):
    """R192B's generated set (2,160 of its 18,432 names were left visible):
    wherever entry_kind calls NAME=<value> a credential's, the value is gone
    from env text, JSON text, the object layer and ACP."""
    leaked = []
    for name in _generated(word):
        kind = entry_kind(name, OPAQUE)
        outs = _carried(name, OPAQUE)
        if kind != "setting" and any(OPAQUE in out for out in outs):
            leaked.append(name)
    assert leaked == []
