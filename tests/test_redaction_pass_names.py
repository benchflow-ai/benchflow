"""*_PASS names that are a test's result, and the passwords beside them.

Review of PR #173, 2026-11-13: the redactor masked SWE-bench's FAIL_TO_PASS
and PASS_TO_PASS. Every value here is made up.

The *_PASS carrier (since d50675db9) cut the leading `[` off a FAIL_TO_PASS
list in text and in a JSON-string value, and the export redactor (since
801a4d3e6, through is_secret_field) masked both lists whole: the lists of
the tests a fix must make pass, a core field of results and exports. The
lead's ruling for `pass`: a last-word PASS after a test or outcome word
(fail_to, pass_to, blockers, deterministic, first, second, final, is,
expected, did, test, tests, check, checks, gate, smoke) is no secret, and
every other last-word PASS is a password's.

FAIL_TO_PASS and PASS_TO_PASS are kept whatever they hold. A name with any
other of those words keeps what a test's result is (a count, a switch word
or a list) and masks any other value (``FIRST_PASS=hunter2FAKE``), as the
redactor's test-count names always have.
"""

from __future__ import annotations

import json

import pytest

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
from tests.test_redaction_shape_matrix import LAYERS as PATH_LAYERS
from tests.test_redaction_shape_matrix import PREFIXES, _dict_key_leaks
from tests.test_redaction_shape_matrix import _leaks as _path_leaks

MASK = "***REDACTED***"
TEST_IDS = ["tests/test_a.py::test_x", "tests/test_a.py::test_y[param-1]"]
MARKER = "tests/test_a.py::test_x"
TEST_LISTS = ["FAIL_TO_PASS", "PASS_TO_PASS"]


def _kept_everywhere(text: str, marker: str) -> bool:
    """*text* comes through text, a dict value, both ACP writers and the
    export redactor unchanged, and *marker* survives result.json's
    ``_masked`` and a serialized line too."""
    return _unchanged(text) == dict.fromkeys(LAYERS, True) and _path_leaks(
        text, marker
    ) == list(PATH_LAYERS)


def _list_texts(name: str) -> list[tuple[str, str]]:
    """(text, a part of it every layer writes as it is)."""
    ids = json.dumps(TEST_IDS)
    forms = [
        f"{name}={ids}",
        f"{name}: {ids}",
        f'"{name}": {ids}',
        f'"{name}": {json.dumps(ids)}',
        json.dumps({name: ids})[1:-1].replace('"', '\\"'),
        f"{name}={MARKER}",
        f"out\\n{name}={ids}",
        f"out\x1b[32m{name}={ids}",
        f"q=%22{name}%22: {ids}",
        f"'{name}': {ids}",
    ]
    return [(text, MARKER) for text in forms] + [(f"{name}: 3", f"{name}: 3")]


@pytest.mark.parametrize("name", TEST_LISTS)
def test_a_swe_bench_test_list_is_kept_in_text(name):
    """Review of PR #173, 2026-11-13: a list of test ids, the same list as
    a JSON string, JSON-escaped, a plain id and a count, after an escaped
    newline, a colour code, a URL-encoded quote and a quote."""
    changed = [
        text for text, marker in _list_texts(name) if not _kept_everywhere(text, marker)
    ]
    assert changed == []


@pytest.mark.parametrize("name", [*TEST_LISTS, "fail_to_pass", "pass_to_pass"])
@pytest.mark.parametrize(
    "value",
    [TEST_IDS, json.dumps(TEST_IDS), MARKER, 3, 1500, 482139],
    ids=["list", "JSON string", "one id", "count", "1500", "482139"],
)
def test_a_swe_bench_test_list_is_kept_as_a_key(name, value):
    """Review of PR #173, 2026-11-13: as an object key, a header and an env
    arg of a hosted run, in trajectory files, both ACP writers and at
    export; a number keeps its type."""
    obj = {name: value}
    assert redact_trajectory_obj(obj) == obj
    assert json.dumps(redact_trajectory_obj(obj)) == json.dumps(obj)
    assert redact_value(obj) == (obj, 0)
    assert not is_secret_field(name, value)
    assert not is_secret_number(name, value)
    assert _masked_env_args({name: value}) == {name: value}
    marker = MARKER if not isinstance(value, int) else str(value)
    assert _dict_leaks(obj, marker) == list(LAYERS)
    if isinstance(value, str):
        assert _dict_key_leaks(name, value, marker) == list(PATH_LAYERS)


def _instance() -> dict:
    """A SWE-bench instance as its dataset row has it (made up)."""
    return {
        "repo": "example-org/widgets",
        "instance_id": "example-org__widgets-4127",
        "base_commit": "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c",
        "patch": (
            "diff --git a/widgets/core.py b/widgets/core.py\n"
            "--- a/widgets/core.py\n+++ b/widgets/core.py\n"
            "@@ -10,7 +10,7 @@ def area(w, h):\n-    return w + h\n+    return w * h\n"
        ),
        "test_patch": "diff --git a/tests/test_core.py b/tests/test_core.py\n",
        "problem_statement": "area() adds instead of multiplying.",
        "version": "1.4",
        "FAIL_TO_PASS": json.dumps(["tests/test_core.py::test_area"]),
        "PASS_TO_PASS": json.dumps(
            ["tests/test_core.py::test_perimeter", "tests/test_core.py::test_repr"]
        ),
        "environment_setup_commit": "1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d",
    }


def test_a_swe_bench_instance_record_comes_through_whole():
    """Review of PR #173, 2026-11-13: a whole instance record, as an object,
    as its lists parsed, as serialized text and inside a result, in
    trajectory files, at export and in the hosted run's records."""
    instance = _instance()
    parsed = {
        **instance,
        "FAIL_TO_PASS": json.loads(instance["FAIL_TO_PASS"]),
        "PASS_TO_PASS": json.loads(instance["PASS_TO_PASS"]),
    }
    for record in (instance, parsed, {"task": {"metadata": parsed}}):
        assert redact_trajectory_obj(record) == record
        assert redact_value(record) == (record, 0)
        text = json.dumps(record)
        assert redact_trajectory_text(text) == text
        assert redact_trajectory_text(json.dumps(text)) == json.dumps(text)
    assert _masked_env_args(instance) == instance


@pytest.mark.parametrize("name", TEST_LISTS)
def test_a_swe_bench_test_list_is_kept_after_every_prefix_of_the_shape_matrix(name):
    """Review of PR #173, 2026-11-13: must-keep rows beside the shape
    matrix's must-mask rows, after each of its prefixes, on its six write
    paths."""
    ids = json.dumps(TEST_IDS)
    kept = {
        context: _path_leaks(f"{prefix}{name}={ids} end", f"{name}=[")
        for context, prefix in PREFIXES.items()
    }
    assert kept == {context: list(PATH_LAYERS) for context in PREFIXES}


# A last-word PASS after a test or outcome word.
OUTCOMES = [
    "ALL_BLOCKERS_PASS",
    "BLOCKERS_PASS",
    "DETERMINISTIC_PASS",
    "FIRST_PASS",
    "SECOND_PASS",
    "FINAL_PASS",
    "IS_PASS",
    "EXPECTED_PASS",
    "DID_PASS",
    "TEST_PASS",
    "TESTS_PASS",
    "CHECK_PASS",
    "CHECKS_PASS",
    "GATE_PASS",
    "SMOKE_PASS",
    "CI_SMOKE_PASS",
    "TOTAL_PASS",
    "N_PASS",
]
RESULTS = ["3", "12/15", "true", "False", json.dumps(TEST_IDS), "[]", "[3, 4]"]
NOT_RESULTS = ["hunter2FAKE", "12345!", "482139", "[hunter2FAKE"]
# After a pure count's word a count has any length (below).
COUNT_NAMES = ["TOTAL_PASS", "N_PASS"]


@pytest.mark.parametrize("name", OUTCOMES)
def test_an_outcome_name_keeps_a_test_result(name):
    """Review of PR #173, 2026-11-13: a count, a switch word or a list under
    the name is kept in text (as written, quoted, after an escaped newline
    and a colour code) and as an object key, in every layer."""
    for value in RESULTS:
        for text in (
            f"{name}={value}",
            f"{name}: {value}",
            f'"{name}": "{value}"',
            f"out\\n{name}={value}",
            f"out\x1b[32m{name}={value}",
        ):
            assert _unchanged(text) == dict.fromkeys(LAYERS, True), text
        assert not is_secret_field(name, value)
        assert redact_value({name: value}) == ({name: value}, 0)
        assert _masked_env_args({name: value}) == {name: value}
    for value in (TEST_IDS, 3, True):
        assert redact_trajectory_obj({name: value}) == {name: value}
        assert redact_value({name: value}) == ({name: value}, 0)


@pytest.mark.parametrize("name", OUTCOMES)
def test_an_outcome_name_masks_any_other_value(name):
    """Review of PR #173, 2026-11-13: a value that is no test result is a
    password's under these names too (``FIRST_PASS=hunter2FAKE``), in text
    and at export."""
    for value in NOT_RESULTS:
        if name in COUNT_NAMES and value.isdigit():
            continue
        for text in (f"{name}={value}", f'"{name}": "{value}"'):
            assert value not in redact_trajectory_text(text), text
        assert is_secret_field(name, value)
        assert redact_value({name: value})[0] == {
            name: "<XXX-benchflow-key-values-XXX>"
        }
        assert _masked_env_args({name: value}) == {name: MASK}


# After a pure count's word (n, num, total, count) a count of any length
# is a test's result (the lead's ruling, review of PR #173, 2026-11-13: the
# classifier's _COUNT_RESULT); other outcome names keep five digits.
COUNT_ROWS = [
    ("TOTAL_PASS", "1234567", False),
    ("N_PASS", "123456", False),
    ("COUNT_PASS", "250000", False),
    ("NUM_PASS", "-123456", False),
    ("TOTAL_PASS", "120000/15", False),
    ("TESTS_PASS", "123456", True),
    ("FIRST_PASS", "123456", True),
    ("TOTAL_PASS", "hunter2FAKE", True),
    ("TOTAL_PASS", "120000/150000", True),
]


@pytest.mark.parametrize("name,value,masked", COUNT_ROWS)
def test_a_count_words_name_keeps_a_count_of_any_length(name, value, masked):
    """Review of PR #173, 2026-11-13: in text (as written, after an escaped
    newline and a colour code, as a JSON string), as an object key, at
    export, in the hosted env and as a number, as the classifier says."""
    text = f"{name}={value}"
    for form in (
        text,
        f"out\\n{text}",
        f"out\x1b[32m{text}",
        json.dumps({name: value}),
    ):
        assert _unchanged(form) == dict.fromkeys(LAYERS, not masked), form
    assert entry_kind(name, value) == ("credential" if masked else "setting")
    assert is_secret_field(name, value) is masked
    assert (redact_trajectory_obj({name: value}) != {name: value}) is masked
    assert (redact_value({name: value})[0] != {name: value}) is masked
    assert (_masked_env_args({name: value}) != {name: value}) is masked
    if value.lstrip("-").isdigit():
        assert is_secret_number(name, int(value)) is masked


# A list under an outcome name is a test's result when it closes on its
# line (the lead's ruling, review of PR #173, 2026-11-13).
CLOSED_LISTS = [
    '["a", "b"]',
    "[1, 2]",
    "[-1, 2.5]",
    "[]",
    "[ ]",
    '["tests/a.py::t[p-1]", "tests/a.py::u"]',
    '["hunter2FAKE"]',
]
MASKED_LISTS = [
    '["hunter2FAKE',
    '["hunter2FAKE", "b"',
    '[ "hunter2FAKE",',
    "['hunter2FAKE",
    "['hunter2FAKE']",
    "[1hunter2FAKE",
    "[1hunter2FAKE]",
]


@pytest.mark.parametrize("value", CLOSED_LISTS)
def test_an_outcome_name_keeps_a_list_that_closes_on_its_line(value):
    """Review of PR #173, 2026-11-13: in text (as written, quoted, after an
    escaped newline and a colour code, with more on the line), as a JSON
    list, as a JSON-string value and as an object key, in every layer."""
    name = "TESTS_PASS"
    for text in (
        f"{name}={value}",
        f"{name}={value} and more ] later",
        f"{name}: {value}",
        f'"{name}": {value}',
        json.dumps({name: value}),
        json.dumps({name: value, "x": "]"}),
        f"out\\n{name}={value}",
        f"out\x1b[32m{name}={value}",
    ):
        assert _unchanged(text) == dict.fromkeys(LAYERS, True), text
    assert redact_trajectory_obj({name: value}) == {name: value}
    assert _dict_leaks({name: value}, json.dumps(value)[1:-1]) == list(LAYERS)


@pytest.mark.parametrize("value", MASKED_LISTS)
def test_an_outcome_name_masks_a_list_cut_off_on_its_line(value):
    """Review of PR #173, 2026-11-13: a list that does not close on its line,
    or is no JSON list (``[1hunter2FAKE]``, Python's ``['a']``), is masked
    with its first item, as written, before a newline or an escaped one, and as a
    JSON-string value with a `]` later on the serialized line; as an object
    key too, and at export."""
    name = "TESTS_PASS"
    for text in (
        f"{name}={value}",
        f"{name}={value}\nnext line ]",
        f"{name}={value}\\nnext line ]",
        f'"{name}": {value}',
        json.dumps({name: value}),
        json.dumps({name: value, "x": "]"}),
        f"out\x1b[32m{name}={value}",
    ):
        assert _boundary_leaks(text, "hunter2FAKE") == [], text
    assert _dict_leaks({name: value}, "hunter2FAKE") == []
    assert is_secret_field(name, value)
    assert redact_value({name: value})[0] == {name: "<XXX-benchflow-key-values-XXX>"}


def test_a_list_that_goes_on_to_the_next_line_is_masked_as_a_cut_off_one():
    """Review of PR #173, 2026-11-13: in text its `[` is masked, as at
    63f22b045; the items on the lines after it hold no value of the name. As
    an object key a JSON list over lines is the classifier's: a setting."""
    text = 'TESTS_PASS=[\n  "a",\n  "b"\n]'
    assert redact_trajectory_text(text) == f'TESTS_PASS={MASK}\n  "a",\n  "b"\n]'
    value = text.split("=", 1)[1]
    assert entry_kind("TESTS_PASS", value) == "setting"
    assert not is_secret_field("TESTS_PASS", value)


def test_a_password_written_as_a_list_is_masked_with_its_first_item():
    """Review of PR #173, 2026-11-13: under a password's name the masked
    value takes the list's first item, not the `[` alone."""
    assert redact_trajectory_text('DB_PASS=["hunter2FAKE"]') == f'DB_PASS={MASK}"]'
    assert redact_trajectory_text("DB_PASS=[hunter2FAKE]") == f"DB_PASS={MASK}"


def test_a_200000_character_list_left_open_stays_linear():
    """Review of PR #173, 2026-11-13: one list of 200,000 characters that
    never closes, and the same list closed, under an outcome name: masked
    and kept, the scan bounded, at the shared check against 50,000 (it was
    well under a second, quiet)."""
    for make, start in (
        (_open_list, f"TESTS_PASS={MASK}"),
        (_closed_list, 'TESTS_PASS=["a"]'),
    ):
        seconds, outs = linear_seconds(redact_trajectory_text, make)
        assert all(out.startswith(start) for out in outs), outs[1][:40]
        assert_linear(seconds)


def _open_list(size: int) -> str:
    return "TESTS_PASS=[" + '"tests/a.py::t", ' * (size // 17)


def _closed_list(size: int) -> str:
    return 'TESTS_PASS=["a"]' + '"tests/a.py::t", ' * (size // 17)


# Passwords, still masked.
PASSWORDS = [
    "DB_PASS",
    "SMTP_PASS",
    "ADMIN_PASS",
    "MYSQL_PWD",
    "LATEST_PASS",
    "BYPASS_PASS",
]
SECRET = "FAKEpw0aZ9bY7Lm2"


@pytest.mark.parametrize("name", PASSWORDS)
def test_a_password_name_is_still_masked(name):
    """Review of PR #173, 2026-11-13: a word glued to TEST is none
    (LATEST_PASS); a string and a PIN-like number are masked in text, as a
    key, at export and in the hosted run's records."""
    for text in (f"{name}={SECRET}", f'"{name}": "{SECRET}"', f"out\\n{name}={SECRET}"):
        layers = {*_boundary_leaks(text, SECRET), *_path_leaks(text, SECRET)}
        assert layers == set(), text
    assert MASK in redact_trajectory_text(f"{name}: 123456")
    assert is_secret_field(name, SECRET)
    assert is_secret_number(name, 123456)
    assert _dict_leaks({name: SECRET}, SECRET) == []
    assert _dict_leaks({name: 123456}, "123456") == []
    assert _masked_env_args({name: SECRET}) == {name: MASK}


# A numbered password's name, and PASS_VALUE alone or after `_` (review of
# PR #173, Finding A): the classifier's ruling makes them credentials.
NUMBERED_AND_VALUE = [
    "PASS_2",
    "PASS_10",
    "DB_PASS_2",
    "SMTP_PASS_01",
    "MYSQL_PWD_2",
    "DB_PASS_VALUE",
    "PASS_VALUE",
    "PASSVALUE",
]


@pytest.mark.parametrize("name", NUMBERED_AND_VALUE)
def test_a_numbered_password_and_a_pass_value_are_masked(name):
    """Review of PR #173, 2026-11-13 (Finding A): the *_PASS rule read only a
    name that ends in _PASS or _PWD, so a value under DB_PASS_2 or
    PASS_VALUE was kept in text and at export. In text (as written, quoted,
    JSON-escaped, after an escaped newline and a colour code), as a key, at
    export and in the hosted run's records."""
    for text in (
        f"{name}={SECRET}",
        f"{name}: {SECRET}",
        f'"{name}": "{SECRET}"',
        f'\\"{name}\\": \\"{SECRET}\\"',
        f"out\\n{name}={SECRET}",
        f"out\x1b[32m{name}={SECRET}",
        f"X_{name}={SECRET}",
    ):
        layers = {*_boundary_leaks(text, SECRET), *_path_leaks(text, SECRET)}
        assert layers == set(), text
    assert is_secret_field(name, SECRET)
    assert _dict_leaks({name: SECRET}, SECRET) == []
    assert _dict_key_leaks(name, SECRET, SECRET) == []
    assert redact_value({name: SECRET})[0] == {name: "<XXX-benchflow-key-values-XXX>"}
    assert _masked_env_args({name: SECRET}) == {name: MASK}


# Names beside them that are no password's: a metric, a count, a test's
# result, a word glued before PASS, a number inside the name, a lower-case
# variable in code; and a numbered test-count name, whose value decides.
NOT_PASSWORDS = [
    "PASS_RATE=0.5",
    "PASS_COUNT=12",
    "PASS_AT_K=1",
    "TESTS_PASS=12",
    "TOTAL_PASS=12/15",
    'FAIL_TO_PASS=["tests/test_a.py::test_x"]',
    'PASS_TO_PASS=["tests/test_a.py::test_x"]',
    "PASS_2_OF_3=yes",
    "X_PASS_2_OF_3=yes",
    "BYPASS_2=1",
    "COMPASS_VALUE=north",
    "first_pass_2=True",
    "PASS_1_RATE=0.5",
    "PASS2=yes",
    "COMPASS_2=north",
    "pass_value=3",
]


@pytest.mark.parametrize("text", NOT_PASSWORDS)
def test_names_beside_a_numbered_password_are_kept(text):
    """Review of PR #173, 2026-11-13 (Finding A): in every layer of text,
    after an escaped newline and a colour code too. As an object key the
    name is the classifier's (``entry_kind``): kept where it says setting;
    PASS_2_OF_3, PASS_1_RATE and PASS2 are credentials to it, and the
    agreement test holds them to it."""
    for form in (text, f"out\\n{text}", f"out\x1b[32m{text}"):
        assert _unchanged(form) == dict.fromkeys(LAYERS, True), form
    name, value = text.split("=", 1)
    if entry_kind(name, value) != "credential":
        assert not is_secret_field(name, value)
        assert redact_value({name: value}) == ({name: value}, 0)


@pytest.mark.parametrize("name", ["TESTS_PASS_2", "FIRST_PASS_2"])
def test_a_numbered_test_count_name_is_a_password_whatever_it_holds(name):
    """Review of PR #173, 2026-11-13 (one rule, the classifier's): a number
    after the last PASS is no test's or outcome's word, so the name is a
    password's whatever it holds (``entry_kind`` says credential), in text
    and as a whole name."""
    for value in (SECRET, "12", "true"):
        assert value not in redact_trajectory_text(f"{name}={value}").split("=", 1)[1]
        assert is_secret_field(name, value)


# A passphrase's name, read as a password's is (review of PR #173,
# 2026-11-13: the redactor had no rule for it): PASSPHRASE after anything,
# PASS_PHRASE as the whole name or after `_`, in any case.
PASSPHRASES = [
    "PASSPHRASE",
    "PASS_PHRASE",
    "DB_PASSPHRASE",
    "SSH_KEY_PASSPHRASE",
    "GPG_PASS_PHRASE",
    "passphrase",
    "Passphrase",
    "db_passphrase",
    "keyPassphrase",
    "gpg_pass_phrase",
]


@pytest.mark.parametrize("name", PASSPHRASES)
def test_a_passphrase_is_masked(name):
    """Review of PR #173, 2026-11-13: in text (as written, quoted,
    JSON-escaped, after an escaped newline and a colour code, after a
    prefix), as a key, at export, in the hosted run's records, and a
    PIN-like number under it."""
    for text in (
        f"{name}={SECRET}",
        f"{name}: {SECRET}",
        f'"{name}": "{SECRET}"',
        f'\\"{name}\\": \\"{SECRET}\\"',
        f"out\\n{name}={SECRET}",
        f"out\x1b[32m{name}={SECRET}",
        f"X_{name}={SECRET}",
    ):
        layers = {*_boundary_leaks(text, SECRET), *_path_leaks(text, SECRET)}
        assert layers == set(), text
    assert is_secret_field(name, SECRET)
    assert _dict_leaks({name: SECRET}, SECRET) == []
    assert _dict_key_leaks(name, SECRET, SECRET) == []
    assert redact_value({name: SECRET})[0] == {name: "<XXX-benchflow-key-values-XXX>"}
    assert _masked_env_args({name: SECRET}) == {name: MASK}
    assert is_secret_number(name, 482139)


NOT_PASSPHRASES = [
    "the passphrase is required",
    "Enter passphrase:",
    "PASSPHRASE_FILE=/run/secrets/x",
    "PASSPHRASE_PATH=/etc/gpg/key.asc",
    "passphrase_required: true",
    "PASSPHRASE_LENGTH=24",
    "passphrase_hint: dog",
    "COMPASS_PHRASE=north",
    "BYPASS_PHRASE=x",
    "mypass_phrase=x",
]


@pytest.mark.parametrize("text", NOT_PASSPHRASES)
def test_text_beside_a_passphrase_is_kept(text):
    """Review of PR #173, 2026-11-13: prose, a name that goes on after
    PASSPHRASE (a file, a path, a flag, a length, a hint), and a word glued
    before PASS_PHRASE, in every layer."""
    for form in (text, f"out\\n{text}", f"out\x1b[32m{text}"):
        assert _unchanged(form) == dict.fromkeys(LAYERS, True), form


@pytest.mark.parametrize(
    "name",
    [
        "PASSPHRASE_FILE",
        "PASSPHRASE_PATH",
        "passphrase_required",
        "PASSPHRASE_LENGTH",
        "passphrase_hint",
        "COMPASS_PHRASE",
        "BYPASS_PHRASE",
    ],
)
def test_a_name_beside_a_passphrase_keeps_its_string_as_a_key(name):
    """Review of PR #173, 2026-11-13: as an object key, in trajectory files
    and at export."""
    assert not is_secret_field(name, "north")
    assert redact_trajectory_obj({name: "north"}) == {name: "north"}
    assert redact_value({name: "north"}) == ({name: "north"}, 0)


@pytest.mark.parametrize(
    "template", ["SSH_KEY_{}=False", "Enter {}: please type it", "COM{}=north"]
)
def test_a_passphrase_name_is_read_as_a_password_name_is(template):
    """Review of PR #173, 2026-11-13: a name that ends in PASSPHRASE is
    masked wherever one that ends in PASSWORD is (a switch word under a
    bare name and prose after it included). The one difference is a
    passphrase's flag holding a switch (below)."""
    for word in ("password", "passphrase", "PASSWORD", "PASSPHRASE"):
        text = template.format(word)
        assert redact_trajectory_text(text) != text, text


# A passphrase's flag holds a switch as no secret (the lead's ruling,
# review of PR #173, 2026-11-13): a flag word right before the passphrase,
# with `_` or `-`.
FLAG_SWITCHES = [
    "has_passphrase=False",
    "requires_passphrase: true",
    "HAS_PASSPHRASE=no",
    "use-passphrase=1",
    "ssh_no_passphrase=0",
    "skip_passphrase=None",
    "NO_PASS_PHRASE=off",
]
NOT_FLAG_SWITCHES = [
    "has_passphrase=hunter2FAKE",
    "SSH_KEY_PASSPHRASE=false",
    "PASSPHRASE=no",
    "DB_PASSPHRASE=0",
    "passphrase=12345",
    "has_passphrase=12",
    "has_passphrase=null",
    "has_passphrase=true.",
    "dbhas_passphrase=false",
    "PASS_PHRASE=on",
]


@pytest.mark.parametrize("text", FLAG_SWITCHES)
def test_a_passphrase_flag_keeps_a_switch(text):
    """Review of PR #173, 2026-11-13: in text (as written, after an escaped
    newline and a colour code, as a JSON string), as an object key, at
    export and in the hosted env, as the classifier says."""
    name, value = text.replace(": ", "=", 1).split("=", 1)
    for form in (
        text,
        f"out\\n{text}",
        f"out\\u2019{text}",
        f"q=%22{text}",
        f"out\x1b[32m{text}",
        f"out\\u001b[32m{text}",
        f"out%1B%5B32m{text}",
        f"ssh_{text}",
        json.dumps({name: value}),
    ):
        assert _unchanged(form) == dict.fromkeys(LAYERS, True), form
    assert entry_kind(name, value) == "setting"
    assert not is_secret_field(name, value)
    assert redact_trajectory_obj({name: value}) == {name: value}
    assert redact_value({name: value}) == ({name: value}, 0)
    assert _masked_env_args({name: value}) == {name: value}


@pytest.mark.parametrize("text", NOT_FLAG_SWITCHES)
def test_any_other_value_under_a_passphrase_name_is_masked(text):
    """Review of PR #173, 2026-11-13: a flag's value that is no switch
    (``12``, ``null``, ``true.``), and a switch under a bare passphrase
    name (SSH_KEY_PASSPHRASE=false, PASSPHRASE=no, DB_PASSPHRASE=0), in
    every layer, as the classifier says."""
    name, value = text.split("=", 1)
    assert redact_trajectory_text(text) == f"{name}={MASK}"
    for form in (f"out\\n{text}", f"out\x1b[32m{text}", json.dumps({name: value})):
        assert _unchanged(form) == dict.fromkeys(LAYERS, False), form
    assert entry_kind(name, value) == "credential"
    assert is_secret_field(name, value)
    assert redact_trajectory_obj({name: value}) == {name: MASK}
    assert redact_value({name: value})[0] == {name: "<XXX-benchflow-key-values-XXX>"}
    assert _masked_env_args({name: value}) == {name: MASK}


def test_a_camel_case_passphrase_flag_is_a_flag_as_a_whole_name_only():
    """Review of PR #173, 2026-11-13: in text the flag word sits before the
    passphrase with `_` or `-` (the lead's ruling); a whole name is read as
    the classifier reads it, in snake case (hasPassphrase, has_passphrase)."""
    assert redact_trajectory_text("hasPassphrase=false") == f"hasPassphrase={MASK}"
    assert entry_kind("hasPassphrase", "false") == "setting"
    assert not is_secret_field("hasPassphrase", "false")
    assert redact_value({"hasPassphrase": "false"}) == ({"hasPassphrase": "false"}, 0)


@pytest.mark.parametrize(
    "unit",
    [
        'FAIL_TO_PASS=["t"] ',
        "TOTAL_PASS=12 ",
        "FIRST_PASS=hunter2FAKE ",
        "_PASS",
        "X_PASS_",
        "DB_PASS_2=",
        "PASS_VALUE=",
        "DB_PASS_2=hunter2FAKE ",
        "PASS_VALUE=hunter2FAKE ",
        "PASSPHRASE=",
        "PASS_PHRASE=",
        "DB_PASSPHRASE=hunter2FAKE ",
        "COMPASS_PHRASE=north ",
        'N_PASS=["a", ',
        'N_PASS=[\\"a\\", ',
        "N_PASS=[1, ",
        "N_PASS=['a', ",
        'TESTS_PASS=["hunter2FAKE" ',
        "has_passphrase=false ",
        "has_passphrase=hunter2FAKE ",
        "NO_PASS_PHRASE=",
        "is_passphrase",
    ],
)
def test_pass_names_repeated_stay_linear(unit):
    """Review of PR #173, 2026-11-13: each repeated to 50,000 and 200,000
    characters: four times the text takes about four times as long."""
    seconds, outs = linear_seconds(
        redact_trajectory_text, lambda size: _repeat(unit, size)
    )
    assert all("hunter2FAKE" not in out for out in outs)
    assert_linear(seconds)


# The lead's ruling (2026-11-14): the timing tests above call the
# shared check (tests/_timing.py). The positive control: the check, on one
# of each test's inputs at its sizes, fails once a quadratic term is added.
def test_the_pass_name_timing_check_catches_a_quadratic():
    assert catches_quadratic(
        redact_trajectory_text, lambda size: _repeat("COMPASS_PHRASE=north ", size)
    )


def test_the_open_list_timing_check_catches_a_quadratic():
    assert catches_quadratic(redact_trajectory_text, _open_list)
