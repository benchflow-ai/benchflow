"""*_PASS and *_PWD names in lower and camel case, and the code beside them.

(commit 8fbdd6fc1). Every value here is made up.

The redactor masked the value of a name that ends in _PASS or _PWD only in
upper case (``DB_PASS=...``), where the classifier
(``credential_names.entry_kind``) reads every case alike: ``db_pass`` and
``dbPass`` were credentials to it and kept by the redactor (the agreement
test's LOWER_CASE_PASS_PENDING, 69 names).

As a whole name (``is_secret_field``: the export redactor, the hosted run's
records, a number's key) a pass name in any case is now the classifier's.
In running text a lower-case pass name is mostly a variable in code, so
its value is masked only where it is a value, not code
(``types._LOWER_PASS``): a mapping's entry (``"db_pass": "v"``) as upper
case is; an env line (``db_pass=v``) unless the value is code; any other
``:`` or ``=`` only for a quoted value, a run no identifier holds or a
number of six digits or more. A name right after ``(``, ``,``, ``{``,
``[`` or ``.`` is code's. Measured on the worktree's files and the
packages in its .venv (19,884 files, 7.8 million lines): the rule changes
8 lines, each a configuration entry.
"""

from __future__ import annotations

import json

import pytest

from benchflow._utils.config_redaction import _should_record_env_entry
from benchflow._utils.credential_names import entry_kind
from benchflow.hosted_env import _masked_env_args
from benchflow.publish.redact import redact_value
from benchflow.trajectories import types
from benchflow.trajectories.types import (
    is_secret_field,
    is_secret_number,
    redact_trajectory_obj,
    redact_trajectory_text,
)
from tests._timing import assert_linear, catches_quadratic, linear_seconds
from tests.test_redaction_ansi_spellings import _repeat
from tests.test_redaction_classifier_agreement import SPELLINGS, _disagreements
from tests.test_redaction_escape_boundary import LAYERS, _dict_leaks, _unchanged
from tests.test_redaction_escape_boundary import _leaks as _boundary_leaks
from tests.test_redaction_shape_matrix import _dict_key_leaks
from tests.test_redaction_shape_matrix import _leaks as _path_leaks

MASK = "***REDACTED***"
EXPORT_MASK = "<XXX-benchflow-key-values-XXX>"
# A password as most are: letters and digits, so it reads as an identifier.
SECRET = "FAKEpw0aZ9bY7Lm2"
# A run no identifier holds.
SYMBOLS = "FAKEpw-0123!x"
PIN = "482139"
TEST_IDS = ["tests/test_a.py::test_x", "tests/test_a.py::test_y[param-1]"]

# Snake case, camel case, numbered, PASS_VALUE and PASS_<n> as upper case
# has them, and a prefix.
NAMES = [
    "db_pass",
    "dbPass",
    "admin_pass",
    "adminPass",
    "smtp_pwd",
    "smtpPwd",
    "db_pass_2",
    "dbPass_2",
    "pass_value",
    "passValue",
    "passvalue",
    "db_pass_value",
    "dbPassValue",
    "pass_2",
    "by_pass",
    "X_db_pass",
]
# The configuration shapes, every one masked: an env line (as written,
# exported, quoted, after an escaped newline and a colour code, with a
# comment, in a URL query, URL-encoded, in a Markdown code span), a
# mapping's entry (JSON, a dict's repr, JSON-escaped, nested), YAML with a
# quoted value, and a TOML line.
SHAPES = [
    "{n}={v}",
    "export {n}={v}",
    '{n}="{v}"',
    "{n}='{v}'",
    "out\\n{n}={v}",
    "out\x1b[32m{n}={v}",
    "{n}={v}  # the main database",
    "https://h.example/x?{n}={v}&page=2",
    "{n}%3D{v}",
    "set `{n}={v}` first",
    '"{n}": "{v}"',
    "'{n}': '{v}'",
    '\\"{n}\\": \\"{v}\\"',
    '{{"env": {{"{n}": "{v}"}}}}',
    '{n}: "{v}"',
    "  {n}: '{v}'",
    '- {n}: "{v}"',
    '{n} = "{v}"',
]
# A value that is no identifier and no count reads as one in every shape:
# YAML and INI unquoted, and a PIN.
VALUES = [
    ("{n}: {v}", SYMBOLS),
    ("{n} = {v}", SYMBOLS),
    ("{n}={v}", SYMBOLS),
    ("{n}={v}", PIN),
    ("{n}: {v}", PIN),
]


@pytest.mark.parametrize("name", NAMES)
def test_a_lower_case_configuration_entry_is_masked_in_every_layer(name):
    """each shape in text, a dict value, both ACP writers, the
    export redactor, result.json's ``_masked`` and a serialized line."""
    cells = [(shape.format(n=name, v=SECRET), SECRET) for shape in SHAPES]
    cells += [(shape.format(n=name, v=value), value) for shape, value in VALUES]
    leaked = [
        (text, layers)
        for text, secret in cells
        if (layers := {*_boundary_leaks(text, secret), *_path_leaks(text, secret)})
    ]
    assert leaked == []


@pytest.mark.parametrize("name", NAMES)
def test_a_lower_case_name_masks_its_value_as_a_whole_name(name):
    """as an object key (trajectory files, both ACP writers, the
    export), a header's key, an env arg of a hosted run, and a PIN-like
    number under it, as the classifier says."""
    assert entry_kind(name, SECRET) == "credential"
    assert is_secret_field(name, SECRET)
    assert is_secret_number(name, int(PIN))
    assert _dict_leaks({name: SECRET}, SECRET) == []
    assert _dict_leaks({name: int(PIN)}, PIN) == []
    assert _dict_key_leaks(name, SECRET, SECRET) == []
    assert redact_value({name: SECRET})[0] == {name: EXPORT_MASK}
    assert _masked_env_args({name: SECRET}) == {name: MASK}


@pytest.mark.parametrize("name", NAMES)
def test_config_json_drops_a_lower_case_password_entry(name):
    """snake and camel case are dropped. Camel case was a strict
    xfail until config.json's env filter read a name's words from the shared
    splitter (2dddc83e8: dbPass, smtpPwd, dbPassValue)."""
    assert _should_record_env_entry(name, SECRET) is False


# Code that must come through every layer as it is: the request's lines,
# the shapes the corpus shows (the worktree and the packages in its .venv),
# and code beside them.
CODE = [
    "first_pass = True",
    "second_pass = run_pass(data)",
    "for n_pass in range(3):",
    "if not is_pass:",
    "self.first_pass: bool = False",
    "def check_pass(x):",
    '{"first_pass": first_pass}',
    "two_pass=True",
    'name_pass = os.environ["NAME_PASS"]',
    'name_pass = input("…")',
    # From the corpus.
    "    deterministic_pass=deterministic_pass,",
    "    all_blockers_pass=blockers,",
    "    tests_pass=review.deterministic_pass,",
    "    deterministic_pass=bool(tests_pass),",
    "    tests_pass: bool | None = None",
    "    all_blockers_pass: bool",
    "    deterministic_pass: bool,",
    '    "tests_pass": scoring.get("tests_pass"),',
    '    "deterministic_pass": self.deterministic_pass,',
    '    "tokens_to_pass": tokens_to_pass,',
    '    "all_blockers_pass": {',
    "    tests_to_pass: list[str] = json.loads(args.tests_to_pass_file.read_text())",
    "    statements_pass: list[ast.stmt] = [hook_impl_test]",
    "_assertion_pass: Callable[[int, str, str], None] | None = None",
    "    on_pass: str = Field(",
    "    if fidelity_pass:",
    'did_pass = r.get("pass", False)',
    "did_pass = trigger_rate >= trigger_threshold",
    "owner_pwd = user_pwd",
    "owner_pwd = (",
    'user_pass = f"{username}:{password}"',
    "user_pass = base64.b64decode(basic_credentials.encode()).decode()",
    "hook_call_pass = ast.Expr(",
    "let imagePass = image.index;",
    "const isPass = label === 'no_rad' || label === 'supported';",
    "is_pass = is_exact_match or is_acceptable",
    "gates_pass = deterministic_pass and all_blockers_pass",
    "all_blockers_pass = not failed_blockers",
    "tests_pass = verifier_reward == 1.0 if verifier_reward is not None else None",
    'only_pass = root / "e2e-rd-pass"',
    "bf.score_weighted_review(rubric: 'Rubric', *, deterministic_pass: 'bool')",
    "#: The same shape with a count of any length (n_pass=123456): what a pass",
    # Beside them: keyword arguments, literals, a shell variable, a
    # template, a Markdown code span, a comparison, a block.
    "connect(host=host, db_pass=db_pass)",
    "f(db_pass=db_pass, retries=3)",
    "first_pass=True, second_pass=False",
    "pass_value=3",
    "first_pass_2=True",
    "max_pass = 3",
    "db_pass=$DB_PASS",
    'db_pass="${DB_PASS}"',
    'db_pass: "{{ vault_db_pass }}"',
    "`first_pass=True` and `tests_pass: bool`",
    "x_pass == y",
    "if is_pass:\n    return count",
]


@pytest.mark.parametrize("text", CODE)
def test_code_with_a_lower_case_pass_name_is_kept_in_every_layer(text):
    """as written, as a line of a longer text, and after an escaped
    newline, in all five layers."""
    for form in (text, f"x = 1\n{text}\n", f"out\\n{text}"):
        assert _unchanged(form) == dict.fromkeys(LAYERS, True), form


# Names that count tests (the classifier's words, in snake and camel case).
OUTCOMES = [
    "tests_pass",
    "testsPass",
    "all_tests_pass",
    "allTestsPass",
    "first_pass",
    "firstPass",
    "deterministic_pass",
    "deterministicPass",
    "all_blockers_pass",
    "allBlockersPass",
    "is_pass",
    "isPass",
    "gates_pass",
    "ci_smoke_pass",
    "ciSmokePass",
    "total_pass",
    "totalPass",
    "n_pass",
    "nPass",
    # The classifier's later words (c74825bd1): all, only and two
    # are outcomes', max a pure count's.
    "all_pass",
    "allPass",
    "only_pass",
    "onlyPass",
    "two_pass",
    "twoPass",
    "max_pass",
    "maxPass",
]
RESULTS = ["3", "12/15", "true", "False", json.dumps(TEST_IDS), "[]", "[3, 4]"]
NOT_RESULTS = ["hunter2FAKE", "12345!", "482139"]
COUNT_NAMES = {"total_pass", "totalPass", "n_pass", "nPass", "max_pass", "maxPass"}


@pytest.mark.parametrize("name", OUTCOMES)
def test_a_lower_case_outcome_name_keeps_a_test_result(name):
    """As review of PR #173 ruled for upper case: a count, a switch
    word or a list under the name is kept in text (as written, quoted,
    after an escaped newline and a colour code) and as an object key, in
    every layer."""
    for value in RESULTS:
        for text in (
            f"{name}={value}",
            f"{name}: {value}",
            f'"{name}": "{value}"',
            f"out\\n{name}={value}",
            f"out\x1b[32m{name}={value}",
        ):
            assert _unchanged(text) == dict.fromkeys(LAYERS, True), text
        assert entry_kind(name, value) == "setting"
        assert not is_secret_field(name, value)
        assert redact_value({name: value}) == ({name: value}, 0)
        assert _masked_env_args({name: value}) == {name: value}
    for value in (TEST_IDS, 3, True):
        assert redact_trajectory_obj({name: value}) == {name: value}
        assert redact_value({name: value}) == ({name: value}, 0)


@pytest.mark.parametrize("name", OUTCOMES)
def test_a_lower_case_outcome_name_masks_any_other_value(name):
    """a value that is no test result is a password's under these
    names too, in an env line, an entry and at export; a list cut off in
    an entry too (in an env line a `[` starts code)."""
    for value in NOT_RESULTS:
        if name in COUNT_NAMES and value.isdigit():
            continue
        for text in (f"{name}={value}", f'"{name}": "{value}"'):
            assert value not in redact_trajectory_text(text), text
        assert entry_kind(name, value) == "credential"
        assert is_secret_field(name, value)
        assert redact_value({name: value})[0] == {name: EXPORT_MASK}
        assert _masked_env_args({name: value}) == {name: MASK}
    entry = f'"{name}": "[hunter2FAKE"'
    assert "hunter2FAKE" not in redact_trajectory_text(entry)


# After a pure count's word a count has any length, as in upper case.
COUNT_ROWS = [
    ("total_pass", "1234567", False),
    ("max_pass", "123456", False),
    ("maxPass", "1234567", False),
    ("two_pass", "123456", True),
    ("onlyPass", "123456", True),
    ("nPass", "123456", False),
    ("count_pass", "250000", False),
    ("num_pass", "-123456", False),
    ("totalPass", "120000/15", False),
    ("tests_pass", "123456", True),
    ("firstPass", "123456", True),
    ("total_pass", "hunter2FAKE", True),
    ("total_pass", "120000/150000", True),
]


@pytest.mark.parametrize("name,value,masked", COUNT_ROWS)
def test_a_lower_case_count_name_keeps_a_count_of_any_length(name, value, masked):
    """in text (as written, after an escaped newline and a colour
    code, in a Markdown code span, as a JSON string), as an object key, at
    export, in the hosted env and as a number, as the classifier says."""
    text = f"{name}={value}"
    for form in (
        text,
        f"out\\n{text}",
        f"out\x1b[32m{text}",
        f"`{text}`",
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


@pytest.mark.parametrize(
    "name",
    [
        "fail_to_pass", "failToPass", "pass_to_pass", "passToPass",
        "tokens_to_pass", "tests_to_pass", "testsToPass", "on_pass", "onPass",
        "fidelity_pass", "fidelityPass", "norm_pass", "normPass",
        "ON_PASS", "TESTS_TO_PASS", "FIDELITY_PASS", "NORM_PASS",
    ],
)  # fmt: skip
def test_a_lower_case_test_list_name_is_kept_whatever_it_holds(name):
    """the classifier's pass names by name (_NOT_SECRET_NAMES:
    FAIL_TO_PASS, PASS_TO_PASS, TOKENS_TO_PASS, and since that change c74825bd1
    tests_to_pass, on_pass, fidelity_pass and norm_pass) are settings
    whatever they hold, in snake and camel case and in upper case, in text
    and as a key."""
    for text in (
        f"{name}={SECRET}",
        f'"{name}": "{SECRET}"',
        f"{name}: {json.dumps(TEST_IDS)}",
    ):
        assert _unchanged(text) == dict.fromkeys(LAYERS, True), text
    assert entry_kind(name, SECRET) == "setting"
    assert not is_secret_field(name, SECRET)
    assert redact_value({name: TEST_IDS}) == ({name: TEST_IDS}, 0)


# The agreement test's LOWER_CASE_PASS_PENDING before that change emptied it.
FORMERLY_PENDING = sorted(
    {
        "ByPass", "adminPass", "admin_pass", "allBlockersPass", "allTestsPass",
        "all_blockers_pass", "all_tests_pass", "blockerPass", "blocker_pass",
        "checkPass", "check_pass", "checksPass", "checks_pass", "dbPass",
        "dbPass2", "db_pass", "db_pass_2", "deterministicPass",
        "deterministic_pass", "didPass", "did_pass", "expectedPass",
        "expected_pass", "finalPass", "final_pass", "firstPass",
        "firstPassYield", "first_pass", "first_pass_yield", "gatePass",
        "gate_pass", "gatesPass", "gates_pass", "isPass", "is_pass", "nPass",
        "n_pass", "numPass", "num_pass", "pass", "pass10", "pass2", "passFail",
        "passThrough", "passValue", "pass_10", "pass_2", "pass_fail",
        "pass_through", "pass_value", "passvalue", "pgPass", "pg_pass", "scorePass",
        "score_pass", "secondPass", "second_pass", "smokePass", "smoke_pass",
        "smtpPass", "smtp_pass", "testPass", "test_pass", "testsPass",
        "tests_pass", "totalPass", "total_pass", "userPass", "user_pass",
    }
)  # fmt: skip


# The classifier's later rulings moved some of them (b1c25bbe4):
# yield, fail and through after pass make a metric or an outcome
# (first_pass_yield, pass_fail, pass_through: settings now), and bypass is
# one word in camel case too (ByPass: no pass word, so the agreement test's
# tables no longer pick it).
NOW_SETTINGS = {
    "firstPassYield", "first_pass_yield", "passFail", "pass_fail",
    "passThrough", "pass_through",
}  # fmt: skip
NOW_NO_PASS_WORD = {"ByPass"}


def test_the_formerly_pending_names_agree_with_the_classifier():
    """each of the 69 names gives the classifier's answer as a whole
    name, for a count and an opaque value, as a string and as a number,
    those the classifier's later rulings moved included; all but ByPass are
    still among the agreement test's spellings."""
    assert len(FORMERLY_PENDING) == 69
    assert set(FORMERLY_PENDING) - set(SPELLINGS) == NOW_NO_PASS_WORD
    assert [name for name in FORMERLY_PENDING if _disagreements(name)] == []
    assert {name for name in NOW_SETTINGS if entry_kind(name, SECRET) == "setting"} == (
        NOW_SETTINGS
    )
    assert entry_kind("ByPass", SECRET) is None


# bypass is one word in camel case (the classifier's _ONE_WORDS); by_pass,
# BY_PASS and by-pass are two, a password's.
ONE_WORD_BYPASS = [
    "ByPass",
    "byPass",
    "isByPass",
    "ByPass_2",
    "byPassValue",
    "x_byPass",
]


# Names that only end in one of those, each a password's: on_pass and
# norm_pass made "ends in" keep COMMON_PASS, SESSION_PASS and abnorm_pass in
# text (found when the classifier's new words were merged); the rule reads
# a whole name now.
ENDS_IN_ONE = [
    "COMMON_PASS", "SESSION_PASS", "DAEMON_PASS", "X_ON_PASS", "X_FIDELITY_PASS",
    "common_pass", "commonPass", "session_pass", "abnorm_pass", "x_on_pass",
]  # fmt: skip


@pytest.mark.parametrize("name", ENDS_IN_ONE)
def test_a_name_that_only_ends_in_a_settings_name_is_a_password(name):
    """in text (an env line, an entry, after an escaped newline) and
    as a whole name, every layer masks it, as the classifier says."""
    assert entry_kind(name, SECRET) == "credential"
    for text in (f"{name}={SECRET}", f'"{name}": "{SECRET}"', f"out\\n{name}={SECRET}"):
        assert {*_boundary_leaks(text, SECRET), *_path_leaks(text, SECRET)} == set(), (
            text
        )
    assert is_secret_field(name, SECRET)
    assert redact_value({name: SECRET})[0] == {name: EXPORT_MASK}


@pytest.mark.parametrize("name", ONE_WORD_BYPASS)
def test_bypass_in_camel_case_is_no_password(name):
    """After the classifier's ruling (b1c25bbe4): in text (an env
    line, an entry, YAML) and as a whole name, every layer keeps it, and
    config.json records it (2dddc83e8)."""
    assert _should_record_env_entry(name, SECRET) is True
    assert entry_kind(name, SECRET) is None
    for text in (f"{name}={SECRET}", f'"{name}": "{SECRET}"', f"{name}: {SYMBOLS}"):
        assert _unchanged(text) == dict.fromkeys(LAYERS, True), text
    assert not is_secret_field(name, SECRET)
    assert redact_value({name: SECRET}) == ({name: SECRET}, 0)
    assert _masked_env_args({name: SECRET}) == {name: SECRET}


@pytest.mark.parametrize(
    "name", ["by_pass", "BY_PASS", "standbyPass", "byPass_db_pass"]
)
def test_by_pass_in_two_words_is_a_password(name):
    """by and pass as two words, and a word glued before by, are a
    password's name, in text and as a whole name."""
    assert entry_kind(name, SECRET) == "credential"
    assert SECRET not in redact_trajectory_text(f"{name}={SECRET}")
    assert is_secret_field(name, SECRET)


# What the text rule leaves, by design: a value that reads as code. Each is
# masked as a whole name (above); in text the same characters are code
# elsewhere in the corpus (`tests_pass: bool`, `owner_pwd = user_pwd`,
# `f(db_pass=db_pass)`, `self.first_pass: bool = False`, `pass_value=3`).
LEFT_TO_CODE = [
    f"db_pass: {SECRET}",
    f"db_pass = {SECRET}",
    f"env db_pass={SECRET} ./run.sh",
    f'f(db_pass="{SECRET}")',
    f'self.db_pass = "{SECRET}"',
    "db_pass=1234",
    f'db_pass=["{SECRET}"]',
    f"db-pass: {SYMBOLS}",
]


@pytest.mark.parametrize("text", LEFT_TO_CODE)
def test_a_lower_case_value_that_reads_as_code_is_kept_in_text(text):
    """The trade-off the report states: an identifier after `:` or
    ` = ` (the corpus has 35 such code lines and 4 configuration entries,
    none a secret), an identifier followed by a space in an env line, a
    name that is a call's argument or an attribute, a count of up to five
    digits, a list, and a hyphenated name. A rule that masks one of these
    changes code; this test fails so the change is a deliberate one."""
    assert redact_trajectory_text(text) == text


def test_the_lower_case_shapes_are_the_classifiers_ended_by_a_backtick_too():
    """_LOWER_LITERAL and the count are the classifier's
    _TEST_RESULT and _COUNT_RESULT with a backtick as one more end, derived
    from them (a Markdown code span: `first_pass=True`). If the
    classifier's shape changes its end, the derivation no longer applies
    and this fails."""
    assert types._COUNT_OR_SWITCH.endswith(types._VALUE_END)
    assert types._VALUE_END in types._COUNT_RESULT
    assert types._LOWER_LITERAL.endswith(types._LOWER_VALUE_END)
    assert types._LOWER_VALUE_END in types._LOWER_COUNT_OF_ANY_LENGTH
    assert redact_trajectory_text("`first_pass=True`") == "`first_pass=True`"


@pytest.mark.parametrize(
    "unit",
    [
        "db_pass=hunter2FAKE\n",
        'dbPass: "hunter2FAKE" ',
        '"smtp_pwd": "hunter2FAKE", ',
        "tests_pass=hunter2FAKE\n",
        "pass_value=hunter2FAKE\n",
        "\\ndb_pass=hunter2FAKE ",
        "\x1b[32mdbPass=hunter2FAKE\n",
        "x_" * 30 + "pass=hunter2FAKE\n",
        "first_pass = True\n",
        "deterministic_pass=deterministic_pass, ",
        '"tests_pass": "12", ',
        "n_pass=123456 ",
        "tests_pass: bool\n",
        "`first_pass=True` ",
        "f(db_pass=",
        "self.x_pass = ",
        "Passed 12 of 12. ",
        "Pass",
        "_pass",
        "pass_",
        "aPass",
        "a" * 70 + "_pass=",
        "(" + "a" * 63 + "Pass=",
        " a" * 40 + "_pass",
        "allTestsPass=",
        'x_pass = "',
    ],
)
def test_lower_case_pass_names_repeated_stay_linear(unit):
    """each repeated to 50,000 and 200,000 characters: four times
    the text takes about four times as long, and a masked value is gone."""
    seconds, outs = linear_seconds(
        redact_trajectory_text, lambda size: _repeat(unit, size)
    )
    assert all("hunter2FAKE" not in out for out in outs)
    assert_linear(seconds)


CODE_AND_PROSE = (
    "def score(first_pass: bool, tests_pass=None):\n"
    "    all_blockers_pass = not failed_blockers\n"
    "    return Scoring(deterministic_pass=deterministic_pass, n_pass=3)\n"
    "PASSED tests/test_a.py::test_x - Pass 12 of 12\n"
    "db_pass=hunter2FAKE\n"
)


def test_a_200000_character_text_of_code_and_prose_stays_quick():
    """Python with lower-case pass names, a log line and an env
    line, 200,000 characters: masked where it holds an entry, at the shared
    check against 50,000 (it was well under a second, quiet)."""
    seconds, outs = linear_seconds(
        redact_trajectory_text, lambda size: _repeat(CODE_AND_PROSE, size)
    )
    text = _repeat(CODE_AND_PROSE, 200_000)
    assert "hunter2FAKE" not in outs[1]
    assert outs[1].count("all_blockers_pass = not failed_blockers") == text.count(
        "all_blockers_pass = not failed_blockers"
    )
    assert_linear(seconds)


def test_the_carrier_reads_lower_case_through_its_words():
    """the *_PASS carrier's words now hold the lower- and camel-case
    spellings, so a text with one runs it, and one with none still runs
    none of the seven (test_redaction_word_shortcut proves the words from
    the pattern)."""
    carrier = next(p for p, _, _ in types._REDACTION_PATTERNS if "_PASS" in p.words)
    assert set(types._LOWER_PASS_NAME_WORDS) <= set(carrier.words)
    for text in ("db_pass=x", "dbPass=x", "smtpPwd=x", "pass_value=x", "passvalue=x"):
        assert carrier.may_match(text, types._folded(text)), text
    assert not carrier.may_match("bypass compass", types._folded("bypass compass"))


# The lead's ruling (2026-11-14): the timing tests above call the
# shared check (tests/_timing.py). The positive control: the check, on one
# of each test's inputs at its sizes, fails once a quadratic term is added.
def test_the_lower_case_pass_name_timing_check_catches_a_quadratic():
    assert catches_quadratic(
        redact_trajectory_text, lambda size: _repeat("_pass", size)
    )


def test_the_code_and_prose_timing_check_catches_a_quadratic():
    assert catches_quadratic(
        redact_trajectory_text, lambda size: _repeat(CODE_AND_PROSE, size)
    )
