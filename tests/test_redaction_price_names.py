"""Credential names with PER_ in them, and the prices per token they are not.

Review of PR #173, 2026-11-13: the price rule turned real credentials into
prices. Every value here is made up.

PR #148 read a price per token as `per`, a word of its own, then at most one
word, then TOKEN (`input_cost_per_token`, `output_cost_per_reasoning_token`).
That made `MCP_PER_USER_TOKEN`, `PER_SERVER_TOKEN`, `GITLAB_PER_PROJECT_TOKEN`
and `per_request_token` prices too, and the value under each was kept in
every layer; before #148 all were masked. The lead's rule:

- a name that ends in PER_TOKEN is a setting;
- one that ends in PER_<word>_TOKEN is a setting only when <word> is a kind
  of token (input, output, prompt, completion, reasoning, cached, audio,
  image, video, text), or the name holds a price word (cost, price, usd);
- every other name with PER_ is a credential's.

A provider's or a credential's word between `per` and the token makes no
price (`cost_per_api_access_token`). The lead's later ruling: text and whole
names follow the same rule, from the classifier's word lists. A whole name
with `per` is decided by the classifier (``name_kind``); running text reads
the price word just before `per` (``_PRICED_TOKEN``). So
``usd_per_request_token=V`` is a price everywhere.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from benchflow._utils.config_redaction import _should_record_env_entry
from benchflow._utils.credential_names import name_kind
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
VALUE = "FAKEperName0aZ9bY7Lm2"
PIN = 482139

# The six names the review measured, and more of their family.
REVIEWED = [
    "MCP_PER_USER_TOKEN",
    "PER_USER_TOKEN",
    "PER_SERVER_TOKEN",
    "GITLAB_PER_PROJECT_TOKEN",
    "PER_ORG_TOKEN",
    "per_request_token",
]
FAMILY = [
    "PER_TENANT_TOKEN",
    "PER_REPO_TOKEN",
    "per-user-token",
    "X-Per-Request-Token",
    "perUserToken",
    "per_user_api_token",
    "per_user_access_token",
]
CREDENTIALS = REVIEWED + FAMILY


def _camel(name: str) -> str:
    words = re.split(r"[_-]+", name)
    if len(words) == 1:
        return name
    words = [word.lower() for word in words]
    return words[0] + "".join(word.capitalize() for word in words[1:])


def _spellings(name: str) -> list[str]:
    """*name* as written, in lower, upper and camel case."""
    return list(dict.fromkeys([name, name.lower(), name.upper(), _camel(name)]))


def _text_forms(name: str) -> list[str]:
    forms = []
    for spelled in _spellings(name):
        forms += [
            f"{spelled}={VALUE}",
            f"{spelled}: {VALUE}",
            f'"{spelled}": "{VALUE}"',
            f'\\"{spelled}\\": \\"{VALUE}\\"',
            f"out\\n{spelled}={VALUE}",
            f"out\x1b[32m{spelled}={VALUE}",
            f"out\\u001b[1;32m{spelled}: {VALUE}",
        ]
    return forms


def _kept_by(text: str, value: str = VALUE) -> list[str]:
    """The layers that keep *value*: text, a dict value, both ACP writers,
    result.json's ``_masked``, a serialized line and the export redactor."""
    return sorted({*_boundary_leaks(text, value), *_path_leaks(text, value)})


@pytest.mark.parametrize("name", CREDENTIALS)
def test_a_value_under_a_per_name_in_text_is_masked(name):
    """Review of PR #173, 2026-11-13: in each case, quoted, JSON-escaped,
    after an escaped newline and after a colour code, in every layer."""
    kept = [(text, layers) for text in _text_forms(name) if (layers := _kept_by(text))]
    assert kept == []


@pytest.mark.parametrize("name", CREDENTIALS)
def test_a_value_under_a_per_name_as_a_key_is_masked(name):
    """Review of PR #173, 2026-11-13: the name as an object key, a header
    and an env name: the string under it is masked in trajectory files, at
    export and in the hosted run's records."""
    for spelled in _spellings(name):
        assert is_secret_field(spelled, VALUE), spelled
        assert _dict_leaks({spelled: VALUE}, VALUE) == [], spelled
        assert _dict_key_leaks(spelled, VALUE, VALUE) == [], spelled
        assert _masked_env_args({spelled: VALUE, "n": PIN}) == {spelled: MASK, "n": PIN}


@pytest.mark.parametrize("name", CREDENTIALS)
def test_a_number_and_an_env_entry_under_a_per_name_are_masked(name):
    """Review of PR #173, 2026-11-13: a PIN-like number under the name is
    masked in trajectory files and at export, and config.json does not
    record the name's env entry (both decided by the classifier, rebuilt to
    the same ruling)."""
    for spelled in _spellings(name):
        assert is_secret_number(spelled, PIN), spelled
        assert _dict_leaks({spelled: PIN}, str(PIN)) == [], spelled
        assert not _should_record_env_entry(spelled, VALUE), spelled


# Prices per token keep their value and its type.
PRICES = [
    "MODEL_COST_PER_TOKEN",
    "COST_PER_INPUT_TOKEN",
    "input_cost_per_token",
    "output_cost_per_reasoning_token",
    "input_cost_per_audio_token",
    "usd_per_token",
    "price_per_output_token",
    "input_cost_per_token_above_200k_tokens",
]
PRICE_VALUES = (3e-06, 0.000015, 1500, 20000)


@pytest.mark.parametrize("name", PRICES)
def test_a_price_per_token_keeps_its_value_and_type(name):
    """Review of PR #173, 2026-11-13: in text (as written, quoted and
    JSON-escaped) and as an object key, a float or an integer, in every
    layer; cost accounting reads them back."""
    for value in PRICE_VALUES:
        for text in (
            f"{name}={value}",
            f"{name}: {value}",
            f'"{name}": {value}',
            f'"{name}": "{value}"',
            f'\\"{name}\\": {value}',
        ):
            assert _unchanged(text) == dict.fromkeys(LAYERS, True), text
            assert _path_leaks(text, str(value)) == list(PATH_LAYERS), text
        for obj in ({name: value}, {"usage": {name: value}}, {"m": {name: value}}):
            kept = redact_trajectory_obj(obj)
            assert kept == obj and json.dumps(kept) == json.dumps(obj)
            assert redact_value(obj) == (obj, 0)
            assert json.dumps(redact_value(obj)[0]) == json.dumps(obj)
        assert not is_secret_field(name, value)
        assert not is_secret_number(name, value)
        assert _masked_env_args({name: value, "s": str(value)}) == {
            name: value,
            "s": str(value),
        }
    assert _masked_env_args({name: "3e-06"}) == {name: "3e-06"}


@pytest.mark.parametrize("name", PRICES)
def test_a_price_per_token_is_kept_after_every_prefix_of_the_shape_matrix(name):
    """Review of PR #173, 2026-11-13: the must-keep rows beside the shape
    matrix's must-mask rows, after each of its prefixes, on its six write
    paths."""
    kept = {
        context: _path_leaks(f"{prefix}{name}=3e-06 end", "3e-06")
        for context, prefix in PREFIXES.items()
    }
    assert kept == {context: list(PATH_LAYERS) for context in PREFIXES}


# The real names: every key of LiteLLM's price table with `per_` and `token`
# in it (the installed version, 179 keys), and benchflow's own.
REAL_PRICES = [
    "citation_cost_per_token",
    "computer_use_input_cost_per_1k_tokens",
    "computer_use_output_cost_per_1k_tokens",
    "input_cost_per_audio_per_second_above_128k_tokens",
    "input_cost_per_audio_token",
    "input_cost_per_audio_token_batches",
    "input_cost_per_audio_token_priority",
    "input_cost_per_character_above_128k_tokens",
    "input_cost_per_image_above_128k_tokens",
    "input_cost_per_image_token",
    "input_cost_per_image_token_batches",
    "input_cost_per_token",
    "input_cost_per_token_above_128k_tokens",
    "input_cost_per_token_above_200k_tokens",
    "input_cost_per_token_above_200k_tokens_priority",
    "input_cost_per_token_above_256k_tokens",
    "input_cost_per_token_above_272k_tokens",
    "input_cost_per_token_above_272k_tokens_flex",
    "input_cost_per_token_above_272k_tokens_priority",
    "input_cost_per_token_above_512k_tokens",
    "input_cost_per_token_batches",
    "input_cost_per_token_cache_hit",
    "input_cost_per_token_flex",
    "input_cost_per_token_priority",
    "input_cost_per_video_per_second_above_128k_tokens",
    "input_cost_per_video_token",
    "input_cost_per_video_token_batches",
    "input_dbu_cost_per_token",
    "output_cost_per_audio_token",
    "output_cost_per_character_above_128k_tokens",
    "output_cost_per_image_token",
    "output_cost_per_reasoning_token",
    "output_cost_per_token",
    "output_cost_per_token_above_128k_tokens",
    "output_cost_per_token_above_200k_tokens",
    "output_cost_per_token_above_200k_tokens_priority",
    "output_cost_per_token_above_256k_tokens",
    "output_cost_per_token_above_272k_tokens",
    "output_cost_per_token_above_272k_tokens_flex",
    "output_cost_per_token_above_272k_tokens_priority",
    "output_cost_per_token_above_512k_tokens",
    "output_cost_per_token_batches",
    "output_cost_per_token_flex",
    "output_cost_per_token_priority",
    "output_cost_per_video_token",
    "output_dbu_cost_per_token",
    # benchflow's own (providers/litellm_config.py, branch_view.py, the
    # pricing snapshot, prime_rl.py).
    "MODEL_COST_PER_TOKEN",
    "custom_cost_per_token",
    "usd_per_token",
    "input_cost_per_token_above_100k_tokens",
    "output_cost_per_token_above_100k_tokens",
    "per_sample_token_count",
]


def _installed_price_names() -> list[str]:
    spec = importlib.util.find_spec("litellm")
    if spec is None or not spec.submodule_search_locations:
        return []
    table = Path(spec.submodule_search_locations[0])
    table /= "model_prices_and_context_window_backup.json"
    if not table.is_file():
        return []
    models = json.loads(table.read_text()).values()
    keys = {key for model in models if isinstance(model, dict) for key in model}
    return sorted(key for key in keys if re.search(r"per_.*token", key))


def test_the_real_price_names_are_the_installed_tables():
    """The list above is LiteLLM's table as installed: a new price name
    there is added to it (and so checked below)."""
    installed = _installed_price_names()
    if not installed:
        pytest.skip("LiteLLM's price table is not installed")
    assert set(installed) <= set(REAL_PRICES)


@pytest.mark.parametrize("name", REAL_PRICES)
def test_every_real_price_name_keeps_its_price(name):
    """Review of PR #173, 2026-11-13: in text, as a key in a trajectory
    file and at export, a price (a float, as the table has it) is kept, and
    so is an integer."""
    for text in (f"{name}=3e-06", f'"{name}": 3e-06', f'"{name}": "3e-06"'):
        assert _unchanged(text) == dict.fromkeys(LAYERS, True), text
    obj = {"model_info": {name: 3e-06}}
    assert redact_trajectory_obj(obj) == obj
    assert redact_value(obj) == (obj, 0)
    assert not is_secret_field(name, 3e-06)
    assert not is_secret_number(name, 3e-06)
    assert not is_secret_number(name, 1500)
    assert redact_value({name: 1500}) == ({name: 1500}, 0)


# A price word, `per` and any one word that is no provider's or
# credential's: a price in text and as a whole name (the lead's ruling,
# review of PR #173, 2026-11-13, which replaced the text-only narrowing).
PRICE_WORD_NAMES = [
    "usd_per_request_token",
    "cost_per_user_token",
    "price_per_org_token",
    "cost_per_input_token",
    "costs_per_user_token",
    "cost_per_1k_token",
]


@pytest.mark.parametrize("name", PRICE_WORD_NAMES)
def test_a_price_word_name_is_a_price_in_text_and_as_a_name(name):
    """Review of PR #173, 2026-11-13 (RPUB's probe `cost_per_user_token=
    "0.000003"`): the value is kept in text, in a trajectory file, at export
    and in a hosted run's records, as the classifier keeps it."""
    assert name_kind(name) == "setting"
    for text in (f"{name}=0.000003", f'"{name}": "0.000003"', f"{name}: 0.002"):
        assert _unchanged(text) == dict.fromkeys(LAYERS, True), text
    # A price's name keeps a number only (the lead's ruling of 2026-11-16 09:58Z).
    assert is_secret_field(name, VALUE)
    assert not is_secret_number(name, PIN)
    for value in (0.002, PIN, "0.000003"):
        assert redact_trajectory_obj({name: value}) == {name: value}
        assert redact_value({name: value}) == ({name: value}, 0)
        assert _masked_env_args({name: value}) == {name: value}


# What each spelling gets: (name, masked in running text, a secret's name as
# a whole name). Text reads `_` or `-` between the words, in any case, and no
# camel case or dot; a whole name is the classifier's, which reads camel
# case, `-` and `.` too.
SPELLINGS = [
    ("input_cost_per_token", False, False),
    ("INPUT_COST_PER_TOKEN", False, False),
    ("input-cost-per-token", False, False),
    ("cost-per-input-token", False, False),
    ("Cost_Per_Input_Token", False, False),
    ("costPerInputToken", True, False),
    ("cost.per.input.token", True, False),
    ("costperinputtoken", True, True),
    ("per_token", False, False),
    ("PER-TOKEN", False, False),
    ("perToken", True, False),
    ("PER_USER_TOKEN", True, True),
    ("per-user-token", True, True),
    ("Per-User-Token", True, True),
    ("perUserToken", True, True),
    ("PerUserToken", True, True),
    ("per.user.token", True, True),
    ("usd_per_request_token", False, False),
    ("usdPerRequestToken", True, False),
    ("USD-PER-REQUEST-TOKEN", False, False),
    ("costs_per_user_token", False, False),
    ("cost_per_1k_token", False, False),
    ("cost_per_api_access_token", True, True),
    ("price_per_refresh_token", True, True),
    ("time_per_output_token", False, False),
]


@pytest.mark.parametrize("name,in_text,as_name", SPELLINGS)
def test_what_each_spelling_of_a_per_name_gets(name, in_text, as_name):
    """Review of PR #173, 2026-11-13: case, `_` and `-`, camel case and the
    whole-name path, as the code treats them. A plural price word counts
    (`costs_per_user_token`), as in the classifier. The columns are what a
    number gets: since the lead's ruling of 2026-11-16 09:58Z a price's name keeps a number only, and an
    opaque value is masked under every spelling."""
    number = "0.000003"
    assert (number not in redact_trajectory_text(f"{name}={number}")) is in_text
    assert is_secret_field(name, number) is as_name
    assert VALUE not in redact_trajectory_text(f"{name}={VALUE}")
    assert is_secret_field(name, VALUE)


@pytest.mark.parametrize("name", REVIEWED)
def test_per_names_repeated_stay_linear(name):
    """Review of PR #173, 2026-11-13: each name repeated to 50,000 and
    200,000 characters, glued and as `NAME=value` pairs: four times the text
    takes about four times as long, and every value is masked."""
    for unit in (name, f"{name}={VALUE} "):
        seconds, outs = linear_seconds(
            redact_trajectory_text, lambda size, unit=unit: _repeat(unit, size)
        )
        assert all(VALUE not in out for out in outs), unit
        assert_linear(seconds)


# The lead's ruling for whole names: a price word and `per_` and a unit is a
# price, whatever the unit (`session` included); so a number under it keeps
# its value and type.
PRICE_UNIT_NAMES = [
    "code_interpreter_cost_per_session",
    "per_session_cost",
    "cost_per_request",
    "usd_per_1k",
    "usd_per_million",
    "price_per_second",
    "input_cost_per_query",
    "input_cost_per_character",
    "output_cost_per_image",
    "costPerSession",
    "input_cost_per_token_batches",
    "input_cost_per_audio_token_batches",
    "input_cost_per_audio_token_priority",
    "input_cost_per_image_token_batches",
    "input_cost_per_token_cache_hit",
    "input_cost_per_token_flex",
    "input_cost_per_token_priority",
    "input_cost_per_video_token_batches",
    "output_cost_per_token_batches",
    "output_cost_per_token_flex",
    "output_cost_per_token_priority",
]


@pytest.mark.parametrize("name", PRICE_UNIT_NAMES)
def test_a_number_under_a_price_with_any_unit_is_kept(name):
    """Review of PR #173, 2026-11-13 (the lead's ruling on whole names): an
    integer under the name keeps its value and type in trajectory files, at
    export and in the hosted run's records."""
    assert name_kind(name) == "setting"
    for value in (1500, 20000, 482139, 0.002):
        obj = {name: value}
        assert not is_secret_number(name, value)
        assert json.dumps(redact_trajectory_obj(obj)) == json.dumps(obj)
        assert redact_value(obj) == (obj, 0)
        assert _masked_env_args(obj) == obj
    assert not is_secret_field(name, 1500)


@pytest.mark.parametrize(
    "name", ["mcp_per_user_token", "per_user_token", "session_token"]
)
def test_a_per_name_with_no_price_word_is_no_price(name):
    """Review of PR #173, 2026-11-13: a `per_` name with no price word is a
    credential's, its number masked."""
    assert name_kind(name) == "credential"
    assert is_secret_field(name, VALUE)
    assert is_secret_number(name, PIN)


@pytest.mark.parametrize(
    "name",
    [
        "cost_per_api_access_token",
        "price_per_refresh_token",
        "usd_per_access_token",
        "cost_per_auth_token",
    ],
)
def test_a_credential_word_between_per_and_token_makes_no_price(name):
    """Review of PR #173, 2026-11-13 (the lead's ruling): the price rule
    yields when a provider's or a credential's word sits between `per` and
    the token; a string and a number under the name are masked in text, as
    a key, at export and in a hosted run's records."""
    assert name_kind(name) == "credential"
    assert is_secret_field(name, VALUE)
    assert is_secret_number(name, PIN)
    assert _kept_by(f"{name}={VALUE}") == []
    assert _dict_leaks({name: VALUE}, VALUE) == []
    assert _dict_leaks({name: PIN}, str(PIN)) == []
    assert _masked_env_args({name: VALUE}) == {name: MASK}


# The lead's ruling (2026-11-14): the timing tests above call the
# shared check (tests/_timing.py). The positive control: the check, on one
# of each test's inputs at its sizes, fails once a quadratic term is added.
def test_the_per_name_timing_check_catches_a_quadratic():
    assert catches_quadratic(
        redact_trajectory_text, lambda size: _repeat(f"{REVIEWED[0]}={VALUE} ", size)
    )
