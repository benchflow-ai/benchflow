"""A price's name keeps a number and nothing else (#289).

The rc kept every value after the 35 price and rate names that names_all
holds (input_cost_per_token, usd_per_token, cost_per_1k_token, PER-TOKEN,
time_per_output_token, ...): a word, a key, a hex or base64 run. 0.9.3 masks
them all. The lead ruled on 2026-11-16 at 09:58Z that the exemption holds for
a number only: an optional sign, digits, an optional decimal part and an
optional exponent, and nothing else; any other value is masked as on rel.
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

# The 35 names that rc f0c7afb90 shows and rel 4ce66f4d1 masks (names_all).
PRICE_NAMES = [
    "COST_PER_INPUT_TOKEN",
    "Cost_Per_Input_Token",
    "INPUT_COST_PER_TOKEN",
    "MODEL_COST_PER_TOKEN",
    "PER-TOKEN",
    "PRICE_PER_TOKEN",
    "USD-PER-REQUEST-TOKEN",
    "_BATCH_TOKEN_ESTIMATE_BYTES_PER_TOKEN",
    "citation_cost_per_token",
    "cost-per-input-token",
    "cost_per_1k_token",
    "cost_per_input_token",
    "cost_per_token",
    "cost_per_user_token",
    "costs_per_user_token",
    "custom_cost_per_token",
    "input-cost-per-token",
    "input_cost_per_audio_token",
    "input_cost_per_image_token",
    "input_cost_per_token",
    "input_cost_per_video_token",
    "input_dbu_cost_per_token",
    "model_cost_per_token",
    "output_cost_per_audio_token",
    "output_cost_per_image_token",
    "output_cost_per_reasoning_token",
    "output_cost_per_token",
    "output_cost_per_video_token",
    "output_dbu_cost_per_token",
    "per_token",
    "price_per_org_token",
    "price_per_output_token",
    "time_per_output_token",
    "usd_per_request_token",
    "usd_per_token",
]
NUMBERS = ["0.0000025", "15", "-3", "2.5e-06", "+1E9"]
NOT_NUMBERS = {
    "hex": "deadbeefcafe0123456789ab",
    "base64": "QUJDREVGR0hJSktMTU5PUFFSU1RVVldY",
    "base64, padded": "aGVsbG8td29ybGQ=",
    "number and unit, glued": "15usd",
    "number and unit": "0.5 $",
    "opaque token": "FAKEq9Zx7Lm2Wv4Kp8Rt",
    "word": "pending",
}
# R632's separators: same line and next line.
SEPARATORS = {
    "eq": "=",
    "colon": ": ",
    "sp_colon": " :",
    "sp_eq": " = ",
    "eq_nl": "=\n",
    "sp_eq_nl2": " =\n\n",
}


def _layers(text: str) -> dict[str, str]:
    return {
        "text": redact_trajectory_text(text),
        "rec": redact_trajectory_text(json.dumps(text)),
        "obj": json.dumps(redact_trajectory_obj({"content": text})),
        "acp": redact_acp_trajectory_jsonl(
            [{"type": "tool_call", "raw_input": {"content": text}}]
        ),
    }


def _shown(text: str, word: str) -> list[str]:
    return [layer for layer, out in _layers(text).items() if word in out]


@pytest.mark.parametrize("separator", list(SEPARATORS.values()), ids=list(SEPARATORS))
@pytest.mark.parametrize("name", PRICE_NAMES)
def test_a_number_after_a_price_name_is_kept(name, separator):
    for number in NUMBERS:
        text = f"{name}{separator}{number}"
        assert _shown(text, number) == ["text", "rec", "obj", "acp"], text


@pytest.mark.parametrize("separator", list(SEPARATORS.values()), ids=list(SEPARATORS))
@pytest.mark.parametrize("name", PRICE_NAMES)
def test_anything_else_after_a_price_name_is_masked(name, separator):
    for kind, value in NOT_NUMBERS.items():
        text = f"{name}{separator}{value}"
        assert _shown(text, value.split()[0]) == [], (kind, text)


@pytest.mark.parametrize("name", ["input_cost_per_token", "usd_per_token", "PER-TOKEN"])
def test_a_quoted_value_and_a_structured_one_follow_the_same_rule(name):
    assert (
        redact_trajectory_text(f'{{"{name}": "2.5e-06"}}') == f'{{"{name}": "2.5e-06"}}'
    )
    assert "FAKEq9Zx7Lm2" not in redact_trajectory_text(f'{{"{name}": "FAKEq9Zx7Lm2"}}')
    for value in (2.5e-06, 15, "15", "-3", {"input": 3e-06, "output": 1.5e-05}):
        assert redact_trajectory_obj({name: value}) == {name: value}
        assert redact_value({name: value}) == ({name: value}, 0)
    for value in ("15usd", "FAKEq9Zx7Lm2", "deadbeefcafe0123"):
        assert redact_trajectory_obj({name: value}) == {name: "***REDACTED***"}
        assert redact_value({name: value})[1] == 1


def test_a_mixed_table_masks_what_is_no_number():
    """A table of prices is kept while every leaf is a number; a leaf that is
    not one is masked in every layer (F2b, 2026-11-16 10:35Z)."""
    record = {"usd_per_token": {"input": 3e-06, "output": "FAKEq9Zx7Lm2Wv4Kp8Rt"}}
    assert redact_trajectory_obj(record) == {
        "usd_per_token": {"input": 3e-06, "output": "***REDACTED***"}
    }
    assert redact_value(record)[1] == 1
    for text in (json.dumps(record), json.dumps(json.dumps(record))):
        out = redact_trajectory_text(text)
        assert "FAKEq9Zx7Lm2Wv4Kp8Rt" not in out and "3e-06" in out, out
    listed = 'cost_per_token: [1e-06, "deadbeefcafe0123", 2e-06]'
    assert redact_trajectory_text(listed) == (
        'cost_per_token: [1e-06, "***REDACTED***", 2e-06]'
    )
    numbers = {"usd_per_token": {"input": 3e-06, "output": 1.5e-05}}
    assert redact_trajectory_text(json.dumps(numbers)) == json.dumps(numbers)
    assert redact_value(numbers) == (numbers, 0)


def test_a_number_with_a_word_after_it_is_still_a_number():
    """A unit is a currency's sign or word; other words after a number are
    prose (`3e-06 end`), and another entry on the line is its own."""
    assert (
        redact_trajectory_text("usd_per_token=3e-06 end") == "usd_per_token=3e-06 end"
    )
    assert (
        redact_trajectory_text("A_COST_PER_TOKEN=3e-06 B=1")
        == "A_COST_PER_TOKEN=3e-06 B=1"
    )
    assert (
        redact_trajectory_text("usd_per_token=0.5 usd")
        == "usd_per_token=***REDACTED*** usd"
    )
