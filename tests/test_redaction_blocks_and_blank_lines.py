"""a value a YAML block scalar or a blank line puts on a later line,
and an object or list held under a credential's name, are masked in every
layer.

R192C's far_probe3 shapes (2026-11-15): before this change a YAML block
(`apiKey: |\\n  <v>`) was shown everywhere, a blank line between a name and
its value was shown in every escaped layer, and `{"password": {"a": "<v>"}}`
was shown in the structured layers. Every value here is made up.
"""

from __future__ import annotations

import json

import pytest

from benchflow.publish.redact import redact_value
from benchflow.trajectories.types import (
    redact_acp_trajectory_jsonl,
    redact_trajectory_obj,
    redact_trajectory_text,
    redact_trajectory_text_with_categories,
)

V = "Qz8Wm3Rt6Yp1Xk4Vn7Jb2Hd5Lc9Fg0Sa"
W = "Hx5Pk9Lm2Nb7Vc4Zq1Rt8Ys3Wd6Gf0Je"


def _layers(text: str) -> dict[str, str]:
    """Each layer's output for ``text``, decoded back to text where it is JSON."""
    rec = redact_trajectory_text(json.dumps({"o": text}))
    acp = redact_acp_trajectory_jsonl([{"type": "tool_call", "raw_output": text}])
    return {
        "text": redact_trajectory_text(text),
        "rec": json.loads(rec)["o"],
        "obj": redact_trajectory_obj({"output": text})["output"],
        "acp": json.loads(acp.splitlines()[0])["raw_output"],
        "publish": redact_value({"output": text})[0]["output"],
    }


def _shown(text: str, *values: str) -> dict[str, list[str]]:
    return {
        layer: [v for v in values if v in out]
        for layer, out in _layers(text).items()
        if any(v in out for v in values)
    }


@pytest.mark.parametrize(
    "text",
    [
        f"apiKey: |\n  {V}",
        f"apiKey: >-\n  {V}",
        f"apiKey: |2+\n  {V}",
        f"apiKey: |\n  {V}\n  {W}",
        f"apiKey: |\n  {V}\n\n  {W}",
        f"apiKey: |\n  {V}\n    {W}",
        f"config:\n  password: |\n    {V}\n    {W}",
        f"GITHUB_TOKEN: >\n  {V} {W}",
        f"apiKey: |   \n  {V}",
    ],
)
def test_a_yaml_block_under_a_credential_name_is_masked_in_every_layer(text):
    assert _shown(text, V, W) == {}


def test_a_yaml_block_ends_at_the_next_less_indented_line():
    out = redact_trajectory_text(f"apiKey: |\n  {V}\nname: kept\n")
    assert V not in out
    assert "name: kept" in out


def test_a_yaml_block_under_another_name_is_kept_and_not_counted():
    text = f"description: |\n  {V}\n  {W}"
    assert redact_trajectory_text_with_categories(text) == (text, {})


@pytest.mark.parametrize(
    "text",
    [
        f"password:\n\n  {V}",
        f"password=\n\n{V}",
        f"api_key:\n\n\n  {V}",
        "{\n" + f'  "password":\n\n    "{V}"\n' + "}",
        f"password:\n  {V}",
        f"GITHUB_TOKEN=\n{V}",
    ],
)
def test_a_value_on_a_later_line_is_masked_in_every_layer(text):
    assert _shown(text, V) == {}


@pytest.mark.parametrize(
    "text",
    [
        # The next line's own entry is no value.
        "GITHUB_TOKEN=\nHOME=/root",
        'api_key:\n  "name": "kept"',
        # Code that prints a name and a break.
        'print("password:\\n")',
        'msg = "token:\\n" + other',
    ],
)
def test_a_later_line_that_is_no_value_is_kept_in_the_escaped_layers(text):
    for layer in ("rec", "obj", "acp"):
        assert _layers(text)[layer] == text, layer


@pytest.mark.parametrize(
    "record",
    [
        {"password": {"a": V, "b": W}},
        {"secret_key": [V, W]},
        {"GITHUB_TOKEN": {"a": {"b": [V, {"c": W}]}}},
        {"api_key": [{"value": V}, W]},
    ],
)
def test_an_object_or_list_under_a_credential_name_masks_every_string(record):
    obj = json.dumps(redact_trajectory_obj(record))
    acp = redact_acp_trajectory_jsonl([{"type": "tool_call", "raw_input": record}])
    published = json.dumps(redact_value(record)[0])
    for out in (obj, acp, published):
        assert V not in out and W not in out


def test_only_string_leaves_are_masked_and_the_shape_is_kept():
    record = {"password": {"enabled": True, "n": 3, "none": None, "empty": "", "v": V}}
    assert redact_trajectory_obj(record) == {
        "password": {
            "enabled": True,
            "n": 3,
            "none": None,
            "empty": "",
            "v": "***REDACTED***",
        }
    }


def test_an_object_under_another_name_is_redacted_as_before():
    record = {"settings": {"theme": "dark", "note": f"api_key={V}"}}
    assert redact_trajectory_obj(record) == {
        "settings": {"theme": "dark", "note": "api_key=***REDACTED***"}
    }
