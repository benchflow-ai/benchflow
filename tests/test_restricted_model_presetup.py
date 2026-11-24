"""A restricted-web model BenchFlow can't admit is refused before any sandbox.

In 0.10.0rc4's dogfood a Gemini 4 Hydrogen run on a no-network task was
refused only after its 2 m 35 s image build and setup, filed as "other
(infrastructure problem)", and its hint named a base URL nobody had set.
The launch's own admission now runs in setup() first, with the same inputs,
and the refusal has its own category.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from benchflow._utils.scoring import (
    ENDING_KEEPS_REWARD,
    ERROR_CATEGORIES,
    RESTRICTED_MODEL_REFUSED,
    classify_error,
)
from benchflow.providers.restricted_models import (
    OPERATOR_ENV,
    RestrictedModelRefused,
    preflight_restricted_route,
)

HYDROGEN = "gemini/gemini-4-hydrogen"


def test_the_refusal_comes_before_the_sandbox():
    with pytest.raises(RestrictedModelRefused) as refused:
        preflight_restricted_route(
            "antigravity",
            HYDROGEN,
            {"GEMINI_API_KEY": "placeholder"},
            disallow_web_tools=True,
            environ={},
        )
    text = str(refused.value)
    assert OPERATOR_ENV in text
    # No base URL was set, so the message names none.
    assert "Use the supported" not in text


def test_an_operator_entry_admits_it_early_too():
    preflight_restricted_route(
        "antigravity",
        HYDROGEN,
        {"GEMINI_API_KEY": "placeholder"},
        disallow_web_tools=True,
        environ={OPERATOR_ENV: HYDROGEN},
    )


def test_an_open_network_needs_no_admission():
    preflight_restricted_route(
        "antigravity", HYDROGEN, {}, disallow_web_tools=False, environ={}
    )


def test_the_refusal_has_its_own_category():
    assert RESTRICTED_MODEL_REFUSED == "restricted_model_refused"
    assert RESTRICTED_MODEL_REFUSED in ERROR_CATEGORIES
    assert ENDING_KEEPS_REWARD[RESTRICTED_MODEL_REFUSED] is False
    for error in (
        f"RestrictedModelRefused: Refused '{HYDROGEN}' with web access restricted: x",
        f"RuntimeError: Refused '{HYDROGEN}' with web access restricted: x",
    ):
        assert classify_error(error) == RESTRICTED_MODEL_REFUSED, error


def test_setup_runs_the_admission_before_it_stages_the_task():
    from benchflow.rollout import Rollout

    source = inspect.getsource(Rollout.setup)
    tree = ast.parse(textwrap.dedent(source))
    calls = [
        n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    ]
    assert "preflight_restricted_model" in calls
    assert source.index("preflight_restricted_model") < source.index(
        "effective_task_path"
    )


def test_the_default_planes_run_the_admission():
    from benchflow.rollout_planes import DefaultRolloutPlanes

    with pytest.raises(RestrictedModelRefused):
        DefaultRolloutPlanes().preflight_restricted_model(
            "antigravity",
            HYDROGEN,
            {"GEMINI_API_KEY": "placeholder"},
            disallow_web_tools=True,
            environ={},
        )
