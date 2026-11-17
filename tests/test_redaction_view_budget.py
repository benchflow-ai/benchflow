"""(rc4-blocking; the lead, 2026-11-15 13:27Z): the decoded views of one redaction run under one
deterministic work budget (5 units of estimated view seconds on the reference VM) with a 120 s wall-clock backstop.
Past either bound, every remaining gated line is masked whole and counted under a category that names the bound:
over-mask, never a leak. Made-up values only."""

from __future__ import annotations

import json
import random

import pytest

from benchflow.publish.redact import (
    format_redaction_breakdown,
    redact_value,
    redact_value_to_stability,
)
from benchflow.trajectories import types
from benchflow.trajectories.types import (
    REDACTION_CATEGORY_VIEW_BACKSTOP,
    REDACTION_CATEGORY_VIEW_BUDGET,
    redact_acp_trajectory_jsonl,
    redact_trajectory_obj,
    redact_trajectory_text_with_categories,
)
from tests._timing import assert_linear, linear_seconds, within_ceiling

B = "\\"


def _key(i: int, rng: random.Random) -> str:
    alphabet = "ABCDEFGHJKMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789"
    return f"FAKE{i:02d}" + "".join(rng.choice(alphabet) for _ in range(36))


def _nested_row(n: int) -> str:
    """A JSON row three copies deep; each row differs (*n*), so no view repeats
    and the per-call view cache can't make the file cheap."""
    row = json.dumps(
        {
            "api_key": "placeholder value",
            "note": "a tool line with a token count",
            "n": n,
        }
    )
    for _ in range(3):
        row = json.dumps(row)[1:-1]
    return row


def _planted(keys: list[str]) -> list[str]:
    """One credential in each escape family the gate opens on."""
    return [
        f"x_token{B}u003d{keys[0]} {B}n",
        f"api_key=&quot;{keys[1]}&quot;",
        f'"Authorization{B}u0022: %22proxyAuthorization%22: "{keys[2]}{B}u0022',
        f'has_authorization: {B}{B}"FAKEpre01234{B}n{keys[3]}{B}{B}"',
        f"note {B}n sk{B}u002dproj-{keys[4]}",
        f"s&#107;-ant-{keys[5]} &amp;",
        f"{B}{B}u0073k-{keys[6]}",
        f"%73k-{keys[7]} {B}n",
        f"{B}x73k-{keys[8]}",
        f"https:{B}/{B}/user:{keys[9]}@host.example",
        f'api{B}u005fkey: "{keys[10]}" {B}n',
        f"&#x73;k-{keys[11]} &quot;x&quot;",
    ]


def _pathological(size: int) -> tuple[str, list[str]]:
    """Lines whose views cost several passes a character (a JSON row three copies
    deep), and past 85% of the text one credential in each escape family."""
    rng = random.Random(686)
    width = len(_nested_row(0)) + 8
    keys = [_key(i, rng) for i in range(12)]
    head = [_nested_row(i) for i in range(int(size * 0.85 / width))]
    tail = [_nested_row(-i) for i in range(1, int(size * 0.15 / width))]
    planted = _planted(keys)
    for i, line in enumerate(planted):
        tail.insert(int(len(tail) * (i + 0.5) / len(planted)), line)
    return "\n".join(head + tail), keys


def test_a_pathological_file_leaks_nothing_and_counts_its_over_mask(monkeypatch):
    """At a budget of one unit (the file's views need about two), the budget runs
    out before the planted lines; the default budget's run is measured in the PR."""
    monkeypatch.setattr(types, "_VIEW_BUDGET_UNITS", 1.0)
    text, keys = _pathological(500_000)
    out, categories = within_ceiling(redact_trajectory_text_with_categories, text)
    assert [k for k in keys if k in out] == []
    assert categories.get(REDACTION_CATEGORY_VIEW_BUDGET, 0) > 0
    assert REDACTION_CATEGORY_VIEW_BACKSTOP not in categories


def test_the_wall_clock_backstop_masks_leaks_nothing_and_says_so(monkeypatch):
    text, keys = _pathological(200_000)
    monkeypatch.setattr(types, "_VIEW_DEADLINE_SECONDS", 0.0)
    out, categories = redact_trajectory_text_with_categories(text)
    assert [k for k in keys if k in out] == []
    assert categories.get(REDACTION_CATEGORY_VIEW_BACKSTOP, 0) > 0
    assert REDACTION_CATEGORY_VIEW_BUDGET not in categories


def test_the_work_budget_is_deterministic():
    text, _ = _pathological(500_000)
    assert redact_trajectory_text_with_categories(
        text
    ) == redact_trajectory_text_with_categories(text)


def test_below_the_budget_nothing_changes(monkeypatch):
    text, _ = _pathological(40_000)
    budgeted = redact_trajectory_text_with_categories(text)
    assert REDACTION_CATEGORY_VIEW_BUDGET not in budgeted[1]
    monkeypatch.setattr(types, "_VIEW_BUDGET_UNITS", 10.0**18)
    assert redact_trajectory_text_with_categories(text) == budgeted


def test_the_row_costs_cover_the_table():
    assert len(types._VIEW_ROW_NS) == len(types._REDACTION_PATTERNS)


def test_the_breakdown_names_the_bound():
    assert (
        format_redaction_breakdown({REDACTION_CATEGORY_VIEW_BUDGET: 3})
        == "3 lines masked whole (view work budget)"
    )
    assert (
        format_redaction_breakdown({REDACTION_CATEGORY_VIEW_BACKSTOP: 1})
        == "1 line masked whole (wall-clock backstop)"
    )


def _gated(size: int) -> str:
    unit = 'line {"api_key": "x\\\\n y", "note": "\\\\"quoted\\\\" text"}\n'
    return (unit * (size // len(unit) + 1))[:size]


@pytest.mark.parametrize("budget", ["unlimited", "default"])
def test_the_counter_checkpoints_stay_linear(budget, monkeypatch):
    """Gated text under the budget (unlimited) and past it (the default, where the
    fallback masks the rest): linear at the shared check."""
    if budget == "unlimited":
        monkeypatch.setattr(types, "_VIEW_BUDGET_UNITS", 10.0**18)
    seconds, _ = linear_seconds(types.redact_trajectory_text, _gated)
    assert_linear(seconds)


# One call, one budget: an object's leaves, an ACP trajectory's events and a
# publish value's fields share it (the 4.4 MB wire log's object path took 11 s
# with a budget per leaf). Made-up values.
LEAVES = {f"k{i}": f"x_token{B}u003dFAKEleaf{i:04d}abcdefgh" for i in range(20)}
CALLS = {
    "object": lambda: redact_trajectory_obj({"outer": [LEAVES, dict(LEAVES)]}),
    "ACP trajectory": lambda: redact_acp_trajectory_jsonl([LEAVES, LEAVES]),
    "publish value": lambda: redact_value({"outer": [LEAVES, LEAVES]}),
    "publish to stability": lambda: redact_value_to_stability({"outer": LEAVES}),
}


@pytest.mark.parametrize("call", CALLS)
def test_one_call_shares_one_budget(call, monkeypatch):
    made = []

    class Counting(types._ViewBudget):
        def __init__(self, units: float, seconds: float) -> None:
            made.append(units)
            super().__init__(units, seconds)

    monkeypatch.setattr(types, "_ViewBudget", Counting)
    CALLS[call]()
    assert len(made) == 1


@pytest.mark.parametrize("call", CALLS)
def test_a_spent_shared_budget_leaks_nothing(call, monkeypatch):
    monkeypatch.setattr(types, "_VIEW_BUDGET_UNITS", 1e-12)
    out = json.dumps(CALLS[call](), default=str)
    assert "FAKEleaf" not in out
