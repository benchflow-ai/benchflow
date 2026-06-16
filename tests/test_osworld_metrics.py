"""Unit tests for the faithful OSWorld metric-function ports."""

from __future__ import annotations

import pytest

from benchflow.adapters.osworld_metrics import (
    check_include_exclude,
    exact_match,
    is_utc_0,
    resolve_metric,
)


class TestIsUtc0:
    def test_utc_line_passes(self) -> None:
        out = (
            "               Local time: Thu 2024-01-25 12:56:06 UTC\n"
            "           Universal time: Thu 2024-01-25 12:56:06 UTC\n"
            "                 RTC time: Thu 2024-01-25 12:56:05\n"
            "                Time zone: Etc/UTC (UTC, +0000)\n"
        )
        assert is_utc_0(out) == 1.0

    def test_non_utc_fails(self) -> None:
        out = (
            "               Local time: Thu 2024-01-25 12:56:06 WET\n"
            "           Universal time: Thu 2024-01-25 12:56:06 UTC\n"
            "                 RTC time: Thu 2024-01-25 12:56:05\n"
            "                Time zone: Atlantic/Faroe (WET, +0100)\n"
        )
        assert is_utc_0(out) == 0.0

    def test_empty_or_short_fails(self) -> None:
        assert is_utc_0("") == 0.0
        assert is_utc_0("one\ntwo\n") == 0.0


class TestCheckIncludeExclude:
    def test_all_include_present_no_exclude_passes(self) -> None:
        rules = {"include": ["check passed"], "exclude": []}
        assert check_include_exclude("Password check passed\n", rules) == 1.0

    def test_missing_an_include_fails(self) -> None:
        rules = {"include": ["check passed"], "exclude": []}
        assert check_include_exclude("Check failed\n", rules) == 0.0

    def test_present_exclude_fails(self) -> None:
        rules = {"include": ["ok"], "exclude": ["error"]}
        assert check_include_exclude("ok but error\n", rules) == 0.0

    def test_none_result_fails(self) -> None:
        assert check_include_exclude(None, {"include": ["x"]}) == 0.0

    def test_multiple_includes_all_required(self) -> None:
        rules = {"include": ["a", "b"], "exclude": []}
        assert check_include_exclude("a only", rules) == 0.0
        assert check_include_exclude("a and b", rules) == 1.0


class TestExactMatch:
    def test_equal_passes(self) -> None:
        assert exact_match("Directory exists.\n", {"expected": "Directory exists.\n"}) == 1.0

    def test_unequal_fails(self) -> None:
        assert exact_match("Directory does not exist.\n", {"expected": "Directory exists.\n"}) == 0.0


class TestResolveMetric:
    def test_resolves_known_func(self) -> None:
        assert resolve_metric("check_include_exclude") is check_include_exclude
        assert resolve_metric("exact_match") is exact_match

    def test_unported_func_raises_actionable_error(self) -> None:
        with pytest.raises(NotImplementedError, match="not ported yet"):
            resolve_metric("compare_pptx")
