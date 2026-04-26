"""Tests for trial YAML parsing helpers."""

from benchflow.trial_yaml import parse_role


def test_parse_role_model_name_fallback():
    """Guards parse_role model_name fallback (issue #4 plan, decision 2C)."""
    role = parse_role({"name": "r", "agent": "a", "model_name": "m"})
    assert role.model == "m"


def test_parse_role_model_takes_precedence():
    """Guards parse_role precedence: model wins when both keys set (issue #4, decision 2C)."""
    role = parse_role({"name": "r", "agent": "a", "model": "m1", "model_name": "m2"})
    assert role.model == "m1"


def test_parse_role_no_model_or_model_name():
    """Guards parse_role None default (issue #4 plan, decision 2C)."""
    role = parse_role({"name": "r", "agent": "a"})
    assert role.model is None
