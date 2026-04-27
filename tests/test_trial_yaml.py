"""Tests for trial YAML parsing helpers."""

from benchflow.trial_yaml import parse_role


def test_parse_role_model_name_fallback():
    """Guards the fix from commit 5431cdd: parse_role accepts model_name as fallback for model (issue #4)."""
    role = parse_role({"name": "r", "agent": "a", "model_name": "m"})
    assert role.model == "m"


def test_parse_role_model_takes_precedence():
    """Guards the fix from commit 5431cdd: parse_role prefers model over model_name when both are set (issue #4)."""
    role = parse_role({"name": "r", "agent": "a", "model": "m1", "model_name": "m2"})
    assert role.model == "m1"


def test_parse_role_no_model_or_model_name():
    """Guards the fix from commit 5431cdd: parse_role returns Role.model=None when neither key is set (issue #4)."""
    role = parse_role({"name": "r", "agent": "a"})
    assert role.model is None
