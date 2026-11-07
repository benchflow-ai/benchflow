"""Guard the S1 e4db9aa7 merge with integ's phase-local setup validation."""

import pytest

from benchflow.task.env import resolve_env_vars


def test_setup_output_wins_without_reading_host(monkeypatch):
    """Guards batch 22: produced values are local, including empty outputs."""
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "")
    monkeypatch.setenv("B22_CAPTURE", "host-value")
    assert resolve_env_vars(
        {"VALUE": "${B22_CAPTURE:-fallback}"}, extra_env={"B22_CAPTURE": ""}
    ) == {"VALUE": ""}
    assert resolve_env_vars(
        {"VALUE": "${B22_CAPTURE}"}, extra_env={"B22_CAPTURE": "setup-value"}
    ) == {"VALUE": "setup-value"}


def test_setup_output_does_not_allow_other_host_values(monkeypatch):
    """Guards batch 22: the S1 refusal survives the setup-output overlay."""
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "")
    monkeypatch.setenv("B22_HOST", "host-value")
    with pytest.raises(ValueError, match="--allow-task-env B22_HOST"):
        resolve_env_vars(
            {"VALUE": "${B22_HOST}"}, extra_env={"B22_CAPTURE": "setup-value"}
        )
    assert resolve_env_vars(
        {"VALUE": "${B22_HOST:-fallback}"},
        extra_env={"B22_CAPTURE": "setup-value"},
    ) == {"VALUE": "fallback"}


def test_explicitly_allowed_host_value_survives_overlay(monkeypatch):
    """Guards batch 22: an explicit operator grant still reads the host."""
    monkeypatch.setenv("B22_HOST", "host-value")
    assert resolve_env_vars(
        {"VALUE": "${B22_HOST}"},
        allowed=["B22_HOST"],
        extra_env={"B22_CAPTURE": "setup-value"},
    ) == {"VALUE": "host-value"}
