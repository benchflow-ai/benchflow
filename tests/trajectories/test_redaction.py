"""Tests for trajectory secret redaction patterns."""

import pytest

from benchflow.trajectories.types import redact_trajectory_text


@pytest.mark.parametrize(
    "label,raw,must_not_contain",
    [
        pytest.param(
            "anthropic sk-ant-",
            '{"key": "sk-ant-api03-abc123XYZ_defghijklmnopqrstuvwxyz0123456789"}',
            "defghijklmnopqrstuvwxyz0123456789",
            id="anthropic",
        ),
        pytest.param(
            "openai sk-proj-",
            '{"key": "sk-proj-abc123XYZ_defghijklmnopqrstuvwxyz0123456789"}',
            "defghijklmnopqrstuvwxyz0123456789",
            id="openai-proj",
        ),
        pytest.param(
            "openai sk-",
            '{"key": "sk-abc1234567defghijklmnopqrstuvwxyz"}',
            "defghijklmnopqrstuvwxyz",
            id="openai-generic",
        ),
        pytest.param(
            "google AIzaSy",
            '{"key": "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx"}',
            "FORTESTSONLYxxxxxxxxxxxxxxx",
            id="google",
        ),
        pytest.param(
            "aws AKIA",
            '{"key": "AKIAIOSFODNN7EXAMPLE"}',
            "IOSFODNN7EXAMPLE",
            id="aws-akia",
        ),
        pytest.param(
            "aws ASIA (STS)",
            '{"key": "ASIAQWERTYUIOPASDFGH"}',
            "QWERTYUIOPASDFGH",
            id="aws-asia",
        ),
        pytest.param(
            "daytona dtn_",
            '{"key": "dtn_abcdefghijklmnop1234567890"}',
            "ghijklmnop1234567890",
            id="daytona",
        ),
        pytest.param(
            "bearer header",
            '{"authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.secret"}',
            "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.secret",
            id="bearer",
        ),
        pytest.param(
            "x-api-key header",
            '{"x-api-key": "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx"}',
            "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx",
            id="x-api-key",
        ),
        pytest.param(
            "api-key header (Azure)",
            '{"api-key": "abc123secret456value"}',
            "abc123secret456value",
            id="api-key-azure",
        ),
    ],
)
def test_redacts_secret_pattern(label, raw, must_not_contain):
    result = redact_trajectory_text(raw)
    assert "***REDACTED***" in result, f"{label}: no redaction applied"
    assert must_not_contain not in result, f"{label}: secret suffix survived"


def test_preserves_non_secret_content():
    raw = '{"role": "user", "content": "Write a Python script to process data"}'
    assert redact_trajectory_text(raw) == raw


@pytest.mark.parametrize(
    "raw",
    [
        # AWS prefix in English words / identifiers (issue: ASIA matched as substring)
        '{"region": "ASIAPACIFIC"}',
        '{"id": "ASIANEWS2024UPDATED"}',
        # Hyphenated slugs containing "sk-" should not be flagged
        '{"queue": "task-sk-us-east-1-foo-bar-baz"}',
        '{"job": "workspace-sk-us-east-1-extra"}',
        # Short Daytona-prefixed identifiers
        '{"label": "dtn_v2_0"}',
        '{"name": "dtn_test"}',
        # Short Google-prefixed values that aren't keys
        '{"name": "AIzaSy"}',
    ],
)
def test_does_not_redact_non_secret_values(raw):
    """Patterns must not corrupt legitimate identifiers that share a prefix."""
    assert redact_trajectory_text(raw) == raw


def test_redacts_multiple_patterns_in_one_string():
    raw = (
        '{"env": "GEMINI_API_KEY=AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx '
        'ANTHROPIC_API_KEY=sk-ant-api03-abc123XYZ_longSecretValueHere"}'
    )
    result = redact_trajectory_text(raw)
    assert "FORTESTSONLYxxxxxxxxxxxxxxx" not in result
    assert "longSecretValueHere" not in result
    assert result.count("***REDACTED***") >= 2


def test_to_jsonl_uses_redaction(tmp_path):
    """End-to-end: Trajectory.to_jsonl applies redact_trajectory_text."""
    from benchflow.trajectories.types import (
        LLMExchange,
        LLMRequest,
        LLMResponse,
        Trajectory,
    )

    trajectory = Trajectory(
        session_id="test",
        exchanges=[
            LLMExchange(
                request=LLMRequest(
                    headers={"x-api-key": "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx"},
                ),
                response=LLMResponse(),
            ),
        ],
    )
    jsonl = trajectory.to_jsonl(redact_keys=True)
    assert "***REDACTED***" in jsonl
    assert "FORTESTSONLYxxxxxxxxxxxxxxx" not in jsonl


def test_to_jsonl_no_redaction_preserves_keys():
    from benchflow.trajectories.types import (
        LLMExchange,
        LLMRequest,
        LLMResponse,
        Trajectory,
    )

    trajectory = Trajectory(
        session_id="test",
        exchanges=[
            LLMExchange(
                request=LLMRequest(
                    headers={"x-api-key": "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx"},
                ),
                response=LLMResponse(),
            ),
        ],
    )
    jsonl = trajectory.to_jsonl(redact_keys=False)
    assert "AIzaSyFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxx" in jsonl
