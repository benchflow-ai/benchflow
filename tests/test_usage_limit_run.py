"""A usage limit fails fast: an unscored trial named by its login, never retried.

Guards the dx/errors fix for the hill-climb smoke of 2026-09-30, where every
trial on a spent Claude subscription surfaced as ``acp_error`` and was retried
twice (three sandboxes per task), and a caller could only find out by
matching ``You've hit your ... limit`` in each trial's error string.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchflow._utils.scoring import USAGE_LIMIT, classify_error
from benchflow.agents.env import login_label
from benchflow.agents.errors import UsageLimitError
from benchflow.diagnostics import UsageLimitDiagnostic
from benchflow.evaluation import RetryConfig
from benchflow.rollout.session_factory_runtime import execute_prompts_session_factory

TOKEN = "sk-ant-oat01-not-a-real-token-but-long-enough-to-scan-for"
WEEKLY = "You've hit your weekly limit · resets Oct 2, 4pm (UTC)"


@pytest.mark.parametrize(
    ("agent", "env", "explicit", "expected"),
    [
        (
            "claude-agent-acp",
            {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN},
            {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN},
            "CLAUDE_CODE_OAUTH_TOKEN (agent_env)",
        ),
        (
            "claude-agent-acp",
            {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN},
            {},
            "CLAUDE_CODE_OAUTH_TOKEN (environment)",
        ),
        (
            "claude-agent-acp",
            {"ANTHROPIC_API_KEY": TOKEN, "CLAUDE_CODE_OAUTH_TOKEN": TOKEN},
            {},
            "ANTHROPIC_API_KEY (environment)",
        ),
        (
            "claude-agent-acp",
            {"_BENCHFLOW_SUBSCRIPTION_AUTH": "1"},
            {},
            "host login (~/.claude/.credentials.json)",
        ),
        (
            "claude-agent-acp",
            {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN, "BENCHFLOW_LOGIN_LABEL": "  WORK "},
            {},
            "WORK",
        ),
        (
            "codex-acp",
            {"CODEX_ACCESS_TOKEN": TOKEN},
            {},
            "CODEX_ACCESS_TOKEN (environment)",
        ),
        ("claude-agent-acp", {}, {}, None),
    ],
)
def test_the_login_is_named_by_a_label_never_its_value(agent, env, explicit, expected):
    label = login_label(agent, None, env, explicit)
    assert label == expected
    assert TOKEN not in (label or "")


def _claude_rollout(tmp_path: Path):
    from benchflow.rollout import Rollout, RolloutConfig

    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task",
            agent="claude-agent-acp",
            model="claude-haiku-4-5-20251001",
            agent_env={"CLAUDE_CODE_OAUTH_TOKEN": TOKEN},
        )
    )
    rollout._agent_env = {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN}
    return rollout


def test_the_rollout_names_the_login_and_records_the_category(tmp_path):
    rollout = _claude_rollout(tmp_path)
    error = UsageLimitError.from_text(
        WEEKLY, now=datetime(2026, 9, 30, 5, 0, tzinfo=UTC)
    )
    assert error is not None
    text = rollout._classify_acp_error(error)
    assert text == (
        "usage limit reached on login CLAUDE_CODE_OAUTH_TOKEN (agent_env): "
        "7-day window, resets 2026-10-02 16:00 UTC (" + WEEKLY + ")"
    )
    assert TOKEN not in text
    assert rollout._diagnostics.category_for_channel("error") == USAGE_LIMIT
    info = rollout._diagnostics.to_result_fields()["usage_limit_info"]
    assert info == {
        "login": "CLAUDE_CODE_OAUTH_TOKEN (agent_env)",
        "window": "7-day",
        "resets_at": "2026-10-02T16:00:00+00:00",
        "detail": WEEKLY,
    }
    assert UsageLimitDiagnostic.format_issue_from_dict("t", info).startswith(
        "t: usage limit reached on login CLAUDE_CODE_OAUTH_TOKEN (agent_env)"
    )


@pytest.mark.parametrize(
    "error",
    [
        "usage limit reached on login X: 7-day window, resets 2026-10-02 16:00 UTC (…)",
        # A result written before this fix (the smoke's own error string).
        "ACP error -32603: Internal error: You've hit your weekly limit · resets Oct 2, 4pm (UTC)",
    ],
)
def test_a_usage_limit_is_never_retried(error):
    assert classify_error(error) == USAGE_LIMIT
    assert RetryConfig().should_retry(error) is False
    # Not even when a caller's own exclude list leaves it out.
    assert RetryConfig(exclude_categories={"timeout"}).should_retry(error) is False
    # Other ACP errors still are.
    assert RetryConfig().should_retry("ACP error -32603: Internal error") is True


async def test_a_session_factory_usage_limit_is_not_turned_into_a_timeout():
    class Spent:
        def __init__(self) -> None:
            self.steps: list = []

        async def prompt(self, _text):
            raise UsageLimitError(WEEKLY)

    with pytest.raises(UsageLimitError):
        await execute_prompts_session_factory(Spent(), ["hi"], timeout=30)
