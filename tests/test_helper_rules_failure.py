"""BenchFlow's helper could not add the agent's uid rules.

Such a trial is the sandbox's setup (sandbox_setup, as on the 0.10 line),
whatever words the helper's reason holds: retried under the run's retry
settings, and once its retries are spent unscored with the helper's reason,
out of score_excl_errors and the scored-only mean reward.
The every-trial score and mean reward (benchflow.score.v1) are unchanged.
"""

from __future__ import annotations

import pytest

from benchflow._utils.scoring import (
    ACP_ERROR,
    SANDBOX_SETUP,
    classify_error,
    mean_reward_every_trial,
    mean_scored_reward,
    score_summary_fields,
)
from benchflow.evaluation import RetryConfig

HELPER = (
    "RuntimeError: Failed to enforce sandbox-user egress firewall: BenchFlow's "
    "helper could not add the rules to main's namespace (status {status})"
)
REASON = (
    ": the metadata firewall helper's script did not report "
    "(exec output receipt not printed (exec status 125))"
)
# Reasons with another category's words: the agent's time limit, an install.
TIMED_OUT = ": the metadata firewall helper's exec timed out after 30 s"
INSTALL_FAILED = ": the helper's iptables install failed (exit 100)"


@pytest.mark.parametrize(
    "error",
    [
        HELPER.format(status="none"),
        HELPER.format(status="1"),
        HELPER.format(status="none") + REASON,
        # The helper's reason may hold another category's words.
        HELPER.format(status="none") + TIMED_OUT,
        HELPER.format(status="none") + INSTALL_FAILED,
        # Without the error's class, and as install_from_helper raises it.
        HELPER.format(status="none").removeprefix("RuntimeError: "),
        "BenchFlow's helper could not add the rules to main's namespace (status 2)",
    ],
    ids=[
        "status-none",
        "status-1",
        "reason",
        "reason-timed-out",
        "reason-install-failed",
        "no-class",
        "bare",
    ],
)
def test_a_helper_rule_install_failure_is_the_sandboxs_setup_and_retried(error):
    """Before, #283's sandbox_setup came after the install, pipe, provider
    and ACP markers: a reason saying "install failed" read as an install
    failure."""
    assert classify_error(error) == SANDBOX_SETUP
    assert RetryConfig().should_retry(error) is True
    # The run's own settings still decide.
    assert RetryConfig(retry_on_infra=False).should_retry(error) is False


def test_an_agents_words_quoting_it_are_not_read_as_it():
    """The match is BenchFlow's own message, from its head: an agent's ACP
    error that quotes it stays the agent's."""
    error = "ACP error -32000: agent said: " + HELPER.format(status="none")
    assert classify_error(error) == ACP_ERROR


@pytest.mark.parametrize(
    "reason",
    [REASON, TIMED_OUT, INSTALL_FAILED],
    ids=["reason", "timed-out", "install"],
)
def test_once_its_retries_are_spent_it_is_unscored_out_of_the_scored_figures(reason):
    """The trial keeps the helper's reason and no reward: out of
    score_excl_errors and the scored-only mean, whatever words the reason
    holds. The every-trial figures (benchflow.score.v1) count it as before."""
    error = HELPER.format(status="none") + reason
    results = [
        {
            "task_name": "helper-refused",
            "trial_name": "helper-refused__1",
            "error": error,
            "error_category": classify_error(error),
            "rewards": None,
        },
        {
            "task_name": "held",
            "trial_name": "held__1",
            "error": None,
            "rewards": {"reward": 1.0},
        },
    ]
    fields = score_summary_fields(results)
    assert fields["errored"] == 1
    assert fields["error_categories"] == {SANDBOX_SETUP: 1}
    assert fields["score_excl_errors_ratio"] == 1.0
    assert mean_scored_reward(results) == 1.0
    # benchflow.score.v1, unchanged: every trial, an unscored one as 0.
    assert fields["score_ratio"] == 0.5
    assert mean_reward_every_trial(results) == 0.5
