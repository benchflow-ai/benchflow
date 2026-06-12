"""The eval console must end-truncate error messages, never mid-token.

A bare ``msg[:50]`` slice rendered::

    [ERR] authored-task (tools=0) (Docker compose command failed for environment auth)

— the embedded task name was cut to ``auth``, which reads as a complete
environment name.  ``truncate_end`` keeps the message a verbatim prefix
of the original, cut at a word boundary and marked with an ellipsis.
"""

from __future__ import annotations

import logging

from benchflow._utils.text import truncate_end
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.models import RunResult

TASK_NAME = "authored-task-for-the-quarterly-regression-suite"
ERROR = (
    f"Docker compose command failed for environment {TASK_NAME}: "
    "exit status 1 while creating the task network"
)


def test_truncate_end_within_budget_is_unchanged():
    assert truncate_end("all good", 8) == "all good"


def test_truncate_end_keeps_word_that_ends_exactly_at_the_cut():
    assert truncate_end("alpha beta gamma", 11) == "alpha beta…"


def test_truncate_end_drops_partial_trailing_token():
    assert truncate_end("alpha beta gamma", 13) == "alpha beta…"


def test_truncate_end_cuts_oversized_single_token_with_marker():
    assert truncate_end("x" * 30, 10) == "x" * 9 + "…"


def test_truncate_end_degenerate_budgets():
    assert truncate_end("long message", 1) == "…"
    assert truncate_end("long message", 0) == ""


def test_truncate_end_never_exceeds_limit_and_keeps_a_verbatim_prefix():
    for limit in range(2, len(ERROR) + 5):
        out = truncate_end(ERROR, limit)
        assert len(out) <= limit
        if len(ERROR) <= limit:
            assert out == ERROR
        else:
            assert out.endswith("…")
            assert ERROR.startswith(out[:-1])


def test_log_and_report_renders_end_truncated_error(tmp_path, caplog):
    evaluation = Evaluation(
        tasks_dir=tmp_path,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(retry=RetryConfig(max_retries=0)),
        job_name="truncation-test",
    )
    result = RunResult(task_name=TASK_NAME, error=ERROR)

    with caplog.at_level(logging.INFO, logger="benchflow.evaluation"):
        evaluation._log_and_report(tmp_path / TASK_NAME, result)

    err_lines = [m for m in caplog.messages if m.startswith("[ERR]")]
    assert err_lines == [
        f"[ERR] {TASK_NAME} (tools=0) (Docker compose command failed for environment…)"
    ]

    line = err_lines[0]
    preview = line[line.rindex("(") + 1 : line.rindex(")")]
    assert preview.endswith("…")
    assert len(preview) <= 50
    # End-truncation only: the kept text is a verbatim prefix of the error
    # made of complete words, so a sliced token can never pose as a name.
    kept = preview[:-1]
    assert ERROR.startswith(kept)
    assert kept.split() == ERROR.split()[: len(kept.split())]
