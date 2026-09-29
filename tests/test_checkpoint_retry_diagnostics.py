"""Retry diagnostics.

With ``--retry-prompt 'The verifier reported that your output failed…'``
a retry child started a fresh session holding only that prompt, so it had no
task to work on, and the retries were recorded as completed with nothing flagging that they did no work. The docs
did not say ``--retry-prompt`` replaces the instruction or how to include it.
Now the retry block records ``tool_calls`` and ``no_work`` (and a warning is
logged), the job summary counts ``no_work``, and ``--retry-prompt`` expands
``@instruction`` (the task instruction) and ``@verifier_feedback`` (the
failed trial's reward and the tail of its verifier output).

The final ``bench eval run`` summary did not mention retries; it now
prints one line with the counts next to the unchanged score.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from benchflow.checkpoint_retry import (
    expand_retry_prompt,
    parse_retry_policy,
    retry_summary,
    run_checkpoint_retry,
)
from tests.test_branch_isolated import IMAGES, IsoRollout
from tests.test_checkpoint_retry import _finished_trial, _result


@pytest.fixture(autouse=True)
def _reset():
    IMAGES.clear()
    IsoRollout.all = []


async def test_a_retry_that_made_no_tool_calls_is_flagged(tmp_path, caplog):
    root, _ = await _finished_trial(tmp_path)
    result = _result(0.0)
    with caplog.at_level(logging.WARNING, logger="benchflow.checkpoint_retry"):
        await run_checkpoint_retry(
            root, result, parse_retry_policy("on-failure", prompt=None)
        )
    # The scripted child reports 0 new tool calls.
    assert result.retry["tool_calls"] == 0
    assert result.retry["no_work"] is True
    saved = json.loads((root._rollout_dir / "result.json").read_text())
    assert saved["retry"]["no_work"] is True
    assert "made no tool calls" in caplog.text


def test_summary_counts_retries_that_did_no_work():
    results = [
        SimpleNamespace(retry={"status": "completed", "reward": 0.0, "no_work": True}),
        SimpleNamespace(retry={"status": "completed", "reward": 1.0, "no_work": False}),
    ]
    assert retry_summary(results)["no_work"] == 1


def test_placeholders_expand(tmp_path):
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "test-stdout.txt").write_text(
        "x" * 5000 + "\nFAILED test_value: 3 != 4\n"
    )
    text = expand_retry_prompt(
        "Task:\n@instruction\n\nVerifier said:\n@verifier_feedback\nFix it.",
        instruction="Write hello.txt",
        rollout_dir=tmp_path,
        reward=0.0,
    )
    assert "Task:\nWrite hello.txt\n" in text
    assert "reward 0.0" in text
    assert "FAILED test_value: 3 != 4" in text
    assert len(text) < 4500  # the tail, not the whole log
    # No verifier output recorded: say so instead of leaving the token.
    bare = expand_retry_prompt(
        "@verifier_feedback", instruction="i", rollout_dir=tmp_path / "nope", reward=0.0
    )
    assert "@verifier_feedback" not in bare and "no verifier output" in bare
    # No placeholder: unchanged.
    assert (
        expand_retry_prompt("plain", instruction="i", rollout_dir=tmp_path, reward=0)
        == "plain"
    )


async def test_the_retry_prompt_expands_placeholders(tmp_path):
    root, _ = await _finished_trial(tmp_path)
    policy = parse_retry_policy("on-failure", prompt="Again: @instruction")
    await run_checkpoint_retry(root, _result(0.0), policy)
    [retry_child] = IsoRollout.all[1:]
    # The scripted trial's instruction is its first resolved prompt.
    assert retry_child.events[3] == "execute:['Again: Draft first.']:['draft']"


def test_final_summary_mentions_retries(capsys):
    from benchflow.cli._shared import _report_eval_result

    result = SimpleNamespace(
        total=2,
        passed=0,
        score=0.0,
        errored=0,
        verifier_errored=0,
        mean_reward=0.4,
        task_failures=[],
        results={
            "a": SimpleNamespace(
                retry={"status": "completed", "reward": 1.0, "no_work": False}
            ),
            "b": SimpleNamespace(
                retry={"status": "completed", "reward": 0.0, "no_work": True}
            ),
        },
    )
    _report_eval_result(result)
    out = capsys.readouterr().out
    assert "Retries from checkpoints: 1/2 passed" in out
    assert "1 made no tool calls" in out
    assert "score above is the trials' own" in out
