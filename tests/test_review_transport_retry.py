"""A reviewer that loses its transport before a verdict gets one fresh retry.

Guards #1144: a Daytona PTY reset (``close_code=1006``) during the review
stage ended the trial with ``reviewer did not produce a readable
review-result.json`` and no retry, although the verifier output was already
on disk and a second reviewer could produce the verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchflow
from benchflow.review.config import load_rubric
from benchflow.review.options import ReviewerConfig
from benchflow.review.runner import run_review
from tests.test_review_runtime import (
    WEIGHTED_RUBRIC,
    FakeRun,
    good_weighted_review,
    make_rollout,
    make_task,
)

_RESET = (
    "Agent connection lost: PTY readline error: websocket closed "
    "(close_code=1006, socket error: ClientConnectionResetError("
    "'Cannot write to closing transport'))"
)


class _FlakyTransport:
    """The first ``failures`` reviewer runs die on the transport, then succeed."""

    def __init__(self, failures: int, *, raw_message: str = _RESET):
        self.failures = failures
        self.raw_message = raw_message
        self.ok = FakeRun(review_payload=good_weighted_review())
        self.calls = 0

    async def __call__(self, config):
        self.calls += 1
        if self.calls > self.failures:
            return await self.ok(config)
        leaf = Path(config.jobs_dir) / "job" / "wrapper__0000"
        leaf.mkdir(parents=True)
        (leaf / "config.json").write_text("{}")
        (leaf / "result.json").write_text(
            json.dumps(
                {
                    "rewards": None,
                    "error": self.raw_message,
                    "error_category": "pipe_closed",
                    "transport_error_info": {
                        "reason": "transport_closed",
                        "raw_message": self.raw_message,
                        "transport_diagnosis": "pty_error",
                    },
                }
            )
        )

        class _Result:
            error = self.raw_message

        return _Result()


async def _review(tmp_path: Path):
    task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
    source = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
    rubric_path = task / "verifier/rubric.json"
    return await run_review(
        source,
        task,
        load_rubric(rubric_path),
        rubric_path,
        ReviewerConfig(agent="codex-acp", model="azure/gpt-5.6-sol"),
        tmp_path / "review-output",
        deterministic_pass=True,
    )


@pytest.mark.asyncio
async def test_transport_reset_is_retried_once_with_a_fresh_reviewer(
    tmp_path, monkeypatch
):
    flaky = _FlakyTransport(failures=1)
    monkeypatch.setattr(benchflow, "run", flaky)
    trial = await _review(tmp_path)
    assert flaky.calls == 2
    assert trial.review_valid and trial.error is None
    assert any("transport" in note and "retried" in note for note in trial.notes)
    # The verdict comes from the second reviewer's own leaf.
    assert Path(trial.reviewer_rollout, "verifier").is_dir()


@pytest.mark.asyncio
async def test_a_second_transport_reset_is_reported_not_retried_again(
    tmp_path, monkeypatch
):
    flaky = _FlakyTransport(failures=5)
    monkeypatch.setattr(benchflow, "run", flaky)
    trial = await _review(tmp_path)
    assert flaky.calls == 2
    assert not trial.review_valid
    assert trial.error
    assert [n for n in trial.notes if "lost its transport" in n][-1].endswith(")")


@pytest.mark.asyncio
async def test_a_readline_timeout_is_not_retried(tmp_path, monkeypatch):
    """A reviewer silent past its budget would only be silent again."""
    flaky = _FlakyTransport(
        failures=5, raw_message="Agent connection lost: PTY readline timeout (900s)"
    )
    monkeypatch.setattr(benchflow, "run", flaky)
    trial = await _review(tmp_path)
    assert flaky.calls == 1
    assert not trial.review_valid


@pytest.mark.asyncio
async def test_an_ordinary_reviewer_error_is_not_retried(tmp_path, monkeypatch):
    fake = FakeRun(review_payload=None, error="reviewer timed out")
    monkeypatch.setattr(benchflow, "run", fake)
    trial = await _review(tmp_path)
    assert len(fake.configs) == 1
    assert not trial.review_valid
