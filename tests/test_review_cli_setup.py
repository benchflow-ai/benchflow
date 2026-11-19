"""``bench review`` takes the reviewer's effort and idle limit, and its report
records the reviewer's setup (a downstream harness patch).

The in-run review and ``bench eval score`` took ``--reviewer-reasoning-effort``
and ``--reviewer-idle-timeout``; ``bench review`` took neither, so its
reviewer ran at the agent's default effort under the 600 s idle limit, and its
report named only the agent, model, sandbox and network.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from typer.testing import CliRunner

import benchflow
from benchflow.cli.main import app
from benchflow.review.wrapper import REVIEWER_IMAGE
from tests.test_review_runtime import FakeRun, good_review, make_rollout, make_task


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "none.env"))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    task = make_task(tmp_path, with_rubric=True)
    rollout = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
    fake = FakeRun(review_payload=good_review())
    monkeypatch.setattr(benchflow, "run", fake)

    def review(*flags, reviewer=("gemini", "gemini/test-model")):
        out = tmp_path / "out"
        result = CliRunner().invoke(
            app,
            [
                "review",
                str(rollout),
                "-a",
                reviewer[0],
                "-m",
                reviewer[1],
                "--tasks-root",
                str(tmp_path / "tasks"),
                "--out-dir",
                str(out),
                *flags,
            ],
        )
        report = out / "review_report.json"
        data = json.loads(report.read_text()) if report.is_file() else None
        return result, data

    return review, fake, tmp_path


def test_the_reviewer_runs_at_the_effort_and_idle_limit_given(reviewed, monkeypatch):
    """On a reviewer that takes the level: the gemini agent declares no
    effort option, so its -e is refused (the last row below)."""
    review, fake, tmp_path = reviewed
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-not-a-key")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Grade the rollout.\n{criteria_guidance}\n")

    result, report = review(
        "--reasoning-effort",
        "HIGH",
        "--idle-timeout",
        "900",
        "--timeout-sec",
        "1200",
        "--prompt",
        str(prompt),
        reviewer=("claude-agent-acp", "anthropic/claude-opus-5-5"),
    )

    assert result.exit_code == 0, result.output
    (config,) = fake.configs
    assert config.reasoning_effort == "high"
    assert config.agent_idle_timeout == 900 and config.timeout == 1200
    reviewer = report["reviewer"]
    assert reviewer["reasoning_effort"] == "high"
    assert reviewer["timeout_sec"] == 1200
    assert reviewer["idle_timeout_sec"] == 900
    assert reviewer["image"] == REVIEWER_IMAGE
    assert reviewer["prompt_sha256"] == hashlib.sha256(prompt.read_bytes()).hexdigest()
    assert reviewer["agent"] == "claude-agent-acp"
    assert reviewer["model"] == "anthropic/claude-opus-5-5"


def test_without_the_flags_the_report_records_the_defaults(reviewed):
    review, fake, _ = reviewed

    result, report = review()

    assert result.exit_code == 0, result.output
    assert fake.configs[0].agent_idle_timeout == 600
    reviewer = report["reviewer"]
    assert reviewer["reasoning_effort"] is None
    assert (reviewer["timeout_sec"], reviewer["idle_timeout_sec"]) == (1800, 600)
    assert reviewer["prompt_sha256"] is None and reviewer["image"] == REVIEWER_IMAGE


def test_none_turns_the_idle_watchdog_off(reviewed):
    review, fake, _ = reviewed

    result, report = review("--idle-timeout", "none")

    assert result.exit_code == 0, result.output
    assert fake.configs[0].agent_idle_timeout is None  # no watchdog
    assert report["reviewer"]["idle_timeout_sec"] == 0


@pytest.mark.parametrize(
    ("flags", "named"),
    [
        (("--reasoning-effort", "loud"), "--reasoning-effort"),
        (("--idle-timeout", "soon"), "--idle-timeout"),
        (
            ("--reasoning-effort", "high"),
            "Refused before launch: reasoning effort high can't reach the gemini "
            "session",
        ),
    ],
)
def test_a_refused_setting_ends_the_command_before_any_review(reviewed, flags, named):
    review, fake, _ = reviewed

    result, report = review(*flags)

    assert result.exit_code == 2, result.output
    assert named in " ".join(result.output.split())
    assert fake.configs == [] and report is None
