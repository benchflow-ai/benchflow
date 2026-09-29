"""Agent integrations that break silently are harness failures, not zeros.

An agent whose install, login or launch broke never does anything, yet the
verifier scores the untouched workspace, usually 0, and the trial reads as a
model that failed the task. The zero-token check (``suspected_api_error``)
skips subscription runs and names no cause.

The trials below are synthetic, shaped like the result folders BenchFlow
writes: an agent whose only event after the prompt is a shim's stderr thought
reporting an exhausted credit balance, one reporting a Node.js version too old
for the agent, and three that must stay as they are: a run that thought at
length and timed out (real work, a real 0), a run that already failed with a
named error, and a run that answered with a refusal (a real answer).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import benchflow as bf
from benchflow._utils.scoring import ERROR_CATEGORIES, score_summary_fields
from benchflow.cli.main import app
from benchflow.diagnostics import DIAGNOSTIC_BY_FIELD, RolloutDiagnostics
from benchflow.integration_health import diagnose, diagnose_trial_dir

runner = CliRunner()

_PROMPT = {"type": "user_message", "text": "Summarise the notes in /app/notes.md."}

# name -> (result.json fields, trajectory events, agent logs)
_TRIALS: dict[str, tuple[dict, list[dict], dict[str, str]]] = {
    "credit-balance": (
        {
            "task_name": "notes-summary",
            "agent": "openclaw",
            "model": "claude-haiku-4-5",
            "rewards": {"reward": 0.0},
            "n_tool_calls": 0,
            "n_prompts": 1,
            "error": None,
            "timing": {"agent_execution": 7.0},
        },
        [
            # Older results did not record the prompt event.
            {
                "type": "agent_thought",
                "text": "[openclaw stderr]\n[agent] run end: isError=true "
                "error=LLM request rejected: Your credit balance is too low to "
                "access the API. Please add credits.",
            }
        ],
        {"agent/openclaw.txt": ""},
    ),
    "node-version": (
        {
            "task_name": "chart-build",
            "agent": "openclaw",
            "model": "gemini-2.5-flash",
            "rewards": {"reward": 0.0},
            "n_tool_calls": 0,
            "n_prompts": 1,
            "error": None,
            "trajectory_source": "acp",
            "timing": {"agent_execution": 30.0},
        },
        [
            _PROMPT,
            {
                "type": "agent_thought",
                "text": "[openclaw stderr]\nopenclaw: Node.js v22.16+ is required "
                "(current: v22.14.0).",
            },
        ],
        {},
    ),
    "long-thought-timeout": (
        {
            "task_name": "physics-derivation",
            "agent": "claude-agent-acp",
            "model": "claude-haiku-4-5",
            "rewards": {"reward": 0.0},
            "n_tool_calls": 0,
            "n_prompts": 1,
            "error": "Agent prompt exceeded wall-clock budget 3600s",
            "error_category": "timeout",
            "trajectory_source": "acp",
            "agent_result": {"n_output_tokens": 256000},
            "timing": {"agent_execution": 3600.0},
        },
        [
            _PROMPT,
            {"type": "agent_thought", "text": "Let me work through this step by step."},
            {"type": "agent_timeout", "reason": "wall_clock_timeout"},
        ],
        {"agent/claude_agent_acp.txt": "[session/query] resume=none\n"},
    ),
    "policy-error": (
        {
            "task_name": "hello-world",
            "agent": "claude-agent-acp",
            "model": "claude-haiku-4-5",
            "rewards": None,
            "n_tool_calls": 0,
            "n_prompts": 1,
            "error": "RuntimeError: Failed to apply no-web policy for claude-agent",
            "error_category": "other",
            "trajectory_source": None,
            "timing": {},
        },
        [],
        {},
    ),
    "refusal": (
        {
            "task_name": "doc-edit",
            "agent": "gemini",
            "model": "gemini-2.5-flash",
            "rewards": {"reward": 0.0},
            "n_tool_calls": 0,
            "n_prompts": 1,
            "error": None,
            "timing": {"agent_execution": 12.0},
        },
        [
            {
                "type": "agent_message",
                "text": "I cannot access your documents from this workspace.",
            }
        ],
        {},
    ),
}


def _write_trial(root: Path, name: str) -> Path:
    result, events, logs = _TRIALS[name]
    d = root / name
    (d / "trajectory").mkdir(parents=True)
    (d / "agent").mkdir()
    (d / "result.json").write_text(json.dumps({"rollout_name": name, **result}))
    (d / "trajectory" / "acp_trajectory.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
    )
    for rel, text in logs.items():
        (d / rel).write_text(text)
    return d


def _load(tmp_path: Path, name: str) -> tuple[Path, dict]:
    d = _write_trial(tmp_path, name)
    return d, json.loads((d / "result.json").read_text())


@pytest.mark.parametrize(
    ("name", "cause", "evidence"),
    [
        ("credit-balance", "agent_auth", "credit balance is too low"),
        ("node-version", "agent_install", "Node.js v22.16+ is required"),
    ],
)
def test_silent_failures_are_named(tmp_path, name, cause, evidence):
    trial_dir, result = _load(tmp_path, name)
    finding = diagnose_trial_dir(trial_dir, result)
    assert finding is not None
    assert finding.cause == cause
    assert evidence in finding.evidence
    assert finding.evidence_source.startswith("trajectory")
    assert finding.activity["tool_calls"] == 0
    assert finding.activity["agent_messages"] == 0


@pytest.mark.parametrize(
    "name",
    ["long-thought-timeout", "policy-error", "refusal"],
)
def test_real_work_and_named_errors_are_left_alone(tmp_path, name):
    trial_dir, result = _load(tmp_path, name)
    assert diagnose_trial_dir(trial_dir, result) is None


def _events(*extra: dict) -> list[dict]:
    return [{"type": "user_message", "text": "Do the task."}, *extra]


@pytest.mark.parametrize(
    ("events", "seconds", "cause"),
    [
        (_events(), 2.0, "empty_trajectory"),
        (_events(), 900.0, "empty_trajectory"),
        (_events({"type": "agent_timeout", "reason": "idle"}), 1.5, "immediate_exit"),
        (_events({"type": "agent_timeout", "reason": "idle"}), 600.0, "no_activity"),
        (
            _events({"type": "agent_thought", "text": "[openclaw stderr]\nboom"}),
            600.0,
            "no_activity",
        ),
        (
            _events(
                {
                    "type": "agent_thought",
                    "text": "[x stderr]\nError: 401 Unauthorized: please run /login",
                }
            ),
            30.0,
            "agent_auth",
        ),
    ],
)
def test_causes(events, seconds, cause):
    finding = diagnose(
        events,
        agent="claude-agent-acp",
        n_tool_calls=0,
        output_tokens=None,
        logs={},
        agent_seconds=seconds,
    )
    assert finding is not None and finding.cause == cause


def test_install_log_names_the_cause():
    finding = diagnose(
        _events(),
        agent="codex-acp",
        n_tool_calls=0,
        output_tokens=0,
        logs={"agent/install-stdout.txt": "npm ERR! code E404\nnpm ERR! 404 Not Found"},
        agent_seconds=5.0,
    )
    assert finding is not None
    assert finding.cause == "agent_install"
    assert finding.evidence_source == "agent/install-stdout.txt"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_tool_calls": 1},
        {"output_tokens": 12},
        {"events": _events({"type": "agent_message", "text": "Done."})},
        {"events": _events({"type": "agent_thought", "text": "Let me think."})},
        {"agent": "oracle"},
        {"agent": "nop"},
        {"events": []},  # nothing captured at all: not judged (PR #886 shape)
        # An old oracle result without its agent name.
        {"agent": "", "events": [{"type": "oracle", "text": ""}]},
    ],
)
def test_activity_controls_and_uncaptured_runs_are_not_judged(kwargs):
    args = {
        "events": _events(),
        "agent": "claude-agent-acp",
        "n_tool_calls": 0,
        "output_tokens": 0,
        "logs": {},
        "agent_seconds": 3.0,
    }
    args.update(kwargs)
    events = args.pop("events")
    assert diagnose(events, **args) is None


def test_truncated_trajectory_at_read_time(tmp_path):
    trial_dir, result = _load(tmp_path / "src", "credit-balance")
    copy = tmp_path / "t"
    shutil.copytree(trial_dir, copy)
    (copy / "trajectory" / "acp_trajectory.jsonl").write_text(
        json.dumps({"type": "user_message", "text": "Go."}) + '\n{"type": "agent_mes'
    )
    for log in (copy / "agent").glob("*"):
        log.unlink()
    finding = diagnose_trial_dir(copy, result)
    assert finding is not None and finding.cause == "truncated_trajectory"


def test_missing_trajectory_folder_is_not_judged(tmp_path):
    """A copied job without its trajectory folders must not be flagged."""
    trial_dir, result = _load(tmp_path / "src", "credit-balance")
    copy = tmp_path / "t"
    shutil.copytree(trial_dir, copy)
    shutil.rmtree(copy / "trajectory")
    for log in (copy / "agent").glob("*"):
        log.unlink()
    assert diagnose_trial_dir(copy, result) is None


def test_diagnostic_and_category_are_registered():
    assert "agent_integration" in ERROR_CATEGORIES
    cls = DIAGNOSTIC_BY_FIELD["integration_failure_info"]
    assert cls.category == "agent_integration"


# Run time: Rollout._maybe_classify_api_error


def _rollout(tmp_path: Path, events: list[dict], agent_env: dict) -> object:
    from benchflow.rollout import Rollout, RolloutConfig

    r = Rollout(
        RolloutConfig(
            task_path=Path("task"),
            agent="claude-agent-acp",
            model="claude-haiku-4-5",
        )
    )
    r._rollout_dir = tmp_path / "trial"
    (r._rollout_dir / "agent").mkdir(parents=True)
    r._executed_prompts = ["p"]
    r._agent_env = agent_env
    r._usage_metrics = {}
    r._rewards = {"reward": 0.0}
    r._trajectory = events
    r._timing = {"agent_execution": 4.0}
    return r


def test_run_time_subscription_run_that_did_nothing_is_unscored(tmp_path):
    r = _rollout(
        tmp_path,
        _events(
            {
                "type": "agent_thought",
                "text": "[openclaw stderr]\nLLM request rejected: Your credit "
                "balance is too low to access the Anthropic API.",
            }
        ),
        {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token"},
    )
    r._maybe_classify_api_error()
    assert r._rewards is None
    assert r._error.startswith("agent integration failure [agent_auth]")
    info = r._diagnostics.to_result_fields()["integration_failure_info"]
    assert info["cause"] == "agent_auth"
    assert info["reward_withheld"] == {"reward": 0.0}
    assert r._diagnostics.category_for_channel("error") == "agent_integration"


def test_run_time_api_key_run_gets_the_named_cause_not_suspected(tmp_path):
    r = _rollout(tmp_path, _events(), {"BENCHFLOW_PROVIDER_NAME": "litellm"})
    r._maybe_classify_api_error()
    assert r._rewards is None
    assert "agent integration failure [empty_trajectory]" in r._error


def test_run_time_real_work_keeps_its_reward(tmp_path):
    r = _rollout(
        tmp_path,
        _events({"type": "agent_thought", "text": "Let me work through this."}),
        {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token"},
    )
    r._usage_metrics = {"total_tokens": 300000, "n_output_tokens": 256000}
    r._maybe_classify_api_error()
    assert r._rewards == {"reward": 0.0}
    assert r._error is None


# Read time: bf.load_job, summaries, metrics, inspect


def _job(tmp_path: Path) -> Path:
    job = tmp_path / "job"
    for name in ("credit-balance", "node-version", "long-thought-timeout", "refusal"):
        _write_trial(job, name)
    return job


def test_load_job_reads_old_results_as_integration_failures(tmp_path):
    job = bf.load_job(_job(tmp_path))
    by = {t.path.name: t for t in job.trials}
    credit = by["credit-balance"]
    assert credit.execution == "integration_failed"
    assert credit.reward is None
    assert credit.assessment == "unscored"
    assert credit.integration_failure["cause"] == "agent_auth"
    assert by["long-thought-timeout"].reward == 0.0
    assert by["long-thought-timeout"].integration_failure is None
    d = job.denominators()
    assert d.integration_failures == 2
    assert d.scored == 2
    assert d.unscored == 2
    doc = job.to_json_dict()
    assert doc["denominators"]["integration_failures"] == 2
    trial_doc = credit.to_json_dict()
    assert trial_doc["integration_failure"]["cause"] == "agent_auth"
    assert trial_doc["execution"] == "integration_failed"


def test_summary_counts_integration_failures_separately():
    rows = [
        {
            "rewards": None,
            "error": "agent integration failure [agent_auth]: credit",
            "error_category": "agent_integration",
            "integration_failure_info": {"cause": "agent_auth"},
        },
        {"rewards": {"reward": 0.0}, "error": None},
    ]
    fields = score_summary_fields(rows)
    assert fields["integration_failures"] == {
        "total": 1,
        "by_cause": {"agent_auth": 1},
    }
    assert fields["failed"] == 1
    assert fields["errored"] == 1


def test_breaker_counts_permanent_integration_failures():
    from benchflow.evaluation import ApiErrorCircuitBreaker
    from benchflow.models import RolloutResult

    breaker = ApiErrorCircuitBreaker(threshold=3)
    for i in range(3):
        breaker.record(
            RolloutResult(
                task_name=f"t{i}",
                error="agent integration failure [agent_auth]: credit balance",
                error_category="agent_integration",
            )
        )
    assert breaker.tripped

    breaker = ApiErrorCircuitBreaker(threshold=3)
    for i in range(3):
        breaker.record(
            RolloutResult(
                task_name=f"t{i}",
                error="agent integration failure [no_activity]: nothing",
                error_category="agent_integration",
            )
        )
    assert not breaker.tripped


def test_inspect_shows_the_cause(tmp_path):
    job = _job(tmp_path)
    result = runner.invoke(app, ["eval", "inspect", str(job)], terminal_width=200)
    assert result.exit_code == 0, result.output
    assert "2 integration failures" in result.output
    assert "agent_auth" in result.output
    assert "agent_install" in result.output


def test_metrics_counts_them(tmp_path):
    job = _job(tmp_path)
    result = runner.invoke(app, ["eval", "metrics", str(job), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["integration_failures"] == {
        "total": 2,
        "by_cause": {"agent_auth": 1, "agent_install": 1},
    }


# bench doctor --agent-start


def test_doctor_agent_start_reports_pass_and_classified_failure(monkeypatch):
    from benchflow import agent_start

    outcomes = {
        "nop": agent_start.AgentStartOutcome(agent="nop", ok=True, seconds=1.0),
        "openclaw": agent_start.AgentStartOutcome(
            agent="openclaw",
            ok=False,
            seconds=3.0,
            error="agent exited: openclaw: Node.js v22.16+ is required",
            cause="agent_install",
            log_tail="openclaw: Node.js v22.16+ is required (current: v22.14.0)",
        ),
    }
    calls = []

    def fake_probe(agent, *, sandbox, model=None):
        calls.append((agent, sandbox))
        return outcomes[agent]

    monkeypatch.setattr(agent_start, "probe_agent_start", fake_probe)
    monkeypatch.setattr(
        "benchflow.doctor.run_doctor",
        lambda **kw: SimpleNamespace(
            checks=[], ok=True, to_dict=lambda: {"checks": []}, counts=lambda: {}
        ),
    )
    result = runner.invoke(
        app,
        [
            "doctor",
            "--sandbox",
            "daytona",
            "--agent-start",
            "nop",
            "--agent-start",
            "openclaw",
            "--json",
        ],
    )
    assert calls == [("nop", "daytona"), ("openclaw", "daytona")]
    assert result.exit_code == 1, result.output
    data = json.loads(result.output)
    starts = {c["id"]: c for c in data["checks"] if c["id"].startswith("agent_start.")}
    assert starts["agent_start.nop"]["status"] == "pass"
    assert starts["agent_start.openclaw"]["status"] == "fail"
    assert "agent_install" in starts["agent_start.openclaw"]["summary"]


def test_diagnostics_round_trip():
    from benchflow.integration_health import IntegrationFailureDiagnostic

    diagnostics = RolloutDiagnostics()
    diagnostics.set(
        IntegrationFailureDiagnostic(
            cause="agent_auth",
            evidence="credit balance is too low",
            evidence_source="trajectory agent_thought",
        )
    )
    fields = diagnostics.to_result_fields()
    back = RolloutDiagnostics.from_result_fields(fields)
    assert back.category_for_channel("error") == "agent_integration"
    block = back.to_results_jsonl_block(
        error_category="agent_integration", verifier_error_category=None
    )
    assert block["error_category"] == "agent_integration"
