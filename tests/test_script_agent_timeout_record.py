"""result.json says a script agent ran to its time limit.

Guards the audit fix on branch ``audit-script-timeout``: a script
stopped at its time limit left ``timed_out: true`` only in the agent's
trajectory event, while result.json had ``agent_timeout_info: null``, so the
trial read as an ordinary wrong answer (seen on a live Docker run). An ACP
agent's wall-clock timeout has always set ``agent_timeout_info``.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import benchflow.rollout as module
from benchflow.rollout import Rollout, RolloutConfig


async def _run(tmp_path, monkeypatch, *, timed_out: bool):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n[agent]\ntimeout_sec = 20\n[environment]\nworkdir = "/app"\n'
    )
    (task / "instruction.md").write_text("Write hello.txt\n")
    script = tmp_path / "sleep.sh"
    script.write_text("sleep 600\n")
    rollout = Rollout(RolloutConfig(task_path=task, agent=f"script:{script}"))
    trial = tmp_path / "trial"
    trial.mkdir()
    rollout._rollout_dir = trial
    rollout._rollout_name = "script-trial"
    rollout._started_at = datetime.now()
    rollout._timeout = 20
    rollout._agent_cwd = "/app"
    rollout._env = SimpleNamespace()
    rollout._egress_denylist = None
    for name in ("setup", "start", "install_agent", "cleanup"):
        setattr(rollout, name, AsyncMock())

    async def run(env, data, label, timeout, user, cwd, variables, **kwargs):
        event = {
            "type": "script",
            "return_code": 124 if timed_out else 0,
            "timed_out": timed_out,
        }
        return [event], label

    async def verify():
        rollout._rewards = {"reward": 0.0}

    rollout.verify = verify
    monkeypatch.setattr(
        rollout._planes, "enforce_agent_egress_firewall", AsyncMock(), raising=False
    )
    monkeypatch.setattr(module, "_run_script", run, raising=False)
    result = await rollout.run()
    return result, json.loads((trial / "result.json").read_text())


@pytest.mark.asyncio
async def test_a_script_stopped_at_its_time_limit_is_recorded(tmp_path, monkeypatch):
    result, saved = await _run(tmp_path, monkeypatch, timed_out=True)

    info = saved["agent_timeout_info"]
    assert info["reason"] == "wall_clock_timeout"
    assert info["timeout_sec"] == 20.0
    assert info["terminal_trajectory_complete"] is True
    # The trial is still graded on what the script left, with no error text.
    assert result.error is None and saved["error"] is None
    assert result.rewards == {"reward": 0.0}
    assert saved["partial_trajectory"] is False


@pytest.mark.asyncio
async def test_a_timed_out_script_is_in_the_timeout_category(tmp_path, monkeypatch):
    """Guards the follow-up to PR #125 on branch
    ``audit-script-timeout-category``: the timeout was recorded in
    ``agent_timeout_info`` with ``error_category: null``, so ``--fail-on
    timeout``, summary.json's ``error_categories`` and the job's "hit agent
    wall-clock timeout" warning, which all read the category, did not see it."""
    from benchflow._utils.scoring import score_summary_fields
    from benchflow.diagnostics import AgentPromptTimeoutDiagnostic, summary_warning
    from benchflow.evaluation import RetryConfig

    result, saved = await _run(tmp_path, monkeypatch, timed_out=True)

    # --fail-on timeout counts results whose error_category is "timeout".
    assert result.error_category == "timeout"
    assert saved["error_category"] == "timeout"
    summary = score_summary_fields([saved])
    assert summary["error_categories"] == {"timeout": 1}
    # Still a scored trial that failed, not an errored one, and not retried.
    assert (summary["failed"], summary["errored"]) == (1, 0)
    assert not RetryConfig().should_retry(result.error, category=result.error_category)
    # The job's per-diagnostic warning is driven by that count.
    assert AgentPromptTimeoutDiagnostic.category == "timeout"
    assert "hit agent wall-clock timeout" in summary_warning(
        AgentPromptTimeoutDiagnostic, summary["error_categories"]["timeout"], 1
    )


@pytest.mark.asyncio
async def test_a_script_that_finished_has_no_timeout_record(tmp_path, monkeypatch):
    result, saved = await _run(tmp_path, monkeypatch, timed_out=False)

    assert saved["agent_timeout_info"] is None
    assert result.error_category is None and saved["error_category"] is None
