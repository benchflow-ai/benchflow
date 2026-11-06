"""Trials of an agent that routes its own model calls are labelled
and never pooled with metered ones.

``model_route = "agent"`` starts no proxy, so nothing meters the agent. Its
trials say so: config.json records ``model_route: "agent"``, summary.json
records it for the job, and the usage source is ``agent-reported`` (what the
agent said over ACP) or ``none``. Neither source is trusted, so summaries keep
those tokens out of metered totals, a mixed scene keeps the agent's report
beside the metered totals instead of in them, and a resumed job sets aside
trials that ran on the other route.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.agents.registry import AGENTS, AgentConfig

OWN_ROUTE = "en191-own-route"


@pytest.fixture
def own_route_agent(monkeypatch):
    monkeypatch.setitem(
        AGENTS,
        OWN_ROUTE,
        AgentConfig(
            name=OWN_ROUTE,
            install_cmd="true",
            launch_cmd=f"{OWN_ROUTE} --acp",
            model_route="agent",
        ),
    )
    return OWN_ROUTE


def _rollout(agent: str, model: str | None = None):
    from benchflow.rollout import Rollout, RolloutConfig

    return Rollout(RolloutConfig(task_path=Path("task"), agent=agent, model=model))


def _usage(**totals):
    return SimpleNamespace(
        input_tokens=totals.get("input", 0),
        output_tokens=totals.get("output", 0),
        total_tokens=totals.get("total"),
        cached_read_tokens=0,
        cached_write_tokens=0,
        thought_tokens=0,
    )


# --- config.json and summary.json ----------------------------------------------


@pytest.mark.parametrize("label", [None, "deepseek-v4-flash"])
def test_config_json_records_the_route_of_an_own_route_agent(
    tmp_path, own_route_agent, label
):
    """config.json says the trial's model calls bypassed
    BenchFlow's proxy, and keeps the model (if any) as the label it is.
    A proxied agent's config.json is unchanged (no model_route key)."""
    from benchflow.rollout import _write_config
    from benchflow.skill_policy import SKILL_MODE_NO_SKILL, resolve_task_skill_policy

    def write(folder: Path, agent: str, model: str | None) -> dict:
        folder.mkdir()
        _write_config(
            folder,
            task_path=tmp_path / "task",
            agent=agent,
            model=model,
            environment="daytona",
            skill_policy=resolve_task_skill_policy(
                task_path=tmp_path / "task",
                skill_mode=SKILL_MODE_NO_SKILL,
                runtime_skills_dir=None,
                declared_sandbox_skills_dir=None,
            ),
            sandbox_user="agent",
            context_root=None,
            timeout=300,
            started_at=datetime(2026, 10, 28),
            agent_env={},
        )
        return json.loads((folder / "config.json").read_text())

    own = write(tmp_path / "own", own_route_agent, label)
    assert own["model_route"] == "agent"
    assert own["model"] == label
    assert "litellm_proxy" not in own
    proxied = write(tmp_path / "proxied", "claude-agent-acp", "claude-haiku-4-5")
    assert "model_route" not in proxied


# --- the usage source: agent-reported or none -----------------------------------


def test_own_route_usage_is_agent_reported_never_native(own_route_agent):
    """What the agent reports over ACP is recorded as
    agent-reported, unpriced, and kept out of the native (trusted) total."""
    from benchflow.acp.session import ACPSession

    rollout = _rollout(own_route_agent, "deepseek-v4-flash")
    session = ACPSession("own-route")
    rollout._session = session
    session.record_prompt_usage(_usage(input=10, output=4, total=14))
    rollout._collect_native_acp_usage()
    session.record_prompt_usage(_usage(input=25, output=9, total=34))
    rollout._collect_native_acp_usage()
    rollout._finalize_usage_metrics()

    usage = rollout._usage_metrics
    assert usage["usage_source"] == "agent-reported"
    assert (usage["n_input_tokens"], usage["n_output_tokens"]) == (25, 9)
    assert usage["total_tokens"] == 34
    assert usage["cost_usd"] is None and usage["price_source"] is None
    assert rollout._native_usage_metrics["usage_source"] == "unavailable"
    assert not rollout._native_usage_agents


def test_own_route_without_a_report_has_usage_source_none(own_route_agent):
    """An agent that routes its own calls and reports nothing
    is labelled none (no meter by design), not unavailable (a meter failed)."""
    rollout = _rollout(own_route_agent)
    rollout._finalize_usage_metrics()
    rollout._finalize_usage_metrics()  # idempotent, as teardown may repeat it
    assert rollout._usage_metrics["usage_source"] == "none"
    assert rollout._usage_metrics["total_tokens"] == 0


def test_mixed_scene_keeps_the_agent_report_beside_metered_totals(own_route_agent):
    """A scene whose other role went through the proxy keeps
    the metered totals; the own-route role's report sits beside them in
    usage_details, never added in, and the cost is unknown (incomplete)."""
    from benchflow.acp.session import ACPSession

    rollout = _rollout("claude-agent-acp", "claude-haiku-4-5")
    rollout._usage_metrics = {
        "n_input_tokens": 100,
        "n_output_tokens": 20,
        "n_cache_read_tokens": 0,
        "n_cache_creation_tokens": 0,
        "total_tokens": 120,
        "cost_usd": 0.01,
        "usage_source": "provider_response",
        "price_source": "litellm",
    }
    rollout._active_role = SimpleNamespace(agent=own_route_agent)
    session = ACPSession("own-route-role")
    rollout._session = session
    session.record_prompt_usage(_usage(input=7, output=3, total=10))
    rollout._collect_native_acp_usage()
    rollout._finalize_usage_metrics()
    rollout._finalize_usage_metrics()

    usage = rollout._usage_metrics
    assert usage["usage_source"] == "provider_response"
    assert usage["total_tokens"] == 120
    assert usage["cost_usd"] is None
    reported = usage["usage_details"]["agent-reported"]
    assert reported["usage_source"] == "agent-reported"
    assert reported["total_tokens"] == 10


def test_summaries_never_pool_agent_reported_with_metered_usage():
    """summary.json's token totals and telemetry coverage count
    only metered (and native) usage; agent-reported and none trials add
    nothing to them."""
    from benchflow._utils.evaluation_results import usage_summary

    def trial(source: str, tokens: int, cost: float | None) -> dict:
        return {
            "rewards": {"reward": 1.0},
            "error": None,
            "verifier_error": None,
            "agent_result": {
                "usage_source": source,
                "n_input_tokens": tokens,
                "n_output_tokens": 0,
                "n_cache_read_tokens": 0,
                "n_cache_creation_tokens": 0,
                "total_tokens": tokens,
                "cost_usd": cost,
            },
        }

    summary = usage_summary(
        {
            "metered": trial("provider_response", 15, 0.01),
            "reported": trial("agent-reported", 1000, None),
            "silent": trial("none", 0, None),
        }
    )
    assert summary["total_tokens"] == 15
    assert summary["total_cost_usd"] == 0.01
    assert summary["telemetry_coverage"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "source, endpoint", [("agent-reported", "agent"), ("none", "none")]
)
def test_result_metadata_names_the_agent_route(source, endpoint):
    """result.json's usage_tracking block validates the new
    sources and says the telemetry came from the agent (or nowhere)."""
    from benchflow.usage_tracking import (
        UsageTrackingConfig,
        is_trusted_usage_source,
        normalize_usage_source,
    )

    assert normalize_usage_source(source) == source
    assert not is_trusted_usage_source(source)
    metadata = UsageTrackingConfig().to_result_metadata(
        environment="daytona", status="unavailable", usage_source=source
    )
    assert metadata["endpoint_kind"] == endpoint
    assert metadata["usage_source"] == source


@pytest.mark.parametrize("source", ["agent-reported", "none"])
def test_missing_llm_trajectory_is_not_an_export_error(tmp_path, source):
    """An agent that routes its own calls has no proxy, so no
    llm_trajectory.jsonl; like a native login, that is not an export error
    (the trial is still not training-ready)."""
    from benchflow.trajectories.results import build_rollout_results_record

    record = build_rollout_results_record(
        tmp_path,
        task_name="t",
        rollout_name="t__1",
        agent=OWN_ROUTE,
        agent_name=OWN_ROUTE,
        model=None,
        n_tool_calls=1,
        prompts=["solve"],
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        error=None,
        verifier_error=None,
        agent_result={"usage_source": source},
    )
    assert record["error"] is None
    unmetered = build_rollout_results_record(
        tmp_path,
        task_name="t",
        rollout_name="t__1",
        agent="openhands",
        agent_name="openhands",
        model="openai/gpt-5.5",
        n_tool_calls=1,
        prompts=["solve"],
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        error=None,
        verifier_error=None,
        agent_result={"usage_source": "unavailable"},
    )
    assert unmetered["error"]["error"] == "missing_llm_trajectory"


# --- resume: the route is part of what a trial measured ---------------------------


def _job_with_trial(tmp_path: Path, config: dict) -> Path:
    trial = tmp_path / "job" / "task-a__1"
    trial.mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps(config))
    (trial / "result.json").write_text(
        json.dumps({"task_name": "task-a", "rollout_name": "task-a__1"})
    )
    return tmp_path / "job"


def test_resume_sets_aside_proxied_trials_of_a_now_own_route_agent(
    tmp_path, own_route_agent
):
    """When an agent's manifest moves it to model_route =
    "agent", its earlier metered trials are never scored with the unmetered
    ones: the resume is refused, and restart_changed sets them aside."""
    from benchflow.evaluation import (
        EvaluationConfig,
        ResumeMismatchError,
        _check_resume_mismatch,
    )

    job = _job_with_trial(tmp_path, {"agent": own_route_agent, "model": None})
    with pytest.raises(ResumeMismatchError, match="model_route='proxy'") as refused:
        _check_resume_mismatch(job, EvaluationConfig(agent=own_route_agent))
    # No setting of the run brings the old route back.
    assert "the BenchFlow version that ran it" in (refused.value.hint or "")
    assert "bench eval resume" not in (refused.value.hint or "")
    dest = _check_resume_mismatch(
        job, EvaluationConfig(agent=own_route_agent), restart_changed=True
    )
    assert dest is not None and not (job / "task-a__1").exists()
    reason = json.loads((dest / "set-aside.json").read_text())["reason"]
    assert reason == "model_route='proxy' (this run: 'agent')"


def test_resume_keeps_trials_on_the_same_route(tmp_path, own_route_agent):
    """A resume on the same route moves nothing, for an own-route
    agent and for a proxied one whose config.json has no model_route."""
    from benchflow.evaluation import EvaluationConfig, _check_resume_mismatch

    job = _job_with_trial(
        tmp_path / "own",
        {"agent": own_route_agent, "model": None, "model_route": "agent"},
    )
    assert _check_resume_mismatch(job, EvaluationConfig(agent=own_route_agent)) is None
    job = _job_with_trial(tmp_path / "proxied", {"agent": "oracle", "model": None})
    assert _check_resume_mismatch(job, EvaluationConfig(agent="oracle")) is None


def test_resume_sets_aside_own_route_trials_of_a_now_proxied_agent(tmp_path):
    """The reverse move, back to the proxy, also sets aside."""
    from benchflow.evaluation import EvaluationConfig, _check_resume_mismatch

    job = _job_with_trial(
        tmp_path, {"agent": "oracle", "model": None, "model_route": "agent"}
    )
    dest = _check_resume_mismatch(
        job, EvaluationConfig(agent="oracle"), restart_changed=True
    )
    assert dest is not None
    reason = json.loads((dest / "set-aside.json").read_text())["reason"]
    assert reason == "model_route='agent' (this run: 'proxy')"


@pytest.mark.asyncio
@pytest.mark.parametrize("own", [True, False], ids=["own-route", "proxied"])
async def test_summary_json_records_an_own_route_job(tmp_path, own_route_agent, own):
    """summary.json names the route of a job whose trials were
    never metered; a proxied job's summary is unchanged."""
    from unittest.mock import AsyncMock

    from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
    from benchflow.models import RunResult

    task_dir = tmp_path / "lone-task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )
    agent = own_route_agent if own else "oracle"
    job = Evaluation(
        tasks_dir=task_dir,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(agent=agent, retry=RetryConfig(max_retries=0)),
        job_name="run",
    )
    job._run_single_task = AsyncMock(
        return_value=RunResult(task_name="lone-task", rewards={"reward": 1.0})
    )
    await job.run()

    summary = json.loads((tmp_path / "jobs" / "run" / "summary.json").read_text())
    assert summary["agent"] == agent
    assert summary.get("model_route") == ("agent" if own else None)
