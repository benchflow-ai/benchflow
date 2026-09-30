"""``bf.rollout_group``: the trainer-facing rollout call.

PostTrain Arena drove every GRPO rollout by shelling out to ``bench eval
run`` with about 20 flags, then re-derived groups, retries, masking and
token data from the job folder. These tests drive ``RolloutGroup`` with a
fake rollout runner that behaves like a real rollout where it matters here:
its model calls go through the policy relay to a mock vLLM server, and it
writes ``result.json``, the gateway's ``llm_trajectory.jsonl`` (built by the
gateway's own import path) and the verifier's CTRF report.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from benchflow.models import RolloutResult
from benchflow.providers.litellm_logging import trajectory_from_litellm_callback_log
from benchflow.rollout import RolloutConfig
from benchflow.training.policy import Policy
from benchflow.training.rollouts import (
    RolloutGroup,
    RolloutGroupError,
    StartupTimeouts,
    attribute,
    pass_fraction,
    read_tests,
)
from tests.fixtures.mock_token_logprobs_server import start_server

TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]
PLAN_CAPTURE = {
    "enabled": True,
    "wire": "openai-chat",
    "provider": "vllm",
    "request": "chat",
    "logprobs": True,
    "token_ids": True,
}


@pytest.fixture
def server():
    srv = start_server(flavor="vllm")
    yield srv
    srv.shutdown()


@pytest.fixture
def task(tmp_path) -> Path:
    path = tmp_path / "tasks" / "sql-000004"
    path.mkdir(parents=True)
    (path / "instruction.md").write_text("answer\n")
    return path


def _member(name: str) -> tuple[int, int]:
    tail = name.rsplit("-r", 1)[1]
    index, attempt = tail.split("-a")
    return int(index), int(attempt)


def fake_runner(plans: dict[Any, dict[str, Any]], *, policy: Policy | None = None):
    """A rollout runner: calls go through the relay; files look like a real rollout's."""

    async def run(config: RolloutConfig) -> RolloutResult:
        name = config.rollout_name or ""
        index, attempt = _member(name)
        plan = plans.get((index, attempt)) or plans.get(index) or plans["default"]
        if plan.get("sleep"):
            await asyncio.sleep(plan["sleep"])
        rollout_dir = Path(config.jobs_dir) / (config.job_name or "job") / name
        (rollout_dir / "trajectory").mkdir(parents=True)
        env = config.agent_env or {}
        records = []
        for call in plan.get(
            "calls", [{"messages": [{"role": "user", "content": "q"}]}]
        ):
            if call.get("bump_version") and policy is not None:
                policy.version = call["bump_version"]
            body = {
                "model": "mock-policy",
                "messages": call["messages"],
                "logprobs": True,
                "return_token_ids": True,
            }
            if call.get("tools", True):
                body["tools"] = TOOLS
            async with httpx.AsyncClient(timeout=30) as client:
                answer = await client.post(
                    env["BENCHFLOW_PROVIDER_BASE_URL"] + "/chat/completions",
                    json=body,
                    headers={
                        "Authorization": "Bearer " + env["BENCHFLOW_PROVIDER_API_KEY"]
                    },
                )
            records.append(
                {
                    "event": "success" if answer.status_code == 200 else "failure",
                    "token_capture": PLAN_CAPTURE,
                    "request": {
                        "method": "POST",
                        "path": "/v1/chat/completions",
                        "body": body,
                    },
                    "response": answer.json(),
                    "call_type": "acompletion",
                    "start_time": "2026-09-30T00:00:00",
                    "end_time": "2026-09-30T00:00:01",
                    "error": None
                    if answer.status_code == 200
                    else {"message": answer.text},
                }
            )
        trajectory = trajectory_from_litellm_callback_log(
            "\n".join(json.dumps(r) for r in records),
            session_id=name,
            agent_name="opencode",
        )
        (rollout_dir / "trajectory" / "llm_trajectory.jsonl").write_text(
            "".join(
                json.dumps(e.model_dump(mode="json"), default=str) + "\n"
                for e in trajectory.exchanges
            )
        )
        if plan.get("tests"):
            (rollout_dir / "verifier").mkdir()
            (rollout_dir / "verifier" / "ctrf.json").write_text(
                json.dumps(
                    {
                        "results": {
                            "tests": [
                                {"name": n, "status": s, "duration": 3}
                                for n, s in plan["tests"]
                            ]
                        }
                    }
                )
            )
        if plan.get("claim"):
            (rollout_dir / "integrity").mkdir()
            (rollout_dir / "integrity" / "claim_verdict.json").write_text(
                json.dumps(plan["claim"])
            )
        result = {
            "task_name": Path(config.task_path).name,
            "rollout_name": name,
            "rewards": plan.get("rewards"),
            "error": plan.get("error"),
            "error_category": plan.get("error_category"),
            "verifier_error": plan.get("verifier_error"),
            **plan.get("extra", {}),
        }
        (rollout_dir / "result.json").write_text(json.dumps(result))
        (rollout_dir / "timing.json").write_text(
            json.dumps({"environment_setup": 1.5, "agent_setup": 2.5, "total": 9})
        )
        return RolloutResult(
            task_name=result["task_name"],
            rollout_name=name,
            rewards=plan.get("rewards"),
            error=plan.get("error"),
            error_category=plan.get("error_category"),
            verifier_error=plan.get("verifier_error"),
            n_tool_calls=plan.get("tool_calls", 1),
            rollout_dir=rollout_dir,
        )

    return run


def group(config, policy, plans, **overrides) -> RolloutGroup:
    options: dict[str, Any] = {
        "n": 2,
        "policy": policy,
        "attempts": 2,
        "on_failure": "mask",
        "concurrency": None,
        "group_id": "step1-sql",
        "advantage": "grpo",
        "drop_zero_variance": False,
        "reward": "verifier",
        "trainable_kinds": ("agent", "subagent", "chat"),
        "integrity": "auto",
        "startup_timeouts": None,
        "sandbox_lifetime": None,
    }
    options.update(overrides)
    return RolloutGroup(config, run=fake_runner(plans, policy=policy), **options)


@pytest.fixture
async def policy(server, monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHFLOW_LEASE_DIR", str(tmp_path / "leases"))
    p = Policy("vllm/mock-policy", base_url=f"{server.base_url}/v1", version=1)
    async with p:
        yield p


@pytest.fixture
def config(task, tmp_path) -> RolloutConfig:
    return RolloutConfig(
        task_path=task,
        agent="opencode",
        environment="docker",
        jobs_dir=tmp_path / "jobs",
    )


async def test_results_arrive_typed_with_tokens_versions_and_attestation(
    config, policy
):
    plans = {"default": {"rewards": {"reward": 1.0}}, 1: {"rewards": {"reward": 0.0}}}
    g = group(config, policy, plans)
    seen = [r async for r in g]
    assert sorted(r.index for r in seen) == [0, 1]
    result = await g.wait()
    first, second = result.rollouts
    assert first.reward == 1.0 and first.passed and first.outcome == "passed"
    assert second.reward == 0.0 and second.outcome == "failed"
    for member in result.rollouts:
        assert member.attribution == "score" and member.attribution_reason == "scored"
        assert member.policy_version == 1 and member.policy_versions == (1,)
        assert member.tokens["attestation"]["status"] == "attested"
        [segment] = member.segments
        assert segment.kind == "agent" and segment.trainable
        assert segment.completion_ids == [1000 + ord(c) for c in "ok!"]
        assert segment.action_mask == [1, 1, 1]
        assert segment.logprobs == [-0.125, -0.25, -0.375]
        assert segment.policy_versions == (1,)
        assert segment.verify()
        relay_log = member.rollout_dir / "trajectory" / "policy_relay.jsonl"
        assert relay_log.is_file()
    # GRPO over the two scored members: +0.707 / -0.707.
    assert first.advantage == pytest.approx(0.7071, abs=1e-3)
    assert second.advantage == pytest.approx(-0.7071, abs=1e-3)
    record = json.loads(
        (Path(config.jobs_dir) / g.job_name / "groups" / "step1-sql.json").read_text()
    )
    assert record["scored"] == 2 and record["dropped"] is None


async def test_helper_calls_are_excluded_and_versions_follow_the_policy(config, policy):
    plans = {
        "default": {
            "rewards": {"reward": 1.0},
            "calls": [
                {"messages": [{"role": "user", "content": "q"}]},
                {"messages": [{"role": "user", "content": "title?"}], "tools": False},
                {"messages": [{"role": "user", "content": "q2"}], "bump_version": 2},
            ],
        }
    }
    g = group(config, policy, plans, n=1)
    result = await g.wait()
    [member] = result.rollouts
    assert [s.kind for s in member.excluded_segments] == ["helper"]
    assert [s.policy_versions for s in member.segments] == [(1,), (2,)]
    assert member.policy_versions == (1, 2)
    assert member.policy_version == 1


async def test_infrastructure_failure_is_retried_then_scored(config, policy):
    plans = {
        "default": {"rewards": {"reward": 1.0}},
        (0, 1): {
            "error": "Sandbox startup failed: Sandbox creation failed after 1 attempt",
            "error_category": "sandbox_setup",
            "calls": [],
            "tool_calls": 0,
        },
    }
    g = group(config, policy, plans)
    result = await g.wait()
    member = result.rollouts[0]
    assert member.attempt == 2 and member.reward == 1.0
    [failed] = member.attempts
    assert failed.attempt == 1 and failed.attribution_reason == "sandbox_start"
    assert failed.rollout.endswith("-r00-a1") and member.rollout.endswith("-r00-a2")


@pytest.mark.parametrize(
    ("on_failure", "reward", "outcome", "reason"),
    [
        ("mask", None, "masked", "sandbox_start"),
        ("zero", 0.0, "failed", "failure_policy_zero"),
    ],
)
async def test_failure_policy_after_the_last_attempt(
    config, policy, on_failure, reward, outcome, reason
):
    plans = {
        "default": {
            "error": "Sandbox startup failed: boom",
            "error_category": "sandbox_setup",
            "calls": [],
            "tool_calls": 0,
        }
    }
    g = group(config, policy, plans, n=1, on_failure=on_failure)
    [member] = (await g.wait()).rollouts
    assert member.reward == reward and member.outcome == outcome
    assert member.attribution_reason == reason
    assert len(member.attempts) == 1


async def test_on_failure_raise_stops_the_group(config, policy):
    plans = {
        "default": {"rewards": {"reward": 1.0}, "sleep": 5},
        0: {
            "error": "Sandbox startup failed: boom",
            "error_category": "sandbox_setup",
            "calls": [],
            "tool_calls": 0,
        },
    }
    g = group(config, policy, plans, attempts=1, on_failure="raise")
    with pytest.raises(RolloutGroupError, match="sandbox_start"):
        async for _ in g:
            pass
    assert all(t.done() for t in g._tasks)


async def test_scored_timeout_is_not_an_infrastructure_failure(config, policy):
    plans = {
        "default": {
            "rewards": {"reward": 0.0},
            "error": "Agent timed out after 900s",
            "error_category": "timeout",
        }
    }
    [member] = (await group(config, policy, plans, n=1).wait()).rollouts
    assert member.attribution == "score" and member.reward == 0.0
    assert member.failure == "policy" and member.error_category == "timeout"
    assert member.attempts == ()


async def test_endpoint_failure_is_masked(config, policy, server):
    server.fail_next.extend([503] * 10)
    plans = {
        "default": {
            "rewards": {"reward": 0.0},
            "error": "agent gave up",
            "error_category": "other",
        }
    }
    [member] = (await group(config, policy, plans, n=1, attempts=1).wait()).rollouts
    assert (
        member.attribution == "mask" and member.attribution_reason == "model_endpoint"
    )
    assert member.reward is None


async def test_partial_credit_from_the_verifier_tests(config, policy):
    plans = {
        "default": {
            "rewards": {"reward": 0.0},
            "tests": [
                ("t1", "passed"),
                ("t2", "passed"),
                ("t3", "failed"),
                ("t4", "skipped"),
            ],
        }
    }
    [member] = (await group(config, policy, plans, n=1, reward="tests").wait()).rollouts
    assert member.reward == pytest.approx(2 / 3)
    assert member.reward_source == "tests" and member.verifier_reward == 0.0
    assert [t.name for t in member.tests] == ["t1", "t2", "t3", "t4"]


async def test_zero_variance_groups_can_be_dropped(config, policy):
    plans = {"default": {"rewards": {"reward": 1.0}}}
    result = await group(config, policy, plans, n=3, drop_zero_variance=True).wait()
    assert result.zero_variance and result.dropped == "zero_variance"
    assert all(r.advantage is None for r in result.rollouts)
    assert result.trainable() == []
    kept = await group(config, policy, plans, n=2, group_id="step2").wait()
    assert kept.zero_variance and kept.dropped is None
    assert [r.advantage for r in kept.rollouts] == [0.0, 0.0]


async def test_integrity_verdicts_make_a_flagged_zero(config, policy):
    plans = {
        "default": {"rewards": {"reward": 1.0}},
        1: {
            "rewards": {"reward": 1.0},
            "claim": {"core_verdict": "AgentViolation", "reason": "read /tests"},
        },
    }
    result = await group(config, policy, plans).wait()
    clean, hacked = result.rollouts
    assert not clean.flagged and clean.reward == 1.0
    assert hacked.flagged and hacked.reward == 0.0 and hacked.verifier_reward == 1.0
    assert hacked.attribution == "score"
    assert hacked.attribution_reason == "integrity_violation"
    assert hacked.integrity == {"exploited": True, "reason": "read /tests"}

    def audit(rollout):
        return {"exploited": rollout.index == 0, "reason": "custom check"}

    custom = await group(
        config, policy, plans, integrity=audit, group_id="custom"
    ).wait()
    assert [r.flagged for r in custom.rollouts] == [True, False]


async def test_startup_failure_is_masked_and_reported(config, policy):
    plans = {
        "default": {
            "error": "TransportClosedError: ACP initialize timed out after 5s before the first prompt",
            "extra": {
                "transport_error_info": {
                    "transport_diagnosis": "acp_initialize_timeout"
                }
            },
            "calls": [],
            "tool_calls": 0,
        }
    }
    g = group(
        config,
        policy,
        plans,
        n=1,
        startup_timeouts=StartupTimeouts(acp_handshake_sec=5, gateway_sec=40),
    )
    [member] = (await g.wait()).rollouts
    assert member.attribution == "mask" and member.attribution_reason == "agent_setup"
    assert member.error_category == "acp_error"
    assert member.startup["failed_phase"] == "acp_initialize"
    assert member.startup["timeouts"]["acp_handshake_sec"] == 5
    assert member.startup["timeouts"]["gateway_sec"] == 40
    assert member.startup["phases_sec"] == {
        "environment_setup": 1.5,
        "agent_setup": 2.5,
    }


async def test_cancel_stops_running_members(config, policy):
    plans = {"default": {"rewards": {"reward": 1.0}, "sleep": 30}}
    g = group(config, policy, plans, n=3)
    await g.start()
    await asyncio.sleep(0.2)
    await g.cancel()
    result = await g.wait()
    assert result.cancelled == 3
    assert all(r.outcome == "cancelled" and r.reward is None for r in result.rollouts)


async def test_group_arguments_are_checked(config, policy, task):
    with pytest.raises(ValueError, match="attempts"):
        group(config, policy, {}, attempts=0)
    with pytest.raises(ValueError, match="on_failure"):
        group(config, policy, {}, on_failure="retry")
    with pytest.raises(ValueError, match="policy="):
        group(config, None, {})
    other = RolloutConfig(task_path=task, agent="opencode", model="vllm/other")
    with pytest.raises(ValueError, match=r"leave RolloutConfig\.model unset"):
        group(other, policy, {})


def test_attribution_rules():
    policy_failures = {
        "timeout, no reward": (
            {"error": "Agent timed out", "error_category": "timeout"},
            "timeout",
        ),
        "crash after acting": (
            {"error": "agent closed stdout", "error_category": "pipe_closed"},
            "run_error",
        ),
        "transport lost after acting": (
            {"error": "connection lost", "error_category": "infra_failure"},
            "run_error",
        ),
        "verifier failed": (
            {"verifier_error": "verifier crashed: exit 1"},
            "verifier_error",
        ),
        "context overflow": (
            {
                "error": "acp error: provider rejected request (HTTP 400)",
                "error_category": "provider_rejected",
            },
            "run_error",
        ),
        "nothing": ({}, "no_reward"),
    }
    for label, (result, reason) in policy_failures.items():
        assert attribute(result, policy_acted=True)[:2] == ("score", reason), label
    masked = {
        "sandbox": (
            {"error_category": "sandbox_setup", "error": "x"},
            True,
            "sandbox_start",
        ),
        "install": (
            {"error_category": "install_failure", "error": "x"},
            True,
            "agent_setup",
        ),
        "api error": (
            {"error_category": "api_error", "error": "x"},
            True,
            "model_endpoint",
        ),
        "503": (
            {
                "error_category": "infra_failure",
                "error": "acp error: provider unavailable (HTTP 503)",
            },
            True,
            "model_endpoint",
        ),
        "dep install": (
            {"verifier_error": "verifier crashed: dependency install failed"},
            True,
            "verifier_infra",
        ),
        "clean run crash": (
            {"verifier_error": "verifier crashed: boom"},
            False,
            "verifier_crash_clean_run",
        ),
        "acp before acting": (
            {"error": "acp error: -32603", "error_category": "acp_error"},
            False,
            "infrastructure",
        ),
    }
    for label, (result, acted, reason) in masked.items():
        assert attribute(result, policy_acted=acted)[:2] == ("mask", reason), label
    scored = {
        "rewards": {"reward": 0.0},
        "error": "Agent timed out",
        "error_category": "timeout",
    }
    assert attribute(scored, policy_acted=True) == ("score", "scored", "policy")
    assert (
        attribute(
            {"rewards": {"reward": 0.25}}, policy_acted=True, endpoint_failed=True
        )[1]
        == "model_endpoint"
    )
    assert (
        attribute(
            {"rewards": {"reward": 1.0}}, policy_acted=True, endpoint_failed=True
        )[1]
        == "scored"
    )


def test_ctrf_reading(tmp_path):
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier" / "ctrf.json").write_text(
        json.dumps(
            {
                "results": {
                    "tests": [
                        {"name": "a", "status": "passed"},
                        {
                            "name": "b",
                            "status": "failed",
                            "message": "E  assert 1 == 2",
                        },
                    ]
                }
            }
        )
    )
    tests = read_tests(tmp_path)
    assert [(t.name, t.status) for t in tests] == [("a", "passed"), ("b", "failed")]
    assert pass_fraction(tests) == 0.5
    assert read_tests(tmp_path / "missing") == ()
    assert pass_fraction(()) is None
