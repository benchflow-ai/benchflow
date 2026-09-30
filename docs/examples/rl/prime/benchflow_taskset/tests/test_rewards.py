"""Reward mapping and drops, through Verifiers' own scoring path.

A rollout scores with ``task.score(trace)`` inside ``boundary(TaskError, "scoring")``
and records any escaping error with ``trace.record_error``; prime-rl then trains only
on traces that are ``ok``. These tests run exactly that path with a scripted session,
so they check what prime-rl will see: a reward, or a failed (dropped) trace.
"""

from __future__ import annotations

import pytest

import verifiers.v1 as vf
from verifiers.v1.errors import RolloutError, TaskError, boundary

from benchflow_taskset import BenchFlowData, BenchFlowInfraError, BenchFlowTask
from benchflow_taskset.session import BridgeError


class ScriptedSession:
    """The part of BenchFlowSession the task's hooks read."""

    def __init__(self, decision=None, *, error: Exception | None = None, **flags) -> None:
        self._decision = decision
        self._error = error
        self.decision = None
        self.verify_result = {"reward": decision.get("reward") if decision else None}
        self.stats = {"bash_calls": 3, "bash_timeouts": 1, "bash_nonzero": 0, "exec_errors": 0, "transient_errors": 0}
        self.policy_acted = flags.get("policy_acted", True)
        self.submitted = flags.get("submitted", False)
        self.infra_error = flags.get("infra_error")
        self.rollout_dir = "/jobs/r1"
        self.budget_spent = flags.get("budget_spent", False)
        self.verify_calls = 0

    async def verify(self):
        self.verify_calls += 1
        if self._error is not None:
            raise self._error
        self.decision = self._decision
        return self._decision

    def over_budget(self) -> bool:
        return self.budget_spent


def make_task() -> BenchFlowTask:
    return BenchFlowTask(BenchFlowData(name="t1", id="fam/t1", prompt="Solve.", task_dir="/tasks/t1"))


def make_trace(task: BenchFlowTask) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="BenchFlowTask", data=task.data),
    )


async def score_like_a_rollout(task: BenchFlowTask) -> vf.Trace:
    """What Rollout.close does with the task's signals: score, capture any failure."""
    trace = make_trace(task)
    try:
        async with boundary(TaskError, "scoring"):
            await task.score(trace)
    except Exception as exc:  # noqa: BLE001 - mirrors Rollout.fail
        trace.record_error(exc)
    else:
        trace.ok = True
    return trace


def decision(reward, reason="scored", *, dropped=False, detail=None):
    return {"reward": reward, "dropped": dropped, "reason": reason, "detail": detail, "flagged": False}


async def test_a_verifier_reward_is_the_training_reward() -> None:
    session = ScriptedSession(decision(1.0))
    trace = await score_like_a_rollout(make_task().bind(session))
    assert trace.ok and not trace.has_error
    assert trace.rewards["benchflow"].score == 1.0
    assert trace.reward == 1.0
    assert trace.info["benchflow"]["decision"]["reason"] == "scored"
    assert trace.metrics["benchflow_bash_calls"] == 3.0
    assert trace.metrics["benchflow_bash_timeouts"] == 1.0
    assert session.verify_calls == 1


@pytest.mark.parametrize("reason", ["timeout", "verifier_error", "run_error", "no_reward", "integrity_violation"])
async def test_policy_failures_score_zero_and_stay_in_the_batch(reason: str) -> None:
    trace = await score_like_a_rollout(make_task().bind(ScriptedSession(decision(0.0, reason))))
    assert trace.ok, "a failure the policy could have caused must train as a 0, not drop"
    assert trace.reward == 0.0
    assert trace.info["benchflow"]["decision"]["reason"] == reason


@pytest.mark.parametrize("reason", ["sandbox_start", "model_endpoint", "verifier_crash_clean_run"])
async def test_infrastructure_failures_drop_the_trace(reason: str) -> None:
    session = ScriptedSession(decision(None, reason, dropped=True, detail="boom"))
    trace = await score_like_a_rollout(make_task().bind(session))
    assert not trace.ok and trace.has_error, "prime-rl trains only on ok traces"
    assert trace.last_error.type == "BenchFlowInfraError"
    assert reason in trace.last_error.message
    assert trace.rewards["benchflow"] is None, "a dropped trace carries no reward"


async def test_the_drop_error_passes_through_verifiers_boundary_unchanged() -> None:
    # boundary() rewraps plain exceptions as TaskError; a RolloutError keeps its type,
    # which is what makes the drop reason visible in prime-rl's error counts.
    assert issubclass(BenchFlowInfraError, vf.SandboxError)
    assert issubclass(BenchFlowInfraError, RolloutError)


async def test_a_failed_bridge_drops_the_trace() -> None:
    session = ScriptedSession(error=BridgeError("the bridge exited during 'verify'"))
    trace = await score_like_a_rollout(make_task().bind(session))
    assert not trace.ok
    assert trace.last_error.type == "BenchFlowInfraError"
    assert "bridge" in trace.last_error.message


async def test_a_missing_reward_never_trains_as_a_number() -> None:
    session = ScriptedSession({"reward": None, "dropped": False, "reason": "scored", "detail": None})
    trace = await score_like_a_rollout(make_task().bind(session))
    assert not trace.ok


async def test_stops_read_the_session() -> None:
    task = make_task()
    trace = make_trace(task)
    quiet = task.bind(ScriptedSession(decision(1.0)))
    assert not await quiet.submitted(trace)
    assert not await quiet.time_budget(trace)
    assert not await quiet.bridge_failed(trace)
    assert await task.bind(ScriptedSession(decision(1.0), submitted=True)).submitted(trace)
    assert await task.bind(ScriptedSession(decision(1.0), budget_spent=True)).time_budget(trace)
    assert await task.bind(ScriptedSession(decision(1.0), infra_error="gone")).bridge_failed(trace)


async def test_stop_names_are_the_stop_conditions_prime_rl_sees() -> None:
    names = {fn.__name__ for fn in make_task().hooks("stop")}
    # None of them ends in "_timeout": prime-rl turns vf stage timeouts (is_timeout)
    # into errors, while a budget stop must score like any other stop.
    assert names == {"submitted", "time_budget", "bridge_failed"}


async def test_an_unbound_task_says_which_env_runs_it() -> None:
    task = make_task()
    with pytest.raises(RuntimeError, match="BenchFlowEnv"):
        await task.submitted(make_trace(task))


def test_binding_copies_and_keeps_the_durable_key() -> None:
    task = make_task()
    bound = task.bind(ScriptedSession(decision(1.0)))
    assert task.session is None and bound.session is not None
    assert bound.key == task.key == "fam/t1"
