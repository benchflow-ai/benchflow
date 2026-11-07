"""Per-trial token and turn budgets, enforced while the trial runs.

Before 0.10.0rc2 BenchFlow capped a trial by wall-clock and idle time only;
``budget.py`` caps a whole job, its tokens checked between rollouts. A trial
budget (``--token-budget``, ``--turn-budget``, a run config's or a task's
``[agent]`` ``token_budget`` and ``turn_budget``) is enforced by the model
proxy, which sees every call of the trial, subagents' included: it refuses
the next call once a measure is spent, records why, and the rollout stops the
agent and has the verifier grade what it left, as after a timeout.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.scoring import TRIAL_BUDGET, classify_error, score_summary_fields
from benchflow.eval_plan import EvalCreateRequest, EvalPlanError, build_eval_plan
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers.litellm_logging import (
    callback_module_source,
    trajectory_from_litellm_callback_log,
)
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.rollout._results import _build_rollout_result
from benchflow.run_card import check_card, trial_budget_text
from benchflow.run_checks import RunChecks, after_run_lines, pool
from benchflow.scoring import score, trial_record
from benchflow.task.config import TaskConfig
from benchflow.trajectories.model_calls import subagent_usage
from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    _exchange_token_usage,
)
from benchflow.trial_budget import (
    TrialBudget,
    TrialBudgetSpentError,
    TrialBudgetUnenforceableError,
    budget_refusal,
    effective_budget,
    stop_text,
    task_budget,
)

# -- the budget itself -------------------------------------------------------


def test_a_budget_takes_positive_whole_numbers_and_says_itself():
    budget = TrialBudget(tokens=200_000, turns="50")
    assert (budget.tokens, budget.turns) == (200_000, 50)
    assert budget.text() == "200,000 tokens and 50 turns"
    assert TrialBudget(turns=1).text() == "1 turn"
    for bad in (0, -3, 1.5, True, "ten"):
        with pytest.raises(ValueError):
            TrialBudget(tokens=bad)
    with pytest.raises(ValueError, match="needs a token budget"):
        TrialBudget()
    assert TrialBudget.coerce({"tokens": None, "turns": None}) is None
    assert TrialBudget.coerce({"turns": 3}) == TrialBudget(turns=3)
    with pytest.raises(ValueError, match="unknown trial budget key"):
        TrialBudget.coerce({"steps": 3})


def test_a_run_measure_replaces_the_tasks_and_each_says_where_it_came_from():
    run = TrialBudget(turns=20)
    task = TrialBudget(tokens=1_000, turns=50)
    assert effective_budget(run, task) == (
        TrialBudget(tokens=1_000, turns=20),
        {"tokens": "task", "turns": "run"},
    )
    assert effective_budget(None, None) == (None, {})
    assert effective_budget(None, task)[1] == {"tokens": "task", "turns": "task"}


def test_the_proxy_reads_the_budget_and_what_earlier_proxies_counted():
    env = TrialBudget(tokens=900, turns=4).proxy_env({"tokens": 120, "turns": 0})
    assert env == {
        "BENCHFLOW_TRIAL_TOKEN_BUDGET": "900",
        "BENCHFLOW_TRIAL_TURN_BUDGET": "4",
        "BENCHFLOW_TRIAL_BUDGET_USED": '{"tokens":120}',
    }
    assert "BENCHFLOW_TRIAL_BUDGET_USED" not in TrialBudget(turns=4).proxy_env()


def test_a_task_sets_its_own_budget_in_its_agent_section():
    config = TaskConfig.model_validate(
        {"agent": {"timeout_sec": 600, "token_budget": 500_000, "turn_budget": 40}}
    )
    assert task_budget(config.agent) == TrialBudget(tokens=500_000, turns=40)
    assert task_budget(TaskConfig.model_validate({}).agent) is None
    for bad in (0, -1, 2.5, True, "40"):
        with pytest.raises(ValueError):
            TaskConfig.model_validate({"agent": {"turn_budget": bad}})


def test_the_stop_reads_as_its_own_error_category_and_is_never_retried():
    record = {
        "budget": "turns",
        "limits": {"turns": 3},
        "used": {"tokens": 4_200, "turns": 3},
    }
    text = stop_text(record)
    assert text.startswith("trial budget reached: the turn budget of 3 is spent")
    assert "4,200 tokens, 3 turns used" in text
    assert classify_error(text) == TRIAL_BUDGET
    assert classify_error(f"TrialBudgetSpentError: {text}") == TRIAL_BUDGET
    # Quoted inside another error, the words claim nothing.
    assert classify_error(f"ACP error -32603: {text}") != TRIAL_BUDGET
    retry = RetryConfig()
    assert not retry.should_retry(text, category=TRIAL_BUDGET)
    assert not retry.reruns_on_resume(text, category=TRIAL_BUDGET)
    assert not RetryConfig(exclude_categories=set()).should_retry(
        text, category=TRIAL_BUDGET
    )


# -- the proxy's enforcement (its callback) ---------------------------------


def _callback(monkeypatch, tmp_path, **env):
    log = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    monkeypatch.delenv("BENCHFLOW_LITELLM_SERVED_MODELS", raising=False)
    for name in (
        "BENCHFLOW_TRIAL_TOKEN_BUDGET",
        "BENCHFLOW_TRIAL_TURN_BUDGET",
        "BENCHFLOW_TRIAL_BUDGET_USED",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    namespace: dict = {}
    exec(callback_module_source(), namespace)
    return namespace, namespace["proxy_handler_instance"], log


def _refused(log: Path) -> list[dict]:
    if not log.exists():
        return []
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    return [r for r in rows if r.get("event") == "refused"]


@pytest.mark.asyncio
async def test_the_proxy_refuses_the_call_after_the_turn_budget(monkeypatch, tmp_path):
    """Two turns allowed: the third call is refused with a 400 before any
    provider sees it, and the refusal record says why and what was used."""
    ns, hook, log = _callback(monkeypatch, tmp_path, BENCHFLOW_TRIAL_TURN_BUDGET="2")
    data = {"model": "m", "messages": [{"role": "user", "content": "go"}]}
    for _ in range(2):
        await hook.async_pre_call_hook(None, None, dict(data), "acompletion")
    with pytest.raises(ns["TrialBudgetSpent"]) as spent:
        await hook.async_pre_call_hook(None, None, dict(data), "acompletion")
    assert spent.value.status_code == 400
    assert spent.value.message.startswith("trial_budget_spent: ")
    [record] = _refused(log)
    assert record["rule"] == "trial-budget"
    assert record["budget"] == "turns"
    assert record["limits"] == {"turns": 2}
    assert record["used"] == {"tokens": 0, "turns": 2}
    assert "go" not in log.read_text()


@pytest.mark.asyncio
async def test_the_token_budget_counts_each_answered_call(monkeypatch, tmp_path):
    """Tokens are known when a call ends: the call that spends the budget
    completes, the next is refused. A token count asks for no answer and is
    no turn."""
    ns, hook, log = _callback(
        monkeypatch, tmp_path, BENCHFLOW_TRIAL_TOKEN_BUDGET="1500"
    )
    data = {"model": "m", "messages": [{"role": "user", "content": "go"}]}
    count = {
        **data,
        "proxy_server_request": {"url": "http://x/v1/messages/count_tokens"},
    }
    await hook.async_pre_call_hook(None, None, dict(count), "anthropic_messages")
    await hook.async_pre_call_hook(None, None, dict(data), "anthropic_messages")
    answer = SimpleNamespace(
        model_dump=lambda **_: {
            "usage": {
                "input_tokens": 900,
                "cache_read_input_tokens": 500,
                "output_tokens": 150,
            }
        }
    )
    kwargs = {"model": "m", "litellm_params": {}, "messages": data["messages"]}
    now = datetime.now()
    await hook.async_log_success_event(kwargs, answer, now, now)
    with pytest.raises(ns["TrialBudgetSpent"]):
        await hook.async_pre_call_hook(None, None, dict(data), "anthropic_messages")
    [record] = _refused(log)
    assert record["budget"] == "tokens"
    assert record["used"] == {"tokens": 1550, "turns": 1}


@pytest.mark.asyncio
async def test_a_proxy_counts_on_from_the_trials_earlier_proxies(monkeypatch, tmp_path):
    ns, hook, log = _callback(
        monkeypatch,
        tmp_path,
        BENCHFLOW_TRIAL_TURN_BUDGET="3",
        BENCHFLOW_TRIAL_BUDGET_USED='{"turns": 3, "tokens": 10}',
    )
    with pytest.raises(ns["TrialBudgetSpent"]):
        await hook.async_pre_call_hook(None, None, {"model": "m"}, "acompletion")
    assert _refused(log)[0]["used"] == {"tokens": 10, "turns": 3}


def test_a_budget_the_proxy_cannot_read_stops_it_from_starting(monkeypatch, tmp_path):
    """Never silently dropped: an unreadable budget fails the proxy's start."""
    with pytest.raises(RuntimeError, match="BENCHFLOW_TRIAL_TURN_BUDGET"):
        _callback(monkeypatch, tmp_path, BENCHFLOW_TRIAL_TURN_BUDGET="many")


@pytest.mark.asyncio
async def test_without_a_budget_the_proxy_counts_and_refuses_nothing(
    monkeypatch, tmp_path
):
    _, hook, log = _callback(monkeypatch, tmp_path)
    for _ in range(5):
        assert (
            await hook.async_pre_call_hook(None, None, {"model": "m"}, "acompletion")
            is None
        )
    assert not _refused(log)


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": 900, "cache_read_input_tokens": 500, "output_tokens": 150},
        {
            "input_tokens": 10,
            "cache_creation_input_tokens": 7,
            "cache_read_input_tokens": 3,
            "output_tokens": 2,
        },
        {
            "prompt_tokens": 1200,
            "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 1000},
        },
        {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
        {
            "promptTokenCount": 70,
            "candidatesTokenCount": 9,
            "thoughtsTokenCount": 4,
            "toolUsePromptTokenCount": 3,
        },
        {
            "input_tokens": 40,
            "output_tokens": 4,
            "input_tokens_details": {"cached_tokens": 30},
        },
    ],
)
def test_the_proxy_counts_a_calls_tokens_as_benchflow_totals_them(
    monkeypatch, tmp_path, usage
):
    """The budget's tokens are the trial's ``total_tokens``: the callback's
    count of one call agrees with the trajectory's, shape by shape."""
    ns, _, _ = _callback(monkeypatch, tmp_path)
    key = "usageMetadata" if "promptTokenCount" in usage else "usage"
    exchange = LLMExchange(
        request=LLMRequest(timestamp=datetime.now(), method="POST", path="/", body={}),
        response=LLMResponse(
            timestamp=datetime.now(), status_code=200, body={key: usage}
        ),
    )
    assert ns["_call_tokens"](usage) == _exchange_token_usage(exchange).total_tokens


def test_the_controller_reads_the_refusal_and_counts_it_apart(caplog):
    record = {
        "event": "refused",
        "rule": "trial-budget",
        "budget": "tokens",
        "limits": {"tokens": 100},
        "used": {"tokens": 130, "turns": 2},
        "refused": 1,
        "request_model": "m",
        "call_type": "acompletion",
        "logged_at": "2026-11-05T00:00:00+00:00",
    }
    trajectory = trajectory_from_litellm_callback_log(
        json.dumps(record), session_id="s", agent_name="a"
    )
    [kept] = trajectory.metadata["refused_requests"]
    assert (kept["budget"], kept["used"], kept["limits"]) == (
        "tokens",
        {"tokens": 130, "turns": 2},
        {"tokens": 100},
    )
    assert budget_refusal(trajectory.metadata["refused_requests"])["refused_calls"] == 1
    # Not the served-model gate's warning: a budget refusal is an ending.
    with caplog.at_level(logging.WARNING, logger=runtime_mod.logger.name):
        runtime_mod._warn_refused_requests(trajectory)
    assert "does not serve" not in caplog.text


def test_the_live_capture_sees_the_refusal_as_it_arrives(tmp_path):
    process = runtime_mod.HostLiteLLMProcess.__new__(runtime_mod.HostLiteLLMProcess)
    process.session_id, process.agent_name = "s", "a"
    line = json.dumps({"event": "refused", "rule": "trial-budget", "budget": "turns"})
    writer = runtime_mod.LiveLLMTrajectoryWriter(tmp_path / "llm.jsonl")
    parsed = process._parse_live_record(line.encode(), writer)
    assert process.trial_budget_refusal() is None
    process._publish_live_record(runtime_mod.Trajectory(session_id="s"), parsed)
    assert process.trial_budget_refusal()["budget"] == "turns"


# -- the rollout: the agent is stopped, the workspace graded ------------------


def _write_task(task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 600\n[environment]\n"
    )
    (task_dir / "instruction.md").write_text("Do the task.\n")


class _Server:
    """A proxy whose live capture reads the refusal record after *after* s."""

    def __init__(self, after: float | None) -> None:
        self.after = after
        self.started = asyncio.get_running_loop().time()

    def trial_budget_refusal(self):
        if self.after is None:
            return None
        if asyncio.get_running_loop().time() - self.started < self.after:
            return None
        return {
            "rule": "trial-budget",
            "budget": "turns",
            "limits": {"turns": 2},
            "used": {"tokens": 3_000, "turns": 2},
        }

    async def settle_live_capture(self, timeout: float = 5.0) -> None:
        await asyncio.sleep(0.05)


def _budget_rollout(tmp_path: Path, agent) -> Rollout:
    task_dir = tmp_path / "task"
    _write_task(task_dir)
    rollout = Rollout(RolloutConfig(task_path=task_dir))
    trial = tmp_path / "trial"
    trial.mkdir()
    rollout._rollout_dir = trial
    rollout._rollout_name = "task__budget"
    rollout._started_at = datetime.now()
    rollout._trial_budget = TrialBudget(turns=2)
    rollout._trial_budget_sources = {"turns": "run"}
    rollout.setup = AsyncMock()
    rollout.start = AsyncMock()
    rollout.install_agent = AsyncMock()

    async def verify():
        rollout._rewards = None
        rollout._phase = "verified"
        return None

    async def cleanup():
        rollout._phase = "cleaned"

    rollout._run_steps = agent
    rollout.verify = AsyncMock(side_effect=verify)
    rollout.cleanup = AsyncMock(side_effect=cleanup)
    return rollout


@pytest.mark.asyncio
async def test_an_agent_that_goes_on_after_the_refusal_is_stopped_and_graded(tmp_path):
    """The agent keeps working after its call was refused (an agent may retry
    it until its time runs out): the rollout stops it, the verifier runs, a
    trial with no reward counts 0 like a timeout, and the ending is recorded."""

    async def agent(_steps):
        rollout._usage_runtime = SimpleNamespace(kind="litellm", server=_Server(0.2))
        await asyncio.sleep(3600)

    rollout = _budget_rollout(tmp_path, agent)
    result = await asyncio.wait_for(rollout.run(), timeout=30)
    rollout.verify.assert_awaited_once()
    assert result.error_category == TRIAL_BUDGET
    assert result.rewards == {"reward": 0.0}
    saved = json.loads((tmp_path / "trial" / "result.json").read_text())
    assert saved["error_category"] == TRIAL_BUDGET
    assert saved["trial_budget"]["stopped_by"] == "turns"
    assert saved["trial_budget"]["stopped_at"] == {"tokens": 3_000, "turns": 2}
    assert saved["trial_budget"]["budget"] == {"tokens": None, "turns": 2}
    assert saved["trial_budget"]["source"] == {"turns": "run"}
    record = trial_record(saved)
    assert (record["budget_stop"], record["outcome"]) == ("turns", "failed")
    assert record["timed_out"] is False


@pytest.mark.asyncio
async def test_an_agent_that_ends_its_turn_on_the_refused_call_is_a_budget_stop(
    tmp_path,
):
    """Claude Code ends its turn on the 400: no cancellation, the same ending."""

    async def agent(_steps):
        rollout._usage_runtime = SimpleNamespace(kind="litellm", server=_Server(0.0))

    rollout = _budget_rollout(tmp_path, agent)
    result = await asyncio.wait_for(rollout.run(), timeout=30)
    assert result.error_category == TRIAL_BUDGET
    saved = json.loads((tmp_path / "trial" / "result.json").read_text())
    assert saved["trial_budget"]["stopped_by"] == "turns"


@pytest.mark.asyncio
async def test_an_agent_error_on_the_refused_call_is_a_budget_stop_and_graded(
    tmp_path,
):
    from benchflow.agents.errors import AgentProtocolError

    async def agent(_steps):
        rollout._usage_runtime = SimpleNamespace(kind="litellm", server=_Server(0.0))
        raise AgentProtocolError("ACP error -32603: Internal error")

    rollout = _budget_rollout(tmp_path, agent)
    result = await asyncio.wait_for(rollout.run(), timeout=30)
    rollout.verify.assert_awaited_once()
    assert result.error_category == TRIAL_BUDGET


@pytest.mark.asyncio
async def test_a_trial_within_its_budget_ends_as_before(tmp_path):
    async def agent(_steps):
        rollout._usage_runtime = SimpleNamespace(kind="litellm", server=_Server(None))

    rollout = _budget_rollout(tmp_path, agent)
    result = await asyncio.wait_for(rollout.run(), timeout=30)
    assert result.error is None
    saved = json.loads((tmp_path / "trial" / "result.json").read_text())
    assert saved["trial_budget"]["stopped_by"] is None
    assert saved["trial_budget"]["refused_calls"] == 0


def test_an_agent_around_the_proxy_is_refused_before_it_starts(tmp_path):
    task_dir = tmp_path / "task"
    _write_task(task_dir)
    rollout = Rollout(RolloutConfig(task_path=task_dir))
    rollout._trial_budget = TrialBudget(tokens=10_000)
    rollout._usage_runtime = None
    with pytest.raises(TrialBudgetUnenforceableError) as refused:
        rollout._require_budget_proxy("claude-agent-acp", "anthropic/claude-x")
    assert refused.value.cause == "agent_budget"
    assert str(refused.value).startswith("agent integration failure [agent_budget]")
    rollout._usage_runtime = SimpleNamespace(kind="litellm")
    rollout._require_budget_proxy("claude-agent-acp", "anthropic/claude-x")
    rollout._trial_budget = None
    rollout._usage_runtime = None
    rollout._require_budget_proxy("claude-agent-acp", "anthropic/claude-x")


def test_the_spent_error_carries_the_proxys_record():
    error = TrialBudgetSpentError({"budget": "tokens", "limits": {"tokens": 5}})
    assert error.record["budget"] == "tokens"
    assert str(error).startswith("trial budget reached: the token budget of 5")


# -- what the trial used: result.json, the subagents' share -------------------


def _claude_log(folder: Path) -> None:
    rows = []
    for i, (side, tokens) in enumerate(((False, 1000), (True, 400), (False, 600))):
        rows.append(
            {
                "type": "assistant",
                "isSidechain": side,
                "timestamp": f"2026-11-05T10:00:0{i}Z",
                "message": {
                    "id": f"msg_{i}",
                    "model": "claude-x",
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": tokens, "output_tokens": 0},
                },
            }
        )
    folder.mkdir(parents=True)
    (folder / "session.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_the_subagents_share_comes_from_the_agents_own_log(tmp_path):
    assert subagent_usage(tmp_path) is None
    _claude_log(tmp_path / "agent" / "claude-sessions")
    assert subagent_usage(tmp_path) == {
        "turns": 1,
        "tokens": 400,
        "of_turns": 3,
        "of_tokens": 2000,
        "source": "claude-code-session-log",
    }


def test_result_json_records_the_budget_what_was_used_and_the_subagents(tmp_path):
    trial = tmp_path / "t__1"
    trial.mkdir()
    _claude_log(trial / "agent" / "claude-sessions")
    (trial / "model-calls.json").write_text(json.dumps({"calls": 3}))
    block = {
        "budget": {"tokens": 1500, "turns": None},
        "source": {"tokens": "task"},
        "stopped_by": "tokens",
        "stopped_at": {"tokens": 2000, "turns": 3},
        "refused_calls": 1,
    }
    result = _build_rollout_result(
        trial,
        task_name="t",
        rollout_name="t__1",
        agent="claude-agent-acp",
        agent_name="claude-agent-acp",
        model="anthropic/m",
        n_tool_calls=2,
        prompts=["go"],
        error=stop_text({"budget": "tokens", "limits": {"tokens": 1500}}),
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 0.0},
        started_at=datetime.now(),
        timing={},
        total_tokens=2000,
        trial_budget=block,
    )
    saved = json.loads((trial / "result.json").read_text())
    assert saved["trial_budget"] == {
        **block,
        "used": {"tokens": 2000, "turns": 3},
        "subagents": {
            "turns": 1,
            "tokens": 400,
            "of_turns": 3,
            "of_tokens": 2000,
            "source": "claude-code-session-log",
        },
    }
    assert result.trial_budget == saved["trial_budget"]
    assert saved["error_category"] == TRIAL_BUDGET
    # BenchFlow ended the prompt, as at a time limit.
    assert (
        saved["agent_outcome"]["ended_by"],
        saved["agent_outcome"]["harness_cause"],
    ) == ("harness", TRIAL_BUDGET)
    # Without a budget the field is there and empty.
    plain = tmp_path / "t__2"
    plain.mkdir()
    _build_rollout_result(
        plain,
        task_name="t",
        rollout_name="t__2",
        agent="a",
        agent_name="a",
        model=None,
        n_tool_calls=0,
        prompts=[],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={},
    )
    assert json.loads((plain / "result.json").read_text())["trial_budget"] is None


# -- configuration: the CLI's plan, run configs, resume -----------------------


def _tasks(tmp_path: Path) -> Path:
    tasks = tmp_path / "tasks"
    _write_task(tasks / "t")
    return tasks


def test_the_flags_reach_the_job_config(tmp_path):
    plan = build_eval_plan(
        EvalCreateRequest(
            tasks_dir=_tasks(tmp_path),
            agent="claude-agent-acp",
            model="anthropic/claude-haiku-4-5",
            agent_env={"ANTHROPIC_API_KEY": "sk-test"},
            token_budget=300_000,
            turn_budget=60,
        )
    )
    assert plan.make_eval_config().trial_budget == TrialBudget(tokens=300_000, turns=60)
    assert (
        build_eval_plan(
            EvalCreateRequest(tasks_dir=_tasks(tmp_path / "b"), agent="oracle")
        )
        .make_eval_config()
        .trial_budget
        is None
    )


@pytest.mark.parametrize(
    ("request_kwargs", "words"),
    [
        ({"token_budget": 0}, "--token-budget must be > 0"),
        ({"turn_budget": -2}, "--turn-budget must be > 0"),
    ],
)
def test_the_plan_refuses_a_budget_that_is_not_positive(
    tmp_path, request_kwargs, words
):
    with pytest.raises(EvalPlanError, match=words):
        build_eval_plan(
            EvalCreateRequest(
                tasks_dir=_tasks(tmp_path), agent="oracle", **request_kwargs
            )
        )


def test_the_plan_refuses_a_budget_an_agent_around_the_proxy_could_not_keep(
    tmp_path, monkeypatch
):
    """A Claude login takes Claude Code around the proxy: no job is made."""
    monkeypatch.setattr(
        "benchflow.agents.env.uses_native_subscription_auth", lambda *a, **k: True
    )
    with pytest.raises(EvalPlanError, match="cannot be enforced on claude-agent-acp"):
        build_eval_plan(
            EvalCreateRequest(
                tasks_dir=_tasks(tmp_path),
                agent="claude-agent-acp",
                model="anthropic/claude-haiku-4-5",
                turn_budget=10,
            )
        )


def test_a_run_config_sets_the_budget_and_writes_it_back(tmp_path):
    raw = {
        "tasks_dir": str(_tasks(tmp_path)),
        "agent": "oracle",
        "token_budget": 50_000,
        "turn_budget": 25,
    }
    evaluation = Evaluation.from_dict(raw)
    assert evaluation._config.trial_budget == TrialBudget(tokens=50_000, turns=25)
    written = evaluation.to_dict()
    assert (written["token_budget"], written["turn_budget"]) == (50_000, 25)
    again = Evaluation.from_dict({**written, "tasks_dir": raw["tasks_dir"]})
    assert again._config.trial_budget == evaluation._config.trial_budget
    assert (
        Evaluation.from_dict(
            {**raw, "token_budget": None, "turn_budget": None}
        )._config.trial_budget
        is None
    )


def test_a_rollout_config_round_trips_the_budget(tmp_path):
    task = tmp_path / "task"
    _write_task(task)
    config = RolloutConfig(task_path=task, trial_budget={"turns": 7})
    assert config.trial_budget == TrialBudget(turns=7)
    restored = RolloutConfig.from_dict(config.to_dict())
    assert restored.trial_budget == TrialBudget(turns=7)
    assert EvaluationConfig(trial_budget={"tokens": 9}).trial_budget == TrialBudget(
        tokens=9
    )


def test_a_sharded_runs_workers_keep_the_budget():
    """Each worker of a sharded run builds its config from the shard payload,
    so the budget must cross it or that worker's trials run unbounded.
    Guards b6cc53616's fix against the gap 44399c360 left: the payload
    carried no budget."""
    from benchflow.eval_sharding import EvalShard, _config_payload
    from benchflow.eval_worker import _evaluation_config

    shard = EvalShard(index=0, task_names=("task",), concurrency=1)
    budget = TrialBudget(tokens=5000, turns=9)
    config = EvaluationConfig(agent="oracle", trial_budget=budget)
    payload = json.loads(json.dumps(_config_payload(config, shard=shard)))
    assert _evaluation_config(payload).trial_budget == budget
    plain = _config_payload(EvaluationConfig(agent="oracle"), shard=shard)
    assert _evaluation_config(json.loads(json.dumps(plain))).trial_budget is None


# -- the run card, before and after; the summaries ----------------------------


def test_the_card_says_the_budget_before_the_run_and_refuses_one_it_cannot_keep():
    block = {
        "run": {"tokens": 200_000, "turns": 50},
        "tokens": {"min": 200_000, "max": 200_000},
        "turns": {"min": 50, "max": 50},
        "tasks_without": 0,
        "sources": ["run"],
        "unenforceable": None,
    }
    text = trial_budget_text(block)
    assert text.startswith("budget 200,000 tokens and 50 turns (the run's), a turn")
    assert "subagents' and side calls included" in text
    assert trial_budget_text(None).startswith("no token or turn budget")
    card = {"trial_budget": {**block, "unenforceable": "it signs in with a login"}}
    card["agent"] = {"name": "claude-agent-acp"}
    with pytest.raises(ValueError, match="cannot be enforced on claude-agent-acp"):
        check_card(card)


def _budget_result(trial: Path, stopped: str | None, tokens: int, sub: int) -> None:
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "rollout_name": trial.name,
                "trial_budget": {
                    "budget": {"tokens": 10_000, "turns": 40},
                    "used": {"tokens": tokens, "turns": 12},
                    "stopped_by": stopped,
                    "subagents": {"tokens": sub, "of_tokens": tokens},
                },
            }
        )
    )


def test_the_run_checks_total_the_budgets_and_the_card_says_them_after(tmp_path):
    checks = RunChecks(None, planned=3)
    _budget_result(tmp_path / "a", "tokens", 10_400, 4_000)
    _budget_result(tmp_path / "b", None, 6_000, 0)
    _budget_result(tmp_path / "c", "turns", 9_000, 2_000)
    for name in "abc":
        checks.record(name, tmp_path / name)
    block = checks.summary()["trial_budget"]
    assert block["trials"] == 3
    assert block["stopped"] == 2
    assert block["stopped_by"] == {"tokens": 1, "turns": 1}
    assert block["stopped_trials"] == {"tokens": ["a"], "turns": ["c"]}
    assert block["used_max"] == {"tokens": 10_400, "turns": 12}
    assert block["used_median"]["tokens"] == 9_000
    assert block["subagent_token_share"] == round(6_000 / 25_400, 4)
    lines = after_run_lines(checks.summary())
    [line] = [line for line in lines if line.startswith("  Per trial:")]
    assert "2 of 3 trials stopped at it (1 by tokens, 1 by turns)" in line
    assert "subagents used 24% of the tokens" in line
    pooled = pool([("j1", checks.summary()), ("j2", checks.summary())])
    assert pooled["trial_budget"]["trials"] == 6
    assert pooled["trial_budget"]["stopped_trials"]["tokens"] == ["j1/a", "j2/a"]
    assert RunChecks(None, planned=1).summary()["trial_budget"] is None


def test_summaries_count_budget_stops_beside_the_score():
    rows = [
        {
            "task_name": "a",
            "rewards": {"reward": 0.0},
            "trial_budget": {"stopped_by": "tokens"},
        },
        {
            "task_name": "b",
            "rewards": {"reward": 1.0},
            "trial_budget": {"stopped_by": None},
        },
    ]
    summary = score_summary_fields(rows)
    assert (summary["budget_stops"], summary["budget_stop_rate"]) == (1, 0.5)
    figures = score(trial_record(row) for row in rows)
    assert figures.budget_stops == 1
    assert figures.budget_stops_text().startswith("1 of 2 trials (50.0%) were stopped")


def test_a_fresh_trials_summary_row_carries_its_endings(tmp_path):
    """summary.json counts a fresh trial from its in-memory result: the
    budget stop and the model refusal reach that row, not only result.json."""
    from benchflow._utils.evaluation_results import rollout_result_payload
    from benchflow.models import RolloutResult

    result = RolloutResult(
        "t",
        rollout_name="t__1",
        rewards={"reward": 0.0},
        trial_budget={"stopped_by": "turns"},
        model_refusal={"by": "last_call"},
    )
    payload = rollout_result_payload(
        result, source_provenance=None, tasks_dir=tmp_path, task_name="t"
    )
    summary = score_summary_fields([payload])
    assert (summary["budget_stops"], summary["model_refusals"]) == (1, 1)
