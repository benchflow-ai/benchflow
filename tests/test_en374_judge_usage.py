"""A trial's LLM judge spend is recorded, apart from its agent's.

Before: ``call_judge`` returned the verdict text only; the llm-judge and
agent-judge strategies never read ``response.usage``, so a judged task's
result.json, summary.json and budget had no judge spend at all (task.md's
judges counted tokens only into verifier/review.json). Now every judge
response's usage is recorded while the rollout's verifier runs, written to
result.json as ``judge_usage`` (agent_result's fields, ``usage_source``
``provider``, the cost at list prices), summed in summary.json's
``judge_totals`` and counted in ``budget.spent``.

The fake SDK responses carry each SDK's own usage fields: Anthropic's
``usage`` (input leaves the cache out), OpenAI's ``usage`` (prompt counts
the cache, ``prompt_tokens_details.cached_tokens``), Gemini's
``usage_metadata`` (prompt counts cached content; thoughts are output).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.rewards import judge_usage
from benchflow.rewards.llm import (
    _call_anthropic,
    _call_google,
    _call_openai,
    call_judge,
)

# Snapshot rates (2026-11-04), USD per token: input, output, cache read.
HAIKU = (0.1e-6, 0.5e-6, 0.01e-6)
SOL = (2e-6, 10e-6, 0.1e-6)
ARGON = (2e-6, 10e-6, 0.1e-6)


def _anthropic_response(model="claude-haiku-5-5"):
    return SimpleNamespace(
        id="msg_01",
        model=model,
        content=[SimpleNamespace(type="text", text='{"score": 1}')],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=1200,
            output_tokens=300,
            cache_read_input_tokens=4000,
            cache_creation_input_tokens=0,
            cache_creation=SimpleNamespace(
                ephemeral_5m_input_tokens=0, ephemeral_1h_input_tokens=0
            ),
            server_tool_use=None,
            service_tier="standard",
        ),
    )


def _patch_anthropic(response):
    client = AsyncMock()
    client.messages.create = AsyncMock(return_value=response)
    fake = type(
        "FakeAnthropic", (), {"AsyncAnthropic": staticmethod(lambda **_: client)}
    )
    return patch.dict("sys.modules", {"anthropic": fake})


def _patch_openai(response):
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    fake = type("FakeOpenAI", (), {"AsyncOpenAI": staticmethod(lambda **_: client)})
    return patch.dict("sys.modules", {"openai": fake})


def _patch_google(response):
    client = AsyncMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)
    genai = type("FakeGenai", (), {"Client": staticmethod(lambda api_key=None: client)})
    google = type("FakeGoogle", (), {"genai": genai})
    return patch.dict("sys.modules", {"google": google, "google.genai": genai})


@pytest.mark.asyncio
async def test_an_anthropic_judge_call_is_counted_and_priced():
    with _patch_anthropic(_anthropic_response()), judge_usage.recording() as calls:
        await _call_anthropic("claude-haiku-5-5", "grade it", 2048)
    block = judge_usage.usage_block(calls)
    assert block is not None
    assert block["calls"] == 1 and block["models"] == {"claude-haiku-5-5": 1}
    assert (
        block["n_input_tokens"],
        block["n_output_tokens"],
        block["n_cache_read_tokens"],
        block["n_cache_write_tokens"],
        block["n_cache_write_5m_tokens"],
        block["n_cache_write_1h_tokens"],
        block["total_tokens"],
    ) == (1200, 300, 4000, 0, 0, 0, 5500)
    assert block["usage_source"] == "provider"
    assert block["cost_source"] == "list_price_estimate"
    assert block["price_source"] == "list_price"
    assert block["price_table_date"] == "2026-11-04"
    assert block["cost_usd"] == pytest.approx(
        1200 * HAIKU[0] + 300 * HAIKU[1] + 4000 * HAIKU[2]
    )


@pytest.mark.asyncio
async def test_an_openai_judge_call_leaves_the_cache_out_of_its_input():
    response = SimpleNamespace(
        model="gpt-6.1-sol",
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"score": 1}'))],
        usage=SimpleNamespace(
            prompt_tokens=2000,
            completion_tokens=500,
            total_tokens=2500,
            prompt_tokens_details=SimpleNamespace(cached_tokens=1500, audio_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=200),
        ),
    )
    with _patch_openai(response), judge_usage.recording() as calls:
        await _call_openai("gpt-6.1-sol", "grade it", 2048, {"OPENAI_API_KEY": "k"})
    block = judge_usage.usage_block(calls)
    assert block is not None
    assert (block["n_input_tokens"], block["n_cache_read_tokens"]) == (500, 1500)
    assert block["n_output_tokens"] == 500 and block["total_tokens"] == 2500
    # Chat Completions reports no cache writes: unknown, not 0.
    assert block["n_cache_write_tokens"] is None
    assert block["cost_usd"] == pytest.approx(
        500 * SOL[0] + 500 * SOL[1] + 1500 * SOL[2]
    )


@pytest.mark.asyncio
async def test_a_gemini_judge_call_counts_its_thinking_as_output():
    response = SimpleNamespace(
        text='{"score": 1}',
        model_version="gemini-4-argon",
        usage_metadata=SimpleNamespace(
            prompt_token_count=3000,
            candidates_token_count=200,
            cached_content_token_count=1000,
            thoughts_token_count=100,
            total_token_count=3300,
        ),
    )
    with _patch_google(response), judge_usage.recording() as calls:
        await _call_google("gemini-4-argon", "grade it", {"GOOGLE_API_KEY": "k"})
    block = judge_usage.usage_block(calls)
    assert block is not None
    assert (block["n_input_tokens"], block["n_cache_read_tokens"]) == (2000, 1000)
    assert block["n_output_tokens"] == 300 and block["total_tokens"] == 3300
    assert block["cost_usd"] == pytest.approx(
        2000 * ARGON[0] + 300 * ARGON[1] + 1000 * ARGON[2]
    )


def test_a_response_without_usage_leaves_the_judges_tokens_unknown():
    calls = [
        judge_usage.JudgeCall(
            "claude-haiku-5-5",
            judge_usage.TokenSlice("claude-haiku-5-5", input=10, output=5),
        ),
        judge_usage.JudgeCall("claude-haiku-5-5", None),
    ]
    block = judge_usage.usage_block(calls)
    assert block is not None
    assert block["calls_without_usage"] == 1
    assert block["total_tokens"] is None and block["cost_usd"] is None
    assert judge_usage.usage_block([]) is None


def test_an_unpriced_judge_model_has_tokens_and_no_cost():
    calls = [
        judge_usage.JudgeCall(
            "judge-x", judge_usage.TokenSlice("judge-x", input=900, output=100)
        )
    ]
    block = judge_usage.usage_block(calls)
    assert block is not None
    assert block["total_tokens"] == 1000
    assert block["cost_usd"] is None and block["cost_source"] is None


@pytest.mark.asyncio
async def test_a_judge_call_outside_a_trial_records_nothing():
    with _patch_anthropic(_anthropic_response()):
        text = await call_judge("claude-haiku-5-5", "grade it", retries=1)
    assert text == '{"score": 1}'


@pytest.mark.asyncio
async def test_a_task_md_judge_session_records_each_call(tmp_path):
    from benchflow.taskmd import judging
    from tests.test_taskmd_judging import _Script, _spec, _submission, _use

    root = tmp_path / "fs"
    (root / "work").mkdir(parents=True)
    (root / "work" / "report.md").write_text("We used maximum likelihood.\n")
    (root / "judge").mkdir()
    files = judging.JudgeFiles(root=root, kept=["/work"])
    script = _Script([[_use("submit_review", _submission(), "submit")]], usage=1000)
    with judge_usage.recording() as calls:
        await judging.run_session(
            _spec(tools=("read", "submit_review")),
            call=script,
            credentials=judging.Credentials("api-key", "placeholder"),
            files=files,
            evidence_text=None,
            runner=None,
        )
    assert len(calls) == len(script.bodies) >= 1
    block = judge_usage.usage_block(calls)
    assert block is not None
    assert block["n_input_tokens"] == 1000 * len(calls)
    assert block["n_output_tokens"] == 10 * len(calls)


@pytest.mark.asyncio
async def test_the_rollout_writes_its_judges_usage_beside_the_agents(
    tmp_path, monkeypatch
):
    """Rollout.verify records the judge calls its verifier makes; the result
    carries them as judge_usage, and agent_result is the agent's alone."""
    from benchflow.rollout import Rollout
    from tests.test_en374_salvage import _rollout

    home = tmp_path / "sandbox-home"
    home.mkdir()
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="native", home=home
    )

    async def judged(self):
        await call_judge("claude-haiku-5-5", "grade it", retries=1)
        await call_judge("claude-haiku-5-5", "grade it again", retries=1)
        self._rewards = {"reward": 1.0}
        return self._rewards

    monkeypatch.setattr(Rollout, "_score_finished_solver", judged)
    with _patch_anthropic(_anthropic_response()):
        assert await rollout.verify() == {"reward": 1.0}
    result = rollout._build_result()

    assert result.judge_usage is not None
    assert result.judge_usage["calls"] == 2
    assert result.judge_usage["total_tokens"] == 2 * 5500
    assert result.total_tokens is None  # the agent's: nothing counted it
    written = json.loads((trial / "result.json").read_text())
    assert written["judge_usage"] == result.judge_usage
    assert written["agent_result"]["total_tokens"] is None
    assert written["agent_result"]["cost_usd"] is None


def _tasks(tmp_path: Path, n: int) -> Path:
    tasks_dir = tmp_path / "tasks"
    for i in range(n):
        (tasks_dir / f"task-{i}").mkdir(parents=True)
        (tasks_dir / f"task-{i}" / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    return tasks_dir


@pytest.mark.asyncio
async def test_summary_and_budget_count_the_judges_spend(tmp_path):
    from benchflow import Budget
    from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
    from benchflow.models import RolloutResult

    judge = judge_usage.usage_block(
        [
            judge_usage.JudgeCall(
                "claude-haiku-5-5",
                judge_usage.TokenSlice("claude-haiku-5-5", input=1200, output=300),
            )
        ]
    )
    assert judge is not None
    job = Evaluation(
        tasks_dir=_tasks(tmp_path, 2),
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            agent="claude-agent-acp",
            model="claude-haiku-5-5",
            concurrency=1,
            retry=RetryConfig(max_retries=0),
        ),
        budget=Budget(max_cost_usd=100.0),
        preflight=False,
    )

    async def fake_run(task_path, _cfg):
        await asyncio.sleep(0)
        rollout_dir = job._jobs_dir / job._job_name / f"{task_path.name}__x"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        agent = {
            "n_input_tokens": 100,
            "n_output_tokens": 50,
            "n_cache_read_tokens": 0,
            "n_cache_creation_tokens": 0,
            "total_tokens": 150,
            "cost_usd": 0.01,
            "usage_source": "proxy",
            "cost_source": "list_price_estimate",
            "price_source": "litellm",
            "price_table_date": None,
            "usage_source_legacy": "provider_response",
        }
        (rollout_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task_path.name,
                    "rollout_name": rollout_dir.name,
                    "rewards": {"reward": 1.0},
                    "agent": "claude-agent-acp",
                    "agent_result": agent,
                    "judge_usage": judge,
                    "error": None,
                    "verifier_error": None,
                }
            )
        )
        return RolloutResult(
            task_name=task_path.name,
            rollout_name=rollout_dir.name,
            rewards={"reward": 1.0},
            agent="claude-agent-acp",
            **{k: v for k, v in agent.items() if k != "usage_source_legacy"},
            judge_usage=judge,
            rollout_dir=rollout_dir,
        )

    job._run_single_task = AsyncMock(side_effect=fake_run)
    await job.run()
    summary = json.loads((job._jobs_dir / job._job_name / "summary.json").read_text())

    # The agent's totals stay the agent's.
    assert summary["total_tokens"] == 300
    assert summary["total_cost_usd"] == pytest.approx(0.02)
    totals = summary["judge_totals"]
    assert totals["trials"] == 2 and totals["calls"] == 2
    assert totals["total_tokens"] == 3000
    assert totals["total_cost_usd"] == pytest.approx(2 * judge["cost_usd"])
    assert totals["usage_sources"] == {"provider": 2}
    assert totals["cost_sources"] == {"list_price_estimate": 2}
    spend = summary["spend_all_attempts"]
    assert spend["judge_cost_usd"] == pytest.approx(2 * judge["cost_usd"])
    assert spend["cost_usd"] == pytest.approx(0.02 + 2 * judge["cost_usd"])
    spent = summary["budget"]["spent"]
    assert spent["cost_usd"] == pytest.approx(spend["cost_usd"])
    assert spent["judge_cost_usd"] == pytest.approx(spend["judge_cost_usd"])
    assert spent["tokens"] == 300 + 3000 and spent["judge_tokens"] == 3000


def test_a_recovery_records_its_judges_inside_the_rollouts_recording():
    """A verifier recovery keeps its own judge calls (for its record) while
    the rollout around it records every call."""
    outer_slice = judge_usage.TokenSlice("claude-haiku-5-5", input=10, output=5)
    with judge_usage.recording() as rollout_calls:
        judge_usage.record("claude-haiku-5-5", outer_slice)
        with judge_usage.recording() as recovery_calls:
            judge_usage.record("claude-haiku-5-5", outer_slice)
    assert len(rollout_calls) == 2 and len(recovery_calls) == 1


def test_a_rescore_publishes_the_recoverys_judge_spend_with_the_solvers(tmp_path):
    """`bench eval score` re-runs a verifier through recovery
    (resume_verification), which can call an LLM judge: the recovery's
    record keeps that spend, and the re-scored result's judge_usage is the
    first verification's and the recovery's together."""
    from benchflow.rollout._verifier_recovery import verification_source

    def block(n: int, cost: float) -> dict:
        calls = [
            judge_usage.JudgeCall(
                "claude-haiku-5-5",
                judge_usage.TokenSlice("claude-haiku-5-5", input=1000, output=100),
            )
            for _ in range(n)
        ]
        made = judge_usage.usage_block(calls)
        assert made is not None
        return {**made, "cost_usd": cost}

    (tmp_path / "solver.json").write_text(
        json.dumps({"task_digest": "d", "judge_usage": block(1, 0.001)})
    )
    attempt = tmp_path / "verifier-recovery" / "abc123"
    attempt.mkdir(parents=True)
    (attempt / "recovery.json").write_text(
        json.dumps(
            {
                "task_digest": "d",
                "status": "complete",
                "rewards": {"reward": 1.0},
                "judge_usage": block(2, 0.002),
                "timing": {"verifier": 1.0},
            }
        )
    )
    (tmp_path / "verification.json").write_text(
        json.dumps({"attempt": "verifier-recovery/abc123"})
    )
    combined = verification_source(tmp_path)["judge_usage"]
    assert combined["calls"] == 3
    assert combined["total_tokens"] == 3 * 1100
    assert combined["cost_usd"] == pytest.approx(0.003)


def test_a_rescore_keeps_every_recoverys_judge_spend(tmp_path):
    """Review 2, #5: an in-run recovery called a judge and failed, then
    `bench eval score` recovered again: only the latest recovery's judge
    spend was combined, so the trial's spend went down. Every recovery's
    counts."""
    from benchflow.rollout._verifier_recovery import verification_source

    calls = [
        judge_usage.JudgeCall(
            "claude-haiku-5-5",
            judge_usage.TokenSlice("claude-haiku-5-5", input=1000, output=100),
        )
    ]
    block = judge_usage.usage_block(calls)
    assert block is not None
    (tmp_path / "solver.json").write_text(
        json.dumps({"task_digest": "d", "judge_usage": block})
    )
    for name, status in (("first1", "failed"), ("second2", "complete")):
        attempt = tmp_path / "verifier-recovery" / name
        attempt.mkdir(parents=True)
        (attempt / "recovery.json").write_text(
            json.dumps(
                {
                    "task_digest": "d",
                    "status": status,
                    "rewards": {"reward": 1.0} if status == "complete" else None,
                    "error": None if status == "complete" else "judge failed",
                    "judge_usage": block,
                    "timing": {"verifier": 1.0},
                }
            )
        )
    (tmp_path / "verification.json").write_text(
        json.dumps({"attempt": "verifier-recovery/second2"})
    )
    combined = verification_source(tmp_path)["judge_usage"]
    assert combined["calls"] == 3
    assert combined["cost_usd"] == pytest.approx(3 * block["cost_usd"])


def test_an_interrupted_recoverys_judge_spend_counts_with_no_pointer(tmp_path):
    """The delta review's nit: every recovery was interrupted, so none
    published a verdict (no verification.json); the re-score counted the
    first verification's judges only. An interrupted recovery's record keeps
    its judge spend, and it counts."""
    from benchflow.rollout._verifier_recovery import verification_source

    calls = [
        judge_usage.JudgeCall(
            "claude-haiku-5-5",
            judge_usage.TokenSlice("claude-haiku-5-5", input=1000, output=100),
        )
    ]
    block = judge_usage.usage_block(calls)
    assert block is not None
    (tmp_path / "solver.json").write_text(
        json.dumps({"task_digest": "d", "judge_usage": block})
    )
    attempt = tmp_path / "verifier-recovery" / "cut1"
    attempt.mkdir(parents=True)
    (attempt / "recovery.json").write_text(
        json.dumps(
            {
                "task_digest": "d",
                "status": "interrupted",
                "judge_usage": block,
                "timing": {"verifier": 1.0},
            }
        )
    )
    assert not (tmp_path / "verification.json").exists()
    assert verification_source(tmp_path)["judge_usage"]["calls"] == 2
