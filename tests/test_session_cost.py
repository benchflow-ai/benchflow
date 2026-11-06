"""USD for a Claude Code rollout that BenchFlow's gateway could not price.

A subscription run (``CLAUDE_CODE_OAUTH_TOKEN``) calls the Anthropic API
from Claude Code itself: BenchFlow recorded its tokens with ``cost_usd``
null, so a ``bf.Budget``'s ``max_cost_usd`` could not count it and the
hill-climb demo (docs/examples/hillclimb/hillclimb_cost.py) priced trials from
Claude Code's session log on its own. BenchFlow now copies that log into the
trial and prices it, labelled as an estimate with its source.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from benchflow._utils.session_cost import (
    CLAUDE_CODE_COST,
    LIST_PRICE,
    estimate_claude_code_cost,
    list_price,
)

HAIKU = "claude-haiku-4-5-20251001"


def _response(msg_id: str, *, input_tokens=1000, output_tokens=50, model=HAIKU):
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    return {
        "type": "assistant",
        "requestId": f"req_{msg_id}",
        "message": {"id": msg_id, "model": model, "usage": usage},
    }


def _state(cost: float, *, input_tokens: int, output_tokens: int, model=HAIKU):
    return {
        "type": "cost-state",
        "totalCostUSD": cost,
        "modelUsage": {
            model: {
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "costUSD": cost,
            }
        },
    }


def _log(root: Path, lines: list[dict], name: str = "session-1") -> Path:
    path = root / "-app" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def test_claude_codes_own_total_wins_when_it_covers_the_log(tmp_path):
    first = _response("msg_1")
    _log(
        tmp_path,
        [
            first,
            first,  # a second content block of the same response
            _response("msg_2"),
            _state(0.0031, input_tokens=2000, output_tokens=100),
        ],
    )
    estimate = estimate_claude_code_cost(tmp_path)
    assert estimate is not None
    assert estimate["method"] == CLAUDE_CODE_COST
    assert estimate["usd"] == pytest.approx(0.0031)
    assert estimate["source"] == "claude-code-session-log"
    assert (estimate["sessions"], estimate["responses"]) == (1, 2)


def test_a_stale_total_falls_back_to_list_prices(tmp_path):
    """The last cost-state predates the last turn: price the logged usage."""
    if list_price(HAIKU) is None:
        pytest.skip("LiteLLM's price table has no claude-haiku-4-5")
    _log(
        tmp_path,
        [
            _response("msg_1"),
            _state(0.00125, input_tokens=1000, output_tokens=50),
            _response("msg_2"),
        ],
    )
    sub = tmp_path / "-app" / "session-1" / "subagents" / "agent-a.jsonl"
    sub.parent.mkdir(parents=True)
    sub.write_text(json.dumps(_response("msg_3")) + "\n")
    estimate = estimate_claude_code_cost(tmp_path)
    assert estimate is not None and estimate["method"] == LIST_PRICE
    price = list_price(HAIKU)
    assert price is not None
    per_call = 1000 * price["input"] + 50 * price["output"]
    assert estimate["usd"] == pytest.approx(3 * per_call)
    assert estimate["responses"] == 3


def test_no_log_or_an_unpriced_model_gives_no_estimate(tmp_path):
    assert estimate_claude_code_cost(tmp_path / "missing") is None
    _log(tmp_path, [_response("msg_1", model="claude-unknown-9")])
    assert estimate_claude_code_cost(tmp_path) is None


TOKEN = "sk-ant-oat01-" + "x" * 40


def _unpriced_claude_rollout(tmp_path, monkeypatch, home: Path, agent_env: dict):
    """A Claude Code rollout the gateway could not price, its log at *home*."""
    from benchflow._utils.task_authoring import task_digest
    from benchflow.rollout import Rollout, RolloutConfig, _session_log
    from benchflow.task import RolloutPaths, Task
    from tests.test_artifacts_collection import LocalTransport

    monkeypatch.setattr(_session_log, "_home", lambda cfg: str(home))

    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Say hi.")

    class Box(LocalTransport):
        async def stop(self, *, delete: bool = True) -> None:
            pass

    rollout = Rollout(
        RolloutConfig(
            task_path=task, agent="claude-agent-acp", task_digest=task_digest(task)
        )
    )
    trial = tmp_path / "trial"
    rollout._task = Task(task)
    rollout._rollout_dir = trial
    rollout._started_at = datetime.now()
    rollout._rollout_paths = RolloutPaths(rollout_dir=trial)
    rollout._rollout_paths.mkdir()
    rollout._env = Box()
    rollout._agent_cwd = str(tmp_path)
    rollout._agent_env = dict(agent_env)
    rollout._native_usage_metrics = {
        **rollout._native_usage_metrics,
        "n_input_tokens": 1000,
        "n_output_tokens": 50,
        "total_tokens": 1050,
        "usage_source": "agent_native_acp",
        "cost_usd": None,
    }
    return rollout, trial


@pytest.mark.asyncio
async def test_cleanup_prices_an_unpriced_claude_rollout_from_its_log(
    tmp_path, monkeypatch
):
    """The rollout's result gets the estimate, labelled, and the redacted log."""
    home = tmp_path / "home"
    token = TOKEN
    leak = {"type": "user", "message": {"content": f"env shows {token}"}}
    _log(
        home / ".claude" / "projects",
        [
            leak,
            _response("msg_1"),
            _state(0.00125, input_tokens=1000, output_tokens=50),
        ],
    )
    rollout, trial = _unpriced_claude_rollout(
        tmp_path, monkeypatch, home, {"CLAUDE_CODE_OAUTH_TOKEN": token}
    )

    await rollout.cleanup()

    usage = rollout._usage_metrics
    assert usage["cost_usd"] == pytest.approx(0.00125)
    assert usage["price_source"] == "agent_session_log"
    estimate = usage["usage_details"]["cost_estimate"]
    assert estimate["method"] == CLAUDE_CODE_COST
    assert estimate["path"] == "agent/claude-sessions"
    copied = trial / "agent" / "claude-sessions" / "-app" / "session-1.jsonl"
    text = copied.read_text()
    assert token not in text and "totalCostUSD" in text


@pytest.mark.asyncio
async def test_an_isolated_branch_child_is_not_priced_from_the_shared_log(
    tmp_path, monkeypatch
):
    """A child restored from the parent's snapshot must not be charged twice.

    `bench eval branch --isolate` runs each child as its own rollout in its own
    sandbox, restored from a snapshot taken after the parent's turns. That
    snapshot carries the parent's Claude Code session file, so pricing the
    child's copy of the log charges every sibling for the parent's turns as
    well as its own (the deterministic tier's branch-parallel scenario saw a
    child priced at 0.005005 where its own turns cost 0.0025).

    The same `_from_branch_snapshot` marker covers a run continued from a kept
    checkpoint, whose sandbox likewise carries the source rollout's log.
    """
    home = tmp_path / "home"
    _log(
        home / ".claude" / "projects",
        [_response("parent_1"), _state(0.00125, input_tokens=1000, output_tokens=50)],
        name="parent-session",
    )
    _log(
        home / ".claude" / "projects",
        [_response("child_1"), _state(0.00125, input_tokens=1000, output_tokens=50)],
        name="child-session",
    )
    rollout, trial = _unpriced_claude_rollout(
        tmp_path, monkeypatch, home, {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN}
    )
    rollout._from_branch_snapshot = True

    await rollout.cleanup()

    usage = rollout._usage_metrics
    assert usage["cost_usd"] is None
    assert usage.get("price_source") != "agent_session_log"
    assert not (trial / "agent" / "claude-sessions").exists()


@pytest.mark.asyncio
async def test_a_public_agent_env_value_survives_the_logs_redaction(
    tmp_path, monkeypatch
):
    """Only credential-bearing agent_env entries are scrubbed from the log.

    Scrubbing every value took the model name with it when the model was
    passed as an env value, and a response whose model reads `[redacted]` has
    no list price, so a session Claude Code's own totals do not cover became
    unpriceable.
    """
    home = tmp_path / "home"
    # A stale cost-state (it counts one response, the log holds two) forces the
    # list-price path, which needs each response's model name.
    _log(
        home / ".claude" / "projects",
        [
            _response("msg_1"),
            _response("msg_2"),
            _state(0.00125, input_tokens=1000, output_tokens=50),
        ],
    )
    rollout, trial = _unpriced_claude_rollout(
        tmp_path,
        monkeypatch,
        home,
        {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN, "ANTHROPIC_MODEL": HAIKU},
    )

    await rollout.cleanup()

    copied = trial / "agent" / "claude-sessions" / "-app" / "session-1.jsonl"
    assert HAIKU in copied.read_text()
    usage = rollout._usage_metrics
    estimate = usage["usage_details"]["cost_estimate"]
    assert estimate["method"] == LIST_PRICE
    assert list(estimate["models"]) == [HAIKU]
    price = list_price(HAIKU)
    assert price is not None
    one = 1000 * price["input"] + 50 * price["output"]
    assert usage["cost_usd"] == pytest.approx(2 * one)
