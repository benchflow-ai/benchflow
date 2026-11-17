"""A session-log cost is the model's price, not Claude Code's default rate
(EN-497, benchflow-ai/benchflow#1240, reported by Benjamin-eecs).

Claude Code prices Claude models only. Pointed at another provider's
Anthropic-compatible endpoint (MiMo, DeepSeek, GLM, Kimi), it costs that
model at its default rate, and its ``cost-state`` lines carry that figure:
for mimo-v2.6-pro $4 / $20 / $0.20 per million input, output and cache-read
tokens, where MiMo's list price is $0.435 / $0.87 / $0.0036. BenchFlow wrote
Claude Code's figure as the trial's ``cost_usd``: 18 to 20 times the list
price on reported MiMo rollouts. The fixture below is the cost-state
Benjamin quoted (pulsar 8e304b18, Claude Code 2.1.280): Claude Code's
$7.07886, MiMo's list price $0.35898.

A session log over teardown's copy limit (64 MiB) got no figure and no word
why. 0.9.3 keeps no copy made while the turns ran (0.10 prices that copy),
so the record says why the cost is unknown.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow._utils.session_cost import estimate_claude_code_cost
from tests.test_en374_salvage import _rollout

# usage_details.cost_estimate.method
CLAUDE_CODE_COST = "claude-code-cost"
CLAUDE_CODE_TOKENS = "claude-code-tokens-at-list-price"
MIMO = "mimo-v2.6-pro"
OPUS = "claude-opus-5-5"
# pulsar 8e304b18's cost-state (benchflow#1240)
PULSAR = {"input": 293_989, "output": 244_790, "cache_read": 5_035_520}
CLAUDE_CODE_USD = 7.07886
# MiMo's list price: $0.435 / $0.87 / $0.0036 per million tokens
MIMO_LIST_USD = (
    PULSAR["input"] * 0.435e-6
    + PULSAR["output"] * 0.87e-6
    + PULSAR["cache_read"] * 0.0036e-6
)


def _usage(fresh: int, output: int, read: int = 0, write: int = 0) -> dict:
    return {
        "input_tokens": fresh,
        "output_tokens": output,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": write,
    }


def _state(model: str, cost: float, fresh: int, output: int, read: int) -> dict:
    return {
        "type": "cost-state",
        "totalCostUSD": cost,
        "modelUsage": {
            model: {
                "inputTokens": fresh,
                "outputTokens": output,
                "cacheReadInputTokens": read,
                "cacheCreationInputTokens": 0,
                "costUSD": cost,
            }
        },
    }


def _session(root: Path, lines: list[dict]) -> Path:
    path = root / ".claude" / "projects" / "-app" / "session-1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def _assistant(index: int, model: str, usage: dict) -> dict:
    return {
        "type": "assistant",
        "requestId": f"req_{index}",
        "message": {"id": f"msg_{index}", "model": model, "usage": usage},
    }


def _pulsar_log(home: Path, model: str = MIMO, cost: float = CLAUDE_CODE_USD) -> Path:
    """Three logged responses, and the cost-state that counts them and the
    calls Claude Code made for itself (which no response line logs)."""
    return _session(
        home,
        [
            _assistant(0, model, _usage(200_000, 150_000, 2_000_000)),
            _assistant(1, model, _usage(60_000, 50_000, 2_000_000)),
            _assistant(2, model, _usage(20_000, 30_000, 900_000)),
            _state(
                model, cost, PULSAR["input"], PULSAR["output"], PULSAR["cache_read"]
            ),
        ],
    )


def test_pulsar_is_priced_at_mimos_list_price_not_claude_codes_rate(tmp_path):
    """benchflow#1240: $0.35898, not Claude Code's $7.07886 (19.7x)."""
    _pulsar_log(tmp_path)
    estimate = estimate_claude_code_cost(tmp_path / ".claude" / "projects")
    assert estimate is not None
    assert estimate["method"] == CLAUDE_CODE_TOKENS
    assert estimate["usd"] == pytest.approx(MIMO_LIST_USD, abs=1e-9)
    assert estimate["usd"] == pytest.approx(0.35898, abs=1e-6)
    assert estimate["claude_code_usd"] == pytest.approx(CLAUDE_CODE_USD)
    assert estimate["repriced_models"] == [MIMO]
    assert CLAUDE_CODE_USD / estimate["usd"] == pytest.approx(19.72, abs=0.01)


def test_a_claude_session_keeps_claude_codes_own_figure(tmp_path):
    """Claude Code prices Claude models at their rate: its figure stands."""
    _pulsar_log(tmp_path, model=OPUS, cost=2.5)
    estimate = estimate_claude_code_cost(tmp_path / ".claude" / "projects")
    assert estimate is not None
    assert estimate["method"] == CLAUDE_CODE_COST
    assert estimate["usd"] == pytest.approx(2.5)
    assert "claude_code_usd" not in estimate


def test_a_model_no_list_price_covers_has_no_figure(tmp_path):
    """A model with no list price gets no cost, never Claude Code's default
    rate; the record names the model and keeps Claude Code's figure."""
    _pulsar_log(tmp_path, model="glm-5.3")
    estimate = estimate_claude_code_cost(tmp_path / ".claude" / "projects")
    assert estimate is not None
    assert estimate["usd"] is None and estimate["method"] is None
    assert estimate["unpriced_models"] == ["glm-5.3"]
    assert estimate["claude_code_usd"] == pytest.approx(CLAUDE_CODE_USD)


def _mimo_direct(tmp_path, monkeypatch):
    """Claude Code over ACP straight against MiMo's Anthropic-compatible
    endpoint: no gateway, so its calls reach BenchFlow only as the
    end-of-turn report (which counts the main session)."""
    home = tmp_path / "sandbox-home"
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    rollout._agent_env = {
        "ANTHROPIC_BASE_URL": "https://token-plan-sgp.xiaomimimo.com/anthropic",
        "ANTHROPIC_API_KEY": "tp-" + "f" * 40,
        "ANTHROPIC_MODEL": MIMO,
    }
    rollout._native_usage_metrics = {
        **rollout._native_usage_metrics,
        "n_input_tokens": 200_000,
        "n_output_tokens": 150_000,
        "n_cache_read_tokens": 2_000_000,
        "n_cache_creation_tokens": 0,
        "total_tokens": 2_350_000,
        "usage_source": "agent_native_acp",
        "cost_usd": None,
    }
    return rollout, trial, home


@pytest.mark.asyncio
async def test_a_mimo_direct_trial_records_mimos_list_price(tmp_path, monkeypatch):
    rollout, _trial, home = _mimo_direct(tmp_path, monkeypatch)
    _pulsar_log(home)

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert usage["cost_usd"] == pytest.approx(MIMO_LIST_USD, abs=1e-9)
    assert usage["price_source"] == "agent_session_log"
    assert usage["cost_source"] == "list_price_estimate"
    assert usage["price_table_date"] == "2026-11-04"  # MiMo's snapshot row
    estimate = usage["usage_details"]["cost_estimate"]
    assert estimate["method"] == CLAUDE_CODE_TOKENS
    assert estimate["claude_code_usd"] == pytest.approx(CLAUDE_CODE_USD)


@pytest.mark.asyncio
async def test_a_mimo_direct_trial_on_an_unpriced_model_has_no_cost(
    tmp_path, monkeypatch
):
    rollout, _trial, home = _mimo_direct(tmp_path, monkeypatch)
    rollout._agent_env["ANTHROPIC_MODEL"] = "glm-5.3"
    _pulsar_log(home, model="glm-5.3")

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert usage["cost_usd"] is None
    assert usage["usage_details"]["cost_estimate"]["unpriced_models"] == ["glm-5.3"]


@pytest.mark.asyncio
async def test_a_log_over_the_copy_limit_with_no_running_copy_says_why(
    tmp_path, monkeypatch
):
    from benchflow.rollout import _session_log

    rollout, _trial, home = _mimo_direct(tmp_path, monkeypatch)
    _pulsar_log(home)
    monkeypatch.setattr(_session_log, "_MAX_BYTES", 256)

    await rollout._cleanup_before_teardown()

    copy = rollout._usage_metrics["usage_details"]["session_log_copy"]
    assert copy["full_copy"].startswith("failed: over the copy limit")
    assert copy["used"] is None


@pytest.mark.parametrize(
    "model",
    [
        "us.anthropic.claude-opus-5-5-v1:0",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ],
)
def test_a_bedrock_claude_id_keeps_claude_codes_own_figure(tmp_path, model):
    """Claude Code calling Bedrock itself logs a Bedrock model id: still a
    Claude model, priced at Claude Code's own figure (ported from 0.10)."""
    _pulsar_log(tmp_path, model=model, cost=0.25)
    estimate = estimate_claude_code_cost(tmp_path / ".claude" / "projects")
    assert estimate is not None
    assert estimate["method"] == CLAUDE_CODE_COST
    assert estimate["usd"] == pytest.approx(0.25)
