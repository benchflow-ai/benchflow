"""A session-log cost is the model's price, not Claude Code's default rate
(benchflow-ai/benchflow#1240, reported by Benjamin-eecs).

Claude Code prices Claude models only. Pointed at another provider's
Anthropic-compatible endpoint (MiMo, DeepSeek, GLM, Kimi), it costs that
model at its default rate, and its ``cost-state`` lines carry that figure:
for mimo-v2.6-pro $4 / $20 / $0.20 per million input, output and cache-read
tokens, where MiMo's list price is $0.435 / $0.87 / $0.0036. BenchFlow wrote
Claude Code's figure as the trial's ``cost_usd``: 18 to 20 times the list
price on reported MiMo rollouts. The fixture below is the cost-state
Benjamin quoted (pulsar 8e304b18, Claude Code 2.1.280): Claude Code's
$7.07886, MiMo's list price $0.35898.

A session log over the copy limit (64 MiB) still gets no figure: pricing it
from the copy made while the turns ran is not in this port.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow._utils.pricing import list_rates
from benchflow._utils.session_cost import estimate_claude_code_cost
from tests.test_en374_salvage import _rollout

# usage_details.cost_estimate.method
CLAUDE_CODE_COST = "claude-code-cost"
CLAUDE_CODE_TOKENS = "claude-code-tokens-at-list-price"
MIMO = "mimo-v2.6-pro"
# A model no list-price table has (its vendor's page gives no per-model
# prices), on an Anthropic-compatible endpoint.
UNPRICED = "qwen3-coder-plus"
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
    _pulsar_log(tmp_path, model=UNPRICED)
    estimate = estimate_claude_code_cost(tmp_path / ".claude" / "projects")
    assert estimate is not None
    assert estimate["usd"] is None and estimate["method"] is None
    assert estimate["unpriced_models"] == [UNPRICED]
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
    # The date of the snapshot row MiMo is priced at.
    assert usage["price_table_date"] == list_rates(MIMO).table_date
    estimate = usage["usage_details"]["cost_estimate"]
    assert estimate["method"] == CLAUDE_CODE_TOKENS
    assert estimate["claude_code_usd"] == pytest.approx(CLAUDE_CODE_USD)


@pytest.mark.asyncio
async def test_a_mimo_direct_trial_on_an_unpriced_model_has_no_cost(
    tmp_path, monkeypatch
):
    rollout, _trial, home = _mimo_direct(tmp_path, monkeypatch)
    rollout._agent_env["ANTHROPIC_MODEL"] = UNPRICED
    _pulsar_log(home, model=UNPRICED)

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert usage["cost_usd"] is None
    assert usage["usage_details"]["cost_estimate"]["unpriced_models"] == [UNPRICED]


@pytest.mark.asyncio
async def test_claude_codes_running_cost_of_mimo_is_not_its_price(
    tmp_path, monkeypatch
):
    """With no log to read, the running cost Claude Code sent over ACP is
    its own figure too: kept for a Claude model, not for MiMo."""
    rollout, _trial, _home = _mimo_direct(tmp_path, monkeypatch)
    rollout._native_usage_metrics = {
        **rollout._native_usage_metrics,
        **dict.fromkeys(
            (
                "n_input_tokens",
                "n_output_tokens",
                "n_cache_read_tokens",
                "n_cache_creation_tokens",
                "total_tokens",
            ),
            None,
        ),
        "usage_source": "unavailable",
    }
    rollout._usage_report_missing = True
    rollout._last_usage_update = {
        "cost": {"amount": CLAUDE_CODE_USD, "currency": "USD"}
    }

    await rollout._cleanup_before_teardown()

    assert rollout._usage_metrics.get("cost_usd") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-5-5",
        "anthropic/claude-sonnet-5-5",
        # Claude Code's own aliases are Claude models it prices.
        "sonnet",
        "opus[1m]",
        "anthropic/haiku",
        "opusplan",
        "fable",
    ],
)
async def test_claude_codes_running_cost_of_a_claude_model_is_kept(
    tmp_path, monkeypatch, model
):
    """The same with a Claude model, by id or by Claude Code's alias: its
    running cost is Claude Code's figure at that model's rate, and kept."""
    rollout, _trial, _home = _mimo_direct(tmp_path, monkeypatch)
    rollout._agent_env = {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-" + "x" * 40}
    monkeypatch.setattr(rollout._config, "model", model, raising=False)
    rollout._native_usage_metrics = {
        **rollout._native_usage_metrics,
        **dict.fromkeys(
            (
                "n_input_tokens",
                "n_output_tokens",
                "n_cache_read_tokens",
                "n_cache_creation_tokens",
                "total_tokens",
            ),
            None,
        ),
        "usage_source": "unavailable",
    }
    rollout._usage_report_missing = True
    rollout._last_usage_update = {"cost": {"amount": 0.42, "currency": "USD"}}

    await rollout._cleanup_before_teardown()

    assert rollout._usage_metrics.get("cost_usd") == pytest.approx(0.42)


def test_a_bedrock_claude_id_keeps_claude_codes_own_figure(tmp_path):
    """Claude Code calling Bedrock itself logs a Bedrock model id: still a
    Claude model, priced at Claude Code's own figure."""
    _pulsar_log(tmp_path, model="us.anthropic.claude-opus-5-5-v1:0", cost=0.024)
    estimate = estimate_claude_code_cost(tmp_path / ".claude" / "projects")
    assert estimate is not None
    assert estimate["method"] == CLAUDE_CODE_COST
    assert estimate["usd"] == pytest.approx(0.024)
