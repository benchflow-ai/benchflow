"""Tests for benchflow.pricing and benchflow.cost (post-hoc cost recomputation)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from benchflow.cost import (
    recompute_rollout_cost,
    update_result_json,
    usage_records_from_llm_trajectory,
    write_cost_report,
)
from benchflow.pricing import (
    LIST_PRICES,
    LITELLM_OBSERVED_FP_2026_09,
    PRICING_VERSION,
    UnknownModelPrice,
    UsageCounts,
    canonical_model,
    compute_cost,
    price_table,
)
from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    Trajectory,
)

MTOK = 1_000_000


# ------------------------------------------------------------------ pricing


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("claude-fable-5-1", "claude-fable-5-1"),
        ("anthropic/claude-fable-5-1", "claude-fable-5-1"),
        ("openrouter/anthropic/claude-fable-5.1", "claude-fable-5-1"),
        ("us.anthropic.claude-opus-5-v1:0", "claude-opus-5"),
        ("global.anthropic.claude-fable-5", "claude-fable-5"),
        ("claude-opus-5[1m]", "claude-opus-5"),
        ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
        ("databricks/databricks-claude-sonnet-5", "claude-sonnet-5"),
        ("", None),
        (None, None),
    ],
)
def test_canonical_model(raw, expected):
    assert canonical_model(raw) == expected


def test_list_prices_are_anthropic_ratios():
    p = LIST_PRICES.get("claude-opus-5")
    assert p is not None
    assert (p.input, p.output, p.cache_read, p.cache_write_5m, p.cache_write_1h) == (
        5.0,
        25.0,
        0.5,
        6.25,
        10.0,
    )
    fable = LIST_PRICES.get("claude-fable-5-1")
    assert fable is not None
    assert fable.cache_read == 0.25  # Fable 5.1 special cache-read price
    assert fable.cache_write_5m == 12.5


def test_compute_cost_breakdown_and_1h_cache():
    usage = UsageCounts(
        input_tokens=1 * MTOK,
        output_tokens=2 * MTOK,
        cache_read_tokens=10 * MTOK,
        cache_creation_tokens=4 * MTOK,
        cache_creation_1h_tokens=1 * MTOK,  # 1 of the 4 written with the 1h TTL
    )
    bd = compute_cost("claude-opus-5", usage)
    assert bd.pricing_version == PRICING_VERSION
    assert bd.input_usd == pytest.approx(5.0)
    assert bd.output_usd == pytest.approx(50.0)
    assert bd.cache_read_usd == pytest.approx(5.0)
    assert bd.cache_write_usd == pytest.approx(3 * 6.25 + 1 * 10.0)
    assert bd.total_usd == pytest.approx(5 + 50 + 5 + 28.75)


def test_compute_cost_unknown_model_raises():
    with pytest.raises(UnknownModelPrice):
        compute_cost("gpt-99", UsageCounts(input_tokens=10))


def test_price_table_lookup():
    assert price_table(None) is LIST_PRICES
    assert price_table("list") is LIST_PRICES
    assert (
        price_table(LITELLM_OBSERVED_FP_2026_09.version) is LITELLM_OBSERVED_FP_2026_09
    )
    with pytest.raises(KeyError):
        price_table("nope")


def test_usage_from_inclusive_input_matches_benchflow_normalization():
    # BenchFlow's n_input_tokens = uncached + cache_read + cache_creation
    u = UsageCounts.from_inclusive_input(
        input_tokens_inclusive=63_728_808,
        output_tokens=424_304,
        cache_read_tokens=41_121_406,
        cache_creation_tokens=11_190_144,
    )
    assert u.input_tokens == 63_728_808 - 41_121_406 - 11_190_144
    assert u.total_input_inclusive == 63_728_808


def test_usage_from_anthropic_usage_block():
    u = UsageCounts.from_anthropic_usage(
        {
            "input_tokens": 2,
            "output_tokens": 219,
            "cache_read_input_tokens": 24614,
            "cache_creation_input_tokens": 7366,
            "cache_creation": {
                "ephemeral_1h_input_tokens": 366,
                "ephemeral_5m_input_tokens": 7000,
            },
        }
    )
    assert (
        u.input_tokens,
        u.output_tokens,
        u.cache_read_tokens,
        u.cache_creation_tokens,
    ) == (2, 219, 24614, 7366)
    assert u.cache_creation_1h_tokens == 366


# ------------------------------------------------------------------ fixtures


def _anthropic_exchange(
    model: str, usage: dict, *, provider_model: str | None = None
) -> LLMExchange:
    return LLMExchange(
        request=LLMRequest(
            timestamp=datetime(2026, 9, 1, 12, 0, 0),
            path="/v1/messages",
            body={"model": model, "messages": []},
        ),
        response=LLMResponse(
            timestamp=datetime(2026, 9, 1, 12, 0, 5),
            status_code=200,
            body={"model": model, "usage": usage, "content": []},
        ),
        duration_ms=5000.0,
        metadata={"provider_model": provider_model or model, "request_model": model},
    )


def _write_rollout(
    tmp_path: Path,
    *,
    model: str,
    exchanges: list[LLMExchange] | None = None,
    result: dict | None = None,
) -> Path:
    rollout = tmp_path / "rollout"
    (rollout / "trajectory").mkdir(parents=True)
    if exchanges is not None:
        traj = Trajectory(
            session_id="s", agent_name="claude-agent-acp", exchanges=exchanges
        )
        (rollout / "trajectory" / "llm_trajectory.jsonl").write_text(
            traj.to_jsonl() + "\n"
        )
    base = {
        "task_name": "t",
        "rollout_name": "t__1",
        "model": model,
        "agent_result": {
            "usage_source": "provider_response",
            "price_source": "litellm",
            "cost_usd": 1.0,
        },
        "final_metrics": {"total_cost_usd": 1.0},
    }
    if result:
        base.update(result)
    (rollout / "result.json").write_text(json.dumps(base))
    return rollout


# ------------------------------------------------------------------ llm_trajectory


def test_recompute_from_llm_trajectory_anthropic_usage(tmp_path):
    exchanges = [
        _anthropic_exchange(
            "anthropic/claude-opus-5",
            {
                "input_tokens": 1000,
                "output_tokens": 100,
                "cache_read_input_tokens": 5000,
                "cache_creation_input_tokens": 2000,
            },
        ),
        _anthropic_exchange(
            "anthropic/claude-opus-5",
            {
                "input_tokens": 500,
                "output_tokens": 50,
                "cache_read_input_tokens": 7000,
                "cache_creation_input_tokens": 0,
            },
        ),
    ]
    # a failed call without usage must not be counted
    failed = _anthropic_exchange("anthropic/claude-opus-5", {})
    failed.response.status_code = 529
    failed.response.body = {"error": {"message": "overloaded"}}
    rollout = _write_rollout(
        tmp_path, model="claude-opus-5", exchanges=[*exchanges, failed]
    )

    report = recompute_rollout_cost(rollout)
    assert report.source == "llm_trajectory"
    assert report.n_records == 2
    assert len(report.per_model) == 1
    m = report.per_model[0]
    assert m.usage == UsageCounts(1500, 150, 12000, 2000)
    expected = (1500 * 5 + 150 * 25 + 12000 * 0.5 + 2000 * 6.25) / MTOK
    assert report.total_cost_usd == pytest.approx(expected)
    assert report.recorded_cost_usd == 1.0
    assert report.delta_usd == pytest.approx(expected - 1.0)


def test_recompute_from_llm_trajectory_openai_style_usage(tmp_path):
    # OpenAI convention: prompt_tokens is cache-INCLUSIVE; cached_tokens is a subset.
    ex = LLMExchange(
        request=LLMRequest(body={"model": "gpt-x", "messages": []}),
        response=LLMResponse(
            body={
                "model": "claude-sonnet-5",  # priced model id on the response
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 10,
                    "prompt_tokens_details": {"cached_tokens": 600},
                },
            }
        ),
        metadata={"provider_model": "claude-sonnet-5"},
    )
    rollout = _write_rollout(tmp_path, model="claude-sonnet-5", exchanges=[ex])
    records = usage_records_from_llm_trajectory(
        rollout / "trajectory" / "llm_trajectory.jsonl"
    )
    assert records == [("claude-sonnet-5", UsageCounts(400, 10, 600, 0))]


def test_recompute_unpriced_model_reports_none(tmp_path):
    ex = _anthropic_exchange(
        "some-vendor/mystery-model", {"input_tokens": 10, "output_tokens": 1}
    )
    rollout = _write_rollout(tmp_path, model="mystery-model", exchanges=[ex])
    report = recompute_rollout_cost(rollout)
    assert report.total_cost_usd is None
    assert report.unpriced_models == ["some-vendor/mystery-model"]
    assert report.per_model[0].unpriced is True


# ------------------------------------------------------------------ otel_usage (OAuth capture)


def test_recompute_from_otel_usage_dedupes_request_ids(tmp_path):
    rollout = _write_rollout(tmp_path, model="claude-fable-5-1")
    rows = [
        {
            "event": "api_request",
            "model": "claude-fable-5-1",
            "input_tokens": 2,
            "output_tokens": 4,
            "cache_read_tokens": 16602,
            "cache_creation_tokens": 0,
            "request_id": "req_a",
            "cost_usd": 0.0043705,
        },
        {
            "event": "api_request",
            "model": "claude-fable-5-1",
            "input_tokens": 2,
            "output_tokens": 4,
            "cache_read_tokens": 16602,
            "cache_creation_tokens": 0,
            "request_id": "req_a",
        },  # duplicate export
        {
            "event": "api_request",
            "model": "claude-haiku-4-5-20251001",
            "input_tokens": 899,
            "output_tokens": 8,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "request_id": "req_b",
            "cost_usd": 0.000939,
        },
        {
            "event": "api_error",
            "model": "claude-fable-5-1",
            "status_code": 529,
            "request_id": "req_c",
        },
    ]
    (rollout / "trajectory" / "otel_usage.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )

    report = recompute_rollout_cost(rollout)
    assert report.source == "otel_usage"
    assert report.n_records == 2
    by_model = {m.model: m for m in report.per_model}
    assert by_model["claude-fable-5-1"].usage == UsageCounts(2, 4, 16602, 0)
    fable = (2 * 10 + 4 * 50 + 16602 * 0.25) / MTOK
    haiku = (899 * 1 + 8 * 5) / MTOK
    # Claude Code's own cost_usd attributes agree with the list table to the micro-dollar
    assert by_model["claude-fable-5-1"].cost_usd == pytest.approx(fable, abs=1e-6)
    assert by_model["claude-haiku-4-5-20251001"].cost_usd == pytest.approx(
        haiku, abs=1e-6
    )
    assert report.total_cost_usd == pytest.approx(fable + haiku)


# ------------------------------------------------------------------ result.json aggregate fallback


def test_recompute_from_result_aggregate_provider_response(tmp_path):
    # Real FrontierPhysics slot (diffuse-ib-rigid-body-fsi / claude-fable-5-1 / max / trial-3)
    result = {
        "model": "claude-fable-5-1",
        "agent_result": {
            "n_input_tokens": 63728808,
            "n_output_tokens": 424304,
            "n_cache_read_tokens": 41121406,
            "n_cache_creation_tokens": 11190144,
            "cost_usd": 173.6434915,
            "usage_source": "provider_response",
            "price_source": "litellm",
        },
        "final_metrics": {"total_cost_usd": 173.6434915},
    }
    rollout = _write_rollout(tmp_path, model="claude-fable-5-1", result=result)
    observed = recompute_rollout_cost(rollout, pricing=LITELLM_OBSERVED_FP_2026_09)
    assert observed.source == "result_aggregate"
    assert observed.total_cost_usd == pytest.approx(
        173.6434915, rel=1e-9
    )  # reproduces LiteLLM's figure
    listed = recompute_rollout_cost(rollout)  # official list prices
    uncached = 63728808 - 41121406 - 11190144
    expected = (uncached * 10 + 424304 * 50 + 41121406 * 0.25 + 11190144 * 12.5) / MTOK
    assert listed.total_cost_usd == pytest.approx(expected)
    assert (
        listed.total_cost_usd > observed.total_cost_usd
    )  # cache writes under-priced by LiteLLM


def test_recompute_from_result_aggregate_native_acp_is_uncached(tmp_path):
    result = {
        "model": "claude-opus-5",
        "agent_result": {
            "n_input_tokens": 100,  # claude-agent-acp: raw (uncached) inputTokens
            "n_output_tokens": 10,
            "n_cache_read_tokens": 1000,
            "n_cache_creation_tokens": 200,
            "cost_usd": None,
            "usage_source": "agent_native_acp",
            "price_source": None,
        },
        "final_metrics": {"total_cost_usd": None},
    }
    rollout = _write_rollout(tmp_path, model="claude-opus-5", result=result)
    report = recompute_rollout_cost(rollout)
    assert report.per_model[0].usage == UsageCounts(100, 10, 1000, 200)
    assert report.recorded_cost_usd is None
    assert report.total_cost_usd == pytest.approx(
        (100 * 5 + 10 * 25 + 1000 * 0.5 + 200 * 6.25) / MTOK
    )


def test_source_override_and_missing_file(tmp_path):
    rollout = _write_rollout(
        tmp_path,
        model="claude-opus-5",
        result={"agent_result": {"n_input_tokens": 10, "n_output_tokens": 1}},
    )
    report = recompute_rollout_cost(rollout, source="otel_usage")
    assert (
        report.source == "otel_usage"
        and report.n_records == 0
        and report.total_cost_usd is None
    )


# ------------------------------------------------------------------ writers


def test_write_report_and_update_result(tmp_path):
    ex = _anthropic_exchange(
        "claude-opus-5", {"input_tokens": MTOK, "output_tokens": 0}
    )
    rollout = _write_rollout(tmp_path, model="claude-opus-5", exchanges=[ex])
    report = recompute_rollout_cost(rollout)
    out = write_cost_report(report)
    data = json.loads(out.read_text())
    assert data["pricing_version"] == PRICING_VERSION
    assert data["total_cost_usd"] == pytest.approx(5.0)
    assert data["usage_totals"]["input_tokens"] == MTOK

    update_result_json(report)
    result = json.loads((rollout / "result.json").read_text())
    assert result["final_metrics"]["total_cost_usd"] == pytest.approx(5.0)
    assert result["agent_result"]["price_source"] == PRICING_VERSION
    assert result["agent_result"]["cost_source"] == "llm_trajectory"
    assert result["agent_result"]["cost_history"][0] == {
        "cost_usd": 1.0,
        "price_source": "litellm",
        "replaced_at": report.computed_at,
    }


# ------------------------------------------------------------------ CLI


def test_cli_cost_command(tmp_path):
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    ex = _anthropic_exchange(
        "claude-opus-5", {"input_tokens": MTOK, "output_tokens": MTOK}
    )
    rollout = _write_rollout(tmp_path, model="claude-opus-5", exchanges=[ex])
    runner = CliRunner()
    res = runner.invoke(app, ["cost", str(rollout)])
    assert res.exit_code == 0, res.output
    assert (
        "30.0000 USD" in res.output
        and "llm_trajectory" in res.output
        and "recorded 1.0000" in res.output
    )

    res = runner.invoke(app, ["cost", str(tmp_path), "--json", "--write"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload[0]["total_cost_usd"] == pytest.approx(30.0)
    assert (rollout / "cost_recompute.json").is_file()

    res = runner.invoke(app, ["cost", "--list-tables"])
    assert res.exit_code == 0, res.output
    assert (
        PRICING_VERSION in res.output
        and LITELLM_OBSERVED_FP_2026_09.version in res.output
    )

    res = runner.invoke(app, ["cost", str(rollout), "--pricing", "bogus"])
    assert res.exit_code == 2
