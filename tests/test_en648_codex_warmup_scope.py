"""EN-648: a Codex trial's tokens say they count its generating calls only.

X4 captured real Codex subscription WebSocket traffic outside BenchFlow on
ten wire cells (rel and rc ``codex-acp-acp-codex-wire-{low,medium,high,xhigh,
max}-run``). Each connection first sends ``response.create`` with
``generate: false``; the provider answers that warm-up with a usage of its
own. The two generating responses match Codex's session log and result.json
exactly (rel low: 25,929 tokens). The session log, BenchFlow's source, never
records the warm-up, and nothing said so: a generating-only total read as all
of the trial's usage. The warm-up's tokens are not observed here (None, never
a figure from outside), and whether the provider bills it is not established,
so its cost is unknown and the trial's cost is a floor.

The session log below is X4's low cell's two generating responses (rc's
hello-wire__59d5bbd6; rel's low cell has the same totals), in the shape Codex
0.159.3 writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchflow.rollout._cost import CODEX_SESSIONS

# rc-codex-acp-acp-codex-wire-low-run, hello-wire__59d5bbd6: Codex's
# token_usage_record lines (input counts the cached tokens).
_X4_GENERATING = [
    {
        "input_tokens": 12890,
        "cached_input_tokens": 11264,
        "cache_write_input_tokens": 0,
        "output_tokens": 60,
        "reasoning_output_tokens": 0,
        "total_tokens": 12950,
    },
    {
        "input_tokens": 12973,
        "cached_input_tokens": 12800,
        "cache_write_input_tokens": 0,
        "output_tokens": 6,
        "reasoning_output_tokens": 0,
        "total_tokens": 12979,
    },
]


def _x4_session_log(trial: Path) -> None:
    rows: list[dict[str, Any]] = [
        {"type": "session_meta", "payload": {"id": "t", "cli_version": "0.159.3"}},
        {"type": "turn_context", "payload": {"model": "gpt-6.1-sol"}},
    ]
    total: dict[str, int] = {}
    for index, usage in enumerate(_X4_GENERATING):
        total = {key: total.get(key, 0) + value for key, value in usage.items()}
        rows.append(
            {
                "type": "token_usage_record",
                "payload": {
                    "response_id": f"resp_{index}",
                    "usage": usage,
                    "thread_token_usage": total,
                },
            }
        )
        rows.append(
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {"last_token_usage": usage, "total_token_usage": total},
                },
            }
        )
    path = trial / CODEX_SESSIONS / "2026" / "11" / "14" / "rollout-t.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _x4_rollout(tmp_path: Path) -> SimpleNamespace:
    from benchflow.task import RolloutPaths

    paths = RolloutPaths(rollout_dir=tmp_path / "trial")
    paths.mkdir()
    _x4_session_log(paths.rollout_dir)
    return SimpleNamespace(
        _config=SimpleNamespace(primary_agent="codex-acp", harness="acp"),
        _native_usage_agents={"codex-acp"},
        _usage_metrics={
            # codex-acp's end-of-turn report: the turn's last call.
            "n_input_tokens": 173,
            "n_output_tokens": 6,
            "n_cache_read_tokens": 12800,
            "total_tokens": 12979,
            "usage_source": "agent_native_acp",
            "cost_usd": None,
            "usage_details": {"thought_tokens": 0},
        },
        _rollout_paths=paths,
    )


@pytest.mark.asyncio
async def test_x4_wire_cell_counts_generating_calls_and_records_the_warmup_apart(
    tmp_path,
):
    from benchflow.rollout._session_log import count_codex_tokens

    rollout = _x4_rollout(tmp_path)
    await count_codex_tokens(rollout)
    metrics = rollout._usage_metrics

    # The generating totals: X4's native session and result.json figure.
    assert metrics["total_tokens"] == 25929
    assert metrics["n_input_tokens"] == 1799
    assert metrics["n_cache_read_tokens"] == 24064
    assert metrics["n_output_tokens"] == 66
    details = metrics["usage_details"]
    assert details["token_count"]["scope"] == (
        "generating calls only; Codex's generate=false warm-up is recorded "
        "separately in usage_details.warmup"
    )
    warmup = details["warmup"]
    assert warmup["request"] == "response.create generate=false"
    # Not observed: unknown, never 0 and never X4's outside figure.
    assert warmup["calls"] is None
    assert warmup["tokens"] is None
    assert warmup["cost_usd"] is None
    assert "not observed" in warmup["reason"]
    assert "not established" in warmup["billing"]
    # The generating scope is whole: no usage_incomplete.
    assert metrics.get("usage_incomplete") is None


def _x4_agent_result(cost: float | None = 0.0066644) -> dict[str, Any]:
    # result.json's agent_result for the cell once its count is recorded.
    from benchflow._utils.result_metadata import codex_warmup

    return {
        "n_input_tokens": 1799,
        "n_output_tokens": 66,
        "n_cache_read_tokens": 24064,
        "total_tokens": 25929,
        "cost_usd": cost,
        "usage_source": "session_log",
        "usage_source_legacy": "agent_native_acp",
        "cost_source": "list_price_estimate",
        "usage_details": {"warmup": codex_warmup()},
    }


def test_an_unpriced_warmup_makes_the_cost_a_floor():
    from benchflow._utils.result_metadata import incomplete_usage_fields

    fields = incomplete_usage_fields(_x4_agent_result())

    assert fields == {
        "cost_incomplete": True,
        "cost_incomplete_reason": (
            "Codex's generate=false warm-up is unpriced: its tokens are not "
            "observed and its billing is not established"
        ),
    }
    # An unknown cost is unknown, not a floor.
    assert incomplete_usage_fields(_x4_agent_result(cost=None)) == {}
    # The same from a result object.
    assert incomplete_usage_fields(SimpleNamespace(**_x4_agent_result())) == fields


def test_the_summary_states_the_scope_and_counts_the_floors():
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow._utils.result_metadata import incomplete_usage_fields

    def row(agent_result: dict[str, Any]) -> dict[str, Any]:
        return {
            "agent_result": {**agent_result, **incomplete_usage_fields(agent_result)}
        }

    proxied = {
        "n_input_tokens": 10,
        "n_output_tokens": 2,
        "total_tokens": 12,
        "cost_usd": 0.001,
        "usage_source": "proxy",
        "usage_source_legacy": "provider_response",
    }
    summary = usage_summary(
        {
            "low": row(_x4_agent_result()),
            "medium": row(_x4_agent_result()),
            "proxied": row(proxied),
        }
    )

    assert summary["total_tokens"] == 2 * 25929 + 12
    assert summary["trials_cost_incomplete"] == 2
    assert summary["usage_scope"] == {
        "generating_calls_only": 2,
        "scope": (
            "token totals count generating calls only; Codex's generate=false "
            "warm-up is recorded per trial in usage_details.warmup (tokens not "
            "observed, cost unknown)"
        ),
    }
    assert usage_summary({"proxied": row(proxied)})["usage_scope"] is None


def test_bench_results_cost_line_reads_the_floor(tmp_path):
    import benchflow as bf
    from benchflow.jobs import _cost_line

    trial = tmp_path / "job" / "hello-wire__59d5bbd6"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "hello-wire",
                "rollout_name": trial.name,
                "rewards": {"reward": 1.0},
                "agent_name": "codex-acp",
                "model": "gpt-6.1-sol",
                "started_at": "2026-11-14T23:34:06+00:00",
                "finished_at": "2026-11-14T23:34:13+00:00",
                "agent_result": {
                    **_x4_agent_result(),
                    "price_source": "list_price",
                },
            }
        )
    )

    assert _cost_line(bf.load_job(tmp_path / "job").trials) == (
        "Cost: $0.0067 over 1 rollout(s); 1 rollout(s) count generating calls "
        "only (Codex's warm-up is unpriced), so the total is a floor"
    )


def test_result_json_states_the_scope_and_the_floor(tmp_path):
    from datetime import datetime

    from benchflow._utils.result_metadata import CODEX_GENERATING_SCOPE, codex_warmup
    from benchflow.rollout import _build_rollout_result

    _build_rollout_result(
        tmp_path,
        task_name="hello-wire",
        rollout_name="hello-wire__59d5bbd6",
        agent="codex-acp",
        agent_name="codex-acp",
        model="gpt-6.1-sol",
        n_tool_calls=1,
        prompts=["write ok"],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={"agent": 6.0},
        n_input_tokens=1799,
        n_output_tokens=66,
        n_cache_read_tokens=24064,
        total_tokens=25929,
        cost_usd=0.0066644,
        usage_source="session_log",
        usage_details={
            "token_count": {
                "source": "codex-session-log",
                "scope": CODEX_GENERATING_SCOPE,
            },
            "warmup": codex_warmup(),
        },
    )
    agent = json.loads((tmp_path / "result.json").read_text())["agent_result"]

    assert agent["total_tokens"] == 25929
    assert agent["cost_usd"] == 0.0066644
    assert agent["cost_incomplete"] is True
    assert agent["cost_incomplete_reason"].startswith("Codex's generate=false warm-up")
    assert "usage_incomplete" not in agent
    assert agent["usage_details"]["token_count"]["scope"] == CODEX_GENERATING_SCOPE
    assert agent["usage_details"]["warmup"]["tokens"] is None
