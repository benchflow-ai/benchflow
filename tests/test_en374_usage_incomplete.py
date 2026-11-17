"""A count that misses calls says so, and so does the cost priced from it.

Review of PR #122 (F2b-R122, 2026-11-13, M1): turn 1 of a Claude Code trial
over ACP reported its usage; turn 2 logged one response 0.2 s in, and the
sandbox was lost before the first 30-second copy of the session log. The
trial recorded turn 1's tokens and their list price as its whole, in
result.json, ATIF, summary.json and the run card, with nothing saying they
were partial. 0.9.3 marks such a count; so does this line now:
``usage_incomplete: true`` with ``usage_incomplete_reason``, and
``cost_incomplete: true`` beside a cost priced from it. summary.json and the
run card count such trials (``trials_usage_incomplete``,
``trials_cost_incomplete``). Every value here is made up.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from benchflow.rollout import _session_mirror
from benchflow.rollout._cost import (
    INCOMPLETE_CODEX_ACP,
    INCOMPLETE_MIRROR,
    INCOMPLETE_UNREPORTED,
    mark_incomplete_usage,
)
from tests.test_codex_acp_tokens import THREE_CALLS, _codex_log, _summed
from tests.test_en374_salvage import OPUS, OPUS_RATES, _rollout, _tokens, _usage
from tests.test_en374_session_mirror import LOST, _acp_rollout

TURN_1 = _usage(1500, 0, 3000, 120)
TURN_2 = _usage(20, 3000, 400, 300)


def _respond(log, message_id: str, usage: dict) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "assistant",
                    "requestId": "req_" + message_id,
                    "message": {"id": message_id, "model": OPUS, "usage": usage},
                }
            )
            + "\n"
        )


async def _cut_off_after_a_reported_turn(tmp_path, monkeypatch):
    """R122's probe: turn 1 reports, turn 2 is cut off 0.2 s in, well before
    the first 30-second copy, and the sandbox is lost."""
    rollout, trial, sandbox, home = _acp_rollout(
        tmp_path, monkeypatch, "claude-agent-acp"
    )
    monkeypatch.setattr(_session_mirror, "MIRROR_INTERVAL_SEC", 30.0)
    rollout._resolved_prompts = ["Build the site.", "Now add tests."]
    object.__setattr__(rollout._config, "model", OPUS)
    log = home / ".claude" / "projects" / "-app" / "session-1.jsonl"

    async def prompts(_client, session, prompt_list, *_args, **_kwargs):
        session.record_user_prompt(prompt_list[0])
        _respond(log, "msg_1", TURN_1)
        session.record_prompt_usage(
            {
                "inputTokens": 1500,
                "outputTokens": 120,
                "cachedReadTokens": 0,
                "cachedWriteTokens": 3000,
                "totalTokens": 4620,
            }
        )
        session.record_user_prompt(prompt_list[1])
        _respond(log, "msg_2", TURN_2)
        await asyncio.sleep(0.2)
        sandbox.lost = True
        raise RuntimeError(LOST)

    monkeypatch.setattr(rollout._planes, "execute_prompts", prompts)
    with pytest.raises(RuntimeError, match=LOST):
        await rollout.execute()
    rollout._env = None
    await rollout._cleanup_before_teardown()
    return rollout, trial


@pytest.mark.asyncio
async def test_a_cut_off_turn_no_log_counted_is_incomplete_in_every_record(
    tmp_path, monkeypatch
):
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow.evaluation import RUN_CARD_USAGE_FIELDS
    from benchflow.models import RolloutResult

    rollout, trial = await _cut_off_after_a_reported_turn(tmp_path, monkeypatch)
    rollout._build_result()
    data = json.loads((trial / "result.json").read_text())
    agent = data["agent_result"]

    # What turn 1 reported, and only that: the count is short by turn 2.
    assert (agent["n_input_tokens"], agent["n_output_tokens"]) == (1500, 120)
    assert agent["n_cache_write_tokens"] == 3000
    assert agent["usage_incomplete"] is True
    assert agent["usage_incomplete_reason"] == INCOMPLETE_UNREPORTED
    # Its cost is the list price of that part: said to be partial too.
    assert agent["cost_usd"] == pytest.approx(
        1500 * OPUS_RATES[0] + 120 * OPUS_RATES[1] + 3000 * OPUS_RATES[3]
    )
    assert agent["cost_source"] == "list_price_estimate"
    assert agent["cost_incomplete"] is True
    extra = data["final_metrics"]["extra"]
    assert extra["usage_incomplete"] is True
    assert extra["usage_incomplete_reason"] == INCOMPLETE_UNREPORTED
    assert extra["cost_incomplete"] is True
    loaded = RolloutResult.load(trial)
    assert loaded.usage_incomplete is True
    assert loaded.usage_incomplete_reason == INCOMPLETE_UNREPORTED

    summary = usage_summary({trial.name: data})
    assert summary["trials_usage_incomplete"] == 1
    assert summary["trials_cost_incomplete"] == 1
    assert summary["total_tokens"] == agent["total_tokens"]
    assert {"trials_usage_incomplete", "trials_cost_incomplete"} <= set(
        RUN_CARD_USAGE_FIELDS
    )


@pytest.mark.asyncio
async def test_turns_that_all_reported_are_not_marked(tmp_path, monkeypatch):
    from benchflow._utils.evaluation_results import usage_summary

    rollout, trial, _sandbox, _home = _acp_rollout(
        tmp_path, monkeypatch, "claude-agent-acp"
    )
    object.__setattr__(rollout._config, "model", OPUS)

    async def prompts(_client, session, prompt_list, *_args, **_kwargs):
        session.record_user_prompt(prompt_list[0])
        session.record_prompt_usage(
            {"inputTokens": 1500, "outputTokens": 120, "totalTokens": 1620}
        )
        return [], 0

    monkeypatch.setattr(rollout._planes, "execute_prompts", prompts)
    await rollout.execute()
    await rollout._cleanup_before_teardown()
    rollout._build_result()
    data = json.loads((trial / "result.json").read_text())
    agent = data["agent_result"]
    assert agent["n_output_tokens"] == 120
    for key in ("usage_incomplete", "usage_incomplete_reason", "cost_incomplete"):
        assert key not in agent
        assert key not in data["final_metrics"]["extra"]
    summary = usage_summary({trial.name: data})
    assert summary["trials_usage_incomplete"] == 0
    assert summary["trials_cost_incomplete"] == 0


@pytest.mark.asyncio
async def test_a_codex_acp_count_with_no_log_is_incomplete(tmp_path, monkeypatch):
    """codex-acp reports each turn's last model call; with no Codex log left
    to count the others, the trial's tokens are short."""
    home = tmp_path / "sandbox-home"
    home.mkdir()
    rollout, _trial = _rollout(
        tmp_path, monkeypatch, agent="codex-acp", harness="acp", home=home
    )
    reported = {
        "usage_source": "agent_native_acp",
        "n_input_tokens": 828,
        "n_output_tokens": 1,
        "n_cache_read_tokens": 12672,
        "n_cache_creation_tokens": None,
        "total_tokens": 828 + 1 + 12672,
        "cost_usd": None,
        "price_source": None,
    }
    rollout._native_usage_metrics = dict(reported)
    rollout._usage_metrics = dict(reported)

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert usage["n_input_tokens"] == 828
    assert usage["usage_incomplete"] is True
    assert usage["usage_incomplete_reason"] == INCOMPLETE_CODEX_ACP


@pytest.mark.asyncio
async def test_a_copy_that_caught_up_counts_as_complete(tmp_path, monkeypatch):
    """The turn ended and its last copy ran; the sandbox was lost after it,
    so teardown counts the copy, which holds every call: not marked."""
    rollout, _trial, sandbox, home = _acp_rollout(tmp_path, monkeypatch, "codex-acp")
    mirror = _session_mirror.start_session_mirror(rollout)
    assert mirror is not None
    _codex_log(home, THREE_CALLS)
    await mirror.end_turn()
    assert mirror.current is True
    sandbox.lost = True
    rollout._env = None

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert _tokens(usage) == _summed(THREE_CALLS)
    assert rollout._mirror_placed == {"codex-sessions": True}
    assert "usage_incomplete" not in usage


@pytest.mark.asyncio
async def test_a_copy_cut_by_the_lost_sandbox_is_incomplete(tmp_path, monkeypatch):
    """The sandbox was lost before the turn's last copy: the copy is what
    the earlier ticks caught, and the count from it says so."""
    rollout, _trial, sandbox, home = _acp_rollout(tmp_path, monkeypatch, "codex-acp")
    mirror = _session_mirror.start_session_mirror(rollout)
    assert mirror is not None
    _codex_log(home, THREE_CALLS)
    await mirror.sync()  # a tick
    sandbox.lost = True
    await mirror.end_turn()  # its copy fails: the sandbox is gone
    assert mirror.current is False
    rollout._env = None

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert rollout._mirror_placed == {"codex-sessions": False}
    assert usage["usage_incomplete"] is True
    assert usage["usage_incomplete_reason"] == INCOMPLETE_MIRROR


def test_a_proxied_count_is_complete_and_the_marker_is_idempotent():
    proxied = SimpleNamespace(
        _usage_metrics={"usage_source": "provider_response", "total_tokens": 10},
        _usage_report_missing=True,
        _usage_salvaged=False,
        _usage_incomplete_reason=None,
    )
    mark_incomplete_usage(proxied)
    assert "usage_incomplete" not in proxied._usage_metrics

    login = SimpleNamespace(
        _usage_metrics={"usage_source": "agent_native_acp", "total_tokens": 10},
        _usage_report_missing=True,
        _usage_salvaged=False,
        _usage_incomplete_reason=None,
    )
    mark_incomplete_usage(login)
    first = dict(login._usage_metrics)
    mark_incomplete_usage(login)
    assert login._usage_metrics == first
    assert first["usage_incomplete_reason"] == INCOMPLETE_UNREPORTED
    # No tokens at all is unknown, not a partial count: never marked.
    nothing = SimpleNamespace(
        _usage_metrics={"usage_source": "unavailable"},
        _usage_report_missing=True,
        _usage_salvaged=False,
        _usage_incomplete_reason=None,
    )
    mark_incomplete_usage(nothing)
    assert "usage_incomplete" not in nothing._usage_metrics


@pytest.mark.asyncio
async def test_the_trial_export_and_bench_results_carry_the_marker(
    tmp_path, monkeypatch
):
    """R122B (N1): result.json carried the marker and the trial export did
    not, so `bench results --json` showed R122's cut-off count (4,620 of the
    8,340 tokens the turns used) as priced and whole, and so did the table."""
    import re
    from pathlib import Path

    import jsonschema
    from typer.testing import CliRunner

    import benchflow as bf
    from benchflow.cli.main import app
    from benchflow.job_export import SCHEMA_MINOR

    rollout, trial = await _cut_off_after_a_reported_turn(tmp_path, monkeypatch)
    rollout._build_result()
    agent = json.loads((trial / "result.json").read_text())["agent_result"]
    assert agent["total_tokens"] == 4620
    marker = {
        "usage_incomplete": True,
        "usage_incomplete_reason": INCOMPLETE_UNREPORTED,
        "cost_incomplete": True,
        "cost_usd_provider": agent.get("cost_usd_provider"),
        "n_calls_cost_estimated": agent.get("n_calls_cost_estimated"),
        "cost_source": "list_price_estimate",
        "cost_usd": agent["cost_usd"],
        "total_tokens": 4620,
    }

    usage = bf.load_trial(trial).to_json_dict()["usage"]
    assert {key: usage.get(key) for key in marker} == marker

    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/reference/schemas/benchflow-trial.v1.schema.json"
        ).read_text()
    )
    assert set(marker) <= set(schema["$defs"]["UsageExport"]["properties"])
    assert SCHEMA_MINOR["trial"] == 5

    runner = CliRunner()
    wide = {"COLUMNS": "300"}
    shown = runner.invoke(app, ["results", str(trial), "--json"], env=wide)
    assert shown.exit_code == 0, shown.output
    document = json.loads(shown.stdout)
    jsonschema.validate(document, schema)
    assert document["schema_minor"] == 5
    assert {key: document["usage"].get(key) for key in marker} == marker

    table = runner.invoke(app, ["results", str(trial.parent)], env=wide)
    assert table.exit_code == 0, table.output
    plain = " ".join(re.sub("[─-╿]", " ", table.stdout).split())
    # "~": no provider's bill; "+": the count missed calls, so it is a floor.
    assert f"~${agent['cost_usd']:.4f}+" in plain
    assert (
        "+ 1 trial(s) counted only some of their calls, so their tokens and "
        f"cost are a floor: {INCOMPLETE_UNREPORTED}"
    ) in plain

    job = bf.load_job(trial.parent)
    assert "1 rollout(s) counted only some of their calls" in job.to_markdown()
