"""``bench backfill-cost``: a finished job's tokens and cost, re-derived from
the logs its trials left.

BenchFlow 0.10.0rc1 to rc3 wrote trials with no cost or 0 tokens. The
records here are in the shapes those builds wrote (result.json fields,
config.json, trainer/atif.json, summary.json, as the rc3 cost matrix of
2026-11-13 left them); the logs are the content-free copies of real logs in
``tests/fixtures/model_calls`` or lines in Claude Code 2.1.316's and Codex's
own shapes. Expected figures are worked out here from the logs' own numbers
(Codex's last running total, each Claude response's usage) and the
2026-11-04 snapshot's rates, not from the code under test.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from benchflow import cost_backfill as cb

FIXTURES = Path(__file__).parent / "fixtures" / "model_calls"
SONNET = "claude-sonnet-5-5"
# Sonnet 5.5 at the 2026-11-04 snapshot: input, output, cache read, 5-minute
# and 1-hour cache writes.
SONNET_RATES = (2e-6, 10e-6, 0.1e-6, 2.5e-6, 4e-6)
# GPT-6.1 Sol: input, output, cached input.
SOL_RATES = (2e-6, 10e-6, 0.1e-6)


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _lines(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _agent_result(**fields: object) -> dict:
    """agent_result as 0.10.0rc3 wrote it (no cost_source, no split)."""
    base = {
        "n_tool_calls": 2,
        "n_skill_invocations": 0,
        "n_prompts": 1,
        "n_input_tokens": 0,
        "n_output_tokens": 0,
        "n_cache_read_tokens": 0,
        "n_cache_creation_tokens": 0,
        "total_tokens": 0,
        "cost_usd": None,
        "usage_source": "unavailable",
        "price_source": None,
    }
    base.update(fields)
    return base


def _trial(
    job: Path,
    name: str,
    *,
    agent: str,
    model: str,
    agent_result: dict,
    harness: str = "acp",
    error: str | None = None,
    error_category: str | None = None,
    effort: dict | None = None,
) -> Path:
    """A trial folder as 0.10.0rc3 wrote it: result.json, config.json and
    trainer/atif.json."""
    trial = job / name
    task = name.split("__")[0]
    result = {
        "task_name": task,
        "rollout_name": name,
        "rewards": {"reward": 0.0} if error is None else None,
        "agent": agent,
        "agent_name": agent,
        "model": model,
        "n_tool_calls": 2,
        "n_prompts": 1,
        "agent_result": agent_result,
        "final_metrics": {
            "total_prompt_tokens": agent_result.get("n_input_tokens"),
            "total_completion_tokens": agent_result.get("n_output_tokens"),
            "total_cached_tokens": agent_result.get("n_cache_read_tokens"),
            "total_cost_usd": agent_result.get("cost_usd"),
        },
        "error": error,
        "error_category": error_category,
        "started_at": "2026-11-13T07:00:01+00:00",
        "finished_at": "2026-11-13T07:01:05+00:00",
    }
    if effort is not None:
        result["effort"] = effort
    _write(trial / "result.json", result)
    _write(
        trial / "config.json",
        {
            "benchflow_version": "0.10.0rc3",
            "task_path": task,
            "agent": agent,
            "model": model,
            "reasoning_effort": None,
            "harness_mode": harness,
            "scenes": [
                {
                    "name": "default",
                    "roles": [{"name": "agent", "agent": agent, "model": model}],
                }
            ],
        },
    )
    _write(
        trial / "trainer" / "atif.json",
        {
            "schema_version": "ATIF-v1.4",
            "session_id": name,
            "agent": {"name": agent, "version": "unknown", "model_name": model},
            "steps": [{"step_id": 1, "source": "user", "message": "task"}],
            "final_metrics": {"total_steps": 1, **result["final_metrics"]},
        },
    )
    return trial


def _job(tmp_path: Path) -> Path:
    job = tmp_path / "jobs" / "j"
    _write(job / "evaluation.json", {"job_name": "j"})
    # summary.json as 0.10.0rc3 wrote it: errored trials left out of the
    # totals, no usage_sources or cost_sources counts.
    (job / "summary.json").write_text(
        json.dumps(
            {
                "job_name": "j",
                "agent": "codex-acp",
                "benchflow_version": "0.10.0rc3",
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": None,
                "telemetry_coverage": 0.0,
            },
            indent=2,
        )
    )
    return job


def _codex_trial(job: Path, name: str = "answer__5b1711e7", **kwargs) -> Path:
    """A codex-acp trial on a ChatGPT login (rc3): the ACP figures are the
    last call's, there is no cost, and Codex's own session log is kept."""
    trial = _trial(
        job,
        name,
        agent="codex-acp",
        model=kwargs.pop("model", "gpt-6.1-sol"),
        agent_result=_agent_result(
            n_input_tokens=6914,
            n_output_tokens=9060,
            n_cache_read_tokens=74496,
            total_tokens=90470,
            usage_source="agent_native_acp",
            usage_details={"thought_tokens": 6062},
        ),
        **kwargs,
    )
    folder = trial / "agent" / "codex-sessions" / "2026" / "11" / "04"
    folder.mkdir(parents=True)
    shutil.copy(FIXTURES / "codex-session.jsonl", folder / "rollout-1.jsonl")
    return trial


def _claude_line(message_id: str, request_id: str, usage: dict, **extra) -> dict:
    """An ``assistant`` line of Claude Code 2.1.316's session log."""
    return {
        "type": "assistant",
        "timestamp": "2026-11-13T07:01:02.000Z",
        "isSidechain": False,
        "requestId": request_id,
        "message": {
            "id": message_id,
            "model": SONNET,
            "stop_reason": "tool_use",
            "usage": usage,
            "content": [],
        },
        **extra,
    }


def _sonnet_usage(fresh: int, write_1h: int, read: int, output: int) -> dict:
    return {
        "input_tokens": fresh,
        "cache_creation_input_tokens": write_1h,
        "cache_read_input_tokens": read,
        "output_tokens": output,
        "output_tokens_details": {"thinking_tokens": 0},
        "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
        "service_tier": "standard",
        "cache_creation": {
            "ephemeral_1h_input_tokens": write_1h,
            "ephemeral_5m_input_tokens": 0,
        },
    }


# The two responses of the rc3 matrix's native Claude trial that hit its
# 60-second limit (forced-timeout__2205b29c), and their list price.
FIRST = _sonnet_usage(2, 3787, 12573, 111)
SECOND = _sonnet_usage(2, 216, 16360, 139)
SESSION_USD = (
    4 * SONNET_RATES[0]
    + 250 * SONNET_RATES[1]
    + (12573 + 16360) * SONNET_RATES[2]
    + (3787 + 216) * SONNET_RATES[4]
)


def _claude_native_timeout(job: Path) -> Path:
    """0.10.0rc3's record of it: 0 tokens beside the session log's cost."""
    trial = _trial(
        job,
        "forced-timeout__2205b29c",
        agent="claude-agent-acp",
        model=SONNET,
        harness="native",
        error="Agent prompt exceeded wall-clock budget 60s",
        error_category="timeout",
        agent_result=_agent_result(
            cost_usd=0.0214133,
            price_source="agent_session_log",
            usage_details={
                "cost_estimate": {
                    "source": "claude-code-session-log",
                    "method": "claude-code-cost",
                    "path": "agent/claude-sessions",
                    "sessions": 1,
                    "responses": 2,
                    "models": {SONNET: 0.021413300000000003},
                    "context_1m": False,
                }
            },
        ),
    )
    _lines(
        trial / "agent" / "claude-sessions" / "-app" / "7c80280b.jsonl",
        [
            {"type": "user", "isSidechain": False},
            # One line per content block: the first block's line carries
            # the output so far, the last the response's whole output.
            _claude_line("msg_1", "req_1", {**FIRST, "output_tokens": 16}),
            _claude_line("msg_1", "req_1", FIRST),
            _claude_line("msg_2", "req_2", SECOND),
            {"type": "cost-state", "totalCostUSD": 0.021413300000000003},
        ],
    )
    return trial


def _snapshot(job: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(p.relative_to(job)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(job.rglob("*"))
        if p.is_file() and not p.name.startswith("backfill-report")
    }


def _line(report: dict, trial: str) -> dict:
    return next(line for line in report["trials"] if line["trial"] == trial)


def test_a_codex_login_trial_counts_every_call_and_gets_a_price(tmp_path):
    """rc3 recorded the last of nine calls (6,914 uncached input, 9,060
    output) and no cost; Codex's running total after the ninth response is
    the trial's: 493,915 input of which 428,800 cached, 10,207 output."""
    job = _job(tmp_path)
    trial = _codex_trial(job)
    before = json.loads((trial / "result.json").read_text())["agent_result"]

    report = cb.backfill_job(job)

    result = json.loads((trial / "result.json").read_text())
    after = result["agent_result"]
    uncached = 493915 - 428800
    assert (
        after["n_input_tokens"],
        after["n_cache_read_tokens"],
        after["n_output_tokens"],
    ) == (uncached, 428800, 10207)
    # OpenAI reports no cache writes: Codex's 0 is no count.
    assert after["n_cache_write_tokens"] is None
    assert after["total_tokens"] == 504122
    expected = uncached * SOL_RATES[0] + 428800 * SOL_RATES[2] + 10207 * SOL_RATES[1]
    assert after["cost_usd"] == pytest.approx(expected, abs=1e-9)
    assert after["cost_source"] == "list_price_estimate"
    assert after["price_table_date"] == "2026-11-04"
    assert (after["usage_source"], after["usage_source_legacy"]) == (
        "session_log",
        "agent_native_acp",
    )
    assert after["usage_details"]["thought_tokens"] == 6354
    assert after["backfill"]["original"] == before
    assert after["backfill"]["filled"] == ["cost_missing", "tokens_recounted"]
    assert after["backfill_version"]["benchflow"]
    assert after["backfilled_at"].endswith("Z")
    # ATIF's terms: the prompt counts the cache.
    for metrics in (
        result["final_metrics"],
        json.loads((trial / "trainer" / "atif.json").read_text())["final_metrics"],
    ):
        assert metrics["total_prompt_tokens"] == 493915
        assert metrics["total_cached_tokens"] == 428800
        assert metrics["total_completion_tokens"] == 10207
        assert metrics["total_cost_usd"] == after["cost_usd"]
        assert metrics["extra"]["n_input_tokens"] == uncached
        assert metrics["extra"]["cost_source"] == "list_price_estimate"
    atif = json.loads((trial / "trainer" / "atif.json").read_text())
    assert atif["final_metrics"]["total_steps"] == 1
    # The effort the calls carried, observed. None was asked for, so the
    # calls are held to the effort the session applied: all nine at xhigh.
    assert result["effort"]["sent"] == {"xhigh": 9}
    assert result["effort"]["applied"] == "xhigh"
    assert result["effort"]["matches"] is True
    line = _line(report, trial.name)
    assert (line["status"], line["source"]) == ("covered", "session_log")
    assert line["before"]["cost_usd"] is None
    assert line["after"]["cost_usd"] == after["cost_usd"]


def test_a_native_claude_timeout_gets_its_tokens_and_keeps_its_cost(tmp_path):
    """rc3 wrote 0 tokens beside a $0.0214133 cost for a native Claude
    trial that hit its limit; the session log has both responses."""
    job = _job(tmp_path)
    trial = _claude_native_timeout(job)

    report = cb.backfill_job(job)

    after = json.loads((trial / "result.json").read_text())["agent_result"]
    assert (
        after["n_input_tokens"],
        after["n_output_tokens"],
        after["n_cache_read_tokens"],
        after["n_cache_write_tokens"],
        after["n_cache_write_1h_tokens"],
        after["n_cache_write_5m_tokens"],
        after["total_tokens"],
    ) == (4, 250, 28933, 4003, 4003, 0, 4 + 250 + 28933 + 4003)
    # The cost was there: kept, and labelled; the snapshot's price of the
    # logged responses is the same figure, so it is dated.
    assert after["cost_usd"] == 0.0214133
    assert after["cost_source"] == "list_price_estimate"
    assert after["price_table_date"] == "2026-11-04"
    assert after["usage_source"] == "session_log"
    assert after["backfill"]["filled"] == ["tokens_missing"]
    line = _line(report, trial.name)
    assert line["status"] == "covered"
    # The logs priced at list price agree with Claude Code's own figure.
    assert line["log_check"]["cost_usd_at_list_price"] == pytest.approx(
        SESSION_USD, abs=1e-9
    )
    assert pytest.approx(0.0214133, abs=1e-9) == SESSION_USD


def test_claude_responses_count_once_across_lines_and_session_files(tmp_path):
    """A response is one (message id, request id): logged once per content
    block, and again in another session file (a resumed copy), it counts
    once, with its whole output. Guards backfill's change to
    ``session_cost._read``, which kept each session's copy (and the last
    line's usage, not the largest)."""
    from benchflow.rollout._salvage import claude_session_usage

    folder = tmp_path / "claude-sessions" / "-app"
    _lines(
        folder / "a.jsonl",
        [
            _claude_line("msg_1", "req_1", {**FIRST, "output_tokens": 16}),
            _claude_line("msg_1", "req_1", FIRST),
        ],
    )
    _lines(
        folder / "b.jsonl",
        [
            _claude_line("msg_1", "req_1", {**FIRST, "output_tokens": 16}),
            _claude_line("msg_2", "req_2", SECOND),
        ],
    )
    _lines(
        folder / "b" / "subagents" / "agent-1.jsonl",
        [_claude_line("msg_3", "req_3", _sonnet_usage(1, 0, 500, 7), isSidechain=True)],
    )

    counted = claude_session_usage(tmp_path / "claude-sessions")

    assert counted is not None
    assert counted["responses"] == 3
    assert counted["n_output_tokens"] == 111 + 139 + 7
    assert counted["n_cache_read_tokens"] == 12573 + 16360 + 500


def _codex_old(folder: Path, name: str, usages: list[dict], repeat: int) -> None:
    """A Codex 0.159 rollout (``token_count`` events only), the event after
    response ``repeat`` sent twice, as Codex sends it again when only its
    rate limits change."""
    total: dict[str, int] = {}
    rows: list[dict] = [{"type": "turn_context", "payload": {"model": "gpt-6.1-sol"}}]
    for index, usage in enumerate(usages):
        total = {key: total.get(key, 0) + value for key, value in usage.items()}
        event = {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {"last_token_usage": usage, "total_token_usage": dict(total)},
            },
        }
        rows += [event, event] if index == repeat else [event]
    _lines(folder / f"rollout-{name}.jsonl", rows)


def test_codex_counts_a_response_once_never_a_repeated_snapshot(tmp_path):
    """Guards backfill's change to ``model_calls._codex_calls``: a
    token_count event repeated with an unmoved running total added its last
    call again, and a response logged in two rollouts counted twice."""
    from benchflow.trajectories.model_calls import _codex_calls

    usage = {"input_tokens": 1000, "cached_input_tokens": 600, "output_tokens": 50}
    old = tmp_path / "old"
    _codex_old(old, "a", [usage, usage], repeat=0)
    assert len(_codex_calls(old)) == 2

    new = tmp_path / "new"
    record = {
        "type": "token_usage_record",
        "payload": {"response_id": "resp_1", "usage": usage},
    }
    _lines(new / "rollout-a.jsonl", [record])
    _lines(new / "rollout-b.jsonl", [record])
    assert len(_codex_calls(new)) == 1


def _proxy_row(
    prompt: int,
    cached: int,
    output: int,
    cost: float | None,
    figure: float | None = None,
    finish: str | None = "tool_calls",
    route: str = "mimo-v2.6-flash",
) -> dict:
    """One line of the proxy's llm_trajectory.jsonl for a MiMo call; ``cost``
    is the provider's bill on the response (usage.cost), when it sent one,
    ``figure`` the gateway's list-price figure (metadata.response_cost)."""
    usage: dict = {
        "completion_tokens": output,
        "prompt_tokens": prompt,
        "total_tokens": prompt + output,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
    if cost is not None:
        usage["cost"] = cost
    return {
        "duration_ms": 6000.0,
        "metadata": {
            "request_model": "mimo-v2.6-flash",
            "provider_model": route,
            "call_type": "anthropic_messages",
            "call_purpose": "agent",
            **({"response_cost": figure} if figure is not None else {}),
        },
        "request": {
            "timestamp": "2026-11-13T07:00:10",
            "body": {"model": "mimo-v2.6-flash", "max_tokens": 128000},
        },
        "response": {
            "status_code": 200,
            "timestamp": "2026-11-13T07:00:16",
            "body": {
                # The gateway's model group, as 0.10.0rc3's proxy logged it.
                "model": "runtime-xiaomi-mimo-v2.6-flash",
                "choices": [{"finish_reason": finish}],
                "usage": usage,
            },
        },
    }


OPENROUTER = "openrouter/xiaomi/mimo-v2.6-flash"


def _billed_row(prompt: int, cached: int, output: int, bill: float, **kw) -> dict:
    """A call on OpenRouter, whose answer's usage.cost is its bill (on other
    routes LiteLLM puts its own list price there, which is no bill)."""
    return _proxy_row(prompt, cached, output, bill, route=OPENROUTER, **kw)


def _proxied(job: Path, name: str, rows: list[dict], **fields) -> Path:
    trial = _trial(
        job,
        name,
        agent="claude-agent-acp",
        model="mimo-v2.6-flash",
        agent_result=_agent_result(
            **{
                "n_input_tokens": sum(
                    r["response"]["body"]["usage"]["prompt_tokens"] for r in rows
                ),
                "n_output_tokens": sum(
                    r["response"]["body"]["usage"]["completion_tokens"] for r in rows
                ),
                "usage_source": "provider_response",
                **fields,
            }
        ),
    )
    _lines(trial / "trajectory" / "llm_trajectory.jsonl", rows)
    return trial


def test_a_proxied_trial_is_its_providers_bill_plus_a_list_price(tmp_path):
    """The mixed case: one call's response carried the provider's bill, the
    other only the gateway's list-price figure. The cost is the two,
    labelled an estimate, with the billed part beside it."""
    job = _job(tmp_path)
    trial = _proxied(
        job,
        "answer__mixed",
        [
            _billed_row(11041, 6656, 150, 0.0006745368),
            _proxy_row(12000, 11000, 80, None, figure=0.0002, route=OPENROUTER),
        ],
    )

    cb.backfill_job(job)

    after = json.loads((trial / "result.json").read_text())["agent_result"]
    assert after["cost_usd"] == pytest.approx(0.0006745368 + 0.0002, abs=1e-12)
    assert after["cost_source"] == "list_price_estimate"
    assert after["price_table_date"] == "2026-11-04"
    assert after["cost_usd_provider"] == pytest.approx(0.0006745368)
    assert after["n_calls_cost_estimated"] == 1
    assert (after["usage_source"], after["usage_source_legacy"]) == (
        "proxy",
        "provider_response",
    )
    # The proxy's input counted the cache; the record's does not.
    assert after["n_input_tokens"] == (11041 - 6656) + (12000 - 11000)
    assert after["n_cache_read_tokens"] == 6656 + 11000


def test_a_proxied_call_with_no_price_leaves_the_cost_unknown(tmp_path):
    """A proxied call with neither a bill nor the gateway's figure is not
    priced from its model's name (the pricing step's rule since dc5d552c0):
    the cost stays unknown, the billed part kept as a subtotal."""
    job = _job(tmp_path)
    trial = _proxied(
        job,
        "answer__partial",
        [_billed_row(100, 0, 10, 0.25), _proxy_row(12000, 11000, 80, None)],
    )

    report = cb.backfill_job(job)

    after = json.loads((trial / "result.json").read_text())["agent_result"]
    assert after["cost_usd"] is None
    assert after["cost_usd_provider"] == 0.25
    assert after["n_output_tokens"] == 90
    line = _line(report, trial.name)
    # The record was rewritten (its tokens), its cost still unknown.
    assert (line["status"], line["reason"]) == ("tokens_filled", "no_price")


def test_a_cost_every_call_billed_is_the_providers(tmp_path):
    job = _job(tmp_path)
    trial = _proxied(
        job,
        "answer__billed",
        [_billed_row(100, 0, 10, 0.25), _billed_row(100, 0, 10, 0.5)],
    )

    cb.backfill_job(job)

    after = json.loads((trial / "result.json").read_text())["agent_result"]
    assert (after["cost_usd"], after["cost_source"], after["cost_usd_provider"]) == (
        0.75,
        "provider",
        0.75,
    )
    assert after["n_calls_cost_estimated"] == 0


_NUMBERS = (*cb.TOKEN_FIELDS, "cost_usd")


def _kept_numbers(after: dict, before: dict) -> bool:
    """The same counts and cost, split its way: the cost and output as
    recorded; where the input counted the cache (``split_converted``), input
    plus cache reads and writes the recorded input and total_tokens the sum
    of the parts; otherwise every figure as recorded, but a cache-write 0
    nobody reported is now unknown."""
    if after.get("cost_usd") != before.get("cost_usd") or after.get(
        "n_output_tokens"
    ) != before.get("n_output_tokens"):
        return False
    if (after.get("backfill") or {}).get("split_converted") is True:
        prompt = (
            after["n_input_tokens"]
            + (after.get("n_cache_read_tokens") or 0)
            + (after.get("n_cache_write_tokens") or 0)
        )
        return (
            prompt == before["n_input_tokens"]
            and after["total_tokens"] == prompt + after["n_output_tokens"]
        )
    unknown = ("n_cache_creation_tokens",)
    return all(
        after.get(key) == before[key]
        or (key in unknown and before[key] == 0 and after.get(key) is None)
        for key in _NUMBERS
        if key in before
    )


def test_a_providers_cost_is_never_replaced(tmp_path):
    """A cost labelled provider stays, even 0 beside positive usage; so does
    a 0.10.0rc2 cost that equals the calls' summed bills, which is labelled
    provider (every number kept)."""
    job = _job(tmp_path)
    labelled = _proxied(
        job,
        "answer__labelled",
        [_proxy_row(100, 0, 10, 0.0)],
        cost_usd=0.0,
        cost_source="provider",
    )
    rc2 = _proxied(
        job,
        "answer__rc2",
        [_billed_row(100, 0, 10, 0.125), _billed_row(100, 0, 10, 0.125)],
        cost_usd=0.25,
        price_source="litellm",
    )
    before = {
        trial: json.loads((trial / "result.json").read_text())["agent_result"]
        for trial in (labelled, rc2)
    }

    report = cb.backfill_job(job)

    for trial, earlier in before.items():
        after = json.loads((trial / "result.json").read_text())["agent_result"]
        assert _kept_numbers(after, earlier)
        assert after["backfill"]["original"] == earlier
        line = _line(report, trial.name)
        assert (line["status"], line["reason"]) == ("labelled", "provider_cost")
    after = json.loads((rc2 / "result.json").read_text())["agent_result"]
    assert (after["cost_source"], after["cost_usd_provider"]) == ("provider", 0.25)
    assert (after["usage_source"], after["usage_source_legacy"]) == (
        "proxy",
        "provider_response",
    )


def test_a_zero_cost_beside_log_usage_is_replaced(tmp_path):
    job = _job(tmp_path)
    trial = _claude_native_timeout(job)
    result = json.loads((trial / "result.json").read_text())
    result["agent_result"]["cost_usd"] = 0.0
    _write(trial / "result.json", result)

    cb.backfill_job(job)

    after = json.loads((trial / "result.json").read_text())["agent_result"]
    assert after["cost_usd"] == pytest.approx(SESSION_USD, abs=1e-9)
    assert after["price_table_date"] == "2026-11-04"
    assert after["backfill"]["filled"] == [
        "cost_zero_beside_log_usage",
        "tokens_missing",
    ]
    assert after["backfill"]["original"]["cost_usd"] == 0.0


def test_every_uncovered_trial_says_why(tmp_path):
    job = _job(tmp_path)
    # Gemini on a key, rc3: zeros, no log of its own.
    gemini = _trial(
        job,
        "answer__02ae8382",
        agent="gemini",
        model="gemini-4-argon",
        agent_result=_agent_result(),
    )
    # A model no price table has: its tokens still count.
    unpriced = _codex_trial(job, "answer__unpriced", model="gpt-unlisted")
    rollout = next((unpriced / "agent" / "codex-sessions").rglob("*.jsonl"))
    rollout.write_text(rollout.read_text().replace("gpt-6.1-sol", "gpt-unlisted"))
    # A log cut off in its first line.
    cut = _trial(
        job,
        "answer__cut",
        agent="claude-agent-acp",
        model=SONNET,
        agent_result=_agent_result(),
    )
    path = cut / "agent" / "claude-sessions" / "-app" / "s.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_claude_line("m", "r", FIRST))[:80])
    # Another harness's trial, and a Harbor trial's result.json.
    _write(job / "en67-lane" / "roll.json", {"rollouts": []})
    _write(
        job / "harbor__1" / "result.json", {"trial_name": "harbor__1", "agent_info": {}}
    )
    # A trial stopped before it wrote result.json (rc1/rc2): its logs say
    # what it spent.
    stopped = job / "answer__stopped"
    _write(stopped / "config.json", {"agent": "codex-acp", "model": "gpt-6.1-sol"})
    rollout = stopped / "agent" / "codex-sessions" / "2026" / "11" / "04" / "r.jsonl"
    rollout.parent.mkdir(parents=True)
    shutil.copy(FIXTURES / "codex-session.jsonl", rollout)
    # An oracle made no model call.
    _trial(
        job,
        "answer__oracle",
        agent="oracle",
        model="",
        agent_result=_agent_result(no_usage_reason="no_model_call"),
    )
    report = cb.backfill_job(job)

    # Nothing counted its tokens: rc3's zeros are unknown, not 0 (review of
    # PR #122, I2; tests/test_cost_backfill_unknown.py).
    after_gemini = json.loads((gemini / "result.json").read_text())["agent_result"]
    assert after_gemini["total_tokens"] is None
    assert after_gemini["no_usage_reason"] == "no_logs"
    expected = {
        "answer__02ae8382": ("uncovered", "no_logs"),
        "answer__unpriced": ("tokens_filled", "no_price"),
        "answer__cut": ("uncovered", "logs_truncated"),
        "en67-lane": ("uncovered", "not_benchflow_trial"),
        "harbor__1": ("uncovered", "not_benchflow_trial"),
        "answer__stopped": ("uncovered", "no_result_json"),
        "answer__oracle": ("not_applicable", "no_model_call"),
    }
    for name, (status, reason) in expected.items():
        line = _line(report, name)
        assert (line["status"], line["reason"]) == (status, reason), name
    spent = _line(report, "answer__stopped")["log_check"]
    assert (spent["n_output_tokens"], spent["total_tokens"]) == (10207, 504122)
    assert spent["cost_usd_at_list_price"] == pytest.approx(0.27518, abs=1e-9)
    after = json.loads((unpriced / "result.json").read_text())["agent_result"]
    assert after["cost_usd"] is None
    assert after["total_tokens"] == 504122
    assert report["job"]["reason_counts"]["uncovered:no_logs"] == 1


def _zstd_available() -> bool:
    return shutil.which("zstd") is not None


@pytest.mark.skipif(not _zstd_available(), reason="no zstd program to compress with")
def test_a_zstd_compressed_log_counts_as_the_plain_one(tmp_path):
    plain_job = _job(tmp_path / "plain")
    packed_job = _job(tmp_path / "packed")
    _claude_native_timeout(plain_job)
    trial = _claude_native_timeout(packed_job)
    log = next((trial / "agent" / "claude-sessions").rglob("*.jsonl"))
    subprocess.run(["zstd", "-q", "--rm", str(log)], check=True)
    assert not log.exists() and log.with_name(log.name + ".zst").exists()

    plain = _line(cb.backfill_job(plain_job), trial.name)
    packed = _line(cb.backfill_job(packed_job), trial.name)

    assert packed["status"] == "covered"
    assert packed["after"] == plain["after"]


def test_a_second_run_changes_nothing(tmp_path):
    job = _job(tmp_path)
    _codex_trial(job)
    trial = _claude_native_timeout(job)
    mtime = (trial / "result.json").stat().st_mtime_ns

    cb.backfill_job(job)
    first = _snapshot(job)
    report = cb.backfill_job(job)

    assert _snapshot(job) == first
    # Attempts of a task are ranked by their result's time: it is kept.
    assert (trial / "result.json").stat().st_mtime_ns == mtime
    assert report["job"]["files_written"] == []
    assert {line["reason"] for line in report["trials"]} == {"backfilled_earlier"}


def test_a_dry_run_writes_only_the_report(tmp_path):
    job = _job(tmp_path)
    _codex_trial(job)
    before = _snapshot(job)
    out = tmp_path / "report.json"

    report = cb.backfill_job(job, dry_run=True, report_path=out)

    assert _snapshot(job) == before
    assert not (job / cb.REPORT_FILE).exists()
    assert json.loads(out.read_text()) == report
    assert report["trials"][0]["status"] == "covered"
    assert report["trials"][0]["after"]["cost_usd"] is not None


def test_summary_totals_come_from_the_summary_builder(tmp_path):
    """summary.json's totals, recomputed by its own builder over the
    backfilled results (the timed-out trial counted); its earlier values
    kept."""
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow._utils.result_paths import load_task_results

    job = _job(tmp_path)
    _codex_trial(job)
    _claude_native_timeout(job)
    old = json.loads((job / "summary.json").read_text())

    report = cb.backfill_job(job)

    summary = json.loads((job / "summary.json").read_text())
    fresh = usage_summary(load_task_results(job, include_terminal_errors=True))
    for key, value in fresh.items():
        assert summary[key] == value, key
    assert summary["total_cost_usd"] == pytest.approx(
        report["job"]["after"]["cost_usd"], abs=1e-9
    )
    assert summary["usage_sources"] == {"session_log": 2}
    assert summary["backfill"]["original"]["total_cost_usd"] is None
    assert summary["backfill"]["original"]["total_tokens"] == old["total_tokens"]
    assert "usage_sources" in summary["backfill"]["added"]
    assert report["job"]["summary_json"] == "updated"


def test_the_command_writes_its_report(tmp_path):
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    job = _job(tmp_path)
    _codex_trial(job)
    out = tmp_path / "out.json"

    done = CliRunner().invoke(app, ["backfill-cost", str(job), "--report", str(out)])

    assert done.exit_code == 0, done.output
    assert "1 covered" in done.output
    assert json.loads(out.read_text())["job"]["status_counts"] == {"covered": 1}


def test_a_trial_nothing_counted_is_counted_from_the_proxys_log(tmp_path):
    """An agent whose usage the record says nothing counted (rc3's
    ``unavailable``, 0 tokens) while its calls went through the proxy: the
    proxy's log counts them."""
    job = _job(tmp_path)
    trial = _trial(
        job,
        "answer__unavailable",
        agent="opencode",
        model="mimo-v2.6-flash",
        agent_result=_agent_result(),
    )
    _lines(
        trial / "trajectory" / "llm_trajectory.jsonl",
        [_proxy_row(2619, 0, 10, None, figure=0.00036666)],
    )

    report = cb.backfill_job(job)

    after = json.loads((trial / "result.json").read_text())["agent_result"]
    assert (after["n_input_tokens"], after["n_output_tokens"]) == (2619, 10)
    assert after["cost_usd"] == pytest.approx(0.00036666, abs=1e-12)
    assert after["usage_source"] == "proxy"
    assert _line(report, trial.name)["status"] == "covered"


def test_a_folder_of_jobs_is_refused(tmp_path):
    """One summary.json cannot total several jobs: pass each job folder."""
    job = _job(tmp_path)
    _codex_trial(job)

    with pytest.raises(ValueError, match="holds other jobs"):
        cb.backfill_job(job.parent)


def _claude_login(
    job: Path,
    name: str,
    *,
    method: str | None,
    version: str = "0.10.0rc3",
    cost: float = 0.0214133,
) -> Path:
    """A Claude trial on a login as 0.10.0rc3 wrote it when its cost was
    found: tokens from the end-of-turn reports, the session log's cost,
    ``agent_native_acp`` and no cost_source."""
    details = (
        {
            "cost_estimate": {
                "source": "claude-code-session-log",
                "method": method,
                "models": {SONNET: 0.0214133},
            }
        }
        if method
        else None
    )
    trial = _trial(
        job,
        name,
        agent="claude-agent-acp",
        model=SONNET,
        agent_result=_agent_result(
            n_input_tokens=4,
            n_output_tokens=250,
            n_cache_read_tokens=28933,
            n_cache_creation_tokens=4003,
            total_tokens=33190,
            cost_usd=cost,
            usage_source="agent_native_acp",
            price_source="agent_session_log",
            **({"usage_details": details} if details else {}),
        ),
    )
    config = json.loads((trial / "config.json").read_text())
    config["benchflow_version"] = version
    _write(trial / "config.json", config)
    _lines(
        trial / "agent" / "claude-sessions" / "-app" / "s.jsonl",
        [_claude_line("msg_1", "req_1", FIRST), _claude_line("msg_2", "req_2", SECOND)],
    )
    return trial


def test_a_kept_cost_gets_its_labels_from_the_evidence(tmp_path):
    """0.10.0rc3 and earlier wrote costs with no cost_source and usage
    sources by their old names: an unlabelled cost let a collector take a
    list price over OpenRouter's bill (FDB PR #17). Each kept cost is
    labelled from the evidence, every number unchanged."""
    job = _job(tmp_path)
    # Claude Code's own figure, which the snapshot reproduces from its log,
    # and one it does not.
    own = _claude_login(job, "a__own", method="claude-code-cost")
    off = _claude_login(job, "a__own_off", method="claude-code-cost", cost=0.03)
    listed = _claude_login(job, "a__listed", method="usage-at-list-price")
    early = _claude_login(
        job, "a__rc1", method="usage-at-list-price", version="0.10.0rc1", cost=0.025
    )
    gateway = _proxied(
        job,
        "a__gateway",
        [_proxy_row(11041, 6656, 150, None, figure=0.0005)],
        cost_usd=0.0005,
        price_source="litellm",
        n_cache_read_tokens=6656,
    )
    partly = _proxied(
        job,
        "a__partly",
        [_billed_row(100, 0, 10, 0.25), _proxy_row(100, 0, 10, None, figure=0.1)],
        cost_usd=0.25,
        price_source="litellm",
    )
    # A list price recorded over the provider's bill: kept, and said so.
    overridden = _proxied(
        job,
        "a__overridden",
        [_billed_row(100, 0, 10, 0.5), _billed_row(100, 0, 10, 0.5)],
        cost_usd=0.8,
        price_source="litellm",
    )
    unknown = _trial(
        job,
        "a__unknown",
        agent="goose",
        model="mimo-v2.6-flash",
        agent_result=_agent_result(
            n_input_tokens=10, n_output_tokens=2, cost_usd=0.01, usage_source="odd"
        ),
    )
    trials = (own, off, listed, early, gateway, partly, overridden, unknown)
    before = {
        t: json.loads((t / "result.json").read_text())["agent_result"] for t in trials
    }

    report = cb.backfill_job(job)

    after = {
        t: json.loads((t / "result.json").read_text())["agent_result"] for t in trials
    }
    for trial in trials:
        assert _kept_numbers(after[trial], before[trial]), trial.name
        assert after[trial]["backfill"]["original"] == before[trial]
        assert _line(report, trial.name)["status"] == "labelled", trial.name
    labels = {
        t.name: (
            after[t]["usage_source"],
            after[t]["cost_source"],
            after[t]["price_table_date"],
            after[t].get("cost_source_reason"),
        )
        for t in trials
    }
    assert labels == {
        # Claude Code's own figure is dated only where the snapshot's price
        # of its logged responses is the same figure.
        "a__own": ("end_of_turn", "list_price_estimate", "2026-11-04", None),
        "a__own_off": ("end_of_turn", "list_price_estimate", None, None),
        "a__listed": ("end_of_turn", "list_price_estimate", "2026-11-04", None),
        # 0.10.0rc1 priced before the dated snapshot.
        "a__rc1": ("end_of_turn", "list_price_estimate", None, None),
        "a__gateway": ("proxy", "list_price_estimate", "2026-11-04", None),
        "a__partly": ("proxy", None, None, "provider_bills_cover_some_calls"),
        "a__overridden": ("proxy", "list_price_estimate", "2026-11-04", None),
        "a__unknown": (None, None, None, "legacy_unlabelled"),
    }
    assert after[partly]["cost_usd_provider"] == 0.25
    assert after[unknown]["usage_source_legacy"] == "odd"
    assert after[own]["usage_source_legacy"] == "agent_native_acp"
    assert after[own]["n_cache_write_tokens"] == 4003
    assert set(after[own]["backfill"]["labelled"]) >= {"usage_source", "cost_source"}
    assert "backfilled_at" not in after[own]["backfill"]["labelled"]
    # A proxied record from before the split: its input counts the cache.
    assert after[gateway]["backfill"]["input_includes_cache"] is True
    assert any(
        "summing to $1.000000" in note
        for note in _line(report, "a__overridden")["notes"]
    )
    # ATIF and final_metrics: the old totals, the new provenance.
    result = json.loads((gateway / "result.json").read_text())
    atif = json.loads((gateway / "trainer" / "atif.json").read_text())
    for metrics in (result["final_metrics"], atif["final_metrics"]):
        assert metrics["total_prompt_tokens"] == 11041
        assert metrics["extra"]["usage_source"] == "proxy"
        assert metrics["extra"]["cost_source"] == "list_price_estimate"
    assert report["job"]["status_counts"]["labelled"] == len(trials)

    first = _snapshot(job)
    cb.backfill_job(job)
    assert _snapshot(job) == first


def test_the_legacy_usage_names_agree_with_usage_tracking():
    from benchflow.usage_tracking import _LEGACY_TO_RESULT

    for name, mapped in _LEGACY_TO_RESULT.items():
        assert cb.LEGACY_USAGE_SOURCES[name] == mapped, name


def _results_row(trial: Path) -> dict:
    """A results.jsonl row as the rollout's row writer lays it out, written
    where the trial ran (another host's path)."""
    from benchflow.trajectories.results import _token_usage_from_agent_result

    result = json.loads((trial / "result.json").read_text())
    return {
        "example_id": 0,
        "prompt": [{"role": "user", "content": "task"}],
        "completion": [{"role": "assistant", "content": "done"}],
        "info": {
            "schema_version": 1,
            "task_id": result["task_name"],
            "task_name": result["task_name"],
            "rollout_name": trial.name,
            "rollout_dir": f"/home/runner/jobs/j/{trial.name}",
            "agent": result["agent"],
            "model": result["model"],
            "source": "benchflow",
        },
        "reward": 0.0,
        "error": None,
        "token_usage": _token_usage_from_agent_result(result["agent_result"]),
        "trajectory": [],
    }


def test_results_rows_and_the_run_card_carry_the_backfill(tmp_path):
    """results.jsonl (the trial's and the job's) and run-card.json's
    ``after_run.usage`` show the backfilled figures, their earlier values
    kept; the rows of a trial whose numbers stayed are left as they were."""
    from benchflow.evaluation import RUN_CARD_USAGE_FIELDS
    from benchflow.trajectories.results import _record_to_redacted_json_line

    job = _job(tmp_path)
    codex = _codex_trial(job)
    claude = _claude_login(job, "a__own", method="claude-code-cost")
    rows = {t: _results_row(t) for t in (codex, claude)}
    for trial, row in rows.items():
        (trial / "results.jsonl").write_text(_record_to_redacted_json_line(row) + "\n")
    (job / "results.jsonl").write_text(
        "".join(
            _record_to_redacted_json_line({**row, "example_id": n}) + "\n"
            for n, row in enumerate(rows.values())
        )
    )
    claude_rows = (claude / "results.jsonl").read_bytes()
    claude_line = (job / "results.jsonl").read_text().splitlines()[1]
    _write(
        job / "run-card.json",
        {"agent": {"name": "codex-acp"}, "after_run": {"run_checks": {"ok": True}}},
    )

    report = cb.backfill_job(job)

    after = json.loads((codex / "result.json").read_text())["agent_result"]
    job_rows = [
        json.loads(line) for line in (job / "results.jsonl").read_text().splitlines()
    ]
    for row in (json.loads((codex / "results.jsonl").read_text()), job_rows[0]):
        assert row["token_usage"]["input_tokens"] == after["n_input_tokens"]
        assert row["token_usage"]["output_tokens"] == 10207
        assert row["token_usage"]["total_tokens"] == 504122
        assert row["info"]["backfill"]["cost_usd"] == after["cost_usd"]
        assert (
            row["info"]["backfill"]["original"]["token_usage"]
            == (rows[codex]["token_usage"])
        )
    assert job_rows[0]["example_id"] == 0
    assert (claude / "results.jsonl").read_bytes() == claude_rows
    assert (job / "results.jsonl").read_text().splitlines()[1] == claude_line
    card = json.loads((job / "run-card.json").read_text())
    summary = json.loads((job / "summary.json").read_text())
    assert card["after_run"]["usage"] == {k: summary[k] for k in RUN_CARD_USAGE_FIELDS}
    assert card["after_run"]["usage"]["total_cost_usd"] is not None
    assert card["after_run"]["run_checks"] == {"ok": True}
    assert card["after_run"]["backfill"]["original"] == {"usage": None}
    assert report["job"]["run_card"] == "updated"
    assert set(report["job"]["results_jsonl_written"]) == {
        f"{codex.name}/results.jsonl",
        "results.jsonl",
    }

    first = _snapshot(job)
    again = cb.backfill_job(job)
    assert _snapshot(job) == first
    assert again["job"]["run_card"] == "current"


def test_the_jobs_folder_summary_copy_is_reported_stale_not_written(tmp_path):
    """A job writes summary.json in its folder and the same text in the
    folder above: the backfill writes only inside the job folder and says
    the copy above is stale, with how to refresh it."""
    job = _job(tmp_path)
    _codex_trial(job)
    shutil.copy(job / "summary.json", job.parent / "summary.json")
    copy = (job.parent / "summary.json").read_bytes()

    report = cb.backfill_job(job)

    assert (job.parent / "summary.json").read_bytes() == copy
    held = report["job"]["jobs_root_summary"]
    assert held["state"] == "stale"
    assert held["path"] == str((job.parent / "summary.json").resolve())
    assert held["refresh"].startswith(f"cp {job.resolve() / 'summary.json'} ")


def test_a_cache_inclusive_input_goes_on_the_split(tmp_path):
    """0.10.0rc3's proxied record of a Claude trial on MiMo (the rc3 cost
    matrix's claude-acp-mimo, docker, answer): 22,001 input tokens that count
    the 17,600 cache reads. On its split the input is 4,401; the counts
    and the cost stay, and final_metrics, results.jsonl and summary.json
    follow. Its answers' usage.cost is LiteLLM's own list price on a MiMo
    route, no bill (the gateway's rule since f8ff1f90b): the cost is labelled
    a list-price estimate."""
    from benchflow.trajectories.results import (
        _record_to_redacted_json_line,
        _token_usage_from_agent_result,
    )

    job = _job(tmp_path)
    trial = _proxied(
        job,
        "answer__mimo",
        [
            _proxy_row(11000, 8800, 40, 0.00034349),
            _proxy_row(11001, 8800, 37, 0.00034349),
        ],
        n_cache_read_tokens=17600,
        total_tokens=22078,
        cost_usd=0.00068698,
        price_source="litellm",
    )
    row = _results_row(trial)
    (trial / "results.jsonl").write_text(_record_to_redacted_json_line(row) + "\n")
    shutil.copy(trial / "results.jsonl", job / "results.jsonl")
    dry = tmp_path / "dry"
    shutil.copytree(tmp_path / "jobs", dry)
    dry_job = dry / "j"
    untouched = _snapshot(dry_job)

    cb.backfill_job(dry_job, dry_run=True, report_path=tmp_path / "dry.json")
    report = cb.backfill_job(job)

    assert _snapshot(dry_job) == untouched
    result = json.loads((trial / "result.json").read_text())
    after = result["agent_result"]
    assert (
        after["n_input_tokens"],
        after["n_cache_read_tokens"],
        after["n_cache_write_tokens"],
        after["n_output_tokens"],
        after["total_tokens"],
    ) == (22001 - 17600, 17600, None, 77, 22078)
    assert (after["cost_usd"], after["cost_source"], after["price_table_date"]) == (
        0.00068698,
        "list_price_estimate",
        "2026-11-04",
    )
    assert after["backfill"]["split_converted"] is True
    assert after["backfill"]["original"]["n_input_tokens"] == 22001
    atif = json.loads((trial / "trainer" / "atif.json").read_text())
    for metrics in (result["final_metrics"], atif["final_metrics"]):
        assert metrics["total_prompt_tokens"] == 22001
        assert metrics["total_cached_tokens"] == 17600
        assert metrics["extra"]["n_input_tokens"] == 4401
    assert atif["final_metrics"]["total_steps"] == 1
    usage = _token_usage_from_agent_result(after)
    for path in (trial / "results.jsonl", job / "results.jsonl"):
        written = json.loads(path.read_text())
        assert written["token_usage"] == usage
        assert (
            written["info"]["backfill"]["original"]["token_usage"]
            == (row["token_usage"])
        )
    summary = json.loads((job / "summary.json").read_text())
    assert summary["total_input_tokens"] == 4401
    assert summary["total_tokens"] == 22078
    assert report["job"]["split_counts"] == {"converted": 1}
    assert _line(report, trial.name)["status"] == "labelled"

    first = _snapshot(job)
    cb.backfill_job(job)
    assert _snapshot(job) == first


def test_a_record_without_its_log_or_unlike_it_is_left_with_a_reason(tmp_path):
    """A proxied record with cache writes and no proxy log cannot be split
    (a bridged answer's writes may sit in its input once or twice); a record
    that is not the old parser's output for its log is left too. Both keep
    their input, their old usage_source name and their final_metrics
    totals, and the report says why."""
    job = _job(tmp_path)
    nolog = _proxied(
        job,
        "answer__nolog",
        [_proxy_row(13000, 6000, 50, None)],
        n_cache_read_tokens=6000,
        n_cache_creation_tokens=3000,
        cost_usd=0.125,
    )
    (nolog / "trajectory" / "llm_trajectory.jsonl").unlink()
    disagrees = _proxied(
        job,
        "answer__disagrees",
        [_proxy_row(100, 0, 10, 0.125)],
        n_input_tokens=900,
        n_cache_read_tokens=500,
        cost_usd=0.125,
    )
    before = json.loads((nolog / "result.json").read_text())

    report = cb.backfill_job(job)

    result = json.loads((nolog / "result.json").read_text())
    after = result["agent_result"]
    assert after["n_input_tokens"] == before["agent_result"]["n_input_tokens"]
    assert after["backfill"]["split_converted"] is False
    assert after["backfill"]["split_reason"] == "cache_writes_without_proxy_log"
    assert after["usage_source"] == "provider_response"
    totals = ("total_prompt_tokens", "total_completion_tokens", "total_cached_tokens")
    for key in totals:
        assert result["final_metrics"][key] == before["final_metrics"][key]
    line = _line(report, nolog.name)
    assert any("cache_writes_without_proxy_log" in note for note in line["notes"])
    other = json.loads((disagrees / "result.json").read_text())["agent_result"]
    assert (other["n_input_tokens"], other["backfill"]["split_reason"]) == (
        900,
        "proxy_log_disagrees",
    )
    assert report["job"]["split_counts"] == {
        "left:cache_writes_without_proxy_log": 1,
        "left:proxy_log_disagrees": 1,
    }

    first = _snapshot(job)
    cb.backfill_job(job)
    assert _snapshot(job) == first


def test_an_input_below_its_cache_is_left_with_a_reason():
    """With no log and no cache writes, the split is the input less the
    reads; an input below its reads did not count them, and stays."""
    before = {
        "n_input_tokens": 100,
        "n_output_tokens": 10,
        "n_cache_read_tokens": 500,
        "n_cache_creation_tokens": 0,
        "usage_source": "provider_response",
    }
    after, split = cb._on_the_split({**before, "n_cache_write_tokens": 0}, before, None)
    assert after["n_input_tokens"] == 100
    assert (split["split_converted"], split["split_reason"]) == (
        False,
        "input_below_cache",
    )
    converted, split = cb._on_the_split(
        {**before, "n_input_tokens": 900, "n_cache_write_tokens": 0},
        {**before, "n_input_tokens": 900},
        None,
    )
    assert (converted["n_input_tokens"], split["split_from"]) == (400, "record")


def test_zero_output_on_a_lost_connection_is_flagged_not_filled(tmp_path):
    """opencode on MiMo (rc3 matrix, daytona, answer): its PTY closed mid-turn
    (``pipe_closed``) after two calls that answered empty: 2,619 input
    tokens, 0 output. The 0 stays; the report flags it."""
    job = _job(tmp_path)
    trial = _proxied(
        job,
        "answer__cut",
        [
            _proxy_row(551, 0, 0, 0.00007714, finish="stop"),
            _proxy_row(2068, 0, 0, 0.00028952, finish="stop"),
        ],
        cost_usd=0.00036666,
        price_source="litellm",
    )
    result = json.loads((trial / "result.json").read_text())
    result.update(
        error="PTY closed by the peer: SIGKILL (close_code=1006, socket error: "
        "ClientConnectionResetError('Cannot write to closing transport'))",
        error_category="pipe_closed",
    )
    _write(trial / "result.json", result)

    report = cb.backfill_job(job)

    after = json.loads((trial / "result.json").read_text())["agent_result"]
    assert (after["n_input_tokens"], after["n_output_tokens"]) == (2619, 0)
    assert after["cost_source"] == "list_price_estimate"
    line = _line(report, trial.name)
    assert line["flags"] == ["zero_output_on_transport_loss"]
    assert report["job"]["flag_counts"] == {"zero_output_on_transport_loss": 1}


# Review of #144 at 2402a2fec: each test below failed on that head.


def test_a_labelled_zero_cost_is_kept(tmp_path):
    """B1: a labelled cost is never replaced, a 0 included: only an
    unlabelled 0 beside logged usage is. A 0.10 record's list-price $0 beside
    a session log that prices to $0.0214133 stays $0, flagged."""
    job = _job(tmp_path)
    trial = _claude_native_timeout(job)
    result = json.loads((trial / "result.json").read_text())
    result["agent_result"].update(
        n_input_tokens=4,
        n_output_tokens=250,
        n_cache_read_tokens=28933,
        n_cache_write_tokens=4003,
        n_cache_creation_tokens=4003,
        n_cache_write_5m_tokens=0,
        n_cache_write_1h_tokens=4003,
        total_tokens=33190,
        cost_usd=0.0,
        usage_source="session_log",
        usage_source_legacy="agent_native_acp",
        cost_source="list_price_estimate",
        price_table_date="2026-11-04",
        cost_usd_provider=None,
        n_calls_cost_estimated=None,
    )
    _write(trial / "result.json", result)
    kept = (trial / "result.json").read_bytes()

    report = cb.backfill_job(job)

    assert (trial / "result.json").read_bytes() == kept
    line = _line(report, trial.name)
    assert (line["status"], line["changes_record"]) == ("already_had_cost", False)
    assert line["flags"] == ["zero_cost_beside_log_usage"]


def test_an_unpriceable_unlabelled_zero_is_kept_and_flagged(tmp_path):
    """B1: an rc-era proxied $0 whose calls carry no bill and no gateway
    figure cannot be priced: the 0 stays (never nulled), labelled
    ``zero_cost_unverified``, and the report says the record changed."""
    job = _job(tmp_path)
    trial = _proxied(
        job,
        "answer__zero",
        [_proxy_row(1000, 600, 20, None)],
        n_cache_read_tokens=600,
        cost_usd=0.0,
        price_source="litellm",
    )

    report = cb.backfill_job(job)

    after = json.loads((trial / "result.json").read_text())["agent_result"]
    assert after["cost_usd"] == 0.0
    assert (after["cost_source"], after["cost_source_reason"]) == (
        None,
        "zero_cost_unverified",
    )
    assert after["backfill"]["original"]["cost_usd"] == 0.0
    line = _line(report, trial.name)
    assert line["changes_record"] is True
    assert line["status"] in ("tokens_filled", "labelled")
    assert line["reason"] == "no_price"
    assert "zero_cost_unverified" in line["flags"]


def _bridged_row(prompt: int, read: int, written: int, output: int) -> dict:
    """An answer from Anthropic through LiteLLM's OpenAI chat route
    (LiteLLM 1.103.3): OpenAI-shaped, its prompt_tokens counting the cache
    reads and writes, the Anthropic counts beside them."""
    row = _proxy_row(prompt, read, output, None, figure=0.001)
    row["response"]["body"]["usage"].update(
        cache_read_input_tokens=read, cache_creation_input_tokens=written
    )
    return row


def test_a_bridged_anthropic_answer_splits_reads_and_writes(tmp_path):
    """S1: an answer from Anthropic through LiteLLM's OpenAI route, prompt
    10,000 with 6,000 cache reads and 3,000 writes inside it. rc3's own
    parser (9b148715f, run on this answer) recorded 13,000 input, 6,000
    reads, 3,000 writes and a total of 10,050: the writes counted twice. The
    record is that parser's output for the logged call, so it goes on the
    split the live path gives the call: 1,000 uncached input. A record that
    is not the old parser's output for its log (10,000 input, no writes)
    stays as recorded, under its old usage_source name."""
    from benchflow._utils.evaluation_results import usage_summary

    job = _job(tmp_path)
    rc3 = _proxied(
        job,
        "answer__bridged",
        [_bridged_row(10000, 6000, 3000, 50)],
        n_input_tokens=13000,
        n_cache_read_tokens=6000,
        n_cache_creation_tokens=3000,
        total_tokens=10050,
        cost_usd=0.001,
        price_source="litellm",
    )
    # Another task's trial (summary.json counts one result a task).
    other = _proxied(
        job,
        "other__1",
        [_bridged_row(10000, 6000, 3000, 50)],
        n_cache_read_tokens=6000,
        cost_usd=0.001,
        price_source="litellm",
    )

    cb.backfill_job(job)

    after = json.loads((rc3 / "result.json").read_text())["agent_result"]
    assert (
        after["n_input_tokens"],
        after["n_cache_read_tokens"],
        after["n_cache_write_tokens"],
        after["total_tokens"],
        after["usage_source"],
    ) == (1000, 6000, 3000, 10050, "proxy")
    assert after["backfill"]["split_from"] == "proxy-log"
    assert after["cost_usd"] == 0.001
    kept = json.loads((other / "result.json").read_text())["agent_result"]
    assert (kept["n_input_tokens"], kept["backfill"]["split_reason"]) == (
        10000,
        "proxy_log_disagrees",
    )
    assert kept["usage_source"] == "provider_response"
    summary = usage_summary({"rc3": json.loads((rc3 / "result.json").read_text())})
    assert summary["total_input_tokens"] == 1000
    job_summary = json.loads((job / "summary.json").read_text())
    assert job_summary["total_input_tokens"] == 1000 + 10000
    assert job_summary["trials_input_includes_cache"] == 1


def test_a_run_stopped_between_writes_keeps_the_originals(tmp_path, monkeypatch):
    """S2: trainer/atif.json is written before result.json (the commit
    point). A run stopped between them redoes the trial and keeps ATIF's
    true originals (6,914 uncached input, not the backfilled 493,915); and
    an ATIF document left stale beside a backfilled result.json is brought
    in line."""
    job = _job(tmp_path)
    trial = _codex_trial(job)
    atif_path = trial / "trainer" / "atif.json"
    pristine = json.loads(atif_path.read_text())
    real = cb._rewrite

    def stop_at_result(path, data):
        if path.name == "result.json":
            raise KeyboardInterrupt
        return real(path, data)

    monkeypatch.setattr(cb, "_rewrite", stop_at_result)
    with pytest.raises(KeyboardInterrupt):
        cb.backfill_job(job)
    monkeypatch.setattr(cb, "_rewrite", real)
    assert (
        "backfill"
        not in json.loads((trial / "result.json").read_text())["agent_result"]
    )

    cb.backfill_job(job)

    original = json.loads(atif_path.read_text())["final_metrics"]["extra"]["backfill"][
        "original"
    ]
    assert (
        original["total_prompt_tokens"]
        == pristine["final_metrics"]["total_prompt_tokens"]
    )
    assert original["total_prompt_tokens"] == 6914
    synced = json.loads(atif_path.read_text())
    _write(atif_path, pristine)
    cb.backfill_job(job)
    again = json.loads(atif_path.read_text())
    assert again["final_metrics"]["total_prompt_tokens"] == 493915
    assert again == synced


def test_a_proxy_that_never_reports_cache_reads_counts_them_unknown(tmp_path):
    """S3a: calls whose answers report no cache reads give a null count, as
    the live path does, never 0."""
    job = _job(tmp_path)
    row = _proxy_row(500, 0, 20, None, figure=0.0002)
    del row["response"]["body"]["usage"]["prompt_tokens_details"]
    trial = _proxied(job, "answer__nocache", [row])
    result = json.loads((trial / "result.json").read_text())
    result["agent_result"]["usage_source"] = "provider_response"
    _write(trial / "result.json", result)

    cb.backfill_job(job)

    after = json.loads((trial / "result.json").read_text())["agent_result"]
    assert (after["n_input_tokens"], after["n_output_tokens"]) == (500, 20)
    assert after["n_cache_read_tokens"] is None
    assert after["n_cache_write_tokens"] is None


def test_a_parser_default_zero_cache_write_is_unknown(tmp_path):
    """S3b: rc3 wrote n_cache_creation_tokens 0 for a proxied call that
    reported no cache writes (claude-acp-mimo, opencode-mimo): unknown, so
    null, and with no log to split by the input stays as recorded. Claude
    Code's own 0 (from Anthropic's usage) is a count and stays."""
    job = _job(tmp_path)
    proxied = _proxied(
        job,
        "answer__default",
        [_proxy_row(22001, 17600, 77, 0.00068698)],
        n_cache_read_tokens=17600,
        cost_usd=0.00068698,
        price_source="litellm",
    )
    (proxied / "trajectory" / "llm_trajectory.jsonl").unlink()
    login = _claude_login(job, "a__login", method="claude-code-cost")
    record = json.loads((login / "result.json").read_text())
    record["agent_result"]["n_cache_creation_tokens"] = 0
    _write(login / "result.json", record)

    cb.backfill_job(job)

    after = json.loads((proxied / "result.json").read_text())["agent_result"]
    assert after["n_cache_write_tokens"] is None
    assert after["n_cache_creation_tokens"] is None
    assert after["backfill"]["split_reason"] == "cache_counts_unknown"
    assert after["n_input_tokens"] == 22001
    own = json.loads((login / "result.json").read_text())["agent_result"]
    assert own["n_cache_write_tokens"] == 0


def test_the_proxy_path_keeps_anthropics_cache_write_split(tmp_path):
    """N1: an Anthropic-shaped answer's 5-minute and 1-hour cache writes."""
    job = _job(tmp_path)
    row = _proxy_row(0, 0, 20, None, figure=0.003)
    row["response"]["body"]["usage"] = {
        "input_tokens": 100,
        "cache_read_input_tokens": 500,
        "cache_creation_input_tokens": 300,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 100,
            "ephemeral_1h_input_tokens": 200,
        },
        "output_tokens": 20,
    }
    trial = _trial(
        job,
        "answer__anthropic",
        agent="claude-agent-acp",
        model=SONNET,
        agent_result=_agent_result(),
    )
    _lines(trial / "trajectory" / "llm_trajectory.jsonl", [row])

    cb.backfill_job(job)

    after = json.loads((trial / "result.json").read_text())["agent_result"]
    assert (
        after["n_input_tokens"],
        after["n_cache_read_tokens"],
        after["n_cache_write_tokens"],
        after["n_cache_write_5m_tokens"],
        after["n_cache_write_1h_tokens"],
    ) == (100, 500, 300, 100, 200)


def test_results_rows_are_matched_by_the_trials_path(tmp_path):
    """N2: a job-level row is the trial its rollout_dir ends with, never
    another trial whose folder has the same name further down."""
    from benchflow.trajectories.results import _record_to_redacted_json_line

    job = _job(tmp_path)
    top = _codex_trial(job, "answer__x")
    other = _claude_native_timeout(job)
    nested = job / "sub" / "answer__x"
    nested.parent.mkdir()
    other.rename(nested)
    row = _results_row(top)
    (job / "results.jsonl").write_text(_record_to_redacted_json_line(row) + "\n")

    cb.backfill_job(job)

    written = json.loads((job / "results.jsonl").read_text())
    after = json.loads((top / "result.json").read_text())["agent_result"]
    assert written["token_usage"]["input_tokens"] == after["n_input_tokens"]
    assert written["token_usage"]["output_tokens"] == 10207


def test_a_bill_the_proxy_recorded_is_the_providers(tmp_path):
    """N4, N5: the proxy's own record of the bill (metadata.provider_cost)
    counts as the response's usage.cost does; a provider-labelled cost is
    named for who billed it (openrouter), not the gateway's table."""
    job = _job(tmp_path)
    rows = [_proxy_row(100, 0, 10, None), _proxy_row(100, 0, 10, None)]
    for row, bill in zip(rows, (0.1, 0.2), strict=True):
        row["metadata"].update(
            provider_cost=bill, provider_model="openrouter/xiaomi/mimo-v2.6-flash"
        )
    trial = _proxied(job, "answer__or", rows, cost_usd=0.3, price_source="litellm")

    report = cb.backfill_job(job)

    after = json.loads((trial / "result.json").read_text())["agent_result"]
    assert (after["cost_source"], after["price_source"]) == ("provider", "openrouter")
    assert after["backfill"]["original"]["price_source"] == "litellm"
    assert _line(report, trial.name)["reason"] == "provider_cost"


def test_each_runs_report_is_kept(tmp_path):
    """N7: backfill-report.<UTC>.json per run, the stable name the latest."""
    job = _job(tmp_path)
    _codex_trial(job)

    first = cb.backfill_job(job)
    second = cb.backfill_job(job)

    kept = sorted(p.name for p in job.glob("backfill-report.*.json"))
    assert kept == sorted([first["report_file"], second["report_file"]])
    latest = json.loads((job / cb.REPORT_FILE).read_text())
    assert latest["report_file"] == second["report_file"]
    assert json.loads((job / first["report_file"]).read_text()) == first


def test_an_openrouter_generation_record_bills_an_unbilled_call(tmp_path):
    """FDB PR #17's Argon trial (0.10.0rc2 through OpenRouter, the route
    named only by the model group): 3 of 34 calls answered with no
    usage.cost (aborted streams), and OpenRouter's own generation records
    (trajectory/or-generations.jsonl, from a bill lookup) say each billed $0:
    every call is billed, the recorded cost is the provider's, and the
    provider is OpenRouter, not the vendor its model id names."""
    job = _job(tmp_path)
    rows = [
        _rc2_openrouter_row("gen-a", "google/gemini-4-argon", 4887, 0, 177, 0.011544),
        _rc2_openrouter_row("gen-b", "google/gemini-4-argon", 47355, 0, 448, None),
    ]
    trial = _proxied(
        job, "answer__argon", rows, cost_usd=0.011544, price_source="litellm"
    )
    _lines(
        trial / "trajectory" / "or-generations.jsonl",
        [{"id": "gen-a", "total_cost": 0.011544}, {"id": "gen-b", "total_cost": 0}],
    )

    report = cb.backfill_job(job)

    after = json.loads((trial / "result.json").read_text())["agent_result"]
    assert (after["cost_usd"], after["cost_source"], after["price_source"]) == (
        0.011544,
        "provider",
        "openrouter",
    )
    assert after["n_calls_cost_estimated"] == 0
    assert _line(report, trial.name)["reason"] == "provider_cost"


def _rc2_openrouter_row(
    gen: str, model: str, prompt: int, cached: int, output: int, cost: float | None
) -> dict:
    """A 0.10.0rc2 proxy line of a call through OpenRouter (FDB PR #17's
    or1112b trials): the route is named only by the model group
    (``runtime-openrouter-...``, also the answer's model); provider_model is
    the bare vendor/model; usage.cost is OpenRouter's bill, absent on an
    aborted stream."""
    group = "runtime-openrouter-" + model.replace("/", "-")
    usage: dict = {
        "completion_tokens": output,
        "prompt_tokens": prompt,
        "total_tokens": prompt + output,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
    if cost is not None:
        usage["cost"] = cost
    return {
        "metadata": {
            "request_model": model,
            "provider_model": model,
            "model_group": group,
            "call_type": "acompletion",
            "call_purpose": "agent",
        },
        "request": {"timestamp": "2026-11-12T08:37:41", "body": {"model": model}},
        "response": {
            "status_code": 200,
            "body": {
                "id": gen,
                "model": group,
                "choices": [{"finish_reason": "stop"}],
                "usage": usage,
            },
        },
    }


def _rc2_trial(job: Path, name: str, rows: list[dict], cost: float) -> Path:
    trial = _proxied(job, name, rows, cost_usd=cost, price_source="litellm")
    config = json.loads((trial / "config.json").read_text())
    config["benchflow_version"] = "0.10.0rc2"
    _write(trial / "config.json", config)
    return trial


def test_rc2_openrouter_bills_are_the_providers_and_mimos_figure_is_not(tmp_path):
    """FDB PR #17's or1112b trials (0.10.0rc2 through OpenRouter): every
    call's usage.cost is OpenRouter's bill, though the route shows only in
    the model group; an aborted stream's call carried none, and OpenRouter's
    generation record says it billed $0. Both costs are the provider's,
    unchanged. A MiMo call's usage.cost is LiteLLM's own list price: an
    estimate."""
    job = _job(tmp_path)
    kimi = _rc2_trial(
        job,
        "startup_autology__04c92e95",
        [
            _rc2_openrouter_row("gen-1", "moonshotai/kimi-k3", 80000, 79000, 900, 1.25),
            _rc2_openrouter_row("gen-2", "moonshotai/kimi-k3", 82000, 81000, 700, 1.5),
        ],
        cost=2.75,
    )
    argon = _rc2_trial(
        job,
        "startup_fourier-health__12a34511",
        [
            _rc2_openrouter_row(
                "gen-a", "google/gemini-4-argon", 4887, 0, 177, 0.011544
            ),
            _rc2_openrouter_row("gen-b", "google/gemini-4-argon", 47355, 0, 448, None),
        ],
        cost=0.011544,
    )
    _lines(
        argon / "trajectory" / "or-generations.jsonl",
        [
            {"id": "gen-a", "total_cost": 0.011544, "native_tokens_prompt": 4887},
            {"id": "gen-b", "total_cost": 0, "native_tokens_prompt": 0},
        ],
    )
    mimo = _proxied(
        job,
        "answer__mimo",
        [_proxy_row(11041, 6656, 150, 0.0006745368)],
        n_cache_read_tokens=6656,
        cost_usd=0.0006745368,
        price_source="litellm",
    )

    report = cb.backfill_job(job)

    for trial, cost in ((kimi, 2.75), (argon, 0.011544)):
        after = json.loads((trial / "result.json").read_text())["agent_result"]
        assert (after["cost_usd"], after["cost_source"], after["price_source"]) == (
            cost,
            "provider",
            "openrouter",
        ), trial.name
        assert (after["cost_usd_provider"], after["n_calls_cost_estimated"]) == (
            cost,
            0,
        )
        line = _line(report, trial.name)
        assert (line["status"], line["reason"]) == ("labelled", "provider_cost")
    after = json.loads((mimo / "result.json").read_text())["agent_result"]
    assert after["cost_source"] == "list_price_estimate"


@pytest.mark.parametrize(
    "shape",
    ["benchflow-group", "api-base", "request-model"],
)
def test_an_openrouter_route_in_any_shape_is_the_biller(tmp_path, shape):
    """N5 (#144 re-review): a route the gateway knows as OpenRouter's by
    0.9.x's ``benchflow-openrouter-`` group, by an openrouter.ai
    ``api_base`` (0.9.x's ``openai/<vendor>/<model>`` id), or only by the
    request's model is billed by OpenRouter: its usage.cost is the bill and
    the cost is named ``openrouter``, never the vendor its id starts with."""
    job = _job(tmp_path)
    row = _proxy_row(1000, 0, 20, 0.004)
    meta = row["metadata"]
    if shape == "benchflow-group":
        meta.update(
            provider_model="openai/anthropic/claude-sonnet-5",
            model_group="benchflow-openrouter-anthropic-claude-sonnet-5",
        )
    elif shape == "api-base":
        meta.update(
            provider_model="openai/anthropic/claude-sonnet-5",
            api_base="https://openrouter.ai/api/v1",
        )
    else:
        meta.update(provider_model="anthropic/claude-sonnet-5")
        meta.pop("request_model")
        row["request"]["body"]["model"] = "openrouter/anthropic/claude-sonnet-5"
    trial = _proxied(job, "answer__or", [row], cost_usd=0.004, price_source="litellm")

    cb.backfill_job(job)

    after = json.loads((trial / "result.json").read_text())["agent_result"]
    assert (after["cost_source"], after["price_source"]) == ("provider", "openrouter")


def test_an_rc_era_mixed_cost_says_its_billed_part(tmp_path):
    """3 (#144 re-review): an rc-era cost that is one call's OpenRouter bill
    ($0.25) plus the gateway's list-price figure for another ($0.10) is a
    list-price estimate, its billed part and its estimated calls beside it,
    from the log, as the mixed case is recorded live."""
    job = _job(tmp_path)
    trial = _proxied(
        job,
        "answer__mixed_rc",
        [
            _billed_row(100, 0, 10, 0.25),
            _proxy_row(100, 0, 10, None, figure=0.1, route=OPENROUTER),
        ],
        cost_usd=0.35,
        price_source="litellm",
    )

    cb.backfill_job(job)

    after = json.loads((trial / "result.json").read_text())["agent_result"]
    assert after["cost_usd"] == 0.35
    assert after["cost_source"] == "list_price_estimate"
    assert (after["cost_usd_provider"], after["n_calls_cost_estimated"]) == (0.25, 1)


def test_summary_counts_the_totals_it_cannot_vouch_for(tmp_path):
    """Nit (#144 re-review): summary.json's total_cost_usd adds a provider
    subtotal and an unverified $0 as it adds any cost, and its input total
    an input still counting its cache: beside the total, how many of each."""
    job = _job(tmp_path)
    _proxied(
        job,
        "subtotal__1",
        [_billed_row(100, 0, 10, 0.25), _proxy_row(100, 0, 10, None, figure=0.1)],
        cost_usd=0.25,
        price_source="litellm",
    )
    _proxied(
        job,
        "zero__1",
        [_proxy_row(1000, 600, 20, None)],
        n_cache_read_tokens=600,
        cost_usd=0.0,
        price_source="litellm",
    )
    nolog = _proxied(
        job,
        "nolog__1",
        [_proxy_row(13000, 6000, 50, None)],
        n_cache_read_tokens=6000,
        n_cache_creation_tokens=3000,
        cost_usd=0.125,
    )
    (nolog / "trajectory" / "llm_trajectory.jsonl").unlink()

    cb.backfill_job(job)

    summary = json.loads((job / "summary.json").read_text())
    keys = list(summary)
    at = keys.index("total_cost_usd")
    assert keys[at + 1 : at + 4] == [
        "trials_cost_provider_subtotal",
        "trials_cost_zero_unverified",
        "trials_input_includes_cache",
    ]
    assert [summary[k] for k in keys[at + 1 : at + 4]] == [1, 1, 1]


def test_a_clean_job_gets_no_write(tmp_path):
    """Nit (#144 review at f0c4362d2): an job that needs nothing (its
    records labelled and on the split, summary.json and the run card as the
    job wrote them) is left as it is: the caveat counts, all 0, are no
    reason to rewrite summary.json, so no byte or mtime in the job folder
    changes."""
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow._utils.result_metadata import final_metrics_from_agent_result
    from benchflow._utils.result_paths import load_task_results
    from benchflow.evaluation import RUN_CARD_USAGE_FIELDS

    job = _job(tmp_path)
    trial = _claude_native_timeout(job)
    result = json.loads((trial / "result.json").read_text())
    result["agent_result"].update(
        n_input_tokens=4,
        n_output_tokens=250,
        n_cache_read_tokens=28933,
        n_cache_write_tokens=4003,
        n_cache_creation_tokens=4003,
        n_cache_write_5m_tokens=0,
        n_cache_write_1h_tokens=4003,
        total_tokens=33190,
        usage_source="session_log",
        usage_source_legacy="agent_native_acp",
        cost_source="list_price_estimate",
        price_table_date="2026-11-04",
        cost_usd_provider=None,
        n_calls_cost_estimated=None,
    )
    result["final_metrics"] = final_metrics_from_agent_result(result["agent_result"])
    _write(trial / "result.json", result)
    summary = {
        "job_name": "j",
        **usage_summary(load_task_results(job, include_terminal_errors=True)),
    }
    (job / "summary.json").write_text(json.dumps(summary, indent=2))
    _write(
        job / "run-card.json",
        {"after_run": {"usage": {k: summary.get(k) for k in RUN_CARD_USAGE_FIELDS}}},
    )
    files = {
        str(p.relative_to(job)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(job.rglob("*"))
        if p.is_file()
    }

    report = cb.backfill_job(job, report_path=tmp_path / "report.json")

    assert {
        str(p.relative_to(job)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(job.rglob("*"))
        if p.is_file()
    } == files
    assert report["job"]["files_written"] == []
    assert report["job"]["summary_json"] == "unchanged"
    assert _line(report, trial.name)["status"] == "already_had_cost"


def test_a_cost_the_backfill_finds_billed_names_the_biller(tmp_path):
    """Nit (#144 review at f0c4362d2): a proxied trial with no cost, priced
    entirely from its calls' OpenRouter bills (the route known only by rc2's
    model group), is named for who billed it, as a kept provider cost is:
    openrouter, not provider."""
    job = _job(tmp_path)
    trial = _proxied(
        job,
        "answer__found",
        [
            _rc2_openrouter_row("gen-1", "moonshotai/kimi-k3", 80000, 79000, 900, 1.25),
            _rc2_openrouter_row("gen-2", "moonshotai/kimi-k3", 82000, 81000, 700, 1.5),
        ],
    )

    report = cb.backfill_job(job)

    after = json.loads((trial / "result.json").read_text())["agent_result"]
    assert (after["cost_usd"], after["cost_source"], after["price_source"]) == (
        2.75,
        "provider",
        "openrouter",
    )
    assert _line(report, trial.name)["status"] == "covered"
