"""A cache-write count no answer reported is null, never 0 (0.9.3).

0.9.3 read six cache-write keys and recorded 0 when none was present, so an
OpenAI chat answer or a MiMo answer said "0 cache writes" where it said
nothing. The count is now null, DashScope's key is read too, and the cost
is a floor only where it matters: on a route whose list price has a
cache-write rate (GPT-6, MiniMax, qwen3.8-flash) and that no provider bill
covers. On a route without one (MiMo, Kimi, DeepSeek) a written token costs
what an input token does, so the cost stays whole.

The MiMo and GPT-6 usage blocks are X4's (2026-11-14): the MiMo answer from
X4's upstream recorder (evidence/outside/rc-harvey-lab-harness-acp-mimo-
hello-default-run, as LiteLLM's callback receives it), the GPT-6 Responses
usage from X4's TLS capture of gpt-6.1-sol (evidence/wire-audit.json, cell
rc-codex-acp-acp-codex-wire-low-run). The DashScope block follows its
documented explicit-cache shape; no capture of it exists yet (EN-663).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytest

from benchflow.providers.litellm_logging import (
    extract_usage_from_trajectory,
    trajectory_from_litellm_callback_log,
)

# X4's MiMo answer, as LiteLLM's callback hands it over (chat-shaped usage).
_MIMO_USAGE = {
    "prompt_tokens": 6005,
    "completion_tokens": 38,
    "total_tokens": 6043,
    "prompt_tokens_details": {"cached_tokens": 5888},
    "completion_tokens_details": {"reasoning_tokens": 17},
}
# X4's TLS capture of gpt-6.1-sol on the Responses API: it names its writes.
_GPT6_RESPONSES_USAGE = {
    "input_tokens": 12890,
    "input_tokens_details": {
        "cached_tokens": 11264,
        "cache_write_tokens": 0,
        "cache_write_tokens_details": {"ttl_30m_tokens": 0, "ttl_12h_tokens": 0},
    },
    "output_tokens": 60,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 12950,
}
# X4's rel Pi MiMo answer (evidence/outside/rel-pi-acp-acp-mimo-hello-default-
# run/0d2f11b8-....response.raw), as the provider sent it.
_MIMO_PI_USAGE = {
    "completion_tokens": 52,
    "prompt_tokens": 1396,
    "total_tokens": 1448,
    "completion_tokens_details": {"reasoning_tokens": 33},
    "prompt_tokens_details": {"cached_tokens": 1280},
}
# The same call's tokens in a chat-completions answer that names no writes.
_GPT6_CHAT_USAGE = {
    "prompt_tokens": 12890,
    "completion_tokens": 60,
    "total_tokens": 12950,
    "prompt_tokens_details": {"cached_tokens": 11264},
}


def _record(model: str, usage: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "event": "success",
        "request_model": model,
        "provider_model": model,
        "request": {
            "method": "POST",
            "path": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role": "user", "content": "hi"}]},
        },
        "response": {"model": model, "usage": usage},
        "usage": usage,
        "response_cost": 0.0066644,
        "start_time": "2026-11-14T23:34:06",
        "end_time": "2026-11-14T23:34:10",
        **extra,
    }


def _usage(*records: dict[str, Any]) -> dict[str, Any]:
    trajectory = trajectory_from_litellm_callback_log(
        "\n".join(json.dumps(r) for r in records),
        session_id="session",
        agent_name="codex-acp",
    )
    return extract_usage_from_trajectory(trajectory)


def _agent_result(usage: dict[str, Any], tmp_path) -> dict[str, Any]:
    from benchflow.rollout import _build_rollout_result

    _build_rollout_result(
        tmp_path,
        task_name="hello",
        rollout_name="hello__0",
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
        timing={"agent": 4.0},
        **usage,
    )
    return json.loads((tmp_path / "result.json").read_text())["agent_result"]


def test_a_priced_route_without_a_write_count_is_null_and_a_floor(tmp_path):
    usage = _usage(_record("openai/gpt-6.1-sol", _GPT6_CHAT_USAGE))

    assert usage["n_cache_creation_tokens"] is None
    assert usage["n_cache_read_tokens"] == 11264
    assert usage["n_input_tokens"] == 1626
    assert usage["total_tokens"] == 12950
    assert usage["usage_details"]["cache_writes"] == {
        "tokens": None,
        "named_tokens": 0,
        "inside_input": True,
        "models": ["openai/gpt-6.1-sol"],
        "priced_models": ["openai/gpt-6.1-sol"],
        "cost_floor": True,
    }
    agent = _agent_result(usage, tmp_path)
    assert agent["n_cache_write_tokens"] is None
    assert agent["cost_usd"] == pytest.approx(0.0066644)
    assert agent["cost_incomplete"] is True
    assert "cache-write rate" in agent["cost_incomplete_reason"]
    assert "usage_incomplete" not in agent


def test_an_unpriced_route_without_a_write_count_is_null_and_whole(tmp_path):
    usage = _usage(_record("mimo-v2.6-flash", _MIMO_USAGE))

    assert usage["n_cache_creation_tokens"] is None
    assert usage["n_cache_read_tokens"] == 5888
    assert usage["usage_details"]["cache_writes"] == {
        "tokens": None,
        "named_tokens": 0,
        "inside_input": True,
        "models": ["mimo-v2.6-flash"],
        "priced_models": [],
        "cost_floor": False,
    }
    agent = _agent_result(usage, tmp_path)
    assert agent["n_cache_write_tokens"] is None
    assert "cost_incomplete" not in agent


def test_an_answer_that_names_its_writes_gives_the_exact_count(tmp_path):
    record = _record("openai/gpt-6.1-sol", _GPT6_RESPONSES_USAGE)
    record["request"]["path"] = "/v1/responses"
    usage = _usage(record)

    # X4's capture names 0 writes: a known 0, not an unknown.
    assert usage["n_cache_creation_tokens"] == 0
    assert usage["n_cache_read_tokens"] == 11264
    assert usage["n_input_tokens"] == 1626
    assert "usage_details" not in usage
    assert "cost_incomplete" not in _agent_result(usage, tmp_path)


def test_dashscope_s_write_key_is_read():
    usage = _usage(
        _record(
            "dashscope/qwen3.8-flash",
            {
                "prompt_tokens": 4000,
                "completion_tokens": 20,
                "total_tokens": 4020,
                "prompt_tokens_details": {
                    "cached_tokens": 1024,
                    "cache_creation_input_tokens": 512,
                },
            },
        )
    )

    assert usage["n_cache_creation_tokens"] == 512
    assert usage["n_cache_read_tokens"] == 1024
    assert usage["n_input_tokens"] == 4000 - 1024 - 512


def test_one_call_without_a_write_count_leaves_the_total_unknown():
    usage = _usage(
        _record("openai/gpt-6.1-sol", _GPT6_RESPONSES_USAGE),
        _record("openai/gpt-6.1-sol", _GPT6_CHAT_USAGE),
    )

    assert usage["n_cache_creation_tokens"] is None
    assert usage["n_cache_write_5m_tokens"] is None
    assert usage["usage_details"]["cache_writes"]["cost_floor"] is True


def test_a_provider_s_bill_is_whole_without_a_write_count(tmp_path):
    """OpenRouter's own bill includes whatever it charged for writes."""
    from benchflow._utils.result_metadata import incomplete_usage_fields

    record = _record(
        "openrouter/openai/gpt-6.1-sol",
        {**_GPT6_CHAT_USAGE, "cost": 0.0071},
        model_group="benchflow-openrouter-openai-gpt-6.1-sol",
        provider_cost=0.0071,
    )
    usage = _usage(record)

    assert usage["cost_source"] == "provider"
    assert usage["cost_usd"] == pytest.approx(0.0071)
    assert usage["n_cache_creation_tokens"] is None
    writes = usage["usage_details"]["cache_writes"]
    assert writes["priced_models"] == ["openrouter/openai/gpt-6.1-sol"]
    assert writes["cost_floor"] is False
    assert "cost_incomplete" not in incomplete_usage_fields(usage)


def test_an_anthropic_answer_keeps_its_write_count():
    usage = _usage(
        _record(
            "anthropic/claude-sonnet-5-5",
            {
                "input_tokens": 10,
                "output_tokens": 3,
                "cache_read_input_tokens": 2,
                "cache_creation_input_tokens": 7,
            },
        )
    )

    assert usage["n_cache_creation_tokens"] == 7
    assert "usage_details" not in usage


def _prompt_total(tmp_path, *records: dict[str, Any]) -> int | None:
    """ATIF's ``total_prompt_tokens`` in the trial's result.json."""
    _agent_result(_usage(*records), tmp_path)
    result = json.loads((tmp_path / "result.json").read_text())
    return result["final_metrics"]["total_prompt_tokens"]


@pytest.mark.parametrize(
    ("model", "usage", "prompt"),
    [
        ("mimo-v2.6-flash", _MIMO_PI_USAGE, 1396),
        ("openai/gpt-6.1-sol", _GPT6_CHAT_USAGE, 12890),
    ],
)
def test_atif_s_prompt_total_stays_known_without_a_write_count(
    tmp_path, model, usage, prompt
):
    """RUSAGE (#261, M1): an answer that names no writes holds its cache in
    its prompt count, so its writes are inside the uncached input already
    and the prompt total is known: 116 + 1,280 and 1,626 + 11,264."""
    assert _prompt_total(tmp_path, _record(model, usage)) == prompt


_ANTHROPIC = {
    "input_tokens": 10,
    "output_tokens": 3,
    "cache_read_input_tokens": 2,
    "cache_creation_input_tokens": 7,
}


def test_an_anthropic_answer_s_prompt_total_is_unchanged(tmp_path):
    record = _record("anthropic/claude-sonnet-5-5", _ANTHROPIC)
    assert _prompt_total(tmp_path, record) == 19


def test_a_mixed_trial_s_prompt_total_adds_the_named_writes(tmp_path):
    """The Anthropic answer's 7 writes stand beside its input; the GPT-6
    answer's unknown writes are inside its input."""
    records = (
        _record("anthropic/claude-sonnet-5-5", _ANTHROPIC),
        _record("openai/gpt-6.1-sol", _GPT6_CHAT_USAGE),
    )
    usage = _usage(*records)
    assert usage["n_cache_creation_tokens"] is None
    assert usage["usage_details"]["cache_writes"]["named_tokens"] == 7
    assert _prompt_total(tmp_path, *records) == 10 + 2 + 7 + 12890


def test_an_input_without_its_cache_and_no_write_count_leaves_the_prompt_unknown(
    tmp_path,
):
    """An Anthropic-shaped input count excludes the cache: writes it did not
    name would stand beside it, so the prompt total is unknown."""
    record = _record(
        "anthropic/claude-sonnet-5-5", {"input_tokens": 3, "output_tokens": 5}
    )
    assert _usage(record)["usage_details"]["cache_writes"]["inside_input"] is False
    assert _prompt_total(tmp_path, record) is None


def _mixed_rollout(login_usage: dict[str, int]):
    """A trial whose proxied role answered on GPT-6 chat (no write count) and
    whose login role reported its own usage over ACP."""
    from pathlib import Path
    from types import SimpleNamespace

    from benchflow.acp.session import ACPSession
    from benchflow.rollout import Rollout, RolloutConfig

    rollout = Rollout(RolloutConfig(task_path=Path("dummy-task")))
    rollout._native_usage_proxy_covered = False
    rollout._native_usage_checkpoint = None
    rollout._session = ACPSession("login")
    rollout._session.record_prompt_usage(SimpleNamespace(**login_usage))
    rollout._collect_native_acp_usage()
    rollout._usage_metrics = _usage(_record("openai/gpt-6.1-sol", _GPT6_CHAT_USAGE))
    rollout._finalize_usage_metrics()
    return rollout._usage_metrics


def test_a_proxied_role_s_unknown_writes_stay_null_beside_a_login(tmp_path):
    """RUSAGE (#261, L3): the proxied + login sum read the proxied role's null
    write count as 0, so result.json said 0 cache writes while
    usage_details.cache_writes said unknown."""
    from benchflow._utils.result_metadata import atif_prompt_tokens

    usage = _mixed_rollout(
        dict(
            input_tokens=80,
            output_tokens=20,
            total_tokens=120,
            cached_read_tokens=10,
            cached_write_tokens=10,
        )
    )

    assert usage["n_cache_creation_tokens"] is None
    assert usage["n_input_tokens"] == 1626 + 80
    assert usage["n_cache_read_tokens"] == 11264 + 10
    writes = usage["usage_details"]["cache_writes"]
    assert writes["tokens"] is None
    assert writes["named_tokens"] == 10
    # The proxied prompt holds its writes; the login's 10 stand beside it.
    assert atif_prompt_tokens(usage) == 12890 + 80 + 10 + 10


def test_a_login_without_a_write_count_leaves_the_prompt_unknown():
    from benchflow._utils.result_metadata import atif_prompt_tokens

    usage = _mixed_rollout(dict(input_tokens=80, output_tokens=20, total_tokens=100))

    assert usage["n_cache_creation_tokens"] is None
    assert usage["usage_details"]["cache_writes"]["inside_input"] is False
    assert atif_prompt_tokens(usage) is None


def test_bench_results_cost_line_words_a_cache_write_floor_by_its_cause(tmp_path):
    """RUSAGE (#261 L2, merge level): with #259 in, jobs._cost_line called
    every floor Codex's warm-up."""
    import benchflow as bf
    from benchflow.jobs import _cost_line

    usage = _usage(_record("openai/gpt-6.1-sol", _GPT6_CHAT_USAGE))
    agent = _agent_result(usage, tmp_path / "built")
    trial = tmp_path / "job" / "hello__0a1b2c3d"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "hello",
                "rollout_name": trial.name,
                "rewards": {"reward": 1.0},
                "agent_name": "codex-acp",
                "model": "gpt-6.1-sol",
                "started_at": "2026-11-14T23:34:06+00:00",
                "finished_at": "2026-11-14T23:34:13+00:00",
                "agent_result": agent,
            }
        )
    )
    line = _cost_line(bf.load_job(tmp_path / "job").trials)

    assert "Codex" not in line
    assert line.endswith(
        "1 rollout(s) left out cache writes their answers did not report (the "
        "route prices them), so the total is a floor"
    )
