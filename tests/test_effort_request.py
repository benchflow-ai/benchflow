"""The effort asked for is applied or refused, never silently downgraded (EN-383, 0.9.3).

Guards the EN-383 findings, which 0.9.2 shares, and a gap of 0.9.2's own:

- ``-m 'gpt-6-astra[ultra]'`` or ``CODEX_CONFIG={"model_reasoning_effort":
  "ultra"}`` ran Astra at medium. codex-acp's model selection
  (``acp/runtime.py`` ``_codex_session_model_id``) read only
  ``reasoning_effort``. It kept the session's current variant when its model
  matched, and otherwise took the first of medium, high, low. For an
  ``--effort`` the session did not offer, 0.9.2 also took medium.
- ``--effort ultra`` was refused outright.
- result.json recorded no applied effort at all (X1). 0.9.3 writes the whole
  ``effort`` block: ``requested``, ``applied``, ``sent`` and ``matches``.

The session states are built from the pinned Codex 0.159.3 catalog
(``agents/codex_models.py``), one ``model[effort]`` id per level, as
codex-acp lists them.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.agents.codex_models import CODEX_CATALOG_VERSION, CODEX_MODEL_EFFORTS
from benchflow.effort_request import (
    CLAUDE_EFFORT_ENV,
    CODEX_CONFIG,
    MODEL_SUFFIX,
    RUN,
    EffortRequestError,
    effort_request,
    settle_effort,
)

ULTRA = json.dumps({"model_reasoning_effort": "ultra"})


def _offered(current: str) -> dict:
    """codex-acp's model state: every catalog model at every level it takes."""
    return {
        "currentModelId": current,
        "availableModels": [
            {"modelId": f"{slug}[{level}]"}
            for slug, (_, levels) in CODEX_MODEL_EFFORTS.items()
            for level in levels
        ],
    }


def _acp(model_state=None, config_options=None):
    from benchflow.acp.client import ACPClient
    from benchflow.acp.session import ACPSession

    session = ACPSession("s1")
    session.model_state = model_state
    session.config_options = list(config_options or [])
    init = MagicMock()
    init.agent_info = None
    acp = AsyncMock(spec=ACPClient)
    acp.initialize = AsyncMock(return_value=init)
    acp.session_new = AsyncMock(return_value=session)
    acp.set_model = AsyncMock()
    acp.set_config_option = AsyncMock()
    return acp, session


async def _connect(tmp_path, acp, *, agent, model, effort, agent_env=None):
    from benchflow.acp.runtime import connect_acp

    with (
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=acp),
    ):
        return await connect_acp(
            env=AsyncMock(),
            agent=agent,
            agent_launch=agent,
            agent_env=agent_env or {},
            sandbox_user=None,
            model=model,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
            reasoning_effort=effort,
        )


# --- the pinned catalog and the one list ---------------------------------------


def test_the_catalog_is_the_pinned_codexs():
    from benchflow.agents.registry import pinned_npm_package

    assert pinned_npm_package("codex")[1] == CODEX_CATALOG_VERSION
    for model in ("gpt-6-astra", "gpt-6.1-sol"):
        assert CODEX_MODEL_EFFORTS[model] == (
            "low",
            ("low", "medium", "high", "xhigh", "max", "ultra"),
        )


def test_every_effort_list_is_the_one_list():
    from benchflow._utils.config import (
        _REASONING_EFFORT_VALUES,
        CLAUDE_EFFORT_LEVELS,
        EFFORT_LEVELS,
    )
    from benchflow.agents import muse_acp_shim
    from benchflow.native_harness.claude_code import CLAUDE_CODE_EFFORTS
    from benchflow.native_harness.codex import CODEX_EFFORTS
    from benchflow.providers import litellm_request_guard
    from benchflow.providers.litellm_config import _PROVIDER_REASONING_EFFORTS

    assert "ultra" in EFFORT_LEVELS
    assert _REASONING_EFFORT_VALUES == CODEX_EFFORTS == _PROVIDER_REASONING_EFFORTS
    assert set(EFFORT_LEVELS) == _REASONING_EFFORT_VALUES
    assert set(CLAUDE_EFFORT_LEVELS) == CLAUDE_CODE_EFFORTS
    # These two run standalone (the model proxy, the Muse shim) and cannot
    # import it; they hold the same levels.
    assert litellm_request_guard._CLAUDE_EFFORTS == CLAUDE_EFFORT_LEVELS
    assert set(muse_acp_shim.EFFORT_LEVELS) <= set(EFFORT_LEVELS)
    for _, levels in CODEX_MODEL_EFFORTS.values():
        assert set(levels) <= set(EFFORT_LEVELS)


# --- every way of asking is the request ----------------------------------------


@pytest.mark.parametrize(
    ("model", "effort", "env", "source"),
    [
        ("gpt-6-astra", "ultra", {}, RUN),
        ("gpt-6-astra[ultra]", None, {}, MODEL_SUFFIX),
        ("gpt-6-astra", None, {"CODEX_CONFIG": ULTRA}, CODEX_CONFIG),
        ("gpt-6.1-sol[ultra]", "ultra", {"CODEX_CONFIG": ULTRA}, None),
    ],
)
def test_each_way_of_asking_is_the_request(model, effort, env, source):
    from benchflow.evaluation import EvaluationConfig
    from benchflow.rollout import RolloutConfig

    asked = effort_request("codex-acp", model, effort, env)
    assert asked.effort == asked.launch == "ultra"
    assert asked.source == (source or f"{MODEL_SUFFIX}, {CODEX_CONFIG}")
    config = RolloutConfig(
        task_path=Path("."),
        agent="codex-acp",
        model=model,
        reasoning_effort=effort,
        agent_env=env,
    )
    assert config.reasoning_effort == "ultra"
    job = EvaluationConfig(
        agent="codex-acp", model=model, reasoning_effort=effort, agent_env=env
    )
    assert job.reasoning_effort == "ultra"


def test_claudes_variable_is_the_request_and_checked_like_effort():
    asked = effort_request(
        "claude-agent-acp", "claude-opus-5-5", None, {CLAUDE_EFFORT_ENV: "MAX"}
    )
    # Claude Code applies it itself; BenchFlow hands the session nothing.
    assert (asked.effort, asked.source, asked.launch) == (
        "max",
        CLAUDE_EFFORT_ENV,
        None,
    )
    with pytest.raises(EffortRequestError, match=CLAUDE_EFFORT_ENV):
        effort_request("claude-agent-acp", "m", None, {CLAUDE_EFFORT_ENV: "extreme"})


# --- refused before launch ------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "effort"), [("gpt-5.5", "ultra"), ("gpt-5.5[max]", None)]
)
def test_a_level_the_model_does_not_take_is_refused_before_launch(model, effort):
    from benchflow.rollout import RolloutConfig

    asked = effort or "max"
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("codex-acp", model, effort, {})
    assert str(caught.value) == (
        f"gpt-5.5 does not take reasoning effort {asked} in the pinned Codex "
        f"0.159.3 (codex-acp); it takes low, medium, high, xhigh"
    )
    with pytest.raises(ValueError, match="does not take reasoning effort"):
        RolloutConfig(
            task_path=Path("."), agent="codex-acp", model=model, reasoning_effort=effort
        )


def test_two_ways_that_differ_are_refused():
    with pytest.raises(EffortRequestError, match="ask for one"):
        settle_effort("codex-acp", "gpt-6-astra[ultra]", "high", {})


def test_effort_ultra_is_a_level():
    from benchflow._utils.config import normalize_reasoning_effort

    assert normalize_reasoning_effort("ULTRA") == "ultra"
    assert settle_effort("codex-acp", "gpt-6-astra", "ultra", {}).effort == "ultra"
    assert settle_effort("codex-acp", "gpt-6.1-sol", "ultra", {}).effort == "ultra"


@pytest.mark.parametrize(
    ("argv", "words"),
    [
        (["-m", "gpt-5.5", "--reasoning-effort", "ultra"], "it takes low, medium"),
        (["-m", "gpt-5.5[ultra]"], "it takes low, medium"),
        (["-m", "gpt-6-astra[ultra]", "--reasoning-effort", "high"], "ask for one"),
    ],
)
def test_bench_eval_run_refuses_before_any_sandbox(tmp_path, monkeypatch, argv, words):
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    started = []
    monkeypatch.setattr(
        "benchflow.cli.main.eval_preflight", lambda **kw: started.append(kw)
    )
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    result = CliRunner().invoke(
        app,
        ["eval", "run", "--tasks-dir", str(tasks), "-a", "codex-acp", *argv],
        env={"COLUMNS": "200", "HARBOR_TELEMETRY": "off"},
    )
    assert result.exit_code == 2, result.output
    assert "Refused before launch" in result.output
    assert words in " ".join(result.output.split())
    assert started == []


# --- the session selects what was asked -----------------------------------------


@pytest.mark.parametrize(
    ("model", "env", "current", "selected"),
    [
        ("gpt-6.1-sol[ultra]", {}, "gpt-6.1-sol[low]", "gpt-6.1-sol[ultra]"),
        ("gpt-6-astra[ultra]", {}, "gpt-6.1-sol[low]", "gpt-6-astra[ultra]"),
        (
            "gpt-6-astra",
            {"CODEX_CONFIG": ULTRA},
            "gpt-6.1-sol[ultra]",
            "gpt-6-astra[ultra]",
        ),
    ],
)
async def test_the_session_selects_the_level_asked_for(
    tmp_path, model, env, current, selected
):
    acp, session = _acp(_offered(current))
    await _connect(
        tmp_path, acp, agent="codex-acp", model=model, effort=None, agent_env=env
    )
    acp.set_model.assert_awaited_once_with(selected)
    assert session.effort_applied == "ultra"


async def test_an_effort_the_session_does_not_offer_is_refused(tmp_path):
    """0.9.2 ran the model's medium variant instead."""
    from benchflow.diagnostics import AgentEffortNotOfferedError

    state = {
        "currentModelId": "gpt-6-astra[low]",
        "availableModels": [
            {"modelId": f"gpt-6-astra[{level}]"} for level in ("low", "medium", "high")
        ],
    }
    acp, _ = _acp(state)
    with pytest.raises(AgentEffortNotOfferedError) as caught:
        await _connect(
            tmp_path, acp, agent="codex-acp", model="gpt-6-astra", effort="xhigh"
        )
    assert str(caught.value) == (
        "agent integration failure [agent_model]: codex-acp does not offer "
        "reasoning effort 'xhigh' for model 'gpt-6-astra'; it offers: low, "
        "medium, high"
    )
    acp.set_model.assert_not_awaited()


async def test_with_nothing_asked_the_session_keeps_its_choice(tmp_path):
    acp, session = _acp(_offered("gpt-6-astra[low]"))
    await _connect(tmp_path, acp, agent="codex-acp", model="gpt-6-astra", effort=None)
    acp.set_model.assert_awaited_once_with("gpt-6-astra[low]")
    assert session.effort_applied == "low"


async def test_an_effort_option_the_agent_does_not_answer_is_not_echoed(tmp_path):
    # The session reported medium before the set; the agent's answer to the
    # set carries no options, so medium is stale, not what it applied.
    option = {
        "id": "effort",
        "currentValue": "medium",
        "options": [{"value": v} for v in ("low", "medium", "high")],
    }
    acp, session = _acp(config_options=[option])
    acp.set_config_option = AsyncMock(return_value={})
    await _connect(tmp_path, acp, agent="opencode", model="m", effort="high")
    acp.set_config_option.assert_any_await("effort", "high")
    assert session.effort_applied is None


# --- recorded, and a mismatch flagged after the run ------------------------------


def _codex_log(trial: Path, effort: str, calls: int) -> None:
    """A Codex 0.159 session rollout: one turn at *effort*, *calls* responses
    (each a token_count event), as copied into agent/codex-sessions."""
    log = trial / "agent/codex-sessions/2026/11/12/rollout-2026-11-12T04-06-48-x.jsonl"
    log.parent.mkdir(parents=True)
    rows = [
        {
            "timestamp": "2026-11-12T04:06:48.000Z",
            "type": "session_meta",
            "payload": {"id": "x", "cli_version": "0.159.3", "source": "exec"},
        },
        {
            "timestamp": "2026-11-12T04:06:49.000Z",
            "type": "turn_context",
            "payload": {"model": "gpt-6-astra", "effort": effort, "summary": "auto"},
        },
    ] + [
        {
            "timestamp": f"2026-11-12T04:07:{10 + n:02d}.000Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {"input_tokens": 100, "output_tokens": 10}
                },
            },
        }
        for n in range(calls)
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _result(trial: Path, *, model: str, env: dict, applied: str | None) -> dict:
    from benchflow.effort_record import effort_block
    from benchflow.rollout._results import _build_rollout_result

    effort = effort_block(
        trial,
        agent="codex-acp",
        model=model,
        reasoning_effort=None,
        agent_env=env,
        applied=applied,
    )
    _build_rollout_result(
        trial,
        task_name="task",
        rollout_name=trial.name,
        agent="codex-acp",
        agent_name="codex-acp",
        model="gpt-6-astra",
        n_tool_calls=1,
        prompts=["task"],
        error=None,
        verifier_error=None,
        trajectory=[{"type": "agent_message", "text": "done"}],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={},
        effort=effort,
    )
    return json.loads((trial / "result.json").read_text())


def test_a_replayed_medium_session_under_an_ultra_request_is_flagged(tmp_path):
    """PR #17's Astra trial, replayed: CODEX_CONFIG asked ultra, the session
    log says each of its 17 responses ran at medium."""
    from benchflow.cli.main import _effort_mismatches
    from benchflow.effort_record import effort_mismatches

    trial = tmp_path / "startup_alpha-sh__5209fe5e"
    _codex_log(trial, "medium", 17)
    result = _result(
        trial, model="gpt-6-astra", env={"CODEX_CONFIG": ULTRA}, applied="medium"
    )
    assert result["effort"] == {
        "requested": "ultra",
        "requested_from": CODEX_CONFIG,
        "applied": "medium",
        "sent": {"medium": 17},
        "sent_source": "codex-session-log",
        "calls_with_effort": 17,
        "calls_with_budget": 0,
        "calls_without_effort": 0,
        "matches": False,
        "reason": "mismatch",
    }
    assert result["warnings"] == [
        "reasoning effort ultra was asked for, but the agent applied medium; the "
        "agent's model calls carried medium (17 calls)"
    ]
    atif = json.loads((trial / "trainer/atif.json").read_text())
    assert atif["extra"]["effort"] == result["effort"]
    assert effort_mismatches({"t": result}) == 1
    assert _effort_mismatches([SimpleNamespace(rollout_dir=trial)]) == 1


def test_an_ultra_session_under_an_ultra_request_matches(tmp_path):
    trial = tmp_path / "startup_alpha-sh__de375d9c"
    _codex_log(trial, "ultra", 34)
    result = _result(trial, model="gpt-6.1-sol[ultra]", env={}, applied="ultra")
    assert result["effort"]["requested_from"] == MODEL_SUFFIX
    assert (result["effort"]["matches"], result["effort"]["sent"]) == (
        True,
        {"ultra": 34},
    )
    assert "warnings" not in result


def _litellm_responses_call(effort, n):
    """The kwargs LiteLLM 1.103.3 hands a callback for a Responses call:
    its own Logging object, filled as ``responses/main.py`` (:1371) fills it
    (``optional_params`` from ``get_optional_params_responses_api``)."""
    import litellm
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.responses.utils import ResponsesAPIRequestUtils
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_responses_api_config(
        model="gpt-6-astra", provider=litellm.LlmProviders.OPENAI
    )
    asked = {"tools": [{"type": "function", "name": "shell", "parameters": {}}]}
    if effort:
        asked["reasoning"] = {"effort": effort, "summary": "auto"}
    params = ResponsesAPIRequestUtils.get_optional_params_responses_api(
        model="gpt-6-astra",
        responses_api_provider_config=config,
        response_api_optional_params=asked,
    )
    logging = Logging(
        model="gpt-6-astra",
        messages=[{"role": "user", "content": f"turn {n}"}],
        stream=False,
        call_type="aresponses",
        start_time=datetime.now(),
        litellm_call_id=f"call-{n}",
        function_id="f",
    )
    logging.update_from_kwargs(
        kwargs={"input": f"turn {n}"},
        model="gpt-6-astra",
        user=None,
        optional_params=dict(params),
        litellm_params={**params, "aresponses": True, "metadata": {}},
        custom_llm_provider="openai",
    )
    return logging.model_call_details


def _through_the_proxy(tmp_path, monkeypatch, efforts):
    """Each call through BenchFlow's proxy logger, then its parser, into a
    trial's trajectory/llm_trajectory.jsonl."""
    import asyncio

    from benchflow.providers.litellm_logging import trajectory_from_litellm_callback_log
    from tests.test_litellm_logging import _callback_namespace

    log = tmp_path / "litellm.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    logger = _callback_namespace()["BenchFlowLiteLLMLogger"]()
    response = {
        "model": "gpt-6-astra",
        "status": "completed",
        "output": [],
        "usage": {},
    }
    for n, effort in enumerate(efforts):
        kwargs = _litellm_responses_call(effort, n)
        asyncio.run(
            logger.async_log_success_event(
                kwargs, response, datetime.now(), datetime.now()
            )
        )
    trajectory = trajectory_from_litellm_callback_log(
        log.read_text(), session_id="s", agent_name="codex-acp"
    )
    trial = tmp_path / "trial"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "trajectory/llm_trajectory.jsonl").write_text(
        "".join(
            json.dumps(e.model_dump(mode="json")) + "\n" for e in trajectory.exchanges
        )
    )
    return trial


def test_a_codex_call_through_the_proxy_keeps_its_effort(tmp_path, monkeypatch):
    """0.9.2's proxy logger kept reasoning_effort, thinking and output_config
    but not the Responses API's ``reasoning``, so every correct Codex run
    through it read sent {} and matches false."""
    from benchflow.effort_record import effort_outcome, sent_efforts

    trial = _through_the_proxy(tmp_path, monkeypatch, ["ultra", "ultra"])
    sent, source, calls = sent_efforts(trial)
    assert (sent, source, calls) == ({"ultra": 2}, "proxy", 2)
    effort = effort_outcome(
        requested="ultra",
        requested_from=RUN,
        applied="ultra",
        sent=sent,
        sent_source=source,
        calls=calls,
    )
    assert (effort["matches"], effort["reason"]) == (True, None)


def test_calls_without_an_effort_through_the_proxy_are_partly_unmeasured(
    tmp_path, monkeypatch
):
    """Fable's RECORDS-3: one call at ultra and 99 with none read as a match."""
    from benchflow.effort_record import effort_outcome, effort_warning, sent_efforts

    trial = _through_the_proxy(tmp_path, monkeypatch, ["ultra"] + [None] * 99)
    sent, source, calls = sent_efforts(trial)
    effort = effort_outcome(
        requested="ultra",
        requested_from=RUN,
        applied="ultra",
        sent=sent,
        sent_source=source,
        calls=calls,
    )
    assert (effort["matches"], effort["reason"]) == (None, "partly unmeasured")
    assert (effort["calls_with_effort"], effort["calls_without_effort"]) == (1, 99)
    assert effort_warning(effort) == (
        "reasoning effort ultra was asked for: 1 of the agent's 100 model calls "
        "carried it and 99 carried no effort, so the effort of those is unmeasured"
    )


@pytest.mark.parametrize("label", ["default", "auto"])
def test_a_session_label_that_is_no_level_is_nothing_applied(label):
    """R383 M1 (0.10's rule here too): a session that answers ``default`` or
    ``auto`` applied no level; the label is kept as said."""
    from benchflow.effort_record import effort_outcome, effort_warning

    effort = effort_outcome(
        requested=None,
        requested_from=None,
        applied=label,
        sent={"medium": 2},
        sent_source="proxy",
        calls=2,
    )
    assert effort["applied"] == label
    assert (effort["matches"], effort["reason"]) == (None, "nothing_to_compare")
    assert effort_warning(effort) is None
    asked = effort_outcome(
        requested="max",
        requested_from=RUN,
        applied=label,
        sent={"max": 2},
        sent_source="proxy",
        calls=2,
    )
    assert (asked["matches"], asked["reason"]) == (True, None)


def test_a_messages_call_records_its_own_effort_not_the_routes_pin():
    """R383 L1: a Bedrock Claude route pins reasoning_effort, which the
    proxy's log copies into each call; an Anthropic Messages call goes
    upstream with its own output_config.effort."""
    from benchflow.effort_record import _body_effort

    pinned = {"reasoning_effort": "high"}
    assert _body_effort({**pinned, "output_config": {"effort": "medium"}}) == "medium"
    assert _body_effort({**pinned, "reasoning": {"effort": "low"}}) == "low"
    assert _body_effort(pinned) == "high"


async def test_summary_counts_a_fresh_trials_effort_mismatch(tmp_path):
    """R383 M3: summary.json counted the effort of resumed trials only; a
    fresh trial's effort lives in its result.json, which the summary payload
    did not copy."""
    from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
    from benchflow.models import RunResult

    task_dir = tmp_path / "lone-task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )
    jobs_dir = tmp_path / "jobs"
    job = Evaluation(
        tasks_dir=task_dir,
        jobs_dir=jobs_dir,
        config=EvaluationConfig(retry=RetryConfig(max_retries=0)),
        job_name="fresh",
    )
    mismatch = {
        "requested": "max",
        "requested_from": RUN,
        "applied": None,
        "sent": {"high": 2},
        "sent_source": "proxy",
        "calls_with_effort": 2,
        "calls_without_effort": 0,
        "matches": False,
        "reason": "mismatch",
    }

    async def run_single(*_args, **_kwargs):
        trial = jobs_dir / "fresh" / "lone-task__fresh"
        trial.mkdir(parents=True, exist_ok=True)
        (trial / "result.json").write_text(json.dumps({"effort": mismatch}))
        return RunResult(
            task_name="lone-task", rewards={"reward": 1.0}, rollout_name=trial.name
        )

    job._run_single_task = run_single
    await job.run()

    summary = json.loads((jobs_dir / "fresh" / "summary.json").read_text())
    assert summary.get("effort_mismatches") == 1, summary


@pytest.mark.parametrize("model", ["deepseek/deepseek-v4-pro", "deepseek-v4-pro"])
@pytest.mark.parametrize("agent", ["claude-agent-acp", "claude"])
def test_an_effort_a_route_turns_into_a_budget_is_refused_before_launch(agent, model):
    """R093 P1, lead ruling: the gateway's DeepSeek route sends Claude Code's
    effort to DeepSeek's Anthropic endpoint as thinking.budget_tokens (max as
    16384) and no effort; the record said max. Refused before launch."""
    with pytest.raises(EffortRequestError) as caught:
        settle_effort(agent, model, "max", {})
    assert str(caught.value) == (
        f"{model} turns reasoning effort into thinking.budget_tokens "
        "(claude-agent-acp): the gateway's deepseek route sends a thinking "
        "budget and no effort (low 1024, medium 2048, high 4096, xhigh 8192, "
        "max 16384), so -e max would run as budget_tokens 16384, not as max"
    )
    assert caught.value.hint == (
        "drop -e: no level reaches this route as asked, and the thinking budget "
        "it sends is recorded"
    )


def test_a_chat_agent_on_deepseek_keeps_its_effort():
    """DeepSeek's chat completions take reasoning_effort: only an Anthropic
    Messages agent's effort is rewritten into a budget."""
    settled = settle_effort("codex-acp", "deepseek/deepseek-v4-pro", "high", {})
    assert settled.effort == "high"


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        ("openrouter/anthropic/claude-opus-4.6", "max"),
        ("openrouter/anthropic/claude-opus-4.8", "max"),
        ("openrouter/anthropic/claude-opus-4.5", "high"),
    ],
)
def test_claude_code_on_a_dotted_4x_id_is_refused_before_launch(model, effort):
    """R093 P2: Claude Code 2.1.280 offers no effort option for OpenRouter's
    dotted 4.x ids ("does not expose reasoning effort config option
    'effort'"), after the sandbox started. OpenRouter lists no dashed Claude
    id (R383C L1), so the refusal names the anthropic/ and Bedrock routes,
    which take the dashed id."""
    anthropic = "anthropic/" + model.rsplit("/", 1)[1].replace("4.", "4-")
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("claude-agent-acp", model, effort, {})
    bedrock = {
        "anthropic/claude-opus-4-5": "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "anthropic/claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
        "anthropic/claude-opus-4-8": "us.anthropic.claude-opus-4-8",
    }[anthropic]
    assert str(caught.value) == (
        f"Claude Code (claude-agent-acp) offers no effort option for {model}, "
        "OpenRouter's dotted id, and OpenRouter lists no dashed Claude id; it "
        f"takes -e {effort} on the anthropic/ route ({anthropic}) or the Bedrock "
        f"route (aws-bedrock/{bedrock})"
    )
    assert caught.value.hint == (
        f"run it as {anthropic} or aws-bedrock/{bedrock}, or drop -e"
    )
    assert settle_effort("claude-agent-acp", anthropic, effort, {}).effort == effort
    # Other agents name the model to the gateway, not to Claude Code.
    assert settle_effort("codex-acp", model, effort, {}).effort == effort


def test_dotted_opus_4_1_takes_no_effort_on_any_route():
    """OpenRouter's dotted Opus 4.1: Anthropic's reference gives Opus 4.1 no
    effort, so it is refused as the model's, on every agent (R383D L1)."""
    with pytest.raises(EffortRequestError) as caught:
        settle_effort(
            "claude-agent-acp", "openrouter/anthropic/claude-opus-4.1", "high", {}
        )
    assert str(caught.value) == (
        "openrouter/anthropic/claude-opus-4.1 takes no reasoning effort "
        "(claude-agent-acp): drop high"
    )
    assert caught.value.hint == "drop -e: the model takes no reasoning effort"


@pytest.mark.parametrize("agent", ["claude-agent-acp", "opencode"])
def test_a_level_the_codex_catalog_does_not_list_is_refused_for_any_agent(agent):
    """R093 P2: gpt-5.5 takes low to xhigh in the pinned Codex catalog;
    -a claude -m openai/gpt-5.5 -e max was forwarded and failed upstream."""
    with pytest.raises(EffortRequestError) as caught:
        settle_effort(agent, "openai/gpt-5.5", "max", {})
    assert str(caught.value) == (
        f"openai/gpt-5.5 does not take reasoning effort max ({agent}); it takes "
        "low, medium, high, xhigh (at most xhigh): ask for one of them"
    )
    # opencode takes no effort on the gateway alias (R383D M1).
    assert (
        settle_effort("claude-agent-acp", "openai/gpt-5.5", "xhigh", {}).effort
        == "xhigh"
    )


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            ("-a", "claude", "-m", "deepseek/deepseek-v4-pro", "-e", "max"),
            "deepseek/deepseek-v4-pro turns reasoning effort into "
            "thinking.budget_tokens (claude-agent-acp)",
        ),
        (
            ("-a", "claude", "-m", "openrouter/anthropic/claude-opus-4.6", "-e", "max"),
            "Claude Code (claude-agent-acp) offers no effort option for "
            "openrouter/anthropic/claude-opus-4.6",
        ),
        (
            ("-a", "claude", "-m", "openai/gpt-5.5", "-e", "max"),
            "openai/gpt-5.5 does not take reasoning effort max (claude-agent-acp)",
        ),
    ],
)
def test_the_dry_run_and_the_run_give_the_same_refusal(
    tmp_path, monkeypatch, args, message
):
    """R093 P1/P2: each passed the dry run and failed after the sandbox
    started (or ran at a budget); now both refuse, with the same words."""
    _shell(
        monkeypatch,
        tmp_path,
        DEEPSEEK_API_KEY="fake-not-a-key",
        OPENROUTER_API_KEY="fake-not-a-key",
        OPENAI_API_KEY="fake-not-a-key",
    )
    texts = []
    for flags in ([], ["--dry-run"]):
        refused, seen = _bench_run(
            tmp_path / ("dry" if flags else "run"), monkeypatch, *args, *flags
        )
        assert refused.exit_code == 2, refused.output
        assert seen == {"preflight": [], "configs": []}
        texts.append(" ".join(refused.output.split()))
    assert message in texts[0]
    assert texts[0] == texts[1]


def test_unknown_is_not_a_match():
    from benchflow.effort_record import effort_outcome

    def matches(applied, sent):
        return effort_outcome(
            requested="max",
            requested_from=RUN,
            applied=applied,
            sent=sent,
            sent_source="proxy" if sent is not None else None,
        )["matches"]

    assert matches("max", None) is None
    assert matches("high", None) is False
    assert matches(None, {"max": 3}) is True
    # The proxy saw calls and none carried an effort: unmeasured, warned.
    assert matches(None, {}) is None


def _outcome(requested, applied, sent, source="proxy"):
    from benchflow.effort_record import effort_outcome, effort_warning

    effort = effort_outcome(
        requested=requested,
        requested_from=RUN if requested else None,
        applied=applied,
        sent=sent,
        sent_source=source if sent is not None else None,
        calls=sum(sent.values()) if sent else None,
    )
    return effort, effort_warning(effort)


def test_an_applied_effort_with_no_calls_observed_is_unmeasured():
    effort, warning = _outcome("max", "max", None)
    assert (effort["matches"], effort["reason"], warning) == (None, "unmeasured", None)


def test_one_call_in_a_hundred_at_another_effort_is_a_mismatch():
    for sent in ({"max": 99, "high": 1}, {"high": 1}):
        effort, warning = _outcome("max", "max", sent)
        assert (effort["matches"], effort["reason"]) == (False, "mismatch")
        assert "high (1 calls)" in (warning or "")


def test_with_no_effort_asked_the_calls_must_agree_with_the_session():
    """PR #17's case: no -e, the session applied medium, the calls carried
    xhigh."""
    effort, warning = _outcome(None, "medium", {"xhigh": 17}, "codex-session-log")
    assert (effort["matches"], effort["reason"]) == (False, "mismatch")
    assert warning == (
        "no reasoning effort was asked for and the agent applied medium, but the "
        "agent's model calls carried xhigh (17 calls)"
    )
    agree, _ = _outcome(None, "medium", {"medium": 17}, "codex-session-log")
    assert (agree["matches"], agree["reason"]) == (True, None)
    nothing, _ = _outcome(None, None, {})
    assert (nothing["matches"], nothing["reason"]) == (None, "nothing_to_compare")


def test_fail_on_effort_mismatch_fails_the_run(tmp_path):
    import typer

    from benchflow.cli.main import _finish_run

    trial = tmp_path / "t__1"
    _codex_log(trial, "medium", 2)
    _result(trial, model="gpt-6-astra[ultra]", env={}, applied="medium")
    result = SimpleNamespace(
        results={"t": SimpleNamespace(error_category=None, rollout_dir=trial)},
        errored=0,
        verifier_errored=0,
        not_run=0,
        score=1.0,
        passed=1,
        total=1,
    )
    request = SimpleNamespace(
        fail_under=None, fail_on=["effort-mismatch"], summary_out=None
    )
    with pytest.raises(typer.Exit) as stop:
        _finish_run(request, result, tmp_path)
    assert stop.value.exit_code == 1


@pytest.mark.parametrize(
    ("model", "effort", "env", "requested_from"),
    [
        ("gpt-6-astra[ultra]", None, {}, MODEL_SUFFIX),
        ("gpt-6-astra", None, {"CODEX_CONFIG": ULTRA}, CODEX_CONFIG),
        ("gpt-6-astra", "ultra", {}, RUN),
    ],
)
async def test_the_rollout_records_the_request_however_asked(
    tmp_path, model, effort, env, requested_from
):
    """Through Rollout.run: result.json's effort.requested is ultra whichever
    way it was asked (0.9.2 wrote no effort block)."""
    from benchflow.rollout import Rollout, RolloutConfig

    task = tmp_path / "t"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n[environment]\n')
    (task / "instruction.md").write_text("Write hello.txt.\n")
    rollout = Rollout(
        RolloutConfig(
            task_path=task,
            agent="codex-acp",
            model=model,
            reasoning_effort=effort,
            agent_env=env,
        )
    )
    trial = tmp_path / "trial"
    trial.mkdir()
    rollout._rollout_dir = trial
    rollout._rollout_name = "t__codex"
    rollout._started_at = datetime.now()
    rollout.setup = AsyncMock()
    rollout.start = AsyncMock()
    rollout.install_agent = AsyncMock(side_effect=RuntimeError("stop before launch"))
    rollout._env = AsyncMock()
    rollout._agent_cwd = "/app"

    async def cleanup():
        rollout._phase = "cleaned"

    rollout.cleanup = AsyncMock(side_effect=cleanup)
    await rollout.run()
    saved = json.loads((trial / "result.json").read_text())
    assert saved["effort"]["requested"] == "ultra"
    assert saved["effort"]["requested_from"] == requested_from


# --- review of 360ca46b8 (lead 10:3xZ) --------------------------------------------


@pytest.mark.parametrize(
    ("agent", "model", "bare", "effort"),
    [
        ("claude-agent-acp", "opus[1m]", "opus[1m]", None),
        ("claude-agent-acp", "claude-sonnet-4-6[1m]", "claude-sonnet-4-6[1m]", None),
        ("codex-acp", "gpt-6-astra[ultra]", "gpt-6-astra", "ultra"),
        ("claude-agent-acp", "claude-opus-4-6[1m][max]", "claude-opus-4-6[1m]", "max"),
    ],
)
def test_a_bracket_suffix_is_read_by_kind(agent, model, bare, effort):
    from benchflow.evaluation import EvaluationConfig
    from benchflow.rollout import RolloutConfig

    settled = settle_effort(agent, model, None, {})
    assert (settled.model, settled.effort) == (bare, effort)
    job = EvaluationConfig(agent=agent, model=model)
    trial = RolloutConfig(task_path=Path("."), agent=agent, model=model)
    assert (job.model, job.reasoning_effort) == (bare, effort)
    assert (trial.model, trial.reasoning_effort) == (bare, effort)


def test_a_suffix_that_is_no_effort_or_size_is_refused():
    with pytest.raises(EffortRequestError, match=r"\[fast\] is neither"):
        settle_effort("codex-acp", "gpt-6-astra[fast]", None, {})


def test_claudes_variable_and_effort_that_differ_are_refused():
    with pytest.raises(EffortRequestError, match="more than one reasoning effort"):
        settle_effort("claude-agent-acp", "m", "high", {CLAUDE_EFFORT_ENV: "low"})


@pytest.mark.parametrize(
    ("effort", "env"),
    [("ultra", {}), (None, {CLAUDE_EFFORT_ENV: "none"})],
)
def test_a_level_claude_code_does_not_take_is_refused_before_launch(effort, env):
    with pytest.raises(EffortRequestError, match="does not take"):
        settle_effort("claude-agent-acp", "claude-opus-5-5", effort, env)


def test_the_proxys_and_opencodes_effort_settings_are_requests():
    from benchflow.effort_request import OPENCODE_CONFIG

    asked = effort_request(
        "openhands", "m", None, {"BENCHFLOW_REASONING_EFFORT": "high"}
    )
    assert (asked.effort, asked.source) == ("high", "BENCHFLOW_REASONING_EFFORT")
    config = json.dumps({"agent": {"build": {"reasoningEffort": "xhigh"}}})
    asked = effort_request("opencode", "p/m", None, {"OPENCODE_CONFIG_CONTENT": config})
    assert (asked.effort, asked.source) == ("xhigh", OPENCODE_CONFIG)


def test_an_invalid_bedrock_thinking_effort_fails_before_any_sandbox():
    from benchflow.providers.litellm_config import _bedrock_thinking_effort

    env = {"BENCHFLOW_BEDROCK_THINKING_EFFORT": "extreme"}
    with pytest.raises(EffortRequestError, match="no Bedrock thinking effort"):
        settle_effort(
            "claude-agent-acp", "aws-bedrock/us.anthropic.claude-opus-4-8", None, env
        )
    with pytest.raises(ValueError, match="must be one of"):
        _bedrock_thinking_effort("us.anthropic.claude-opus-4-8", env)


@pytest.mark.parametrize(
    ("agent", "model", "effort", "who", "levels"),
    [
        (
            "antigravity",
            "gemini-3.1-pro",
            "max",
            "the Antigravity CLI (agy)",
            "low, medium, high",
        ),
        (
            "antigravity",
            "gemini-3.1-pro",
            "minimal",
            "the Antigravity CLI (agy)",
            "low, medium, high",
        ),
        (
            "antigravity",
            "gemini-3.1-pro",
            "ultra",
            "the Antigravity CLI (agy)",
            "low, medium, high",
        ),
        (
            "muse-code",
            "muse-1",
            "none",
            "Muse",
            "minimal, low, medium, high, xhigh, max, ultra",
        ),
        (
            "opencode",
            "anthropic/claude-opus-5-5",
            "minimal",
            "anthropic/claude-opus-5-5",
            "low, medium, high, xhigh, max",
        ),
        (
            "pi-acp",
            "openrouter/anthropic/claude-opus-5.5",
            "ultra",
            "openrouter/anthropic/claude-opus-5.5",
            "low, medium, high, xhigh, max",
        ),
        # R383 M2: LiteLLM's Bedrock map sent Opus 4.6's xhigh as max.
        (
            "openhands",
            "aws-bedrock/us.anthropic.claude-opus-4-6-v1",
            "xhigh",
            "aws-bedrock/us.anthropic.claude-opus-4-6-v1",
            "low, medium, high, max",
        ),
        (
            "claude-agent-acp",
            "anthropic/claude-opus-4-6",
            "xhigh",
            "anthropic/claude-opus-4-6",
            "low, medium, high, max",
        ),
        # R383 L3: Claude Code refused these itself, after its sandbox started.
        (
            "claude-agent-acp",
            "aws-bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
            "max",
            "aws-bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
            "low, medium, high",
        ),
        (
            "claude-agent-acp",
            "aws-bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
            "xhigh",
            "aws-bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
            "low, medium, high",
        ),
        # R383 L4: OpenRouter's dotted ids.
        (
            "pi-acp",
            "openrouter/anthropic/claude-opus-4.8",
            "ultra",
            "openrouter/anthropic/claude-opus-4.8",
            "low, medium, high, xhigh, max",
        ),
        (
            "opencode",
            "openrouter/anthropic/claude-opus-4.8",
            "minimal",
            "openrouter/anthropic/claude-opus-4.8",
            "low, medium, high, xhigh, max",
        ),
    ],
)
def test_a_known_downgrade_is_refused_before_launch(agent, model, effort, who, levels):
    """F2b 18:46Z (lead ruling 1): a level a path is known to run as another
    or not to take is refused before launch, naming what it takes and the
    highest, never run lower with a warning (rc3 ran agy at high for max and
    recorded "provider clamp: antigravity max→high")."""
    with pytest.raises(EffortRequestError) as caught:
        settle_effort(agent, model, effort, {})
    top = levels.split(", ")[-1]
    assert str(caught.value) == (
        f"{who} does not take reasoning effort {effort} ({agent}); it takes "
        f"{levels} (at most {top}): ask for one of them"
    )


@pytest.mark.parametrize(
    ("agent", "model", "effort"),
    [
        ("antigravity", "gemini-3.1-pro", "high"),
        # Muse runs ultra at xhigh only where its provider lacks it: unseen,
        # so left to the calls to show.
        ("muse-code", "muse-1", "ultra"),
        ("codex-acp", "anthropic/claude-opus-5-5", "max"),
        ("claude-agent-acp", "aws-bedrock/us.anthropic.claude-opus-4-8", "max"),
    ],
)
def test_a_level_the_path_takes_is_run_as_asked(agent, model, effort):
    """No clamp is assumed: Bedrock's Claude 4.x ids take max (the LiteLLM
    1.88.0rc1 limit that lowered it to high is gone, EN-501)."""
    from benchflow.providers.litellm_config import resolve_litellm_route

    env = {"AWS_BEARER_TOKEN_BEDROCK": "fake-not-a-key", "AWS_REGION": "us-east-1"}
    assert settle_effort(agent, model, effort, env).source == RUN
    if model.startswith("aws-bedrock/"):
        from benchflow.effort_request import with_route_effort

        sent = with_route_effort(agent, model, effort, env)
        route = resolve_litellm_route(model, sent)
        assert route.litellm_params["reasoning_effort"] == effort


def test_summary_counts_the_unmeasured():
    from benchflow.effort_record import effort_unmeasured

    results = {
        "a": {"effort": {"requested": "max", "reason": "partly unmeasured"}},
        "b": {"effort": {"requested": None, "reason": "unmeasured"}},
    }
    assert effort_unmeasured(results) == 1


async def test_codex_applied_is_the_variant_the_session_took(tmp_path):
    acp, session = _acp(_offered("gpt-6.1-sol[low]"))
    await _connect(
        tmp_path, acp, agent="codex-acp", model="gpt-6-astra", effort="ultra"
    )
    acp.set_model.assert_awaited_once_with("gpt-6-astra[ultra]")
    assert session.model_state["currentModelId"] == "gpt-6.1-sol[low]"
    assert session.effort_applied == "ultra"


async def test_with_nothing_asked_codex_runs_the_models_own_default(tmp_path):
    acp, session = _acp(_offered("gpt-6.1-sol[low]"))
    await _connect(tmp_path, acp, agent="codex-acp", model="gpt-6-astra", effort=None)
    acp.set_model.assert_awaited_once_with("gpt-6-astra[low]")
    assert session.effort_applied == "low"


def test_every_ladder_is_the_one_list():
    from benchflow._utils.config import EFFORT_LEVELS
    from benchflow.agents import antigravity_acp_shim
    from benchflow.providers.litellm_config import _BEDROCK_EFFORT_LADDER

    assert EFFORT_LEVELS[1:7] == _BEDROCK_EFFORT_LADDER
    assert set(antigravity_acp_shim._EFFORT_ALIASES) == set(EFFORT_LEVELS)


def _matrix_run(tmp_path, monkeypatch, entries: str, *flags: str, agent="oracle"):
    from typer.testing import CliRunner

    from benchflow.cli.main import app
    from benchflow.evaluation import Evaluation
    from tests.test_eval_artifact_cli import _write_rollout, _write_task

    tasks = tmp_path / "tasks"
    _write_task(tasks / "task-a")
    # The dry run loads each task as the run does (EN-632): a task.toml
    # task needs its instruction.md.
    (tasks / "task-a" / "instruction.md").write_text("Do the task.\n")
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("models:\n" + entries)
    seen: list = []

    async def fake_run(self):
        seen.append(self._config)
        _write_rollout(self._jobs_dir / self._job_name / "task-a__abc", "task-a")
        return SimpleNamespace(
            job_name=self._job_name,
            passed=1,
            failed=0,
            errored=0,
            verifier_errored=0,
            total=1,
            score=1.0,
            score_excl_errors=1.0,
        )

    monkeypatch.setattr(Evaluation, "run", fake_run)
    monkeypatch.setattr("benchflow.cli.main.eval_preflight", lambda **kw: None)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(tasks),
            "--agent",
            agent,
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--matrix",
            str(matrix),
            *flags,
        ],
        env={"COLUMNS": "200", "HARBOR_TELEMETRY": "off"},
    )
    return result, seen


def test_a_matrix_entrys_effort_and_harness_are_honoured(tmp_path, monkeypatch):
    """0.9.2 loaded reasoning_effort, effort and harness from a --matrix entry
    and never read them."""
    from benchflow.effort_request import MATRIX

    result, seen = _matrix_run(
        tmp_path,
        monkeypatch,
        "  astra:\n    model: gpt-6-astra\n    agent: codex-acp\n"
        "    reasoning_effort: ultra\n    harness: native\n"
        "  sol:\n    model: gpt-6.1-sol[ultra]\n    agent: codex-acp\n",
        "-m",
        "gpt-5.5",
        agent="codex-acp",
    )
    assert result.exit_code == 0, result.output
    assert [(c.model, c.reasoning_effort, c.harness) for c in seen] == [
        ("gpt-6-astra", "ultra", "native"),
        ("gpt-6.1-sol", "ultra", "acp"),
    ]
    assert seen[0].reasoning_effort_source == MATRIX


@pytest.mark.parametrize(
    ("entry", "flags", "words"),
    [
        (
            "    model: gpt-5.5\n    agent: codex-acp\n    reasoning_effort: ultra\n",
            (),
            "gpt-5.5 does not take reasoning effort ultra",
        ),
        (
            "    model: gpt-6-astra\n    agent: codex-acp\n    reasoning_effort: ultra\n",
            ("--reasoning-effort", "high"),
            "differs from the run's --effort high",
        ),
    ],
)
def test_a_matrix_entry_is_refused_before_any_trial(
    tmp_path, monkeypatch, entry, flags, words
):
    result, seen = _matrix_run(tmp_path, monkeypatch, "  bad:\n" + entry, *flags)
    assert result.exit_code == 2, result.output
    assert words in " ".join(result.output.split())
    assert seen == []


# --- review of 24b59cf40 ----------------------------------------------------------


@pytest.mark.parametrize(
    "agent", ["pi-acp", "gemini", "openhands", "deepagents", "mimo"]
)
def test_a_proxy_effort_never_reaches_a_session_and_is_refused_where_nothing_applies_it(
    agent,
):
    env = {"BENCHFLOW_REASONING_EFFORT": "high"}
    with pytest.raises(EffortRequestError, match=f"nothing applies it to {agent}"):
        settle_effort(agent, "openai/gpt-5.5", None, env)
    azure = {**env, "AZURE_RESOURCE": "r", "AZURE_API_KEY": "k"}
    assert (
        settle_effort(agent, "azure-foundry-openai/gpt-5.5", None, azure).effort is None
    )


def test_a_bedrock_thinking_effort_that_differs_from_effort_is_refused():
    with pytest.raises(EffortRequestError, match="more than one reasoning effort"):
        settle_effort(
            "claude-agent-acp",
            "aws-bedrock/us.anthropic.claude-opus-4-8",
            "max",
            {"BENCHFLOW_BEDROCK_THINKING_EFFORT": "low"},
        )


def test_a_dry_run_checks_the_matrix_entries_as_the_run_does(tmp_path, monkeypatch):
    result, _ = _matrix_run(
        tmp_path,
        monkeypatch,
        "  bad:\n    model: gpt-5.5\n    agent: codex-acp\n    reasoning_effort: ultra\n",
        "--dry-run",
    )
    assert result.exit_code == 2, result.output
    assert "gpt-5.5 does not take reasoning effort ultra" in " ".join(
        result.output.split()
    )


def test_a_matrix_entrys_effort_is_named_as_the_entrys(tmp_path, monkeypatch):
    result, _ = _matrix_run(
        tmp_path,
        monkeypatch,
        "  astra:\n    model: gpt-6-astra\n    agent: codex-acp\n    effort: ultra\n"
        '    agent_env:\n      CODEX_CONFIG: \'{"model_reasoning_effort":"high"}\'\n',
    )
    assert result.exit_code == 2, result.output
    text = " ".join(result.output.split())
    assert "(--matrix entry ultra, CODEX_CONFIG model_reasoning_effort high)" in text


def test_openhands_applies_its_own_llm_reasoning_effort():
    settled = settle_effort(
        "openhands", "openai/gpt-5.5", None, {"LLM_REASONING_EFFORT": "high"}
    )
    assert settled.effort is None
    with pytest.raises(EffortRequestError, match="nothing applies it to pi-acp"):
        settle_effort(
            "pi-acp", "openai/gpt-5.5", None, {"LLM_REASONING_EFFORT": "high"}
        )


def test_the_proxys_azure_route_puts_the_effort_on_every_call():
    from benchflow.providers.litellm_config import resolve_litellm_route

    route = resolve_litellm_route(
        "azure-foundry-openai/gpt-5.5",
        {
            "BENCHFLOW_REASONING_EFFORT": "high",
            "AZURE_RESOURCE": "r",
            "AZURE_API_KEY": "k",
        },
    )
    assert route.litellm_params["reasoning_effort"] == "high"


# --- review 4 of b483b235f: the env the trial reads; a route's own default -------

_OPUS_48 = "aws-bedrock/us.anthropic.claude-opus-4-8"
#: Made-up values, for the shell's environment in these runs.
_BEDROCK_SHELL = {
    "AWS_BEARER_TOKEN_BEDROCK": "fake-bearer-not-a-key",
    "AWS_REGION": "us-east-1",
}


def _shell(monkeypatch, tmp_path, **env):
    """The host's environment for a run: *env* and no .env file."""
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "no.env"))
    for name in (
        "BENCHFLOW_BEDROCK_THINKING_EFFORT",
        "BENCHFLOW_REASONING_EFFORT",
        "AZURE_API_ENDPOINT",
        "AZURE_RESOURCE",
        "AZURE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def _bench_run(tmp_path, monkeypatch, *argv, trial=None):
    """``bench run <one task> *argv`` through the real CLI. Evaluation.run is
    replaced: it keeps the config, awaits *trial* (given the Evaluation),
    and reports one pass; nothing starts a sandbox."""
    from typer.testing import CliRunner

    from benchflow.cli.main import app
    from benchflow.evaluation import Evaluation
    from tests.test_eval_artifact_cli import _write_rollout

    task = tmp_path / "tasks" / "task-a"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n[environment]\n')
    (task / "instruction.md").write_text("Write hello.txt.\n")
    seen: dict = {"preflight": [], "configs": []}

    async def fake_run(self):
        seen["configs"].append(self._config)
        if trial is not None:
            await trial(self)
        _write_rollout(self._jobs_dir / self._job_name / "task-a__abc", "task-a")
        return SimpleNamespace(
            job_name=self._job_name,
            passed=1,
            failed=0,
            errored=0,
            verifier_errored=0,
            total=1,
            score=1.0,
            score_excl_errors=1.0,
        )

    monkeypatch.setattr(Evaluation, "run", fake_run)
    monkeypatch.setattr(
        "benchflow.cli.main.eval_preflight", lambda **kw: seen["preflight"].append(kw)
    )
    result = CliRunner().invoke(
        app,
        ["run", str(tmp_path / "tasks"), "--jobs-dir", str(tmp_path / "jobs"), *argv],
        env={"COLUMNS": "200", "HARBOR_TELEMETRY": "off"},
    )
    return result, seen


@pytest.mark.parametrize("dry_run", [True, False])
def test_a_bedrock_effort_in_the_shell_beside_effort_is_refused_before_any_sandbox(
    tmp_path, monkeypatch, dry_run
):
    """Review 4's blocker: with BENCHFLOW_BEDROCK_THINKING_EFFORT=low in the
    shell, ``-e max`` passed the dry run and the run, whose trials ran at low.
    The checks read --agent-env only; the run also reads .env and the
    shell."""
    from benchflow.evaluation import EvaluationConfig

    _shell(
        monkeypatch, tmp_path, **_BEDROCK_SHELL, BENCHFLOW_BEDROCK_THINKING_EFFORT="low"
    )
    flags = ["--dry-run"] if dry_run else []
    result, seen = _bench_run(
        tmp_path,
        monkeypatch,
        "-a",
        "claude-agent-acp",
        "-m",
        _OPUS_48,
        "-e",
        "max",
        *flags,
    )
    assert result.exit_code == 2, result.output
    text = " ".join(result.output.split())
    assert "Refused before launch" in text
    assert (
        "(--effort max, BENCHFLOW_BEDROCK_THINKING_EFFORT low (from the host's "
        "environment)); ask for one"
    ) in text
    assert seen == {"preflight": [], "configs": []}
    # A config or the SDK is refused the same way.
    with pytest.raises(EffortRequestError, match="from the host's environment"):
        EvaluationConfig(
            agent="claude-agent-acp", model=_OPUS_48, reasoning_effort="max"
        )


def test_bedrock_low_is_sent_to_the_route_and_the_trial_records_a_match(
    tmp_path, monkeypatch
):
    """Review 4, older than these PRs: ``-e low`` on a Bedrock Claude route was
    accepted, but the route set its own high on every call, so the trial ran
    at high (and, 0.9.x keeping no route, recorded the calls' low as a
    match). A request wins over a route's default (lead ruling, 2026-11-13).
    Through the CLI with the shell's AWS settings: the env Rollout.setup
    resolves for the trial, the route made from it, the Bedrock body
    LiteLLM's own transform builds in the proxy, and the record."""
    import benchflow.task.plan_validation as plan_validation
    from benchflow.agents.env import resolve_agent_env
    from benchflow.effort_record import effort_outcome
    from benchflow.providers.litellm_config import resolve_litellm_route
    from benchflow.rollout import Rollout
    from tests.test_bedrock_thinking import _wire_output_config_effort

    _shell(monkeypatch, tmp_path, **_BEDROCK_SHELL)
    argv = ("-a", "claude-agent-acp", "-m", _OPUS_48, "-e", "low")
    trial_env: dict = {}

    class Stop(Exception):
        pass

    def stop(*args, **kwargs):
        # The plan gate is the first thing after the trial's env is resolved.
        trial_env.update(kwargs["agent_env"])
        raise Stop

    real_gate = plan_validation.validate_task_plan
    monkeypatch.setattr(plan_validation, "validate_task_plan", stop)

    async def trial(evaluation):
        task = evaluation._get_task_dirs()[0]
        rollout = Rollout(evaluation._rollout_config(task, evaluation._config))
        with pytest.raises(Stop):
            await rollout.setup()

    result, seen = _bench_run(tmp_path / "run", monkeypatch, *argv, trial=trial)
    assert result.exit_code == 0, result.output
    cfg = seen["configs"][0]
    assert (cfg.model, cfg.reasoning_effort) == (_OPUS_48, "low")
    route = resolve_litellm_route(_OPUS_48, trial_env)
    assert (
        trial_env.get("BENCHFLOW_BEDROCK_THINKING_EFFORT"),
        route.litellm_params["reasoning_effort"],
    ) == ("low", "low")
    # The proxy runs with the trial's env (and the route's config): LiteLLM's
    # Bedrock transform, with BenchFlow's patch, puts low on the wire.
    assert (
        _wire_output_config_effort(
            str(route.litellm_params["reasoning_effort"]),
            env_override=trial_env["BENCHFLOW_BEDROCK_THINKING_EFFORT"],
        )
        == "low"
    )
    # Nothing asked: the route keeps its own default.
    plain = resolve_agent_env("claude-agent-acp", _OPUS_48, {})
    assert resolve_litellm_route(_OPUS_48, plain).litellm_params[
        "reasoning_effort"
    ] == ("high")
    effort = effort_outcome(
        requested=cfg.reasoning_effort,
        requested_from=cfg.reasoning_effort_source,
        applied=None,
        sent={"low": 12},
        sent_source="proxy",
        calls=12,
    )
    assert (effort["matches"], effort["reason"]) == (True, None)
    # The dry run runs the run's own plan gate (EN-632): the real one.
    monkeypatch.setattr(plan_validation, "validate_task_plan", real_gate)
    dry, _ = _bench_run(tmp_path / "dry", monkeypatch, *argv, "--dry-run")
    assert dry.exit_code == 0, dry.output
    assert "effort_warning" not in json.loads(dry.stdout)


@pytest.mark.parametrize("dry_run", [True, False])
def test_an_azure_endpoint_in_the_shell_is_no_false_refusal(
    tmp_path, monkeypatch, dry_run
):
    """Review 4: with AZURE_API_ENDPOINT and AZURE_API_KEY in the shell, which
    the run uses, BENCHFLOW_REASONING_EFFORT on an Azure OpenAI route was
    refused as no Azure OpenAI route: the check resolved the route from
    --agent-env alone."""
    from benchflow.agents.env import resolve_agent_env
    from benchflow.providers.litellm_config import resolve_litellm_route

    _shell(
        monkeypatch,
        tmp_path,
        AZURE_API_ENDPOINT="https://example-resource.openai.azure.com",
        AZURE_API_KEY="fake-azure-key",
    )
    model = "azure-foundry-openai/gpt-5.5"
    flags = ["--dry-run"] if dry_run else []
    result, seen = _bench_run(
        tmp_path,
        monkeypatch,
        "-a",
        "pi-acp",
        "-m",
        model,
        "--agent-env",
        "BENCHFLOW_REASONING_EFFORT=high",
        *flags,
    )
    assert result.exit_code == 0, result.output
    assert "Refused" not in result.output
    if dry_run:
        return
    cfg = seen["configs"][0]
    assert cfg.reasoning_effort is None  # the route sets it, not the session
    asked = effort_request(cfg.agent, cfg.model, cfg.reasoning_effort, cfg.agent_env)
    assert (asked.effort, asked.source) == ("high", "BENCHFLOW_REASONING_EFFORT")
    route = resolve_litellm_route(
        model, resolve_agent_env(cfg.agent, model, cfg.agent_env)
    )
    assert route.litellm_params["reasoning_effort"] == "high"


def test_a_level_a_provider_lowers_unannounced_is_a_measured_mismatch(tmp_path):
    """Review 4, with F2b 18:46Z: no clamp is assumed or recorded. A provider
    that runs another level than the one asked for, where BenchFlow could not
    know it, shows in the calls: ``mismatch``, a warning, and the run's
    ``--fail-on effort-mismatch`` count. (rc3 named an assumed clamp, and
    calls at low under it read as the clamp.)"""
    from benchflow.cli.main import _effort_mismatches

    trial = tmp_path / "t__1"
    _codex_log(trial, "low", 6)
    result = _clamped_result(trial)
    assert (result["effort"]["matches"], result["effort"]["reason"]) == (
        False,
        "mismatch",
    )
    assert "clamp" not in result["effort"]
    assert result["warnings"] == [
        "reasoning effort max was asked for, but the agent's model calls carried "
        "low (6 calls)"
    ]
    assert _effort_mismatches([SimpleNamespace(rollout_dir=trial)]) == 1


def _clamped_result(trial: Path) -> dict:
    from benchflow.effort_record import effort_block
    from benchflow.rollout._results import _build_rollout_result

    effort = effort_block(
        trial,
        agent="claude-agent-acp",
        model=_OPUS_48,
        reasoning_effort="max",
        agent_env={},
        applied=None,
    )
    _build_rollout_result(
        trial,
        task_name="task",
        rollout_name=trial.name,
        agent="claude-agent-acp",
        agent_name="claude-agent-acp",
        model=_OPUS_48,
        n_tool_calls=1,
        prompts=["task"],
        error=None,
        verifier_error=None,
        trajectory=[{"type": "agent_message", "text": "done"}],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={},
        effort=effort,
    )
    return json.loads((trial / "result.json").read_text())


@pytest.mark.parametrize("dry_run", [True, False])
def test_a_known_downgrade_is_refused_in_the_dry_run_and_the_run(
    tmp_path, monkeypatch, dry_run
):
    """F2b 18:46Z: ``-e max`` on Antigravity ran at high with a warning; it is
    refused before any sandbox, the dry run as the run. ``-e max`` on
    Bedrock's Opus 4.8, which it takes, passes both."""
    _shell(monkeypatch, tmp_path, **_BEDROCK_SHELL)
    flags = ["--dry-run"] if dry_run else []
    refused, seen = _bench_run(
        tmp_path / "agy",
        monkeypatch,
        "-a",
        "antigravity",
        "-m",
        "gemini-3.1-pro",
        "-e",
        "max",
        *flags,
    )
    assert refused.exit_code == 2, refused.output
    text = " ".join(refused.output.split())
    assert "Refused before launch" in text
    assert "it takes low, medium, high (at most high): ask for one of them" in text
    assert seen == {"preflight": [], "configs": []}
    taken, _ = _bench_run(
        tmp_path / "bedrock",
        monkeypatch,
        "-a",
        "claude-agent-acp",
        "-m",
        _OPUS_48,
        "-e",
        "max",
        *flags,
    )
    assert taken.exit_code == 0, taken.output
    assert "provider clamp" not in taken.output


@pytest.mark.parametrize("effort", ["ultra", "none"])
def test_a_level_a_bedrock_route_cannot_take_is_refused(monkeypatch, effort):
    """The route is sent the level asked for, so one a Claude model cannot
    take is refused before launch, never run at the route's default."""
    monkeypatch.delenv("BENCHFLOW_BEDROCK_THINKING_EFFORT", raising=False)
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("opencode", _OPUS_48, effort, dict(_BEDROCK_SHELL))
    assert str(caught.value) == (
        f"{_OPUS_48} does not take reasoning effort {effort} (opencode); it takes "
        "low, medium, high, xhigh, max (at most max): ask for one of them"
    )
    from benchflow.effort_request import with_route_effort

    # Bedrock models without a route effort of their own are left alone.
    plain = with_route_effort("claude-agent-acp", _OPUS_48, None, dict(_BEDROCK_SHELL))
    assert "BENCHFLOW_BEDROCK_THINKING_EFFORT" not in plain
    old = "aws-bedrock/us.anthropic.claude-opus-4-7"
    assert "BENCHFLOW_BEDROCK_THINKING_EFFORT" not in with_route_effort(
        "claude-agent-acp", old, "low", dict(_BEDROCK_SHELL)
    )


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-sonnet-4-5",
        "aws-bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    ],
)
def test_a_claude_model_that_takes_no_effort_refuses_every_level(model):
    """Sonnet 4.5 and Haiku 4.5 take no effort (Anthropic's model reference):
    an -e for one ran at whatever the provider made of it."""
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("openhands", model, "high", {})
    assert str(caught.value) == (
        f"{model} takes no reasoning effort (openhands): drop high"
    )


@pytest.mark.parametrize(
    ("agent", "model", "effort"),
    [
        ("codex-acp", "aws-bedrock/us.anthropic.claude-opus-4-6-v1", "max"),
        (
            "claude-agent-acp",
            "aws-bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
            "high",
        ),
        ("codex-acp", "openrouter/anthropic/claude-opus-4.8", "xhigh"),
        ("codex-acp", "openrouter/anthropic/claude-opus-4.6", "max"),
    ],
)
def test_a_level_a_claude_4x_model_takes_is_run_as_asked(agent, model, effort):
    assert settle_effort(agent, model, effort, {}).source == RUN


@pytest.mark.parametrize("agent", ["pi-acp", "pi"])
def test_an_agent_whose_session_takes_no_effort_is_refused_before_launch(agent):
    """R383 L3: pi-acp's registry entry declares no effort option, so an -e
    for its session failed after the sandbox started ("does not declare or
    advertise an ACP effort config option"). pi-acp 0.0.34 does offer
    thought_level, which BenchFlow doesn't use yet (R383D L3)."""
    with pytest.raises(EffortRequestError) as caught:
        settle_effort(agent, "openrouter/anthropic/claude-opus-5.5", "high", {})
    assert str(caught.value) == (
        f"reasoning effort high can't reach the {agent} session: its registry "
        "entry declares no ACP effort option, so BenchFlow can't hand it one; "
        "drop the effort.\nNothing ran, so nothing was recorded."
    )


def test_a_proxy_effort_still_reaches_an_agent_whose_session_takes_none():
    """The Azure OpenAI route puts BENCHFLOW_REASONING_EFFORT on every call;
    pi-acp's session is never handed it."""
    env = {
        "BENCHFLOW_REASONING_EFFORT": "high",
        "AZURE_RESOURCE": "r",
        "AZURE_API_KEY": "k",
    }
    settled = settle_effort("pi-acp", "azure-foundry-openai/gpt-5.5", None, env)
    assert (settled.effort, settled.source) == (None, "BENCHFLOW_REASONING_EFFORT")


#: The registered agents whose entry declares no ACP effort option (R383C).
#: Gemini CLI 0.42.0 builds no config options, MiMo 0.1.15 offers model and
#: mode only, BenchFlow's deepagents and harvey-lab shims offer none, and
#: OpenHands takes its effort through LLM_REASONING_EFFORT. pi-acp 0.0.34 does
#: offer thought_level, which its entry doesn't declare yet (R383D L3).
_NO_EFFORT_OPTION = frozenset(
    {"deepagents", "gemini", "harvey-lab-harness", "mimo", "openhands", "pi-acp"}
)


def test_every_registered_agent_takes_an_effort_or_is_refused_one():
    """R383C: an agent takes an effort only where its registry entry declares
    one; every other is refused before launch, so a new agent that declares
    none can't reach a sandbox with -e."""
    from benchflow.agents.registry import _CORE_AGENT_CONFIGS, AGENTS

    taken, refused = set(), set()
    for name in AGENTS:
        try:
            settle_effort(name, "openai/gpt-5.5", "high", {})
        except EffortRequestError as error:
            assert "can't reach the" in str(error), (name, error)
            refused.add(name)
        else:
            taken.add(name)
    # OpenCode's option needs a model with variants, and openai/gpt-5.5 is
    # handed to it as the gateway alias, which has none (R383D M1).
    declared = {
        name
        for name, config in AGENTS.items()
        if config.acp_effort_config_id and not config.acp_effort_needs_variants
    }
    assert taken == declared | {"codex-acp"}
    assert refused == set(AGENTS) - taken
    assert refused & set(_CORE_AGENT_CONFIGS) == _NO_EFFORT_OPTION | {"opencode"}


def test_a_new_agent_takes_an_effort_only_if_its_entry_declares_one(monkeypatch):
    from benchflow.agents.registry import AGENTS, AgentConfig

    bare = AgentConfig(name="new-agent", install_cmd="true", launch_cmd="new-agent")
    monkeypatch.setitem(AGENTS, "new-agent", bare)
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("new-agent", "openai/gpt-5.5", "high", {})
    assert str(caught.value) == (
        "reasoning effort high can't reach the new-agent session: its registry "
        "entry declares no ACP effort option, so BenchFlow can't hand it one; "
        "drop the effort.\nNothing ran, so nothing was recorded."
    )
    declared = AgentConfig(
        name="new-agent",
        install_cmd="true",
        launch_cmd="new-agent",
        acp_effort_config_id="effort",
    )
    monkeypatch.setitem(AGENTS, "new-agent", declared)
    assert settle_effort("new-agent", "openai/gpt-5.5", "high", {}).effort == "high"


@pytest.mark.parametrize(
    ("spec", "refused"),
    [
        ("oh", True),
        ("acp:mimo", True),
        ("acpx/openhands", True),
        # acpx's runtime key is no codex-acp: the runtime hands its session no
        # model[effort] id, and it declares no effort option.
        ("acpx/codex", True),
        ("acpx/claude", False),
        ("acp:claude", False),
        ("codex", False),
    ],
)
def test_every_way_of_naming_an_agent_reads_its_registry_entry(spec, refused):
    model = "openai/gpt-5.5"
    if refused:
        with pytest.raises(EffortRequestError, match="declares no ACP effort option"):
            settle_effort(spec, model, "high", {})
    else:
        assert settle_effort(spec, model, "high", {}).effort == "high"


def test_scripted_and_unknown_agents_are_left_to_their_own_checks():
    for agent in ("oracle", "nop", "script:solve.sh", "no-such-agent"):
        assert settle_effort(agent, "openai/gpt-5.5", "high", {}).effort == "high"


def test_openhands_refusal_names_its_own_setting():
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("openhands", "openai/gpt-5.5", "xhigh", {})
    assert str(caught.value) == (
        "reasoning effort xhigh can't reach the openhands session: its registry "
        "entry declares no ACP effort option, so BenchFlow can't hand it one; "
        "ask through its own setting, --agent-env LLM_REASONING_EFFORT=xhigh, "
        "or drop the effort.\nNothing ran, so nothing was recorded."
    )
    assert caught.value.hint == (
        "ask through its own setting, --agent-env LLM_REASONING_EFFORT=xhigh, or "
        "drop the effort"
    )
    # A suffix asks the session too.
    with pytest.raises(EffortRequestError, match="LLM_REASONING_EFFORT=high"):
        settle_effort("openhands", "openai/gpt-5.5[high]", None, {})


@pytest.mark.parametrize(
    "agent", ["openhands", "mimo", "deepagents", "gemini", "harvey-lab-harness"]
)
def test_an_agent_without_an_effort_option_is_refused_by_the_dry_run_as_by_the_run(
    tmp_path, monkeypatch, agent
):
    """R383C: -e passed the dry run (exit 0, "effort: high") and failed after
    the sandbox and the agent install, with exit 1 ("does not declare or
    advertise an ACP effort config option"). Now both refuse, with the same
    words, before any preflight."""
    _shell(monkeypatch, tmp_path, OPENAI_API_KEY="fake-not-a-key")
    texts = []
    for flags in ([], ["--dry-run"]):
        refused, seen = _bench_run(
            tmp_path / ("dry" if flags else "run"),
            monkeypatch,
            "-a",
            agent,
            "-m",
            "openai/gpt-5.5",
            "-e",
            "high",
            *flags,
        )
        assert refused.exit_code == 2, refused.output
        assert seen == {"preflight": [], "configs": []}
        texts.append(" ".join(refused.output.split()))
    assert f"reasoning effort high can't reach the {agent} session" in texts[0]
    assert "Nothing ran, so nothing was recorded." in texts[0]
    if agent == "openhands":
        assert "--agent-env LLM_REASONING_EFFORT=high" in texts[0]
    assert texts[0] == texts[1]


def test_openhands_own_setting_runs_and_the_trial_records_it_as_asked(
    tmp_path, monkeypatch
):
    """The path openhands' refusal names. --agent-env LLM_REASONING_EFFORT
    passes the dry run and the run and reaches the trial's config (OpenHands
    writes it into its LLM settings). The record reads it as the request, and
    the calls that carried it as a match. ``applied`` stays the session's own
    word, which OpenHands' session doesn't give."""
    from benchflow.effort_record import PROXY_LOG, effort_block

    _shell(monkeypatch, tmp_path, OPENAI_API_KEY="fake-not-a-key")
    argv = (
        "-a",
        "openhands",
        "-m",
        "openai/gpt-5.5",
        "--agent-env",
        "LLM_REASONING_EFFORT=high",
    )
    ran, seen = _bench_run(tmp_path / "run", monkeypatch, *argv)
    assert ran.exit_code == 0, ran.output
    dry, _ = _bench_run(tmp_path / "dry", monkeypatch, *argv, "--dry-run")
    assert dry.exit_code == 0, dry.output
    cfg = seen["configs"][0]
    assert (cfg.reasoning_effort, cfg.reasoning_effort_source) == (
        None,
        "LLM_REASONING_EFFORT",
    )
    assert cfg.agent_env["LLM_REASONING_EFFORT"] == "high"
    trial = tmp_path / "trial"
    log = trial / PROXY_LOG
    log.parent.mkdir(parents=True)
    body = {"tools": [{"type": "function"}], "reasoning_effort": "high"}
    row = {"request": {"body": {**body, "upstream": {"reasoning_effort": "high"}}}}
    log.write_text((json.dumps(row) + "\n") * 3)
    effort = effort_block(
        trial,
        agent=cfg.agent,
        model=cfg.model,
        reasoning_effort=cfg.reasoning_effort,
        agent_env=cfg.agent_env,
        applied=None,
        requested_from=cfg.reasoning_effort_source,
    )
    assert {key: effort[key] for key in ("requested", "requested_from")} == {
        "requested": "high",
        "requested_from": "LLM_REASONING_EFFORT",
    }
    assert (effort["applied"], effort["sent"], effort["matches"]) == (
        None,
        {"high": 3},
        True,
    )


def test_the_budgets_the_deepseek_refusal_names_are_the_gateways():
    """R383C N1: the refusal gives the budget each level becomes, which is
    LiteLLM's (the gateway's pin) DEFAULT_REASONING_EFFORT_*_THINKING_BUDGET."""
    from litellm import constants

    from benchflow.effort_request import _THINKING_BUDGETS

    assert {
        level: getattr(
            constants, f"DEFAULT_REASONING_EFFORT_{level.upper()}_THINKING_BUDGET"
        )
        for level in _THINKING_BUDGETS
    } == _THINKING_BUDGETS


def test_the_deepseek_refusal_says_to_drop_e_not_to_ask_for_another_level(
    tmp_path, monkeypatch
):
    """R383C N1: its next: line said "ask for a level the model takes", and no
    level is taken on that route."""
    _shell(monkeypatch, tmp_path, DEEPSEEK_API_KEY="fake-not-a-key")
    refused, _ = _bench_run(
        tmp_path,
        monkeypatch,
        "-a",
        "claude",
        "-m",
        "deepseek/deepseek-v4-pro",
        "-e",
        "high",
        "--dry-run",
    )
    assert refused.exit_code == 2, refused.output
    text = " ".join(refused.output.split())
    assert "so -e high would run as budget_tokens 4096, not as high" in text
    assert "drop -e: no level reaches this route as asked" in text
    assert "ask for a level the model takes" not in text


def test_a_call_with_only_a_thinking_budget_is_counted_apart():
    """R383C I4: calls_with_effort counted budget-only calls as calls with an
    effort; they carried no level."""
    budget_only = _effort_counts(None, {"budget_tokens=63999": 2}, 2)
    assert budget_only == {
        "calls_with_effort": 0,
        "calls_with_budget": 2,
        "calls_without_effort": 0,
        "matches": None,
        "reason": "nothing_to_compare",
    }
    mixed = _effort_counts("high", {"high": 3, "budget_tokens=4096": 1}, 5)
    assert mixed == {
        "calls_with_effort": 3,
        "calls_with_budget": 1,
        "calls_without_effort": 1,
        "matches": False,
        "reason": "mismatch",
    }
    levels = _effort_counts("high", {"high": 4}, 4)
    assert (levels["calls_with_budget"], levels["matches"]) == (0, True)


def _effort_counts(requested, sent, calls):
    from benchflow.effort_record import effort_outcome

    effort = effort_outcome(
        requested=requested,
        requested_from=RUN if requested else None,
        applied=None,
        sent=sent,
        sent_source="proxy",
        calls=calls,
    )
    keys = (
        "calls_with_effort",
        "calls_with_budget",
        "calls_without_effort",
        "matches",
        "reason",
    )
    return {key: effort[key] for key in keys}


def test_a_bedrock_effort_in_the_shell_off_a_bedrock_route_asks_for_nothing(
    tmp_path, monkeypatch
):
    """Only a Bedrock Claude route reads BENCHFLOW_BEDROCK_THINKING_EFFORT: from
    the shell, off one, it changes nothing, so it is no request and no
    refusal. From --agent-env it is still refused there."""
    _shell(monkeypatch, tmp_path, BENCHFLOW_BEDROCK_THINKING_EFFORT="medium")
    model = "anthropic/claude-opus-4-8"
    asked = effort_request("claude-agent-acp", model, "low", {})
    assert (asked.effort, asked.source) == ("low", RUN)
    assert settle_effort("claude-agent-acp", model, None, {}).source is None
    with pytest.raises(EffortRequestError, match="only on a Bedrock Claude route"):
        settle_effort(
            "claude-agent-acp",
            model,
            None,
            {"BENCHFLOW_BEDROCK_THINKING_EFFORT": "medium"},
        )
    # On a Bedrock Claude route it is the request, from wherever it came.
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fake-bearer-not-a-key")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    asked = effort_request("claude-agent-acp", _OPUS_48, None, {})
    assert (asked.effort, asked.source) == (
        "medium",
        "BENCHFLOW_BEDROCK_THINKING_EFFORT",
    )
