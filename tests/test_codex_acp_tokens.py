"""A codex-acp trial counts every model call's tokens, not its last call's.

codex-acp 2.0.1 answers each prompt with the usage of the turn's last model
call, and its ``usage_update`` notifications carry only how full the context
is. On 0.10.0rc1 a codex-acp trial of 75 tool calls on gpt-6.1-sol recorded
110,848 cached + 362 input + 84 output tokens: its last call. The trial now
counts Codex's own session log, one record per response.

The ACP side here is what codex-acp sends (``PromptResponse.usage`` is
``toPromptUsage(toTokenCount(tokenUsage.last))``); the Codex side is a small
synthetic log in the shape Codex 0.159.3 writes, not a real one.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.acp.session import ACPSession
from benchflow.trajectories.model_calls import CODEX_SESSIONS

CONTEXT_WINDOW = 400000


def _usage(fresh: int, cached: int, output: int, reasoning: int = 0) -> dict:
    """One response's usage as Codex logs it: input counts the cached tokens,
    output counts the reasoning."""
    return {
        "input_tokens": fresh + cached,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": 0,
        "output_tokens": output,
        "reasoning_output_tokens": reasoning,
        "total_tokens": fresh + cached + output,
    }


# One prompt turn of three model calls: read the task, run a tool, answer.
THREE_CALLS = [
    _usage(12000, 0, 300, 200),
    _usage(896, 11904, 150, 64),
    _usage(828, 12672, 84),
]
# That trial's last call alone.
ONE_CALL = [_usage(362, 110848, 84)]


def _codex_log(
    home: Path,
    usages: list[dict],
    *,
    model: str = "gpt-6.1-sol",
    records: bool = True,
    repeated: bool = False,
) -> None:
    """The sandbox's Codex session log (``~/.codex/sessions``) for one thread.

    ``records=False`` is an older Codex's log, its ``token_count`` events
    only; ``repeated`` writes each event twice, as Codex does when only its
    rate limits changed (the same running total again)."""
    total: dict[str, int] = {}
    rows: list[dict] = [
        {
            "type": "session_meta",
            "payload": {"id": "thread-1", "cli_version": "0.159.3"},
        },
        {
            "type": "turn_context",
            "payload": {"model": model, "effort": "xhigh"},
        },
    ]
    for index, usage in enumerate(usages):
        total = {key: total.get(key, 0) + value for key, value in usage.items()}
        info = {
            "last_token_usage": usage,
            "total_token_usage": total,
            "model_context_window": CONTEXT_WINDOW,
        }
        if records:
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
        event = {"type": "event_msg", "payload": {"type": "token_count", "info": info}}
        rows += [event, event] if repeated else [event]
    path = (
        home / ".codex" / "sessions" / "2026" / "11" / "07" / "rollout-thread-1.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(
            json.dumps({"timestamp": f"2026-11-07T15:13:{n:02d}Z", **row}) + "\n"
            for n, row in enumerate(rows)
        )
    )


def _codex_acp_session(usages: list[dict]) -> ACPSession:
    """The ACP session codex-acp 2.0.1 leaves after one prompt turn."""
    session = ACPSession("thread-1")
    session.record_user_prompt("Fix the bug.")
    for usage in usages:
        # createUsageUpdate: the last call's total as the context's fill.
        session.handle_update(
            {
                "sessionUpdate": "usage_update",
                "used": usage["total_tokens"],
                "size": CONTEXT_WINDOW,
            }
        )
    last = usages[-1]
    session.record_prompt_usage(
        {
            "totalTokens": last["total_tokens"],
            "inputTokens": last["input_tokens"] - last["cached_input_tokens"],
            "cachedReadTokens": last["cached_input_tokens"],
            "outputTokens": last["output_tokens"],
            "thoughtTokens": last["reasoning_output_tokens"],
        }
    )
    session.mark_prompt_end()
    return session


def _codex_rollout(tmp_path: Path, monkeypatch, home: Path, *, harness: str = "acp"):
    """A codex-acp rollout after its agent's turn, its sandbox at *home*."""
    from benchflow._utils.task_authoring import task_digest
    from benchflow.rollout import Rollout, RolloutConfig, _session_log
    from benchflow.task import RolloutPaths, Task
    from tests.test_artifacts_collection import LocalTransport

    monkeypatch.setattr(_session_log, "_home", lambda cfg: str(home))
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Fix the bug.")

    class Box(LocalTransport):
        async def stop(self, *, delete: bool = True) -> None:
            pass

    rollout = Rollout(
        RolloutConfig(
            task_path=task,
            agent="codex-acp",
            task_digest=task_digest(task),
            harness=harness,
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
    rollout._agent_env = {}
    return rollout, trial


def _tokens(metrics: dict) -> dict:
    return {
        key: metrics[key]
        for key in (
            "n_input_tokens",
            "n_output_tokens",
            "n_cache_read_tokens",
            "n_cache_creation_tokens",
            "total_tokens",
        )
    }


def _summed(usages: list[dict]) -> dict:
    cached = sum(u["cached_input_tokens"] for u in usages)
    fresh = sum(u["input_tokens"] for u in usages) - cached
    output = sum(u["output_tokens"] for u in usages)
    return {
        "n_input_tokens": fresh,
        "n_output_tokens": output,
        "n_cache_read_tokens": cached,
        # OpenAI reports no cache writes: unknown, not 0.
        "n_cache_creation_tokens": None,
        "total_tokens": fresh + cached + output,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("usages", [THREE_CALLS, ONE_CALL], ids=["three", "one"])
async def test_a_codex_acp_trial_counts_every_call(tmp_path, monkeypatch, usages):
    """Guards a reported case (0.10.0rc1, 2026-11-07): a codex-acp trial's
    tokens were its ACP PromptResponse.usage, which codex-acp fills with the
    last call's. Three calls give their sum (13,584 tokens on ACP, 38,834 in
    all); one call still gives that call's usage."""
    home = tmp_path / "sandbox-home"
    _codex_log(home, usages)
    rollout, trial = _codex_rollout(tmp_path, monkeypatch, home)
    rollout._session = _codex_acp_session(usages)
    rollout._collect_native_acp_usage()
    rollout._session = None
    last_call = _tokens(rollout._native_usage_metrics)
    assert last_call == _summed(usages[-1:])

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert _tokens(usage) == _summed(usages)
    if usages is THREE_CALLS:
        assert usage["total_tokens"] == 38834 and last_call["total_tokens"] == 13584
    # the count is the session log's, and it is priced at
    # gpt-6.1-sol's list price.
    assert usage["usage_source"] == "session_log"
    assert usage["usage_source_legacy"] == "agent_native_acp"
    summed = _summed(usages)
    assert usage["cost_usd"] == pytest.approx(
        summed["n_input_tokens"] * 2e-6
        + summed["n_cache_read_tokens"] * 0.1e-6
        + summed["n_output_tokens"] * 10e-6
    )
    assert usage["cost_source"] == "list_price_estimate"
    assert usage["usage_details"]["thought_tokens"] == sum(
        u["reasoning_output_tokens"] for u in usages
    )
    assert usage["usage_details"]["token_count"] == {
        "source": "codex-session-log",
        "path": "agent/codex-sessions",
        "calls": len(usages),
        "acp_reported": {
            **last_call,
            "thought_tokens": usages[-1]["reasoning_output_tokens"],
        },
    }
    # The per-call record agrees with the totals.
    summary = json.loads((trial / "model-calls.json").read_text())
    assert summary["calls"] == len(usages)
    assert summary["tokens"]["output_tokens"] == usage["n_output_tokens"]


@pytest.mark.asyncio
async def test_codex_s_native_harness_counts_its_log_too(tmp_path, monkeypatch):
    """``harness: native`` runs Codex's CLI, whose turns report Codex's
    running total; counts both harnesses from the session log, so
    both count alike. The log's sum replaces the report, never adds to it."""
    home = tmp_path / "sandbox-home"
    _codex_log(home, THREE_CALLS)
    rollout, _ = _codex_rollout(tmp_path, monkeypatch, home, harness="native")
    rollout._session = _codex_acp_session(THREE_CALLS)
    rollout._collect_native_acp_usage()
    rollout._session = None

    await rollout._cleanup_before_teardown()

    assert _tokens(rollout._usage_metrics) == _summed(THREE_CALLS)
    assert rollout._usage_metrics["usage_source"] == "session_log"
    assert rollout._usage_metrics["usage_details"]["token_count"]["calls"] == 3


def _counted_rollout(tmp_path: Path, **overrides) -> SimpleNamespace:
    """A rollout whose Codex log is already copied: three calls, and the
    last one's tokens from ACP."""
    from benchflow.task import RolloutPaths

    trial = tmp_path / "trial"
    paths = RolloutPaths(rollout_dir=trial)
    paths.mkdir()
    _codex_log(tmp_path / "home", THREE_CALLS)
    shutil.move(tmp_path / "home" / ".codex" / "sessions", trial / CODEX_SESSIONS)
    fields = {
        "_config": SimpleNamespace(primary_agent="codex-acp", harness="acp"),
        "_native_usage_agents": {"codex-acp"},
        "_usage_metrics": {
            **_summed(THREE_CALLS[-1:]),
            "usage_source": "agent_native_acp",
            "cost_usd": None,
            "usage_details": {"thought_tokens": 0},
        },
        "_rollout_paths": paths,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "counted"),
    [
        ({}, True),
        ({"_native_usage_agents": {"codex-acp", "codex"}}, True),
        ({"_native_usage_agents": {"codex-acp", "claude-agent-acp"}}, False),
        ({"_usage_metrics": {"usage_source": "provider_response"}}, False),
        # a codex-acp trial that reported nothing is counted too.
        ({"_usage_metrics": {"usage_source": "unavailable"}}, True),
        ({"_from_branch_snapshot": True}, False),
        ({"_branch_child_active": True}, False),
    ],
)
async def test_only_codex_acp_s_own_report_is_replaced(tmp_path, overrides, counted):
    """Another agent's calls, a proxy's metering, no usage at all, or a
    sandbox that holds another rollout's turns keep the trial's figures."""
    from benchflow.rollout._session_log import count_codex_tokens

    rollout = _counted_rollout(tmp_path, **overrides)
    before = dict(rollout._usage_metrics)
    await count_codex_tokens(rollout)
    if counted:
        assert _tokens(rollout._usage_metrics) == _summed(THREE_CALLS)
    else:
        assert rollout._usage_metrics == before


@pytest.mark.asyncio
async def test_without_a_codex_log_the_acp_figure_stays(tmp_path):
    from benchflow.rollout._session_log import count_codex_tokens

    rollout = _counted_rollout(tmp_path)
    shutil.rmtree(tmp_path / "trial" / CODEX_SESSIONS)
    before = dict(rollout._usage_metrics)
    await count_codex_tokens(rollout)
    assert rollout._usage_metrics == before


# rc3's codex-acp trial on a ChatGPT login, cut off by its 60 s agent timeout
# (2026-11-13, f2-matrix ``codex-acp-sub__docker__forced-timeout``,
# gpt-6-astra): three calls. Its result.json recorded the last one, 230
# input + 61 output + 17,152 cached; Codex's log held 10,350 + 199 + 41,344.
RC3_FORCED_TIMEOUT = [
    _usage(9928, 7168, 64),
    _usage(192, 17024, 74),
    _usage(230, 17152, 61),
]


@pytest.mark.asyncio
async def test_rc3_s_timed_out_codex_login_trial_counts_every_call(
    tmp_path, monkeypatch
):
    home = tmp_path / "sandbox-home"
    _codex_log(home, RC3_FORCED_TIMEOUT, model="gpt-6-astra")
    rollout, trial = _codex_rollout(tmp_path, monkeypatch, home)
    rollout._session = _codex_acp_session(RC3_FORCED_TIMEOUT)
    rollout._collect_native_acp_usage()
    rollout._session = None
    assert _tokens(rollout._native_usage_metrics) == {
        "n_input_tokens": 230,
        "n_output_tokens": 61,
        "n_cache_read_tokens": 17152,
        "n_cache_creation_tokens": None,
        "total_tokens": 17443,
    }
    rollout._usage_report_missing = True  # execute() saw the agent timeout

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert _tokens(usage) == {
        "n_input_tokens": 10350,
        "n_output_tokens": 199,
        "n_cache_read_tokens": 41344,
        "n_cache_creation_tokens": None,
        "total_tokens": 51893,
    }
    assert usage["usage_source"] == "session_log"
    assert usage["usage_details"]["token_count"]["calls"] == 3
    # gpt-6-astra at the 2026-11-04 snapshot's list price.
    assert usage["cost_usd"] == pytest.approx(
        10350 * 10e-6 + 41344 * 1e-6 + 199 * 50e-6
    )
    assert usage["cost_source"] == "list_price_estimate"
    assert usage["price_table_date"] == "2026-11-04"
    summary = json.loads((trial / "model-calls.json").read_text())
    assert summary["calls"] == 3
    assert summary["models_served"] == {"gpt-6-astra": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape", ["logged-again", "repeated-event", "older-logged-again"]
)
async def test_a_response_logged_twice_counts_once(tmp_path, monkeypatch, shape):
    """Each response counts once: a record whose response id was
    read already, in this log or another, is not counted again; an older
    Codex's ``token_count`` event repeated with the same running total (only
    its rate limits changed) is no new response; and an older Codex's
    session logged in two files counts once, by its running totals."""
    home = tmp_path / "sandbox-home"
    day = home / ".codex" / "sessions" / "2026" / "11" / "07"
    if shape == "repeated-event":
        _codex_log(home, THREE_CALLS, records=False, repeated=True)
    else:
        _codex_log(home, THREE_CALLS, records=shape == "logged-again")
        shutil.copy(day / "rollout-thread-1.jsonl", day / "rollout-thread-1-b.jsonl")
    rollout, trial = _codex_rollout(tmp_path, monkeypatch, home)
    rollout._session = _codex_acp_session(THREE_CALLS)
    rollout._collect_native_acp_usage()
    rollout._session = None

    await rollout._cleanup_before_teardown()

    assert _tokens(rollout._usage_metrics) == _summed(THREE_CALLS)
    assert rollout._usage_metrics["usage_details"]["token_count"]["calls"] == 3
    summary = json.loads((trial / "model-calls.json").read_text())
    assert summary["calls"] == 3
