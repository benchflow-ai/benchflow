"""The effort asked for is applied or refused, never silently downgraded.

Guards three findings on 0.10.0rc3 (X2, live, and frontier-design-benchmark
PR #17):

- ``-m 'gpt-6-astra[ultra]'`` ran at medium and ``-m 'gpt-6.1-sol[ultra]'``
  at low, both exit 0, with ``effort.requested`` null: codex-acp's model
  selection (``acp/runtime.py`` ``_codex_session_model_id``) read only
  ``reasoning_effort``, kept the session's current variant when its model
  matched (Sol's default, low) and otherwise took the first of medium,
  high, low (Astra).
- ``CODEX_CONFIG={"model_reasoning_effort":"ultra"}`` ran Sol at ultra and
  Astra at medium: the session opened on Codex's default model at ultra, so
  for Astra the same ladder picked medium; BenchFlow never knew ultra was
  asked for.
- ``--effort ultra`` was refused outright: ``ultra`` was not among the levels
  ``normalize_reasoning_effort`` knew, though the pinned Codex lists it for
  both models.

The session states below are built from the pinned Codex 0.159.3 catalog
(``agents/codex_models.py``), one ``model[effort]`` id per level, as
codex-acp lists them.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
from tests.test_effort_applied import _acp, _connect

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


# --- the pinned catalog ------------------------------------------------------


def test_the_catalog_is_the_pinned_codexs():
    from benchflow.agents.registry import pinned_npm_package

    assert pinned_npm_package("codex")[1] == CODEX_CATALOG_VERSION
    assert CODEX_MODEL_EFFORTS["gpt-6-astra"] == (
        "low",
        ("low", "medium", "high", "xhigh", "max", "ultra"),
    )
    assert CODEX_MODEL_EFFORTS["gpt-6.1-sol"] == (
        "low",
        ("low", "medium", "high", "xhigh", "max", "ultra"),
    )


# --- every way of asking is the request --------------------------------------


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
    # ultra thinks as long as max: BenchFlow's longer idle limit.
    assert config.agent_idle_timeout == 1800
    job = EvaluationConfig(
        agent="codex-acp", model=model, reasoning_effort=effort, agent_env=env
    )
    assert job.reasoning_effort == "ultra"


def test_claude_codes_own_variable_is_the_request():
    asked = effort_request(
        "claude-agent-acp", "claude-opus-5-5", None, {CLAUDE_EFFORT_ENV: "max"}
    )
    # Claude Code applies it itself; BenchFlow hands the session nothing.
    assert (asked.effort, asked.source, asked.launch) == (
        "max",
        CLAUDE_EFFORT_ENV,
        None,
    )
    assert effort_request("claude-agent-acp", "m", "high", {}).effort == "high"
    assert (
        effort_request(
            "claude-agent-acp", "m", None, {CLAUDE_EFFORT_ENV: "auto"}
        ).effort
        is None
    )


def test_nothing_asked_is_no_request():
    assert effort_request("codex-acp", "gpt-6-astra", None, {}).effort is None
    assert settle_effort("codex-acp", "gpt-6-astra", None, {}).effort is None


# --- refused before launch ---------------------------------------------------


@pytest.mark.parametrize(
    ("model", "effort"), [("gpt-5.5", "ultra"), ("gpt-5.5[max]", None)]
)
def test_a_level_the_model_does_not_take_is_refused_before_launch(model, effort):
    from benchflow.rollout import RolloutConfig

    asked = effort or "max"
    expected = (
        f"gpt-5.5 does not take reasoning effort {asked} in the pinned Codex "
        f"0.159.3 (codex-acp); it takes low, medium, high, xhigh"
    )
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("codex-acp", model, effort, {})
    assert str(caught.value) == expected
    with pytest.raises(ValueError, match="does not take reasoning effort"):
        RolloutConfig(
            task_path=Path("."), agent="codex-acp", model=model, reasoning_effort=effort
        )


def test_two_ways_that_differ_are_refused():
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("codex-acp", "gpt-6-astra[ultra]", "high", {})
    assert str(caught.value) == (
        "the run asks for more than one reasoning effort (--effort high, "
        "model[effort] ultra); ask for one"
    )


def test_a_level_nobody_knows_is_refused():
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("codex-acp", "gpt-6-astra[extreme]", None, {})
    assert str(caught.value) == (
        "model 'gpt-6-astra[extreme]': [extreme] is neither a reasoning effort "
        "(none, minimal, low, medium, high, xhigh, max, ultra) nor a context size "
        "such as [1m]"
    )


def test_effort_ultra_is_a_level():
    """rc3 refused --effort ultra outright, for every model."""
    from benchflow._utils.config import normalize_reasoning_effort

    assert normalize_reasoning_effort("ULTRA") == "ultra"
    assert settle_effort("codex-acp", "gpt-6-astra", "ultra", {}).effort == "ultra"
    assert settle_effort("codex-acp", "gpt-6.1-sol", "ultra", {}).effort == "ultra"
    # A model the catalog does not hold is left to the session to refuse.
    assert settle_effort("codex-acp", "gpt-7-unknown", "ultra", {}).effort == "ultra"


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


# --- the session selects what was asked --------------------------------------


@pytest.mark.parametrize(
    ("model", "env", "current", "selected"),
    [
        # X2: the suffix was dropped; Sol kept its current low variant, Astra
        # got the ladder's medium.
        ("gpt-6.1-sol[ultra]", {}, "gpt-6.1-sol[low]", "gpt-6.1-sol[ultra]"),
        ("gpt-6-astra[ultra]", {}, "gpt-6.1-sol[low]", "gpt-6-astra[ultra]"),
        # PR #17: CODEX_CONFIG opened the session on Codex's default model at
        # ultra; Astra got the ladder's medium.
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
    from unittest.mock import patch

    from benchflow.acp.runtime import connect_acp

    with (
        patch("benchflow.acp.runtime.ContainerTransport"),
        patch("benchflow.acp.runtime.ACPClient", return_value=acp),
    ):
        await connect_acp(
            env=AsyncMock(),
            agent="codex-acp",
            agent_launch="codex-acp",
            agent_env=env,
            sandbox_user=None,
            model=model,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
            reasoning_effort=None,
        )
    acp.set_model.assert_awaited_once_with(selected)
    assert session.effort_applied == "ultra"


async def test_with_nothing_asked_the_session_keeps_its_choice(tmp_path):
    acp, session = _acp(_offered("gpt-6-astra[low]"))
    await _connect(tmp_path, acp, agent="codex-acp", model="gpt-6-astra", effort=None)
    acp.set_model.assert_awaited_once_with("gpt-6-astra[low]")
    assert session.effort_applied == "low"


# --- recorded, and a mismatch flagged after the run ----------------------------


def _codex_log(trial: Path, effort: str, calls: int) -> None:
    """A Codex 0.159 session rollout: one turn at *effort*, *calls* responses
    (each a token_count event), as codex-acp leaves it in agent/codex-sessions."""
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


def _result(trial: Path, *, requested, requested_from, applied) -> dict:
    from benchflow.rollout._results import _build_rollout_result
    from benchflow.trajectories.model_calls import write_model_calls

    (trial / "config.json").write_text(json.dumps({"reasoning_effort": requested}))
    write_model_calls(trial)
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
        reasoning_effort=requested,
        effort_requested_from=requested_from,
        effort_applied=applied,
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={},
    )
    return json.loads((trial / "result.json").read_text())


def test_a_replayed_medium_session_under_an_ultra_request_is_flagged(tmp_path):
    """PR #17's Astra trial, replayed: CODEX_CONFIG asked ultra, the session
    log says each of its 17 responses ran at medium."""
    from benchflow._utils.evaluation_results import model_call_summary
    from benchflow.cli.main import _effort_mismatches

    trial = tmp_path / "startup_alpha-sh__5209fe5e"
    _codex_log(trial, "medium", 17)
    result = _result(
        trial, requested="ultra", requested_from=CODEX_CONFIG, applied="medium"
    )
    assert result["effort"] == {
        "requested": "ultra",
        "requested_from": CODEX_CONFIG,
        "applied": "medium",
        "route": None,
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
    # The ATIF export carries the same block.
    atif = json.loads((trial / "trainer/atif.json").read_text())
    assert atif["extra"]["effort"] == result["effort"]
    # summary.json counts it, and --fail-on effort-mismatch counts it too.
    assert model_call_summary({"t": result})["model_calls"]["effort_mismatches"] == 1
    assert _effort_mismatches([SimpleNamespace(rollout_dir=trial)]) == 1


def test_an_ultra_session_under_an_ultra_request_matches(tmp_path):
    trial = tmp_path / "startup_alpha-sh__de375d9c"
    _codex_log(trial, "ultra", 34)
    result = _result(
        trial, requested="ultra", requested_from=MODEL_SUFFIX, applied="ultra"
    )
    assert (result["effort"]["matches"], result["effort"]["sent"]) == (
        True,
        {"ultra": 34},
    )
    assert "warnings" not in result


def test_fail_on_effort_mismatch_fails_the_run(tmp_path):
    import typer

    from benchflow.cli.main import _finish_run

    trial = tmp_path / "t__1"
    _codex_log(trial, "medium", 2)
    _result(trial, requested="ultra", requested_from=MODEL_SUFFIX, applied="medium")
    result = SimpleNamespace(
        results={"t": SimpleNamespace(error_category=None, rollout_dir=trial)},
        errored=0,
        verifier_errored=0,
        not_run=0,
        score=1.0,
        passed=1,
        total=1,
        run_checks=None,
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
    way it was asked (rc3: null for the suffix and CODEX_CONFIG)."""
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


@pytest.mark.parametrize(
    ("model", "env", "source"),
    [
        ("gpt-6-astra[ultra]", {}, MODEL_SUFFIX),
        ("gpt-6-astra", {"CODEX_CONFIG": ULTRA}, CODEX_CONFIG),
    ],
)
def test_the_run_card_says_what_was_asked_and_the_run_checks_hold_it(
    tmp_path, model, env, source
):
    """The card's model.effort is what the run checks compare the calls
    with: rc3 left it null for a suffix or a CODEX_CONFIG effort, so calls at
    medium passed them."""
    from benchflow.run_card import build_run_card, card_lines
    from benchflow.run_checks import (
        compare_sent,
        read_calls,
        requested_settings,
        settings_sent,
    )
    from benchflow.trajectories.model_calls import write_model_calls
    from tests.test_run_card import _evaluation, _tasks

    tasks = _tasks(tmp_path, 1)
    evaluation = _evaluation(
        tmp_path, tasks, agent="codex-acp", model=model, agent_env=env
    )
    card = build_run_card(evaluation, sorted(tasks.iterdir()))
    assert (card["model"]["effort"], card["model"]["effort_source"]) == (
        "ultra",
        source,
    )
    assert any(
        f"effort: ultra (from {source})" in line for line in card_lines(card, tmp_path)
    )
    wanted = [w for w in requested_settings(card) if w["setting"] == "effort"]
    assert wanted == [
        {"setting": "effort", "value": "ultra", "source": "run", "fails": True}
    ]
    trial = tmp_path / "trial"
    _codex_log(trial, "medium", 17)
    write_model_calls(trial)
    sent = settings_sent(read_calls(trial), "gpt-6-astra")
    assert compare_sent(wanted[0], sent) == {
        "kind": "mismatch",
        "sent": {"medium": 17},
        "calls": 17,
        "of": 17,
    }


# --- the lead's review (07:02Z): one list, nothing echoed, unknown no match ----


def test_every_effort_list_is_the_one_list():
    """Seven lists disagreed on ultra (Fable's audit of 1633fea3)."""
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
    from benchflow.setting_groups import EFFORT_ORDER

    assert "ultra" in EFFORT_LEVELS
    assert _REASONING_EFFORT_VALUES == CODEX_EFFORTS == _PROVIDER_REASONING_EFFORTS
    assert set(EFFORT_LEVELS) == _REASONING_EFFORT_VALUES
    assert EFFORT_ORDER == EFFORT_LEVELS
    assert set(CLAUDE_EFFORT_LEVELS) == CLAUDE_CODE_EFFORTS
    # These two run standalone (the model proxy, the Muse shim) and cannot
    # import it; they hold the same levels.
    assert litellm_request_guard._CLAUDE_EFFORTS == CLAUDE_EFFORT_LEVELS
    assert set(muse_acp_shim.EFFORT_LEVELS) <= set(EFFORT_LEVELS)
    for _, levels in CODEX_MODEL_EFFORTS.values():
        assert set(levels) <= set(EFFORT_LEVELS)


def test_claudes_variable_is_checked_like_effort():
    with pytest.raises(EffortRequestError, match=CLAUDE_EFFORT_ENV):
        effort_request("claude-agent-acp", "m", None, {CLAUDE_EFFORT_ENV: "extreme"})
    assert (
        effort_request(
            "claude-agent-acp", "m", None, {CLAUDE_EFFORT_ENV: "HIGH"}
        ).effort
        == "high"
    )


async def test_an_effort_option_the_agent_does_not_answer_is_not_echoed(tmp_path):
    """The agent set its effort option and said nothing back: applied is
    unknown, not the request echoed (rc3 recorded the request)."""
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


def test_unknown_is_not_a_match():
    from benchflow.trajectories.model_calls import effort_outcome

    session_log = {"calls": 3, "source": "claude-code-session-log", "agent_effort": {}}
    assert (
        effort_outcome(session_log, requested="max", applied="max")["matches"] is None
    )
    assert (
        effort_outcome(session_log, requested="max", applied="high")["matches"] is False
    )
    proxy = {"calls": 3, "source": "proxy", "agent_effort": {"max": 3}}
    assert effort_outcome(proxy, requested="max", applied=None)["matches"] is True


# --- Fable's second message (07:4xZ): nothing measured is no match --------------


def _calls(source="proxy", calls=100, **efforts):
    return {"calls": calls, "source": source, "agent_effort": efforts}


def test_an_applied_effort_with_no_calls_observed_is_unmeasured():
    from benchflow.trajectories.model_calls import effort_outcome, effort_warning

    effort = effort_outcome(
        _calls(source="claude-code-session-log"), requested="max", applied="max"
    )
    assert (effort["matches"], effort["reason"]) == (None, "unmeasured")
    assert effort_warning(effort) is None


def test_one_call_in_a_hundred_at_another_effort_is_a_mismatch():
    from benchflow.trajectories.model_calls import effort_outcome, effort_warning

    for sent in ({"max": 99, "high": 1}, {"high": 1}):
        effort = effort_outcome(_calls(**sent), requested="max", applied="max")
        assert (effort["matches"], effort["reason"]) == (False, "mismatch")
        assert "high (1 calls)" in (effort_warning(effort) or "")


def test_with_no_effort_asked_the_calls_must_agree_with_the_session():
    """PR #17's case: no -e, the session applied medium, the calls carried
    xhigh. rc3 said null and warned nothing."""
    from benchflow.trajectories.model_calls import effort_outcome, effort_warning

    effort = effort_outcome(
        _calls(source="codex-session-log", xhigh=17), requested=None, applied="medium"
    )
    assert (effort["matches"], effort["reason"]) == (False, "mismatch")
    assert effort_warning(effort) == (
        "no reasoning effort was asked for and the agent applied medium, but the "
        "agent's model calls carried xhigh (17 calls)"
    )
    agree = effort_outcome(
        _calls(source="codex-session-log", medium=17), requested=None, applied="medium"
    )
    assert (agree["matches"], agree["reason"]) == (True, None)


def test_calls_without_an_effort_are_no_agreement():
    from benchflow.trajectories.model_calls import effort_outcome

    dropped = effort_outcome(_calls(), requested="max", applied="max")
    assert (dropped["sent"], dropped["matches"], dropped["reason"]) == (
        {},
        None,
        "unmeasured: no call carried an effort",
    )
    silent = effort_outcome(_calls(), requested=None, applied="medium")
    assert (silent["matches"], silent["reason"]) == (
        None,
        "unmeasured: no call carried an effort",
    )
    nothing = effort_outcome(_calls(), requested=None, applied=None)
    assert (nothing["matches"], nothing["reason"]) == (None, "nothing_to_compare")


# --- a --matrix entry's effort and harness ---------------------------------------


def _matrix_run(tmp_path, monkeypatch, entries: str, *flags: str, agent="oracle"):
    from typer.testing import CliRunner

    from benchflow.cli.main import app
    from benchflow.evaluation import Evaluation
    from tests.test_eval_artifact_cli import _write_rollout, _write_task

    tasks = tmp_path / "tasks"
    _write_task(tasks / "task-a")
    # The dry run loads each task as the run does: a task.toml
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
    """rc3 loaded reasoning_effort, effort and harness from a --matrix entry
    and never read them: every entry ran at the run's own."""
    result, seen = _matrix_run(
        tmp_path,
        monkeypatch,
        "  astra:\n    model: gpt-6-astra\n    agent: codex-acp\n"
        "    reasoning_effort: ultra\n    harness: native\n"
        "  sol:\n    model: gpt-6.1-sol\n    agent: codex-acp\n    effort: high\n",
    )
    assert result.exit_code == 0, result.output
    assert [(c.model, c.reasoning_effort, c.harness) for c in seen] == [
        ("gpt-6-astra", "ultra", "native"),
        ("gpt-6.1-sol", "high", "acp"),
    ]
    # ultra thinks long: the effort's idle limit follows it.
    assert seen[0].agent_idle_timeout == 1800


@pytest.mark.parametrize(
    ("entry", "words"),
    [
        (
            "    model: gpt-5.5\n    agent: codex-acp\n    reasoning_effort: ultra\n",
            "gpt-5.5 does not take reasoning effort ultra",
        ),
        (
            "    model: gpt-6-astra\n    agent: codex-acp\n    reasoning_effort: high\n"
            "    effort: max\n",
            "reasoning_effort and effort differ",
        ),
        ("    model: m\n    agent: opencode\n    harness: native\n", "native harness"),
    ],
)
def test_a_matrix_entry_is_refused_before_any_trial(
    tmp_path, monkeypatch, entry, words
):
    result, seen = _matrix_run(tmp_path, monkeypatch, "  bad:\n" + entry)
    assert result.exit_code == 2, result.output
    text = " ".join(result.output.split())
    assert "Refused before launch: matrix model 'bad'" in text
    assert words in text
    assert seen == []


# --- a Claude Code alias is the model it names ---------------------------


def test_the_alias_list_is_the_pinned_claude_codes():
    from benchflow.agents.claude_models import CLAUDE_CODE_CATALOG_VERSION
    from benchflow.agents.registry import pinned_npm_package

    assert pinned_npm_package("claude-code")[1] == CLAUDE_CODE_CATALOG_VERSION


@pytest.mark.parametrize(
    ("requested", "served", "same"),
    [
        ("sonnet", "claude-sonnet-5-5", True),
        ("sonnet", "claude-sonnet-5-5-20261101", True),
        ("anthropic/sonnet", "claude-sonnet-4-6", True),  # a gateway's sonnet
        ("opus[1m]", "claude-opus-5-5", True),
        ("best", "claude-fable-5-1", True),
        ("sonnet", "claude-haiku-5-5", False),
        ("haiku", "claude-sonnet-5-5", False),
    ],
)
def test_a_claude_code_alias_is_the_model_it_names(requested, served, same):
    from benchflow.run_checks import same_model

    assert same_model(requested, served, agent="claude-agent-acp") is same


@pytest.mark.parametrize(
    ("served", "fails"), [("claude-sonnet-5-5", False), ("claude-haiku-5-5", True)]
)
async def test_a_run_with_m_sonnet_passes_the_model_check(tmp_path, served, fails):
    """X2: `-m sonnet` solved the task and the run failed its model check,
    the calls having carried claude-sonnet-5-5."""
    from tests.test_run_checks import _calls, _job, _summary, _trials_that

    job = _job(tmp_path, 3, model="sonnet")
    job._run_single_task = _trials_that(
        job, _calls(model="sonnet", served=served, route="anthropic/sonnet")
    )
    await job.run()
    checks = _summary(job)["run_checks"]
    assert checks["failed"] is fails, checks
    if fails:
        assert checks["failures"][0].startswith("model: requested sonnet;")


# --- review of 362bef7e8 (lead 10:3xZ) --------------------------------------------


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
    """[1m] is a context size, part of the model id: 362bef7e8 read every
    bracket as an effort and refused opus[1m] ('asks for effort 1m')."""
    from benchflow.evaluation import EvaluationConfig
    from benchflow.rollout import RolloutConfig

    settled = settle_effort(agent, model, None, {})
    assert (settled.model, settled.effort) == (bare, effort)
    job = EvaluationConfig(agent=agent, model=model)
    trial = RolloutConfig(task_path=Path("."), agent=agent, model=model)
    # The effort suffix is consumed: the model id goes upstream bare.
    assert (job.model, job.reasoning_effort) == (bare, effort)
    assert (trial.model, trial.reasoning_effort) == (bare, effort)


def test_a_suffix_that_is_no_effort_or_size_is_refused():
    with pytest.raises(EffortRequestError, match=r"\[fast\] is neither"):
        settle_effort("codex-acp", "gpt-6-astra[fast]", None, {})
    with pytest.raises(EffortRequestError, match="names two reasoning efforts"):
        settle_effort("codex-acp", "gpt-6-astra[high][max]", None, {})


def test_claudes_variable_and_effort_that_differ_are_refused():
    """-e high with CLAUDE_CODE_EFFORT_LEVEL=low passed, recorded low and
    handed the session high."""
    from benchflow.rollout import RolloutConfig

    with pytest.raises(EffortRequestError) as caught:
        settle_effort("claude-agent-acp", "m", "high", {CLAUDE_EFFORT_ENV: "low"})
    assert str(caught.value) == (
        "the run asks for more than one reasoning effort (--effort high, "
        "CLAUDE_CODE_EFFORT_LEVEL low); ask for one"
    )
    with pytest.raises(ValueError, match="more than one reasoning effort"):
        RolloutConfig(
            task_path=Path("."),
            agent="claude-agent-acp",
            model="m",
            reasoning_effort="high",
            agent_env={CLAUDE_EFFORT_ENV: "low"},
        )
    same = settle_effort("claude-agent-acp", "m", "high", {CLAUDE_EFFORT_ENV: "high"})
    assert same.effort == "high"


@pytest.mark.parametrize(
    ("effort", "env"),
    [
        ("ultra", {}),
        (None, {CLAUDE_EFFORT_ENV: "none"}),
        (None, {CLAUDE_EFFORT_ENV: "minimal"}),
    ],
)
def test_a_level_claude_code_does_not_take_is_refused_before_launch(effort, env):
    with pytest.raises(
        EffortRequestError, match="Claude Code \\(claude-agent-acp\\) does not take"
    ):
        settle_effort("claude-agent-acp", "claude-opus-5-5", effort, env)


@pytest.mark.parametrize(
    ("env", "source"),
    [
        ({"BENCHFLOW_REASONING_EFFORT": "high"}, "BENCHFLOW_REASONING_EFFORT"),
        ({"LLM_REASONING_EFFORT": "high"}, "LLM_REASONING_EFFORT"),
    ],
)
def test_the_proxys_effort_settings_are_requests(env, source):
    asked = effort_request("openhands", "m", None, env)
    assert (asked.effort, asked.source) == ("high", source)
    with pytest.raises(EffortRequestError, match="more than one reasoning effort"):
        effort_request("openhands", "m", "max", env)


def test_opencodes_config_effort_is_a_request():
    from benchflow.effort_request import OPENCODE_CONFIG

    config = json.dumps(
        {
            "provider": {
                "p": {"models": {"m": {"options": {"reasoningEffort": "xhigh"}}}}
            }
        }
    )
    asked = effort_request("opencode", "p/m", None, {"OPENCODE_CONFIG_CONTENT": config})
    assert (asked.effort, asked.source) == ("xhigh", OPENCODE_CONFIG)


def test_an_invalid_bedrock_thinking_effort_fails_before_any_sandbox():
    """It became high."""
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
    1.88.0rc1 limit that lowered it to high is gone)."""
    from benchflow.providers.litellm_config import resolve_litellm_route

    env = {"AWS_BEARER_TOKEN_BEDROCK": "fake-not-a-key", "AWS_REGION": "us-east-1"}
    assert settle_effort(agent, model, effort, env).source == RUN
    if model.startswith("aws-bedrock/"):
        from benchflow.effort_request import with_route_effort

        sent = with_route_effort(agent, model, effort, env)
        route = resolve_litellm_route(model, sent)
        assert route.litellm_params["reasoning_effort"] == effort


def test_calls_without_an_effort_through_the_proxy_are_partly_unmeasured(
    tmp_path, monkeypatch
):
    """Fable's RECORDS-3, through the proxy's logger and BenchFlow's parser:
    one Responses call at ultra and 99 with no effort read as matches true."""
    import asyncio
    from datetime import datetime as dt

    from benchflow.providers.litellm_logging import trajectory_from_litellm_callback_log
    from benchflow.trajectories.model_calls import (
        effort_outcome,
        effort_warning,
        write_model_calls,
    )
    from tests.test_litellm_logging import _callback_namespace

    log = tmp_path / "litellm.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    logger = _callback_namespace()["BenchFlowLiteLLMLogger"]()
    tools = [{"type": "function", "name": "shell", "parameters": {}}]
    for n in range(100):
        optional = {"tools": tools}
        if n == 0:
            optional["reasoning"] = {"effort": "ultra", "summary": "auto"}
        kwargs = {
            "model": "gpt-6-astra",
            "input": [{"role": "user", "content": f"turn {n}"}],
            "call_type": "aresponses",
            "optional_params": optional,
            "litellm_params": {"metadata": {}},
        }
        response = {
            "model": "gpt-6-astra",
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
        asyncio.run(
            logger.async_log_success_event(kwargs, response, dt.now(), dt.now())
        )
    trajectory = trajectory_from_litellm_callback_log(
        log.read_text(), session_id="s", agent_name="codex-acp"
    )
    trial = tmp_path / "trial"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "trajectory" / "llm_trajectory.jsonl").write_text(
        "".join(
            json.dumps(exchange.model_dump(mode="json")) + "\n"
            for exchange in trajectory.exchanges
        )
    )
    summary = write_model_calls(trial)
    effort = effort_outcome(summary, requested="ultra", applied="ultra")
    assert effort["sent"] == {"ultra": 1}
    assert (effort["matches"], effort["reason"]) == (None, "partly unmeasured")
    assert (effort["calls_with_effort"], effort["calls_without_effort"]) == (1, 99)
    assert effort_warning(effort) == (
        "reasoning effort ultra was asked for: 1 of the agent's 100 model calls "
        "carried it and 99 carried no effort, so the effort of those is unmeasured"
    )


def test_summary_counts_the_unmeasured_beside_the_mismatches():
    from benchflow._utils.evaluation_results import model_call_summary

    results = {
        "a": {
            "effort": {
                "requested": "max",
                "matches": None,
                "reason": "partly unmeasured",
            }
        },
        "b": {"effort": {"requested": "max", "matches": False, "reason": "mismatch"}},
        "c": {"effort": {"requested": None, "matches": None, "reason": "unmeasured"}},
    }
    block = model_call_summary(results)["model_calls"]
    assert (block["effort_mismatches"], block["effort_unmeasured"]) == (1, 1)


def test_a_claude_alias_is_resolved_only_for_claude_code():
    from benchflow.run_checks import same_model

    assert same_model("sonnet", "claude-sonnet-5-5", agent="claude-agent-acp")
    assert not same_model("sonnet", "claude-sonnet-5-5", agent="opencode")
    assert not same_model("sonnet", "claude-sonnet-5-5")


async def test_codex_applied_is_the_variant_the_session_took(tmp_path):
    """applied is the model[effort] id set_model took; BenchFlow writes
    nothing into the session's own state (turn_context is the check)."""
    acp, session = _acp(_offered("gpt-6.1-sol[low]"))
    await _connect(
        tmp_path, acp, agent="codex-acp", model="gpt-6-astra", effort="ultra"
    )
    acp.set_model.assert_awaited_once_with("gpt-6-astra[ultra]")
    assert session.model_state["currentModelId"] == "gpt-6.1-sol[low]"
    assert session.effort_applied == "ultra"


async def test_with_nothing_asked_codex_runs_the_models_own_default(tmp_path):
    """rc3's ladder took medium for Astra, whose pinned default is low."""
    acp, session = _acp(_offered("gpt-6.1-sol[low]"))
    await _connect(tmp_path, acp, agent="codex-acp", model="gpt-6-astra", effort=None)
    acp.set_model.assert_awaited_once_with("gpt-6-astra[low]")
    assert session.effort_applied == "low"


def test_every_ladder_is_the_one_list():
    from benchflow._utils.config import EFFORT_LEVELS
    from benchflow.agents import antigravity_acp_shim, muse_acp_shim
    from benchflow.providers import litellm_bedrock_patch
    from benchflow.providers.litellm_config import _BEDROCK_EFFORT_LADDER

    assert EFFORT_LEVELS[1:7] == _BEDROCK_EFFORT_LADDER
    assert litellm_bedrock_patch._EFFORT_LADDER == _BEDROCK_EFFORT_LADDER
    # Each shim takes every level BenchFlow sends it, never an error; the run
    # refuses before launch the levels a shim runs as another.
    assert set(antigravity_acp_shim._EFFORT_ALIASES) == set(EFFORT_LEVELS)
    assert set(muse_acp_shim.EFFORT_LEVELS) | set(muse_acp_shim._EFFORT_ALIASES) == set(
        EFFORT_LEVELS
    )


def test_a_matrix_entrys_effort_that_differs_from_the_runs_is_refused(
    tmp_path, monkeypatch
):
    result, seen = _matrix_run(
        tmp_path,
        monkeypatch,
        "  astra:\n    model: gpt-6-astra\n    agent: codex-acp\n    reasoning_effort: ultra\n",
        "--reasoning-effort",
        "high",
    )
    assert result.exit_code == 2, result.output
    assert "differs from the run's --effort high" in " ".join(result.output.split())
    assert seen == []


def test_a_matrix_entrys_own_model_and_env_build_its_trials(tmp_path, monkeypatch):
    """-a codex-acp -m gpt-5.5 with an entry {model: gpt-6-astra, CODEX_CONFIG
    ultra} exited 1 with no trials: the trial config paired gpt-5.5 with the
    entry's ultra before the entry's model was set."""
    from benchflow.effort_request import MATRIX

    result, seen = _matrix_run(
        tmp_path,
        monkeypatch,
        "  astra:\n    model: gpt-6-astra\n    agent: codex-acp\n"
        '    agent_env:\n      CODEX_CONFIG: \'{"model_reasoning_effort":"ultra"}\'\n'
        "  sol:\n    model: gpt-6.1-sol[ultra]\n    agent: codex-acp\n"
        "  luna:\n    model: gpt-6-luna\n    agent: codex-acp\n    effort: max\n",
        "-m",
        "gpt-5.5",
        agent="codex-acp",
    )
    assert result.exit_code == 0, result.output
    assert [(c.model, c.reasoning_effort) for c in seen] == [
        ("gpt-6-astra", "ultra"),
        ("gpt-6.1-sol", "ultra"),
        ("gpt-6-luna", "max"),
    ]
    assert seen[2].reasoning_effort_source == MATRIX


# --- review of 0477ad1f3 ----------------------------------------------------------


@pytest.mark.parametrize(
    "agent", ["pi-acp", "gemini", "openhands", "deepagents", "mimo"]
)
def test_a_proxy_effort_never_reaches_a_session_and_is_refused_where_nothing_applies_it(
    agent,
):
    """0477ad1f3 handed BENCHFLOW_REASONING_EFFORT to the session, and an agent
    with no effort option died after its sandbox started. It is a request,
    applied where it acts (the proxy's Azure OpenAI route); where nothing
    applies it, refused before launch."""
    env = {"BENCHFLOW_REASONING_EFFORT": "high"}
    with pytest.raises(EffortRequestError) as caught:
        settle_effort(agent, "openai/gpt-5.5", None, env)
    assert str(caught.value) == (
        f"BENCHFLOW_REASONING_EFFORT=high: nothing applies it to {agent} on "
        "openai/gpt-5.5: the model proxy sets it only on an Azure OpenAI route; "
        "ask with --effort where the agent takes one, or drop it"
    )
    azure = {**env, "AZURE_RESOURCE": "r", "AZURE_API_KEY": "k"}
    settled = settle_effort(agent, "azure-foundry-openai/gpt-5.5", None, azure)
    assert settled.effort is None  # the session gets nothing
    asked = effort_request(agent, "azure-foundry-openai/gpt-5.5", None, azure)
    assert (asked.effort, asked.source) == ("high", "BENCHFLOW_REASONING_EFFORT")


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
    """Where BenchFlow itself sets the effort, a call without it is a bug."""
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


def test_a_bedrock_thinking_effort_that_differs_from_effort_is_refused():
    """-e max with BENCHFLOW_BEDROCK_THINKING_EFFORT=low passed, warned high and
    ran at low."""
    model = "aws-bedrock/us.anthropic.claude-opus-4-8"
    with pytest.raises(EffortRequestError) as caught:
        settle_effort(
            "claude-agent-acp",
            model,
            "max",
            {"BENCHFLOW_BEDROCK_THINKING_EFFORT": "low"},
        )
    assert str(caught.value) == (
        "the run asks for more than one reasoning effort (--effort max, "
        "BENCHFLOW_BEDROCK_THINKING_EFFORT low); ask for one"
    )
    alone = settle_effort(
        "claude-agent-acp", model, None, {"BENCHFLOW_BEDROCK_THINKING_EFFORT": "low"}
    )
    assert alone.effort is None
    with pytest.raises(EffortRequestError, match="only on a Bedrock Claude route"):
        settle_effort(
            "claude-agent-acp",
            "anthropic/claude-opus-4-8",
            None,
            {"BENCHFLOW_BEDROCK_THINKING_EFFORT": "low"},
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


def test_a_dry_run_checks_the_matrix_entries_as_the_run_does(tmp_path, monkeypatch):
    """A bad entry exited 0 in --dry-run and 2 in the run."""
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
    # The dry run checks the entry's own agent, and refuses a missing key
    # as each trial does: a made-up one.
    monkeypatch.setenv("OPENAI_API_KEY", "fake-not-a-key")
    good, _ = _matrix_run(
        tmp_path / "ok",
        monkeypatch,
        "  astra:\n    model: gpt-6-astra\n    agent: codex-acp\n    effort: ultra\n",
        "--dry-run",
    )
    assert good.exit_code == 0, good.output


# --- review 4 of f74177187: the env the trial reads; a route's own default -------

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
    shell, ``-e max`` passed the dry run and the run, whose trials ran at low
    and recorded a Bedrock clamp. The checks read
    --agent-env only; the run also reads .env and the shell."""
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
    at high and recorded a mismatch. A request wins over a route's default
    (lead ruling, 2026-11-13). Through the CLI with the shell's AWS settings:
    the dry run's card, the env Rollout.setup resolves for the trial, the
    route made from it, the Bedrock body LiteLLM's own transform builds in
    the proxy, and the record."""
    import benchflow.task.plan_validation as plan_validation
    from benchflow.agents.env import resolve_agent_env
    from benchflow.providers.litellm_config import resolve_litellm_route
    from benchflow.rollout import Rollout
    from benchflow.trajectories.model_calls import effort_outcome
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
        _calls(low=12, calls=12),
        requested=cfg.reasoning_effort,
        applied=None,
        route=str(route.litellm_params["reasoning_effort"]),
        requested_from=cfg.reasoning_effort_source,
    )
    assert (effort["matches"], effort["reason"]) == (True, None)
    # The dry run's card says the same. It runs the run's own plan gate
    # , so it gets the real one back.
    monkeypatch.setattr(plan_validation, "validate_task_plan", real_gate)
    dry, _ = _bench_run(tmp_path / "dry", monkeypatch, *argv, "--dry-run")
    assert dry.exit_code == 0, dry.output
    card = json.loads(dry.stdout)["run_card"]
    row = next(
        s for s in card["settings"] if s["name"] == "BENCHFLOW_BEDROCK_THINKING_EFFORT"
    )
    assert (row["value"], row["source"]) == ("low", "run_effort")
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
    from benchflow.rollout._results import _build_rollout_result
    from benchflow.trajectories.model_calls import write_model_calls

    (trial / "config.json").write_text(json.dumps({"reasoning_effort": "max"}))
    write_model_calls(trial)
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
        reasoning_effort="max",
        effort_requested_from=RUN,
        effort_route="max",
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={},
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
        ("openrouter/anthropic/claude-opus-4.6", "low"),
        ("openrouter/anthropic/claude-opus-4.8", "max"),
        ("openrouter/anthropic/claude-opus-4.8", "xhigh"),
    ],
)
def test_claude_code_runs_a_dotted_id_at_the_level_asked(model, effort):
    """R383C M1: Claude Code 2.1.316 sends these levels upstream on
    OpenRouter's dotted ids (R383C on 5b41cb45b: max x2, matches true), and
    OpenRouter lists no dashed id; 965ba94b1 refused them."""
    assert settle_effort("claude-agent-acp", model, effort, {}).effort == effort


@pytest.mark.parametrize(
    "model",
    ["openrouter/anthropic/claude-opus-4.6", "openrouter/anthropic/claude-opus-4.8"],
)
def test_claude_code_on_a_dotted_id_runs_max_in_the_dry_run_and_the_run(
    tmp_path, monkeypatch, model
):
    _shell(monkeypatch, tmp_path, OPENROUTER_API_KEY="fake-not-a-key")
    argv = ("-a", "claude", "-m", model, "-e", "max")
    ran, seen = _bench_run(tmp_path / "run", monkeypatch, *argv)
    assert ran.exit_code == 0, ran.output
    assert (seen["configs"][0].model, seen["configs"][0].reasoning_effort) == (
        model,
        "max",
    )
    dry, _ = _bench_run(tmp_path / "dry", monkeypatch, *argv, "--dry-run")
    assert dry.exit_code == 0, dry.output


def test_claude_code_on_dotted_opus_4_5_is_refused_before_launch():
    """Claude Code 2.1.316 applies high on OpenRouter's dotted Opus 4.5 and
    sends no effort upstream (R383C: record "unmeasured: no call carried an
    effort"); the dashed id on the anthropic/ route takes it."""
    model = "openrouter/anthropic/claude-opus-4.5"
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("claude-agent-acp", model, "high", {})
    assert str(caught.value) == (
        f"Claude Code 2.1.316 (claude-agent-acp) sends no effort for {model}: it "
        "runs at its own level with none upstream; it takes -e high on the "
        "anthropic/ route (anthropic/claude-opus-4-5) or the Bedrock route "
        "(aws-bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0)"
    )
    assert caught.value.hint == (
        "run it as anthropic/claude-opus-4-5 or "
        "aws-bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0, or drop -e"
    )
    assert (
        settle_effort(
            "claude-agent-acp", "anthropic/claude-opus-4-5", "high", {}
        ).effort
        == "high"
    )
    # Other agents name the model to the gateway, not to Claude Code.
    assert settle_effort("codex-acp", model, "high", {}).effort == "high"


def test_the_claude_effort_table_was_checked_on_the_pinned_claude_code():
    """The table's claude_code levels hold for the Claude Code they were
    checked on; a bump of the pin fails here until the table is re-checked."""
    from benchflow.agents.registry import pinned_npm_package
    from benchflow.effort_request import CLAUDE_CODE_CHECKED

    assert pinned_npm_package("claude-code")[1] == CLAUDE_CODE_CHECKED


def test_the_claude_effort_table_is_well_formed():
    from benchflow._utils.config import CLAUDE_EFFORT_LEVELS
    from benchflow.effort_request import CLAUDE_EFFORTS

    for model, entry in CLAUDE_EFFORTS.items():
        assert set(entry.claude_code) <= set(entry.docs), model
        assert list(entry.docs) == [
            level for level in CLAUDE_EFFORT_LEVELS if level in entry.docs
        ], model
    # Anthropic's current models (2026-11-13; Mythos, for verified
    # organizations only, aside) each have a row.
    for model in (
        "claude-fable-5-2",
        "claude-fable-5-1",
        "claude-fable-5",
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-5-5",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-haiku-5-5",
        "claude-haiku-4-5",
    ):
        assert model in CLAUDE_EFFORTS


@pytest.mark.parametrize(
    ("model", "row"),
    [
        ("aws-bedrock/us.anthropic.claude-opus-4-6-v1", "claude-opus-4-6"),
        ("anthropic/claude-opus-4-5-20251101", "claude-opus-4-5"),
        ("openrouter/anthropic/claude-opus-4.5", "claude-opus-4.5"),
        ("openrouter/anthropic/claude-fable-5.2", "claude-fable-5-2"),
        ("claude-opus-4-6[1m]", "claude-opus-4-6"),
        ("openrouter/anthropic/claude-opus-5.5", "claude-opus-5-5"),
        ("claude-opus-5", "claude-opus-5"),
    ],
)
def test_a_claude_id_reads_its_row(model, row):
    from benchflow.effort_request import CLAUDE_EFFORTS, _claude_entry

    assert _claude_entry(model) is CLAUDE_EFFORTS[row]


@pytest.mark.parametrize(
    "model",
    [
        "claude-mythos-5-2",
        "gpt-5.5",
    ],
)
def test_a_model_the_table_does_not_name_reads_no_row(model):
    from benchflow.effort_request import _claude_entry

    assert _claude_entry(model) is None


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
            (
                "-a",
                "claude",
                "-m",
                "openrouter/anthropic/claude-opus-4.5",
                "-e",
                "high",
            ),
            "Claude Code 2.1.316 (claude-agent-acp) sends no effort for "
            "openrouter/anthropic/claude-opus-4.5",
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
    writes it into its LLM settings). The trial's record, from the request
    the rollout reads off that config, takes it as the request, and calls
    that carried it as a match. ``applied`` stays the session's own word,
    which OpenHands' session doesn't give."""
    from benchflow.effort_request import effort_source
    from benchflow.trajectories.model_calls import effort_outcome

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
    asked = effort_request(cfg.agent, cfg.model, cfg.reasoning_effort, cfg.agent_env)
    effort = effort_outcome(
        _calls(calls=3, high=3),
        requested=asked.effort,
        requested_from=effort_source(cfg.reasoning_effort_source, asked.source),
        applied=None,
    )
    assert {key: effort[key] for key in ("requested", "requested_from")} == {
        "requested": "high",
        "requested_from": "LLM_REASONING_EFFORT",
    }
    assert (effort["applied"], effort["matches"]) == (None, True)


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
    from benchflow.trajectories.model_calls import effort_outcome

    summary = {
        "calls": calls,
        "source": "proxy",
        "agent_effort": sent,
        "agent_effort_calls": calls,
    }
    effort = effort_outcome(summary, requested=requested, applied=None)
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
