"""The context window on the run card, and Claude Code's compactions per trial.

Claude Code assumes 200K tokens for a model id it does not recognize, as
every non-Claude model is through BenchFlow's proxy, and compacts there:
models served with 1M compacted 8 to 35 times in a published run's sample.
The card puts the window Claude Code assumes beside the model's, warns under
half, and counts each trial's compactions after the run.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.evaluation import Evaluation, EvaluationConfig
from benchflow.rollout._session_log import compactions_in, count_compactions
from benchflow.run_card import (
    build_run_card,
    card_lines,
    claude_code_window,
    served_window,
)
from benchflow.run_checks import RunChecks, after_run_lines


@pytest.mark.parametrize(
    "model_id,env,expected",
    [
        ("benchflow-xiaomi-mimo-v2.6-flash", {}, (200_000, "unrecognized")),
        ("benchflow-anthropic-claude-sonnet-5", {}, (1_000_000, "claude_1m")),
        ("us.anthropic.claude-opus-4-8", {}, (1_000_000, "claude_1m")),
        ("claude-sonnet-4-5", {}, (200_000, "claude_200k")),
        ("claude-opus-4-6[1m]", {}, (1_000_000, "claude_1m")),
        ("mimo-v2.6-pro[1m]", {}, (1_000_000, "1m_suffix")),
        (
            "benchflow-moonshot-kimi-k3",
            {"CLAUDE_CODE_MAX_CONTEXT_TOKENS": "1048576"},
            (1_048_576, "max_context_tokens"),
        ),
        # Only an id Claude Code does not recognize takes the variable alone.
        (
            "claude-sonnet-4-5",
            {"CLAUDE_CODE_MAX_CONTEXT_TOKENS": "1048576"},
            (200_000, "claude_200k"),
        ),
        (
            "claude-fable-5-1",
            {"CLAUDE_CODE_DISABLE_1M_CONTEXT": "1"},
            (200_000, "disabled_1m"),
        ),
    ],
)
def test_claude_code_s_window_for_a_model_id(model_id, env, expected):
    assert claude_code_window(model_id, env) == expected


def test_the_served_window_comes_from_litellm_s_map_or_the_provider_registry():
    assert served_window("gemini/gemini-2.5-pro") == (1_048_576, "litellm_model_map")
    assert served_window("zai-coding/glm-5.3") == (200_000, "provider_registry")
    assert served_window("someone/unknown-model-9") is None


def _card(tmp_path: Path, **config) -> tuple[dict, list[str]]:
    tasks = tmp_path / "tasks" / "task-0"
    tasks.mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"\n[agent]\ntimeout_sec = 60\n')
    (tasks / "instruction.md").write_text("task\n")
    job = Evaluation(
        tasks_dir=str(tasks.parent),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(**{"agent": "claude-agent-acp", **config}),
        preflight=False,
    )
    card = build_run_card(job, job._get_task_dirs())
    return card, card_lines(card, tmp_path / "job")


def test_a_window_under_half_the_model_s_is_flagged_with_the_setting(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "no.env"))
    card, lines = _card(tmp_path, model="gemini/gemini-2.5-pro")
    context = card["context"]
    assert (context["window"], context["window_source"]) == (200_000, "unrecognized")
    assert context["model_id"] == "runtime-gemini-gemini-2.5-pro"
    assert (context["served_window"], context["under_half"]) == (1_048_576, True)
    assert (
        "  Context:   Claude Code assumes 200000 tokens (Claude Code's assumed "
        "window for a model id it does not recognize); the model serves 1048576 "
        "(LiteLLM's model map): under half, so it compacts early; --agent-env "
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS=1048576 gives it the model's window"
    ) in lines
    given, _lines = _card(
        tmp_path / "given",
        model="gemini/gemini-2.5-pro",
        agent_env={"CLAUDE_CODE_MAX_CONTEXT_TOKENS": "1048576"},
    )
    assert given["context"]["window"] == 1_048_576
    assert given["context"]["under_half"] is False


def test_a_claude_model_s_own_window_is_not_flagged(tmp_path, monkeypatch):
    """LiteLLM 1.103.3's map gives Sonnet 4.5 1,000,000 input tokens (1.91.5's
    gave 200,000): the window of Anthropic's long-context beta, which Claude
    Code asks for with the [1m] suffix alone. Without it the model serves the
    run 200K, the window Claude Code assumes."""
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "no.env"))
    card, _lines = _card(tmp_path, model="anthropic/claude-sonnet-4-5")
    context = card["context"]
    assert (context["window"], context["served_window"]) == (200_000, 200_000)
    assert context["served_source"] == "claude_200k"
    assert context["under_half"] is False


def test_a_1m_claude_model_with_its_1m_window_turned_off_is_flagged(
    tmp_path, monkeypatch
):
    """Fable 5.1 serves 1M with no beta, so Claude Code held to 200K by
    CLAUDE_CODE_DISABLE_1M_CONTEXT compacts early."""
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "no.env"))
    card, _lines = _card(
        tmp_path,
        model="anthropic/claude-fable-5-1",
        agent_env={"CLAUDE_CODE_DISABLE_1M_CONTEXT": "1"},
    )
    context = card["context"]
    assert (context["window"], context["window_source"]) == (200_000, "disabled_1m")
    assert (context["served_window"], context["under_half"]) == (1_000_000, True)


# A compaction as Claude Code writes it in its session log.
_BOUNDARY = {
    "parentUuid": None,
    "logicalParentUuid": "6c1f",
    "isSidechain": False,
    "userType": "external",
    "cwd": "/app",
    "sessionId": "s-1",
    "version": "2.1.316",
    "type": "system",
    "subtype": "compact_boundary",
    "content": "Conversation compacted",
    "isMeta": False,
    "timestamp": "2026-11-05T10:00:00.000Z",
    "uuid": "u-1",
    "level": "info",
    "compactMetadata": {"trigger": "auto", "preTokens": 167000},
}


def _log_lines() -> list[str]:
    manual = {
        **_BOUNDARY,
        "compactMetadata": {"trigger": "manual", "preTokens": 90000},
    }
    subagent = {**_BOUNDARY, "isSidechain": True}
    other = {"type": "assistant", "message": {"content": "compact_boundary"}}
    return [json.dumps(row) for row in (_BOUNDARY, manual, subagent, other)]


def test_compactions_are_read_from_claude_code_s_boundary_lines():
    assert compactions_in(_log_lines()) == {
        "count": 2,
        "auto": 1,
        "manual": 1,
        "subagent": 1,
        "pre_tokens": [167000, 90000],
        "source": "claude-code-session-log",
    }


def _rollout(tmp_path: Path, env) -> SimpleNamespace:
    from benchflow.task.paths import RolloutPaths

    return SimpleNamespace(
        _config=SimpleNamespace(primary_agent="claude-agent-acp", sandbox_user="agent"),
        _env=env,
        _rollout_paths=RolloutPaths(tmp_path / "trial"),
        _compactions=None,
    )


async def test_teardown_counts_the_compactions_in_the_sandbox(tmp_path):
    asked: list[str] = []

    async def exec(command, **_kwargs):
        asked.append(command)
        return SimpleNamespace(return_code=0, stdout="\n".join(_log_lines()), stderr="")

    rollout = _rollout(tmp_path, SimpleNamespace(exec=exec))
    await count_compactions(rollout)
    assert rollout._compactions["count"] == 2
    assert "/home/agent/.claude/projects" in asked[0]


async def test_a_copied_session_log_is_read_without_the_sandbox(tmp_path):
    async def exec(command, **_kwargs):
        raise AssertionError("the copied log is read on the host")

    rollout = _rollout(tmp_path, SimpleNamespace(exec=exec))
    logs = rollout._rollout_paths.agent_dir / "claude-sessions" / "-app"
    logs.mkdir(parents=True)
    (logs / "s-1.jsonl").write_text("\n".join(_log_lines()) + "\n")
    await count_compactions(rollout)
    assert rollout._compactions["pre_tokens"] == [167000, 90000]


async def test_another_agent_s_trial_counts_none(tmp_path):
    rollout = _rollout(tmp_path, SimpleNamespace(exec=None))
    rollout._config.primary_agent = "codex-acp"
    await count_compactions(rollout)
    assert rollout._compactions is None


def test_the_card_after_the_run_counts_the_trials_that_compacted(tmp_path):
    checks = RunChecks(None, planned=3)
    for name, compactions in (
        ("a", {"count": 3, "pre_tokens": [190000, 195000, 199000]}),
        ("b", {"count": 0, "pre_tokens": []}),
        ("c", None),
    ):
        trial = tmp_path / name
        trial.mkdir()
        (trial / "result.json").write_text(
            json.dumps({"rollout_name": name, "compactions": compactions})
        )
        checks.record(name, trial)
    summary = checks.summary()
    assert summary["compactions"] == {
        "trials": 2,
        "compacted": 1,
        "compactions": 3,
        "max_pre_tokens": 199000,
        "compacted_trials": ["a"],
    }
    assert (
        "  Compacted: 1 of 2 trials compacted their context (3 compactions, the "
        "largest at 199000 tokens)"
    ) in after_run_lines(summary)


def test_the_trial_s_compactions_reach_result_json(tmp_path):
    from datetime import datetime

    from benchflow.rollout._results import _build_rollout_result

    counted = compactions_in(_log_lines())
    trial = tmp_path / "trial"
    trial.mkdir()
    _build_rollout_result(
        trial,
        task_name="task",
        rollout_name="trial",
        agent="claude-agent-acp",
        agent_name="claude-agent-acp",
        model="gemini/gemini-2.5-pro",
        n_tool_calls=0,
        prompts=["task"],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        compactions=counted,
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={},
    )
    result = json.loads((trial / "result.json").read_text())
    assert result["compactions"] == counted
