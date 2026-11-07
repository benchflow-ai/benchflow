"""The run card names every variable that changes what a trial measures,
with its value and where it came from (benchflow.run_card.env_settings)."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.evaluation import Evaluation, EvaluationConfig
from benchflow.run_card import build_run_card, card_lines


def _tasks(tmp_path: Path) -> Path:
    root = tmp_path / "tasks"
    task = root / "task-0"
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text(
        'version = "1.0"\n[agent]\ntimeout_sec = 600\n[verifier]\ntimeout_sec = 120\n'
    )
    (task / "instruction.md").write_text("Write 42 to /app/answer.txt.\n")
    (task / "tests" / "test.sh").write_text("#!/bin/sh\n")
    return root


def _card(tmp_path: Path, **config) -> tuple[dict, list[str]]:
    base = {"agent": "claude-agent-acp", "model": "claude-sonnet-5"}
    base.update(config)
    job = Evaluation(
        tasks_dir=str(_tasks(tmp_path)),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(**base),
        preflight=False,
    )
    card = build_run_card(job, job._get_task_dirs())
    return card, card_lines(card, tmp_path / "job")


@pytest.fixture(autouse=True)
def _no_operator_env(monkeypatch, tmp_path):
    """Nothing from the machine running the tests: no .env, none of the
    variables the card reads."""
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "no.env"))
    for name in (
        "BENCHFLOW_BEDROCK_THINKING_EFFORT",
        "BENCHFLOW_AGENT_PENDING_TOOL_GRACE_SEC",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
        "MAX_THINKING_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)


def _setting(card: dict, name: str) -> dict | None:
    return next((s for s in card["settings"] if s["name"] == name), None)


def _line(lines: list[str], head: str) -> str:
    return next(line for line in lines if line.startswith(head))


def test_benchflow_s_own_output_cap_is_named_as_its_default(tmp_path):
    card, lines = _card(tmp_path)
    cap = _setting(card, "CLAUDE_CODE_MAX_OUTPUT_TOKENS")
    assert cap is not None
    assert (cap["value"], cap["source"]) == ("128000", "benchflow_default")
    assert card["agent"]["output_cap_source"] == "benchflow_default"
    assert (
        "caps Claude Code's output at 128000 tokens (CLAUDE_CODE_MAX_OUTPUT_TOKENS), "
        "BenchFlow's default;" in _line(lines, "  Sampling:")
    )
    assert _line(lines, "  Settings:") == (
        "  Settings:  CLAUDE_CODE_MAX_OUTPUT_TOKENS 128000 (BenchFlow's default)"
    )


def test_agent_env_settings_are_named_as_the_run_s(tmp_path):
    card, lines = _card(
        tmp_path,
        agent_env={
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000",
            "MAX_THINKING_TOKENS": "31999",
        },
    )
    assert card["agent"]["output_cap_source"] == "agent_env"
    thinking = _setting(card, "MAX_THINKING_TOKENS")
    assert thinking is not None
    assert (thinking["value"], thinking["source"]) == ("31999", "agent_env")
    assert thinking["reads"] == "Claude Code"
    assert _line(lines, "  Settings:") == (
        "  Settings:  CLAUDE_CODE_MAX_OUTPUT_TOKENS 128000 (from --agent-env); "
        "MAX_THINKING_TOKENS 31999 (from --agent-env)"
    )


def test_a_host_export_the_agent_never_gets_is_not_its_setting(tmp_path, monkeypatch):
    """BenchFlow copies only some names from the host into the agent's
    environment: an exported CLAUDE_CODE_MAX_OUTPUT_TOKENS stays on the host,
    and the agent runs with BenchFlow's default."""
    monkeypatch.setenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "32000")
    card, _lines = _card(tmp_path)
    cap = _setting(card, "CLAUDE_CODE_MAX_OUTPUT_TOKENS")
    assert cap is not None
    assert (cap["value"], cap["source"]) == ("128000", "benchflow_default")


def test_an_inherited_setting_names_the_host_or_the_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_BEDROCK_THINKING_EFFORT", "medium")
    card, lines = _card(tmp_path)
    effort = _setting(card, "BENCHFLOW_BEDROCK_THINKING_EFFORT")
    assert effort is not None
    assert (effort["value"], effort["source"]) == ("medium", "host_env")
    assert "BENCHFLOW_BEDROCK_THINKING_EFFORT medium (from the host's environment)" in (
        _line(lines, "  Settings:")
    )
    # .env wins over the host, as the agent's environment is resolved.
    dotenv = tmp_path / ".env"
    dotenv.write_text("BENCHFLOW_BEDROCK_THINKING_EFFORT=low\n")
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(dotenv))
    card, _lines = _card(tmp_path / "again")
    effort = _setting(card, "BENCHFLOW_BEDROCK_THINKING_EFFORT")
    assert effort is not None
    assert (effort["value"], effort["source"]) == ("low", "dotenv")


def test_a_bedrock_claude_model_shows_the_proxy_s_default_effort(tmp_path):
    card, _lines = _card(tmp_path, model="aws-bedrock/us.anthropic.claude-opus-4-8")
    effort = _setting(card, "BENCHFLOW_BEDROCK_THINKING_EFFORT")
    assert effort is not None
    assert (effort["value"], effort["source"]) == ("high", "benchflow_default")


def test_the_watchdog_s_own_setting_is_read_from_the_host(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_AGENT_PENDING_TOOL_GRACE_SEC", "3600")
    card, _lines = _card(tmp_path)
    grace = _setting(card, "BENCHFLOW_AGENT_PENDING_TOOL_GRACE_SEC")
    assert grace is not None
    assert (grace["value"], grace["source"]) == ("3600", "host_env")
    assert grace["reads"] == "BenchFlow's idle watchdog"


def test_an_agent_s_setting_is_not_on_another_agent_s_card(tmp_path):
    card, lines = _card(
        tmp_path,
        agent="openhands",
        model="openai/gpt-5.5",
        agent_env={"MAX_THINKING_TOKENS": "1000"},
    )
    assert card["settings"] == []
    assert card["agent"]["output_cap"] is None
    assert "none of the variables BenchFlow knows" in _line(lines, "  Settings:")


def test_a_value_that_is_not_a_plain_setting_is_not_written(tmp_path):
    card, _lines = _card(tmp_path, agent_env={"CLAUDE_CODE_SUBAGENT_MODEL": "a b; c=d"})
    subagent = _setting(card, "CLAUDE_CODE_SUBAGENT_MODEL")
    assert subagent is not None
    assert subagent["value"] == "(set)"
