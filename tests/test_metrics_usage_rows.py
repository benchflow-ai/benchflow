"""`bench eval metrics` shows a job's usage and model calls.

Usage sat in report.md and each trial's model-calls.json, and no bench command
printed it: getting a Claude Code run's usage took a file path. The metrics
table now has a Tokens row (with the usage's source), a Cost row (report.md's
line, which says what was estimated from the agent's session log) and a Model
calls row (calls, cuts at the output cap, refusals, provider errors, and the
source). The two trials are the same run as the proxy saw it and as a
subscription run around the proxy records it: tokens over ACP, the cost
estimated from Claude Code's session log.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from benchflow.cli._errors import run_cli
from benchflow.trajectories import model_calls as mc

FIXTURES = Path(__file__).parent / "fixtures" / "model_calls"


def _trial(
    job: Path, name: str, *, fixture: str, kind: str, cost: float | None = None
) -> None:
    trial = job / name
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": name.split("__")[0],
                "rollout_name": name,
                "rewards": {"reward": 1.0},
                "started_at": "2026-11-04T18:32:47+00:00",
                "finished_at": "2026-11-04T18:38:53+00:00",
                "agent_name": "claude-agent-acp",
                "agent_result": {
                    "n_input_tokens": 577009,
                    "n_output_tokens": 7747,
                    "cost_usd": cost,
                    "usage_source": kind,
                    "price_source": "agent_session_log" if cost is not None else None,
                },
            }
        )
    )
    if kind == "provider_response":
        (trial / "trajectory").mkdir()
        shutil.copy(FIXTURES / fixture, trial / "trajectory" / "llm_trajectory.jsonl")
    else:
        folder = trial / "agent" / "claude-sessions" / "-app"
        folder.mkdir(parents=True)
        shutil.copy(FIXTURES / fixture, folder / "s.jsonl")
    mc.write_model_calls(trial)


def test_metrics_shows_tokens_and_model_calls_with_their_sources(tmp_path, capsys):
    job = tmp_path / "job"
    _trial(
        job, "essay-task__aaaa", fixture="trunc-proxy.jsonl", kind="provider_response"
    )
    _trial(
        job,
        "essay-task__bbbb",
        fixture="trunc-claude-session.jsonl",
        kind="agent_native_acp",
        cost=0.0421,
    )
    from benchflow.cli.main import app

    with pytest.raises(SystemExit):
        run_cli(app, ["eval", "metrics", str(job)])
    # The table wraps long cells at the console's width; read it as one line.
    out = " ".join(re.sub(r"[│┃]", " ", capsys.readouterr().out).split())
    assert (
        "Tokens 1,154,018 in, 15,494 out over 2 of 2 trials "
        "(from agent_native_acp, provider_response)"
    ) in out
    assert (
        "Cost $0.0421 over 1 rollout(s) (1 estimated from the agent's session log); "
        "1 rollout(s) reported no USD, so the total is a floor"
    ) in out
    assert "Model calls 60 over 2 trials; 20 cut at the output cap" in out
    assert "from claude-code-session-log (1), proxy (1)" in out
