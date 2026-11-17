"""`bench eval metrics` shows a job's usage and model calls.

Usage sat in report.md and each trial's model-calls.json, and no bench command
printed it: getting a Claude Code run's usage took a file path. The metrics
table now has a Tokens row (with the usage's source), two cost rows (the kept
trials' total_cost_usd and every attempt's spend_all_attempts) and a Model
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
    # Two labelled figures, as summary.json names them: what the
    # kept trials cost and what every attempt spent, unknown counted apart.
    assert (
        "Cost (total_cost_usd) kept trials: $0.0421 over 1 trial; "
        "1 unknown (no cost reported)"
    ) in out
    assert (
        "Spend (spend_all_attempts) every attempt: $0.0421 over 2 attempts, "
        "retries and judges included; 1 unknown (no cost reported)"
    ) in out
    assert "Model calls 60 over 2 trials; 20 cut at the output cap" in out
    assert "from claude-code-session-log (1), proxy (1)" in out


def _priced(job: Path, name: str, cost: float, *, incomplete: bool = False) -> None:
    trial = job / name
    trial.mkdir(parents=True)
    agent = {
        "n_input_tokens": 1000,
        "n_output_tokens": 100,
        "total_tokens": 1100,
        "cost_usd": cost,
        "usage_source": "end_of_turn",
        "cost_source": "list_price_estimate",
        "price_source": "list_price",
    }
    if incomplete:
        agent.update(
            usage_incomplete=True,
            usage_incomplete_reason="a turn ended without its usage report",
            cost_incomplete=True,
        )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": name.split("__")[0],
                "rollout_name": name,
                "rewards": {"reward": 1.0},
                "started_at": "2026-11-14T08:00:00+00:00",
                "finished_at": "2026-11-14T08:01:00+00:00",
                "agent_name": "claude-agent-acp",
                "model": "claude-opus-5-5",
                "agent_result": agent,
            }
        )
    )


def test_a_cost_from_a_count_that_missed_calls_is_a_floor_in_every_reader(
    tmp_path, capsys
):
    """Review of PR #147 (F2b-R147, M1): #147's Cost and Spend rows replaced
    ``_cost_line``, and neither knew a cost priced from a count that missed
    calls, so ``bench eval metrics`` dropped #122's "the total is a floor".
    The table, ``--json``, the run card and ``bench results`` now agree."""
    from typer.testing import CliRunner

    from benchflow._utils.evaluation_results import usage_summary
    from benchflow.cli.main import app
    from benchflow.evaluation import RUN_CARD_USAGE_FIELDS

    job = tmp_path / "job"
    _priced(job, "one__00000001", 0.01)
    _priced(job, "two__00000002", 0.02, incomplete=True)

    with pytest.raises(SystemExit):
        run_cli(app, ["eval", "metrics", str(job)])
    out = " ".join(re.sub(r"[│┃]", " ", capsys.readouterr().out).split())
    assert (
        "kept trials: $0.0300 over 2 trials (list_price_estimate 2); 1 trial "
        "counted only some of their calls, so the total is a floor"
    ) in out
    assert "1 attempt counted only some of their calls, so the total is a floor" in out

    with pytest.raises(SystemExit):
        run_cli(app, ["eval", "metrics", str(job), "--json"])
    figures = json.loads(capsys.readouterr().out)
    assert figures["trials_cost_incomplete"] == 1
    assert figures["trials_usage_incomplete"] == 1
    assert figures["spend_all_attempts"]["attempts_cost_incomplete"] == 1

    rows = {p.parent.name: json.loads(p.read_text()) for p in job.glob("*/result.json")}
    card = {k: usage_summary(rows).get(k) for k in RUN_CARD_USAGE_FIELDS}
    assert (card["trials_cost_incomplete"], card["trials_usage_incomplete"]) == (1, 1)

    shown = CliRunner().invoke(app, ["results", str(job)], env={"COLUMNS": "300"})
    plain = " ".join(re.sub("[─-╿]", " ", shown.stdout).split())
    assert "~$0.0200+" in plain
    assert "1 trial(s) counted only some of their calls" in plain
