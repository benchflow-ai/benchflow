"""A cost that is no provider's bill is shown as an estimate.

Review of PR #122 (F2b-R122B, 2026-11-13, L-a): `bench results` and the
cost line of report.md and `bench eval metrics` put "~" before a cost only
when the agent's session log priced it. Every other list-price estimate
(the model gateway's LiteLLM figure on an API-key route, a login's tokens at
list prices, a hosted run) printed as `$x.xxxx`, as an OpenRouter bill does,
and a hosted run has no other place its cost appears. Every cost whose
``cost_source`` is not ``provider`` is now marked. Every value is made up.
"""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

import benchflow as bf
from benchflow.cli.main import app
from benchflow.jobs import _cost_line

TRIALS = {
    # name: (cost_usd, cost_source, price_source)
    "billed": (0.01, "provider", "openrouter"),
    "gateway": (0.0065, "list_price_estimate", "litellm"),
    "hosted": (0.0123, "list_price_estimate", "list_price"),
    "session": (0.0421, "list_price_estimate", "agent_session_log"),
}


def _job(tmp_path):
    job = tmp_path / "job"
    for name, (cost, cost_source, price_source) in TRIALS.items():
        trial = job / f"{name}__0a1b2c3d"
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": name,
                    "rollout_name": trial.name,
                    "rewards": {"reward": 1.0},
                    "agent_name": "claude-agent-acp",
                    "model": "claude-opus-5-5",
                    "started_at": "2026-11-13T18:32:47+00:00",
                    "finished_at": "2026-11-13T18:38:53+00:00",
                    "agent_result": {
                        "n_input_tokens": 1000,
                        "n_output_tokens": 100,
                        "cost_usd": cost,
                        "usage_source": "end_of_turn",
                        "cost_source": cost_source,
                        "price_source": price_source,
                    },
                }
            )
        )
    return job


def _rows(output: str) -> dict[str, str]:
    rows = {}
    for line in output.splitlines():
        cells = [c.strip() for c in re.split("[│┃]", line)]
        if len(cells) > 2 and cells[1] in TRIALS:
            rows[cells[1]] = next(c for c in cells if "$" in c)
    return rows


def test_bench_results_marks_every_cost_that_is_no_bill(tmp_path):
    shown = CliRunner().invoke(
        app, ["results", str(_job(tmp_path))], env={"COLUMNS": "300"}
    )
    assert shown.exit_code == 0, shown.output
    assert _rows(shown.stdout) == {
        "billed": "$0.0100",
        "gateway": "~$0.0065",
        "hosted": "~$0.0123",
        "session": "~$0.0421",
    }
    assert "~ an estimate (tokens at list prices, or the agent's own figure)" in (
        " ".join(shown.stdout.split())
    )


def test_the_cost_line_counts_every_estimate(tmp_path):
    job = bf.load_job(_job(tmp_path))
    assert _cost_line(job.trials) == (
        "Cost: $0.0709 over 4 rollout(s) (3 estimated, not a provider's bill; "
        "1 of them from the agent's session log)"
    )
    assert _cost_line(job.trials) in job.to_markdown()
    billed = [t for t in job.trials if t.result.cost_source == "provider"]
    assert _cost_line(billed) == "Cost: $0.0100 over 1 rollout(s)"
