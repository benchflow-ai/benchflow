"""A hosted run with a model gets a cost (review of PR #122, L3; the
founder's rule: tokens and cost on every run).

A verifiers v1 run's traces count its tokens, and result.json recorded them
with no cost and no reason. They are now priced by the pricing step every
trial's tokens go through, labelled ``list_price_estimate`` with the price
table's date; v1 reports no cache split, so every input token is priced as
uncached input. The count is the runner's own (``usage_source`` end_of_turn,
``usage_source_legacy`` agent-reported), kept out of metered totals as an
agent that routes its own calls is.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._utils.pricing import list_rates
from benchflow.hosted_env import run_hosted_env
from tests.test_hosted_env_v1 import FIXTURE, _config, _dist_info


def _run(tmp_path, monkeypatch, **config):
    def fake_run(cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        if cmd[1:3] == ["venv", "--python"]:
            _dist_info(Path(cmd[-1]), "0.3.2.dev137", "verifiers.v1.cli.eval.main:main")
        if cmd[0].endswith("vf-eval"):
            out = Path(cmd[cmd.index("--output-dir") + 1]) / "benchflow"
            out.mkdir(parents=True)
            shutil.copy(FIXTURE, out / "traces.jsonl")
            return SimpleNamespace(returncode=0, stdout="rollout done\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("benchflow.hosted_env.shutil.which", lambda b: f"/bin/{b}")
    monkeypatch.setattr("benchflow.hosted_env.subprocess.run", fake_run)
    result = run_hosted_env(
        _config(jobs_dir=tmp_path, verifiers_version="0.3.2.dev137", **config)
    )
    return json.loads((result.run_dir / "result.json").read_text())


def test_a_hosted_run_s_tokens_are_priced(tmp_path, monkeypatch):
    payload = _run(tmp_path, monkeypatch, model="claude-opus-5-5")
    agent = payload["agent_result"]
    rates = list_rates("anthropic/claude-opus-5-5")
    assert rates is not None and rates.table_date is not None
    # The fixture's traces: 5 input and 1 output token.
    assert (agent["n_input_tokens"], agent["n_output_tokens"]) == (5, 1)
    assert agent["cost_usd"] == pytest.approx(5 * rates.input + 1 * rates.output)
    assert agent["cost_source"] == "list_price_estimate"
    assert agent["price_table_date"] == rates.table_date
    assert agent["usage_source"] == "end_of_turn"
    assert agent["usage_source_legacy"] == "agent-reported"
    extra = payload["final_metrics"]["extra"]
    assert extra["cost_source"] == "list_price_estimate"
    assert payload["final_metrics"]["total_cost_usd"] == agent["cost_usd"]


def test_a_model_with_no_list_price_keeps_an_unknown_cost(tmp_path, monkeypatch):
    payload = _run(tmp_path, monkeypatch, model="acme/unlisted-model-7")
    agent = payload["agent_result"]
    assert agent["total_tokens"] == 6
    assert agent["cost_usd"] is None
    assert agent["cost_source"] is None
