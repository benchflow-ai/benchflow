"""Training exports from real rollouts: reward vectors, group advantages,
token-coverage. Reads the ``--matrix`` job of ``test_results.py`` (nine
fake-agent rollouts: three tasks x three trials) and the agent branch job."""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from tests.e2e import harness as h
from tests.e2e.test_results import matrix_dir  # noqa: F401  (shared fixture)


def test_convert_with_reward_vector_and_group_advantage(
    matrix_dir: Path,  # noqa: F811
    jobs_root: Path,
    ledger: h.Ledger,
):
    out = jobs_root / "train-signal.jsonl"
    manifest = jobs_root / "train-signal.manifest.json"
    run = h.bench(
        "train", "convert", matrix_dir, "--out", out, "--manifest", manifest,
        "--reward-vector", "--group-advantage", "grpo", "--group-by", "agent,model",
    )  # fmt: skip
    ledger.record("train convert --reward-vector --group-advantage grpo", surface="CLI",
                  seconds=run.seconds)  # fmt: skip
    h.assert_exit(run, 0)
    rows = h.read_jsonl(out)
    assert len(rows) == 9, len(rows)
    rewards = []
    for row in rows:
        signal = {
            k: row[k] for k in ("reward_vector", "advantage", "group") if k in row
        }
        h.validate(signal, "benchflow-training-signal.v1.schema.json")
        assert row["reward_vector"]["source"] == "verifier"
        rewards.append(row["reward"])
    # One group (agent, model) of nine: rewards 1,1,1, 0,0,0, 0.5,0.5,0.5.
    assert sorted(rewards) == [0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1]
    mean = statistics.mean(rewards)
    std = statistics.stdev(rewards)
    for row in rows:
        assert row["advantage"] == pytest.approx((row["reward"] - mean) / (std + 1e-4))
        assert row["group"]["scored"] == 9
    assert h.read_json(manifest)["training_signal"]
    run = h.bench("train", "validate", out, "--expected-rows", "9")
    h.assert_exit(run, 0)


def test_token_coverage_reports(
    matrix_dir: Path,  # noqa: F811
    jobs_root: Path,
    ledger: h.Ledger,
):
    """Gateway rollouts without token capture are not training-grade, with a reason."""
    job = sorted(p.parent for p in matrix_dir.glob("trial-*/*/results.jsonl"))[0]
    run, report = h.bench_json("train", "token-coverage", job, "--json")
    ledger.record(
        "train token-coverage on gateway rollouts", surface="CLI", seconds=run.seconds
    )
    assert run.returncode in (0, 1), run.tail()
    text = str(report)
    assert "training" in text.lower() or "rollouts" in text.lower(), report
    assert report, report
