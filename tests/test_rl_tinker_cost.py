"""The Tinker cookbook's cost accounting (docs/examples/rl/tinker/tinker_cost.py).

Stdlib only, no network: a run's cost comes from the token tallies its records
carry, at list price. Covers the price arithmetic, which episodes pay for
training, the three kinds of run folders, and the budget gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "docs" / "examples" / "rl" / "tinker"
sys.path.insert(0, str(EXAMPLE))
import tinker_cost as tc  # noqa: E402

MODEL = "Qwen/Qwen3.8-27B"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def tokens(prefill: int, reusable: int, sampled: int, train: int) -> dict:
    return {
        "prefill": prefill,
        "reusable": reusable,
        "sampled": sampled,
        "train": train,
    }


def test_cached_prefill_bills_at_a_fifth_and_the_bound_ignores_the_cache():
    cost = tc.usd(MODEL, tokens(1_000_000, 750_000, 100_000, 2_000_000))
    prefill, sample, train = tc.PRICES[MODEL]
    common = 0.1 * sample + 2 * train
    assert cost["usd"] == pytest.approx(0.25 * prefill + 0.75 * prefill * 0.2 + common)
    assert cost["usd_no_cache"] == pytest.approx(prefill + common)
    assert tc.PRICES[MODEL] == (1.86, 5.595, 4.103)


def make_training_log(root: Path) -> Path:
    log = root / "train"
    write_jsonl(
        log / "trials" / "rollouts.jsonl",
        [
            # step 0: task a trained (mixed), task b dropped (constant group)
            {
                "step": "train-0000",
                "task_id": "a",
                "reward": 1.0,
                "tokens": tokens(100, 60, 10, 110),
            },
            {
                "step": "train-0000",
                "task_id": "a",
                "reward": 0.0,
                "tokens": tokens(100, 60, 10, 110),
            },
            {
                "step": "train-0000",
                "task_id": "b",
                "reward": 1.0,
                "tokens": tokens(50, 0, 5, 55),
            },
            # an infrastructure drop in a trained group: sampled, never trained
            {
                "step": "train-0000",
                "task_id": "a",
                "reward": None,
                "tokens": tokens(20, 0, 2, 22),
            },
            # step 1
            {
                "step": "train-0001",
                "task_id": "c",
                "reward": 0.5,
                "tokens": tokens(200, 150, 20, 220),
            },
            # not training (an evaluation record in the same folder)
            {
                "step": "eval-base",
                "task_id": "z",
                "reward": 1.0,
                "tokens": tokens(999, 0, 9, 0),
            },
        ],
    )
    write_jsonl(
        log / "groups.jsonl",
        [
            {"where": "train-0000", "task": "a", "kind": "mixed", "dropped": False},
            {"where": "train-0000", "task": "b", "kind": "all_solved", "dropped": True},
            {"where": "train-0001", "task": "c", "kind": "mixed", "dropped": False},
        ],
    )
    return log


def test_every_sampled_episode_pays_prefill_and_only_trained_ones_pay_training(
    tmp_path,
):
    steps = tc.training_steps(make_training_log(tmp_path))
    assert [s["step"] for s in steps] == [0, 1]
    first = steps[0]
    assert (first["episodes"], first["trained"]) == (4, 2)
    assert (first["prefill"], first["reusable"], first["sampled"]) == (270, 120, 27)
    assert first["train"] == 220
    assert steps[1]["train"] == 220


def test_cost_of_reads_training_logs_tinker_eval_results_and_evaluator_folders(
    tmp_path,
):
    log = make_training_log(tmp_path)
    training = tc.cost_of(log, model=MODEL)
    assert training["kind"] == "training"
    assert training["tokens"] == {
        "prefill": 470,
        "reusable": 270,
        "sampled": 47,
        "train": 440,
    }

    result = tmp_path / "calibration" / "eval.json"
    result.parent.mkdir()
    result.write_text(
        json.dumps(
            {
                "evaluations": [
                    {
                        "episodes": {
                            "t": [
                                {"tokens": tokens(100, 80, 10, 500)},
                                {"tokens": tokens(50, 0, 5, 60)},
                            ]
                        }
                    }
                ]
            }
        )
    )
    calibration = tc.cost_of(result, model=MODEL)
    # An evaluation trains nothing.
    assert calibration["tokens"] == {
        "prefill": 150,
        "reusable": 80,
        "sampled": 15,
        "train": 0,
    }

    folder = tmp_path / "eval-base"
    write_jsonl(
        folder / "episodes.jsonl",
        [
            {"prompt_tokens": 1000, "completion_tokens": 100},
            {"prompt_tokens": 3000, "completion_tokens": 300},
        ],
    )
    evaluated = tc.cost_of(folder, model=MODEL, reusable_fraction=0.5)
    assert evaluated["kind"] == "evaluate"
    assert evaluated["tokens"] == {
        "prefill": 4000,
        "reusable": 2000,
        "sampled": 400,
        "train": 0,
    }
    assert evaluated["usd"] < evaluated["usd_no_cache"]

    with pytest.raises(FileNotFoundError):
        tc.cost_of(tmp_path / "nothing", model=MODEL)


def gate_args(
    tmp_path: Path, log: Path, *, budget: float, cap: float | None = None
) -> list[str]:
    folder = tmp_path / "eval-base"
    write_jsonl(
        folder / "episodes.jsonl",
        [{"prompt_tokens": 1_000_000, "completion_tokens": 100_000}],
    )
    args = ["gate", "--model", MODEL, "--train", str(log), "--steps", "20"]
    args += ["--spent", str(folder), "--reserve", str(folder), "--budget", str(budget)]
    if cap is not None:
        args += ["--cap", str(cap)]
    return [*args, "--out", str(tmp_path / "gate.json")]


def test_the_gate_projects_from_the_measured_steps_and_stops_past_the_budget(tmp_path):
    log = make_training_log(tmp_path)
    assert tc.main(gate_args(tmp_path, log, budget=1000)) == 0
    doc = json.loads((tmp_path / "gate.json").read_text())
    steps = tc.training_steps(log)
    mean = sum(tc.usd(MODEL, s)["usd"] for s in steps) / len(steps)
    evaluation = tc.cost_of(
        tmp_path / "eval-base", model=MODEL, reusable_fraction=doc["reusable_share"]
    )["usd"]
    assert doc["projected"]["usd"] == pytest.approx(2 * evaluation + 20 * mean)
    assert doc["reusable_share"] == pytest.approx(270 / 470)
    assert doc["go"] and doc["reasons"] == []

    # The estimate fits the budget, but the no-cache bound passes the cap.
    assert tc.main(gate_args(tmp_path, log, budget=1000, cap=0.01)) == tc.STOP
    assert "cap" in json.loads((tmp_path / "gate.json").read_text())["reasons"][0]
    assert tc.main(gate_args(tmp_path, log, budget=0.01)) == tc.STOP


def test_set_aside_steps_count_as_spent_and_a_cut_line_is_skipped(tmp_path):
    log = make_training_log(tmp_path)
    write_jsonl(
        log / "trials" / "rollouts.discarded.jsonl",
        [
            {
                "step": "train-0001",
                "task_id": "c",
                "reward": 0.5,
                "tokens": tokens(1000, 0, 100, 1100),
            }
        ],
    )
    write_jsonl(
        log / "groups.discarded.jsonl",
        [{"where": "train-0001", "task": "c", "kind": "mixed", "dropped": False}],
    )
    with (log / "trials" / "rollouts.jsonl").open("a") as f:
        f.write(
            '{"step": "train-0001", "task_id": "c", "rew'
        )  # a writer killed mid-line
    cost = tc.cost_of(log, model=MODEL)
    assert cost["tokens"] == {
        "prefill": 1470,
        "reusable": 270,
        "sampled": 147,
        "train": 1540,
    }
    assert cost["set_aside_tokens"] == tokens(1000, 0, 100, 1100)
    # The gate's mean step leaves them out; its spend does not.
    assert [s["step"] for s in tc.training_steps(log)] == [0, 1]
    assert tc.main(gate_args(tmp_path, log, budget=1000)) == 0
    doc = json.loads((tmp_path / "gate.json").read_text())
    aside = tc.usd(MODEL, tokens(1000, 0, 100, 1100))
    assert doc["set_aside"]["usd"] == pytest.approx(aside["usd"])
    mean = sum(tc.usd(MODEL, s)["usd"] for s in tc.training_steps(log)) / 2
    spent = sum(c["usd"] for c in doc["spent"]) + sum(c["usd"] for c in doc["reserve"])
    assert doc["projected"]["usd"] == pytest.approx(spent + aside["usd"] + 20 * mean)
