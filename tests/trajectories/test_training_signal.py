"""Reward vectors and group-relative advantages in training exports.

Training on rubric-graded tasks needs more than one scalar per rollout:
per-criterion credit and group scoring. ``bench train convert --reward-vector --group-advantage`` and the
matching keyword arguments of the Prime-SFT and TRL exporters emit the
rubric's per-criterion scores with their names and weights, and GRPO-style or
leave-one-out advantages over the rollouts of the same task, recording the
grouping key and normalisation. Unscored rollouts stay unscored (null, never
0) and never enter a group's baseline.

The numbers in ``test_two_scored_two_unscored_group`` are four rollouts of one
task: 0.0, 1.0 and two ``pipe_closed`` rollouts without a reward.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.trajectories.export_prime_sft import (
    convert_benchflow_rollouts_to_prime_sft_rows,
    export_prime_sft_jsonl,
)
from benchflow.trajectories.export_trl_sft import (
    convert_benchflow_rollouts_to_trl_sft_rows,
)
from benchflow.trajectories.training_signal import rollout_training_signals
from tests.trajectories.test_export_prime_sft import _exchange

runner = CliRunner()


def _rollout(
    job: Path,
    name: str,
    *,
    reward: float | None,
    task: str = "demo-task",
    agent: str = "opencode",
    model: str = "vendor/model-a",
    rewards: dict | None = None,
    error: str | None = None,
    extra: dict | None = None,
) -> Path:
    d = job / name
    d.mkdir(parents=True)
    result = {
        "task_name": task,
        "rollout_name": name,
        "agent": agent,
        "model": model,
        "rewards": rewards
        if rewards is not None
        else ({"reward": reward} if reward is not None else None),
        "error": error,
        "error_category": "pipe_closed" if error else None,
        **(extra or {}),
    }
    (d / "result.json").write_text(json.dumps(result))
    (d / "trajectory").mkdir()
    (d / "trajectory" / "llm_trajectory.jsonl").write_text(
        "\n".join(
            json.dumps(e) for e in (_exchange(final=False), _exchange(final=True))
        )
        + "\n"
    )
    return d


def _by_rollout(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for row in rows:
        path = row.get("source_rollout_dir") or str(Path(row["source_path"]).parents[1])
        out[Path(path).name] = row
    return out


def test_two_scored_two_unscored_group(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _rollout(job, "demo__00000001", reward=0.0)
    _rollout(job, "demo__00000002", reward=1.0)
    _rollout(job, "demo__00000003", reward=None, error="transport closed")
    _rollout(job, "demo__00000004", reward=None, error="transport closed")

    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(
        job, reward_vector=True, group_advantage="grpo"
    )
    by = _by_rollout(rows)
    assert sorted(by) == ["demo__00000001", "demo__00000002"]
    std = math.sqrt(0.5)  # sample std of (0, 1)
    assert by["demo__00000002"]["advantage"] == pytest.approx(0.5 / (std + 1e-4))
    assert by["demo__00000001"]["advantage"] == pytest.approx(-0.5 / (std + 1e-4))
    group = by["demo__00000002"]["group"]
    assert group["id"] == ("task=demo-task|agent=opencode|model=vendor/model-a")
    assert group["by"] == ["task", "agent", "model"]
    assert group["normalisation"] == "grpo"
    assert group["rollouts"] == 4
    assert group["scored"] == 2
    assert group["mean"] == pytest.approx(0.5)
    assert group["std"] == pytest.approx(std)
    assert group["std_ddof"] == 1
    assert group["eps"] == 1e-4
    assert by["demo__00000002"]["reward_vector"] == {
        "source": "verifier",
        "names": ["reward"],
        "kinds": ["verifier"],
        "weights": [None],
        "values": [1.0],
        "formula": "reward = rewards.reward (as written by the verifier)",
    }

    signal = stats.training_signal
    assert signal["group_advantage"] == "grpo"
    assert signal["group_by"] == ["task", "agent", "model"]
    (entry,) = signal["groups"]
    members = {m["rollout"]: m for m in entry["members"]}
    assert members["demo__00000003"]["reward"] is None
    assert members["demo__00000003"]["advantage"] is None
    assert members["demo__00000003"]["excluded"] == "unscored"
    assert members["demo__00000002"]["advantage"] == pytest.approx(0.5 / (std + 1e-4))

    rows, _ = convert_benchflow_rollouts_to_prime_sft_rows(job, group_advantage="loo")
    by = _by_rollout(rows)
    assert by["demo__00000002"]["advantage"] == pytest.approx(1.0)
    assert by["demo__00000001"]["advantage"] == pytest.approx(-1.0)
    assert by["demo__00000001"]["group"]["normalisation"] == "loo"
    assert "reward_vector" not in by["demo__00000001"]


def test_leave_one_out_three_scored(tmp_path: Path) -> None:
    job = tmp_path / "job"
    for name, reward in (("a", 0.0), ("b", 0.5), ("c", 1.0)):
        _rollout(job, name, reward=reward)
    rows, _ = convert_benchflow_rollouts_to_prime_sft_rows(job, group_advantage="loo")
    by = _by_rollout(rows)
    assert by["a"]["advantage"] == pytest.approx(-0.75)
    assert by["b"]["advantage"] == pytest.approx(0.0)
    assert by["c"]["advantage"] == pytest.approx(0.75)


def test_single_scored_rollout_and_equal_rewards(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _rollout(job, "solo", reward=1.0, task="t1")
    _rollout(job, "x", reward=1.0, task="t2")
    _rollout(job, "y", reward=1.0, task="t2")
    for method in ("grpo", "loo"):
        rows, _ = convert_benchflow_rollouts_to_prime_sft_rows(
            job, group_advantage=method
        )
        by = _by_rollout(rows)
        assert by["solo"]["advantage"] is None
        assert by["solo"]["group"]["excluded"] == "single_scored_rollout"
        assert by["x"]["advantage"] == 0.0
        assert by["y"]["advantage"] == 0.0
        assert by["x"]["group"]["std"] == 0.0


def test_group_by_keeps_models_apart_and_is_configurable(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _rollout(job, "d0", reward=0.0)
    _rollout(job, "d1", reward=1.0)
    _rollout(job, "o1", reward=1.0, agent="claude-agent-acp", model="vendor/model-b")
    _rollout(job, "o2", reward=1.0, agent="claude-agent-acp", model="vendor/model-b")
    rows, _ = convert_benchflow_rollouts_to_prime_sft_rows(job, group_advantage="grpo")
    by = _by_rollout(rows)
    assert by["o1"]["advantage"] == 0.0
    assert by["d1"]["advantage"] > 0

    rows, _ = convert_benchflow_rollouts_to_prime_sft_rows(
        job, group_advantage="grpo", group_by=("task",)
    )
    by = _by_rollout(rows)
    assert by["o1"]["group"]["id"] == "task=demo-task"
    assert by["o1"]["group"]["scored"] == 4
    assert by["o1"]["advantage"] == pytest.approx(0.25 / (0.5 + 1e-4))

    with pytest.raises(ValueError, match="group_by"):
        convert_benchflow_rollouts_to_prime_sft_rows(
            job, group_advantage="grpo", group_by=("task", "seed")
        )


def test_min_reward_does_not_move_the_baseline(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _rollout(job, "lo", reward=0.0)
    _rollout(job, "hi", reward=1.0)
    rows, _ = convert_benchflow_rollouts_to_prime_sft_rows(
        job, group_advantage="grpo", min_reward=1.0
    )
    (row,) = rows
    assert row["group"]["mean"] == pytest.approx(0.5)
    assert row["advantage"] > 0


def test_verifier_multi_key_rewards_become_a_vector(tmp_path: Path) -> None:
    """Two reward shapes verifiers write: a flat second key and a
    nested ``metrics`` dict."""
    job = tmp_path / "job"
    _rollout(
        job,
        "flat",
        reward=None,
        rewards={"reward": 0.0, "task_success": 0.0, "flag": True},
    )
    _rollout(
        job,
        "nested",
        reward=None,
        rewards={"reward": 1, "metrics": {"task_success": 1, "nan": float("nan")}},
    )
    rows, _ = convert_benchflow_rollouts_to_prime_sft_rows(
        job, reward_vector=True, group_advantage="grpo"
    )
    by = _by_rollout(rows)
    assert by["flat"]["reward_vector"]["names"] == ["reward", "task_success"]
    assert by["flat"]["reward_vector"]["values"] == [0.0, 0.0]
    assert by["nested"]["reward_vector"]["names"] == [
        "reward",
        "metrics.task_success",
    ]
    assert by["nested"]["reward_vector"]["values"] == [1.0, 1.0]
    assert by["nested"]["group"]["vector_names_differ"] is True
    # Per-component advantages use only the members that have the component.
    assert by["nested"]["advantage_vector"][0] > 0
    assert by["nested"]["advantage_vector"][1] is None


RUBRIC = {
    "criteria": [
        {"name": "answer_correct", "blocker": 1, "weight": 1, "description": "d"},
        {"name": "explanation_clear", "blocker": 0, "weight": 3, "description": "d"},
        {"name": "tests_added", "blocker": 0, "weight": 1, "description": "d"},
    ]
}


def _rubric_rollout(job: Path, name: str, checks: dict, rubric_reward: float) -> Path:
    scoring = {
        "schema_version": 1,
        "policy": "tests-blockers-quality-v1",
        "status": "complete",
        "passed": True,
        "tests_pass": True,
        "all_blockers_pass": True,
        "failed_blockers": [],
        "verifier_reward": 1.0,
        "rubric_reward": rubric_reward,
        "reviewer_run": "reviews/r/run",
        "revision": "scoring/r1.json",
        "error": None,
    }
    d = _rollout(
        job,
        name,
        reward=None,
        rewards={
            "reward": rubric_reward,
            "verifier_reward": 1.0,
            "rubric_reward": rubric_reward,
        },
        extra={"scoring": scoring},
    )
    (d / "reviews" / "r1").mkdir(parents=True)
    (d / "reviews" / "r1" / "rubric.json").write_text(json.dumps(RUBRIC))
    (d / "scoring").mkdir()
    (d / "scoring" / "r1.json").write_text(
        json.dumps(
            {
                "attempt": "r1",
                "contract": "v0.2",
                "rubric_sha256": "9ac3",
                "rubric_snapshot": "reviews/r1/rubric.json",
                "checks": checks,
                "scoring": scoring,
            }
        )
    )
    return d


def test_rubric_criteria_become_a_weighted_vector(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _rubric_rollout(
        job,
        "good",
        {
            "answer_correct": {"outcome": "pass", "explanation": "e"},
            "explanation_clear": {"score": 1, "explanation": "e"},
            "tests_added": {"score": 2, "explanation": "e"},
        },
        0.625,
    )
    _rubric_rollout(
        job,
        "partial",
        {
            "answer_correct": {"outcome": "pass", "explanation": "e"},
            "explanation_clear": {"score": 0, "explanation": "e"},
            "tests_added": {"explanation": "no score recorded"},
        },
        0.0,
    )
    rows, _ = convert_benchflow_rollouts_to_prime_sft_rows(
        job, reward_vector=True, group_advantage="loo"
    )
    by = _by_rollout(rows)
    vector = by["good"]["reward_vector"]
    assert vector["source"] == "rubric"
    assert vector["revision"] == "scoring/r1.json"
    assert vector["rubric_sha256"] == "9ac3"
    assert vector["names"] == [
        "tests",
        "answer_correct",
        "explanation_clear",
        "tests_added",
    ]
    assert vector["kinds"] == ["gate", "blocker", "scored", "scored"]
    assert vector["weights"] == [None, None, 3, 1]
    assert vector["values"] == [1.0, 1.0, 0.5, 1.0]
    assert by["good"]["reward"] == 0.625
    # A criterion without a score is null, never 0.
    assert by["partial"]["reward_vector"]["values"] == [1.0, 1.0, 0.0, None]
    assert by["good"]["advantage_vector"] == pytest.approx([0.0, 0.0, 0.5, None])
    assert by["good"]["advantage"] == pytest.approx(0.625)


def test_missing_rubric_revision_falls_back_to_scoring(tmp_path: Path) -> None:
    job = tmp_path / "job"
    d = _rubric_rollout(job, "r", {}, 0.625)
    (d / "scoring" / "r1.json").unlink()
    rows, _ = convert_benchflow_rollouts_to_prime_sft_rows(job, reward_vector=True)
    vector = rows[0]["reward_vector"]
    assert vector["source"] == "scoring"
    assert vector["names"] == ["tests", "rubric_reward"]
    assert vector["values"] == [1.0, 0.625]
    assert "scoring/r1.json" in vector["note"]


def test_trl_rows_carry_the_same_signal(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _rollout(job, "lo", reward=0.0)
    _rollout(job, "hi", reward=1.0)
    rows, stats = convert_benchflow_rollouts_to_trl_sft_rows(
        job, row_mode="rollout", reward_vector=True, group_advantage="loo"
    )
    by = _by_rollout(rows)
    assert by["hi"]["advantage"] == pytest.approx(1.0)
    assert by["hi"]["reward_vector"]["values"] == [1.0]
    assert stats.training_signal["groups"][0]["scored"] == 2


def test_python_helper_reads_groups_without_trajectories(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _rollout(job, "lo", reward=0.0)
    _rollout(job, "hi", reward=1.0)
    (job / "hi" / "trajectory" / "llm_trajectory.jsonl").unlink()
    signals = rollout_training_signals(job, group_advantage="grpo")
    assert signals.groups[0]["scored"] == 2
    assert signals.for_rollout(job / "hi")["advantage"] > 0


def test_cli_flags_manifest_and_refusals(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _rollout(job, "lo", reward=0.0)
    _rollout(job, "hi", reward=1.0)
    out, manifest = tmp_path / "out.jsonl", tmp_path / "m.json"
    result = runner.invoke(
        app,
        [
            "train",
            "convert",
            str(job),
            "--out",
            str(out),
            "--manifest",
            str(manifest),
            "--reward-vector",
            "--group-advantage",
            "loo",
            "--group-by",
            "task,model",
        ],
    )
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert sorted(r["advantage"] for r in rows) == [-1.0, 1.0]
    assert rows[0]["group"]["by"] == ["task", "model"]
    signal = json.loads(manifest.read_text())["training_signal"]
    assert signal["group_advantage"] == "loo"
    assert signal["reward_vector"] is True

    for extra, message in (
        (["--group-by", "task,seed"], "--group-by"),
        (["--format", "branch-tree"], "branch-tree"),
    ):
        result = runner.invoke(
            app,
            [
                "train",
                "convert",
                str(job),
                "--out",
                str(tmp_path / "x.jsonl"),
                "--group-advantage",
                "grpo",
                *extra,
            ],
        )
        assert result.exit_code == 1, result.output
        assert message in result.output

    result = runner.invoke(
        app,
        [
            "train",
            "convert",
            str(out),
            "--out",
            str(tmp_path / "y.jsonl"),
            "--reward-vector",
        ],
    )
    assert result.exit_code == 1
    assert "rollout or jobs directory" in result.output


def test_export_function_takes_the_options(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _rollout(job, "lo", reward=0.0)
    _rollout(job, "hi", reward=1.0)
    out = tmp_path / "o.jsonl"
    export_prime_sft_jsonl(job, out, group_advantage="grpo", group_by=("task",))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["group"]["id"] for r in rows} == {"task=demo-task"}


SCHEMAS = Path(__file__).resolve().parents[2] / "docs/reference/schemas"


def test_schema_file_is_current_and_rows_validate(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    from benchflow.trajectories import training_signal

    name = "benchflow-training-signal.v1.schema.json"
    schema = json.loads((SCHEMAS / name).read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema == json.loads(json.dumps(training_signal.SCHEMA))

    job = tmp_path / "job"
    _rollout(job, "lo", reward=0.0)
    _rollout(job, "hi", reward=1.0)
    _rollout(job, "solo", reward=1.0, task="other")
    _rubric_rollout(
        job,
        "rub",
        {"answer_correct": {"outcome": "pass", "explanation": "e"}},
        0.5,
    )
    rows, _ = convert_benchflow_rollouts_to_prime_sft_rows(
        job, reward_vector=True, group_advantage="grpo"
    )
    assert len(rows) == 4
    for row in rows:
        jsonschema.validate(row, schema, cls=jsonschema.Draft202012Validator)
