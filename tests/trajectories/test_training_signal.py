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
    """Changed with whole groups: the four-rollout group is withheld whole.

    A group with a dropped member is withheld from the trainer, never
    partially passed. Before, its two scored rollouts were written with
    advantages over the two of them. The formula is still checked on the
    two scored rollouts alone, in a job of their own.
    """
    job = tmp_path / "job"
    _rollout(job, "demo__00000001", reward=0.0)
    _rollout(job, "demo__00000002", reward=1.0)
    _rollout(job, "demo__00000003", reward=None, error="transport closed")
    _rollout(job, "demo__00000004", reward=None, error="transport closed")

    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(
        job, reward_vector=True, group_advantage="grpo"
    )
    assert rows == []
    signal = stats.training_signal
    (entry,) = signal["groups"]
    assert entry["withheld"].startswith("unscored:")
    assert "demo__00000003" in entry["withheld"]
    assert (signal["withheld_groups"], signal["rows_withheld"]) == (1, 2)
    members = {m["rollout"]: m for m in entry["members"]}
    assert members["demo__00000003"]["excluded"] == "unscored"
    assert members["demo__00000001"]["excluded"] == "group_withheld"
    assert members["demo__00000001"]["advantage"] is None
    assert entry["mean"] == pytest.approx(0.5)

    job = tmp_path / "scored"
    _rollout(job, "demo__00000001", reward=0.0)
    _rollout(job, "demo__00000002", reward=1.0)
    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(
        job, reward_vector=True, group_advantage="grpo"
    )
    by = _by_rollout(rows)
    assert sorted(by) == ["demo__00000001", "demo__00000002"]
    std = math.sqrt(0.5)  # sample std of (0, 1)
    assert by["demo__00000002"]["advantage"] == pytest.approx(0.5 / (std + 1e-4))
    assert by["demo__00000001"]["advantage"] == pytest.approx(-0.5 / (std + 1e-4))
    group = by["demo__00000002"]["group"]
    assert group["id"] == (
        "task=demo-task|agent=opencode|model=vendor/model-a|job=scored"
    )
    assert group["by"] == ["task", "agent", "model", "job"]
    assert group["normalisation"] == "grpo"
    assert "withheld" not in group
    assert group["rollouts"] == 2
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
    assert signal["group_by"] == ["task", "agent", "model", "job"]
    assert (signal["withheld_groups"], signal["rows_withheld"]) == (0, 0)
    (entry,) = signal["groups"]
    members = {m["rollout"]: m for m in entry["members"]}
    assert "excluded" not in members["demo__00000002"]
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
    """Restored after review: advantages are computed on the whole
    group, then ``min_reward`` filters rows. A scored member under the bar
    keeps its reward in the baseline and does not withhold its group."""
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
        assert result.exit_code == 2, result.output
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


def test_groups_stay_inside_a_job_and_skip_retried_attempts(tmp_path: Path) -> None:
    """A baseline never mixes jobs by default, nor an attempt a retry replaced.

    Guards the dx/sdk fix of the training signal from bf6e8412 (SDK update),
    whose default group (task, agent, model) pooled rollouts across jobs,
    which may have run different policy checkpoints under one model name,
    and across an Evaluation's retried attempts.
    """
    root = tmp_path / "jobs"
    _rollout(root / "step-1", "a1", reward=0.0)
    _rollout(root / "step-1", "a2", reward=1.0)
    _rollout(root / "step-2", "b1", reward=1.0)
    _rollout(root / "step-2", "b2", reward=1.0)
    # An Evaluation job: its first attempt of demo-task hung and was retried.
    retried = _rollout(root / "eval", "c1", reward=0.0, error="idle timeout")
    _rollout(root / "eval", "c2", reward=1.0)
    (root / "eval" / "evaluation.json").write_text("{}")
    import os
    import time

    now = time.time()
    os.utime(retried / "result.json", (now - 60, now - 60))

    signals = rollout_training_signals(root, group_advantage="grpo")
    groups = {g["key"]["job"]: g for g in signals.groups}
    assert set(groups) == {"step-1", "step-2", "eval"}
    assert groups["step-2"]["mean"] == 1.0  # not pooled with step-1's 0.0
    ev = groups["eval"]
    assert (ev["rollouts"], ev["scored"], ev["mean"]) == (1, 1, 1.0)
    members = {m["rollout"]: m for m in ev["members"]}
    assert members["c1"]["excluded"] == "retried"
    assert members["c1"]["advantage"] is None

    pooled = rollout_training_signals(
        root, group_advantage="grpo", group_by=("task", "agent", "model")
    )
    (only,) = pooled.groups
    assert (only["rollouts"], only["scored"]) == (5, 5)


def _whole_and_cut_job(job: Path) -> None:
    """A group whose third member made no model calls (scored 0, no row),
    and a whole group of another task.

    Changed after review: the cut member used to end with an error,
    which now keeps its group whole when it is scored (see
    test_a_scored_member_that_ended_with_an_error_keeps_its_group_whole).
    """
    _rollout(job, "a", reward=1.0)
    _rollout(job, "b", reward=0.0)
    _rollout(job, "c", reward=0.0)
    (job / "c" / "trajectory" / "llm_trajectory.jsonl").write_text("")
    _rollout(job, "x", reward=1.0, task="other")
    _rollout(job, "y", reward=0.0, task="other")


@pytest.mark.parametrize(
    "convert",
    [
        convert_benchflow_rollouts_to_prime_sft_rows,
        convert_benchflow_rollouts_to_trl_sft_rows,
    ],
    ids=["prime-sft", "trl-sft"],
)
def test_a_group_with_a_member_that_wrote_no_row_is_withheld_whole(
    tmp_path: Path, convert
) -> None:
    """A group with a dropped member is withheld from the trainer, never
    partially passed.

    Before, a rollout that wrote no row (here: no model calls) had
    its group's other members written with advantages whose baseline
    counted it, so a trainer got part of the group.
    """
    job = tmp_path / "job"
    _whole_and_cut_job(job)

    rows, stats = convert(job, group_advantage="grpo")

    assert sorted(_by_rollout(rows)) == ["x", "y"]
    signal = stats.training_signal
    groups = {g["key"]["task"]: g for g in signal["groups"]}
    assert groups["demo-task"]["withheld"].startswith("no row:")
    assert "c (no model calls recorded)" in groups["demo-task"]["withheld"]
    assert "withheld" not in groups["other"]
    assert signal["withheld_groups"] == 1
    assert signal["rows_withheld"] > 0
    assert stats.rows_written == len(rows)
    assert stats.sources == [row["source_path"] for row in rows]


def test_without_group_advantage_nothing_is_withheld(tmp_path: Path) -> None:
    """Plain SFT rows have no groups to keep whole (whole groups change only grouped rows)."""
    job = tmp_path / "job"
    _whole_and_cut_job(job)

    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(job)

    assert sorted(_by_rollout(rows)) == ["a", "b", "x", "y"]
    assert stats.training_signal is None


def test_skyrl_rows_keep_groups_whole(tmp_path: Path) -> None:
    """verl-sft and skyrl-sft build on the TRL rows, so they withhold the same groups."""
    from benchflow.trajectories.export_skyrl_sft import export_skyrl_sft_jsonl

    job = tmp_path / "job"
    _whole_and_cut_job(job)
    out = tmp_path / "skyrl" / "train.jsonl"

    stats = export_skyrl_sft_jsonl(job, out, group_advantage="grpo")

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {Path(r["source_rollout_dir"]).name for r in rows} == {"x", "y"}
    assert stats.training_signal["withheld_groups"] == 1


def test_cli_reports_withheld_groups(tmp_path: Path) -> None:
    """The summary line names the withheld groups; the manifest has them."""
    job = tmp_path / "job"
    _whole_and_cut_job(job)
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
            "--group-advantage",
            "grpo",
        ],
    )

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["task_name"] for r in rows} == {"other"}
    assert "1 group withheld" in result.output
    signal = json.loads(manifest.read_text())["training_signal"]
    assert signal["withheld_groups"] == 1


_CONVERTERS = pytest.mark.parametrize(
    "convert",
    [
        convert_benchflow_rollouts_to_prime_sft_rows,
        convert_benchflow_rollouts_to_trl_sft_rows,
    ],
    ids=["prime-sft", "trl-sft"],
)


def _trial_config(root: Path, task: str = "demo-task") -> None:
    """The ``config.json`` a trial writes when it starts, before any result."""
    (root / "config.json").write_text(
        json.dumps(
            {
                "task_path": f"tasks/{task}",
                "agent": "opencode",
                "model": "vendor/model-a",
            }
        )
    )


def _four_and_a_whole_group(job: Path) -> None:
    for i, reward in enumerate([1.0, 0.0, 1.0, 0.0]):
        _trial_config(_rollout(job, f"demo__a{i}", reward=reward))
    _rollout(job, "x", reward=1.0, task="other")
    _rollout(job, "y", reward=0.0, task="other")


@_CONVERTERS
@pytest.mark.parametrize("damage", ["truncated", "missing"])
def test_a_member_without_a_readable_result_withholds_its_group(
    tmp_path: Path, convert, damage: str
) -> None:
    """The export sees a member whose result.json is cut short or missing.

    Guards a review finding: a trial killed while writing its
    result (or before) was invisible to the export, which wrote the other
    three rows with advantages over three rewards instead of four (+0.577,
    -1.155, +0.577 where the full group gives +-0.866), and withheld nothing.
    """
    job = tmp_path / "job"
    _four_and_a_whole_group(job)
    result = job / "demo__a3" / "result.json"
    if damage == "truncated":
        result.write_text('{"task_name": "demo-task", "rewa')
    else:
        result.unlink()

    rows, stats = convert(job, group_advantage="grpo")

    assert sorted(_by_rollout(rows)) == ["x", "y"]
    signal = stats.training_signal
    groups = {g["key"]["task"]: g for g in signal["groups"]}
    held = groups["demo-task"]
    assert held["withheld"].startswith("no result:")
    assert "demo__a3" in held["withheld"]
    assert held["rollouts"] == 4
    assert "withheld" not in groups["other"]
    assert signal["withheld_groups"] == 1


def test_a_missing_result_in_an_evaluation_folder_is_a_replaced_attempt(
    tmp_path: Path,
) -> None:
    """An Evaluation retried a trial whose first attempt left no result: the
    attempt is not a sample, and the scored retry keeps the group whole."""
    job = tmp_path / "job"
    job.mkdir()
    (job / "evaluation.json").write_text("{}")
    _trial_config(_rollout(job, "demo-task__1", reward=1.0))
    _trial_config(_rollout(job, "demo-task__2", reward=0.0, agent="other-agent"))
    first = _rollout(job, "demo-task__0", reward=None)
    _trial_config(first)
    (first / "result.json").unlink()

    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(
        job, group_advantage="grpo", group_by=("task", "model")
    )

    assert sorted(_by_rollout(rows)) == ["demo-task__1", "demo-task__2"]
    assert stats.training_signal["withheld_groups"] == 0


def _task_package_rollout(job: Path, name: str, reward: float) -> Path:
    """A trial in the task.toml-package runner's format: its reward under
    ``verifier_result.rewards``, its agent and model under ``agent_info``."""
    d = job / name
    (d / "trajectory").mkdir(parents=True)
    (d / "result.json").write_text(
        json.dumps(
            {
                "task_name": "org/demo-task",
                "trial_name": name,
                "agent_info": {
                    "name": "opencode",
                    "model_info": {"name": "model-a", "provider": "vendor"},
                },
                "verifier_result": {"rewards": {"reward": reward}},
                "exception_info": None,
            }
        )
    )
    (d / "trajectory" / "llm_trajectory.jsonl").write_text(
        "\n".join(
            json.dumps(e) for e in (_exchange(final=False), _exchange(final=True))
        )
        + "\n"
    )
    return d


@_CONVERTERS
def test_task_package_results_are_scored(tmp_path: Path, convert) -> None:
    """The signal pass reads results as the row pass does (load_rollout_result).

    Guards a review finding: it read the raw file, so both rollouts
    were unscored and their group withheld, and the export wrote no rows.
    """
    job = tmp_path / "job"
    _task_package_rollout(job, "demo-task__c0", 1.0)
    _task_package_rollout(job, "demo-task__c1", 0.0)

    rows, stats = convert(job, group_advantage="grpo")

    by = _by_rollout(rows)
    assert sorted(by) == ["demo-task__c0", "demo-task__c1"]
    assert by["demo-task__c0"]["advantage"] == pytest.approx(0.5 / (0.5**0.5 + 1e-4))
    (group,) = stats.training_signal["groups"]
    assert group["scored"] == 2 and "withheld" not in group

    signals = rollout_training_signals(job, group_advantage="grpo")
    assert signals.groups[0]["scored"] == 2


@_CONVERTERS
def test_a_scored_member_that_ended_with_an_error_keeps_its_group_whole(
    tmp_path: Path, convert
) -> None:
    """A rollout that ended with an error but was scored is a whole member.

    Guards a review finding: the export withheld this group while
    the stream passed it with every member's advantage. Time-outs cluster
    on hard tasks, so the export dropped hard tasks' groups. Its rows now
    carry the error, so a trainer can mask them, and the advantages are the
    stream's.
    """
    from benchflow.trajectories.rollout_stream import stream_rollouts

    job = tmp_path / "job"
    _rollout(job, "ok1", reward=1.0)
    _rollout(job, "ok2", reward=0.0)
    _rollout(
        job, "err", reward=0.0, error="Agent prompt exceeded wall-clock budget 600s"
    )

    rows, stats = convert(job, group_advantage="grpo")

    by = _by_rollout(rows)
    assert sorted(by) == ["err", "ok1", "ok2"]
    (group,) = stats.training_signal["groups"]
    assert "withheld" not in group
    assert by["err"]["group"]["error"] == "pipe_closed"
    assert "error" not in by["ok1"]["group"]
    streamed = {
        r.rollout: r.advantage for r in stream_rollouts(job, follow=False, group_size=3)
    }
    assert {k: by[k]["advantage"] for k in by} == pytest.approx(streamed)
    assert by["ok1"]["advantage"] == pytest.approx(1.1547, abs=1e-3)


def test_without_group_advantage_an_errored_rollout_still_writes_no_row(
    tmp_path: Path,
) -> None:
    """Plain SFT rows still leave out a rollout that ended with an error."""
    job = tmp_path / "job"
    _rollout(job, "ok", reward=1.0)
    _rollout(job, "err", reward=0.0, error="agent timed out")

    rows, _ = convert_benchflow_rollouts_to_prime_sft_rows(job)

    assert sorted(_by_rollout(rows)) == ["ok"]
