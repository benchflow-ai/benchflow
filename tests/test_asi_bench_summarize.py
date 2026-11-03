"""Offline summary contract for ASI-Bench BenchFlow jobs."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "benchmarks" / "asi-bench" / "run_asi_bench.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("asi_bench_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _job(
    root: Path,
    name: str,
    *,
    task: str = "math.demo",
    level: str = "B1",
    reward: float | None = None,
    evaluator_error: bool = False,
    verifier_error: bool = False,
) -> Path:
    job = root / name
    verifier = job / "verifier"
    verifier.mkdir(parents=True)
    (job / "result.json").write_text(
        json.dumps({"verifier_error": "scorer crashed" if verifier_error else None})
    )
    if reward is not None:
        (verifier / "reward.json").write_text(
            json.dumps(
                {
                    "reward": reward,
                    "metadata": {
                        "benchmark": "ASI-Bench",
                        "seed": 31415,
                        "official": False,
                        "task_id": task,
                        "instance_id": f"{task}__seed31415",
                        "prompt_level": level,
                        "asi_revision": "a" * 40,
                        "hf_revision": "b" * 40,
                        "final_score": reward * 100,
                        "max_score": 100,
                    },
                }
            )
        )
    if evaluator_error:
        (verifier / "asi_error.json").write_text(
            json.dumps(
                {
                    "scorer_internal_error": True,
                    "failure_kind": "evaluator_runtime_error",
                }
            )
        )
    return job


def test_summarize_excludes_evaluator_errors_and_keeps_valid_zero(
    tmp_path: Path,
) -> None:
    _job(tmp_path, "a", reward=1.0)
    _job(tmp_path, "b", reward=0.0, level="B2")
    _job(tmp_path, "c", task="robotics.other", reward=0.5, level="B1")
    _job(tmp_path, "d", evaluator_error=True, verifier_error=True)
    _job(tmp_path, "e")

    report = _load_runner().summarize(tmp_path)

    assert report["scope"] == "seed31415 local, non-official"
    assert report["asi_revision"] == "a" * 40
    assert report["hf_revision"] == "b" * 40
    assert report["job_count"] == 5
    assert report["scored_instance_count"] == 3
    assert report["scorer_error_count"] == 1
    assert report["unscored_count"] == 1
    assert report["overall_mean_score"] == 50.0
    assert report["by_prompt_level"] == {"B1": 75.0, "B2": 0.0}
    assert report["by_task"] == [
        {
            "task_id": "math.demo",
            "scored_instances": 2,
            "mean_score": 50.0,
            "scores_by_level": {"B1": 100.0, "B2": 0.0},
        },
        {
            "task_id": "robotics.other",
            "scored_instances": 1,
            "mean_score": 50.0,
            "scores_by_level": {"B1": 50.0},
        },
    ]


def test_summarize_all_invalid_has_no_numeric_score(tmp_path: Path) -> None:
    _job(tmp_path, "failed", evaluator_error=True, verifier_error=True)
    report = _load_runner().summarize(tmp_path)
    assert report["overall_mean_score"] is None
    assert report["by_task"] == []
    assert report["by_prompt_level"] == {}
    assert report["scorer_error_count"] == 1


def test_summarize_all_unscored_does_not_display_zero(tmp_path: Path) -> None:
    _job(tmp_path, "unfinished")
    report = _load_runner().summarize(tmp_path)
    assert report["overall_mean_score"] is None
    assert report["by_task"] == []
    assert report["unscored_count"] == 1


def test_summarize_rejects_wrong_seed_and_conflicting_error(tmp_path: Path) -> None:
    job = _job(tmp_path, "wrong-seed", reward=0.5)
    reward_file = job / "verifier" / "reward.json"
    payload = json.loads(reward_file.read_text())
    payload["metadata"]["seed"] = 42
    reward_file.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="seed31415"):
        _load_runner().summarize(tmp_path)

    payload["metadata"]["seed"] = 31415
    reward_file.write_text(json.dumps(payload))
    (job / "verifier" / "asi_error.json").write_text(
        json.dumps({"scorer_internal_error": True})
    )
    with pytest.raises(ValueError, match="both reward and evaluator error"):
        _load_runner().summarize(tmp_path)


def test_summarize_rejects_reward_without_result_or_matching_score(
    tmp_path: Path,
) -> None:
    job = _job(tmp_path, "bad-reward", reward=0.5)
    (job / "result.json").unlink()
    with pytest.raises(ValueError, match=r"no persisted result\.json"):
        _load_runner().summarize(tmp_path)

    (job / "result.json").write_text("{}")
    reward_file = job / "verifier" / "reward.json"
    payload = json.loads(reward_file.read_text())
    payload["reward"] = 0.9
    reward_file.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Reward disagrees"):
        _load_runner().summarize(tmp_path)


def test_summarize_cli_prints_json_report(tmp_path: Path) -> None:
    _job(tmp_path, "scored", reward=0.25)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "summarize", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["overall_mean_score"] == 25.0


def test_summarize_rejects_mixed_source_revisions(tmp_path: Path) -> None:
    _job(tmp_path, "old", reward=1.0)
    new_job = _job(tmp_path, "new", reward=0.5)
    reward_file = new_job / "verifier" / "reward.json"
    payload = json.loads(reward_file.read_text())
    payload["metadata"]["hf_revision"] = "c" * 40
    reward_file.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different ASI/HF source revisions"):
        _load_runner().summarize(tmp_path)
