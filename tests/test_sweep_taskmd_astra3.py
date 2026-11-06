"""Guards the PY8 fixes against the task.md cache at 363d815b."""

import json
import shutil

import pytest

from benchflow.rewards.validation import validate_reward_map
from benchflow.task import Task
from benchflow.task.formats import materialize_task_dir
from benchflow.taskmd import grading
from tests._taskmd_helpers import EXAMPLES


@pytest.fixture
def package(tmp_path, monkeypatch):
    source = tmp_path / "task"
    shutil.copytree(EXAMPLES / "hello-world", source)
    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "cache"))
    return source


@pytest.mark.parametrize("headline", ["partial", "strict"])
@pytest.mark.parametrize("points", [[2], [-2], [3, -2]])
def test_sum_rubric_materializes_the_emitted_reward_contract(package, headline, points):
    """Final validation accepts every attainable total and auxiliary score."""
    rubric = {
        "$schema": "https://task.md/schema/rubric-1.json",
        "version": "1.0.0",
        "scoring": {"method": "sum", "headline": headline},
        "criteria": [
            {
                "id": f"c{i}",
                "text": "correct",
                "judge": "test",
                "check": f"t{i}",
                "points": p,
            }
            for i, p in enumerate(points)
        ],
    }
    (package / "verifier" / "rubric.json").write_text(json.dumps(rubric))
    native = materialize_task_dir(package)
    task = Task(native)
    import itertools

    for values in itertools.product([True, False], repeat=len(points)):
        result = grading.score(
            rubric,
            rubric["criteria"],
            dict(zip([c["id"] for c in rubric["criteria"]], values, strict=True)),
        )
        rewards = {
            "reward": grading.headline(rubric, result),
            "partial": float(result.partial),
            "strict": float(result.strict),
        }
        assert (
            validate_reward_map(rewards, reward_range=task.config.verifier.reward_range)
            == rewards
        )
