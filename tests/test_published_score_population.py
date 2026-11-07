"""Guards the fix at 363d815b."""

import json
from types import SimpleNamespace

import pytest

from benchflow.cli.eval_artifacts import postprocess_eval_artifacts


@pytest.mark.parametrize(
    "rewards,expected",
    [([1, None], 50), ([None, None], 0), ([0.75, None, 0], 25), ([-1, None], -50)],
)
def test_publish_counts_every_canonical_task(tmp_path, monkeypatch, rewards, expected):
    """Guards FF-sw-trajectories: unscored tasks remain in the denominator."""
    for index, reward in enumerate(rewards):
        rollout = tmp_path / f"task-{index}"
        rollout.mkdir()
        (rollout / "result.json").write_text(
            json.dumps(
                {
                    "task_name": str(index),
                    "rewards": {"reward": reward} if reward is not None else None,
                    "error": "dummy failure" if reward is None else None,
                }
            )
        )
    request = SimpleNamespace(
        task_manifest_out=None,
        run_config_out=None,
        health_summary_out=None,
        canonicalize="none",
        publish_hf=None,
        publish_bucket=None,
        eval_results_model="dummy/model",
        eval_results_dataset="dummy/tasks",
        eval_results_task="all",
    )
    captured = []
    monkeypatch.setattr(
        "benchflow.publish.huggingface.open_eval_results_pr",
        lambda **kw: captured.append(kw),
    )
    postprocess_eval_artifacts(
        SimpleNamespace(request=request),
        tmp_path,
        SimpleNamespace(source_provenance={}),
        tmp_path,
    )
    assert len(captured) == 1
    assert captured[0]["value"] == expected
    assert f"n = {len(rewards)}" in captured[0]["notes"]
    assert "unscored as 0" in captured[0]["notes"]
