"""End-to-end vertical slice: ClawsBench manifest → scored trajectory →
Verifiers/ORS export.

The data-path test proves the thread with no Docker. A full live rollout is
run manually:

    bench eval create --tasks-dir benchmarks/clawsbench/tasks/<task> \\
      --environment-manifest benchmarks/clawsbench/environment.toml \\
      --agent claude-agent-acp --model claude-haiku-4-5
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow._utils.task_authoring import check_task
from benchflow.environment.manifest import load_manifest
from benchflow.evaluation import Evaluation, EvaluationConfig
from benchflow.rewards.protocol import VerifyResult
from benchflow.task import (
    TaskDocument,
    VerifierDocument,
    export_task_to_split_layout,
)
from benchflow.trajectories.export import (
    export_trajectories_to_jsonl,
    trajectory_to_verifiers_record,
)

MANIFEST = Path("benchmarks/clawsbench/environment.toml")
TASK = Path("benchmarks/clawsbench/tasks/archive-amazon-shipping")


def test_manifest_drives_a_scored_exportable_record(tmp_path):
    """The ClawsBench manifest loads, and a scored trajectory from it
    exports to a valid Verifiers dataset line — the slice's data path."""
    manifest = load_manifest(MANIFEST)
    assert manifest.name == "clawsbench"

    # A scored trajectory, as Rollout.verify() would produce one.
    messages = [
        {"role": "user", "content": "Archive Alice's email."},
        {"role": "assistant", "content": "Archived 1 email."},
    ]
    result = VerifyResult(reward=1.0, items={"exact_match": 1.0})

    record = trajectory_to_verifiers_record(
        task_id="clawsbench/archive-alice",
        messages=messages,
        verify_result=result,
        model="claude-haiku-4-5",
        environment=manifest.name,
    )
    out = tmp_path / "clawsbench_dataset.jsonl"
    export_trajectories_to_jsonl([record], out)

    parsed = json.loads(out.read_text().strip())
    assert parsed["reward"] == 1.0
    assert parsed["info"]["environment"] == "clawsbench"
    assert parsed["prompt"] and parsed["completion"]
    assert parsed["metrics"] == {"exact_match": 1.0}


def test_archive_amazon_shipping_is_publication_grade_native_task():
    """Guards PR #1's service-backed ClawsBench task.md dogfood slice."""
    assert (TASK / "task.md").is_file()
    assert not (TASK / "task.toml").exists()
    assert not (TASK / "instruction.md").exists()
    assert not (TASK / "tests").exists()
    assert not (TASK / "solution").exists()
    assert (TASK / "environment" / "Dockerfile").is_file()
    assert (TASK / "verifier" / "test.sh").is_file()
    assert (TASK / "verifier" / "evaluate.py").is_file()
    assert (TASK / "verifier" / "verifier.md").is_file()
    assert (TASK / "verifier" / "rubrics" / "gmail-state.md").is_file()
    assert (TASK / "oracle" / "solve.sh").is_file()

    document = TaskDocument.from_path(TASK / "task.md")
    assert document.config.task is not None
    assert document.config.task.name == "clawsbench/archive-amazon-shipping"
    assert document.benchflow["environment"]["manifest"] == "../../environment.toml"
    assert document.benchflow["environment"]["services"] == ["gmail"]
    assert "http://localhost:9001" in document.instruction

    verifier = VerifierDocument.from_verifier_dir(TASK / "verifier")
    assert verifier.selected_strategy.command == "./test.sh"
    assert verifier.outputs.reward_json == "/logs/verifier/reward.json"
    assert verifier.outputs.details_json == "/logs/verifier/reward-details.json"

    assert check_task(TASK, validation_level="publication-grade") == []


def test_eval_create_enumerates_archive_amazon_shipping_native_task(tmp_path):
    """Eval discovery must treat the native service-backed task as runnable."""
    ev = Evaluation(
        tasks_dir=str(TASK),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(agent="oracle"),
    )

    assert ev._get_task_dirs() == [TASK]


@pytest.mark.asyncio
async def test_native_task_manifest_is_loaded_for_rollout(tmp_path):
    """Guards PR #1's ClawsBench task.md-declared environment manifest contract."""
    ev = Evaluation(
        tasks_dir=str(TASK),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(agent="oracle"),
    )
    captured = {}

    async def fake_create(config):
        captured["config"] = config
        rollout = MagicMock()
        rollout.run = AsyncMock(return_value=MagicMock())
        return rollout

    with patch("benchflow.rollout.Rollout.create", side_effect=fake_create):
        await ev._run_single_task(TASK, ev._config)

    assert captured["config"].environment_manifest is not None
    assert captured["config"].environment_manifest.name == "clawsbench"


def test_archive_amazon_shipping_exports_to_harbor_split_layout(tmp_path):
    """Native ClawsBench packages still export for split-layout consumers."""
    out_dir = tmp_path / "exported"

    report = export_task_to_split_layout(TASK, out_dir, target="harbor")

    assert report.status == "degraded"
    assert (out_dir / "task.toml").is_file()
    assert (out_dir / "instruction.md").is_file()
    assert (out_dir / "environment" / "Dockerfile").is_file()
    assert (out_dir / "tests" / "test.sh").is_file()
    assert (out_dir / "tests" / "evaluate.py").is_file()
    assert (out_dir / "tests" / "verifier.md").is_file()
    assert (out_dir / "tests" / "rubrics" / "gmail-state.md").is_file()
    assert (out_dir / "solution" / "solve.sh").is_file()


def test_archive_amazon_shipping_verifier_writes_structured_reward(tmp_path):
    """Verifier evidence includes reward JSON and reason-bearing details."""
    state = {
        "users": {
            "me": {
                "messages": [
                    {
                        "id": "msg-1",
                        "sender": "shipment-tracking@amazon.com",
                        "labelIds": ["STARRED"],
                    }
                ]
            }
        }
    }
    state_path = tmp_path / "state.json"
    reward_txt = tmp_path / "reward.txt"
    reward_json = tmp_path / "reward.json"
    details_json = tmp_path / "reward-details.json"
    state_path.write_text(json.dumps(state))

    result = subprocess.run(
        [
            sys.executable,
            str(TASK / "verifier" / "evaluate.py"),
            "--state",
            str(state_path),
            "--output",
            str(reward_txt),
            "--reward-json",
            str(reward_json),
            "--details-json",
            str(details_json),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert reward_txt.read_text().strip() == "1.0"
    assert json.loads(reward_json.read_text()) == {"reward": 1.0}
    assert json.loads(details_json.read_text())["reason"] == "target-archived"
