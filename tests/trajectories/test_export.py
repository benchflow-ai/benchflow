"""Verifiers / ORS trajectory export.

Record shape pinned against the Verifiers RolloutOutput type
(willccbb/verifiers, verifiers/types.py): prompt, completion, reward,
metrics, is_completed, is_truncated, example_id, info.
"""

import json

from benchflow.rewards.events import RewardEvent
from benchflow.rewards.protocol import VerifyResult
from benchflow.trajectories.export import (
    export_trajectories_to_jsonl,
    trajectory_to_verifiers_record,
)


def _sample_trajectory():
    return [
        {"role": "user", "content": "Archive the email from Alice."},
        {"role": "assistant", "content": "Done — archived 1 email."},
    ]


def test_record_has_prompt_completion_reward():
    rec = trajectory_to_verifiers_record(
        task_id="clawsbench/archive-alice",
        messages=_sample_trajectory(),
        verify_result=VerifyResult(reward=1.0, items={"exact_match": 1.0}),
        model="claude-haiku-4-5",
        environment="clawsbench",
    )
    assert rec["reward"] == 1.0
    assert rec["prompt"] == [
        {"role": "user", "content": "Archive the email from Alice."}
    ]
    assert rec["completion"] == [
        {"role": "assistant", "content": "Done — archived 1 email."}
    ]
    assert rec["info"]["task_id"] == "clawsbench/archive-alice"
    assert rec["info"]["environment"] == "clawsbench"
    assert rec["info"]["model"] == "claude-haiku-4-5"


def test_record_carries_verifiers_required_fields():
    rec = trajectory_to_verifiers_record(
        task_id="t",
        messages=_sample_trajectory(),
        verify_result=VerifyResult(reward=0.5, items={"a": 0.5}),
        model="m",
        environment="clawsbench",
        example_id=7,
    )
    assert rec["example_id"] == 7
    assert rec["is_completed"] is True
    assert rec["is_truncated"] is False
    assert rec["metrics"] == {"a": 0.5}


def test_record_carries_ors_reward_metadata():
    rec = trajectory_to_verifiers_record(
        task_id="t",
        messages=_sample_trajectory(),
        verify_result=VerifyResult(reward=0.5, items={"a": 0.5}),
        model="m",
        environment="clawsbench",
    )
    assert rec["info"]["reward_metadata"]["items"] == {"a": 0.5}
    assert rec["info"]["reward_valid"] is True


def test_invalid_reward_clamped_to_zero():
    rec = trajectory_to_verifiers_record(
        task_id="t",
        messages=_sample_trajectory(),
        verify_result=VerifyResult(reward=float("nan"), items={}),
        model="m",
        environment="clawsbench",
    )
    assert rec["reward"] == 0.0
    assert rec["info"]["reward_valid"] is False


def test_empty_messages_yield_empty_prompt_and_completion():
    rec = trajectory_to_verifiers_record(
        task_id="t",
        messages=[],
        verify_result=VerifyResult(reward=0.0, items={}),
        model="m",
        environment="clawsbench",
    )
    assert rec["prompt"] == []
    assert rec["completion"] == []


def test_export_writes_one_json_object_per_line(tmp_path):
    records = [
        trajectory_to_verifiers_record(
            task_id=f"t{i}",
            messages=_sample_trajectory(),
            verify_result=VerifyResult(reward=float(i) / 2, items={}),
            model="m",
            environment="clawsbench",
            example_id=i,
        )
        for i in range(3)
    ]
    out = tmp_path / "dataset.jsonl"
    export_trajectories_to_jsonl(records, out)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        parsed = json.loads(line)
        assert parsed["example_id"] == i


def test_export_creates_parent_directory(tmp_path):
    out = tmp_path / "nested" / "deep" / "dataset.jsonl"
    export_trajectories_to_jsonl([], out)
    assert out.exists()


def test_record_preserves_reward_space_and_granularity_tags():
    """Trainer JSONL keeps ``(space, granularity)`` per event and headline.

    Regression for issue #391: ORS reward-event export dropped both tags,
    so memory/action/reasoning process events lost their evaluation-space
    metadata at the trainer seam.
    """
    events = [
        RewardEvent(
            type="dense",
            reward=0.4,
            source="memory-scorer",
            step=2,
            space="memory",
            granularity="step",
        ),
    ]
    rec = trajectory_to_verifiers_record(
        task_id="t",
        messages=_sample_trajectory(),
        verify_result=VerifyResult(
            reward=0.4,
            items={"memory-scorer": 0.4},
            events=events,
            space="memory",
            granularity="step",
        ),
        model="m",
        environment="clawsbench",
    )
    meta = rec["info"]["reward_metadata"]
    assert meta["space"] == "memory"
    assert meta["granularity"] == "step"
    assert meta["events"][0]["space"] == "memory"
    assert meta["events"][0]["granularity"] == "step"
