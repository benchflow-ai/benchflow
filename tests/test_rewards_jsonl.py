"""Tests for _write_rewards_jsonl — dense reward persistence."""

import json
from datetime import datetime
from pathlib import Path

from benchflow.sdk import _write_rewards_jsonl


def test_terminal_reward_written(tmp_path: Path) -> None:
    rewards = {"reward": 1.0}
    ts = datetime(2026, 4, 17, 15, 0, 0)
    _write_rewards_jsonl(tmp_path, rewards, ts)
    path = tmp_path / "rewards.jsonl"
    assert path.exists()
    lines = [json.loads(ln) for ln in path.read_text().strip().splitlines()]
    assert len(lines) == 1
    assert lines[0]["type"] == "terminal"
    assert lines[0]["source"] == "verifier"
    assert lines[0]["value"] == 1.0
    assert lines[0]["tag"] == "reward"


def test_no_file_when_rewards_none(tmp_path: Path) -> None:
    _write_rewards_jsonl(tmp_path, None, datetime.now())
    assert not (tmp_path / "rewards.jsonl").exists()


def test_no_file_when_rewards_empty(tmp_path: Path) -> None:
    _write_rewards_jsonl(tmp_path, {}, datetime.now())
    assert not (tmp_path / "rewards.jsonl").exists()


def test_extra_keys_in_meta(tmp_path: Path) -> None:
    rewards = {"reward": 0.75, "exact_match": 1.0, "partial": 0.5}
    _write_rewards_jsonl(tmp_path, rewards, datetime.now())
    lines = [
        json.loads(ln)
        for ln in (tmp_path / "rewards.jsonl").read_text().strip().splitlines()
    ]
    assert lines[0]["value"] == 0.75
    assert lines[0]["meta"]["exact_match"] == 1.0
    assert lines[0]["meta"]["partial"] == 0.5
    assert "reward" not in lines[0]["meta"]


def test_rubric_items_emitted_as_process(tmp_path: Path) -> None:
    rewards = {
        "reward": 0.75,
        "rubric": [
            {"name": "file_exists", "score": 1.0, "weight": 1.0},
            {"name": "content_correct", "score": 0.5, "weight": 1.0},
        ],
    }
    _write_rewards_jsonl(tmp_path, rewards, datetime.now())
    lines = [
        json.loads(ln)
        for ln in (tmp_path / "rewards.jsonl").read_text().strip().splitlines()
    ]
    assert len(lines) == 3
    assert lines[0]["type"] == "process"
    assert lines[0]["source"] == "verifier_rubric"
    assert lines[0]["value"] == 1.0
    assert lines[0]["tag"] == "file_exists"
    assert lines[0]["step_index"] == 0
    assert lines[0]["meta"]["weight"] == 1.0
    assert lines[1]["type"] == "process"
    assert lines[1]["tag"] == "content_correct"
    assert lines[1]["value"] == 0.5
    assert lines[1]["step_index"] == 1
    assert lines[2]["type"] == "terminal"
    assert lines[2]["value"] == 0.75


def test_rubric_without_terminal_still_works(tmp_path: Path) -> None:
    rewards = {
        "rubric": [{"name": "check_a", "score": 0.8}],
    }
    _write_rewards_jsonl(tmp_path, rewards, datetime.now())
    lines = [
        json.loads(ln)
        for ln in (tmp_path / "rewards.jsonl").read_text().strip().splitlines()
    ]
    assert len(lines) == 1
    assert lines[0]["type"] == "process"
    assert lines[0]["value"] == 0.8


def test_rubric_plus_terminal_no_rubric_in_meta(tmp_path: Path) -> None:
    rewards = {"reward": 0.6, "rubric": [{"name": "x", "score": 0.6}]}
    _write_rewards_jsonl(tmp_path, rewards, datetime.now())
    lines = [
        json.loads(ln)
        for ln in (tmp_path / "rewards.jsonl").read_text().strip().splitlines()
    ]
    terminal = next(ln for ln in lines if ln["type"] == "terminal")
    assert "rubric" not in terminal["meta"]


def test_empty_rubric_list_only_terminal(tmp_path: Path) -> None:
    rewards = {"reward": 1.0, "rubric": []}
    _write_rewards_jsonl(tmp_path, rewards, datetime.now())
    lines = [
        json.loads(ln)
        for ln in (tmp_path / "rewards.jsonl").read_text().strip().splitlines()
    ]
    assert len(lines) == 1
    assert lines[0]["type"] == "terminal"


# Space / granularity tags (issue #399)
#
# The architecture tags every reward record with ``(space, granularity, value)``.
# Native ``rewards.jsonl`` must surface those tags as first-class fields so
# dashboard / trainer / monitor consumers can distinguish Output / Action /
# Reasoning / Memory / Latent signals without unpacking ``meta``.


def test_terminal_carries_default_space_and_granularity(tmp_path: Path) -> None:
    _write_rewards_jsonl(tmp_path, {"reward": 1.0}, datetime.now())
    line = json.loads((tmp_path / "rewards.jsonl").read_text().strip())
    assert line["space"] == "output"
    assert line["granularity"] == "terminal"


def test_terminal_preserves_verifier_supplied_tags(tmp_path: Path) -> None:
    rewards = {"reward": 0.4, "space": "memory", "granularity": "terminal"}
    _write_rewards_jsonl(tmp_path, rewards, datetime.now())
    line = json.loads((tmp_path / "rewards.jsonl").read_text().strip())
    assert line["space"] == "memory"
    assert line["granularity"] == "terminal"
    # Promoted out of meta, not duplicated underneath it.
    assert "space" not in line["meta"]
    assert "granularity" not in line["meta"]


def test_rubric_items_default_to_step_granularity(tmp_path: Path) -> None:
    rewards = {
        "reward": 0.5,
        "rubric": [{"name": "check_a", "score": 0.5}],
    }
    _write_rewards_jsonl(tmp_path, rewards, datetime.now())
    lines = [
        json.loads(ln)
        for ln in (tmp_path / "rewards.jsonl").read_text().strip().splitlines()
    ]
    process = next(ln for ln in lines if ln["type"] == "process")
    assert process["space"] == "output"
    assert process["granularity"] == "step"


def test_rubric_items_preserve_per_item_tags(tmp_path: Path) -> None:
    rewards = {
        "reward": 0.25,
        "rubric": [
            {
                "name": "asked_when_needed",
                "score": 0.25,
                "space": "action",
                "granularity": "step",
            }
        ],
    }
    _write_rewards_jsonl(tmp_path, rewards, datetime.now())
    lines = [
        json.loads(ln)
        for ln in (tmp_path / "rewards.jsonl").read_text().strip().splitlines()
    ]
    process = next(ln for ln in lines if ln["type"] == "process")
    assert process["space"] == "action"
    assert process["granularity"] == "step"
    # First-class fields, not buried under meta.
    assert "space" not in process["meta"]
    assert "granularity" not in process["meta"]
