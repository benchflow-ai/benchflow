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
    lines = [json.loads(ln) for ln in (tmp_path / "rewards.jsonl").read_text().strip().splitlines()]
    assert lines[0]["value"] == 0.75
    assert lines[0]["meta"]["exact_match"] == 1.0
    assert lines[0]["meta"]["partial"] == 0.5
    assert "reward" not in lines[0]["meta"]
