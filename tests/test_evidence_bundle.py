"""`bench bundle`: one shareable, redacted zip of a run's evidence.

From a benchmark's review: partners asked for each run's trajectories,
review report, gate records and workspace by hand, file by file.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.evidence_bundle import MANIFEST, BundleError, build_bundle

SECRET = "sk-ant-api03-" + "Zk3plT9xQ2mVb7Rr8sLw4nHq" * 2


def _job(tmp_path: Path) -> Path:
    job = tmp_path / "jobs" / "run"
    trial = job / "task-a__abc123"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "verifier").mkdir()
    (trial / "reviews" / "r1").mkdir(parents=True)
    (trial / "evidence" / "workspace").mkdir(parents=True)
    (trial / "agent").mkdir()
    (job / "run-card.json").write_text('{"agent": {"name": "oracle"}}\n')
    (job / "summary.json").write_text('{"total": 1}\n')
    (trial / "result.json").write_text(
        json.dumps({"task_name": "task-a", "rewards": {"reward": 1.0}})
    )
    (trial / "trajectory" / "acp_trajectory.jsonl").write_text(
        json.dumps({"type": "tool", "text": "ok"})
        + "\n"
        + json.dumps({"type": "tool", "text": f"export ANTHROPIC_API_KEY={SECRET}"})
        + "\n"
    )
    (trial / "verifier" / "test-stdout.txt").write_text(f"token was {SECRET}\n")
    (trial / "verifier" / "reward.txt").write_text("1\n")
    (trial / "reviews" / "r1" / "review_report.json").write_text('{"score": 2}\n')
    (trial / "evidence" / "manifest.json").write_text('{"workspace": "/app"}\n')
    (trial / "evidence" / "workspace" / "answer.txt").write_text("42\n")
    (trial / "agent" / "claude-sessions.jsonl").write_text("not bundled\n")
    (trial / "verifier" / "elsewhere").symlink_to(tmp_path)
    return job


def _members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_a_job_bundle_holds_the_evidence_with_secrets_masked(tmp_path):
    job = _job(tmp_path)
    result = build_bundle(job, tmp_path / "run.zip")

    members = _members(result.path)
    trial = "task-a__abc123"
    assert set(members) == {
        MANIFEST,
        "run-card.json",
        "summary.json",
        f"{trial}/result.json",
        f"{trial}/trajectory/acp_trajectory.jsonl",
        f"{trial}/verifier/test-stdout.txt",
        f"{trial}/verifier/reward.txt",
        f"{trial}/reviews/r1/review_report.json",
        f"{trial}/evidence/manifest.json",
        f"{trial}/evidence/workspace/answer.txt",
    }
    assert not any(SECRET.encode() in data for data in members.values())
    manifest = json.loads(members[MANIFEST])
    assert manifest["trials"] == [trial]
    assert sum(manifest["masked"].values()) == sum(result.masked.values()) >= 2
    for item in manifest["files"]:
        assert hashlib.sha256(members[item["path"]]).hexdigest() == item["sha256"]
    assert manifest["left_out"] == [
        {"path": f"{trial}/verifier/elsewhere", "reason": "a link"}
    ]


def test_the_bundle_is_the_same_bytes_every_time(tmp_path):
    job = _job(tmp_path)
    first = build_bundle(job, tmp_path / "a.zip")
    second = build_bundle(job, tmp_path / "b.zip")
    assert first.sha256 == second.sha256


def test_no_workspace_leaves_evidence_out_and_a_trial_folder_bundles_alone(tmp_path):
    job = _job(tmp_path)
    result = build_bundle(job / "task-a__abc123", tmp_path / "t.zip", workspace=False)
    members = _members(result.path)
    assert "task-a__abc123/result.json" in members
    assert not any("/evidence/" in name for name in members)
    assert "run-card.json" not in members


def test_a_folder_without_trials_is_refused(tmp_path):
    with pytest.raises(BundleError, match="holds no trial"):
        build_bundle(tmp_path, tmp_path / "x.zip")


def test_bench_bundle_writes_beside_the_folder(tmp_path):
    from benchflow.cli.main import app

    job = _job(tmp_path)
    result = CliRunner().invoke(app, ["bundle", str(job)])
    assert result.exit_code == 0, result.output
    assert (job.parent / "run.bundle.zip").is_file()
    assert "value(s) masked" in result.output
