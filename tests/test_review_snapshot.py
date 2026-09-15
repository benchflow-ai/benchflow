"""Guards immutable reviewer inputs after baseline PR #1126."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import benchflow
from benchflow.review import automatic
from benchflow.review.config import load_rubric_snapshot
from tests.test_review_automatic import auth as auth
from tests.test_review_automatic import prepared as prepared
from tests.test_review_automatic import saved as saved
from tests.test_review_runtime import FakeRun, good_weighted_review


def test_rubric_snapshot_parse_and_hash_use_one_read(prepared, monkeypatch):
    """Guards provenance against a second file read after PR #1126."""
    path = prepared.rubric_path
    original = path.read_bytes()
    reads = []
    read_bytes = Path.read_bytes

    def mutate_after_read(candidate):
        contents = read_bytes(candidate)
        if candidate == path:
            reads.append(candidate)
            replacement = json.loads(contents)
            replacement["criteria"][0]["guidance"] = "Different rubric revision"
            path.write_text(json.dumps(replacement))
        return contents

    monkeypatch.setattr(Path, "read_bytes", mutate_after_read)
    rubric, contents = load_rubric_snapshot(path)
    assert reads == [path]
    assert contents == original
    assert rubric == prepared.rubric
    assert hashlib.sha256(contents).hexdigest() == prepared.rubric_sha256


@pytest.mark.asyncio
async def test_task_edit_during_reviewer_queue_cannot_replace_evidence(
    prepared, saved, monkeypatch
):
    """Guards queued reviewer provenance after PR #1126."""
    original = (prepared.task_path / "task.md").read_bytes()
    observed = []
    fake = FakeRun(review_payload=good_weighted_review(saved.name))

    class Queue:
        async def __aenter__(self):
            (prepared.task_path / "task.md").write_text("Changed while queued")

        async def __aexit__(self, *args):
            return False

    async def capture(config):
        task_upload = next(
            Path(source)
            for source, destination in config.uploads.items()
            if destination == "/evidence/task"
        )
        observed.append((task_upload / "task.md").read_bytes())
        return await fake(config)

    monkeypatch.setattr(automatic, "_review_limit", lambda config: Queue())
    monkeypatch.setattr(benchflow, "run", capture)
    scoring = await automatic.finish_review(prepared, saved)
    assert scoring.status == "complete"
    assert observed == [original]
    details = json.loads(next((saved / "scoring").glob("*.json")).read_text())
    assert (saved / details["task_input"] / "task.md").read_bytes() == original
    assert (
        hashlib.sha256((saved / details["rubric_snapshot"]).read_bytes()).hexdigest()
        == prepared.rubric_sha256
    )
    assert details["task_digest"] == prepared.task_digest


@pytest.mark.asyncio
async def test_mutation_during_task_copy_rejects_review(prepared, saved, monkeypatch):
    """Guards copied task identity after PR #1126."""
    copytree = shutil.copytree

    def corrupt_copy(source, target, *args, **kwargs):
        copied = copytree(source, target, *args, **kwargs)
        if Path(source) == prepared.task_path:
            (Path(target) / "task.md").write_text("New task revision")
        return copied

    fake = FakeRun(review_payload=good_weighted_review(saved.name))
    monkeypatch.setattr(shutil, "copytree", corrupt_copy)
    monkeypatch.setattr(benchflow, "run", fake)
    scoring = await automatic.finish_review(prepared, saved)
    assert scoring.status == "error"
    assert "Task contents changed" in scoring.error
    assert not fake.configs
    assert not list((saved / "reviews").glob("*/task-input"))
