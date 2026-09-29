"""``bench sandbox cleanup`` also removes stale Docker branch/checkpoint images.

Automatic checkpoints (``--checkpoints``) and ``--retain-snapshots`` keep
``bf-snap-*`` images on a Docker host after the run; nothing removed them,
while Daytona snapshots were already reaped by owner and age. The cleanup now
lists ``bf-snap-*`` images and removes those older than ``--max-age``
(``--dry-run`` only lists). An image a container still uses is left alone
(``docker rmi`` without force refuses it).
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta

from benchflow.cli import sandbox as sandbox_cli


def _listing(now: datetime) -> str:
    def row(image_id, name, age_min):
        created = (now - timedelta(minutes=age_min)).strftime("%Y-%m-%d %H:%M:%S %z")
        return f"{image_id}\t{name}\t{created} UTC"

    return "\n".join(
        [
            row("aaa", "bf-snap-task-old:latest", 3000),
            row("bbb", "bf-snap-task-new:latest", 5),
        ]
    )


def test_stale_snapshot_images_are_removed(monkeypatch):
    now = datetime.now(UTC)
    calls = []

    def fake_run(args, **_):
        calls.append(args)
        if args[:2] == ["docker", "images"]:
            return subprocess.CompletedProcess(args, 0, _listing(now), "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(sandbox_cli.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(sandbox_cli.subprocess, "run", fake_run)
    counts = sandbox_cli.cleanup_docker_snapshots(dry_run=False, max_age_minutes=60)
    assert counts == {"found": 2, "deleted": 1, "skipped": 1, "failed": 0}
    assert ["docker", "rmi", "bf-snap-task-old:latest"] in calls
    assert not any("bf-snap-task-new:latest" in c for c in calls if c[1] == "rmi")


def test_dry_run_only_lists(monkeypatch):
    now = datetime.now(UTC)
    calls = []

    def fake_run(args, **_):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, _listing(now), "")

    monkeypatch.setattr(sandbox_cli.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(sandbox_cli.subprocess, "run", fake_run)
    counts = sandbox_cli.cleanup_docker_snapshots(dry_run=True, max_age_minutes=60)
    assert counts["deleted"] == 1
    assert all(c[1] != "rmi" for c in calls)


def test_no_docker_is_a_no_op(monkeypatch):
    monkeypatch.setattr(sandbox_cli.shutil, "which", lambda _: None)
    assert (
        sandbox_cli.cleanup_docker_snapshots(dry_run=False, max_age_minutes=0) is None
    )


def test_an_image_in_use_is_counted_as_failed(monkeypatch):
    now = datetime.now(UTC)

    def fake_run(args, **_):
        if args[:2] == ["docker", "images"]:
            return subprocess.CompletedProcess(args, 0, _listing(now), "")
        return subprocess.CompletedProcess(args, 1, "", "image is being used")

    monkeypatch.setattr(sandbox_cli.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(sandbox_cli.subprocess, "run", fake_run)
    counts = sandbox_cli.cleanup_docker_snapshots(dry_run=False, max_age_minutes=60)
    assert (counts["deleted"], counts["failed"]) == (0, 1)
