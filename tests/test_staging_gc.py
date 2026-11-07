"""Garbage collection never re-enters the staging lock.

A sandbox's weakref finalizer used to call ``StagedEnvironment.remove``,
which takes the non-reentrant ``_live_lock``. When the collector ran it on
the thread that already held that lock (inside ``stage_environment``'s
``_live.add``), the thread waited for itself and the run hung for good
(batch 23, ``staging-base-probe.py``). The finalizer now removes the copy
only when the lock is free and otherwise queues the root; the next holder
of the lock, a job's end or exit removes it.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow.task import staging

ROOT = Path(__file__).resolve().parents[1]

# The batch 23 probe: a real sandbox whose real cleanup finalizer is only
# reachable through a cycle, collected exactly where stage_environment
# holds the lock. faulthandler ends a hang with its stack in the output.
_PROBE = """
import faulthandler, gc, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from benchflow.task import staging
from tests.test_cloud_metadata_block import OPEN, _sandbox_for

work = Path(sys.argv[2])
gc.disable()
before = set(staging._live)
sandbox = _sandbox_for(work / "probe", OPEN, block=False)
(collected,) = set(staging._live) - before
sandbox._gc_probe_cycle = sandbox
del sandbox


class CollectingSet(set):
    def add(self, value):
        gc.collect()
        return super().add(value)


staging._live = CollectingSet(staging._live)
environment = work / "second" / "task" / "environment"
environment.mkdir(parents=True)
faulthandler.dump_traceback_later(20, exit=True)
second = staging.stage_environment(environment)
faulthandler.cancel_dump_traceback_later()
queued = collected in staging._released and collected.is_dir()
staging._live = set(staging._live)
second.remove()
print(json.dumps({
    "queued": queued,
    "removed": not collected.exists() and collected not in staging._live,
}), flush=True)
"""


def test_collection_while_staging_holds_the_lock_does_not_hang(tmp_path):
    try:
        done = subprocess.run(
            [sys.executable, "-c", _PROBE, str(ROOT), str(tmp_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - the bug
        pytest.fail(f"staging hung under garbage collection: {exc.stdout!r}")
    assert done.returncode == 0, done.stdout + done.stderr
    # The finalizer really ran: it queued the collected sandbox's copy
    # (the lock was held) and the next holder of the lock removed it.
    assert json.loads(done.stdout.splitlines()[-1]) == {
        "queued": True,
        "removed": True,
    }


def _environment(root: Path, name: str) -> Path:
    environment = root / name / "task" / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    return environment


def test_release_takes_no_lock(tmp_path):
    staged = staging.stage_environment(_environment(tmp_path, "a"))
    # Held by this thread, as stage_environment holds it during _live.add.
    with staging._live_lock:
        staged.release()
    assert staged.root.is_dir()  # queued, never waited for the lock
    staging.stage_environment(_environment(tmp_path, "b")).remove()
    assert not staged.root.exists()


def test_release_removes_at_once_when_the_lock_is_free(tmp_path):
    staged = staging.stage_environment(_environment(tmp_path, "a"))
    staged.release()
    assert not staged.root.exists()
    assert staged.root not in staging._live
    assert staged.root not in staging._released


def test_removal_queued_during_staging_is_applied_at_the_next_call(
    tmp_path, monkeypatch
):
    live = staging._live
    first = staging.stage_environment(_environment(tmp_path, "a"))

    class ReleasingSet(set):
        # A finalizer that runs while stage_environment holds the lock.
        def add(self, value):
            first.release()
            return super().add(value)

    monkeypatch.setattr(staging, "_live", ReleasingSet(live))
    second = staging.stage_environment(_environment(tmp_path, "b"))
    # Queued after this call drained: still live, still on disk.
    assert first.root in staging._live
    assert first.root.is_dir()

    # Back to the real set, with what the stand-in gained and lost.
    live.clear()
    live.update(staging._live)
    monkeypatch.setattr(staging, "_live", live)
    third = staging.stage_environment(_environment(tmp_path, "c"))
    assert first.root not in live
    assert not first.root.exists()
    assert {second.root, third.root} <= live

    first.remove()  # already released: a no-op, never a second removal
    second.remove()
    third.remove()
    assert not second.root.exists()
    assert not third.root.exists()
    assert not ({first.root, second.root, third.root} & live)


def test_finalizer_releases_and_explicit_remove_drains(tmp_path):
    first = staging.stage_environment(_environment(tmp_path, "a"))
    with staging._live_lock:
        first.release()
    second = staging.stage_environment(_environment(tmp_path, "b"))
    # stage_environment already drained the first; queue one more and let
    # an explicit remove take it with its own root.
    third = staging.stage_environment(_environment(tmp_path, "c"))
    with staging._live_lock:
        third.release()
    second.remove()
    assert not first.root.exists()
    assert not second.root.exists()
    assert not third.root.exists()
    assert not ({first.root, second.root, third.root} & staging._live)


def _queued(tmp_path: Path, name: str) -> staging.StagedEnvironment:
    """A staged copy whose finalizer found the lock held, so it waits."""
    staged = staging.stage_environment(_environment(tmp_path, name))
    with staging._live_lock:
        staged.release()
    assert staged.root.is_dir()
    return staged


def test_a_failed_clean_up_never_fails_another_trials_staging(tmp_path, monkeypatch):
    broken = _queued(tmp_path, "broken")
    rmtree = staging._rmtree

    def failing(root):
        if root == broken.root:
            raise PermissionError(13, "Permission denied", str(root))
        rmtree(root)

    monkeypatch.setattr(staging, "_rmtree", failing)
    other = staging.stage_environment(_environment(tmp_path, "other"))
    assert other.environment_dir.is_dir()
    # Still on disk, so queued again for the next holder of the lock.
    assert broken.root.is_dir()
    assert broken.root in staging._released

    monkeypatch.setattr(staging, "_rmtree", rmtree)
    other.remove()
    assert not broken.root.exists()
    assert broken.root not in staging._released


def test_an_interrupt_between_drain_and_delete_loses_no_copy(tmp_path, monkeypatch):
    first = _queued(tmp_path, "a")
    second = _queued(tmp_path, "b")
    rmtree = staging._rmtree
    interrupts = [KeyboardInterrupt()]

    def interrupted(root):
        if interrupts:
            raise interrupts.pop()
        rmtree(root)

    monkeypatch.setattr(staging, "_rmtree", interrupted)
    with pytest.raises(KeyboardInterrupt):
        staging.stage_environment(_environment(tmp_path, "c"))
    # Neither drained root is lost: each is gone (the failed staging's own
    # clean-up retried it) or queued again, never only on disk.
    for root in (first.root, second.root):
        assert not root.exists() or root in staging._released

    monkeypatch.setattr(staging, "_rmtree", rmtree)
    staging.remove_released()
    assert not first.root.exists()
    assert not second.root.exists()


def test_a_jobs_end_removes_queued_copies(tmp_path, monkeypatch):
    from benchflow.evaluation import Evaluation

    queued = _queued(tmp_path, "a")

    class Lock:
        def release(self):
            pass

    async def finished():
        return "result"

    evaluation = Evaluation.__new__(Evaluation)
    evaluation._preflight = False
    monkeypatch.setattr(evaluation, "_check_verifier_sandbox_plan", lambda: None)
    monkeypatch.setattr(evaluation, "_settle_auto_job_name", lambda: None)
    monkeypatch.setattr(evaluation, "_acquire_job_lock", Lock)
    monkeypatch.setattr(evaluation, "_run_unlocked", finished)
    assert asyncio.run(evaluation.run()) == "result"
    assert not queued.root.exists()
    assert queued.root not in staging._live


def test_bf_run_end_removes_queued_copies(tmp_path):
    from benchflow.runtime import run

    queued = _queued(tmp_path, "a")
    with pytest.raises(ValueError, match="task_path required"):
        asyncio.run(run("oracle"))
    assert not queued.root.exists()
    assert queued.root not in staging._live


def test_a_direct_rollout_run_removes_queued_copies(tmp_path, monkeypatch):
    from benchflow.rollout import Rollout

    queued = _queued(tmp_path, "a")

    async def scored():
        return "result"

    rollout = Rollout.__new__(Rollout)
    monkeypatch.setattr(rollout, "_run_with_scoring", scored)
    assert asyncio.run(rollout.run()) == "result"
    assert not queued.root.exists()
    assert queued.root not in staging._live
