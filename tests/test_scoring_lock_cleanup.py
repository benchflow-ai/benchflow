"""The scoring lock leaves nothing behind in a finished trial (REPORT B17)."""

from __future__ import annotations

import fcntl
from pathlib import Path

import pytest

from benchflow.review import persistence
from benchflow.review.persistence import scoring_lock

# The literal name, not the module constant: the test reads the trial folder.
SCORING_LOCK_NAME = ".scoring.lock"


def test_a_released_lock_leaves_no_file(tmp_path: Path) -> None:
    with scoring_lock(tmp_path):
        assert (tmp_path / SCORING_LOCK_NAME).is_file()
    assert not (tmp_path / SCORING_LOCK_NAME).exists()
    assert list(tmp_path.iterdir()) == []


def test_a_failed_scoring_leaves_no_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="verifier broke"), scoring_lock(tmp_path):
        raise RuntimeError("verifier broke")
    assert not (tmp_path / SCORING_LOCK_NAME).exists()


def test_a_second_scorer_is_still_refused_and_keeps_the_holders_file(
    tmp_path: Path,
) -> None:
    with scoring_lock(tmp_path):
        with pytest.raises(ValueError, match="already running"), scoring_lock(tmp_path):
            pass
        # The refused scorer must not remove the holder's lock file.
        assert (tmp_path / SCORING_LOCK_NAME).is_file()
    assert not (tmp_path / SCORING_LOCK_NAME).exists()


def test_a_lock_left_by_a_crashed_scorer_is_taken_and_removed(tmp_path: Path) -> None:
    # A killed process leaves its (unlocked) file behind; the next scorer takes it.
    (tmp_path / SCORING_LOCK_NAME).write_text("")
    with scoring_lock(tmp_path):
        pass
    assert not (tmp_path / SCORING_LOCK_NAME).exists()


def test_a_scorer_that_locked_the_removed_file_locks_a_fresh_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finished holder removed the file between this scorer's open and its
    lock: the lock it got is on a file no other scorer can see, so it must take
    the lock again on the file the name now points to."""
    lock_path = tmp_path / SCORING_LOCK_NAME
    real_flock = fcntl.flock
    calls: list[int] = []

    def flock_after_holder_left(stream, operation):
        if operation & fcntl.LOCK_EX and not calls:
            # The previous holder's release: the name goes away.
            lock_path.unlink()
        calls.append(operation)
        return real_flock(stream, operation)

    monkeypatch.setattr(persistence.fcntl, "flock", flock_after_holder_left)
    with scoring_lock(tmp_path):
        assert lock_path.is_file()
        # A scorer arriving now sees the live lock and is refused.
        monkeypatch.setattr(persistence.fcntl, "flock", real_flock)
        with pytest.raises(ValueError, match="already running"), scoring_lock(tmp_path):
            pass
    exclusive = [op for op in calls if op & fcntl.LOCK_EX]
    assert len(exclusive) == 2, calls
    assert not lock_path.exists()
