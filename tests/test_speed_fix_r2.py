"""SPEED-FIX r2: regressions introduced by 2b5c678e6's process-wide memo."""

import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from types import SimpleNamespace

import pytest

from benchflow.trajectories import types


@pytest.mark.parametrize("character", ["x", "\U0001f680"])
def test_memo_retains_at_most_four_mib_including_unicode(character):
    """Guard SPEED-FIX r2 against 2b5c678e6's 64M-character bound."""
    with types.trial_redaction_memo() as memo:
        for i in range(80):
            types.redact_trajectory_text_with_categories(character * 20000 + str(i))
        retained = sys.getsizeof(memo.entries)
        for text, value in memo.entries.items():
            retained += sys.getsizeof(text) + sys.getsizeof(value)
            retained += sys.getsizeof(value[0]) + sys.getsizeof(value[1])
            retained += sum(
                sys.getsizeof(k) + sys.getsizeof(v) for k, v in value[1].items()
            )
        assert memo.entries  # The bound must still permit useful caching.
        assert retained <= memo.size <= 4 << 20
        huge = character * (4 << 20)
        types.redact_trajectory_text_with_categories(huge)
        assert huge not in memo.entries


def test_threads_charge_duplicate_misses_once(monkeypatch):
    """Guard SPEED-FIX r2 against 2b5c678e6's unsynchronized counter updates."""
    barrier = threading.Barrier(8)
    pattern, replacement, category = types._REDACTION_PATTERNS[0]

    class RacingPattern:
        def subn(self, replacement, text):
            barrier.wait(timeout=10)
            return pattern.subn(replacement, text)

    monkeypatch.setattr(
        types, "_REDACTION_PATTERNS", [(RacingPattern(), replacement, category)]
    )
    with types.trial_redaction_memo() as memo, ThreadPoolExecutor(8) as pool:
        contexts = [copy_context() for _ in range(8)]
        futures = [
            pool.submit(context.run, types.redact_trajectory_text, "same secret text")
            for context in contexts
        ]
        assert [f.result(timeout=15) for f in futures] == ["same secret text"] * 8
        assert len(memo.entries) == 1
        size = memo.size
        memo.close()
        assert memo.size == 0 and not memo.entries
    with types.trial_redaction_memo() as single:
        # No regex/barrier: compare the charge for one insertion with eight misses.
        single.remember("same secret text", "same secret text", {})
        assert single.size == size


@pytest.mark.parametrize("exit_kind", ["success", "error", "cancel", "cleanup_error"])
async def test_trial_exit_clears_secrets_even_in_inherited_contexts(
    monkeypatch, exit_kind
):
    """Guard SPEED-FIX r2: 2b5c678e6 retains secrets after Rollout.run exits."""
    from benchflow.rollout import Rollout
    from benchflow.task import staging

    captured = []
    secret = "Authorization: Bearer " + "s" * 40

    async def scored():
        types.redact_trajectory_text(secret)
        memo = types._redaction_memo.get()
        assert memo is not None and secret in memo.entries
        captured.append((memo, copy_context()))
        if exit_kind == "error":
            raise RuntimeError("trial failed")
        if exit_kind == "cancel":
            raise asyncio.CancelledError
        return "result"

    def cleanup():
        types.redact_trajectory_text("cleanup text")
        if exit_kind == "cleanup_error":
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(staging, "remove_released", cleanup)
    rollout = SimpleNamespace(
        _run_with_scoring=scored, _config=SimpleNamespace(purpose="eval")
    )
    if exit_kind == "success":
        assert await Rollout.run(rollout) == "result"
    else:
        expected = asyncio.CancelledError if exit_kind == "cancel" else RuntimeError
        with pytest.raises(expected):
            await Rollout.run(rollout)
    memo, inherited = captured[0]
    assert memo.closed and memo.size == 0 and not memo.entries
    inherited.run(types.redact_trajectory_text, secret)
    assert memo.size == 0 and not memo.entries
    assert types._redaction_memo.get() is None
    types.redact_trajectory_text(secret)
    assert types._redaction_memo.get() is None


async def test_concurrent_trials_keep_separate_memos(monkeypatch):
    """Guard SPEED-FIX r2 against 2b5c678e6 sharing secrets between SDK trials."""
    from benchflow.rollout import Rollout
    from benchflow.task import staging

    first_done = asyncio.Event()
    both_started = asyncio.Barrier(2)
    memos = {}

    async def trial(name):
        async def scored():
            await asyncio.to_thread(types.redact_trajectory_text, name)
            memo = types._redaction_memo.get()
            assert memo is not None and name in memo.entries
            memos[name] = memo
            await both_started.wait()
            assert set(memo.entries) == {name}
            if name == "second":
                await first_done.wait()
                assert not memos["first"].entries
                assert name in memo.entries

        rollout = SimpleNamespace(
            _run_with_scoring=scored, _config=SimpleNamespace(purpose="eval")
        )
        await Rollout.run(rollout)
        if name == "first":
            first_done.set()

    monkeypatch.setattr(staging, "remove_released", lambda: None)
    await asyncio.wait_for(asyncio.gather(trial("first"), trial("second")), timeout=10)
    assert all(m.closed and not m.entries for m in memos.values())


def test_inflight_redaction_cannot_refill_a_finished_trial(monkeypatch):
    """Guard SPEED-FIX r2: closing must beat a late 2b5c678e6-style insertion."""
    started = threading.Event()
    finish = threading.Event()
    pattern, replacement, category = types._REDACTION_PATTERNS[0]

    class PausedPattern:
        def subn(self, replacement, text):
            started.set()
            assert finish.wait(timeout=10)
            return pattern.subn(replacement, text)

    monkeypatch.setattr(
        types, "_REDACTION_PATTERNS", [(PausedPattern(), replacement, category)]
    )
    with ThreadPoolExecutor(1) as pool:
        with types.trial_redaction_memo() as memo:
            future = pool.submit(
                copy_context().run, types.redact_trajectory_text, "secret"
            )
            try:
                assert started.wait(timeout=10)
            finally:
                memo.close()
                finish.set()
        assert future.result(timeout=15) == "secret"
        assert not memo.entries and memo.size == 0
