"""Verifier lifetime fault injection for SV4-B91-R1-1 at 70b40639."""

from __future__ import annotations

import asyncio
import threading

import pytest

from benchflow import budget
from benchflow.rollout import _separate_verifier as separate
from benchflow.rollout import _setup
from tests.test_separate_verifier import (
    FakeVerifierSandbox,
    _agent_trial,
    _fake_rollout,
    _task_dir,
)


class InjectedFailure(RuntimeError):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona"])
@pytest.mark.parametrize("cached", [False, True], ids=["cache-off", "cache-on"])
@pytest.mark.parametrize("fault", ["cancel", "raise", "none"])
@pytest.mark.parametrize(
    "phase",
    [
        "start",
        "upload",
        "readiness",
        "trajectory",
        "scoring",
        "publication",
        "fallback-publication",
    ],
)
async def test_verifier_lifetime_owns_every_exit(
    tmp_path, monkeypatch, backend, cached, fault, phase
):
    """Guards SV4-B91-R1-1: teardown precedes settlement at every await.

    Publication uses the real worker and cancellation drain, with dummy files.
    A readiness error reaches fallback publication before the first cancel.
    Backend transports are local stand-ins; no daemon or credentials are used.
    """
    monkeypatch.setenv("BENCHFLOW_VERIFIER_LAYER", "1" if cached else "0")
    _, workspace, trial = await _agent_trial(tmp_path)
    task_dir = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task_dir, str(workspace))
    rollout._config.environment = backend
    events = []
    entered = asyncio.Event()
    release = asyncio.Event()
    worker_entered = threading.Event()
    worker_release = threading.Event()
    settlement = []

    async def checkpoint(name):
        events.append(name)
        if phase == name and fault != "none":
            entered.set()
            await release.wait()
            raise InjectedFailure(name)

    class Box(FakeVerifierSandbox):
        def use_verifier_layer(self, recipe):
            assert (recipe is not None) == cached

        async def start(self, **kwargs):
            await super().start(**kwargs)
            await checkpoint("start")

        async def upload_file(self, *args, **kwargs):
            await checkpoint("upload")
            await super().upload_file(*args, **kwargs)

        async def exec(self, *args, **kwargs):
            await checkpoint("readiness")
            if phase == "fallback-publication":
                raise separate.SeparateVerifierError("dummy readiness failure")
            return await super().exec(*args, **kwargs)

        async def stop(self, **kwargs):
            events.append("stop-start")
            # Teardown really suspends; an un-awaited task cannot pass.
            await asyncio.sleep(0)
            await super().stop(**kwargs)
            events.append("stop-finished")

    box = Box(tmp_path / "verifier-box")
    start_charge = budget.start_verifier_charge

    def tracked_charge():
        charge = start_charge()
        finish = charge.finish

        def settled():
            settlement.append(list(events))
            finish()

        monkeypatch.setattr(charge, "finish", settled)
        return charge

    monkeypatch.setattr(budget, "start_verifier_charge", tracked_charge)

    async def trajectory(*args, **kwargs):
        await checkpoint("trajectory")

    async def scoring(env, task, paths, *args, **kwargs):
        await checkpoint("scoring")
        (paths.verifier_dir / "reward.txt").write_text("1\n")
        return {"reward": 1.0}, None, None

    monkeypatch.setattr(_setup, "_publish_trajectory_for_verifier", trajectory)
    monkeypatch.setattr(_setup, "_verify_rollout", scoring)
    publish = separate._publish_verifier_outputs

    def publication(source, *args):
        name = "fallback-publication" if source is None else "publication"
        events.append(name)
        if name == phase and fault != "none":
            worker_entered.set()
            if not worker_release.wait(10):
                raise AssertionError("test did not release publication")
            if fault == "raise":
                raise InjectedFailure(name)
        publish(source, *args)
        events.append(f"{name}-finished")

    monkeypatch.setattr(separate, "_publish_verifier_outputs", publication)
    running = asyncio.create_task(
        separate.run_separate_verifier(rollout, create_environment=lambda *a: box)
    )
    try:
        if fault != "none":
            if phase.endswith("publication"):
                async with asyncio.timeout(10):
                    while not worker_entered.is_set() and not running.done():
                        await asyncio.sleep(0.001)
                assert worker_entered.is_set()
            else:
                await asyncio.wait_for(entered.wait(), timeout=10)
            if fault == "cancel":
                running.cancel()
                if phase.endswith("publication"):
                    # A second cancel must also wait for the publishing worker.
                    await asyncio.sleep(0)
                    running.cancel()
                    await asyncio.sleep(0)
                    assert not settlement
            release.set()
            worker_release.set()
        if fault == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(running, timeout=10)
        else:
            rewards, error = await asyncio.wait_for(running, timeout=10)
            if fault == "none" and phase != "fallback-publication":
                assert rewards == {"reward": 1.0} and error is None
                assert (trial / "verifier" / "reward.txt").read_text() == "1\n"
    finally:
        release.set()
        worker_release.set()
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

    assert box.started and box.stopped == [True]
    assert events.count("stop-start") == events.count("stop-finished") == 1
    assert len(settlement) == 1 and settlement[0][-1] == "stop-finished"
    assert phase in events


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona"])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("prior_cancel", [False, True])
async def test_repeated_cancellation_waits_for_teardown(
    tmp_path, monkeypatch, backend, cached, prior_cancel
):
    """Guards SV4-B91-R1-1's sibling: cancellation inside teardown itself."""
    monkeypatch.setenv("BENCHFLOW_VERIFIER_LAYER", "1" if cached else "0")
    _, workspace, trial = await _agent_trial(tmp_path)
    task_dir = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task_dir, str(workspace))
    rollout._config.environment = backend
    verifying = asyncio.Event()
    stopping = asyncio.Event()
    release = asyncio.Event()
    charges = []
    start_charge = budget.start_verifier_charge

    def tracked_charge():
        charge = start_charge()
        charges.append(charge)
        return charge

    monkeypatch.setattr(budget, "start_verifier_charge", tracked_charge)

    class Box(FakeVerifierSandbox):
        def use_verifier_layer(self, recipe):
            assert (recipe is not None) == cached

        async def stop(self, **kwargs):
            stopping.set()
            await release.wait()
            await super().stop(**kwargs)

    box = Box(tmp_path / "verifier-box")

    async def verify(*args, **kwargs):
        verifying.set()
        if prior_cancel:
            await asyncio.Event().wait()
        return {"reward": 1.0}, None

    running = asyncio.create_task(
        separate.run_separate_verifier(
            rollout, create_environment=lambda *a: box, verify=verify
        )
    )
    try:
        await asyncio.wait_for(verifying.wait(), timeout=10)
        if prior_cancel:
            running.cancel()
        await asyncio.wait_for(stopping.wait(), timeout=10)
        for _ in range(3):
            running.cancel()
            await asyncio.sleep(0)
        unfinished = not running.done() and charges[0].finished is None
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, timeout=10)
    finally:
        release.set()
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
    assert unfinished, "charge settled while teardown was still allocated"
    assert box.stopped == [True]
    assert charges[0].finished is not None
