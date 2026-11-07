"""The Daytona start path, round 4 (DAYTONA-START "Round 4").

The round-3 review's findings, each as a case that fails at 9cb125baf:

M1. A fresh sandbox retries only what a fresh sandbox can fix: a provider
    start failure (state ERROR), or a BUILD_FAILED of a recipe with no
    fallback. A recipe with cached install layers that fails to build gets
    one build, then the plain task image.
L1. ``bench sandbox cleanup --all`` never deletes a STARTED sandbox, lapsed
    lease or not; ``--id`` names one to delete it.
L2. The leftovers cleanup does its lookups and deletes outside the file
    lock, so a job on the same host never waits on them.
L3. A delete cancelled at a stop bound, a failed delete before a snapshot
    restore and a failed delete of the previous sandbox before a create
    retry each go to the ledger.
L4. The error a trial gets when Daytona stopped its sandbox names the
    inactivity stop, and stays an infrastructure error.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("daytona")

from daytona.common.errors import DaytonaError

from benchflow.sandbox import daytona as daytona_mod
from benchflow.sandbox import daytona_start
from tests import test_daytona_start_r3
from tests.test_daytona_install_layers import TASK_IMAGE, FakeDaytona, _start
from tests.test_daytona_start_r3 import (
    _cleanup,
    _Client,
    _err,
    _params,
    _resource,
    _sandbox,
    _Server,
)

# Round 3's fixture: no create backoff, short watch and settle times.
fast = test_daytona_start_r3.fast


def _start_failure(state: str) -> str:
    return (
        "Failed to create sandbox: Failure during waiting for sandbox to start: "
        f"Sandbox sb-failed failed to start with state: SandboxState.{state}, "
        'error reason: process "/bin/sh -c printf ... | base64 -d > /tmp/x && '
        'bash /tmp/x" did not complete successfully: exit code: 127'
    )


class _ScriptedDaytona(FakeDaytona):
    """Raises the SDK's own DaytonaError for the scripted creates, by recipe
    kind (layered: has a RUN line; plain: FROM only); then builds."""

    def __init__(self, *, layered: list[str], plain: list[str]) -> None:
        super().__init__()
        self.script = {"layered": list(layered), "plain": list(plain)}

    async def create(self, params, timeout=None):
        kind = "layered" if "RUN " in params.image.dockerfile() else "plain"
        if self.script[kind]:
            self.requests.append(params)
            self.timeouts.append(timeout)
            raise DaytonaError(_start_failure(self.script[kind].pop(0)))
        return await super().create(params, timeout)

    def list(self, query=None):
        async def empty():
            if False:
                yield None

        return empty()


async def _builds(monkeypatch, tmp_path, fake) -> tuple[int, int, object]:
    monkeypatch.setattr(daytona_start, "wait_sec", lambda *_: 0.0)
    manager = SimpleNamespace(get_client=AsyncMock(return_value=fake))
    monkeypatch.setattr(
        daytona_mod,
        "DaytonaClientManager",
        SimpleNamespace(get_instance=AsyncMock(return_value=manager)),
    )
    sandbox = await _start(fake, tmp_path / "trial-1")
    recipes = [request.image.dockerfile() for request in fake.requests]
    layered = sum("RUN " in r for r in recipes)
    return layered, len(recipes) - layered, sandbox


# ── M1 ──────────────────────────────────────────────────────────────────


async def test_a_layer_build_failure_costs_one_build_before_the_fallback(
    monkeypatch, tmp_path
):
    """The round-3 review's reproducer: the SDK's BUILD_FAILED for the
    layered recipe. At 9cb125baf: 3 layered builds, then the plain one."""
    fake = _ScriptedDaytona(layered=["BUILD_FAILED"] * 3, plain=[])
    layered, plain, sandbox = await _builds(monkeypatch, tmp_path, fake)
    assert (layered, plain) == (1, 1)
    assert fake.requests[-1].image.dockerfile() == f"FROM {TASK_IMAGE}\n"
    assert sandbox._install_layers_record["fallback"] == "layer build failed"


async def test_a_provider_start_failure_of_the_layered_recipe_is_retried(
    monkeypatch, tmp_path
):
    """ERROR is the provider's (run 2's sysbox deadline): a fresh sandbox
    of the same layered recipe, no fallback."""
    fake = _ScriptedDaytona(layered=["ERROR"], plain=[])
    layered, plain, sandbox = await _builds(monkeypatch, tmp_path, fake)
    assert (layered, plain) == (2, 0)
    assert "fallback" not in sandbox._install_layers_record


async def test_the_plain_build_after_the_fallback_keeps_the_fresh_sandbox_retry(
    monkeypatch, tmp_path
):
    """The plain task image has no fallback, so its BUILD_FAILED (run 2's
    transient curl) still gets a fresh sandbox."""
    fake = _ScriptedDaytona(layered=["BUILD_FAILED"], plain=["BUILD_FAILED"])
    layered, plain, _ = await _builds(monkeypatch, tmp_path, fake)
    assert (layered, plain) == (1, 2)


@pytest.mark.asyncio
async def test_a_plain_build_failure_still_gets_three_creates(fast):
    """No install layers: rule E as in round 3."""
    client = _Client([_err("DaytonaError", _start_failure("BUILD_FAILED"))] * 5)
    client.get.return_value = SimpleNamespace(id="x", delete=AsyncMock())
    sandbox = _sandbox(client)
    with pytest.raises(Exception, match="BUILD_FAILED"):
        await sandbox._create_sandbox(params=_params())
    assert client.creates == 3


def test_the_retry_rule_reads_the_env_being_created():
    def state(exc, falls_back):
        env = SimpleNamespace(_layer_build_falls_back=falls_back)
        outcome = SimpleNamespace(exception=lambda: exc)
        return SimpleNamespace(args=(env,), outcome=outcome)

    build = DaytonaError(_start_failure("BUILD_FAILED"))
    error = DaytonaError(_start_failure("ERROR"))
    assert not daytona_start.create_should_retry(state(build, True))
    assert daytona_start.create_should_retry(state(build, False))
    assert daytona_start.create_should_retry(state(error, True))
    assert daytona_start.create_should_retry(state(error, False))


# ── L1 ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("scenario", ["controller-paused", "writer-clock-behind"])
def test_cleanup_all_keeps_a_started_sandbox_whose_lease_reads_lapsed(
    monkeypatch, capsys, scenario
):
    """The round-3 review's L1 cases: a paused controller's sandbox, and a
    living run whose lease was written by a clock 4,201 s behind."""
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", "me")
    now = int(time.time())
    lease = now - 601 - 600 if scenario == "controller-paused" else now - 4201 + 3600
    server = _Server([_resource("started-lapsed", lease=lease, owner="me")])
    _cleanup(monkeypatch, server, all_mine=True)
    assert server.deleted == []
    out = " ".join(capsys.readouterr().out.split())
    assert "1 STARTED" in out and "--id" in out

    _cleanup(monkeypatch, server, names=frozenset({"started-lapsed"}))
    assert server.deleted == ["started-lapsed"]


@pytest.mark.parametrize("state", ["STARTED", "SandboxState.STARTED"])
def test_cleanup_all_keeps_a_started_sandbox_with_no_lease(monkeypatch, state):
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", "me")
    running = _resource("running", lease=0, owner="me", state=state)
    del running.labels["benchflow.lease"]
    stopped = _resource("stopped", lease=0, owner="me", state="STOPPED")
    del stopped.labels["benchflow.lease"]
    server = _Server([running, stopped])
    _cleanup(monkeypatch, server, all_mine=True)
    assert server.deleted == ["stopped"]


# ── L2 ──────────────────────────────────────────────────────────────────


def test_cleanup_does_not_hold_the_leftovers_lock_across_its_lookups():
    """The round-3 review's L2 case, in one process: the cleanup's one lookup
    takes 1.5 s, and a job's loop opens a ledger entry meanwhile. At
    9cb125baf the loop stalled for the whole lookup."""
    from benchflow.sandbox import daytona_ledger as ledger

    ledger._append(
        {
            "op": "open",
            "key": "attempt:old",
            "kind": "attempt",
            "labels": {ledger.CREATE_ATTEMPT_LABEL: "old"},
            "sandbox_id": None,
            "managed": "1:me",
            # Old enough that an empty lookup closes it.
            "opened_at": time.time() - 2 * ledger.FORGET_EMPTY_ATTEMPT_SEC,
        }
    )
    in_lookup = threading.Event()

    class Slow:
        def list(self, query):
            in_lookup.set()
            time.sleep(1.5)
            return []

    counts: dict = {}
    cleanup = threading.Thread(
        target=lambda: counts.update(ledger.cleanup_leftovers(Slow(), "1:me"))
    )
    cleanup.start()
    assert in_lookup.wait(10)

    async def job() -> float:
        ticks: list[float] = []

        async def ticker():
            for _ in range(20):
                ticks.append(time.monotonic())
                await asyncio.sleep(0.05)

        task = asyncio.ensure_future(ticker())
        await asyncio.sleep(0.1)
        ledger.LEDGER._open(
            ledger.Entry(
                key="attempt:new",
                kind="attempt",
                labels={ledger.CREATE_ATTEMPT_LABEL: "new"},
                sandbox_id=None,
                managed="1:me",
                opened_at=time.time(),
                settle_at=time.monotonic(),
                reason="test",
            )
        )
        await task
        return max(b - a for a, b in itertools.pairwise(ticks))

    stall = asyncio.run(job())
    cleanup.join(30)
    ledger.LEDGER.entries.clear()
    assert stall < 0.5, stall
    # The compaction keeps what the job opened during the cleanup.
    assert counts["closed"] == 1
    assert [r["key"] for r in ledger.read_leftovers()] == ["attempt:new"]


# ── L3 ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_delete_cancelled_at_the_stop_bound_goes_to_the_ledger(fast):
    """The round-3 review's L3 case: the separate verifier bounds stop()
    with wait_for; a delete still in flight is cancelled."""
    from benchflow.sandbox import daytona_ledger as ledger

    client = _Client()
    gate = asyncio.Event()

    async def slow_delete(*_a, **_kw):
        await gate.wait()

    client.made.delete.side_effect = slow_delete
    sandbox = _sandbox(client)
    await sandbox._create_sandbox(params=_params())
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(sandbox.stop(delete=True), timeout=0.2)
    assert sandbox._sandbox is None
    assert [r["sandbox_id"] for r in ledger.read_leftovers()] == ["sb-ok"]
    assert "sandbox:sb-ok" in ledger.LEDGER.entries or any(
        e.sandbox_id == "sb-ok" for e in ledger.LEDGER.entries.values()
    )


@pytest.mark.asyncio
async def test_a_failed_delete_of_the_previous_sandbox_goes_to_the_ledger(fast):
    """Guards 9b0320f: retain ownership and refuse replacement after failed deletion."""
    from benchflow.sandbox import daytona_ledger as ledger

    client = _Client()
    sandbox = _sandbox(client)
    sandbox._sandbox = SimpleNamespace(
        id="previous", delete=AsyncMock(side_effect=RuntimeError("502 Bad Gateway"))
    )
    # Guards 9b0320f: keep the previous handle and do not overlap a new create
    # while its deletion is unconfirmed; the durable ledger still retries it.
    previous = sandbox._sandbox
    from benchflow.sandbox.protocol import SandboxStartupError

    with pytest.raises(SandboxStartupError, match="502 Bad Gateway"):
        await sandbox._create_sandbox(params=_params())
    assert sandbox._sandbox is previous
    assert client.creates == 0
    assert [r["sandbox_id"] for r in ledger.read_leftovers()] == ["previous"]


@pytest.mark.asyncio
async def test_a_failed_delete_before_a_restore_goes_to_the_ledger(monkeypatch):
    """Guards 9b0320f: restore cannot overlap a sandbox whose deletion failed."""
    from benchflow.sandbox import daytona_ledger as ledger
    from benchflow.sandbox.protocol import SandboxImage

    sandbox = _sandbox(_Client())
    sandbox._sandbox = SimpleNamespace(
        id="before-restore",
        delete=AsyncMock(side_effect=RuntimeError("502 Bad Gateway")),
    )
    sandbox._auto_delete_interval = 0
    sandbox._auto_stop_interval = 15
    sandbox._network_block_all = False
    monkeypatch.setattr(daytona_mod, "_checkpoint_startup_env", lambda _image: {})

    # Guards 9b0320f: failed deletion must retain ownership before replacement.
    previous = sandbox._sandbox
    sandbox._create_sandbox_without_toolbox_bash_env = AsyncMock()
    with pytest.raises(RuntimeError, match="502 Bad Gateway"):
        await sandbox._strategy.restore(
            SandboxImage(provider="daytona", ref="snap-1", meta={})
        )
    assert sandbox._sandbox is previous
    sandbox._create_sandbox_without_toolbox_bash_env.assert_not_awaited()
    assert [r["sandbox_id"] for r in ledger.read_leftovers()] == ["before-restore"]


# ── L4 ──────────────────────────────────────────────────────────────────


async def test_a_stopped_sandbox_error_names_the_inactivity_stop():
    from benchflow._utils.scoring import INFRA_ERROR, classify_error
    from benchflow._utils.text import describe_exception
    from benchflow.sandbox.daytona_pty import stamp_transient_transport

    daytona_said = _err(
        "DaytonaValidationError",
        "Failed to get session command: bad request: failed to get session "
        "command: Is the Sandbox started? [status_code=400, "
        "error_code=SANDBOX_NOT_RUNNING]",
        400,
    )

    @stamp_transient_transport
    async def poll():
        raise daytona_said

    with pytest.raises(Exception) as caught:
        await poll()
    text = describe_exception(caught.value)
    # The trial's recorded error; before round 4 it was Daytona's text alone.
    assert text.startswith("DaytonaSandboxStoppedError: ")
    assert "Daytona stopped the sandbox" in text
    assert "15 min without activity" in text
    assert "paused, asleep or offline" in text
    assert "SANDBOX_NOT_RUNNING" in text
    assert caught.value.__cause__ is daytona_said
    assert classify_error(text) == INFRA_ERROR


async def test_other_daytona_errors_are_not_renamed():
    from benchflow.sandbox.daytona_pty import stamp_transient_transport

    rejected = _err("DaytonaValidationError", "bad request: invalid path", 400)

    @stamp_transient_transport
    async def call():
        raise rejected

    with pytest.raises(type(rejected)):
        await call()
