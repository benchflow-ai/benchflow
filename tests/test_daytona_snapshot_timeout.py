"""Daytona checkpoint snapshots: the create's timeout, and a snapshot that outlives it.

0.10.0 called ``_experimental_create_snapshot(name)`` with the SDK's 60 s
default. A claude-agent-acp sandbox (about 3 GiB of install) takes about 4
minutes, so ``--checkpoints`` failed with ``DaytonaTimeoutError`` while Daytona
went on to finish the snapshot: 2.48 GB and 2.495 GB, ACTIVE, recorded nowhere
(two live runs on 0.10.0). While it was being made Daytona also refused to
delete the sandbox, which was left running after the job.

These rows run the real ``_DaytonaDirect.snapshot``/``stop`` and
``checkpoints.after_prompt`` against a fake SDK whose snapshot service answers
scripted states. No key, no network.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

daytona_sdk = pytest.importorskip("daytona")

from benchflow.checkpoints import after_prompt, parse_checkpoint_policy  # noqa: E402
from benchflow.sandbox import daytona as sdk  # noqa: E402
from benchflow.sandbox._snapshot_credentials import StashedCredential  # noqa: E402
from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDirect  # noqa: E402
from benchflow.sandbox.daytona_reaper import owner_snapshot_prefix  # noqa: E402
from tests import test_image_trees  # noqa: E402

sdk._load_daytona_sdk()
# The production look and poll, before any row shortens them.
REAL_LOOK = vars(_DaytonaDirect).get("snapshot_look_sec")
REAL_STOP_LOOK = vars(_DaytonaDirect).get("snapshot_stop_look_sec")
REAL_POLL = vars(_DaytonaDirect).get("snapshot_poll_sec")

ABSENT = "absent"
LOOKUP_FAILS = "lookup fails"
TIMED_OUT = "Failed to create snapshot: Function 'create_snapshot' exceeded timeout of 60 seconds."


class Snapshots:
    """Daytona's snapshot service. Each ``get`` answers the next scripted
    state (the last one repeats); a ``delete`` Daytona accepts makes the
    snapshot ``removing``, then gone. ``events`` is shared with the test."""

    def __init__(
        self, script: list[str], events: list, *, refuse_deletes: bool = False
    ):
        self.script = list(script)
        self.events = events
        self.refuse_deletes = refuse_deletes
        self.gets: list[str] = []
        self.deletes: list[str] = []
        self.block: asyncio.Event | None = None
        # When set, the state comes from this instead of the script.
        self.state_now = None

    async def get(self, name: str):
        if self.block is not None:
            await self.block.wait()
        self.gets.append(name)
        if self.state_now is not None:
            state = self.state_now()
        else:
            state = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        self.events.append(("get", state))
        if state == ABSENT:
            raise daytona_sdk.DaytonaNotFoundError(f"Snapshot {name} not found")
        if state == LOOKUP_FAILS:
            raise daytona_sdk.DaytonaError(
                "upstream not found: bad gateway", status_code=502
            )
        return SimpleNamespace(name=name, state=sdk.SnapshotState(state))

    async def delete(self, snapshot) -> None:
        self.events.append(("delete", snapshot.name))
        if self.refuse_deletes:
            raise daytona_sdk.DaytonaError("Snapshot is being created", status_code=409)
        self.deletes.append(snapshot.name)
        self.script = ["removing", ABSENT]


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A Daytona sandbox whose create is scripted per row, and its trial folder."""
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", "snapshot-owner")
    monkeypatch.delenv("BENCHFLOW_DAYTONA_SNAPSHOT_TIMEOUT_SEC", raising=False)
    # The look after a failed create, and again at stop(), in test time.
    monkeypatch.setattr(_DaytonaDirect, "snapshot_look_sec", 0.3, raising=False)
    monkeypatch.setattr(_DaytonaDirect, "snapshot_stop_look_sec", 0.3, raising=False)
    monkeypatch.setattr(_DaytonaDirect, "snapshot_poll_sec", 0.01, raising=False)
    events: list = []
    snapshots = Snapshots(["snapshotting"], events)
    client = SimpleNamespace(snapshot=snapshots)

    class Manager:
        async def get_client(self):
            return client

    async def get_instance():
        return Manager()

    monkeypatch.setattr(sdk.DaytonaClientManager, "get_instance", get_instance)
    trial = tmp_path / "primes__abc"
    trial.mkdir()
    creates: list[tuple[str, dict]] = []

    env = DaytonaSandbox.__new__(DaytonaSandbox)
    env.logger = logging.getLogger("benchflow")
    env.environment_name = "primes"
    env._toolbox_startup_receipt = dict.fromkeys(sdk.SYSTEM_SHELL_EMPTY_VARS, "")
    env._client_manager = None
    env.rollout_paths = SimpleNamespace(rollout_dir=trial)

    def failing_create(error: BaseException):
        async def create(name, **kwargs):
            creates.append((name, kwargs))
            events.append(("create", name))
            raise error

        return create

    # The sandbox's own states, one per refresh (the last one repeats).
    sandbox_states = ["started"]

    def sandbox(create):
        handle = SimpleNamespace(
            id="sb-1",
            _experimental_create_snapshot=create,
            # The toolbox exec that clears Daytona's session files first.
            process=SimpleNamespace(
                exec=AsyncMock(return_value=SimpleNamespace(exit_code=0))
            ),
        )

        async def refresh_data():
            state = (
                sandbox_states.pop(0) if len(sandbox_states) > 1 else sandbox_states[0]
            )
            handle.state = state
            events.append(("sandbox state", state))

        handle.refresh_data = refresh_data
        env._sandbox = handle

    async def stop_sandbox(handle):
        events.append(("sandbox delete", handle.id))

    env._stop_sandbox = stop_sandbox
    env._log_no_sandbox_to_stop = lambda: None
    env._log_sandbox_left_behind = lambda exc, sandbox=None, **_: events.append(
        ("left behind", getattr(sandbox, "id", None))
    )
    env._strategy = _DaytonaDirect(env)
    monkeypatch.setattr(env._strategy, "_credential_ops", lambda: SimpleNamespace())
    stash: list[StashedCredential] = []

    async def scrub(ops):
        events.append(("scrub",))
        return list(stash)

    async def put_back(ops, items):
        events.append(("put back", len(items)))

    monkeypatch.setattr("benchflow.sandbox.daytona_strategies.scrub_credentials", scrub)
    monkeypatch.setattr(
        "benchflow.sandbox.daytona_strategies.put_back_credentials", put_back
    )
    sandbox(failing_create(daytona_sdk.DaytonaTimeoutError(TIMED_OUT)))
    rollout = SimpleNamespace(
        _config=SimpleNamespace(
            checkpoints=parse_checkpoint_policy("prompt:1", keep=1)
        ),
        _env=env,
        _rollout_dir=trial,
        _rollout_name="primes__abc",
        _cursor=SimpleNamespace(id="n6"),
    )
    return SimpleNamespace(
        env=env,
        rollout=rollout,
        snapshots=snapshots,
        events=events,
        creates=creates,
        stash=stash,
        trial=trial,
        sandbox=sandbox,
        sandbox_states=sandbox_states,
        failing_create=failing_create,
    )


def _row(world) -> dict:
    document = json.loads((world.trial / "checkpoints.json").read_text())
    [row] = document["checkpoints"]
    return row


def _name(world) -> str:
    return world.creates[0][0]


# K1: the create gets 600 s, not the SDK's 60.
async def test_the_create_is_given_600_seconds_by_default(world):
    world.snapshots.script = ["active"]
    await after_prompt(world.rollout, 1)
    [(_, kwargs)] = world.creates
    assert kwargs.get("timeout") == 600


# K2: the setting, and what an unusable value gets.
@pytest.mark.parametrize(
    ("raw", "seconds", "warned"),
    [
        ("1800", 1800.0, False),
        (" 90.5 ", 90.5, False),
        ("0", 600.0, True),
        ("-5", 600.0, True),
        ("soon", 600.0, True),
        ("nan", 600.0, True),
        ("inf", 600.0, True),
        ("-inf", 600.0, True),
        ("1e400", 600.0, True),
    ],
)
async def test_the_timeout_is_BENCHFLOW_DAYTONA_SNAPSHOT_TIMEOUT_SEC(
    world, monkeypatch, caplog, raw, seconds, warned
):
    monkeypatch.setenv("BENCHFLOW_DAYTONA_SNAPSHOT_TIMEOUT_SEC", raw)
    world.snapshots.script = ["active"]
    with caplog.at_level(logging.WARNING, logger="benchflow"):
        await after_prompt(world.rollout, 1)
    [(_, kwargs)] = world.creates
    assert kwargs.get("timeout") == seconds
    assert ("BENCHFLOW_DAYTONA_SNAPSHOT_TIMEOUT_SEC" in caplog.text) is warned


# K3: the 2.48 GB case. The create times out, Daytona finishes the snapshot:
# it is the checkpoint, and its handle says it came late.
async def test_a_snapshot_active_after_the_create_timed_out_is_the_checkpoint(world):
    world.snapshots.script = ["snapshotting", "snapshotting", "active"]
    await after_prompt(world.rollout, 1)
    row = _row(world)
    name = _name(world)
    assert (row["status"], row["ref"], row["provider"]) == ("kept", name, "daytona")
    assert "leftover" not in row
    late = json.loads(row["meta"]["benchflow_snapshot_late"])
    assert late["cause"] == "DaytonaTimeoutError"
    assert late["timeout_sec"] == 600
    assert world.snapshots.deletes == []
    assert set(world.snapshots.gets) == {name}
    assert world.env._strategy._snapshot_leftovers == {}


# K3b (guard): a lookup that fails (a 502 whose text says "not found") is not
# "no snapshot": the look goes on, and the late snapshot is still found.
async def test_a_failed_lookup_is_not_taken_for_no_snapshot(world):
    world.snapshots.script = [
        LOOKUP_FAILS,
        LOOKUP_FAILS,
        LOOKUP_FAILS,
        LOOKUP_FAILS,
        "active",
    ]
    await after_prompt(world.rollout, 1)
    row = _row(world)
    assert (row["status"], row["ref"]) == ("kept", _name(world))
    assert world.snapshots.deletes == []


# K4: the create times out and the snapshot fails on Daytona: deleted, checked
# gone, and the trial's record names it.
async def test_a_snapshot_that_failed_after_the_create_timed_out_is_deleted(world):
    world.snapshots.script = ["snapshotting", "error"]
    await after_prompt(world.rollout, 1)
    row = _row(world)
    name = _name(world)
    assert (row["status"], row["ref"]) == ("failed", None)
    assert row["error"] == {"type": "DaytonaTimeoutError", "code": "snapshot_failed"}
    assert row["leftover"]["name"] == name
    assert (row["leftover"]["outcome"], row["leftover"]["state"]) == (
        "removed",
        "error",
    )
    assert world.snapshots.deletes == [name]
    assert world.snapshots.gets[-1] == name and world.events[-1] == ("get", ABSENT)
    assert world.env._strategy._snapshot_leftovers == {}


# K5: still being made when the look ends. Nothing is deleted
# then (a delete during SNAPSHOTTING does not hold on Daytona); the stop waits
# for it, deletes it before the sandbox, and the record ends `removed`.
async def test_a_snapshot_still_being_made_is_removed_at_the_stop(world):
    world.snapshots.script = ["snapshotting"]
    await after_prompt(world.rollout, 1)
    name = _name(world)
    row = _row(world)
    assert row["status"] == "failed"
    assert (row["leftover"]["name"], row["leftover"]["outcome"]) == (
        name,
        "delete_deferred",
    )
    assert row["leftover"]["state"] == "snapshotting"
    assert world.snapshots.deletes == []
    assert ("delete", name) not in world.events

    world.snapshots.script = ["snapshotting", "active"]
    await world.env._strategy.stop(delete=True)
    assert world.snapshots.deletes == [name]
    order = [e[0] for e in world.events]
    assert order.index("delete") < order.index("sandbox delete")
    leftover = _row(world)["leftover"]
    assert (leftover["name"], leftover["outcome"], leftover["at"]) == (
        name,
        "removed",
        "stop",
    )
    assert world.env._strategy._snapshot_leftovers == {}


# K6: one that never goes stays named, under the owner prefix the
# hub's sweep and `bench sandbox cleanup` match; the sandbox is still deleted.
async def test_a_leftover_that_never_goes_stays_named(world, caplog):
    world.snapshots.script = ["snapshotting"]
    await after_prompt(world.rollout, 1)
    name = _name(world)
    with caplog.at_level(logging.WARNING, logger="benchflow"):
        await world.env._strategy.stop(delete=True)
    leftover = _row(world)["leftover"]
    assert (leftover["name"], leftover["outcome"], leftover["at"]) == (
        name,
        "delete_deferred",
        "stop",
    )
    assert name.startswith(owner_snapshot_prefix())
    assert name in caplog.text
    assert ("sandbox delete", "sb-1") in world.events
    assert world.snapshots.deletes == []


# K7: the agent's credential files are kept out of the snapshot and go back
# once, after the create call has ended and before after_prompt returns, in
# every case: late, still being made, failed, never there. Measured on Daytona:
# a snapshot's content is fixed when the create starts. A file written
# 0.2 s after the call (it returned 4.7 s later, writes wait while Daytona
# captures) and files written up to 186 s into a 293 s SNAPSHOTTING were not in
# the snapshot; one written just before the call was. So files put back after
# the call can't be captured, and the next prompt has its login.
@pytest.mark.parametrize(
    "script",
    [
        ["snapshotting", "active"],
        ["snapshotting"],
        ["snapshotting", "error"],
        [ABSENT],
    ],
    ids=["late", "still-snapshotting", "error", "absent"],
)
async def test_credentials_go_back_once_after_the_create_in_every_case(world, script):
    world.stash.append(
        StashedCredential(
            path="/home/agent/.claude/.credentials.json",
            uid="1001",
            gid="1001",
            mode="600",
            content=b"{}",
        )
    )
    world.snapshots.script = list(script)
    await after_prompt(world.rollout, 1)
    kinds = [e[0] for e in world.events]
    assert kinds.count("put back") == 1
    assert kinds.index("scrub") < kinds.index("create") < kinds.index("put back")


# K8: a cancelled create leaves its name to the stop, which deletes the snapshot.
async def test_a_cancelled_create_is_removed_at_the_stop(world):
    world.sandbox(world.failing_create(asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await after_prompt(world.rollout, 1)
    name = _name(world)
    assert name in getattr(world.env._strategy, "_snapshot_leftovers", {})
    world.snapshots.script = ["active"]
    await world.env._strategy.stop(delete=True)
    assert world.snapshots.deletes == [name]
    assert world.env._strategy._snapshot_leftovers == {}


# K9 (guard): a create Daytona refused (4xx), or one that never reached it,
# made nothing: one look, no delete.
@pytest.mark.parametrize(
    "error",
    [
        daytona_sdk.DaytonaError("invalid name", status_code=400),
        TypeError("unexpected keyword argument"),
    ],
    ids=["refused-400", "not-sent"],
)
async def test_a_create_that_made_nothing_looks_once_and_deletes_nothing(world, error):
    world.sandbox(world.failing_create(error))
    world.snapshots.script = [ABSENT]
    await after_prompt(world.rollout, 1)
    row = _row(world)
    assert row["error"]["type"] == type(error).__name__
    assert row["leftover"]["outcome"] == "none_found"
    assert world.snapshots.gets == [_name(world)]
    assert world.snapshots.deletes == []


# K10 (guard): a create that returns is unchanged: no lookup, no late mark.
async def test_a_create_that_returns_is_unchanged(world):
    async def create(name, **kwargs):
        world.creates.append((name, kwargs))

    world.sandbox(create)
    await after_prompt(world.rollout, 1)
    row = _row(world)
    assert row["status"] == "kept"
    assert "benchflow_snapshot_late" not in row["meta"]
    assert world.snapshots.gets == []


# K11: strict stops the agent's sandbox before the verifier under a 300 s
# bound, and cleanup stops it again. A stop cancelled while it waits for a
# leftover raises at once, hands the sandbox to the ledger (which retries its
# delete while the job lives) and leaves the leftover to the next stop, which
# deletes it and records that.
async def test_a_stop_cancelled_mid_wait_leaves_the_leftover_to_the_next_stop(world):
    world.snapshots.script = ["snapshotting"]
    await after_prompt(world.rollout, 1)
    name = _name(world)
    world.snapshots.block = asyncio.Event()
    stop = asyncio.create_task(world.env._strategy.stop(delete=True))
    await asyncio.sleep(0.05)
    stop.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stop
    assert ("left behind", "sb-1") in world.events
    assert name in world.env._strategy._snapshot_leftovers
    world.snapshots.block = None
    world.snapshots.script = ["active"]
    await world.env._strategy.stop(delete=True)
    assert world.snapshots.deletes == [name]
    leftover = _row(world)["leftover"]
    assert (leftover["outcome"], leftover["at"]) == ("removed", "stop")
    assert world.env._strategy._snapshot_leftovers == {}


# K12 is a live run on Daytona, not a row here.


class FakeClock:
    """``asyncio`` as daytona_snapshot_create sees it, on a clock that moves
    only when the module sleeps: a look of minutes runs in no time."""

    CancelledError = asyncio.CancelledError

    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def get_running_loop(self) -> FakeClock:
        return self

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        await asyncio.sleep(0)


# K13: at the production look and poll (not the fixture's 0.3 s), a snapshot
# that is active 240 s after the create failed (live runs took 150-240 s) is
# found and kept, by polls about 10 s apart; and the stop's look ends inside
# strict's 300 s bound on the stop before the verifier.
async def test_the_production_look_finds_a_snapshot_active_240_s_later(
    world, monkeypatch
):
    assert REAL_LOOK is not None and REAL_LOOK >= 240
    assert REAL_POLL is not None and 1 <= REAL_POLL <= 30
    assert REAL_STOP_LOOK is not None and 180 <= REAL_STOP_LOOK < 300
    monkeypatch.setattr(_DaytonaDirect, "snapshot_look_sec", REAL_LOOK)
    monkeypatch.setattr(_DaytonaDirect, "snapshot_poll_sec", REAL_POLL)
    clock = FakeClock()
    monkeypatch.setattr("benchflow.sandbox.daytona_snapshot_create.asyncio", clock)
    world.snapshots.state_now = lambda: "snapshotting" if clock.now < 240 else "active"
    await after_prompt(world.rollout, 1)
    row = _row(world)
    assert (row["status"], row["ref"]) == ("kept", _name(world))
    assert 20 <= len(world.snapshots.gets) <= 30, len(world.snapshots.gets)
    assert json.loads(row["meta"]["benchflow_snapshot_late"])["late_sec"] >= 240


# K14: a create that fails with a 5xx or on the transport may have started the
# snapshot too: it is looked for, and kept when it turns active.
@pytest.mark.parametrize(
    "error",
    [
        daytona_sdk.DaytonaError("502 Bad Gateway", status_code=502),
        ConnectionResetError("websocket closed"),
    ],
    ids=["http-502", "transport"],
)
async def test_a_5xx_or_transport_failure_is_looked_for(world, error):
    world.sandbox(world.failing_create(error))
    world.snapshots.script = ["snapshotting", "snapshotting", "active"]
    await after_prompt(world.rollout, 1)
    row = _row(world)
    assert (row["status"], row["ref"]) == ("kept", _name(world))


# K15: a delete Daytona refuses is tried again; one that never goes stays
# `delete_deferred`, named under the owner prefix in the row, the trial's
# leftovers file and the log, and the sandbox is still deleted.
async def test_a_refused_delete_is_retried_and_one_that_never_goes_stays_named(
    world, caplog
):
    world.snapshots.script = ["snapshotting", "error"]
    world.snapshots.refuse_deletes = True
    await after_prompt(world.rollout, 1)
    name = _name(world)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="benchflow"):
        await world.env._strategy.stop(delete=True)
    # Tried again within each look, not once per look.
    assert world.events.count(("delete", name)) >= 3
    leftover = _row(world)["leftover"]
    assert (leftover["name"], leftover["outcome"], leftover["at"]) == (
        name,
        "delete_deferred",
        "stop",
    )
    assert leftover["delete_tries"] >= 3
    assert "being created" in leftover["delete_error"]
    prefix = owner_snapshot_prefix()
    assert prefix is not None and name.startswith(prefix)
    assert _leftovers_file(world) == [name]
    assert f"Daytona snapshot {name} could not be deleted" in caplog.text
    assert ("sandbox delete", "sb-1") in world.events


# K15b: a delete refused twice, then accepted, ends `removed`.
async def test_a_delete_refused_twice_then_accepted_is_removed(world):
    world.snapshots.script = ["snapshotting", "error"]
    world.snapshots.refuse_deletes = True
    accepted = world.snapshots.delete
    tries = {"n": 0}

    async def refuse_twice(snapshot):
        tries["n"] += 1
        world.snapshots.refuse_deletes = tries["n"] <= 2
        await accepted(snapshot)

    world.snapshots.delete = refuse_twice
    await after_prompt(world.rollout, 1)
    leftover = _row(world)["leftover"]
    assert (leftover["outcome"], leftover["delete_tries"]) == ("removed", 3)
    assert world.snapshots.deletes == [_name(world)]


# K16: a delete Daytona accepts but that doesn't hold (seen live: removing,
# then active again) is made again; `removed` only once the snapshot is gone.
async def test_a_delete_that_does_not_hold_is_made_again(world):
    world.snapshots.script = ["snapshotting", "error"]
    accepted = world.snapshots.delete
    tries = {"n": 0}

    async def first_does_not_hold(snapshot):
        await accepted(snapshot)
        tries["n"] += 1
        if tries["n"] == 1:
            world.snapshots.script = ["removing", "active"]

    world.snapshots.delete = first_does_not_hold
    await after_prompt(world.rollout, 1)
    leftover = _row(world)["leftover"]
    assert tries["n"] >= 2
    assert leftover["outcome"] == "removed"
    assert world.events[-1] == ("get", ABSENT)


# K17: a cancel during the look (not the create: K8) leaves the name to the
# stop, which deletes the snapshot.
async def test_a_cancel_during_the_look_is_removed_at_the_stop(world):
    world.snapshots.block = asyncio.Event()
    task = asyncio.create_task(after_prompt(world.rollout, 1))
    await asyncio.sleep(0.05)
    assert world.creates
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    name = _name(world)
    assert name in world.env._strategy._snapshot_leftovers
    world.snapshots.block = None
    world.snapshots.script = ["active"]
    await world.env._strategy.stop(delete=True)
    assert world.snapshots.deletes == [name]
    assert _leftovers_file(world) == [name]


# K18: a cancelled create whose snapshot can't be removed at the stop, and a
# branch snapshot (no checkpoints.json row) left behind, are named in the
# trial's daytona-snapshot-leftovers.json.
@pytest.mark.parametrize("how", ["cancelled-checkpoint", "branch"])
async def test_a_leftover_no_row_names_is_in_the_trials_leftovers_file(world, how):
    world.snapshots.script = ["snapshotting"]
    if how == "cancelled-checkpoint":
        world.sandbox(world.failing_create(asyncio.CancelledError()))
        with pytest.raises(asyncio.CancelledError):
            await after_prompt(world.rollout, 1)
        assert not (world.trial / "checkpoints.json").exists()
    else:
        with pytest.raises(daytona_sdk.DaytonaTimeoutError):
            await world.env.snapshot()
    await world.env._strategy.stop(delete=True)
    [record] = json.loads(
        (world.trial / "daytona-snapshot-leftovers.json").read_text()
    )["leftovers"]
    assert (record["name"], record["outcome"], record["at"]) == (
        _name(world),
        "delete_deferred",
        "stop",
    )


# K19: the in-run branch path (rollout_branch -> checkpoint_composed ->
# sandbox.snapshot()) takes the same snapshot: a late one is the fork's image.
async def test_a_late_branch_snapshot_is_the_forks_image(world):
    world.snapshots.script = ["snapshotting", "active"]
    image = await world.env.snapshot()
    assert image.ref == _name(world)
    assert "benchflow_snapshot_late" in image.meta


# K20: a late checkpoint carries what an in-time one does (trust lineage,
# toolbox start-up, PATH inspector), plus the late mark, so a fork from it
# restores the same way.
async def test_a_late_checkpoint_has_an_in_time_ones_meta(world):
    world.rollout._config.checkpoints = parse_checkpoint_policy("prompt:1,2", keep=2)
    world.snapshots.script = ["snapshotting", "active"]
    await after_prompt(world.rollout, 1)

    async def in_time(name, **kwargs):
        world.creates.append((name, kwargs))

    world.sandbox(in_time)
    await after_prompt(world.rollout, 2)
    late, on_time = json.loads((world.trial / "checkpoints.json").read_text())[
        "checkpoints"
    ]
    assert set(late["meta"]) - {"benchflow_snapshot_late"} == set(on_time["meta"])
    assert "benchflow_snapshot_late" in late["meta"]


# K21 (guard): an in-time checkpoint is untouched by the stop: no lookup, no
# delete, no leftover, no leftovers file. With credential files too: the name
# registered for the stop while they go back (K28) is taken off once they're
# back, so the stop doesn't delete the kept checkpoint.
@pytest.mark.parametrize(
    "with_credentials", [False, True], ids=["plain", "credentials"]
)
async def test_an_in_time_checkpoint_survives_the_stop(world, with_credentials):
    if with_credentials:
        world.stash.append(
            StashedCredential(
                path="/home/agent/.claude/.credentials.json",
                uid="1001",
                gid="1001",
                mode="600",
                content=b"{}",
            )
        )

    async def in_time(name, **kwargs):
        world.creates.append((name, kwargs))

    world.sandbox(in_time)
    await after_prompt(world.rollout, 1)
    assert [e[0] for e in world.events].count("put back") == int(with_credentials)
    assert world.env._strategy._snapshot_leftovers == {}
    await world.env._strategy.stop(delete=True)
    row = _row(world)
    assert (row["status"], row["ref"]) == ("kept", _name(world))
    assert "leftover" not in row
    assert world.snapshots.gets == [] and world.snapshots.deletes == []
    assert not (world.trial / "daytona-snapshot-leftovers.json").exists()


# K22: lookups that all fail (an API outage: 403s or 503s) never read as "no
# snapshot": the name stays for the stop, and the record says why the look
# found nothing.
async def test_lookups_that_all_fail_are_recorded_and_left_to_the_stop(world):
    world.snapshots.script = [LOOKUP_FAILS]
    await after_prompt(world.rollout, 1)
    leftover = _row(world)["leftover"]
    assert leftover["outcome"] == "delete_deferred"
    assert leftover["lookups_failed"] >= 3
    assert (
        "502" in leftover["lookup_error"] or "bad gateway" in leftover["lookup_error"]
    )
    assert _name(world) in world.env._strategy._snapshot_leftovers


# K23: after a late `active`, the run goes on only once the sandbox has left
# SNAPSHOTTING too (the SDK's create waits for the sandbox, not the snapshot).
async def test_a_late_snapshot_waits_for_its_sandbox_to_leave_snapshotting(world):
    world.snapshots.script = ["snapshotting", "active"]
    world.sandbox_states[:] = ["snapshotting", "snapshotting", "started"]
    await after_prompt(world.rollout, 1)
    states = [e[1] for e in world.events if e[0] == "sandbox state"]
    assert states == ["snapshotting", "snapshotting", "started"]
    assert _row(world)["status"] == "kept"


def _leftovers_file(world) -> list[str]:
    path = world.trial / "daytona-snapshot-leftovers.json"
    return [r["name"] for r in json.loads(path.read_text())["leftovers"]]


# K26: the in-run branch's own record (tree.json's fork) names what became of
# the snapshot of a fork checkpoint whose create failed.
async def test_a_failed_fork_checkpoint_records_its_leftover(tmp_path):
    from benchflow.rollout import Rollout, RolloutConfig, Scene

    leftover = {"name": "bf-snap2-fork", "outcome": "delete_deferred"}

    class LeftoverSandbox:
        supports_snapshot = True

        async def snapshot(self, name=None):
            exc = daytona_sdk.DaytonaTimeoutError(TIMED_OUT)
            exc.benchflow_snapshot_leftover = dict(leftover)
            raise exc

    (tmp_path / "task").mkdir()
    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy --agent")]
        )
    )
    rollout._env = LeftoverSandbox()
    rollout._rollout_dir = tmp_path / "trial"
    rollout._rollout_dir.mkdir()

    async def run_child(child):
        return 1.0

    with pytest.raises(daytona_sdk.DaytonaTimeoutError):
        await rollout.branch(2, run_child=run_child, snapshot_layers={"sandbox"})
    [fork] = json.loads((tmp_path / "trial" / "tree.json").read_text())["forks"]
    assert fork["status"] == "failed"
    assert fork["snapshot"]["leftover"] == leftover


# K27: a snapshot that shows up only after the first lookups find nothing is
# still found when the create may have reached Daytona (a timeout, a 5xx), at
# the production poll; after a 4xx, one lookup and nothing deleted.
@pytest.mark.parametrize(
    ("error", "kept"),
    [
        (daytona_sdk.DaytonaTimeoutError(TIMED_OUT), True),
        (daytona_sdk.DaytonaError("502 Bad Gateway", status_code=502), True),
        (daytona_sdk.DaytonaError("invalid name", status_code=400), False),
    ],
    ids=["timeout", "http-502", "refused-400"],
)
async def test_a_snapshot_listed_late_is_found_unless_the_create_was_refused(
    world, monkeypatch, error, kept
):
    monkeypatch.setattr(_DaytonaDirect, "snapshot_look_sec", REAL_LOOK)
    monkeypatch.setattr(_DaytonaDirect, "snapshot_poll_sec", REAL_POLL)
    monkeypatch.setattr(
        "benchflow.sandbox.daytona_snapshot_create.asyncio", FakeClock()
    )
    world.sandbox(world.failing_create(error))
    world.snapshots.script = [ABSENT, ABSENT, "snapshotting", "active"]
    await after_prompt(world.rollout, 1)
    row = _row(world)
    if kept:
        assert (row["status"], row["ref"]) == ("kept", _name(world))
    else:
        assert row["leftover"]["outcome"] == "none_found"
        assert world.snapshots.gets == [_name(world)]
    assert world.snapshots.deletes == []


# K28: a cancel while the credential files go back after a create that
# returned in time: the snapshot exists and no row names it, so the stop
# deletes it.
async def test_a_cancel_during_the_put_back_leaves_the_snapshot_to_the_stop(
    world, monkeypatch
):
    world.stash.append(
        StashedCredential(
            path="/root/.codex/auth.json", uid="0", gid="0", mode="600", content=b"{}"
        )
    )

    async def in_time(name, **kwargs):
        world.creates.append((name, kwargs))

    world.sandbox(in_time)
    blocked = asyncio.Event()

    async def put_back(ops, items):
        world.events.append(("put back", len(items)))
        await blocked.wait()

    monkeypatch.setattr(
        "benchflow.sandbox.daytona_strategies.put_back_credentials", put_back
    )
    task = asyncio.create_task(after_prompt(world.rollout, 1))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    name = _name(world)
    assert name in world.env._strategy._snapshot_leftovers
    world.snapshots.script = ["active"]
    await world.env._strategy.stop(delete=True)
    assert world.snapshots.deletes == [name]


# K29: with --checkpoint-keep 1, a cancel while the older checkpoint is being
# deleted leaves the new one recorded in checkpoints.json.
async def test_a_cancel_during_the_rotation_leaves_the_new_checkpoint_recorded(
    world,
):
    world.rollout._config.checkpoints = parse_checkpoint_policy("prompt:1,2", keep=1)

    async def in_time(name, **kwargs):
        world.creates.append((name, kwargs))

    world.sandbox(in_time)
    await after_prompt(world.rollout, 1)
    world.snapshots.block = asyncio.Event()
    task = asyncio.create_task(after_prompt(world.rollout, 2))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    rows = json.loads((world.trial / "checkpoints.json").read_text())["checkpoints"]
    assert [r["ref"] for r in rows] == [world.creates[0][0], world.creates[1][0]]


# K30: the leftovers file keeps every stop's records: one an earlier stop
# removed stays named after a later stop settles another, and an unreadable
# earlier file is moved aside and named, never blocking the new records.
async def test_the_leftovers_file_keeps_every_stops_records(world):
    world.snapshots.script = ["snapshotting"]
    await after_prompt(world.rollout, 1)
    first = _name(world)
    world.snapshots.script = ["active"]
    await world.env._strategy.stop(delete=True)
    assert world.snapshots.deletes == [first]

    world.sandbox(world.failing_create(daytona_sdk.DaytonaTimeoutError(TIMED_OUT)))
    world.creates.clear()
    world.snapshots.script = ["snapshotting"]
    with pytest.raises(daytona_sdk.DaytonaTimeoutError):
        await world.env.snapshot()
    second = _name(world)
    await world.env._strategy.stop(delete=True)
    leftovers_file = world.trial / "daytona-snapshot-leftovers.json"
    records = {
        r["name"]: r["outcome"]
        for r in json.loads(leftovers_file.read_text())["leftovers"]
    }
    assert records == {first: "removed", second: "delete_deferred"}

    leftovers_file.write_text("{not json")
    world.sandbox(world.failing_create(daytona_sdk.DaytonaTimeoutError(TIMED_OUT)))
    world.creates.clear()
    with pytest.raises(daytona_sdk.DaytonaTimeoutError):
        await world.env.snapshot()
    third = _name(world)
    await world.env._strategy.stop(delete=True)
    document = json.loads(leftovers_file.read_text())
    # The second, still deferred, is settled again by this stop.
    assert {r["name"] for r in document["leftovers"]} == {second, third}
    assert (world.trial / document["unreadable_earlier_file"]).read_text() == (
        "{not json"
    )


# K31: a late snapshot whose sandbox never leaves SNAPSHOTTING doesn't hold the
# run past the look: it is kept and the run goes on, with a warning.
async def test_a_sandbox_still_snapshotting_after_a_late_active_holds_nothing(
    world, caplog
):
    world.snapshots.script = ["snapshotting", "active"]
    world.sandbox_states[:] = ["snapshotting"]
    with caplog.at_level(logging.WARNING, logger="benchflow"):
        await asyncio.wait_for(after_prompt(world.rollout, 1), 5)
    assert _row(world)["status"] == "kept"
    assert "its sandbox is still snapshotting" in caplog.text


# K32: the most credit one Daytona snapshot gets against the hard deadline is
# its own worst case, from the settings: the create's timeout and the look
# after it, and a tenth more for the credential ops (990 s at the defaults).
def test_a_daytona_snapshots_deadline_credit_is_its_worst_case(world, monkeypatch):
    monkeypatch.setattr(_DaytonaDirect, "snapshot_look_sec", REAL_LOOK)
    assert world.env._strategy.snapshot_budget_sec() == pytest.approx(990.0)
    monkeypatch.setenv("BENCHFLOW_DAYTONA_SNAPSHOT_TIMEOUT_SEC", "900")
    assert world.env._strategy.snapshot_budget_sec() == pytest.approx(1320.0)


# K33: a Daytona checkpoint wedged below its own bounds (a create that ignores
# its timeout) still trips the hard deadline, once its credit (timeout + look)
# is used: here a 0.3 s deadline, a 1 s create timeout and a 0.5 s look.
async def test_a_wedged_daytona_checkpoint_still_trips_the_deadline(world, monkeypatch):
    from pathlib import Path

    from benchflow import checkpoints
    from benchflow.models import RolloutResult
    from benchflow.rollout import _deadline

    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "0.3")
    monkeypatch.setenv("BENCHFLOW_DAYTONA_SNAPSHOT_TIMEOUT_SEC", "1")
    monkeypatch.setattr(_DaytonaDirect, "snapshot_look_sec", 0.5)
    monkeypatch.setattr(_deadline, "ABANDON_CLEANUP_BOUND_SEC", 0.1)

    async def wedged(name, **kwargs):
        world.creates.append((name, kwargs))
        await asyncio.Event().wait()

    world.sandbox(wedged)

    async def lifecycle():
        await after_prompt(world.rollout, 1)
        return RolloutResult(task_name="primes", rollout_name="primes__abc")

    result = await asyncio.wait_for(
        _deadline.enforce_hard_deadline(
            lifecycle(),
            config=SimpleNamespace(task_path=Path("primes")),
            paused_sec=lambda: checkpoints.seconds_spent(world.rollout),
        ),
        timeout=10,
    )
    assert result.error is not None and "hard deadline" in result.error


class _CutChannels:
    """The command-channel set a snapshot cut: its PTY websocket closed and the
    restarts tried while the sandbox was snapshotting used up its budget, so
    every run loses its exec to the transport (what the post-turn walk hit)."""

    cleanup_task = None
    finalized = False

    def __init__(self, provider, events) -> None:
        self.provider = provider
        self.events = events

    def retire(self) -> None:
        self.events.append(("channels retired",))

    async def close(self) -> None:
        self.events.append(("channels closed",))

    async def run(self, *args, **kwargs):
        from benchflow.diagnostics import TransientSandboxTransportError

        raise TransientSandboxTransportError(
            "the sandbox's command channel stopped: websocket closed (close_code=1006)"
        )


# K34: once a snapshot's create (and the look after a failed one) has ended, in
# time, late or failed, the command-channel set it cut is retired, before the
# credential files go back, and channels stay on for the next command.
@pytest.mark.parametrize("how", ["in-time", "late", "failed"])
async def test_a_snapshot_retires_the_channel_set_it_cut(world, how):
    world.stash.append(
        StashedCredential(
            path="/root/.codex/auth.json", uid="0", gid="0", mode="600", content=b"{}"
        )
    )
    if how == "in-time":

        async def in_time(name, **kwargs):
            world.creates.append((name, kwargs))
            world.events.append(("create", name))

        world.sandbox(in_time)
    world.snapshots.script = (
        ["snapshotting", "active"] if how == "late" else ["snapshotting", "error"]
    )
    world.env._command_channels = _CutChannels(world.env._sandbox, world.events)
    await after_prompt(world.rollout, 1)
    kinds = [e[0] for e in world.events]
    assert (
        kinds.index("create")
        < kinds.index("channels retired")
        < kinds.index("put back")
    )
    assert world.env._command_channels is None
    assert world.env._command_channels_closing is False
    # The cut set is closed in the background (its PTY session, its helper).
    await asyncio.sleep(0.05)
    assert ("channels closed",) in world.events
    assert world.env._retired_command_channels == []


# K35: after that, the next root command (the post-turn walk's is one) runs,
# and on a new channel set, where 0.10.0 sent it to the cut set and it lost its
# exec to the transport.
async def test_after_a_snapshot_the_next_command_starts_a_new_channel(monkeypatch):
    from benchflow.sandbox import daytona_channel
    from benchflow.sandbox._base import KIT_PROBE_ROUTE
    from tests.daytona_channel_fakes import _lifecycle_sandbox

    box, _ = _lifecycle_sandbox(monkeypatch)
    box._sandbox.process.create_pty_session = AsyncMock()
    events: list = []
    started: list = []

    class NewChannels:
        def __init__(self, sandbox, route) -> None:
            self.provider = sandbox._sandbox
            started.append(route)

        async def run(self, *args, **kwargs):
            return None  # still starting: this exec takes the session path

    monkeypatch.setattr(daytona_channel, "DaytonaChannels", NewChannels)
    box._command_channels = _CutChannels(box._sandbox, events)
    box._command_channels_closing = False
    assert daytona_channel.eligible(box, KIT_PROBE_ROUTE)

    box._renew_command_channels()
    result = await box._sandbox_session_exec(
        "echo answer",
        timeout_sec=5,
        route=KIT_PROBE_ROUTE,
        cleanup_session=False,
        use_channel=True,
    )
    assert result.return_code == 0
    assert started == [KIT_PROBE_ROUTE]
    assert events == [("channels retired",)]


def _walk_env(exec_side_effect):
    """A sandbox whose root execs answer in turn from *exec_side_effect*, with
    a record from before the turn (a Python walk)."""
    from benchflow.sandbox import _image_manifest as im
    from tests.test_image_trees import _manifest

    before = _manifest(
        interpreters={
            "/usr/bin/python3": [0o100755, 0, 0, 10, 5 * 10**9, 6 * 10**9, "", "a"]
        },
        ids={"/usr/bin/python3": [1, 2, 1]},
    )
    env = SimpleNamespace(exec=AsyncMock(side_effect=exec_side_effect))
    env._benchflow_image_manifest = before
    after = im.ImageManifest(taken="post-turn", files={}, trees=before.trees)
    return env, before, after


def _cut(message="websocket closed (close_code=1006)"):
    from benchflow.diagnostics import TransientSandboxTransportError

    return TransientSandboxTransportError(
        f"the sandbox's command channel stopped: {message}"
    )


# K36: the post-turn walk's exec that loses its exec to the transport (a
# snapshot cut the channel) is asked again, as bfx's reads are, and the walk
# completes: the record is checked, not walk-failed.
async def test_the_post_turn_walk_is_asked_again_after_a_transport_loss():
    from benchflow.sandbox import _image_manifest as im

    env, before, after = _walk_env([])
    env.exec.side_effect = [
        _cut(),
        SimpleNamespace(return_code=0, stdout=after.to_json(), stderr=""),
    ]
    walked = await im.walk_after(env, timeout_sec=5)
    assert walked is not None and walked.taken == "post-turn"
    assert env.exec.await_count == 2
    record = im.detector_record("audit", before, walked)
    assert record["status"] == "checked"


# K37: when every attempt loses its exec, the walk is failed, and the record
# says so (walk-failed, flagged), never quiet.
async def test_a_walk_that_loses_every_attempt_is_walk_failed():
    from benchflow.sandbox import _image_manifest as im
    from benchflow.sandbox._verifier_links import READ_ATTEMPTS

    env, before, _ = _walk_env([_cut() for _ in range(READ_ATTEMPTS + 2)])
    assert await im.walk_after(env, timeout_sec=5) is None
    assert env.exec.await_count == READ_ATTEMPTS
    record = im.detector_record(
        "audit",
        before,
        None,
        walk_failed=await im._walk_failed(env, before, timeout_sec=5),
    )
    assert record["status"] == "flagged"
    assert [f["category"] for f in record["flags"]] == ["walk-failed"]


# K38: the shell-only look at the interpreters that comes back empty or cut
# (a channel dropped mid-output), or with a line that can't be read, reads as
# not checked, named on the walk-failed flag, never as "edited"; a whole line
# that differs still flags.
@pytest.mark.parametrize(
    ("output", "kinds", "unchecked"),
    [
        ("", [], ["/usr/bin/python3"]),
        ("/usr/bin/python3|81ed|0|0", [], ["/usr/bin/python3"]),
        ("/usr/bin/python3|81ed|0|0\n", [], ["/usr/bin/python3"]),
        ("/usr/bin/python3|81ed|0|0|ten|5|6\n", [], ["/usr/bin/python3"]),
        ("/usr/bin/python3|81ed|0|0|10|5|99|1|2|1\n", ["edited"], None),
    ],
    ids=["empty", "cut", "short-line", "unreadable-field", "whole-and-changed"],
)
async def test_an_empty_or_cut_interpreter_look_is_not_checked(
    output, kinds, unchecked
):
    from benchflow.sandbox import _image_manifest as im

    env, before, _ = _walk_env(
        [SimpleNamespace(return_code=0, stdout=output, stderr="")]
    )
    found = await im._walk_failed(env, before, timeout_sec=5)
    assert found[0]["category"] == "walk-failed"
    assert [f["kind"] for f in found[1:]] == kinds
    assert found[0].get("interpreters_not_checked") == unchecked


# K39: a snapshot records when it ran on the sandbox (its create's start to the
# end of the create and the look after a failed one), for the ctime rule below.
async def test_a_snapshot_records_its_window(world):
    import time as _time

    before = _time.time()
    world.snapshots.script = ["snapshotting", "active"]
    await after_prompt(world.rollout, 1)
    [(start, end)] = world.env._snapshot_windows
    assert before <= start <= end <= _time.time()


# K40-K42: measured on Daytona, a snapshot sets the ctime of files in
# the sandbox's writable layer (the runtime kit's interpreter among them) to its
# start, while their content, mode, owner, size and mtime stay. The shell look
# after a failed walk compares ctime, the one field a process can't set back,
# except when it is the only change and falls inside a snapshot window the
# trial recorded: then the walk-failed flag names it as not compared.
@pytest.mark.parametrize(
    ("windows", "line", "kinds", "noted"),
    [
        ([(1000.0, 1100.0)], "/usr/bin/python3|81ed|0|0|10|5|1050", [], True),
        ([(1000.0, 1100.0)], "/usr/bin/python3|81ed|0|0|10|5|2000", ["edited"], False),
        ([], "/usr/bin/python3|81ed|0|0|10|5|1050", ["edited"], False),
        ([(1000.0, 1100.0)], "/usr/bin/python3|81ed|0|0|10|7|1050", ["edited"], False),
    ],
    ids=["inside-a-window", "outside-it", "no-snapshot", "inside-but-mtime-too"],
)
async def test_a_ctime_moved_by_a_snapshot_is_noted_not_flagged(
    windows, line, kinds, noted
):
    from benchflow.sandbox import _image_manifest as im

    env, before, _ = _walk_env(
        [SimpleNamespace(return_code=0, stdout=line + "|1|2|1\n", stderr="")]
    )
    env._snapshot_windows = windows
    found = await im._walk_failed(env, before, timeout_sec=5)
    assert found[0]["category"] == "walk-failed"
    assert [f["kind"] for f in found[1:]] == kinds
    notes = found[0].get("ctime_not_compared")
    if noted:
        [note] = notes
        assert note["path"] == "/usr/bin/python3"
        assert note["note"] == (
            "ctime moved during the snapshot at 1970-01-01T00:16:40+00:00; not compared"
        )
    else:
        assert notes is None


# K43: inside a recorded window only a ctime-only change is excused in the
# shell look: a changed size, mode (a setuid bit), owner or group still flags.
@pytest.mark.parametrize(
    "line",
    [
        "/usr/bin/python3|81ed|0|0|11|5|1050",
        "/usr/bin/python3|89ed|0|0|10|5|1050",
        "/usr/bin/python3|81ed|1001|0|10|5|1050",
        "/usr/bin/python3|81ed|0|1001|10|5|1050",
    ],
    ids=["size", "mode-setuid", "owner", "group"],
)
async def test_inside_a_window_any_other_change_still_flags(line):
    from benchflow.sandbox import _image_manifest as im

    env, before, _ = _walk_env(
        [SimpleNamespace(return_code=0, stdout=line + "|1|2|1\n", stderr="")]
    )
    env._snapshot_windows = [(1000.0, 1100.0)]
    found = await im._walk_failed(env, before, timeout_sec=5)
    assert [f["kind"] for f in found[1:]] == ["edited"]
    assert "ctime_not_compared" not in found[0]


# K44: every snapshot records its window, however its create
# ends: in time, late, still snapshotting when the look ends, failed, or never
# seen (each may have moved the ctime). K39 is the late path alone.
@pytest.mark.parametrize(
    "case", ["in-time", "late", "still-snapshotting", "error", "absent"]
)
async def test_every_snapshot_records_its_window(world, case):
    import time as _time

    t0 = _time.time()
    if case == "in-time":

        async def in_time(name, **kwargs):
            world.creates.append((name, kwargs))

        world.sandbox(in_time)
    else:
        world.snapshots.script = {
            "late": ["snapshotting", "active"],
            "still-snapshotting": ["snapshotting"],
            "error": ["snapshotting", "error"],
            "absent": [ABSENT],
        }[case]
    await after_prompt(world.rollout, 1)
    [(start, end)] = world.env._snapshot_windows
    assert t0 <= start <= end <= _time.time()


# K45: the window covers the create call itself, where Daytona's
# snapshot begins and sets the ctime: a window stamped at the create's end, or
# of no length at its start, would miss a long create's mark.
async def test_the_window_covers_the_create_call(world):
    import time as _time

    seen = {}

    async def create(name, **kwargs):
        world.creates.append((name, kwargs))
        seen["at"] = _time.time()
        await asyncio.sleep(0.2)

    world.sandbox(create)
    await after_prompt(world.rollout, 1)
    [(start, end)] = world.env._snapshot_windows
    assert start <= seen["at"] and seen["at"] + 0.2 <= end


# K46: a look cut inside its last number (no closing newline)
# still reads as six numbers; only lines that end with a newline, as each the
# look prints does, are read, so it is not checked, never "edited".
async def test_a_look_cut_inside_the_ctime_is_not_checked():
    from benchflow.sandbox import _image_manifest as im

    env, before, _ = _walk_env(
        [
            SimpleNamespace(
                return_code=0, stdout="/usr/bin/python3|81ed|0|0|10|5|6", stderr=""
            )
        ]
    )
    found = await im._walk_failed(env, before, timeout_sec=5)
    assert [f["kind"] for f in found[1:]] == []
    assert found[0].get("interpreters_not_checked") == ["/usr/bin/python3"]


# K46b: output with CRLF line ends (a terminal's) reads as with LF: a gone
# interpreter is removed, an unchanged one is checked.
async def test_a_look_with_crlf_line_ends_reads_the_same():
    from benchflow.sandbox import _image_manifest as im

    env, before, _ = _walk_env(
        [SimpleNamespace(return_code=0, stdout="/usr/bin/python3|gone\r\n", stderr="")]
    )
    found = await im._walk_failed(env, before, timeout_sec=5)
    assert [(f["path"], f["kind"]) for f in found[1:]] == [
        ("/usr/bin/python3", "removed")
    ]
    assert "interpreters_not_checked" not in found[0]


# K47 (guard): only a transport loss is asked again; an rc other
# than 0, a timeout or any other failure fails the walk at once.
@pytest.mark.parametrize("kind", ["rc", "timeout", "other"])
async def test_only_a_transport_loss_is_asked_again(kind):
    from benchflow.sandbox import _image_manifest as im
    from benchflow.sandbox.protocol import ExecTimeoutError

    first = {
        "rc": SimpleNamespace(return_code=1, stdout="", stderr="boom"),
        "timeout": ExecTimeoutError(5),
        "other": RuntimeError("boom"),
    }[kind]
    env, _, _ = _walk_env([first, SimpleNamespace(return_code=0, stdout="", stderr="")])
    assert await im.walk_after(env, timeout_sec=5) is None
    assert env.exec.await_count == 1


def _full_walk(section, path, was, now, windows, ids=((1, 2, 1), (1, 2, 1))):
    """detect() on a walk after the turn that ran, with *path* in *section*
    as *was* before the turn and *now* after it; *ids* are its inode, device
    and link count in each walk."""
    from benchflow.sandbox import _image_manifest as im
    from tests.test_image_trees import _manifest

    def walk(entry, identity):
        # A start-up file is test.sh's (BASH_ENV names it), so an edit flags.
        read = [path] if section == "startup" else []
        return _manifest(
            **{section: {path: entry}},
            owned={},
            processes=[],
            startup_env=read,
            ids={path: list(identity)} if identity else {},
        )

    before = walk(was, ids[0])
    after = im.ImageManifest(taken="post-turn", files={}, trees=walk(now, ids[1]).trees)
    env = SimpleNamespace(_benchflow_image_manifest=before)
    if windows is not None:
        env._snapshot_windows = windows
    return env, after


_S = 10**9


# K48: the walk after the turn compares an entry the walk before didn't hash
# (a PATH tool, a Python-tree file) by its fingerprint. A ctime that alone
# moved, inside a recorded snapshot window, on a file only root can write, is a
# snapshot-ctime change with the note, not a flag. Size, mode, owner and mtime
# must still match, bytes too where both walks hashed it, and the ctime must be
# inside a window.
@pytest.mark.parametrize(
    ("section", "path"),
    [
        ("path_tools", "/usr/local/bin/pytest"),
        ("python_trees", "/s/_pytest/main.py"),
        ("interpreters", "/usr/local/bin/python3"),
        ("startup", "/root/.bashrc"),
    ],
)
@pytest.mark.parametrize(
    ("now", "windows", "excused"),
    [
        ([0o100755, 0, 0, 100, 5 * _S, 1050 * _S, "", "h"], [(1000.0, 1100.0)], True),
        ([0o100755, 0, 0, 100, 5 * _S, 2000 * _S, "", "h"], [(1000.0, 1100.0)], False),
        ([0o100755, 0, 0, 100, 5 * _S, 1050 * _S, "", "h"], None, False),
        ([0o100755, 0, 0, 100, 7 * _S, 1050 * _S, "", "h"], [(1000.0, 1100.0)], False),
        ([0o100755, 0, 0, 101, 5 * _S, 1050 * _S, "", "h"], [(1000.0, 1100.0)], False),
        ([0o104755, 0, 0, 100, 5 * _S, 1050 * _S, "", "h"], [(1000.0, 1100.0)], False),
        (
            [0o100755, 1001, 0, 100, 5 * _S, 1050 * _S, "", "h"],
            [(1000.0, 1100.0)],
            False,
        ),
    ],
    ids=["inside", "outside", "no-snapshot", "mtime", "size", "setuid", "owner"],
)
async def test_a_snapshot_ctime_is_not_an_edit_in_the_walk_that_runs(
    section, path, now, windows, excused, monkeypatch
):
    from benchflow.sandbox import _image_manifest as im

    # An interpreter or start-up file the walk before couldn't read ("?") is
    # compared by its fingerprint too.
    was = [
        0o100755,
        0,
        0,
        100,
        5 * _S,
        6 * _S,
        "",
        "?" if section in ("interpreters", "startup") else "",
    ]
    env, after = _full_walk(section, path, was, now, windows)

    async def walked(env, **kwargs):
        return after

    monkeypatch.setattr(im, "walk_after", walked)
    record = await im.detect(env, "audit", timeout_sec=5, test_script="pytest -q")
    assert record["after"] is not None
    moved = [c for c in record["changes"] if c["category"] == "snapshot-ctime"]
    if excused:
        assert record["flags"] == [], record["flags"]
        [row] = moved
        assert (row["where"], row["path"], row["kind"]) == (section, path, "edited")
        assert row["note"] == (
            "ctime moved during the snapshot at 1970-01-01T00:16:40+00:00; not compared"
        )
    else:
        assert moved == []
        assert [(f["path"], f["kind"]) for f in record["flags"]] == [(path, "edited")]


# K48b (guard): where both walks hashed the entry, its bytes decide: different
# bytes with only the ctime moved inside a window is still an edit.
async def test_a_hashed_entry_with_new_bytes_is_an_edit_inside_a_window(monkeypatch):
    from benchflow.sandbox import _image_manifest as im

    path = "/usr/local/bin/python3"
    was = [0o100755, 0, 0, 100, 5 * _S, 6 * _S, "", "a"]
    now = [0o100755, 0, 0, 100, 5 * _S, 1050 * _S, "", "b"]
    env, after = _full_walk("interpreters", path, was, now, [(1000.0, 1100.0)])

    async def walked(env, **kwargs):
        return after

    monkeypatch.setattr(im, "walk_after", walked)
    record = await im.detect(env, "audit", timeout_sec=5)
    assert [(f["path"], f["kind"]) for f in record["flags"]] == [(path, "edited")]
    assert [c for c in record["changes"] if c["category"] == "snapshot-ctime"] == []


# K49: a ctime counts as the snapshot's within 5 s either side of its window
# (the host's clock against the sandbox's), not 6.
@pytest.mark.parametrize(
    ("ctime", "excused"),
    [(996, True), (994, False), (1104, True), (1106, False)],
    ids=["4s-before", "6s-before", "4s-after", "6s-after"],
)
async def test_the_window_margin_is_five_seconds(ctime, excused):
    from benchflow.sandbox import _image_manifest as im

    env, before, _ = _walk_env(
        [
            SimpleNamespace(
                return_code=0,
                stdout=f"/usr/bin/python3|81ed|0|0|10|5|{ctime}|1|2|1\n",
                stderr="",
            )
        ]
    )
    env._snapshot_windows = [(1000.0, 1100.0)]
    found = await im._walk_failed(env, before, timeout_sec=5)
    assert [f["kind"] for f in found[1:]] == ([] if excused else ["edited"])


# K48c (guard): with a snapshot window recorded, an entry planted or removed
# in the turn is what it is without one (a planted pytest file flags). Also
# when both walks recorded its identity (through another section: the map is
# one per walk, by path), and the missing side never escapes as an error.
@pytest.mark.parametrize("ids", [False, True], ids=["no-identity", "identity"])
@pytest.mark.parametrize("kind", ["planted", "removed"])
async def test_a_planted_or_removed_entry_is_unchanged_by_a_window(
    kind, ids, monkeypatch
):
    from benchflow.sandbox import _image_manifest as im
    from tests.test_image_trees import _manifest

    path = "/s/_pytest/main.py"
    entry = [0o100644, 0, 0, 100, 5 * _S, 1050 * _S, "", "h"]

    def walk(trees):
        return _manifest(
            python_trees=trees,
            owned={},
            processes=[],
            startup_env=[],
            ids={path: [1, 2, 1]} if ids else {},
        )

    before = walk({} if kind == "planted" else {path: entry})
    after = im.ImageManifest(
        taken="post-turn",
        files={},
        trees=walk({path: entry} if kind == "planted" else {}).trees,
    )
    env = SimpleNamespace(
        _benchflow_image_manifest=before, _snapshot_windows=[(1000.0, 1100.0)]
    )

    async def walked(env, **kwargs):
        return after

    monkeypatch.setattr(im, "walk_after", walked)
    record = await im.detect(env, "audit", timeout_sec=5)
    rows = record["flags"] + record["changes"]
    assert [(r["path"], r["kind"]) for r in rows] == [(path, kind)]
    assert rows[0]["category"] != "snapshot-ctime"


def _walk_detect(
    section, path, was, now, windows, monkeypatch, ids=((1, 2, 1), (1, 2, 1))
):
    from benchflow.sandbox import _image_manifest as im

    env, after = _full_walk(section, path, was, now, windows, ids)

    async def walked(env, **kwargs):
        return after

    monkeypatch.setattr(im, "walk_after", walked)
    return im.detect(env, "audit", timeout_sec=5, test_script="pytest -q")


_ROOT = [0o100755, 0, 0, 100, 5 * _S, 6 * _S, "", ""]


# K50: a snapshot sets the ctime of an edited file too, so only a file the
# agent user couldn't have edited in place (root's, with no group or other
# write bit) is excused; a group change, a link's new target, or a read that
# failed after the turn still flags; a root-owned link whose target is the
# same is excused (its own mode is always 0777).
@pytest.mark.parametrize(
    ("section", "path", "was", "now", "excused"),
    [
        (
            "path_tools",
            "/usr/local/bin/pytest",
            [0o100644, 1000, 1000, 100, 5 * _S, 6 * _S, "", ""],
            [0o100644, 1000, 1000, 100, 5 * _S, 1050 * _S, "", "h"],
            False,
        ),
        (
            "python_trees",
            "/s/_pytest/main.py",
            [0o100644, 1000, 1000, 100, 5 * _S, 6 * _S, "", ""],
            [0o100644, 1000, 1000, 100, 5 * _S, 1050 * _S, "", "h"],
            False,
        ),
        (
            "path_tools",
            "/usr/local/bin/pytest",
            [0o100775, 0, 0, 100, 5 * _S, 6 * _S, "", ""],
            [0o100775, 0, 0, 100, 5 * _S, 1050 * _S, "", "h"],
            False,
        ),
        (
            "path_tools",
            "/usr/local/bin/pytest",
            [0o100757, 0, 0, 100, 5 * _S, 6 * _S, "", ""],
            [0o100757, 0, 0, 100, 5 * _S, 1050 * _S, "", "h"],
            False,
        ),
        (
            "path_tools",
            "/usr/local/bin/pytest",
            _ROOT,
            [0o100755, 0, 1001, 100, 5 * _S, 1050 * _S, "", "h"],
            False,
        ),
        (
            "path_tools",
            "/usr/local/bin/pytest",
            [0o120777, 0, 0, 14, 5 * _S, 6 * _S, "/opt/x/pytest1", ""],
            [0o120777, 0, 0, 14, 5 * _S, 1050 * _S, "/tmp/x/pytest1", ""],
            False,
        ),
        (
            "path_tools",
            "/usr/local/bin/pytest",
            [0o120777, 0, 0, 14, 5 * _S, 6 * _S, "/opt/x/pytest1", ""],
            [0o120777, 0, 0, 14, 5 * _S, 1050 * _S, "/opt/x/pytest1", ""],
            True,
        ),
        (
            "interpreters",
            "/usr/local/bin/python3",
            [0o100755, 0, 0, 100, 5 * _S, 6 * _S, "", "a"],
            [0o100755, 0, 0, 100, 5 * _S, 1050 * _S, "", "?"],
            False,
        ),
        (
            "path_tools",
            "/usr/local/bin/pytest",
            _ROOT,
            [0o100755, 0, 0, 100, 5 * _S, 1050 * _S, "", "?"],
            False,
        ),
    ],
    ids=[
        "agent-owned-tool",
        "agent-owned-pytest",
        "root-group-writable",
        "root-other-writable",
        "group",
        "link-retargeted",
        "root-link-same-target",
        "hashed-before-unreadable-after",
        "unreadable-after",
    ],
)
async def test_only_a_file_only_root_can_write_is_excused(
    section, path, was, now, excused, monkeypatch
):
    record = await _walk_detect(
        section, path, was, now, [(1000.0, 1100.0)], monkeypatch
    )
    moved = [c["path"] for c in record["changes"] if c["category"] == "snapshot-ctime"]
    if excused:
        assert record["flags"] == [] and moved == [path], record["flags"]
    else:
        assert moved == []
        assert [(f["path"], f["kind"]) for f in record["flags"]] == [(path, "edited")]


# K51: the shell look after a failed walk follows the same rule: an
# interpreter the agent user owns, or one a group can write, whose ctime alone
# moved inside a window, is still edited.
@pytest.mark.parametrize(
    ("entry", "line"),
    [
        (
            [0o100755, 1000, 1000, 10, 5 * _S, 6 * _S, "", "a"],
            "/usr/bin/python3|81ed|1000|1000|10|5|1050",
        ),
        (
            [0o100775, 0, 0, 10, 5 * _S, 6 * _S, "", "a"],
            "/usr/bin/python3|81fd|0|0|10|5|1050",
        ),
    ],
    ids=["agent-owned", "group-writable"],
)
async def test_the_shell_look_excuses_only_a_file_only_root_can_write(entry, line):
    from benchflow.sandbox import _image_manifest as im
    from tests.test_image_trees import _manifest

    # The record's identity matches the look's, so the owner rule decides.
    before = _manifest(
        interpreters={"/usr/bin/python3": entry}, ids={"/usr/bin/python3": [1, 2, 1]}
    )
    env = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout=line + "|1|2|1\n", stderr="")
            ]
        ),
        _benchflow_image_manifest=before,
        _snapshot_windows=[(1000.0, 1100.0)],
    )
    found = await im._walk_failed(env, before, timeout_sec=5)
    assert [f["kind"] for f in found[1:]] == ["edited"]
    assert "ctime_not_compared" not in found[0]


# K52: the same through the real walk, in a container. A file the agent user
# owns (as a venv in its workspace is), edited in place with its size kept and
# its mtime set back, gets its ctime moved like an untouched file's when a
# snapshot follows; it still flags. A root-owned file no other user can write,
# whose ctime alone moved (a chmod to its own mode), is a snapshot-ctime change.
_EDIT_IN_PLACE = (
    "python3 -c 'import os, sys\n"
    'p = sys.argv[1]; st = os.lstat(p); d = bytearray(open(p, "rb").read())\n'
    "d[-1] = 32 if d[-1] != 32 else 10\n"
    'f = open(p, "r+b"); f.write(bytes(d)); f.close()\n'
    "os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))' "
)
_STAT = (
    "python3 -c 'import os, sys; s = os.lstat(sys.argv[1]); "
    "print(s.st_size, s.st_mtime_ns, s.st_ctime_ns)' "
)


box = test_image_trees.box


def _window_around_ctime(box, path):
    ctime = int(box.sh(_STAT + path).split()[2]) / 1e9
    box._snapshot_windows = [(ctime - 2.0, ctime + 2.0)]


@pytest.mark.parametrize("which", ["pytest-own", "path-tool"])
def test_an_agent_owned_file_edited_before_a_snapshot_still_flags(box, which):
    from tests.test_image_trees import SITE, TEST_SH, _detect, _record

    path = (
        f"{SITE}/_pytest/__init__.py" if which == "pytest-own" else "/usr/local/bin/pip"
    )
    box.sh(f"chown 1000:1000 {path}")
    _record(box)
    was = box.sh(_STAT + path).split()
    box.sh(_EDIT_IN_PLACE + path)
    now = box.sh(_STAT + path).split()
    assert was[:2] == now[:2] and was[2] != now[2]
    _window_around_ctime(box, path)
    record = _detect(box, test_script=TEST_SH)
    assert (path, "edited", which) in {
        (f["path"], f["kind"], f["category"]) for f in record["flags"]
    }, [(c["path"], c["category"]) for c in record["changes"] if c["path"] == path]


def test_a_root_only_file_whose_ctime_alone_moved_is_excused(box):
    from tests.test_image_trees import SITE, TEST_SH, _detect, _record

    path = f"{SITE}/_pytest/__init__.py"
    box.sh(f"chown 0:0 {path} && chmod 644 {path}")
    _record(box)
    box.sh(f"chmod 644 {path}")
    _window_around_ctime(box, path)
    record = _detect(box, test_script=TEST_SH)
    assert [f for f in record["flags"] if f["path"] == path] == []
    assert [c["category"] for c in record["changes"] if c["path"] == path] == [
        "snapshot-ctime"
    ]


# K57: the shell look after a failed walk, on a real file: its stat's inode,
# device and link count are what the Python walk before recorded. An
# interpreter only root can write whose ctime alone moved is noted; the same
# interpreter swapped for a copy with the same size, mode and mtime (three
# renames: another inode) is edited, and so is one given a hard link (its link
# count, which a snapshot keeps). The interpreter is copied up first (a chmod to its own
# mode): on overlayfs the copy-up of an image file with a second hard link
# gives it a new inode and link count, and a snapshot copies nothing up. The
# look compares whole seconds, hence the wait.
@pytest.mark.parametrize("case", ["ctime-only", "swapped", "hard-linked"])
def test_the_look_finds_the_identity_the_walk_before_recorded(box, case):
    from benchflow.sandbox import _image_manifest as im
    from tests.test_image_trees import _record

    path, other = "/usr/local/bin/python3.6", "/usr/local/bin/python3.9"
    box.sh(f"chmod 755 {path} && cp -p {path} {other}")
    before = _record(box)
    box.sh("sleep 1.2")
    if case == "ctime-only":
        box.sh(f"chmod 755 {path}")
    elif case == "hard-linked":
        box.sh(f"ln {path} /usr/local/bin/.linked")
    else:
        box.sh(
            f"mv {path} /usr/local/bin/t && mv {other} {path} "
            f"&& mv /usr/local/bin/t {other}"
        )
    _window_around_ctime(box, path)
    found = asyncio.run(im._walk_failed(box, before, timeout_sec=120))
    edited = sorted(f["path"] for f in found[1:] if f["kind"] == "edited")
    noted = [n["path"] for n in found[0].get("ctime_not_compared", [])]
    assert "interpreters_not_checked" not in found[0], found[0]
    if case == "ctime-only":
        assert edited == [] and path in noted, (edited, noted)
    elif case == "hard-linked":
        assert edited == [path] and noted == [], (edited, noted)
    else:
        assert path in edited and other in edited and noted == [], (edited, noted)


# K53: the excuse needs the same file: its inode, device and link count as the
# walk before recorded them (a snapshot keeps them). Two root-only files with
# the same fingerprint swapped with renames, or one replaced by `ln -f`, change
# them; an entry with no identity recorded (an older record) is not excused.
# A link the agent user owns is not excused either, its target unchanged.
@pytest.mark.parametrize(
    ("was", "ids", "excused"),
    [
        (_ROOT, ((1, 2, 1), (1, 2, 1)), True),
        (_ROOT, ((1, 2, 1), (7, 2, 1)), False),
        (_ROOT, ((1, 2, 1), (1, 2, 2)), False),
        (_ROOT, ((1, 2, 1), (1, 3, 1)), False),
        (_ROOT, (None, (1, 2, 1)), False),
        (_ROOT, (None, None), False),
        (
            [0o120777, 1000, 1000, 14, 5 * _S, 6 * _S, "/opt/x/pytest1", ""],
            ((1, 2, 1), (1, 2, 1)),
            False,
        ),
    ],
    ids=[
        "same-file",
        "swapped",
        "hard-linked",
        "other-device",
        "no-identity",
        "no-identity-either",
        "agent-owned-link",
    ],
)
async def test_the_excuse_needs_the_same_file(was, ids, excused, monkeypatch):
    path = "/usr/local/bin/pytest"
    now = [*was[:5], 1050 * _S, was[6], "h" if not was[6] else ""]
    record = await _walk_detect(
        "path_tools", path, was, now, [(1000.0, 1100.0)], monkeypatch, ids
    )
    moved = [c["path"] for c in record["changes"] if c["category"] == "snapshot-ctime"]
    if excused:
        assert record["flags"] == [] and moved == [path], record["flags"]
    else:
        assert moved == []
        assert [(f["path"], f["kind"]) for f in record["flags"]] == [(path, "edited")]


# K54: the shell walk keeps the mode as text (its type letter, then the octal
# bits). The owner rule reads it; one it can't read is not root-only, and the
# excuse never raises: the entry is compared as without a snapshot.
@pytest.mark.parametrize(
    ("mode", "uid", "root_only"),
    [
        ("f755", 0, True),
        ("f775", 0, False),
        ("f757", 0, False),
        ("l777", 0, True),
        ("f4755", 0, True),
        ("f644", 1000, False),
        ("x", 0, False),
        ("", 0, False),
        (None, 0, False),
        (0o100755, 0, True),
        (0o100775, 0, False),
        (0o120777, 0, True),
    ],
)
def test_the_owner_rule_reads_either_walks_mode(mode, uid, root_only):
    from benchflow.sandbox import _image_trees as trees

    assert trees.only_root_writes(mode, uid) is root_only


@pytest.mark.parametrize(
    ("section", "path"),
    [
        ("path_tools", "/usr/local/bin/pytest"),
        ("python_trees", "/s/_pytest/main.py"),
        ("interpreters", "/usr/local/bin/python3"),
        ("startup", "/root/.bashrc"),
    ],
)
@pytest.mark.parametrize(
    ("mode", "excused"),
    [("f755", True), ("f775", False), ("x", False)],
    ids=["text-root-only", "text-group-writable", "text-unreadable"],
)
async def test_a_shell_walked_entry_is_judged_never_a_crash(
    section, path, mode, excused, monkeypatch
):
    was = [mode, 0, 0, 100, 5 * _S, 6 * _S, "", ""]
    now = [mode, 0, 0, 100, 5 * _S, 1050 * _S, "", ""]
    record = await _walk_detect(
        section, path, was, now, [(1000.0, 1100.0)], monkeypatch
    )
    moved = [c["path"] for c in record["changes"] if c["category"] == "snapshot-ctime"]
    assert moved == ([path] if excused else [])
    if not excused:
        assert [(f["path"], f["kind"]) for f in record["flags"]] == [(path, "edited")]


# K55: the shell look prints no link's target, so it never excuses a link.
async def test_the_shell_look_never_excuses_a_link():
    from benchflow.sandbox import _image_manifest as im
    from tests.test_image_trees import _manifest

    before = _manifest(
        interpreters={
            "/usr/bin/python3": [0o120777, 0, 0, 9, 5 * _S, 6 * _S, "python3.6", ""]
        },
        ids={"/usr/bin/python3": [1, 2, 1]},
    )
    env = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(
                    return_code=0,
                    stdout="/usr/bin/python3|a1ff|0|0|9|5|1050|1|2|1\n",
                    stderr="",
                )
            ]
        ),
        _benchflow_image_manifest=before,
        _snapshot_windows=[(1000.0, 1100.0)],
    )
    found = await im._walk_failed(env, before, timeout_sec=5)
    assert [f["kind"] for f in found[1:]] == ["edited"]


# K56: the same through the real walks, in a container. An image without a
# Python is walked from the shell: a root-only PATH tool whose ctime alone
# moved (a chmod to its own mode) is a snapshot-ctime change, not a crash. Two
# root-only files with the same fingerprint swapped with renames still flag.
bare = test_image_trees.bare


def test_a_shell_walked_tool_whose_ctime_alone_moved_is_excused(bare):
    from tests.test_image_trees import _detect, _record

    path = "/usr/bin/bash"
    before = _record(bare)
    assert before.trees["walker"] == "shell"
    bare.sh(f"chmod 755 {path}")
    ctime = int(bare.sh(f"stat -c %Z {path}").strip())
    bare._snapshot_windows = [(ctime - 2.0, ctime + 2.0)]
    record = _detect(bare, test_script="bash run.sh\n")
    rows = [r for r in record["flags"] + record["changes"] if r["path"] == path]
    assert [r["category"] for r in rows] == ["snapshot-ctime"], rows


def test_two_root_only_files_swapped_still_flag(box):
    from tests.test_image_trees import SITE, TEST_SH, _detect, _record

    d = f"{SITE}/_pytest"
    a, b = f"{d}/aa.py", f"{d}/bb.py"
    box.sh(
        f"printf 'x = 1\\n' > {a} && printf 'y = 2\\n' > {b} && chmod 644 {a} {b} "
        f"&& touch -d @1600000000 {a} {b}"
    )
    _record(box)
    box.sh(f"mv {a} {d}/t && mv {b} {a} && mv {d}/t {b}")
    ctime = int(box.sh(f"stat -c %Z {a}").strip())
    box._snapshot_windows = [(ctime - 2.0, ctime + 2.0)]
    record = _detect(box, test_script=TEST_SH)
    flagged = {(f["path"], f["kind"], f["category"]) for f in record["flags"]}
    assert {(a, "edited", "pytest-own"), (b, "edited", "pytest-own")} <= flagged


async def test_an_entry_it_cannot_read_is_compared_never_a_crash(monkeypatch):
    """A field the excuse can't read (a ctime that isn't a number) means not
    excused: the entry is compared as without a snapshot."""
    path = "/usr/local/bin/pytest"
    was = [0o100755, 0, 0, 100, 5 * _S, 6 * _S, "", ""]
    now = [0o100755, 0, 0, 100, 5 * _S, "bad", "", ""]
    record = await _walk_detect(
        "path_tools", path, was, now, [(1000.0, 1100.0)], monkeypatch
    )
    assert [(f["path"], f["kind"]) for f in record["flags"]] == [(path, "edited")]


async def test_the_shell_look_needs_the_same_file():
    """In the look too: a root-only interpreter whose inode differs from the
    record's is edited, its ctime inside a window or not."""
    from benchflow.sandbox import _image_manifest as im

    env, before, _ = _walk_env(
        [
            SimpleNamespace(
                return_code=0,
                stdout="/usr/bin/python3|81ed|0|0|10|5|1050|9|2|1\n",
                stderr="",
            )
        ]
    )
    env._snapshot_windows = [(1000.0, 1100.0)]
    found = await im._walk_failed(env, before, timeout_sec=5)
    assert [f["kind"] for f in found[1:]] == ["edited"]
