"""Guards FF-sw-other against log alias writes introduced before 363d815b."""

import json
import os
from pathlib import Path

import pytest

from benchflow.native_harness.client import NativeCLIClient
from tests.reachable_fixture import directory


@pytest.mark.parametrize("method", ["turns", "stream", "log"])
@pytest.mark.parametrize(
    "alias", ["absolute", "relative", "hardlink", "fifo", "parent"]
)
def test_native_logs_never_write_through_agent_alias(tmp_path, method, alias):
    victim = tmp_path / "dummy-marker"
    victim.write_text("DUMMY_MARKER")
    agent = tmp_path / "agent"
    agent.mkdir()
    held = directory(agent)
    path = agent / "native-log"
    if alias == "parent":
        agent.rmdir()
        agent.symlink_to(tmp_path, target_is_directory=True)
        path = agent / victim.name
    elif alias == "hardlink":
        os.link(victim, path)
    elif alias == "fifo":
        os.mkfifo(path)
    else:
        path.symlink_to(victim if alias == "absolute" else Path("..") / victim.name)
    client = NativeCLIClient.__new__(NativeCLIClient)
    client._turns_path = client._stream_path = client._log_path = held / path.name
    client._log_file = None
    client._stream_file = None
    client.turns = [{"turn": 1}]
    try:
        if method == "turns":
            client._write_turns()
        elif method == "stream":
            client._write_stream({"type": "dummy"})
        else:
            client._log("dummy")
    except OSError:
        pass  # Unsafe parents may be refused; they must never be followed.
    finally:
        for attr in ("_log_file", "_stream_file"):
            handle = getattr(client, attr, None)
            if handle:
                handle.close()
    assert victim.read_text() == "DUMMY_MARKER"
    if alias != "parent":
        assert not path.is_symlink()
        assert path.is_file()
        if method == "turns":
            assert json.loads(path.read_text()) == [{"turn": 1}]


@pytest.fixture(autouse=True)
def bounded_alias_repro():
    import signal

    def expired(*args):
        raise TimeoutError("unsafe file open blocked on agent FIFO")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(3)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize(
    "alias", ["absolute", "relative", "hardlink", "parent", "fifo"]
)
def test_native_turn_history_never_reads_aliases(tmp_path, alias):
    """Guards FF-sw-other reconnect reads against the same 363d815b alias class."""
    from benchflow.native_harness.client import _load_turns

    victim = tmp_path / "dummy-history"
    victim.write_text(json.dumps([{"turn": 99, "marker": "DUMMY_PRIVATE_MARKER"}]))
    agent = tmp_path / "agent"
    agent.mkdir()
    held = directory(agent)
    path = agent / "native-turns.json"
    if alias == "parent":
        agent.rmdir()
        agent.symlink_to(tmp_path, target_is_directory=True)
        path = agent / victim.name
    elif alias == "hardlink":
        os.link(victim, path)
    elif alias == "fifo":
        os.mkfifo(path)
    else:
        path.symlink_to(victim if alias == "absolute" else Path("..") / victim.name)
    assert _load_turns(held / path.name) == []
