"""Guards CONFIRM-taskmd-rewards-R5-1 (c61088c6) and its writer siblings."""

import os
from unittest.mock import AsyncMock

import pytest

from benchflow.acp.container_transport import ContainerTransport
from benchflow.task.paths import RolloutPaths


@pytest.mark.asyncio
@pytest.mark.parametrize("site", ["receive", "shutdown"])
@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("prior_output", [False, True])
async def test_acp_log_never_opens_a_sandbox_link(
    tmp_path, site, relative, prior_output
):
    """Guards the two ACP writer bypasses still present at c61088c6 (R5-1)."""
    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    victim = tmp_path / "controller-marker"
    victim.write_text("DUMMY_CONTROLLER_MARKER")
    entry = tmp_path / "trial" / "agent" / "deepagents.txt"
    process = AsyncMock()
    process.stderr_tail = "completion metrics\n" if site == "shutdown" else ""
    process.readline.side_effect = [
        b"first diagnostic\n",
        b'{"jsonrpc":"2.0","id":1,"result":{}}\n',
        b"late diagnostic\n",
        b'{"jsonrpc":"2.0","id":2,"result":{}}\n',
    ]
    transport = ContainerTransport(
        process, "deepagents", agent_log_path=paths.agent_dir / entry.name
    )
    await transport.start()
    if prior_output:
        await transport.receive()
    entry.unlink(missing_ok=True)
    entry.symlink_to(os.path.relpath(victim, entry.parent) if relative else victim)
    try:
        if site == "receive":
            await transport.receive()
    finally:
        await transport.close()
    assert victim.read_text() == "DUMMY_CONTROLLER_MARKER"
    assert not entry.is_symlink()
    expected = "completion metrics" if site == "shutdown" else "diagnostic"
    assert expected in entry.read_text()
    if prior_output:
        assert "first diagnostic" in entry.read_text()


@pytest.mark.parametrize("imported", [False, True])
@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize(
    "writer",
    [
        'self.destination.open("w")',
        'open(self.destination, "a")',
        'self.destination.write_text("x")',
        'self.destination.write_bytes(b"x")',
        'shutil.copyfile("private", self.destination)',
        'shutil.copy("private", self.destination)',
        'shutil.copy2("private", self.destination)',
        'shutil.copytree("private", self.destination)',
        'os.replace("private", self.destination)',
    ],
)
def test_scan_tracks_constructor_propagated_destinations(
    tmp_path, monkeypatch, imported, keyword, writer
):
    """Guards c61088c6: a neutral constructor parameter must not hide a writer."""
    from tests import test_reachable_write_scan as scan

    consumer = f"""import os, shutil
class Writer:
    def __init__(self, destination):
        self.destination = destination
    def finish(self):
        {writer}
"""
    argument = "destination=" if keyword else ""
    caller = f"""def run(rollout_dir):
    name = rollout_dir / "agent" / "deepagents.txt"
    Writer({argument}name)
"""
    if imported:
        (tmp_path / "consumer.py").write_text(consumer)
        caller = "from benchflow.consumer import Writer\n" + caller
    else:
        caller = consumer + caller
    (tmp_path / "caller.py").write_text(caller)
    monkeypatch.setattr(scan, "SRC", tmp_path)
    monkeypatch.setattr(scan, "HELPER", tmp_path / "_utils" / "reachable_write.py")
    sites, _ = scan._scan()
    assert len(sites) == 1
    assert sites[0][1] == "Writer.finish"


def test_reachable_directories_cannot_escape_as_filesystem_paths(tmp_path):
    """Guards 961fddc9: constructors cannot unwrap a held capability into a path."""
    from pathlib import Path

    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    for directory in (paths.agent_dir, paths.verifier_dir, paths.artifacts_dir):
        assert directory is getattr(paths, directory.name + "_dir")
        for entry in (directory, directory / "nested" / "log.txt"):
            with pytest.raises(TypeError):
                os.fspath(entry)
            with pytest.raises(TypeError):
                Path(entry)
            with pytest.raises(TypeError), open(entry, "w"):
                pass
            assert str(tmp_path) not in str(entry)


@pytest.mark.parametrize(
    "name",
    [
        "deepagents.txt",
        "install-stdout.txt",
        "install-native-stdout.txt",
        "acp_trajectory.jsonl",
        "output.txt",
        "native-turns.json",
        "claude.jsonl",
        "reward.txt",
        "reward.json",
        "reward-details.json",
        "criteria-runner-manifest.json",
        "test-stdout.txt",
        "test-stderr.txt",
        "judgments.jsonl",
        "review.json",
        "ctrf.json",
        "taskmd-judge",
        "claude-sessions",
        "declared/artifact.txt",
        ".benchflow-mount-probe-dummy",
        "guard-marker",
    ],
)
@pytest.mark.parametrize("root", ["agent", "verifier", "artifacts"])
@pytest.mark.parametrize("link", ["absolute", "relative", "hard", "fifo"])
def test_all_writer_names_use_held_publication(tmp_path, name, root, link):
    """Guards 961fddc9 across writer names and all backend-neutral host roots."""
    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    held = getattr(paths, root + "_dir") / name
    disk = tmp_path / "trial" / root / name
    disk.parent.mkdir(parents=True, exist_ok=True)
    victim = tmp_path / "dummy-marker"
    victim.write_text("DUMMY")
    if link == "hard":
        os.link(victim, disk)
    elif link == "fifo":
        os.mkfifo(disk)
    else:
        disk.symlink_to(
            os.path.relpath(victim, disk.parent) if link == "relative" else victim
        )
    held.write("published")
    assert victim.read_text() == "DUMMY"
    assert disk.read_text() == "published"
    assert not disk.is_symlink()


@pytest.mark.parametrize("operation", ["write", "append", "publish", "log"])
def test_opened_root_is_not_reopened_after_swap(tmp_path, operation):
    """Guards 961fddc9: a retained capability must refuse a replacement root."""
    from benchflow._utils.reachable_write import ReachableWriteError

    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    held = paths.agent_dir / "deepagents.txt"
    disk = tmp_path / "trial" / "agent"
    disk.rename(disk.with_name("old-agent"))
    disk.mkdir()
    (disk / "deepagents.txt").write_text("REPLACEMENT")
    with pytest.raises(ReachableWriteError):
        if operation == "write":
            held.write("bad")
        elif operation == "append":
            held.append("bad")
        elif operation == "publish":
            with held.private_folder() as private:
                (private / "x").write_text("bad")
                held.publish(private / "x")
        else:
            with held.log() as log:
                log.write("bad")
    assert (disk / "deepagents.txt").read_text() == "REPLACEMENT"


@pytest.mark.parametrize("site", ["receive", "shutdown"])
@pytest.mark.parametrize("kind", ["hardlink", "fifo", "directory"])
async def test_acp_log_special_entries_do_not_change_private_stream(
    tmp_path, site, kind
):
    """Guards 961fddc9's special-entry siblings at both ACP logging sites."""
    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    victim = tmp_path / "dummy-marker"
    victim.write_text("DUMMY")
    disk = tmp_path / "trial/agent/deepagents.txt"
    cp = AsyncMock()
    cp.stderr_tail = "metrics" if site == "shutdown" else ""
    cp.readline.side_effect = [b"log\n", b'{"jsonrpc":"2.0","id":1,"result":{}}\n']
    transport = ContainerTransport(
        cp, "agent", agent_log_path=paths.agent_dir / disk.name
    )
    await transport.start()
    if kind == "hardlink":
        os.link(victim, disk)
    elif kind == "fifo":
        os.mkfifo(disk)
    else:
        disk.mkdir()
        (disk / "trap").symlink_to(victim)
    if site == "receive":
        await transport.receive()
    await transport.close()
    assert victim.read_text() == "DUMMY"
    assert disk.read_text() == ("metrics" if site == "shutdown" else "log\n")


def test_missing_read_only_roots_are_not_created(tmp_path):
    """Guards 961fddc9's handle migration against mutating an inspected job."""
    paths = RolloutPaths(tmp_path / "missing")
    assert not paths.agent_dir.exists()
    assert not paths.verifier_dir.is_dir()
    assert not paths.artifacts_dir.is_file()
    assert not list(tmp_path.iterdir())


def test_one_capability_keeps_one_root_handle(tmp_path, monkeypatch):
    """Guards 961fddc9: repeated writes reuse the opened root for descendants."""
    from benchflow._utils import reachable_write as rw

    root = tmp_path / "trial/agent"
    opened = []
    original = rw._open_dir

    def observe(name, dir_fd=None):
        if name == root:
            opened.append(name)
        return original(name, dir_fd)

    monkeypatch.setattr(rw, "_open_dir", observe)
    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    paths.agent_dir.__truediv__("one").write("one")
    (paths.agent_dir / "nested/two").write("two")
    (paths.agent_dir / "one").append("three")
    assert opened == [root]


def test_windows_publication_pins_root_and_parents(tmp_path, monkeypatch):
    """Guards 961fddc9: Windows renames run only while ancestors are held.

    Uses the existing Windows no-reparse API's interface; native Windows
    syscall behavior remains covered by its platform-specific tests.
    """
    from contextlib import contextmanager
    from pathlib import Path
    from types import SimpleNamespace

    from benchflow._utils import reachable_write as rw
    from benchflow.rewards import _safe_evidence

    root = tmp_path / "trial/agent"
    root.mkdir(parents=True)
    active = {1: root}
    opened = []

    class API:
        def open(self, part, *, directory, parent):
            assert directory and parent in active
            path = active[parent] / part
            assert not path.is_symlink()
            handle = len(opened) + 2
            opened.append(path)
            active[handle] = path
            return handle

        def close(self, handle):
            del active[handle]

    @contextmanager
    def windows_root(path, *, evidence_root):
        assert path == evidence_root == root
        yield SimpleNamespace(api=API(), handle=1)

    monkeypatch.setattr(rw, "os", SimpleNamespace(**(vars(os) | {"name": "nt"})))
    monkeypatch.setattr(rw, "_held_handles_available", lambda: False)
    monkeypatch.setattr(_safe_evidence, "open_directory", windows_root)
    original = os.replace
    moved = []

    def rename(source, destination):
        destination = Path(destination)
        assert destination.parent in active.values()
        moved.append(destination)
        return original(source, destination)

    monkeypatch.setattr(rw.os, "replace", rename)
    held = rw.ReachablePath.directory(root)
    victim = tmp_path / "dummy-marker"
    victim.write_text("DUMMY")
    (root / "nested").mkdir()
    (root / "nested/log").symlink_to(victim)
    (held / "nested/log").write("kept")
    assert moved == [root / "nested/log"]
    assert active == {1: root}
    assert opened == [root / "nested"]
    assert victim.read_text() == "DUMMY"
    assert (root / "nested/log").read_text() == "kept"


@pytest.mark.parametrize(
    "name", ["..", "../marker", "a/../../marker", "/tmp/dummy-marker"]
)
def test_capability_names_cannot_leave_the_root(tmp_path, name):
    """Guards 961fddc9 against absolute names and every traversal position."""
    paths = RolloutPaths(tmp_path / "trial")
    with pytest.raises(ValueError):
        paths.agent_dir / name


@pytest.mark.parametrize(
    "name", ["NUL", "con.txt", "a:stream", "tail.", "tail ", "aux", "COM1.log"]
)
def test_windows_name_aliases_are_rejected_by_the_handle(tmp_path, monkeypatch, name):
    """Guards 961fddc9's Windows capability boundary before any file operation."""
    paths = RolloutPaths(tmp_path / "trial")
    held = paths.agent_dir
    monkeypatch.setattr(held._anchor, "windows", object())
    with pytest.raises(ValueError):
        held / name


def test_declared_judge_inputs_preserve_capabilities(tmp_path):
    """Guards 961fddc9 against a read-side resolver leaking raw writer targets."""
    from benchflow.task.verifier_judge_inputs import _local_rollout_input_path

    for root in ("agent", "verifier", "artifacts"):
        for spelling in (f"{root}/output.txt", f"/logs/{root}/output.txt"):
            entry = _local_rollout_input_path(spelling, rollout_dir=tmp_path)
            with pytest.raises(TypeError):
                os.fspath(entry)
            with pytest.raises(ValueError):
                entry.open("w")


def test_private_evidence_copy_preserves_directory_and_file_metadata(tmp_path):
    """Guards 78f87c0e against the mode/mtime copy regression in 961fddc9."""
    from benchflow._utils.reachable_write import ReachablePath

    raw = tmp_path / "trial/verifier"
    folder = raw / "records"
    folder.mkdir(parents=True)
    file = folder / "executable"
    file.write_text("record")
    file.chmod(0o751)
    folder.chmod(0o750)
    moment = 1_234_567_890_123_456_789
    for path in (file, folder):
        os.utime(path, ns=(moment, moment))
    held = ReachablePath.directory(raw)
    private = tmp_path / "private-copy"
    held.copy_to(private)
    for original in (folder, file):
        copied = private / original.relative_to(raw)
        assert copied.stat().st_mode == original.stat().st_mode
        assert copied.stat().st_mtime_ns == original.stat().st_mtime_ns
    assert (private / "records/executable").read_text() == "record"


@pytest.mark.parametrize("judge", ["agent", "ors"])
@pytest.mark.parametrize("root", ["agent", "verifier", "artifacts"])
@pytest.mark.parametrize("ancestor", ["", "/output"])
@pytest.mark.parametrize("prefix", ["", "/logs/"])
async def test_unsafe_input_ancestors_keep_the_judge_error_contract(
    tmp_path, judge, root, ancestor, prefix
):
    """Guards 961fddc9's migration against leaking handle errors to trial code."""
    from benchflow.task import Verifier
    from benchflow.task.verifier_errors import (
        AgentJudgeInputError,
        ORSEpisodeInputError,
    )

    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    marker = tmp_path / "dummy-controller"
    marker.mkdir()
    (marker / "data").write_text("DUMMY")
    link = tmp_path / "trial" / (root + ancestor)
    if link.exists():
        link.rmdir()
    link.symlink_to(marker, target_is_directory=True)
    verifier = Verifier.__new__(Verifier)
    verifier._rollout_paths = paths
    verifier._sandbox = AsyncMock()
    resolve = (
        verifier._resolve_agent_judge_input
        if judge == "agent"
        else verifier._resolve_ors_episode_input
    )
    error = AgentJudgeInputError if judge == "agent" else ORSEpisodeInputError
    with pytest.raises(error, match="unsafe local path"):
        await resolve(
            prefix + root + "/output/data", inputs_dir=tmp_path / "private", index=0
        )
    assert not verifier._sandbox.mock_calls
    assert (marker / "data").read_text() == "DUMMY"
