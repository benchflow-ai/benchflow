"""Guards the fix at 363d815b."""

import os
import signal
import socket
import zipfile

import pytest

from benchflow.publish.traj_capture import (
    attach_workspace_archive,
    stage_trajectory_artifacts,
)


@pytest.mark.parametrize("kind", ["fifo", "socket", "link", "hardlink"])
def test_workspace_archives_only_regular_files(tmp_path, kind):
    """Guards FF-sw-trajectories: special files never block ZIP input."""
    session = tmp_path / "session.jsonl"
    session.write_text('{"type":"message","text":"dummy"}\n')
    folder = tmp_path / "ws"
    folder.mkdir()
    (folder / "normal").write_text("public")
    special = folder / "special"
    server = None
    if kind == "fifo":
        os.mkfifo(special)
    elif kind == "socket":
        server = socket.socket(socket.AF_UNIX)
        server.bind(str(special))
    elif kind == "link":
        special.symlink_to(folder / "normal")
    else:
        outside = tmp_path / "dummy"
        outside.write_text("DUMMY_PRIVATE_MARKER")
        os.link(outside, special)

    def timed_out(*args):
        raise AssertionError("workspace input blocked on a special file")

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(2)
    try:
        with stage_trajectory_artifacts(session, source_id="dummy") as artifacts:
            result = attach_workspace_archive(artifacts, folder)
            assert result.attached is not None
            with zipfile.ZipFile(result.attached.local_path) as archive:
                assert archive.namelist() == ["normal"]
                assert archive.read("normal") == b"public"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        if server:
            server.close()


@pytest.mark.parametrize(
    "replacement", ["leaf-link", "leaf-fifo", "ancestor-link", "growth"]
)
def test_workspace_rechecks_opened_objects_and_actual_byte_limit(
    tmp_path, monkeypatch, replacement
):
    """Guards FF-sw-trajectories against replacements after workspace discovery."""
    from benchflow.publish import traj_capture

    session = tmp_path / "session.jsonl"
    session.write_text('{"type":"message","text":"dummy"}\n')
    folder = tmp_path / "ws"
    (folder / "sub").mkdir(parents=True)
    (folder / "normal").write_text("public")
    entry = folder / "sub" / "entry"
    entry.write_text("short")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "entry").write_text("DUMMY_PRIVATE_MARKER")
    collect = traj_capture._collect_workspace_files

    def changed(*args, **kwargs):
        result = collect(*args, **kwargs)
        if replacement == "ancestor-link":
            entry.parent.rename(folder / "old")
            (folder / "sub").symlink_to(outside, target_is_directory=True)
        elif replacement == "growth":
            entry.write_text("x" * 1000)
        else:
            entry.unlink()
            if replacement == "leaf-link":
                entry.symlink_to(outside / "entry")
            else:
                os.mkfifo(entry)
        return result

    monkeypatch.setattr(traj_capture, "_collect_workspace_files", changed)

    def timed_out(*args):
        raise AssertionError("archive blocked after file replacement")

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(2)
    try:
        with stage_trajectory_artifacts(session, source_id="dummy") as artifacts:
            result = attach_workspace_archive(artifacts, folder, max_bytes=500)
            if replacement == "growth":
                assert result.attached is None
                assert "byte limit" in result.skipped_reason
            else:
                assert result.attached is not None
                with zipfile.ZipFile(result.attached.local_path) as archive:
                    assert archive.namelist() == ["normal"]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
