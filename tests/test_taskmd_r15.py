"""Publication lifecycle findings of review B on 537e2c00c (TASKMD-R15).

B-R14-M1 a cancelled Docker staging open (or cleanup await) is never
joined, P-M1 a second cancellation interrupts the copier's join, B-R14-M2
a directory commit deletes the old output before the rename that can fail.
"""

import asyncio
import contextlib
import errno
import io
import threading
from pathlib import Path

import pytest

from benchflow._utils import reachable_write as rw
from benchflow.sandbox import docker
from benchflow.sandbox._download import private_download
from tests.test_taskmd_r14 import _archive, _old_output, _sandbox, _staging_left

# --- P-M1: repeated cancellation ---------------------------------------------


@pytest.mark.asyncio
async def test_drain_absorbs_every_cancellation_until_the_worker_ends():
    from benchflow.sandbox._download import drain

    release = threading.Event()
    finished = []

    def work():
        assert release.wait(10)
        finished.append(True)
        return "done"

    task = asyncio.create_task(drain(asyncio.to_thread(work)))
    await asyncio.sleep(0.05)
    for _ in range(3):
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done(), "drain returned while its worker still ran"
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished == [True]


@pytest.mark.asyncio
async def test_drain_without_cancellation_keeps_the_result_and_error():
    from benchflow.sandbox._download import drain

    assert await drain(asyncio.to_thread(lambda: 7)) == 7

    def fail():
        raise OSError(errno.ENOSPC, "full")

    with pytest.raises(OSError) as error:
        await drain(asyncio.to_thread(fail))
    assert error.value.errno == errno.ENOSPC


@pytest.mark.asyncio
async def test_double_cancel_returns_after_the_copier(tmp_path, monkeypatch):
    """Guards P-M1 (repeated cancellation), present at 537e2c00c."""
    entered, release, done = threading.Event(), threading.Event(), threading.Event()

    class Transport:
        @private_download(directory=False)
        async def download_file(self, source_path, target_path):
            Path(target_path).write_text("NEW")

    out = tmp_path / "out"
    out.write_text("OLD")
    original = rw._copy_opened

    def copy(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        try:
            return original(*args, **kwargs)
        finally:
            done.set()

    monkeypatch.setattr(rw, "_copy_opened", copy)
    rw.pin_destination(tmp_path)
    task = asyncio.create_task(Transport().download_file("/answer", out))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0.1)
    task.cancel()
    await asyncio.sleep(0.1)
    early = task.done()
    release.set()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(done.wait, 5)
    assert not early, "cancellation returned while the copier still ran"


# --- B-R14-M1: every Docker staging operation is joined ----------------------


@pytest.mark.asyncio
async def test_cancel_during_staging_open_cleans_up(tmp_path, monkeypatch):
    """Guards B-R14-M1 introduced by 537e2c00c: the real opening seam."""
    entered, release = threading.Event(), threading.Event()
    original = rw._open_beneath

    def open_stage(parent, name, *args, **kwargs):
        if str(name).startswith(".benchflow-download-") and not entered.is_set():
            entered.set()
            assert release.wait(10)
        return original(parent, name, *args, **kwargs)

    monkeypatch.setattr(rw, "_open_beneath", open_stage)
    payload = tmp_path / "stream.tar"
    payload.write_bytes(_archive())
    sandbox, _ = _sandbox(monkeypatch, 'cat "$1"', payload)
    out = _old_output(tmp_path)
    held = rw.pin_destination(tmp_path)
    task = asyncio.create_task(sandbox.download_dir("/output", held / "out"))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0.15)
    early = task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not early, "cancellation returned while the opener still ran"
    assert _staging_left(tmp_path) == []
    assert sorted(p.name for p in out.iterdir()) == ["old"]


SEAMS = [
    "enter",
    "extract",
    "wait",
    "stderr",
    "close",
    "commit",
    "exit-ok",
    "exit-error",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("seam", SEAMS)
async def test_repeated_cancel_at_each_await_joins_and_cleans(
    tmp_path, monkeypatch, seam
):
    """Two cancellations at each lifecycle await: nothing returns early or leaks."""
    entered, release = threading.Event(), threading.Event()
    payload = tmp_path / "stream.tar"
    payload.write_bytes(_archive())
    sandbox, spawned = _sandbox(monkeypatch, 'cat "$1"', payload)
    out = _old_output(tmp_path)
    held = rw.pin_destination(tmp_path)
    stages = []

    def pause(at):
        if at == seam:
            entered.set()
            assert release.wait(10)

    class Stage(rw.StagedTar):
        def __enter__(self):
            stages.append(self)
            pause("enter")
            return super().__enter__()

        def extract(self, reader):
            pause("extract")
            if seam == "exit-error":
                raise OSError(errno.ENOSPC, "injected extraction failure")
            return super().extract(reader)

        def commit(self):
            pause("commit")
            return super().commit()

        def __exit__(self, *args):
            pause("exit-ok" if args[0] is None else "exit-error")
            return super().__exit__(*args)

    monkeypatch.setattr(docker, "StagedTar", Stage)
    spawn = docker.asyncio.create_subprocess_exec

    async def gated(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        if seam == "wait":
            wait = process.wait

            async def paused_wait():
                entered.set()
                while not release.is_set():
                    await asyncio.sleep(0.01)
                return await wait()

            process.wait = paused_wait
        if seam == "stderr":
            read = process.stderr.read

            async def paused_read(*a):
                data = await read(*a)
                entered.set()
                while not release.is_set():
                    await asyncio.sleep(0.01)
                return data

            process.stderr.read = paused_read
        return process

    monkeypatch.setattr(docker.asyncio, "create_subprocess_exec", gated)
    if seam == "close":
        fdopen = docker.os.fdopen

        class Reader:
            def __init__(self, inner):
                self.inner = inner

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def close(self):
                pause("close")
                return self.inner.close()

        def open_reader(fd, *args, **kwargs):
            inner = fdopen(fd, *args, **kwargs)
            on_loop = threading.current_thread() is threading.main_thread()
            return Reader(inner) if args == ("rb",) and on_loop else inner

        monkeypatch.setattr(docker.os, "fdopen", open_reader)
    task = asyncio.create_task(
        sandbox._stream_download("/output", held / "out", directory=True)
    )
    assert await asyncio.to_thread(entered.wait, 5), seam
    task.cancel()
    await asyncio.sleep(0.1)
    task.cancel()
    await asyncio.sleep(0.1)
    early = task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not early, f"returned while the {seam} operation still ran"
    assert _staging_left(tmp_path) == []
    assert all(stage._work_fd is None for stage in stages)
    assert spawned[0].returncode is not None
    # The commit is the irrevocable point: joined, it publishes the new tree.
    expected = ["first"] if seam in ("commit", "exit-ok") else ["old"]
    assert sorted(p.name for p in out.iterdir()) == expected


# --- B-R14-M2: transactional directory commit --------------------------------


PREVIOUS = getattr(rw, "_PREVIOUS", "previous")


def _fail_rename(monkeypatch, *, sources):
    rename = rw.os.rename

    def fail(src, dst, *args, **kwargs):
        if src in sources:
            raise OSError(errno.ENOSPC, f"injected ENOSPC renaming {src}")
        return rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(rw.os, "rename", fail)
    # TASKMD-R16: the rename-back is a no-replace rename, not os.rename.
    monkeypatch.setattr(rw, "_rename_noreplace", fail, raising=False)


@pytest.mark.parametrize("directory", [True, False], ids=["folder", "file"])
def test_commit_enospc_keeps_existing_destination(tmp_path, monkeypatch, directory):
    """Guards B-R14-M2, present since round 13: errno 28 at the final rename."""
    out = tmp_path / "out"
    if directory:
        _old_output(tmp_path)
    else:
        out.write_text("OLD")
    held = rw.pin_destination(tmp_path)
    _fail_rename(monkeypatch, sources={"payload"})
    with pytest.raises(OSError) as error:
        rw.publish_tar_stream(
            io.BytesIO(_archive(directory=directory)), held / "out", directory=directory
        )
    assert error.value.errno == errno.ENOSPC
    assert (out / "old" if directory else out).read_text() == "OLD"
    if directory:
        assert sorted(p.name for p in out.iterdir()) == ["old"]
    assert _staging_left(tmp_path) == []


def test_failed_restore_keeps_old_folder_in_staging(tmp_path, monkeypatch, caplog):
    """If the old folder cannot be put back, staging is its only copy: keep it."""
    _old_output(tmp_path)
    held = rw.pin_destination(tmp_path)
    _fail_rename(monkeypatch, sources={"payload", PREVIOUS})
    with pytest.raises(OSError) as error:
        rw.publish_tar_stream(io.BytesIO(_archive()), held / "out", directory=True)
    assert error.value.errno == errno.ENOSPC
    assert "payload" in str(error.value)
    assert any("previous" in note for note in error.value.__notes__)
    (kept,) = _staging_left(tmp_path)
    assert (tmp_path / kept / PREVIOUS / "old").read_text() == "OLD"
    assert kept in caplog.text


def test_commit_over_existing_folder_replaces_it(tmp_path):
    """Control: a successful commit installs the payload and removes the old."""
    out = _old_output(tmp_path)
    held = rw.pin_destination(tmp_path)
    rw.publish_tar_stream(io.BytesIO(_archive()), held / "out", directory=True)
    assert sorted(p.name for p in out.iterdir()) == ["first"]
    assert (out / "first").read_text() == "NEW"
    assert _staging_left(tmp_path) == []
