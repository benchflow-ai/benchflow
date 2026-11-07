"""Document-copy regression at 49de99935a.

Original control: 616001d88. Keep all earlier acceptance inputs unchanged.
Exercise actual parent I/O through admission, copying and saved judge results.
"""

import contextlib
import errno
import json
import os
import stat
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow._utils import reachable_write as rw
from benchflow.rewards import file_readers
from tests._os_capabilities import patch_os
from tests.test_judge_reference_storage_provenance import (
    catch_verification,
    has_cause,
    judge_environment,
)
from tests.test_r5_release_contract import observe
from tests.test_verifier_own_failures import _assess


def fd_path(fd):
    return Path(os.readlink(f"/proc/self/fd/{fd}"))


class StreamProxy:
    def __init__(self, raw):
        self.raw = raw

    def __getattr__(self, name):
        return getattr(self.raw, name)


COPY_CASES = [
    pytest.param("identity-stat", errno.EACCES, False, id="post-copy-stat-eacces"),
    pytest.param("parent-open", errno.EACCES, False, id="parent-open-eacces"),
    pytest.param("parent-close", errno.EIO, False, id="post-copy-close-eio"),
    pytest.param("identity-stat", errno.EACCES, True, id="second-document-stat"),
]


@pytest.mark.parametrize("prior", [False, True], ids=["no-prior", "prior"])
@pytest.mark.parametrize("operation,number,multiple", COPY_CASES)
async def test_document_copy_private_parent_failure_is_saved_unscored(
    tmp_path, monkeypatch, prior, operation, number, multiple
):
    sandbox, verifier, calls = await judge_environment(
        tmp_path, monkeypatch, "llm", prior
    )
    expected = b"the answer is 42\n"
    companion = "the earlier document survived"
    if multiple:
        (sandbox.root / "app/a-first.txt").write_text(companion)
    errors = catch_verification(monkeypatch, verifier)
    fault = OSError(number, f"fixture document private {operation} failed")
    state = {"reads": b"", "writes": b"", "flushed": False, "payload_done": False}
    reached, gathered, prompts = [], {}, []
    real_copy, real_create = rw._snapshot_stream, rw.create_private_file
    real_stat, real_open, real_close = os.stat, os.open, os.close
    real_openat2, real_path_open = rw._openat2, Path.open
    real_reader = file_readers.read_file_as_text

    def copy(source, target, **kwargs):
        if not (
            isinstance(target, Path)
            and target.name == "answer.txt"
            and target.parent.name.startswith("benchflow-deliverable-")
        ):
            return real_copy(source, target, **kwargs)
        info = os.fstat(source.fileno())
        source_path = fd_path(source.fileno())
        assert stat.S_ISREG(info.st_mode) and info.st_size == len(expected)
        assert source_path.name == "answer.txt" and source_path.parent.name == "app"
        assert source_path.parent.parent.name.startswith("benchflow-native-judge-")
        sandbox.download_file.assert_awaited_once()
        state.update(target=target, source=source_path, source_fd=source.fileno())

        class Source(StreamProxy):
            def read(self, *args):
                data = self.raw.read(*args)
                state["reads"] += data
                return data

        state["copying"] = True
        try:
            return real_copy(Source(source), target, **kwargs)
        except OSError as exc:
            state["copy_error"] = exc
            raise
        finally:
            state["copying"] = False

    @contextlib.contextmanager
    def create(target):
        with real_create(target) as raw:
            if target != state.get("target"):
                yield raw
                return
            assert fd_path(raw.fileno()) == target
            state["output"] = raw

            class Output(StreamProxy):
                def write(self, data):
                    count = self.raw.write(data)
                    state["writes"] += data[:count]
                    return count

                def flush(self):
                    self.raw.flush()
                    with real_path_open(target, "rb") as copied:
                        assert copied.read() == expected
                    state["flushed"] = True

            yield Output(raw)
            # The actual stream/held-parent context has not exited yet.
            state["payload_done"] = True

    def selected(parent, path):
        return (
            state.get("copying")
            and not reached
            and parent is not None
            and fd_path(parent) / path == state["target"].parent
        )

    def hit(parent, path, *, closed=False):
        assert bool(getattr(sandbox, "_benchflow_test_command_issued", False)) is prior
        assert not calls and not prompts
        if operation == "parent-open":
            assert not state["reads"] and not state["writes"]
            assert not state["flushed"] and not state["payload_done"]
        else:
            assert state["reads"] == state["writes"] == expected
            assert state["flushed"] and state["payload_done"]
            assert state["output"].closed
        reached.append(
            {
                "operation": operation,
                "fd": parent,
                "path": path,
                "closed_before_fault": closed,
            }
        )
        raise fault

    def inspect(path, *args, **kwargs):
        parent = kwargs.get("dir_fd")
        if (
            operation == "identity-stat"
            and state["payload_done"]
            and selected(parent, path)
        ):
            assert kwargs["follow_symlinks"] is False
            hit(parent, fd_path(parent) / path)
        return real_stat(path, *args, **kwargs)

    def openat2(parent, path, flags, mode):
        if operation == "parent-open" and selected(parent, path):
            # Reach the actual no-follow fallback os.open in all source arms.
            return None
        return real_openat2(parent, path, flags, mode)

    def opened(path, flags, *args, **kwargs):
        parent = kwargs.get("dir_fd")
        if operation == "parent-open" and selected(parent, path):
            assert flags & os.O_DIRECTORY and flags & os.O_NOFOLLOW
            hit(parent, fd_path(parent) / path)
        return real_open(path, flags, *args, **kwargs)

    def closed(fd):
        if (
            operation == "parent-close"
            and state.get("copying")
            and state["payload_done"]
            and not reached
            and fd_path(fd) == state["target"].parent
        ):
            where = fd_path(fd)
            real_close(fd)
            with pytest.raises(OSError) as closed_error:
                os.fstat(fd)
            assert closed_error.value.errno == errno.EBADF
            hit(fd, where, closed=True)
        return real_close(fd)

    def read(path, **kwargs):
        result = real_reader(path, **kwargs)
        gathered[path.name] = result
        return result

    async def provider(model, prompt, **kwargs):
        calls.append("document")
        prompts.append(prompt)
        return '{"verdict":"pass"}'

    monkeypatch.setattr(rw, "_snapshot_stream", copy)
    monkeypatch.setattr(rw, "create_private_file", create)
    monkeypatch.setattr(rw, "_openat2", openat2)
    patch_os(monkeypatch, "stat", inspect)
    patch_os(monkeypatch, "open", opened)
    patch_os(monkeypatch, "close", closed)
    monkeypatch.setattr(file_readers, "read_file_as_text", read)
    provider_mock = AsyncMock(side_effect=provider)
    monkeypatch.setattr("benchflow.rewards.llm.call_judge", provider_mock)
    rewards, error, warnings = await _assess(sandbox, verifier)

    # Retain the actual bad-source outcome before any acceptance assertion.
    (tmp_path / "operation.json").write_text(
        json.dumps(
            {
                "reached": reached,
                "source": state.get("source"),
                "source_fd": state.get("source_fd"),
                "target": state.get("target"),
                "read_bytes": len(state["reads"]),
                "written_bytes": len(state["writes"]),
                "flushed": state["flushed"],
                "payload_done": state["payload_done"],
                "gathered": gathered,
                "prompts": prompts,
                "calls": calls,
            },
            default=str,
            indent=2,
        )
    )
    observe(tmp_path, rewards, error, "unscored", "private")
    assert len(reached) == 1 and has_cause(state["copy_error"], fault)
    assert errors == [state["copy_error"]]
    assert isinstance(errors[0], rw.PrivateStorageError)
    assert errors[0].errno == number
    assert str(fault) in error and "provider failed" not in error
    assert not calls and not prompts
    provider_mock.assert_not_awaited()
    assert gathered == ({"a-first.txt": companion} if multiple else {})
    assert not verifier._rollout_paths.reward_json_path.exists()
    sandbox.upload_file.assert_not_awaited()
    assert not warnings
