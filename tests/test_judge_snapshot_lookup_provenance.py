"""Operation regressions against frozen 616001d88.

Keep the earlier acceptance modules byte-identical. These fixtures retain real
admission, copying, judge consumers and result serialization; faults target
individual private operations after admission, before provider invocation.
"""

import errno
import json
import os
import stat
import tarfile
import tomllib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow._utils import reachable_write as rw
from benchflow.rewards import file_readers, rubric_config
from benchflow.rewards.builtins import JudgeScoringError
from benchflow.taskmd import verify as taskmd
from tests._os_capabilities import patch_os
from tests.test_judge_reference_storage_provenance import (
    catch_verification,
    has_cause,
    judge_environment,
)
from tests.test_r5_release_contract import observe
from tests.test_verifier_own_failures import _assess


def causes(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__


def fd_path(fd):
    return Path(os.readlink(f"/proc/self/fd/{fd}"))


class StreamProxy:
    def __init__(self, raw):
        self.raw = raw

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.raw.close()


def trace_capture(monkeypatch, sandbox):
    state = {"archive_bytes": 0, "parents": [], "leaf_probe": False}
    original_open = rw.ReachablePath.open
    original_folders = taskmd._real_folders

    def opened(target, *args, **kwargs):
        raw = original_open(target, *args, **kwargs)
        if target.name != "copy.tar":
            return raw

        class Archive(StreamProxy):
            def read(self, *args):
                data = self.raw.read(*args)
                state["archive_bytes"] += len(data)
                return data

        return Archive(raw)

    def folders(root, parts, path):
        state["preparing"] = True
        try:
            result = original_folders(root, parts, path)
        finally:
            state["preparing"] = False
        # This call follows complete tar inventory, including member admission.
        assert state["archive_bytes"] > 0
        sandbox.download_file.assert_awaited_once()
        state["parents"].append(result)
        if result == root / "app":
            state["app"] = result
            state["anchor"] = rw.destination(result / "main.py")._anchor.path
        return result

    monkeypatch.setattr(rw.ReachablePath, "open", opened)
    monkeypatch.setattr(taskmd, "_real_folders", folders)
    return state


LOOKUP_CASES = [
    ("regular-stat", errno.EACCES),
    ("hardlink-pre-stat", errno.EACCES),
    ("hardlink-post-stat", errno.EACCES),
    ("file-parent-open", errno.EACCES),
    ("folder-open", errno.EACCES),
    ("root-dup", errno.EACCES),
    ("folder-identity-stat", errno.EACCES),
    ("root-identity-stat", errno.EACCES),
    ("held-folder-fstat", errno.EACCES),
    ("post-write-identity-stat", errno.EACCES),
    ("folder-close", errno.EACCES),
    ("regular-stat", errno.EIO),
    ("folder-open", errno.ENOSPC),
]


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("operation,number", LOOKUP_CASES)
async def test_private_capture_lookup_is_saved_unscored(
    tmp_path, monkeypatch, prior, operation, number
):
    sandbox, verifier, calls = await judge_environment(
        tmp_path, monkeypatch, "native", prior
    )
    if operation.startswith("hardlink"):
        os.link(sandbox.root / "app/main.py", sandbox.root / "app/second.py")
    errors = catch_verification(monkeypatch, verifier)
    state = trace_capture(monkeypatch, sandbox)
    fault = OSError(number, f"fixture private {operation} failed")
    reached = []
    real_stat, real_open, real_dup = os.stat, os.open, os.dup
    real_fstat = os.fstat
    real_chmod, real_link, real_close = os.fchmod, os.link, os.close
    real_openat2 = rw._openat2

    def hit(detail):
        assert state["archive_bytes"] > 0 and state["parents"]
        assert not state["preparing"]
        assert bool(getattr(sandbox, "_benchflow_test_command_issued", False)) is prior
        reached.append((operation, detail))
        raise fault

    def inspect(path, *args, **kwargs):
        parent = kwargs.get("dir_fd")
        if (
            operation == "root-identity-stat"
            and state.get("app")
            and not state["preparing"]
            and not reached
            and path == state["anchor"]
            and kwargs.get("follow_symlinks") is False
        ):
            hit(path)
        if (
            state.get("app")
            and not state["preparing"]
            and parent is not None
            and not reached
        ):
            where = fd_path(parent)
            if where == state["app"] and path in ("main.py", "second.py"):
                assert kwargs["follow_symlinks"] is False
                state["leaf_probe"] = True
                if operation == "regular-stat" and path == "main.py":
                    hit((where, path))
                if path == "second.py" and (
                    (operation == "hardlink-pre-stat" and not state.get("linked"))
                    or (operation == "hardlink-post-stat" and state.get("linked"))
                ):
                    assert (state["app"] / "main.py").read_bytes() == b"print(42)\n"
                    hit((where, path))
            if (
                where == state["app"].parent
                and path == "app"
                and (
                    operation == "folder-identity-stat"
                    or (
                        operation == "post-write-identity-stat" and state.get("written")
                    )
                )
            ):
                assert kwargs["follow_symlinks"] is False
                hit((where, path))
        return real_stat(path, *args, **kwargs)

    def selected_open(parent, path):
        return (
            state.get("app")
            and not state["preparing"]
            and not reached
            and parent is not None
            and path == "app"
            and fd_path(parent) == state["app"].parent
            and (
                operation == "file-parent-open"
                or (operation == "folder-open" and state.get("written"))
            )
        )

    def openat2(parent, path, flags, mode):
        if selected_open(parent, path):
            # Exercise the real no-follow fallback open, in both source arms.
            return None
        return real_openat2(parent, path, flags, mode)

    def opened(path, flags, *args, **kwargs):
        parent = kwargs.get("dir_fd")
        if selected_open(parent, path):
            assert flags & os.O_NOFOLLOW and flags & os.O_DIRECTORY
            hit((fd_path(parent), path))
        return real_open(path, flags, *args, **kwargs)

    def fstat(fd):
        if (
            operation == "held-folder-fstat"
            and state.get("app")
            and not state["preparing"]
            and not reached
            and fd_path(fd) == state["app"]
        ):
            hit(state["app"])
        return real_fstat(fd)

    def duplicate(fd):
        if (
            operation == "root-dup"
            and state.get("app")
            and not state["preparing"]
            and not reached
            and fd_path(fd) == state["anchor"]
        ):
            hit(fd_path(fd))
        return real_dup(fd)

    def chmod(fd, mode):
        result = real_chmod(fd, mode)
        if state.get("app") and fd_path(fd) == state["app"] / "main.py":
            state["written"] = True
        return result

    def link(source, target, *args, **kwargs):
        result = real_link(source, target, *args, **kwargs)
        if target == "second.py" and "dst_dir_fd" in kwargs:
            assert fd_path(kwargs["dst_dir_fd"]) == state["app"]
            state["linked"] = True
        return result

    def close(fd):
        selected = (
            operation == "folder-close"
            and state["leaf_probe"]
            and not reached
            and fd_path(fd) == state["app"]
        )
        result = real_close(fd)
        if selected:
            hit(state["app"])
        return result

    patch_os(monkeypatch, "stat", inspect)
    monkeypatch.setattr(rw, "_openat2", openat2)
    patch_os(monkeypatch, "open", opened)
    patch_os(monkeypatch, "fstat", fstat)
    patch_os(monkeypatch, "dup", duplicate)
    patch_os(monkeypatch, "fchmod", chmod)
    patch_os(monkeypatch, "link", link)
    patch_os(monkeypatch, "close", close)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert len(reached) == 1
    observe(tmp_path, rewards, error, "unscored", "PrivateStorageError")
    assert not calls
    assert len(errors) == 1 and has_cause(errors[0], fault)
    assert any(isinstance(item, rw.PrivateStorageError) for item in causes(errors[0]))
    sandbox.download_file.assert_awaited_once()
    sandbox.upload_file.assert_not_awaited()
    assert operation in error and not warnings


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize(
    "case", ["unsafe-replacement", "custody-EACCES", "custody-EPERM"]
)
async def test_capture_destination_refusal_still_saves_zero(
    tmp_path, monkeypatch, prior, case
):
    sandbox, verifier, calls = await judge_environment(
        tmp_path, monkeypatch, "native", prior
    )
    errors = catch_verification(monkeypatch, verifier)
    state = trace_capture(monkeypatch, sandbox)
    reached = []
    original_check, original_stat = rw._check_in_place, os.stat
    refusal = rw.ReachableWriteError(
        errno.EACCES if case.endswith("EACCES") else errno.EPERM,
        "fixture held private folder custody refused",
    )

    def checked(root, parts, handles, **kwargs):
        original_check(root, parts, handles, **kwargs)
        if (
            case.startswith("custody")
            and state.get("app")
            and not state["preparing"]
            and not reached
            and fd_path(handles[-1]) == state["app"]
        ):
            reached.append(case)
            raise refusal

    def inspect(path, *args, **kwargs):
        parent = kwargs.get("dir_fd")
        if (
            case == "unsafe-replacement"
            and state.get("app")
            and not state["preparing"]
            and not reached
            and path == "main.py"
            and parent is not None
            and fd_path(parent) == state["app"]
        ):
            assert kwargs["follow_symlinks"] is False
            os.mkdir("main.py", dir_fd=parent)
            reached.append(case)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(rw, "_check_in_place", checked)
    patch_os(monkeypatch, "stat", inspect)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert reached == [case] and state["archive_bytes"] > 0
    observe(tmp_path, rewards, error, 0)
    assert not calls
    assert len(errors) == 1
    assert not any(
        isinstance(item, rw.PrivateStorageError) for item in causes(errors[0])
    )
    if case.startswith("custody"):
        assert has_cause(errors[0], refusal)
    sandbox.upload_file.assert_not_awaited()
    assert warnings


@pytest.mark.parametrize("prior", [False, True])
async def test_incoming_member_read_denial_still_saves_zero(
    tmp_path, monkeypatch, prior
):
    sandbox, verifier, calls = await judge_environment(
        tmp_path, monkeypatch, "native", prior
    )
    state = trace_capture(monkeypatch, sandbox)
    errors = catch_verification(monkeypatch, verifier)
    fault = PermissionError(errno.EACCES, "fixture incoming member read denied")
    reached = []
    original = tarfile.ExFileObject.read

    def read(stream, *args):
        assert state["app"] and state["archive_bytes"] > 0
        assert (state["app"] / "main.py").is_file()
        reached.append("incoming member read")
        raise fault

    monkeypatch.setattr(tarfile.ExFileObject, "read", read)
    rewards, error, warnings = await _assess(sandbox, verifier)
    monkeypatch.setattr(tarfile.ExFileObject, "read", original)
    assert reached == ["incoming member read"]
    observe(tmp_path, rewards, error, 0)
    assert not calls
    assert len(errors) == 1 and has_cause(errors[0], fault)
    assert not any(
        isinstance(item, rw.PrivateStorageError) for item in causes(errors[0])
    )
    sandbox.upload_file.assert_not_awaited()
    assert warnings


def rubric_source(tmp_path, verifier, format):
    path = tmp_path / "task/tests" / f"rubric.{format}"
    if format == "toml":
        path.write_text('[[criterion]]\nname = "c1"\ndescription = "ok"\n')
    verifier._task.config.verifier.judge.rubric_path = f"tests/rubric.{format}"
    return path


def trace_snapshots(monkeypatch):
    copied = {}
    original_copy, original_open = rw._snapshot_stream, Path.open

    def copy(source, target, **kwargs):
        source_info = os.fstat(source.fileno())
        assert stat.S_ISREG(source_info.st_mode)
        result = original_copy(source, target, **kwargs)
        if isinstance(target, Path) and target.parent.name.startswith(
            ("benchflow-snapshot-", "benchflow-deliverable-")
        ):
            with original_open(target, "rb") as stream:
                contents = stream.read()
            assert len(contents) == source_info.st_size
            copied[target] = contents
        return result

    monkeypatch.setattr(rw, "_snapshot_stream", copy)
    return copied, original_open


READ_CASES = [
    ("text", "open", errno.EACCES),
    ("text", "read", errno.EACCES),
    ("text", "close", errno.EACCES),
    ("text", "read", errno.EIO),
    ("json", "open", errno.EACCES),
    ("json", "read", errno.EACCES),
    ("json", "close", errno.EACCES),
    ("toml", "open", errno.EACCES),
    ("toml", "read", errno.EACCES),
    ("toml", "close", errno.EACCES),
    ("json", "read", errno.EIO),
    ("toml", "open", errno.ENOSPC),
]


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("kind,operation,number", READ_CASES)
async def test_completed_private_snapshot_read_is_saved_unscored(
    tmp_path, monkeypatch, prior, kind, operation, number
):
    sandbox, verifier, calls = await judge_environment(
        tmp_path, monkeypatch, "llm", prior
    )
    source = rubric_source(tmp_path, verifier, "json" if kind == "text" else kind)
    expected = b"the answer is 42\n" if kind == "text" else source.read_bytes()
    copied, original_open = trace_snapshots(monkeypatch)
    errors = catch_verification(monkeypatch, verifier)
    fault = OSError(number, f"fixture completed snapshot {operation} failed")
    reached = []

    def opened(path, *args, **kwargs):
        selected = path in copied and (
            path.name == "answer.txt" if kind == "text" else path.name == source.name
        )
        if not selected or reached:
            return original_open(path, *args, **kwargs)
        assert copied[path] == expected
        assert bool(getattr(sandbox, "_benchflow_test_command_issued", False)) is prior
        if operation == "open":
            reached.append((operation, path))
            raise fault
        raw = original_open(path, *args, **kwargs)

        class Snapshot(StreamProxy):
            def read(self, *args):
                if operation == "read":
                    reached.append((operation, path))
                    raise fault
                return self.raw.read(*args)

            def close(self):
                self.raw.close()
                if operation == "close" and not reached:
                    reached.append((operation, path))
                    raise fault

        return Snapshot(raw)

    monkeypatch.setattr(Path, "open", opened)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert len(reached) == 1
    observe(tmp_path, rewards, error, "unscored", f"snapshot {operation}")
    assert not calls
    assert len(errors) == 1 and has_cause(errors[0], fault)
    assert isinstance(errors[0], rw.PrivateStorageError)
    assert errors[0].__cause__ is fault
    assert not verifier._rollout_paths.reward_json_path.exists()
    sandbox.download_file.assert_awaited_once()
    assert "provider failed" not in error and not warnings


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("format", ["json", "toml"])
@pytest.mark.parametrize("score", [0, 1])
async def test_completed_document_snapshots_keep_numeric_verdicts(
    tmp_path, monkeypatch, prior, format, score
):
    sandbox, verifier, calls = await judge_environment(
        tmp_path, monkeypatch, "llm", prior
    )
    source = rubric_source(tmp_path, verifier, format)
    copied, _ = trace_snapshots(monkeypatch)

    async def provider(model, prompt, **kwargs):
        assert "the answer is 42" in prompt and "(error reading" not in prompt
        assert any(path.name == source.name for path in copied)
        assert any(path.name == "answer.txt" for path in copied)
        calls.append("document")
        return json.dumps({"verdict": "pass" if score else "fail"})

    monkeypatch.setattr(
        "benchflow.rewards.llm.call_judge", AsyncMock(side_effect=provider)
    )
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert calls == ["document"]
    observe(tmp_path, rewards, error, score)
    assert not warnings


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("format", ["json", "toml"])
async def test_malformed_rubric_is_parsing_not_private_io(
    tmp_path, monkeypatch, prior, format
):
    sandbox, verifier, calls = await judge_environment(
        tmp_path, monkeypatch, "llm", prior
    )
    source = rubric_source(tmp_path, verifier, format)
    source.write_text("[")
    copied, _ = trace_snapshots(monkeypatch)
    errors = catch_verification(monkeypatch, verifier)
    original = rubric_config.json.loads if format == "json" else tomllib.loads
    parsed = []

    def parse(text, *args, **kwargs):
        if text == "[":
            assert any(
                path.name == source.name and data == b"["
                for path, data in copied.items()
            )
            parsed.append(text)
        return original(text, *args, **kwargs)

    if format == "json":
        monkeypatch.setattr(rubric_config.json, "loads", parse)
    else:
        monkeypatch.setattr(tomllib, "loads", parse)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert parsed == ["["] and not calls
    assert len(errors) == 1 and not isinstance(errors[0], rw.PrivateStorageError)
    assert "DecodeError" in error and "PrivateStorageError" not in error
    # A malformed task rubric is a configuration error, not submitted evidence.
    observe(tmp_path, rewards, error, "unscored", "DecodeError")
    assert not warnings


@pytest.mark.parametrize("prior", [False, True])
async def test_provider_failure_after_valid_snapshots_stays_sanitized(
    tmp_path, monkeypatch, prior
):
    sandbox, verifier, calls = await judge_environment(
        tmp_path, monkeypatch, "llm", prior
    )
    copied, _ = trace_snapshots(monkeypatch)
    errors = catch_verification(monkeypatch, verifier)
    fault = ValueError("PRIVATE_PROVIDER_RESPONSE_DETAIL")

    async def provider(model, prompt, **kwargs):
        assert "the answer is 42" in prompt
        assert any(path.name == "answer.txt" for path in copied)
        calls.append("document")
        raise fault

    monkeypatch.setattr(
        "benchflow.rewards.llm.call_judge", AsyncMock(side_effect=provider)
    )
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert calls == ["document"] and len(errors) == 1
    assert isinstance(errors[0], JudgeScoringError) and errors[0].__cause__ is fault
    observe(tmp_path, rewards, error, "unscored", "JudgeScoringError")
    assert "PRIVATE_PROVIDER_RESPONSE_DETAIL" not in error and not warnings


@pytest.mark.parametrize(
    "case", ["direct-denial", "invalid-text", "invalid-rich", "unsupported-rich"]
)
def test_direct_and_format_reader_policies_are_preserved(tmp_path, monkeypatch, case):
    path = tmp_path / ("input.docx" if case.endswith("rich") else "input.txt")
    path.write_bytes(b"answer\xff")
    reached = []
    if case == "direct-denial":
        original = Path.open

        def denied(target, *args, **kwargs):
            if target == path:
                reached.append(case)
                raise PermissionError(errno.EACCES, "fixture untrusted source denied")
            return original(target, *args, **kwargs)

        monkeypatch.setattr(Path, "open", denied)
        assert "error reading input.txt" in file_readers.read_file_as_text(path)
    elif case == "invalid-text":
        path.write_bytes(b"answer\xff\r\nnext\rlast")
        assert file_readers.find_deliverables(tmp_path) == {
            "input.txt": "answer\ufffd\nnext\nlast"
        }
        reached.append(case)
    else:

        def decode(snapshot):
            assert snapshot.parent.name.startswith("benchflow-deliverable-")
            assert snapshot.read_bytes() == b"answer\xff"
            reached.append(case)
            if case == "invalid-rich":
                raise ValueError("fixture invalid document encoding")
            return "(unsupported document format)"

        monkeypatch.setattr(file_readers, "_read_docx", decode)
        result = file_readers.find_deliverables(tmp_path)["input.docx"]
        assert result.startswith(
            "(error reading" if case == "invalid-rich" else "(unsupported"
        )
    assert reached == [case]
