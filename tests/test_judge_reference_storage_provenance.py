"""Regressions for the judge boundaries at frozen 10529b8de.

Guards provider references and private judge allocation/archive operations.
Keep the R5/R6/R7 acceptance inputs unchanged. Scoring cases are saved and
reloaded; direct helper cases verify refusals and interfaces. No live provider.
"""

import contextlib
import errno
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from benchflow._utils import reachable_write as rw
from benchflow.taskmd import judging
from benchflow.taskmd import verify as taskmd
from tests._os_capabilities import patch_os
from tests.test_native_provider_provenance import native, response, transport
from tests.test_r5_release_contract import observe, prime
from tests.test_verifier_own_failures import _assess, _task


def reference_response(channel, path, *, score=1, quote="print(42)"):
    value = response()
    tool = value["content"][0]
    if channel == "read":
        tool.update(name="read", input={"path": path})
    else:
        verdict = tool["input"]["verdicts"][0]
        verdict["verdict"] = "pass" if score else "fail"
        verdict["citations"] = [{"source": channel, "path": path, "quote": quote}]
    return value


def trace_references(monkeypatch):
    reached = []
    read, cite = judging.read_result, judging.check_citation

    def read_call(files, assignment, step, fence, args, *extra):
        reached.append(("read", args.get("path")))
        return read(files, assignment, step, fence, args, *extra)

    def cite_call(value, *args):
        reached.append((value["source"], value.get("path")))
        return cite(value, *args)

    monkeypatch.setattr(judging, "read_result", read_call)
    monkeypatch.setattr(judging, "check_citation", cite_call)
    return reached


REFERENCE_CASES = [
    ("read", "/app/" + "a" * 256, False),
    ("read", "/app/" + "a" * 256 + "/child", False),
    ("read", "/app/" + "é" * 128, False),
    ("file", "/app/" + "a" * 256, False),
    ("file", "é" * 128, False),
    ("view", "/judge/" + "a" * 256, False),
    ("read", "/app/main.py", True),
    ("file", "/app/main.py", True),
    ("view", "/judge/instruction.md", True),
    ("view", "/judge/../app/main.py", False),
]


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize(
    "channel,path,lookup_limit",
    REFERENCE_CASES,
    ids=[
        "read-leaf",
        "read-parent",
        "read-utf8",
        "file-leaf",
        "file-relative-utf8",
        "view-leaf",
        "read-os-limit",
        "file-os-limit",
        "view-os-limit",
        "view-parent-reference",
    ],
)
async def test_provider_reference_failure_is_saved_sanitized(
    tmp_path, monkeypatch, prior, channel, path, lookup_limit
):
    sandbox, verifier = native(tmp_path, monkeypatch, prior)
    reached = trace_references(monkeypatch)
    calls, lookups = [], []
    original = rw.ReachablePath.stat

    def stat(target, *args, **kwargs):
        if lookup_limit and calls and target.name == Path(path).name:
            lookups.append(target.name)
            raise OSError(errno.ENAMETOOLONG, "PRIVATE_REFERENCE_OS_DETAIL")
        return original(target, *args, **kwargs)

    monkeypatch.setattr(rw.ReachablePath, "stat", stat)

    async def handle(request):
        calls.append(json.loads(request.content))
        assert bool(getattr(sandbox, "_benchflow_test_command_issued", False)) is prior
        return httpx.Response(200, json=reference_response(channel, path))

    transport(monkeypatch, handle)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert len(calls) == 1 and reached == [(channel, path)]
    assert bool(lookups) is lookup_limit
    observe(
        tmp_path,
        rewards,
        error,
        "unscored",
        "provider selected an invalid view path"
        if "/../" in path
        else "provider selected an unrepresentable path",
    )
    assert "PRIVATE_REFERENCE_OS_DETAIL" not in error and path not in error
    assert not warnings


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("channel", ["read", "file", "view"])
@pytest.mark.parametrize("score", [0, 1])
async def test_valid_native_references_keep_numeric_results(
    tmp_path, monkeypatch, prior, channel, score
):
    sandbox, verifier = native(tmp_path, monkeypatch, prior)
    reached = trace_references(monkeypatch)
    calls = []
    path = "/judge/instruction.md" if channel == "view" else "/app/main.py"
    quote = (
        verifier._task.instruction.splitlines()[0] if channel == "view" else "print(42)"
    )

    async def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        assert bool(getattr(sandbox, "_benchflow_test_command_issued", False)) is prior
        if channel == "read" and len(calls) == 1:
            value = reference_response("read", path)
        else:
            if channel == "read":
                assert "print(42)" in body["messages"][-1]["content"][0]["content"]
            value = reference_response(
                "file" if channel == "read" else channel, path, score=score, quote=quote
            )
        return httpx.Response(200, json=value)

    transport(monkeypatch, handle)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert len(calls) == (2 if channel == "read" else 1)
    assert reached == (
        [("read", path), ("file", path)] if channel == "read" else [(channel, path)]
    )
    observe(tmp_path, rewards, error, score)
    assert not warnings
    review = json.loads((tmp_path / "trial/verifier/review.json").read_text())
    assert review["verdicts"][0]["citations"][0]["verified"] is True


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize(
    "channel,private_view",
    [("read", False), ("file", False), ("read", True), ("view", True)],
)
async def test_reference_source_denial_and_private_view_failure_stay_distinct(
    tmp_path, monkeypatch, prior, channel, private_view
):
    sandbox, verifier = native(tmp_path, monkeypatch, prior)
    calls, reads = [], []
    path = "/judge/instruction.md" if private_view else "/app/main.py"
    original = rw.ReachablePath.read_bytes

    def read(target):
        if calls and target.name == Path(path).name:
            reads.append(target.name)
            raise PermissionError(errno.EACCES, "fixture reference read denied")
        return original(target)

    monkeypatch.setattr(rw.ReachablePath, "read_bytes", read)

    async def handle(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=reference_response(channel, path))

    transport(monkeypatch, handle)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert calls == ["/v1/messages"] and reads == [Path(path).name]
    observe(tmp_path, rewards, error, "unscored" if private_view else 0)
    assert bool(warnings) is not private_view
    if private_view:
        assert "PrivateStorageError" in error and "reference read denied" in error


async def judge_environment(tmp_path, monkeypatch, kind, prior):
    if kind == "native":
        sandbox, verifier = native(tmp_path, monkeypatch, False)
    else:
        from tests.test_taskmd_verify import _Sandbox
        from tests.test_verifier_strategies import _AGENT_JUDGE_VERIFIER_MD

        _, verifier = _task(tmp_path, "llm-judge" if kind == "llm" else "test-script")
        root = tmp_path / "sandbox"
        (root / "app").mkdir(parents=True)
        (root / "app/answer.txt").write_text("the answer is 42\n")
        sandbox = _Sandbox(root)
        sandbox.is_mounted = True
        sandbox.upload_dir = AsyncMock()
        verifier._sandbox = sandbox
        verifier._task.config.verifier.judge.input_dir = "/app"
        if kind.startswith("agent"):
            document = _AGENT_JUDGE_VERIFIER_MD
            if kind == "agent-remote":
                document = document.replace("artifacts/answer.txt", "/app/answer.txt")
            (tmp_path / "task/tests/verifier.md").write_text(document)
            (verifier._rollout_paths.artifacts_dir / "answer.txt").write_text("42")
    original_exec = sandbox.exec
    await prime(tmp_path / "prior", sandbox, prior)
    sandbox.exec = AsyncMock(wraps=original_exec)
    sandbox.download_file = AsyncMock(wraps=sandbox.download_file)
    sandbox.upload_file = AsyncMock(wraps=sandbox.upload_file)
    assert bool(getattr(sandbox, "_benchflow_test_command_issued", False)) is prior
    calls = []

    async def handle(request):
        calls.append("native")
        return httpx.Response(200, json=response())

    async def document_provider(*args, **kwargs):
        calls.append("document")
        return '{"verdict":"pass"}'

    transport(monkeypatch, handle)
    monkeypatch.setattr(
        "benchflow.rewards.llm.call_judge", AsyncMock(side_effect=document_provider)
    )
    return sandbox, verifier, calls


def catch_verification(monkeypatch, verifier):
    errors = []
    original = verifier.verify

    async def verify(*args, **kwargs):
        try:
            return await original(*args, **kwargs)
        except Exception as exc:
            errors.append(exc)
            raise

    monkeypatch.setattr(verifier, "verify", verify)
    return errors


def has_cause(error, wanted):
    seen = set()
    while error is not None and id(error) not in seen:
        if error is wanted:
            return True
        seen.add(id(error))
        error = error.__cause__
    return False


DIRECTORY_CASES = [
    ("native", "benchflow-judge-copy-", "EACCES"),
    ("native", "benchflow-judge-kept-", "EACCES"),
    ("agent-local", "benchflow-agent-judge-", "EACCES"),
    ("llm", "benchflow-native-judge-", "EACCES"),
    ("llm", "benchflow-snapshot-", "EACCES"),
    ("llm", "benchflow-deliverable-", "EACCES"),
    ("agent-remote", "benchflow-judge-copy-", "EACCES"),
    ("native", "benchflow-judge-kept-", "EIO"),
    ("agent-local", "benchflow-agent-judge-", "ENOSPC"),
]


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("kind,prefix,code", DIRECTORY_CASES)
async def test_private_judge_directory_create_is_saved_unscored(
    tmp_path, monkeypatch, prior, kind, prefix, code
):
    sandbox, verifier, calls = await judge_environment(
        tmp_path, monkeypatch, kind, prior
    )
    errors = catch_verification(monkeypatch, verifier)
    fault = OSError(getattr(errno, code), "fixture exact private mkdir failed")
    reached = []
    original = os.mkdir

    def mkdir(path, *args, **kwargs):
        if Path(path).name.startswith(prefix):
            reached.append(os.fspath(path))
            assert not calls
            raise fault
        return original(path, *args, **kwargs)

    patch_os(monkeypatch, "mkdir", mkdir)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert len(reached) == 1 and not calls and len(errors) == 1
    assert has_cause(errors[0], fault)
    assert bool(getattr(sandbox, "_benchflow_test_command_issued", False)) is prior
    observe(tmp_path, rewards, error, "unscored", "PrivateStorageError")
    assert "exact private mkdir failed" in error and not warnings
    if prefix == "benchflow-judge-copy-":
        assert any("BF copy" in c.args[0] for c in sandbox.exec.await_args_list)
        sandbox.download_file.assert_not_awaited()
    if prefix == "benchflow-judge-kept-":
        sandbox.download_file.assert_awaited_once()
        sandbox.upload_file.assert_not_awaited()


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize(
    "kind,leaf",
    [
        ("native", "taskmd-judge"),
        ("native", "fs"),
        ("native", "judge"),
        ("agent-remote", "0-capture"),
    ],
)
async def test_private_judge_child_directory_create_is_saved_unscored(
    tmp_path, monkeypatch, prior, kind, leaf
):
    sandbox, verifier, calls = await judge_environment(
        tmp_path, monkeypatch, kind, prior
    )
    fault = PermissionError(errno.EACCES, "fixture private child mkdir failed")
    errors = catch_verification(monkeypatch, verifier)
    reached = []
    original = os.mkdir

    def mkdir(path, *args, **kwargs):
        if Path(path).name == leaf:
            reached.append(leaf)
            raise fault
        return original(path, *args, **kwargs)

    patch_os(monkeypatch, "mkdir", mkdir)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert reached and not calls and len(errors) == 1
    assert has_cause(errors[0], fault)
    observe(tmp_path, rewards, error, "unscored", "PrivateStorageError")
    assert not warnings


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize(
    "kind,prefix,provider_calls",
    [
        ("native", "benchflow-judge-kept-", 0),
        ("agent-local", "benchflow-agent-judge-", 0),
        ("llm", "benchflow-native-judge-", 1),
    ],
)
async def test_private_judge_cleanup_retains_its_cause(
    tmp_path, monkeypatch, prior, kind, prefix, provider_calls
):
    sandbox, verifier, calls = await judge_environment(
        tmp_path, monkeypatch, kind, prior
    )
    errors = catch_verification(monkeypatch, verifier)
    fault = OSError(errno.EIO, "fixture private directory cleanup failed")
    reached = []
    original = os.rmdir

    def rmdir(path, *args, **kwargs):
        if Path(path).name.startswith(prefix):
            # TemporaryDirectory removes its root by its full pathname.
            assert Path(path).is_absolute()
            reached.append(Path(path))
            assert len(calls) == provider_calls
            raise fault
        return original(path, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch_os(patch, "rmdir", rmdir)
            rewards, error, warnings = await _assess(sandbox, verifier)
    finally:
        for path in reached:
            if path.exists():
                shutil.rmtree(path)
    assert len(reached) == 1 and len(calls) == provider_calls
    assert len(errors) == 1 and has_cause(errors[0], fault)
    observe(tmp_path, rewards, error, "unscored", "PrivateStorageError")
    assert not warnings


@pytest.mark.parametrize("prior", [False, True])
async def test_private_tests_view_read_is_saved_unscored(tmp_path, monkeypatch, prior):
    sandbox, verifier, calls = await judge_environment(
        tmp_path, monkeypatch, "native", prior
    )
    errors = catch_verification(monkeypatch, verifier)
    fault = PermissionError(errno.EACCES, "fixture private tests view read failed")
    reached = []
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path.name == "tests.json" and path.parent.name == "judge":
            reached.append(path.name)
            raise fault
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert reached == ["tests.json"] and not calls
    assert len(errors) == 1 and has_cause(errors[0], fault)
    observe(tmp_path, rewards, error, "unscored", "PrivateStorageError")
    assert not warnings


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize(
    "prefix,leaf",
    [("benchflow-snapshot-", "rubric.json"), ("benchflow-deliverable-", "answer.txt")],
)
async def test_document_judge_preserves_private_snapshot_write_cause(
    tmp_path, monkeypatch, prior, prefix, leaf
):
    sandbox, verifier, calls = await judge_environment(
        tmp_path, monkeypatch, "llm", prior
    )
    errors = catch_verification(monkeypatch, verifier)
    fault = PermissionError(errno.EACCES, "fixture private snapshot leaf failed")
    reached = []
    original = rw.create_private_file

    @contextlib.contextmanager
    def create(target):
        if target.name == leaf and target.parent.name.startswith(prefix):
            reached.append(target.name)
            raise fault
        with original(target) as stream:
            yield stream

    monkeypatch.setattr(rw, "create_private_file", create)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert reached == [leaf] and not calls
    assert len(errors) == 1 and has_cause(errors[0], fault)
    observe(tmp_path, rewards, error, "unscored", "PrivateSnapshotWriteError")
    assert "private snapshot leaf failed" in error and not warnings


ARCHIVE_CASES = [
    (phase, "EACCES")
    for phase in (
        "create",
        "header",
        "data",
        "trailer",
        "tell",
        "flush",
        "close",
        "stat",
    )
]
ARCHIVE_CASES += [("create", "EIO"), ("data", "ENOSPC")]


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("phase,code", ARCHIVE_CASES)
async def test_outgoing_custody_archive_failure_is_saved_unscored(
    tmp_path, monkeypatch, prior, phase, code
):
    sandbox, verifier, calls = await judge_environment(
        tmp_path, monkeypatch, "native", prior
    )
    fault = OSError(getattr(errno, code), "fixture outgoing archive operation failed")
    errors = catch_verification(monkeypatch, verifier)
    reached = []
    opened = []
    original_open, original_fdopen = os.open, os.fdopen
    original_openat2, original_stat = rw._openat2, Path.stat

    def private_fd(fd):
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        return path.name == "kept.tar" and path.parent.name.startswith(
            "benchflow-judge-kept-"
        )

    def trip(operation):
        if operation == phase and not reached:
            reached.append(operation)
            assert not calls
            raise fault

    def openat2(directory, name, flags, mode):
        # Exercise the supported no-follow fallback at this exact exclusive
        # create, so the OS operation can be observed in both frozen arms.
        if name == "kept.tar" and flags & os.O_CREAT:
            return None
        return original_openat2(directory, name, flags, mode)

    def open_file(path, flags, *args, **kwargs):
        if os.fspath(path) == "kept.tar" and flags & os.O_CREAT:
            assert flags & os.O_EXCL and flags & os.O_NOFOLLOW
            parent = Path(os.readlink(f"/proc/self/fd/{kwargs['dir_fd']}"))
            assert parent.name.startswith("benchflow-judge-kept-")
            opened.append(parent.name)
            trip("create")
        return original_open(path, flags, *args, **kwargs)

    class Stream:
        def __init__(self, raw):
            self.raw = raw

        def __getattr__(self, name):
            return getattr(self.raw, name)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

        def write(self, data):
            if data == b"print(42)\n":
                trip("data")
            elif len(data) >= 1024 and not data.strip(b"\0"):
                trip("trailer")
            elif len(data) == 512:
                trip("header")
            return self.raw.write(data)

        def tell(self):
            trip("tell")
            return self.raw.tell()

        def flush(self):
            trip("flush")
            return self.raw.flush()

        def close(self):
            try:
                # Buffered close flushes in both arms, even when no caller
                # explicitly flushes the outgoing file beforehand.
                self.flush()
                trip("close")
            finally:
                self.raw.close()

    def fdopen(fd, *args, **kwargs):
        outgoing = private_fd(fd)
        raw = original_fdopen(fd, *args, **kwargs)
        return Stream(raw) if outgoing else raw

    def stat(path, *args, **kwargs):
        if path.name == "kept.tar" and path.parent.name.startswith(
            "benchflow-judge-kept-"
        ):
            trip("stat")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(rw, "_openat2", openat2)
    patch_os(monkeypatch, "open", open_file)
    patch_os(monkeypatch, "fdopen", fdopen)
    monkeypatch.setattr(Path, "stat", stat)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert reached == [phase] and len(opened) == 1 and not calls
    assert len(errors) == 1 and has_cause(errors[0], fault)
    sandbox.download_file.assert_awaited_once()
    sandbox.upload_file.assert_not_awaited()
    observe(tmp_path, rewards, error, "unscored", "PrivateStorageError")
    assert "outgoing archive operation failed" in error and not warnings


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize(
    "operation", ["create", "write", "file-mode", "folder-mode", "link"]
)
async def test_native_capture_destination_operations_keep_private_provenance(
    tmp_path, monkeypatch, prior, operation
):
    sandbox, verifier, calls = await judge_environment(
        tmp_path, monkeypatch, "native", prior
    )
    if operation == "link":
        os.link(sandbox.root / "app/main.py", sandbox.root / "app/second.py")
    errors = catch_verification(monkeypatch, verifier)
    fault = PermissionError(
        errno.EACCES, "fixture capture destination operation failed"
    )
    reached = []
    original_create = taskmd.create_private_file
    original_chmod, original_link = os.fchmod, os.link

    @contextlib.contextmanager
    def create(target):
        selected = target.name == "main.py"
        if selected and operation == "create":
            reached.append(operation)
            raise fault
        with original_create(target) as raw:
            if not selected or operation != "write":
                yield raw
                return

            class Output:
                def __getattr__(self, name):
                    return getattr(raw, name)

                def write(self, data):
                    assert data == b"print(42)\n"
                    reached.append(operation)
                    raise fault

            yield Output()

    def chmod(fd, mode):
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        leaf = "main.py" if operation == "file-mode" else "app"
        if operation in ("file-mode", "folder-mode") and path.name == leaf:
            assert "/fs/app" in str(path)
            reached.append(operation)
            raise fault
        return original_chmod(fd, mode)

    def link(source, target, *args, **kwargs):
        if operation == "link" and target == "second.py":
            assert kwargs["follow_symlinks"] is False
            assert "src_dir_fd" in kwargs and "dst_dir_fd" in kwargs
            reached.append(operation)
            raise fault
        return original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(taskmd, "create_private_file", create)
    patch_os(monkeypatch, "fchmod", chmod)
    patch_os(monkeypatch, "link", link)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert reached == [operation] and not calls
    assert len(errors) == 1 and has_cause(errors[0], fault)
    sandbox.download_file.assert_awaited_once()
    sandbox.upload_file.assert_not_awaited()
    observe(tmp_path, rewards, error, "unscored", "PrivateStorageError")
    assert not warnings


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("kind", ["native", "agent-local", "agent-remote", "llm"])
async def test_valid_judge_preparation_reaches_provider_and_saves_one(
    tmp_path, monkeypatch, prior, kind
):
    sandbox, verifier, calls = await judge_environment(
        tmp_path, monkeypatch, kind, prior
    )
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert calls == ["native" if kind == "native" else "document"]
    observe(tmp_path, rewards, error, 1)
    assert not warnings
    if kind == "native":
        sandbox.upload_file.assert_awaited_once()


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "stash-source",
        "incoming-archive",
        "agent-source",
        "malformed-reward",
        "download-shape",
        "download-absent",
        "malformed-archive",
    ],
)
async def test_submitted_evidence_failures_still_save_zero(
    tmp_path, monkeypatch, prior, case
):
    kind = "agent-local" if case == "agent-source" else "native"
    sandbox, verifier, calls = await judge_environment(
        tmp_path, monkeypatch, kind, prior
    )
    reached = []
    archive_calls = []
    original = rw.ReachablePath.open
    original_archive = taskmd.custody_archive

    def archive(*args, **kwargs):
        archive_calls.append("custody archive")
        return original_archive(*args, **kwargs)

    monkeypatch.setattr(taskmd, "custody_archive", archive)

    def open_file(target, *args, **kwargs):
        selected = (
            (case == "incoming-archive" and target.name == "copy.tar")
            or (case == "stash-source" and archive_calls and target.name == "main.py")
            or (case == "agent-source" and target.name == "answer.txt")
        )
        if selected:
            reached.append(target.name)
            raise PermissionError(errno.EACCES, "fixture submitted source unreadable")
        return original(target, *args, **kwargs)

    if case.startswith("download-") or case == "malformed-archive":
        from benchflow.sandbox._download import RemoteFileAbsentError

        async def download(source, target):
            assert source == taskmd.COPY_TAR
            reached.append(case)
            if case == "download-shape":
                raise rw.UnsafeDownloadEntryError(
                    "fixture archive source is a directory"
                )
            if case == "download-absent":
                raise RemoteFileAbsentError("fixture archive source verified absent")
            target.write_bytes(b"not a tar archive")

        sandbox.download_file = AsyncMock(side_effect=download)
    elif case == "malformed-reward":
        _, verifier = _task(tmp_path / "reward")
        verifier._sandbox = sandbox
        original_exec = sandbox.exec

        async def execute(command, **kwargs):
            if "test-stdout.txt" in command:
                reached.append("reward command")
                verifier._rollout_paths.reward_text_path.write_text("not a number")
            return await original_exec(command, **kwargs)

        sandbox.exec = AsyncMock(side_effect=execute)
    else:
        monkeypatch.setattr(rw.ReachablePath, "open", open_file)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert reached and not calls
    if case == "stash-source":
        assert archive_calls == ["custody archive"]
        sandbox.upload_file.assert_not_awaited()
    elif case in (
        "incoming-archive",
        "download-shape",
        "download-absent",
        "malformed-archive",
    ):
        sandbox.download_file.assert_awaited_once()
    observe(tmp_path, rewards, error, 0)
    assert warnings


@pytest.mark.parametrize("refusal", ["location", "mount", "custody"])
def test_private_storage_refusals_do_not_become_own_errors(
    tmp_path, monkeypatch, refusal
):
    """Only attempted private I/O acquires storage provenance."""
    from benchflow.task.verifier_errors import verifier_infrastructure_failure

    source = tmp_path / "source"
    source.write_text("ordinary readable evidence")
    reached = []
    problem = rw.ReachableWriteError("fixture unsafe private location")
    if refusal == "location":

        def gettempdir():
            reached.append("location")
            raise problem

        monkeypatch.setattr(tempfile, "gettempdir", gettempdir)
    elif refusal == "mount":

        def inside_mount(_):
            reached.append("mount")
            return True

        monkeypatch.setattr(rw, "_inside_declared_mount", inside_mount)
    else:

        @contextlib.contextmanager
        def refused(_):
            reached.append("custody")
            raise problem
            yield

        monkeypatch.setattr(taskmd, "create_private_file", refused)
    with pytest.raises(OSError) as caught:
        if refusal == "custody":
            taskmd.custody_archive(None, [], tmp_path / "kept.tar")
        else:
            # Select the production consumer, without importing a helper that
            # does not exist in F: paired failures must not be import failures.
            with rw.snapshot_file(source):
                pass
    assert reached == [refusal]
    assert not isinstance(caught.value, rw.PrivateStorageError)
    assert not verifier_infrastructure_failure(caught.value)


def test_private_snapshot_publication_keeps_primary_detail_and_notes(
    tmp_path, monkeypatch, caplog
):
    """Guards the primary detail and notes, keeping the existing restore assertion."""
    from benchflow.task.verifier_errors import verifier_infrastructure_failure
    from tests.test_taskmd_r15 import PREVIOUS, _archive, _old_output, _staging_left

    _old_output(tmp_path)
    held = rw.pin_destination(tmp_path)
    primary = OSError(errno.ENOSPC, "fixture renaming payload failed")
    restoration = OSError(errno.EACCES, "fixture restoring previous failed")
    primary.add_note("fixture primary operation note")
    reached = []
    rename = os.rename

    def replace(source, target, **kwargs):
        if source == "payload":
            reached.append(source)
            raise primary
        return rename(source, target, **kwargs)

    def restore(source, target, **kwargs):
        assert source == PREVIOUS
        reached.append(source)
        raise restoration

    patch_os(monkeypatch, "rename", replace)
    monkeypatch.setattr(rw, "_rename_noreplace", restore)
    with pytest.raises(rw.PrivateSnapshotWriteError) as caught:
        rw.publish_tar_stream(io.BytesIO(_archive()), held / "out", directory=True)
    assert reached == ["payload", PREVIOUS]
    assert caught.value.__cause__ is primary
    assert caught.value.errno == errno.ENOSPC
    assert "renaming payload failed" in str(caught.value)
    assert verifier_infrastructure_failure(caught.value)
    assert all(note in caught.value.__notes__ for note in primary.__notes__)
    assert any("previous" in note for note in caught.value.__notes__)
    (kept,) = _staging_left(tmp_path)
    assert (tmp_path / kept / PREVIOUS / "old").read_text() == "OLD"
    assert kept in caplog.text


def test_private_spool_closed_tracks_underlying_stream():
    """Guards the spool's closed state, including closure through the real stream."""
    raw = io.BytesIO(b"private bytes")
    wrapped = rw._PrivateSpool(raw)
    assert wrapped.closed is False
    raw.close()
    assert wrapped.closed is True
    with rw._private_spool() as anonymous:
        assert anonymous.closed is False
        assert os.fstat(anonymous.fileno()).st_nlink == 0
    assert anonymous.closed is True
