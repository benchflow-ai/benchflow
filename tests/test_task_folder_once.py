"""A trial's setup reads its task folder once.

Earlier a Daytona trial's setup read a large task folder up to four
times: ``TaskRuntimeView`` hashed every source file each time it was built,
and the launch checks build one twice per trial for its paths alone; the
keep-services digest listed and measured the whole folder before applying its
64 MiB cap; and both content digests spent most of their time in one
``Path.relative_to`` per file. On an 877 MB, 14,561-file folder that was
8.5 s of ``Rollout.setup()`` on an idle 16-vCPU machine (2.5 s after), and a
52.5 s median with 60 trials on 16 vCPUs.

These tests guard the fix: the digests keep their exact construction
(written out below from each docstring, as ``test_task_digest`` does), the
launch check hashes nothing, and an over-cap folder is neither read nor
walked to the end.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from benchflow._utils.task_authoring import task_digest
from benchflow.task import keep_services
from benchflow.task import runtime_view as runtime_view_module
from benchflow.task.runtime_view import TaskRuntimeView


def _registry_digest(entries: dict[str, bytes]) -> str:
    """``task_digest``'s construction: sorted by POSIX path string, each file
    adding ``path_utf8 + b"\\x00" + sha256(bytes).digest()``."""
    h = hashlib.sha256()
    for rel in sorted(entries):
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(hashlib.sha256(entries[rel]).digest())
    return f"sha256:{h.hexdigest()}"


def _package_digest(entries: dict[bytes, bytes]) -> str:
    """``keep_services.task_digest``'s construction: sorted by path bytes, each
    file adding ``path + b"\\0" + sha256 hex + b"\\n"``."""
    h = hashlib.sha256()
    for rel in sorted(entries):
        h.update(rel + b"\0")
        h.update(hashlib.sha256(entries[rel]).hexdigest().encode())
        h.update(b"\n")
    return f"sha256:{h.hexdigest()}"


def _split_task(root: Path) -> Path:
    task = root / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text('version = "1.0"\n\n[verifier]\ntimeout_sec = 60\n')
    (task / "instruction.md").write_text("Do the thing.\n")
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    return task


# The runtime view and the launch check


def test_view_hashes_sources_on_first_read_only(tmp_path, monkeypatch):
    """Building the view reads no file content; the first read of
    ``source_hashes`` hashes once, with the same values as before."""
    task = _split_task(tmp_path)
    expected = runtime_view_module._source_hashes(runtime_view_module.TaskPaths(task))
    calls: list[Path] = []
    real = runtime_view_module._source_hashes

    def counting(paths):
        calls.append(paths.task_dir)
        return real(paths)

    monkeypatch.setattr(runtime_view_module, "_source_hashes", counting)
    view = TaskRuntimeView.from_task_dir(task)
    assert calls == []
    assert view.source_hashes == expected
    assert view.source_hashes == expected
    assert calls == [task.resolve()]
    assert set(expected) == {
        "environment/Dockerfile",
        "instruction.md",
        "task.toml",
        "tests/test.sh",
    }


def test_launch_check_hashes_no_source_file(tmp_path, monkeypatch):
    """The plan check and ``create_environment`` both run this check; it needs
    the task's paths and capabilities, never its file hashes."""
    from benchflow.sandbox.setup import _validate_task_runtime_for_launch
    from benchflow.task.task import Task

    task_dir = _split_task(tmp_path)

    def refuse(_paths):
        raise AssertionError("the launch check hashed the task's files")

    monkeypatch.setattr(runtime_view_module, "_source_hashes", refuse)
    _validate_task_runtime_for_launch(
        Task(task_dir), sandbox_type="docker", task_path=task_dir
    )


# keep_services.task_digest


def test_package_digest_construction_on_a_mixed_tree(tmp_path):
    """Which files count, exactly: the published files and folders (a linked
    published folder is walked, a link to a file counts as the file), minus
    the default ignores at any depth; never a linked folder inside one, a
    broken link, a FIFO or an unpublished file."""
    task = _split_task(tmp_path)
    outside = tmp_path / "outside"
    (outside / "deep").mkdir(parents=True)
    (outside / "deep" / "step.sh").write_bytes(b"echo step\n")
    (task / "steps").symlink_to(outside, target_is_directory=True)
    env = task / "environment"
    (env / "nested" / "__pycache__").mkdir(parents=True)
    (env / "nested" / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"pyc")
    (env / "nested" / "keep.py").write_bytes(b"print(1)\n")
    (env / "nested" / "keep.pyc").write_bytes(b"ignored by name")
    (env / "nested" / "notes.txt~").write_bytes(b"backup")
    (env / ".DS_Store").write_bytes(b"finder")
    (env / "empty.bin").write_bytes(b"")
    (env / "café.txt").write_bytes(b"accent")
    (env / "link-to-file").symlink_to(task / "instruction.md")
    (env / "link-to-dir").symlink_to(outside, target_is_directory=True)
    (env / "broken").symlink_to(task / "missing")
    os.mkfifo(env / "pipe")
    (task / "unpublished.txt").write_bytes(b"not in the package")

    expected = _package_digest(
        {
            b"task.toml": (task / "task.toml").read_bytes(),
            b"instruction.md": b"Do the thing.\n",
            b"environment/Dockerfile": b"FROM ubuntu:24.04\n",
            b"environment/nested/keep.py": b"print(1)\n",
            b"environment/empty.bin": b"",
            "environment/café.txt".encode(): b"accent",
            b"environment/link-to-file": b"Do the thing.\n",
            b"tests/test.sh": b"#!/bin/bash\nexit 0\n",
            b"steps/deep/step.sh": b"echo step\n",
        }
    )
    assert keep_services.task_digest(task) == expected


def test_package_digest_keeps_raw_name_bytes(tmp_path):
    """A name that is not UTF-8 is hashed as its own bytes."""
    task = _split_task(tmp_path)
    raw = b"environment/data-\xff.bin"
    (task / os.fsdecode(raw)).write_bytes(b"raw")
    expected = _package_digest(
        {
            b"task.toml": (task / "task.toml").read_bytes(),
            b"instruction.md": b"Do the thing.\n",
            b"environment/Dockerfile": b"FROM ubuntu:24.04\n",
            raw: b"raw",
            b"tests/test.sh": b"#!/bin/bash\nexit 0\n",
        }
    )
    assert keep_services.task_digest(task) == expected


def test_package_digest_stops_listing_at_the_cap(tmp_path, monkeypatch):
    """A folder over the cap gets None after the file that passes it: no
    file is read and the rest is not listed."""
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    for name in ("a", "b", "c", "d"):
        (task / "environment" / name).write_bytes(b"x" * 8)
    monkeypatch.setattr(keep_services, "_DIGEST_MAX_BYTES", 12)
    listed: list[str] = []
    real = keep_services._ignored_by_default

    def counting(rel: str) -> bool:
        listed.append(rel)
        return real(rel)

    def no_read(*_args, **_kwargs):
        raise AssertionError("a file of an over-cap folder was read")

    monkeypatch.setattr(keep_services, "_ignored_by_default", counting)
    monkeypatch.setattr(keep_services.hashlib, "file_digest", no_read)
    assert keep_services.task_digest(task) is None
    assert len(listed) == 2


def test_package_digest_at_the_cap_is_hashed(tmp_path, monkeypatch):
    """Exactly the cap is still hashed; one byte more is not."""
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "a").write_bytes(b"x" * 6)
    (task / "environment" / "b").write_bytes(b"y" * 6)
    monkeypatch.setattr(keep_services, "_DIGEST_MAX_BYTES", 12)
    assert keep_services.task_digest(task) == _package_digest(
        {b"environment/a": b"x" * 6, b"environment/b": b"y" * 6}
    )
    (task / "environment" / "c").write_bytes(b"z")
    assert keep_services.task_digest(task) is None


# _utils.task_authoring.task_digest


def test_registry_digest_construction_on_a_mixed_tree(tmp_path):
    """Every regular file at any depth (no link, FIFO or source record, at
    any depth), large files included, in POSIX path-string order."""
    task = tmp_path / "task"
    (task / "environment" / "deep" / "deeper").mkdir(parents=True)
    big = os.urandom(3 * 1024 * 1024 + 17)
    (task / "environment" / "deep" / "deeper" / "big.bin").write_bytes(big)
    (task / "environment" / "deep" / ".benchflow-source.json").write_text("{}")
    (task / ".benchflow-source.json").write_text("{}")
    (task / "task.md").write_text("# A task\n")
    (task / "environment" / "café.txt").write_bytes(b"accent")
    (task / "environment" / "empty").write_bytes(b"")
    (task / "environment" / "link").symlink_to(task / "task.md")
    os.mkfifo(task / "environment" / "pipe")
    expected = _registry_digest(
        {
            "environment/deep/deeper/big.bin": big,
            "task.md": b"# A task\n",
            "environment/café.txt": b"accent",
            "environment/empty": b"",
        }
    )
    assert task_digest(task) == expected


def test_registry_digest_relative_task_path(tmp_path, monkeypatch):
    """A relative task path gives the same digest as its absolute path."""
    task = _split_task(tmp_path)
    monkeypatch.chdir(task.parent)
    assert task_digest(Path("task")) == task_digest(task)
    monkeypatch.chdir(task)
    assert task_digest(Path(".")) == task_digest(task)


def test_registry_digest_still_refuses_a_name_that_is_not_utf8(tmp_path):
    """The registry construction encodes each path as UTF-8, so a name that
    is not UTF-8 raises, as it did before."""
    task = _split_task(tmp_path)
    (task / os.fsdecode(b"environment/data-\xff.bin")).write_bytes(b"raw")
    with pytest.raises(UnicodeEncodeError):
        task_digest(task)
