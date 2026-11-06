"""Guards CONFIRM-taskmd-rewards-R3-1 against 9662e9ee's live destination writes."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from benchflow.taskmd import verify
from tests.reachable_fixture import directory, reachable
from tests.test_taskmd_agent_judge import _grade, _ScriptedJudge


def _trees(tmp_path):
    fs = tmp_path / "private" / "fs"
    (fs / "app" / "out").mkdir(parents=True)
    (fs / "app" / "out" / "a.txt").write_text("padding")
    (fs / "app" / "out" / "z.txt").write_text("agent bytes")
    public = tmp_path / "trial" / "verifier" / verify.JUDGE_DIR
    public.mkdir(parents=True)
    return fs, public


@pytest.mark.parametrize("ancestor", ["fs", "fs/app", "fs/app/out"])
@pytest.mark.parametrize("relative", [False, True])
def test_copy_time_ancestor_swap_cannot_overwrite_controller_files(
    tmp_path, monkeypatch, ancestor, relative
):
    """Guards R3-1: every destination ancestor, absolute and relative links."""
    fs, public = _trees(tmp_path)
    destination = reachable(public)
    victim_root = tmp_path / "victim-a"
    suffix = Path("fs/app/out/z.txt").relative_to(ancestor)
    victim = victim_root / suffix
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_text("controller marker")
    original = shutil.copyfile

    def copy(source, destination, *args, **kwargs):
        destination = Path(destination)
        if destination.name == "z.txt" and public in destination.parents:
            path = public / ancestor
            path.rename(path.with_name(path.name + "-old"))
            path.symlink_to(
                os.path.relpath(victim_root, path.parent) if relative else victim_root,
                target_is_directory=True,
            )
        return original(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", copy)
    verify._publish_judge_copy(fs, destination)
    assert victim.read_text() == "controller marker"


@pytest.mark.parametrize("kind", ["link", "hardlink", "fifo", "directory"])
def test_copy_time_leaf_swap_never_opens_an_existing_inode(tmp_path, monkeypatch, kind):
    """Guards R3-1's leaf siblings, including a FIFO with a reader already open."""
    fs, public = _trees(tmp_path)
    destination = reachable(public)
    victim = tmp_path / "victim-a"
    victim.write_text("controller marker")
    original = shutil.copyfile
    readers = []

    def copy(source, destination, *args, **kwargs):
        destination = Path(destination)
        if destination.name == "z.txt" and public in destination.parents:
            leaf = destination.with_name("z.txt")
            if kind == "link":
                leaf.symlink_to(victim)
            elif kind == "hardlink":
                os.link(victim, leaf)
            elif kind == "fifo":
                os.mkfifo(leaf)
                readers.append(os.open(leaf, os.O_RDONLY | os.O_NONBLOCK))
            else:
                leaf.mkdir()
        return original(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", copy)
    try:
        verify._publish_judge_copy(fs, destination)
        assert victim.read_text() == "controller marker"
        assert all(os.read(reader, 4096) == b"" for reader in readers)
        assert (public / "fs/app/out/z.txt").is_file()
        assert (public / "fs/app/out/z.txt").read_text() == "agent bytes"
    finally:
        for reader in readers:
            os.close(reader)


@pytest.mark.parametrize("kind", ["abs-link", "relative-link", "file", "fifo", "tree"])
def test_preexisting_publication_entry_is_replaced_without_following_it(tmp_path, kind):
    """Guards R3-1: a retry replaces the directory entry, never its descendants."""
    fs, public = _trees(tmp_path)
    destination = reachable(public)
    public.rmdir()
    victim = tmp_path / "victim-a"
    victim.mkdir()
    (victim / "marker").write_text("controller marker")
    if kind.endswith("link"):
        public.symlink_to(
            victim if kind == "abs-link" else os.path.relpath(victim, public.parent),
            target_is_directory=True,
        )
    elif kind == "file":
        public.write_text("stale")
    elif kind == "fifo":
        os.mkfifo(public)
    else:
        (public / "fs/app/out").mkdir(parents=True)
        (public / "fs/app/out/z.txt").write_text("stale")
    verify._publish_judge_copy(fs, destination)
    assert (victim / "marker").read_text() == "controller marker"
    assert sorted(p.name for p in victim.iterdir()) == ["marker"]
    assert (public / "fs/app/out/z.txt").read_text() == "agent bytes"


@pytest.mark.parametrize("case,expected", [("oracle", 1.0), ("nop", 0.0)])
async def test_judge_records_remain_private_until_all_writes_finish(
    tmp_path, monkeypatch, case, expected
):
    """Guards R3-1's session-record sibling with the real taskmd verifier loop."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-not-a-key")
    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "cache"))
    victim = tmp_path / "victim-a"
    victim.mkdir()
    record = victim / "agent-rubric-s1-a1.json"
    record.write_text("controller marker")
    public = tmp_path / f"trial-{case}" / "verifier" / verify.JUDGE_DIR

    class SwappingJudge(_ScriptedJudge):
        swapped = False

        async def call(self, body, timeout):
            if not self.swapped:
                if public.exists():
                    public.rename(public.with_name("old-judge"))
                public.symlink_to(victim, target_is_directory=True)
                self.swapped = True
            return await super().call(body, timeout)

    result, trial, _ = await _grade(tmp_path, case, SwappingJudge(), monkeypatch)
    assert result.rewards["reward"] == expected
    assert record.read_text() == "controller marker"
    assert not public.is_symlink()
    assert (public / "agent-rubric-s1-a1.json").is_file()
    assert (public / "fs/judge/instruction.md").is_file()
    assert (trial / "verifier/review.json").is_file()


def test_ordinary_publication_preserves_empty_dirs_modes_and_inert_links(tmp_path):
    """Guards R3-1's fix against changing legitimate evidence semantics."""
    fs, public = _trees(tmp_path)
    destination = reachable(public)
    (fs / "empty").mkdir()
    script = fs / "app/out/z.txt"
    script.chmod(0o755)
    (fs / "link").symlink_to("app/out/z.txt")
    verify._publish_judge_copy(fs, destination)
    saved = public / "fs"
    assert (saved / "empty").is_dir()
    assert (saved / "app/out/z.txt").stat().st_mode & 0o777 == 0o755
    assert (saved / "link").is_symlink()
    assert os.readlink(saved / "link") == "app/out/z.txt"


@pytest.mark.parametrize("kind", ["abs-link", "relative-link", "tree", "fifo"])
def test_swap_between_retirement_and_installation_cannot_redirect_publication(
    tmp_path, monkeypatch, kind
):
    """Guards R3-1 against a live writer at the final directory-entry boundary."""
    fs, public = _trees(tmp_path)
    destination = reachable(public)
    victim = tmp_path / "victim-a"
    victim.mkdir()
    (victim / "marker").write_text("controller marker")
    rename = os.rename
    attacked = []

    def swap(source, target, *args, **kwargs):
        result = rename(source, target, *args, **kwargs)
        if str(target).startswith(".retired-"):
            attacked.append(True)
            if kind.endswith("link"):
                public.symlink_to(
                    victim
                    if kind == "abs-link"
                    else os.path.relpath(victim, public.parent),
                    target_is_directory=True,
                )
            elif kind == "tree":
                (public / "fs/app/out").mkdir(parents=True)
                (public / "fs/app/out/z.txt").symlink_to(victim / "marker")
            else:
                os.mkfifo(public)
        return result

    monkeypatch.setattr(os, "rename", swap)
    monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, swap})
    verify._publish_judge_copy(fs, destination)
    assert attacked
    assert (victim / "marker").read_text() == "controller marker"
    assert sorted(p.name for p in victim.iterdir()) == ["marker"]


@pytest.mark.parametrize("failure", ["unsupported", "cross-device", "denied"])
def test_optional_publication_refuses_without_recursive_copy(
    tmp_path, monkeypatch, caplog, failure
):
    """Guards R3-1: missing capabilities and failed rename have no copy fallback."""
    import errno

    fs, public = _trees(tmp_path)
    destination = reachable(public)
    if failure == "unsupported":
        monkeypatch.setattr(os, "supports_dir_fd", set())
    else:

        def denied(*args, **kwargs):
            raise OSError(
                errno.EXDEV if failure == "cross-device" else errno.EACCES, failure
            )

        monkeypatch.setattr(os, "rename", denied)
        monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, denied})

    def copied(*args, **kwargs):
        pytest.fail("publication must never copy into the sandbox mount")

    monkeypatch.setattr(shutil, "copytree", copied)
    verify._publish_judge_copy(fs, destination)
    assert "was not published" in caplog.text
    assert list(public.iterdir()) == []


@pytest.mark.parametrize("ending", ["success", "error", "cancel"])
async def test_private_record_custody_is_released_on_every_exit(
    tmp_path, monkeypatch, ending
):
    """Guards R3-1's private custody lifetime on return, error and cancellation."""
    import asyncio
    from types import SimpleNamespace

    verifier_dir = tmp_path / "trial/verifier"
    verifier_dir.mkdir(parents=True)
    verifier = SimpleNamespace(
        _rollout_paths=SimpleNamespace(verifier_dir=directory(verifier_dir))
    )
    held = []

    async def grade(verifier, strategy, fs):
        held.append(fs.parent)
        fs.mkdir()
        (fs / "answer").write_text("evidence")
        private = fs.parent / verify.JUDGE_DIR
        private.mkdir()
        (private / "session.json").write_text("session")
        assert not (verifier_dir / verify.JUDGE_DIR).exists()
        if ending == "error":
            raise RuntimeError("dummy judge failure")
        if ending == "cancel":
            raise asyncio.CancelledError()
        return "result"

    monkeypatch.setattr(verify, "_verify_taskmd", grade)
    if ending == "success":
        assert await verify.verify_taskmd(verifier, None) == "result"
    else:
        with pytest.raises(
            RuntimeError if ending == "error" else asyncio.CancelledError
        ):
            await verify.verify_taskmd(verifier, None)
    assert held and all(not path.exists() for path in held)
    assert (verifier_dir / verify.JUDGE_DIR / "fs/answer").read_text() == "evidence"
    assert (verifier_dir / verify.JUDGE_DIR / "session.json").read_text() == "session"


def test_controller_owned_parent_aliases_preserve_publication(tmp_path):
    """Guards R3-1's fix on controller aliases such as macOS /var and /tmp."""
    _fs, public = _trees(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    from benchflow._utils.reachable_write import pin_destination

    _pinned = pin_destination(alias)
    verify._publish_judge_copy(
        alias / "private/fs", reachable(alias / "trial/verifier" / verify.JUDGE_DIR)
    )
    assert (public / "fs/app/out/z.txt").read_text() == "agent bytes"


def test_verifier_mount_root_cannot_be_a_link(tmp_path, caplog):
    """Guards R3-1's destination anchor before any shared entry can be retired."""
    fs, public = _trees(tmp_path)
    destination = reachable(public)
    public.rmdir()
    public.parent.rmdir()
    victim = tmp_path / "victim-a"
    victim.mkdir()
    (victim / "marker").write_text("controller marker")
    public.parent.symlink_to(victim, target_is_directory=True)
    verify._publish_judge_copy(fs, destination)
    assert sorted(p.name for p in victim.iterdir()) == ["marker"]
    assert "was not published" in caplog.text
