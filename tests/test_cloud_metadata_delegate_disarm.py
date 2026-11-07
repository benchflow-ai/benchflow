"""What the delegate inventory counts is disarmed for the
agent's turn and put back before grading, and only on unchanged files.

Every stock Debian and Ubuntu image ships ``su`` setuid root, which the
inventory counts (S2-CONF-8), so refusing them all is not workable. Root
records each counted file's identity (owner, link count, size, sha256,
capability; device and inode for the record), takes its set-ID bits and
capability off, and inventories again; after the kill before grading it
puts them back on a file reached at its path without a link whose owner,
size, sha256 and link count are unchanged, a sandbox restored from a
snapshot (new inodes) included. A file the agent changed keeps nothing
back. Where the image has no Python, BenchFlow stages its own
pinned CPython, so no image is refused for lacking tools.

Here: the root program (``_delegate_disarm.DISARM_PROGRAM``) run in-process
on real files in ``tmp_path`` (set-ID bits on the test user's own files are
real; ``security.capability`` needs root, so it is stubbed), the block's
disarm with scripted sandbox answers, the restore step, the rollout's
records, and choosing or staging root's Python.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._receipt_fakes import receipt_aware
from tests.test_cloud_metadata_block import (
    AGENT_REFUSED,
    DELEGATES_CLEAN,
    ROOT_OK,
    FakeEnv,
    _rollout,
)
from tests.test_cloud_metadata_delegates import sandbox as sandbox

linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="the shell route runs on Linux sandboxes"
)

PYTHON_CLEAN = DELEGATES_CLEAN.replace("reader=getcap", "reader=python3")


def _caps_stub(monkeypatch, start: dict[str, bytes] | None = None) -> dict:
    """A stand-in for ``security.capability`` keyed by inode (setting the
    real attribute needs root). Returns the live table."""
    table: dict[tuple[int, int], bytes] = {}
    for path, value in (start or {}).items():
        st = os.stat(path)
        table[(st.st_dev, st.st_ino)] = value

    def key(target):
        st = os.fstat(target) if isinstance(target, int) else os.stat(target)
        return (st.st_dev, st.st_ino)

    def getxattr(target, name, *, follow_symlinks=True):
        assert name == "security.capability"
        if key(target) in table:
            return table[key(target)]
        raise OSError(errno.ENODATA, "No data available")

    def removexattr(target, name, *, follow_symlinks=True):
        assert name == "security.capability"
        table.pop(key(target))

    def setxattr(target, name, value, flags=0, *, follow_symlinks=True):
        assert name == "security.capability"
        table[key(target)] = bytes(value)

    monkeypatch.setattr(os, "getxattr", getxattr, raising=False)
    monkeypatch.setattr(os, "removexattr", removexattr, raising=False)
    monkeypatch.setattr(os, "setxattr", setxattr, raising=False)
    return table


def _run(monkeypatch, *argv: str, proc: Path | None = None) -> tuple[str, int]:
    """Run DISARM_PROGRAM as root's sandbox would, in-process; the test's
    own uid stands in for root's. A strip's paths (``argv[2:]``) go on its
    stdin, as root's command sends them. A restore looks at *proc* for live
    processes: an empty stand-in /proc by default (this host's own has
    processes of other users without no_new_privs)."""
    from benchflow.sandbox import _delegate_disarm

    action, record, *rest = argv
    extra: list[str] = []
    stdin = b""
    if action == "strip":
        stdin = "".join(f"{h}\n" for h in rest).encode()
    else:
        extra = [str(proc if proc is not None else Path(tempfile.mkdtemp()))]
    out = io.BytesIO()
    saved = sys.argv, sys.stdout, sys.stdin
    sys.argv = ["-c", action, record, f"0,{os.getuid()}", *extra]
    sys.stdout = SimpleNamespace(buffer=out)
    sys.stdin = SimpleNamespace(buffer=io.BytesIO(stdin))
    try:
        with pytest.raises(SystemExit) as ended:
            exec(
                compile(_delegate_disarm.DISARM_PROGRAM, "<disarm>", "exec"),
                {"__name__": "__main__"},
            )
    finally:
        sys.argv, sys.stdout, sys.stdin = saved
    return out.getvalue().decode(), ended.value.code


def _proc(tmp_path: Path, *statuses: str) -> Path:
    """A stand-in /proc with one process per status text (pids 101, 102...)."""
    root = tmp_path / "proc"
    root.mkdir(exist_ok=True)
    (root / "self").mkdir(exist_ok=True)
    for pid, status in enumerate(statuses, start=101):
        (root / str(pid)).mkdir()
        (root / str(pid) / "status").write_text(status)
    return root


def _status(name: str, uids: str, *, nnp: str | None = None, state: str = "S") -> str:
    return f"Name:\t{name}\nState:\t{state} (sleeping)\nUid:\t{uids}\n" + (
        f"NoNewPrivs:\t{nnp}\n" if nnp is not None else ""
    )


def _file(path: Path, mode: int, body: bytes = b"#!/bin/sh\nexit 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    path.chmod(mode)
    return path


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def _hex(path: Path) -> str:
    return str(path).encode().hex()


V2_NET_ADMIN = (
    (0x02000001).to_bytes(4, "little") + (1 << 12).to_bytes(4, "little") + (b"\0" * 12)
)


def test_strip_records_each_files_identity_and_takes_its_privilege_off(
    monkeypatch, tmp_path
):
    """Option A: the record (path, mode, owner, inode, link
    count, size, sha256, capability) is written before anything changes, and
    afterwards no set-ID bit or capability is left on the files the inventory
    counted."""
    su = _file(tmp_path / "usr/bin/su", 0o4755)
    wall = _file(tmp_path / "usr/bin/tool", 0o2755)
    ptp = _file(tmp_path / "usr/lib/gst-ptp-helper", 0o755)
    record = tmp_path / "opt/benchflow/delegates"
    caps = _caps_stub(monkeypatch, {str(ptp): V2_NET_ADMIN})

    out, code = _run(monkeypatch, "strip", str(record), _hex(su), _hex(wall), _hex(ptp))

    assert code == 0, out
    assert out.splitlines()[-1] == "BF metadata-delegates-strip-done 3 0"
    assert (_mode(su), _mode(wall), _mode(ptp)) == (0o755, 0o755, 0o755)
    assert caps == {}
    assert _mode(record) == 0o700
    saved = json.loads((record / "record.json").read_text())["entries"]
    by_path = {bytes.fromhex(e["paths"][0]).decode(): e for e in saved}
    st = os.stat(su)
    assert by_path[str(su)] == {
        "paths": [_hex(su)],
        "dev": st.st_dev,
        "ino": st.st_ino,
        "links": 1,
        "stripped_mode": 0o755,
        "uid": st.st_uid,
        "gid": st.st_gid,
        "size": st.st_size,
        "mode": 0o4755,
        "sha256": hashlib.sha256(su.read_bytes()).hexdigest(),
        "caps": None,
    }
    assert by_path[str(ptp)]["caps"] == V2_NET_ADMIN.hex()
    assert f"BF metadata-delegates-stripped {_hex(su)} 4755 " in out


def test_restore_puts_back_the_exact_mode_and_capability(monkeypatch, tmp_path):
    """Option A: before grading each unchanged file gets its
    exact recorded mode and capability back, checked by fstat and a read
    back, and the record is gone."""
    su = _file(tmp_path / "usr/bin/su", 0o4755)
    ptp = _file(tmp_path / "usr/lib/gst-ptp-helper", 0o755)
    record = tmp_path / "opt/benchflow/delegates"
    caps = _caps_stub(monkeypatch, {str(ptp): V2_NET_ADMIN})
    _run(monkeypatch, "strip", str(record), _hex(su), _hex(ptp))
    assert caps == {}

    out, code = _run(monkeypatch, "restore", str(record))

    assert code == 0, out
    assert f"BF metadata-delegates-restored {_hex(su)} 4755:exact" in out
    assert _mode(su) == 0o4755
    st = os.stat(ptp)
    assert caps[(st.st_dev, st.st_ino)] == V2_NET_ADMIN
    assert out.splitlines()[-1] == "BF metadata-delegates-restore-done 2 0 0"
    assert not (record / "record.json").exists()
    # Run again: nothing is left to do.
    out, code = _run(monkeypatch, "restore", str(record))
    assert (code, out) == (0, "BF metadata-delegates-restore-done 0 0 0\n")


def _owned_by_another(monkeypatch, path: Path, field: str) -> None:
    """Make fstat report *path*'s inode with another owner uid or gid (a
    test cannot chown to another user without root)."""
    real, ino = os.fstat, os.stat(path).st_ino

    def fstat(fd):
        st = real(fd)
        if st.st_ino != ino:
            return st
        fields = list(st[:10])
        fields[4 if field == "uid" else 5] += 1
        return os.stat_result(fields)

    monkeypatch.setattr(os, "fstat", fstat)


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ("content", "changed:size,sha256"),
        ("extra-hard-link", "changed:links"),
        ("owner", "changed:uid"),
        ("group", "changed:gid"),
        ("symlink", "gone-or-linked:"),
        ("linked-folder", "gone-or-linked:"),
    ],
)
def test_a_file_the_agent_changed_gets_nothing_back(monkeypatch, tmp_path, change, why):
    """Option A's identity rule: a stripped file whose content,
    owner or group changed, that gained a hard link (a second name the agent
    could keep for the set-ID inode), or whose path now has a link in its
    place or in a folder's, is never given set-ID bits back; it stays
    stripped and is reported with what changed."""
    su = _file(tmp_path / "srv/bin/su", 0o4755)
    other = _file(tmp_path / "srv/other", 0o755)
    record = tmp_path / "opt/benchflow/delegates"
    _caps_stub(monkeypatch)
    _run(monkeypatch, "strip", str(record), _hex(su))
    if change == "content":
        with open(su, "ab") as f:
            f.write(b"# changed\n")
    elif change == "extra-hard-link":
        os.link(su, tmp_path / "srv/kept-name")
    elif change in ("owner", "group"):
        _owned_by_another(monkeypatch, su, "uid" if change == "owner" else "gid")
    elif change == "symlink":
        su.unlink()
        su.symlink_to(other)
    else:
        moved = tmp_path / "srv/bin.real"
        (tmp_path / "srv/bin").rename(moved)
        (tmp_path / "srv/bin").symlink_to(moved)

    out, code = _run(monkeypatch, "restore", str(record))

    assert code == 0, out
    assert f"BF metadata-delegates-kept {_hex(su)} {why}" in out
    assert out.splitlines()[-1] == "BF metadata-delegates-restore-done 0 1 0"
    assert not any(
        os.lstat(p).st_mode & (stat.S_ISUID | stat.S_ISGID)
        for p in (tmp_path / "srv").rglob("*")
        if not os.path.islink(p)
    )


def test_a_file_restored_from_a_snapshot_gets_its_bits_back(monkeypatch, tmp_path):
    """Option A's content rule on a sandbox restored from a
    snapshot (a branch's child or parent), where every file has a new inode
    by construction: same path, owner, size, sha256 and link count, so the
    exact mode and the capability go back on the new inodes."""
    su = _file(tmp_path / "usr/bin/su", 0o4755)
    ptp = _file(tmp_path / "usr/lib/gst-ptp-helper", 0o755)
    record = tmp_path / "opt/benchflow/delegates"
    caps = _caps_stub(monkeypatch, {str(ptp): V2_NET_ADMIN})
    _run(monkeypatch, "strip", str(record), _hex(su), _hex(ptp))
    inodes = {os.stat(su).st_ino, os.stat(ptp).st_ino}
    # The snapshot's copies: the stripped bytes and mode on new inodes, all
    # made before any is renamed over its path, so no freed inode number can
    # be reused for one of them; the record stays, as a snapshot keeps it.
    copies = [
        (_file(p.with_name(p.name + ".snap"), 0o755, p.read_bytes()), p)
        for p in (su, ptp)
    ]
    for copy, path in copies:
        os.replace(copy, path)
    assert not inodes & {os.stat(su).st_ino, os.stat(ptp).st_ino}
    assert caps == {}

    out, code = _run(monkeypatch, "restore", str(record))

    assert code == 0, out
    assert f"BF metadata-delegates-restored {_hex(su)} 4755:exact" in out
    assert f"BF metadata-delegates-restored {_hex(ptp)} 755:exact" in out
    assert _mode(su) == 0o4755
    st = os.stat(ptp)
    assert caps == {(st.st_dev, st.st_ino): V2_NET_ADMIN}


def test_a_strip_lists_what_roots_record_held_already(monkeypatch, tmp_path):
    """Option A on a snapshot: a strip lists, as ``recorded``
    lines, the files the record held already (disarmed before the snapshot),
    with or without files of its own to strip, so the host remembers them
    and they are put back before this sandbox's grading."""
    from benchflow.sandbox import _delegate_disarm

    su = _file(tmp_path / "usr/bin/su", 0o4755)
    tool = _file(tmp_path / "usr/bin/tool", 0o2755)
    record = tmp_path / "opt/benchflow/delegates"
    _caps_stub(monkeypatch)
    _run(monkeypatch, "strip", str(record), _hex(su))

    out, code = _run(monkeypatch, "strip", str(record))
    assert code == 0, out
    st = su.stat()
    assert out.splitlines() == [
        f"BF metadata-delegates-recorded {_hex(su)} 4755 {st.st_uid} {st.st_gid} "
        f"{st.st_ino} 1 {st.st_size} {hashlib.sha256(su.read_bytes()).hexdigest()} -",
        "BF metadata-delegates-strip-done 1 0",
    ]
    listed = _delegate_disarm.read_strip(out, code)
    assert listed.ok and [(e["path"], e["earlier"]) for e in listed.stripped] == [
        (str(su), True)
    ]

    out, code = _run(monkeypatch, "strip", str(record), _hex(tool))
    assert code == 0 and out.splitlines()[-1] == "BF metadata-delegates-strip-done 2 0"
    assert [line.split()[1] for line in out.splitlines()[:-1]] == [
        "metadata-delegates-stripped",
        "metadata-delegates-recorded",
    ]
    out, code = _run(monkeypatch, "restore", str(record))
    assert code == 0 and (_mode(su), _mode(tool)) == (0o4755, 0o2755), out


def test_a_hard_link_is_one_entry_restored_once(monkeypatch, tmp_path):
    """Option A: two names of one inode share one record entry,
    so the second name never records the stripped mode over the first."""
    su = _file(tmp_path / "usr/bin/su", 0o4755)
    alias = tmp_path / "usr/sbin/su"
    alias.parent.mkdir(parents=True)
    os.link(su, alias)
    record = tmp_path / "opt/benchflow/delegates"
    _caps_stub(monkeypatch)

    _run(monkeypatch, "strip", str(record), _hex(su), _hex(alias))
    saved = json.loads((record / "record.json").read_text())["entries"]
    assert [sorted(e["paths"]) for e in saved] == [sorted([_hex(su), _hex(alias)])]
    assert saved[0]["mode"] == 0o4755

    out, code = _run(monkeypatch, "restore", str(record))
    assert code == 0 and _mode(su) == _mode(alias) == 0o4755, out


def test_a_strip_that_fails_names_the_file_and_fails(monkeypatch, tmp_path):
    """Option A: a file that cannot be disarmed (a read-only
    mount, say) is named with its errno and the strip fails, so the block
    refuses the start."""
    su = _file(tmp_path / "usr/bin/su", 0o4755)
    record = tmp_path / "opt/benchflow/delegates"
    _caps_stub(monkeypatch)
    real = os.fchmod

    def fchmod(fd, mode):
        if os.fstat(fd).st_ino == os.stat(su).st_ino:
            raise OSError(errno.EROFS, "Read-only file system")
        return real(fd, mode)

    monkeypatch.setattr(os, "fchmod", fchmod)
    out, code = _run(monkeypatch, "strip", str(record), _hex(su))

    assert code == 7
    assert f"BF metadata-delegates-strip-failed {_hex(su)} EROFS" in out
    assert _mode(su) == 0o4755


def test_the_record_folder_must_be_roots_alone(monkeypatch, tmp_path):
    """Option A: a record anyone else could write is no
    record; a folder others may write on its way (not sticky) fails it."""
    su = _file(tmp_path / "usr/bin/su", 0o4755)
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    _caps_stub(monkeypatch)

    out, code = _run(monkeypatch, "strip", str(shared / "delegates"), _hex(su))

    assert code == 7
    assert "BF metadata-delegates-strip-failed" in out or "disarm-error" in out
    assert _mode(su) == 0o4755


# --- the review of be4b7369 ---------------------------------------------------

_NOBODY = "65534\t65534\t65534\t65534"


@pytest.mark.parametrize(
    ("status", "withheld"),
    [
        (_status("sshd", _NOBODY), True),
        (_status("www-data", "33\t33\t33\t33", nnp="0"), True),
        (_status("svc", _NOBODY, nnp="1"), False),
        (_status("cron", "0\t0\t0\t0"), False),
        (_status("half", "65534\t65534\t0\t65534"), False),
        (_status("gone", _NOBODY, state="Z"), False),
    ],
    ids=["no-nnp-line", "nnp-0", "nnp", "root", "saved-uid-0", "zombie"],
)
def test_a_live_process_outside_no_new_privs_withholds_the_restore(
    monkeypatch, tmp_path, status, withheld
):
    """Review 1(b): processes survive the kill before grading
    (kept services, every other user's). While one that is not root (no real,
    effective or saved uid 0) runs without no_new_privs, nothing is put back:
    it could use a restored su or capability during grading. The process is
    named (pid, uid, comm), the files stay disarmed and recorded. A process
    under no_new_privs, root's, one with a saved uid 0 and a zombie do not
    count."""
    from benchflow.sandbox import _delegate_disarm

    su = _file(tmp_path / "usr/bin/su", 0o4755)
    record = tmp_path / "opt/benchflow/delegates"
    _caps_stub(monkeypatch)
    _run(monkeypatch, "strip", str(record), _hex(su))

    out, code = _run(monkeypatch, "restore", str(record), proc=_proc(tmp_path, status))

    assert code == 0, out
    report = _delegate_disarm.read_restore(out, code)
    assert not report.failed
    if withheld:
        name = status.split("\t", 1)[1].split("\n", 1)[0]
        uid = status.split("Uid:\t", 1)[1].split("\t", 1)[0]
        assert out.splitlines() == [
            f"BF metadata-delegates-live 101 {uid} {name.encode().hex()}",
            f"BF metadata-delegates-kept {_hex(su)} live-non-nnp-process",
            "BF metadata-delegates-restore-done 0 1 0",
        ]
        assert report.live == [{"pid": 101, "uid": int(uid), "comm": name}]
        assert _mode(su) == 0o755 and (record / "record.json").exists()
    else:
        assert f"BF metadata-delegates-restored {_hex(su)} 4755:exact" in out
        assert report.live == [] and _mode(su) == 0o4755


def test_an_unreadable_proc_is_benchflows_failure_and_restores_nothing(
    monkeypatch, tmp_path
):
    """Review 1(b) and 3: with no /proc to look at, nothing is
    put back and the restore is BenchFlow's failure (unscored), not a file
    the agent changed."""
    from benchflow.sandbox import _delegate_disarm

    su = _file(tmp_path / "usr/bin/su", 0o4755)
    record = tmp_path / "opt/benchflow/delegates"
    _caps_stub(monkeypatch)
    _run(monkeypatch, "strip", str(record), _hex(su))

    out, code = _run(monkeypatch, "restore", str(record), proc=tmp_path / "no-proc")

    assert (out, code) == ("BF metadata-delegates-disarm-error ENOENT\n", 7)
    assert _mode(su) == 0o755 and (record / "record.json").exists()
    report = _delegate_disarm.read_restore(out, code)
    assert [r["outcome"] for r in report.rows] == ["failed"]


def test_both_names_of_a_hard_link_overlay_split_get_their_bits_back(
    monkeypatch, tmp_path
):
    """Review 2 (the reviewer's overlay_split.py): on overlayfs
    with index=off a strip copies each name of a hard-linked set-ID file up to
    a file of its own, while both report one inode number. The restore checks
    and puts back every recorded name; it stopped after the first, leaving
    /usr/sbin/su at 755 with no warning."""
    a = _file(tmp_path / "usr/bin/su", 0o4755)
    b = _file(tmp_path / "usr/sbin/su", 0o4755)
    split = {(os.stat(p).st_dev, os.stat(p).st_ino) for p in (a, b)}
    real_fstat = os.fstat

    def one_number(fd):
        st = real_fstat(fd)
        if (st.st_dev, st.st_ino) not in split:
            return st
        fields = list(st[:10])
        fields[1] = 424242
        return os.stat_result(fields)

    monkeypatch.setattr(os, "fstat", one_number)
    _caps_stub(monkeypatch)
    record = tmp_path / "opt/benchflow/delegates"

    out, code = _run(monkeypatch, "strip", str(record), _hex(a), _hex(b))
    assert code == 0 and _mode(a) == _mode(b) == 0o755, out
    saved = json.loads((record / "record.json").read_text())["entries"]
    assert [sorted(e["paths"]) for e in saved] == [sorted([_hex(a), _hex(b)])]

    out, code = _run(monkeypatch, "restore", str(record))
    assert code == 0, out
    assert f"BF metadata-delegates-restored {_hex(a)} 4755:exact" in out
    assert f"BF metadata-delegates-restored {_hex(b)} 4755:exact" in out
    assert _mode(a) == _mode(b) == 0o4755


@pytest.mark.parametrize("err", [errno.EIO, errno.EACCES, errno.EMFILE])
def test_an_error_of_benchflows_own_is_a_failure_and_stays_recorded(
    monkeypatch, tmp_path, err
):
    """Review 3: an I/O error, EACCES or EMFILE opening a file
    is BenchFlow's failure (``restore-failed``, the trial unscored), not the
    agent's change (it was ``kept gone-or-linked``); the file stays recorded.
    A file that is gone is the agent's change."""
    from benchflow.sandbox import _delegate_disarm

    su = _file(tmp_path / "usr/bin/su", 0o4755)
    tool = _file(tmp_path / "usr/bin/tool", 0o4755)
    record = tmp_path / "opt/benchflow/delegates"
    _caps_stub(monkeypatch)
    _run(monkeypatch, "strip", str(record), _hex(su), _hex(tool))
    real_open = os.open

    def failing(path, flags, mode=0o777, *, dir_fd=None):
        if path == b"su" and dir_fd is not None:
            raise OSError(err, os.strerror(err))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", failing)
    tool.unlink()

    out, code = _run(monkeypatch, "restore", str(record))

    name = errno.errorcode[err]
    assert code == 7, out
    assert f"BF metadata-delegates-restore-failed {_hex(su)} {name}" in out
    assert f"BF metadata-delegates-kept {_hex(tool)} gone-or-linked:ENOENT" in out
    report = _delegate_disarm.read_restore(out, code)
    assert [(r["outcome"], r["why"]) for r in report.failed] == [("failed", name)]
    left = json.loads((record / "record.json").read_text())["entries"]
    assert sorted(h for e in left for h in e["paths"]) == sorted([_hex(su), _hex(tool)])


def test_a_mode_changed_since_the_strip_keeps_the_file_disarmed(monkeypatch, tmp_path):
    """Review 5: the restore also needs the mode the strip
    left. A file whose mode changed since (the agent fixed its own tool, say)
    keeps it and gets no set-ID bit back."""
    tool = _file(tmp_path / "srv/bin/tool", 0o4755)
    record = tmp_path / "opt/benchflow/delegates"
    _caps_stub(monkeypatch)
    _run(monkeypatch, "strip", str(record), _hex(tool))
    tool.chmod(0o700)

    out, code = _run(monkeypatch, "restore", str(record))

    assert code == 0, out
    assert f"BF metadata-delegates-kept {_hex(tool)} changed:mode" in out
    assert _mode(tool) == 0o700


def test_the_sandbox_users_own_set_id_files_count_only_beyond_its_identity(
    monkeypatch, tmp_path
):
    """Review 5: a set-ID file the sandbox user owns is not
    counted (never disarmed, never "restored" over the agent's own fix) only
    when it grants nothing beyond that user's own identity: setuid to it, or
    setgid to a group it already belongs to, primary or supplementary, read
    in the sandbox. Setgid to any other group (root, disk, shadow) counts,
    and a file capability counts whoever owns the file."""
    import pwd

    from benchflow.sandbox.lockdown import _delegate_report

    me = pwd.getpwuid(os.getuid())
    mine = os.getgrouplist(me.pw_name, me.pw_gid)
    setuid = _file(tmp_path / "home/agent/tool", 0o4755)
    own_group = _file(tmp_path / "home/agent/own-group", 0o2755)
    os.chown(own_group, -1, me.pw_gid)
    own_group.chmod(0o2755)
    supplementary = next((g for g in mine if g != me.pw_gid), None)
    extra = tmp_path / "home/agent/extra-group"
    if supplementary is not None:
        _file(extra, 0o755)
        os.chown(extra, -1, supplementary)
        extra.chmod(0o2755)
    capped = _file(tmp_path / "home/agent/capped", 0o755)
    _caps_stub(monkeypatch, {str(capped): V2_NET_ADMIN})

    report = _delegate_report(*_read_inventory(tmp_path, sandbox_user=me.pw_name))
    assert report.complete, report.problems
    assert [f["path"] for f in report.findings] == [str(capped)]
    assert report.sandbox_owned == (3 if supplementary is not None else 2)
    assert report.summary()["sandbox_owned_setid_not_counted"] == report.sandbox_owned

    # The same files when their group is none of the sandbox user's (root,
    # disk, shadow: a test cannot chgrp to a group it is not in, so the
    # sandbox user's groups stand in): setgid counts, setuid still does not.
    monkeypatch.setattr(os, "getgrouplist", lambda name, gid: [4242424])
    report = _delegate_report(*_read_inventory(tmp_path, sandbox_user=me.pw_name))
    found = {f["path"]: f["why"] for f in report.findings}
    assert found.pop(str(own_group)) == "setgid"
    if supplementary is not None:
        assert found.pop(str(extra)) == "setgid"
    assert found == {str(capped): "file capabilities cap_net_admin=ep"}
    assert report.sandbox_owned == 1

    # Without the sandbox user, every one of them counts.
    report = _delegate_report(*_read_inventory(tmp_path))
    assert str(setuid) in {f["path"] for f in report.findings}


def test_entries_of_different_snapshot_generations_never_collapse(
    monkeypatch, tmp_path
):
    """Review 7: the record was read into a dict keyed by
    device and inode, so two entries sharing them (from different snapshot
    generations) became one and one file was never put back."""
    record = tmp_path / "opt/benchflow/delegates"
    record.mkdir(parents=True)
    record.chmod(0o700)
    files = [
        _file(tmp_path / "usr/bin/su", 0o755),
        _file(tmp_path / "usr/bin/newgrp", 0o755, b"#!/bin/sh\nexit 2\n"),
    ]

    def entry(path):
        st = os.stat(path)
        return {
            "paths": [_hex(path)],
            "dev": 7,
            "ino": 9,
            "links": 1,
            "uid": st.st_uid,
            "gid": st.st_gid,
            "size": st.st_size,
            "mode": 0o4755,
            "stripped_mode": 0o755,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "caps": None,
        }

    (record / "record.json").write_text(
        json.dumps({"version": 1, "entries": [entry(f) for f in files]})
    )
    _caps_stub(monkeypatch)

    out, code = _run(monkeypatch, "restore", str(record))

    assert code == 0, out
    assert [_mode(f) for f in files] == [0o4755, 0o4755]


def test_strip_takes_its_paths_on_stdin_however_many(monkeypatch, tmp_path):
    """Review 8: the paths to strip come on the program's
    stdin, so its arguments stay the same size; 1,500 paths (about 280 KB of
    hex, past Linux's 128 KiB limit on one argument) all strip."""
    from benchflow.sandbox import _delegate_disarm

    files = [_file(tmp_path / "many" / f"f{i:04d}", 0o4755) for i in range(1500)]
    record = tmp_path / "opt/benchflow/delegates"
    _caps_stub(monkeypatch)

    out, code = _run(monkeypatch, "strip", str(record), *(_hex(f) for f in files))

    assert (
        code == 0 and out.splitlines()[-1] == "BF metadata-delegates-strip-done 1500 0"
    )
    assert {_mode(f) for f in files} == {0o755}
    command = _delegate_disarm.disarm_cmd(
        "/opt/runtime/rt/py/python3",
        "strip",
        targets=tuple(str(f).encode() for f in files),
    )
    program_args = command[: command.index("<<'BF_DELEGATE_TARGETS'")]
    assert _hex(files[0]) not in program_args and _hex(files[-1]) in command


@pytest.mark.asyncio
async def test_a_snapshots_record_with_no_python_to_restore_it_refuses_retryably(
    monkeypatch,
):
    """Review 4: an isolated branch child or a checkpoint retry
    whose sandbox holds root's record (files disarmed before the snapshot),
    whose inventory finishes through getcap and where no Python can be had,
    used to start with nothing to put the files back: it now refuses, and a
    retry may find a Python."""
    from benchflow._utils.scoring import is_unrecoverable_startup_error
    from benchflow.sandbox import _delegate_disarm
    from benchflow.sandbox.lockdown import CloudMetadataPowerError, block_cloud_metadata

    async def no_python(env):
        return None, "its staging folder could not be made (no report)"

    monkeypatch.setattr(_delegate_disarm, "find_python", no_python)

    class Env(FakeEnv):
        @receipt_aware
        async def exec(self, command, user=None, **kwargs):
            if "BF metadata-delegates-scan" in command:
                return SimpleNamespace(
                    stdout="BF metadata-delegates-record\n" + DELEGATES_CLEAN,
                    return_code=0,
                )
            return await super().exec(command, user=user, **kwargs)

    env = Env(ROOT_OK, AGENT_REFUSED, "BF metadata-hits 1\n")
    with pytest.raises(CloudMetadataPowerError) as refused:
        await block_cloud_metadata(env, "agent")

    message = str(refused.value)
    assert "root's record of files disarmed before this sandbox's snapshot" in message
    assert refused.value.record["state"] == "refused"
    assert not is_unrecoverable_startup_error(message)


# --- a Python whose os module has no xattr calls ------------------------------

_NO_OS_XATTR = ("getxattr", "setxattr", "removexattr", "listxattr")


def _read_inventory(root: Path, sandbox_user: str = "") -> tuple[str, int]:
    """Run the inventory's Python program on *root* in-process, as is."""
    from benchflow.sandbox import lockdown

    program = lockdown._DELEGATE_PY_READER.replace("@NAMES@", repr(lockdown._CAP_NAMES))
    buf = io.BytesIO()
    saved = sys.argv, sys.stdout
    sys.argv = ["-c", str(root), "2000", *([sandbox_user] if sandbox_user else [])]
    sys.stdout = SimpleNamespace(buffer=buf)
    try:
        with pytest.raises(SystemExit) as ended:
            exec(
                compile(program, "<delegate reader>", "exec"), {"__name__": "__main__"}
            )
    finally:
        sys.argv, sys.stdout = saved
    return buf.getvalue().decode(), ended.value.code


@linux_only
def test_the_c_librarys_xattr_calls_stand_in_for_os_ones(monkeypatch, tmp_path):
    """Guards the fix on BenchFlow's staged musl CPython, whose os module has
    no getxattr (python-build-standalone's musl builds leave it out): root's
    programs reach the C library's calls through ctypes, with os's values
    and errnos, values longer than the first buffer included, and never
    through a link. Without it the Alpine inventory crashed and refused."""
    from benchflow.sandbox import _delegate_disarm

    real_set, real_get = os.setxattr, os.getxattr
    target = tmp_path / "f"
    target.write_bytes(b"x")
    link = tmp_path / "l"
    link.symlink_to(target)
    small, big = b"\x01\x00\x00\x02" * 5, bytes(range(256)) * 2
    try:
        real_set(target, "user.bf-small", small)
        real_set(target, "user.bf-big", big)
    except OSError as e:
        pytest.skip(f"no user extended attributes on {tmp_path} ({e})")
    for name in _NO_OS_XATTR:
        monkeypatch.delattr(os, name, raising=False)
    calls: dict = {"os": os, "errno": errno}
    exec(compile(_delegate_disarm.XATTR_SHIM, "<xattr shim>", "exec"), calls)
    lget, fget, fset, frem = (
        calls[n] for n in ("lgetxattr", "fgetxattr", "fsetxattr", "fremovexattr")
    )

    assert lget(str(target).encode(), "user.bf-small") == small
    assert lget(str(target).encode(), "user.bf-big") == big
    with pytest.raises(OSError) as no_link:
        lget(str(link).encode(), "user.bf-small")
    assert no_link.value.errno == errno.ENODATA
    fd = os.open(target, os.O_RDONLY)
    try:
        assert fget(fd, "user.bf-big") == big
        fset(fd, "user.bf-new", b"\x00v\x00")
        assert real_get(target, "user.bf-new") == b"\x00v\x00"
        frem(fd, "user.bf-new")
        for call in (fget, frem):
            with pytest.raises(OSError) as gone:
                call(fd, "user.bf-new")
            assert gone.value.errno == errno.ENODATA
    finally:
        os.close(fd)


@linux_only
def test_inventory_strip_and_restore_run_on_a_python_without_os_xattr(
    monkeypatch, tmp_path
):
    """End to end on such a Python (the C library's calls,
    real files): the inventory is complete and finds the setuid file, the
    strip takes the bit off and the restore puts the exact mode back."""
    from benchflow.sandbox.lockdown import _delegate_report

    su = _file(tmp_path / "usr/bin/su", 0o4755)
    _file(tmp_path / "usr/bin/plain", 0o755)
    for name in _NO_OS_XATTR:
        monkeypatch.delattr(os, name, raising=False)

    out, code = _read_inventory(tmp_path / "usr")
    report = _delegate_report(out, code)
    assert report.complete and report.reader == "python3", (out, report.problems)
    assert [f["path"] for f in report.findings] == [str(su)]

    record = str(tmp_path / "opt/benchflow/delegates")
    out, code = _run(monkeypatch, "strip", record, _hex(su))
    assert code == 0 and _mode(su) == 0o755, out
    out, code = _run(monkeypatch, "restore", record)
    assert code == 0 and "4755:exact" in out and _mode(su) == 0o4755, out


def test_a_python_with_no_way_to_read_attributes_reads_nothing_and_says_so(
    monkeypatch, tmp_path
):
    """Guards the fix against a fail-open: on a Python with neither os's
    xattr calls nor a working ctypes, a missing call is never taken for a
    file without capabilities. The inventory reads nothing and is
    incomplete, which refuses, and the strip refuses to run, each saying why."""
    from benchflow.sandbox import _delegate_disarm, lockdown

    su = _file(tmp_path / "usr/bin/su", 0o4755)
    for name in _NO_OS_XATTR:
        monkeypatch.delattr(os, name, raising=False)
    monkeypatch.setitem(sys.modules, "ctypes", None)

    out, code = _read_inventory(tmp_path)
    report = lockdown._delegate_report(out, code)
    assert code == 7 and report.reader == "none", out
    assert report.problems == [lockdown._DELEGATE_WHY["xattr"]]
    assert report.deterministic and not report.findings
    assert "no way to read file capabilities" in lockdown._delegate_refusal(report)

    out, code = _run(monkeypatch, "strip", str(tmp_path / "rec"), _hex(su))
    assert (out, code) == ("BF metadata-delegates-disarm-error xattr\n", 7)
    assert _mode(su) == 0o4755 and not (tmp_path / "rec").exists()
    strip = _delegate_disarm.read_strip(out, code)
    assert strip.problems == [_delegate_disarm._DISARM_ERRORS["xattr"]]


# --- the block disarms instead of refusing ------------------------------------


def _python_env(monkeypatch, answers: dict[str, list]) -> FakeEnv:
    """A FakeEnv whose sandbox has root's Python, answering the inventory,
    the strip and the restore from *answers* in turn."""
    from benchflow.sandbox import _delegate_disarm

    async def kit(env):
        return _delegate_disarm.DelegatePython("/opt/runtime/rt/py/python3", "kit"), ""

    monkeypatch.setattr(_delegate_disarm, "find_python", kit)

    class Env(FakeEnv):
        @receipt_aware
        async def exec(self, command, user=None, **kwargs):
            tail = re.search(r"' (strip|restore) /", command)
            if tail is not None:
                name = tail[1]
            elif "BF metadata-delegates-scan" in command:
                name = "inventory"
            else:
                return await super().exec(command, user=user, **kwargs)
            self.seen.append(name)
            self.sent.append(command)
            out, code = answers[name].pop(0)
            return SimpleNamespace(stdout=out, return_code=code)

    env = Env(ROOT_OK, AGENT_REFUSED, "BF metadata-hits 1\n")
    env.seen = []
    env.sent = []
    return env


_SU_HEX = b"/usr/bin/su".hex()
_SU_LINE = f"BF metadata-delegates-setid u-0 {_SU_HEX}\n"
_SHA = "ab" * 32
_STRIPPED = (
    f"BF metadata-delegates-stripped {_SU_HEX} 4755 0 0 1234 1 55680 {_SHA} -\n"
    "BF metadata-delegates-strip-done 1 0\n"
)


@pytest.mark.asyncio
async def test_a_counted_launcher_is_disarmed_and_the_agent_starts(monkeypatch):
    """Option A against refusing every stock Debian and Ubuntu
    image: su is disarmed, the second inventory is clean, the block is on,
    and the record names what was disarmed with its identity."""
    from benchflow.sandbox import _delegate_disarm
    from benchflow.sandbox.lockdown import block_cloud_metadata

    env = _python_env(
        monkeypatch,
        {
            "inventory": [(_SU_LINE + PYTHON_CLEAN, 0), (PYTHON_CLEAN, 0)],
            "strip": [(_STRIPPED, 0)],
        },
    )
    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "on", record
    assert env.seen == ["inventory", "strip", "inventory"]
    assert record["delegates_disarmed"] == [
        {
            "path": "/usr/bin/su",
            "why": "setuid root",
            "mode": "4755",
            "uid": 0,
            "gid": 0,
            "inode": 1234,
            "links": 1,
            "size": 55680,
            "sha256": _SHA,
        }
    ]
    assert record["delegate_inventory"]["python"] == "kit"
    state = getattr(env, _delegate_disarm.DISARMED_ATTR)
    assert state["python"] == "/opt/runtime/rt/py/python3"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("strip", "after", "named"),
    [
        (
            (
                f"BF metadata-delegates-strip-failed {_SU_HEX} EROFS\n"
                "BF metadata-delegates-strip-done 1 1\n",
                7,
            ),
            None,
            "could not take the set-ID bit or file capability off /usr/bin/su (EROFS)",
        ),
        ((_STRIPPED, 0), (_SU_LINE + PYTHON_CLEAN, 0), "still counts /usr/bin/su"),
        (("", 125), None, "could not take the set-ID bit"),
    ],
    ids=["strip-failed", "still-counted", "no-report"],
)
async def test_a_file_that_cannot_be_disarmed_refuses_by_name(
    monkeypatch, strip, after, named
):
    """Option A: any strip or second-inventory failure refuses
    the start (fail closed), naming the file, with the refusal's record."""
    from benchflow.sandbox.lockdown import CloudMetadataPowerError, block_cloud_metadata

    answers = {"inventory": [(_SU_LINE + PYTHON_CLEAN, 0)], "strip": [strip]}
    if after is not None:
        answers["inventory"].append(after)
    env = _python_env(monkeypatch, answers)

    with pytest.raises(CloudMetadataPowerError) as refused:
        await block_cloud_metadata(env, "agent")

    assert named in str(refused.value)
    assert refused.value.record["state"] == "refused"
    assert "--allow-cloud-metadata" in str(refused.value)


@pytest.mark.asyncio
async def test_a_sandbox_restored_from_a_snapshot_lists_roots_record(monkeypatch):
    """Option A on a branch: a sandbox restored from a snapshot
    has su disarmed already, so the inventory counts nothing, but it reports
    root's record. Root lists it (a strip with no files to strip, and no
    second inventory), the host remembers it so the restore runs before this
    sandbox's grading, and the record names it as disarmed ``earlier``. A
    clean sandbox with no record costs no extra exec."""
    from benchflow.sandbox import _delegate_disarm
    from benchflow.sandbox.lockdown import block_cloud_metadata

    recorded = (
        f"BF metadata-delegates-recorded {_SU_HEX} 4755 0 0 1234 1 55680 {_SHA} -\n"
        "BF metadata-delegates-strip-done 1 0\n"
    )
    env = _python_env(
        monkeypatch,
        {
            "inventory": [("BF metadata-delegates-record\n" + PYTHON_CLEAN, 0)],
            "strip": [(recorded, 0)],
        },
    )
    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "on", record
    assert env.seen == ["inventory", "strip"]
    # Nothing to strip: the paths on its stdin are none.
    assert env.sent[1].endswith(
        "strip /opt/runtime/delegates "
        + _delegate_disarm._trusted_uids()
        + " <<'BF_DELEGATE_TARGETS'\nBF_DELEGATE_TARGETS\n)"
    ), env.sent[1][-200:]
    assert record["delegates_disarmed"] == [
        {
            "path": "/usr/bin/su",
            "why": "setuid root",
            "mode": "4755",
            "uid": 0,
            "gid": 0,
            "inode": 1234,
            "links": 1,
            "size": 55680,
            "sha256": _SHA,
            "earlier": True,
        }
    ]
    assert getattr(env, _delegate_disarm.DISARMED_ATTR)["stripped"]

    clean = _python_env(monkeypatch, {"inventory": [(PYTHON_CLEAN, 0)]})
    record = await block_cloud_metadata(clean, "agent")
    assert record["state"] == "on" and "delegates_disarmed" not in record
    assert clean.seen == ["inventory"]
    assert getattr(clean, _delegate_disarm.DISARMED_ATTR, None) is None


# --- restore before grading, and the trial's records ---------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("out", "code", "outcomes"),
    [
        (
            f"BF metadata-delegates-restored {_SU_HEX} 4755:exact\n"
            "BF metadata-delegates-restore-done 1 0 0\n",
            0,
            [("restored", "4755:exact")],
        ),
        (
            f"BF metadata-delegates-kept {_SU_HEX} changed:sha256,size\n"
            "BF metadata-delegates-restore-done 0 1 0\n",
            0,
            [("kept_stripped", "changed:sha256,size")],
        ),
        (
            f"BF metadata-delegates-restore-failed {_SU_HEX} EIO\n"
            "BF metadata-delegates-restore-done 0 0 1\n",
            7,
            [("failed", "EIO")],
        ),
        ("", 125, [("failed", "the restore did not report (status 125)")]),
    ],
    ids=["restored", "changed", "eio", "no-report"],
)
async def test_the_restore_reports_each_file_and_raises_only_for_its_own_failure(
    out, code, outcomes
):
    """Option A (review 3): the restore runs root's same
    checked Python and reports each file. A file the agent changed stays
    disarmed and grading goes on; BenchFlow's own failure (an I/O error, a
    restore that did not report) raises DelegateRestoreError, a verifier
    infrastructure error: the trial is unscored, never 0."""
    from benchflow.sandbox import _delegate_disarm
    from benchflow.task.verifier_errors import verifier_infrastructure_failure

    commands: list[str] = []

    class Env:
        @receipt_aware
        async def exec(self, command, user=None, **kwargs):
            commands.append(command)
            return SimpleNamespace(stdout=out, return_code=code)

    env = Env()
    _delegate_disarm.remember(
        env,
        _delegate_disarm.DelegatePython("/opt/runtime/rt/py/python3", "kit"),
        [{"path": "/usr/bin/su", "inode": 1234}],
    )
    if outcomes[0][0] == "failed":
        with pytest.raises(_delegate_disarm.DelegateRestoreError) as failed:
            await _delegate_disarm.restore_disarmed_delegates(env)
        assert verifier_infrastructure_failure(failed.value)
        report = getattr(env, _delegate_disarm.RESTORED_ATTR)
    else:
        report = await _delegate_disarm.restore_disarmed_delegates(env)
        assert getattr(env, _delegate_disarm.RESTORED_ATTR) is report

    assert [(r["outcome"], r["why"]) for r in report.rows] == outcomes
    # The host's memory outlives the restore: a grading after a snapshot
    # restore (a branch) runs it again; root's record decides what is left.
    assert getattr(env, _delegate_disarm.DISARMED_ATTR)["stripped"]
    assert " restore " in commands[0] and "/opt/runtime/rt/py/python3" in commands[0]
    # Nothing was disarmed: nothing to restore, no exec.
    assert await _delegate_disarm.restore_disarmed_delegates(Env()) is None


@pytest.mark.parametrize("case", ["restored", "changed", "live", "failed"])
def test_the_trial_records_what_was_put_back_and_what_was_not(tmp_path, case):
    """Option A and review 1(b), 3: cloud-metadata.json gets
    each file's restore outcome, and result.json names a file that stayed
    disarmed while the verifier ran, with a trial warning that says why: the
    agent changed it, a process outside no_new_privs was alive (named), or
    BenchFlow failed (the trial is unscored)."""
    from benchflow.sandbox import _delegate_disarm
    from benchflow.sandbox.cloud_metadata import CloudMetadataPlan

    trial = _rollout(tmp_path, CloudMetadataPlan(armed=True))
    (tmp_path / "cloud-metadata.json").write_text(json.dumps({"state": "on"}))
    trial._cloud_metadata_result = {
        "state": "on",
        "delegates_disarmed": ["/usr/bin/su"],
    }
    outcome, why = {
        "restored": ("restored", "4755:exact"),
        "changed": ("kept_stripped", "changed:sha256"),
        "live": ("kept_stripped", "live-non-nnp-process"),
        "failed": ("failed", "EIO"),
    }[case]
    row = {"path": "/usr/bin/su", "outcome": outcome, "why": why}
    live = [{"pid": 4242, "uid": 33, "comm": "apache2"}] if case == "live" else []
    setattr(
        trial._env,
        _delegate_disarm.RESTORED_ATTR,
        _delegate_disarm.RestoreReport([row], live),
    )

    trial._record_delegate_restore()

    record = json.loads((tmp_path / "cloud-metadata.json").read_text())
    assert record["delegates_restored"] == [row]
    assert record.get("restore_withheld_for") == (live or None)
    result = trial._cloud_metadata_result
    assert result["delegates_restored"] == (1 if case == "restored" else 0)
    assert ("delegates_not_restored" in result) is (case != "restored")
    warnings = " | ".join(getattr(trial, "_warnings", []))
    expected = {
        "restored": None,
        "changed": "it is not the file it disarmed any more",
        "live": "pid 4242 uid 33 apache2",
        "failed": "the trial is unscored",
    }[case]
    if expected is None:
        assert "/usr/bin/su" not in warnings
    else:
        assert expected in warnings, warnings


@pytest.mark.asyncio
async def test_a_restore_benchflow_could_not_do_leaves_the_trial_unscored(tmp_path):
    """Review 3 end to end in grading: hardening's restore
    raising DelegateRestoreError (a timeout, a failed exec, an I/O error) is
    a verifier crash, rewards None, never a 0 and never the agent's fault."""
    from benchflow.rollout._setup import _verify_rollout
    from benchflow.sandbox import _delegate_disarm

    class Planes:
        async def harden_before_verify(self, env, task, sandbox_user, **kwargs):
            raise _delegate_disarm.DelegateRestoreError(
                "BenchFlow could not put back ...: /usr/bin/su (EIO)"
            )

        def verifier(self, **kwargs):  # never reached
            raise AssertionError("the verifier must not run")

    task = SimpleNamespace(
        name="t", config=SimpleNamespace(verifier=SimpleNamespace(timeout_sec=60))
    )
    paths = SimpleNamespace(verifier_dir=tmp_path / "verifier", rollout_dir=tmp_path)

    rewards, error, _diag = await _verify_rollout(
        SimpleNamespace(), task, paths, {}, Planes(), sandbox_user="agent"
    )

    assert rewards is None
    assert error.startswith("verifier crashed: "), error
    assert "could not put back" in error


def test_a_reconnect_keeps_what_the_trial_disarmed_earlier(tmp_path):
    """Option A's records on a branch. A sandbox restored from
    a snapshot has su disarmed already, so the reconnect's inventory counts
    nothing. Before, its record overwrote cloud-metadata.json and result.json
    without the files this trial disarmed (seen on the deterministic branch
    tests). A refusal never inherits them."""
    from benchflow.sandbox.cloud_metadata import CloudMetadataPlan

    trial = _rollout(tmp_path, CloudMetadataPlan(armed=True))
    su = {"path": "/usr/bin/su", "mode": "4755", "inode": 7, "why": "setuid root"}
    record_path = tmp_path / "cloud-metadata.json"

    trial._record_cloud_metadata(
        tmp_path, {"state": "on", "why": "w", "delegates_disarmed": [su]}, ""
    )
    trial._record_cloud_metadata(tmp_path, {"state": "on", "why": "w"}, "")

    assert json.loads(record_path.read_text())["delegates_disarmed"] == [su]
    assert trial._cloud_metadata_result == {
        "state": "on",
        "delegates_disarmed": ["/usr/bin/su"],
    }
    # A reconnect that listed root's record names su as disarmed earlier: the
    # entry the disarming connect wrote stays; a new file is added.
    listed = {**su, "earlier": True, "inode": 9}
    tool = {"path": "/usr/bin/tool", "mode": "2755", "why": "setgid"}
    trial._record_cloud_metadata(
        tmp_path, {"state": "on", "why": "w", "delegates_disarmed": [listed, tool]}, ""
    )
    assert json.loads(record_path.read_text())["delegates_disarmed"] == [su, tool]

    trial._record_cloud_metadata(tmp_path, {"state": "refused", "why": "w"}, "")
    assert "delegates_disarmed" not in json.loads(record_path.read_text())
    trial._record_cloud_metadata(tmp_path, {"state": "on", "why": "w"}, "")
    assert "delegates_disarmed" not in json.loads(record_path.read_text())


# --- root's Python: kit, managed, staged ----------------------------------------


def _probe(managed="-", staged="-", arch="x86_64", libc="gnu", tar="1") -> str:
    return (
        f"BF delegate-python managed={managed} staged={staged} arch={arch} "
        f"libc={libc} tar={tar}\n"
    )


class _StageEnv:
    def __init__(self, probe: str):
        self.probe = probe
        self.commands: list[str] = []
        self.uploads: list[tuple[str, str]] = []

    @receipt_aware
    async def exec(self, command, user=None, **kwargs):
        self.commands.append(command)
        if "BF delegate-python managed=" in command:
            return SimpleNamespace(stdout=self.probe, return_code=0)
        if "BF delegate-python prepared" in command:
            return SimpleNamespace(
                stdout="BF delegate-python prepared\n", return_code=0
            )
        if "BF delegate-python staged" in command:
            return SimpleNamespace(stdout="BF delegate-python staged\n", return_code=0)
        raise AssertionError(command[:200])

    async def upload_file(self, source, target, *, admission):
        # BenchFlow's own bytes, admitted as a trusted transfer: the
        # sandbox's guard refuses an upload without an admission.
        assert admission.reason == "BenchFlow's pinned CPython"
        self.uploads.append((str(source), target))


@pytest.mark.asyncio
async def test_without_a_python_in_the_image_benchflow_stages_its_own(
    monkeypatch, tmp_path
):
    """Option A's bar (no stock image refused for lacking
    tools): with no kit or managed Python, BenchFlow's pinned CPython for the
    image's machine and C library goes from the host's cache into a root-only
    folder, is unpacked and checked, and becomes root's Python."""
    from benchflow.sandbox import _delegate_disarm

    tarball = tmp_path / "cpython.tar.gz"
    tarball.write_bytes(b"pinned build")
    picked = []

    def host(arch, libc):
        picked.append((arch, libc))
        return tarball

    monkeypatch.setattr(_delegate_disarm, "host_tarball", host)
    env = _StageEnv(_probe())

    python, why = await _delegate_disarm.find_python(env)

    assert why == "" and python is not None
    assert picked == [("x86_64", "gnu")]
    sha = _delegate_disarm.STAGED_PYTHON_SHA256[("x86_64", "gnu")]
    assert python.path == f"/opt/runtime/staged-python/{sha}/python/bin/python3.12"
    assert python.source == "staged"
    assert python.tree == f"/opt/runtime/staged-python/{sha}"
    assert env.uploads == [
        (str(tarball), f"/opt/runtime/staged-python/{sha}/python.tar.gz")
    ]
    unpack = env.commands[-1]
    assert sha in unpack and "-xzf" in unpack and "chown -R 0:0" in unpack


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("probe", "source", "why"),
    [
        (
            _probe(managed=b"/opt/runtime/python/cpython/bin/python3.12".hex()),
            "managed",
            "",
        ),
        (
            _probe(staged=b"/opt/runtime/staged-python/x/python/bin/python3.12".hex()),
            "staged",
            "",
        ),
        (
            _probe(arch="other", libc="none"),
            None,
            "has no build of BenchFlow's CPython",
        ),
        (_probe(tar="0"), None, "no tar to unpack BenchFlow's CPython"),
    ],
    ids=["managed", "staged-before", "no-build", "no-tar"],
)
async def test_root_python_is_found_or_the_reason_given(
    monkeypatch, probe, source, why
):
    """Option A: BenchFlow's managed CPython (the model proxy's
    runtime) or an earlier staged one is used as found; an image whose
    machine or C library has no build, or that has no tar, gives a reason
    instead of a Python, and nothing is uploaded."""
    from benchflow.sandbox import _delegate_disarm

    monkeypatch.setattr(
        _delegate_disarm,
        "host_tarball",
        lambda *a: pytest.fail("no upload expected"),
    )
    env = _StageEnv(probe)
    python, reason = await _delegate_disarm.find_python(env)

    if source is None:
        assert python is None and why in reason
    else:
        assert python is not None and python.source == source and reason == ""
    assert env.uploads == []


def test_the_hosts_copy_is_fetched_once_and_only_with_the_pinned_sha256(
    monkeypatch, tmp_path
):
    """Option A: the staged CPython is BenchFlow's pinned build
    and nothing else; a download whose sha256 is not the pin is refused, and
    a good copy is fetched once and then read from the cache."""
    from benchflow.sandbox import _delegate_disarm

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    good = b"the pinned build"
    pins = dict(_delegate_disarm.STAGED_PYTHON_SHA256)
    pins[("x86_64", "gnu")] = hashlib.sha256(good).hexdigest()
    monkeypatch.setattr(_delegate_disarm, "STAGED_PYTHON_SHA256", pins)
    fetched: list[str] = []

    def fetch(body):
        def run(url, dest):
            fetched.append(url)
            Path(dest).write_bytes(body)

        return run

    with pytest.raises(
        _delegate_disarm.StagedPythonUnavailable, match="not the pinned"
    ):
        _delegate_disarm.host_tarball("x86_64", "gnu", fetch=fetch(b"tampered"))
    path = _delegate_disarm.host_tarball("x86_64", "gnu", fetch=fetch(good))
    assert path.read_bytes() == good
    assert _delegate_disarm.host_tarball("x86_64", "gnu", fetch=fetch(b"x")) == path
    assert fetched[-1].endswith(
        "cpython-3.12.15%2B20261013-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
    )
    assert len(fetched) == 3  # two mirrors refused the tampered copy, one good


def test_the_staged_pins_are_the_releases_published_sums():
    """The four pins are python-build-standalone 20261013's own SHA256SUMS
    entries for the CPython the uv pin installs (3.12.15)."""
    from benchflow._uv_pin import PYTHON_VERSION
    from benchflow.sandbox import _delegate_disarm

    assert _delegate_disarm.STAGED_PYTHON_VERSION == PYTHON_VERSION == "3.12.15"
    assert set(_delegate_disarm.STAGED_PYTHON_SHA256) == {
        ("x86_64", "gnu"),
        ("aarch64", "gnu"),
        ("x86_64", "musl"),
        ("aarch64", "musl"),
    }
    assert all(
        len(v) == 64 and int(v, 16) >= 0
        for v in _delegate_disarm.STAGED_PYTHON_SHA256.values()
    )


# --- the real shell route --------------------------------------------------------


@linux_only
def test_strip_and_restore_run_through_roots_shell_route(sandbox, monkeypatch):
    """Option A end to end in a real shell: the command root
    sends checks its Python's custody, strips a setuid file, and the restore
    puts the exact mode back (this test's user stands in for root). Then as
    on a branch: the inventory reports root's record, the file comes back
    from a "snapshot" on a new inode, a strip with nothing to strip lists
    it, and the restore puts the mode back on the new inode. The Python is a
    root-owned system one: a runner's own venv may sit in a folder others
    can write, which the custody check refuses."""
    from benchflow.sandbox import _delegate_disarm, lockdown

    python = shutil.which("python3", path="/usr/bin:/bin")
    if python is None:
        pytest.skip("no root-owned python3 in /usr/bin or /bin")

    su = _file(sandbox.root / "usr/bin/su", 0o4755)
    record = str(sandbox.root / "opt/benchflow/delegates")
    # This host's own /proc has other users' processes without no_new_privs.
    no_procs = str(_proc(sandbox.root.parent))

    def run(command: str) -> tuple[str, int]:
        done = subprocess.run(
            ["/bin/sh", "-c", command],
            capture_output=True,
            text=True,
            env={"PATH": f"{sandbox.bin}:{sandbox._system_bin()}"},
        )
        return done.stdout, done.returncode

    out, code = run(
        _delegate_disarm.disarm_cmd(
            python, "strip", targets=(str(su).encode(),), record_dir=record
        )
    )
    assert code == 0, out
    assert _mode(su) == 0o755
    monkeypatch.setattr(
        lockdown, "_CLOUD_METADATA_ROOT_PATH", f"{sandbox.bin}:{sandbox._system_bin()}"
    )
    out, code = run(
        lockdown._cloud_metadata_delegate_check_cmd(
            str(sandbox.root / "usr"), python=python, record=f"{record}/record.json"
        )
    )
    report = lockdown._delegate_report(out, code)
    assert report.complete and report.recorded and not report.findings, out
    copy = _file(su.with_name("su.snap"), 0o755, su.read_bytes())
    old_inode = os.stat(su).st_ino
    os.replace(copy, su)
    assert os.stat(su).st_ino != old_inode
    out, code = run(_delegate_disarm.disarm_cmd(python, "strip", record_dir=record))
    listed = _delegate_disarm.read_strip(out, code)
    assert listed.ok and [(e["path"], e["earlier"]) for e in listed.stripped] == [
        (str(su), True)
    ], out

    out, code = run(
        _delegate_disarm.disarm_cmd(
            python, "restore", record_dir=record, proc_root=no_procs
        )
    )
    assert code == 0, out
    assert "4755:exact" in out and _mode(su) == 0o4755
    out, code = run(
        lockdown._cloud_metadata_delegate_check_cmd(
            str(sandbox.root / "usr"), python=python, record=f"{record}/record.json"
        )
    )
    assert not lockdown._delegate_report(out, code).recorded, out
