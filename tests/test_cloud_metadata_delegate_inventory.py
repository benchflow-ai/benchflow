"""The cloud metadata block's delegate inventory fails closed and
names what it found.

Earlier the inventory needed the image's ``getcap`` and ``find``. Without
either, after a timeout or exec failure, or on any unreadable report, it
recorded ``delegate_privilege_audit: "unavailable"`` and the agent started,
so a file carrying ``cap_net_admin`` in its ``security.capability`` attribute
went unnoticed on every image without libcap's getcap (batch 8, 70400f55, made
that the rule). A refusal named nothing and wrote no record, so result.json
did not say which file to fix (git-multibranch, 2026-10-28: ``/usr/bin/su``
with getcap present). The same image passed in one trial and was refused in
three: its ``find`` failed once, which counted as "unavailable".

Here: the protected Python reader (run in-process on a temp tree, its
attribute reads stubbed: a test cannot set ``security.capability``), the
shell inventory on a real temp tree with a stub getcap, the host's report
classifier, the block's refusal and its record, the rollout writing that
record before the error goes on, and the retry classification.
"""

from __future__ import annotations

import errno
import io
import json
import os
import subprocess
import sys
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
from tests.test_cloud_metadata_delegates import _isolate_inventory
from tests.test_cloud_metadata_delegates import sandbox as sandbox

linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="the inventory's shell runs on Linux sandboxes"
)


# --- the protected Python reader ------------------------------------------------


def _vfs_caps(prm: int, *, effective: bool = True, rev: int = 2, inh: int = 0) -> bytes:
    """A ``security.capability`` value as linux/capability.h lays it out."""
    magic = (rev << 24) | int(effective)
    words = [magic, prm & 0xFFFFFFFF, inh & 0xFFFFFFFF]
    if rev >= 2:
        words += [prm >> 32, inh >> 32]
    if rev == 3:
        words.append(1000)
    return b"".join(w.to_bytes(4, "little") for w in words)


def _run_reader(
    monkeypatch,
    root: Path,
    *,
    caps: dict[str, bytes] | None = None,
    xattr_errors: dict[str, int] | None = None,
    gone_dirs: tuple[str, ...] = (),
    limit: int = 2000,
) -> tuple[str, int]:
    """Run the reader's program text as root's sandbox would, in-process,
    with each file's capability attribute stubbed by path."""
    from benchflow.sandbox import lockdown

    caps = {str(root / k).encode(): v for k, v in (caps or {}).items()}
    errors = {str(root / k).encode(): v for k, v in (xattr_errors or {}).items()}
    gone = {str(root / d).encode() for d in gone_dirs}

    def getxattr(path, name, *, follow_symlinks=True):
        assert name == "security.capability" and follow_symlinks is False
        if path in errors:
            raise OSError(errors[path], os.strerror(errors[path]))
        if path in caps:
            return caps[path]
        raise OSError(errno.ENODATA, "No data available")

    real_scandir = os.scandir

    def scandir(path):
        if path in gone:
            raise FileNotFoundError(errno.ENOENT, "No such file or directory")
        return real_scandir(path)

    monkeypatch.setattr(os, "getxattr", getxattr, raising=False)
    monkeypatch.setattr(os, "scandir", scandir)
    program = lockdown._DELEGATE_PY_READER.replace("@NAMES@", repr(lockdown._CAP_NAMES))
    out = io.BytesIO()
    monkeypatch.setattr(sys, "argv", ["-c", str(root), str(limit)])
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=out))
    with pytest.raises(SystemExit) as ended:
        exec(compile(program, "<delegate reader>", "exec"), {"__name__": "__main__"})
    return out.getvalue().decode(), ended.value.code


def _tree(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 1\n")
        path.chmod(0o755)


def test_a_capability_file_is_found_and_named_without_getcap(monkeypatch, tmp_path):
    """(a) Guards against the fail-open batch 8 (70400f55) pinned: an
    image without getcap had its file capabilities never read, and started.
    The protected Python reads each attribute itself and names the file."""
    from benchflow.sandbox.lockdown import _delegate_report

    _tree(tmp_path, "opt/tools/fw", "usr/bin/plain")
    out, code = _run_reader(
        monkeypatch, tmp_path, caps={"opt/tools/fw": _vfs_caps(1 << 12 | 1 << 7)}
    )

    assert code == 0, out
    report = _delegate_report(out, code)
    assert report.complete, report.problems
    assert report.reader == "python3"
    assert report.findings == [
        {
            "path": f"{tmp_path}/opt/tools/fw",
            "why": "file capabilities cap_setuid,cap_net_admin=ep",
        }
    ]


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (_vfs_caps(1 << 13), "cap_net_raw=ep"),
        (_vfs_caps(1 << 13, rev=1), "cap_net_raw=ep"),
        (_vfs_caps(1 << 12, effective=False), "cap_net_admin=p"),
        (_vfs_caps(0, inh=1 << 21), "cap_sys_admin=i"),
        (_vfs_caps(1 << 12, inh=1 << 12), "cap_net_admin=eip"),
        (_vfs_caps(1 << 40), "cap_checkpoint_restore=ep"),
        (_vfs_caps(1 << 63), "63=ep"),
        (_vfs_caps(1 << 13, rev=3), "cap_net_raw=ep [rootid=1000]"),
        (_vfs_caps(0), "="),
        (b"\x01\x00\x00\x02" + b"\x00" * 12, "invalid"),
        (b"\x01\x00\x00\x05" + b"\x00" * 16, "invalid"),
    ],
)
def test_every_capability_revision_is_read_as_getcap_prints_it(
    monkeypatch, tmp_path, value, text
):
    """The reader prints a one-clause capability as libcap's
    getcap does (checked against getcap on install-windows-3.11's
    gst-ptp-helper), so only ping's own ``cap_net_raw=ep`` can match the
    narrow exemption, and an attribute it cannot read still counts."""
    _tree(tmp_path, "bin/x")
    out, code = _run_reader(monkeypatch, tmp_path, caps={"bin/x": value})

    assert code == 0
    assert (
        f"BF metadata-delegates-cap {str(tmp_path / 'bin/x').encode().hex()} {text}\n"
        in out
    )


def test_names_that_go_away_mid_walk_are_counted_not_failed(monkeypatch, tmp_path):
    """Guards the nondeterminism fix: git-multibranch passed in 1 of 4
    trials because its whole-filesystem find failed once (a name gone between
    a folder's listing and its own look, as a running proxy's bytecode temp
    files do), which counted as "unavailable". A name that went away has
    nothing left to run: it is counted, and the inventory still finishes."""
    from benchflow.sandbox.lockdown import _delegate_report

    _tree(tmp_path, "srv/__pycache__/m.cpython-312.pyc.140", "srv/tmp/x", "srv/a")
    out, code = _run_reader(
        monkeypatch,
        tmp_path,
        xattr_errors={"srv/__pycache__/m.cpython-312.pyc.140": errno.ENOENT},
        gone_dirs=("srv/tmp",),
    )

    assert code == 0, out
    report = _delegate_report(out, code)
    assert report.complete, report.problems
    assert report.vanished == 2
    assert report.findings == []


@pytest.mark.parametrize("err", [errno.EACCES, errno.EIO])
def test_an_unreadable_file_makes_the_inventory_incomplete(monkeypatch, tmp_path, err):
    """(c) A file whose attribute cannot be read is never a
    clean file; the inventory ends incomplete (exit 7) and names it."""
    from benchflow.sandbox.lockdown import _delegate_report

    _tree(tmp_path, "usr/bin/a", "usr/bin/b")
    out, code = _run_reader(monkeypatch, tmp_path, xattr_errors={"usr/bin/b": err})

    assert code == 7
    report = _delegate_report(out, code)
    assert not report.complete
    assert (
        f"getxattr {tmp_path}/usr/bin/b: {errno.errorcode[err]}" in report.problems[0]
    )


def test_more_report_lines_than_the_limit_is_incomplete(monkeypatch, tmp_path):
    """An inventory is never cut short into a clean one."""
    from benchflow.sandbox.lockdown import _delegate_report

    for i in range(5):
        _tree(tmp_path, f"opt/s{i}")
        (tmp_path / f"opt/s{i}").chmod(0o4755)
    out, code = _run_reader(monkeypatch, tmp_path, limit=3)

    report = _delegate_report(out, code)
    assert code == 7 and not report.complete
    assert "more than" in report.problems[0]


# --- the shell inventory (find, getcap) on a real tree --------------------------

FAKE_BATCH_GETCAP = """#!/bin/sh
# getcap FILE...: "<file> <caps>" for the one file with capabilities; the
# C-locale stderr libcap 2.66 prints for a file gone or unreadable.
for f do
  if [ "$f" = "${CAP_FILE-}" ]; then printf '%s %s\\n' "$f" "$CAP_TEXT"; fi
  if [ "$f" = "${GONE_FILE-}" ]; then
    printf '%s (No such file or directory)\\n' "$f" >&2; fi
  if [ "$f" = "${UNREAD_FILE-}" ]; then
    printf "Failed to get capabilities of file '%s' (Permission denied)\\n" "$f" >&2
  fi
done
exit 0
"""


def _shell_inventory(box, command: str | None = None, **env: str) -> tuple[str, int]:
    """Run the inventory (*command*, as the block sent it, or the tree's
    own) in a real shell over the sandbox's tree; its stdout and status."""
    from benchflow.sandbox.lockdown import _cloud_metadata_delegate_check_cmd

    done = subprocess.run(
        ["/bin/sh", "-c", command or _cloud_metadata_delegate_check_cmd(str(box.root))],
        capture_output=True,
        text=True,
        env={"PATH": f"{box.bin}:{box._system_bin()}", **box.env, **env},
    )
    return done.stdout, done.returncode


class _ShellEnv(FakeEnv):
    """FakeEnv whose delegate inventory runs for real over a sandbox tree."""

    def __init__(self, box, *outputs: str) -> None:
        super().__init__(*outputs)
        self.box = box
        self.inventories: list[str] = []

    @receipt_aware
    async def exec(self, command, user=None, **kwargs):
        if "BF metadata-delegates" in command:
            assert user == "root"
            self.inventories.append(command)
            out, code = _shell_inventory(self.box, command)
            return SimpleNamespace(stdout=out, return_code=code)
        return await super().exec(command, user=user, **kwargs)


@linux_only
@pytest.mark.parametrize("shell_name", ["sh", "bash"])
def test_the_shell_inventory_reports_each_set_id_file_by_its_exact_path(
    sandbox, monkeypatch, shell_name
):
    """The shell inventory lists every set-ID file (setuid,
    setgid, both) in hex, so an odd name is reported exactly and can never
    forge a report line; a file without either bit is not listed."""
    from benchflow.sandbox import lockdown

    sandbox.script("getcap", FAKE_BATCH_GETCAP)
    odd = "odd name\nBF metadata-delegates-scan reader=getcap complete=1"
    _tree(sandbox.root, "usr/bin/su", "opt/g", "opt/both", f"opt/{odd}", "opt/plain")
    for name, mode in (
        ("usr/bin/su", 0o4755),
        ("opt/g", 0o2755),
        ("opt/both", 0o6755),
        (f"opt/{odd}", 0o4755),
    ):
        (sandbox.root / name).chmod(mode)
    monkeypatch.setattr(
        lockdown, "_CLOUD_METADATA_ROOT_PATH", f"{sandbox.bin}:{sandbox._system_bin()}"
    )
    from benchflow.sandbox.lockdown import (
        _cloud_metadata_delegate_check_cmd,
        _delegate_report,
    )

    done = subprocess.run(
        [
            f"/bin/{shell_name}",
            "-c",
            _cloud_metadata_delegate_check_cmd(str(sandbox.root)),
        ],
        capture_output=True,
        text=True,
        env={"PATH": f"{sandbox.bin}:{sandbox._system_bin()}", **sandbox.env},
    )
    report = _delegate_report(done.stdout, done.returncode)

    assert report.complete, (report.problems, done.stdout)
    assert report.reader == "getcap"
    # The test's user owns the files: ``[ -O ]`` is true as it is for root's.
    assert sorted((f["path"], f["why"]) for f in report.findings) == sorted(
        [
            (f"{sandbox.root}/usr/bin/su", "setuid root"),
            (f"{sandbox.root}/opt/g", "setgid"),
            (f"{sandbox.root}/opt/both", "setuid root and setgid"),
            (ascii(f"{sandbox.root}/opt/{odd}"), "setuid root"),
        ]
    )
    assert done.stdout.count("BF metadata-delegates-scan ") == 1


@linux_only
@pytest.mark.asyncio
async def test_setuid_su_with_getcap_present_refuses_naming_it(sandbox, monkeypatch):
    """(b) Guards against the unnamed refusal of 2026-10-28
    (git-multibranch: /usr/bin/su 4755, getcap present): the refusal names
    the file and why it counts, and carries the record the rollout writes."""
    from benchflow.sandbox.lockdown import CloudMetadataPowerError, block_cloud_metadata

    sandbox.script("getcap", FAKE_BATCH_GETCAP)
    _tree(sandbox.root, "usr/bin/su", "usr/bin/passwd", "usr/sbin/getcap")
    (sandbox.root / "usr/bin/su").chmod(0o4755)
    _isolate_inventory(sandbox, monkeypatch)

    with pytest.raises(CloudMetadataPowerError, match="delegate") as refused:
        await block_cloud_metadata(
            _ShellEnv(sandbox, ROOT_OK, AGENT_REFUSED, "BF metadata-hits 1\n"), "agent"
        )

    message = str(refused.value)
    assert message.startswith("sandbox startup refused: the image holds privileges")
    assert f"{sandbox.root}/usr/bin/su (setuid root)" in message
    assert "chmod ug-s <file>" in message and "--allow-cloud-metadata" in message
    record = refused.value.record
    assert record["state"] == "refused"
    assert record["delegates"] == [
        {"path": f"{sandbox.root}/usr/bin/su", "why": "setuid root"}
    ]
    assert record["delegate_privilege_audit"] == "complete"
    assert record["delegate_inventory"]["reader"] == "getcap"


@linux_only
@pytest.mark.parametrize(
    ("env", "complete", "vanished"),
    [
        ({"CAP_FILE": "{root}/usr/bin/ping", "CAP_TEXT": "cap_net_raw=ep"}, True, None),
        ({"GONE_FILE": "{root}/usr/bin/ping"}, True, 1),
        ({"UNREAD_FILE": "{root}/usr/bin/ping"}, False, None),
    ],
    ids=["capability", "gone", "unreadable"],
)
def test_getcaps_stderr_is_read_and_only_a_vanished_name_is_forgiven(
    sandbox, monkeypatch, env, complete, vanished
):
    """(c) getcap exits 0 even for a file it could not read,
    reporting it only on stderr, which the old inventory dropped. A name
    that went away is counted; any other getcap error is an incomplete
    inventory. (ping's capability is exempt only at the system path, so at
    the test tree's path it counts.)"""
    from benchflow.sandbox import lockdown
    from benchflow.sandbox.lockdown import _delegate_report

    sandbox.script("getcap", FAKE_BATCH_GETCAP)
    _tree(sandbox.root, "usr/bin/ping")
    monkeypatch.setattr(
        lockdown, "_CLOUD_METADATA_ROOT_PATH", f"{sandbox.bin}:{sandbox._system_bin()}"
    )
    out, code = _shell_inventory(
        sandbox, **{k: v.format(root=sandbox.root) for k, v in env.items()}
    )
    report = _delegate_report(out, code)

    assert report.complete is complete, (report.problems, out)
    assert report.vanished == vanished
    if "CAP_FILE" in env:
        assert report.findings == [
            {
                "path": f"{sandbox.root}/usr/bin/ping",
                "why": "file capabilities cap_net_raw=ep",
            }
        ]
    if not complete:
        assert "getcap could not read 1 file(s)" in report.problems[0]
        assert code == 0  # getcap's own exit status says nothing


@linux_only
@pytest.mark.asyncio
async def test_no_capability_reader_refuses_with_the_remedy_and_opt_outs(
    sandbox, monkeypatch
):
    """(a, c) Guards against the fail-open batch 8 (70400f55) pinned:
    an image without getcap and without a protected Python, whose file
    capabilities nothing can read, is refused, not started as "unavailable";
    the message says what to add and how to opt out, and a retry would only
    repeat it."""
    from benchflow._utils.scoring import is_unrecoverable_startup_error
    from benchflow.sandbox.lockdown import CloudMetadataPowerError, block_cloud_metadata

    (sandbox.bin / "getcap").unlink()
    _isolate_inventory(sandbox, monkeypatch)
    env = _ShellEnv(sandbox, ROOT_OK, AGENT_REFUSED, "BF metadata-hits 1\n")

    with pytest.raises(CloudMetadataPowerError) as refused:
        await block_cloud_metadata(env, "agent")

    message = str(refused.value)
    assert "could not finish the file privilege inventory" in message
    assert "the image has no getcap (libcap)" in message
    assert "libcap2-bin on Debian and Ubuntu" in message
    assert (
        "allow_cloud_metadata = true" in message and "--allow-cloud-metadata" in message
    )
    assert refused.value.record["delegate_privilege_audit"] == "incomplete"
    assert refused.value.record["delegate_inventory"]["reader"] == "none"
    assert len(env.inventories) == 1  # the image decides it: no second attempt
    assert is_unrecoverable_startup_error(message)


@linux_only
@pytest.mark.asyncio
async def test_a_clean_image_still_starts(sandbox, monkeypatch):
    """(d) Guards against refusing ordinary images: a tree with no
    set-ID file and no file capability that counts passes an inventory that
    finished, and the block goes on as before."""
    from benchflow.sandbox.lockdown import block_cloud_metadata

    sandbox.script("getcap", FAKE_BATCH_GETCAP)
    _tree(sandbox.root, "usr/bin/plain", "opt/app/run")
    _isolate_inventory(sandbox, monkeypatch)

    record = await block_cloud_metadata(
        _ShellEnv(sandbox, ROOT_OK, AGENT_REFUSED, "BF metadata-hits 1\n"), "agent"
    )

    assert record["state"] == "on", record
    assert record["delegate_privilege_audit"] == "complete"
    assert record["delegate_inventory"]["reader"] == "getcap"
    assert "delegates" not in record


# --- the host's report classifier ------------------------------------------------

_SU = "BF metadata-delegates-setid u-0 " + b"/usr/bin/su".hex() + "\n"


@pytest.mark.parametrize(
    ("out", "code", "problem"),
    [
        ("", 0, "it did not report"),
        (DELEGATES_CLEAN * 2, 0, "it reported more than once"),
        ("BF metadata-delegates power=0\n", 0, "it did not report"),
        ("ls: cannot access\n" + DELEGATES_CLEAN, 0, "line(s) it never prints"),
        (DELEGATES_CLEAN, 7, "status 7, not 0"),
        (DELEGATES_CLEAN, None, "status None, not 0"),
        (DELEGATES_CLEAN.replace("why=-", "why=caps"), 7, "find or getcap failed"),
        (DELEGATES_CLEAN.replace("why=-", "why=caps"), 0, "status 0, not 7"),
        ("BF metadata-delegates-setid u-0 7a\n" + DELEGATES_CLEAN, 0, "never prints"),
    ],
    ids=[
        "empty",
        "repeated",
        "old-report",
        "stray-line",
        "status",
        "no-status",
        "failed-pass",
        "failed-pass-status-0",
        "relative-path",
    ],
)
def test_the_report_fails_closed(out, code, problem):
    """(c) Only a complete, well-formed report with its own
    exit status is a finished inventory; anything else is incomplete."""
    from benchflow.sandbox.lockdown import _delegate_report

    report = _delegate_report(out, code)
    assert not report.complete
    assert any(problem in p for p in report.problems), report.problems


def test_only_the_narrow_ping_capability_at_its_system_path_is_exempt():
    """Guards the host classifier against widening S2-CONF-8's rule:
    ping's ``cap_net_raw=ep`` is exempt at /usr/bin only; the same line at
    another path, another capability on ping, and a getcap line that cannot
    be read all count."""
    from benchflow.sandbox.lockdown import _delegate_report

    lines = [
        "/usr/bin/ping cap_net_raw=ep",
        "/opt/ping cap_net_raw=ep",
        "/usr/bin/ping6 cap_net_admin,cap_net_raw=ep",
        "/usr/bin/x = cap_setuid+ep",
        "/usr/bin/arping cap_net_raw=ep [rootid=1000]",
        "not a getcap line",
    ]
    blob = ("\n".join(lines) + "\n").encode().hex()
    report = _delegate_report(
        f"BF metadata-delegates-getcap {blob}\n" + DELEGATES_CLEAN, 0
    )

    assert report.complete, report.problems
    assert report.findings == [
        {"path": "/opt/ping", "why": "file capabilities cap_net_raw=ep"},
        {
            "path": "/usr/bin/ping6",
            "why": "file capabilities cap_net_admin,cap_net_raw=ep",
        },
        {"path": "/usr/bin/x", "why": "file capabilities cap_setuid+ep"},
        {
            "path": "/usr/bin/arping",
            "why": "file capabilities cap_net_raw=ep [rootid=1000]",
        },
        {"path": "not a getcap line", "why": "a getcap line BenchFlow could not read"},
    ]


# --- retries, and the rollout's record ------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first", "attempts", "refused"),
    [
        ((DELEGATES_CLEAN.replace("why=-", "why=setid"), 7), 2, False),
        ((DELEGATES_CLEAN.replace("why=-", "why=reader"), 7), 1, True),
        ((_SU + DELEGATES_CLEAN, 0), 1, True),
        (TimeoutError("dummy inventory timeout"), 1, True),
    ],
    ids=["transient-then-clean", "no-reader", "found", "timeout"],
)
async def test_one_more_attempt_follows_only_a_transient_incomplete_inventory(
    first, attempts, refused
):
    """(c) A find that failed once (BusyBox has no
    -ignore_readdir_race) gets one more attempt, which decides; positive
    evidence, a missing reader and a failed or timed-out exec are never
    repeated, and none of them is ever a clean report."""
    from benchflow.sandbox.lockdown import CloudMetadataPowerError, block_cloud_metadata

    answers = [first, (DELEGATES_CLEAN, 0)]
    seen: list[str] = []

    class Env(FakeEnv):
        @receipt_aware
        async def exec(self, command, user=None, **kwargs):
            if "BF metadata-delegates" in command:
                seen.append(command)
                answer = answers.pop(0)
                if isinstance(answer, BaseException):
                    raise answer
                return SimpleNamespace(stdout=answer[0], return_code=answer[1])
            return await super().exec(command, user=user, **kwargs)

    env = Env(ROOT_OK, AGENT_REFUSED, "BF metadata-hits 1\n")
    if refused:
        with pytest.raises(CloudMetadataPowerError) as error:
            await block_cloud_metadata(env, "agent")
        record = error.value.record
        assert record["state"] == "refused"
    else:
        record = await block_cloud_metadata(env, "agent")
        assert record["state"] == "on"
    assert record["delegate_inventory"]["attempts"] == attempts
    assert len(seen) == attempts


@pytest.mark.asyncio
async def test_the_rollout_writes_the_refusals_record_before_the_error_goes_on(
    tmp_path,
):
    """(2) Guards against the refusal of 2026-10-28, which left no
    cloud-metadata.json and no result.json field naming the file to fix."""
    from unittest.mock import patch

    from benchflow.sandbox.cloud_metadata import CloudMetadataPlan
    from benchflow.sandbox.lockdown import CloudMetadataPowerError, block_cloud_metadata

    trial = _rollout(tmp_path, CloudMetadataPlan(armed=True))

    class Env(FakeEnv):
        @receipt_aware
        async def exec(self, command, user=None, **kwargs):
            if "BF metadata-delegates" in command:
                return SimpleNamespace(stdout=_SU + DELEGATES_CLEAN, return_code=0)
            return await super().exec(command, user=user, **kwargs)

    env = Env(ROOT_OK, AGENT_REFUSED, "BF metadata-hits 1\n")

    async def block(_env, user):
        return await block_cloud_metadata(env, user)

    with (
        patch.object(trial._planes, "block_cloud_metadata", side_effect=block),
        pytest.raises(CloudMetadataPowerError) as refused,
    ):
        await trial._apply_cloud_metadata_block(tmp_path)

    record = json.loads((tmp_path / "cloud-metadata.json").read_text())
    assert record["state"] == "refused"
    assert record["delegates"] == [{"path": "/usr/bin/su", "why": "setuid root"}]
    assert record["why"] == (
        "the image holds privileges a firewall delegate could use to gain root or "
        "NET_ADMIN and remove the cloud metadata block: /usr/bin/su (setuid root); "
        "BenchFlow's own Python could not be had to disarm them (the image's "
        "machine (other) or C library (none) has no build of BenchFlow's CPython)"
    )
    assert str(refused.value).startswith(f"sandbox startup refused: {record['why']}. ")
    assert trial._cloud_metadata_result == {
        "state": "refused",
        "delegates": [{"path": "/usr/bin/su", "why": "setuid root"}],
    }
    assert "cloud_metadata_block" in trial._timing


@pytest.mark.asyncio
async def test_a_failed_inventory_exec_stays_a_retryable_refusal():
    """Guards the retry classification: a positive finding is the
    image's and a retry repeats it (unrecoverable, as for a firewall tool);
    an inventory whose exec failed may finish on a fresh sandbox."""
    from benchflow._utils.scoring import is_unrecoverable_startup_error
    from benchflow.sandbox.lockdown import CloudMetadataPowerError, block_cloud_metadata

    class Env(FakeEnv):
        def __init__(self, answer, *outputs):
            super().__init__(*outputs)
            self.answer_with = answer

        @receipt_aware
        async def exec(self, command, user=None, **kwargs):
            if "BF metadata-delegates" in command:
                if isinstance(self.answer_with, BaseException):
                    raise self.answer_with
                return SimpleNamespace(stdout=self.answer_with, return_code=0)
            return await super().exec(command, user=user, **kwargs)

    messages = {}
    for name, answer in (
        ("found", _SU + DELEGATES_CLEAN),
        ("exec", ConnectionError("dummy lost connection")),
    ):
        with pytest.raises(CloudMetadataPowerError) as refused:
            await block_cloud_metadata(
                Env(answer, ROOT_OK, AGENT_REFUSED, "BF metadata-hits 1\n"), "agent"
            )
        messages[name] = str(refused.value)

    assert is_unrecoverable_startup_error(messages["found"])
    assert (
        "its exec failed (ConnectionError: dummy lost connection)" in messages["exec"]
    )
    assert not is_unrecoverable_startup_error(messages["exec"])
