"""The record of the quiesce after grading, ``kill-after-grading.json`` (0.10.1).

In integrity audit and off the agent's processes live through grading and the
rollout ends them after it (``planes.quiesce_agent``). That kill left no
record: no ``kill-before-grading.json`` (there is no kill before grading),
nothing else. Now the trial writes ``kill-after-grading.json`` beside it:
the account, the kill's count as strict's record has it (``killed``,
``killed_total``), when the quiesce started and ended, and each process root
listed just before it (pid, account, command, start time) with the signal
that ended it, inferred from its SIGTERM disposition, not measured
(``signal_source``). A trial with nothing left records an empty list.

Rows here: the listing on a stand-in /proc (each disposition, the accounts'
rule, a command that tries to end the listing early), and the record written
by ``_quiesce_after_grading`` with fake planes. The Docker rows, with real
leftovers and an independent listing, are in
``test_kill_after_grading_record_docker.py``.
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._utils import kill_record
from benchflow.rollout import _setup
from benchflow.sandbox import _quiesce_listing

AGENT = 1000
BTIME = 1_800_000_000
TERM = "0000000000004000"
NONE = "0000000000000000"


def _proc(
    root: Path,
    pid: int,
    *,
    uids: tuple[int, int, int] = (AGENT, AGENT, AGENT),
    state: str = "S",
    name: str = "sleep",
    cmdline: bytes = b"sleep\x00600\x00",
    ign: str = NONE,
    cgt: str = NONE,
    blocked: tuple[str, ...] = (NONE,),
    start: int = 12345,
) -> None:
    entry = root / str(pid)
    for index, mask in enumerate(blocked):
        task = entry / "task" / str(pid + index)
        task.mkdir(parents=True)
        (task / "status").write_text(f"Name:\t{name}\nSigBlk:\t{mask}\n")
    ruid, euid, suid = uids
    (entry / "status").write_text(
        f"Name:\t{name}\nState:\t{state} (x)\nUid:\t{ruid}\t{euid}\t{suid}\t{ruid}\n"
        f"Threads:\t{len(blocked)}\nSigBlk:\t{blocked[0]}\nSigIgn:\t{ign}\n"
        f"SigCgt:\t{cgt}\n"
    )
    # Fields 3-21, then starttime (22); the name may hold ") ".
    (entry / "stat").write_text(
        f"{pid} ({name}) {state} " + " ".join(["0"] * 18) + f" {start} 0 0\n"
    )
    (entry / "cmdline").write_bytes(cmdline)


def _list(root: Path, cwd: Path | None = None) -> list[dict]:
    (root / "stat").write_text(f"cpu  1 2 3\nbtime {BTIME}\nprocesses 9\n")
    command = _quiesce_listing.list_command([("agent", AGENT)], proc=str(root))
    done = subprocess.run(
        ["sh", "-c", command], capture_output=True, text=True, timeout=60, cwd=cwd
    )
    assert done.returncode == 0, done.stderr
    return _quiesce_listing.parse_listing(done.stdout)


def test_the_listing_reads_each_sigterm_disposition(tmp_path):
    """Each process of the account, with the signal its SIGTERM disposition
    leads to; root's, another account's and a finished zombie are left out."""
    proc = tmp_path / "proc"
    _proc(proc, 101)
    _proc(proc, 102, ign=TERM, name="py ignored")
    _proc(proc, 103, cgt=TERM, name="py caught")
    _proc(proc, 104, blocked=(TERM, TERM), name="blocked both")
    _proc(proc, 106, blocked=(TERM, NONE), name="blocked one")
    _proc(proc, 108, blocked=(TERM,), cgt=TERM, name="blocked caught")
    _proc(proc, 110, uids=(0, 0, 0), name="root")
    _proc(proc, 111, uids=(1001, 1001, 1001), name="other")
    _proc(proc, 112, uids=(0, AGENT, 0), name="effective")
    _proc(proc, 113, state="Z", name="reaped")
    _proc(proc, 114, state="Z", blocked=(NONE, NONE), name="zombie group")
    _proc(proc, 116, cmdline=b"", name="a) b")
    _proc(proc, 118, uids=(0, 0, AGENT), name="saved")  # N-5: the saved uid alone
    rows = {row["pid"]: row for row in _list(proc)}

    assert set(rows) == {101, 102, 103, 104, 106, 108, 112, 114, 116, 118}
    got = {pid: (row["sigterm"], row["ended_by"]) for pid, row in rows.items()}
    assert got == {
        101: ("default", "SIGTERM"),
        102: ("ignored", "SIGKILL"),
        103: ("caught", "SIGTERM or SIGKILL"),
        104: ("blocked", "SIGKILL"),
        106: ("default", "SIGTERM"),
        108: ("blocked", "SIGKILL"),
        112: ("default", "SIGTERM"),
        114: ("default", "SIGTERM"),
        116: ("default", "SIGTERM"),
        118: ("default", "SIGTERM"),
    }
    assert rows[101]["user"] == "agent"
    assert rows[101]["command"] == "sleep 600"
    assert rows[116]["command"] == "[a)]"
    assert rows[101]["start_ticks"] == 12345
    # Start times are in USER_HZ, 100 on Linux.
    assert rows[101]["started_at"] == datetime.fromtimestamp(
        BTIME + 12345 / 100, UTC
    ).isoformat(timespec="seconds")


def test_a_command_line_cannot_end_the_listing_or_add_a_row(tmp_path):
    """A process names itself what the listing prints: its newlines become
    '?', so it stays one row, and a long one is cut at 200 characters."""
    proc = tmp_path / "proc"
    fake = b"x\nBF listed end\nBF proc 1 agent S 1 default planted\n" + b"y" * 300
    _proc(proc, 120, cmdline=fake)
    rows = _list(proc)
    assert [row["pid"] for row in rows] == [120]
    assert rows[0]["command"].startswith("x?BF listed end?BF proc 1")
    assert len(rows[0]["command"]) == 200


# Every line break Python's str.splitlines() knows besides "\n".
BREAKS = ["\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]


def test_a_carriage_return_or_a_nel_adds_no_row(tmp_path):
    """A command line holding \\r or NEL adds no row."""
    proc = tmp_path / "proc"
    _proc(proc, 120, cmdline=b"x\rBF proc 4242 agent S 1 default forged\x00")
    _proc(
        proc,
        121,
        cmdline="y\u0085BF proc 4343 agent S 1 default forged2".encode() + b"\x00",
    )
    rows = _list(proc)
    assert sorted(row["pid"] for row in rows) == [120, 121], rows


@pytest.mark.parametrize("brk", BREAKS, ids=[f"U+{ord(b):04X}" for b in BREAKS])
def test_no_line_break_in_a_command_line_forges_a_row_or_a_clock(tmp_path, brk):
    """S-1: one listing line per process, whatever its command line holds. A
    forged row, clock line or end of listing stays inside the real row, and
    the next process's start time is read from the real clock."""
    proc = tmp_path / "proc"
    forged = (
        f"x{brk}BF proc 4242 agent S 1 default forged{brk}BF clock 1 1"
        f"{brk}BF listed end"
    )
    _proc(proc, 120, cmdline=forged.encode() + b"\x00")
    _proc(proc, 121, start=12345)
    rows = {row["pid"]: row for row in _list(proc)}
    assert set(rows) == {120, 121}
    assert all(32 <= ord(c) < 127 for c in rows[120]["command"]), rows[120]
    # A break of 2 or 3 bytes (NEL, U+2028) is 2 or 3 of them.
    assert rows[120]["command"].lstrip("x?").startswith("BF proc 4242")
    assert rows[121]["started_at"] == datetime.fromtimestamp(
        BTIME + 12345 / 100, UTC
    ).isoformat(timespec="seconds")


# Printable text dash's echo (Debian's and Ubuntu's sh) reads as escapes
# (S-4): \n and \0012 would print a forged row, \c in a middle process would
# hide the next row, and in the last one the end of the listing.
ESCAPES = {
    "backslash-n": {140: b"x\\nBF proc 4242 agent S 1 default forged\x00"},
    "backslash-c-middle": {141: b"x\\c\x00", 142: b"sleep\x00900\x00"},
    "backslash-c-last": {999: b"y\\c\x00"},
    "octal": {150: b"x\\0012BF proc 4343 agent S 1 default forged\x00"},
}


def add_escape_cases(proc: Path, *cases: str) -> list[int]:
    """The ESCAPES processes in a stand-in /proc; their pids."""
    pids = []
    for case in cases or tuple(ESCAPES):
        for pid, cmdline in ESCAPES[case].items():
            _proc(proc, pid, cmdline=cmdline)
            pids.append(pid)
    return sorted(pids)


@pytest.mark.parametrize("case", sorted(ESCAPES))
def test_a_backslash_escape_in_a_command_line_stays_text(tmp_path, case):
    """S-4: each row is printed with printf, never echo, so a command line's
    \\n, \\c or \\0nnn stays the text it is: no forged row, no hidden row, the
    end of the listing kept. (The host's sh on the test VMs is dash.)
    Mutant: echo back."""
    proc = tmp_path / "proc"
    pids = add_escape_cases(proc, case)
    rows = _list(proc)
    assert sorted(row["pid"] for row in rows) == pids, rows
    want = {
        pid: cmdline.rstrip(b"\x00").replace(b"\x00", b" ").decode()
        for pid, cmdline in ESCAPES[case].items()
    }
    assert {row["pid"]: row["command"] for row in rows} == want


def test_a_name_cannot_set_the_clock(tmp_path):
    """The kernel writes a newline in a Name as the two
    characters \\n; printed as text, they can't start a clock line."""
    proc = tmp_path / "proc"
    _proc(proc, 130, cmdline=b"", name="\\nBF clock 9 9\\n")
    _proc(proc, 131, start=12345)
    rows = {row["pid"]: row for row in _list(proc)}
    assert rows[131]["started_at"] == datetime.fromtimestamp(
        BTIME + 12345 / 100, UTC
    ).isoformat(timespec="seconds")


# A comm with a newline, as a process can set it (prctl, /proc/self/comm):
# stat prints it raw, status as the two characters \n (N-6).
COMMS = {"glob": ") * * * * * *\n", "fields": ") 7 7 7 7 7 7\n", "plain": "x\n"}


def add_comm_newline(proc: Path, comm: str, pid: int = 160) -> None:
    _proc(proc, pid, name="x", start=12345)
    (proc / str(pid) / "stat").write_text(
        f"{pid} ({comm}) S " + " ".join(["0"] * 18) + " 12345 0 0\n"
    )
    status = (proc / str(pid) / "status").read_text()
    (proc / str(pid) / "status").write_text(
        status.replace("Name:\tx\n", "Name:\t" + comm.replace("\n", "\\n") + "\n", 1)
    )


def digit_named_files(folder: Path) -> Path:
    """Root's cwd in a trial is the image's WORKDIR, where the agent may name
    files: digits, for a glob in the stat fields to expand to."""
    folder.mkdir()
    for name in ("77", "78", "79", "80"):
        (folder / name).write_text("")
    return folder


def add_globbing_stat(proc: Path, pid: int = 161) -> None:
    """A stat whose fields after the comm are *: no kernel writes that; it
    shows the split never globs (set -f), which the joined lines otherwise
    leave nothing to test."""
    _proc(proc, pid, start=12345)
    (proc / str(pid) / "stat").write_text(
        f"{pid} (x) S " + " ".join(["*"] * 18) + " 12345 0 0\n"
    )


def add_stat_case(proc: Path, case: str) -> None:
    if case == "fields-are-globs":
        add_globbing_stat(proc)
    else:
        add_comm_newline(proc, COMMS[case])


STAT_CASES = [*sorted(COMMS), "fields-are-globs"]


@pytest.mark.parametrize("case", STAT_CASES)
def test_a_newline_in_the_comm_cannot_choose_the_start(tmp_path, case):
    """Every line of stat is read and its fields split with globbing off, so a
    comm's newline, fields or glob can't set the start time. Run from a folder
    of files named by digits. Mutants: one line of stat only; globbing left on."""
    proc = tmp_path / "proc"
    add_stat_case(proc, case)
    (row,) = _list(proc, cwd=digit_named_files(tmp_path / "cwd"))
    assert row["start_ticks"] == 12345, row


def test_a_name_with_a_line_break_stays_in_its_row(tmp_path):
    """A process with no command line is named by its status Name, which it
    sets too: mapped the same way."""
    proc = tmp_path / "proc"
    _proc(proc, 130, cmdline=b"", name="n\u0085BF")
    rows = _list(proc)
    assert [(row["pid"], row["command"]) for row in rows] == [(130, "[n??BF]")]


def test_the_parser_ends_a_line_at_a_newline_only():
    """S-1, the parser's side: a line break other than "\\n" inside a row is
    part of that row (the shell side leaves none; both must hold)."""
    out = (
        "BF clock 1800000000 100\n"
        "BF proc 7 agent S 5 default a\rBF proc 8 agent S 5 default b\n"
        "BF proc 9 agent S 5 default c\u2028BF clock 1 1\n"
        "BF listed end\n"
    )
    rows = _quiesce_listing.parse_listing(out)
    assert [row["pid"] for row in rows] == [7, 9]
    assert rows[1]["started_at"] == datetime.fromtimestamp(
        1800000000 + 5 / 100, UTC
    ).isoformat(timespec="seconds")


def test_a_listing_that_did_not_finish_is_not_read():
    with pytest.raises(ValueError, match="did not finish"):
        _quiesce_listing.parse_listing(
            "BF clock 1 100\nBF proc 5 agent S 1 default x\n"
        )


def _row(pid: int, command: str, sigterm: str = "default") -> dict:
    return {
        "pid": pid,
        "user": "agent",
        "command": command,
        "state": "S",
        "started_at": None,
        "start_ticks": pid * 10,
        "sigterm": sigterm,
        "ended_by": _quiesce_listing.ENDED_BY[sigterm],
    }


async def _quiesce(
    tmp_path, monkeypatch, *, listings, killed=0, fails=None, env=None
) -> dict | None:
    """Run ``_quiesce_after_grading`` with *listings* (before, after; an
    exception raises) and a fake quiesce that records *killed* as the shell
    kill does, or raises *fails*."""
    answers = list(listings)

    async def list_processes(env, accounts, *, timeout_sec):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return [dict(row) for row in answer]

    monkeypatch.setattr(_quiesce_listing, "list_processes", list_processes)
    calls = []

    async def quiesce_agent(env, sandbox_user):
        calls.append(sandbox_user)
        if fails is not None:
            raise fails
        kill_record.record_kill_before_grading(sandbox_user, killed)

    planes = SimpleNamespace(quiesce_agent=quiesce_agent)
    env = env if env is not None else SimpleNamespace()
    await _setup._quiesce_after_grading(planes, env, "agent", tmp_path)
    assert calls == ["agent"]
    path = tmp_path / kill_record.KILL_AFTER_GRADING_RECORD
    return json.loads(path.read_text()) if path.exists() else None


@pytest.mark.asyncio
async def test_the_record_names_the_processes_the_quiesce_ended(tmp_path, monkeypatch):
    before = [_row(41, "sleep 600"), _row(42, "python3 ignore.py", "ignored")]
    record = await _quiesce(tmp_path, monkeypatch, listings=[before, []], killed=2)
    assert record is not None
    assert record["user"] == "agent"
    assert record["accounts"] == ["agent"]
    assert record["killed"] == 2 and record["killed_total"] == 2
    assert record["signal_source"] == "sigterm_disposition"
    assert record["signals"] == _quiesce_listing.SIGNALS
    assert record["started_at"] <= record["ended_at"]
    assert record["left"] == []
    assert [
        (r["pid"], r["command"], r["ended_by"], r["ended"]) for r in record["processes"]
    ] == [
        (41, "sleep 600", "SIGTERM", True),
        (42, "python3 ignore.py", "SIGKILL", True),
    ]
    assert "error" not in record and "listing_error" not in record


@pytest.mark.asyncio
async def test_a_trial_with_nothing_left_records_an_empty_list(tmp_path, monkeypatch):
    record = await _quiesce(tmp_path, monkeypatch, listings=[[], []], killed=0)
    assert record is not None
    assert record["processes"] == []
    assert record["left"] == []
    assert record["killed"] == 0


@pytest.mark.asyncio
async def test_a_failed_quiesce_is_recorded_with_what_is_left(tmp_path, monkeypatch):
    before = [_row(41, "sleep 600"), _row(43, "survivor")]
    record = await _quiesce(
        tmp_path,
        monkeypatch,
        listings=[before, [_row(43, "survivor")]],
        fails=RuntimeError("Sandbox-user writers remain: 43 (survivor)"),
    )
    assert record is not None
    assert "Sandbox-user writers remain" in record["error"]
    assert "killed" not in record
    assert [(r["pid"], r["ended"]) for r in record["processes"]] == [
        (41, True),
        (43, False),
    ]
    # A process that did not end has no signal.
    assert [r["ended_by"] for r in record["processes"]] == ["SIGTERM", None]
    assert [r["pid"] for r in record["left"]] == [43]


@pytest.mark.asyncio
async def test_a_failed_listing_still_writes_the_kill(tmp_path, monkeypatch):
    record = await _quiesce(
        tmp_path,
        monkeypatch,
        listings=[RuntimeError("exec failed"), RuntimeError("exec failed")],
        killed=3,
    )
    assert record is not None
    assert record["killed"] == 3
    assert record["processes"] is None and record["left"] is None
    assert record["listing_error"].startswith("before the quiesce: ")
    assert "; after the quiesce: " in record["listing_error"]


@pytest.mark.asyncio
async def test_a_reused_pid_is_another_process_left_running(tmp_path, monkeypatch):
    """The same pid with another start time is another process: the
    listed one ended, the new one is left. Mutant: matching by pid alone."""
    old = {**_row(41, "sleep 600"), "start_ticks": 100}
    new = {**_row(41, "sleep 900"), "start_ticks": 900}
    record = await _quiesce(tmp_path, monkeypatch, listings=[[old], [new]], killed=1)
    assert record is not None
    assert [(r["pid"], r["start_ticks"], r["ended"]) for r in record["processes"]] == [
        (41, 100, True)
    ]
    assert record["processes"][0]["ended_by"] == "SIGTERM"
    assert [(r["pid"], r["start_ticks"]) for r in record["left"]] == [(41, 900)]


@pytest.mark.asyncio
async def test_the_switched_accounts_are_listed_too(tmp_path, monkeypatch):
    seen = []

    async def list_processes(env, accounts, *, timeout_sec):
        seen.append(list(accounts))
        return []

    monkeypatch.setattr(_quiesce_listing, "list_processes", list_processes)

    async def quiesce_agent(env, sandbox_user):
        kill_record.record_kill_before_grading(sandbox_user, 0)

    env = SimpleNamespace(agent_identity_switch=("builder",))
    await _setup._quiesce_after_grading(
        SimpleNamespace(quiesce_agent=quiesce_agent), env, "agent", tmp_path
    )
    assert seen == [["builder", "agent"], ["builder", "agent"]]
    record = json.loads((tmp_path / kill_record.KILL_AFTER_GRADING_RECORD).read_text())
    assert record["accounts"] == ["builder", "agent"]


RECORD = {"user": "agent", "killed": 0, "killed_total": 0, "processes": [], "left": []}


def test_a_bundle_of_an_audit_trial_carries_the_record(tmp_path):
    """``bench bundle`` keeps the record beside kill-before-grading's
    (``evidence_bundle.TRIAL_FILES``)."""
    from benchflow.evidence_bundle import build_bundle

    job = tmp_path / "jobs" / "run"
    trial = job / "task-a__abc123"
    trial.mkdir(parents=True)
    (job / "run-card.json").write_text('{"agent": {"name": "oracle"}}\n')
    (trial / "result.json").write_text(
        json.dumps({"task_name": "task-a", "rewards": {"reward": 1.0}})
    )
    (trial / kill_record.KILL_AFTER_GRADING_RECORD).write_text(json.dumps(RECORD))
    result = build_bundle(job, tmp_path / "run.zip")
    with zipfile.ZipFile(result.path) as archive:
        member = f"task-a__abc123/{kill_record.KILL_AFTER_GRADING_RECORD}"
        assert json.loads(archive.read(member)) == RECORD


@pytest.mark.parametrize("review", ["first", "again"])
def test_a_reviewers_copy_of_the_trial_keeps_the_record(tmp_path, review):
    """A reviewer's copy of the trial keeps the record. A re-review's
    copy is held to the rollout layout (``review/wrapper._REVIEW_LAYOUT``)."""
    from benchflow.review.config import Rubric
    from benchflow.review.wrapper import assemble_review_task

    rubric = Rubric.model_validate(
        {"criteria": [{"name": "sound", "description": "d", "guidance": "PASS"}]}
    )
    rollout = tmp_path / "rollout-a"
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "config.json").write_text("{}")
    solver = {"task_name": "t", "rewards": {"reward": 1.0}, "verifier_error": None}
    result = dict(solver)
    if review == "again":
        (rollout / "solver.json").write_text(json.dumps(solver))
        result["scoring"] = {"status": "complete", "rubric_reward": 0.5}
    (rollout / "result.json").write_text(json.dumps(result))
    (rollout / kill_record.KILL_AFTER_GRADING_RECORD).write_text(json.dumps(RECORD))
    dest, _ = assemble_review_task(rollout, None, rubric, tmp_path / "w")
    copy = dest / "evidence" / "trial" / kill_record.KILL_AFTER_GRADING_RECORD
    assert json.loads(copy.read_text()) == RECORD
