"""After test.sh exits, only the names it wrote are put back (#1247).

In integrity off and audit a process left in the sandbox (the agent's, or one
a root ``solve.sh`` left) lives through grading, and on Docker the mounted
``/logs/verifier`` goes back to the host user once test.sh has exited: a
process with that uid, or root's, can add names there. The rewards are read
from the copy taken at exit (``verifier-at-exit/``), so a plant never scores;
but on 0.10.0 put_back undid only regular files with a reward's name. Every
other name stayed in the trial's ``verifier/`` unflagged: a file, a folder, a
file in one of test.sh's folders, a link to a host path, a ``reward.json``
link. Now each name test.sh did not leave and BenchFlow did not write is
removed from ``verifier/`` and named in ``integrity/verifier-inputs.json``.

BenchFlow's own records written after the exit stay as written: the
verifier's, the rollout's (``trusted-path.json``, ``network-policy.json``) and
the task.md judge's (``reward.json``, ``reward.txt``, ``review.json``,
``judgments.jsonl``, ``taskmd-judge/``). On 0.10.0 the judge's reward files
were taken for planted: moved to ``verifier-after-exit/`` and flagged.

Mutant: put back every name (0.10.0's rule: only reward-named files are
undone).
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from benchflow.task import verifier_outputs as vo
from benchflow.task.paths import RolloutPaths

# test.sh's outputs at its exit: files, a folder with a file, an empty folder
# and a link of its own.
TEST_SH_FILES = {"reward.txt": "1\n", "out.txt": "ran\n", "logs/a.txt": "log\n"}


def _trial(tmp_path: Path) -> tuple[Path, RolloutPaths]:
    trial = tmp_path / "trial"
    paths = RolloutPaths(rollout_dir=trial)
    paths.mkdir()
    verifier = trial / "verifier"
    for rel, text in TEST_SH_FILES.items():
        path = verifier / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (verifier / "empty").mkdir()
    (verifier / "latest").symlink_to("out.txt")
    (trial / "integrity").mkdir(exist_ok=True)
    (trial / "integrity" / "verifier-inputs.json").write_text(
        json.dumps({"status": "checked", "flags": [], "sections": {}})
    )
    return trial, paths


def _verifier(paths: RolloutPaths):
    """A Verifier as the rollout's verify leaves it at test.sh's exit."""
    from benchflow.task.verifier_core import Verifier

    verifier = Verifier.__new__(Verifier)
    verifier._rollout_paths = paths
    verifier._sandbox = MagicMock()
    verifier._reward_range = None
    verifier.outputs_at_exit = None
    verifier._written_after_exit = {}
    verifier._take_outputs_at_exit()
    return verifier


def _plant(verifier_dir: Path, kind: str, host: Path) -> str:
    """What a process left in the sandbox adds after test.sh's exit; returns
    the planted name."""
    if kind == "file":
        (verifier_dir / "extra.txt").write_text("x")
        return "extra.txt"
    if kind == "file-in-test-sh-folder":
        (verifier_dir / "logs" / "planted.txt").write_text("z")
        return "logs/planted.txt"
    if kind == "folder":
        (verifier_dir / "newdir" / "deeper").mkdir(parents=True)
        (verifier_dir / "newdir" / "f.txt").write_text("y")
        (verifier_dir / "newdir" / "deeper" / "g.txt").write_text("y")
        return "newdir"
    if kind == "empty-folder":
        (verifier_dir / "emptydir").mkdir()
        return "emptydir"
    if kind == "link-to-host":
        (verifier_dir / "hostlink").symlink_to(host)
        return "hostlink"
    if kind == "reward-link":
        (verifier_dir / "reward.json").symlink_to(host)
        return "reward.json"
    if kind == "dangling-link":
        (verifier_dir / "gone").symlink_to(verifier_dir / "no-such-name")
        return "gone"
    if kind == "fifo":
        os.mkfifo(verifier_dir / "pipe")
        return "pipe"
    raise ValueError(kind)


PLANTS = (
    "file",
    "file-in-test-sh-folder",
    "folder",
    "empty-folder",
    "link-to-host",
    "reward-link",
    "dangling-link",
    "fifo",
)


@pytest.mark.parametrize("kind", PLANTS)
def test_a_name_test_sh_did_not_write_is_removed_and_flagged(tmp_path, kind):
    """E1. Each name a process adds after test.sh's exit is gone from the
    trial's verifier/ once the rollout puts the copy back, and the integrity
    record names it; the reward read from the copy is unchanged."""
    trial, paths = _trial(tmp_path)
    host = tmp_path / "host-secret.json"
    host.write_text('{"reward": 1}')
    verifier = _verifier(paths)
    planted = _plant(trial / "verifier", kind, host)

    flags = vo.put_back_and_record(trial, verifier._sandbox, sandbox_gone=True)

    left = trial / "verifier" / planted
    assert not (left.exists() or left.is_symlink()), (
        f"{planted} ({kind}) stays in the trial's verifier/"
    )
    named = {(f["path"], f["kind"]) for f in flags}
    assert (f"/logs/verifier/{planted}", "planted") in named, flags
    record = json.loads((trial / "integrity" / "verifier-inputs.json").read_text())
    assert record["status"] == "flagged", record
    assert any(
        f["path"] == f"/logs/verifier/{planted}" and f["kind"] == "planted"
        for f in record["flags"]
    ), record
    # The reward is the one test.sh left at its exit.
    assert verifier._parse_reward_text() == {"reward": 1.0}
    # The host's file a link named is untouched.
    assert host.read_text() == '{"reward": 1}'


def test_every_plant_at_once_and_a_second_put_back_finds_nothing(tmp_path):
    """E1, all the plants in one trial: each named once; the rollout's second
    put_back (at teardown) removes and adds nothing."""
    trial, paths = _trial(tmp_path)
    host = tmp_path / "host.txt"
    host.write_text("host")
    verifier = _verifier(paths)
    planted = {_plant(trial / "verifier", kind, host) for kind in PLANTS}

    flags = vo.put_back_and_record(trial, verifier._sandbox, sandbox_gone=True)

    assert {f["path"] for f in flags if f["kind"] == "planted"} == {
        f"/logs/verifier/{name}" for name in planted
    }, flags
    assert vo.put_back_and_record(trial, verifier._sandbox, sandbox_gone=True) == []
    record = json.loads((trial / "integrity" / "verifier-inputs.json").read_text())
    assert len([f for f in record["flags"] if f["kind"] == "planted"]) == len(planted)


def test_the_names_test_sh_left_stay(tmp_path):
    """Guard row: test.sh's own files, folders (an empty one too) and link
    stay as test.sh left them, unflagged."""
    trial, paths = _trial(tmp_path)
    verifier = _verifier(paths)

    assert vo.put_back_and_record(trial, verifier._sandbox) == []

    verifier_dir = trial / "verifier"
    for rel, text in TEST_SH_FILES.items():
        assert (verifier_dir / rel).read_text() == text
    assert (verifier_dir / "empty").is_dir()
    assert os.readlink(verifier_dir / "latest") == "out.txt"
    record = json.loads((trial / "integrity" / "verifier-inputs.json").read_text())
    assert record["status"] == "checked" and record["flags"] == [], record


def test_a_link_test_sh_left_that_points_elsewhere_after_exit_is_removed(tmp_path):
    """E1: test.sh's own link, pointed at a host path after its exit, is
    removed and flagged with both targets; the link as test.sh left it is a
    name test.sh wrote, never a pass for whatever it points at later."""
    trial, paths = _trial(tmp_path)
    verifier = _verifier(paths)
    latest = trial / "verifier" / "latest"
    latest.unlink()
    latest.symlink_to("/etc/hostname")

    flags = vo.put_back_and_record(trial, verifier._sandbox, sandbox_gone=True)

    assert not (latest.exists() or latest.is_symlink())
    (row,) = [f for f in flags if f["path"] == "/logs/verifier/latest"]
    assert row["kind"] == "edited", row
    assert row["before"] == {"type": "link", "target": "out.txt"}, row
    assert row["after"]["target"] == "/etc/hostname", row


def test_a_name_two_benchflow_writers_wrote_is_kept_in_either_version(tmp_path):
    """The verifier's own record and another BenchFlow writer can both write
    one name after the exit (a task.md judge's reward.json beside the
    verifier's): either version is BenchFlow's and stays. Anything else is
    put back to the last writer's version and flagged."""
    trial, paths = _trial(tmp_path)
    verifier = _verifier(paths)
    verifier._write_verifier_file(paths.reward_json_path, '{"reward": 0.5}')
    vo.wrote(paths.reward_json_path, b'{"reward": 1.0}')
    reward_json = trial / "verifier" / "reward.json"
    for kept in ('{"reward": 0.5}', '{"reward": 1.0}'):
        reward_json.write_text(kept)
        assert (
            vo.put_back(trial, verifier.outputs_at_exit, verifier._written_after_exit)
            == []
        )
        assert reward_json.read_text() == kept

    reward_json.write_text('{"reward": 9}')
    flags = vo.put_back(trial, verifier.outputs_at_exit, verifier._written_after_exit)

    assert reward_json.read_text() == '{"reward": 1.0}'
    assert [(f["path"], f["kind"]) for f in flags] == [
        ("/logs/verifier/reward.json", "edited")
    ]


def test_benchflows_own_records_after_the_exit_stay(tmp_path):
    """Guard row: what BenchFlow itself writes to verifier/ after test.sh's
    exit stays as written, unflagged: the verifier's own records and the
    rollout's (trusted-path.json, network-policy.json)."""
    from benchflow.rollout import _setup

    trial, paths = _trial(tmp_path)
    verifier = _verifier(paths)
    verifier._write_verifier_file(paths.verifier_dir / "verifier-network.json", "{}")
    verifier._write_verifier_file(paths.test_stdout_path, "out\n")
    report = SimpleNamespace(to_record=lambda: {"kept": []}, absent_root_only=())
    _setup._write_trusted_path_record(paths, report, False, ())
    _setup._write_network_policy_record({"held": False}, paths)

    assert vo.put_back_and_record(trial, verifier._sandbox) == []

    verifier_dir = trial / "verifier"
    assert (verifier_dir / "verifier-network.json").read_text() == "{}"
    assert (verifier_dir / "test-stdout.txt").read_text() == "out\n"
    assert json.loads((verifier_dir / "trusted-path.json").read_text())["kept"] == []
    assert json.loads((verifier_dir / "network-policy.json").read_text()) == {
        "held": False
    }


def test_the_taskmd_judges_records_stay_and_are_not_planted(tmp_path):
    """E1, the task.md judge (draft 2): after test.sh's exit it moves test.sh's
    reward aside (``script-reward.txt``) and writes the trial's own
    ``reward.json``, ``reward.txt``, ``review.json``, ``judgments.jsonl`` and
    ``taskmd-judge/``. All stay as the judge wrote them, unflagged. On 0.10.0
    its reward files were taken for planted: moved to verifier-after-exit/,
    two ``planted`` flags, the trial ``flagged`` (measured on Docker with
    ``examples/hello-world`` as the oracle in audit)."""
    from benchflow._utils.reachable_write import staging_for
    from benchflow.taskmd import verify as tv

    trial, paths = _trial(tmp_path)
    verifier = _verifier(paths)
    for name in ("reward.txt", "reward.json"):
        path = paths.verifier_dir / name
        if path.exists():
            path.rename_entry(f"script-{name}")
    (paths.verifier_dir / tv.JUDGMENTS).unlink(missing_ok=True)
    tv._append_jsonl(paths.verifier_dir / tv.JUDGMENTS, {"criterion": "c1"})
    target = paths.verifier_dir / tv.JUDGE_DIR
    with staging_for(target, same_filesystem=True) as host_copy:
        fs = host_copy / "fs"
        (fs / "app").mkdir(parents=True)
        (fs / "app" / "hello.txt").write_text("Hello, world!")
        tv._publish_judge_copy(fs, target)
    tv._write_json(paths.verifier_dir / "review.json", {"format": "task.md draft 2"})
    tv._write_json(paths.reward_json_path, {"reward": 1.0})
    tv._host_write(paths.reward_text_path, "1.0\n")

    flags = vo.put_back_and_record(trial, verifier._sandbox)

    assert flags == [], flags
    verifier_dir = trial / "verifier"
    assert json.loads((verifier_dir / "reward.json").read_text()) == {"reward": 1.0}
    assert (verifier_dir / "reward.txt").read_text() == "1.0\n"
    assert (verifier_dir / "script-reward.txt").read_text() == "1\n"
    assert json.loads((verifier_dir / "review.json").read_text())["format"]
    assert (verifier_dir / tv.JUDGMENTS).read_text().strip()
    assert (verifier_dir / tv.JUDGE_DIR / "fs" / "app" / "hello.txt").is_file()
    assert not (trial / vo.AFTER_EXIT).exists()
    record = json.loads((trial / "integrity" / "verifier-inputs.json").read_text())
    assert record["status"] == "checked" and record["flags"] == [], record


def test_a_new_test_command_forgets_an_earlier_copy(tmp_path):
    """A multi-step task's earlier verify left its copy on the sandbox; once a
    new test command is issued, that copy is never put back over the new
    outputs (the new command's own copy, if it exits, replaces it)."""
    trial, paths = _trial(tmp_path)
    verifier = _verifier(paths)
    holder = verifier._sandbox
    (trial / "verifier" / "timeout-reward.txt").write_text("0\n")

    vo.forget(holder)

    assert holder._benchflow_verifier_outputs_at_exit is None
    assert vo.put_back_and_record(trial, holder) == []
    assert (trial / "verifier" / "timeout-reward.txt").read_text() == "0\n"


def test_both_places_a_test_command_is_issued_forget_the_earlier_copy():
    """Each issue of a test command (test.sh's, and a service's) forgets
    the copy first."""
    import inspect

    from benchflow.task import verifier_core

    source = inspect.getsource(verifier_core)
    issues = source.count("self._sandbox._benchflow_test_command_issued = True")
    assert issues == 2
    assert source.count("verifier_outputs.forget(self._sandbox)") == issues


def test_a_benchflow_write_through_a_held_folder_is_kept_without_a_call(tmp_path):
    """Every BenchFlow write to verifier/ after the exit goes through a held
    capability, which notes it: a later writer that never calls
    verifier_outputs (here, as Rollout._record_taskmd_timed_out writes
    reward.json) keeps its record."""
    from benchflow._utils.reachable_write import destination

    trial, paths = _trial(tmp_path)
    verifier = _verifier(paths)
    destination(paths.reward_json_path).write('{"reward": 1.0, "timed_out": false}\n')

    assert vo.put_back_and_record(trial, verifier._sandbox) == []
    assert "timed_out" in (trial / "verifier" / "reward.json").read_text()


def test_put_backs_own_restores_are_not_noted_as_benchflows(tmp_path):
    """put_back's own writes are test.sh's outputs going back, never
    BenchFlow's later records."""
    trial, paths = _trial(tmp_path)
    verifier = _verifier(paths)
    (trial / "verifier" / "reward.txt").write_text("9\n")

    flags = vo.put_back_and_record(trial, verifier._sandbox)

    assert [(f["path"], f["kind"]) for f in flags] == [
        ("/logs/verifier/reward.txt", "edited")
    ]
    assert "reward.txt" not in vo._journal(trial)


# -- the observer -----------------------------------------------------------------


def test_an_observer_that_raises_never_fails_a_write(tmp_path, monkeypatch):
    from benchflow._utils import reachable_write as rw

    def broken(folder, parts, data):
        raise RuntimeError("an observer's own fault")

    monkeypatch.setattr(rw, "_WRITE_OBSERVERS", [broken, *rw._WRITE_OBSERVERS])
    folder = tmp_path / "held"
    folder.mkdir()
    target = rw.ReachablePath.directory(folder) / "record.json"
    target.write("{}")
    assert (folder / "record.json").read_text() == "{}"


def test_a_put_back_restore_never_shields_a_later_plant_of_that_name(tmp_path):
    """put_back writes test.sh's file back; a process that replaces that name
    again (here with a link to a host path) is undone again by the next
    put_back, and the name is never BenchFlow's."""
    trial, paths = _trial(tmp_path)
    verifier = _verifier(paths)
    out = trial / "verifier" / "out.txt"
    for _ in range(2):
        out.unlink()
        out.symlink_to("/etc/hostname")
        flags = vo.put_back(
            trial, verifier.outputs_at_exit, verifier._written_after_exit, remove=True
        )
        assert out.is_file() and not out.is_symlink()
        assert out.read_text() == "ran\n"
        assert ("/logs/verifier/out.txt", "edited") in {
            (f["path"], f["kind"]) for f in flags
        }
        assert "out.txt" not in vo._journal(trial)


def test_the_taskmd_timed_out_record_stays(tmp_path):
    """Rollout._record_taskmd_timed_out rewrites reward.json after the verify
    to add timed_out; it calls nothing of verifier_outputs, and its record
    stays (the Docker row with examples/hello-world checks it in a trial)."""
    from benchflow.rollout import Rollout

    trial, paths = _trial(tmp_path)
    verifier = _verifier(paths)
    from benchflow.taskmd.materialize import SOURCE_FILE

    # A materialized task.md package, as taskmd_metadata reads one.
    package = tmp_path / "package"
    package.mkdir()
    (package / SOURCE_FILE).write_text("{}")
    (package / "task.md").write_text("---\nmetadata:\n  taskmd: {name: t}\n---\nbody\n")
    rollout = Rollout.__new__(Rollout)
    rollout._rollout_paths = paths
    rollout._rewards = {"reward": 1.0}
    rollout._config = SimpleNamespace(task_path=package)
    rollout._agent_ran_out_of_time = False

    rollout._record_taskmd_timed_out()
    flags = vo.put_back_and_record(trial, verifier._sandbox)

    assert flags == [], flags
    assert json.loads((trial / "verifier" / "reward.json").read_text()) == {
        "reward": 1.0,
        "timed_out": False,
    }


def test_a_write_that_bypasses_the_held_folder_is_planted(tmp_path):
    """A writer that goes around reachable_write (a plain path) is not known
    as BenchFlow's: its name is removed and flagged, so such a writer fails
    loud, never open."""
    trial, paths = _trial(tmp_path)
    verifier = _verifier(paths)
    (trial / "verifier" / "late-record.json").write_text("{}")

    flags = vo.put_back_and_record(trial, verifier._sandbox, sandbox_gone=True)

    assert not (trial / "verifier" / "late-record.json").exists()
    assert [(f["path"], f["kind"]) for f in flags] == [
        ("/logs/verifier/late-record.json", "planted")
    ]


# -- while the sandbox lives ---------------------------------------------------------


def test_while_the_sandbox_lives_nothing_is_removed_and_every_plant_is_named(tmp_path):
    """The first put_back runs after grading with the sandbox still up, where
    a root process left in it may still hold a planted folder (its cwd): it
    names every plant and removes none. The pass after the sandbox's stop
    removes them, and the record's flags say so."""
    trial, paths = _trial(tmp_path)
    host = tmp_path / "host.txt"
    host.write_text("host")
    verifier = _verifier(paths)
    planted = {_plant(trial / "verifier", kind, host) for kind in PLANTS}

    first = vo.put_back_and_record(trial, verifier._sandbox)

    for name in planted:
        there = trial / "verifier" / name
        assert there.exists() or there.is_symlink(), (
            f"{name} removed while the sandbox lives"
        )
    rows = {f["path"]: f for f in first if f["kind"] == "planted"}
    assert set(rows) == {f"/logs/verifier/{n}" for n in planted}, first
    assert all(r["removed"] is False and r["pending"] for r in rows.values()), rows

    vo.put_back_and_record(trial, verifier._sandbox, sandbox_gone=True)

    for name in planted:
        there = trial / "verifier" / name
        assert not (there.exists() or there.is_symlink()), f"{name} left after the stop"
    record = json.loads((trial / "integrity" / "verifier-inputs.json").read_text())
    planted_rows = [f for f in record["flags"] if f["kind"] == "planted"]
    assert len(planted_rows) == len(planted)
    assert all(f["removed"] is True and "pending" not in f for f in planted_rows), (
        planted_rows
    )


def test_a_planted_reward_file_is_kept_aside_and_removed(tmp_path):
    """A regular reward.json a process adds after the exit is
    copied to verifier-after-exit/ for the record, then removed."""
    trial, paths = _trial(tmp_path)
    verifier = _verifier(paths)
    (trial / "verifier" / "reward.json").write_text('{"reward": 1}')

    flags = vo.put_back_and_record(trial, verifier._sandbox, sandbox_gone=True)

    assert not (trial / "verifier" / "reward.json").exists()
    assert (trial / vo.AFTER_EXIT / "reward.json").read_text() == '{"reward": 1}'
    assert [(f["path"], f["kind"], f["removed"]) for f in flags] == [
        ("/logs/verifier/reward.json", "planted", True)
    ]


def test_a_name_added_to_a_tree_benchflow_published_is_planted(tmp_path):
    """A tree BenchFlow publishes after the exit (the task.md
    judge's taskmd-judge/) is listed then; a name added in it later is
    planted, and the published names stay."""
    from benchflow._utils.reachable_write import staging_for
    from benchflow.taskmd import verify as tv

    trial, paths = _trial(tmp_path)
    verifier = _verifier(paths)
    target = paths.verifier_dir / tv.JUDGE_DIR
    with staging_for(target, same_filesystem=True) as host_copy:
        fs = host_copy / "fs"
        (fs / "app").mkdir(parents=True)
        (fs / "app" / "hello.txt").write_text("Hello, world!")
        tv._publish_judge_copy(fs, target)
    judge = trial / "verifier" / tv.JUDGE_DIR
    (judge / "fs" / "app" / "planted.txt").write_text("x")

    flags = vo.put_back_and_record(trial, verifier._sandbox, sandbox_gone=True)

    assert (judge / "fs" / "app" / "hello.txt").read_text() == "Hello, world!"
    assert not (judge / "fs" / "app" / "planted.txt").exists()
    assert [(f["path"], f["kind"]) for f in flags] == [
        (f"/logs/verifier/{tv.JUDGE_DIR}/fs/app/planted.txt", "planted")
    ]


def _exchange():
    """renameat2(RENAME_EXCHANGE) by syscall, or None where it isn't known."""
    import ctypes
    import platform

    number = {"x86_64": 316, "aarch64": 276}.get(platform.machine())
    if not sys.platform.startswith("linux") or number is None:
        return None
    libc = ctypes.CDLL(None, use_errno=True)

    def exchange(a: bytes, b: bytes) -> None:
        libc.syscall(number, -100, a, -100, b, 2)

    return exchange


def test_a_swapped_link_under_a_read_only_folder_never_reaches_a_host_file(tmp_path):
    """A planted folder holds a 0555
    folder (whose names the host can't unlink until it is writable) beside a
    link to a host folder, and something swaps the two in a loop while the
    folder is removed. Removal goes through the held folder's descriptors
    only, so no attempt deletes the host's file or changes its folder's mode.
    The mutant, a removal that makes the folder writable and retries by path,
    deletes it (3 in 36 attempts on a review VM)."""
    import threading
    import time

    exchange = _exchange()
    if exchange is None:
        pytest.skip("renameat2 by syscall is Linux x86_64/aarch64 only")
    host = tmp_path / "host-folder"
    host.mkdir()
    (host / "victim").write_text("host data")
    host_mode = host.stat().st_mode
    for attempt in range(60):
        trial, paths = _trial(tmp_path / f"t{attempt}")
        verifier = _verifier(paths)
        planted = trial / "verifier" / "newdir"
        (planted / "a").mkdir(parents=True)
        (planted / "a" / "victim").write_text("planted")
        os.chmod(planted / "a", 0o555)
        os.symlink(str(host), planted / "s")
        stop = threading.Event()
        a, s = str(planted / "a").encode(), str(planted / "s").encode()

        def swap(stop=stop, a=a, s=s) -> None:
            while not stop.is_set():
                exchange(a, s)

        racer = threading.Thread(target=swap)
        racer.start()
        time.sleep(0.001)
        try:
            vo.put_back_and_record(trial, verifier._sandbox, sandbox_gone=True)
        finally:
            stop.set()
            racer.join()
        assert (host / "victim").read_text() == "host data", f"attempt {attempt}"
        assert host.stat().st_mode == host_mode, f"attempt {attempt}"
        for name in ("a", "s"):
            with contextlib.suppress(FileNotFoundError):
                if (planted / name).is_dir() and not (planted / name).is_symlink():
                    os.chmod(planted / name, 0o755)


# Swaps two names (renameat2 RENAME_EXCHANGE) until killed; prints "go" after
# the first swap. Its own process, so it swaps while the pass runs: a thread
# waits on the GIL and swapped in about 1 of 60 attempts at the write.
_SWAPPER = """
import sys
sys.path.insert(0, sys.argv[1])
from tests.test_verifier_put_back_names import _exchange
exchange = _exchange()
a, s = sys.argv[2].encode(), sys.argv[3].encode()
exchange(a, s)
print("go", flush=True)
while True:
    exchange(a, s)
"""


def test_a_folder_swapped_for_a_link_during_the_first_pass_reaches_no_host_file(
    tmp_path,
):
    """The first put_back (the sandbox alive, nothing
    removed) writes test.sh's files back under verifier/. A folder test.sh
    left (logs/), swapped over and over with a link to a host folder while that
    pass restores logs/a.txt, leaves every host file and mode untouched.
    Mutant: that restore written by plain path."""
    if _exchange() is None:
        pytest.skip("renameat2 by syscall is Linux x86_64/aarch64 only")
    root = str(Path(__file__).resolve().parents[1])
    host = tmp_path / "host-folder"
    host.mkdir()
    host_mode = host.stat().st_mode
    for attempt in range(60):
        trial, paths = _trial(tmp_path / f"t{attempt}")
        verifier = _verifier(paths)
        logs = trial / "verifier" / "logs"
        (logs / "a.txt").unlink()  # removed after the exit: restored
        os.symlink(str(host), trial / "verifier" / "swap")
        swap = str(trial / "verifier" / "swap")
        racer = subprocess.Popen(
            [sys.executable, "-c", _SWAPPER, root, str(logs), swap],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert racer.stdout is not None and racer.stdout.readline() == "go\n"
            vo.put_back_and_record(trial, verifier._sandbox)  # the first pass
        finally:
            racer.kill()
            racer.wait()
        assert sorted(os.listdir(host)) == [], f"attempt {attempt}: {os.listdir(host)}"
        assert host.stat().st_mode == host_mode, f"attempt {attempt}"
