"""The steps from the agent's end to the verifier's start, made faster (PERF-G)
without changing what any of them decides or the order a check needs.

- The K1 kill's walk of ``/proc`` stops reading a status file once it has the
  fields it tests (State, the UIDs, and Threads for a zombie). It lists the
  same processes as the full read on every kernel layout below.
- Independent steps run side by side where the sandbox allows it
  (``sandbox/_in_order.py``), and one after another everywhere else; either
  way the first failure in order is the one raised.
- ``disconnect()`` runs the agent's pkill once per connection when K1's kill
  stopped every process of the sandbox user first.
- Hardening keeps its order: the kill, then the reads, then the first change.

Everything runs against stand-ins: a fake ``/proc`` folder, fake sandboxes
that record calls. Nothing here signals, kills or removes anything real.
"""

from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import lockdown

UID = "4242"

# The block the walk gained (PERF-G). The old walk is the same script without it.
_EARLY_STOP = """                # Stop once the fields the test below reads are in (Threads
                # only for a zombie): the shell reads the file a byte at a
                # time, and the ~45 lines after Uid made each walk cost
                # ~15 ms in a trial's container (PERF-G).
                if [ -n "$state" ] && [ -n "$euid" ] &&
                    { [ "$state" != Z ] || [ -n "$threads" ]; }; then
                    break
                fi
"""


def _kernel_status(state: str, uids: str, threads: str = "1") -> str:
    """A status file in the kernel's own layout (Linux 6.x, field order)."""
    return (
        "Name:\tsleep\nUmask:\t0022\n"
        f"State:\t{state}\n"
        "Tgid:\t77\nNgid:\t0\nPid:\t77\nPPid:\t1\nTracerPid:\t0\n"
        f"Uid:\t{uids}\nGid:\t{uids}\n"
        "FDSize:\t64\nGroups:\t \nNStgid:\t77\nNSpid:\t77\nNSpgid:\t77\nNSsid:\t1\n"
        "Kthread:\t0\nVmPeak:\t    2384 kB\nVmSize:\t    2384 kB\nVmLck:\t       0 kB\n"
        "VmPin:\t       0 kB\nVmHWM:\t     896 kB\nVmRSS:\t     896 kB\n"
        "RssAnon:\t      88 kB\nRssFile:\t     808 kB\nRssShmem:\t       0 kB\n"
        "VmData:\t     224 kB\nVmStk:\t     132 kB\nVmExe:\t      16 kB\n"
        "VmLib:\t    1532 kB\nVmPTE:\t      44 kB\nVmSwap:\t       0 kB\n"
        "HugetlbPages:\t       0 kB\nCoreDumping:\t0\nTHP_enabled:\t1\n"
        f"Threads:\t{threads}\n"
        "SigQ:\t0/31529\nSigPnd:\t0000000000000000\nShdPnd:\t0000000000000000\n"
        "SigBlk:\t0000000000000000\nSigIgn:\t0000000000000000\n"
        "SigCgt:\t0000000000000000\nCapInh:\t0000000000000000\n"
        "CapPrm:\t0000000000000000\nCapEff:\t0000000000000000\n"
        "CapBnd:\t000001ffffffffff\nCapAmb:\t0000000000000000\nNoNewPrivs:\t1\n"
        "Seccomp:\t2\nSeccomp_filters:\t1\nSpeculation_Store_Bypass:\tthread vulnerable\n"
        "Cpus_allowed:\tffff\nCpus_allowed_list:\t0-15\nMems_allowed:\t1\n"
        "Mems_allowed_list:\t0\nvoluntary_ctxt_switches:\t2\n"
        "nonvoluntary_ctxt_switches:\t0\n"
    )


def _zombie_layout(threads: str) -> str:
    """A zombie's status: no Vm lines, Threads still there."""
    full = _kernel_status("Z (zombie)", f"{UID}\t{UID}\t{UID}\t{UID}", threads)
    return "".join(
        line + "\n" for line in full.splitlines() if not line.startswith(("Vm", "Rss"))
    )


# Every shape the walk tells apart, in the kernel's layout and in the short
# three-line one the kill's own tests use.
_STATUSES = {
    "live": _kernel_status("S (sleeping)", f"{UID}\t{UID}\t{UID}\t{UID}"),
    "running": _kernel_status("R (running)", f"{UID}\t{UID}\t{UID}\t{UID}"),
    "stopped": _kernel_status("T (stopped)", f"{UID}\t{UID}\t{UID}\t{UID}"),
    "zombie": _zombie_layout("1"),
    "zombie leader with live threads": _zombie_layout("3"),
    "root": _kernel_status("S (sleeping)", "0\t0\t0\t0"),
    "another user": _kernel_status("S (sleeping)", "5678\t5678\t5678\t5678"),
    "setuid the agent started": _kernel_status("S (sleeping)", f"{UID}\t0\t0\t0"),
    "saved uid only": _kernel_status("S (sleeping)", f"0\t0\t{UID}\t0"),
    "effective uid only": _kernel_status("S (sleeping)", f"0\t{UID}\t0\t0"),
    "short live": f"State:\tS (sleeping)\nUid:\t{UID}\t{UID}\t{UID}\t{UID}\nThreads:\t1\n",
    "short zombie leader": f"State:\tZ (zombie)\nUid:\t{UID}\t{UID}\t{UID}\t{UID}\nThreads:\t3\n",
    "uid before state": f"Uid:\t{UID}\t{UID}\t{UID}\t{UID}\nState:\tS (sleeping)\n",
    "threads before state": f"Threads:\t3\nUid:\t{UID}\t{UID}\t{UID}\t{UID}\nState:\tZ (zombie)\n",
    "no state": f"Name:\tx\nUid:\t{UID}\t{UID}\t{UID}\t{UID}\n",
    "no uid": "Name:\tx\nState:\tS (sleeping)\n",
    "empty": "",
    "no trailing newline": f"State:\tS (sleeping)\nUid:\t{UID}\t{UID}\t{UID}\t{UID}",
}


def _walk_script(proc: str, *, early_stop: bool) -> str:
    """The kill's prelude with fixed numbers (as ``kill_prelude`` gives
    it), then one walk, its output and its status. *early_stop* False is
    the same prelude without the early stop: the full read."""
    prelude = lockdown._KILL_PRELUDE
    if not early_stop:
        assert _EARLY_STOP in prelude
        prelude = prelude.replace(_EARLY_STOP, "")
    start = prelude.index("uid=$(id -u ")
    end = prelude.index("me=\n", start)
    prelude = (
        prelude[:start]
        + f'uid={UID}\ngid={UID}\n[ -r "$P/self/status" ] || exit 2\n'
        + prelude[end:]
    )
    return (
        prelude.replace("__PROC__", proc).replace("__NEED_ROOT__", "0")
        + 'out=$(live); rc=$?; printf "%s\\n" "$out" | sort; echo "rc=$rc"\n'
    )


def _fake_proc(tmp_path, statuses: dict[str, str | None]):
    root = tmp_path / "fakeproc"
    (root / "self").mkdir(parents=True)
    # Not root, so the prelude probes no helper (nothing to switch to here).
    (root / "self/status").write_text(
        "State:\tR (running)\nUid:\t5678\t5678\t5678\t5678\n"
    )
    for pid, (_name, status) in enumerate(statuses.items(), start=100):
        entry = root / str(pid)
        entry.mkdir()
        if status is None:
            (entry / "status").mkdir()  # an entry whose status cannot be read
        else:
            (entry / "status").write_text(status)
    return root


def _walk(root, *, early_stop: bool) -> str:
    result = subprocess.run(
        ["/bin/sh", "-c", _walk_script(str(root), early_stop=early_stop)],
        capture_output=True,
        text=True,
        timeout=30,
        env={"PATH": "/usr/bin:/bin"},
    )
    return result.stdout + result.stderr


@pytest.mark.parametrize("name", sorted(_STATUSES))
def test_the_walk_lists_each_process_as_the_full_read_did(tmp_path, name):
    root = _fake_proc(tmp_path, {name: _STATUSES[name]})
    assert _walk(root, early_stop=True) == _walk(root, early_stop=False)


def test_the_walk_lists_a_whole_table_as_the_full_read_did(tmp_path):
    # Without the entries that cannot be inspected (no State or no Uid),
    # which end the walk (the next test).
    readable = {
        name: status
        for name, status in _STATUSES.items()
        if name not in ("no state", "no uid", "empty")
    }
    root = _fake_proc(tmp_path, readable)
    new = _walk(root, early_stop=True)
    assert new == _walk(root, early_stop=False)
    # The user's live ones: live, running, stopped, the zombie leader (three
    # layouts), the setuid and saved-UID ones, the effective-UID one tagged r,
    # the short live one, Uid before State, and no trailing newline.
    listed = [line for line in new.splitlines() if line and not line.startswith("rc=")]
    assert len(listed) == 12, new
    assert sum(line.startswith("r") for line in listed) == 1
    assert new.rstrip().endswith("rc=0")


def test_an_unreadable_entry_still_fails_the_walk(tmp_path):
    root = _fake_proc(tmp_path, {"live": _STATUSES["live"], "unreadable": None})
    new = _walk(root, early_stop=True)
    assert new == _walk(root, early_stop=False)
    assert "Cannot inspect process 101" in new and "rc=2" in new.splitlines()


def test_the_walk_stops_reading_once_it_has_its_fields(tmp_path):
    """Fails before PERF-G: the walk read every status file to its end.

    The probe is a file the kernel never writes, with a second State line far
    below Uid. The full read takes the last one (a zombie of one thread, not
    listed); the walk now stops at Uid with the first (sleeping, listed).
    On a real kernel no field repeats, so both read the same values (the
    tests above)."""
    probe = _kernel_status("S (sleeping)", f"{UID}\t{UID}\t{UID}\t{UID}") + (
        "State:\tZ (zombie)\n"
    )
    root = _fake_proc(tmp_path, {"probe": probe})
    assert _walk(root, early_stop=True).splitlines()[0] == "100"


def test_the_kill_script_still_has_the_early_stop_in_its_one_walk():
    script = lockdown._KILL_SANDBOX_USER_PROCS_CMD_TEMPLATE
    assert script.count(_EARLY_STOP) == 1
    assert script.count('done <"$entry/status"') == 1


# --- in_order -----------------------------------------------------------------


class _Sequential:
    pass


class _SideBySide:
    concurrent_steps = True


def _recorder():
    events: list[str] = []

    def step(name, delay, *, fail=None, value=None):
        async def run():
            events.append(f"start {name}")
            await asyncio.sleep(delay)
            events.append(f"end {name}")
            if fail is not None:
                raise fail
            return value if value is not None else name

        return run

    return events, step


@pytest.mark.asyncio
async def test_in_order_runs_one_after_another_by_default():
    from benchflow.sandbox._in_order import in_order, runs_steps_at_once

    events, step = _recorder()
    assert not runs_steps_at_once(_Sequential())
    assert await in_order(_Sequential(), step("a", 0.01), step("b", 0)) == ["a", "b"]
    assert events == ["start a", "end a", "start b", "end b"]


@pytest.mark.asyncio
async def test_in_order_stops_at_the_first_failure_one_after_another():
    from benchflow.sandbox._in_order import in_order

    events, step = _recorder()
    with pytest.raises(ValueError, match="a broke"):
        await in_order(
            _Sequential(), step("a", 0, fail=ValueError("a broke")), step("b", 0)
        )
    # The later step was never made, as before.
    assert events == ["start a", "end a"]


@pytest.mark.asyncio
async def test_in_order_runs_side_by_side_where_the_sandbox_allows_it():
    from benchflow.sandbox._in_order import in_order, runs_steps_at_once

    events, step = _recorder()
    assert runs_steps_at_once(_SideBySide())
    assert await in_order(_SideBySide(), step("a", 0.05), step("b", 0)) == ["a", "b"]
    assert events[:2] == ["start a", "start b"]


@pytest.mark.asyncio
async def test_in_order_raises_the_first_failure_in_order_not_in_time():
    from benchflow.sandbox._in_order import in_order

    events, step = _recorder()
    with pytest.raises(ValueError, match="a broke"):
        await in_order(
            _SideBySide(),
            step("a", 0.05, fail=ValueError("a broke")),
            step("b", 0, fail=RuntimeError("b broke first")),
        )
    # Every step ran to its end before the failure was raised.
    assert sorted(events) == ["end a", "end b", "start a", "start b"]


@pytest.mark.asyncio
async def test_in_order_ignores_a_flag_set_on_an_instance_only():
    from benchflow.sandbox._in_order import runs_steps_at_once

    sandbox = _Sequential()
    sandbox.concurrent_steps = True  # a mock's attribute is not the backend's
    assert not runs_steps_at_once(sandbox)


def test_only_docker_runs_steps_side_by_side():
    import inspect

    from benchflow.sandbox._base import BaseSandbox
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    # tests/conftest.py turns it off for the scripted Docker tests, so the
    # class's own line is what is checked.
    assert "concurrent_steps: ClassVar[bool] = True" in inspect.getsource(DockerSandbox)
    assert issubclass(RemoteDockerSandbox, DockerSandbox)
    assert getattr(BaseSandbox, "concurrent_steps", False) is False
    for name in ("daytona", "modal_impl", "agentcore", "apple_container"):
        try:
            module = __import__(f"benchflow.sandbox.{name}", fromlist=["_"])
        except ImportError:
            continue
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and issubclass(value, BaseSandbox)
                and not issubclass(value, DockerSandbox)
            ):
                assert getattr(value, "concurrent_steps", False) is not True, value


# --- the agent's pkill, once per quiesced connection ---------------------------


def _rollout(sandbox_user, *, keep=None, monkeypatch):
    from benchflow.rollout import Rollout

    monkeypatch.setattr(
        "benchflow.task.keep_services.trial_keep", lambda *a: keep, raising=False
    )
    rollout = Rollout.__new__(Rollout)
    calls: list[str] = []

    async def exec_(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    quiesce = AsyncMock()
    rollout.__dict__.update(
        _is_session_factory=False,
        _capture_partial_acp_trajectory=lambda: None,
        _collect_native_acp_usage=lambda: None,
        _acp_client=None,
        _config=SimpleNamespace(sandbox_user=sandbox_user),
        _connection_needs_quiescence=True,
        _muse_credentials_staged=False,
        _task=SimpleNamespace(config=SimpleNamespace()),
        _planes=SimpleNamespace(quiesce_agent=quiesce),
        _agent_launch="/opt/runtime/bin/claude-agent-acp",
        _env=SimpleNamespace(exec=exec_),
        _credential_ledger=[],
        _phase="running",
        _agent_quiesced=False,
        _agent_pkill_done=False,
    )
    return rollout, calls, quiesce


def _pkills(calls):
    return [c for c in calls if c.startswith("pkill -f")]


@pytest.mark.asyncio
async def test_a_second_disconnect_after_the_quiesce_runs_no_second_pkill(monkeypatch):
    """Fails before PERF-G: the verifier's capture disconnects again and ran
    the agent's pkill a second time, after K1's kill had stopped every
    process of the sandbox user and nothing had been launched since."""
    rollout, calls, quiesce = _rollout("agent", monkeypatch=monkeypatch)
    await rollout.disconnect()
    await rollout.disconnect()
    assert quiesce.await_count == 1
    assert len(_pkills(calls)) == 1


@pytest.mark.asyncio
async def test_a_new_connection_gets_its_own_pkill(monkeypatch):
    rollout, calls, quiesce = _rollout("agent", monkeypatch=monkeypatch)
    await rollout.disconnect()
    # What connect() and connect_as() mark for the next connection.
    rollout._connection_needs_quiescence = True
    rollout._agent_pkill_done = rollout._agent_quiesced = False
    await rollout.disconnect()
    assert quiesce.await_count == 2
    assert len(_pkills(calls)) == 2


@pytest.mark.asyncio
async def test_a_root_agent_keeps_every_pkill(monkeypatch):
    rollout, calls, quiesce = _rollout(None, monkeypatch=monkeypatch)
    await rollout.disconnect()
    await rollout.disconnect()
    assert quiesce.await_count == 0
    assert len(_pkills(calls)) == 2


@pytest.mark.asyncio
async def test_kept_services_keep_every_pkill(monkeypatch):
    """No UID-wide kill runs at a kept-service disconnect, so a process that
    matches the agent may have started since: the pkill stays."""
    keep = SimpleNamespace(ports=(5000,))
    rollout, calls, quiesce = _rollout("agent", keep=keep, monkeypatch=monkeypatch)
    await rollout.disconnect()
    await rollout.disconnect()
    assert quiesce.await_count == 0
    assert len(_pkills(calls)) == 2


@pytest.mark.asyncio
async def test_connect_clears_the_pkill_mark():
    """connect() and connect_as() both clear the marks before anything else
    can launch the agent (checked in the source: they set them beside the
    quiescence flag)."""
    import inspect

    from benchflow.rollout import Rollout

    for method in (Rollout.connect, Rollout.connect_as):
        source = inspect.getsource(method)
        flag = source.index("self._connection_needs_quiescence = True")
        assert (
            source.index("self._agent_pkill_done = self._agent_quiesced = False") > flag
        )


# --- hardening keeps its order ---------------------------------------------------


class _HardenSandbox(_SideBySide):
    verifier_prep_script = True


def _harden_fakes(monkeypatch, events, *, bash_fail=None, guard_fail=None):
    def timed(name, delay=0.0, value=None, fail=None):
        async def run(*args, **kwargs):
            events.append(f"start {name}")
            await asyncio.sleep(delay)
            events.append(f"end {name}")
            if fail is not None:
                raise fail
            return value

        return run

    monkeypatch.setattr(lockdown, "_kill_sandbox_user_procs", timed("kill", 0.02, 0))
    monkeypatch.setattr(lockdown, "kill_switched_accounts", timed("switched", 0, 0))
    monkeypatch.setattr(
        "benchflow.agents.credentials.muse_login_present", timed("muse", 0, False)
    )
    monkeypatch.setattr(
        lockdown,
        "_trusted_verifier_bash_env",
        timed("bash_env", 0.03, "/etc/bash.bashrc", bash_fail),
    )

    async def blocked(env, prefixes):
        name = "aliases" if prefixes == ("/app",) else "guard"
        events.append(f"start {name}")
        await asyncio.sleep(0.01)
        events.append(f"end {name}")
        if name == "guard" and guard_fail is not None:
            raise guard_fail
        return (*prefixes, f"/alias-of-{name}")

    monkeypatch.setattr(lockdown, "_blocked_before_freeze", blocked)
    # On integ since batch 32b the guard's read is main's path exclusions
    # (V3); the judged interpreter comes after the reads.
    monkeypatch.setattr(lockdown, "_path_exclusions_before_freeze", blocked)

    async def judged(env, blocked, sandbox_user):
        events.append("root_python")
        return lockdown.RootPython("/usr/bin/python3")

    monkeypatch.setattr(lockdown, "_judge_root_python", judged)
    seen: dict[str, object] = {}

    async def prep(env, steps, seconds):
        events.append("prep")
        seen["steps"] = steps

    monkeypatch.setattr(lockdown, "_run_prep_steps", prep)

    def cleanup_step(hardening, workspace, aliases):
        seen["aliases"] = aliases
        return ("cleanup", "true")

    monkeypatch.setattr(lockdown, "_cleanup_step", cleanup_step)

    async def build(env, task, sandbox_user, workspace, **kwargs):
        events.append("build")
        seen["build"] = kwargs
        return {"PATH": "/usr/bin"}

    monkeypatch.setattr(lockdown, "_build_verifier_env", build)
    monkeypatch.setattr(
        "benchflow.task.keep_services.trial_keep", lambda *a: None, raising=False
    )
    return seen


def _task(tmp_path):
    return SimpleNamespace(
        task_dir=tmp_path,
        config=SimpleNamespace(verifier=SimpleNamespace(env={}, service="main")),
    )


@pytest.mark.asyncio
async def test_hardening_reads_come_after_the_kill_and_before_the_first_change(
    tmp_path, monkeypatch
):
    events: list[str] = []
    seen = _harden_fakes(monkeypatch, events)
    task = _task(tmp_path)
    await lockdown._harden_steps(
        _HardenSandbox(), task, "agent", "/app", False, None, {}
    )
    reads = [e for e in events if e.split()[-1] in ("bash_env", "guard", "aliases")]
    # The three reads ran side by side, the exclusions first: they write
    # their private inode index (merge wave batch 35) ...
    assert [e for e in events if e.startswith("start")][-3:] == [
        "start guard",
        "start bash_env",
        "start aliases",
    ]
    # ... after the kill ended, and all ended before the prep's first change.
    assert events.index("end kill") < events.index(reads[0])
    assert max(events.index(e) for e in reads) < events.index("prep")
    assert events[-1] == "build"
    assert max(events.index(e) for e in reads) < events.index("root_python")
    # Each result reached the step that uses it, as before.
    assert seen["aliases"] == ("/app", "/alias-of-aliases")
    assert seen["build"]["bash_env"] == "/etc/bash.bashrc"
    assert seen["build"]["blocked"][-1] == "/alias-of-guard"
    assert task.config.verifier.env == {"PATH": "/usr/bin"}


@pytest.mark.asyncio
async def test_a_failed_read_fails_hardening_as_before(tmp_path, monkeypatch):
    """The path exclusions come first in order (they write their inode index,
    merge wave batch 35): their failure is the one raised, even when the
    BASH_ENV read failed too, and nothing is changed."""
    events: list[str] = []
    _harden_fakes(
        monkeypatch,
        events,
        bash_fail=RuntimeError("Verifier hardening failed: bash env"),
        guard_fail=RuntimeError(
            "Verifier hardening failed: reading workspace mount aliases"
        ),
    )
    with pytest.raises(RuntimeError, match="mount aliases"):
        await lockdown._harden_steps(
            _HardenSandbox(), _task(tmp_path), "agent", "/app", False, None, {}
        )
    assert "prep" not in events and "build" not in events


@pytest.mark.asyncio
async def test_a_failed_bash_env_read_alone_fails_hardening_as_before(
    tmp_path, monkeypatch
):
    events: list[str] = []
    _harden_fakes(
        monkeypatch,
        events,
        bash_fail=RuntimeError("Verifier hardening failed: bash env"),
    )
    with pytest.raises(RuntimeError, match="bash env"):
        await lockdown._harden_steps(
            _HardenSandbox(), _task(tmp_path), "agent", "/app", False, None, {}
        )
    assert "prep" not in events and "build" not in events


@pytest.mark.asyncio
async def test_a_failed_mount_read_fails_hardening_as_before(tmp_path, monkeypatch):
    events: list[str] = []
    _harden_fakes(
        monkeypatch,
        events,
        guard_fail=RuntimeError(
            "Verifier hardening failed: reading workspace mount aliases"
        ),
    )
    with pytest.raises(RuntimeError, match="mount aliases"):
        await lockdown._harden_steps(
            _HardenSandbox(), _task(tmp_path), "agent", "/app", False, None, {}
        )
    assert "prep" not in events and "build" not in events


@pytest.mark.asyncio
async def test_a_failed_kill_runs_no_read(tmp_path, monkeypatch):
    events: list[str] = []
    _harden_fakes(monkeypatch, events)

    async def kill(*args, **kwargs):
        events.append("kill")
        raise RuntimeError("Sandbox-user writers remain: 7 (sleep)")

    monkeypatch.setattr(lockdown, "_kill_sandbox_user_procs", kill)
    with pytest.raises(RuntimeError, match="writers remain"):
        await lockdown._harden_steps(
            _HardenSandbox(), _task(tmp_path), "agent", "/app", False, None, {}
        )
    assert events == ["kill"]


@pytest.mark.asyncio
async def test_sequential_and_side_by_side_hardening_give_the_same_result(
    tmp_path, monkeypatch
):
    results = []
    for sandbox in (_Sequential(), _HardenSandbox()):
        if isinstance(sandbox, _Sequential):
            sandbox.verifier_prep_script = True
        events: list[str] = []
        seen = _harden_fakes(monkeypatch, events)
        task = _task(tmp_path)
        await lockdown._harden_steps(sandbox, task, "agent", "/app", False, None, {})
        build = {k: v for k, v in seen["build"].items() if k != "image_pythonpath"}
        results.append((seen["aliases"], build, task.config.verifier.env))
    assert results[0] == results[1]


# --- publication: the two files go up side by side -------------------------------


class _SideBySideRecording:
    """RecordingEnv's shape on a sandbox that runs steps side by side."""

    concurrent_steps = True

    def __init__(self, fail_upload: str | None = None) -> None:
        self.execs: list[tuple[str, str | None]] = []
        self.events: list[str] = []
        self.fail_upload = fail_upload

    async def exec(self, command, user=None, timeout_sec=None):
        self.execs.append((command, user))
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload_file(self, source, target, **_kwargs):
        name = target.rsplit("/", 1)[-1]
        self.events.append(f"start {name}")
        await asyncio.sleep(0.02)
        self.events.append(f"end {name}")
        if name == self.fail_upload:
            raise OSError("docker cp failed")


@pytest.mark.asyncio
async def test_the_published_files_go_up_side_by_side(tmp_path):
    from benchflow.rollout._setup import _publish_trajectory_for_verifier
    from tests.reachable_fixture import directory

    env = _SideBySideRecording()
    await _publish_trajectory_for_verifier(
        env, [{"type": "agent_message", "text": "done"}], directory(tmp_path / "agent")
    )
    assert env.events[:2] == ["start acp_trajectory.jsonl", "start output.txt"]
    # The copy into place comes after both, and the stage is removed last.
    commands = [c for c, _ in env.execs]
    assert commands[-2].startswith("rc=0;") and commands[-1].startswith("rm -rf ")


@pytest.mark.asyncio
async def test_a_failed_upload_still_fails_publication(tmp_path):
    from benchflow.rollout._setup import _publish_trajectory_for_verifier
    from tests.reachable_fixture import directory

    env = _SideBySideRecording(fail_upload="output.txt")
    with pytest.raises(RuntimeError, match="publication upload failed"):
        await _publish_trajectory_for_verifier(
            env,
            [{"type": "agent_message", "text": "done"}],
            directory(tmp_path / "agent"),
        )
    commands = [c for c, _ in env.execs]
    # Nothing was copied into place, and the stage was removed.
    assert not any(c.startswith("rc=0;") for c in commands)
    assert commands[-1].startswith("rm -rf /logs/.runtime-publish-")


# --- publication beside the artifact collection ----------------------------------


def _verify_stand_in(*, artifacts=(), mounted=True, side_by_side=True):
    from benchflow.rollout import Rollout

    cls = _SideBySide if side_by_side else _Sequential
    env = cls()
    env.is_mounted = mounted
    rollout = Rollout.__new__(Rollout)
    rollout.__dict__.update(
        _env=env,
        _task=SimpleNamespace(config=SimpleNamespace(artifacts=list(artifacts))),
        _branch_child_active=False,
    )
    return rollout


def test_publication_overlaps_the_collection_only_when_they_share_nothing():
    from benchflow.rollout import Rollout

    assert Rollout._publish_beside_collection(_verify_stand_in())
    # A declared artifact may name /logs/agent (or a link to it): one after
    # the other, as before.
    assert not Rollout._publish_beside_collection(
        _verify_stand_in(artifacts=["/logs/agent"])
    )
    assert not Rollout._publish_beside_collection(_verify_stand_in(artifacts=["out"]))
    # Unmounted /logs (the collection copies out of the sandbox), and
    # sandboxes that run one step at a time.
    assert not Rollout._publish_beside_collection(_verify_stand_in(mounted=False))
    assert not Rollout._publish_beside_collection(_verify_stand_in(side_by_side=False))


@pytest.mark.asyncio
@pytest.mark.parametrize("beside", [False, True])
async def test_a_failed_publication_beside_the_collection_fails_as_before(
    tmp_path, monkeypatch, beside
):
    """The publication's failure is handled where it was awaited before: a
    verifier error, and the grader never runs, in both orders.
    Beside the collection, the trajectory is scraped first and the collection
    still runs to its end."""
    from pathlib import Path
    from unittest.mock import MagicMock

    import benchflow.rollout as mod
    from benchflow.rollout import Rollout, _setup

    events: list[str] = []

    async def capture(_r):
        events.append("capture")

    async def collect(_r):
        events.append("collect start")
        await asyncio.sleep(0.02)
        events.append("collect end")

    async def publish(*a, **k):
        events.append("publish")
        raise _setup.VerifierPublicationError("verifier evidence publication failed")

    async def scrape(_r):
        events.append("scrape")

    monkeypatch.setattr(mod, "capture_terminal_workspace", capture)
    monkeypatch.setattr(mod, "collect_rollout_artifacts", collect)
    monkeypatch.setattr(mod, "_publish_trajectory_for_verifier", publish)
    monkeypatch.setattr(Rollout, "_scrape_trajectory_if_missing", scrape)
    monkeypatch.setattr(Rollout, "_publish_beside_collection", lambda _r: beside)
    monkeypatch.setattr(mod, "require_safe_branch_world", lambda _r: None)
    grader = AsyncMock()
    monkeypatch.setattr(mod, "_verify_rollout", grader)
    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.recovery_ineligible_reason",
        lambda _r: "not eligible",
    )
    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.mark_solver_complete", lambda _r: None
    )
    rollout = Rollout.__new__(Rollout)
    rollout._config = SimpleNamespace(
        primary_agent="opencode", sandbox_user="agent", task_path=Path("/tmp/task")
    )
    rollout._task = SimpleNamespace(config=None)
    rollout._trajectory = [{"type": "agent_message", "text": "done"}]
    rollout._env = object()
    rollout._rollout_paths = SimpleNamespace(agent_dir=tmp_path / "agent")
    rollout._timing = {}
    rollout._planes = object()
    rollout._agent_cwd = "/app"
    rollout._diagnostics = MagicMock()
    assert await rollout.verify() is None
    assert "publication failed" in rollout._verifier_error
    grader.assert_not_awaited()
    if beside:
        assert events == [
            "capture",
            "scrape",
            "collect start",
            "publish",
            "collect end",
        ]
    else:
        assert events == [
            "capture",
            "collect start",
            "collect end",
            "scrape",
            "publish",
        ]


# --- the image's PYTHONPATH, read beside BASH_ENV ----------------------------------


@pytest.mark.asyncio
async def test_an_early_pythonpath_read_is_used_and_its_check_runs_later(monkeypatch):
    calls: list[str] = []

    async def read(env, command, *args, **kwargs):
        calls.append(command)
        raise AssertionError("the image's PYTHONPATH was read again")

    async def judge(env, raw, blocked, agent):
        # The ownership check (V3's judge, on integ since batch 32b): both
        # entries trusted.
        calls.append(f"judge {raw}")
        return SimpleNamespace(kept=("/opt/lib", "/srv/py"))

    monkeypatch.setattr(lockdown, "_checked_read", read)
    monkeypatch.setattr(lockdown, "_judge_trusted_entries", judge)
    # The value as the byte-encoded read gives it (V3); plain paths encode
    # as themselves.
    early = lockdown._EarlyRead("/opt/lib:/srv/py\n")
    got = await lockdown._trusted_verifier_pythonpath(
        object(), "agent", image_pythonpath=early
    )
    assert got == "/opt/lib:/srv/py"
    # The value was not read again; only the check ran.
    assert calls == ["judge /opt/lib:/srv/py"]


@pytest.mark.asyncio
async def test_an_early_pythonpath_failure_is_raised_where_the_read_was():
    early = lockdown._EarlyRead(
        error=RuntimeError("Command timed out after 60 seconds")
    )
    with pytest.raises(RuntimeError, match="timed out"):
        await lockdown._trusted_verifier_pythonpath(
            object(), "agent", image_pythonpath=early
        )


@pytest.mark.asyncio
async def test_the_early_read_never_raises(monkeypatch):
    async def broken(env, command, *args, **kwargs):
        raise RuntimeError("channel lost")

    monkeypatch.setattr(lockdown, "_checked_read", broken)
    early = await lockdown._early_image_pythonpath(object())
    with pytest.raises(RuntimeError, match="channel lost"):
        early.get()


@pytest.mark.asyncio
async def test_hardening_reads_pythonpath_early_only_side_by_side(
    tmp_path, monkeypatch
):
    for sandbox, expect in ((_HardenSandbox(), True), (_Sequential(), False)):
        if isinstance(sandbox, _Sequential):
            sandbox.verifier_prep_script = True
        events: list[str] = []
        seen = _harden_fakes(monkeypatch, events)
        await lockdown._harden_steps(
            sandbox, _task(tmp_path), "agent", "/app", False, None, {}
        )
        early = seen["build"]["image_pythonpath"]
        assert isinstance(early, lockdown._EarlyRead) is expect
