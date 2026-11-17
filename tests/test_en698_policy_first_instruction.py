"""EN-698's pieces without a daemon: the start holds a run asks for, the
uid rules' reset, a model agent's arm-time check, and the start hold's
hand-over. The real-Docker trials are in
tests/test_en698_policy_first_instruction_docker.py."""

from __future__ import annotations

import random
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox import verifier_network as vn


def _task_config(**sandbox):
    from benchflow.task.config import TaskConfig

    raw = {"schema_version": "1.3", "sandbox": sandbox}
    return TaskConfig.model_validate(raw)


# --- the uid rules ------------------------------------------------------------


def test_the_uid_rules_reset_the_uids_tcp_before_refusing_the_rest():
    """N4: a TCP packet of the uid is answered with a reset, ahead of the
    plain REJECT, IPv4 and IPv6, so a connection opened before the rules is
    cut at its next packet instead of stalled with its data queued."""
    cmd = lockdown._agent_egress_firewall_cmd("agent", refuse_dns=True, uid=1500)
    owner = '-m owner --uid-owner "$agent_uid"'
    for tool in ("iptables", "ip6tables"):
        reset = f"{tool} -A OUTPUT -p tcp {owner} -j REJECT --reject-with tcp-reset"
        plain = f"{tool} -A OUTPUT {owner} -j REJECT"
        assert reset in cmd, tool
        assert cmd.index(reset) < cmd.index(plain), tool


# --- the start holds a run asks for -------------------------------------------


@pytest.mark.parametrize(
    ("sandbox", "mode"),
    [
        ({"network_mode": "allowlist", "allowed_hosts": ["192.0.2.0/24"]}, "allowlist"),
        ({"network_mode": "denylist", "blocked_hosts": ["example.com"]}, "denylist"),
        ({"allow_internet": False}, "no-network"),
    ],
)
def test_a_restricted_policy_asks_for_every_hold(sandbox, mode):
    from benchflow.sandbox.service_cover import ask_start_holds

    box = SimpleNamespace()
    ask_start_holds(box, _task_config(**sandbox), None, every_process=True)
    assert box._hold_every_service is True
    assert box._start_cover["mode"] == mode
    assert box._start_hold_mode == mode
    if mode == "allowlist":
        assert box._start_hold_allow == ("192.0.2.0/24",)


def test_a_model_agent_asks_for_the_services_only():
    """R1: a model agent's main keeps its open start (its install and model
    proxy); its services are still held and covered before release."""
    from benchflow.sandbox.service_cover import ask_start_holds

    box = SimpleNamespace()
    ask_start_holds(
        box,
        _task_config(network_mode="denylist", blocked_hosts=["x.org"]),
        None,
        every_process=False,
    )
    assert box._hold_every_service is True
    assert not hasattr(box, "_start_hold_mode")


def test_an_open_network_asks_for_nothing():
    from benchflow.sandbox.service_cover import ask_start_holds

    box = SimpleNamespace()
    ask_start_holds(box, _task_config(), None, every_process=True)
    assert vars(box) == {}


# --- the start hold's hand-over -----------------------------------------------


class _Helper:
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def metadata_rules_install(self, command: str):
        self.commands.append(command)
        return SimpleNamespace(stdout="BF verifier-network removed status=0"), ""


@pytest.mark.asyncio
async def test_the_start_hold_gives_way_to_the_verifiers_rules():
    """Its own rules only (what it added) come out, under the hold."""
    env = _Helper()
    env._benchflow_start_hold = (["-d 10.0.0.1 -j REJECT"], [], [])
    assert vn._handover_pending(env)
    await vn.remove_start_hold(env)
    assert env._benchflow_start_hold is None
    assert "-D OUTPUT" in env.commands[0] and "10.0.0.1" in env.commands[0]


@pytest.mark.asyncio
async def test_a_start_hold_that_stays_stops_hardening():
    env = _Helper()

    async def install(command):
        return SimpleNamespace(stdout=""), ""

    env.metadata_rules_install = install
    env._benchflow_start_hold = (["-j REJECT"], [], [])
    with pytest.raises(RuntimeError, match="could not give way"):
        await vn.remove_start_hold(env)


# --- R1: main is quiet when a model agent starts ------------------------------

# Root's scan (lockdown._TASK_PROCS_SH): pid, ppid, uid, open connections,
# state, start time, then the program and the command line.
SCAN = (
    "BF proc 1 0 0 0 S 100\t/usr/bin/dash\tsh -c sleep infinity\n"
    "BF proc 7 1 0 1 S 120\t/usr/bin/bash\t/bin/bash /probe.sh main-entry-root hold\n"
    "BF proc 9 1 1500 0 S 121\t/usr/bin/bash\t/bin/bash /probe.sh quiet\n"
    "BF proc 20 0 0 0 S 200\t/.bfx/bfx\t/.bfx/bfx serve\n"
    "BF proc 21 20 0 2 S 201\t/usr/bin/python3\tpython3 -c x\n"
    "BF proc 30 1 0 3 S 300\t/usr/bin/python3.12\t/opt/runtime/proxy/abc/venv/bin/python litellm\n"
    "BF proc 31 30 0 1 S 301\t/usr/bin/python3.12\tpython3 worker\n"
    "BF proc 40 1 0 0 S 101\t/usr/bin/sleep\tsleep infinity\n"
    "BF proc 50 1 0 1 S 130\t/opt/benchflow/python/cpython-3.12/bin/python3.12\tpython\n"
    "BF proc 60 1 0 0 Z 140\t\t\n"
    "BF proc 70 1 1500 0 S 150\t/opt/runtime/bin/sleep\t/opt/runtime/bin/sleep 600\n"
    "BF task-procs status=0\n"
)
# The model proxy, as the host recorded it when BenchFlow started it.
PROXY = ((30, 300),)


def _pids(rows):
    return [r["pid"] for r in rows]


def test_task_processes_are_all_but_benchflows_own_known_from_the_host():
    """BenchFlow's own: what an engine exec started (ppid 0) and what runs
    under it, a daemon on the host's record (pid and start time) and what
    runs under it, the keep-alive. Everything else is the task's, whatever
    its uid, open connections or program path (50 and 70 run from
    BenchFlow's own folders); a zombie runs nothing."""
    assert _pids(lockdown._task_processes(SCAN, PROXY)) == [7, 9, 50, 70]


def test_a_daemon_not_on_the_hosts_record_is_a_task_process():
    """No record, or a record whose start time is another process's (the
    pid was reused): the process at that pid is the task's."""
    assert _pids(lockdown._task_processes(SCAN)) == [7, 9, 30, 31, 50, 70]
    assert _pids(lockdown._task_processes(SCAN, ((30, 299),))) == [
        7,
        9,
        30,
        31,
        50,
        70,
    ]


@pytest.mark.parametrize(
    ("rows", "task"),
    [
        # The keep-alive shell's own sleep, and a second one the ENTRYPOINT
        # left: one sleep is the keep-alive's, the other the task's.
        (
            "BF proc 1 0 0 0 S 1\t/usr/bin/dash\tsh -c sleep infinity\n"
            "BF proc 8 1 0 0 S 5\t/usr/bin/sleep\tsleep infinity\n"
            "BF proc 9 1 0 0 S 6\t/usr/bin/sleep\tsleep infinity\n",
            [9],
        ),
        # An init over the keep-alive.
        (
            "BF proc 1 0 0 0 S 1\t/usr/bin/tini\t/usr/bin/tini -- sh -c sleep infinity\n"
            "BF proc 6 1 0 0 S 2\t/usr/bin/dash\tsh -c sleep infinity\n"
            "BF proc 7 6 0 0 S 3\t/usr/bin/sleep\tsleep infinity\n",
            [],
        ),
        # An ENTRYPOINT that runs its command instead of exec'ing it: PID 1
        # is the task's, and so is all it runs.
        (
            "BF proc 1 0 0 0 S 1\t/usr/bin/bash\t/bin/bash /entry.sh sh -c sleep infinity\n"
            "BF proc 6 1 0 0 S 2\t/usr/bin/dash\tsh -c sleep infinity\n"
            "BF proc 7 6 0 0 S 3\t/usr/bin/sleep\tsleep infinity\n",
            [1, 6, 7],
        ),
        # Only the keep-alive: quiet.
        ("BF proc 1 0 0 0 S 1\t/usr/bin/sleep\tsleep infinity\n", []),
    ],
    ids=["second-sleep", "init", "entrypoint-pid-1", "sleep-pid-1"],
)
def test_the_keep_alive_is_pid_1_and_the_one_sleep_under_it(rows, task):
    assert _pids(lockdown._task_processes(rows + "BF task-procs status=0\n")) == task


def test_a_pids_first_row_stands():
    """A row for a pid already read (no command line can print one: the
    scan prints tabs and newlines in it as spaces) changes nothing."""
    scan = (
        "BF proc 1 0 0 0 S 1\t/usr/bin/sleep\tsleep infinity\n"
        "BF proc 7 1 0 0 S 9\t/usr/bin/sleep\tsleep 600\n"
        "BF proc 7 0 0 0 S 9\t/.bfx/bfx\t/.bfx/bfx serve\n"
    )
    assert _pids(lockdown._task_processes(scan)) == [7]


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the scan reads Linux /proc"
)
def test_the_scan_prints_a_command_line_on_its_own_row(tmp_path):
    """The real scan over this machine's /proc: a process whose command line
    carries a newline and a forged row reads as one row, its own."""
    forged = "x\nBF proc 1 0 0 0 S 1\t/.bfx/bfx\tforged"
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", forged],
        stdout=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.2)
        out = subprocess.run(
            ["sh", "-c", lockdown._TASK_PROCS_SH],
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
    finally:
        proc.kill()
        proc.wait()
    assert "BF task-procs status=0" in out
    rows = lockdown._scan_rows(out)
    assert rows[proc.pid]["argv"].endswith("forged")
    assert rows[proc.pid]["start"] > 0
    assert not any("forged" in r["argv"] for p, r in rows.items() if p != proc.pid)


class _Main:
    """A Docker sandbox at a model agent's launch, its scan answered from
    *scan* (and a start-time read from *stat*)."""

    def __init__(self, scan: str, entrypoint=("/entry.sh",), stat="30 300\n") -> None:
        self._bootstrap_configs = {"main": {"Entrypoint": list(entrypoint)}}
        self.scan = scan
        self.stat = stat
        self.commands: list[str] = []

    async def metadata_rules_install(self, command):  # pragma: no cover
        raise AssertionError

    async def exec(self, command, **_):
        self.commands.append(command)
        if command.startswith("kill"):
            return SimpleNamespace(stdout="", stderr="", return_code=0)
        if "/proc/$p/stat" in command:
            return SimpleNamespace(stdout=self.stat, stderr="", return_code=0)
        return SimpleNamespace(stdout=self.scan, stderr="", return_code=0)


@pytest.fixture
def plain_reads(monkeypatch):
    async def read(env, command, **kwargs):
        return await env.exec(command, **kwargs)

    monkeypatch.setattr(lockdown, "_read_exec", read)


@pytest.mark.asyncio
async def test_main_not_quiet_refuses_naming_each_process(plain_reads):
    """Every task process is named (pid, uid, command line), the agent's own
    account's quiet one too, and the refusal says what the task's author
    changes."""
    main = _Main(SCAN)
    main._benchflow_own_processes = list(PROXY)
    with pytest.raises(RuntimeError) as refused:
        await lockdown.refuse_task_processes(main)
    text = str(refused.value)
    assert text.startswith(
        "sandbox startup refused: main must be quiet when a model agent starts "
        "under the task's network policy"
    )
    assert "4 task processes are still running there" in text
    assert "started by main's image ENTRYPOINT (/entry.sh)" in text
    for named in (
        "pid 7 (uid 0): /bin/bash /probe.sh main-entry-root hold",
        "pid 9 (uid 1500): /bin/bash /probe.sh quiet",
        "pid 50 (uid 0): python",
        "pid 70 (uid 1500): /opt/runtime/bin/sleep 600",
    ):
        assert named in text, named
    assert "pid 30 " not in text and "pid 31 " not in text
    assert "drop the image's ENTRYPOINT, or make it exec its command" in text


@pytest.mark.asyncio
async def test_main_with_benchflows_own_processes_only_runs(plain_reads):
    quiet = (
        "BF proc 1 0 0 0 S 100\t/usr/bin/dash\tsh -c sleep infinity\n"
        "BF proc 20 0 0 0 S 200\t/.bfx/bfx\t/.bfx/bfx serve\n"
        "BF proc 30 1 0 3 S 300\t/usr/bin/python3.12\tlitellm\n"
        "BF proc 40 1 0 0 S 101\t/usr/bin/sleep\tsleep infinity\n"
        "BF task-procs status=0\n"
    )
    main = _Main(quiet)
    main._benchflow_own_processes = list(PROXY)
    await lockdown.refuse_task_processes(main)
    assert not any(c.startswith("kill") for c in main.commands)


@pytest.mark.asyncio
async def test_without_an_entrypoint_main_is_still_checked(plain_reads):
    """A process a task command left (no ENTRYPOINT) is refused too."""
    scan = (
        "BF proc 1 0 0 0 S 100\t/usr/bin/sleep\tsleep infinity\n"
        "BF proc 12 1 999 0 S 150\t/usr/lib/postgresql/bin/postgres\tpostgres -D /db\n"
        "BF task-procs status=0\n"
    )
    with pytest.raises(RuntimeError) as refused:
        await lockdown.refuse_task_processes(_Main(scan, entrypoint=()))
    text = str(refused.value)
    assert "1 task process is still running there, started by the task's own" in text
    assert "pid 12 (uid 999): postgres -D /db" in text


@pytest.mark.asyncio
async def test_every_process_held_or_not_docker_skips_the_check(monkeypatch):
    async def read(env, command, **kwargs):  # pragma: no cover
        raise AssertionError("no scan")

    monkeypatch.setattr(lockdown, "_read_exec", read)
    held = _Main(SCAN)
    held._benchflow_start_hold = ([], [], [])
    await lockdown.refuse_task_processes(held)

    class _Other:
        pass

    await lockdown.refuse_task_processes(_Other())


@pytest.mark.asyncio
async def test_a_scan_that_cannot_run_refuses(plain_reads):
    with pytest.raises(RuntimeError, match="could not be read"):
        await lockdown.refuse_task_processes(_Main("BF task-procs status=2\n"))


@pytest.mark.asyncio
async def test_a_refusal_stops_mains_task_processes_at_once(plain_reads):
    """The looping-connect check: a refused trial's task processes are
    killed at the refusal, not left running (and connecting) until the
    sandbox is torn down; one started meanwhile is found and killed too.
    PID 1, which nothing in main can kill, is left to the teardown."""
    late = SCAN.splitlines()[0] + "\nBF proc 61 1 0 1 S 160\t/usr/bin/bash\tbash late\n"
    scans = iter([SCAN, late + "BF task-procs status=0\n"])

    class _Scans(_Main):
        async def exec(self, command, **kwargs):
            if "BF proc" in command:
                self.commands.append("scan")
                return SimpleNamespace(
                    stdout=next(scans, "BF task-procs status=0\n"),
                    stderr="",
                    return_code=0,
                )
            return await super().exec(command, **kwargs)

    main = _Scans(SCAN)
    main._benchflow_own_processes = list(PROXY)
    with pytest.raises(RuntimeError, match="sandbox startup refused"):
        await lockdown.refuse_task_processes(main)
    kills = [c for c in main.commands if c.startswith("kill")]
    assert kills == [
        "kill -KILL 7 9 50 70 2>/dev/null; true",
        "kill -KILL 61 2>/dev/null; true",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scan", [SCAN, "BF task-procs status=2\n"], ids=["busy", "unread"]
)
async def test_r1s_refusal_is_not_retried(plain_reads, scan):
    """The task's image starts the same processes again: R1's refusal (and
    its could-not-read refusal) is one another attempt only repeats."""
    from benchflow._utils.scoring import is_unrecoverable_startup_error

    with pytest.raises(RuntimeError, match="sandbox startup refused") as refused:
        await lockdown.refuse_task_processes(_Main(scan))
    assert is_unrecoverable_startup_error(f"RuntimeError: {refused.value}")


@pytest.mark.asyncio
async def test_the_host_records_benchflows_daemon_with_its_start_time(plain_reads):
    main = _Main(SCAN, stat="30 300\n")
    await lockdown.note_own_process(main, pid_file="/opt/runtime-otel/receiver.pid")
    await lockdown.note_own_process(main, pid=30)
    assert lockdown.own_processes(main) == ((30, 300),)
    assert "read -r p < /opt/runtime-otel/receiver.pid" in main.commands[0]
    unread = _Main(SCAN, stat="no such process\n")
    await lockdown.note_own_process(unread, pid=30)
    assert lockdown.own_processes(unread) == ()

    class _Other:
        pass

    other = _Other()
    await lockdown.note_own_process(other, pid=30)
    assert lockdown.own_processes(other) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_env", "checked"),
    [
        ({"BENCHFLOW_DISALLOW_WEB_TOOLS": "1"}, True),
        ({lockdown.EGRESS_DENYLIST_ENV: "1"}, True),
        ({}, False),
    ],
    ids=["no-web", "denylist", "open"],
)
async def test_r1_runs_before_the_first_launch_under_a_policy(
    monkeypatch, agent_env, checked
):
    """Before the agent's process first starts, under a policy that holds
    only the agent; once per sandbox (a later connection finds what the
    agent itself left)."""
    seen: list[object] = []

    async def scan(env):
        seen.append(env)

    monkeypatch.setattr(lockdown, "refuse_task_processes", scan)
    main = _Main(SCAN)
    await lockdown.refuse_task_processes_before_launch(main, agent_env)
    await lockdown.refuse_task_processes_before_launch(main, agent_env)
    assert seen == ([main] if checked else [])


@pytest.mark.asyncio
async def test_the_launchs_firewall_no_longer_scans(monkeypatch):
    """The firewall the launch puts in only puts in its rules: R1 ran before
    the agent's process started."""
    calls: list[str] = []

    async def uid(env, user, **kwargs):
        calls.append(f"uid {user}")

    async def scan(env):  # pragma: no cover
        raise AssertionError("no scan")

    monkeypatch.setattr(lockdown, "enforce_sandbox_uid_egress", uid)
    monkeypatch.setattr(lockdown, "refuse_task_processes", scan)
    await lockdown.enforce_agent_egress_firewall(
        _Main(SCAN),
        "agent",
        {
            "BENCHFLOW_DISALLOW_WEB_TOOLS": "1",
            "BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:4000/v1",
        },
    )
    assert calls == ["uid agent"]


# --- the uid arm before the model proxy (the lead's 07:13Z go) ----------------


class _Arms:
    def __init__(self) -> None:
        self.calls: list[tuple] = []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sandbox_user", "no_web", "policy", "expected"),
    [
        ("agent", True, None, ("agent", None, True, None)),
        (None, True, None, ("root", None, True, 61997)),
        (
            "agent",
            False,
            SimpleNamespace(allow_mode=False),
            ("agent", None, False, None),
        ),
        (
            "agent",
            False,
            SimpleNamespace(allow_mode=True, allow_networks=("192.0.2.0/24",)),
            ("agent", ("192.0.2.0/24",), False, None),
        ),
        ("agent", False, None, None),
        (None, False, SimpleNamespace(allow_mode=False), None),
    ],
    ids=["no-web", "no-web-root", "denylist", "allowlist", "open", "denylist-root"],
)
async def test_the_early_arm_puts_in_the_launchs_rules(
    monkeypatch, sandbox_user, no_web, policy, expected
):
    """The rules the launch's own call would put in, read from the task's
    policy, before the proxy starts; nothing where the launch arms nothing or
    refuses (a denylist without a sandbox user). No scan: R1 is at the
    launch."""
    seen = _Arms()

    async def uid(env, user, *, allow_networks=None, root_gid=None, refuse_dns=False):
        seen.calls.append((user, allow_networks, refuse_dns, root_gid))

    async def scan(env):  # pragma: no cover
        raise AssertionError("no scan at the arm")

    monkeypatch.setattr(lockdown, "enforce_sandbox_uid_egress", uid)
    monkeypatch.setattr(lockdown, "refuse_task_processes", scan)
    armed = await lockdown.arm_agent_uid_before_its_proxy(
        object(), sandbox_user, no_web=no_web, egress_denylist=policy
    )
    if expected is None:
        assert armed is False and seen.calls == []
    else:
        assert armed is True
        assert seen.calls == [expected]


@pytest.mark.asyncio
async def test_the_early_arm_costs_the_launchs_execs_once_more(monkeypatch):
    """Exec counts: before the proxy, the early arm makes the launch's own
    calls, so a no-web or denylist trial runs them twice (the launch's are
    ``-C`` checked and add no rule): main's uid read and one helper run for
    the rules. An open network adds none (the counted fake trials are open:
    unchanged)."""
    from benchflow.sandbox import verifier_network

    seen: list[str] = []

    async def read(env, command, **kwargs):
        seen.append("scan" if "BF proc" in command else command)
        if command.startswith("id -u"):
            return SimpleNamespace(stdout="1500\n", stderr="", return_code=0)
        return SimpleNamespace(
            stdout="BF task-procs status=0\n", stderr="", return_code=0
        )

    async def helper(env, command, *, phase):
        seen.append(f"helper {phase}")

    monkeypatch.setattr(lockdown, "_read_exec", read)
    monkeypatch.setattr(verifier_network, "install_from_helper", helper)
    env = _Main("")
    env.rules_by_helper = True
    assert await lockdown.arm_agent_uid_before_its_proxy(env, "agent", no_web=True)
    assert [s for s in seen if not s.startswith("helper")] == ["id -u agent"]
    assert seen.count("helper agent-uid") == 1
    seen.clear()
    assert not await lockdown.arm_agent_uid_before_its_proxy(env, "agent", no_web=False)
    assert seen == []


# --- readiness execs end at their timeout --------------------------------------


def _running(marker: str) -> list[int]:
    """Live processes whose command line holds *marker*."""
    found = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            state = (d / "stat").read_text().rsplit(") ", 1)[1].split()[0]
        except (OSError, IndexError, UnicodeDecodeError):
            continue
        if marker in argv and state != "Z":
            found.append(int(d.name))
    return found


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_a_readiness_command_past_its_timeout_ends_with_its_group():
    """A readiness command still running at its timeout, and a process it
    started, are no longer running afterwards: the exec's process group
    ends (run here in a session of its own, as a docker exec is)."""
    from benchflow.sandbox._inert_bootstrap import bounded_readiness

    tag = random.randint(100, 999)
    kept, child = f"3601.{tag}", f"3602.{tag}"
    command = shlex.join(["sh", "-c", f"sleep {child} & exec sleep {kept}"])
    started = time.monotonic()
    done = subprocess.run(
        ["sh", "-c", bounded_readiness(command, 1.0)],
        start_new_session=True,
        timeout=60,
    )
    assert done.returncode == -9
    assert time.monotonic() - started < 10
    time.sleep(0.3)
    assert _running(kept) == [] and _running(child) == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
@pytest.mark.parametrize(("command", "status"), [("true", 0), ("exit 3", 3)])
def test_a_readiness_command_in_time_keeps_its_status(command, status):
    from benchflow.sandbox._inert_bootstrap import bounded_readiness

    started = time.monotonic()
    done = subprocess.run(
        ["sh", "-c", bounded_readiness(command, 5.0)],
        start_new_session=True,
        timeout=60,
    )
    assert done.returncode == status
    assert time.monotonic() - started < 3


@pytest.mark.asyncio
async def test_wait_healthy_runs_the_readiness_command_bounded(monkeypatch):
    """The healthcheck's exec carries its own timeout (the task's 3 s), and
    the host waits past it only as a backstop."""
    from benchflow.sandbox import _inert_bootstrap as bootstrap

    seen: dict[str, object] = {}

    async def wait_for(awaitable, timeout):
        seen["host_timeout"] = timeout
        return await awaitable

    async def route(*args, **kwargs):
        return SimpleNamespace(kit=False)

    def argv(route, command, **kwargs):
        seen["command"] = command
        return ["exec", command]

    async def cli(argv, check=False):
        return SimpleNamespace(return_code=0)

    monkeypatch.setattr(bootstrap.asyncio, "wait_for", wait_for)
    monkeypatch.setattr(bootstrap, "system_shell_route", route)
    monkeypatch.setattr("benchflow.sandbox.docker.route_exec_argv", argv)
    await bootstrap.wait_healthy(
        SimpleNamespace(_docker_cli=cli),
        "c" * 64,
        {"Healthcheck": {"Test": ["CMD", "/probe"], "Timeout": 3_000_000_000}},
    )
    assert seen["command"] == bootstrap.bounded_readiness("/usr/bin/env -- /probe", 3.0)
    assert seen["host_timeout"] == 3.0 + bootstrap.READINESS_GRACE_SEC


# --- M1: the uid rules right after release, before the install ----------------


@pytest.mark.asyncio
async def test_the_arm_at_release_makes_the_sandbox_user_first(monkeypatch):
    """Before the agent's install the sandbox user may not exist yet: the
    arm makes it (keeping one the image has), then puts the rules in."""
    from benchflow.sandbox import verifier_network

    seen: list[str] = []

    async def read(env, command, **kwargs):
        seen.append(command)
        return SimpleNamespace(stdout="1500\n", stderr="", return_code=0)

    async def helper(env, command, *, phase):
        seen.append(f"helper {phase}")

    monkeypatch.setattr(lockdown, "_read_exec", read)
    monkeypatch.setattr(verifier_network, "install_from_helper", helper)
    env = _Main("")
    env.rules_by_helper = True
    assert await lockdown.arm_agent_uid_before_its_proxy(
        env, "agent", no_web=True, create_user=True, timeout_sec=99
    )
    assert env.commands == [
        lockdown._create_sandbox_user_cmd("agent", above_regular_ids=True)
    ]
    assert seen == ["id -u agent", "helper agent-uid"]


@pytest.mark.asyncio
async def test_a_user_that_cannot_be_made_arms_nothing_at_release(monkeypatch):
    async def uid(env, user, **kwargs):  # pragma: no cover
        raise AssertionError("no rules without the user")

    class _NoUser(_Main):
        async def exec(self, command, **kwargs):
            self.commands.append(command)
            return SimpleNamespace(stdout="", stderr="no useradd", return_code=1)

    monkeypatch.setattr(lockdown, "enforce_sandbox_uid_egress", uid)
    assert not await lockdown.arm_agent_uid_before_its_proxy(
        _NoUser(""), "agent", no_web=True, create_user=True
    )


@pytest.mark.asyncio
async def test_the_start_asks_main_to_hold_the_agent_uid_before_release(monkeypatch):
    """Under no-network or denylist the rollout's start asks the sandbox for
    the agent's uid rules before main's release; the Docker start's hook
    puts them in through the same arm, the sandbox user made first, and
    records that they are in (the arm after release then has nothing to
    do)."""
    from benchflow.sandbox import service_cover
    from benchflow.sandbox.docker import DockerSandbox

    calls: list[tuple] = []

    class _Box:
        async def start(self, force_build=False):
            calls.append(("start", getattr(self, "_start_uid_hold", None)))

    box = _Box()
    ask = {"user": "agent", "no_web": True, "policy": None, "timeout_sec": 77}
    await service_cover.start_sandbox(box, SimpleNamespace(), agent=None, agent_uid=ask)
    assert calls == [("start", ask)]

    async def arm(env, user, **kwargs):
        calls.append(("arm", user, kwargs))
        return True

    monkeypatch.setattr(lockdown, "arm_agent_uid_before_its_proxy", arm)
    calls.clear()
    await DockerSandbox._hold_agent_uid_before_release(box)
    assert calls == [
        (
            "arm",
            "agent",
            {
                "no_web": True,
                "egress_denylist": None,
                "create_user": True,
                "timeout_sec": 77,
            },
        )
    ]
    assert box._agent_uid_held_at_start is True
    unasked = SimpleNamespace()
    await DockerSandbox._hold_agent_uid_before_release(unasked)
    assert unasked._agent_uid_held_at_start is False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="runs sh")
@pytest.mark.parametrize(
    ("login_defs", "adduser_conf", "first"),
    [
        (None, None, 60001),
        ("UID_MIN 1000\nUID_MAX 60000\nGID_MAX 60000\n", None, 60001),
        ("UID_MAX 62000\nGID_MAX 61000\n", "LAST_UID=59999\nLAST_GID = 63000\n", 63001),
    ],
    ids=["defaults", "login.defs", "highest-wins"],
)
def test_the_early_user_takes_the_first_id_above_the_images_regular_range(
    tmp_path, login_defs, adduser_conf, first
):
    """The start id a user made before the ENTRYPOINT runs counts up from:
    one past the highest of login.defs' UID_MAX and GID_MAX, adduser.conf's
    LAST_UID and LAST_GID, and 60000."""
    etc = tmp_path / "etc"
    etc.mkdir()
    script = lockdown._ABOVE_REGULAR_IDS_SH
    if login_defs is not None:
        (etc / "login.defs").write_text(login_defs)
    if adduser_conf is not None:
        (etc / "adduser.conf").write_text(adduser_conf)
    script = script.replace("/etc/", f"{etc}/")
    done = subprocess.run(
        ["sh", "-c", script + '; echo "$bf_uid"'],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.stdout.strip() == str(first), done.stderr


@pytest.mark.parametrize(
    ("rows", "task"),
    [
        # A command line is what a process says of itself: `exec -a sleep
        # bash infinity` reads as the keep-alive and runs bash.
        (
            "BF proc 1 0 0 0 S 1\t/usr/bin/bash\tbash\tsleep infinity\n",
            [1],
        ),
        # An "init" by its argv[0] only.
        (
            "BF proc 1 0 0 0 S 1\t/usr/bin/python3.12\tpython3\ttini /net.py\n",
            [1],
        ),
        # A real init with a connection out is no keep-alive either.
        (
            "BF proc 1 0 0 2 S 1\t/usr/bin/tini\ttini\t/usr/bin/tini -- sh -c sleep infinity\n",
            [1],
        ),
        # Another user's PID 1, whose program root may not read: its kernel
        # name decides.
        ("BF proc 1 0 1500 0 S 1\t\tsleep\tsleep infinity\n", []),
        ("BF proc 1 0 1500 0 S 1\t\tpython3\tsleep infinity\n", [1]),
        # The keep-alive's own programs, a shell's and sleep's.
        (
            "BF proc 1 0 0 0 S 1\t/bin/busybox\tsh\tsh -c sleep infinity\n"
            "BF proc 5 1 0 0 S 2\t/bin/busybox\tsleep\tsleep infinity\n",
            [],
        ),
    ],
    ids=[
        "argv-sleep",
        "argv0-tini",
        "init-with-socket",
        "other-user",
        "other-user-python",
        "busybox",
    ],
)
def test_the_keep_alive_and_an_init_are_known_by_their_program(rows, task):
    """R698B N1: neither a command line (`exec -a`) nor argv[0] makes a
    process BenchFlow's keep-alive or an init; its program does (the
    executable root reads, or the kernel's name for another user's)."""
    assert _pids(lockdown._task_processes(rows + "BF task-procs status=0\n")) == task


@pytest.mark.asyncio
async def test_a_refused_pid_1_is_stopped_from_the_host(plain_reads):
    """R698B N2: a task process that is PID 1, which nothing in main may
    kill, is stopped from the host at the refusal; background ones are
    killed from inside as before."""
    killed: list[str] = []

    class _Pid1(_Main):
        async def kill_main(self):
            killed.append("main")

    scan = (
        "BF proc 1 0 0 1 S 1\t/usr/bin/python3.12\tpython3\tpython3 /net.py\n"
        "BF task-procs status=0\n"
    )
    with pytest.raises(RuntimeError, match=re.escape("pid 1 (uid 0): python3 /net.py")):
        await lockdown.refuse_task_processes(_Pid1(scan))
    assert killed == ["main"]
    quiet = _Pid1(SCAN)
    quiet._benchflow_own_processes = list(PROXY)
    killed.clear()
    with pytest.raises(RuntimeError):
        await lockdown.refuse_task_processes(quiet)
    assert killed == []
