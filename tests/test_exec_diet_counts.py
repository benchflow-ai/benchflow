"""How many execs a trial runs inside its sandbox (PY8 U12, research brief Part 3 A2).

Every exec costs a round trip: on Docker a ``docker compose exec`` process
(about 0.1 s each), on Daytona a session and its create, execute, status and
logs requests. U12 runs commands that need nothing from the host between them
in one exec (``sandbox/_exec_merge.py``), keeps the default user's ``pwd`` and
uid from the start, skips parent ``mkdir``s that cannot matter, and sends
short Daytona execs synchronously. These tests count what a trial makes; on
the base (``integ/py-0.8`` fc2b9e9a plus U5b) each one fails by its count, not
by a missing name.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._bfx import locate
from benchflow._utils.trusted_path import PATH_QUERY_CMD, PYTHONPATH_QUERY_CMD
from tests._trust_admission import admitted
from tests.exec_diet_fakes import MERGED_MARK, Exec, caller, docker_available
from tests.integration.deterministic import harness as h

EXEC_LIST_ENV = "BENCHFLOW_EXEC_LIST_REPORT"

# The environment reads and the start probe's commands, as their readers run
# them alone.
_READS = {
    PATH_QUERY_CMD,
    PYTHONPATH_QUERY_CMD,
    "cat /etc/os-release 2>/dev/null || true",
    "if command -v python3 >/dev/null 2>&1; then exit 0; else exit 1; fi",
}


def _is_read(command: str) -> bool:
    from benchflow.sandbox._exec_receipt import unwrap_receipt_command

    payload = unwrap_receipt_command(command)
    return (
        command in _READS
        or MERGED_MARK in command
        or (payload is not None and payload[0] in _READS)
    )


_READERS = {
    # V3's encoded PYTHONPATH read uses _checked_read (99d90d2cd);
    # PERF-G moved it into _read_image_pythonpath (1d659a374).
    # _is_read still requires the exact read at the receipt's start.
    "_checked_read",
    "_build_verifier_env",
    "_probe_verifier_reads",
    "_trusted_verifier_path",
    "_trusted_verifier_pythonpath",
    "_distro_pip_env",
    "_python3_absent",
}


def _group(e: Exec) -> str:
    """Which of U12's groups an exec belongs to ("other" for the rest)."""
    command = e.command
    if (
        e.caller == "_prepare_log_dirs_after_up"
        or (e.caller == "_resolve_agent_cwd" and command == "pwd")
        or (e.caller == "_verifier_uid" and e.user is None)
    ):
        return "start_probe"
    # Since V7 the snapshot's copy rides in front of the bash-env inspectors'
    # root exec, so that exec is the snapshot's (with U12, the manifest too).
    if e.caller == "_snapshot_build_config" or (
        e.caller == "_prepare_bash_env_inspector"
        and "runtime_build_snapshot" in command
    ):
        return "snapshot"
    if e.caller == "_seed_verifier_workspace":
        return "seed"
    if e.caller == "_run_oracle" and "oracle.txt" in command:
        return "oracle"
    if e.caller in _READERS and _is_read(command):
        return "environment_reads"
    if "verifier-started-" not in command and (
        ("chmod +x" in command and "test.sh" in command)
        or "mkdir -p /run/runtime" in command
    ):
        return "verifier_start"
    if command == "mkdir -p /":
        return "upload_root_mkdir"
    if e.caller in ("_publish_trajectory_for_verifier", "_publish_step") or (
        e.caller == "upload_file" and "/logs/.runtime-publish-" in command
    ):
        return "publish"
    return "other"


def _report(execs: list[Exec], name: str) -> None:
    path = os.environ.get(EXEC_LIST_ENV)
    if not path:
        return
    out = Path(path)
    data = json.loads(out.read_text()) if out.exists() else {}
    data[name] = [
        {
            "group": _group(e),
            "caller": e.caller,
            "user": e.user,
            "timeout_sec": e.timeout_sec,
            "command": e.command[:200],
        }
        for e in execs
    ]
    out.write_text(json.dumps(data, indent=1))


@pytest.mark.deterministic
async def test_an_oracle_trial_on_docker_makes_fewer_execs(monkeypatch, tmp_path):
    """The deterministic task's oracle trial on Docker, through
    ``install_agent``, ``_run_oracle`` and ``verify``, with every exec
    recorded (command, user, budget, caller). Guards the exec accounting
    after V3 ac543f619 and V2 039774d05/a4d6f6aa5 merged in batch 32.
    Each U12 group makes one exec
    where the base made more; the oracle and verifier start keep their two."""
    if not docker_available():
        pytest.skip("needs a Docker daemon")
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.sandbox.docker import DockerSandbox

    execs: list[Exec] = []
    original = DockerSandbox.exec

    async def recording(
        self,
        command,
        cwd=None,
        env=None,
        timeout_sec=None,
        user=None,
        service="main",
        **kwargs,
    ):
        execs.append(Exec(command, user, timeout_sec, cwd, caller()))
        return await original(
            self,
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
            service=service,
            **kwargs,
        )

    monkeypatch.setattr(DockerSandbox, "exec", recording)
    task = tmp_path / "deterministic"
    shutil.copytree(h.TEMPLATE_TASK, task)
    result = await Rollout(
        RolloutConfig(
            task_path=task,
            agent="oracle",
            environment="docker",
            jobs_dir=tmp_path / "jobs",
            job_name="u12-counts",
            rollout_name="deterministic__u12",
        )
    ).run()
    _report(execs, "docker-oracle-deterministic")
    assert result.error is None, result.error
    assert result.rewards == {"reward": 1.0}
    groups = Counter(_group(e) for e in execs)
    assert {
        name: groups[name]
        for name in (
            "start_probe",
            "snapshot",
            "seed",
            "oracle",
            "environment_reads",
            "verifier_start",
            "upload_root_mkdir",
            "publish",
        )
    } == {
        "start_probe": 1,
        "snapshot": 1,
        "seed": 1,
        "oracle": 2,
        # The merged root reads, and PYTHONPATH on the task route (C8).
        "environment_reads": 2,
        "verifier_start": 2,
        "upload_root_mkdir": 0,
        # The staged publish (V6): the stage with the defensive
        # /logs/agent mkdir in one root exec, upload_file's parent mkdir,
        # chmod, the copy as the sandbox user, the stage's removal (6 on
        # integ/py-0.8 008da0e0). Why the last four stay apart:
        # tests/test_publish_exec_fold.py.
        "publish": 5,
    }, [(e.caller, e.command[:60]) for e in execs]
    # 47 on integ/py-0.8 363d815b alone, 9 fewer with U12: the start probe 3
    # to 1, upload's mkdir -p / gone, the manifest in the snapshot's exec
    # (which V7's bash-env inspectors share), the seed 3 to 1, the reads 3 to
    # 1, the defensive publish mkdir in the stage's exec.
    # 38 on integ/py-0.8 cfca0513; the mount provenance read before each
    # ownership freeze (_blocked_before_freeze, CONFIRM-lockdown-3 and -5)
    # runs twice in this trial: two execs more.
    # 39 on integ/py-0.8 5653de15, which holds both 43a69cb78 and C8. The
    # verifier's preparation is one exec where it was seven (the
    # /logs/verifier reset, mkdir /app, the cache reclaim, two workspace
    # freezes, the workspace check, the conftest/sitecustomize/.pth sweep):
    # six fewer. C8 adds five: the protected kit's build and seal, the PATH
    # lock, the PYTHONPATH read on the task route and a uid read. The
    # workspace trust replaces the git safe.directory exec one for one.
    # 44 on integ/py-0.8 fb00ee747: V3 ac543f619 adds three execs
    # (read excluded inode identities, make their directory, write them).
    # V2 039774d05 adds the agent's group read; a4d6f6aa5 adds the image
    # pytest-config read. All 39 old steps remain, including preparation;
    # the PATH judgment and plugin discovery have replacement commands.
    # 45 with PERF-SBX 661670103 (batch 42): one exec more, _clear_system_route,
    # the kit Python's walk of the whole system route; a CLEAR result lets
    # the sh walks after it be skipped.
    # 46 where bfx is delivered, with TR-P2's hook clean-up on bfx
    # (8a674c8e4, 6fd1e5756): the host decides the clean-up from test.sh's
    # search-path inputs, read in their own receipt exec after the
    # preparation (_hook_cleanup.INPUTS_CMD). A checkout without the pinned
    # binaries keeps the clean-up inside the preparation: 45.
    want = 46 if locate.available(platform.machine()) else 45
    assert len(execs) == want, [(e.caller, e.command[:60]) for e in execs]
    # The preparation still runs, once, as root.
    prep = [e for e in execs if "BENCHFLOW-PREP" in e.command]
    assert len(prep) == 1 and prep[0].user == "root", [e.caller for e in prep]
    # No redundant default-user pwd or uid read after the start probe.
    assert not [e for e in execs if e.command in ("pwd", "id -u") and e.user is None]
    # Each kept its user and budget: the merged start probe runs as the
    # default user with pwd's 10 s; the reads and verifier start run as root.
    by_group = {_group(e): e for e in execs}
    assert by_group["start_probe"].user is None
    assert by_group["start_probe"].timeout_sec == 10
    assert by_group["environment_reads"].user == "root"
    assert by_group["environment_reads"].timeout_sec == 180
    assert by_group["verifier_start"].user == "root"
    assert by_group["verifier_start"].timeout_sec == 10
    assert [e.timeout_sec for e in execs if _group(e) == "verifier_start"] == [60, 10]
    assert by_group["snapshot"].user == by_group["seed"].user == "root"


# --- Docker's upload parents ---------------------------------------------------


def _docker_sandbox(calls: list):
    from benchflow.sandbox.docker import DockerSandbox

    sandbox = DockerSandbox.__new__(DockerSandbox)

    async def exec_(command, **kwargs):
        calls.append(("exec", command, kwargs.get("user")))
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    async def compose(args, check=True, timeout_sec=None):
        calls.append(("compose", args[0], args[-1]))
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    async def mode(target, mode):
        return None

    sandbox.exec = exec_
    sandbox._run_docker_compose_command = compose
    sandbox._apply_upload_mode = mode
    return sandbox


async def test_docker_upload_to_the_root_folder_runs_no_mkdir(tmp_path):
    calls: list = []
    source = tmp_path / "instruction.md"
    source.write_text("do it")
    sandbox = _docker_sandbox(calls)
    await sandbox.upload_file(source, "/instruction.md", admission=admitted(sandbox))
    assert calls == [("compose", "cp", "main:/instruction.md")]


async def test_docker_upload_still_makes_a_missing_parent(tmp_path):
    calls: list = []
    source = tmp_path / "b"
    source.write_text("b")
    sandbox = _docker_sandbox(calls)
    await sandbox.upload_file(source, "/a/b", admission=admitted(sandbox))
    assert calls == [("exec", "mkdir -p /a", "root"), ("compose", "cp", "main:/a/b")]


# The staged trajectory publish: tests/test_publish_exec_fold.py.


# --- Daytona: which execs go synchronous, and what they request --------------


class _Process:
    """Records each request; a synchronous execute answers with the result,
    an asynchronous one is done at its first status request."""

    def __init__(self) -> None:
        self.requests: list[str] = []
        self.run_async: list[bool] = []

    async def create_session(self, session_id):
        self.requests.append("create")

    async def delete_session(self, session_id):
        self.requests.append("delete")

    async def execute_session_command(self, session_id, request, timeout=None):
        self.requests.append("execute")
        self.run_async.append(request.run_async)
        if request.run_async is False:
            return SimpleNamespace(
                cmd_id="c", exit_code=0, stdout="out\n", stderr="", output="out\n"
            )
        return SimpleNamespace(cmd_id="c", exit_code=None)

    async def get_session_command(self, session_id, command_id):
        self.requests.append("status")
        return SimpleNamespace(id=command_id, exit_code=0)

    async def get_session_command_logs(self, session_id, command_id):
        self.requests.append("logs")
        return SimpleNamespace(stdout="out\n", stderr="")


def _daytona(rollout_dir: Path | None = None):
    pytest.importorskip("daytona")
    from benchflow.sandbox.daytona import DaytonaSandbox, _load_daytona_sdk
    from benchflow.sandbox.daytona_strategies import _DaytonaDirect

    _load_daytona_sdk()
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    process = _Process()
    sandbox._sandbox = SimpleNamespace(process=process)
    sandbox._strategy = _DaytonaDirect(sandbox)
    sandbox.rollout_paths = (
        SimpleNamespace(rollout_dir=rollout_dir) if rollout_dir is not None else None
    )
    return sandbox, process


@pytest.mark.parametrize("budget", [1, 10, 30])
async def test_a_short_daytona_exec_is_synchronous_when_enabled(budget, monkeypatch):
    from benchflow.sandbox import daytona as daytona_module

    monkeypatch.setattr(daytona_module, "_DAYTONA_SYNC_EXEC_MAX_SEC", 30)
    sandbox, process = _daytona()
    result = await sandbox.exec("echo out", timeout_sec=budget)
    assert (result.return_code, result.stdout) == (0, "out\n")
    assert process.run_async == [False]
    # The answer holds the result: no status or logs request.
    assert process.requests == ["create", "execute"]


@pytest.mark.parametrize("budget", [None, 0, -1, 31, 600])
async def test_other_daytona_execs_keep_the_asynchronous_poll(budget):
    sandbox, process = _daytona()
    result = await sandbox.exec("echo out", timeout_sec=budget)
    assert (result.return_code, result.stdout) == (0, "out\n")
    assert process.run_async == [True]
    assert process.requests == ["create", "execute", "status", "logs"]


async def test_a_short_transient_read_is_synchronous_and_cleaned_up(monkeypatch):
    from benchflow.sandbox import daytona as daytona_module

    monkeypatch.setattr(daytona_module, "_DAYTONA_SYNC_EXEC_MAX_SEC", 30)
    sandbox, process = _daytona()
    await sandbox.exec_transient("cat /tmp/gateway.log", timeout_sec=10)
    assert process.run_async == [False]
    assert process.requests == ["create", "execute", "delete"]


async def test_the_constant_at_zero_sends_every_exec_as_before(monkeypatch):
    from benchflow.sandbox import daytona as daytona_module

    monkeypatch.setattr(daytona_module, "_DAYTONA_SYNC_EXEC_MAX_SEC", 0, raising=False)
    sandbox, process = _daytona()
    for budget in (1, 10, 30, None):
        await sandbox.exec("echo out", timeout_sec=budget)
    assert process.run_async == [True] * 4
    assert process.requests.count("status") == 4


async def test_the_record_counts_synchronous_execs(tmp_path, monkeypatch):
    from benchflow.sandbox import daytona as daytona_module

    monkeypatch.setattr(daytona_module, "_DAYTONA_SYNC_EXEC_MAX_SEC", 30)
    monkeypatch.setattr(
        daytona_module, "_DAYTONA_COMMAND_POLL_FIRST_SEC", 0.001, raising=False
    )
    sandbox, _ = _daytona(tmp_path)
    sandbox._strategy.stop = _noop_stop
    await sandbox.exec("true", timeout_sec=10)
    await sandbox.exec("true", timeout_sec=None)
    await sandbox.exec_transient("true", timeout_sec=5)
    await sandbox.stop(delete=True)
    record = json.loads((tmp_path / "daytona-execs.json").read_text())
    assert record["sync_exec_calls"] == 2
    # U5b's fields as before: two ordinary execs, one transient; only the
    # asynchronous one asked for its status.
    assert (record["exec_calls"], record["transient_exec_calls"]) == (2, 1)
    assert (record["status_requests"], record["transient_status_requests"]) == (1, 0)


async def _noop_stop(delete):
    await asyncio.sleep(0)
