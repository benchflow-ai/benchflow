"""Exec calls and poll requests of one trial on a fake Daytona API (PY8 U5b).

Every Daytona exec is a session command BenchFlow polls until it is done
(``DaytonaSandbox._poll_response``). The poll used to wait a fixed 1.0 s
between two status requests; it now starts at 0.05 s and doubles up to 1.0 s.
This test runs one real oracle trial of the deterministic task
(``tests/integration/deterministic/task``) with ``--sandbox daytona``, through
the real ``Rollout``, ``DaytonaSandbox`` and Daytona SDK types, against a fake
Daytona API: the client, sandbox, ``process`` and ``fs`` objects BenchFlow
calls are stand-ins that run each command in a Docker container of the task's
image and answer the status requests the way Daytona does (no exit code until
the command has finished). The trial runs twice, with the old fixed wait and
with the new schedule, and the test counts for each run: exec calls, status
and log requests, sessions, uploads and downloads, and the time each exec
waited after its command had finished (the time the poll costs). Both runs
must make the same execs with the same exit codes and reward; the new
schedule must cut the waiting.

Set ``BENCHFLOW_EXEC_COUNT_REPORT`` to a file path to keep the counts and the
list of execs (which code made each one) as JSON.

Since PY8 U12 the fake also answers ``run_async=False`` the way SDK 0.184.0's
toolbox does (the exit code, stdout and stderr in the execute answer), and
the U5b comparison runs with U12's synchronous execs off in both arms, so it
still compares the poll schedules alone. U12's own trial
(``test_one_fake_trial_with_merged_execs``) uses production defaults (sync
disabled); ``BENCHFLOW_EXEC_COUNT_REPORT_U12`` keeps its counts.

Docker and the ``sandbox-daytona`` extra are needed; without them the test
skips. An oracle trial runs nothing as the sandbox user, so it installs no
daemon guard and no agent: an agent trial makes those execs on top of these.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
import traceback
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.integration.deterministic import harness as h

# Both trials share the module's image, which the fixture removes when its
# module is done: under -n (CI's -n auto) two workers would each build it and
# one would remove it under the other's trial (PY8 U12).
pytestmark = [
    pytest.mark.deterministic,
    pytest.mark.xdist_group("fake_daytona_image"),
]

REPORT_ENV = "BENCHFLOW_EXEC_COUNT_REPORT"
IMAGE = "benchflow-test-fake-daytona:deterministic"


@dataclass
class _Exec:
    caller: str
    command: str
    started: float
    finished: float | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    status_requests: int = 0
    logs_at: float | None = None
    # Sent with run_async=False (answered with the result).
    sync: bool = False


@dataclass
class _Counts:
    execs: list[_Exec] = field(default_factory=list)
    sessions_created: int = 0
    sessions_deleted: int = 0
    uploads: int = 0
    downloads: int = 0
    searches: int = 0
    file_infos: int = 0
    # The trial's own daytona-execs.json.
    record: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        lag = [
            e.logs_at - e.finished
            for e in self.execs
            if e.logs_at is not None and e.finished is not None
        ]
        return {
            "exec_calls": len(self.execs),
            "status_requests": sum(e.status_requests for e in self.execs),
            "log_requests": sum(1 for e in self.execs if e.logs_at is not None),
            "sync_execs": sum(1 for e in self.execs if e.sync),
            # Status requests of synchronous execs whose answer lacked the
            # result (counted in status_requests too).
            "sync_status_requests": sum(
                e.status_requests for e in self.execs if e.sync
            ),
            "sessions_created": self.sessions_created,
            "sessions_deleted": self.sessions_deleted,
            "uploads": self.uploads,
            "downloads": self.downloads,
            "searches": self.searches,
            "file_infos": self.file_infos,
            "command_seconds": round(
                sum(
                    e.finished - e.started for e in self.execs if e.finished is not None
                ),
                3,
            ),
            "wait_after_done_seconds": round(sum(lag), 3),
        }


def _caller() -> str:
    """The BenchFlow code that asked for this exec: the two innermost frames
    in the package outside the Daytona backend and the exec receipt."""
    import benchflow

    package = str(Path(benchflow.__file__).parent) + os.sep
    backend = (
        os.path.join(package, "sandbox", "daytona"),
        os.path.join(package, "sandbox", "_base.py"),
        os.path.join(package, "sandbox", "_exec_receipt.py"),
        # exec_raw passes every machine read through (the readers' own
        # frames name the exec), like the receipt.
        os.path.join(package, "sandbox", "_exec_output.py"),
    )
    frames = [
        f
        for f in traceback.extract_stack()
        if f.filename.startswith(package) and not f.filename.startswith(backend)
    ]
    names = [f"{Path(f.filename).stem}.{f.name}" for f in frames[-2:]]
    return " < ".join(reversed(names)) or "?"


async def _docker(*args: str, stdin: bytes | None = None) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = await proc.communicate(stdin)
    return (
        proc.returncode or 0,
        out.decode(errors="replace"),
        err.decode(errors="replace"),
    )


class _FakeProcess:
    """``sandbox.process``: session commands run with ``docker exec``."""

    def __init__(self, sandbox: _FakeSandbox) -> None:
        self._sandbox = sandbox
        self._commands: dict[str, _Exec] = {}
        self._sessions: dict[str, str] = {}
        self._tasks: set[asyncio.Task] = set()

    async def create_session(self, session_id):
        self._sandbox.counts.sessions_created += 1

    async def delete_session(self, session_id):
        self._sandbox.counts.sessions_deleted += 1

    async def execute_session_command(self, session_id, request, timeout=None):
        record = _Exec(
            caller=_caller(), command=request.command, started=time.monotonic()
        )
        self._sandbox.counts.execs.append(record)
        cmd_id = uuid.uuid4().hex

        async def run() -> None:
            code, out, err = await _docker(
                "exec", self._sandbox.id, "sh", "-c", request.command
            )
            record.stdout, record.stderr, record.exit_code = out, err, code
            record.finished = time.monotonic()

        self._commands[cmd_id] = record
        self._sessions[session_id] = cmd_id
        if getattr(request, "run_async", True) is False:
            # Answered when the command is done, with its result.
            record.sync = True
            await run()
            return SimpleNamespace(
                cmd_id=cmd_id,
                exit_code=record.exit_code,
                stdout=record.stdout,
                stderr=record.stderr,
                output=record.stdout + record.stderr,
            )
        task = asyncio.ensure_future(run())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return SimpleNamespace(cmd_id=cmd_id)

    async def get_session(self, session_id):
        cmd_id = self._sessions.get(session_id)
        commands = []
        if cmd_id is not None:
            record = self._commands[cmd_id]
            record.status_requests += 1
            commands.append(
                SimpleNamespace(
                    id=cmd_id,
                    exit_code=record.exit_code if record.finished is not None else None,
                )
            )
        return SimpleNamespace(commands=commands)

    async def exec(self, command, env=None, timeout=None):
        # The toolbox's direct exec: V7 reads the shells' BASH_ENV with it
        # right after create. Not a session command, so not counted.
        flags = [arg for k, v in (env or {}).items() for arg in ("-e", f"{k}={v}")]
        code, out, err = await _docker(
            "exec", *flags, self._sandbox.id, "sh", "-c", command
        )
        return SimpleNamespace(exit_code=code, result=out + err)

    async def get_session_command(self, session_id, command_id):
        record = self._commands[command_id]
        record.status_requests += 1
        exit_code = record.exit_code if record.finished is not None else None
        return SimpleNamespace(id=command_id, exit_code=exit_code)

    async def get_session_command_logs(self, session_id, command_id):
        record = self._commands[command_id]
        record.logs_at = time.monotonic()
        return SimpleNamespace(stdout=record.stdout, stderr=record.stderr)


class _FakeFs:
    """``sandbox.fs``: uploads and downloads with tar through ``docker exec``."""

    def __init__(self, sandbox: _FakeSandbox) -> None:
        self._sandbox = sandbox

    def _not_found(self, path: str) -> Exception:
        from daytona import DaytonaNotFoundError

        return DaytonaNotFoundError(f"{path}: not found")

    async def _put(self, source: Any, destination: str) -> None:
        data = source if isinstance(source, bytes) else Path(source).read_bytes()
        code, _, err = await _docker(
            "exec",
            "-i",
            self._sandbox.id,
            "sh",
            "-c",
            'mkdir -p "$(dirname "$1")" && cat > "$1"',
            "sh",
            destination,
            stdin=data,
        )
        if code != 0:
            raise RuntimeError(f"fake upload of {destination} failed: {err}")

    async def upload_file(self, source, destination, timeout=None, **_kwargs):
        self._sandbox.counts.uploads += 1
        await self._put(source, destination)

    async def upload_files(self, files, timeout=None):
        self._sandbox.counts.uploads += 1
        for upload in files:
            await self._put(upload.source, upload.destination)

    async def _get(self, source: str) -> bytes:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            self._sandbox.id,
            "sh",
            "-c",
            '[ -f "$1" ] && cat "$1"',
            "sh",
            source,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            raise self._not_found(source)
        return out

    async def download_file(self, source, destination=None, timeout=None):
        self._sandbox.counts.downloads += 1
        data = await self._get(source)
        if destination is None:
            return data
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(data)
        return None

    async def download_files(self, files, timeout=None):
        self._sandbox.counts.downloads += 1
        responses = []
        for request in files:
            try:
                data = await self._get(request.source)
            except Exception as exc:
                responses.append(SimpleNamespace(source=request.source, error=str(exc)))
                continue
            if request.destination:
                Path(request.destination).parent.mkdir(parents=True, exist_ok=True)
                Path(request.destination).write_bytes(data)
            responses.append(SimpleNamespace(source=request.source, error=None))
        return responses

    async def search_files(self, path, pattern):
        self._sandbox.counts.searches += 1
        _, out, _ = await _docker(
            "exec",
            self._sandbox.id,
            "find",
            path,
            "-mindepth",
            "1",
            "-name",
            pattern,
        )
        return SimpleNamespace(files=[line for line in out.splitlines() if line])

    async def get_file_info(self, path):
        self._sandbox.counts.file_infos += 1
        code, out, _ = await _docker(
            "exec", self._sandbox.id, "sh", "-c", 'stat -c %F "$1"', "sh", path
        )
        if code != 0:
            raise self._not_found(path)
        return SimpleNamespace(is_dir=out.strip() == "directory")

    async def list_files(self, path):
        _, out, _ = await _docker(
            "exec",
            self._sandbox.id,
            "find",
            path,
            "-mindepth",
            "1",
            "-maxdepth",
            "1",
            "-type",
            "d",
            "-printf",
            "%f\n",
        )
        return [SimpleNamespace(name=name, is_dir=True) for name in out.splitlines()]


class _FakeSandbox:
    def __init__(self, container_id: str, counts: _Counts) -> None:
        self.id = container_id
        self.counts = counts
        self.process = _FakeProcess(self)
        self.fs = _FakeFs(self)

    async def delete(self, timeout=60, wait=False):
        await _docker("rm", "-f", self.id)


class _FakeClient:
    def __init__(self, counts: _Counts) -> None:
        self.counts = counts
        self.sandboxes: list[_FakeSandbox] = []

    async def create(self, params=None, timeout=None):
        code, out, err = await _docker(
            "run",
            "-d",
            "--label",
            "benchflow.test=fake-daytona",
            IMAGE,
            "sleep",
            "infinity",
        )
        if code != 0:
            raise RuntimeError(f"fake create failed: {err}")
        sandbox = _FakeSandbox(out.strip(), self.counts)
        self.sandboxes.append(sandbox)
        return sandbox

    async def close(self):
        return None


def _manager_for(client: _FakeClient) -> type:
    class _FakeManager:
        @classmethod
        async def get_instance(cls):
            return cls()

        async def get_client(self):
            return client

    return _FakeManager


@pytest.fixture(scope="module")
def fake_daytona_image() -> Iterator[str]:
    if not h._docker_available():
        pytest.skip("needs a Docker daemon to back the fake Daytona API")
    pytest.importorskip("daytona")
    subprocess.run(
        ["docker", "build", "-q", "-t", IMAGE, str(h.TEMPLATE_TASK / "environment")],
        check=True,
        capture_output=True,
        timeout=900,
    )
    yield IMAGE
    subprocess.run(["docker", "rmi", "-f", IMAGE], capture_output=True, timeout=120)


async def _one_trial(monkeypatch, tmp_path: Path, arm: str) -> tuple[Any, _Counts]:
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.sandbox import daytona as daytona_module

    counts = _Counts()
    client = _FakeClient(counts)
    monkeypatch.setattr(daytona_module, "DaytonaClientManager", _manager_for(client))
    task = tmp_path / arm / "deterministic"
    shutil.copytree(h.TEMPLATE_TASK, task)
    try:
        rollout = Rollout(
            RolloutConfig(
                task_path=task,
                agent="oracle",
                environment="daytona",
                jobs_dir=tmp_path / arm / "jobs",
                job_name=f"u5b-{arm}",
                rollout_name=f"deterministic__{arm}",
            )
        )
        result = await rollout.run()
    finally:
        for sandbox in client.sandboxes:
            await sandbox.delete()
    records = list((tmp_path / arm / "jobs").rglob("daytona-execs.json"))
    assert len(records) == 1, records
    counts.record = json.loads(records[0].read_text())
    return result, counts


async def test_one_fake_trial_counts_execs_and_polls_before_and_after(
    fake_daytona_image, monkeypatch, tmp_path
) -> None:
    """Guards PY8 U5b (``_command_poll_waits`` in ``sandbox/daytona.py``):
    the same trial makes the same execs with the same exit codes, and its
    execs wait far less after their commands are done."""
    from benchflow.sandbox import daytona as daytona_module

    runs: dict[str, tuple[Any, _Counts]] = {}
    # U12's synchronous execs off in both arms: this compares the poll
    # schedules alone.
    with monkeypatch.context() as before:
        # The old schedule: every wait 1.0 s.
        before.setattr(
            daytona_module, "_DAYTONA_COMMAND_POLL_FIRST_SEC", 1.0, raising=False
        )
        before.setattr(daytona_module, "_DAYTONA_SYNC_EXEC_MAX_SEC", 0, raising=False)
        runs["before"] = await _one_trial(before, tmp_path, "before")
    with monkeypatch.context() as after:
        after.setattr(daytona_module, "_DAYTONA_SYNC_EXEC_MAX_SEC", 0, raising=False)
        runs["after"] = await _one_trial(after, tmp_path, "after")

    summary = {arm: counts.summary() for arm, (_, counts) in runs.items()}
    for arm, (_, counts) in runs.items():
        summary[arm]["trial_record"] = counts.record
    report = os.environ.get(REPORT_ENV)
    if report:
        Path(report).write_text(
            json.dumps(
                {
                    "summary": summary,
                    "execs": {
                        arm: [
                            {
                                "caller": e.caller,
                                "exit_code": e.exit_code,
                                "status_requests": e.status_requests,
                                "command_seconds": round(
                                    (e.finished or e.started) - e.started, 3
                                ),
                                "wait_after_done_seconds": round(
                                    (e.logs_at or 0) - (e.finished or 0), 3
                                )
                                if e.logs_at is not None
                                else None,
                                "command": e.command[:160],
                            }
                            for e in counts.execs
                        ]
                        for arm, (_, counts) in runs.items()
                    },
                },
                indent=1,
            )
        )
    print(json.dumps(summary, indent=1))

    for arm, (result, _) in runs.items():
        assert result.error is None, (arm, result.error)
        assert result.rewards == {"reward": 1.0}, (arm, result.rewards)
    before_execs = runs["before"][1].execs
    after_execs = runs["after"][1].execs
    assert [(e.caller, e.exit_code) for e in after_execs] == [
        (e.caller, e.exit_code) for e in before_execs
    ]
    assert summary["after"]["exec_calls"] == summary["before"]["exec_calls"] > 10
    assert summary["after"]["log_requests"] == summary["before"]["log_requests"]
    # Every exec waits at least one 1.0 s step under the old schedule unless
    # its command was done at the first status request; under the new one
    # a short command waits 50 to 150 ms.
    waited_before = summary["before"]["wait_after_done_seconds"]
    waited_after = summary["after"]["wait_after_done_seconds"]
    assert waited_after < waited_before / 3, summary
    # The trial folder records the same execs and status requests, the
    # transient ones (session deleted afterward) apart.
    for arm, (_, counts) in runs.items():
        record = counts.record
        assert (
            record["exec_calls"] + record["transient_exec_calls"]
            == summary[arm]["exec_calls"]
        ), arm
        assert (
            record["status_requests"] + record["transient_status_requests"]
            == summary[arm]["status_requests"]
        ), arm
        assert record["transient_exec_calls"] == summary[arm]["sessions_deleted"]

    def seconds(arm: str) -> float:
        record = runs[arm][1].record
        return record["exec_seconds"] + record["transient_exec_seconds"]

    assert seconds("after") < seconds("before") / 2


# The execs U12 merges (PY8 U12), told apart by their command and the
# function that made them; on U5b's tip this trial makes 3, 2, 3, 3 and 2,
# and on integ/py-0.8 008da0e0 the staged trajectory publish makes 5.
U12_GROUPS = (
    "start",
    "snapshot",
    "seed",
    "environment_reads",
    "verifier_start",
    "publish",
)
U12_REPORT_ENV = "BENCHFLOW_EXEC_COUNT_REPORT_U12"
_READERS = (
    "lockdown._trusted_verifier_path",
    "lockdown._trusted_verifier_pythonpath",
    "lockdown._distro_pip_env",
    "lockdown._python3_absent",
    "lockdown._probe_verifier_reads",
)


def _u12_group(e: _Exec) -> str:
    innermost = e.caller.split(" < ")[0]
    command = e.command
    if (
        "chmod 777 /logs/agent /logs/verifier" in command
        or (innermost == "_setup._resolve_agent_cwd" and command.endswith(" pwd"))
        or (innermost == "lockdown._verifier_uid" and "su " not in command)
    ):
        return "start"
    if innermost in ("_setup._publish_step", "_setup._publish_trajectory_for_verifier"):
        return "publish"
    # Since V7 the snapshot copy rides in front of the bash-env inspectors'
    # root exec, so that exec is the snapshot's (with U12, the manifest too).
    if innermost == "lockdown._snapshot_build_config" or (
        innermost == "lockdown._prepare_bash_env_inspector"
        and "benchflow_build_snapshot" in command
    ):
        return "snapshot"
    if innermost == "lockdown._seed_verifier_workspace":
        return "seed"
    if innermost in _READERS and any(
        read in command
        for read in (
            'printf "%s\\n" "${_BENCHFLOW_IMAGE_PATH-$PATH}"',
            "${PYTHONPATH-}",
            "cat /etc/os-release",
            "command -v python3",
        )
    ):
        return "environment_reads"
    if ("chmod +x" in command and "test.sh" in command) or (
        "mkdir -p /run/benchflow" in command and "verifier-started" not in command
    ):
        return "verifier_start"
    return "other"


async def test_one_fake_trial_with_merged_execs(
    fake_daytona_image, monkeypatch, tmp_path
) -> None:
    """Guards PY8 U12 at 71f4d05f, with its two findings fixed: 38 to 31
    execs (the start 3 to 1, snapshot 2 to 1, seed 3 to 1, environment reads
    3 to 1). The verifier start keeps its 2 execs and every command uses the
    asynchronous poll, with the same reward.

    Since the merge of integ/py-0.8 008da0e0 the trajectory is published
    through a root staging folder (V6), 5 execs there and 35 in
    all with U12's other merges; U12 folds the defensive /logs/agent mkdir
    into the stage's exec: 34. The other three stay their own execs, which
    folding would weaken: the chmod must follow the upload (a host call, so
    no exec can hold both), the copy runs as the sandbox user so a link the
    agent left at the name reaches only what the agent could write, and the
    stage's removal is root's after that copy. One exec holding the copy and
    a root step would need a root shell to drop to the sandbox user itself,
    which on Docker is not the backend's own switch (``docker exec -u``);
    see tests/test_publish_exec_fold.py.

    Since the merge of integ/py-0.8 363d815b: 36 (integ alone: 44). Integ
    added two root execs to this trial, both kept apart: AG's Muse login
    check and V7's read of the image's BASH_ENV (_image_bash_env), which
    runs in the verifier's service with that exec's own environment. V7's
    bash-env inspectors add none: they run after the snapshot's copy and
    manifest steps in that same root exec, before the agent starts. The
    publish stage's retry adds none when its first mkdir answers.

    On py8/parity-triage (C8): 39. The PATH lock runs as its own root exec
    on the system shell; the image's PYTHONPATH is read on the task route
    with a builtin, apart from the merged root reads (root's plumbing never
    has it); and the first exec as the sandbox user looks its uid up once
    (exec_route, inside the publish's copy). Round 8: 41, the
    protected runtime kit's build and seal, one root exec each.

    Since K1's merge of integ/py-0.8 7a2387ac: 37. The daemon log is made
    root's in a root exec of its own whose answer is checked (the image's
    default user may not be root, so it cannot ride in the start's
    default-user exec); it counts under "other".

    DAYTONA-EXEC (57d28d656): 46. This trial could not run on the fake from
    the time its commands passed one argument's limit (the fake's ``docker
    exec`` got the session command as one host argument: E2BIG, then the
    poll's timeout), so the count was not kept while verifier hardening
    gained its reads (mount provenance, excluded inodes, the index of
    exclusions, the tool-state isolation, the image's pytest.ini, the output
    folder's clearing, the guard's markers). Under the transport contract
    every session command fits its bound; the five over it (the kit's build
    and seal, the preparation script, plugin discovery, the guard's
    install) are each one exec still, their text uploaded as root's staged
    file: five uploads, no exec added. The image's PYTHONPATH is read in the
    hardening's own checked read now, not with the environment reads.

    Since 88ac33cde, the trusted verifier PATH verdict also exceeds the
    transport bound and is staged. Six payload uploads still use 46 execs;
    the command-size and one-upload-per-payload checks remain below.
    """
    result, counts = await _one_trial(monkeypatch, tmp_path, "u12")
    summary = counts.summary()
    summary["trial_record"] = counts.record
    groups: dict[str, int] = {}
    for e in counts.execs:
        group = _u12_group(e)
        groups[group] = groups.get(group, 0) + 1
    summary["u12_groups"] = groups
    report = os.environ.get(U12_REPORT_ENV)
    if report:
        Path(report).write_text(
            json.dumps(
                {
                    "summary": summary,
                    "execs": [
                        {
                            "caller": e.caller,
                            "group": _u12_group(e),
                            "sync": e.sync,
                            "exit_code": e.exit_code,
                            "status_requests": e.status_requests,
                            "logs": e.logs_at is not None,
                            "command": e.command[:160],
                        }
                        for e in counts.execs
                    ],
                },
                indent=1,
            )
        )
    print(json.dumps(summary, indent=1))
    assert result.error is None, result.error
    assert result.rewards == {"reward": 1.0}
    # 36 on integ/py-0.8 cfca0513; the mount provenance read before each
    # ownership freeze (_blocked_before_freeze, CONFIRM-lockdown-3 and -5)
    # runs twice in this trial: two execs more. K1's root-only daemon-log
    # exec of its own: one more.
    assert summary["exec_calls"] == 46, groups
    assert {g: groups.get(g, 0) for g in U12_GROUPS} == {
        "start": 1,
        "snapshot": 1,
        "seed": 1,
        "environment_reads": 1,
        "verifier_start": 2,
        "publish": 5,
    }, groups
    # The transport contract: every session command within its bound; the
    # staged ones are one exec each, their text one upload each.
    from benchflow.sandbox._transport import COMMAND_BOUND

    assert max(len(e.command.encode()) for e in counts.execs) <= COMMAND_BOUND
    # A staged command opens its file only after root's shell proved the
    # file's route (DAYTONA-GAPS rule 4): ``if ( <proof> ); then { ...``.
    staged = [e for e in counts.execs if e.command.startswith("if ( ")]
    assert len(staged) == 6, [e.caller for e in staged]
    assert any("_judge_trusted_entries" in e.caller for e in staged)
    assert summary["uploads"] == 4 + len(staged)
    sync = [e for e in counts.execs if e.sync]
    assert not sync
    assert all(e.status_requests >= 1 and e.logs_at is not None for e in counts.execs)
    assert counts.record["sync_exec_calls"] == 0
    assert all(e.exit_code is not None for e in counts.execs)
