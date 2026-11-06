"""Kept services go on just before the test command, with the tests closed (K2).

The kill before grading leaves the services a task keeps frozen, so nothing
the agent left moves while root prepares the verifier. The scoring path lets
them go on through the verifier's ``before_test`` hook: after the verifier's
code is uploaded, before its test command, with that folder made root's
alone first (a kept service reads no test).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow._utils import kill_record
from benchflow.rollout._setup import _verify_rollout
from benchflow.task.verifier import Verifier


def _verifier(tmp_path, order):
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir()
    from benchflow.task.paths import RolloutPaths

    rollout_paths = RolloutPaths(tmp_path)
    task = MagicMock()
    task.config.verifier.type = "test-script"
    task.config.verifier.service = "main"
    task.config.verifier.user = "root"
    task.config.verifier.env = None
    task.config.verifier.timeout_sec = 60
    task.paths.tests_dir = tmp_path / "tests"
    (tmp_path / "tests").mkdir()
    task.paths.test_path = tmp_path / "tests" / "test.sh"
    task.paths.uses_native_verifier_dir = False

    sandbox = MagicMock()
    sandbox.is_mounted = True

    async def upload_dir(**kwargs):
        order.append(("upload", kwargs["target_dir"]))

    async def fake_exec(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("command", "")
        if "test.sh" in cmd and "chmod" not in cmd:
            order.append(("test", cmd))
            rollout_paths.reward_text_path.write_text("1")
        return MagicMock(exit_code=0, returncode=0)

    sandbox.upload_dir = AsyncMock(side_effect=upload_dir)
    sandbox.upload_file = AsyncMock()
    sandbox.exec = AsyncMock(side_effect=fake_exec)
    return Verifier(task=task, rollout_paths=rollout_paths, sandbox=sandbox)


@pytest.mark.asyncio
async def test_before_test_runs_after_the_upload_and_before_the_test(tmp_path):
    order: list = []
    verifier = _verifier(tmp_path, order)

    async def release(code_dir: str) -> None:
        order.append(("release", code_dir))

    verifier.before_test = release
    await verifier._verify_test_script()
    kinds = [k for k, _ in order]
    assert kinds == ["upload", "release", "test"], order
    assert order[1][1] == order[0][1] == "/tests"


class _Planes:
    """Planes whose hardening reports one kept service, and a verifier that
    runs its hook as the real one does."""

    def __init__(self):
        self.released: list = []

    async def harden_before_verify(self, env, task, sandbox_user, **kwargs):
        kill_record.record_kill_before_grading(sandbox_user, 3)
        record = kill_record._current.get()
        assert record is not None
        record["kept_services"] = [
            {
                "port": 5000,
                "pid": 4242,
                "command": "python3 app.py",
                "source": "the task",
            }
        ]

    async def release_kept_services(self, env, sandbox_user, pids, close=()):
        self.released.append((sandbox_user, tuple(pids), tuple(close)))
        return len(pids)

    async def end_kept_services(self, env, sandbox_user, opened=()):
        order = getattr(self, "order", None)
        if order is not None:
            order.append("end")
        return 2

    def verifier(self, **kwargs):
        planes = self

        class _V:
            execution_receipt = None
            before_test = None

            async def verify(self):
                hook = getattr(self, "before_test", None)
                if hook is not None:
                    await hook("/tests")
                planes.order.append("test")
                return SimpleNamespace(rewards={"reward": 1.0})

        return _V()


@pytest.mark.asyncio
async def test_the_scoring_path_releases_the_kept_services_before_the_test(tmp_path):
    planes = _Planes()
    planes.order = []
    original = planes.release_kept_services

    async def release(*args, **kwargs):
        planes.order.append("release")
        return await original(*args, **kwargs)

    planes.release_kept_services = release
    task = SimpleNamespace(
        name="svc",
        task_dir=None,
        config=SimpleNamespace(
            verifier=SimpleNamespace(
                timeout_sec=30, type="test-script", reward_range=None
            )
        ),
    )
    rollout_paths = SimpleNamespace(
        verifier_dir=tmp_path / "verifier", rollout_dir=tmp_path
    )
    _rewards, error, _ = await _verify_rollout(
        object(),
        task,
        rollout_paths,
        {},
        planes,
        sandbox_user="agent",
        workspace="/app",
    )
    assert error is None, error
    # K2-R3-A-1: nothing kept lives past grading.
    assert planes.order == ["release", "test", "end"]
    assert planes.released == [("agent", (4242,), ("/tests",))]
    written = (tmp_path / "kill-before-grading.json").read_text()
    assert '"kept_services"' in written and '"pid": 4242' in written
    assert json.loads(written)["kept_services_ended"] == 2


@pytest.mark.asyncio
async def test_on_unmounted_logs_the_release_closes_what_the_clear_opened(tmp_path):
    """K2-R1-A-1: on a sandbox whose logs are not mounted (Daytona, remote
    Docker) the verifier clears its output folder again after the hardening
    closed it (which left it 0777 before K2-R4-A-1). The release runs after
    that clear, closes /logs again and only then lets the kept service go on,
    so the test command starts with /logs/verifier 0755 and /logs 0700."""
    from benchflow.sandbox.lockdown import release_kept_services
    from tests.test_keep_services_kill import _Async, _NetProc
    from tests.test_verifier_process_quiescence import _LIVE

    proc = _NetProc(tmp_path)
    pid = proc.add(_LIVE)
    proc.place(pid)
    proc.run(f"kill -s STOP {pid}")
    logs = tmp_path / "logs"
    (logs / "verifier").mkdir(parents=True)
    logs.chmod(0o700)
    (logs / "verifier").chmod(0o755)

    order: list = []
    verifier = _verifier(tmp_path, order)
    verifier._sandbox.is_mounted = False
    verifier._sandbox.download_dir = AsyncMock()
    test_exec = verifier._sandbox.exec.side_effect
    seen: dict = {}

    async def exec_(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("command", "")
        if "find /logs/verifier -mindepth 1" in cmd:
            # The clear, as root runs it in the sandbox.
            order.append(("clear", ""))
            opened = cmd.endswith("chmod 777 /logs/verifier")
            (logs / "verifier").chmod(0o777 if opened else 0o755)
            logs.chmod(0o755)
        if "test.sh" in cmd and "chmod" not in cmd:
            seen["logs"] = logs.stat().st_mode & 0o777
            seen["verifier"] = (logs / "verifier").stat().st_mode & 0o777
            seen["frozen"] = proc.stopped(pid)
        return await test_exec(*args, **kwargs)

    verifier._sandbox.exec = AsyncMock(side_effect=exec_)

    async def release(code_dir: str) -> None:
        order.append(("release", code_dir))
        # A stand-in for the code folder: the real one is not on this host.
        await release_kept_services(
            _Async(proc), "agent", (pid,), (str(tmp_path / "code"),), logs_dir=str(logs)
        )

    verifier.before_test = release
    await verifier._verify_test_script()
    kinds = [k for k, _ in order]
    assert kinds == ["clear", "upload", "release", "test"], order
    assert seen == {"logs": 0o700, "verifier": 0o755, "frozen": False}


@pytest.mark.asyncio
async def test_a_criteria_runner_runner_starts_after_the_release(tmp_path):
    """K2-R1-A-2: a criteria-runner runner is a command in the sandbox too, so the
    kept services go on (with the code folder closed) before it starts."""
    from tests.test_verifier_strategies import _criteria_runner_verifier, _criteria_toml

    verifier, _ = _criteria_runner_verifier(
        tmp_path,
        criteria_toml=_criteria_toml("weighted_mean", {"c1": 1.0}),
        reward_payload={"metrics": {"c1": 1.0}},
    )
    verifier._task.config.verifier.user = "root"  # the file is root's to read
    order: list = []
    runner = verifier._sandbox.exec.side_effect

    async def exec_(*args, **kwargs):
        if "criteria_runner/reward.py" in (kwargs.get("command") or ""):
            order.append(("runner", ""))
        return await runner(*args, **kwargs)

    verifier._sandbox.exec = AsyncMock(side_effect=exec_)
    uploads: list = []

    async def upload_file(source_path, target_path, *, mode=None, **_kwargs):
        uploads.append((target_path, mode, Path(source_path).read_text()))

    verifier._sandbox.upload_file = AsyncMock(side_effect=upload_file)

    async def release(code_dir: str) -> None:
        order.append(("release", code_dir))

    verifier.before_test = release
    result = await verifier.verify()
    assert result.rewards["reward"] == 1.0
    assert order == [("release", "/verifier"), ("runner", "")]
    # K2-R4-B-1: the runner's env (BENCHFLOW_* and [verifier.env]) comes from
    # a root-only file, not the exec's env (the shell's command line).
    ((target, mode, body),) = uploads
    assert mode == "600" and "export BENCHFLOW_REWARD_JSON=" in body
    (runner,) = [
        c
        for c in verifier._sandbox.exec.await_args_list
        if "criteria_runner/reward.py" in (c.kwargs.get("command") or "")
    ]
    assert runner.kwargs["env"] is None
    assert f"if . {target}; then /bin/rm -rf " in runner.kwargs["command"]


@pytest.mark.asyncio
async def test_a_rubric_with_no_script_releases_before_the_judges(
    tmp_path, monkeypatch
):
    """K2-R1-A-2: a task.md rubric with no test command runs its judges'
    commands in the sandbox; the kept services go on before them, after the
    copy of the files the judges read."""
    from benchflow.task.formats import materialize_task_dir
    from benchflow.taskmd import verify
    from tests._taskmd_helpers import EXAMPLES
    from tests.test_taskmd_verify import _Sandbox, _strategy, _Verifier

    native = materialize_task_dir(EXAMPLES / "analysis-judge")
    container = tmp_path / "container"
    (container / "work").mkdir(parents=True)
    trial = tmp_path / "trial"
    sandbox = _Sandbox(container)
    verifier = _Verifier(native, trial, sandbox, report=None)
    order: list = []

    async def release(code_dir: str) -> None:
        order.append(("release", any("then echo link" in c for c in sandbox.commands)))

    verifier.before_test = release

    class _Stop(Exception):
        pass

    async def judges(**kwargs):
        order.append(("judges", None))
        raise _Stop

    monkeypatch.setattr(verify, "_run_judges", judges)
    strategy = _strategy(native)
    config = {k: v for k, v in dict(strategy.config).items() if k != "command"}
    assert config.get("grading") == "rubric"
    with pytest.raises(_Stop):
        await verify.verify_taskmd(verifier, SimpleNamespace(config=config))
    assert order == [("release", True), ("judges", None)]
    assert verifier.calls == []


@pytest.mark.asyncio
async def test_a_dropped_kept_service_is_recorded(tmp_path):
    """K2-R1-B-1: a kept process the release killed instead is listed in
    kill-before-grading.json with why."""
    from benchflow.sandbox.lockdown import KeptRelease

    planes = _Planes()
    planes.order = []

    async def release(*args, **kwargs):
        planes.order.append("release")
        return KeptRelease(released=0, dropped=((4242, "holds a folder"),))

    planes.release_kept_services = release
    task = SimpleNamespace(
        name="svc",
        task_dir=None,
        config=SimpleNamespace(
            verifier=SimpleNamespace(
                timeout_sec=30, type="test-script", reward_range=None
            )
        ),
    )
    rollout_paths = SimpleNamespace(
        verifier_dir=tmp_path / "verifier", rollout_dir=tmp_path
    )
    _rewards, error, _ = await _verify_rollout(
        object(),
        task,
        rollout_paths,
        {},
        planes,
        sandbox_user="agent",
        workspace="/app",
    )
    assert error is None, error
    written = json.loads((tmp_path / "kill-before-grading.json").read_text())
    assert written["kept_services_dropped"] == [{"pid": 4242, "why": "holds a folder"}]


@pytest.mark.asyncio
async def test_lost_and_unfrozen_kept_services_are_recorded_at_the_first_release(
    tmp_path,
):
    """K2p-R1-A-1: the first release of a trial adds the kept pids that had
    exited (``kept_services_lost``) and those no longer frozen
    (``kept_services_unfrozen``, with the state) to kill-before-grading.json;
    a verifier retried in place releases again and finds the services
    running by then, which is not recorded."""
    from benchflow.rollout._setup import _release_kept_services
    from benchflow.sandbox.lockdown import KeptRelease

    outcomes = [
        KeptRelease(released=1, lost=(4243,), unfrozen=((4242, "S"),)),
        KeptRelease(released=1, unfrozen=((4242, "S"), (4244, "R"))),
    ]

    class _Releasing:
        async def release_kept_services(self, env, sandbox_user, pids, close=()):
            return outcomes.pop(0)

    record: dict = {"kept_services": []}
    rollout_paths = SimpleNamespace(rollout_dir=tmp_path)
    for _ in range(2):
        await _release_kept_services(
            _Releasing(),
            object(),
            "agent",
            (4242, 4243),
            record,
            rollout_paths.rollout_dir,
            "/t",
        )
    written = json.loads((tmp_path / "kill-before-grading.json").read_text())
    assert written["kept_services_lost"] == [4243]
    assert written["kept_services_unfrozen"] == [{"pid": 4242, "state": "S"}]
    assert written["kept_services_released"] == 1


@pytest.mark.asyncio
async def test_a_verifier_retried_in_place_keeps_the_output_folder_closed(
    tmp_path, monkeypatch
):
    """K2-R4-A-1: the first test command never starts (no start receipt), so
    the verifier runs again in place, with the kept services running since
    the first release. Its clear must not open /logs/verifier to them: a
    service holding the folder through a descriptor in flight would create
    reward.txt of its own there and write into it after test.sh did."""
    import asyncio
    import shutil

    from benchflow.rollout import _setup

    monkeypatch.setattr(_setup, "_VERIFIER_START_GRACE_SEC", 0.2)
    monkeypatch.setattr(_setup, "_VERIFIER_RECEIPT_POLL_SEC", 0.01)

    async def never_started(env, verifier):
        return False

    monkeypatch.setattr(_setup, "_verifier_started", never_started)
    logs = tmp_path / "logs"
    out = logs / "verifier"
    out.mkdir(parents=True)
    out.chmod(0o755)

    order: list = []
    verifier = _verifier(tmp_path, order)
    verifier._write_execution_receipt = True
    verifier._sandbox.is_mounted = False
    verifier._sandbox.download_dir = AsyncMock()
    test_exec = verifier._sandbox.exec.side_effect
    running = {"services": False, "tests": 0}
    clears: list = []
    planted: list = []

    async def exec_(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("command", "")
        if "find /logs/verifier -mindepth 1" in cmd:
            # The clear, as root runs it in the sandbox: empty the folder,
            # then leave it with the mode the command sets.
            for child in out.iterdir():
                shutil.rmtree(child) if child.is_dir() else child.unlink()
            mode = 0o777 if cmd.endswith("chmod 777 /logs/verifier") else 0o755
            assert mode == 0o777 or cmd.endswith(
                "chown root:root /logs/verifier && chmod 755 /logs/verifier"
            ), cmd
            out.chmod(mode)
            clears.append(mode)
            if running["services"] and mode & 0o002:
                # A kept service holding the folder through a descriptor in
                # flight: the folder is open to it, so it makes its own file.
                (out / "reward.txt").write_text("1\n")
                planted.append("reward.txt")
        if "test.sh" in cmd and "chmod" not in cmd:
            running["tests"] += 1
            if running["tests"] == 1:
                await asyncio.sleep(30)  # lost by the exec layer
        return await test_exec(*args, **kwargs)

    verifier._sandbox.exec = AsyncMock(side_effect=exec_)

    async def release(code_dir: str) -> None:
        order.append(("release", code_dir))
        running["services"] = True

    verifier.before_test = release
    verifier.verify = verifier._verify_test_script
    await _setup._await_verifier(MagicMock(exec=AsyncMock()), verifier, 30)
    assert running["tests"] == 2
    assert [k for k, _ in order].count("release") == 2, order
    assert clears == [0o755, 0o755]
    assert planted == []


_FAKE_KEY = "bf-fake-judge-key-not-a-secret-7f3a91"


def _shows_key(text: str) -> bool:
    """Whether ``text`` (a command line) holds the fake key, as is or inside
    a base64 run (how every backend's exec carries an env)."""
    import base64
    import re

    if _FAKE_KEY in text:
        return True
    for run in re.findall(r"[A-Za-z0-9+/=]{16,}", text):
        try:
            decoded = base64.b64decode(run + "=" * (-len(run) % 4)).decode(
                errors="replace"
            )
        except ValueError:
            continue
        if _FAKE_KEY in decoded:
            return True
    return False


@pytest.mark.asyncio
async def test_kept_services_never_see_the_verifier_env_on_a_command_line(tmp_path):
    """K2-R4-B-1: while services are kept, the verifier's env (an LLM judge's
    key in [verifier.env]) reaches the test command through a file only root
    can read, uploaded before the release, never through an exec's env
    (which every backend puts, base64-encoded, on the shell's command line,
    readable by agent code in /proc/<pid>/cmdline)."""
    order: list = []
    verifier = _verifier(tmp_path, order)
    verifier._task.config.verifier.env = {"JUDGE_API_KEY": _FAKE_KEY}
    test_exec = verifier._sandbox.exec.side_effect
    calls: list = []
    uploads: list = []

    async def exec_(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("command", "")
        calls.append((cmd, kwargs.get("env"), kwargs.get("user")))
        return await test_exec(*args, **kwargs)

    async def upload_file(source_path, target_path, *, mode=None, **_kwargs):
        uploads.append((target_path, mode, Path(source_path).read_text()))
        order.append(("upload_env", target_path))

    verifier._sandbox.exec = AsyncMock(side_effect=exec_)
    verifier._sandbox.upload_file = AsyncMock(side_effect=upload_file)

    async def release(code_dir: str) -> None:
        order.append(("release", code_dir))

    verifier.before_test = release
    await verifier._verify_test_script()

    for cmd, env, _user in calls:
        assert not _shows_key(cmd), cmd
        assert not env or _FAKE_KEY not in env.values(), env
    ((target, mode, body),) = uploads
    assert mode == "600" and f"export JUDGE_API_KEY={_FAKE_KEY}\n" in body
    folder = target.rsplit("/", 1)[0]
    # K2-R6-A-1: never under /tmp, whose folders anyone may rename when its
    # sticky bit is gone; under a parent only root can enter, set up and
    # checked first.
    assert folder.startswith("/var/lib/runtime/verifier-env/.runtime-verifier-env-")
    cmds = [c for c, _, _ in calls]
    made = [c for c in cmds if f"mkdir {folder}" in c]
    assert made == [f"umask 077 && mkdir {folder}"], made
    from benchflow.task.verifier_core import _VERIFIER_ENV_PARENT_CMD

    parent = cmds.index(_VERIFIER_ENV_PARENT_CMD)
    assert parent < cmds.index(made[0])
    assert calls[parent][2] == "root"
    assert "for d in / /var /var/lib;" in _VERIFIER_ENV_PARENT_CMD
    assert "/tmp" not in _VERIFIER_ENV_PARENT_CMD
    assert [k for k, _ in order] == ["upload", "upload_env", "release", "test"]
    test_cmd = order[-1][1]
    from benchflow.task.verifier_core import _root_file_env_prefix

    assert test_cmd.startswith(_root_file_env_prefix(folder)), test_cmd
    # An EXIT trap would replace the exec wrapper's, which removes the
    # sandbox's own env file (K2-R5-A-3).
    assert "trap" not in test_cmd.split("\n", 1)[0], test_cmd


@pytest.mark.asyncio
async def test_the_verifier_env_never_goes_into_a_folder_that_already_exists(
    tmp_path, monkeypatch
):
    """K2-R5-A-1: on a verifier retried in place the kept services run, read
    the env step's command line in /proc, and can make the folder it names
    (with an open file inside) before root's mkdir runs. The step must then
    fail and pick a new name, never accept and close a folder made first."""
    from benchflow.task import verifier_core

    monkeypatch.setattr(verifier_core, "_SETUP_RETRY_DELAYS_SEC", (0.0, 0.0))
    order: list = []
    verifier = _verifier(tmp_path, order)
    verifier._task.config.verifier.env = {"JUDGE_API_KEY": _FAKE_KEY}
    test_exec = verifier._sandbox.exec.side_effect
    planted: set[str] = set()
    made: list[str] = []
    uploads: list[str] = []

    async def exec_(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("command", "")
        if "mkdir" in cmd and ".runtime-verifier-env-" in cmd:
            folder = cmd.split()[-1]
            made.append(folder)
            if not planted:
                # A kept service wins the race for the first name it sees.
                planted.add(folder)
            exists = folder in planted
            plain = " -p " not in f" {cmd} "
            return MagicMock(exit_code=1 if exists and plain else 0, returncode=0)
        return await test_exec(*args, **kwargs)

    async def upload_file(source_path, target_path, *, mode=None, **_kwargs):
        uploads.append(target_path)

    verifier._sandbox.exec = AsyncMock(side_effect=exec_)
    verifier._sandbox.upload_file = AsyncMock(side_effect=upload_file)

    async def release(code_dir: str) -> None:
        order.append(("release", code_dir))

    verifier.before_test = release
    await verifier._verify_test_script()

    assert len(made) == 2 and made[0] != made[1], made
    (target,) = uploads
    folder = target.rsplit("/", 1)[0]
    assert folder not in planted, (folder, planted)
    assert folder == made[1]
    from benchflow.task.verifier_core import _root_file_env_prefix

    assert order[-1][1].startswith(_root_file_env_prefix(folder))


@pytest.mark.asyncio
async def test_with_nothing_kept_the_verifier_env_goes_as_before(tmp_path):
    """Without kept services no agent process is alive during grading, so the
    verifier's env still goes through exec's env (K2-R4-B-1)."""
    order: list = []
    verifier = _verifier(tmp_path, order)
    verifier._task.config.verifier.env = {"JUDGE_API_KEY": _FAKE_KEY}
    envs: list = []
    test_exec = verifier._sandbox.exec.side_effect

    async def exec_(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("command", "")
        if "test.sh" in cmd and "chmod" not in cmd:
            envs.append(kwargs.get("env"))
        return await test_exec(*args, **kwargs)

    verifier._sandbox.exec = AsyncMock(side_effect=exec_)
    verifier._sandbox.upload_file = AsyncMock()
    await verifier._verify_test_script()
    # The pins ride in the command, not the env (b127).
    assert envs == [{"JUDGE_API_KEY": _FAKE_KEY}]
    verifier._sandbox.upload_file.assert_not_awaited()
