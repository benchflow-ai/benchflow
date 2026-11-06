"""The rules of U12's merged execs and kept answers (PY8 U12).

``tests/test_exec_diet_goldens.py`` shows the results are the base's;
``tests/test_exec_diet_counts.py`` counts a trial's execs. These pin the
rules in between, with sandboxes that answer from a script
(``ScriptedSandbox``): when the default user's ``pwd`` and uid from the start
are used and when they are not (explicit users, ``/``, another generation, a
failed or garbled probe), what each merged exec does when it gets no answer,
and the Daytona synchronous path's edge cases (missing exit data, a transport
error, a command that overran, cancellation, no replay).
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from benchflow.rollout._setup import _resolve_agent_cwd
from benchflow.sandbox import _exec_merge, lockdown
from benchflow.sandbox._exec_merge import (
    MergedStep,
    merged_budget,
    merged_script,
    new_tag,
    parse_merged,
)
from benchflow.sandbox.docker import DockerSandbox
from tests._receipt_fakes import inner_command
from tests.exec_diet_fakes import (
    MERGED_MARK,
    UBUNTU,
    BoxSandbox,
    ScriptedSandbox,
    box_factory,  # noqa: F401  (fixture)
    merged_steps,
)

# --- the script and its parser ------------------------------------------------


def test_parse_splits_each_step_with_its_status():
    tag = new_tag()
    stdout = f"{tag}:0:begin\na\nb\n\n{tag}:0:end:0\n{tag}:1:begin\n\n{tag}:1:end:3\n"
    answers = parse_merged(stdout, tag, 2)
    assert answers[0].output == "a\nb\n"
    assert answers[0].return_code == 0
    assert (answers[1].output, answers[1].return_code) == ("", 3)


@pytest.mark.parametrize(
    "stdout",
    [
        "{t}:0:begin\nx\n{t}:0:end:0\n{t}:0:begin\ny\n{t}:0:end:0\n",  # repeated
        "{t}:0:begin\nx {t} y\n{t}:0:end:0\n",  # tag inside an answer
        "{t}:0:begin\nx\n{t}:0:end:zero\n",  # garbled status
        "{t}:0:begin\nx\n",  # no end
        "{t}:1:begin\nx\n{t}:1:end:0\n{t}:0:begin\nx\n{t}:0:end:0\n",  # order
        "junk{t}:0:begin\nx\n{t}:0:end:0\n",  # not at a line start
    ],
)
def test_a_malformed_answer_gives_nothing(stdout):
    tag = new_tag()
    assert parse_merged(stdout.format(t=tag), tag, 2) == {}


def test_a_step_that_did_not_run_is_missing():
    tag = new_tag()
    answers = parse_merged(f"{tag}:0:begin\n\n{tag}:0:end:1\n", tag, 2)
    assert list(answers) == [0]
    assert parse_merged(None, tag, 1) == {}
    assert parse_merged(object(), tag, 1) == {}


def test_the_budget_is_the_smallest_step_budget():
    assert merged_budget(None, 10, 180) == 10
    assert merged_budget(60, 10) == 10
    assert merged_budget(None, None) is None
    assert merged_budget(0, -1) is None


def test_each_tag_is_new():
    assert new_tag() != new_tag()
    assert new_tag().startswith(MERGED_MARK)


def test_the_script_runs_in_sh_with_each_status(tmp_path):
    """The real shell: each step's own status, an exit in a step ends only
    that step, a failed gate stops the script, a captured step's output is
    readable by the next."""
    import subprocess

    tag = new_tag()
    script = merged_script(
        [
            "echo one; exit 4",
            MergedStep("echo two", capture=True),
            'echo "saw $__bf_out1"',
            MergedStep("false", gate=True),
            "echo never",
        ],
        tag,
    )
    out = subprocess.run(
        ["/bin/sh", "-c", script], capture_output=True, text=True, cwd=tmp_path
    ).stdout
    answers = parse_merged(out, tag, 5)
    assert [(a.output, a.return_code) for a in answers.values()] == [
        ("one\n", 4),
        ("two", 0),
        ("saw two\n", 0),
        ("", 1),
    ]


# --- the kept pwd and uid -------------------------------------------------------


class _Probed(ScriptedSandbox):
    """A sandbox with Docker's start (the log folders' chmod and the probe)."""

    _prepare_log_dirs_after_up = DockerSandbox._prepare_log_dirs_after_up

    async def _probe_verifier_log_mount(self) -> None:
        return None


def _answers(pwd="/work\n", uid="1000\n", pwd_rc=0, uid_rc=0):
    def handler(command, user):
        if command == "pwd":
            return pwd_rc, pwd
        if command == "id -u":
            return uid_rc, uid if user is None else "0\n"
        return 0, ""

    return handler


def _task(workdir=None):
    return SimpleNamespace(
        config=SimpleNamespace(sandbox=SimpleNamespace(workdir=workdir))
    )


async def _start(sandbox):
    sandbox._forget_kept_identity()
    await sandbox._prepare_log_dirs_after_up()


async def test_the_kept_pwd_and_uid_are_what_the_execs_answer():
    sandbox = _Probed(_answers())
    await _start(sandbox)
    assert len(sandbox.execs) == 1
    assert sandbox.execs[0].user is None
    assert sandbox.execs[0].timeout_sec == 10
    assert await _resolve_agent_cwd(sandbox, _task()) == "/work"
    assert await lockdown._verifier_uid(sandbox, None) == "1000"
    assert len(sandbox.execs) == 1
    # Without the probe the same readers answer the same, with their execs.
    plain = _Probed(_answers())
    assert await _resolve_agent_cwd(plain, _task()) == "/work"
    assert await lockdown._verifier_uid(plain, None) == "1000"
    assert [inner_command(e.command) for e in plain.execs] == ["pwd", "id -u"]


class _ProbedBox(BoxSandbox):
    """A real container with Docker's start probe."""

    _prepare_log_dirs_after_up = DockerSandbox._prepare_log_dirs_after_up

    async def _probe_verifier_log_mount(self) -> None:
        return None


@pytest.mark.deterministic
async def test_the_kept_answers_are_a_real_containers(box_factory):  # noqa: F811
    """A container whose default user is uid 1000 with WORKDIR /work: what
    the start probe keeps is what the readers' own execs answer there."""
    box = box_factory(UBUNTU, "--user", "1000:1000", "-w", "/work")
    probed = _ProbedBox(box)
    probed._forget_kept_identity()
    await probed._prepare_log_dirs_after_up()
    kept = probed.kept_identity()
    assert kept is not None
    assert (kept.pwd, kept.uid) == ("/work\n", "1000\n")
    assert await _resolve_agent_cwd(probed, _task()) == "/work"
    assert await lockdown._verifier_uid(probed, None) == "1000"
    assert len(probed.execs) == 1
    plain = BoxSandbox(box)
    assert await _resolve_agent_cwd(plain, _task()) == "/work"
    assert await lockdown._verifier_uid(plain, None) == "1000"
    assert [inner_command(e.command) for e in plain.execs] == ["pwd", "id -u"]
    # root, asked for by name, is still asked.
    assert await lockdown._verifier_uid(probed, "root") == "0"


@pytest.mark.parametrize("user", ["root", 0, "0", "agent"])
async def test_an_explicit_verifier_user_runs_its_own_id(user):
    sandbox = _Probed(_answers())
    await _start(sandbox)
    assert await lockdown._verifier_uid(sandbox, user) == "0"
    assert [(e.command, e.user) for e in sandbox.execs[1:]] == [("id -u", user)]


async def test_a_root_workdir_takes_the_fallback():
    sandbox = _Probed(_answers(pwd="/\n"))
    await _start(sandbox)
    assert await _resolve_agent_cwd(sandbox, _task()) in ("", "/root")
    assert [e.command for e in sandbox.execs[1:]] == [
        "mkdir -p /root && cd /root && pwd"
    ]


async def test_a_configured_workdir_reads_no_pwd():
    sandbox = _Probed(_answers())
    await _start(sandbox)
    await _resolve_agent_cwd(sandbox, _task("/srv/app"))
    assert [e.command for e in sandbox.execs[1:]] == [
        "mkdir -p /srv/app && cd /srv/app && pwd"
    ]


@pytest.mark.parametrize(
    ("answers", "reads"),
    [
        (_answers(pwd_rc=1), ["pwd"]),
        (_answers(uid="nobody\n"), ["id -u"]),
        (_answers(uid_rc=1), ["id -u"]),
    ],
)
async def test_a_failed_step_is_not_kept(answers, reads):
    sandbox = _Probed(answers)
    await _start(sandbox)
    await _resolve_agent_cwd(sandbox, _task())
    await lockdown._verifier_uid(sandbox, None)
    assert [e.command for e in sandbox.execs[1:]] == reads


_CHMOD_ERR = "chmod: changing permissions of '/logs/agent': Operation not permitted\n"


class _Interleaved(_Probed):
    """Docker's merged stream with stderr text that reached the host between
    a step's markers (stdout and stderr are separate pipes with no order
    between them, PY8 U12-R1-B-1): *into* -> (step, text, before the body)."""

    def __init__(self, handler, into):
        super().__init__(handler)
        self.into = into

    async def exec(self, command, **kwargs):
        result = await super().exec(command, **kwargs)
        if MERGED_MARK not in command:
            return result
        stdout = result.stdout
        tag, _ = merged_steps(command)
        for step, (text, before) in self.into.items():
            if before:
                begin = f"{tag}:{step}:begin\n"
                at = stdout.index(begin) + len(begin)
            else:
                at = stdout.index(f"\n{tag}:{step}:end:")
            stdout = stdout[:at] + text + stdout[at:]
        return result.model_copy(update={"stdout": stdout})


@pytest.mark.parametrize(
    "into",
    [
        {1: (_CHMOD_ERR, True)},  # the chmod's message, then /work
        {1: (_CHMOD_ERR, False)},  # /work, then the message
        {1: ("work", True)},  # a fragment glued to the front: not a path
        {1: (_CHMOD_ERR, True), 2: (_CHMOD_ERR, True)},
    ],
)
async def test_stderr_inside_the_pwd_answer_is_not_kept(into):
    """Text from another stream inside ``pwd``'s markers is not a working
    directory: nothing is kept and ``_resolve_agent_cwd`` runs its own
    ``pwd``, as the base did (PY8 U12-R1-B-1)."""
    sandbox = _Interleaved(_answers(), into)
    await _start(sandbox)
    kept = sandbox.kept_identity()
    assert kept is None or kept.pwd is None
    assert await _resolve_agent_cwd(sandbox, _task()) == "/work"
    assert await lockdown._verifier_uid(sandbox, None) == "1000"
    reads = ["pwd"] + (["id -u"] if 2 in into else [])
    assert [e.command for e in sandbox.execs[1:]] == reads


def test_the_start_commands_stderr_stays_in_its_own_answer():
    """With stdout and stderr on separate pipes (Docker without a TTY), the
    start command's stderr is in its own answer, never another step's."""
    import subprocess

    tag = new_tag()
    script = merged_script(
        [
            MergedStep("echo out; echo err >&2; exit 1", stderr_in_answer=True),
            "echo /work",
        ],
        tag,
    )
    run = subprocess.run(["/bin/sh", "-c", script], capture_output=True, text=True)
    assert run.stderr == ""
    answers = parse_merged(run.stdout, tag, 2)
    assert (answers[0].output, answers[0].return_code) == ("out\nerr\n", 1)
    assert (answers[1].output, answers[1].return_code) == ("/work\n", 0)
    with pytest.raises(ValueError):
        MergedStep("pwd", capture=True, stderr_in_answer=True)


async def test_the_start_probe_keeps_its_commands_stderr_in_its_answer():
    sandbox = _Probed(_answers())
    await _start(sandbox)
    assert "\nchmod 777 /logs/agent /logs/verifier\n) 2>&1\n" in (
        sandbox.execs[0].command
    )


async def test_a_garbled_probe_runs_the_chmod_again_and_keeps_nothing():
    sandbox = _Probed(_answers(), garble=True)
    await _start(sandbox)
    assert sandbox.execs[1].command == "chmod 777 /logs/agent /logs/verifier"
    assert sandbox.execs[1].timeout_sec is None
    await _resolve_agent_cwd(sandbox, _task())
    await lockdown._verifier_uid(sandbox, None)
    assert [e.command for e in sandbox.execs[2:]] == ["pwd", "id -u"]


async def test_a_probe_that_raises_runs_the_chmod_alone():
    def handler(command, user):
        if command == "chmod 777 /logs/agent /logs/verifier":
            return 0, ""
        return _answers()(command, user)

    class _Raising(_Probed):
        async def exec(self, command, **kwargs):
            if MERGED_MARK in command:
                self.execs.append(None)
                raise RuntimeError("Command timed out after 10 seconds")
            return await super().exec(command, **kwargs)

    sandbox = _Raising(handler)
    await _start(sandbox)
    assert sandbox.kept_identity() is None
    assert sandbox.execs[1].command == "chmod 777 /logs/agent /logs/verifier"


async def test_the_chmod_alone_still_raises_as_before():
    class _Dead(_Probed):
        async def exec(self, command, **kwargs):
            raise RuntimeError("container is gone")

    with pytest.raises(RuntimeError, match="container is gone"):
        await _start(_Dead(_answers()))


async def test_another_generation_or_identity_runs_its_own_reads():
    sandbox = _Probed(_answers())
    await _start(sandbox)
    sandbox._forget_kept_identity()  # a stop, a restart or a restore
    await _resolve_agent_cwd(sandbox, _task())
    assert sandbox.execs[-1].command == "pwd"

    sandbox = _Probed(_answers())
    await _start(sandbox)
    sandbox._persistent_env = {"PWD": "/elsewhere"}  # not the probe's env
    await _resolve_agent_cwd(sandbox, _task())
    assert sandbox.execs[-1].command == "pwd"

    sandbox = _Probed(_answers())
    await _start(sandbox)
    sandbox.default_user = "agent"  # not the probe's user
    await lockdown._verifier_uid(sandbox, None)
    assert (sandbox.execs[-1].command, sandbox.execs[-1].user) == ("id -u", None)


async def test_a_sandbox_without_the_hook_has_nothing_kept():
    assert _exec_merge.kept_default_identity(SimpleNamespace()) is None
    from unittest.mock import MagicMock

    assert _exec_merge.kept_default_identity(MagicMock()) is None


async def test_docker_lifecycle_drops_what_was_kept(monkeypatch):
    from benchflow.sandbox import docker as docker_module
    from benchflow.sandbox.protocol import (
        SandboxImage,
        SandboxSnapshotNotSupported,
    )

    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox._keep_identity(user=None, env=None, pwd="/w\n", uid="0\n")
    assert sandbox.kept_identity() is not None
    with pytest.raises(SandboxSnapshotNotSupported):
        await sandbox.restore(SandboxImage(provider="daytona", ref="x"))
    assert sandbox.kept_identity() is None

    sandbox._keep_identity(user=None, env=None, pwd="/w\n", uid="0\n")

    def no_project(name):
        raise RuntimeError("start stopped here")

    monkeypatch.setattr(docker_module, "claim_project", no_project)
    monkeypatch.setattr(
        DockerSandbox, "compose_project_name", property(lambda self: "p")
    )
    with pytest.raises(RuntimeError, match="start stopped here"):
        await sandbox.start(force_build=False)
    assert sandbox.kept_identity() is None  # a failed start keeps nothing

    sandbox._keep_identity(user=None, env=None, pwd="/w\n", uid="0\n")
    sandbox._compose_up_attempted = False
    sandbox._keep_containers = True
    sandbox._snapshot_credentials = {}
    sandbox._deferred_snapshot_refs = set()

    async def compose(*args, **kwargs):
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    sandbox._run_docker_compose_command = compose
    await sandbox.stop(delete=False)
    assert sandbox.kept_identity() is None


async def test_daytona_lifecycle_drops_what_was_kept():
    pytest.importorskip("daytona")
    from benchflow.sandbox.daytona import DaytonaSandbox

    class _Strategy:
        async def start(self, force_build):
            assert sandbox.kept_identity() is None

        async def stop(self, delete):
            return None

        async def restore(self, image):
            return None

    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox._strategy = _Strategy()
    sandbox.rollout_paths = None
    for step in (
        lambda: sandbox.start(force_build=False),
        lambda: sandbox.stop(delete=True),
        lambda: sandbox.restore(SimpleNamespace(provider="daytona", ref="x", meta={})),
    ):
        sandbox._keep_identity(user=None, env=None, pwd="/w\n", uid="0\n")
        await step()
        assert sandbox.kept_identity() is None


# --- the snapshot: what the host does without a readable answer ------------------


async def test_the_snapshot_copy_without_an_answer_runs_both_execs_again():
    def handler(command, user):
        if "cp --preserve=all" in command:
            return 0, "setup.py=present\n"
        return 0, ""

    sandbox = ScriptedSandbox(handler, garble=True)
    await lockdown._snapshot_build_config(sandbox, "/app")
    assert len(sandbox.execs) == 3
    assert sandbox.execs[1].command.startswith(f"mkdir -p {lockdown._SNAPSHOT_DIR}")
    write = sandbox.execs[2].command
    assert write.startswith('echo \'{"setup.py": true, "pyproject.toml": false')
    assert all(e.user == "root" for e in sandbox.execs)


@pytest.mark.parametrize(("status", "written"), [(1, ""), (0, '{"setup.py": true}')])
async def test_a_wrong_or_failed_manifest_write_is_written_by_the_host(status, written):
    def handler(command, user):
        if "cp --preserve=all" in command:
            return 0, "setup.py=present\nMakefile=absent\n"
        return status, written

    sandbox = ScriptedSandbox(handler)
    await lockdown._snapshot_build_config(sandbox, "/app")
    assert len(sandbox.execs) == 2
    assert '"setup.py": true' in sandbox.execs[1].command
    assert '"Makefile": false' in sandbox.execs[1].command


async def test_the_seed_is_one_root_exec_and_runs_every_command():
    seen: list[str] = []

    def handler(command, user):
        seen.append(command)
        return 1, "failed"

    sandbox = ScriptedSandbox(handler)
    await lockdown._seed_verifier_workspace(sandbox, "/app", "agent")
    assert [(e.user, e.timeout_sec) for e in sandbox.execs] == [("root", None)]
    assert len(seen) == 3


# --- the verifier's environment reads -------------------------------------------


async def test_the_reads_are_one_root_exec_and_are_used_only_inside_it():
    def handler(command, user):
        return {
            lockdown._IMAGE_PATH_CMD: (0, "/usr/bin:/bin\n"),
            lockdown._PRINTENV_PYTHONPATH_CMD: (0, "\n"),
            "cat /etc/os-release 2>/dev/null || true": (0, "ID=fedora\n"),
        }.get(command, (1, ""))

    sandbox = ScriptedSandbox(handler)
    answers = await lockdown._probe_verifier_reads(sandbox)
    # PYTHONPATH is read on the task route (C8), not in this root exec.
    assert set(answers) == {"path", "os_release", "python3"}
    assert answers["python3"].return_code == 1
    assert [(e.user, e.timeout_sec) for e in sandbox.execs] == [("root", 180)]
    # Outside _build_verifier_env, each reader runs its own exec.
    assert await lockdown._distro_pip_env(sandbox) == {"PIP_PREFIX": "/usr/local"}
    assert len(sandbox.execs) == 2
    token = lockdown._verifier_reads.set((sandbox, answers))
    try:
        assert await lockdown._python3_absent(sandbox) is True
        assert await lockdown._distro_pip_env(sandbox) == {"PIP_PREFIX": "/usr/local"}
        # Another sandbox does not take this one's answers.
        other = ScriptedSandbox(handler)
        assert await lockdown._python3_absent(other) is True
        assert len(other.execs) == 1
    finally:
        lockdown._verifier_reads.reset(token)
    assert len(sandbox.execs) == 2


# --- the verifier start -------------------------------------------------------------


async def test_the_receipt_folder_waits_for_chmod():
    seen: list[str] = []

    def handler(command, user):
        seen.append(command)
        return (1, "") if command.startswith("chmod +x") else (0, "")

    from unittest.mock import MagicMock

    from benchflow.task.verifier_core import Verifier

    verifier = Verifier(MagicMock(), MagicMock(), ScriptedSandbox(handler))
    verifier._write_execution_receipt = True

    async def no_wait(what, run):
        result = await run()
        assert result.return_code == 1
        raise RuntimeError("stop")

    verifier._setup_step = no_wait
    with pytest.raises(RuntimeError, match="stop"):
        await verifier._chmod_and_prepare_receipt("chmod +x /tests/test.sh", "main")
    # Chmod failed: its separate receipt step never ran.
    assert seen == ["chmod +x /tests/test.sh"]
    assert verifier._sandbox.execs[0].timeout_sec == 60


async def test_a_chmod_that_fails_every_try_leaves_the_receipt_as_it_was(
    monkeypatch,
):
    """As the separate execs did: the receipt step is never reached, so a
    previous attempt's receipt is not cleared (the recovery probe reads it)."""
    from unittest.mock import MagicMock

    from benchflow.task import verifier_core
    from benchflow.task.verifier_core import Verifier
    from benchflow.task.verifier_errors import VerifierOutputParseError

    monkeypatch.setattr(verifier_core, "_SETUP_RETRY_DELAYS_SEC", (0.0, 0.0))
    sandbox = ScriptedSandbox(
        lambda command, user: (1, "") if command.startswith("chmod +x") else (0, "")
    )
    verifier = Verifier(MagicMock(), MagicMock(), sandbox)
    verifier._write_execution_receipt = True
    verifier.execution_receipt = ("/run/benchflow/verifier-started-old", "main")
    with pytest.raises(
        VerifierOutputParseError,
        match=r"^Verifier setup failed: chmod exited with rc=1$",
    ):
        await verifier._chmod_and_prepare_receipt("chmod +x /tests/test.sh", "main")
    assert verifier.execution_receipt == ("/run/benchflow/verifier-started-old", "main")
    # Every try keeps chmod's own budget.
    assert [e.timeout_sec for e in sandbox.execs] == [60, 60, 60]


# --- Daytona's synchronous path ----------------------------------------------------


@pytest.fixture
def sync_exec_enabled(monkeypatch):
    """Exercise the dormant path explicitly; production keeps the poll."""
    from benchflow.sandbox import daytona

    monkeypatch.setattr(daytona, "_DAYTONA_SYNC_EXEC_MAX_SEC", 30)


class _SyncProcess:
    def __init__(self, execute, session=None) -> None:
        self.execute = execute
        self.session = session
        self.requests: list[str] = []

    async def create_session(self, session_id):
        self.requests.append("create")

    async def delete_session(self, session_id):
        self.requests.append("delete")

    async def execute_session_command(self, session_id, request, timeout=None):
        self.requests.append(f"execute:{request.run_async}")
        return await self.execute(timeout)

    async def get_session_command(self, session_id, command_id):
        self.requests.append("status")
        return SimpleNamespace(id=command_id, exit_code=5)

    async def get_session_command_logs(self, session_id, command_id):
        self.requests.append("logs")
        return SimpleNamespace(stdout="from logs", stderr="e")

    async def get_session(self, session_id):
        self.requests.append("session")
        if isinstance(self.session, BaseException):
            raise self.session
        return self.session


def _daytona(process):
    pytest.importorskip("daytona")
    from benchflow.sandbox.daytona import DaytonaSandbox, _load_daytona_sdk
    from benchflow.sandbox.daytona_strategies import _DaytonaDirect

    _load_daytona_sdk()
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox._sandbox = SimpleNamespace(process=process)
    sandbox._strategy = _DaytonaDirect(sandbox)
    sandbox.rollout_paths = None
    return sandbox


def _answer(**fields):
    async def execute(timeout):
        return SimpleNamespace(**fields)

    return execute


@pytest.mark.parametrize(
    "response",
    [
        {"cmd_id": "c", "exit_code": None, "stdout": "", "stderr": ""},
        {"cmd_id": "c", "exit_code": "0", "stdout": "x", "stderr": ""},
        {"cmd_id": "c", "exit_code": 0, "stdout": None, "stderr": None},
        {"cmd_id": "c", "exit_code": 0, "stdout": "", "stderr": "", "output": "x"},
    ],
)
@pytest.mark.usefixtures("sync_exec_enabled")
async def test_an_incomplete_answer_polls_its_command_without_sending_it_again(
    response,
):
    process = _SyncProcess(_answer(**response))
    result = await _daytona(process).exec("true", timeout_sec=10)
    assert (result.return_code, result.stdout) == (5, "from logs")
    assert process.requests == ["create", "execute:False", "status", "logs"]


@pytest.mark.usefixtures("sync_exec_enabled")
async def test_an_answer_without_a_command_id_fails_as_before():
    process = _SyncProcess(_answer(cmd_id=None, exit_code=None))
    with pytest.raises(RuntimeError, match="Cannot find command ID"):
        await _daytona(process).exec("true", timeout_sec=10)


@pytest.mark.usefixtures("sync_exec_enabled")
async def test_a_transport_error_in_the_budget_is_the_request_error():
    from benchflow.diagnostics import TransientSandboxTransportError

    async def execute(timeout):
        raise TimeoutError("read timed out")

    process = _SyncProcess(execute)
    with pytest.raises(TransientSandboxTransportError):
        await _daytona(process).exec("true", timeout_sec=10)
    assert process.requests == ["create", "execute:False"]


def _overran(error=None):
    async def execute(timeout):
        await asyncio.sleep(timeout)
        raise error or TimeoutError("read timed out")

    return execute


@pytest.mark.parametrize(
    ("command", "outcome"),
    [
        (SimpleNamespace(id="c", exit_code=None), "timed out"),
        (SimpleNamespace(id="c", exit_code=124), "result"),
        (SimpleNamespace(id="c", exit_code=0), "result"),
    ],
)
@pytest.mark.usefixtures("sync_exec_enabled")
async def test_a_request_that_failed_after_the_budget_asks_the_session(
    command, outcome
):
    process = _SyncProcess(_overran(), SimpleNamespace(commands=[command]))
    sandbox = _daytona(process)
    if outcome == "timed out":
        with pytest.raises(RuntimeError, match=r"^Command timed out after 1 seconds$"):
            await sandbox.exec("sleep 9", timeout_sec=1)
        assert process.requests == ["create", "execute:False", "session"]
    else:
        result = await sandbox.exec("true", timeout_sec=1)
        assert (result.return_code, result.stdout) == (command.exit_code, "from logs")
        assert process.requests == ["create", "execute:False", "session", "logs"]


@pytest.mark.parametrize(
    "session", [SimpleNamespace(commands=[]), ConnectionResetError("gone")]
)
@pytest.mark.usefixtures("sync_exec_enabled")
async def test_without_a_command_to_find_the_request_error_stands(session):
    from benchflow.diagnostics import TransientSandboxTransportError

    process = _SyncProcess(_overran(), session)
    with pytest.raises(TransientSandboxTransportError):
        await _daytona(process).exec("true", timeout_sec=1)
    assert process.requests.count("execute:False") == 1


@pytest.mark.usefixtures("sync_exec_enabled")
async def test_a_completed_124_after_the_budget_is_still_a_result():
    """A completed status wins over HTTP elapsed time."""

    async def execute(timeout):
        await asyncio.sleep(timeout)
        return SimpleNamespace(cmd_id="c", exit_code=124, stdout="", stderr="")

    process = _SyncProcess(execute)
    result = await _daytona(process).exec("sleep 9", timeout_sec=1)
    assert result.return_code == 124


@pytest.mark.usefixtures("sync_exec_enabled")
async def test_cancellation_is_not_caught_and_nothing_is_sent_again():
    started = asyncio.Event()

    async def execute(timeout):
        started.set()
        await asyncio.sleep(60)

    process = _SyncProcess(execute)
    task = asyncio.ensure_future(_daytona(process).exec("true", timeout_sec=30))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.requests == ["create", "execute:False"]


@pytest.mark.usefixtures("sync_exec_enabled")
async def test_the_pinned_sdk_answer_is_read_as_the_result():
    """The SDK's own ``AsyncProcess.execute_session_command`` (pinned 0.184.0)
    over a stub API client: a synchronous answer's exit code, stdout and
    stderr reach the exec result."""
    pytest.importorskip("daytona")
    from daytona._async.process import AsyncProcess

    calls: list = []

    class _Api:
        async def session_execute_command(self, session_id, request, _request_timeout):
            calls.append((request.run_async, _request_timeout))
            return SimpleNamespace(
                cmd_id="c",
                output="out\nerr\n",
                stdout="out\n",
                stderr="err\n",
                exit_code=3,
                additional_properties={},
            )

    process = AsyncProcess.__new__(AsyncProcess)
    process._api_client = _Api()

    class _Wrapper:
        def __getattr__(self, name):
            return getattr(process, name)

        async def create_session(self, session_id):
            return None

    started = time.monotonic()
    result = await _daytona(_Wrapper()).exec("x", timeout_sec=7)
    assert time.monotonic() - started < 5
    assert (result.return_code, result.stdout, result.stderr) == (3, "out\n", "err\n")
    assert calls[0][0] is False
