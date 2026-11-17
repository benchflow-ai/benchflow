"""a PID 1 of the sandbox user's on a host whose Yama lets the user trace it.

On Debian 12 (``kernel.yama.ptrace_scope`` 0 by default) a task image whose
USER is the sandbox user's account (Terminal-Bench 4.1's rs-archive-clone)
errored in every trial at the verifier's hardening, after the agent's whole
turn: ``Sandbox-user writers remain: 1 (sleep infinity )``, PID 1 itself,
which the kill cannot end and may not vouch for when the agent's processes
could have attached to it. The sandbox is now refused when it starts, with the
two ways out, never retried; the dry run refuses the same task with the same
text where it can read the host's scope and says when it cannot; every
result.json records the scope. The Docker trials are in
tests/test_kill_image_user_init_docker.py.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from benchflow._utils.scoring import (
    SANDBOX_SETUP,
    classify_error,
    is_unrecoverable_startup_error,
)
from benchflow.cli.main import app
from benchflow.sandbox import pid1_ptrace
from benchflow.sandbox.pid1_ptrace import NO_YAMA, Pid1TraceableError

RS_ARCHIVE_CLONE = (
    "FROM python:3.12-slim\n"
    "RUN useradd --create-home --uid 10001 agent && mkdir -p /app\n"
    "WORKDIR /app\n"
    "USER agent\n"
)
ROOT_IMAGE = "FROM python:3.12-slim\nRUN mkdir -p /app\nWORKDIR /app\n"


def _answer(scope: str, pid1: str, user: str) -> str:
    return f"BF pid1-ptrace scope={scope} pid1={pid1} user={user}\n"


@pytest.mark.parametrize(
    ("scope", "pid1", "user", "read", "refused"),
    [
        # rs-archive-clone on Debian: PID 1 is agent (10001), Yama 0.
        ("0", "10001 10001 10001", "10001", 0, True),
        # A kernel without Yama leaves ptrace to the same-uid rule: as 0.
        (NO_YAMA, "10001 10001 10001", "10001", NO_YAMA, True),
        # Yama 1 to 3: the kill's PID 1 rule can vouch for it.
        ("1", "10001 10001 10001", "10001", 1, False),
        ("2", "10001 10001 10001", "10001", 2, False),
        ("3", "10001 10001 10001", "10001", 3, False),
        # The kill's walk matches the effective uid too (a setuid PID 1).
        ("0", "0 10001 0", "10001", 0, True),
        ("0", "0 0 10001", "10001", 0, True),
        # A root PID 1 (every root-USER image) is never the user's.
        ("0", "0 0 0", "1000", 0, False),
        # Another non-root USER (risk-scorer-replay's nobody) is not either.
        ("0", "65534 65534 65534", "10001", 0, False),
        # No such account yet: BenchFlow creates it with a uid nothing uses.
        ("0", "10001 10001 10001", "-", 0, False),
        # PID 1's uids could not be read: the scope is still recorded.
        ("0", "- - -", "10001", 0, False),
        # An unreadable scope refuses nothing (the kill still fails closed).
        ("unreadable", "10001 10001 10001", "10001", None, False),
    ],
)
def test_the_start_decision_is_the_kill_s_pid_1_rule(scope, pid1, user, read, refused):
    assert pid1_ptrace.decide(_answer(scope, pid1, user), "agent") == (read, refused)


def test_no_answer_refuses_nothing_and_records_nothing():
    assert pid1_ptrace.decide("", "agent") == (None, False)
    # A stand-in sandbox's answer that is not text (a mock) is no answer.
    assert pid1_ptrace.decide(None, "agent") == (None, False)
    assert pid1_ptrace.decide(object(), "agent") == (None, False)
    assert pid1_ptrace.decide(_answer("0", "10001 10001 10001", "10001"), None) == (
        0,
        False,
    )


def test_the_probe_runs_on_a_plain_shell_and_reads_this_kernel():
    """The probe as root's shell would run it, here: the scope it prints is
    this kernel's file, and a user the machine lacks prints ``-``."""
    out = subprocess.run(
        ["sh", "-c", pid1_ptrace.probe_command("bf-no-such-user")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    path = Path(pid1_ptrace.PTRACE_SCOPE_FILE)
    expected = pid1_ptrace.parse_scope(path.read_text()) if path.exists() else NO_YAMA
    assert pid1_ptrace.decide(out, "bf-no-such-user") == (expected, False), out
    assert out.rstrip().endswith("user=-"), out


@pytest.mark.parametrize("scope", [0, NO_YAMA])
def test_the_refusal_names_both_ways_out_and_is_never_retried(scope):
    text = str(pid1_ptrace.refusal("agent", scope))
    assert text.startswith(
        "sandbox startup refused: PID 1 of the task's image runs as the sandbox "
        "user 'agent'"
    )
    assert "Set kernel.yama.ptrace_scope=1 on the host" in text
    assert "end the task's image on another USER than 'agent'" in text
    assert classify_error(text) == SANDBOX_SETUP
    assert is_unrecoverable_startup_error(text)
    assert is_unrecoverable_startup_error(f"Pid1TraceableError: {text}")
    # The job's retry policy, with its infra retries on, never reruns it.
    from benchflow.evaluation import RetryConfig

    assert (
        RetryConfig(retry_on_infra=True).should_retry(f"Pid1TraceableError: {text}")
        is False
    )
    assert (
        RetryConfig(retry_on_infra=True).should_retry("sandbox startup failed: EOF")
        is True
    )


class _Env:
    def __init__(self, stdout: str | None = None, fails: bool = False) -> None:
        self.stdout = stdout
        self.fails = fails
        self.calls: list[tuple[str, dict]] = []

    async def exec(self, command: str, **kwargs):
        self.calls.append((command, kwargs))
        if self.fails:
            raise RuntimeError("the sandbox's command channel stopped")
        return SimpleNamespace(stdout=self.stdout, stderr="", return_code=0)


async def test_the_start_check_refuses_rs_archive_clone_on_yama_0():
    env = _Env(_answer("0", "10001 10001 10001", "10001"))
    with pytest.raises(Pid1TraceableError) as raised:
        await pid1_ptrace.check_at_start(env, "agent")
    assert str(raised.value) == str(pid1_ptrace.refusal("agent", 0))
    # The refused trial's record keeps the scope that refused it.
    assert raised.value.scope == 0
    ((command, kwargs),) = env.calls
    assert kwargs["user"] == "root" and "id -u agent" in command


async def test_the_start_check_records_the_scope_it_lets_through():
    assert (
        await pid1_ptrace.check_at_start(
            _Env(_answer("1", "10001 10001 10001", "10001")), "agent"
        )
        == 1
    )
    assert (
        await pid1_ptrace.check_at_start(_Env(_answer("0", "0 0 0", "1000")), "agent")
        == 0
    )
    # A root agent: read, recorded, never refused.
    assert await pid1_ptrace.check_at_start(_Env(_answer("0", "0 0 0", "0")), None) == 0
    # An exec that fails records nothing and refuses nothing.
    assert await pid1_ptrace.check_at_start(_Env(fails=True), "agent") is None


@pytest.fixture
def no_docker(monkeypatch):
    """Base images are read only from the local image store: none here."""
    monkeypatch.setattr(pid1_ptrace, "_local_image_user", lambda ref: None)


@pytest.mark.usefixtures("no_docker")
@pytest.mark.parametrize(
    ("dockerfile", "user", "runs_as"),
    [
        (RS_ARCHIVE_CLONE, "agent", True),
        (ROOT_IMAGE, "", False),
        # By the uid the Dockerfile gives the account.
        (
            "FROM debian:12\nRUN useradd -m -u 10001 -s /bin/bash agent\nUSER 10001\n",
            "10001",
            True,
        ),
        (
            "FROM debian:12\nRUN useradd -m -u 10001 agent\nUSER 10001:10001\n",
            "10001:10001",
            True,
        ),
        # Another account, and an uid that is not the agent's.
        ("FROM debian:12\nUSER nobody\n", "nobody", False),
        ("FROM debian:12\nRUN useradd -u 1000 builder\nUSER 1000\n", "1000", False),
        # Through an earlier stage, and ended as root again.
        (
            "FROM python:3.12-slim AS base\nRUN useradd agent\nUSER agent\n"
            "FROM base\nRUN true\n",
            "agent",
            True,
        ),
        (RS_ARCHIVE_CLONE + "USER root\n", "root", False),
        # An ARG names the user.
        ("FROM debian:12\nARG U=agent\nRUN useradd $U\nUSER ${U}\n", "agent", True),
        # A base image this host does not hold: unknown, noted, never refused.
        ("FROM ghcr.io/example/base:1\nRUN true\n", None, None),
        # A uid the account gets only when the image builds (R647's L1).
        ("FROM debian:12\nRUN useradd -m agent\nUSER 1000\n", "1000", None),
        # A uid with no account the Dockerfile makes: a base image may hold it.
        ("FROM debian:12\nUSER 1000\n", "1000", None),
    ],
)
def test_the_dry_run_reads_main_s_user_from_the_task_files(dockerfile, user, runs_as):
    found, accounts = pid1_ptrace.dockerfile_user(dockerfile)
    assert found == user
    assert pid1_ptrace.runs_as(found, accounts, "agent") is runs_as


def _task(root: Path, name: str, dockerfile: str, compose: str | None = None) -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(dockerfile)
    if compose is not None:
        (task / "environment" / "docker-compose.yaml").write_text(compose)
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("#!/bin/sh\ntrue\n")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[agent]\ntimeout_sec = 300\n\n'
        "[verifier]\ntimeout_sec = 120\n\n[environment]\n"
    )
    (task / "instruction.md").write_text("Do nothing.\n")
    return task


@pytest.mark.usefixtures("no_docker")
def test_compose_s_user_for_main_wins_over_the_image(tmp_path):
    from benchflow.task.task import Task

    as_agent = _task(
        tmp_path,
        "compose-agent",
        ROOT_IMAGE + "RUN useradd -u 10001 agent\n",
        "services:\n  main:\n    user: agent\n",
    )
    as_root = _task(
        tmp_path,
        "compose-root",
        RS_ARCHIVE_CLONE,
        "services:\n  main:\n    user: '0'\n",
    )
    assert pid1_ptrace.runs_as(*pid1_ptrace.task_image_user(Task(as_agent)), "agent")
    assert not pid1_ptrace.runs_as(*pid1_ptrace.task_image_user(Task(as_root)), "agent")


def _scope_file(tmp_path, monkeypatch, value: str | None) -> None:
    path = tmp_path / "ptrace_scope"
    if value is not None:
        path.write_text(value + "\n")
    monkeypatch.setattr(pid1_ptrace, "PTRACE_SCOPE_FILE", str(path))
    monkeypatch.setattr(pid1_ptrace.sys, "platform", "linux")


@pytest.mark.parametrize(
    ("value", "scope"), [("0", 0), ("1", 1), ("3", 3), (None, NO_YAMA)]
)
def test_the_host_scope_is_read_for_a_local_docker_daemon(
    tmp_path, monkeypatch, value, scope
):
    _scope_file(tmp_path, monkeypatch, value)
    environ = {"DOCKER_CONFIG": str(tmp_path / "docker-config")}
    assert pid1_ptrace.host_scope("docker", environ) == (scope, None)
    environ["DOCKER_HOST"] = "unix:///var/run/docker.sock"
    assert pid1_ptrace.host_scope("docker", environ) == (scope, None)


@pytest.mark.parametrize(
    ("sandbox", "environ", "platform"),
    [
        ("docker", {"DOCKER_HOST": "tcp://10.0.0.5:2376"}, "linux"),
        ("docker", {"DOCKER_HOST": "ssh://user@builder"}, "linux"),
        ("docker", {"DOCKER_CONTEXT": "builder"}, "linux"),
        (
            "docker",
            {"DOCKER_HOST": "unix:///home/u/.docker/desktop/docker.sock"},
            "linux",
        ),
        ("docker", {}, "darwin"),
        ("daytona", {}, "linux"),
        ("remote-docker", {}, "linux"),
    ],
)
def test_the_host_scope_is_not_read_for_another_kernel(
    tmp_path, monkeypatch, sandbox, environ, platform
):
    _scope_file(tmp_path, monkeypatch, "0")
    monkeypatch.setattr(pid1_ptrace.sys, "platform", platform)
    environ = {"DOCKER_CONFIG": str(tmp_path / "docker-config"), **environ}
    scope, why = pid1_ptrace.host_scope(sandbox, environ)
    assert scope is None and why


def test_a_current_docker_context_is_another_kernel(tmp_path, monkeypatch):
    _scope_file(tmp_path, monkeypatch, "0")
    config = tmp_path / "docker-config"
    config.mkdir()
    (config / "config.json").write_text(json.dumps({"currentContext": "vm"}))
    scope, why = pid1_ptrace.host_scope("docker", {"DOCKER_CONFIG": str(config)})
    assert scope is None and "'vm'" in why


# --- bench eval run --dry-run: the run's refusal, before any sandbox ----------


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """``bench eval run`` of the oracle on tmp_path/tasks, the sandbox's
    preflight skipped; the dry run reads the scope from *scope*."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
    monkeypatch.setattr(pid1_ptrace, "_local_image_user", lambda ref: None)

    def run(*flags: str, scope=0, why=None):
        monkeypatch.setattr(
            pid1_ptrace, "host_scope", lambda sandbox, environ=None: (scope, why)
        )
        return CliRunner().invoke(
            app,
            [
                "eval",
                "run",
                "--tasks-dir",
                str(tmp_path / "tasks"),
                "--agent",
                "oracle",
                "--sandbox",
                "docker",
                "--jobs-dir",
                str(tmp_path / "jobs"),
                "--dry-run",
                *flags,
            ],
            prog_name="bench",
        )

    return run


def _plan(result) -> dict:
    text = result.stdout
    return json.JSONDecoder().raw_decode(text[text.index("{") :])[0]


@pytest.mark.parametrize("scope", [0, NO_YAMA])
def test_the_dry_run_refuses_rs_archive_clone_with_the_run_s_text(
    tmp_path, bench, scope
):
    _task(tmp_path / "tasks", "rs-archive-clone", RS_ARCHIVE_CLONE)
    dry = bench(scope=scope)
    assert dry.exit_code == 1, dry.output
    refused = str(pid1_ptrace.refusal("agent", scope))
    assert _plan(dry)["errors"] == {
        "rs-archive-clone": f"Pid1TraceableError: {refused}"
    }


@pytest.mark.parametrize("scope", [1, 2, 3])
def test_the_dry_run_lets_rs_archive_clone_through_under_yama(tmp_path, bench, scope):
    _task(tmp_path / "tasks", "rs-archive-clone", RS_ARCHIVE_CLONE)
    dry = bench(scope=scope)
    assert dry.exit_code == 0, dry.output
    plan = _plan(dry)
    assert "errors" not in plan and "unchecked" not in plan


@pytest.mark.parametrize(
    ("name", "dockerfile"),
    [
        # R647's L1: the account's uid is known only once the image builds.
        ("pid1-numeric", "FROM debian:12\nRUN useradd -m agent\nUSER 1000\n"),
        # A base image this host does not hold, which may end as the account.
        ("pid1-base", "FROM ghcr.io/example/base:1\nRUN true\n"),
    ],
)
@pytest.mark.parametrize("scope", [0, NO_YAMA])
def test_the_dry_run_notes_an_image_user_it_cannot_tell(
    tmp_path, bench, name, dockerfile, scope
):
    """Dry run = run: the run may refuse it at Yama 0, so the dry run lists the
    task as unchecked with the reason instead of passing it in silence."""
    _task(tmp_path / "tasks", name, dockerfile)
    dry = bench(scope=scope)
    assert dry.exit_code == 0, dry.output
    plan = _plan(dry)
    assert "errors" not in plan
    ((note,),) = [plan["unchecked"]]
    assert note["target"] == name
    assert "the image's user cannot be told" in note["reason"]
    assert "kernel.yama.ptrace_scope" in note["reason"]


@pytest.mark.parametrize("scope", [1, 2, 3])
def test_an_image_user_it_cannot_tell_is_not_noted_under_yama(tmp_path, bench, scope):
    """At Yama 1 to 3 the run refuses no image: nothing to note."""
    _task(
        tmp_path / "tasks",
        "pid1-numeric",
        "FROM debian:12\nRUN useradd -m agent\nUSER 1000\n",
    )
    dry = bench(scope=scope)
    assert dry.exit_code == 0, dry.output
    assert "unchecked" not in _plan(dry)


def test_an_unknown_user_and_an_unreadable_scope_give_both_reasons(tmp_path, bench):
    _task(
        tmp_path / "tasks",
        "pid1-numeric",
        "FROM debian:12\nRUN useradd -m agent\nUSER 1000\n",
    )
    dry = bench(scope=None, why="DOCKER_HOST names a remote Docker daemon")
    ((note,),) = [_plan(dry)["unchecked"]]
    assert "the image's user cannot be told" in note["reason"]
    assert "DOCKER_HOST names a remote Docker daemon" in note["reason"]


@pytest.mark.parametrize("scope", [0, 1, NO_YAMA])
def test_a_root_user_image_dry_runs_as_before_under_any_scope(tmp_path, bench, scope):
    _task(tmp_path / "tasks", "plain", ROOT_IMAGE)
    dry = bench(scope=scope)
    assert dry.exit_code == 0, dry.output
    plan = _plan(dry)
    assert "errors" not in plan and "unchecked" not in plan


def test_the_dry_run_says_when_it_cannot_read_the_host_s_scope(tmp_path, bench):
    _task(tmp_path / "tasks", "rs-archive-clone", RS_ARCHIVE_CLONE)
    dry = bench(scope=None, why="DOCKER_HOST names a remote Docker daemon")
    assert dry.exit_code == 0, dry.output
    plan = _plan(dry)
    assert "errors" not in plan
    ((note,),) = [plan["unchecked"]]
    assert note["target"] == "rs-archive-clone"
    assert "kernel.yama.ptrace_scope" in note["reason"]
    assert "DOCKER_HOST names a remote Docker daemon" in note["reason"]


def test_a_root_sandbox_user_is_never_refused(tmp_path, bench):
    _task(tmp_path / "tasks", "rs-archive-clone", RS_ARCHIVE_CLONE)
    dry = bench("--sandbox-user", "root", scope=0)
    assert dry.exit_code == 0, dry.output
    assert "errors" not in _plan(dry)


# --- result.json: every record holds the scope ---------------------------------


@pytest.mark.parametrize("scope", [0, 1, NO_YAMA, None])
def test_every_result_records_the_host_s_ptrace_scope(tmp_path, scope):
    from benchflow.rollout._results import _build_rollout_result

    kwargs = {} if scope is None else {"host_ptrace_scope": scope}
    _build_rollout_result(
        tmp_path,
        task_name="task",
        rollout_name=tmp_path.name,
        agent="oracle",
        agent_name="oracle",
        model=None,
        n_tool_calls=0,
        prompts=["task"],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={},
        **kwargs,
    )
    recorded = json.loads((tmp_path / "result.json").read_text())
    assert "host_ptrace_scope" in recorded
    assert recorded["host_ptrace_scope"] == scope


def test_the_rollout_passes_its_scope_to_the_result():
    """Rollout._build_result hands the scope start() read to the record, and
    a rollout that never started records None (the class default)."""
    import inspect

    from benchflow.rollout import Rollout

    assert Rollout._host_ptrace_scope is None
    source = inspect.getsource(Rollout._build_result)
    assert "host_ptrace_scope=self._host_ptrace_scope" in source
    start = inspect.getsource(Rollout.start)
    assert "pid1_ptrace.check_at_start" in start
    assert "self._host_ptrace_scope = refused.scope" in start


async def test_a_refused_start_records_the_scope_that_refused_it(monkeypatch):
    """Rollout.start keeps the refusal's scope for result.json, then raises."""
    from benchflow.rollout import Rollout

    async def started(*args, **kwargs):
        return None

    monkeypatch.setattr("benchflow.rollout._start_env_and_upload", started)
    rollout = Rollout.__new__(Rollout)
    rollout._env = _Env(_answer("0", "10001 10001 10001", "10001"))
    rollout._config = SimpleNamespace(
        task_path=Path("task"),
        uploads=None,
        primary_agent="oracle",
        sandbox_user="agent",
    )
    rollout._timing = {}
    rollout._task = None
    rollout._env_externally_owned = False
    rollout._from_branch_snapshot = False
    rollout._skills_agent_derived = False
    rollout._skills_baked = False
    for name in ("_configure_daytona_layers", "_request_layers", "_reserve_host_proxy"):
        monkeypatch.setattr(rollout, name, lambda: None, raising=False)
    with pytest.raises(Pid1TraceableError):
        await Rollout.start(rollout)
    assert rollout._host_ptrace_scope == 0
