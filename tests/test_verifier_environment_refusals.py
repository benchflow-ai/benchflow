"""Behavioral regressions for V3's findings at 39a8ed41 and 7c108b94 and for
V3-CONFIRM-1..6 at 548ac115.

All sandbox scripts run only in disposable Docker containers, without host
mounts. The transport uses DockerSandbox.exec, including its outer shell and
env-file wrapper, not an already-sanitized test shell.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from benchflow._runtime_tree import KIT_BIN
from benchflow.rollout._setup import _verify_rollout
from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._shell import SYSTEM_PATH
from benchflow.sandbox._task_route import as_task_command
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.lockdown import (
    REFUSAL_STUBS_PREFIX,
    _build_verifier_env,
    _image_pytest_ini,
    _trusted_verifier_path,
    harden_before_verify,
    setup_sandbox_user,
)
from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig
from tests.test_verifier_environment_checks import _task as _harden_task
from tests.test_verifier_trusted_path import _rollout_paths

BASE = "alpine@sha256:d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc"
DOCKERFILE = r"""
FROM BASE
RUN apk add --no-cache acl \
 && mkdir -p /opt/probe/bin /opt/tools/bin /opt/vendor /app /testbed /testbed_verify /tests /logs/verifier \
 && chmod 777 /opt/probe/bin \
 && printf '#!/bin/sh\necho image-tool\n' > /opt/tools/bin/tool \
 && chmod 755 /opt/tools/bin/tool \
 && cp /opt/tools/bin/tool /opt/vendor/image-tool \
 && chown 1000:1000 /opt/vendor/image-tool
VOLUME /opt/vendor
ENV PATH=/opt/probe/bin:/opt/tools/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
WORKDIR /app
""".replace("FROM BASE", f"FROM {BASE}")


def _run(args, *, input=None):
    return subprocess.run(
        args, input=input, capture_output=True, text=True, timeout=300
    )


@pytest.fixture(scope="module")
def image():
    if shutil.which("docker") is None or _run(["docker", "info"]).returncode:
        pytest.skip("Docker required")
    tag = f"bf-v3-r4-{uuid.uuid4().hex[:12]}"
    made = _run(
        ["docker", "build", "--label", "target=V3", "-q", "-t", tag, "-"],
        input=DOCKERFILE,
    )
    assert made.returncode == 0, made.stderr
    try:
        yield tag
    finally:
        _run(["docker", "rmi", tag])


def _container(image, path=None):
    name = f"bf-v3-r4-box-{uuid.uuid4().hex[:12]}"
    made = _run(
        [
            "docker",
            "run",
            "-d",
            "--network",
            "none",
            "--label",
            "target=V3",
            *(["-e", f"PATH={path}"] if path is not None else []),
            "--name",
            name,
            image,
            "/bin/sleep",
            "600",
        ]
    )
    assert made.returncode == 0, made.stderr
    return name


def _sandbox(containers):
    """A DockerSandbox whose Compose services are the *containers* by name."""
    sandbox = DockerSandbox.__new__(DockerSandbox)
    # Exercise bootstrap decoding even on probes with no explicit environment.
    sandbox._persistent_env = {"BF_FIXTURE": "1"}
    sandbox.default_user = None
    sandbox._logs_are_mounted = False
    sandbox.logger = logging.getLogger("bf-v3-fixture")
    sandbox._main_container = containers.get("main")
    # Downloads stream ``docker cp`` with Compose's process environment:
    # here the real docker CLI with this process's.
    sandbox._docker_compose_env = lambda: dict(os.environ)
    sandbox._compose_transport_env = lambda: {}

    async def transport(command, **kwargs):
        if command[:2] == ["ps", "-q"]:
            # A download from another service asks Compose for its container.
            done = _run(["docker", "inspect", "-f", "{{.Id}}", containers[command[2]]])
            return ExecResult(
                stdout=done.stdout, stderr=done.stderr, return_code=done.returncode
            )
        if command[0] == "cp":
            # Uploads and downloads: ``<service>:<path>`` names a container.
            args = [
                containers[a.split(":", 1)[0]] + ":" + a.split(":", 1)[1]
                if a.split(":", 1)[0] in containers
                else a
                for a in command[1:]
            ]
            done = _run(["docker", "cp", *args])
            return ExecResult(
                stdout=done.stdout, stderr=done.stderr, return_code=done.returncode
            )
        assert command[:2] == ["exec", "-T"]
        # Replace just the Compose transport; keep the backend's chosen shell,
        # command and env-file bootstrap exactly as production constructed them.
        # The route's shell: /bin/sh, or the protected kit's once it exists.
        index = (
            next(
                i
                for i, a in enumerate(command)
                if a.endswith("/sh") and command[i + 1 : i + 2] == ["-c"]
            )
            - 1
        )
        flags = command[2:index]
        # The exec's env arrives on stdin: pass it on.
        stdin = kwargs.get("stdin_data")
        if isinstance(stdin, bytes):
            stdin = stdin.decode()
        done = _run(
            [
                "docker",
                "exec",
                *(["-i"] if stdin is not None else []),
                *flags,
                containers[command[index]],
                *command[index + 1 :],
            ],
            input=stdin,
        )
        return ExecResult(
            stdout=done.stdout, stderr=done.stderr, return_code=done.returncode
        )

    sandbox._run_docker_compose_command = transport
    return sandbox


@contextlib.contextmanager
def _box(image, path=None):
    name = _container(image, path)
    try:
        yield SimpleNamespace(name=name, env=_sandbox({"main": name}))
    finally:
        # -v removes only this container's anonymous image VOLUME.
        _run(["docker", "rm", "-f", "-v", name])


@pytest.fixture
def box(image):
    with _box(image) as made:
        yield made


def _sh(box, script):
    # The fixture tools come from the system folders, whatever the box PATH.
    done = _run(
        [
            "docker",
            "exec",
            "-e",
            f"PATH={SYSTEM_PATH}",
            box.name,
            "/bin/sh",
            "-c",
            script,
        ]
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def _write(box, path, content):
    done = _run(
        [
            "docker",
            "exec",
            "-i",
            box.name,
            "/bin/sh",
            "-c",
            '/bin/cat > "$1"',
            "bf-write",
            path,
        ],
        input=content,
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.asyncio
async def test_trust_probes_never_execute_image_path_stubs(box):
    """Forged ls/cat and outer sh never run."""
    for tool in ("sh", "base64", "ls", "cat", "id", "readlink", "getfacl"):
        body = f'#!/bin/sh\nprintf "{tool}\\n" >> /tmp/probe-used\n'
        if tool == "ls":
            body += (
                'case "$*" in *"-ldn /opt/probe/bin"*) '
                'echo "drwxr-xr-x 2 0 0 4096 Jan 1 00:00 /opt/probe/bin"; '
                "exit 0;; esac\n"
            )
        if tool == "cat":
            body += 'case "$1" in /testbed/pytest.ini) set -- /testbed_verify/pytest.ini;; esac\n'
        body += f'exec /bin/{tool} "$@"\n'
        _write(box, f"/opt/probe/bin/{tool}", body)
    _sh(box, "chmod 755 /opt/probe/bin/*; chmod 700 /testbed_verify")
    _write(box, "/testbed_verify/pytest.ini", "[pytest]\naddopts = -q\n")
    _write(
        box,
        "/testbed/pytest.ini",
        "[pytest]\naddopts = --deselect=tests/test_required.py\n",
    )

    path = await _trusted_verifier_path(box.env, None, "/app")
    ini = await _image_pytest_ini(box.env, "/testbed")

    assert "/opt/probe/bin" not in path.split(":")
    assert ini is None
    assert _sh(box, "test ! -e /tmp/probe-used; echo clean").strip() == "clean"


@pytest.mark.asyncio
async def test_image_addopts_argument_files_are_an_explicit_error(box):
    """Guards the fix at 39a8ed41, even with a byte-identical ini."""
    ini = "[pytest]\naddopts = @args\n"
    _write(box, "/testbed_verify/pytest.ini", ini)
    _write(box, "/testbed/pytest.ini", ini)
    _write(box, "/testbed/args", "--deselect=tests/test_answer.py::test_required\n")
    _sh(box, "chmod 700 /testbed_verify")

    with pytest.raises(
        RuntimeError, match=r"verifier refused pytest argument files .*@file"
    ):
        await _image_pytest_ini(box.env, "/testbed")


@pytest.mark.asyncio
@pytest.mark.parametrize("unsafe", ["mode", "owner", "acl", "link-target", "parent"])
async def test_root_owned_path_folder_does_not_trust_unsafe_entries(box, unsafe):
    """0755 folders do not freeze their tools."""
    _sh(box, "adduser -D -u 1001 agent")
    changes = {
        "mode": "chmod 777 /opt/tools/bin/tool",
        "owner": "chown agent /opt/tools/bin/tool",
        "acl": "setfacl -m u:agent:rwx /opt/tools/bin/tool",
        "link-target": "mkdir -p /opt/unsafe; cp /opt/tools/bin/tool /opt/unsafe/tool; "
        "chmod 777 /opt/unsafe/tool; ln -s /opt/unsafe/tool /opt/tools/bin/link",
        "parent": "chmod 777 /opt/tools",
    }
    _sh(box, changes[unsafe])

    path = await _trusted_verifier_path(box.env, "agent", "/app")

    assert "/opt/tools/bin" not in path.split(":")


@pytest.mark.asyncio
async def test_uid_inventory_includes_image_volume_files(box):
    """Uid/gid 1000 already own a VOLUME tool."""
    await setup_sandbox_user(box.env, "agent", "/app", guard_daemon=False)

    uid, gid = _sh(box, "id -u agent; id -g agent").split()
    assert uid != "1000"
    assert gid != "1000"
    assert _sh(box, "stat -c '%u %g' /opt/vendor/image-tool").strip() == "1000 1000"


@pytest.mark.asyncio
async def test_incomplete_uid_inventory_refuses_setup(box):
    """Find failure cannot look like an empty image."""
    _sh(box, "rm /usr/bin/find")
    _write(
        box,
        "/usr/bin/find",
        '#!/bin/sh\nprintf "inventory unavailable\\n" >&2\nexit 1\n',
    )
    _sh(box, "chmod 755 /usr/bin/find")

    # PT's kit preparation (it copies root's tools) refuses first now; either
    # way setup refuses and no user is made.
    with pytest.raises(
        RuntimeError,
        match=r"inventory could not complete|could not prepare its protected runtime",
    ):
        await setup_sandbox_user(box.env, "agent", "/app", guard_daemon=False)
    assert (
        _sh(
            box, "if id agent >/dev/null 2>&1; then echo created; else echo absent; fi"
        ).strip()
        == "absent"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_error", [False, True])
async def test_caught_missing_program_clears_reward_and_names_refusal(
    box, tmp_path, existing_error, caplog
):
    """Caught command-not-found is not scored 0."""
    _write(box, "/opt/probe/bin/bf-required", "#!/bin/sh\nexit 0\n")
    _sh(box, "chmod 755 /opt/probe/bin/bf-required")
    test_sh = (
        "#!/bin/sh\nif bf-required; then echo 1; else echo 0; fi"
        " > /logs/verifier/reward.txt\n"
    )
    _write(box, "/tests/test.sh", test_sh)
    paths = _rollout_paths(tmp_path)
    task = _script_task(tmp_path, test_sh)

    class CaughtFailureVerifier:
        execution_receipt = None
        test_return_code = None

        async def verify(self):
            # The task route, as verifier_core runs test.sh (PT's C8).
            with as_task_command(box.env):
                done = await box.env.exec(
                    "/bin/sh /tests/test.sh", env=task.config.verifier.env
                )
            self.test_return_code = done.return_code
            paths.test_stdout_path.write_text((done.stdout or "") + (done.stderr or ""))
            assert done.return_code == 0
            reward = float(_sh(box, "cat /logs/verifier/reward.txt"))
            assert reward == 0
            if existing_error:
                raise RuntimeError("earlier verifier error")
            return SimpleNamespace(rewards={"reward": reward})

    rewards, error, _ = await _verify_rollout(
        box.env,
        task,
        paths,
        {},
        _refusing_planes(CaughtFailureVerifier()),
        "agent",
        "/app",
    )

    assert "/opt/probe/bin" in caplog.text
    if existing_error:
        assert rewards is None
        assert "earlier verifier error" in error
    else:
        assert rewards == {"reward": 0.0}
        assert error is None
    record = json.loads((paths.verifier_dir / "trusted-path.json").read_text())
    # The evidence is the refusal stub that ran, not the
    # wording of the output.
    assert record["refused_programs_run"] == [
        {"program": "bf-required", "folder": "/opt/probe/bin"}
    ]


def _refusing_planes(verifier):
    """The verifier environment hardening builds, as test.sh gets it."""

    async def harden(env, task, sandbox_user, **kwargs):
        task.config.verifier.env = await _build_verifier_env(
            env, task, sandbox_user, "/app"
        )

    return SimpleNamespace(
        harden_before_verify=harden, verifier=lambda **kwargs: verifier
    )


def _script_task(tmp_path, test_sh):
    task = _harden_task(tmp_path, {"test.sh": test_sh})
    task.name = "path-task"
    task.config.verifier.timeout_sec = 60
    task.config.verifier.reward_range = None
    return task


class _ScriptVerifier:
    """Runs /tests/test.sh as a script verifier does and reads its reward."""

    execution_receipt = None
    test_return_code = None

    def __init__(self, box, task, paths):
        self.box, self.task, self.paths = box, task, paths

    async def verify(self):
        # On the task route, as verifier_core runs test.sh (PT's C8): root's
        # plumbing would pin its own PATH over the verifier's.
        with as_task_command(self.box.env):
            done = await self.box.env.exec(
                "/bin/sh /tests/test.sh", env=self.task.config.verifier.env
            )
        self.test_return_code = done.return_code
        self.paths.test_stdout_path.write_text(
            (done.stdout or "") + (done.stderr or "")
        )
        reward = float(_sh(self.box, "cat /logs/verifier/reward.txt"))
        return SimpleNamespace(rewards={"reward": reward})


async def _verify(box, tmp_path, test_sh, *, optional_tools=(), required_tools=()):
    _write(box, "/tests/test.sh", test_sh)
    paths = _rollout_paths(tmp_path)
    task = _script_task(tmp_path, test_sh)
    task.config.verifier.optional_tools = list(optional_tools)
    task.config.verifier.required_tools = (
        None if required_tools is None else list(required_tools)
    )
    rewards, error, _ = await _verify_rollout(
        box.env,
        task,
        paths,
        {},
        _refusing_planes(_ScriptVerifier(box, task, paths)),
        "agent",
        "/app",
    )
    record_path = paths.verifier_dir / "trusted-path.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else None
    return rewards, error, record


async def _prepare_inspector(box) -> None:
    """What a trial does before the agent's turn, after the user: the
    pre-agent PATH inspector (the build-config snapshot records it)."""
    await lockdown._prepare_bash_env_inspector(
        box.env, blocked=lockdown._blocked_verifier_path_prefixes("agent", "/app")
    )


def _plant_system_folder(box) -> None:
    """Batch 32b review A M-1: the agent replaced every tool the protected
    kit holds in /usr/sbin, the first of the system folders. Root's steps
    take the kit's copies; one that re-pinned the system PATH would run
    these, and they log it."""
    kit = _sh(box, f"/bin/ls {KIT_BIN}").split()
    planted = []
    for tool in kit:
        if tool == "sh" or _sh(box, f"[ -e /usr/sbin/{tool} ] && echo y || true"):
            continue
        _write(
            box,
            f"/usr/sbin/{tool}",
            f'#!/bin/sh\nprintf "sys-{tool}\\n" >> /tmp/probe-used\n'
            f'for d in /bin /usr/bin /sbin; do [ -x "$d/{tool}" ] '
            f'&& exec "$d/{tool}" "$@"; done\nexit 127\n',
        )
        planted.append(f"/usr/sbin/{tool}")
    # By absolute path: chmod itself may be among the plants.
    if planted:
        _sh(box, "/bin/chmod 755 " + " ".join(planted))


# Every program hardening could reach through the image's PATH. Each stub in
# the agent-writable /opt/probe/bin logs that root ran it, then runs the real
# one; python3 is not in the image, so its stub only logs.
_ROOT_TOOLS = [
    *("cat", "id", "python3", "find", "rm", "chmod", "chown", "mkdir", "cp"),
    *("ln", "pkill", "sleep", "stat", "ls", "readlink", "getfacl", "sed", "awk"),
    *("xargs", "head", "tr", "base64", "sort", "wc", "test", "env"),
]


@pytest.mark.asyncio
async def test_hardening_never_runs_image_path_programs_as_root(box, tmp_path):
    """After the PATH probe, hardening's
    distro, uid, Python, guard and clean-up commands found ``cat``, ``id``,
    ``python3``, ``find`` and others in an agent-writable PATH folder."""
    await setup_sandbox_user(box.env, "agent", "/app", guard_daemon=False)
    await _prepare_inspector(box)
    # PT's PATH lock closed /opt/probe/bin to root during setup; a lock
    # that did not hold (a folder it could not close) leaves it open, and
    # V3's judge must still keep root off it.
    _sh(box, "/bin/chmod 777 /opt/probe/bin")
    for tool in _ROOT_TOOLS:
        body = f'#!/bin/sh\nprintf "{tool}\\n" >> /tmp/probe-used\n'
        if tool == "python3":
            body += "exit 127\n"
        else:
            body += (
                f'for d in /bin /usr/bin /sbin /usr/sbin; do [ -x "$d/{tool}" ] '
                f'&& exec "$d/{tool}" "$@"; done\nexit 127\n'
            )
        _write(box, f"/opt/probe/bin/{tool}", body)
    _sh(box, "chmod 755 /opt/probe/bin/*")
    _plant_system_folder(box)
    task = _harden_task(tmp_path, {"test.sh": "#!/bin/sh\necho 1\n"})

    failure = None
    try:
        await harden_before_verify(box.env, task, "agent", workspace="/app")
    except RuntimeError as exc:  # a planted program can also break hardening
        failure = exc

    used = _sh(box, "/bin/cat /tmp/probe-used 2>/dev/null || true").split()
    assert used == []
    assert failure is None
    assert "/opt/probe/bin" not in task.config.verifier.env["PATH"].split(":")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "helper", "workdir"),
    [
        ("bin:/usr/local/bin:/usr/bin:/bin", "/app/bin/check-answer", None),
        (":/usr/local/bin:/usr/bin:/bin", "/app/check-answer", None),
        (".:/usr/local/bin:/usr/bin:/bin", "/app/check-answer", None),
        ("bin:/usr/local/bin:/usr/bin:/bin", "/app/sub/bin/check-answer", "/app/sub"),
        ("", "/app/check-answer", None),
    ],
    ids=["bin", "empty-entry", "dot", "bin-after-cd", "empty-path"],
)
async def test_relative_path_entries_are_a_verifier_error(
    image, tmp_path, path, helper, workdir
):
    """Guards the fix at 7c108b94 and V3-CONFIRM-2 at 548ac115: a
    relative entry names another folder in every folder the verifier's
    commands change to, and an explicitly empty PATH is the working folder.
    Neither can be inventoried, so a caught failure of a helper found only
    through one (here after ``cd``) scored 0; hardening now refuses them."""
    with _box(image, path) as box:
        _sh(box, f"mkdir -p {helper.rsplit('/', 1)[0]}")
        _write(box, helper, "#!/bin/sh\nexit 0\n")
        _sh(box, f"chmod 755 {helper}")
        go = f"cd {workdir}\n" if workdir else ""
        rewards, error, _ = await _verify(
            box,
            tmp_path,
            f"#!/bin/sh\n{go}if check-answer; then echo 1; else echo 0; fi"
            " > /logs/verifier/reward.txt\nexit 0\n",
            required_tools=None,
        )

    assert rewards is None
    assert error is not None and "relative" in error


@pytest.mark.asyncio
@pytest.mark.parametrize("probe_first", [False, True])
async def test_redirected_or_prechecked_refusal_is_not_scored_zero(
    box, tmp_path, probe_first, caplog
):
    """A refusal sent to the verifier's own
    log, or a silent ``command -v`` check, left no trace in test-stdout.txt."""
    _write(box, "/opt/probe/bin/cargo", "#!/bin/sh\nexit 0\n")
    _sh(box, "chmod 755 /opt/probe/bin/cargo")
    check = (
        "command -v cargo >/dev/null 2>&1 || "
        "{ echo 0 > /logs/verifier/reward.txt; exit 0; }\n"
        if probe_first
        else ""
    )
    rewards, error, record = await _verify(
        box,
        tmp_path,
        "#!/bin/sh\n"
        + check
        + "if cargo test > /logs/verifier/cargo.log 2>&1; then echo 1; "
        "else echo 0; fi > /logs/verifier/reward.txt\nexit 0\n",
    )

    assert rewards == {"reward": 0.0}
    assert error is None
    assert "cargo (in /opt/probe/bin)" in caplog.text
    assert "Verifier scored 0" in caplog.text
    assert record["refused_programs_run"] == [
        {"program": "cargo", "folder": "/opt/probe/bin"}
    ]


@pytest.mark.asyncio
async def test_honest_not_found_output_keeps_its_score(box, tmp_path):
    """An honest pass that prints "command
    not found" lost its score because an unused PATH folder was left out."""
    _write(box, "/opt/probe/bin/unused-tool", "#!/bin/sh\nexit 0\n")
    _sh(box, "chmod 755 /opt/probe/bin/unused-tool")

    rewards, error, record = await _verify(
        box,
        tmp_path,
        "#!/bin/sh\necho 'checked expected diagnostic: command not found'\n"
        "echo 1 > /logs/verifier/reward.txt\nexit 0\n",
    )

    assert (rewards, error) == ({"reward": 1.0}, None)
    assert record["verifier_missed_a_program"] is True
    assert record["refused_programs_run"] == []


def _lifecycle_task(tmp_path, test_sh):
    """A task.toml task whose tests/ package the real Verifier uploads."""
    package = tmp_path / "task" / "tests"
    package.mkdir(parents=True)
    (package / "test.sh").write_text(test_sh)
    task = MagicMock()
    task.name = "path-task"
    task.task_dir = package.parent
    task.paths.task_dir = package.parent
    task.paths.tests_dir = package
    task.paths.test_path = package / "test.sh"
    task.paths.uses_native_verifier_dir = False
    task.config = TaskConfig.model_validate_toml(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\nrequired_tools = []\n'
    )
    return task


@pytest.mark.asyncio
async def test_verifier_setup_never_runs_image_path_programs_as_root(box, tmp_path):
    """Guards V3-CONFIRM-1 at 548ac115: after hardening, the Verifier's own
    root commands (the upload's ``mkdir``, ``chmod +x`` of test.sh, the start
    receipt's folder, the download's ``chown``) found their programs on the
    image's PATH. A ``chmod`` planted in an agent-writable PATH folder ran as
    root after the trusted test.sh was uploaded and replaced it."""
    await setup_sandbox_user(box.env, "agent", "/app", guard_daemon=False)
    await _prepare_inspector(box)
    # PT's PATH lock closed /opt/probe/bin to root during setup; a lock
    # that did not hold (a folder it could not close) leaves it open, and
    # V3's judge must still keep root off it.
    _sh(box, "/bin/chmod 777 /opt/probe/bin")
    for tool in _ROOT_TOOLS:
        body = f'#!/bin/sh\nprintf "{tool}\\n" >> /tmp/probe-used\n'
        if tool == "chmod":
            body += (
                'case "$*" in *test.sh*) printf "#!/bin/sh\\necho 1 > '
                '/logs/verifier/reward.txt\\n" > /tests/test.sh;; esac\n'
            )
        if tool == "python3":
            body += "exit 127\n"
        else:
            body += (
                f'for d in /bin /usr/bin /sbin /usr/sbin; do [ -x "$d/{tool}" ] '
                f'&& exec "$d/{tool}" "$@"; done\nexit 127\n'
            )
        _write(box, f"/opt/probe/bin/{tool}", body)
    _sh(box, "chmod 755 /opt/probe/bin/*")
    # rm too: the guard markers' clear-out in front of test.sh takes main's
    # kit rm, not test.sh's PATH (merge wave 32b round-2 review A, L-2).
    _plant_system_folder(box)
    task = _lifecycle_task(tmp_path, "#!/bin/sh\necho 0 > /logs/verifier/reward.txt\n")
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    planes = SimpleNamespace(
        harden_before_verify=harden_before_verify,
        verifier=lambda **kwargs: Verifier(**kwargs),
    )

    rewards, error, _ = await _verify_rollout(
        box.env, task, paths, {}, planes, "agent", "/app"
    )

    used = _sh(box, "/bin/cat /tmp/probe-used 2>/dev/null || true").split()
    assert (used, rewards, error) == ([], {"reward": 0.0}, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["check@v1", ".check", "-check", "check answer"])
async def test_any_program_name_in_a_left_out_folder_is_refused(
    box, tmp_path, name, caplog
):
    """Guards V3-CONFIRM-3 at 548ac115: names outside ``[A-Za-z0-9._+-]``,
    names starting with ``-`` and hidden names got no refusal stub, so a
    caught failure of the only such program scored 0."""
    program = f"/opt/probe/bin/{name}"
    _write(box, program, "#!/bin/sh\nexit 0\n")
    _sh(box, f"chmod 755 {shlex.quote(program)}")

    rewards, error, record = await _verify(
        box,
        tmp_path,
        f"#!/bin/sh\nif {shlex.quote(name)}; then echo 1; else echo 0; fi"
        " > /logs/verifier/reward.txt\nexit 0\n",
    )

    assert rewards == {"reward": 0.0}
    assert error is None
    assert f"{name} (in /opt/probe/bin)" in caplog.text
    assert "Verifier scored 0" in caplog.text
    assert record["refused_programs_run"] == [
        {"program": name, "folder": "/opt/probe/bin"}
    ]


@pytest.mark.asyncio
async def test_a_program_name_no_record_can_hold_refuses_hardening(
    box, tmp_path, caplog
):
    """Guards V3-CONFIRM-3 at 548ac115: a name with a newline was skipped
    without a stub; hardening now refuses it instead of losing the evidence."""
    _sh(
        box,
        "f=\"/opt/probe/bin/bad$(printf '\\nname')\"; "
        'printf \'#!/bin/sh\\nexit 0\\n\' > "$f"; chmod 755 "$f"',
    )

    rewards, error, _ = await _verify(
        box, tmp_path, "#!/bin/sh\necho 0 > /logs/verifier/reward.txt\nexit 0\n"
    )

    assert rewards == {"reward": 0.0}
    assert error is None
    assert "newline" in caplog.text


@pytest.mark.asyncio
async def test_excess_program_names_cannot_drop_the_trial(
    box, tmp_path, monkeypatch, caplog
):
    """FIX-49 r4: filling a writable PATH directory is evidence, not infrastructure."""
    monkeypatch.setattr(lockdown, "_REFUSAL_STUBS_MAX", 1)
    for name in ("first", "second"):
        _write(box, f"/opt/probe/bin/{name}", "#!/bin/sh\nexit 0\n")
        _sh(box, f"chmod 755 /opt/probe/bin/{name}")
    rewards, error, _ = await _verify(
        box, tmp_path, "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    assert rewards == {"reward": 0.0}
    assert error is None
    assert "more than 1 programs" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(("score", "kept"), [("1", True), ("0.5", True)])
async def test_an_optional_probe_of_a_refused_program_keeps_full_marks(
    box, tmp_path, score, kept
):
    """Guards V3-CONFIRM-6 at 548ac115: an honest verifier that probes
    ``cargo --version``, falls back to another check and passes lost its score
    because the probe ran a refusal stub. RECONFIRM-3 at 76a0723c requires
    an explicit optional-tools contract, independent of the score."""
    _write(box, "/opt/probe/bin/cargo", "#!/bin/sh\nexit 0\n")
    _sh(box, "chmod 755 /opt/probe/bin/cargo")

    rewards, error, record = await _verify(
        box,
        tmp_path,
        "#!/bin/sh\nif cargo --version >/dev/null 2>&1; then echo cargo; fi\n"
        f"echo {score} > /logs/verifier/reward.txt\nexit 0\n",
        optional_tools=("cargo",),
    )

    if kept:
        assert (rewards, error) == ({"reward": float(score)}, None)
    else:
        assert rewards is None
        assert error is not None and "cargo (in /opt/probe/bin)" in error
    assert record["refused_programs_run"] == [
        {"program": "cargo", "folder": "/opt/probe/bin"}
    ]


@pytest.mark.asyncio
async def test_a_target_service_verifier_gets_its_own_refusals(image, tmp_path, caplog):
    """Guards V3-CONFIRM-5 at 548ac115: with ``[verifier].service = "target"``
    the PATH was judged and the refusal stubs made in ``main``; target's PATH
    named a stubs folder target does not have, and the hits were read in
    ``main``, so target's caught refusal scored 0."""
    names = {"main": _container(image), "target": _container(image)}
    try:
        env = _sandbox(names)
        main = SimpleNamespace(name=names["main"], env=env)
        target = SimpleNamespace(name=names["target"], env=env)
        for each in (main, target):
            _write(each, "/opt/probe/bin/cargo", "#!/bin/sh\nexit 0\n")
            _sh(each, "chmod 755 /opt/probe/bin/cargo")
        test_sh = (
            "#!/bin/sh\nif cargo test; then echo 1; else echo 0; fi"
            " > /logs/verifier/reward.txt\nexit 0\n"
        )
        _write(target, "/tests/test.sh", test_sh)
        # The pre-agent PATH inspectors a trial records for main and the
        # verifier's service (the build-config snapshot, merge wave 32b).
        blocked = lockdown._blocked_verifier_path_prefixes("agent", "/app")
        await lockdown._prepare_bash_env_inspector(env, blocked=blocked)
        await lockdown._prepare_bash_env_inspector(
            env, service="target", blocked=blocked
        )
        paths = _rollout_paths(tmp_path)
        task = _script_task(tmp_path, test_sh)
        task.config.verifier.service = "target"

        class TargetVerifier:
            execution_receipt = None
            test_return_code = None

            async def verify(self):
                # The task route, as verifier_core runs test.sh (PT's C8).
                with as_task_command(env):
                    done = await env.exec(
                        "/bin/sh /tests/test.sh",
                        env=task.config.verifier.env,
                        service="target",
                    )
                self.test_return_code = done.return_code
                paths.test_stdout_path.write_text(
                    (done.stdout or "") + (done.stderr or "")
                )
                reward = float(_sh(target, "cat /logs/verifier/reward.txt"))
                return SimpleNamespace(rewards={"reward": reward})

        rewards, error, _ = await _verify_rollout(
            env, task, paths, {}, _refusing_planes(TargetVerifier()), "agent", "/app"
        )
        made_in_main = _sh(main, f"ls -d /{REFUSAL_STUBS_PREFIX}* 2>/dev/null || true")
    finally:
        for name in names.values():
            _run(["docker", "rm", "-f", "-v", name])

    assert rewards == {"reward": 0.0}
    assert error is None
    assert "cargo (in /opt/probe/bin)" in caplog.text
    assert "Verifier scored 0" in caplog.text
    record = json.loads((paths.verifier_dir / "trusted-path.json").read_text())
    assert record["service"] == "target"
    assert made_in_main.strip() == ""
