"""Guards DT2-R12-1 at 458ddafa: a verifier on a backend that switches users
with su is the account that backend's launcher selects, which its spelling
does not decide: at 458ddafa's integ (a662659d) su read ``"0"`` as the
account named 0 (UID 1001 here); since FF-sw-launchers (45b1691b,
``sandbox._su.su_command``) a decimal is a UID found through getent, and
``"1001"`` reaches the account named 0. Setup must decide from the
measured UID before any root-only restriction, and preflight must not refuse
a verifier spelling whose UID depends on the backend.
"""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import subprocess
import uuid
from types import SimpleNamespace
from unittest import mock

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.task import keep_services
from tests.receipt_fakes import receipt_answer
from tests.test_check_run_plan import _isolate, _task  # noqa: F401
from tests.test_service_identity_docker import PYTHON

BUSYBOX = "bash@sha256:a54fb4422b18f05dd3107c36f39d67b26334fda7ec89f4126052b45e228e2f15"
SPELLING = "another spelling"
RESOLVE = "cannot resolve"

#: Trusted accounts in every image below: digit names are names under su.
PASSWD = (
    "root:x:0:0::/root:/bin/bash\n"
    "agent:x:1000:1000::/tmp:/bin/bash\n"
    "0:x:1001:1001::/tmp:/bin/bash\n"
    "00:x:1002:1002::/tmp:/bin/bash\n"
    "1000:x:0:0::/root:/bin/bash\n"
    "toor:x:0:0::/root:/bin/bash\n"
)
GROUP = "root:x:0:\ng1000:x:1000:\ng1001:x:1001:\ng1002:x:1002:\n"

#: (verifier selector, what setup gives under su beside a non-root agent):
#: "kept", "none" (graded, nothing kept), or a refusal's words.
SU_ROWS = [
    (None, "kept"),
    ("root", "kept"),
    (0, "kept"),
    ("0", SPELLING),  # getent passwd 0: root, measured UID 0
    ("00", SPELLING),
    ("agent", "none"),
    ("1000", "none"),  # getent passwd 1000: agent, measured non-root
    ("1001", "none"),  # the account named 0, measured UID 1001
    ("1002", "none"),  # the account named 00
    ("toor", SPELLING),
    ("0:0", RESOLVE),  # su_command takes a plain name or a decimal only
    (":1000", RESOLVE),
    ("root:root", RESOLVE),
    ("2000", RESOLVE),
    ("missing", RESOLVE),
]
#: Every verifier shape preflight could see, for every backend.
SHAPES = [None, "", "root", 0, "0", "000", "+0", "-00", "0:0", ":1000"]
SHAPES += ["root:root", "toor", "1000", "00", "agent", "checker"]
#: The shapes docker exec -u runs as UID 0 without reading the image.
DOCKER_ZERO = {0, "0", "00", "000", "+0", "-00", "0:0"}


def trial(verifier_user, service="main", agent_user="agent"):
    return SimpleNamespace(
        config=SimpleNamespace(
            metadata={"x-keep-services": [{"port": 8080}]},
            agent=SimpleNamespace(user=agent_user),
            verifier=SimpleNamespace(user=verifier_user, service=service),
        )
    )


@pytest.mark.parametrize("service", ["main", "grader"])
@pytest.mark.parametrize("user", SHAPES, ids=repr)
@pytest.mark.parametrize(
    "sandbox",
    [
        "docker",
        "remote-docker",
        "daytona",
        "modal",
        "apple-container",
        "agentcore",
        None,
    ],
)
def test_preflight_refuses_only_what_docker_fixes(sandbox, user, service):
    """Guards DT2-R12-1: a backend-dependent verifier spelling is deferred."""
    task = trial(user, service)
    for agent, sandbox_user in (("nop", "agent"), ("claude-agent-acp", "agent")):
        refusal = keep_services.static_refusal(task, agent, sandbox_user, sandbox)
        docker = sandbox in ("docker", "remote-docker")
        if not docker or user not in DOCKER_ZERO or (service == "main" and user == 0):
            assert refusal is None, (sandbox, user, service, refusal)
        else:
            assert refusal is not None and keep_services.KEY in refusal
            assert ("service 'grader'" if service == "grader" else SPELLING) in refusal
        # The oracle and a root agent are never refused before setup.
        assert (
            keep_services.static_refusal(task, "oracle", sandbox_user, sandbox) is None
        )
        assert keep_services.static_refusal(task, agent, None, sandbox) is None


@pytest.mark.parametrize("sandbox", ["daytona", "modal"])
@pytest.mark.parametrize("user", ['"0"', '"0:0"', '"root:root"', '"1000"'])
def test_tasks_check_defers_a_su_backends_verifier(tmp_path, sandbox, user):
    """Guards DT2-R12-1 through ``bench tasks check --sandbox``."""
    toml = (
        '[agent]\nuser = "agent"\n'
        "[metadata]\nx-keep-services = [{port = 8080}]\n"
        f"[verifier]\nuser = {user}\n"
    )
    task = _task(tmp_path / "task", toml)
    result = CliRunner().invoke(
        app,
        ["tasks", "check", str(task), "--agent", "nop", "--sandbox", sandbox],
    )
    text = " ".join(result.output.split())
    assert SPELLING not in text and "write user" not in text, result.output


class SuEnv:
    """A backend whose exec switches users the way Daytona (without DinD),
    Modal and AgentCore do (``sandbox._su.su_command``): a decimal or an
    integer is a UID found through getent, anything else an account name,
    and a selector that is neither is refused before su runs."""

    def __init__(self, passwd=PASSWD):
        self.users = [line.split(":") for line in passwd.splitlines()]

    def _named(self, name):
        return next((r for r in self.users if r[0] == name), None)

    async def exec(self, command, user=None, service="main", **kwargs):
        # Account reads cross as a receipt (its NUL separator).
        return receipt_answer(command, lambda read: self._answer(read, user))

    def _answer(self, command, user):
        if user is not None:
            from benchflow.sandbox._su import su_command

            su_command(command, user)  # raises ValueError as production does
            if str(user).isdigit():
                row = next((r for r in self.users if r[2] == str(int(user))), None)
                if row is None:
                    return SimpleNamespace(return_code=97, stdout="", stderr="")
                user = row[0]
        words = shlex.split(command)
        if words[:3] == ["su", "-s", "/bin/bash"]:  # the oracle's own su
            user, command = words[3], words[5]
        record = self._named("root" if user is None else user)
        if record is None:
            return SimpleNamespace(return_code=1, stdout="", stderr="su: no user")
        if "benchflow-id" in command:
            out = f"benchflow-id:{record[2]}:{record[3]}\n"
        elif command.startswith("cat /etc/passwd"):
            out = PASSWD + "\0" + GROUP
        elif command.startswith("id -u"):
            out = f"{record[2]}\n{record[3]}\n"
        else:
            out = ""
        return SimpleNamespace(return_code=0, stdout=out, stderr="")


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["oracle", "nop"])
@pytest.mark.parametrize("user,expected", SU_ROWS, ids=[repr(r[0]) for r in SU_ROWS])
async def test_su_backend_setup_uses_the_measured_verifier(user, expected, agent):
    """Guards DT2-R12-1: the measured UID decides before any spelling rule."""
    task = trial(user)
    env = SuEnv()
    # The oracle runs as its [agent] user with --sandbox-user none; another
    # agent runs as --sandbox-user agent. Both are UID 1000 here.
    inherit = agent == "oracle"
    prepare = keep_services.prepare_identities(env, task, "agent", inherit=inherit)
    if expected in (SPELLING, RESOLVE):
        with pytest.raises(RuntimeError, match=expected):
            await prepare
        return
    await prepare
    assert task._benchflow_service_identities[0].uid == 1000
    kept = keep_services.kept_by_identity(task)
    assert (kept is not None) == (expected == "kept"), (user, kept)


class _Process:
    def __init__(self, proc):
        self._proc = proc
        self.stdout = SimpleNamespace(read=SimpleNamespace(aio=self._out))
        self.stderr = SimpleNamespace(read=SimpleNamespace(aio=self._err))
        self.wait = SimpleNamespace(aio=self._wait)

    async def _out(self):
        return self._proc.stdout

    async def _err(self):
        return self._proc.stderr

    async def _wait(self):
        return self._proc.returncode


class _Container:
    """Modal's sandbox object, backed by a mount-free local container."""

    def __init__(self, name):
        self.name = name
        self.exec = SimpleNamespace(aio=self._exec)

    async def _exec(self, *argv, workdir=None, secrets=(), timeout=None):
        # Modal sets each Secret's values in the process's environment: the
        # exec's env and the loader names it clears (empty values).
        env = [
            arg
            for secret in secrets
            for k, v in secret.env_dict.items()
            for arg in ("-e", f"{k}={v}")
        ]
        proc = await asyncio.to_thread(
            subprocess.run,
            ["docker", "exec", *env, self.name, *argv],
            capture_output=True,
            text=True,
        )
        return _Process(proc)


@contextlib.contextmanager
def modal_container(image):
    from benchflow.sandbox.modal_impl import ModalSandbox

    name = "dt2-su-" + uuid.uuid4().hex[:12]
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--network",
            "none",
            "--label",
            "target=DT2",
            image,
            "sleep",
            "300",
        ],
        check=True,
        capture_output=True,
    )
    try:
        users = "".join(
            line + "\n" for line in PASSWD.splitlines() if not line.startswith("root:")
        )
        shadow = "".join(
            f"{line.split(':')[0]}:*:19000:0:99999:7:::\n"
            for line in PASSWD.splitlines()
            if not line.startswith("root:")
        )
        groups = "".join(
            line + "\n" for line in GROUP.splitlines() if not line.startswith("root:")
        )
        subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                "0:0",
                name,
                "/bin/sh",
                "-c",
                "[ -e /bin/bash ] || ln -s /usr/local/bin/bash /bin/bash; "
                f"printf %s '{users}' >> /etc/passwd && "
                f"printf %s '{shadow}' >> /etc/shadow && "
                f"printf %s '{groups}' >> /etc/group",
            ],
            check=True,
            capture_output=True,
        )
        env = ModalSandbox.__new__(ModalSandbox)
        env.default_user = None
        env._persistent_env = {}
        env._sandbox = _Container(name)
        # The Secret keeps its values where _Container reads them; a real
        # one would load them into Modal's servers.
        with mock.patch(
            "modal.Secret.from_dict",
            lambda env_dict: SimpleNamespace(env_dict=dict(env_dict)),
        ):
            yield env
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=True, capture_output=True)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("image", [BUSYBOX, PYTHON], ids=["busybox", "util-linux"])
async def test_real_su_backend_grades_a_numeric_named_verifier(image):
    """Guards DT2-R12-1 with Modal's production exec and real su."""
    with modal_container(image) as env:
        su = await env.exec('readlink -f "$(command -v su)"')
        assert ("busybox" in su.stdout) == (image == BUSYBOX), su.stdout
        for user, expected in SU_ROWS:
            for agent in ("oracle", "nop"):
                task = trial(user)
                vars(env).pop("_benchflow_launched_identities", None)
                prepare = keep_services.prepare_identities(
                    env, task, "agent", inherit=agent == "oracle"
                )
                if expected in (SPELLING, RESOLVE):
                    with pytest.raises(RuntimeError, match=expected):
                        await prepare
                    continue
                await prepare
                agent_id, verifier = task._benchflow_service_identities
                assert agent_id.uid == 1000, (user, agent)
                kept = keep_services.kept_by_identity(task)
                assert (kept is not None) == (expected == "kept"), (user, agent)
                # The verifier's own command runs as the measured account.
                ran = await env.exec("id -u", user=user)
                assert ran.stdout.strip() == str(verifier.uid), (user, ran)
