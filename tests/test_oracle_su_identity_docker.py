"""Guards DT2-R11-1 at b2b50251 with real BusyBox and util-linux su.

An oracle's ``[agent] user`` reaches ``su`` as an account name, digits
included: ``"1000"`` is the account named 1000, which a trusted image may give
UID 0. Setup must measure the account through the oracle's own su, never read
a UID from the selector's spelling, and check must defer it.
"""

import contextlib
import shlex
import subprocess
import uuid
from types import SimpleNamespace

import pytest

from benchflow.sandbox._base import SandboxConfig
from benchflow.sandbox._identities import Identities, Identity
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task import keep_services
from tests.test_service_identity_docker import PYTHON, RemoteServices, Services

#: Alpine with Bash: su is BusyBox's.
BUSYBOX = "bash@sha256:a54fb4422b18f05dd3107c36f39d67b26334fda7ec89f4126052b45e228e2f15"
#: Debian: su is util-linux's (PAM).
UTIL_LINUX = PYTHON

#: _run_oracle's own su form, spelled out so this file runs before the fix.
PROBE = 'printf \'benchflow-id:%s:%s\\n\' "$(id -u)" "$(id -g)"'


def solve_sh_su(account, command):
    return f"su -s /bin/bash {shlex.quote(account)} -c {shlex.quote(command)}"


#: Trusted accounts; every name below is what su receives.
ACCOUNTS = {
    "1000": Identity(0, 0),
    "0": Identity(1001, 1001),
    "00": Identity(1002, 1002),
    "toor": Identity(0, 0),
    "agent": Identity(1000, 1000),
    "root": Identity(0, 0),
}
PASSWD = (
    "1000:x:0:0::/root:/bin/bash\n"
    "0:x:1001:1001::/tmp:/bin/bash\n"
    "00:x:1002:1002::/tmp:/bin/bash\n"
    "toor:x:0:0::/root:/bin/bash\n"
    "agent:x:1000:1000::/tmp:/bin/bash\n"
)


@contextlib.contextmanager
def container(tmp_path, image, backend):
    cls = Services if backend == "docker" else RemoteServices
    env = cls.__new__(cls)
    DockerSandbox.__init__(
        env,
        environment_dir=tmp_path,
        environment_name="dt2-oracle-su",
        session_id="dt2-oracle-su",
        rollout_paths=None,
        task_env_config=SandboxConfig(docker_image=image),
    )
    env._compose_project = "dt2-su-" + uuid.uuid4().hex[:12]
    name = env.compose_project_name + "-main"
    env.names = {"main": name}
    try:
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
                "--label",
                f"com.docker.compose.project={env.compose_project_name}",
                "--label",
                "com.docker.compose.service=main",
                image,
                "sleep",
                "300",
            ],
            check=True,
            capture_output=True,
        )
        shadow = "".join(
            f"{line.split(':')[0]}:*:19000:0:99999:7:::\n"
            for line in PASSWD.splitlines()
        )
        groups = "".join(f"g{gid}:x:{gid}:\n" for gid in (1000, 1001, 1002))
        subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                "0:0",
                name,
                "/bin/sh",
                "-c",
                # The official bash image keeps it in /usr/local/bin; an Alpine
                # task that adds bash with apk has /bin/bash, as su -s needs.
                "[ -e /bin/bash ] || ln -s /usr/local/bin/bash /bin/bash; "
                f"printf %s '{PASSWD}' >> /etc/passwd && "
                f"printf %s '{shadow}' >> /etc/shadow && "
                f"printf %s '{groups}' >> /etc/group",
            ],
            check=True,
            capture_output=True,
        )
        yield env
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=True, capture_output=True)


def oracle_task(account, verifier_user=None):
    return SimpleNamespace(
        config=SimpleNamespace(
            metadata={"x-keep-services": [{"port": 8080}]},
            agent=SimpleNamespace(user=account),
            verifier=SimpleNamespace(user=verifier_user, service="main"),
        )
    )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote"])
@pytest.mark.parametrize("image", [BUSYBOX, UTIL_LINUX], ids=["busybox", "util-linux"])
async def test_oracle_account_is_the_one_su_selects(tmp_path, image, backend):
    """Guards DT2-R11-1: every launcher of solve.sh agrees with setup."""
    with container(tmp_path, image, backend) as env:
        su = await env.exec('readlink -f "$(command -v su)"')
        assert ("busybox" in su.stdout) == (image == BUSYBOX), su.stdout
        for account, expected in ACCOUNTS.items():
            # What solve.sh itself would run as, through _run_oracle's su.
            ran = await env.exec(solve_sh_su(account, PROBE))
            assert f"benchflow-id:{expected.uid}:{expected.gid}" in ran.stdout, (
                account,
                ran.stdout,
            )
            for verifier_user in (None, "root", 0, "0"):
                trial = oracle_task(account, verifier_user)
                # Check and the plan gate defer every oracle account.
                assert keep_services.static_refusal(trial, "oracle", None) is None
                assert keep_services.static_refusal(trial, "oracle", "agent") is None
                if expected.uid and verifier_user == "0":
                    with pytest.raises(RuntimeError, match="another spelling"):
                        await keep_services.prepare_identities(
                            env, trial, account, inherit=True
                        )
                    continue
                await keep_services.prepare_identities(
                    env, trial, account, inherit=True
                )
                agent, verifier = trial._benchflow_service_identities
                assert agent == expected, account
                assert verifier.uid == 0
                kept = keep_services.kept_by_identity(trial)
                assert (kept is not None) == bool(expected.uid), account
        # The spelling would have said UID 1000; su says root.
        table = vars(env)["_benchflow_identities"]
        assert isinstance(table, Identities)
        assert table.resolve("1000").uid == 1000
        for absent in ("2000", "+0", "missing"):
            with pytest.raises(RuntimeError, match="cannot resolve"):
                await keep_services.prepare_identities(
                    env, oracle_task(absent, "root"), absent, inherit=True
                )
