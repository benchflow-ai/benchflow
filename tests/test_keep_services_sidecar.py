"""Guards DT2-CONF-10 at a2172ffe and the narrowed retention matrix.

Kept services are agent code alive while the verifier runs. A root verifier in
a service other than ``main`` gets its env on its shell's command line, which
a kept process reads through /proc when that service shares main's process
namespace (Compose ``pid: "service:main"``). Setup refuses that combination
before any agent runs, and refuses root spellings other than ``root``, ``0``
or none, instead of matching them.
"""

import contextlib
import subprocess
import uuid

import pytest

from benchflow.sandbox._base import SandboxConfig
from benchflow.sandbox._identities import Identity
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task import keep_services
from benchflow.task.verifier_core import VerifierOutputParseError
from tests.test_keep_services_verify import _verifier
from tests.test_service_identity_docker import ALPINE, PYTHON, RemoteServices, Services
from tests.test_service_identity_scope import sandbox, task

BACKENDS = ["docker", "remote", "dind"]
# Every spelling that resolves to UID 0 in the mock accounts but is not one
# of the supported three.
OTHER_ROOT_SPELLINGS = [
    "0",
    "000",
    "+0",
    "-00",
    "0:0",
    "root:root",
    "root:team",
    ":",
    ":1000",
    "toor",
    "toor:root",
]


def with_toor(accounts):
    for service in accounts:
        accounts[service] += "toor:x:0:0::/:/bin/sh\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("user", [None, "", "root", 0, "checker", "0", "toor"])
async def test_root_sidecar_verifier_with_kept_services_is_refused(
    tmp_path, backend, user
):
    """Guards CONF-10: no kept agent process runs beside a root sidecar."""
    env, accounts, configs, _ = sandbox(tmp_path, backend, grader_uid=0)
    with_toor(accounts)
    if user is None or user == "":
        configs["b" * 64]["User"] = "root"
    trial = task(user, "grader")
    with pytest.raises(RuntimeError, match=r"service 'grader'.*service = \"main\""):
        await keep_services.prepare_identities(env, trial, "agent")
    with pytest.raises(RuntimeError, match="service 'grader'"):
        keep_services.trial_keep(trial, "agent")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("user", ["checker", "2000", None])
async def test_nonroot_sidecar_verifier_still_grades(tmp_path, backend, user):
    """Guards CONF-8 under the narrowed policy: nothing kept, no refusal."""
    env, _, _, _ = sandbox(tmp_path, backend)
    trial = task(user, "grader")
    await keep_services.prepare_identities(env, trial, "agent")
    assert trial._benchflow_service_identities[1] == Identity(2000, 4000)
    assert keep_services.trial_keep(trial, "agent") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("user", [None, "", "root", 0])
async def test_main_root_verifier_keeps_services(tmp_path, backend, user):
    """The supported spellings: unset, ``root`` and the number 0."""
    env, _, _, _ = sandbox(tmp_path, backend)
    trial = task(user)
    await keep_services.prepare_identities(env, trial, "agent")
    assert keep_services.trial_keep(trial, "agent") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("user", OTHER_ROOT_SPELLINGS)
async def test_other_root_spellings_are_refused(tmp_path, backend, user):
    """Root by any other spelling is refused before the agent, not matched."""
    env, accounts, _, _ = sandbox(tmp_path, backend)
    with_toor(accounts)
    trial = task(user)
    with pytest.raises(RuntimeError, match=r"another spelling.*user = \"root\""):
        await keep_services.prepare_identities(env, trial, "agent")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("default", ["0", "000", "0:team", "toor"])
async def test_unset_user_inherits_any_root_default_in_main(tmp_path, backend, default):
    """An unset selector takes the trusted image's default, however spelled."""
    env, accounts, configs, _ = sandbox(tmp_path, backend)
    with_toor(accounts)
    configs["a" * 64]["User"] = default
    trial = task(None)
    await keep_services.prepare_identities(env, trial, "agent")
    assert trial._benchflow_service_identities[1].uid == 0
    assert keep_services.trial_keep(trial, "agent") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("user", ["0", "toor"])
async def test_other_root_spellings_without_kept_services_still_run(
    tmp_path, backend, user
):
    """The refusal is only for retention: a task keeping nothing is unchanged."""
    env, accounts, _, _ = sandbox(tmp_path, backend)
    with_toor(accounts)
    trial = task(user)
    trial.config.metadata = {}
    await keep_services.prepare_identities(env, trial, "agent")
    assert keep_services.trial_keep(trial, "agent") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["test", "clear", "policy"])
async def test_verifier_never_releases_kept_services_outside_main(tmp_path, step):
    """Guards CONF-10 in the verifier too, should setup ever be bypassed."""
    order: list = []
    verifier = _verifier(tmp_path, order)
    verifier._task.config.verifier.service = "grader"

    async def release(code_dir: str) -> None:
        order.append(("release", code_dir))

    verifier.before_test = release
    with pytest.raises(VerifierOutputParseError, match="service 'grader'"):
        if step == "test":
            await verifier._verify_test_script()
        elif step == "clear":
            await verifier._clear_output_dir("grader")
        else:
            verifier._keeps_services("grader")
    assert [kind for kind, _ in order if kind in ("release", "test")] == []
    assert verifier._keeps_services("main") is True


@contextlib.contextmanager
def shared_pid_services(tmp_path, backend):
    """main and a grader that joins main's PID namespace, both mount-free."""
    cls = Services if backend == "docker" else RemoteServices
    env = cls.__new__(cls)
    DockerSandbox.__init__(
        env,
        environment_dir=tmp_path,
        environment_name="dt2-sidecar",
        session_id="dt2-sidecar",
        rollout_paths=None,
        task_env_config=SandboxConfig(docker_image=PYTHON),
    )
    env._compose_project = "dt2-sidecar-" + uuid.uuid4().hex[:12]
    env.names = {}
    try:
        for service, image in (("main", PYTHON), ("grader", ALPINE)):
            name = f"{env.compose_project_name}-{service}"
            flags = (
                ["--pid", f"container:{env.names['main']}"]
                if service == "grader"
                else []
            )
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
                    f"com.docker.compose.service={service}",
                    *flags,
                    image,
                    "sleep",
                    "300",
                ],
                check=True,
                capture_output=True,
            )
            env.names[service] = name
        subprocess.run(
            ["docker", "exec", env.names["main"], "useradd", "-u", "1000", "agent"],
            check=True,
            capture_output=True,
        )
        yield env
    finally:
        for name in env.names.values():
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote"])
@pytest.mark.parametrize("user", [None, "root"])
async def test_shared_pid_root_sidecar_is_refused_before_agent(tmp_path, backend, user):
    """Guards CONF-10 with real containers that share one PID namespace."""
    with shared_pid_services(tmp_path, backend) as env:
        main_ns = await env.exec("readlink /proc/self/ns/pid")
        grader_ns = await env.exec("readlink /proc/self/ns/pid", service="grader")
        assert main_ns.stdout.strip() == grader_ns.stdout.strip()
        trial = task(user, "grader")
        with pytest.raises(RuntimeError, match="service 'grader'"):
            await keep_services.prepare_identities(env, trial, "agent")
        # Refused before any agent code: the agent user has no process.
        ps = await env.exec("ps -o user= -e 2>/dev/null || ps -e", service="grader")
        assert "agent" not in ps.stdout and "1000" not in ps.stdout
