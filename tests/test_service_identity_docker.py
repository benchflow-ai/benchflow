"""Guards DT2-CONF-8/9 at e28ebc01 with real trusted service accounts."""

import asyncio
import contextlib
import os
import subprocess
import uuid

import pytest

from benchflow.sandbox._base import ExecResult, SandboxConfig
from benchflow.sandbox._identities import Identity
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from benchflow.task import keep_services
from tests.test_service_identity_scope import task

ALPINE = (
    "alpine@sha256:4bcff63911fcb4448bd4fdacec207030997caf25e9bea4045fa6c8c44de311d1"
)
PYTHON = (
    "python@sha256:5560e9ab8709f459489e5b8aa696eda8a07ef821e14bb122be62d91234bfa98b"
)


class Services(DockerSandbox):
    """Production exec/config with mount-free containers as Compose services."""

    async def _docker_cli(self, args, check=True, *, raw_output=False, unset_env=()):
        proc = await asyncio.to_thread(
            subprocess.run,
            ["docker", *args],
            capture_output=True,
            text=True,
            check=check,
            env={k: v for k, v in os.environ.items() if k not in unset_env},
        )
        return ExecResult(
            return_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr
        )

    async def _run_docker_compose_command(
        self, argv, check=True, timeout_sec=None, *, unset_env=(), **kwargs
    ):
        assert argv[0] == "exec"
        index = next(i for i, value in enumerate(argv) if value in self.names)
        flags = [arg for arg in argv[1:index] if arg != "-T"]
        return await self._docker_cli(
            ["exec", *flags, self.names[argv[index]], *argv[index + 1 :]],
            check=check,
            unset_env=unset_env,
        )


class RemoteServices(Services, RemoteDockerSandbox):
    """Exercise the remote backend's inherited identity transport locally."""


@contextlib.contextmanager
def services(tmp_path, backend, family):
    cls = Services if backend == "docker" else RemoteServices
    env = cls.__new__(cls)
    DockerSandbox.__init__(
        env,
        environment_dir=tmp_path,
        environment_name="dt2-service-identity",
        session_id="dt2-service-identity",
        rollout_paths=None,
        task_env_config=SandboxConfig(docker_image=ALPINE),
    )
    env._compose_project = "dt2-scope-" + uuid.uuid4().hex[:12]
    env.names = {}
    try:
        for service in ["main"] if family == "bashless" else ["main", "grader"]:
            name = env.compose_project_name + "-" + service
            image = ALPINE if service == "grader" or family == "bashless" else PYTHON
            # Config.User must exist before startup. Numeric User can be set
            # before the trusted passwd entry is installed via root exec.
            flags = ["--user", "2000:4000"] if service == "grader" else []
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
            if service == "grader":
                setup = "addgroup -g 4000 team; adduser -D -u 2000 -G team checker"
            elif family == "bashless":
                setup = "addgroup -g 3000 team; adduser -D -u 1000 -G team agent"
            else:
                setup = "groupadd -g 3000 team; useradd -u 1000 -g team agent"
                if family == "conflicting":
                    setup += "; printf 'checker:x:0:0::/:/bin/sh\\n' >> /etc/passwd"
            subprocess.run(
                [
                    "docker",
                    "exec",
                    "-u",
                    "0:0",
                    name,
                    "/bin/sh",
                    "-c",
                    setup + "; mkdir -p /app; chmod 777 /app",
                ],
                check=True,
                capture_output=True,
            )
        yield env
    finally:
        for name in env.names.values():
            subprocess.run(
                ["docker", "rm", "-f", name], check=True, capture_output=True
            )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote"])
@pytest.mark.parametrize("family", ["selected", "inherited", "conflicting", "bashless"])
async def test_trusted_services_grade_without_bash_requirement(
    tmp_path, backend, family
):
    """Guards CONF-8/9: valid accounts and POSIX graders remain usable."""
    with services(tmp_path, backend, family) as env:
        verifier_service = "main" if family == "bashless" else "grader"
        user = (
            "root"
            if family == "bashless"
            else (None if family == "inherited" else "checker")
        )
        trial = task(user, verifier_service)
        await keep_services.prepare_identities(env, trial, "agent")
        assert trial._benchflow_service_identities[0] == Identity(1000, 3000)
        expected = Identity(0, 0) if family == "bashless" else Identity(2000, 4000)
        assert trial._benchflow_service_identities[1] == expected
        assert bool(keep_services.trial_keep(trial, "agent")) == (family == "bashless")
        assert (
            await env.exec("printf dummy-answer > /app/answer", user="1000:3000")
        ).return_code == 0
        assert (await env.exec("cat /app/answer")).stdout == "dummy-answer"
        assert (
            await env.exec(
                "printf 0",
                user=f"{expected.uid}:{expected.gid}",
                service=verifier_service,
            )
        ).stdout == "0"
