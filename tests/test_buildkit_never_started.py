"""A build BuildKit never started is built again (the helper's and a task's).

BuildKit's gateway forwarder waits 3 s for a build to register before it
answers the client's first call, ``Ping`` (moby/buildkit v0.26.2,
``control/gateway/gateway.go``, ``lookupForwarder``). On a loaded daemon the
build registers later, and ``docker build`` fails with ``NotFound: forwarding
Ping: no such job <id>`` before any step ran. Seen in the release loops under
load ~30 on 32 vCPUs: the trial was refused at sandbox start. Another
attempt is safe there, and only there: a step that failed is never built again.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _metadata_helper as helper_mod
from benchflow.sandbox import docker as docker_mod
from benchflow.sandbox.image_build import ImageBuildError, image_build_error

IMAGE_ID = "sha256:" + "ab" * 32
# The line `docker build --quiet` printed in run b-13 (release loop, VM 2).
NEVER_STARTED = "ERROR: failed to build: NotFound: forwarding Ping: no such job {job}\n"
# What run b-33 (release loop, VM 2) raised from the task image's build.
COMPOSE_NEVER_STARTED = (
    "Docker compose command failed for environment perf-f-task. Command: docker "
    "--config /tmp/cfg compose --project-name perf-f-task__ec64e688-95efac46 build. "
    "Stderr: NotFound: forwarding Ping: no such job 8ihioh8k3ncimyi06lc56jgbn"
)
RECIPE_FAILED = (
    'ERROR: failed to build: failed to solve: process "/bin/sh -c apk add '
    '--no-cache iptables=1.8.13-r0" did not complete successfully: exit code: 1\n'
)


class BuildAnswers:
    """The docker CLI for the helper's build: each ``build`` gives the next
    scripted stderr (``None``: the build succeeds and prints the image ID)."""

    def __init__(self, *answers: str | None) -> None:
        self.answers = list(answers)
        self.calls: list[list[str]] = []

    async def __call__(self, args: list[str], *, timeout: float, stdin=None):
        self.calls.append(list(args))
        if args[:1] == ["build"]:
            answer = self.answers.pop(0) if self.answers else None
            if answer is None:
                return SimpleNamespace(stdout=f"{IMAGE_ID}\n", stderr="", return_code=0)
            return SimpleNamespace(stdout="", stderr=answer, return_code=1)
        # docker info: no daemon ID, so no record is read or written.
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    def builds(self) -> list[list[str]]:
        return [c for c in self.calls if c[:1] == ["build"]]


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(helper_mod, "_READY", {})
    monkeypatch.setattr(helper_mod, "_FAILED", {})
    monkeypatch.setattr(helper_mod, "_BUILD_LOCKS", {})
    monkeypatch.setattr(helper_mod, "_PREBUILDS", {}, raising=False)
    # No waiting between tries here; the delays themselves are checked below.
    monkeypatch.setattr(
        helper_mod, "_BUILD_RETRY_DELAYS_SEC", (0.0, 0.0), raising=False
    )


@pytest.mark.asyncio
async def test_a_helper_build_buildkit_never_started_is_built_again():
    docker = BuildAnswers(NEVER_STARTED.format(job="o09ngx3tx486r7b6n42l4x4ln"), None)

    assert await helper_mod.ensure_image(docker, "local") == IMAGE_ID
    assert len(docker.builds()) == 2
    assert helper_mod.build_failure("local") == ""


@pytest.mark.asyncio
async def test_a_helper_recipe_that_failed_is_never_built_again():
    docker = BuildAnswers(RECIPE_FAILED, None)

    with pytest.raises(helper_mod.HelperUnavailable, match="exit code: 1"):
        await helper_mod.ensure_image(docker, "local")
    assert len(docker.builds()) == 1
    # Held for _RETRY_BUILD_SEC as before (a mirror that was down).
    with pytest.raises(helper_mod.HelperUnavailable, match="exit code: 1"):
        await helper_mod.ensure_image(docker, "local")
    assert len(docker.builds()) == 1


@pytest.mark.asyncio
async def test_the_helper_builds_a_bounded_number_of_times_and_keeps_each_reason():
    jobs = [
        "job1aaaaaaaaaaaaaaaaaaaa",
        "job2bbbbbbbbbbbbbbbbbbbb",
        "job3cccccccccccccccccccc",
    ]
    docker = BuildAnswers(*(NEVER_STARTED.format(job=j) for j in jobs), None)

    with pytest.raises(helper_mod.HelperUnavailable) as raised:
        await helper_mod.ensure_image(docker, "local")
    assert len(docker.builds()) == 3
    for job in jobs:
        assert job in str(raised.value)
    assert str(raised.value) == helper_mod.build_failure("local")


@pytest.mark.asyncio
async def test_a_helper_build_buildkit_never_started_does_not_hold_back_the_next_start():
    """A daemon too busy to start three builds in a row refuses this start;
    the next start builds again instead of reading that refusal for
    _RETRY_BUILD_SEC (no step ran: there is no mirror to wait for)."""
    docker = BuildAnswers(*(NEVER_STARTED.format(job=f"j{i}") for i in range(3)), None)

    with pytest.raises(helper_mod.HelperUnavailable):
        await helper_mod.ensure_image(docker, "local")
    assert await helper_mod.ensure_image(docker, "local") == IMAGE_ID
    assert len(docker.builds()) == 4


@pytest.mark.parametrize(
    "message",
    [
        NEVER_STARTED.format(job="xyz"),
        # Compose's own wording around the same BuildKit answer.
        "failed to solve: NotFound: forwarding Ping: no such job s0m3j0b\n",
    ],
)
def test_a_task_build_buildkit_never_started_is_retried_and_not_the_tasks(message):
    assert docker_mod._is_retryable_docker_build_error(message)
    assert image_build_error(message) is None


def test_a_task_build_step_that_failed_is_still_the_tasks():
    assert not docker_mod._is_retryable_docker_build_error(RECIPE_FAILED)
    assert image_build_error(RECIPE_FAILED) is not None


def _compose_box(*failures: str) -> tuple[docker_mod.DockerSandbox, list[list[str]]]:
    """A Docker sandbox whose compose ``build`` fails with each of *failures*
    in turn, then succeeds."""
    box = docker_mod.DockerSandbox.__new__(docker_mod.DockerSandbox)
    box.task_env_config = SimpleNamespace(build_timeout_sec=600)
    box.environment_name = "perf-f-task"
    box.logger = logging.getLogger("benchflow.test")
    calls: list[list[str]] = []
    pending = list(failures)

    async def compose(command, *, timeout_sec):
        calls.append(list(command))
        if pending:
            raise RuntimeError(pending.pop(0))

    box._run_docker_compose_command = compose  # type: ignore[method-assign]
    return box, calls


@pytest.mark.asyncio
async def test_a_task_image_build_buildkit_never_started_is_built_again(monkeypatch):
    monkeypatch.setattr(docker_mod, "_DOCKER_BUILD_RETRY_DELAYS_SEC", (0.0, 0.0))
    box, calls = _compose_box(COMPOSE_NEVER_STARTED)

    await box._run_docker_compose_build_attempts()

    assert calls == [["build"], ["build"]]


@pytest.mark.asyncio
async def test_a_task_image_build_whose_step_failed_is_built_once(monkeypatch):
    monkeypatch.setattr(docker_mod, "_DOCKER_BUILD_RETRY_DELAYS_SEC", (0.0, 0.0))
    box, calls = _compose_box(f"Docker compose command failed. Stderr: {RECIPE_FAILED}")

    with pytest.raises(ImageBuildError):
        await box._run_docker_compose_build_attempts()

    assert calls == [["build"]]
