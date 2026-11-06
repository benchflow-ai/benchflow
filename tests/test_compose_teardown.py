"""Compose teardown: kill at once, and delete only the images the trial built.

Teardown used ``compose down -t 5`` (``--rmi all`` with delete). The engine
bench (2026-10-16) saw SIGTERM to a ``sleep infinity`` PID 1 that ignores it,
then SIGKILL 5.01 s later, one container after another: 5 s per container on
every trial. ``--rmi all`` also deleted the operator's own ``ubuntu:24.04``
(a task's ``docker_image``), and with ``--remove-orphans`` Compose removes the
first tag (in name order) of each image the project built, which took the
verifier-recovery lease (``benchflow-recovery-lease:...`` sorts before
``bf__<env>``) off the original image.

``FakeDaemon`` below holds an image store and answers BenchFlow's compose and
docker calls the way Docker 29 with Compose 2.40.3 did on a build VM
(``compose down --rmi`` as in Compose's ``getServiceImagesToRemove``: custom
``image:`` names only with ``all``; with ``--remove-orphans`` the first tag of
every image labelled with the project; ``<project>-<service>`` for services
without ``image:``). Tests check what is left in the store.
"""

from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.daytona import _DaytonaDinD
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.remote_docker import (
    REMOTE_DOCKER_HOST_ENV,
    RemoteDockerSandbox,
)
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths
from tests import _fake_docker_cli

PROJECT_LABEL = "com.docker.compose.project"
SERVICE_LABEL = "com.docker.compose.service"
LEASE = "benchflow-recovery-lease:" + "0" * 32
# What every compose backend tears down with: SIGKILL to all at once, then
# down without a stop timeout (and without --rmi).
KILL = ["kill"]
DOWN = ["down", "--remove-orphans", "-t", "0"]


class FakeDaemon:
    """An image store and running containers for one compose project."""

    def __init__(self, project: str, services: dict[str, str | None]) -> None:
        # services: name -> custom ``image:`` (None: Compose's default name).
        self.project = project
        self.services = services
        self.images: dict[str, dict] = {}
        self.running = set(services)
        self.containers = set(services)
        self.events: list[tuple[str, str]] = []
        self.compose_calls: list[list[str]] = []
        self.cli_calls: list[list[str]] = []

    def add_image(
        self, image_id: str, tags: list[str], project: str | None = None, **labels
    ) -> None:
        if project is not None:
            labels[PROJECT_LABEL] = project
        self.images[image_id] = {"tags": list(tags), "labels": labels}

    def built(self, image_id: str, tag: str, service: str) -> None:
        self.add_image(image_id, [tag], self.project, **{SERVICE_LABEL: service})

    def tags(self) -> set[str]:
        return {t for image in self.images.values() for t in image["tags"]}

    def _find(self, ref: str) -> str | None:
        ref = ref if ":" in ref or ref.startswith("sha256") else f"{ref}:latest"
        for image_id, image in self.images.items():
            if ref == image_id or ref in image["tags"]:
                return image_id
        return None

    def _rm(self, ref: str) -> bool:
        image_id = self._find(ref)
        if image_id is None:
            return False
        image = self.images[image_id]
        if ref in image["tags"] or f"{ref}:latest" in image["tags"]:
            tag = ref if ref in image["tags"] else f"{ref}:latest"
            image["tags"].remove(tag)
        elif len(image["tags"]) > 1:
            return False  # by ID with several tags: Docker wants --force
        else:
            image["tags"].clear()
        if not image["tags"]:
            del self.images[image_id]
        return True

    async def compose(
        self, command: list[str], check: bool = True, timeout_sec: int | None = None
    ) -> ExecResult:
        self.compose_calls.append(list(command))
        if command[0] == "kill":
            for service in sorted(self.running):
                self.events.append(("SIGKILL", service))
            self.running.clear()
        elif command[0] == "down":
            grace = command[command.index("-t") + 1] if "-t" in command else "10"
            for service in sorted(self.running):
                if grace != "0":
                    self.events.append((f"SIGTERM, wait {grace} s", service))
                self.events.append(("SIGKILL", service))
            self.running.clear()
            self.containers.clear()
            if "--rmi" in command:
                self._compose_rmi(
                    command[command.index("--rmi") + 1],
                    "--remove-orphans" in command,
                )
        return ExecResult(stdout="", stderr="", return_code=0)

    def _compose_rmi(self, mode: str, orphans: bool) -> None:
        refs: list[str] = []
        local = [s for s, image in self.services.items() if image is None]
        for image in self.services.values():
            if image is not None and mode == "all" and self._find(image):
                refs.append(image)
        for image in list(self.images.values()):
            if image["labels"].get(PROJECT_LABEL) != self.project or not image["tags"]:
                continue
            if orphans or image["labels"].get(SERVICE_LABEL) in local:
                refs.append(sorted(image["tags"])[0])
        for service in local:
            if self._find(f"{self.project}-{service}"):
                refs.append(f"{self.project}-{service}")
        for ref in dict.fromkeys(refs):
            self._rm(ref)

    async def cli(
        self, args: list[str], check: bool = True, *, raw_output=False
    ) -> ExecResult:
        self.cli_calls.append(list(args))
        if args[:2] == ["image", "ls"]:
            wanted = args[args.index("--filter") + 1].split("=", 1)[1]
            key, _, value = wanted.partition("=")
            lines = [
                image_id if "--quiet" in args else f"{image_id} {tag}"
                for image_id, image in self.images.items()
                if image["labels"].get(key) == value
                for tag in (image["tags"] or ["<none>:<none>"])
            ]
            return ExecResult(stdout="\n".join(lines), stderr="", return_code=0)
        if args[:2] == ["image", "inspect"]:
            infos = [
                {"Id": ref, "RepoTags": self.images[ref]["tags"], "RepoDigests": []}
                for ref in args[2:]
                if ref in self.images
            ]
            return ExecResult(stdout=json.dumps(infos), stderr="", return_code=0)
        if args[:2] == ["image", "tag"]:
            image_id = self._find(args[2])
            if image_id is None:
                return ExecResult(stdout="", stderr="No such image", return_code=1)
            self.images[image_id]["tags"].append(args[3])
            return ExecResult(stdout="", stderr="", return_code=0)
        if args[:2] == ["image", "rm"]:
            missing = [
                ref
                for ref in args[2:]
                if not ref.startswith("--") and not self._rm(ref)
            ]
            return ExecResult(
                stdout="", stderr=" ".join(missing), return_code=1 if missing else 0
            )
        return ExecResult(stdout="", stderr="", return_code=0)


def _docker_sandbox(
    tmp_path: Path,
    services: dict[str, str | None] | None = None,
    *,
    docker_image: str | None = None,
    keep_containers: bool = False,
) -> tuple[DockerSandbox, FakeDaemon]:
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    sandbox = DockerSandbox(
        environment_dir=environment,
        environment_name="hello",
        session_id="hello__abc",
        rollout_paths=paths,
        task_env_config=SandboxConfig(docker_image=docker_image),
        keep_containers=keep_containers,
    )
    main = docker_image if docker_image else sandbox._env_vars.main_image_name
    daemon = FakeDaemon(
        sandbox.compose_project_name, {"main": main, **(services or {})}
    )
    sandbox._use_prebuilt = docker_image is not None
    sandbox._compose_up_attempted = True
    sandbox._chown_to_host_user = AsyncMock()
    sandbox._run_docker_compose_command = daemon.compose
    sandbox._docker_cli = daemon.cli
    sandbox._built_image_tags.update(
        f"{sandbox.compose_project_name}-{service}:latest"
        if image is None
        else image
        if ":" in image.rsplit("/", 1)[-1]
        else f"{image}:latest"
        for service, image in (services or {}).items()
    )
    return sandbox, daemon


# --- kill at once ------------------------------------------------------------


async def test_teardown_kills_every_container_at_once_without_a_stop_timeout(
    tmp_path,
):
    sandbox, daemon = _docker_sandbox(tmp_path, {"api": None})
    daemon.built("sha256:main", "bf__hello:latest", "main")

    await sandbox.stop(delete=True)

    # One SIGKILL per container, all from one `compose kill`, and no SIGTERM
    # grace waited out.
    assert daemon.events == [("SIGKILL", "api"), ("SIGKILL", "main")]
    assert daemon.compose_calls[0] == KILL
    # No `rm --volumes`: Docker would take inherited volumes too (r2).
    down = daemon.compose_calls[1]
    assert down[0] == "down"
    assert down[down.index("-t") + 1] == "0"
    assert not daemon.containers


async def test_teardown_without_delete_also_kills_at_once(tmp_path):
    sandbox, daemon = _docker_sandbox(tmp_path)
    daemon.built("sha256:main", "bf__hello:latest", "main")

    await sandbox.stop(delete=False)

    assert daemon.events == [("SIGKILL", "main")]
    assert not daemon.containers
    # Without delete, nothing is removed but the containers and the network.
    assert daemon.tags() == {"bf__hello:latest"}
    assert "--volumes" not in daemon.compose_calls[-1]


async def test_nothing_started_means_no_kill(tmp_path):
    sandbox, daemon = _docker_sandbox(tmp_path)
    sandbox._compose_up_attempted = False

    await sandbox.stop(delete=True)

    assert [c[0] for c in daemon.compose_calls] == ["down"]


# --- only our own images -------------------------------------------------------


async def test_a_pulled_task_image_stays_after_teardown(tmp_path):
    """The engine bench's case: a task with ``docker_image = "ubuntu:24.04"``."""
    sandbox, daemon = _docker_sandbox(tmp_path, docker_image="ubuntu:24.04")
    daemon.add_image("sha256:ubuntu", ["ubuntu:24.04"])

    await sandbox.stop(delete=True)

    assert "ubuntu:24.04" in daemon.tags()
    assert not daemon.containers


async def test_a_service_on_a_pulled_image_keeps_it_after_teardown(tmp_path):
    sandbox, daemon = _docker_sandbox(tmp_path, {"db": "ubuntu:24.04"})
    daemon.built("sha256:main", "bf__hello:latest", "main")
    daemon.add_image("sha256:ubuntu", ["ubuntu:24.04"])

    await sandbox.stop(delete=True)

    # U10-CONFIRM-3: shared names stay; deleting by ID could take a new tag.
    assert daemon.tags() == {"ubuntu:24.04", "bf__hello:latest"}


async def test_built_images_are_still_deleted_with_delete(tmp_path):
    sandbox, daemon = _docker_sandbox(
        tmp_path, {"api": None, "web": "example.test/web:1"}
    )
    project = sandbox.compose_project_name
    daemon.built("sha256:main", "bf__hello:latest", "main")
    daemon.built("sha256:api", f"{project}-api:latest", "api")
    daemon.built("sha256:web", "example.test/web:1", "web")
    # A build whose tag a later build of the same name took.
    daemon.add_image("sha256:old", [], project, **{SERVICE_LABEL: "main"})
    # Guards U10-CONFIRM-3 after 22672877: builds can be collected once their
    # shared names have moved, using only private temporary cleanup aliases.
    for image in daemon.images.values():
        image["tags"].clear()

    await sandbox.stop(delete=True)

    assert daemon.images == {}


async def test_another_trials_build_is_not_ours(tmp_path):
    sandbox, daemon = _docker_sandbox(tmp_path)
    daemon.built("sha256:main", "bf__hello:latest", "main")
    daemon.add_image(
        "sha256:other",
        ["bf__other:latest"],
        "other__xyz-12345678",
        **{SERVICE_LABEL: "main"},
    )

    await sandbox.stop(delete=True)

    assert daemon.tags() == {"bf__hello:latest", "bf__other:latest"}


async def test_the_recovery_lease_keeps_the_original_image(tmp_path):
    """The lease tag sorts before ``bf__<env>``; teardown must not take it."""
    sandbox, daemon = _docker_sandbox(tmp_path)
    daemon.add_image(
        "sha256:main",
        ["bf__hello:latest", LEASE],
        sandbox.compose_project_name,
        **{SERVICE_LABEL: "main"},
    )

    await sandbox.stop(delete=True)

    assert daemon.tags() == {"bf__hello:latest", LEASE}


async def test_a_lease_that_is_the_builds_only_tag_stays(tmp_path):
    """Another trial of the same environment built ``bf__hello`` later and
    took the tag: the lease is all that is left on this trial's build, and
    the rollout releases it after scoring, not teardown."""
    sandbox, daemon = _docker_sandbox(tmp_path)
    daemon.add_image(
        "sha256:main",
        [LEASE],
        sandbox.compose_project_name,
        **{SERVICE_LABEL: "main"},
    )

    await sandbox.stop(delete=True)

    assert daemon.tags() == {LEASE}


async def test_a_recovery_sandbox_removes_no_image_but_its_own_volumes(tmp_path):
    """The lease is the parent's image; the volumes carry the recovery
    project's own label (DOCKER-ANON-VOLUME-LEAK)."""
    sandbox, daemon = _docker_sandbox(tmp_path)
    daemon.add_image(
        "sha256:main",
        [LEASE],
        sandbox.compose_project_name,
        **{SERVICE_LABEL: "main"},
    )
    sandbox._recovery_baseline = SimpleNamespace(image_id="sha256:main")

    await sandbox.stop(delete=True)

    assert daemon.tags() == {LEASE}
    assert not any(c[:2] == ["image", "rm"] for c in daemon.cli_calls)
    label = f"label=com.docker.compose.project={sandbox.compose_project_name}"
    assert ["volume", "ls", "-q", "--filter", label] in daemon.cli_calls
    assert "--volumes" not in daemon.compose_calls[-1]
    assert "--rmi" not in daemon.compose_calls[-1]


TRIAL_VOLUME = "bfvol1_main_0"


def _labelled_cli(daemon: FakeDaemon, project: str):
    """The fake daemon's CLI, holding one volume named for the trial with
    *project*'s label (and no other volume)."""

    async def cli(args, check=True, *, raw_output=False):
        if args[:2] == ["volume", "ls"]:
            daemon.cli_calls.append(list(args))
            label = f"label=com.docker.compose.project={project}"
            listed = f"{project}_{TRIAL_VOLUME}" if label in args else ""
            return ExecResult(stdout=listed, stderr="", return_code=0)
        return await daemon.cli(args, check, raw_output=raw_output)

    return cli


async def test_teardown_reads_no_container_to_find_the_volumes(tmp_path):
    """DOCKER-ANON-VOLUME-LEAK r3: the volumes carry the trial's label from
    their creation, so teardown neither lists nor inspects containers, for a
    completed start or one that ended early."""
    sandbox, daemon = _docker_sandbox(tmp_path)
    project = sandbox.compose_project_name
    sandbox._docker_cli = _labelled_cli(daemon, project)

    await sandbox.stop(delete=True)

    assert not any(c[0] in {"ps", "inspect"} for c in daemon.cli_calls)
    assert ["volume", "rm", f"{project}_{TRIAL_VOLUME}"] in daemon.cli_calls


async def test_the_force_kill_fallback_removes_the_labelled_volumes(tmp_path):
    sandbox, daemon = _docker_sandbox(tmp_path)
    project = sandbox.compose_project_name
    sandbox._docker_cli = _labelled_cli(daemon, project)

    async def wedged(command, check=True, timeout_sec=None):
        daemon.compose_calls.append(list(command))
        raise RuntimeError(f"Command timed out after {timeout_sec} seconds")

    sandbox._run_docker_compose_command = wedged
    sandbox._force_kill_project = AsyncMock()

    await sandbox.stop(delete=True)

    sandbox._force_kill_project.assert_awaited_once()
    assert ["volume", "rm", f"{project}_{TRIAL_VOLUME}"] in daemon.cli_calls


async def test_a_snapshot_with_blanked_labels_stays(tmp_path):
    sandbox, daemon = _docker_sandbox(tmp_path)
    daemon.built("sha256:main", "bf__hello:latest", "main")
    daemon.add_image(
        "sha256:snap", ["bf-snap-hello-1:latest"], "", **{SERVICE_LABEL: ""}
    )

    await sandbox.stop(delete=True)

    assert daemon.tags() == {"bf__hello:latest", "bf-snap-hello-1:latest"}


async def test_kept_containers_stop_as_before(tmp_path):
    sandbox, daemon = _docker_sandbox(tmp_path, keep_containers=True)
    daemon.built("sha256:main", "bf__hello:latest", "main")

    await sandbox.stop(delete=True)

    assert daemon.compose_calls == [["stop", "-t", "5"]]
    assert daemon.cli_calls == []
    assert daemon.tags() == {"bf__hello:latest"}


async def test_a_wedged_kill_falls_through_to_the_force_kill(tmp_path):
    sandbox, daemon = _docker_sandbox(tmp_path)

    async def wedged(command, check=True, timeout_sec=None):
        daemon.compose_calls.append(list(command))
        raise RuntimeError(f"Command timed out after {timeout_sec} seconds")

    sandbox._run_docker_compose_command = wedged
    sandbox._force_kill_project = AsyncMock()

    await sandbox.stop(delete=True)

    assert daemon.compose_calls == [["kill"]]
    sandbox._force_kill_project.assert_awaited_once()


# --- one teardown for every compose backend ------------------------------------


def _remote_sandbox(tmp_path: Path) -> RemoteDockerSandbox:
    environment = tmp_path / "task" / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM alpine:3.20\n")
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    return RemoteDockerSandbox(
        environment_dir=environment,
        environment_name="remote-task",
        session_id="remote-task__abc123",
        rollout_paths=paths,
        task_env_config=SandboxConfig(),
    )


def _compose_subcommand(argv: list[str]) -> list[str]:
    rest = argv[1:]
    while rest and rest[0].startswith("-"):
        flag = rest.pop(0)
        if flag in ("--project-name", "-p", "--project-directory", "-f"):
            rest.pop(0)
    return rest


async def test_remote_docker_tears_down_with_the_same_commands(tmp_path, monkeypatch):
    for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY"):
        monkeypatch.delenv(name, raising=False)
    fake = _fake_docker_cli.install(tmp_path, monkeypatch)
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "ssh://builder@builder.example.test")
    sandbox = _remote_sandbox(tmp_path)
    await sandbox.start(force_build=False)
    started = len(fake.argvs())

    await sandbox.stop(delete=True)

    argvs = fake.argvs()[started:]
    compose = [_compose_subcommand(a) for a in argvs if a[0] == "compose"]
    assert compose[0] == KILL
    assert compose[1] == DOWN
    label = f"label=com.docker.compose.project={sandbox.compose_project_name}"
    assert any(a[:2] == ["image", "ls"] and label in a for a in argvs)


async def test_daytona_dind_tears_down_with_the_same_commands():
    strategy = _DaytonaDinD.__new__(_DaytonaDinD)
    env = SimpleNamespace(
        logger=logging.getLogger("test.compose.teardown"),
        _sandbox=object(),
        _stop_sandbox=AsyncMock(),
        _client_manager=object(),
        session_id="dind-project",
    )
    strategy._env = env
    strategy._compose_file_flags = lambda: ["-f", "/benchflow/compose/a.yaml"]
    strategy._compose_env_vars = lambda: {}
    commands: list[str] = []

    async def vm_exec(command, cwd=None, env=None, timeout_sec=None, **kwargs):
        commands.append(command)
        return ExecResult(stdout="", stderr="", return_code=0)

    strategy._vm_exec = vm_exec

    await strategy.stop(delete=True)

    assert len(commands) == 1
    kill, down = (shlex.split(part) for part in commands[0].split("; "))
    assert _compose_subcommand(kill[1:]) == KILL
    assert _compose_subcommand(down[1:]) == DOWN
    env._stop_sandbox.assert_awaited_once()
