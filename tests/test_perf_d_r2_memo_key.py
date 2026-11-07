"""PERF-D rounds 2 and 3: the kept Compose check answers as a fresh check would.

The key covers what ``check_compose`` reads: the whole task folder that
staging copies and refuses on (not only ``environment/``; names by their raw
bytes, links by where they land), the operator's docker config.json and
current context, and the Compose plugin that runs. With a named folder in
play nothing is kept. A check that could not run Compose, that staging found
changing, or whose folder changed while it was checked is not kept. The
factory's early prebuilt overlay goes with a sandbox dropped without stop(),
and stays with kept containers.
"""

from __future__ import annotations

import gc
import json
import os
import re
import sys
from pathlib import Path

import pytest

from benchflow.task import compose_env

MAIN = "services:\n  main:\n    image: fixture-main:latest\n"


def _task(root: Path) -> Path:
    environment = root / "task" / "environment"
    environment.mkdir(parents=True)
    (environment / "docker-compose.yaml").write_text(MAIN)
    (environment / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (root / "task" / "tests").mkdir()
    (root / "task" / "tests" / "test.sh").write_text("exit 0\n")
    return environment


def _project_directory(command: list[str]) -> str:
    return command[command.index("--project-directory") + 1]


def _main(**extra) -> dict:
    return {"services": {"main": {"image": "fixture-main:latest", **extra}}}


@pytest.fixture(autouse=True)
def own_docker_config(tmp_path, monkeypatch):
    """The key reads the docker config folder: the developer's own (a
    ``docker context use`` during the run) must not change it."""
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "own-docker-config"))


@pytest.fixture
def resolve(monkeypatch):
    """Fake ``docker compose ... config`` (no daemon): the project each call
    resolves is ``project(command)``."""
    state = {"project": lambda command: _main(), "calls": 0}

    def fake(command, env, *, project_only=False):
        state["calls"] += 1
        return compose_env._Resolved(
            project=state["project"](command), variables={}, environment=frozenset()
        )

    monkeypatch.setattr(compose_env, "_run_config", fake)
    return state


def _shown(check) -> list[str]:
    # Each check stages its own private copy: its name is not the answer.
    return [re.sub(r"benchflow-stage-[^/]+", "STAGE", r.shown) for r in check.refused]


def _fresh(environment: Path, *args, **kwargs) -> list[str]:
    return _shown(compose_env._check_compose(environment, *args, **kwargs))


def test_a_link_out_added_under_tests_after_a_pass_is_refused(tmp_path, resolve):
    environment = _task(tmp_path)
    assert compose_env.check_compose(environment).refused == []

    (environment.parent / "tests" / "outside").symlink_to("/etc")
    refused = _shown(compose_env.check_compose(environment))
    assert refused and any("tests/outside" in r for r in refused)
    assert refused == _fresh(environment)


def test_a_link_out_under_tests_removed_after_a_refusal_passes(tmp_path, resolve):
    environment = _task(tmp_path)
    link = environment.parent / "tests" / "outside"
    link.symlink_to("/etc")
    assert compose_env.check_compose(environment).refused

    link.unlink()
    assert compose_env.check_compose(environment).refused == []
    assert _fresh(environment) == []


def test_a_change_anywhere_in_the_task_checks_again(tmp_path, resolve):
    environment = _task(tmp_path)
    compose_env.check_compose(environment)
    compose_env.check_compose(environment)
    assert resolve["calls"] == 1
    (environment.parent / "solution").mkdir()
    (environment.parent / "solution" / "solve.sh").write_text("true\n")
    compose_env.check_compose(environment)
    assert resolve["calls"] == 2
    (environment.parent / "tests" / "test.sh").write_text("exit 1\n")
    compose_env.check_compose(environment)
    assert resolve["calls"] == 3
    compose_env.check_compose(environment)
    assert resolve["calls"] == 3


@pytest.mark.skipif(sys.platform != "linux", reason="the filesystem proof is Linux's")
def test_a_saved_docker_context_after_a_pass_is_refused(tmp_path, resolve, monkeypatch):
    """``docker context use <remote>`` writes config.json's currentContext:
    the task's bind mount is then a path on another machine."""
    for name in ("DOCKER_HOST", "DOCKER_CONTEXT"):
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "docker-config"
    config.mkdir()
    monkeypatch.setenv("DOCKER_CONFIG", str(config))
    environment = _task(tmp_path)
    resolve["project"] = lambda command: _main(
        volumes=[
            {
                "type": "bind",
                "source": _project_directory(command),
                "target": "/srv",
                "read_only": True,
            }
        ]
    )

    assert compose_env.check_compose(environment).refused == []

    (config / "config.json").write_text(json.dumps({"currentContext": "elsewhere"}))
    refused = _shown(compose_env.check_compose(environment))
    assert refused and refused == _fresh(environment)

    (config / "config.json").write_text(json.dumps({}))
    assert compose_env.check_compose(environment).refused == []


def test_another_compose_plugin_checks_again(tmp_path, monkeypatch):
    """The Compose binary is replaced (an upgrade) in a running process: its
    answer is the new one's."""
    for name in ("DOCKER_HOST", "DOCKER_CONTEXT"):
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "docker-config"
    (config / "cli-plugins").mkdir(parents=True)
    plugin = config / "cli-plugins" / "docker-compose"
    plugin.write_text("compose v1\n")
    monkeypatch.setenv("DOCKER_CONFIG", str(config))

    def fake(command, env, *, project_only=False):
        if "v2" in plugin.read_text():
            return compose_env._Resolved(error="no such service: main")
        return compose_env._Resolved(
            project=_main(), variables={}, environment=frozenset()
        )

    monkeypatch.setattr(compose_env, "_run_config", fake)
    environment = _task(tmp_path)
    assert compose_env.check_compose(environment).refused == []

    replacement = config / "cli-plugins" / "docker-compose.new"
    replacement.write_text("compose v2, longer\n")
    os.replace(replacement, plugin)
    refused = _shown(compose_env.check_compose(environment))
    assert refused and refused == _fresh(environment)


def test_a_named_folder_that_moves_checks_again(tmp_path, resolve, monkeypatch):
    """A named variable's path is a link the operator points elsewhere: the
    task's bind of the old place is then outside every allowed folder."""
    shared = tmp_path / "shared-a"
    shared.mkdir()
    (tmp_path / "shared-b").mkdir()
    link = tmp_path / "shared"
    link.symlink_to(shared)
    monkeypatch.setenv("PERFD_SHARED", str(link))
    environment = _task(tmp_path)
    resolve["project"] = lambda command: _main(
        volumes=[
            {"type": "bind", "source": str(shared), "target": "/srv", "read_only": True}
        ]
    )

    assert (
        compose_env.check_compose(environment, allowed=["PERFD_SHARED"]).refused == []
    )

    link.unlink()
    link.symlink_to(tmp_path / "shared-b")
    refused = _shown(compose_env.check_compose(environment, allowed=["PERFD_SHARED"]))
    assert refused and refused == _fresh(environment, allowed=["PERFD_SHARED"])


def test_a_mount_below_a_named_folder_checks_again(tmp_path, resolve, monkeypatch):
    """A writable nonrecursive bind volume of a named folder is refused once
    a mount appears below it."""
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.setenv("PERFD_SHARED", str(shared))
    environment = _task(tmp_path)
    resolve["project"] = lambda command: {
        "services": {
            "main": {
                "image": "fixture-main:latest",
                "volumes": [{"type": "volume", "source": "data", "target": "/srv"}],
            }
        },
        "volumes": {
            "data": {
                "name": f"{command[command.index('--project-name') + 1]}_data",
                "driver_opts": {"type": "none", "o": "bind", "device": str(shared)},
            }
        },
    }
    points = {"/", "/proc"}
    monkeypatch.setattr(compose_env, "_mountpoints", lambda text: set(points))
    allowed = ["PERFD_SHARED"]

    assert compose_env.check_compose(environment, allowed=allowed).refused == []
    points.add("/var/lib/docker/overlay2/x/merged")
    assert compose_env.check_compose(environment, allowed=allowed).refused == []

    points.add(str(shared.resolve() / "inner"))
    refused = _shown(compose_env.check_compose(environment, allowed=allowed))
    assert refused and refused == _fresh(environment, allowed=allowed)


def test_a_compose_that_could_not_run_is_not_kept(tmp_path, monkeypatch):
    """docker compose could not start (an OSError; a timeout takes the same
    path): the next check runs it again rather than keep that refusal for the
    rest of the process."""

    def broken(*args, **kwargs):
        raise OSError(24, "Too many open files")

    environment = _task(tmp_path)
    monkeypatch.setattr(compose_env.subprocess, "Popen", broken)
    refused = _shown(compose_env.check_compose(environment))
    assert refused and any("could not run docker compose" in r for r in refused)

    def fake(command, env, *, project_only=False):
        return compose_env._Resolved(
            project=_main(), variables={}, environment=frozenset()
        )

    monkeypatch.setattr(compose_env, "_run_config", fake)
    assert compose_env.check_compose(environment).refused == []


def test_the_early_overlay_goes_with_a_dropped_sandbox(tmp_path, monkeypatch):
    from benchflow.sandbox.docker import DockerSandbox
    from tests.test_docker_direct_exec import _sandbox

    monkeypatch.setattr(DockerSandbox, "_uses_compose", property(lambda self: True))
    sandbox = _sandbox(tmp_path)
    sandbox._write_prebuilt_overlay("fixture-main:latest")
    overlay = sandbox._runtime_compose_path
    assert overlay is not None and overlay.exists()
    del sandbox
    gc.collect()
    assert not overlay.exists()


def test_a_removed_overlay_is_not_removed_again(tmp_path, monkeypatch):
    """Once the sandbox removed its overlay, its finalizer is gone: a later
    file of the same name (another sandbox's) stays."""
    from benchflow.sandbox.docker import DockerSandbox
    from tests.test_docker_direct_exec import _sandbox

    monkeypatch.setattr(DockerSandbox, "_uses_compose", property(lambda self: True))
    sandbox = _sandbox(tmp_path)
    sandbox._write_prebuilt_overlay("fixture-main:latest")
    overlay = sandbox._runtime_compose_path
    sandbox._write_prebuilt_overlay(None)
    assert sandbox._runtime_compose_path is None and not overlay.exists()
    overlay.write_text("another sandbox's\n")
    del sandbox
    gc.collect()
    assert overlay.exists()
    overlay.unlink()


def test_the_key_does_not_call_shutil_which(tmp_path, resolve, monkeypatch):
    """Tests and callers stand in for ``shutil.which`` with a one-argument
    function (test_check_run_sweep, test_taskmd_cli): the key must not break
    them."""
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/docker")
    environment = _task(tmp_path)
    assert compose_env.check_compose(environment).refused == []
    assert compose_env.check_compose(environment).refused == []
    assert resolve["calls"] == 1


# --- round 3 ---------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "linux", reason="macOS refuses non-UTF-8 names")
def test_a_name_that_is_not_utf8_passes_and_is_keyed(tmp_path, resolve):
    """A fixture named in Latin-1 (``tests/caf\\xe9.txt``) passes the check as
    a fresh check passes it, and its bytes are in the key."""
    environment = _task(tmp_path)
    fixture = os.path.join(os.fsencode(environment.parent / "tests"), b"caf\xe9.txt")
    with open(fixture, "wb") as out:
        out.write(b"one\n")
    assert compose_env.check_compose(environment).refused == []
    assert _fresh(environment) == []
    calls = resolve["calls"]
    compose_env.check_compose(environment)
    assert resolve["calls"] == calls
    with open(fixture, "wb") as out:
        out.write(b"two\n")
    assert compose_env.check_compose(environment).refused == []
    assert resolve["calls"] == calls + 1
    # The UTF-8 spelling of the same word is another name: another key.
    os.rename(fixture, os.path.join(os.path.dirname(fixture), "café.txt".encode()))
    compose_env.check_compose(environment)
    assert resolve["calls"] == calls + 2


def test_a_task_over_the_caps_reads_no_file(tmp_path, resolve, monkeypatch):
    """The sizes are taken first: a task over the caps costs no reads before
    the memo turns off for it."""
    environment = _task(tmp_path)
    for n in range(4):
        (environment.parent / "tests" / f"data{n}").write_bytes(b"x" * 1000)
    monkeypatch.setattr(compose_env, "_CHECK_MEMO_MAX_BYTES", 2500)
    opened: list[str] = []
    real_open = open

    def counting(path, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(compose_env, "open", counting, raising=False)
    assert compose_env._folder_digest(environment.parent) is None
    assert opened == []
    compose_env.check_compose(environment)
    compose_env.check_compose(environment)
    assert resolve["calls"] == 2


def test_a_link_that_lands_elsewhere_through_an_outside_link_checks_again(
    tmp_path, resolve
):
    """``tests/data -> ../../alias/tests/real.txt``, where ``alias`` (beside
    the task) points at the task: the link lands inside. Point ``alias``
    elsewhere and the same link text lands outside: refused."""
    environment = _task(tmp_path)
    task = environment.parent
    (task / "tests" / "real.txt").write_text("real\n")
    other = tmp_path / "other"
    (other / "tests").mkdir(parents=True)
    (other / "tests" / "real.txt").write_text("real\n")
    alias = tmp_path / "alias"
    alias.symlink_to(task)
    (task / "tests" / "data").symlink_to("../../alias/tests/real.txt")
    assert compose_env.check_compose(environment).refused == []

    alias.unlink()
    alias.symlink_to(other)
    refused = _shown(compose_env.check_compose(environment))
    assert refused and refused == _fresh(environment)


def test_a_named_folder_keeps_no_answer(tmp_path, resolve, monkeypatch):
    """The check resolves what a task binds below a named folder: a link
    there pointed outside is refused on the next check."""
    shared = tmp_path / "shared"
    (shared / "v1" / "data").mkdir(parents=True)
    (tmp_path / "outside" / "data").mkdir(parents=True)
    (shared / "current").symlink_to("v1")
    monkeypatch.setenv("PERFD_SHARED", str(shared))
    environment = _task(tmp_path)
    resolve["project"] = lambda command: _main(
        volumes=[
            {
                "type": "bind",
                "source": str(shared / "current" / "data"),
                "target": "/srv",
                "read_only": True,
            }
        ]
    )
    allowed = ["PERFD_SHARED"]
    assert compose_env.check_compose(environment, allowed=allowed).refused == []

    (shared / "current").unlink()
    (shared / "current").symlink_to(tmp_path / "outside")
    refused = _shown(compose_env.check_compose(environment, allowed=allowed))
    assert refused and refused == _fresh(environment, allowed=allowed)


def test_staging_that_found_the_task_changing_is_not_kept(
    tmp_path, resolve, monkeypatch
):
    """A file gone between staging's listing and its copy (an editor's temp
    file): "try again" is the answer once, not for the rest of the process."""
    from benchflow.task import staging

    environment = _task(tmp_path)
    real_copy = staging._copy_file
    calls: list[int] = []

    def once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise FileNotFoundError(2, "No such file or directory")
        return real_copy(*args, **kwargs)

    monkeypatch.setattr(staging, "_copy_file", once)
    refused = _shown(compose_env.check_compose(environment))
    assert refused and any("try again" in r for r in refused)
    assert compose_env.check_compose(environment).refused == []


def test_a_task_that_changed_while_checked_keeps_no_answer(
    tmp_path, resolve, monkeypatch
):
    """The link goes between the key and the check's own staging: the pass
    belongs to the folder without it, so the folder with it checks again."""
    from benchflow.task import staging

    environment = _task(tmp_path)
    link = environment.parent / "tests" / "outside"
    link.symlink_to("/etc")
    real_stage = staging.stage_environment

    def racing(*args, **kwargs):
        if link.is_symlink():
            link.unlink()
        return real_stage(*args, **kwargs)

    monkeypatch.setattr(staging, "stage_environment", racing)
    assert compose_env.check_compose(environment).refused == []
    monkeypatch.setattr(staging, "stage_environment", real_stage)

    link.symlink_to("/etc")
    refused = _shown(compose_env.check_compose(environment))
    assert refused and refused == _fresh(environment)


@pytest.mark.asyncio
async def test_kept_containers_keep_their_overlay(tmp_path, monkeypatch):
    """stop() with kept containers leaves the Compose files for a later
    ``docker compose -f ...``; the runtime overlay stays past the sandbox."""
    from tests.test_docker_subnets import _Compose, _FakeDaemon, _local, _sandbox

    _local(monkeypatch, _FakeDaemon("here", []))
    sandbox = _sandbox(tmp_path)
    sandbox._keep_containers = True
    sandbox._run_docker_compose_command = _Compose(sandbox)  # type: ignore[method-assign]
    await sandbox.start(force_build=False)
    sandbox._write_prebuilt_overlay("fixture-main:latest")
    overlay = sandbox._runtime_compose_path
    assert overlay is not None and overlay.exists()
    await sandbox.stop(delete=False)
    sandbox._release_subnets()
    del sandbox
    gc.collect()
    assert overlay.exists()
    overlay.unlink()
