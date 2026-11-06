"""Guards the fixes after c7e75fb5 (dummy markers only)."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import pytest

from tests.test_compose_launch import _from_task
from tests.test_compose_oracle import _compose_available, _compose_task, _gate

compose = pytest.mark.skipif(not _compose_available(), reason="needs Compose config")


@pytest.fixture(autouse=True)
def _operator(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "")
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("key", ["env_file", "label_file"])
@pytest.mark.parametrize("merged", [False, True])
def test_r4_1_every_long_form_path_is_checked(tmp_path, key, merged):
    """An explicit or inherited unsafe path matters."""
    from benchflow.task.compose_load import read_compose

    outside = tmp_path / "operator.env"
    outside.write_text("S1_DUMMY=operator-marker\n")
    safe, unsafe = (outside, "own.env") if merged else ("own.env", outside)
    task = _compose_task(
        tmp_path / "task",
        f"x-env: &safe {{path: {safe}}}\nservices:\n  main:\n"
        f"    {key}:\n      - <<: *safe\n        path: {unsafe}\n",
        files={"own.env": "SAFE=1\n"},
    )
    loaded = read_compose(
        task / "environment",
        task / "environment/docker-compose.yaml",
        host_access=False,
        host_remedy="opt in",
    )
    assert any(str(outside) in r.describe() for r in loaded.refused)


@pytest.mark.parametrize("options", ["bind,ro", "ro,bind", "rbind,ro", "bind"])
@pytest.mark.parametrize("remote", [False, True])
def test_r4_2_a_relative_daemon_device_is_never_a_task_path(options, remote):
    """Guards the fix for local, remote and DinD's shared checker."""
    from benchflow.task.compose_env import _ProjectCheck

    check = _ProjectCheck("/tmp/task/environment", ("/tmp/task",), remote_binds=remote)
    assert check.check(
        {
            "volumes": {
                "loot": {
                    "driver": "local",
                    "driver_opts": {
                        "type": "none",
                        "o": options,
                        "device": ".",
                    },
                }
            }
        }
    )


@compose
@pytest.mark.parametrize("kind", ["bind", "build", "secret", "config", "device"])
def test_r4_3_task_dependencies_survive_original_replacement(tmp_path, kind):
    """Every launch consumer uses its private task copy."""
    snippets = {
        "bind": "services:\n  main:\n    volumes: [../payload:/loot:ro, '..:/task-files:ro']\n",
        "build": "services:\n  main:\n    build: {context: ../payload}\n",
        "secret": "services:\n  main:\n    secrets: [payload]\nsecrets:\n  payload: {file: ../payload/marker}\n",
        "config": "services:\n  main:\n    configs: [payload]\nconfigs:\n  payload: {file: ../payload/marker}\n",
        "device": "services:\n  main:\n    volumes: [loot:/loot]\nvolumes:\n  loot:\n    driver: local\n    driver_opts: {type: none, o: bind, device: ../payload}\n",
    }
    task = _compose_task(tmp_path / "task", snippets[kind])
    payload = task / "payload"
    payload.mkdir()
    (payload / "marker").write_text("task-marker")
    (payload / "Dockerfile").write_text("FROM alpine:3.20\n")
    sandbox = _from_task(task).inner
    project = json.loads(sandbox._launch.path.read_text())
    if kind == "bind":
        raw = next(
            v["source"]
            for v in project["services"]["main"]["volumes"]
            if v["target"] == "/loot"
        )
    elif kind == "build":
        raw = project["services"]["main"]["build"]["context"]
    elif kind == "device":
        raw = project["volumes"]["loot"]["driver_opts"]["device"]
    else:
        raw = project[kind + "s"]["payload"]["file"]
    outside = tmp_path / "operator"
    outside.mkdir()
    (outside / "marker").write_text("operator-marker")
    payload.rename(task / "old-payload")
    payload.symlink_to(outside, target_is_directory=True)
    copied = Path(raw) if kind in ("secret", "config") else Path(raw) / "marker"
    assert copied.read_text() == "task-marker"
    assert copied.is_relative_to(sandbox._staged.root)


@pytest.mark.parametrize("root_swap", [False, True])
def test_r4_4_snapshot_never_follows_a_swapped_ancestor(
    monkeypatch, tmp_path, root_swap
):
    """Guards the fix by swapping a directory at its enumeration boundary."""
    from benchflow.task.staging import StagingRefused, copy_tree

    source = tmp_path / "source"
    (source / "payload").mkdir(parents=True)
    (source / "payload/marker").write_text("task-marker")
    outside = tmp_path / "operator"
    (outside / "payload").mkdir(parents=True)
    (outside / "payload/marker").write_text("operator-marker")
    (outside / "marker").write_text("operator-marker")
    original = os.scandir
    swapped = False

    def scandir(path):
        nonlocal swapped
        named = (
            os.readlink(f"/proc/self/fd/{path}") if isinstance(path, int) else str(path)
        )
        victim = source if root_swap else source / "payload"
        if not swapped and named == str(victim):
            swapped = True
            victim.rename(tmp_path / "held")
            victim.symlink_to(outside, target_is_directory=True)
        return original(path)

    monkeypatch.setattr(os, "scandir", scandir)
    destination = tmp_path / "destination"
    with contextlib.suppress(StagingRefused):
        copy_tree(source, destination)
    assert swapped
    assert all(p.read_text() != "operator-marker" for p in destination.rglob("marker"))


@compose
@pytest.mark.parametrize("top", ["secrets", "configs"])
def test_r4_5_environment_sources_keep_exact_dotenv_content(monkeypatch, tmp_path, top):
    """Guards the fix without forwarding dotenv into host programs."""
    monkeypatch.delenv("S1_DATA", raising=False)
    value = "task-value\nsecond=$literal"
    task = _compose_task(
        tmp_path / "task",
        f"services:\n  main:\n    {top}: [payload]\n{top}:\n  payload:\n    environment: S1_DATA\n",
        files={".env": "S1_DATA='task-value\nsecond=$literal'\nPATH=task-only\n"},
    )
    sandbox = _from_task(task).inner
    source = json.loads(sandbox._launch.path.read_text())[top]["payload"]
    assert "file" not in source
    assert sandbox._launch.env[source["environment"]] == value
    assert "S1_DATA" not in sandbox._launch.env
    assert sandbox._launch.env.get("PATH") != "task-only"


@compose
def test_r4_5_missing_environment_source_is_refused_early(monkeypatch, tmp_path):
    """Guards the before-allocation availability check."""
    from benchflow.task.env import TaskEnvNotAllowedError

    monkeypatch.delenv("S1_DATA", raising=False)
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    secrets: [payload]\nsecrets:\n  payload: {environment: S1_DATA}\n",
    )
    with pytest.raises(TaskEnvNotAllowedError, match="S1_DATA"):
        _from_task(task)


@pytest.mark.parametrize("dependency", ["include", "extends"])
def test_r4_6_no_compose_refuses_unresolved_dependencies(
    monkeypatch, tmp_path, dependency
):
    """Guards the fix before any Daytona allocation."""
    from benchflow.task import compose_env
    from benchflow.task.env import TaskEnvNotAllowedError

    monkeypatch.setattr(compose_env, "_compose_cli_available", lambda: False)
    root = (
        "include: [child.yaml]\nservices:\n  main: {}\n"
        if dependency == "include"
        else "services:\n  main:\n    extends: {file: child.yaml, service: side}\n"
    )
    task = _compose_task(
        tmp_path / "task",
        root,
        files={
            "child.yaml": "services:\n  side:\n    image: alpine:3.20\n    privileged: true\n"
        },
    )
    with pytest.raises(TaskEnvNotAllowedError, match="docker compose"):
        _gate(task, sandbox_type="daytona")


@pytest.mark.parametrize(
    "body,toml",
    [
        ("services:\n  main:\n    volumes: ['/:/host']\n", ""),
        ("services:\n  main:\n    build: {context: /tmp/operator}\n", ""),
        (
            "services:\n  main:\n    secrets: [data]\nsecrets:\n  data: {file: /tmp/operator-marker}\n",
            "",
        ),
        (
            "services:\n  main: {}\n",
            '[environment.env]\nCONTEXT_DIR = "/tmp/operator"\n',
        ),
    ],
    ids=["mount", "build", "secret", "runtime-path-override"],
)
def test_r4_6_no_compose_cannot_approve_unresolved_path_consumers(
    monkeypatch, tmp_path, body, toml
):
    """Guards the same before-allocation rule for root paths."""
    from benchflow.task import compose_env
    from benchflow.task.env import TaskEnvNotAllowedError

    monkeypatch.setattr(compose_env, "_compose_cli_available", lambda: False)
    task = _compose_task(tmp_path / "task", body, toml)
    with pytest.raises(TaskEnvNotAllowedError, match="docker compose"):
        _gate(task, sandbox_type="daytona")


@compose
@pytest.mark.parametrize(
    "source", ["/benchflow/task/environment/data", "${CONTEXT_DIR}/data"]
)
@pytest.mark.parametrize("absolute_dependency", [False, True])
def test_r4_7_daytona_checks_vm_coordinates(tmp_path, source, absolute_dependency):
    """A valid VM-local source passes local preflight."""
    task = _compose_task(
        tmp_path / "task", f"services:\n  main:\n    volumes: ['{source}:/data:ro']\n"
    )
    (task / "environment/data").mkdir()
    if absolute_dependency:
        root = task / "environment/docker-compose.yaml"
        root.write_text(
            root.read_text() + "    env_file: /benchflow/task/environment/task.env\n"
        )
        (task / "environment/task.env").write_text("MODE=task-value\n")
    _gate(task, sandbox_type="daytona")
