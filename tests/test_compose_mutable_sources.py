"""Guards a finding after 62c80723: immutable launch path entries."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.task.compose_env import _ProjectCheck, check_compose_in_vm
from tests.test_compose_launch import _from_task
from tests.test_compose_oracle import _compose_available, _compose_task, _gate


def _bind(path, target="/data", **kwargs):
    return {"type": "bind", "source": str(path), "target": target, **kwargs}


def _project(root, source, writer, consumer):
    main = {"volumes": []}
    project = {"services": {"main": main}}
    mount = _bind(root, "/workspace")
    if writer == "bind":
        main["volumes"].append(mount)
    elif writer == "sidecar":
        project["services"]["side"] = {"volumes": [mount]}
    elif writer.startswith("volume"):
        main["volumes"].append(
            {"type": "volume", "source": "work", "target": "/workspace"}
        )
        project["volumes"] = {
            "work": {
                "driver": "local",
                "driver_opts": {
                    "type": "none" if writer == "volume" else "ext4",
                    "o": "bind" if writer == "volume" else "rw,rbind",
                    "device": str(root),
                },
            }
        }
    else:
        mount["read_only"] = True
        project["services"]["holder"] = {"volumes": [mount]}
        project["services"]["middle"] = {"volumes_from": ["holder:ro"]}
        main["volumes_from"] = ["middle:rw"]
    if consumer == "bind":
        main["volumes"].append(_bind(source, read_only=True))
    elif consumer in ("secrets", "configs"):
        project[consumer] = {"payload": {"file": str(source)}}
        main[consumer] = ["payload"]
    elif consumer in ("env_file", "label_file"):
        main[consumer] = [{"path": str(source)}]
    elif consumer == "device":
        project.setdefault("volumes", {})["data"] = {
            "driver_opts": {"type": "none", "o": "bind,ro", "device": str(source)}
        }
        main["volumes"].append(
            {"type": "volume", "source": "data", "target": "/data", "read_only": True}
        )
    else:
        build = {
            "context": str(root.parent / "build"),
            "dockerfile_inline": "FROM scratch",
        }
        main["build"] = build
        if consumer == "context":
            build["context"] = str(source)
        elif consumer == "dockerfile":
            del build["dockerfile_inline"]
            build["dockerfile"] = str(source)
        elif consumer == "additional_contexts":
            build[consumer] = {"data": str(source)}
        elif consumer == "oci_context":
            build["additional_contexts"] = [f"data=oci-layout://{source}"]
        elif consumer in ("cache_from", "cache_to"):
            build[consumer] = [f"type=local,src={source},dest={source}"]
        elif consumer == "ssh":
            build[consumer] = [f"default={source}"]
    return project


@pytest.mark.parametrize(
    "writer", ["bind", "sidecar", "volume", "volume-rbind", "inherited"]
)
@pytest.mark.parametrize(
    "consumer",
    [
        "bind",
        "secrets",
        "configs",
        "env_file",
        "label_file",
        "device",
        "context",
        "dockerfile",
        "additional_contexts",
        "oci_context",
        "cache_from",
        "cache_to",
        "ssh",
    ],
)
@pytest.mark.parametrize(
    "alias",
    [
        "plain",
        "dot",
        "root-alias",
        "enter-and-leave",
        "descendant-alias",
        "descendant-exit",
        "leaf-alias",
        "hidden-chain",
    ],
)
def test_launch_sources_cannot_be_replaced_through_writable_mounts(
    tmp_path, writer, consumer, alias
):
    """Guards the entire path-consumer class from the sol review of 62c80723."""
    root = tmp_path / "work"
    root.mkdir()
    (root / "data").mkdir()
    source = root / "data"
    if alias == "dot":
        source = root / "unused" / ".." / "data"
    elif alias == "root-alias":
        (tmp_path / "alias").symlink_to(root, target_is_directory=True)
        root = tmp_path / "alias"
    elif alias == "enter-and-leave":
        outside = tmp_path / "other"
        outside.mkdir()
        (root / "exit").symlink_to(outside, target_is_directory=True)
        (tmp_path / "alias").symlink_to(root, target_is_directory=True)
        source = tmp_path / "alias" / "exit" / "payload"
    elif alias.startswith("descendant"):
        (tmp_path / "alias").symlink_to(root / "data", target_is_directory=True)
        source = tmp_path / "alias" / "payload"
        if alias == "descendant-exit":
            outside = tmp_path / "other"
            outside.mkdir()
            (root / "data/exit").symlink_to(outside, target_is_directory=True)
            source = tmp_path / "alias/exit/payload"
    elif alias in ("leaf-alias", "hidden-chain"):
        target = root / "data"
        if alias == "hidden-chain":
            outside = tmp_path / "other"
            outside.mkdir()
            (root / "exit").symlink_to(outside, target_is_directory=True)
            target = root / "exit/data"
        (tmp_path / "alias").symlink_to(target, target_is_directory=True)
        source = tmp_path / "alias"
    project = _project(root, source, writer, consumer)
    check = _ProjectCheck(str(tmp_path), (str(tmp_path),))
    refusals = check.check(project)
    assert any("writable" in r.describe() for r in refusals)


@pytest.mark.parametrize("relation", ["same", "sibling", "prefix", "readonly-parent"])
def test_independent_and_readonly_mounts_remain_valid(tmp_path, relation):
    """Guards 62c80723's valid mounts while restricting only replaceable entries."""
    root = tmp_path / "work"
    root.mkdir()
    source = {
        "same": root,
        "sibling": tmp_path / "data",
        "prefix": tmp_path / "work-other",
        "readonly-parent": root / "data",
    }[relation]
    project = _project(root, source, "bind", "bind")
    if relation == "readonly-parent":
        project["services"]["main"]["volumes"][0]["read_only"] = True
    assert not _ProjectCheck(str(tmp_path), (str(tmp_path),)).check(project)


@pytest.mark.parametrize("consumer", ["bind", "secrets", "configs", "device"])
def test_remote_host_sources_stay_refused(tmp_path, consumer):
    """Guards the finding's remote Docker sibling at 62c80723."""
    project = _project(tmp_path / "work", tmp_path / "work/data", "bind", consumer)
    assert _ProjectCheck(str(tmp_path), (str(tmp_path),), remote_binds=True).check(
        project
    )


@pytest.mark.parametrize("subpath", ["data", "./data", "first/../data", "data/nested"])
def test_bind_backed_volume_subpaths_are_launch_sources(tmp_path, subpath):
    """Guards the finding's named-volume subpath sibling at 62c80723."""
    project = _project(tmp_path / "work", tmp_path / "separate", "volume", "bind")
    project["services"]["main"]["volumes"].append(
        {
            "type": "volume",
            "source": "work",
            "target": "/subpath",
            "read_only": True,
            "volume": {"subpath": subpath},
        }
    )
    assert _ProjectCheck(str(tmp_path), (str(tmp_path),)).check(project)


@pytest.mark.parametrize("read_only", [True, False])
def test_inherited_readonly_and_cycles_are_bounded(tmp_path, read_only):
    """Guards inherited access without treating a readonly chain as a writer."""
    project = _project(tmp_path / "work", tmp_path / "work/data", "bind", "bind")
    project["services"]["main"]["volumes"][0]["read_only"] = True
    project["services"]["main"]["volumes_from"] = ["side:ro"]
    project["services"]["side"] = {
        "volumes_from": ["main:ro" if read_only else "main:rw"]
    }
    refused = _ProjectCheck(str(tmp_path), (str(tmp_path),)).check(project)
    assert bool(refused) is not read_only


@pytest.mark.asyncio
@pytest.mark.parametrize("writer", ["bind", "volume", "inherited"])
async def test_dind_resolves_every_ancestor_before_checking(
    tmp_path, monkeypatch, writer
):
    """Guards the VM path batch against aliases that enter and leave a writer."""
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)
    project = _project(
        Path("/tmp/task/work"), Path("/tmp/task/alias/exit/data"), writer, "bind"
    )
    requested = set()

    async def run(command):
        if "config" in shlex.split(command):
            return SimpleNamespace(return_code=0, stdout=json.dumps(project))
        if shlex.split(command)[:2] == ["sh", "-c"]:
            paths = shlex.split(command)[4:]
            requested.update(paths)

            # /alias enters /work; /alias/exit leaves it for another allowed tree.
            def resolve(path):
                if path == "/tmp/task/alias":
                    return "/tmp/task/work"
                return path.replace("/tmp/task/alias/exit", "/tmp/task/other")

            return SimpleNamespace(return_code=0, stdout="\n".join(map(resolve, paths)))
        return SimpleNamespace(return_code=0, stdout="x$$y")

    refused = await check_compose_in_vm(
        run, ["docker", "compose"], "/tmp/task", ["/tmp/task"]
    )
    assert "/tmp/task/alias" in requested
    assert any("writable over the launch source" in r.describe() for r in refused)


@pytest.mark.skipif(not _compose_available(), reason="needs Compose config")
@pytest.mark.parametrize("backend", ["docker", "daytona"])
@pytest.mark.parametrize("spelling", ["short", "long", "inherited", "volume"])
def test_resolved_gate_rejects_mutable_sources(
    tmp_path, monkeypatch, backend, spelling
):
    """Guards real Compose normalization and both launch gates after 62c80723."""
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "")
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.chdir(tmp_path)
    mounts = {
        "short": "    volumes: ['./work:/workspace', './work/data:/data:ro']\n",
        "long": "    volumes:\n      - {type: bind, source: ./work, target: /workspace, read_only: false}\n      - {type: bind, source: ./work/data, target: /data, read_only: true}\n",
        "inherited": "    volumes: ['./work/data:/data:ro']\n    volumes_from: ['holder:rw']\n  holder:\n    image: busybox\n    volumes: ['./work:/workspace:ro']\n",
        "volume": "    volumes: ['work:/workspace', './work/data:/data:ro']\nvolumes:\n  work:\n    driver_opts: {type: none, o: bind, device: '${CONTEXT_DIR}/work'}\n",
    }
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n" + mounts[spelling],
        files={"work/data/marker": "task-marker"},
    )
    with pytest.raises(ValueError, match="writable over the launch source"):
        _gate(task, backend)


@pytest.mark.skipif(not _compose_available(), reason="needs Compose config")
def test_agent_swap_never_reaches_a_frozen_restore_source(tmp_path, monkeypatch):
    """Guards the finding with dummy markers, without any host mount."""
    from benchflow.task.env import TaskEnvNotAllowedError

    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.chdir(tmp_path)
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    volumes: ['../workspace:/testbed', '../workspace/data:/data:ro']\n",
    )
    (task / "workspace/data").mkdir(parents=True)
    (task / "workspace/data/marker").write_text("task-marker")
    outside = tmp_path / "dummy-operator"
    outside.mkdir()
    (outside / "marker").write_text("operator-marker")
    try:
        sandbox = _from_task(task).inner
    except TaskEnvNotAllowedError as exc:
        assert "writable over the launch source" in str(exc)
        return
    project = json.loads(sandbox._launch.path.read_text())
    mounts = {
        m["target"]: Path(m["source"]) for m in project["services"]["main"]["volumes"]
    }
    (mounts["/testbed"] / "data").rename(mounts["/testbed"] / "data.old")
    (mounts["/testbed"] / "data").symlink_to(outside, target_is_directory=True)
    assert (mounts["/data"] / "marker").read_text() == "task-marker", (
        "restore source now resolves to dummy operator-marker"
    )
