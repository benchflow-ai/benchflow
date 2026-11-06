"""One build per layer key across processes, and no prune under a hit (PY8 U13).

Two OS processes ask for the same cold layer through a stand-in ``docker``
on PATH (a small script that keeps its images in a JSON file, logs every
command, and holds each build until the test lets it finish). The test lets
the second process in only after the first has started building, so the
second's outcome does not depend on scheduling: with a per-process lock it
misses and builds the layer again; with the host lock it waits, then hits.

The prune race runs in one process against the in-memory stand-in of
``tests/test_env_layers_agent.py``: a hit is paused between its inspect and
its ``docker tag`` while another trial's post-build prune evicts that key.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from benchflow.sandbox import env_layers
from benchflow.sandbox.env_layers import (
    AGENT,
    AgentLayerRecipe,
    DockerLayers,
    LayerError,
    layer_tag,
)
from tests.test_env_layers_agent import FakeDocker


def _optional(module: str) -> Any:
    """``module``, or None on a BenchFlow without it (the fails-before base),
    so these tests fail there on what BenchFlow does, not on an import."""
    try:
        return importlib.import_module(module)
    except ImportError:
        return None


_layer_lock = _optional("benchflow.sandbox._layer_lock")
LOCK_DIR_ENV = "BENCHFLOW_LAYER_LOCK_DIR"

FAKE_DOCKER = r'''#!PYTHON
"""A stand-in docker CLI for tests: images in a JSON file, every call logged
with the pid of the process that ran it."""
import fcntl, hashlib, json, os, re, sys, time

STATE = os.environ["FAKE_DOCKER_STATE"]
LOG = os.environ["FAKE_DOCKER_LOG"]
GATE = os.environ.get("FAKE_DOCKER_GATE", "")
argv = sys.argv[1:]


def log(text):
    with open(LOG, "a") as handle:
        handle.write(f"{os.getppid()} {text}\n")


class State:
    def __enter__(self):
        self.fd = os.open(STATE + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        with open(STATE) as handle:
            self.data = json.load(handle)
        return self.data

    def __exit__(self, *exc):
        with open(STATE, "w") as handle:
            json.dump(self.data, handle)
        os.close(self.fd)


def find(data, ref):
    if ref in data["tags"]:
        return data["images"][data["tags"][ref]]
    return data["images"].get(ref)


def info_of(data, image_id):
    image = data["images"][image_id]
    tags = [t for t, i in data["tags"].items() if i == image_id]
    return {
        "Id": image_id,
        "RepoTags": tags,
        "Architecture": "amd64",
        "Os": "linux",
        "Size": image["size"],
        "Created": "2026-10-16T00:00:00.000000000Z",
        "RootFS": {"Type": "layers", "Layers": image["layers"]},
        "Config": {"User": "", "Env": ["PATH=/bin"], "Labels": image["labels"]},
    }


log(" ".join(argv))
words = [a for a in argv]
if words[:1] == ["info"]:
    if "{{.ID}}" in words:
        print("fake-daemon-race")
        sys.exit(0)
    sys.exit(1)
if words[:2] == ["image", "inspect"]:
    refs = words[2:]
    with State() as data:
        if refs and refs[0] == "--format":
            image = find(data, refs[2])
            label = re.search(r'"([^"]+)"', refs[1]).group(1)
            print(image["labels"].get(label, "") if image else "")
            sys.exit(0 if image else 1)
        found = [find(data, r) for r in refs]
        infos = [info_of(data, i["id"]) for i in found if i]
    print(json.dumps(infos))
    if not all(found):
        print("Error: No such image", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
if words[:2] == ["image", "ls"]:
    filters = [words[i + 1] for i, w in enumerate(words) if w == "--filter"]
    with State() as data:
        rows = []
        for tag, image_id in data["tags"].items():
            labels = data["images"][image_id]["labels"]
            if all(
                labels.get(f[6:].partition("=")[0]) == f[6:].partition("=")[2]
                for f in filters
            ):
                rows.append((tag, image_id))
    if "--quiet" in words:
        print("\n".join(sorted({i for _, i in rows})))
    else:
        for tag, image_id in rows:
            repo, _, name = tag.rpartition(":")
            print(json.dumps({"Repository": repo, "Tag": name, "ID": image_id[7:19],
                              "CreatedAt": "2026-10-16 00:00:00 +0000 UTC"}))
    sys.exit(0)
if words[:1] == ["run"]:
    print("glibc")
    sys.exit(0)
if words[:1] == ["tag"]:
    with State() as data:
        image = find(data, words[1])
        if image is None:
            print("Error: No such image", file=sys.stderr)
            sys.exit(1)
        data["tags"][words[2]] = image["id"]
    sys.exit(0)
if words[:2] == ["image", "rm"]:
    with State() as data:
        if words[2] not in data["tags"]:
            print("Error: No such image", file=sys.stderr)
            sys.exit(1)
        del data["tags"][words[2]]
    sys.exit(0)
if words[:1] == ["ps"]:
    sys.exit(0)
if words[:1] == ["build"]:
    dockerfile = sys.stdin.read()
    tag = words[words.index("--tag") + 1]
    labels = dict(words[i + 1].partition("=")[::2] for i, w in enumerate(words) if w == "--label")
    log(f"build-start {tag}")
    deadline = time.time() + 60
    while GATE and not os.path.exists(GATE) and time.time() < deadline:
        time.sleep(0.05)
    base_ref = dockerfile.splitlines()[0].removeprefix("FROM ")
    with State() as data:
        base = find(data, base_ref)
        image_id = "sha256:" + hashlib.sha256((tag + dockerfile + str(time.time())).encode()).hexdigest()
        data["images"][image_id] = {
            "id": image_id,
            "labels": {**base["labels"], **labels},
            "layers": base["layers"] + ["sha256:" + hashlib.sha256(dockerfile.encode()).hexdigest()],
            "size": 1000,
        }
        data["tags"][tag] = image_id
    # BenchFlow trusts a layer by the id its own build reports.
    if "--iidfile" in words:
        with open(words[words.index("--iidfile") + 1], "w") as handle:
            handle.write(image_id)
    log(f"build-end {tag}")
    sys.exit(0)
if words[:2] == ["builder", "du"]:
    sys.exit(0)
if words[:2] == ["builder", "prune"]:
    sys.exit(0)
print("fake docker: unknown command " + " ".join(words), file=sys.stderr)
sys.exit(2)
'''

DRIVER = r"""
import asyncio, json, os, sys, time
from benchflow.sandbox.env_layers import AgentLayerRecipe, DockerLayers, ProxyLayerRecipe

kind, go, out = sys.argv[1:4]
while not os.path.exists(go):
    time.sleep(0.02)
if kind == "agent":
    recipe = AgentLayerRecipe(agent="fake-agent", version="1.0.0", install_cmd="true")
else:
    recipe = ProxyLayerRecipe(
        group="litellm-proxy", requirements_name="proxy.txt",
        requirements_text="litellm==1.0\n", install_cmd="true",
        python="3.12", uv_version="0.13.4",
    )

async def main():
    layers = DockerLayers()
    if kind == "agent":
        result = await layers.ensure_agent_layer(recipe, "task:1", prune=False)
    else:
        result = await layers.ensure_proxy_layer(recipe, "task:1", prune=False)
    return result.record()

with open(out, "w") as handle:
    json.dump(asyncio.run(main()), handle)
"""


def _wait_for(check, what: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.02)


def _log_lines(log: Path, pid: int) -> list[str]:
    if not log.exists():
        return []
    prefix = f"{pid} "
    return [
        line[len(prefix) :]
        for line in log.read_text().splitlines()
        if line.startswith(prefix)
    ]


@pytest.mark.parametrize("kind", ["agent", "proxy"])
def test_two_processes_build_a_cold_key_once(kind, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER.replace("PYTHON", sys.executable, 1))
    docker.chmod(0o755)
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "images": {
                    "sha256:" + "1" * 64: {
                        "id": "sha256:" + "1" * 64,
                        "labels": {},
                        "layers": ["sha256:" + "0" * 64],
                        "size": 500,
                    }
                },
                "tags": {"task:1": "sha256:" + "1" * 64},
            }
        )
    )
    log = tmp_path / "docker.log"
    gate = tmp_path / "builds-may-finish"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "FAKE_DOCKER_STATE": str(state),
        "FAKE_DOCKER_LOG": str(log),
        "FAKE_DOCKER_GATE": str(gate),
        LOCK_DIR_ENV: str(tmp_path / "layer-locks"),
        "BENCHFLOW_AGENT_LAYER_MIN_FREE_GB": "0",
    }
    procs = []
    for name in ("a", "b"):
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    DRIVER,
                    kind,
                    str(tmp_path / f"go-{name}"),
                    str(tmp_path / f"out-{name}.json"),
                ],
                env=env,
                cwd=tmp_path,
            )
        )
    a, b = procs
    try:
        (tmp_path / "go-a").touch()
        # The first process is inside its build, holding whatever lock it has.
        _wait_for(
            lambda: any(x.startswith("build-start") for x in _log_lines(log, a.pid)),
            "the first build to start",
        )
        (tmp_path / "go-b").touch()
        # The second has reached its lock (host lock: it names the daemon,
        # then waits) or, with a per-process lock only, its own hit check.
        _wait_for(
            lambda: any(
                x.startswith("info --format {{.ID}}")
                or x.startswith(f"image inspect {env_layers.LAYER_REPOSITORY}:")
                for x in _log_lines(log, b.pid)
            ),
            "the second process to reach its lock or its hit check",
        )
        time.sleep(0.3)
        gate.touch()
        for proc in procs:
            assert proc.wait(timeout=120) == 0
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
    builds = [
        line
        for proc in procs
        for line in _log_lines(log, proc.pid)
        if line.startswith("build-start")
    ]
    records = {
        name: json.loads((tmp_path / f"out-{name}.json").read_text())
        for name in ("a", "b")
    }
    assert len(builds) == 1, f"{len(builds)} builds of one {kind} key: {builds}"
    assert records["a"]["status"] == "miss"
    assert records["b"]["status"] == "hit"
    assert records["b"]["key"] == records["a"]["key"]
    assert records["b"]["lock"] == "host"
    assert records["b"]["lock_wait_sec"] > 0
    assert records["a"]["lock"] == "host"


# -- prune against a hit ---------------------------------------------------

RECIPE = AgentLayerRecipe(agent="fake-agent", version="1.2.3", install_cmd="true")
OTHER = AgentLayerRecipe(agent="other-agent", version="9.9.9", install_cmd="true")


class PausingDocker(FakeDocker):
    """Pauses ``docker tag <layer> <alias>`` until :attr:`resume` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.paused = asyncio.Event()
        self.resume = asyncio.Event()
        self.pause_alias: str | None = None

    async def run(self, *args, stdin=None, timeout=0):
        argv = list(args)
        if argv[:1] == ["info"] and "{{.ID}}" in argv:
            self.calls.append(argv)
            return env_layers._Run(stdout="fake-daemon-1\n", stderr="", code=0)
        if argv[:1] == ["tag"] and self.pause_alias and argv[-1] == self.pause_alias:
            self.paused.set()
            await self.resume.wait()
        return await super().run(*args, stdin=stdin, timeout=timeout)


@pytest.fixture
def pausing(monkeypatch, tmp_path_factory):
    monkeypatch.setenv(LOCK_DIR_ENV, str(tmp_path_factory.mktemp("held") / "locks"))
    DockerLayers._claims.clear()
    DockerLayers._libc_cache.clear()
    for name in ("_daemon_ids", "_key_locks"):
        table = getattr(DockerLayers, name, None)
        if table is not None:
            table.clear()
    if _layer_lock is not None:
        _layer_lock.clear_process_locks()
    fake = PausingDocker()

    async def run(self, *args, stdin=None, timeout=0):
        return await fake.run(*args, stdin=stdin, timeout=timeout)

    monkeypatch.setattr(DockerLayers, "_docker", run)
    yield fake
    DockerLayers._claims.clear()


async def test_a_prune_never_removes_a_layer_a_hit_is_tagging(pausing):
    docker = pausing
    docker.add_image("task:1")
    docker.add_image("task:2", layers=["sha256:" + "9" * 64])
    layers = DockerLayers()
    first = await layers.ensure_agent_layer(RECIPE, "task:1", prune=False)
    DockerLayers.release(first.key, daemon=first.daemon_id)
    # A hit on the cached layer stops between its inspect and its tag.
    docker.pause_alias = "bf__layer__trial-x:1"
    hit = asyncio.create_task(
        layers.ensure_agent_layer(
            RECIPE, "task:1", alias="bf__layer__trial-x:1", prune=False
        )
    )
    await asyncio.wait_for(docker.paused.wait(), 5)
    # Another trial misses another key; its post-build prune's byte bound
    # (one layer's worth, which its own held layer takes) evicts the
    # idle-looking first key.
    other = await layers.ensure_agent_layer(OTHER, "task:2", prune=False)
    assert other.status == "miss"
    report = await layers.prune(AGENT, keep=64, max_bytes=docker.build_size)
    docker.resume.set()
    try:
        result = await asyncio.wait_for(hit, 5)
    except LayerError as e:
        pytest.fail(f"the hit lost its layer to the prune: {e}")
    assert first.key not in report.removed
    assert result.status == "hit"
    assert layer_tag(AGENT, first.key) in docker.tags
    assert (
        docker.tags["bf__layer__trial-x:1"] is docker.tags[layer_tag(AGENT, first.key)]
    )
    assert "lock" in getattr(report, "skipped", {}).get(first.key, "")
