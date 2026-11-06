"""The agent layer against a real Docker daemon (env_layers, PY8 U3a).

Builds tiny task images on pinned alpine and busybox images (musl, glibc, static),
builds an agent layer ``FROM`` each, reuses it, lists the files the layer
adds, and prunes it by key. Every image and key here is this test's own (a
fresh id in the tags and the recipe), so nothing else on the daemon is
touched. Skipped when no Docker daemon is available.

``prune(only=[key])`` limits removal, while ``in_use`` reports all held
layers on the daemon (they count toward its bounds). Parallel tests may
hold other keys: require our key in that report and verify its image and
subsequent removal directly, without requiring an otherwise idle daemon.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile
import uuid
from pathlib import Path

import pytest

from benchflow.sandbox import _docker_sweep, env_layers
from benchflow.sandbox.env_layers import (
    AGENT,
    ALIAS_PREFIX,
    LAYER_LABEL,
    AgentLayerRecipe,
    DockerLayers,
    alias_owner_tag,
    layer_tag,
)

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

# Task image bases by the libc the probe must find: alpine 3.20 (musl), busybox
# 1.37 built on glibc, and busybox 1.37 built on musl, which is static (no
# loader at all).
BASES = {
    "musl": "alpine@sha256:d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc",
    "none": "busybox@sha256:5cec3fc171c87218698e85a52af7087de727372aae264a787b8112901a5b0092",
    "glibc": "busybox@sha256:42a9ff19af33811ee466b24c66384461d37c32dcc28d1d27d49b8aca816bd5ee",
}
CANARY = "sk-ant-api03-u3a-canary-" + "x" * 24


def _docker(
    *args: str, stdin: str | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    run = subprocess.run(
        ["docker", *args], input=stdin, capture_output=True, text=True, timeout=600
    )
    if check and run.returncode != 0:
        raise AssertionError(f"docker {' '.join(args)}: {run.stderr[-2000:]}")
    return run


def _docker_ready() -> bool:
    try:
        return _docker("info", check=False).returncode == 0
    except Exception:
        return False


def _build_task(tag: str, libc: str, project: str) -> str:
    # The compose label stands in for what `compose build` puts on a task
    # image; the layer must not carry it (teardown's `down --rmi all`
    # deletes images labelled with its project).
    dockerfile = (
        f"FROM {BASES[libc]}\n"
        "RUN echo task > /task.txt\n"
        f"LABEL com.docker.compose.project={project}\n"
        "WORKDIR /app\n"
    )
    _docker("build", "--quiet", "--tag", tag, "-", stdin=dockerfile)
    return _docker("image", "inspect", "--format", "{{.Id}}", tag).stdout.strip()


def _layer_files(tag: str, workdir: Path) -> tuple[list[str], bytes]:
    """The paths the image's last layer adds, and the whole saved image."""
    archive = workdir / "image.tar"
    _docker("save", "--output", str(archive), tag)
    raw = archive.read_bytes()
    with tarfile.open(archive) as outer:
        listing = outer.extractfile("manifest.json")
        assert listing is not None
        manifest = json.load(listing)
        last = outer.extractfile(manifest[0]["Layers"][-1])
        assert last is not None
        with tarfile.open(fileobj=io.BytesIO(last.read()), mode="r:*") as layer:
            names = [m.name for m in layer.getmembers()]
    return names, raw


@pytest.fixture
def run_id(monkeypatch, tmp_path):
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    # What a trial has around it: keys in the environment, credential files,
    # skills and trial logs in the working folder. None may reach a layer.
    monkeypatch.setenv("ANTHROPIC_API_KEY", CANARY)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(f"OPENAI_API_KEY={CANARY}\n")
    (tmp_path / "auth.json").write_text(json.dumps({"token": CANARY}))
    (tmp_path / "skills" / "demo").mkdir(parents=True)
    (tmp_path / "skills" / "demo" / "SKILL.md").write_text(CANARY)
    (tmp_path / "logs" / "agent").mkdir(parents=True)
    rid = uuid.uuid4().hex[:12]
    yield rid
    listing = _docker("image", "ls", "--format", "{{.Repository}}:{{.Tag}}").stdout
    layers = _docker(
        "image",
        "ls",
        "--filter",
        f"label=benchflow.layer.agent=fake-agent-{rid}",
        "--format",
        "{{.Repository}}:{{.Tag}}",
    ).stdout.split()
    for ref in [*layers, *(line for line in listing.splitlines() if rid in line)]:
        _docker("image", "rm", "--force", ref, check=False)
    for ref in layers:
        key = ref.rpartition("-")[2]
        _docker(
            "builder",
            "prune",
            "--all",
            "--force",
            "--filter",
            f'description~="benchflow[.]layer=agent key={key};"',
            check=False,
        )
    DockerLayers._claims.clear()


def _recipe(rid: str) -> AgentLayerRecipe:
    return AgentLayerRecipe(
        agent=f"fake-agent-{rid}",
        version="1.0.0",
        install_cmd=(
            "mkdir -p /opt/runtime/bin && "
            f"printf '#!/bin/sh\\necho fake-agent 1.0.0 {rid}\\n' > /opt/runtime/bin/fake-agent && "
            "chmod 755 /opt/runtime/bin/fake-agent"
        ),
    )


@pytest.mark.parametrize("libc", ["musl", "glibc", "none"])
async def test_layer_built_from_the_task_image_reused_and_clean(run_id, libc, tmp_path):
    task = f"bf-u3a-test-{run_id}:{libc}"
    _build_task(task, libc, project=f"trial-{run_id}")
    layers = DockerLayers()
    recipe = _recipe(run_id)

    first = await layers.ensure_agent_layer(recipe, task, prune=False)
    assert first.status == "miss"
    assert first.base.libc == libc
    second = await layers.ensure_agent_layer(recipe, task, prune=False)
    assert (second.status, second.key) == ("hit", first.key)
    assert second.seconds < first.seconds

    # FROM the task image: its layers, then exactly one more.
    base_layers = json.loads(
        _docker("image", "inspect", "--format", "{{json .RootFS.Layers}}", task).stdout
    )
    built = json.loads(_docker("image", "inspect", first.tag).stdout)[0]
    assert built["RootFS"]["Layers"][: len(base_layers)] == base_layers
    assert len(built["RootFS"]["Layers"]) == len(base_layers) + 1
    labels = built["Config"]["Labels"]
    assert labels[LAYER_LABEL] == AGENT
    assert labels["com.docker.compose.project"] == ""
    assert built["Config"]["WorkingDir"] == "/app"

    # The layer adds the agent and nothing else: no credential, no per-trial
    # file (logs, the working folder, /tmp), no skill.
    names, raw = _layer_files(first.tag, tmp_path)
    assert "opt/runtime/bin/fake-agent" in names, names
    for name in names:
        assert name.rstrip("/") in {
            "opt",
            "opt/runtime",
            "opt/runtime/bin",
        } or name.startswith("opt/runtime/bin/"), (
            f"unexpected path in the layer: {name}"
        )
    for bad in (
        "auth",
        ".env",
        "credential",
        "skills",
        "SKILL",
        "logs",
        "app/",
        "tmp/",
        "root/",
        "home/",
    ):
        assert not any(bad in n for n in names), (bad, names)
    assert CANARY.encode() not in raw
    history = _docker("image", "history", "--no-trunc", first.tag).stdout
    assert CANARY not in history

    out = _docker("run", "--rm", first.tag, "/opt/runtime/bin/fake-agent").stdout
    assert f"fake-agent 1.0.0 {run_id}" in out


async def test_a_rebuilt_task_image_reuses_the_layer(run_id):
    """Teardown removes the built task image; the next trial rebuilds it
    from BuildKit's cache with a new id. The layer is still a hit."""
    task = f"bf-u3a-test-{run_id}:rebuild"
    first_id = _build_task(task, "musl", project=f"trial-a-{run_id}")
    layers = DockerLayers()
    first = await layers.ensure_agent_layer(_recipe(run_id), task, prune=False)
    _docker("image", "rm", task)
    second_id = _build_task(task, "musl", project=f"trial-b-{run_id}")
    assert second_id != first_id
    again = await layers.ensure_agent_layer(_recipe(run_id), task, prune=False)
    assert (again.status, again.key) == ("hit", first.key)


async def test_the_prune_removes_the_layer_and_its_build_records(run_id):
    task = f"bf-u3a-test-{run_id}:prune"
    _build_task(task, "musl", project=f"trial-{run_id}")
    layers = DockerLayers()
    result = await layers.ensure_agent_layer(_recipe(run_id), task, prune=False)
    DockerLayers.release(result.key, daemon=result.daemon_id)
    du = _docker("builder", "du", "--verbose", check=False)
    if du.returncode != 0:
        pytest.skip("this daemon lists no BuildKit records (`docker builder du`)")
    assert result.key in du.stdout

    report = await layers.prune(keep=0, only=[result.key])
    assert report.removed == [result.key]
    assert (
        _docker(
            "image", "inspect", layer_tag(AGENT, result.key), check=False
        ).returncode
        != 0
    )
    assert result.key not in _docker("builder", "du", "--verbose").stdout
    # The task image is not BenchFlow's layer and stays.
    assert _docker("image", "inspect", task, check=False).returncode == 0


async def test_a_layer_a_container_uses_is_not_pruned(run_id):
    task = f"bf-u3a-test-{run_id}:inuse"
    _build_task(task, "musl", project=f"trial-{run_id}")
    layers = DockerLayers()
    result = await layers.ensure_agent_layer(
        _recipe(run_id), task, alias=f"u3a-other-{run_id}:1", prune=False
    )
    DockerLayers.release(result.key, daemon=result.daemon_id)
    name = f"bf-u3a-test-{run_id}"
    # Through a tag that is not a trial's: only the container says it is used.
    _docker("create", "--name", name, f"u3a-other-{run_id}:1", "true")
    try:
        report = await layers.prune(keep=0, only=[result.key])
        assert result.key in report.in_use and not report.removed
        assert (
            _docker(
                "image", "inspect", layer_tag(AGENT, result.key), check=False
            ).returncode
            == 0
        )
    finally:
        _docker("rm", "--force", name, check=False)


async def test_a_leased_layer_and_an_image_committed_from_it_survive_the_prune(
    run_id,
):
    """A verifier-recovery lease tags the layer after its trial's teardown,
    and a snapshot committed from a layer container inherits the layer's
    labels. prune(keep=0) keeps the leased layer with both tags; once the
    lease goes the layer is removed and the committed image stays whole."""
    task = f"bf-u3a-test-{run_id}:lease"
    _build_task(task, "musl", project=f"trial-{run_id}")
    layers = DockerLayers()
    result = await layers.ensure_agent_layer(_recipe(run_id), task, prune=False)
    DockerLayers.release(result.key, daemon=result.daemon_id)
    tag = layer_tag(AGENT, result.key)
    lease = f"benchflow-recovery-lease:{run_id}"
    snap = f"bf-snap-u3a-{run_id}:1"
    name = f"bf-u3a-test-{run_id}"
    _docker("tag", tag, lease)
    _docker("create", "--name", name, tag, "true")
    try:
        _docker("commit", name, snap)  # labels inherited, as any commit
    finally:
        _docker("rm", "--force", name, check=False)
    inherited = json.loads(
        _docker("image", "inspect", "--format", "{{json .Config.Labels}}", snap).stdout
    )
    assert inherited[LAYER_LABEL] == AGENT

    report = await layers.prune(keep=0, only=[result.key])
    assert result.key in report.in_use and not report.removed
    lease_tags = _docker(
        "image", "inspect", "--format", "{{json .RepoTags}}", lease
    ).stdout
    assert tag in lease_tags and lease in lease_tags
    assert _docker("image", "inspect", snap, check=False).returncode == 0

    _docker("image", "rm", lease)
    report = await layers.prune(keep=0, only=[result.key])
    assert report.removed == [result.key] and report.bytes_removed > 0
    assert _docker("image", "inspect", tag, check=False).returncode != 0
    out = _docker("run", "--rm", snap, "/opt/runtime/bin/fake-agent").stdout
    assert f"fake-agent 1.0.0 {run_id}" in out


async def test_a_dead_processs_trial_tag_is_untagged_and_its_layer_pruned(
    run_id, monkeypatch
):
    """A trial whose process was killed before teardown left its own tag on
    the layer. The prune untags it (its process is gone, no container uses
    it) and removes the layer; a live process's tag would have kept it."""
    task = f"bf-u3a-test-{run_id}:stale"
    _build_task(task, "musl", project=f"trial-{run_id}")
    layers = DockerLayers()
    result = await layers.ensure_agent_layer(_recipe(run_id), task, prune=False)
    DockerLayers.release(result.key, daemon=result.daemon_id)
    dead_pid = 4_000_001
    real = _docker_sweep._pid_exists
    monkeypatch.setattr(
        _docker_sweep, "_pid_exists", lambda pid: pid != dead_pid and real(pid)
    )
    host = env_layers._host_hash(_docker_sweep.current_process().host)
    dead = f"{host}-{dead_pid}-1-00000000-0000-0000-0000-000000000000"
    stale = f"{ALIAS_PREFIX}u3c-{run_id}:{dead}"
    live = f"{ALIAS_PREFIX}u3c-live-{run_id}:{alias_owner_tag()}"
    tag = layer_tag(AGENT, result.key)
    _docker("tag", tag, stale)
    _docker("tag", tag, live)

    report = await layers.prune(keep=0, only=[result.key])
    assert stale in report.aliases_removed
    assert result.key in report.in_use and not report.removed
    assert _docker("image", "inspect", stale, check=False).returncode != 0

    _docker("image", "rm", live)
    report = await layers.prune(keep=0, only=[result.key])
    assert report.removed == [result.key]
    assert _docker("image", "inspect", tag, check=False).returncode != 0
