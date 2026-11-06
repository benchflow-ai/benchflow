"""Guards U3c-CONFIRM-1 (fd85c046) against default mounts hiding installs.

Execute the actual Dockerfile RUN command with dummy install bytes under
pytest's temporary directory. Docker's metadata/build transport is simulated;
the filesystem checks and installer are real shell commands.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import PurePosixPath

import pytest

from benchflow.sandbox import env_layers
from benchflow.sandbox.env_layers import AgentLayerRecipe, DockerLayers, LayerError
from benchflow.task.paths import SandboxPaths
from tests.test_env_layers_agent import FakeDocker


@pytest.fixture
def mounts(tmp_path, monkeypatch):
    paths = {}
    for name in ("agent", "verifier", "artifacts"):
        path = tmp_path / "logs" / name
        monkeypatch.setattr(SandboxPaths, f"{name}_dir", PurePosixPath(path))
        paths[name] = path
    return paths


async def build_recipe(monkeypatch, tmp_path, install, *, remote=False, on_proxy=False):
    docker = FakeDocker()
    docker.add_image("task")
    recipe = AgentLayerRecipe(agent="dummy", version="1.0.0", install_cmd=install)

    async def run(layers, *args, **kwargs):
        if args[0] == "build":
            text = kwargs["stdin"].decode()
            line = next(line for line in text.splitlines() if line.startswith("RUN "))
            command = json.loads(line[4:])
            result = subprocess.run(
                command, capture_output=True, text=True, cwd=tmp_path
            )
            if result.returncode:
                return env_layers._Run(result.stdout, result.stderr, result.returncode)
        return await docker.run(*args, **kwargs)

    monkeypatch.setattr(DockerLayers, "_docker", run)
    layers = DockerLayers(
        env={"DOCKER_HOST": "ssh://dummy.invalid"} if remote else None
    )
    base = await layers.inspect_base("task") if on_proxy else None
    result = await layers.ensure_agent_layer(recipe, "task", base=base, prune=False)
    DockerLayers.release(result.key, daemon=result.daemon_id)
    reused = await layers.ensure_agent_layer(recipe, "task", base=base, prune=False)
    assert reused.status == "hit"
    DockerLayers.release(reused.key, daemon=reused.daemon_id)
    return result


@pytest.mark.parametrize("mount", ["agent", "verifier", "artifacts"])
@pytest.mark.parametrize(
    "variant",
    [
        "direct",
        "hidden",
        "hidden-directory",
        "dangling-entry",
        "fifo",
        "absolute-link",
        "relative-link",
        "chain",
        "ancestor-link",
        "preinstalled",
        "mount-link",
        "logs-link",
        "mount-file",
        "recipe-exit",
    ],
)
@pytest.mark.parametrize("remote", [False, True], ids=["local", "remote"])
@pytest.mark.parametrize("on_proxy", [False, True], ids=["task", "proxy"])
async def test_default_mounts_cannot_hide_cached_bytes(
    tmp_path, monkeypatch, mounts, mount, variant, remote, on_proxy
):
    """Guards fd85c046 for every built-in mount, aliases, and inherited bytes."""
    destination = mounts[mount]
    destination.parent.mkdir(parents=True)
    install_path = destination / "dummy-agent"
    if variant in {"absolute-link", "relative-link", "chain", "ancestor-link"}:
        alias = tmp_path / "bin"
        if variant == "relative-link":
            alias.symlink_to(f"logs/{mount}")
        elif variant == "chain":
            (tmp_path / "hop").symlink_to(destination)
            alias.symlink_to("hop")
        else:
            alias.symlink_to(destination)
        install_path = alias / (
            "nested/dummy-agent" if variant == "ancestor-link" else "dummy-agent"
        )
        destination.mkdir()
    elif variant == "hidden":
        install_path = destination / ".dummy-agent"
    elif variant == "hidden-directory":
        install_path = destination / "..cache" / "dependency"
    elif variant == "preinstalled":
        destination.mkdir()
        install_path.write_text("trusted preinstalled dependency")
        install_path = tmp_path / "bin" / "dummy-agent"
    elif variant == "mount-link":
        (tmp_path / "bin").mkdir()
        destination.symlink_to(tmp_path / "bin")
        install_path = tmp_path / "bin" / "dummy-agent"
    elif variant == "logs-link":
        destination.parent.rmdir()
        (tmp_path / "real-logs").mkdir()
        destination.parent.symlink_to(tmp_path / "real-logs")
        install_path = tmp_path / "bin" / "dummy-agent"
    elif variant == "mount-file":
        destination.write_text("not a directory")
        install_path = tmp_path / "bin" / "dummy-agent"
    elif variant in {"dangling-entry", "fifo"}:
        destination.mkdir()
        if variant == "dangling-entry":
            install_path.symlink_to("missing")
        else:
            os.mkfifo(install_path)
        install_path = tmp_path / "bin" / "dummy-agent"
    install = f"mkdir -p {shlex.quote(str(install_path.parent))} && printf dummy > {shlex.quote(str(install_path))}"
    if variant == "recipe-exit":
        install += "; exit 0"
    with pytest.raises(LayerError, match="runtime log mount"):
        await build_recipe(
            monkeypatch, tmp_path, install, remote=remote, on_proxy=on_proxy
        )


@pytest.mark.parametrize("existing", [False, True])
async def test_missing_or_empty_log_mounts_preserve_layer_reuse(
    tmp_path, monkeypatch, mounts, existing
):
    """Guards U3c-CONFIRM-1 without disabling ordinary agent layers."""
    if existing:
        for path in mounts.values():
            path.mkdir(parents=True)
    adjacent = tmp_path / "logs" / "agent-adjacent"
    adjacent.mkdir(parents=True)
    (adjacent / "keep").write_text("not hidden by a built-in mount")
    install = f"printf dummy > {shlex.quote(str(tmp_path / 'dummy-agent'))}"
    result = await build_recipe(monkeypatch, tmp_path, install)
    assert result.status == "miss"


async def test_failed_installer_cannot_be_masked_by_mount_guard(
    tmp_path, monkeypatch, mounts
):
    """Guards U3c-CONFIRM-1's wrapper without changing recipe errexit."""
    install = "set -e\nfalse\nprintf should-not-run"
    with pytest.raises(LayerError):
        await build_recipe(monkeypatch, tmp_path, install)


def test_mount_destinations_are_part_of_the_agent_cache_key(monkeypatch):
    """Guards U3c-CONFIRM-1 when the default mount layout changes later."""
    from tests.test_env_layers_agent import BASE, RECIPE

    before = env_layers.agent_layer_key(RECIPE, BASE)
    monkeypatch.setattr(SandboxPaths, "agent_dir", PurePosixPath("/other-agent-logs"))
    assert env_layers.agent_layer_key(RECIPE, BASE) != before
