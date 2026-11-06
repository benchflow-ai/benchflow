"""Guards FF-sw-daytona-C1/C2/C3 against the gaps at 07f2ec9d."""

import hashlib
import subprocess
from types import SimpleNamespace

import pytest

from benchflow.sandbox import daytona_layers as layers
from benchflow.sandbox import daytona_reaper as reaper
from benchflow.sandbox.daytona_content_snapshots import check_recipe, owner_prefix
from tests.test_daytona_snapshot_owner import _Client, _snap


@pytest.mark.parametrize("owner", ["a", "team", "Team_A", "abc.def", "z" * 20])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_snapshot_owner_field_cannot_be_extended(monkeypatch, owner, depth):
    """Guard C2 at 07f2ec9d: complete owner fields precede variable text."""
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", owner)
    own_name = reaper.benchflow_snapshot_name("task")
    own_prefix = reaper.owner_snapshot_prefix()
    other = owner
    for _ in range(depth):
        other += "-" + hashlib.sha256(other.encode()).hexdigest()[:6]
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", other)
    foreign_name = reaper.benchflow_snapshot_name("task")
    assert not foreign_name.startswith(own_prefix)
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", owner)
    client = _Client([[_snap(own_name, 10), _snap(foreign_name, 10)]])
    reaper.reap_stale_snapshots(client, max_age_minutes=0)
    assert client.deleted == [own_name]


@pytest.mark.parametrize("owner", ["a", "team", "Team_A"])
def test_snapshot_cleanup_leaves_legacy_names(monkeypatch, owner):
    """Guard C2: old ambiguous names never authorize automatic deletion."""
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", owner)
    tag = hashlib.sha256(owner.encode()).hexdigest()[:6]
    legacy = f"bf-snap-{reaper._snapshot_name_part(owner)}-{tag}-task-0123456789ab"
    client = _Client([[_snap(legacy, 10)]])
    reaper.reap_stale_snapshots(client, max_age_minutes=0)
    assert client.deleted == []
    assert not owner_prefix().startswith(f"bf-img-{reaper._snapshot_name_part(owner)}-")


@pytest.mark.parametrize("keyword", ["ONBUILD", "onbuild", "OnBuIlD"])
@pytest.mark.parametrize(
    "instruction",
    [
        "RUN --mount=from=alpine:latest,target=/src cat /src/version",
        "RUN --mount=type=cache,from=toolchain:latest,target=/tc true",
        "RUN --mount=type=cache,target=/cache --mount=from=alpine,target=/src true",
        "RUN \\\n --mount=from=alpine,target=/src true",
        "COPY --from=alpine:latest /version /version",
        "ADD https://example.invalid/data /data",
        "RUN echo harmless",
        "COPY seed /seed",
    ],
)
def test_deferred_instructions_are_not_snapshot_inputs(keyword, instruction):
    """Guard C3 at 07f2ec9d: unmodeled deferred instructions always fall back."""
    recipe = "FROM example/base@sha256:" + "1" * 64 + " AS base\n"
    recipe += f"{keyword} {instruction}\nFROM base\n"
    assert check_recipe(recipe)[1] is not None


def test_proxy_recipe_keeps_its_runtime_outside_opt():
    """Guard C1: /opt workspaces cannot own a cached proxy or its Python."""
    plan = layers.plan_install_layers(agent="openhands", model="openai/gpt-4o")
    assert plan.proxy.venv.startswith("/opt/benchflow/")
    assert "UV_PYTHON_INSTALL_DIR=/opt/benchflow/" in plan.proxy.script


@pytest.mark.parametrize("entry", ["bin/python", "bin/litellm", "lib", "pyvenv.cfg"])
async def test_agent_owned_proxy_is_never_selected(tmp_path, monkeypatch, entry):
    """Guard C1 at 07f2ec9d using real shell metadata and dummy executables."""
    root = tmp_path / "cache"
    venv = root / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "lib").mkdir()
    (venv / "pyvenv.cfg").write_text("dummy")
    for name in ("python", "litellm"):
        path = venv / "bin" / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    (venv / entry).chmod(0o777)
    monkeypatch.setattr(layers, "PROXY_LAYER_VENV", str(venv))
    monkeypatch.setattr(layers, "PROXY_LAYER_ROOT", str(root), raising=False)
    monkeypatch.setattr(layers, "LAYER_KEY_DIR", str(root))
    layer = layers.InstallLayer(
        kind="proxy", name="litellm", script="dummy", venv=str(venv)
    )
    from pathlib import Path

    Path(layer.marker).write_text("dummy-key")

    class Sandbox:
        def cached_install_layer(self, *_):
            return layer, "dummy-key"

        async def exec(self, command, **_):
            result = subprocess.run(
                ["sh", "-c", command], capture_output=True, text=True
            )
            return SimpleNamespace(return_code=result.returncode, stdout=result.stdout)

    assert (
        await layers.cached_layer_in_sandbox(
            Sandbox(), "proxy", "litellm", script="dummy"
        )
        is None
    )
