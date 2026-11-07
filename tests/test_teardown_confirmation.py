"""Guards U10's confirmation fixes after 22672877; all deletion is simulated."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _recovery_baseline
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.env_layers import DockerLayers
from tests.test_compose_teardown import FakeDaemon, _docker_sandbox
from tests.test_docker_recovery_baseline import _lease


class InventoryDaemon(FakeDaemon):
    """Classic-store reference semantics, including digests and parent pruning."""

    def __init__(self, project):
        super().__init__(project, {})
        self.digests = {}
        self.parent_present = True

    async def cli(self, args, check=True, *, raw_output=False):
        if args[:2] == ["image", "inspect"]:
            self.cli_calls.append(list(args))
            ids = list(dict.fromkeys(self._find(ref) for ref in args[2:]))
            infos = [
                {
                    "Id": image_id,
                    "RepoTags": self.images[image_id]["tags"],
                    "RepoDigests": self.digests.get(image_id, []),
                }
                for image_id in ids
                if image_id is not None
            ]
            return ExecResult(stdout=json.dumps(infos), stderr="", return_code=0)
        if args[:2] == ["image", "tag"]:
            self.cli_calls.append(list(args))
            image_id = self._find(args[2])
            if image_id is None:
                return ExecResult(stdout="", stderr="No such image", return_code=1)
            self.images[image_id]["tags"].append(args[3])
            return ExecResult(stdout="", stderr="", return_code=0)
        if args[:2] == ["image", "rm"]:
            self.cli_calls.append(list(args))
            for ref in args[2:]:
                if ref.startswith("--"):
                    continue
                image_id = self._find(ref)
                if image_id is None:
                    continue
                self._rm(ref)
                if image_id not in self.images and "--no-prune" not in args:
                    self.parent_present = False
            return ExecResult(stdout="", stderr="", return_code=0)
        result = await super().cli(args, check=check)
        if args[:2] == ["image", "ls"]:
            if "--quiet" in args:
                result.stdout = "\n".join(
                    image_id
                    for image_id, image in self.images.items()
                    if image["labels"].get("com.docker.compose.project") == self.project
                )
            else:
                # The old formatter emits repo:<none> for a digest-only image.
                for image_id, digests in self.digests.items():
                    if not self.images[image_id]["tags"] and digests:
                        result.stdout = result.stdout.replace(
                            f"{image_id} <none>:<none>",
                            f"{image_id} {digests[0].split('@')[0]}:<none>",
                        )
        return result


def rig(tmp_path):
    sandbox, _ = _docker_sandbox(tmp_path)
    daemon = InventoryDaemon(sandbox.compose_project_name)
    sandbox._docker_cli = daemon.cli
    return sandbox, daemon


@pytest.mark.parametrize("prebuilt", [False, True])
async def test_layer_release_keeps_prebuilt_and_reassigned_base_names(
    tmp_path, prebuilt
):
    """Guards U10-CONFIRM-1: layered teardown must not resolve a borrowed name."""
    sandbox, daemon = rig(tmp_path)
    base = "ubuntu:24.04" if prebuilt else "bf__hello:latest"
    daemon.add_image("sha256:borrowed", [base], "other-project")
    alias, cached = "bf__layer__trial:latest", "benchflow-layer:agent-cache"
    daemon.add_image("sha256:layer", [cached, alias])
    sandbox._layers_base_ref = base
    sandbox._agent_layer = SimpleNamespace(
        daemon_id="dummy-daemon", key="test-key", image=alias, tag=cached
    )

    await sandbox.stop(delete=True)

    assert daemon._find(base) == "sha256:borrowed"
    assert daemon.tags() == {base, cached}


@pytest.mark.parametrize("tags", [[], ["bf__hello:latest"]])
async def test_digest_retention_survives_teardown(tmp_path, tags):
    """Guards U10-CONFIRM-2: RepoDigests retain builds with or without tags."""
    sandbox, daemon = rig(tmp_path)
    daemon.add_image("sha256:retained", tags, sandbox.compose_project_name)
    digest = "registry.test/archive/hello@sha256:" + "d" * 64
    daemon.digests["sha256:retained"] = [digest]

    await sandbox.stop(delete=True)

    assert "sha256:retained" in daemon.images
    assert not any(c[:2] == ["image", "rm"] for c in daemon.cli_calls)


@pytest.mark.parametrize("at", ["inventory", "pin", "remove"])
async def test_a_new_retained_tag_survives_every_cleanup_boundary(tmp_path, at):
    """Guards U10-CONFIRM-3: a sole tag added after listing must survive."""
    sandbox, daemon = rig(tmp_path)
    daemon.add_image("sha256:old", [], sandbox.compose_project_name)
    archive = "archive/hello:keep"
    added = False

    async def cli(args, check=True, *, raw_output=False):
        nonlocal added
        if not added and (
            (at == "pin" and args[:2] in (["image", "tag"], ["image", "rm"]))
            or (at == "remove" and args[:2] == ["image", "rm"])
        ):
            daemon.images["sha256:old"]["tags"].append(archive)
            added = True
        result = await daemon.cli(args, check=check)
        if at == "inventory" and args[:2] == ["image", "ls"]:
            daemon.images["sha256:old"]["tags"].append(archive)
        return result

    sandbox._docker_cli = cli
    await sandbox.stop(delete=True)

    assert daemon._find(archive) == "sha256:old"
    removals = [c for c in daemon.cli_calls if c[:2] == ["image", "rm"]]
    assert all("sha256:old" not in c for c in removals)


async def test_shared_build_name_is_retained_but_dangling_build_is_deleted(tmp_path):
    """Guards U10-CONFIRM-3 with a positive deletion control and a shared name."""
    sandbox, daemon = rig(tmp_path)
    daemon.built("sha256:shared", "bf__hello:latest", "main")
    daemon.add_image("sha256:dangling", [], sandbox.compose_project_name)

    await sandbox.stop(delete=True)

    assert daemon.tags() == {"bf__hello:latest"}
    assert "sha256:dangling" not in daemon.images
    assert daemon.parent_present


@pytest.mark.parametrize("path", ["layer", "snapshot", "deferred-snapshot", "baseline"])
async def test_every_deferred_release_preserves_an_unowned_parent(
    tmp_path, monkeypatch, path
):
    """Guards U10-CONFIRM-4 across the complete release lifecycle."""
    sandbox, daemon = rig(tmp_path)
    ref = "bf-snap-trial:release"
    daemon.add_image("sha256:child", [ref])
    if path == "layer":
        sandbox._agent_layer = SimpleNamespace(
            daemon_id="dummy-daemon", key="test-key", image=ref, tag="cached"
        )
        await sandbox._release_layers(delete=True)
    elif path == "snapshot":
        assert await sandbox._remove_snapshot_image(ref)
    elif path == "deferred-snapshot":
        sandbox._deferred_snapshot_refs.add(ref)
        await sandbox._delete_deferred_snapshots()
        assert not sandbox._deferred_snapshot_refs
    else:
        baseline = _lease()
        daemon.images["sha256:child"]["tags"] = [baseline.lease_tag]

        async def docker(*args):
            result = await daemon.cli(list(args))
            return result.return_code, result.stdout

        monkeypatch.setattr(_recovery_baseline, "_docker", docker)
        assert await _recovery_baseline.release_baseline(baseline)

    assert "sha256:child" not in daemon.images
    assert daemon.parent_present
    assert all("--no-prune" in c for c in daemon.cli_calls if c[:2] == ["image", "rm"])


@pytest.mark.parametrize("kind", ["agent", "proxy", "cache"])
async def test_layer_build_and_cache_release_do_not_prune_parents(monkeypatch, kind):
    """Guards U10-CONFIRM-4's layer-cache and temporary base-pin siblings."""
    from tests.test_env_layers_agent import RECIPE, FakeDocker
    from tests.test_env_layers_proxy import RECIPE as PROXY_RECIPE
    from tests.test_env_layers_proxy import ProxyFakeDocker

    daemon = ProxyFakeDocker() if kind == "proxy" else FakeDocker()
    removals = []

    async def docker(*args, **kwargs):
        if args[:2] == ("image", "rm"):
            removals.append(args)
            # The legacy double predates --no-prune; preserve its tag semantics.
            args = tuple(arg for arg in args if arg != "--no-prune")
        return await daemon.run(*args, **kwargs)

    monkeypatch.setattr(DockerLayers, "_docker", staticmethod(docker))
    monkeypatch.setattr(DockerLayers, "_claims", {})
    daemon.add_image("bf__task")
    layers = DockerLayers()
    if kind == "proxy":
        await layers.ensure_proxy_layer(PROXY_RECIPE, "bf__task", prune=False)
    else:
        layer = await layers.ensure_agent_layer(RECIPE, "bf__task", prune=False)
        if kind == "cache":
            DockerLayers.release(layer.key, daemon=layer.daemon_id)
            removals.clear()
            await layers.prune(keep=0)
    assert removals
    assert all("--no-prune" in c for c in removals)
