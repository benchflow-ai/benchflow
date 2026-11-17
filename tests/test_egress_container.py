"""The verifier's egress proxy container runs only BenchFlow's pinned helper image.

The lead approved the proxy's own container as an own-image exec
(tests/test_c8_r8_routes.py OWN_IMAGE_EXECS) on these conditions: its image is
the helper image this process builds from the pinned recipe, and a test fails
if it is anything else.
"""

from __future__ import annotations

import pytest

from benchflow.sandbox import _egress_container as ec
from benchflow.sandbox import _metadata_helper as helper_mod
from tests.test_metadata_helper import (  # noqa: F401
    IMAGE_ID,
    MAIN,
    FakeDocker,
    fresh_images,
)

PINNED_BASE = "alpine:3.24@sha256:294b683cb724975bec92580e1e685676bd4b50bda910ddb8c51d4cabeaec77e6"


def test_the_helper_recipe_is_pinned_by_digest():
    assert helper_mod.HELPER_BASE == PINNED_BASE
    assert helper_mod.HELPER_DOCKERFILE.startswith(f"FROM {PINNED_BASE}\n")
    for package in helper_mod.HELPER_PACKAGES:
        assert "=" in package, package  # every package at a pinned version


@pytest.mark.asyncio
async def test_the_egress_container_runs_only_the_pinned_helper_image(monkeypatch):
    """The run names the image ID this process's build of the pinned recipe
    gave (never a tag, never the task's image); a different image fails."""
    docker = FakeDocker()
    docker.exec_out = "placed\n"
    helper_mod._READY["d"] = IMAGE_ID
    box = ec.EgressContainer(docker, daemon="d", name="t-bf-egress", labels={})
    await box.start(MAIN, {"proxy.py": b"x"}, ["python3", "proxy.py"], lifetime_sec=60)
    (run,) = docker.runs()
    image = run[run.index("--entrypoint") + 2]
    assert image == IMAGE_ID
    assert helper_mod.HELPER_TAG not in run and "alpine" not in " ".join(run)
    assert run[run.index("--network") + 1] == f"container:{MAIN}"
    assert [run[i + 1] for i, a in enumerate(run) if a == "--cap-add"] == ["NET_ADMIN"]
    # Every exec after the run targets that container by the id it printed.
    assert all(
        c[1] == box.container or c[2] == box.container for c in docker.of("exec")
    )

    async def other_image(docker_, daemon):
        return "sha256:" + "ee" * 32

    monkeypatch.setattr(helper_mod, "ensure_image", other_image)
    other = ec.EgressContainer(docker, daemon="d", name="t-bf-egress", labels={})
    await other.start(MAIN, {}, ["python3"], lifetime_sec=60)
    last = docker.runs()[-1]
    assert last[last.index("--entrypoint") + 2] != IMAGE_ID  # the guard sees it
