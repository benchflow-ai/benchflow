"""Workspace capture through ``bfx tar-out`` in a real container: the pinned
bfx bound read-only at ``/.bfx`` gives the same manifest and exclusions as
the image's own python3 capture, and runs no program of the image.

Skipped without a Docker daemon or a pinned bfx in this checkout.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid

import pytest

from benchflow.review.evidence import capture_workspace
from tests.test_c8_protected_runtime_container import _docker_ready, _sandbox

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

IMAGE = "python:3.12-slim"
LABEL = os.environ.get("BENCHFLOW_TEST_CONTAINER_LABEL", "target=TR-CORE-capture")
PREFIX = os.environ.get("BENCHFLOW_TEST_CONTAINER_PREFIX", "trcore-cap-")

_TREE = r"""
set -e
useradd -m agent
mkdir -p /app/src/pkg /app/empty /app/.codex /app/.cache/pip
echo 'x = 1' > /app/src/pkg/mod.py
echo 42 > /app/answer.txt
printf '#!/bin/sh\necho hi\n' > /app/run.sh && chmod 755 /app/run.sh
ln -s answer.txt /app/link-to-answer
ln /app/answer.txt /app/hardlink.txt
ln -s /etc/hostname /app/escape
echo '{"token": "x"}' > /app/.codex/auth.json
echo cached > /app/.cache/pip/blob
mkfifo /app/pipe
chown -R agent:agent /app
"""


@pytest.fixture
def box():
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    from benchflow._bfx import delivery
    from benchflow._bfx.locate import BfxUnavailable

    try:
        mount = delivery.run_args(
            delivery.BIND, source=str(delivery.host_dir("x86_64"))
        )
    except BfxUnavailable as exc:
        pytest.skip(f"no pinned bfx in this checkout: {exc}")
    name = PREFIX + uuid.uuid4().hex[:8]
    key, _, value = LABEL.partition("=")
    run = subprocess.run(
        ["docker", "run", "-d", "--name", name, "--label", f"{key}={value}",
         "--network", "none", *mount, IMAGE, "sleep", "600"],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start {IMAGE}: {run.stderr.strip()}")
    try:
        made = subprocess.run(
            ["docker", "exec", name, "/bin/sh", "-c", _TREE],
            capture_output=True,
            text=True,
        )
        assert made.returncode == 0, made.stderr
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _entries(manifest) -> list[tuple]:
    return [(e.path, e.kind, e.size, e.sha256, e.link_target) for e in manifest.entries]


@pytest.mark.asyncio
async def test_bfx_capture_matches_the_image_python_capture(box, tmp_path):
    plain = _sandbox(box)
    before = await capture_workspace(plain, "/app", tmp_path / "python")

    with_bfx = _sandbox(box)
    with_bfx.bfx_delivery = "bind"
    commands: list[list[str]] = []
    run = with_bfx._run_docker_container_command

    async def recorded(argv, *args, **kwargs):
        commands.append(list(argv))
        return await run(argv, *args, **kwargs)

    with_bfx._run_docker_container_command = recorded
    after = await capture_workspace(with_bfx, "/app", tmp_path / "bfx")

    assert after.workspace == before.workspace == "/app"
    assert _entries(after) == _entries(before)
    assert sorted(x.model_dump_json() for x in after.exclusions) == sorted(
        x.model_dump_json() for x in before.exclusions
    )
    reasons = {e.original_path: e.reason for e in after.exclusions}
    assert reasons["/app/escape"] == "symlink_escape"
    assert reasons["/app/pipe"] == "special_file"
    assert any(
        path.startswith("/app/.codex") and why == "credential"
        for path, why in reasons.items()
    ), reasons
    # No exec of the image's programs: only bfx ran in the container.
    assert not any("python3" in " ".join(argv) for argv in commands), commands
    assert (tmp_path / "bfx" / "workspace" / "hardlink.txt").read_text() == "42\n"


@pytest.mark.asyncio
async def test_a_linked_workspace_path_is_captured_as_before(box, tmp_path):
    subprocess.run(["docker", "exec", box, "ln", "-s", "/app", "/work"], check=True)
    with_bfx = _sandbox(box)
    with_bfx.bfx_delivery = "bind"
    manifest = await capture_workspace(with_bfx, "/work", tmp_path / "bfx")
    assert manifest.workspace == "/app"
