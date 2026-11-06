"""Real Docker/Compose publication checks for R12 fixes (7c24f4d88).

Opt in on the builder VM with BENCHFLOW_R13_DOCKER=1. The sole bind contains
only this test's fresh temporary files; no socket or host namespace is mounted.
"""

import os
import shutil
import subprocess
import uuid

import pytest

from benchflow._utils import reachable_write as rw
from benchflow.sandbox.docker import DockerSandbox


@pytest.fixture
def compose_output(tmp_path):
    if os.environ.get("BENCHFLOW_R13_DOCKER") != "1":
        pytest.skip("requires explicit real-Docker builder stage")
    if not shutil.which("docker"):
        pytest.fail("real-Docker stage requested without Docker")
    shared = tmp_path / "shared"
    shared.mkdir()
    project = "taskmd-r13-" + uuid.uuid4().hex[:12]
    config = tmp_path / "compose.yaml"
    config.write_text(
        "services:\n  main:\n    image: py8-gate:e0a3cd4aff60\n"
        f"    user: '{os.getuid()}:{os.getgid()}'\n"
        "    network_mode: none\n    command: [sleep, infinity]\n"
        f"    volumes:\n      - '{shared}:/work-shared'\n"
    )
    command = ["docker", "compose", "-p", project, "-f", str(config)]

    def run(*arguments):
        return subprocess.run(
            [*command, *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

    try:
        run("up", "-d")
        container = run("ps", "-q", "main").stdout.strip()
        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox._main_container = container
        sandbox._docker_compose_global_args = lambda: []
        sandbox._docker_compose_env = lambda: dict(os.environ)
        sandbox._compose_transport_env = lambda: {}
        yield shared, sandbox, run
    finally:
        run("down", "--volumes", "--remove-orphans")


@pytest.mark.asyncio
async def test_real_docker_hardlinks_and_plain_file(compose_output, tmp_path):
    """Guards R12-4 with Moby's actual TypeLink archive, not a tar fake."""
    _, sandbox, run = compose_output
    run(
        "exec",
        "-T",
        "main",
        "sh",
        "-c",
        "mkdir /tmp/output; printf answer > /tmp/output/a; ln /tmp/output/a /tmp/output/b",
    )
    destination = tmp_path / "download"
    await sandbox.download_dir("/tmp/output", destination)
    assert (
        (destination / "a").read_bytes()
        == (destination / "b").read_bytes()
        == b"answer"
    )
    assert (destination / "a").stat().st_ino != (destination / "b").stat().st_ino
    await sandbox.download_file("/tmp/output/a", tmp_path / "ordinary")
    assert (tmp_path / "ordinary").read_bytes() == b"answer"


@pytest.mark.asyncio
async def test_real_compose_agent_link_is_refused(compose_output, tmp_path):
    """Guards R12-1 with a same-UID agent planting through a real Compose bind."""
    shared, sandbox, run = compose_output
    # The caller chooses its root before letting the agent run.
    _pinned = rw.destination(shared / "unused", root=shared)
    private = tmp_path / "private"
    private.mkdir()
    marker = private / "marker.json"
    marker.write_text("original")
    run("exec", "-T", "main", "ln", "-s", str(private), "/work-shared/results")
    run("exec", "-T", "main", "sh", "-c", "printf changed > /tmp/answer")
    with pytest.raises(rw.ReachableWriteError):
        await sandbox.download_file("/tmp/answer", shared / "results/marker.json")
    assert marker.read_text() == "original"


def test_real_compose_log_staging_cannot_redirect(compose_output, tmp_path):
    """Guards R12-2 with the jobs parent shared into the same-UID container."""
    shared, _, run = compose_output
    (shared / "logs").mkdir()
    marker = tmp_path / "marker.json"
    marker.write_text("original")
    target = rw.ReachablePath.directory(shared / "logs") / "agent.log"
    with target.log() as log:
        # Legacy staging appeared under this exact Compose bind. R13 keeps
        # the log anonymous, so there is no named log/published pair to plant.
        run(
            "exec",
            "-T",
            "main",
            "sh",
            "-c",
            'for d in /work-shared/.benchflow-private-*; do [ ! -d "$d" ] || ln -s "$1" "$d/published"; done; :',
            "plant",
            str(marker),
        )
        log.write("safe\n")
        log.flush()
    assert marker.read_text() == "original"
    assert (shared / "logs/agent.log").read_text() == "safe\n"
