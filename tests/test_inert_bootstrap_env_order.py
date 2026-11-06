"""The inert bootstrap's env rows have one order.

Compose builds a service's Env from a Go map, so a prototype's Config.Env
order changes between identical runs, and the bootstrap copied that order
into main's argv and its receipt label. Both now carry the rows sorted by
key, stably: rows that share a key keep their order, so ``env --`` (which
assigns left to right) still gives the same environment.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import uuid

import pytest

from benchflow.sandbox import _inert_bootstrap as bootstrap

CONFIG = {
    "Entrypoint": None,
    "Cmd": ["sh", "-c", "sleep infinity"],
    "Env": ["PERF_E=a$b c", "EMPTY=", "PATH=/usr/bin:/bin", "B=2"],
    "Healthcheck": None,
    "Shell": None,
}


def _rows(argv: list[str]) -> list[str]:
    return argv[4 : -len(CONFIG["Cmd"])]


def test_the_env_rows_are_sorted_by_key_in_argv_and_receipt():
    rows = ["B=2", "EMPTY=", "PATH=/usr/bin:/bin", "PERF_E=a$b c"]
    assert _rows(bootstrap.entrypoint(CONFIG)) == rows
    receipt = json.loads(base64.b64decode(bootstrap.config_receipt(CONFIG)))
    assert receipt["Env"] == rows
    # Any order Docker was given gives the same argv and receipt.
    shuffled = {**CONFIG, "Env": list(reversed(CONFIG["Env"]))}
    assert bootstrap.entrypoint(shuffled) == bootstrap.entrypoint(CONFIG)
    assert bootstrap.config_receipt(shuffled) == bootstrap.config_receipt(CONFIG)


def test_rows_that_share_a_key_keep_their_order():
    """``env -- A=1 A=2`` sets A=2: the later row must stay later."""
    config = {**CONFIG, "Env": ["Z=0", "A=1", "M=x", "A=2"]}
    assert _rows(bootstrap.entrypoint(config)) == ["A=1", "A=2", "M=x", "Z=0"]
    flipped = {**CONFIG, "Env": ["A=2", "Z=0", "A=1"]}
    assert _rows(bootstrap.entrypoint(flipped)) == ["A=2", "A=1", "Z=0"]


def test_a_receipt_round_trips_to_the_same_argv():
    """A checkpoint reads the receipt back (original_config) and wraps it
    again: the order is a fixed point."""
    labels = {bootstrap.LABEL: bootstrap.config_receipt(CONFIG)}
    restored = bootstrap.original_config({"Labels": labels, "Env": []})
    assert bootstrap.entrypoint(restored) == bootstrap.entrypoint(CONFIG)
    assert bootstrap.config_receipt(restored) == bootstrap.config_receipt(CONFIG)


# --- On a real daemon: six Compose-path starts, one argv and one receipt -------

IMAGE = "ubuntu:24.04"


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "image", "inspect", IMAGE], capture_output=True, timeout=20
        )
    except Exception:
        return False
    return probe.returncode == 0


@pytest.mark.skipif(
    not _docker_ready(), reason=f"needs a Docker daemon holding {IMAGE}"
)
@pytest.mark.asyncio
async def test_repeated_compose_starts_give_one_argv_and_receipt(tmp_path, monkeypatch):
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task.config import SandboxConfig
    from benchflow.task.paths import RolloutPaths

    # The Compose path itself (the direct path is off in tests by default).
    monkeypatch.setattr(DockerSandbox, "direct_docker", False, raising=False)
    environment = {"D_FOUR": "4", "A_ONE": "1", "C_THREE": "x$y", "B_TWO": ""}
    seen = set()
    for _ in range(6):
        name = f"perf-e-env-{uuid.uuid4().hex[:8]}"
        env_dir = tmp_path / name / "environment"
        env_dir.mkdir(parents=True)
        (env_dir / "Dockerfile").write_text(f"FROM {IMAGE}\n")
        compose = {"services": {"main": {"image": IMAGE, "environment": environment}}}
        (env_dir / "docker-compose.yaml").write_text(
            json.dumps(compose).replace("$", "$$")
        )
        paths = RolloutPaths(rollout_dir=tmp_path / name / "run")
        paths.mkdir()
        sandbox = DockerSandbox(
            environment_dir=env_dir,
            environment_name=name,
            session_id=name,
            rollout_paths=paths,
            task_env_config=SandboxConfig(cpus=1, memory_mb=512, docker_image=IMAGE),
        )
        await sandbox.start(force_build=False)
        try:
            main = sandbox._main_container
            assert main is not None
            config = json.loads(
                subprocess.run(
                    ["docker", "inspect", main],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            )[0]["Config"]
            seen.add((tuple(config["Entrypoint"]), config["Labels"][bootstrap.LABEL]))
        finally:
            await sandbox.stop(delete=True)
    assert len(seen) == 1
    ((argv, _),) = seen
    rows = [row for row in argv if row.partition("=")[0] in environment]
    assert rows == ["A_ONE=1", "B_TWO=", "C_THREE=x$y", "D_FOUR=4"]
