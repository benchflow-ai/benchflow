"""Docker without Compose, through a real ``docker`` executable (PR #146).

Guards the fix in PR #146: on a fresh Ubuntu 24.04 VM with Ubuntu's docker.io and no
docker-compose-v2, `bench doctor` passed Docker, and the run it recommended
failed at its first task, because every Docker sandbox runs through
`docker compose`. A fake `docker` first on PATH answers the daemon, context and
buildx probes as that VM did and has no `compose` command; doctor's real
probes (subprocesses, PATH lookup) run against it.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow import doctor as doctor_mod
from benchflow.cli import main as cli_main
from benchflow.cli.main import app
from benchflow.doctor import Check, DoctorProbes, check_docker

FAKE_DOCKER = """#!{python}
import json, sys

args = sys.argv[1:]
if args[:1] == ["info"]:
    print(json.dumps({{
        "ServerVersion": "28.2.2",
        "NCPU": 4,
        "MemTotal": 16 * 1024**3,
        "OperatingSystem": "Ubuntu 24.04.3 LTS",
        "DockerRootDir": "/var/lib/docker",
    }}))
elif args[:2] == ["context", "show"]:
    print("default")
elif args[:2] == ["buildx", "version"]:
    print("github.com/docker/buildx 0.29.1 0.29.1-0ubuntu1~24.04.1")
elif args[:1] == ["compose"]:
    # What docker.io's CLI says without the docker-compose-v2 plugin.
    print("docker: unknown command: docker compose\\n\\n"
          "Run 'docker --help' for more information", file=sys.stderr)
    sys.exit(1)
# Anything else (the leftover-container listings): nothing.
"""


@pytest.fixture
def docker_without_compose(tmp_path, monkeypatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER.format(python=sys.executable))
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "BENCHFLOW_SKIP_PREFLIGHT"):
        monkeypatch.delenv(name, raising=False)
    return docker


def test_doctor_fails_docker_without_compose(docker_without_compose):
    probes = dataclasses.replace(DoctorProbes.from_host(), system="Linux")
    checks = {c.id: c for c in check_docker(probes, required=True)}
    assert checks["docker"].status == "pass"
    compose = checks["docker-compose"]
    assert compose.status == "fail"
    assert "docker: unknown command: docker compose" in compose.summary
    assert "docker-compose-v2" in compose.fix
    assert "docker-compose-plugin" in compose.fix
    assert "docker-buildx" not in checks


def test_the_run_stops_before_a_job_exists(
    docker_without_compose, tmp_path, monkeypatch
):
    """`bench eval run --sandbox docker` now refuses up front, as doctor
    does, instead of failing at the first task's `docker compose`."""
    calls: list = []
    monkeypatch.setattr(
        cli_main, "run_batch_eval", lambda *args, **kwargs: calls.append(args)
    )
    monkeypatch.setattr(
        doctor_mod,
        "check_daytona",
        lambda probes, *, required, offline, for_run=False: Check(
            "daytona", "sandbox", "daytona", "ok", "stubbed"
        ),
    )
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(task),
            "--agent",
            "oracle",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--sandbox",
            "docker",
        ],
    )
    assert result.exit_code == 1, result.output
    assert calls == []
    assert not (tmp_path / "jobs").exists()
    assert "no job was created" in result.stderr
    assert "Docker Compose not found" in result.stderr
