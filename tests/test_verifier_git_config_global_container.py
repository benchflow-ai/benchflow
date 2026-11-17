"""Root's `git config --global` in test.sh leaves /dev/null a device (EN-708).

The verifier's git had ``GIT_CONFIG_GLOBAL=/dev/null``. SWT-bench's run scripts
do ``git config --global --add safe.directory /testbed`` as root: git writes
``/dev/null.lock`` and renames it over the name, so /dev/null became a regular
file. Every later ``>/dev/null`` wrote into it, and git then stopped with "bad
config line 1 in file /dev/null"; the gold run's test patch never applied and a
correct solution scored 0 (astropy-14598, matplotlib-23299). The verifier now
gets an empty, writable gitconfig of its own in its fresh tool state.
In a real container, as root; skipped without Docker.
"""

from __future__ import annotations

import shlex
import subprocess
import uuid

import pytest

from benchflow.sandbox import _verifier_tool_state as tool_state
from benchflow.sandbox.lockdown import VERIFIER_ENV
from tests.exec_diet_fakes import docker_available

FRESH = "/var/lib/bf-verifier-tool-state"
SCRIPT = (
    "rm -rf /tmp/repo && git init -q /tmp/repo && "
    "env {env} git config --global --add safe.directory /testbed; "
    "echo noise > /dev/null; "
    "env {env} git -C /tmp/repo status --short; echo git=$?; "
    "if [ -c /dev/null ]; then echo devnull=char; else echo devnull=file; fi"
)


@pytest.fixture(scope="module")
def image(tmp_path_factory):
    if not docker_available():
        pytest.skip("needs a Docker daemon")
    context = tmp_path_factory.mktemp("git-image")
    (context / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "RUN apt-get update -qq && apt-get install -y -qq git >/dev/null && "
        "rm -rf /var/lib/apt/lists/*\n"
    )
    tag = "benchflow-test-git:" + uuid.uuid4().hex[:8]
    built = subprocess.run(
        ["docker", "build", "-q", "-t", tag, str(context)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if built.returncode != 0:
        pytest.skip(f"could not build an image with git: {built.stderr[-300:]}")
    yield tag
    subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


def _run_as_root(image: str, command: str) -> subprocess.CompletedProcess:
    """*command* as root in a new container of *image* (each test its own:
    the control breaks /dev/null)."""
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-u",
            "0",
            image,
            "sh",
            "-c",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )


def _git_env(global_config: str) -> str:
    """The verifier's git variables as shell words, with *global_config*."""
    env = {k: v for k, v in VERIFIER_ENV.items() if k.startswith("GIT_")}
    env["GIT_CONFIG_GLOBAL"] = global_config
    return " ".join(shlex.quote(f"{k}={v}") for k, v in env.items())


def test_root_git_config_global_keeps_dev_null_a_device(image):
    overrides = tool_state.overrides({"HOME": "/root"}, ("/root",), FRESH)
    config = overrides["GIT_CONFIG_GLOBAL"]
    assert config == f"{FRESH}/{tool_state.GIT_CONFIG}"

    ran = _run_as_root(
        image,
        f"mkdir -p {FRESH} && : > {config} && "
        + SCRIPT.format(env=_git_env(config))
        + f" && cat {config}",
    )

    assert "devnull=char" in ran.stdout, ran.stdout + ran.stderr
    assert "git=0" in ran.stdout, ran.stdout + ran.stderr
    assert "directory = /testbed" in ran.stdout, ran.stdout


def test_dev_null_as_the_global_config_was_the_fault(image):
    """The control: with GIT_CONFIG_GLOBAL=/dev/null the same script turns
    /dev/null into a file and git stops reading its config."""
    ran = _run_as_root(image, SCRIPT.format(env=_git_env("/dev/null")))

    out = ran.stdout + ran.stderr
    assert "devnull=file" in ran.stdout, out
    # Git's words vary by version: "bad config line 1 in file /dev/null"
    # (SWT-bench's images), "key does not contain a section" (newer).
    assert "bad config line 1" in out or "does not contain a section" in out, out
