"""Guards SCRIPT-AGENT-DOCKER-F1 at fb00ee747 with real Docker.

Every ``script:`` agent run on Docker ended as an infrastructure error ("the
runner exited 0 before the script started"), ordinary images included: S1's
inert bootstrap leaves loader names bare in Config.Env, ``protected_exec_argv``
revived each as ``-e NAME=``, and glibc, which acts on LD_SHOW_AUXV merely
being present, printed its aux vector over the runner's marker read.
"""

import base64
import shutil
import subprocess
from pathlib import Path

import pytest

import benchflow as bf
from tests.test_script_agent_docker import started_box

ORDINARY = "python:3.12-slim"

# Appends the UID of every process that loads it to /tmp/f1-hook-ran.
HOOK_C = r"""
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
__attribute__((constructor)) static void ran(void) {
    char line[32];
    int n = snprintf(line, sizeof line, "%u\n", (unsigned) getuid());
    int fd = open("/tmp/f1-hook-ran", O_WRONLY | O_APPEND | O_CREAT, 0666);
    if (fd >= 0) {
        if (write(fd, line, n) < 0) {}
        close(fd);
    }
}
"""


def docker_or_skip():
    if not shutil.which("docker"):
        pytest.skip("Docker is unavailable")
    if subprocess.run(["docker", "info"], capture_output=True, check=False).returncode:
        pytest.skip("Docker daemon is unavailable")
    if subprocess.run(
        ["docker", "image", "inspect", ORDINARY], capture_output=True, check=False
    ).returncode:
        pulled = subprocess.run(
            ["docker", "pull", ORDINARY], capture_output=True, check=False
        )
        if pulled.returncode:
            pytest.skip(f"{ORDINARY} is unavailable")


def hook_library(tmp_path: Path) -> bytes:
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("a C compiler is needed to build the planted LD_PRELOAD hook")
    source = tmp_path / "hook.c"
    source.write_text(HOOK_C)
    library = tmp_path / "hook.so"
    subprocess.run(
        [compiler, "-shared", "-fPIC", "-O2", "-o", str(library), str(source)],
        check=True,
        capture_output=True,
    )
    return library.read_bytes()


def planting_script(hook: bytes) -> str:
    """The agent's script: plant the hook in its own workspace, show that it
    works for the agent (its UID is recorded), then write the answer."""
    encoded = base64.b64encode(hook).decode()
    # Unset in the shell first: with the image's LD_SHOW_AUXV, base64 would
    # write the aux vector into the library it decodes.
    return (
        "(unset LD_SHOW_AUXV LD_PRELOAD\n"
        f" printf %s {encoded} | base64 -d > /app/hook.part\n"
        " chmod 755 /app/hook.part && mv /app/hook.part /app/hook.so)\n"
        "LD_PRELOAD=/app/hook.so /bin/true\n"
        "printf hello > /app/hello.txt\n"
    )


def write_task(root: Path, *, compose_env: str = "", dockerfile: str = "") -> Path:
    task = root / "task"
    for directory in ("tests", "environment"):
        (task / directory).mkdir(parents=True, exist_ok=True)
    image = f"docker_image: '{ORDINARY}', " if not dockerfile else ""
    (task / "task.md").write_text(
        "---\nschema_version: '1.3'\n"
        f"sandbox: {{{image}workdir: /app}}\n"
        "---\n\n## prompt\n\nWrite hello to /app/hello.txt.\n"
    )
    if dockerfile:
        (task / "environment/Dockerfile").write_text(dockerfile)
        service = "    build: .\n"
    else:
        service = f"    image: {ORDINARY}\n"
    (task / "environment/docker-compose.yaml").write_text(
        "services:\n  main:\n"
        + service
        + "    network_mode: none\n    command: [sleep, infinity]\n"
        "    working_dir: /app\n" + compose_env
    )
    # Root's verifier: the answer, the agent's own hook record when one is
    # expected, and never a root (UID 0) load of the agent's hook.
    (task / "tests/test.sh").write_text(
        "#!/bin/bash\n"
        '[ "$(/bin/cat /app/hello.txt)" = hello ] || exit 0\n'
        "if [ -e /app/hook.so ]; then\n"
        "  /bin/grep -qvx 0 /tmp/f1-hook-ran || exit 0\n"
        "  ! /bin/grep -qx 0 /tmp/f1-hook-ran || exit 0\n"
        "fi\n"
        "echo 1 > /logs/verifier/reward.txt\n"
    )
    return task


def run_script_agent(root: Path, task: Path, body: str):
    script = root / "attempt.sh"
    script.write_text(body)
    return bf.run_sync(
        bf.RolloutConfig(
            task_path=task,
            agent=f"script:{script}",
            environment="docker",
            sandbox_user="agent",
            jobs_dir=root / "jobs",
            # No metadata probing; the task's network is none.
            allow_cloud_metadata=True,
        )
    )


def assert_no_loader_output(jobs: Path):
    """Nothing BenchFlow recorded shows glibc's aux vector, and nothing but
    the script's own output (agent/ and the trajectory that records it)
    shows a loader message: the script gets the image's LD_PRELOAD back, as
    ACP agents do, so its bash may report the hook it has
    not planted yet; root's steps never load it."""
    for path in jobs.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            assert b"AT_EXECFN" not in data, path
            if path.parent.name not in ("agent", "trajectory"):
                assert b"ld.so" not in data, path


@pytest.mark.integration
@pytest.mark.parametrize("direct", ["1", "0"])
def test_ordinary_image_script_agent_reaches_a_score(tmp_path, monkeypatch, direct):
    """The full public-SDK rollout on python:3.12-slim, direct Docker and
    Compose: the script runs as the agent and the trial is scored."""
    docker_or_skip()
    monkeypatch.setenv("BENCHFLOW_DIRECT_DOCKER", direct)
    task = write_task(tmp_path)
    result = run_script_agent(tmp_path, task, "printf hello > /app/hello.txt\n")
    assert result.error is None, result.error
    assert result.reward == 1.0
    assert_no_loader_output(tmp_path / "jobs")


@pytest.mark.integration
@pytest.mark.parametrize("direct", ["1", "0"])
@pytest.mark.parametrize("where", ["image", "compose"])
def test_planted_loader_variables_neither_run_nor_break(
    tmp_path, monkeypatch, direct, where
):
    """LD_PRELOAD naming a file the agent plants in its workspace, and
    LD_SHOW_AUXV, set in the image's ENV or the task's Compose environment:
    the trial still scores, no root process loads the agent's hook, and no
    aux vector reaches anything BenchFlow records. The image the "image"
    case builds is removed by the sandbox's own teardown, which removes
    only the images its Compose project built."""
    docker_or_skip()
    hook = hook_library(tmp_path)
    monkeypatch.setenv("BENCHFLOW_DIRECT_DOCKER", direct)
    if where == "image":
        task = write_task(
            tmp_path,
            dockerfile=(
                f"FROM {ORDINARY}\n"
                "ENV LD_PRELOAD=/app/hook.so LD_SHOW_AUXV=1\n"
                "WORKDIR /app\n"
            ),
        )
    else:
        task = write_task(
            tmp_path,
            compose_env=(
                "    environment:\n"
                "      LD_PRELOAD: /app/hook.so\n"
                '      LD_SHOW_AUXV: "1"\n'
            ),
        )
    result = run_script_agent(tmp_path, task, planting_script(hook))
    assert result.error is None, result.error
    assert result.reward == 1.0
    assert_no_loader_output(tmp_path / "jobs")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_runner_root_execs_ignore_valued_loader_variables(tmp_path):
    """Without the inert bootstrap Config.Env keeps the values: root's runner
    removes them before even env loads, the agent keeps its environment."""
    from benchflow.rollout._setup import _run_script

    hook = hook_library(tmp_path)
    with started_box(env=("LD_PRELOAD=/app/hook.so", "LD_SHOW_AUXV=1")) as box:
        events, _ = await _run_script(
            box,
            planting_script(hook).encode(),
            "script:attempt.sh",
            30,
            "agent",
            "/app",
            {},
        )
        assert events[0]["return_code"] == 0
        # Read without the image's values (the fixture's own exec keeps them).
        uid, ran = (
            subprocess.run(
                [
                    *("docker", "exec", "-e", "LD_PRELOAD", "-e", "LD_SHOW_AUXV"),
                    *(box.name, "/bin/sh", "-c", command),
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
            for command in ("id -u agent", "cat /tmp/f1-hook-ran")
        )
        # The agent's own processes ran with the image's LD_PRELOAD (the
        # hook works); root's runner steps after the script never loaded it.
        assert uid[0] in ran and "0" not in ran, ran
