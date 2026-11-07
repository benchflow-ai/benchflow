"""Guards SCRIPT-AGENT-DOCKER-F1 round 2 at 5572a868a with real Docker.

- A ``script:`` agent lost the image's values of every name S1's bootstrap
  clears (LD_LIBRARY_PATH, PYTHONPATH, ...), which ACP agents and the oracle
  get back: a tool that needs the image's own LD_LIBRARY_PATH failed in the
  script (review A M-1, pre-existing since 6ad32f11d).
- A sidecar that is the verifier's service, with ``ENV
  LD_TRACE_LOADED_OBJECTS=1``, left the trial unscored: root's system shell
  listed its libraries instead of running (review B M3).
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_script_agent_docker_f1 import ORDINARY, docker_or_skip, run_script_agent

GREET_LIB_C = 'const char *greeting(void) { return "hello"; }\n'
GREET_C = r"""
#include <stdio.h>
const char *greeting(void);
int main(void) { fputs(greeting(), stdout); return 0; }
"""
HOLD_C = "#include <unistd.h>\nint main(void) { for (;;) pause(); }\n"


def compiler_or_skip() -> str:
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("a C compiler is needed to build the test image's tools")
    return compiler


def build_greet(context: Path) -> None:
    """``greet`` prints hello through ``libgreet.so``, linked without an
    rpath, so it runs only with the image's LD_LIBRARY_PATH."""
    compiler = compiler_or_skip()
    (context / "greet-lib.c").write_text(GREET_LIB_C)
    (context / "greet.c").write_text(GREET_C)
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            "-o",
            context / "libgreet.so",
            context / "greet-lib.c",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            compiler,
            "-o",
            context / "greet",
            context / "greet.c",
            f"-L{context}",
            "-lgreet",
        ],
        check=True,
        capture_output=True,
    )


@pytest.mark.integration
@pytest.mark.parametrize("direct", ["1", "0"])
def test_script_agent_gets_the_images_library_path(tmp_path, monkeypatch, direct):
    """The image's LD_LIBRARY_PATH and PYTHONPATH reach the script, as they
    reach ACP agents; its LD_BIND_NOW (a loader switch) does not. test.sh
    runs without the image's LD_LIBRARY_PATH (V3), and result.json says so
    by name, never by value."""
    docker_or_skip()
    monkeypatch.setenv("BENCHFLOW_DIRECT_DOCKER", direct)
    task = tmp_path / "task"
    for directory in ("tests", "environment"):
        (task / directory).mkdir(parents=True)
    build_greet(task / "environment")
    (task / "environment/Dockerfile").write_text(
        f"FROM {ORDINARY}\n"
        "COPY greet /opt/mytool/bin/greet\n"
        "COPY libgreet.so /opt/mytool/lib/libgreet.so\n"
        "ENV LD_LIBRARY_PATH=/opt/mytool/lib PYTHONPATH=/opt/mytool/py LD_BIND_NOW=1\n"
        "WORKDIR /app\n"
    )
    (task / "environment/docker-compose.yaml").write_text(
        "services:\n  main:\n    build: .\n    network_mode: none\n"
        "    command: [sleep, infinity]\n    working_dir: /app\n"
    )
    (task / "task.md").write_text(
        "---\nschema_version: '1.3'\nsandbox: {workdir: /app}\n---\n\n"
        "## prompt\n\nWrite hello to /app/hello.txt with greet.\n"
    )
    (task / "tests/test.sh").write_text(
        "#!/bin/bash\n"
        '[ "$(/bin/cat /app/hello.txt)" = hello ] || exit 0\n'
        "/bin/grep -qx PYTHONPATH=/opt/mytool/py /app/agent-env.txt || exit 0\n"
        "! /bin/grep -q ^LD_BIND_NOW= /app/agent-env.txt || exit 0\n"
        "echo 1 > /logs/verifier/reward.txt\n"
    )
    result = run_script_agent(
        tmp_path,
        task,
        "/usr/bin/env > /app/agent-env.txt\n/opt/mytool/bin/greet > /app/hello.txt\n",
    )
    assert result.error is None, result.error
    assert result.reward == 1.0
    (record,) = (tmp_path / "jobs").rglob("result.json")
    warnings = json.loads(record.read_text()).get("warnings") or []
    assert any("LD_LIBRARY_PATH" in w for w in warnings), warnings
    assert not any("/opt/mytool" in w for w in warnings), warnings


@pytest.mark.integration
def test_sidecar_verifier_with_a_trace_flag_is_scored(tmp_path, monkeypatch):
    """Review B's sidecar: a static PID 1 (healthy under the flag) and ``ENV
    LD_TRACE_LOADED_OBJECTS=1``, the verifier's service. Root's steps there
    run their commands, so the trial is scored."""
    docker_or_skip()
    compiler = compiler_or_skip()
    monkeypatch.setenv("BENCHFLOW_DIRECT_DOCKER", "0")
    task = tmp_path / "task"
    sidecar = task / "environment" / "sidecar"
    for directory in (task / "tests", sidecar):
        directory.mkdir(parents=True)
    (sidecar / "hold.c").write_text(HOLD_C)
    built = subprocess.run(
        [compiler, "-static", "-o", sidecar / "hold", sidecar / "hold.c"],
        capture_output=True,
        check=False,
    )
    if built.returncode:
        pytest.skip("a static C library is needed for the sidecar's PID 1")
    (sidecar / "Dockerfile").write_text(
        f"FROM {ORDINARY}\nCOPY hold /opt/hold\n"
        'ENV LD_TRACE_LOADED_OBJECTS=1\nCMD ["/opt/hold"]\n'
    )
    (task / "environment/Dockerfile").write_text(f"FROM {ORDINARY}\nWORKDIR /app\n")
    (task / "environment/docker-compose.yaml").write_text(
        "services:\n"
        "  main:\n    build: .\n    network_mode: none\n"
        "    command: [sleep, infinity]\n    working_dir: /app\n"
        "  sidecar:\n    build: ./sidecar\n    network_mode: none\n"
    )
    (task / "instruction.md").write_text("Write hello.\n")
    (task / "task.toml").write_text(
        'version = "1.0"\n[agent]\ntimeout_sec = 60\n'
        '[verifier]\ntimeout_sec = 60\nservice = "sidecar"\n'
    )
    (task / "tests/test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    result = run_script_agent(tmp_path, task, "printf hello > /app/hello.txt\n")
    assert result.error is None, result.error
    assert result.reward == 1.0
