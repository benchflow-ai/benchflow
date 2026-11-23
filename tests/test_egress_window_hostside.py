"""The agent user's network rules are in before the agent starts, from host-side records only.

Before 0.10, a no-web agent's uid rules went in during ACP connect, after its process had started: on the 0.9.2
wheel a connection the agent opened at start-up (+0.08 s) reached the host and survived the rules. This is the
release gate's window check (it was a hand-run VM probe through 0.10.0rc4), as a test:

- the stub agent of tests/test_cloud_metadata_failclosed_docker.py, whose launch script first connects to a
  listener on the Docker bridge gateway, before it answers any ACP request;
- dockerd's own event stream, read live, for when the rules exec and the agent's launch exec ran;
- the host listener, for whether that connect was accepted;
- where ``sudo -n`` works, a root poller of the rules in the sandbox's network namespace.

The readout is tests/window/readout.py (its planted faults are in tests/test_window_readout.py). A positive
control, a plain container's connect to the same listener, must be accepted before and after the runs, or "no
accept" proves nothing. ``BENCHFLOW_WINDOW_RUNS`` sets the runs per user (3 on a pull request, 5 at night).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import pytest

from tests import test_cloud_metadata_failclosed_docker as fc
from tests.window import readout
from tests.window.listener import Listener

pytestmark = [pytest.mark.deterministic, pytest.mark.window]

RUNS = int(os.environ.get("BENCHFLOW_WINDOW_RUNS", "3"))
HERE = Path(__file__).resolve().parent

EARLY_SH = r"""#!/bin/bash
h=$(awk '$2=="00000000"{print $3; exit}' /proc/net/route)
gw=$(printf '%d.%d.%d.%d' 0x${h:6:2} 0x${h:4:2} 0x${h:2:2} 0x${h:0:2})
echo "gw $gw start $(date +%s%N) ids $(id -u) $(id -g)"
if timeout 3 bash -c "exec 3<>/dev/tcp/$gw/__PORT__"; then echo "early-connect ok $(date +%s%N)"
else echo "early-connect failed $(date +%s%N)"; fi
"""

EARLY_ACP_SH = fc.NOWEB_ACP_SH.replace(
    "echo launched > /logs/agent/launched\n",
    "echo launched > /logs/agent/launched\n"
    "/bin/bash /opt/bfprobe/early.sh > /logs/agent/early.txt 2>&1 < /dev/null &\n",
)
assert EARLY_ACP_SH != fc.NOWEB_ACP_SH

DOCKERFILE = fc.DOCKERFILE.replace(
    "COPY acp.sh probe.sh /opt/bfprobe/\nRUN chmod 755 /opt/bfprobe /opt/bfprobe/acp.sh /opt/bfprobe/probe.sh\n",
    "COPY acp.sh probe.sh early.sh /opt/bfprobe/\n"
    "RUN chmod 755 /opt/bfprobe /opt/bfprobe/acp.sh /opt/bfprobe/probe.sh /opt/bfprobe/early.sh\n",
)
assert DOCKERFILE != fc.DOCKERFILE


def _docker() -> bool:
    return shutil.which("docker") is not None and fc._docker_ready()


@pytest.fixture(scope="module")
def listener(tmp_path_factory):
    server = Listener(tmp_path_factory.mktemp("window") / "listen.log")
    try:
        yield server
    finally:
        server.close()


@pytest.fixture(scope="module")
def window_image(tmp_path_factory, listener):
    if not _docker():
        pytest.skip("no Docker daemon answers")
    context = tmp_path_factory.mktemp("bfprobe-window")
    (context / "Dockerfile").write_text(DOCKERFILE)
    (context / "acp.sh").write_text(EARLY_ACP_SH)
    (context / "probe.sh").write_text(fc.NOWEB_PROBE_SH)
    (context / "early.sh").write_text(EARLY_SH.replace("__PORT__", str(listener.port)))
    tag = f"bf-test-mdprobe-window:{uuid.uuid4().hex[:12]}"
    built = subprocess.run(
        ["docker", "build", "--quiet", "--label", f"{fc.LABEL}=image", "-t", tag, "."],
        cwd=context,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if built.returncode != 0:
        pytest.fail(
            f"cannot build the window probe image: {built.stderr.strip()[-300:]}",
            pytrace=False,
        )
    try:
        yield tag
    finally:
        subprocess.run(["docker", "rmi", "--force", tag], capture_output=True)


@pytest.fixture(scope="module")
def kernel_log(tmp_path_factory):
    """The root poller, where sudo works without a password (GitHub's runners, the build VMs)."""
    if subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode != 0:
        yield None
        return
    log = tmp_path_factory.mktemp("netwatch") / "netwatch.log"
    log.touch()
    proc = subprocess.Popen(
        [
            "sudo",
            "-n",
            "python3",
            "-I",
            str(HERE / "window" / "netwatch.py"),
            str(log),
            "md-probe",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield log
    finally:
        subprocess.run(["sudo", "-n", "kill", str(proc.pid)], capture_output=True)
        proc.wait(timeout=30)


class _Events:
    """dockerd's events, read live for the length of one run."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.proc = subprocess.Popen(
            [
                "docker",
                "events",
                "--format",
                "{{json .}}",
                "--filter",
                "type=container",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        time.sleep(0.5)

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.append(line)

    def close(self) -> list[str]:
        time.sleep(0.5)
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.thread.join(timeout=10)
        return self.lines


def _control(image: str, port: int) -> None:
    subprocess.run(
        ["docker", "run", "--rm", "--label", f"{fc.LABEL}=control", "--entrypoint", "bash", image, "-c",
         f"h=$(awk '$2==\"00000000\"{{print $3; exit}}' /proc/net/route); "
         f"gw=$(printf '%d.%d.%d.%d' 0x${{h:6:2}} 0x${{h:4:2}} 0x${{h:2:2}} 0x${{h:0:2}}); "
         f"exec 3<>/dev/tcp/$gw/{port}"],
        capture_output=True, timeout=120,
    )  # fmt: skip


def test_the_listener_hears_a_plain_container(window_image, listener):
    start = time.time_ns()
    _control(window_image, listener.port)
    end = time.time_ns()
    assert readout.control_seen(readout.parse_listener(listener.lines()), start, end), (
        "the positive control's connect was not accepted: the listener is not reachable from a "
        "container, so no window run can be read"
    )


@pytest.mark.parametrize("run", range(RUNS))
@pytest.mark.parametrize("user", ["root", "agent"])
def test_the_rules_are_in_before_the_agent_starts(
    tmp_path, window_image, listener, kernel_log, user, run
):
    events = _Events()
    try:
        done, result, _record, _probe, _mock = fc._run(
            tmp_path, window_image, "--sandbox-user", user, no_web=True
        )
    finally:
        lines = events.close()
    kernel = (
        readout.parse_kernel(kernel_log.read_text().splitlines())
        if kernel_log
        else None
    )
    verdict = readout.analyze(
        readout.parse_events(lines), readout.parse_listener(listener.lines()), kernel
    )
    early = list((tmp_path / "jobs").rglob("agent/early.txt"))
    said = early[0].read_text() if early else "(no early.txt)"
    assert verdict.passed, (verdict, said, result.get("error"), done.stderr[-2000:])
    assert "early-connect failed" in said, said
    print(
        f"window {user} run {run}: gap {verdict.gap_s:.3f}s, kernel {verdict.kernel_s}, rules execs {verdict.rules_execs}"
    )


def test_the_listener_still_hears_a_plain_container_after_the_runs(
    window_image, listener
):
    test_the_listener_hears_a_plain_container(window_image, listener)
