"""A root oracle's leftovers are stopped when solve.sh ends (RCI on
011d84c9f), on real Docker: every process of main's container started from
solve.sh's start tick on (a background loop, a setsid daemon, a double fork,
another uid's), and nothing else: not main's earlier processes, not a
sidecar's process in main's pid namespace (another cgroup)."""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid

import pytest

from benchflow.rollout._setup import _STOP_ORACLE_PROCS
from benchflow.sandbox._exec_receipt import (
    new_receipt_nonce,
    read_receipt,
    receipt_command,
)
from benchflow.sandbox.protocol import ExecResult

IMAGE = "ubuntu:24.04"


def _docker() -> bool:
    return shutil.which("docker") is not None and (
        subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    )


def _sh(container: str, script: str, user: str = "0:0") -> str:
    done = subprocess.run(
        ["docker", "exec", "-u", user, container, "sh", "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return done.stdout + done.stderr


def _cmdlines(container: str) -> list[str]:
    out = _sh(
        container,
        'for d in /proc/[0-9]*; do [ -r "$d/cmdline" ] && { tr "\\000" " " < "$d/cmdline"; echo; }; done',
    )
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


@pytest.fixture
def pair():
    if not _docker():
        pytest.skip("Docker required")
    tag = uuid.uuid4().hex[:8]
    main, side = f"bf-stop-main-{tag}", f"bf-stop-side-{tag}"
    try:
        subprocess.run(["docker", "pull", "-q", IMAGE], capture_output=True, check=True)
        subprocess.run(
            ["docker", "run", "-d", "--name", main, IMAGE, "sleep", "infinity"],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                side,
                "--pid",
                f"container:{main}",
                IMAGE,
                "sleep",
                "infinity",
            ],
            capture_output=True,
            check=True,
        )
        yield main, side
    finally:
        subprocess.run(["docker", "rm", "-f", main, side], capture_output=True)


def test_the_stop_takes_solve_shs_processes_and_nothing_else(pair):
    main, side = pair
    _sh(main, "nohup sleep 1001 >/dev/null 2>&1 &")  # main's, before solve.sh
    up = _sh(main, 'read -r up rest < /proc/uptime; echo "$up"').strip()
    secs, frac = up.split(".")
    tick = int(secs) * 100 + int(frac[:2])
    # solve.sh's leftovers, as root and another uid.
    _sh(main, "nohup sh -c 'while :; do sleep 0.2; done' >/dev/null 2>&1 &")
    _sh(main, "setsid sh -c 'exec sleep 1002' >/dev/null 2>&1 < /dev/null &")
    _sh(main, "sh -c 'nohup sleep 1003 >/dev/null 2>&1 &' &")
    _sh(main, "nohup sleep 1004 >/dev/null 2>&1 &", user="65534:65534")
    # A sidecar's process in main's pid namespace, after the tick: never.
    _sh(side, "nohup sleep 1005 >/dev/null 2>&1 &")
    before = _cmdlines(main)
    assert any("sleep 1002" in c for c in before) and any(
        "sleep 1005" in c for c in before
    )

    # As the oracle's stop runs it: through the receipt's subshell and encoder
    # (which it must spare: they are this exec's own tree).
    nonce = new_receipt_nonce()
    wrapped = receipt_command(_STOP_ORACLE_PROCS.replace("__T0__", str(tick)), nonce)
    done = subprocess.run(
        ["docker", "exec", main, "sh", "-c", wrapped],
        capture_output=True,
        text=True,
        timeout=120,
    )
    read = read_receipt(
        ExecResult(stdout=done.stdout, stderr=done.stderr, return_code=done.returncode),
        nonce,
    )
    out = read.stdout or ""
    found = re.search(r"BF oracle-stop killed=(\d+) left=([0-9 ]*)", out)
    assert found, (out, done.stdout[-500:], done.stderr[-500:])
    assert int(found[1]) >= 5 and found[2].strip() == "", out

    after = " | ".join(_cmdlines(main))
    for gone in ("sleep 0.2", "sleep 1002", "sleep 1003", "sleep 1004"):
        assert gone not in after, after
    for kept in ("sleep infinity", "sleep 1001", "sleep 1005"):
        assert kept in after, after
