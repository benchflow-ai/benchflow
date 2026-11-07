"""``bench doctor`` names the sandboxes that ended runs left on Docker.

A run stopped hard (killed, a crash, the laptop asleep) leaves its compose
containers. The next run removes only the stopped ones (the leftover sweep,
``sandbox/_docker_sweep.py``), so a container left running keeps its CPU,
memory and ports until someone notices. Doctor now counts the containers
whose ``benchflow.process`` label names a process that is gone, and says how
to remove them (Rust 0.8.2's doctor reports its own the same way).
"""

from __future__ import annotations

import socket

from benchflow.doctor import CommandResult, run_doctor
from benchflow.sandbox._docker_sweep import current_process
from tests.test_doctor import by_id, make_probes

_LISTING = (
    "docker",
    "ps",
    "-a",
    "--no-trunc",
    "--filter",
    "label=benchflow.owned=process",
    "--format",
    '{{.ID}}\t{{.Label "benchflow.process"}}\t'
    '{{.Label "com.docker.compose.project"}}\t{{.State}}',
)
# A pid no process holds (above Linux's pid_max default ceiling).
_GONE_PID = 4_194_304 + 4242


def _gone_token() -> str:
    # Another boot on this host, or (off Linux) a pid nobody holds: gone.
    return f"{socket.gethostname()}:{_GONE_PID}:other-boot:1"


def _live_token() -> str:
    return current_process().token()


def _row(cid: str, token: str, project: str, state: str) -> str:
    return f"{cid}\t{token}\t{project}\t{state}"


def test_containers_of_ended_runs_are_named(tmp_path):
    listing = "\n".join(
        [
            _row("a" * 64, _gone_token(), "hello__1", "running"),
            _row("b" * 64, _gone_token(), "hello__1", "exited"),
            _row("c" * 64, _gone_token(), "other__2", "running"),
            # A live run's container, another host's, and an unlabelled one.
            _row("d" * 64, _live_token(), "live__3", "running"),
            _row("e" * 64, "elsewhere:12:boot:1", "remote__4", "running"),
            _row("f" * 64, "", "old__5", "running"),
        ]
    )
    probes = make_probes(tmp_path, commands={_LISTING: CommandResult(0, listing, "")})

    report = run_doctor(sandbox="docker", offline=True, probes=probes)

    row = by_id(report)["docker-leftovers"]
    assert row.status == "warn"
    assert row.summary == "3 containers left by 2 ended runs (2 still running)"
    assert "docker rm -f " + " ".join(c * 12 for c in "abc") in row.fix
    assert row.details["containers"] == ["a" * 12, "b" * 12, "c" * 12]
    assert "d" * 12 not in row.fix and "e" * 12 not in row.fix


def test_nothing_left_means_no_row(tmp_path):
    listing = _row("d" * 64, _live_token(), "live__3", "running")
    probes = make_probes(tmp_path, commands={_LISTING: CommandResult(0, listing, "")})

    report = run_doctor(sandbox="docker", offline=True, probes=probes)

    assert "docker-leftovers" not in by_id(report)


def test_a_failed_listing_adds_no_row(tmp_path):
    probes = make_probes(tmp_path)

    report = run_doctor(sandbox="docker", offline=True, probes=probes)

    assert "docker-leftovers" not in by_id(report)
