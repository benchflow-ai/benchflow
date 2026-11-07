"""The leftover sweep lists stopped containers, running containers and
networks at once.

Every job sweeps before its first trial and after its last; each listing is a
docker CLI start plus a daemon call (about 15 ms on the engine bench's VM), and
none depends on another (the running listing is DF-p3-eval-runner-9's). All still finish before the sweep reads
which projects are live (tests/test_docker_sweep.py covers what it removes).
"""

from __future__ import annotations

import subprocess
import threading

from benchflow.sandbox import _docker_sweep as sweep


def test_the_three_listings_run_at_the_same_time(monkeypatch):
    all_started = threading.Barrier(3, timeout=2)
    listed: list[str] = []
    live_read_after: list[list[str]] = []

    def run(argv, **kwargs):
        if argv[1:3] == ["network", "ls"] or argv[1] == "ps":
            # Each listing waits for the others: sequential listings time out.
            try:
                all_started.wait()
            except threading.BrokenBarrierError:
                listed.append(f"{argv[1]} alone")
            else:
                listed.append(argv[1])
        return subprocess.CompletedProcess(argv, 0, "", "")

    def live_projects():
        live_read_after.append(sorted(listed))
        return frozenset()

    monkeypatch.setattr(sweep.subprocess, "run", run)
    monkeypatch.setattr(sweep, "live_projects", live_projects)

    result = sweep.sweep_leftovers()

    assert sorted(listed) == ["network", "ps", "ps"]
    assert live_read_after == [["network", "ps", "ps"]]
    assert (result.containers, result.networks) == ([], [])
