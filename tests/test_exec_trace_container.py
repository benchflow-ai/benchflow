"""The exec-trace helper (tests/_exec_trace.py) in real containers: it
records each root ``docker exec`` with its exact user and argv, the root
processes those execs start, and nothing that ran before it opened or ran
as another account; ``disallowed`` keeps only programs outside the allowed
set. With bfx delivered read-only at /.bfx, a trace of bfx's own commands
names only /.bfx/bfx.

Skipped without a Docker daemon.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid

import pytest

from benchflow._bfx import delivery, locate
from tests._exec_trace import LABEL, PREFIX, ExecTrace, disallowed, program

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

IMAGE = "python:3.12-slim"


def _docker_ready() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=10
            ).returncode
            == 0
        )
    except Exception:
        return False


@pytest.fixture
def box():
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    made: list[str] = []

    def make(*extra: str) -> str:
        name = PREFIX + uuid.uuid4().hex[:8]
        key, _, value = LABEL.partition("=")
        run = subprocess.run(
            ["docker", "run", "-d", "--name", name, "--label", f"{key}={value}",
             "--network", "none", *extra, IMAGE, "sleep", "600"],
            capture_output=True,
            text=True,
        )  # fmt: skip
        if run.returncode != 0:
            pytest.skip(f"cannot start {IMAGE}: {run.stderr.strip()}")
        made.append(name)
        return name

    try:
        yield make
    finally:
        for name in made:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _dx(name: str, *argv: str, user: str = "0") -> None:
    done = subprocess.run(
        ["docker", "exec", "-u", user, name, *argv], capture_output=True, timeout=60
    )
    assert done.returncode == 0, done.stderr


def test_a_root_exec_is_recorded_with_its_user_and_argv(box):
    name = box()
    with ExecTrace(name) as trace:
        _dx(name, "/bin/sh", "-c", "exit 0")
    execs = [s for s in trace.root_starts() if s.source == "exec"]
    assert [(s.argv, s.user) for s in execs] == [(("/bin/sh", "-c", "exit 0"), "0")]


def test_an_exec_as_another_account_is_not_a_root_start(box):
    name = box()
    with ExecTrace(name) as trace:
        _dx(name, "/bin/sleep", "0.3", user="1000:1000")
    assert trace.root_starts() == []


def test_what_a_root_exec_starts_is_sampled(box):
    name = box()
    with ExecTrace(name) as trace:
        _dx(name, "/bin/sh", "-c", "/bin/sleep 0.5 & wait")
    sampled = [s.argv for s in trace.root_starts() if s.source == "sampled"]
    assert ("/bin/sleep", "0.5") in sampled, sampled


def test_processes_from_before_the_trace_are_not_reported(box):
    name = box()
    _dx(name, "/bin/sh", "-c", "setsid /bin/sleep 30 </dev/null >/dev/null 2>&1 &")
    with ExecTrace(name) as trace:
        pass
    assert trace.root_starts() == []


def test_disallowed_keeps_only_programs_outside_the_allowed_set(box):
    name = box()
    with ExecTrace(name) as trace:
        _dx(name, "/bin/sleep", "0.3")
        _dx(name, "/usr/bin/env", "-i", "/bin/sleep", "0.3")
    starts = trace.root_starts()
    assert {program(s) for s in disallowed(starts, {"/bin/sleep"})} == {"/usr/bin/env"}
    assert disallowed(starts, {"/bin/sleep", "/usr/bin/env"}) == []


def test_a_trace_of_bfx_commands_names_only_bfx(box, tmp_path, monkeypatch):
    if not locate.available("x86_64"):
        pytest.skip("no pinned bfx in this checkout (make bfx)")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    folder = delivery.host_dir("x86_64")
    name = box(*delivery.run_args(delivery.BIND, source=str(folder)))
    with ExecTrace(name) as trace:
        _dx(name, "/.bfx/bfx", "version")
        _dx(name, "/.bfx/bfx", "stat", "--json", '{"root":"/","paths":["etc"]}')
        _dx(
            name,
            "/.bfx/bfx",
            "exec",
            "--json",
            '{"argv":["/bin/sleep","0.3"],"env":{},"uid":0,"gid":0,"groups":[]}',
        )
    starts = trace.root_starts()
    assert [program(s) for s in starts if s.source == "exec"] == ["/.bfx/bfx"] * 3
    # One-shot exec replaces bfx with the program, in place: both are starts.
    assert {program(s) for s in disallowed(starts, {"/.bfx/bfx"})} == {"/bin/sleep"}
