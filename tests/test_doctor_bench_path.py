"""``bench doctor``'s ``bench on PATH`` line, with stand-in programs on PATH.

Both BenchFlow's Python package and the Rust engine install a program called
``bench``. These tests put stand-ins for each in temporary folders,
point PATH at them in different orders, and check that doctor names the one a
shell runs and warns about every one it hides. The stand-ins are real files
read the way doctor reads them: a Python entry script as uv writes it, a Rust
binary (bytes holding the Rust engine's marker), another program called
``bench``, and a runnable script standing in for another BenchFlow install.
"""

from __future__ import annotations

import os
import re
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow import __version__, doctor
from benchflow.doctor import CommandResult, DoctorProbes, run_doctor

RUST_VERSION_LINE = "bench 0.8.2 (5180090b 2026-10-15)\n"
INSTALL_ANCHOR = "docs/benchflow/getting-started/install#which-bench-runs"


def _make_program(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _entry_script(interpreter: Path) -> bytes:
    # What uv (and pip) write for ``[project.scripts] bench``.
    return (
        f"#!{interpreter}\n"
        "# -*- coding: utf-8 -*-\n"
        "import sys\n"
        "from benchflow.cli.main import main\n"
        'if __name__ == "__main__":\n'
        "    sys.exit(main())\n"
    ).encode()


def _rust_binary(path: Path) -> Path:
    # An ELF-shaped file with the marker well past the first read.
    body = (
        b"\x7fELF"
        + b"\0" * (2 * 1024 * 1024)
        + b"label ai.benchflow.owner="
        + b"\0" * 64
    )
    return _make_program(path, body)


class Machine:
    """A home folder with this install in uv's tool layout, plus stand-ins."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.home = tmp_path / "home"
        self.home.mkdir()
        tool_bin = self.home / ".local/share/uv/tools/benchflow/bin"
        self.python = tool_bin / "python"
        self.tool_bench = _make_program(tool_bin / "bench", _entry_script(self.python))
        self.local_bin = self.home / ".local/bin"
        self.local_bin.mkdir(parents=True)
        # uv links the tool's entry points into ~/.local/bin.
        (self.local_bin / "bench").symlink_to(self.tool_bench)
        self.runs: list[tuple[str, ...]] = []
        self.rust_dir = self.home / ".bench/bin"

    def add_rust(self, folder: Path | None = None) -> Path:
        return _rust_binary((folder or self.rust_dir) / "bench")

    def probes(
        self,
        path_dirs: Sequence[Path],
        *,
        entry: Path | None = None,
        python: Path | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> DoctorProbes:
        def run(argv: Sequence[str], timeout: float) -> CommandResult:
            self.runs.append(tuple(argv))
            if (
                argv[0].endswith("/bench")
                and Path(argv[0]).read_bytes()[:4] == b"\x7fELF"
            ):
                return CommandResult(0, RUST_VERSION_LINE, "")
            if argv[0].endswith("/bench"):
                # Another BenchFlow install's stand-in really runs.
                return doctor._run_command(argv, timeout)
            return CommandResult(None, "", "not found")

        environ = {"PATH": os.pathsep.join(str(d) for d in path_dirs)}
        environ.update(extra_env or {})
        probes = DoctorProbes(
            environ=environ,
            origins={},
            home=self.home,
            which=lambda name: None,
            run=run,
            dist_version=lambda dist: None,
            python_executable=str(python or self.python),
        )
        probes.entry_point = str(entry or self.local_bin / "bench")
        return probes


def bench_line(report) -> doctor.Check:
    lines = [c for c in report.checks if c.id == "bench-path"]
    assert lines, "doctor printed no 'bench on PATH' line"
    return lines[0]


def _line(machine: Machine, path_dirs: Sequence[Path], **kwargs) -> doctor.Check:
    report = run_doctor(probes=machine.probes(path_dirs, **kwargs), offline=True)
    return bench_line(report)


@pytest.fixture
def machine(tmp_path, monkeypatch) -> Machine:
    m = Machine(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(m.local_bin / "bench"), "doctor"])
    return m


def test_this_install_alone_on_path_passes(machine):
    line = _line(machine, [machine.local_bin, Path("/usr/bin")])
    assert line.status == "pass"
    assert line.summary == "~/.local/bin/bench, this install"
    assert line.group == "runtime"
    [entry] = line.details["on_path"]
    assert entry["this_install"] is True
    assert entry["version"] == __version__
    assert entry["resolves_to"] == os.path.realpath(machine.tool_bench)


def test_rust_bench_later_on_path_is_named_as_hidden(machine):
    rust = machine.add_rust()
    line = _line(machine, [machine.local_bin, machine.rust_dir])
    assert line.status == "warn"
    assert line.summary == (
        "~/.local/bin/bench, this install; it hides ~/.bench/bin/bench (Rust bench)"
    )
    assert "full path" in line.fix and INSTALL_ANCHOR in line.fix
    assert [e["kind"] for e in line.details["on_path"]] == ["python", "rust"]
    # Neither the Rust stand-in nor this install was run.
    assert not any(run[0] == str(rust) for run in machine.runs)
    assert not any(run[0] == str(machine.local_bin / "bench") for run in machine.runs)
    assert line.details["on_path"][1]["version"] is None


def test_rust_bench_first_on_path_shadows_this_install(machine):
    machine.add_rust()
    line = _line(machine, [machine.rust_dir, machine.local_bin])
    assert line.status == "warn"
    assert line.summary == (
        "bench runs ~/.bench/bin/bench (Rust bench), "
        "not this install (~/.local/bin/bench)"
    )
    assert line.fix.startswith("benchflow always runs this install")
    assert f"put {machine.local_bin} before {machine.rust_dir} on PATH" in line.fix


def test_rust_bench_replacing_uvs_link_is_first_and_this_install_is_off_path(
    machine,
):
    # The Rust install script with --force put its binary where uv's link
    # was; this install still runs as ~/.local/bin/benchflow.
    (machine.local_bin / "bench").unlink()
    machine.add_rust(machine.local_bin)
    entry = _make_program(
        machine.local_bin / "benchflow", _entry_script(machine.python)
    )
    line = _line(machine, [machine.local_bin], entry=entry)
    assert line.status == "warn"
    assert line.summary == (
        "bench runs ~/.local/bin/bench (Rust bench), "
        "not this install (~/.local/bin/benchflow)"
    )
    # Moving folders cannot help when both would sit in one folder. The
    # reinstall is the user's own command, so its version, extras and
    # constraints stay; a bare one dropped them.
    assert "run the install command you installed it with again" in line.fix
    assert f"--force (benchflow=={__version__} and the same extras" in line.fix
    assert f"which replaces {machine.local_bin}/bench" in line.fix
    assert "--force benchflow," not in line.fix
    assert f"before {machine.local_bin}" not in line.fix


def test_python_m_benchflow_off_path_is_told_how_to_run_itself(machine, tmp_path):
    # .venv/bin/python -m benchflow doctor, venv not activated, Rust bench on
    # PATH: no reinstall gives this copy the name, and benchflow is not it.
    python = tmp_path / "project/.venv/bin/python"
    site = tmp_path / "project/.venv/lib/python3.12/site-packages"
    entry = site / "benchflow/__main__.py"
    machine.add_rust()
    line = _line(machine, [machine.rust_dir], entry=entry, python=python)
    assert line.status == "warn"
    assert line.summary == (
        "bench runs ~/.bench/bin/bench (Rust bench), "
        f"not this install ({python} -m benchflow)"
    )
    assert line.fix.startswith(
        f"This install has no bench on PATH; run it as {python} -m benchflow"
    )
    assert "--force" not in line.fix
    assert "always runs this install" not in line.fix


def test_another_python_install_first_is_named_but_never_run(machine, tmp_path):
    other_dir = tmp_path / "project/.venv/bin"
    other = _make_program(
        other_dir / "bench",
        # Runs as a shell script; read as an entry script (0.7's import line).
        b"#!/bin/sh\necho 'benchflow 0.7.9'\nexit 0\n"
        b"from benchflow.cli.main import app\n",
    )
    line = _line(machine, [other_dir, machine.local_bin])
    assert line.status == "warn"
    assert line.summary.startswith(
        f"bench runs {other_dir / 'bench'} (BenchFlow (Python)), not this install"
    )
    assert not any(run[0] == str(other) for run in machine.runs)


def test_a_hidden_script_with_benchflows_import_line_is_never_run(machine, tmp_path):
    """A project's bin/ later on PATH (absolute, so not skipped
    as relative) holds a shell script with an unreachable BenchFlow import
    line. Doctor read it as BenchFlow and ran it for its version, on the
    host, under --offline too, although the shell never runs it."""
    project_bin = tmp_path / "d4-review/project/bin"
    witness = tmp_path / "witness"
    other = _make_program(
        project_bin / "bench",
        f"#!/bin/sh\ntouch {witness}\nexit 0\n"
        "from benchflow.cli.main import app\n".encode(),
    )
    line = _line(machine, [machine.local_bin, project_bin])
    assert line.status == "warn"
    assert line.summary.endswith(f"it hides {other} (BenchFlow (Python))")
    assert not witness.exists()
    assert not any(run[0] == str(other) for run in machine.runs)


def test_another_program_called_bench_is_never_run(machine, tmp_path):
    other_dir = tmp_path / "opt/bin"
    other = _make_program(other_dir / "bench", b"#!/bin/sh\necho unrelated\n")
    line = _line(machine, [machine.local_bin, other_dir])
    assert line.status == "warn"
    assert line.summary.endswith(f"it hides {other} (another program called bench)")
    assert not any(run[0] == str(other) for run in machine.runs)


def test_a_script_that_only_mentions_benchflow_is_never_run(machine, tmp_path):
    # A user's own wrapper: the word is there, the entry point's import is not.
    other_dir = tmp_path / "wrappers"
    other = _make_program(
        other_dir / "bench",
        b"#!/bin/bash\ncd /tmp/made-up/benchflow && echo 'benchflow 9.9.9'\n",
    )
    line = _line(machine, [machine.local_bin, other_dir])
    assert line.status == "warn"
    assert line.summary.endswith(f"it hides {other} (another program called bench)")
    assert not any(run[0] == str(other) for run in machine.runs)


def test_one_program_reached_from_two_folders_hides_nothing(machine, tmp_path):
    alias_dir = tmp_path / "aliases"
    alias_dir.mkdir()
    (alias_dir / "bench").symlink_to(machine.tool_bench)
    line = _line(machine, [machine.local_bin, alias_dir, machine.local_bin])
    assert line.status == "pass"
    assert len(line.details["on_path"]) == 2


def test_no_bench_on_path_points_at_update_shell(machine):
    line = _line(machine, [Path("/nonexistent-bin")])
    assert line.status == "warn"
    assert line.summary == "no bench on PATH; this one is ~/.local/bin/bench"
    assert "uv tool update-shell" in line.fix


def test_a_script_install_is_known_by_its_interpreter(machine, tmp_path):
    # pip install --user: the entry script is not beside the interpreter.
    interpreter = tmp_path / "usr/bin/python3.12"
    user_bin = machine.home / "user-scripts"
    script = _make_program(user_bin / "bench", _entry_script(interpreter))
    line = _line(machine, [user_bin], entry=script, python=interpreter)
    assert line.status == "pass"
    assert line.summary == "~/user-scripts/bench, this install"


def test_folders_and_non_executables_are_not_programs(machine, tmp_path):
    folder = tmp_path / "has-a-folder"
    (folder / "bench").mkdir(parents=True)
    plain = tmp_path / "plain-file"
    plain.mkdir()
    (plain / "bench").write_bytes(b"#!/bin/sh\n")
    line = _line(machine, [folder, plain, machine.local_bin])
    assert line.status == "pass"
    assert len(line.details["on_path"]) == 1


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="root can search a mode-000 folder",
)
def test_a_path_folder_this_user_cannot_search_is_skipped(machine, tmp_path):
    # ENV PATH=/root/.local/bin:$PATH, then USER app: the folder is not ours.
    locked = tmp_path / "locked-bin"
    _make_program(locked / "bench", _entry_script(machine.python))
    locked.chmod(0)
    try:
        report = run_doctor(
            probes=machine.probes([locked, machine.local_bin]), offline=True
        )
    finally:
        locked.chmod(0o755)
    line = bench_line(report)
    assert line.status == "pass"
    assert line.summary == "~/.local/bin/bench, this install"
    # The rest of the report is still there.
    assert len(report.checks) > 1


def test_relative_path_folders_are_neither_read_nor_run(machine, tmp_path, monkeypatch):
    # Doctor run inside a downloaded folder, with PATH holding relative
    # entries (binstub setups): their bench files depend on the working
    # folder and must not run.
    downloaded = tmp_path / "downloaded-task"
    planted = (
        b"#!/bin/sh\necho planted > ran.txt\necho 'benchflow 6.6.6'\nexit 0\n"
        b"from benchflow.cli.main import main\n"
    )
    for sub in ("bin", ".", "node_modules/.bin"):
        _make_program(downloaded / sub / "bench", planted)
    monkeypatch.chdir(downloaded)
    rel = [Path("bin"), Path("."), Path("node_modules/.bin")]
    line = _line(machine, [machine.local_bin, *rel])
    assert line.status == "pass"
    assert line.summary == "~/.local/bin/bench, this install"
    assert len(line.details["on_path"]) == 1
    assert not (downloaded / "ran.txt").exists()
    benches = [run[0] for run in machine.runs if run[0].endswith("bench")]
    assert all(os.path.isabs(b) for b in benches), benches


def test_read_bench_file_finds_the_marker_across_a_read_boundary(tmp_path):
    marker = doctor.RUST_BENCH_MARKER
    # The marker straddles the 4096-byte first read and the next chunk.
    body = b"\x7fELF" + b"\0" * (4096 - 4 - 5) + marker + b"\0" * 10
    path = _make_program(tmp_path / "bench", body)
    assert doctor.read_bench_file(path).kind == "rust"
    assert doctor.read_bench_file(_make_program(tmp_path / "x", b"\x7fELF")).kind == (
        "other"
    )


def test_read_bench_file_reads_the_sh_exec_form_with_a_space(tmp_path):
    # uv writes this form when the interpreter's path has a space in it.
    script = (
        b"#!/bin/sh\n"
        b"'''exec' '/opt/my tools/bin/python' \"$0\" \"$@\"\n"
        b"' '''\n"
        b"from benchflow.cli.main import main\n"
    )
    found = doctor.read_bench_file(_make_program(tmp_path / "bench", script))
    assert found.kind == "python"
    assert found.interpreter == "/opt/my tools/bin/python"


def test_doctor_command_prints_the_line(machine, monkeypatch):
    from benchflow.cli.main import app

    machine.add_rust()
    probes = machine.probes([machine.rust_dir, machine.local_bin])
    monkeypatch.setattr(DoctorProbes, "from_host", classmethod(lambda cls: probes))
    result = CliRunner().invoke(app, ["doctor", "--offline"])
    out = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert re.search(
        r"WARN\s+bench on PATH\s+bench runs ~/\.bench/bin/bench \(Rust bench\), "
        r"not this install \(~/\.local/bin/bench\)",
        out,
    ), out
    assert INSTALL_ANCHOR in out


def test_windows_is_skipped_not_reported_as_missing(machine):
    probes = machine.probes([machine.local_bin])
    probes.system = "Windows"
    line = bench_line(run_doctor(probes=probes, offline=True))
    assert line.status == "skip"
    assert line.summary == "not checked on Windows"
