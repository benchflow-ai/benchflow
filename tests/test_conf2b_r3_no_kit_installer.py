"""PT-C8-CONF-2b round 3 (review B of round 2, M1, second line): the guard
installer's no-kit branch never starts an interpreter that is, or resolves
to, a file under a blocked prefix (the agent's folders, or what the
transfer brought into a separate verifier). It prints it as unrecorded, with
a ``benchflow:`` line, and the verifier's PYTHONPATH carries its guard.

The installer runs in sh here; its "interpreters" are scripts that leave a
mark when started. The real-Docker test is
``test_conf2b_r3_separate_verifier_container.py``.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from benchflow.sandbox import lockdown

pytestmark = pytest.mark.skipif(os.name != "posix", reason="sh")

NAME = "_bf_guard_conf2b_r3"


def _interpreter(path: Path, marks: Path) -> Path:
    """A "python" that marks any start, then fails the version check."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\necho "$0" >> {marks}\nexit 3\n')
    path.chmod(0o755)
    return path


def _install(tmp_path: Path, path: str, blocked: tuple[str, ...]):
    guard = tmp_path / "guard.py"
    guard.write_text("GUARD = 1\n")
    script = (
        lockdown._INSTALL_PYTEST_PLUGIN_GUARD_CMD_TEMPLATE.replace(
            "__GUARD__", shlex.quote(str(guard))
        )
        .replace("__NAME__", shlex.quote(NAME))
        .replace("__PATH__", shlex.quote(path))
        .replace(
            "__SCRIPT__", shlex.quote(lockdown._INSTALL_PYTEST_PLUGIN_GUARD_SCRIPT)
        )
        # No kit: its record is not there.
        .replace("__KIT_RECORD__", shlex.quote(str(tmp_path / "no-kit-record")))
        .replace("__BLOCKED__", lockdown._guard_blocked_words(blocked))
        .replace("__KIT_INSTALL__", ":")
    )
    return subprocess.run(
        ["/bin/sh", "-c", script], capture_output=True, text=True, timeout=60
    )


def _started(marks: Path) -> list[str]:
    return marks.read_text().splitlines() if marks.exists() else []


def test_a_link_into_a_blocked_folder_is_not_started(tmp_path):
    marks = tmp_path / "started"
    agent = _interpreter(tmp_path / "app" / ".venv" / "bin" / "python", marks)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "python3").symlink_to(agent)
    done = _install(tmp_path, str(tmp_path / "bin"), (str(tmp_path / "app"),))
    assert done.returncode == 0, done.stderr
    assert _started(marks) == []
    python = tmp_path / "bin" / "python3"
    assert f"unrecorded {python}" in done.stdout.splitlines()
    assert f"benchflow: the pytest plugin guard reaches {python} " in done.stdout


def test_a_pytest_shebang_naming_a_transferred_python_is_not_started(tmp_path):
    marks = tmp_path / "started"
    agent = _interpreter(tmp_path / "app" / ".venv" / "bin" / "python", marks)
    (tmp_path / "bin").mkdir()
    pytest_script = tmp_path / "bin" / "pytest"
    pytest_script.write_text(f"#!{agent}\nimport pytest\n")
    pytest_script.chmod(0o755)
    # The prefix as blocked lists may spell it: with a trailing slash.
    done = _install(tmp_path, str(tmp_path / "bin"), (str(tmp_path / "app") + "/",))
    assert done.returncode == 0, done.stderr
    assert _started(marks) == []
    assert f"unrecorded {agent}" in done.stdout.splitlines()


def test_an_interpreter_outside_the_blocked_folders_is_still_checked(tmp_path):
    marks = tmp_path / "started"
    python = _interpreter(tmp_path / "bin" / "python3", marks)
    done = _install(tmp_path, str(tmp_path / "bin"), (str(tmp_path / "app"),))
    assert done.returncode == 0, done.stderr
    # Started by the version check, as before; not reported.
    assert _started(marks) == [str(python)]
    assert "unrecorded" not in done.stdout


def test_with_no_blocked_prefixes_every_interpreter_is_checked_as_before(tmp_path):
    # A main sandbox without a kit (no sandbox user): the installer is
    # given no prefixes, and checks each interpreter as it did.
    marks = tmp_path / "started"
    agent = _interpreter(tmp_path / "app" / ".venv" / "bin" / "python", marks)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "python3").symlink_to(agent)
    done = _install(tmp_path, str(tmp_path / "bin"), ())
    assert done.returncode == 0, done.stderr
    assert _started(marks) == [str(tmp_path / "bin" / "python3")]
    assert "unrecorded" not in done.stdout


@pytest.mark.parametrize(
    ("blocked", "words"),
    [
        (("/app", "/logs"), "/app /logs"),
        (("/app/", "/app"), "/app"),
        # "/" and "" would match every path; relative entries match none.
        (("/", "", "app", "/tmp"), "/tmp"),
        (("/opt/my env",), "'/opt/my env'"),
        ((), ""),
    ],
)
def test_the_blocked_words(blocked, words):
    assert lockdown._guard_blocked_words(blocked) == words
