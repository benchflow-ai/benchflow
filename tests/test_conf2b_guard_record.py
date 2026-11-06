"""PT-C8-CONF-2b: the pytest guard installer's kit
branch, in sh against a record, without starting any Python.

With a kit, the installer reads each interpreter's search path from the
record taken before the agent's turn (``IMAGE_PYTHON_GUARD``) and writes and
resolves the guard by the rules of ``_INSTALL_PYTEST_PLUGIN_GUARD_SCRIPT``.
The interpreters here are scripts that leave a mark when started: none may
be. The real-Docker test is ``test_conf2b_guard_install_container.py``.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from benchflow._image_python_paths import (
    GUARD_ID_SH,
    GUARD_SEARCH,
    GUARD_SYSTEM_PATH,
    IMAGE_PYTHON_GUARD,
)
from benchflow.sandbox import lockdown
from benchflow.sandbox._hardening_python import TRUSTED_SYSTEM_PATH

pytestmark = pytest.mark.skipif(os.name != "posix", reason="sh and stat")

NAME = "_bf_guard_conf2b"


def _python(tmp_path: Path, name: str = "python3") -> Path:
    """An "interpreter" that marks any start and fails."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    python = bin_dir / name
    python.write_text(f"#!/bin/sh\necho started >> {tmp_path / 'started'}\nexit 3\n")
    python.chmod(0o755)
    return python


def _identity(python: Path) -> str:
    done = subprocess.run(
        ["/bin/sh", "-c", GUARD_ID_SH + '\nbf_guard_id "$1"', "sh", str(python)],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout


def _record(
    tmp_path: Path, python: Path, body: str, identity: str | None = None
) -> Path:
    record = tmp_path / "IMAGE_PYTHON_GUARD"
    with record.open("a") as handle:
        handle.write(
            f"exe {python}\nid {_identity(python) if identity is None else identity}\n"
        )
        handle.write(body)
    return record


def _install(
    tmp_path: Path,
    python: Path,
    record: Path,
    *,
    path: str | None = None,
):
    guard = tmp_path / "guard.py"
    if not guard.exists():
        guard.write_text("GUARD = 1\n")
    kit = lockdown._install_guard_kit_cmd(str(guard), NAME).replace(
        shlex.quote(IMAGE_PYTHON_GUARD), shlex.quote(str(record))
    )
    return subprocess.run(
        ["/bin/sh", "-c", kit + '\ninstall_guard "$1"', "sh", str(python)],
        capture_output=True,
        text=True,
        env=None if path is None else {"PATH": path},
    )


def test_a_recorded_interpreter_gets_its_guard_without_a_start(tmp_path):
    python = _python(tmp_path)
    locked, first, second = tmp_path / "locked", tmp_path / "first", tmp_path / "second"
    for folder in (locked, first, second):
        folder.mkdir()
    locked.chmod(0o555)
    try:
        record = _record(
            tmp_path,
            python,
            f"py3 1\ndir {tmp_path / 'absent'}\ndir {locked}\ndir {first}\n"
            f"dir {second}\nend\n",
        )
        done = _install(tmp_path, python, record)
    finally:
        locked.chmod(0o755)
    if os.geteuid() == 0:
        pytest.skip("root writes the read-only folder")
    assert done.returncode == 0, done.stderr
    assert done.stdout == f"guarded {python}\n"
    target = first / f"{NAME}.py"
    assert target.read_text() == "GUARD = 1\n"
    assert oct(target.stat().st_mode & 0o777) == "0o444"
    assert not (second / f"{NAME}.py").exists()
    assert not (tmp_path / "started").exists()


def test_a_second_interpreter_on_the_same_folder_shares_the_copy(tmp_path):
    first = _python(tmp_path, "python3")
    second = _python(tmp_path, "python3.12")
    site = tmp_path / "site"
    site.mkdir()
    _record(tmp_path, first, f"py3 1\ndir {site}\nend\n")
    record = _record(tmp_path, second, f"py3 1\ndir {site}\nend\n")
    assert _install(tmp_path, first, record).returncode == 0
    done = _install(tmp_path, second, record)
    assert done.returncode == 0, done.stderr
    assert done.stdout == f"guarded {second}\n"
    assert not (tmp_path / "started").exists()


def test_a_recorded_python_2_is_skipped_as_before(tmp_path):
    python = _python(tmp_path)
    record = _record(tmp_path, python, "py3 0\n")
    done = _install(tmp_path, python, record)
    assert (done.returncode, done.stdout, done.stderr) == (0, "", "")
    assert not (tmp_path / "started").exists()


def test_a_record_that_stopped_before_end_fails_as_before(tmp_path):
    python = _python(tmp_path)
    site = tmp_path / "site"
    site.mkdir()
    record = _record(tmp_path, python, f"py3 1\ndir {site}\n")
    done = _install(tmp_path, python, record)
    assert done.returncode == 1
    assert f"Cannot install the pytest plugin guard for {python}" in done.stderr
    assert not (site / f"{NAME}.py").exists()


def test_a_different_module_in_the_way_is_an_error(tmp_path):
    python = _python(tmp_path)
    site = tmp_path / "site"
    site.mkdir()
    (site / f"{NAME}.py").write_text("OTHER = 1\n")
    record = _record(tmp_path, python, f"py3 1\ndir {site}\nend\n")
    done = _install(tmp_path, python, record)
    assert done.returncode == 1
    assert "a different module occupies" in done.stderr


def test_no_folder_on_the_search_path_is_an_error(tmp_path):
    python = _python(tmp_path)
    record = _record(tmp_path, python, f"py3 1\ndir {tmp_path / 'absent'}\nend\n")
    done = _install(tmp_path, python, record)
    assert done.returncode == 1
    assert "no site directory on its search path" in done.stderr


def test_an_earlier_module_of_the_same_name_fails_the_resolution(tmp_path):
    """The finder's order: an extension module in an earlier folder of the
    search path wins over the copy, so the copy would never load."""
    python = _python(tmp_path)
    earlier, site = tmp_path / "earlier", tmp_path / "site"
    earlier.mkdir()
    site.mkdir()
    (earlier / f"{NAME}.cpython-312-x86_64-linux-gnu.so").write_bytes(b"\0")
    earlier.chmod(0o555)
    try:
        record = _record(tmp_path, python, f"py3 1\ndir {earlier}\ndir {site}\nend\n")
        done = _install(tmp_path, python, record)
    finally:
        earlier.chmod(0o755)
    if os.geteuid() == 0:
        pytest.skip("root writes the read-only folder")
    assert done.returncode == 1
    assert "does not resolve" in done.stderr


def test_a_linked_folder_resolving_to_the_copy_passes(tmp_path):
    python = _python(tmp_path)
    site = tmp_path / "site"
    site.mkdir()
    (tmp_path / "alias").symlink_to(site)
    record = _record(
        tmp_path, python, f"py3 1\ndir {site}\ndir {tmp_path / 'alias'}\nend\n"
    )
    done = _install(tmp_path, python, record)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("why", ["unrecorded", "changed", "uncheckable"])
def test_an_interpreter_not_as_recorded_is_never_started(tmp_path, why):
    """Not in the record, in it as another file, or in it with an id that
    cannot be checked (no stat): not started by anyone, no copy, reported as
    unrecorded with a benchflow: line, so the verifier's PYTHONPATH carries
    its guard (CONF-2b round 2)."""
    python = _python(tmp_path)
    site = tmp_path / "site"
    site.mkdir()
    other = _python(tmp_path, "python3.99")
    if why == "unrecorded":
        record = _record(tmp_path, other, f"py3 1\ndir {site}\nend\n")
    elif why == "changed":
        record = _record(
            tmp_path, python, f"py3 1\ndir {site}\nend\n", identity="2:3:4 - - "
        )
    else:
        record = _record(tmp_path, python, f"py3 1\ndir {site}\nend\n")
    path = _tools_without_stat(tmp_path) if why == "uncheckable" else None
    done = _install(tmp_path, python, record, path=path)
    assert done.returncode == 0, done.stderr
    assert done.stdout == (
        f"unrecorded {python}\n"
        f"benchflow: the pytest plugin guard reaches {python} through PYTHONPATH "
        "only: it is not in the pre-agent record as this file\n"
    )
    assert not (tmp_path / "started").exists()
    assert not (site / f"{NAME}.py").exists()


def _tools_without_stat(tmp_path: Path) -> str:
    """A PATH with the installer's tools but no stat: an image without one."""
    tools = tmp_path / "tools"
    tools.mkdir()
    for tool in ("cat", "realpath", "chmod"):
        found = subprocess.run(
            ["/bin/sh", "-c", f"command -v {tool}"], capture_output=True, text=True
        ).stdout.strip()
        if found:
            (tools / tool).symlink_to(found)
    return str(tools)


def test_an_id_without_stat_is_not_the_recorded_one(tmp_path):
    python = _python(tmp_path)
    done = subprocess.run(
        ["/bin/sh", "-c", GUARD_ID_SH + '\nbf_guard_id "$1"', "sh", str(python)],
        capture_output=True,
        text=True,
        env={"PATH": _tools_without_stat(tmp_path)},
    )
    assert done.stdout.split() == ["?", "-", "-"]


def test_the_id_leaves_out_the_device(tmp_path):
    """An overlay's device differs between two containers of one image; a kit
    restored into another one must still match its record."""
    python = _python(tmp_path)
    inode, size, mtime = _identity(python).split()[0].split(":")
    stat = python.stat()
    assert (int(inode), int(size), int(mtime)) == (
        stat.st_ino,
        stat.st_size,
        int(stat.st_mtime),
    )


def test_a_record_cut_off_right_after_py3_1_fails_as_before(tmp_path):
    """The rule raised before its first folder (an unreadable pyvenv.cfg):
    the old install script failed there, so hardening fails, not skips."""
    python = _python(tmp_path)
    record = _record(tmp_path, python, "py3 1\n")
    done = _install(tmp_path, python, record)
    assert done.returncode == 1
    assert f"Cannot install the pytest plugin guard for {python}" in done.stderr
    assert not (tmp_path / "started").exists()


def test_the_identity_follows_links_and_names_both_venv_files(tmp_path):
    python = _python(tmp_path)
    (tmp_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
    link = tmp_path / "bin" / "python"
    link.symlink_to(python)
    plain, linked = _identity(python).split(), _identity(link).split()
    assert plain == linked
    assert plain[1] == "-" and plain[2] != "-"


def test_the_capture_reads_the_fallback_verifier_path():
    assert GUARD_SYSTEM_PATH == TRUSTED_SYSTEM_PATH


def test_the_search_rule_is_the_one_both_installers_use():
    from benchflow._image_python_paths import GUARD_SEARCH_RULE

    assert GUARD_SEARCH_RULE in lockdown._INSTALL_PYTEST_PLUGIN_GUARD_SCRIPT
    assert GUARD_SEARCH_RULE in GUARD_SEARCH
    rendered = lockdown._install_guard_kit_cmd("/g/x.py", NAME)
    # The installer starts no interpreter: nothing to run them with.
    assert shlex.quote(GUARD_SEARCH) not in rendered
    assert "setpriv" not in rendered


@pytest.mark.parametrize("unrecorded", [(), ("/opt/late/bin/python3",)])
async def test_an_unrecorded_python_puts_the_guard_on_pythonpath(
    monkeypatch, unrecorded
):
    """A Python the pre-agent record does not vouch for counts as one without
    a copy, even when it or every other Python took one (CONF-2b round 2)."""
    from unittest.mock import AsyncMock

    from benchflow._utils.trusted_path import TrustedPathReport
    from tests.test_sandbox_hardening import _make_env, _make_task

    for name, result in (
        ("_probe_verifier_reads", None),
        ("_judge_verifier_path", TrustedPathReport(path="/usr/bin:/bin")),
        ("_isolate_verifier_tool_state", {}),
        ("_discover_pytest_plugins", ([], [])),
        (
            "_install_pytest_plugin_guard",
            ("/guard", "-p guard", ("/usr/bin/python3",), unrecorded),
        ),
        ("_trusted_verifier_pythonpath", "/image"),
        ("_distro_pip_env", {}),
        ("_verifier_uid", 0),
    ):
        monkeypatch.setattr(lockdown, name, AsyncMock(return_value=result))
    monkeypatch.setattr(
        lockdown, "_verifier_may_run_uncopied_python", lambda *_args: False
    )
    sandbox = _make_env()
    env = await lockdown._build_verifier_env(
        sandbox, _make_task(), "agent", "/app", bash_env=""
    )
    assert env["PYTHONPATH"] == ("/guard:/image" if unrecorded else "/image")
    # For the verifier's guard-load error (CONF-2b round 3).
    assert sandbox._benchflow_guard_unrecorded == unrecorded
