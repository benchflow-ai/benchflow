"""PT round 14: unit checks next to ``test_c8_r14_container`` (real Docker).

- PT-R13-B3: an empty entry in a Python/Node/Perl path list (``ENV
  PYTHONPATH=${PYTHONPATH}:/opt/vendor``) is dropped, the rest kept; the
  loader's names keep refusing it.
- PT-R13-B1: the guard installer works out the tests' interpreter's search
  path as data and installs where that path finds it.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from benchflow._root_only import ROOT_CHAIN_FUNCS
from tests._readonly_stdlib_python import python_version, readonly_stdlib_python


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/sh", "-c", script, "sh", *args],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )


def _value_ok(name: str, value: str) -> bool:
    from benchflow.sandbox._base import env_value_check_func

    script = ROOT_CHAIN_FUNCS + env_value_check_func() + 'bf_value_ok "$1" "$2"'
    return _run(script, name, value).returncode == 0


def _clean(name: str, value: str) -> str:
    from benchflow.sandbox._base import env_value_clean_func

    return _run(env_value_clean_func() + 'bf_value_clean "$1" "$2"', name, value).stdout


@pytest.mark.skipif(
    os.getuid() == 0 or os.stat("/usr/lib").st_uid != 0,
    reason="needs a root-owned /usr/lib and a non-root test user",
)
@pytest.mark.parametrize(
    ("name", "value", "ok"),
    [
        ("PYTHONPATH", ":/usr/lib", True),  # fails on e4ec66fa
        ("PYTHONPATH", "/usr/lib::/usr/share", True),  # fails on e4ec66fa
        ("PYTHONPATH", "/usr/lib:", True),  # fails on e4ec66fa
        ("NODE_PATH", ":/usr/lib", True),  # fails on e4ec66fa
        ("PERL5LIB", ":/usr/lib", True),  # fails on e4ec66fa
        ("PYTHONPATH", ":", False),  # nothing left to restore
        ("PYTHONPATH", ":{own}", False),  # what is left must be root's
        ("PYTHONPATH", ":relative", False),
        ("LD_LIBRARY_PATH", "/usr/lib", True),
        # PT-R13-A-1 then PT-R14-B1: an empty entry anywhere in a loader list
        # (leading, trailing, doubled) is dropped, not the whole value
        # refused; the trailing one fails on 899eca7e. ';', which glibc
        # splits on too, is still refused, as is a value with nothing left.
        ("LD_LIBRARY_PATH", ":/usr/lib", True),
        ("LD_LIBRARY_PATH", "/usr/lib:", True),
        ("LD_LIBRARY_PATH", "/usr/lib::/usr/share", True),
        ("LD_LIBRARY_PATH", "/usr/lib;/usr/share", False),
        ("LD_LIBRARY_PATH", ":", False),
        ("LD_LIBRARY_PATH", "/usr/lib:{own}", False),
        ("LD_PRELOAD", "/usr/lib/x.so:", True),
        ("GCONV_PATH", "/usr/lib:", True),
        # ... and in a language list it is dropped wherever it is.
        ("PYTHONPATH", "/usr/lib:", True),
        ("PYTHONPATH", ":/usr/lib:", True),
        ("NODE_PATH", "/usr/lib::/usr/share", True),
    ],
)
def test_an_empty_entry_is_dropped_from_every_path_list(tmp_path, name, value, ok):
    assert _value_ok(name, value.format(own=tmp_path)) is ok


@pytest.mark.parametrize(
    ("name", "value", "cleaned"),
    [
        ("PYTHONPATH", ":/opt/vendor", "/opt/vendor"),
        ("PYTHONPATH", "/opt/vendor:", "/opt/vendor"),
        ("PYTHONPATH", "/opt/a::/opt/b", "/opt/a:/opt/b"),
        ("NODE_PATH", "::/opt/node::", "/opt/node"),
        ("PYTHONPATH", "/opt/a::/opt/b:", "/opt/a:/opt/b"),
        ("PERL5LIB", ":/opt/perl", "/opt/perl"),
        ("PYTHONPATH", "/opt/with space", "/opt/with space"),
        ("LD_LIBRARY_PATH", "/opt/lib", "/opt/lib"),
        ("LD_LIBRARY_PATH", "/opt/vendor/lib:", "/opt/vendor/lib"),  # PT-R14-B1
        ("LD_LIBRARY_PATH", ":/opt/a::/opt/b", "/opt/a:/opt/b"),
        ("LD_PRELOAD", "/opt/a.so /opt/b.so", "/opt/a.so:/opt/b.so"),
    ],
)
def test_root_gets_the_value_without_its_empty_entries(name, value, cleaned):
    assert _clean(name, value) == cleaned


def test_the_restore_assigns_no_variable_of_its_own():
    """bf_value_clean's scratch names live in a subshell: a task variable
    named like them keeps its value in the command."""
    from benchflow.sandbox._base import env_value_clean_func

    done = _run(
        "bf_o=task-o; v=task-v; "
        + env_value_clean_func()
        + 'x=$(bf_value_clean PYTHONPATH ":/opt/vendor"); printf "%s|%s|%s" "$x" "$bf_o" "$v"'
    )
    assert done.stdout == "/opt/vendor|task-o|task-v"


# --- PT-R13-B1: the guard's target, worked out as data ----------------------


def _guard_install(python: str, tmp_path, name: str = "_bf_guard_probe"):
    from benchflow.sandbox.lockdown import _INSTALL_PYTEST_PLUGIN_GUARD_SCRIPT

    source = tmp_path / "guard.py"
    source.write_text("GUARD = 1\n")
    return subprocess.run(
        [python, "-I", "-S", "-B", "-c", _INSTALL_PYTEST_PLUGIN_GUARD_SCRIPT,
         str(source), name],
        capture_output=True,
        text=True,
    )  # fmt: skip


def test_a_venvs_guard_goes_into_its_own_site_packages(tmp_path):
    """The folder site.py puts on a venv's path, and the venv imports it.

    The venv's base is a Python whose standard library the test cannot write
    the installer writes into the first folder of the path it can,
    and the standard library comes first, so a writable one (a Python the
    test user installed, as CI's ``uv python install``) took the guard into
    that shared folder instead of the venv's."""
    base = readonly_stdlib_python()
    if base is None:
        pytest.skip("no Python whose standard library this user cannot write")
    venv = tmp_path / "venv"
    made = subprocess.run(
        [base, "-m", "venv", "--without-pip", str(venv)],
        capture_output=True,
    )
    if made.returncode != 0:
        pytest.skip("cannot make a venv here")
    python = str(venv / "bin" / "python")
    done = _guard_install(python, tmp_path)
    assert done.returncode == 0, done.stderr
    version = python_version(python)
    target = venv / "lib" / version / "site-packages" / "_bf_guard_probe.py"
    assert target.is_file()
    imported = subprocess.run(
        [python, "-c", "import _bf_guard_probe, sys; print(_bf_guard_probe.__file__)"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert imported.returncode == 0, imported.stderr
    assert os.path.realpath(imported.stdout.strip()) == os.path.realpath(target)
