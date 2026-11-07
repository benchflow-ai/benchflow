"""The conftest sweep keeps an installed package's own conftest.py and still
removes every planted one.

Guards this branch's fix against the failure round 3 found by
running the scripted-users example: between rounds, the sweep removed
beartype's ``_conf/conftest.py`` (a module, not a pytest file) from the
openhands install tree, so openhands could not start again for round 2
(``ModuleNotFoundError: No module named 'beartype._conf.conftest'``).

A conftest.py now stays only when a ``*.dist-info/RECORD`` in its own
site-packages or dist-packages folder lists it with its exact sha256, and
never inside the workspace. Both forms are checked on the same tree: the bfx
form on the host, and the shell form's script on real files.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow._root_python import FLAGS, _script
from benchflow._runtime_tree import KIT_RECORD
from benchflow.sandbox import _hook_cleanup, lockdown
from tests.test_hook_cleanup_bfx import INPUTS, _FakeFs

SITE = "/opt/runtime/uv-tools/openhands/lib/python3.12/site-packages"
OWN_PATH = f"{SITE}/beartype/_conf/conftest.py"
OWN = b"from beartype._conf.confcls import BeartypeConf\n"
PLANTED = b"def pytest_runtest_makereport(item, call):\n    pass\n"


def _record_line(rel: str, data: bytes) -> bytes:
    return f"{rel},{_hook_cleanup.record_digest(data)},{len(data)}\n".encode()


def _tree() -> dict[str, bytes]:
    return {
        # An installed package's own file, listed by its RECORD.
        OWN_PATH: OWN,
        f"{SITE}/beartype-0.21.0.dist-info/RECORD": (
            b"beartype/__init__.py,sha256=AAAA,10\n"
            + _record_line("beartype/_conf/conftest.py", OWN)
        ),
        # A package's own file the agent edited: its RECORD names other bytes.
        "/usr/lib/python3/dist-packages/jeepney/tests/conftest.py": PLANTED,
        "/usr/lib/python3/dist-packages/jeepney-0.8.0.dist-info/RECORD": (
            _record_line("jeepney/tests/conftest.py", b"honest\n")
        ),
        # Planted in a site folder, listed by no RECORD.
        "/usr/local/lib/python3.12/site-packages/conftest.py": PLANTED,
        # Planted outside any site folder.
        "/srv/conftest.py": PLANTED,
        # Planted in the workspace, with a RECORD forged to list it.
        "/app/lib/python3.12/site-packages/evil/conftest.py": PLANTED,
        "/app/lib/python3.12/site-packages/evil-1.0.dist-info/RECORD": (
            _record_line("evil/conftest.py", PLANTED)
        ),
    }


REMOVED = (
    "/usr/lib/python3/dist-packages/jeepney/tests/conftest.py",
    "/usr/local/lib/python3.12/site-packages/conftest.py",
    "/srv/conftest.py",
    "/app/lib/python3.12/site-packages/evil/conftest.py",
)


def test_record_digest_is_the_record_format():
    """RECORD lines hold urlsafe base64 sha256 digests without padding."""
    digest = _hook_cleanup.record_digest(b"")
    assert digest == "sha256=47DEQpj8HBSa-_TImW-5JCeuQeRkm5NMpJWZG3hSuFU"
    lines = _hook_cleanup.record_lines(
        b'a/conftest.py,sha256=x,3\n"b,c/conftest.py",sha256=y,4\nc.pyc,,\n'
    )
    assert lines == {"a/conftest.py": {"sha256=x"}, "b,c/conftest.py": {"sha256=y"}}
    assert _hook_cleanup.site_folder_of(OWN_PATH) == (
        SITE,
        "beartype/_conf/conftest.py",
    )
    assert _hook_cleanup.site_folder_of("/srv/conftest.py") is None


@pytest.mark.asyncio
async def test_bfx_form_keeps_a_packages_own_file_and_removes_planted_ones():
    fs = _FakeFs(
        {
            **{path.encode(): data for path, data in _tree().items()},
            b"/tests/conftest.py": b"# the task's own\n",
        }
    )
    loop = asyncio.get_running_loop()
    await _hook_cleanup.run(
        fs.call,
        _hook_cleanup.parse_inputs(INPUTS.replace("a:b", "")),
        cleanup_conftests=True,
        workspace="/app",
        workspace_aliases=(),
        verifier_dir="/opt/runtime/verifier",
        deadline=loop.time() + 30,
    )
    assert fs.tree[OWN_PATH.encode()] == OWN
    assert b"/tests/conftest.py" in fs.tree
    for path in REMOVED:
        assert path.encode() not in fs.tree, path


def _materialize(root: Path) -> dict[str, Path]:
    made = {}
    for path, data in _tree().items():
        real = root / path.lstrip("/")
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_bytes(data)
        made[path] = real
    return made


def test_shell_form_script_applies_the_same_rule(tmp_path):
    made = _materialize(tmp_path)
    conftests = [p for p in made.values() if p.name == "conftest.py"]
    ran = subprocess.run(
        [
            sys.executable,
            *FLAGS,
            "-c",
            _script(lockdown._CONFTEST_SWEEP_SCRIPT),
            str(tmp_path / "app"),
        ],
        input=b"\0".join(os.fsencode(p) for p in conftests) + b"\0",
        capture_output=True,
        cwd="/",
    )
    assert ran.returncode == 0, ran.stderr
    assert made[OWN_PATH].read_bytes() == OWN
    for path in REMOVED:
        assert not made[path].exists(), path


def _sweep_under(root: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    """The real sweep command, walking *root* instead of ``/``."""
    command = lockdown._conftest_sweep_cmd(workspace=str(root / "app"))
    assert command.count("find / -path /proc") == 2
    command = command.replace("find / -path /proc", f"find {root} -path /proc")
    return subprocess.run(
        ["sh", "-c", command], capture_output=True, text=True, env=env, cwd="/"
    )


@pytest.mark.skipif(os.path.exists(KIT_RECORD), reason="a kit is installed here")
def test_the_sweep_keeps_a_packages_own_file_with_a_judged_python(tmp_path):
    made = _materialize(tmp_path)
    env = {"PATH": "/usr/bin:/bin", "__bf_root_python": sys.executable}
    ran = _sweep_under(tmp_path, env)
    assert ran.returncode == 0, ran.stderr
    assert made[OWN_PATH].read_bytes() == OWN
    for path in REMOVED:
        assert not made[path].exists(), path


@pytest.mark.skipif(os.path.exists(KIT_RECORD), reason="a kit is installed here")
def test_without_a_protected_python_every_conftest_still_goes(tmp_path):
    """Fail closed: no protected Python (127) means the plain sweep, which
    removes the package's own file too, as before this change."""
    made = _materialize(tmp_path)
    ran = _sweep_under(tmp_path, {"PATH": "/usr/bin:/bin"})
    assert ran.returncode == 0, ran.stderr
    assert not made[OWN_PATH].exists()
    for path in REMOVED:
        assert not made[path].exists(), path
