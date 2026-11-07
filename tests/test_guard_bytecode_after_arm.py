"""The plugin guard and byte-code caches Python writes after the arm.

Terminal-Bench 4.1's verifier image ships its packages with no byte code, and
its test.sh imports them with ``python3 -I``, which ignores
PYTHONDONTWRITEBYTECODE. The first import writes ``anyio/__pycache__``, which
changed the package folder after the guard was armed, so the guard refused
the image's own ``anyio`` plugin and the reference solution scored 0. A
folder whose only change is caches that compile from its own sources is
unchanged; anything else that changed still is.
"""

from __future__ import annotations

import importlib.util
import marshal
import os
import py_compile
import sys
import time
from pathlib import Path

import pytest

from benchflow.sandbox import _pytest_plugin_guard as guard

TAG = sys.implementation.cache_tag


def _package(root: Path) -> Path:
    pkg = root / "site-packages" / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("from pkg import mod\n")
    (pkg / "mod.py").write_text("VALUE = 1\n")
    (pkg / "sub" / "__init__.py").write_text("SUB = 2\n")
    return pkg


@pytest.fixture
def armed(tmp_path, monkeypatch):
    """A package made before the arm, then the guard's key file (the arm)."""
    pkg = _package(tmp_path)
    time.sleep(0.05)
    key = tmp_path / "key"
    key.write_text("k")
    monkeypatch.setattr(guard, "_BENCHFLOW_KEY_FILE", str(key))
    time.sleep(0.05)
    return pkg


def _cache(source: Path, optimize: int = -1) -> Path:
    target = Path(
        importlib.util.cache_from_source(
            str(source), optimization="" if optimize < 1 else optimize
        )
    )
    py_compile.compile(str(source), cfile=str(target), doraise=True, optimize=optimize)
    return target


def test_caches_python_wrote_from_the_sources_leave_the_folder_unchanged(armed):
    for source in (
        armed / "__init__.py",
        armed / "mod.py",
        armed / "sub" / "__init__.py",
    ):
        _cache(source)
    _cache(armed / "mod.py", optimize=1)
    assert guard._after_arm(str(armed)) is False


def test_a_cache_that_runs_other_code_is_a_change(armed):
    pyc = _cache(armed / "mod.py")
    data = pyc.read_bytes()
    forged = compile("VALUE = 2\n", str(armed / "mod.py"), "exec", dont_inherit=True)
    pyc.write_bytes(data[:16] + marshal.dumps(forged))
    assert guard._after_arm(str(armed)) is True


def test_a_module_added_after_the_arm_is_a_change(armed):
    _cache(armed / "mod.py")
    (armed / "planted.py").write_text("import os\n")
    assert guard._after_arm(str(armed)) is True


def test_a_module_added_below_a_cached_subpackage_is_a_change(armed):
    _cache(armed / "sub" / "__init__.py")
    _cache(armed / "mod.py")
    (armed / "sub" / "planted.py").write_text("import os\n")
    assert guard._after_arm(str(armed)) is True


def test_a_pytest_rewrite_cache_is_a_change(armed):
    pyc = _cache(armed / "mod.py")
    rewrite = pyc.with_name(f"mod.{TAG}-pytest-8.4.1.pyc")
    rewrite.write_bytes(pyc.read_bytes())
    assert guard._after_arm(str(armed)) is True


def test_a_cache_folder_that_is_a_link_is_a_change(armed, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.symlink(elsewhere, armed / "__pycache__")
    assert guard._after_arm(str(armed)) is True


def test_a_folder_changed_with_no_new_caches_is_a_change(armed):
    (armed / "mod.py").unlink()
    assert guard._after_arm(str(armed)) is True


def test_a_file_changed_after_the_arm_is_still_a_change(armed):
    (armed / "mod.py").write_text("VALUE = 3\n")
    assert guard._after_arm(str(armed / "mod.py")) is True


def test_an_unarmed_guard_trusts_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "_BENCHFLOW_KEY_FILE", "")
    assert guard._after_arm(str(_package(tmp_path))) is True


def test_the_armed_guard_writes_and_reads_no_byte_code(armed, tmp_path, monkeypatch):
    """`python3 -I` leaves PYTHONDONTWRITEBYTECODE unapplied, so the armed
    guard applies it: pytest's rewriter writes no cache the guard would then
    refuse. Caches are read only under a folder that is never made, in the
    guard's own folder, never where the environment points: under -I the
    variable is unapplied on purpose, and its folder may be the agent's."""
    agent_folder = tmp_path / "app" / "bytecache"
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.setattr(sys, "pycache_prefix", None)
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(agent_folder))
    guard._keep_byte_code_out()
    assert sys.dont_write_bytecode is True
    assert sys.pycache_prefix == str(tmp_path / ".no-pycache")
    assert not os.path.lexists(sys.pycache_prefix)


def test_a_no_cache_folder_that_exists_is_refused(armed, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "pycache_prefix", None)
    (tmp_path / ".no-pycache").mkdir()
    with pytest.raises(guard.Rejected, match="never made"):
        guard._keep_byte_code_out()
    assert sys.pycache_prefix is None


SOURCE = (
    "def allowed():\n    return 'allowed'\n\n"
    "def denied():\n    return 'denied'\n\n"
    "def run():\n    return allowed()\n"
)


def _specialized(name):
    import dis
    import opcode

    for table in (
        getattr(dis, "_all_opmap", None),
        getattr(opcode, "_specialized_opmap", None),
    ):
        if table and name in table:
            return table[name]
    return None


@pytest.mark.skipif(sys.version_info < (3, 11), reason="no specialized instructions")
def test_a_cache_that_compares_equal_but_runs_other_code_is_not_the_source(tmp_path):
    """Code objects compare with specializations undone: a cache whose global
    load is specialized, with its index pointing at another global, is
    ``==`` the compilation of its source yet would call ``denied()``."""
    import dis

    special = _specialized("LOAD_GLOBAL_MODULE")
    if special is None:
        pytest.skip("this Python names no LOAD_GLOBAL_MODULE")
    source = tmp_path / "victim.py"
    source.write_text(SOURCE)
    pyc = _cache(source)
    data = bytearray(pyc.read_bytes())
    compiled = compile(SOURCE, str(source), "exec", dont_inherit=True)
    raw = next(
        c
        for c in marshal.loads(bytes(data[16:])).co_consts
        if getattr(c, "co_name", "") == "run"
    ).co_code
    base = bytes(data).find(raw, 16)
    at = next(i for i in range(0, len(raw), 2) if raw[i] == dis.opmap["LOAD_GLOBAL"])
    data[base + at] = special
    slot = base + at + 2 + 2 * (1 if sys.version_info < (3, 13) else 3)
    data[slot : slot + 2] = (9).to_bytes(2, "little")
    pyc.write_bytes(bytes(data))
    if marshal.loads(bytes(data[16:])) != compiled:
        pytest.skip("on this Python the forged cache does not compare equal")
    assert guard._pyc_matches(str(pyc), str(source), -1) is False
    assert guard._pyc_matches(str(_cache(source)), str(source), -1) is True


def test_a_cache_folder_renamed_into_place_is_a_change(tmp_path, monkeypatch):
    """Caches made before the arm in another folder, renamed into the
    package after it: every cache in a cache folder made after the arm must
    be the compilation of the source beside it, however old."""
    pkg = _package(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    other = tmp_path / "other" / "mod.py"
    other.parent.mkdir()
    other.write_text("VALUE = 2\n")
    py_compile.compile(
        str(other), cfile=str(elsewhere / f"mod.{TAG}.pyc"), doraise=True
    )
    time.sleep(0.05)
    key = tmp_path / "key"
    key.write_text("k")
    monkeypatch.setattr(guard, "_BENCHFLOW_KEY_FILE", str(key))
    time.sleep(0.05)
    os.rename(elsewhere, pkg / "__pycache__")
    if os.lstat(pkg / "__pycache__").st_ctime < os.stat(key).st_ctime:
        pytest.skip("this filesystem keeps a renamed folder's change time")
    assert guard._after_arm(str(pkg)) is True
