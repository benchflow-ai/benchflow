"""The pytest plugin guard runs on the oldest Python a registered task's pytest uses, and its own failure is never a refusal.

The guard is imported by the task's own pytest, in the task's own Python.
Among the registered SWE-family tasks whose test.sh runs pytest, the oldest
Python is 3.6: SWE-bench Verified's astropy 1.3 and scikit-learn 0.20-0.22 (29
tasks, by the harness's specs; the images agree). Django's 3.5 tasks run
runtests.py, which never loads the guard.

``zip(strict=True)`` (Python 3.10) in the byte-code check raised TypeError on
3.6. The check caught it as "not this source's compilation", so astropy-7166's
own plugins were refused and its oracle scored 0. The guard now needs nothing
newer than 3.6. A failure of the guard's own code is a ``GuardError``: it
stops pytest through the ``crashed`` record, naming the file, so the run is
not scored and never a refusal's 0. Bytes an agent can plant still make the
file checks answer "not trusted".
"""

from __future__ import annotations

import ast
import importlib.util
import marshal
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow.sandbox import _pytest_plugin_guard as guard
from benchflow.sandbox import lockdown
from tests.exec_diet_fakes import docker_available

GUARD = Path(guard.__file__)
FLOOR = (3, 6)
FLOOR_IMAGE = "python:3.6-slim"


def test_the_guard_parses_at_the_floor():
    ast.parse(GUARD.read_text(), feature_version=FLOOR)
    # And what ships (no comments, no docstrings), which the sandbox runs.
    ast.parse(lockdown._DISCOVER_PYTEST_PLUGINS_SCRIPT, feature_version=FLOOR)


def test_the_guard_needs_nothing_newer_than_the_floor():
    """Library features too (``zip(strict=)``, ``str.removeprefix``,
    ``importlib.metadata`` without its backport), which no parse catches. A
    use behind its own version check carries ``# novermin``."""
    done = subprocess.run(
        [
            str(Path(sys.executable).with_name("vermin")),
            f"-t={FLOOR[0]}.{FLOOR[1]}-",
            "--violations",
            "--no-tips",
            str(GUARD),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"Minimum required versions: {FLOOR[0]}.{FLOOR[1]}" in done.stdout, (
        done.stdout
    )


def _cache(tmp_path: Path, text: str, name: str = "mod") -> tuple[Path, Path]:
    source = tmp_path / f"{name}.py"
    source.write_text(text)
    pyc = Path(importlib.util.cache_from_source(str(source)))
    py_compile.compile(str(source), cfile=str(pyc), doraise=True)
    return source, pyc


def _header() -> int:
    return 16 if sys.version_info >= (3, 7) else 12


def test_a_cache_of_the_source_is_its_compilation(tmp_path):
    source, pyc = _cache(tmp_path, "def f():\n    return [x for x in (1, 2)]\n")
    assert guard._pyc_matches(str(pyc), str(source), 0) is True


def test_a_cache_of_other_code_is_not(tmp_path):
    source, pyc = _cache(tmp_path, "def f():\n    return 1\n")
    _, other_pyc = _cache(tmp_path, "def f():\n    return 2\n", name="other")
    pyc.write_bytes(other_pyc.read_bytes())
    assert guard._pyc_matches(str(pyc), str(source), 0) is False


@pytest.mark.parametrize(
    "payload",
    [b"\xff\xfe garbage", marshal.dumps(7), marshal.dumps(("not", "code")), b""],
    ids=["not-marshal", "an-int", "a-tuple", "empty"],
)
def test_bytes_an_agent_planted_stay_untrusted(tmp_path, payload):
    """Whatever a planted cache holds is 'not trusted', never the guard's failure."""
    source, pyc = _cache(tmp_path, "x = 1\n")
    pyc.write_bytes(pyc.read_bytes()[: _header()] + payload)
    assert guard._pyc_matches(str(pyc), str(source), 0) is False


def test_an_unreadable_source_is_untrusted(tmp_path):
    source, pyc = _cache(tmp_path, "x = 1\n")
    source.unlink()
    assert guard._pyc_matches(str(pyc), str(source), 0) is False


def test_the_guards_own_failure_is_a_guard_error(tmp_path, monkeypatch):
    source, pyc = _cache(tmp_path, "x = 1\n")

    def broken(cached, compiled):
        raise TypeError("zip() got an unexpected keyword argument 'strict'")

    monkeypatch.setattr(guard, "_same_code", broken)
    with pytest.raises(guard.GuardError) as raised:
        guard._pyc_matches(str(pyc), str(source), 0)
    assert str(raised.value) == (
        "guard_error: TypeError: zip() got an unexpected keyword argument "
        f"'strict' while judging {pyc}"
    )


def test_a_module_judgment_that_fails_is_a_guard_error(monkeypatch):
    def broken(name, blocked):
        raise AttributeError("'NoneType' object has no attribute 'origin'")

    monkeypatch.setattr(guard, "trusted_module", broken)
    with pytest.raises(guard.GuardError) as raised:
        guard._judged_module("astropy.tests.plugins.display", ())
    assert str(raised.value) == (
        "guard_error: AttributeError: 'NoneType' object has no attribute "
        "'origin' while judging astropy.tests.plugins.display"
    )


def test_a_guard_error_is_never_a_refusal(monkeypatch):
    """_validate lets it through, and the hook records ``crashed`` (with a
    ``guard_error`` verdict), not ``refused``."""
    error = guard.GuardError("guard_error: TypeError: forced while judging pkg.mod")

    def broken(name, blocked):
        raise error

    monkeypatch.setattr(guard, "trusted_module", broken)
    with pytest.raises(guard.GuardError):
        guard._validate(["pkg.mod"], entry_points=False)
    marks = []
    monkeypatch.setattr(guard, "_mark", lambda kind, detail="": marks.append(kind))
    monkeypatch.setattr(guard, "_unmark", lambda kind: None)
    guard._stopping(error)
    assert marks == ["verdicts", "crashed"]
    guard._stopping(guard.Rejected("Verifier plugin trust rejected: pkg.mod (x)"))
    assert marks == ["verdicts", "crashed", "refused"]


def _shipped(tmp_path: Path) -> Path:
    """A folder holding the guard as it ships (lockdown's script), under the
    file's name: the sandbox never runs the source file itself."""
    folder = tmp_path / "shipped-guard"
    folder.mkdir()
    (folder / GUARD.name).write_text(lockdown._DISCOVER_PYTEST_PLUGINS_SCRIPT)
    return folder


@pytest.fixture
def floor_python():
    if not docker_available():
        pytest.skip("needs a Docker daemon")
    pulled = subprocess.run(
        ["docker", "pull", "-q", FLOOR_IMAGE], capture_output=True, text=True
    )
    if pulled.returncode != 0:
        pytest.skip(f"cannot pull {FLOOR_IMAGE}: {pulled.stderr.strip()[:200]}")
    return FLOOR_IMAGE


_AT_THE_FLOOR = """
import importlib.util, py_compile, sys, os
assert sys.version_info[:2] == (3, 6), sys.version
spec = importlib.util.spec_from_file_location("guard", "/guard/_pytest_plugin_guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
os.makedirs("/work", exist_ok=True)
with open("/work/mod.py", "w") as handle:
    handle.write("def f():\\n    return [x for x in (1, 2)]\\n")
pyc = importlib.util.cache_from_source("/work/mod.py")
py_compile.compile("/work/mod.py", cfile=pyc, doraise=True)
print("own", guard._pyc_matches(pyc, "/work/mod.py", 0))
with open("/work/other.py", "w") as handle:
    handle.write("def f():\\n    return 2\\n")
other = importlib.util.cache_from_source("/work/other.py")
py_compile.compile("/work/other.py", cfile=other, doraise=True)
with open(other, "rb") as left, open(pyc, "wb") as right:
    right.write(left.read())
print("planted", guard._pyc_matches(pyc, "/work/mod.py", 0))
"""


def test_the_guard_judges_byte_code_on_python_3_6(floor_python, tmp_path):
    """A real Python 3.6: the guard imports, a cache of its own source is the
    source's compilation (the 3.6 TypeError called it foreign), and another
    module's cache is not."""
    done = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{_shipped(tmp_path)}:/guard:ro",
            floor_python,
            "python",
            "-c",
            _AT_THE_FLOOR,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split("\n")[:-1] == ["own True", "planted False"], done.stdout


async def test_a_guard_failure_inside_pytest_is_recorded_never_refused(
    tmp_path, monkeypatch
):
    """The armed guard in a real pytest, its file check made to fail. It
    leaves the ``crashed`` record, naming the failure and the module, and no
    ``refused`` one: never a refusal. How the verifier scores a crashed guard
    (today a 0 it cannot trust, unscored with the guard_error rule) reads that
    record."""
    from tests.test_verifier_plugin_guard_issues import (
        IMAGE_FIXTURES,
        USES_IMAGE_FIXTURE,
    )
    from tests.test_verifier_uv_state import GUARD_OWNERSHIP, score, uvx

    def image(layout):
        package = layout.workspace / "imgpkg" / "testing"
        package.mkdir(parents=True)
        (layout.workspace / "imgpkg/__init__.py").write_text("")
        (package / "__init__.py").write_text("")
        (package / "fixtures.py").write_text(IMAGE_FIXTURES)

    def image_conftest(env, layout):
        (layout.tmp / "verifier-tests" / "conftest.py").write_text(
            "pytest_plugins = 'imgpkg.testing.fixtures'\n"
        )
        return [*uvx(env, layout), layout.workspace]

    await score(
        tmp_path,
        monkeypatch,
        USES_IMAGE_FIXTURE,
        image_conftest,
        before=image,
        workspace_name="testbed",
        guard_model=GUARD_OWNERSHIP
        + "def _untrusted_reason(path, blocked):\n"
        + "    raise TypeError('forced by the test')\n",
    )

    crashed = sorted(tmp_path.rglob("*_guard_*.crashed"))
    assert crashed, sorted(p.name for p in tmp_path.rglob("*_guard_*"))
    record = crashed[0].read_text()
    # The first plugin file judged names it (here pytest's own, registered
    # first): every judgment fails in this model.
    assert (
        "GuardError: guard_error: TypeError: forced by the test while judging /"
    ) in record, record
    assert not list(tmp_path.rglob("*_guard_*.refused"))
