"""Each place the plugin guard catches an error: what a file's bytes cause is
"not trusted", and a failure of the guard's own code is a GuardError.

An agent can plant any file the guard reads, so whatever those bytes make a
check raise (OSError, ValueError, EOFError, SyntaxError, UnicodeError,
RecursionError, MemoryError, OverflowError, and an ImportError where a module
is looked up) must stay a refusal. Anything else (a TypeError, AttributeError
or NameError from the guard's own code, say on a Python older than that code
expects) is a GuardError. It names the file and stops pytest through the
``crashed`` record, with a ``guard_error`` entry in the ``verdicts`` record,
never a refusal's 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from benchflow.sandbox import _pytest_plugin_guard as guard


def _raise(error):
    def broken(*args, **kwargs):
        raise error

    return broken


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    ws = tmp_path / "testbed"
    ws.mkdir()
    (ws / "plugin.py").write_text("# the image's\n")
    copy = tmp_path / "testbed_verify"
    copy.mkdir()
    (copy / "plugin.py").write_text("# the image's\n")
    monkeypatch.setattr(guard, "_BENCHFLOW_ARMED", True)
    monkeypatch.setattr(guard, "_BENCHFLOW_WORKSPACE", str(ws))
    monkeypatch.setattr(guard, "_BENCHFLOW_SNAPSHOT", str(copy))
    monkeypatch.setattr(guard, "_private_to_root", lambda directory: True)
    return ws


# untrusted_reason (every caller's way in)


def test_untrusted_reason_reads_a_files_error_as_untrusted(monkeypatch):
    monkeypatch.setattr(guard, "_untrusted_reason", _raise(OSError("EIO")))
    assert guard.untrusted_reason("/testbed/x.py", ()) == (
        "/testbed/x.py cannot be judged (OSError)"
    )


def test_untrusted_reason_turns_its_own_failure_into_a_guard_error(monkeypatch):
    monkeypatch.setattr(guard, "_untrusted_reason", _raise(TypeError("bug")))
    with pytest.raises(guard.GuardError) as raised:
        guard.untrusted_reason("/testbed/x.py", ())
    assert raised.value.path == "/testbed/x.py"
    assert raised.value.cause == "TypeError: bug"


# _image_bytes


def test_image_bytes_reads_a_missing_copy_as_not_the_images(workspace):
    (workspace.parent / "testbed_verify" / "plugin.py").unlink()
    assert guard._image_bytes(str(workspace / "plugin.py")) is False


def test_image_bytes_turns_its_own_failure_into_a_guard_error(workspace, monkeypatch):
    monkeypatch.setattr(guard, "_private_to_root", _raise(AttributeError("bug")))
    with pytest.raises(guard.GuardError) as raised:
        guard._image_bytes(str(workspace / "plugin.py"))
    assert raised.value.path == str(workspace / "plugin.py")


# _pyc_matches (the source's compilation; the comparison is in the floor tests)


def test_pyc_matches_turns_a_compile_failure_of_its_own_into_a_guard_error(
    tmp_path, monkeypatch
):
    import importlib.util
    import py_compile

    source = tmp_path / "m.py"
    source.write_text("x = 1\n")
    pyc = importlib.util.cache_from_source(str(source))
    py_compile.compile(str(source), cfile=pyc, doraise=True)
    monkeypatch.setattr(guard, "compile", _raise(TypeError("bug")), raising=False)
    with pytest.raises(guard.GuardError) as raised:
        guard._pyc_matches(pyc, str(source), 0)
    assert raised.value.path == pyc


def test_pyc_matches_reads_source_a_file_cannot_compile_as_untrusted(tmp_path):
    import importlib.util
    import py_compile

    source = tmp_path / "m.py"
    source.write_text("x = 1\n")
    pyc = importlib.util.cache_from_source(str(source))
    py_compile.compile(str(source), cfile=pyc, doraise=True)
    source.write_text("def (\n")
    assert guard._pyc_matches(pyc, str(source), 0) is False


# _caches_harmless


@pytest.mark.parametrize(
    ("error", "guard_error"),
    [(ValueError("bad optimization"), False), (TypeError("bug"), True)],
)
def test_caches_harmless(monkeypatch, error, guard_error):
    monkeypatch.setattr(guard.importlib.util, "cache_from_source", _raise(error))
    if guard_error:
        with pytest.raises(guard.GuardError) as raised:
            guard._caches_harmless("/testbed/x.py", "/testbed/x.py", ())
        assert raised.value.path == "/testbed/x.py"
    else:
        assert guard._caches_harmless("/testbed/x.py", "/testbed/x.py", ()) is False


# trusted_module's lookup, and _judged_module


class _Finder:
    def __init__(self, error):
        self.error = error

    def find_spec(self, *args, **kwargs):
        raise self.error


@pytest.mark.parametrize(
    ("error", "guard_error"),
    [
        (ImportError("planted"), False),
        (ValueError("x"), False),
        (AttributeError("bug"), True),
    ],
)
def test_trusted_module_lookup(monkeypatch, error, guard_error):
    monkeypatch.setattr(guard, "PathFinder", _Finder(error))
    if guard_error:
        with pytest.raises(guard.GuardError) as raised:
            guard.trusted_module("pkg.mod", ())
        assert raised.value.path == "pkg"
    else:
        assert guard.trusted_module("pkg.mod", ()) is False


def test_judged_module_reads_a_files_error_as_untrusted(monkeypatch):
    monkeypatch.setattr(guard, "trusted_module", _raise(ValueError("x")))
    assert guard._judged_module("pkg.mod", ()) is False


# The record helpers: whose problem a refusal is, never trust. A GuardError
# passes through them; a file's error leaves their answer at "no evidence".


@pytest.mark.parametrize(
    ("helper", "args", "quiet"),
    [
        (
            "_installed_by_verifier",
            ("/venv/lib/python3.12/site-packages/x.py",),
            False,
        ),
        ("_installed_during_verification", (["x"], False, (), ()), []),
        ("_unrecorded", (["x"], False, (), ()), None),
    ],
)
def test_record_helpers(monkeypatch, helper, args, quiet):
    monkeypatch.setattr(guard, "_record", _raise(ValueError("x")))
    assert getattr(guard, helper)(*args) == quiet
    monkeypatch.setattr(
        guard, "_record", _raise(guard._guard_error(TypeError("bug"), "/x"))
    )
    with pytest.raises(guard.GuardError):
        getattr(guard, helper)(*args)


def test_why_refused(monkeypatch):
    monkeypatch.setattr(guard, "_plugin_paths", _raise(ValueError("x")))
    assert guard._why_refused("x", False, (), ()) == "cannot inspect: ValueError"
    monkeypatch.setattr(
        guard, "_plugin_paths", _raise(guard._guard_error(TypeError("bug"), "/x"))
    )
    with pytest.raises(guard.GuardError):
        guard._why_refused("x", False, (), ())


# What the guard leaves for the verifier.


def test_a_guard_error_leaves_a_guard_error_verdict_and_a_crashed_record(monkeypatch):
    marks = []
    monkeypatch.setattr(
        guard, "_mark", lambda kind, detail="": marks.append((kind, detail))
    )
    monkeypatch.setattr(guard, "_unmark", lambda kind: None)
    error = guard._guard_error(
        TypeError("zip() does not take keyword arguments"),
        "/testbed/astropy/__init__.py",
    )
    guard._stopping(error)
    kinds = [kind for kind, _ in marks]
    assert kinds == ["verdicts", "crashed"]
    python = ".".join(str(part) for part in sys.version_info[:3])
    assert json.loads(marks[0][1]) == [
        {
            "v": 1,
            "guard": guard.__name__,
            "kind": "guard_error",
            "path": "/testbed/astropy/__init__.py",
            "sha256": "",
            "why": "TypeError: zip() does not take keyword arguments",
            "verdict": "guard_error",
            "python": python,
            "executable": sys.executable,
        }
    ]
    assert (
        "GuardError: guard_error: TypeError: zip() does not take keyword"
        in (marks[1][1])
    )


def test_every_entry_names_the_python_and_the_bytes(monkeypatch, tmp_path):
    marks = []
    monkeypatch.setattr(
        guard, "_mark", lambda kind, detail="": marks.append((kind, detail))
    )
    monkeypatch.setattr(guard, "_BENCHFLOW_MANIFEST", "")
    path = Path(tmp_path) / "x.py"
    path.write_text("x = 1\n")
    guard._record_verdicts([str(path)], "why")
    python = ".".join(str(part) for part in sys.version_info[:3])
    import hashlib

    assert json.loads(marks[0][1]) == [
        {
            "v": 1,
            "guard": guard.__name__,
            "kind": "rejected",
            "path": str(path),
            "sha256": hashlib.sha256(b"x = 1\n").hexdigest(),
            "why": "why",
            "verdict": "no-manifest",
            "python": python,
            "executable": sys.executable,
        }
    ]
