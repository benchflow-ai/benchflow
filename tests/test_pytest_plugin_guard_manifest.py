"""The pytest plugin guard trusts a file whose bytes the image manifest records.

A file the agent's sandbox or its transfer could have written stays
distrusted (the separate verifier sandbox distrusts the whole transferred
workspace). There is one exception: the image manifest, recorded before the
agent's turn or the transfer and written back by BenchFlow before the guard
is armed, records that path with the same size and sha256. So a repository's
own plugin loads, such as the ``sphinx.testing.fixtures`` a Sphinx conftest.py
declares. An edited file is ``changed`` and a planted one ``new``, and both are
refused. Without a manifest the guard trusts no such file, as before. A
byte-code cache the manifest records is the image's even when it is not its
source's compilation (a stale one the image shipped). The source is judged by
itself, so an edited source behind the image's cache is still refused. Every
refusal leaves a ``verdicts`` record with what the manifest says of each
refused file.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import py_compile
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _image_manifest, lockdown
from benchflow.sandbox import _pytest_plugin_guard as guard

PYTHON = ".".join(str(part) for part in sys.version_info[:3])


def _record(manifest: Path, files: list[Path]) -> None:
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "taken": "pre-transfer",
                "files": {
                    str(p): {
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        "size": p.stat().st_size,
                    }
                    for p in files
                },
            }
        )
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    workspace = tmp_path / "testbed"
    (workspace / "pkg/testing").mkdir(parents=True)
    (workspace / "pkg/__init__.py").write_text("")
    (workspace / "pkg/testing/__init__.py").write_text("")
    (workspace / "pkg/testing/fixtures.py").write_text("# the image's\n")
    folder = tmp_path / "var-lib-benchflow"
    folder.mkdir()
    manifest = folder / "image-manifest.json"
    _record(manifest, sorted(workspace.rglob("*.py")))
    monkeypatch.setattr(guard, "_BENCHFLOW_ARMED", True)
    monkeypatch.setattr(guard, "_BENCHFLOW_OWNERSHIP", False)
    monkeypatch.setattr(guard, "_BENCHFLOW_MANIFEST", str(manifest))
    monkeypatch.setattr(guard, "_MANIFEST", [])
    # pytest's own temporary folders are under /tmp, which the manifest
    # prunes in a real image; here they stand for the workspace.
    monkeypatch.setattr(guard, "_MANIFEST_PRUNED", ())
    # Nothing here is root: the folder and file stand for ones only root reaches.
    monkeypatch.setattr(guard, "_private_to_root", lambda d: Path(d).is_dir())
    monkeypatch.setattr(guard, "_private_file", lambda p: Path(p).is_file())
    monkeypatch.setattr(sys, "path", [str(workspace), *sys.path])
    blocked = lockdown._separate_verifier_blocked_prefixes((str(workspace),))
    return SimpleNamespace(
        workspace=workspace, manifest=manifest, blocked=blocked, tmp=tmp_path
    )


def _why(repo, relative: str) -> str:
    path = f"{repo.workspace}/{relative}"
    return f"{path} is under {repo.workspace}, " + guard._blocked_why()


def test_a_module_the_manifest_records_loads(repo):
    assert guard._manifest_verdict(str(repo.workspace / "pkg/testing/fixtures.py")) == (
        "identical"
    )
    assert guard.trusted_module("pkg.testing.fixtures", repo.blocked)


def test_an_edited_image_module_is_changed_and_refused(repo):
    target = repo.workspace / "pkg/testing/fixtures.py"
    target.write_text("# the agent's\n")
    assert guard._manifest_verdict(str(target)) == "changed"
    assert not guard.trusted_module("pkg.testing.fixtures", repo.blocked)
    assert guard.untrusted_reason(str(target), repo.blocked) == _why(
        repo, "pkg/testing/fixtures.py"
    )


def test_an_edit_of_the_same_size_is_changed(repo):
    target = repo.workspace / "pkg/testing/fixtures.py"
    target.write_text("# the agent!\n")
    assert guard._manifest_verdict(str(target)) == "changed"


def test_a_planted_module_is_new_and_refused(repo):
    # A package the agent planted is what Python imports in place of the
    # image's fixtures.py; neither it nor its folder is in the manifest.
    planted = repo.workspace / "pkg/testing/fixtures"
    planted.mkdir()
    (planted / "__init__.py").write_text("# planted\n")
    assert guard._manifest_verdict(str(planted / "__init__.py")) == "new"
    assert guard._manifest_verdict(str(planted)) == "new"
    assert not guard.trusted_module("pkg.testing.fixtures", repo.blocked)


def test_a_link_to_an_image_file_is_not_the_image_file(repo):
    link = repo.workspace / "pkg/testing/linked.py"
    link.symlink_to(repo.workspace / "pkg/testing/fixtures.py")
    assert guard._manifest_verdict(str(link)) == "new"


@pytest.mark.parametrize("how", ["missing", "unset", "not-root", "bad-schema"])
def test_without_a_manifest_no_workspace_module_loads(repo, monkeypatch, how):
    if how == "missing":
        repo.manifest.unlink()
    elif how == "unset":
        monkeypatch.setattr(guard, "_BENCHFLOW_MANIFEST", "")
    elif how == "not-root":
        monkeypatch.setattr(guard, "_private_file", lambda p: False)
    else:
        repo.manifest.write_text(json.dumps({"version": 2, "files": {}}))
    assert guard._manifest_verdict(str(repo.workspace / "pkg/testing/fixtures.py")) is (
        None
    )
    assert not guard.trusted_module("pkg.testing.fixtures", repo.blocked)


def _stale_cache(repo) -> tuple[Path, Path]:
    """A cache the image shipped of an older fixtures.py: Python uses it (its
    header names the source's size and mtime), but it is not the source's
    compilation."""
    source = repo.workspace / "pkg/testing/fixtures.py"
    source.write_text("VALUE = 1  # old\n")
    pyc = Path(importlib.util.cache_from_source(str(source)))
    py_compile.compile(str(source), cfile=str(pyc), doraise=True)
    stamp = source.stat()
    source.write_text("VALUE = 2  # new\n")
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert not guard._pyc_matches(str(pyc), str(source), 0)
    return source, pyc


def test_a_stale_cache_the_image_shipped_is_the_images(repo):
    _, pyc = _stale_cache(repo)
    _record(repo.manifest, [*sorted(repo.workspace.rglob("*.py")), pyc])
    assert guard._manifest_verdict(str(pyc)) == "identical"
    assert guard.trusted_module("pkg.testing.fixtures", repo.blocked)


def test_the_images_cache_does_not_vouch_for_an_edited_source(repo):
    source, pyc = _stale_cache(repo)
    _record(repo.manifest, [*sorted(repo.workspace.rglob("*.py")), pyc])
    source.write_text("VALUE = 3  # agent\n")
    assert guard._manifest_verdict(str(pyc)) == "identical"
    assert not guard.trusted_module("pkg.testing.fixtures", repo.blocked)


def test_a_planted_cache_is_refused(repo):
    source, pyc = _stale_cache(repo)
    # The image's source, and a cache the image never had.
    _record(repo.manifest, sorted(repo.workspace.rglob("*.py")))
    guard._MANIFEST.clear()
    assert guard._manifest_verdict(str(source)) == "identical"
    assert guard._manifest_verdict(str(pyc)) == "new"
    assert not guard.trusted_module("pkg.testing.fixtures", repo.blocked)
    assert guard.untrusted_reason(str(source), repo.blocked) == (
        f"{source} has byte code Python or pytest would run in its place that "
        "is not its source's compilation"
    )


def _pairs(entries):
    return [(e["path"], e["verdict"]) for e in entries]


def _fields_ok(entries, kind="rejected"):
    for e in entries:
        assert e["v"] == 1 and e["guard"] == guard.__name__, e
        assert e["kind"] == kind and e["python"] == PYTHON, e
        path = Path(e["path"])
        if path.is_file() and not path.is_symlink():
            assert e["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), e
        else:
            assert e["sha256"] == "", e


def test_a_refusal_records_each_files_verdict(repo, monkeypatch):
    marks = []
    monkeypatch.setattr(
        guard, "_mark", lambda kind, detail="": marks.append((kind, detail))
    )
    (repo.workspace / "pkg/testing/fixtures.py").write_text("# the agent's\n")
    with pytest.raises(guard.Rejected) as raised:
        guard._validate(["pkg.testing.fixtures"], entry_points=False)
    (verdicts,) = [json.loads(d) for k, d in marks if k == "verdicts"]
    _fields_ok(verdicts)
    assert {e["why"] for e in verdicts} == {str(raised.value)[:300]}
    w = repo.workspace
    assert _pairs(verdicts) == [
        (str(w / "pkg/__init__.py"), "identical"),
        (str(w / "pkg"), "identical"),
        (str(w / "pkg/testing/__init__.py"), "identical"),
        (str(w / "pkg/testing"), "identical"),
        (str(w / "pkg/testing/fixtures.py"), "changed"),
    ]


def test_without_a_manifest_the_record_says_so(repo, monkeypatch):
    marks = []
    monkeypatch.setattr(
        guard, "_mark", lambda kind, detail="": marks.append((kind, detail))
    )
    monkeypatch.setattr(guard, "_BENCHFLOW_MANIFEST", "")
    target = str(repo.workspace / "pkg/testing/fixtures.py")
    guard._record_verdicts([target], "Verifier plugin trust rejected: x (y)")
    ((kind, detail),) = marks
    entries = json.loads(detail)
    assert kind == "verdicts"
    _fields_ok(entries)
    assert _pairs(entries) == [(target, "no-manifest")]
    assert entries[0]["why"] == "Verifier plugin trust rejected: x (y)"


@pytest.mark.parametrize(
    "path",
    [
        "/opt/conda/lib/python3.9/site-packages/plug.py",
        "/usr/lib/python3/dist-packages/plug.py",
        "/usr/lib/python3.12/plug.py",
        "/tmp/plug.py",
        "/root/.cache/plug.py",
        "/var/lib/benchflow-image/image-manifest.json",
        _image_manifest.VERIFIER_DIR + "/verifier.env",
    ],
)
def test_paths_the_manifest_prunes_are_no_manifest(repo, monkeypatch, path):
    monkeypatch.setattr(guard, "_MANIFEST_PRUNED", _image_manifest.PRUNED)
    assert guard._outside_manifest(path)
    assert guard._manifest_verdict(path) is None


def test_a_repository_folder_named_like_a_pruned_one_is_still_judged(repo):
    assert not guard._outside_manifest(str(repo.workspace / "pkgs/x.py"))
    assert not guard._outside_manifest(str(repo.workspace / "lib/pythonic.py"))


def test_the_guard_source_names_the_manifest():
    source = lockdown._pytest_plugin_guard_source(
        "_runtime_guard_test",
        ("/testbed",),
        [],
        ownership=False,
        manifest=_image_manifest.MANIFEST_PATH,
    )
    assert f"\n_BENCHFLOW_MANIFEST = {_image_manifest.MANIFEST_PATH!r}" in source
    # The pruned trees come from the manifest's own list, never a copy.
    assert f"\n_MANIFEST_PRUNED = {tuple(_image_manifest.PRUNED)!r}" in source
    assert "/var/lib/benchflow-image" in _image_manifest.PRUNED
    plain = lockdown._pytest_plugin_guard_source(
        "_runtime_guard_test", ("/testbed",), []
    )
    assert "\n_BENCHFLOW_MANIFEST = ''" in plain


def test_the_verdicts_record_is_read_back():
    assert "verdicts" in lockdown.PYTEST_PLUGIN_GUARD_MARKER_KINDS


def test_the_record_names_a_planted_cache(repo, monkeypatch):
    source, pyc = _stale_cache(repo)
    _record(repo.manifest, sorted(repo.workspace.rglob("*.py")))
    guard._MANIFEST.clear()
    marks = []
    monkeypatch.setattr(
        guard, "_mark", lambda kind, detail="": marks.append((kind, detail))
    )
    guard._record_verdicts([str(source)])
    entries = json.loads(marks[0][1])
    _fields_ok(entries)
    assert _pairs(entries) == [(str(source), "identical"), (str(pyc), "new")]
