"""The reference pages label what is new against what was actually published (tools/published_releases.json).

0.10.0rc5 went to PyPI on 2026-11-18 while the generator still compared with 0.10.0rc3, a hand-set constant: the
pages called rc4's and rc5's commands unreleased, and named rc3 as the current candidate. ("rc4 was tagged but
never published.") Now the candidate the labels compare with is the newest pre-release the record says PyPI
serves, no newer than this tree's version (pyproject.toml), and its baselines are pinned in the record by hash, so
a baseline and its version move together.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path

from packaging.version import Version

from tools import docs_errors as E
from tools import docs_reference as D

ROOT = Path(__file__).resolve().parents[1]
RECORD = json.loads((ROOT / "tools" / "published_releases.json").read_text())


def _version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def test_the_candidate_is_the_newest_published_one():
    current = Version(_version())
    published = [
        Version(v)
        for v in RECORD["releases"]
        if Version(v).is_prerelease and Version(v) <= current
    ]
    assert str(max(published)) == D.PRE_RELEASE
    assert _version() == D.CURRENT


def test_the_baselines_are_that_releases_pinned_by_hash():
    for path, field in (
        (D.PRE_BASELINE, "cli_baseline_sha256"),
        (D.PRE_NAMES_BASELINE, "names_baseline_sha256"),
    ):
        assert D.PRE_RELEASE in path.name, path
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == RECORD["releases"][D.PRE_RELEASE][field]
        )


def test_each_recorded_release_names_its_wheel_on_pypi():
    for version, entry in RECORD["releases"].items():
        assert entry["wheel"] == f"benchflow-{version}-py3-none-any.whl"
        assert len(entry["wheel_sha256"]) == 64


def test_a_newer_release_on_pypi_than_the_record_is_named():
    pypi = {
        "releases": {
            "0.10.0rc5": [{"filename": "benchflow-0.10.0rc5-py3-none-any.whl"}],
            "0.10.0rc6": [{"filename": "benchflow-0.10.0rc6-py3-none-any.whl"}],
            "0.10.1.dev20261118": [{"filename": "x.whl"}],
            "0.10.0rc4": [],
        }
    }
    assert D.newer_on_pypi(pypi, {"0.9.3", "0.10.0rc5"}) == ["0.10.0rc6"]
    assert D.newer_on_pypi(pypi, {"0.10.0rc5", "0.10.0rc6"}) == []


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_release_is_the_newest_published_final_one():
    """The labels said what 0.9.2 lacked after 0.9.3 shipped."""
    finals = [Version(v) for v in RECORD["releases"] if not Version(v).is_prerelease]
    assert str(max(finals)) == D.BASELINE_RELEASE
    for path, field in (
        (D.BASELINE, "cli_baseline_sha256"),
        (D.NAMES_BASELINE, "names_baseline_sha256"),
    ):
        assert D.BASELINE_RELEASE in path.name, path
        assert _sha256(path) == RECORD["releases"][D.BASELINE_RELEASE][field]


def test_the_error_page_compares_with_the_same_releases():
    """errors.md labelled its messages against 0.9.2 and 0.10.0rc3, set by hand, after rc5 shipped."""
    paths = E.baseline_paths()
    assert list(paths) == [D.BASELINE_RELEASE, D.PRE_RELEASE]
    for version, path in paths.items():
        assert _sha256(path) == RECORD["releases"][version]["errors_baseline_sha256"], (
            path
        )
    page = (ROOT / "docs" / "reference" / "errors.md").read_text()
    head = f"A message {D.BASELINE_RELEASE} doesn't print has `(0.10)` after it"
    assert head + ("." if D.RELEASING else f" when {D.PRE_RELEASE} prints it") in page


def test_the_notes_on_the_release_were_checked_on_it():
    """The hand-written notes on what the release does differently (its --dry-run, its trainer formats, its
    task.md keys) were checked on 0.9.2; with 0.9.3 the release, they would still name 0.9.2. Moving the release
    on fails here until they are checked again on the new one."""
    assert D.NOTES_CHECKED_ON == D.BASELINE_RELEASE
    notes = [*D._CHANGED_SINCE.values(), *D._RUNS_FROM.values()]
    finals = {
        v for note in notes for v in re.findall(r"(?<![\w.])\d+\.\d+\.\d+(?!\w)", note)
    }
    assert finals == {D.BASELINE_RELEASE}, finals


def test_a_final_version_being_cut_is_the_release():
    """At 0.10.0's cut the tree is a final release PyPI doesn't serve yet: the pages called 0.9.3 the current
    release and 0.10.0rc5 the candidate, and labelled what rc5 lacks `(unreleased)`, all of which ships."""
    current = Version(_version())
    finals = [Version(v) for v in RECORD["releases"] if not Version(v).is_prerelease]
    cutting = (
        not current.is_prerelease
        and not current.is_devrelease
        and current > max(finals)
    )
    assert D.RELEASING is cutting
    if not cutting:
        return
    pages = ("commands.md", "python.md", "cli-python.md", "files.md", "errors.md")
    for name in pages:
        assert "(unreleased)" not in (ROOT / "docs" / "reference" / name).read_text(), (
            name
        )
    changelog = (ROOT / "docs" / "versions" / "changelog.md").read_text()
    assert f"{D.CURRENT} is the current release" in changelog
    assert "current release candidate" not in changelog.split("<!--", 1)[0]
