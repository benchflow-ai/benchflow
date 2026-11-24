"""Nightly and release-line versions (tools/release_version.py).

Between releases a line carries the next candidate with ``.dev0``
(``0.10.1rc1.dev0``, docs/contributing/release.md). Before this change the
preview computation raised on such a version instead of publishing or
skipping it, so a preview run on such a commit went red. Nightlies are
date-stamped (``X.Y.Z.devYYYYMMDD``): they sort after the last release and
before the next one, and ``pip install benchflow`` never takes them.
"""

from __future__ import annotations

import datetime as dt

import pytest
from packaging.version import Version

from tools.release_version import (
    InternalPreviewDecision,
    compute_internal_preview_version,
    compute_nightly_version,
    main,
)

DAY = dt.date(2026, 11, 18)


@pytest.mark.parametrize(
    ("version", "run_number", "expected"),
    [
        ("0.10.1rc1.dev0", 7, "0.10.1rc1.dev7"),
        ("0.11.0a2.dev0", "12", "0.11.0a2.dev12"),
    ],
)
def test_a_candidates_dev_line_publishes_a_preview(version, run_number, expected):
    assert compute_internal_preview_version(version, run_number) == (
        InternalPreviewDecision(publish=True, version=expected)
    )


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.10.1.dev0", "0.10.1.dev20261118"),
        ("0.10.1rc1.dev0", "0.10.1rc1.dev20261118"),
        ("1.0.dev3", "1.0.dev20261118"),
    ],
)
def test_the_nightly_is_date_stamped(version, expected):
    decision = compute_nightly_version(version, DAY)
    assert decision == InternalPreviewDecision(publish=True, version=expected)
    nightly = Version(decision.version)
    assert nightly.is_prerelease
    assert nightly < Version(Version(version).base_version)


@pytest.mark.parametrize("version", ["0.10.0", "0.10.0rc6", "0.9.3"])
def test_a_release_commit_publishes_no_nightly(version):
    assert compute_nightly_version(version, DAY) == InternalPreviewDecision(
        publish=False
    )


def test_a_nightly_sorts_after_the_release_before_it():
    assert Version("0.10.0") < Version("0.10.1.dev20261118") < Version("0.10.1rc1")


def test_the_nightly_command_writes_its_outputs(tmp_path, monkeypatch, capsys):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "0.10.1.dev0"\n')
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    assert main(["nightly", "--pyproject", str(pyproject), "--date", "2026-11-18"]) in (
        0,
        None,
    )
    out = capsys.readouterr().out
    assert "publish=true" in out and "version=0.10.1.dev20261118" in out
