from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from tools.release_version import (
    CitationRelease,
    InternalPreviewDecision,
    ReleaseVersionError,
    compute_internal_preview_version,
    main,
    read_citation_release,
    validate_public_release_citation,
    validate_public_release_version,
)

CITATION_TEMPLATE = """cff-version: 1.2.0
message: "If you use benchflow in your research, please cite it as below."
title: "BenchFlow"
type: software
authors:
  - name: "BenchFlow team"
license: Apache-2.0
version: {version}
date-released: {date_released}
"""


def _citation(
    tmp_path: Path, *, version: str = "0.5.1", date_released: str = "2026-01-02"
) -> Path:
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        CITATION_TEMPLATE.format(version=version, date_released=date_released)
    )
    return citation


@pytest.mark.parametrize(
    ("version", "run_number", "expected"),
    [
        ("0.5.1.dev0", "123", "0.5.1.dev123"),
        ("0.5.1.dev7", 124, "0.5.1.dev124"),
        ("1.0.dev0", "00125", "1.0.dev125"),
    ],
)
def test_internal_preview_computes_version(
    version: str, run_number: str | int, expected: str
) -> None:
    """Guards PR #621 internal preview version policy."""
    assert compute_internal_preview_version(
        version,
        run_number,
    ) == InternalPreviewDecision(publish=True, version=expected)


@pytest.mark.parametrize("version", ["0.5.1", "0.5.1.post1"])
def test_internal_preview_skips_final_public_versions(version: str) -> None:
    """Guards PR #621 release staging skip policy."""
    assert compute_internal_preview_version(
        version,
        "123",
    ) == InternalPreviewDecision(publish=False)


@pytest.mark.parametrize(
    "version",
    [
        "0.5.1a1",
        "0.5.1b1",
        "0.5.1rc1",
        "0.5.1+local",
        "0.5.1rc1.dev0",
    ],
)
def test_internal_preview_rejects_ambiguous_versions(version: str) -> None:
    """Guards PR #621 against publishing ambiguous preview bases."""
    with pytest.raises(ReleaseVersionError, match="Internal preview releases"):
        compute_internal_preview_version(version, "123")


@pytest.mark.parametrize("run_number", ["0", "-1", "abc", "1.5"])
def test_internal_preview_rejects_invalid_run_numbers(run_number: str) -> None:
    """Guards PR #621 against malformed GitHub run numbers."""
    with pytest.raises(ReleaseVersionError, match="positive integer"):
        compute_internal_preview_version("0.5.1.dev0", run_number)


@pytest.mark.parametrize("version", ["0.5.1", "0.5.1.post1"])
def test_public_release_accepts_matching_final_versions(version: str) -> None:
    """Guards PR #621 public release tag validation."""
    assert validate_public_release_version(f"v{version}", version) == version


@pytest.mark.parametrize("version", ["0.5.1.dev0", "0.5.1rc1", "0.5.1+local"])
def test_public_release_rejects_non_final_versions(version: str) -> None:
    """Guards PR #621 against non-final public release versions."""
    with pytest.raises(ReleaseVersionError, match="final PEP 440"):
        validate_public_release_version(f"v{version}", version)


def test_public_release_rejects_tag_mismatch() -> None:
    """Guards PR #621 against publishing mismatched release tags."""
    with pytest.raises(ReleaseVersionError, match="does not match"):
        validate_public_release_version("v0.5.2", "0.5.1")


def test_internal_preview_cli_writes_github_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #621 GitHub output contract for internal preview releases."""
    pyproject = tmp_path / "pyproject.toml"
    output = tmp_path / "github-output"
    pyproject.write_text('[project]\nversion = "0.5.1.dev0"\n')
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert (
        main(
            [
                "internal-preview",
                "--pyproject",
                str(pyproject),
                "--run-number",
                "321",
            ]
        )
        == 0
    )

    assert output.read_text() == "publish=true\nversion=0.5.1.dev321\n"


def test_public_release_cli_writes_github_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #621 GitHub output contract for public releases."""
    pyproject = tmp_path / "pyproject.toml"
    output = tmp_path / "github-output"
    pyproject.write_text('[project]\nversion = "0.5.1"\n')
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert (
        main(
            [
                "public-release",
                "--pyproject",
                str(pyproject),
                "--citation",
                str(_citation(tmp_path)),
                "--tag",
                "v0.5.1",
            ]
        )
        == 0
    )

    assert output.read_text() == "version=0.5.1\n"


def test_citation_release_reads_unquoted_scalars(tmp_path: Path) -> None:
    """Guards the Zenodo archiving citation gate against YAML scalar surprises."""
    citation = _citation(tmp_path, version="0.5.1", date_released="2026-01-02")

    assert read_citation_release(citation) == CitationRelease(
        version="0.5.1", date_released=date(2026, 1, 2)
    )


def test_citation_release_reads_quoted_scalars(tmp_path: Path) -> None:
    """Guards the Zenodo archiving citation gate for quoted CITATION.cff values."""
    citation = _citation(tmp_path, version='"1.0"', date_released='"2026-01-02"')

    assert read_citation_release(citation) == CitationRelease(
        version="1.0", date_released=date(2026, 1, 2)
    )


def test_citation_release_reads_float_shaped_version(tmp_path: Path) -> None:
    """Guards the Zenodo archiving citation gate for a `1.0` release line.

    YAML parses an unquoted `version: 1.0` as a float, so the reader must accept
    that shape instead of rejecting a legitimate release.
    """
    citation = _citation(tmp_path, version="1.0")

    assert read_citation_release(citation).version == "1.0"


def test_citation_release_rejects_missing_file(tmp_path: Path) -> None:
    """Guards the Zenodo archiving citation gate against a deleted citation file."""
    with pytest.raises(ReleaseVersionError, match="is missing"):
        read_citation_release(tmp_path / "CITATION.cff")


def test_citation_release_rejects_malformed_yaml(tmp_path: Path) -> None:
    """Guards the Zenodo archiving citation gate against unparsable YAML."""
    citation = tmp_path / "CITATION.cff"
    citation.write_text("version: [0.5.1\n")

    with pytest.raises(ReleaseVersionError, match="as YAML"):
        read_citation_release(citation)


def test_citation_release_rejects_non_mapping(tmp_path: Path) -> None:
    """Guards the Zenodo archiving citation gate against a non-mapping document."""
    citation = tmp_path / "CITATION.cff"
    citation.write_text("- 0.5.1\n")

    with pytest.raises(ReleaseVersionError, match="YAML mapping"):
        read_citation_release(citation)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("date-released: 2026-01-02\n", "`version`"),
        ("version: 0.5.1\n", "`date-released`"),
        ("version: 0.5.1\ndate-released: not-a-date\n", "unparsable"),
    ],
)
def test_citation_release_rejects_incomplete_metadata(
    tmp_path: Path, body: str, match: str
) -> None:
    """Guards the Zenodo archiving citation gate against incomplete metadata."""
    citation = tmp_path / "CITATION.cff"
    citation.write_text(body)

    with pytest.raises(ReleaseVersionError, match=match):
        read_citation_release(citation)


def test_public_release_citation_accepts_matching_release() -> None:
    """Guards the Zenodo archiving citation gate happy path."""
    citation = CitationRelease(version="0.5.1", date_released=date(2026, 1, 2))

    validate_public_release_citation("0.5.1", citation, date(2026, 1, 2))


def test_public_release_citation_accepts_pep440_equivalent_version() -> None:
    """Guards the Zenodo archiving citation gate against padding-only mismatches."""
    citation = CitationRelease(version="0.5.1.0", date_released=date(2026, 1, 2))

    validate_public_release_citation("0.5.1", citation, date(2026, 1, 2))


def test_public_release_citation_rejects_stale_version() -> None:
    """Guards the Zenodo archiving citation gate against archiving a stale version."""
    citation = CitationRelease(version="0.5.0", date_released=date(2026, 1, 2))

    with pytest.raises(ReleaseVersionError, match=r"CITATION\.cff records version"):
        validate_public_release_citation("0.5.1", citation, date(2026, 1, 2))


def test_public_release_citation_tolerates_releaser_timezone_ahead_of_utc() -> None:
    """Guards the Zenodo archiving citation gate against a UTC+N false rejection.

    A releaser east of UTC writes tomorrow's UTC date when tagging after local
    midnight, so one day of skew must pass.
    """
    citation = CitationRelease(version="0.5.1", date_released=date(2026, 1, 3))

    validate_public_release_citation("0.5.1", citation, date(2026, 1, 2))


def test_public_release_citation_rejects_future_date() -> None:
    """Guards the Zenodo archiving citation gate against fabricated release dates."""
    citation = CitationRelease(version="0.5.1", date_released=date(2026, 2, 1))

    with pytest.raises(ReleaseVersionError, match="in the future"):
        validate_public_release_citation("0.5.1", citation, date(2026, 1, 2))


def test_public_release_cli_rejects_stale_citation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards the Zenodo archiving citation gate at the workflow boundary."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "0.5.1"\n')
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "github-output"))

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "public-release",
                "--pyproject",
                str(pyproject),
                "--citation",
                str(_citation(tmp_path, version="0.5.0")),
                "--tag",
                "v0.5.1",
            ]
        )

    assert excinfo.value.code == 1


def test_internal_preview_ignores_citation_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards `main`'s .devN line against tag-time citation rules.

    CITATION.cff names the last published release, so while main sits on a
    `.devN` version the citation file is legitimately behind pyproject.
    """
    pyproject = tmp_path / "pyproject.toml"
    output = tmp_path / "github-output"
    pyproject.write_text('[project]\nversion = "0.5.2.dev0"\n')
    _citation(tmp_path, version="0.5.1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert (
        main(
            [
                "internal-preview",
                "--pyproject",
                str(pyproject),
                "--run-number",
                "321",
            ]
        )
        == 0
    )

    assert output.read_text() == "publish=true\nversion=0.5.2.dev321\n"
