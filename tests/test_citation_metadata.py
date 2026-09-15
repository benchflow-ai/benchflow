"""Parity coverage for the citation and Zenodo archiving metadata files."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent

# Zenodo derives a release's version and publication date from the git tag and
# the GitHub Release, so pinning either here would go stale on every release --
# the exact drift the tag-time CITATION.cff gate exists to prevent.
ZENODO_DERIVED_KEYS = ("version", "publication_date")

CREATORS_PLACEHOLDER = "TODO BEFORE MERGE"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def citation() -> dict:
    return yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def zenodo() -> dict:
    return json.loads((REPO_ROOT / ".zenodo.json").read_text(encoding="utf-8"))


def test_zenodo_declares_software_upload(zenodo: dict) -> None:
    """Guards the Zenodo archiving deposit shape."""
    assert zenodo["upload_type"] == "software"
    assert zenodo["access_right"] == "open"


def test_zenodo_matches_citation_prose(citation: dict, zenodo: dict) -> None:
    """Guards the Zenodo archiving metadata against drifting from CITATION.cff."""
    assert zenodo["title"] == citation["title"]
    assert zenodo["description"] == citation["abstract"]
    assert zenodo["keywords"] == citation["keywords"]


def test_license_is_declared_identically_everywhere(
    pyproject: dict, citation: dict, zenodo: dict
) -> None:
    """Guards the Zenodo archiving license claim against a three-way mismatch."""
    assert pyproject["project"]["license"]["text"] == "Apache-2.0"
    assert citation["license"] == "Apache-2.0"
    assert zenodo["license"] == "Apache-2.0"


def test_zenodo_omits_release_derived_fields(zenodo: dict) -> None:
    """Guards the Zenodo archiving metadata against hardcoding a release version."""
    for key in ZENODO_DERIVED_KEYS:
        assert key not in zenodo


def test_citation_never_names_an_unpublished_version(
    pyproject: dict, citation: dict
) -> None:
    """Guards CITATION.cff against naming a version that was never released.

    CITATION.cff names the last published release, so on `main` it is either
    equal to the staged public version or behind the next `.devN` line. This
    catches a citation file that has run ahead of the project; a citation file
    that lags several releases behind is caught at tag time instead, by
    `tools/release_version.py public-release`.
    """
    project_release = Version(pyproject["project"]["version"]).base_version

    assert Version(str(citation["version"])) <= Version(project_release)


def test_zenodo_creators_are_resolved(zenodo: dict) -> None:
    """Blocks merging the Zenodo archiving PR with placeholder authorship.

    Zenodo mints a permanent DOI from `.zenodo.json`, and a published DOI cannot
    be withdrawn. This test is expected to fail until the agreed author list --
    names, ORCIDs, affiliations -- replaces the placeholder.
    """
    creators = zenodo["creators"]

    assert creators, ".zenodo.json must declare at least one creator."
    for creator in creators:
        assert CREATORS_PLACEHOLDER not in creator["name"], (
            "Replace the .zenodo.json creators placeholder with the agreed "
            "author list before merging; Zenodo authorship is permanent."
        )
