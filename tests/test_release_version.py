from __future__ import annotations

from pathlib import Path

import pytest

from tools.release_version import (
    InternalPreviewDecision,
    ReleaseVersionError,
    compute_internal_preview_version,
    main,
    validate_public_release_version,
)


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


@pytest.mark.parametrize("version", ["0.5.1a1", "0.5.1b1", "0.5.1rc1"])
def test_internal_preview_skips_public_prerelease_versions(version: str) -> None:
    """A release candidate on main is a public-release staging version, as a
    final one is: it publishes no preview (the 0.10 pre-release change, which
    narrows PR #621's rejection of a/b/rc bases)."""
    assert compute_internal_preview_version(
        version,
        "123",
    ) == InternalPreviewDecision(publish=False)


@pytest.mark.parametrize("version", ["0.5.1+local", "0.5.1rc1.dev0"])
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


@pytest.mark.parametrize("version", ["0.5.1a1", "0.5.1b2", "0.10.0rc2"])
def test_public_release_accepts_matching_prerelease_versions(version: str) -> None:
    """An a/b/rc pre-release publishes publicly (the 0.10 pre-release change;
    PR #621 allowed final versions only)."""
    assert validate_public_release_version(f"v{version}", version) == version


@pytest.mark.parametrize(
    "version", ["0.5.1.dev0", "0.5.1rc1.dev0", "0.5.1+local", "0.5.1rc1+local"]
)
def test_public_release_rejects_dev_and_local_versions(version: str) -> None:
    """Guards PR #621 against publishing .dev and local versions publicly."""
    with pytest.raises(ReleaseVersionError, match="final or a/b/rc PEP 440"):
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
                "--tag",
                "v0.5.1",
            ]
        )
        == 0
    )

    assert output.read_text() == "version=0.5.1\nprerelease=false\n"


def test_public_release_cli_marks_a_prerelease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The workflow marks the GitHub release of an a/b/rc version as a
    pre-release from this output (the 0.10 pre-release change)."""
    pyproject = tmp_path / "pyproject.toml"
    output = tmp_path / "github-output"
    pyproject.write_text('[project]\nversion = "0.10.0rc2"\n')
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert (
        main(["public-release", "--pyproject", str(pyproject), "--tag", "v0.10.0rc2"])
        == 0
    )

    assert output.read_text() == "version=0.10.0rc2\nprerelease=true\n"
