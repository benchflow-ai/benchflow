"""Release version policy helpers for GitHub Actions workflows."""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from packaging.version import InvalidVersion, Version


class ReleaseVersionError(ValueError):
    """Raised when a release version does not match BenchFlow policy."""


@dataclass(frozen=True)
class InternalPreviewDecision:
    """Internal preview publication decision."""

    publish: bool
    version: str = ""


def read_project_version(pyproject_path: Path) -> str:
    """Return the `[project].version` value from a pyproject file."""
    try:
        pyproject = tomllib.loads(pyproject_path.read_text())
        version = pyproject["project"]["version"]
    except (KeyError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseVersionError(
            f"Could not read [project].version from {pyproject_path}."
        ) from exc
    if not isinstance(version, str) or not version:
        raise ReleaseVersionError(
            f"[project].version in {pyproject_path} must be a non-empty string."
        )
    return version


def compute_internal_preview_version(
    version_text: str, run_number: str | int
) -> InternalPreviewDecision:
    """Compute the internal preview version for a tested main workflow run."""
    current = _parse_version(version_text)
    normalized_run_number = _normalize_run_number(run_number)

    if current.is_devrelease and current.pre is None and current.local is None:
        return InternalPreviewDecision(
            publish=True,
            version=f"{current.base_version}.dev{normalized_run_number}",
        )
    if not current.is_prerelease and current.local is None:
        return InternalPreviewDecision(publish=False)
    raise ReleaseVersionError(
        "Internal preview releases must be based on a plain .dev version "
        f"or a final public-release staging version, got {current}."
    )


def validate_public_release_version(tag_name: str, version_text: str) -> str:
    """Validate that a public release tag matches a final project version."""
    version = _parse_version(version_text)

    if tag_name != f"v{version_text}":
        raise ReleaseVersionError(
            f"Tag {tag_name!r} does not match pyproject.toml version 'v{version_text}'."
        )
    if version.is_prerelease or version.local is not None:
        raise ReleaseVersionError(
            f"Public releases must use a final PEP 440 version, got {version_text!r}."
        )
    return version_text


def write_github_output(values: dict[str, str]) -> None:
    """Write GitHub Actions outputs, or print key-value lines outside Actions."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path is None:
        for key, value in values.items():
            print(f"{key}={value}")
        return

    with Path(output_path).open("a", encoding="utf-8") as output_file:
        for key, value in values.items():
            output_file.write(f"{key}={value}\n")


def _parse_version(version_text: str) -> Version:
    try:
        return Version(version_text)
    except InvalidVersion as exc:
        raise ReleaseVersionError(
            f"Invalid PEP 440 version: {version_text!r}."
        ) from exc


def _normalize_run_number(run_number: str | int) -> str:
    run_number_text = str(run_number)
    if not run_number_text.isdecimal() or int(run_number_text) <= 0:
        raise ReleaseVersionError(
            f"GitHub workflow run number must be a positive integer, got {run_number!r}."
        )
    return str(int(run_number_text))


def _cmd_internal_preview(args: argparse.Namespace) -> int:
    run_number = args.run_number or os.environ.get("TEST_RUN_NUMBER")
    if run_number is None:
        raise ReleaseVersionError(
            "TEST_RUN_NUMBER must be set or passed with --run-number."
        )

    decision = compute_internal_preview_version(
        read_project_version(args.pyproject),
        run_number,
    )
    write_github_output(
        {
            "publish": "true" if decision.publish else "false",
            "version": decision.version,
        }
    )
    return 0


def _cmd_public_release(args: argparse.Namespace) -> int:
    tag_name = args.tag or os.environ.get("TAG_NAME")
    if tag_name is None:
        raise ReleaseVersionError("TAG_NAME must be set or passed with --tag.")

    version = validate_public_release_version(
        tag_name,
        read_project_version(args.pyproject),
    )
    write_github_output({"version": version})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    internal_preview = subparsers.add_parser(
        "internal-preview",
        help="Compute the internal preview version for a main test workflow run.",
    )
    internal_preview.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml.",
    )
    internal_preview.add_argument(
        "--run-number",
        help="Successful test workflow run number. Defaults to TEST_RUN_NUMBER.",
    )
    internal_preview.set_defaults(func=_cmd_internal_preview)

    public_release = subparsers.add_parser(
        "public-release",
        help="Validate a public release tag against pyproject.toml.",
    )
    public_release.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml.",
    )
    public_release.add_argument(
        "--tag",
        help="Release tag name. Defaults to TAG_NAME.",
    )
    public_release.set_defaults(func=_cmd_public_release)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ReleaseVersionError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    sys.exit(main())
