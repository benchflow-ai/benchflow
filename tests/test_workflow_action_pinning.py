"""Regression guard: every GitHub Actions `uses:` must be SHA-pinned.

Tag-pinned actions (`actions/checkout@v4`) follow mutable refs — a compromised
maintainer or registry can swap the underlying commit without our PR review.
Pinning to a 40-char SHA closes that supply-chain hole. The accompanying tag
in a trailing comment (e.g. `# v4.2.2`) is for humans and Dependabot, not the
runner.

This test walks `.github/workflows/*.yml`, finds every `uses:` line, and
asserts the value matches `<owner>/<repo>@<40-hex-sha>` (optionally with a
subpath like `actions/cache/save`). Local actions (`uses: ./.github/...`)
and Docker actions (`uses: docker://...`) are exempt — they don't pull from
the marketplace.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# owner/repo[/subpath]@<40-hex-sha>
_SHA_PINNED = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._/-]+)?@[0-9a-f]{40}$"
)
_LOCAL_OR_DOCKER = re.compile(r"^(?:\./|docker://)")
_VERSION_COMMENT = re.compile(r"#\s*v\d+\.\d+")


def _iter_uses(node: object):
    """Yield every `uses:` string found anywhere inside a parsed workflow."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "uses" and isinstance(value, str):
                yield value
            else:
                yield from _iter_uses(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_uses(item)


def _iter_uses_lines(workflow: Path) -> list[str]:
    """Return raw `uses:` lines (preserving trailing comments) from *workflow*."""
    return [
        line
        for line in workflow.read_text().splitlines()
        if re.search(r"^\s*-?\s*uses:", line)
    ]


def _workflow_files() -> list[Path]:
    if not WORKFLOWS_DIR.is_dir():
        return []
    return sorted(p for p in WORKFLOWS_DIR.iterdir() if p.suffix in {".yml", ".yaml"})


def test_workflows_directory_exists() -> None:
    """If this fails, the test layout drifted; update WORKFLOWS_DIR."""
    assert WORKFLOWS_DIR.is_dir(), f"missing {WORKFLOWS_DIR}"
    assert _workflow_files(), "no workflow files found — did the suite move?"


@pytest.mark.parametrize("workflow", _workflow_files(), ids=lambda p: p.name)
def test_workflow_actions_are_sha_pinned(workflow: Path) -> None:
    """Every `uses:` in this workflow must be SHA-pinned, not tag-pinned."""
    data = yaml.safe_load(workflow.read_text())
    unpinned: list[str] = []
    for ref in _iter_uses(data):
        if _LOCAL_OR_DOCKER.match(ref):
            continue
        if not _SHA_PINNED.match(ref):
            unpinned.append(ref)
    assert not unpinned, (
        f"{workflow.name} uses non-SHA-pinned actions: {unpinned}. "
        "Pin to a 40-char commit SHA and leave the tag in a trailing comment."
    )


@pytest.mark.parametrize("workflow", _workflow_files(), ids=lambda p: p.name)
def test_sha_pins_have_version_comment(workflow: Path) -> None:
    """Every SHA-pinned `uses:` must have a trailing `# vX.Y.Z` comment.

    Guards PR #450 review finding: Dependabot and human readers rely on the
    version comment to know which release a SHA corresponds to.
    """
    missing: list[str] = []
    for line in _iter_uses_lines(workflow):
        value = line.split("uses:", 1)[1].strip()
        ref = value.split("#")[0].strip() if "#" in value else value.strip()
        if _LOCAL_OR_DOCKER.match(ref):
            continue
        if not _SHA_PINNED.match(ref):
            continue
        if not _VERSION_COMMENT.search(line):
            missing.append(ref)
    assert not missing, (
        f"{workflow.name} has SHA-pinned actions without a trailing version "
        f"comment (e.g. `# v4.2.2`): {missing}"
    )


@pytest.mark.parametrize("workflow", _workflow_files(), ids=lambda p: p.name)
def test_workflow_declares_permissions(workflow: Path) -> None:
    """Workflows must declare an explicit top-level `permissions:` block.

    Without it, jobs inherit the repo-wide default GITHUB_TOKEN scope, which
    is broader than most jobs need. Least privilege is the supply-chain
    posture we want.
    """
    data = yaml.safe_load(workflow.read_text())
    assert isinstance(data, dict), f"{workflow.name} is not a mapping"
    assert "permissions" in data, (
        f"{workflow.name} is missing a top-level `permissions:` block; "
        "declare least-privilege scopes explicitly (e.g. `contents: read`)."
    )
