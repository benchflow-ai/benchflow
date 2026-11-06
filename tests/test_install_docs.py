"""The install page: what it must tell a user, and that doctor's links reach it.

``bench doctor``'s ``bench on PATH`` line sends people to the published
install page's ``#which-bench-runs``; a renamed heading would leave that link
pointing nowhere, so the anchor is checked against the page's headings.
"""

from __future__ import annotations

import re
from pathlib import Path

from benchflow import doctor

_REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL = _REPO_ROOT / "docs" / "start" / "install.md"
STARTED = _REPO_ROOT / "docs" / "start" / "quick-start.md"
_UV_TOOL_INSTALL_BENCHFLOW = re.compile(
    r"uv tool install\b[^\n`]*\bbenchflow(?:\[[^\]\n`]+\])?\b"
)


def _slug(heading: str) -> str:
    # GitHub's anchor for a Markdown heading.
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def _anchors(text: str) -> set[str]:
    return {
        _slug(line.lstrip("#"))
        for line in text.splitlines()
        if re.match(r"#{1,6} ", line)
    }


def test_install_page_gives_install_upgrade_uninstall_and_constraints():
    text = INSTALL.read_text()
    # A pinned install with the constraints of that same release.
    assert re.search(
        r"uv tool install --python 3\.12 'benchflow==(?P<version>[^']+)' \\\n"
        r"\s*--constraints https://raw\.githubusercontent\.com/benchflow-ai/"
        r"benchflow/v(?P=version)/constraints\.txt",
        text,
    ), "the install page pins a version with that release's constraints.txt"
    # A pinned install moves only when the install line runs again.
    assert "run the install line again with the other version in both places" in text
    assert "uv tool uninstall benchflow" in text
    for command in _UV_TOOL_INSTALL_BENCHFLOW.finditer(text):
        assert "--python 3.12" in command.group(0), command.group(0)


def test_install_page_says_which_bench_runs_and_how_to_keep_both():
    text = INSTALL.read_text()
    section = text.split("## Which bench runs", 1)[1]
    assert 'tar -xzf "$F" -C ~/.bench/bin' in section
    assert "`benchflow` always runs this package" in section
    assert 'export PATH="$HOME/.bench/bin:$PATH"' in section
    assert "bench doctor" in section


def test_the_rust_bench_is_a_checked_download_not_a_piped_script():
    """The page piped the installer from the mutable bench-rs
    branch into sh, unchecked; BENCH_VERSION pinned only what that script
    then fetched. The page now downloads the pinned release and checks its
    published sha256 before anything is unpacked, and pipes nothing to a
    shell."""
    text = INSTALL.read_text()
    assert re.search(r"\|\s*(?:[A-Z_]+=\S+\s+)*(?:ba|z|da)?sh\b", text) is None
    section = text.split("## Which bench runs", 1)[1]
    assert "releases/download/bench-v$V" in section
    check = section.index('shasum -a 256 -c "$F.sha256" &&')
    assert check < section.index('tar -xzf "$F"')


def test_the_reinstall_advice_keeps_the_install_command():
    """A bare reinstall drops the version, extras and
    constraints the user installed with."""
    text = INSTALL.read_text()
    assert (
        "run the install line again with the other version in both places "
        "and the extras you use"
    ) in text


def test_doctor_links_to_a_heading_of_the_install_page():
    """Doctor links the published page, which is
    docs/getting-started/install.md, not a repo path."""
    assert (
        doctor.INSTALL_GUIDE_URL
        == "https://www.benchflow.ai/docs/benchflow/getting-started/install"
    )
    assert "which-bench-runs" in _anchors(INSTALL.read_text())


def test_readme_and_getting_started_point_at_the_install_page():
    readme = (_REPO_ROOT / "README.md").read_text()
    started = STARTED.read_text()
    assert "(./docs/start/install.md)" in readme
    assert "(./install.md)" in started
