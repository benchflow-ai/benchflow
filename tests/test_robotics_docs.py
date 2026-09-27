"""docs/robotics.md stays in step with the robotics states and commands.

Regression test: no docs page said how to read
``trial-record.json`` or use ``python -m benchflow.robotics``. The page now
exists; these checks fail when a state or a flag is added without it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow._utils.scoring import ASSESSMENT_STATUSES
from benchflow.robotics.outcome import EXECUTION_STATUSES

_ROOT = Path(__file__).resolve().parents[1]
_PAGE = _ROOT / "docs" / "robotics.md"


def test_page_documents_every_execution_and_assessment_state():
    page = _PAGE.read_text()
    for status in EXECUTION_STATUSES | ASSESSMENT_STATUSES:
        assert re.search(rf"^\| `{status}` \|", page, re.MULTILINE), status


@pytest.mark.parametrize("command", ["index", "report", "score"])
def test_page_documents_every_flag_of_the_saved_trial_commands(command):
    help_text = subprocess.run(
        [sys.executable, "-m", "benchflow.robotics", command, "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    ).stdout
    flags = set(re.findall(r"--[a-z][a-z-]+", help_text)) - {"--help"}
    assert flags
    page = _PAGE.read_text()
    assert f"python -m benchflow.robotics {command} " in page
    for flag in flags:
        assert f"`{flag}" in page, flag


def test_readme_links_the_page():
    assert "(./docs/robotics.md)" in (_ROOT / "README.md").read_text()
