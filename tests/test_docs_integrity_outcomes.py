"""the guide, the help and errors.md say what the integrity rules do.

``provenance_incomplete`` (an agent user's attempt scores 0 when a check of
where its reward came from does not complete) and ``integrity_unavailable``
(BenchFlow can't check at all: unscored) are named where a reader meets
them, and no page says the reward never changes or that the image's tools
are never used.
"""

from __future__ import annotations

import pytest
import typer.main
from typer.core import TyperGroup

from benchflow.cli.main import app
from tools.docs_errors import FIXES, PAGE, _norm, curated, render

OUTCOMES = ("integrity_unavailable", "provenance_incomplete")
REWARD_HACKING = PAGE.parents[1] / "tasks" / "reward-hacking.md"
# errors.md's verifier messages whose fix is written in the TSV.
PROVENANCE_MESSAGES = (
    "the links into /logs/verifier could not all be looked for: …",
    "the links into /logs/verifier could not be looked for: …",
    "the verifier user's uid could not be read",
)


def test_the_error_index_has_a_causes_row_for_both():
    page = render()
    for key in OUTCOMES:
        assert f"| `{key}` |" in page, key


def test_no_page_tells_readers_to_keep_readlink_instead_of_bfx():
    assert "keep `readlink` in the image" not in PAGE.read_text(encoding="utf-8")


def _fixes() -> dict[str, str]:
    """The verifier section's written fixes (tools/docs_errors_fixes.tsv)."""
    assert FIXES.exists()
    return {
        message: fix
        for (section, _key, message), fix in curated().items()
        if section == "verifier"
    }


@pytest.mark.parametrize("message", PROVENANCE_MESSAGES)
def test_each_provenance_fix_names_the_outcome(message):
    fix = _fixes()[_norm(message)]
    assert "`provenance_incomplete`" in fix, fix
    assert "unverifiable for this trial" not in fix, fix


def test_the_link_walk_fixes_name_benchflow_s_own_gap():
    fixes = _fixes()
    for message in PROVENANCE_MESSAGES[:2]:
        fix = fixes[_norm(message)]
        assert "`integrity_unavailable`" in fix, fix


def test_the_reward_hacking_guide_matches_the_rules():
    text = REWARD_HACKING.read_text(encoding="utf-8")
    assert "The reward itself never changes" not in text
    assert "never the image's tools" not in text
    # The one image program left in the link check (kernel_resolve).
    assert "`readlink`" in text
    for key in OUTCOMES:
        assert f"`{key}`" in text, key


def _integrity_help(*path: str) -> str:
    command = typer.main.get_command(app)
    for word in path:
        assert isinstance(command, TyperGroup)
        command = command.commands[word]
    for param in command.params:
        if "--integrity" in getattr(param, "opts", ()):
            return " ".join(str(getattr(param, "help", "") or "").split())
    raise AssertionError(f"{path} has no --integrity")


@pytest.mark.parametrize("path", [("run",), ("eval", "run")])
def test_the_integrity_help_names_both_outcomes(path):
    text = _integrity_help(*path)
    for key in OUTCOMES:
        assert key in text, (path, text)
    assert "bench explain" in text, text
