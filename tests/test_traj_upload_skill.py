"""Drift guards for the public trajectory-upload operating skill."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import click
import typer
import yaml

from benchflow.cli.main import app

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKILL_DIR = _REPO_ROOT / ".agents" / "skills" / "benchflow-traj-upload"
_SKILL_MD = _SKILL_DIR / "SKILL.md"
_CLI_CONTRACT = _SKILL_DIR / "references" / "cli-contract.md"


def _frontmatter(text: str) -> dict[str, object]:
    assert text.startswith("---\n")
    _, raw, _ = text.split("---", maxsplit=2)
    parsed = yaml.safe_load(raw)
    assert isinstance(parsed, dict)
    return parsed


def _traj_upload_flags() -> set[str]:
    command = typer.main.get_command(app)
    for segment in ("traj", "upload"):
        command = cast("click.Group", command).commands[segment]
    flags = {
        option
        for parameter in command.params
        for option in getattr(parameter, "opts", ())
        if option.startswith("--")
    }
    flags.discard("--help")
    return flags


def test_traj_upload_skill_uses_current_skill_creator_shape() -> None:
    """Guards the skill packaging update from PR #1008 against metadata drift."""
    skill_text = _SKILL_MD.read_text(encoding="utf-8")
    metadata = _frontmatter(skill_text)

    assert set(metadata) == {"name", "description"}
    assert metadata["name"] == "benchflow-traj-upload"
    description = str(metadata["description"])
    for trigger in ("upload", "test", "troubleshoot", "manifest", "production"):
        assert trigger in description
    assert "[references/cli-contract.md](references/cli-contract.md)" in skill_text


def test_traj_upload_skill_documents_every_live_cli_flag() -> None:
    """Guards PR #1008 so new trajectory-upload flags cannot bypass the skill."""
    contract = _CLI_CONTRACT.read_text(encoding="utf-8")
    documented = {
        flag for flag in _traj_upload_flags() if f"`{flag}`" in contract
    }

    assert documented == _traj_upload_flags(), (
        "bench traj upload flags missing from the skill contract: "
        f"{sorted(_traj_upload_flags() - documented)}"
    )


def test_traj_upload_skill_pins_safety_and_report_contract() -> None:
    """Guards the production/safety contract established by PR #1008."""
    package = "\n".join(
        (
            _SKILL_MD.read_text(encoding="utf-8"),
            _CLI_CONTRACT.read_text(encoding="utf-8"),
        )
    )

    required = (
        "<XXX-benchflow-key-values-XXX>",
        "total_steps = thinking_steps + tool_call_steps + human_steps",
        "first 100",
        "schema `1.2.0`",
        "manifest last",
        "sources/community/<digest>/",
        "A dry run proves only",
        "does not use public quarantine or the public validator",
        "Never print signed PUT URLs",
    )
    for invariant in required:
        assert invariant in package


def test_traj_upload_skill_evals_are_parseable_and_cover_three_modes() -> None:
    """Guards the forward-test fixtures added with PR #1008."""
    evals = yaml.safe_load(
        (_SKILL_DIR / "evals" / "evals.json").read_text(encoding="utf-8")
    )

    assert evals["skill_name"] == "benchflow-traj-upload"
    assert [case["id"] for case in evals["evals"]] == [1, 2, 3]
    prompts = " ".join(case["prompt"] for case in evals["evals"])
    for mode in ("--dry-run", "bench traj upload", "--direct"):
        assert mode in prompts
