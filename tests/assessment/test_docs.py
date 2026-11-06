"""The assessment docs and their example protocol stay true to the code.

Docs v2 moved docs/jobs/assessment.md's walkthrough out: the human
preferences page names ``bench assess`` and sends readers to its ``--help``,
and the generated commands reference documents every command.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer.main

from benchflow.assessment import records as r
from benchflow.cli.main import app

REPO = Path(__file__).resolve().parents[2]
GUIDE = REPO / "docs/tasks/human-preferences.md"
CLI_MD = REPO / "docs/reference/commands.md"
PROTOCOL = REPO / "docs/examples/assessment/en161-blind-pair.protocol.json"


def test_every_command_is_documented() -> None:
    group = typer.main.get_command(app).commands["assess"]  # type: ignore[attr-defined]
    guide, reference = GUIDE.read_text(), CLI_MD.read_text()
    section = reference[reference.index("## bench assess") :]
    section = section[: section.index("\n## ", 3)]
    for name in group.commands:
        assert f"`{name}`" in section or f"`{name}`," in section, name
        assert f"bench assess {name}" in guide or f"`{name}`" in section, name


def test_the_example_protocol_is_a_valid_protocol() -> None:
    document = json.loads(PROTOCOL.read_text())
    r.check_protocol_document(document)
    assert document["collector"]["commit"] == "f2650b0542ac788401e8f5f8e48af121ac320249"
    assert set(document["collector"]["sources"]) >= {"model.py", "app.py", "web/app.js"}
