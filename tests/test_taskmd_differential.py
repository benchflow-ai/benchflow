"""BenchFlow parses a task.md draft 2 package exactly as the reference parser does.

The recorded half needs nothing: golden/<name>.json is what the pinned commit's
``tools/taskmd.py json`` printed for each fixture. The live half runs a task-md
checkout's own tool (``TASKMD_REPO``) on every example package and compares
the whole parse, diagnostics included.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.taskmd import package_json
from tests._taskmd_helpers import EXAMPLES, GOLDEN, reference_tool, require_taskmd_repo

GOLDEN_NAMES = sorted(p.stem for p in GOLDEN.glob("*.json"))


def test_there_are_goldens_for_every_fixture() -> None:
    assert sorted(p.name for p in EXAMPLES.iterdir() if p.is_dir()) == GOLDEN_NAMES


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_parse_equals_the_recorded_reference_output(name: str, monkeypatch) -> None:
    monkeypatch.chdir(EXAMPLES)
    assert package_json(name) == json.loads((GOLDEN / f"{name}.json").read_text())


def _package_roots(root: Path) -> list[Path]:
    """Every package under root: the shallowest folders holding task.md."""
    found: list[Path] = []
    for task_md in sorted(root.rglob("task.md")):
        folder = task_md.parent
        if not any(parent in found for parent in folder.parents):
            found.append(folder)
    return found


def test_parse_equals_the_reference_tool_on_every_example(monkeypatch) -> None:
    repo = require_taskmd_repo()
    examples = repo / "examples"
    monkeypatch.chdir(examples)
    roots = _package_roots(examples)
    assert len(roots) >= 20
    mismatched = []
    for folder in roots:
        rel = str(folder.relative_to(examples))
        run = reference_tool(repo, "taskmd.py", "json", rel, cwd=examples)
        expected = json.loads(run.stdout)
        got = package_json(rel)
        if got != expected or got["ok"] != (run.returncode == 0):
            mismatched.append(rel)
    assert mismatched == []
