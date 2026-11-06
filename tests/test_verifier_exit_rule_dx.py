"""One rule for a verifier that exits nonzero after writing a reward (DF-6).

Dogfood p2-task-author (2026-10-18): ``tasks/b3-exit1`` writes 1 and then
exits 1. BenchFlow scores it (pytest exits 1 when a test fails, so a
reward written by the run is its verdict), as 0.7.11 did and as
``docs/task-standard.md`` said. The scaffold's test.sh comment and
``docs/task-authoring-task-md.md`` said a nonzero exit is a verifier
infrastructure failure instead. They now state the scoring rule, and so does
``docs/tasks/verifiers.md`` in docs v2.
"""

from __future__ import annotations

from pathlib import Path

from benchflow._utils.task_authoring.scaffolding import scaffold_task

DOCS = Path(__file__).resolve().parents[1] / "docs"
_WRONG = "nonzero exit means verifier infrastructure failure"


def _one_line(text: str) -> str:
    return " ".join(text.replace("#", " ").split())


def test_the_scaffold_states_the_scoring_rule(tmp_path: Path) -> None:
    for task_format in ("task-md", "legacy"):
        result = scaffold_task(
            f"t-{task_format}", parent_dir=tmp_path, task_format=task_format
        )
        scripts = list(result.task_dir.glob("*/test.sh"))
        assert len(scripts) == 1, scripts
        text = _one_line(scripts[0].read_text())
        assert _WRONG not in text
        assert "A reward this run wrote is scored whatever the exit code" in text
        assert "a nonzero exit with no reward scores 0 with a warning" in text


def test_the_authoring_docs_state_the_scoring_rule() -> None:
    # Docs v2's layout: the verifiers page states the rule once; the
    # task format page took over the task-package reference.
    verifier = _one_line((DOCS / "tasks" / "verifiers.md").read_text())
    package = _one_line((DOCS / "tasks" / "task-format.md").read_text())
    assert _WRONG not in verifier and _WRONG not in package
    assert "A reward the script wrote counts, whatever its exit code" in verifier
    assert (
        "A script that exits with an error and writes nothing scores 0, "
        "with a warning" in verifier
    )
