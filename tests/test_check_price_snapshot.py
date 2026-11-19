"""The release gate refuses a price snapshot past a vendor's announced change.

The unit tests read the snapshot as of its own date, so they pass on any
day; ``tools/check_price_snapshot.py`` compares the snapshot with the day a
release is cut, and both release workflows run it before they publish.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from tools.check_price_snapshot import main

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def test_the_gate_passes_while_every_dated_rate_is_current(capsys):
    assert main(["--today", "2026-11-30"]) == 0
    assert "current" in capsys.readouterr().out


def test_the_gate_names_a_rate_past_its_change(capsys):
    assert main(["--today", "2026-12-01"]) == 1
    err = capsys.readouterr().err
    assert "gemini/gemini-4-argon" in err
    assert "2026-11-30" in err
    assert "gemini-3.8-flash" not in err


def _steps(workflow: str, job: str) -> list[dict]:
    loaded = yaml.load((WORKFLOWS / workflow).read_text(), Loader=yaml.BaseLoader)
    return loaded["jobs"][job]["steps"]


def test_both_release_workflows_run_the_gate_before_they_publish():
    for workflow, job, publish in (
        ("public-release.yml", "publish", "Publish public release"),
        ("internal-preview-release.yml", "publish", "Publish internal preview"),
    ):
        names = [step.get("name") for step in _steps(workflow, job)]
        gate = next(
            i
            for i, step in enumerate(_steps(workflow, job))
            if "tools/check_price_snapshot.py" in step.get("run", "")
        )
        assert gate < names.index(publish), workflow
