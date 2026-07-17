from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = (
    ROOT
    / ".agents"
    / "skills"
    / "benchflow-experiment-review"
    / "scripts"
    / "validate_run_artifacts.py"
)
FIXTURES = (
    ROOT / ".agents" / "skills" / "benchflow-experiment-review" / "evals" / "files"
)


def _run_validator(fixture: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(FIXTURES / fixture), "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_experiment_review_validator_accepts_complete_fixture() -> None:
    """Guards PR #827's validator cleanup from breaking healthy artifact acceptance."""
    result = _run_validator("clean-pass")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["healthy"] is True
    assert payload["summary"] == {"checked": 1, "healthy": 1, "unhealthy": 0}


def test_experiment_review_validator_rejects_missing_llm_fixture() -> None:
    """Guards PR #827's validator cleanup from losing the missing-LLM fail-closed gate."""
    result = _run_validator("missing-llm-trajectory")

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["healthy"] is False
    assert payload["summary"] == {"checked": 1, "healthy": 0, "unhealthy": 1}
    assert "missing required artifact" in payload["rollouts"][0]["issues"][0]
