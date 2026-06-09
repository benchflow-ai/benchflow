import json
import subprocess
import sys
from pathlib import Path

SKILL_ROOT = Path(".claude/skills/benchflow-experiment-review")
EXTRACT_SKILLS = SKILL_ROOT / "scripts" / "extract_harness_skills.py"
FIXTURES = SKILL_ROOT / "evals" / "files"


def _run_extract(fixture: str) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            str(EXTRACT_SKILLS),
            str(FIXTURES / fixture / "trajectory" / "llm_trajectory.jsonl"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_experiment_review_extracts_with_skill_loading() -> None:
    """Guards PR #1's trajectory-health review helper for task skill catalogs."""
    extracted = _run_extract("clean-pass")

    assert extracted["harness"] == "openhands"
    assert extracted["task_skill_mode"] == "with_skills"
    assert extracted["expected_task_skills"] == ["data-cleaning"]
    assert extracted["task_skills_loading"] == 1
    assert extracted["task_skills_loading_status"] == "complete"
    assert extracted["no_skill_leakage_detected"] is False
    assert len(extracted["checked_files"]) == 2


def test_experiment_review_detects_no_skill_task_skill_leakage() -> None:
    """Guards PR #1 against accepting no-skill runs that read task skills."""
    extracted = _run_extract("no-skill-leak")

    assert extracted["harness"] == "codex"
    assert extracted["task_skill_mode"] == "without_skills"
    assert extracted["expected_task_skills"] == ["task-helper"]
    assert extracted["task_skills_loading"] == 0
    assert extracted["task_skills_loading_status"] == "not_loaded_without_skills"
    assert extracted["no_skill_leakage_detected"] is True
    assert extracted["manual_review_required"] is True
    assert any("task-helper" in item for item in extracted["no_skill_leakage_evidence"])


def test_experiment_review_ignores_benign_no_skill_prompt_markers() -> None:
    """Guards PR #1 against false positives from no-skill prompt text."""
    extracted = _run_extract("no-skill-benign-marker")

    assert extracted["harness"] == "codex"
    assert extracted["task_skill_mode"] == "without_skills"
    assert extracted["expected_task_skills"] == ["task-helper"]
    assert extracted["task_skills_loading"] == 0
    assert extracted["no_skill_leakage_detected"] is False
    assert extracted["no_skill_leakage_evidence"] == []
