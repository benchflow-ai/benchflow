"""Path traversal regression tests for benchflow.skill_eval.

Covers issues:
    * #361 — case id used as task directory segment
    * #403 — case id used in GEPA trace filenames
    * #405 — skill_name used as ``skills/<name>`` directory segment

Each test confirms a ValueError is raised before any file is written and
asserts no escape artifact lands outside the intended output root.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.skill_eval import (
    CaseResult,
    EvalCase,
    EvalDataset,
    SkillEvalResult,
    export_gepa_traces,
    generate_tasks,
    load_eval_dataset,
)


def _write_skill_dir(
    root: Path,
    *,
    skill_name: str | None = None,
    case_id: str = "ok-case",
) -> Path:
    """Build a minimal skill directory with one eval case."""
    skill = root / "skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: audit-skill\n---\n# Audit skill\n")
    payload: dict = {"cases": [{"id": case_id, "question": "q"}]}
    if skill_name is not None:
        payload["skill_name"] = skill_name
    (skill / "evals" / "evals.json").write_text(json.dumps(payload))
    return skill


# ---------------------------------------------------------------------------
# #361 — case id as task_dir segment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_case_id",
    ["../escaped-case", "/tmp/escape", "a/b", "..", "."],
)
def test_load_eval_dataset_rejects_traversal_case_id(
    tmp_path: Path, bad_case_id: str
) -> None:
    skill = _write_skill_dir(tmp_path, case_id=bad_case_id)
    with pytest.raises(ValueError, match="case id"):
        load_eval_dataset(skill)


def test_generate_tasks_rejects_traversal_case_id_at_call_time(
    tmp_path: Path,
) -> None:
    # Build EvalDataset by hand to bypass load-time validation, simulating a
    # caller that constructs the dataset programmatically.
    skill = _write_skill_dir(tmp_path)
    dataset = EvalDataset(
        skill_name="audit-skill",
        skill_dir=skill,
        cases=[EvalCase(id="../escaped-case", question="q")],
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="case id"):
        generate_tasks(dataset, out)
    # Nothing escaped the requested output root.
    assert not (tmp_path / "escaped-case").exists()
    assert not (tmp_path / "escaped-case" / "task.toml").exists()


# ---------------------------------------------------------------------------
# #405 — skill_name as skills/<name> segment
# ---------------------------------------------------------------------------


def test_load_eval_dataset_rejects_traversal_skill_name(tmp_path: Path) -> None:
    skill = _write_skill_dir(tmp_path, skill_name="../../../../escaped-skill-copy")
    with pytest.raises(ValueError, match="skill name"):
        load_eval_dataset(skill)


def test_generate_tasks_rejects_traversal_skill_name_at_call_time(
    tmp_path: Path,
) -> None:
    skill = _write_skill_dir(tmp_path)
    dataset = EvalDataset(
        skill_name="../../../../escaped-skill-copy",
        skill_dir=skill,
        cases=[EvalCase(id="ok-case", question="q")],
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="skill name"):
        generate_tasks(dataset, out, with_skill=True)
    # No SKILL.md landed outside the intended task tree.
    assert not (tmp_path / "escaped-skill-copy").exists()


# ---------------------------------------------------------------------------
# #403 — case id as GEPA trace filename component
# ---------------------------------------------------------------------------


def test_export_gepa_rejects_traversal_case_id(tmp_path: Path) -> None:
    skill = _write_skill_dir(tmp_path)
    dataset = EvalDataset(
        skill_name="audit-skill",
        skill_dir=skill,
        cases=[EvalCase(id="ok-case", question="q")],
    )
    # CaseResult is independent of load_eval_dataset, so the export sink
    # needs its own validation step.
    cr = CaseResult(
        case_id="../../escaped-gepa",
        agent="agent-a",
        model="m",
        with_skill=True,
        reward=1.0,
    )
    result = SkillEvalResult(
        skill_name="audit-skill",
        n_cases=1,
        agents=["agent-a"],
        case_results=[cr],
    )
    output = tmp_path / "gepa-out"
    with pytest.raises(ValueError, match="case id"):
        export_gepa_traces(result, dataset, output)
    # No escaped JSON file landed at the sibling location.
    assert not list(tmp_path.glob("escaped-gepa-*.json"))
