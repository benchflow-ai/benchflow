"""The opt-in Evidence lines in the reviewer's criteria guidance.

A rubric criterion's ``description`` is author-facing and never reaches the
reviewer, yet rubrics such as FrontierPhysics' name there, in an ``Evidence:``
sentence, the records a criterion is judged from (the paper, code, results,
the trajectory); without them the reviewer picks its own. These tests guard
``ReviewerConfig.criterion_evidence``: the sentence reaches the guidance only
when asked, nothing else of the description ever does, and the default
guidance is unchanged (tests/test_reviewer_pdf_tools.py holds the default
reviewer task to a61a26f4's, byte for byte).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import benchflow
from benchflow.cli.main import app
from benchflow.review.config import (
    CRITERION_EVIDENCE_HEADER,
    Rubric,
    build_criteria_guidance,
    criterion_evidence,
)
from benchflow.review.options import ReviewerConfig
from benchflow.review.prompts import render_review_instruction
from benchflow.review.runner import run_reviews
from tests.test_review_runtime import FakeRun, make_rollout, make_task

EVIDENCE_RUBRIC = {
    "criteria": [
        {
            "name": "critical_definitions",
            "blocker": 1,
            "weight": 5,
            "description": "Category: theory. Evidence: paper.pdf and trajectory.",
            "guidance": "Pass when the definitions are right; fail otherwise.",
        },
        {
            "name": "pose_degeneracy",
            "blocker": 0,
            "weight": 3,
            "description": (
                "Category: research. Evidence: paper.pdf, aligned_image.npy. "
                "Grades whether the near-degeneracy is quantified."
            ),
            "guidance": "Score 2, 1, or 0.",
        },
        {
            "name": "honest_citations",
            "blocker": 0,
            "weight": 2,
            "description": "Category: using_evidence.",
            "guidance": "Score 2, 1, or 0.",
        },
    ]
}

PLAIN_LINES = [
    "- critical_definitions [BLOCKER; answer pass or fail; weight 5 does not "
    "enter weighted quality]: Pass when the definitions are right; fail otherwise.",
    "- pose_degeneracy [SCORED; weight 3; answer 0, 1, or 2]: Score 2, 1, or 0.",
    "- honest_citations [SCORED; weight 2; answer 0, 1, or 2]: Score 2, 1, or 0.",
]


@pytest.mark.parametrize(
    ("description", "named"),
    [
        (
            "Category: research. Evidence: paper.pdf and trajectory.",
            "paper.pdf and trajectory",
        ),
        ("Category: theory. Evidence: paper.pdf", "paper.pdf"),
        (
            "Category: results. Evidence: paper.pdf, 1.csv, 2.csv",
            "paper.pdf, 1.csv, 2.csv",
        ),
        (
            "Category: research. Evidence: paper.pdf. Grades the writing quality.",
            "paper.pdf",
        ),
        (
            "Evidence: paper.pdf;\n  the slots a and b in Statements.lean.",
            "paper.pdf; the slots a and b in Statements.lean",
        ),
        ("Category: using_evidence.", None),
        ("Internal note for rubric authors.", None),
        ("Evidence: .", None),
        ("", None),
    ],
)
def test_the_evidence_sentence_is_read_from_the_description(description, named):
    assert criterion_evidence(description) == named


def test_the_guidance_is_unchanged_unless_asked():
    rubric = Rubric.model_validate(EVIDENCE_RUBRIC)
    assert build_criteria_guidance(rubric).splitlines() == PLAIN_LINES
    assert build_criteria_guidance(rubric, evidence=False).splitlines() == PLAIN_LINES


def test_each_named_evidence_goes_under_its_criterion():
    rubric = Rubric.model_validate(EVIDENCE_RUBRIC)
    assert build_criteria_guidance(rubric, evidence=True).splitlines() == [
        CRITERION_EVIDENCE_HEADER,
        PLAIN_LINES[0],
        "  Evidence: paper.pdf and trajectory.",
        PLAIN_LINES[1],
        "  Evidence: paper.pdf, aligned_image.npy.",
        PLAIN_LINES[2],
    ]


def test_a_rubric_naming_no_evidence_keeps_its_guidance():
    rubric = Rubric.model_validate(
        {
            "criteria": [
                {
                    "name": "a",
                    "description": "Internal note.",
                    "guidance": "PASS when ok.",
                }
            ]
        }
    )
    assert build_criteria_guidance(rubric, evidence=True) == build_criteria_guidance(
        rubric
    )


def test_the_instruction_carries_only_the_evidence_sentence():
    rubric = Rubric.model_validate(EVIDENCE_RUBRIC)
    plain = render_review_instruction(rubric, output_schema={"type": "object"})
    shown = render_review_instruction(
        rubric, output_schema={"type": "object"}, criterion_evidence=True
    )
    assert "Evidence:" not in plain
    assert CRITERION_EVIDENCE_HEADER in shown
    assert "  Evidence: paper.pdf and trajectory." in shown
    for hidden in ("Category:", "Grades whether", "using_evidence"):
        assert hidden not in shown
    removed = shown.replace(f"{CRITERION_EVIDENCE_HEADER}\n", "")
    for line in (
        "  Evidence: paper.pdf and trajectory.\n",
        "  Evidence: paper.pdf, aligned_image.npy.\n",
    ):
        removed = removed.replace(line, "")
    assert removed == plain


def test_the_header_says_what_it_promises():
    assert "Evidence line" in CRITERION_EVIDENCE_HEADER
    assert "read those records before you judge it" in CRITERION_EVIDENCE_HEADER
    assert "{" not in CRITERION_EVIDENCE_HEADER
    assert chr(0x2014) not in CRITERION_EVIDENCE_HEADER  # no em dash


@pytest.mark.parametrize("asked", [False, True])
async def test_the_lines_reach_the_reviewer_and_the_record(
    tmp_path, monkeypatch, asked
):
    payload = {
        "trial_name": "rollout-a",
        "summary": "Reviewed.",
        "checks": {
            "critical_definitions": {"explanation": "ok", "outcome": "pass"},
            "pose_degeneracy": {"explanation": "ok", "score": 2},
            "honest_citations": {"explanation": "ok", "score": 1},
        },
    }
    fake = FakeRun(review_payload=payload)
    monkeypatch.setattr(benchflow, "run", fake)
    task = make_task(tmp_path, with_rubric=True, rubric_data=EVIDENCE_RUBRIC)
    rollout = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
    _, report_path = await run_reviews(
        rollout,
        agent="gemini",
        model="gemini/test-model",
        out_dir=tmp_path / "out",
        tasks_root=tmp_path / "tasks",
        criterion_evidence=asked,
    )
    (entry,) = json.loads(report_path.read_text())["trials"]
    assert entry["reviewer_environment"]["criterion_evidence"] is asked
    assert ("  Evidence: paper.pdf and trajectory." in fake.task_docs[0]) is asked
    assert (CRITERION_EVIDENCE_HEADER in fake.task_docs[0]) is asked
    assert "Grades whether" not in fake.task_docs[0]


def test_eval_run_takes_the_flag(tmp_path, monkeypatch):
    from benchflow.evaluation import Evaluation

    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    captured = []

    async def capture(self):
        captured.append(self._config.reviewer)
        return SimpleNamespace(
            passed=1, total=1, score=1.0, errored=0, verifier_errored=0
        )

    monkeypatch.setattr(Evaluation, "run", capture)
    ran = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(task),
            "--agent",
            "oracle",
            "--reviewer-criterion-evidence",
        ],
    )
    assert ran.exit_code == 0, ran.output
    assert captured == [ReviewerConfig(criterion_evidence=True)]


@pytest.mark.parametrize(
    ("flags", "expected"), [([], False), (["--criterion-evidence"], True)]
)
def test_bench_review_takes_the_flag(tmp_path, monkeypatch, flags, expected):
    import benchflow.review as review

    seen = {}

    async def fake_run_reviews(path, **kwargs):
        seen.update(kwargs)
        raise review.ReviewRunError("stop here")

    monkeypatch.setattr(review, "run_reviews", fake_run_reviews)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    CliRunner().invoke(app, ["review", str(tmp_path), *flags])
    assert seen["criterion_evidence"] is expected
