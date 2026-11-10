"""The opt-in submitted-file rule, and the prompt's hashes in the record.

Some rubric reviews of a paper judged its TeX sources, or text the agent had
extracted before later edits, instead of the PDF it submitted: a61a26f4's
review template never says which file is the deliverable. These tests guard
``ReviewerConfig.submitted_file_rule`` (the rule goes into the template only
when asked), and the template and instruction sha256 every review now
records. tests/test_reviewer_pdf_tools.py holds the default instruction to
a61a26f4's, byte for byte.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import benchflow
from benchflow.cli.main import app
from benchflow.review.options import ReviewerConfig
from benchflow.review.prompts import (
    REVIEW_TEMPLATE,
    SUBMITTED_FILE_RULE,
    render_review_instruction,
    review_template,
)
from benchflow.review.runner import run_reviews
from benchflow.task import Task
from tests.test_review_runtime import (
    RUBRIC,
    FakeRun,
    good_review,
    good_weighted_review,
    make_rollout,
    make_task,
)
from tests.test_reviewer_pdf_tools import GOLDEN, _automatic, _record

ANCHOR = "Work through the criteria one at a time."


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class InstructionRun(FakeRun):
    """FakeRun that also keeps the instruction benchflow reads from the
    wrapper task, while the wrapper still exists."""

    async def __call__(self, config):
        self.instructions = getattr(self, "instructions", [])
        self.instructions.append(Task(config.task_path).instruction)
        return await super().__call__(config)


def test_the_rule_goes_before_the_criteria_paragraph_only_when_asked():
    assert review_template() is REVIEW_TEMPLATE
    assert review_template(submitted_file_rule=False) is REVIEW_TEMPLATE

    ruled = review_template(submitted_file_rule=True)
    assert ruled.count(SUBMITTED_FILE_RULE) == 1
    assert f"{SUBMITTED_FILE_RULE}\n\n{ANCHOR}" in ruled
    assert ruled.replace(f"{SUBMITTED_FILE_RULE}\n\n", "") == REVIEW_TEMPLATE
    assert review_template(ruled, submitted_file_rule=True) == ruled


def test_a_custom_template_takes_the_rule_at_its_end():
    custom = "Review {trial_path}.\n\nGuidance:\n{criteria_guidance}\n"
    assert review_template(custom) is custom
    assert review_template(custom, submitted_file_rule=True) == (
        f"{custom.rstrip()}\n\n{SUBMITTED_FILE_RULE}\n"
    )


def test_the_rule_says_what_it_promises():
    """Generic (no task's file name), the deliverable judged first and by
    page, the PDF over its sources, and other records still required."""
    for phrase in (
        "written deliverable",
        "judge the file the agent submitted",
        "Read it before you judge",
        "page images",
        "Cite the page numbers",
        "compiled PDF wins over its TeX or Markdown sources",
        "still require those records",
    ):
        assert phrase in SUBMITTED_FILE_RULE
    assert "paper.pdf" not in SUBMITTED_FILE_RULE
    assert "{" not in SUBMITTED_FILE_RULE and "}" not in SUBMITTED_FILE_RULE
    assert chr(0x2014) not in SUBMITTED_FILE_RULE  # no em dash


def test_the_rendered_instruction_carries_the_rule_when_asked():
    from benchflow.review.config import Rubric

    rubric = Rubric.model_validate(RUBRIC)
    plain = render_review_instruction(rubric)
    ruled = render_review_instruction(rubric, submitted_file_rule=True)
    assert SUBMITTED_FILE_RULE not in plain
    assert ruled.replace(f"{SUBMITTED_FILE_RULE}\n\n", "") == plain


async def test_the_rule_reaches_the_reviewer_and_the_record(tmp_path, monkeypatch):
    fake = InstructionRun(review_payload=good_weighted_review())
    _, _, out = await _automatic(
        tmp_path,
        monkeypatch,
        ReviewerConfig(agent="codex", model="m", submitted_file_rule=True),
        fake=fake,
    )
    assert f"{SUBMITTED_FILE_RULE}\n\n{ANCHOR}" in fake.task_docs[0]
    record = _record(out)
    assert record["submitted_file_rule"] is True
    assert record["template_sha256"] == _sha256(
        review_template(submitted_file_rule=True)
    )
    assert record["instruction_sha256"] == _sha256(fake.instructions[0])


async def test_the_default_record_hashes_the_default_template(tmp_path, monkeypatch):
    """The instruction hash is of the golden instruction a61a26f4 wrote."""
    fake = InstructionRun(review_payload=good_weighted_review())
    _, _, out = await _automatic(
        tmp_path,
        monkeypatch,
        ReviewerConfig(agent="codex", model="azure/gpt5.6terra"),
        fake=fake,
    )
    record = _record(out)
    assert record["submitted_file_rule"] is False
    assert record["template_sha256"] == _sha256(REVIEW_TEMPLATE)
    golden = GOLDEN / "automatic.md"
    golden_task = tmp_path / "golden-task"
    golden_task.mkdir()
    (golden_task / "task.md").write_text(golden.read_text(encoding="utf-8"))
    assert record["instruction_sha256"] == _sha256(Task(golden_task).instruction)


async def test_a_custom_prompt_is_hashed_as_given(tmp_path, monkeypatch):
    custom = tmp_path / "prompt.md"
    custom.write_text("Review {trial_path}.\n\nGuidance:\n{criteria_guidance}\n")
    monkeypatch.setattr(benchflow, "run", FakeRun(review_payload=good_review()))
    task = make_task(tmp_path, with_rubric=True)
    rollout = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
    _, report_path = await run_reviews(
        rollout,
        agent="gemini",
        model="gemini/test-model",
        prompt_path=custom,
        out_dir=tmp_path / "out",
        tasks_root=tmp_path / "tasks",
        submitted_file_rule=True,
    )
    (entry,) = json.loads(report_path.read_text())["trials"]
    environment = entry["reviewer_environment"]
    assert environment["submitted_file_rule"] is True
    assert environment["template_sha256"] == _sha256(
        review_template(custom.read_text(), submitted_file_rule=True)
    )


def test_eval_run_takes_the_rule_flag(tmp_path, monkeypatch):
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
            "--reviewer-submitted-file-rule",
            "--reviewer-pdf-tools",
        ],
    )
    assert ran.exit_code == 0, ran.output
    assert captured == [ReviewerConfig(pdf_tools=True, submitted_file_rule=True)]


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], {"pdf_tools": False, "submitted_file_rule": False}),
        (
            ["--pdf-tools", "--submitted-file-rule"],
            {"pdf_tools": True, "submitted_file_rule": True},
        ),
    ],
)
def test_bench_review_takes_both_flags(tmp_path, monkeypatch, flags, expected):
    import benchflow.review as review

    seen = {}

    async def fake_run_reviews(path, **kwargs):
        seen.update(kwargs)
        raise review.ReviewRunError("stop here")

    monkeypatch.setattr(review, "run_reviews", fake_run_reviews)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    CliRunner().invoke(app, ["review", str(tmp_path), *flags])
    assert {key: seen[key] for key in expected} == expected
