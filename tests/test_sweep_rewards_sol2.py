"""Guards PY8 taskmd/rewards sweep fixes against base commit 363d815b."""

from pathlib import Path

import pytest

from benchflow.rewards.builtins import LLMJudgeRewardFunc
from benchflow.rewards.file_readers import find_deliverables
from benchflow.rewards.rubric_config import Criterion


@pytest.mark.parametrize("kind", ["binary", "numeric", "likert"])
@pytest.mark.parametrize(
    "answer",
    [
        {},
        {"score": None},
        {"score": "no"},
        {"score": True},
        {"verdict": "unknown"},
        {"verdict": True},
    ],
)
def test_malformed_judge_answers_are_errors(kind, answer):
    """Absence or a malformed value is never a graded verdict."""
    criterion = Criterion(name="c", description="correct", type=kind, min=-1, max=0)
    with pytest.raises(ValueError):
        LLMJudgeRewardFunc._extract_score(criterion, answer)


def test_plain_deliverables_and_valid_scores_still_work(tmp_path: Path):
    (tmp_path / "answer.txt").write_text("correct")
    assert find_deliverables(tmp_path) == {"answer.txt": "correct"}
    assert (
        LLMJudgeRewardFunc._extract_score(
            Criterion(name="c", description="ok"), {"verdict": "pass"}
        )
        == 1
    )
