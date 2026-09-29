"""A reviewer with no default model must name the reviewer's own option.

Guards the reviewer preflight added by PR #1126: a task with a weighted
``rubric.json`` run without ``--reviewer-model`` was refused with the solver's
message ("pass --model"), although ``--model`` sets the solver and cannot fix
the refusal.
"""

from __future__ import annotations

import pytest

from benchflow.review import automatic
from benchflow.review.options import ReviewerConfig
from tests.test_review_runtime import WEIGHTED_RUBRIC, make_task


def test_reviewer_without_a_model_names_the_reviewer_option(tmp_path):
    task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
    with pytest.raises(ValueError) as caught:
        automatic.prepare_review(task, ReviewerConfig())
    message = str(caught.value)
    assert "--reviewer-model" in message
    assert "pass --model" not in message
    assert "ReviewerConfig(model=" in message
